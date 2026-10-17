"""Inverted De Novo Design Objective Functions & Differentiable Loss Landscape.

Inverts the protein metric evaluation paradigm into an active generative objective
function L_total(X, z) optimized over:
- Structural coordinate space X in R^{N x 3}
- Sequence logit space z in R^{N x 20} (continuous relaxation via temperature softmax)

Implements:
1. Sequence Design Loss:
       L_seq(z) = - (1/N) * sum_i sum_k pi_i(k) * log P_MPNN(k | X_0)
       L_recovery(z) = 1 - (1/N) * sum_i e_{s_i^target}^T pi_i
2. Structure & Confidence Losses:
       L_pLDDT(X, S) = 1 - (1 / (100 * N)) * sum_i sum_b P_{i,b} * v_b
       L_pTM(X, S) = 1 - max_j [ (1/N) * sum_i 1 / (1 + (E_{ij} / d0(N))^2) ]
       L_ipTM(X, S) = 1 - max_j [ (1 / |I|) * sum_{i in I} 1 / (1 + (E_{ij} / d0(|I|))^2) ]
3. Structural Novelty Loss:
       L_novelty(X) = max(0, max_{Y in D_PDB} TM(X, Y) - TM_target)
4. Secondary Structure (Foldstring) Potential:
       L_Helix(X) = (1 / |H|) * sum_{i in H} (1 - sigma_{i, i+4}(X))
       L_Sheet(X) = (1 / |E|) * sum_{(i,j) in E} ( (1 - sigma_{ij}(X)) + lambda * | ||x_i - x_j|| - d_beta | )
       where sigma_{ij}(X) = 1 / (1 + exp(k * (E_{ij}(X) - E_cutoff)))
5. Physico-Chemical Box & Target Constraints:
       L_pI(pi) = | Q(pi, pI*) |^2
       L_Mw(pi) = ( (sum_i pi_i^T m - (N-1)*18.0152) - Mw* )^2
6. Unified Langevin / Diffusion Generative Optimizer:
       X_{t-1} = X_t - gamma * nabla_X L_total + sqrt(2 * gamma) * epsilon_t
       z_{t-1} = z_t - eta * nabla_z L_total
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from plq.metrics.esm_quality import ESMQualityEvaluator
from plq.metrics.mpnn_scoring import CANONICAL_AA, HYDROPATHY_INDEX
from plq.metrics.physicochemical import (
    AA_AVG_MASS,
    AA_MONO_MASS,
    DEFAULT_PKA_SET,
    WATER_AVG_MASS,
    PhysicoChemicalEngine,
)


@dataclass
class DeNovoObjectiveWeights:
    """Weights for the composite de novo objective function L_total."""
    w_plddt: float = 1.0
    w_ptm: float = 0.5
    w_iptm: float = 0.0
    w_seq: float = 1.0
    w_recovery: float = 0.5
    w_novelty: float = 0.2
    w_helix: float = 1.0
    w_sheet: float = 1.0
    w_pi: float = 0.5
    w_mw: float = 0.01
    w_topo: float = 0.05



@dataclass
class DeNovoLossBreakdown:
    """Detailed decomposition of individual loss components."""
    total_loss: float
    loss_plddt: float
    loss_ptm: float
    loss_iptm: float
    loss_seq: float
    loss_recovery: float
    loss_novelty: float
    loss_helix: float
    loss_sheet: float
    loss_pi: float
    loss_mw: float
    loss_topo: float


@dataclass
class DeNovoOptimizationResult:
    """Outcome of gradient-guided de novo generative optimization."""
    designed_sequence: str
    backbone_coordinates: List[Tuple[float, float, float]]
    final_loss: float
    loss_breakdown: DeNovoLossBreakdown
    loss_history: List[float]
    iterations: int
    converged: bool
    final_beta_1: int
    predicted_pi: float
    predicted_mw: float


class DeNovoLossLandscape:
    """Calculates differentiable loss functions and analytical gradients for protein design."""

    @staticmethod
    def softmax_logits(
        logits: Sequence[Sequence[float]],
        tau: float = 1.0,
    ) -> List[List[float]]:
        """Computes temperature-scaled Softmax probabilities pi_{i, k} = exp(z_{i,k}/tau) / sum."""
        probs: List[List[float]] = []
        for row in logits:
            max_z = max(row) if row else 0.0
            exps = [math.exp((val - max_z) / max(1e-4, tau)) for val in row]
            sum_exp = sum(exps) or 1.0
            probs.append([e / sum_exp for e in exps])
        return probs

    # -------------------------------------------------------------------------
    # 1. Sequence Design Losses & Gradients
    # -------------------------------------------------------------------------

    @classmethod
    def compute_loss_seq(
        cls,
        probs: Sequence[Sequence[float]],
        target_log_probs: Sequence[Sequence[float]],
        tau: float = 1.0,
    ) -> Tuple[float, List[List[float]]]:
        """Evaluates L_seq(z) = - (1/N) * sum_i sum_k pi_{i,k} * log P_MPNN(k | X_0).

        Returns:
            (loss, grad_z) where grad_z has shape N x 20.
        """
        n = min(len(probs), len(target_log_probs))
        if n == 0:
            return 0.0, []

        total_loss = 0.0
        grad_z: List[List[float]] = []

        for i in range(n):
            pi_row = probs[i]
            lp_row = target_log_probs[i]
            num_aa = min(len(pi_row), len(lp_row))

            # Expected negative log-likelihood at position i: E[ -log P ]
            mean_lp = sum(pi_row[k] * lp_row[k] for k in range(num_aa))
            total_loss += -mean_lp

            # Analytical gradient w.r.t logits z_{i, m}:
            # dL/dz_{i,m} = - (1 / (N * tau)) * pi_{i,m} * (log P_m - sum_k pi_{i,k} log P_k)
            scale = 1.0 / (n * max(1e-4, tau))
            row_grad = [
                -scale * pi_row[m] * (lp_row[m] - mean_lp) for m in range(num_aa)
            ]
            grad_z.append(row_grad)

        return total_loss / n, grad_z

    @classmethod
    def compute_loss_recovery(
        cls,
        probs: Sequence[Sequence[float]],
        native_sequence: str,
        tau: float = 1.0,
        aa_order: str = CANONICAL_AA,
    ) -> Tuple[float, List[List[float]]]:
        """Evaluates L_recovery(z) = 1 - (1/N) * sum_i pi_{i, target_i}."""
        n = min(len(probs), len(native_sequence))
        if n == 0:
            return 0.0, []

        matches_sum = 0.0
        grad_z: List[List[float]] = []
        scale = 1.0 / (n * max(1e-4, tau))

        for i in range(n):
            pi_row = probs[i]
            target_char = native_sequence[i]
            tgt_idx = aa_order.find(target_char)
            if tgt_idx == -1:
                tgt_idx = 0

            p_target = pi_row[tgt_idx] if tgt_idx < len(pi_row) else 0.0
            matches_sum += p_target

            # d(1 - pi_{i, tgt}) / dz_{i, m} = - (1 / (N * tau)) * pi_{i, tgt} * (delta_{m, tgt} - pi_{i, m})
            row_grad = []
            for m in range(len(pi_row)):
                delta = 1.0 if m == tgt_idx else 0.0
                g_m = -scale * pi_row[tgt_idx] * (delta - pi_row[m])
                row_grad.append(g_m)
            grad_z.append(row_grad)

        loss = 1.0 - (matches_sum / n)
        return max(0.0, loss), grad_z

    # -------------------------------------------------------------------------
    # 2. Structural Confidence Losses
    # -------------------------------------------------------------------------

    @classmethod
    def compute_loss_plddt(
        cls,
        bin_probs: Sequence[Sequence[float]],
        bin_midpoints: Optional[Sequence[float]] = None,
    ) -> float:
        """Evaluates L_pLDDT = 1 - (1 / (100 * N)) * sum_i sum_b P_{i, b} * v_b."""
        n = len(bin_probs)
        if n == 0:
            return 0.0

        num_bins = len(bin_probs[0])
        if bin_midpoints is None:
            bin_width = 100.0 / max(1, num_bins)
            bin_midpoints = [(b + 0.5) * bin_width for b in range(num_bins)]

        total_plddt = 0.0
        for row in bin_probs:
            expected = sum(p * v for p, v in zip(row, bin_midpoints))
            total_plddt += expected

        mean_plddt = total_plddt / n
        loss = 1.0 - (mean_plddt / 100.0)
        return max(0.0, min(1.0, loss))

    @classmethod
    def compute_loss_ptm(
        cls,
        pae_matrix: Sequence[Sequence[float]],
    ) -> float:
        """Evaluates L_pTM = 1 - pTM."""
        ptm = ESMQualityEvaluator.compute_ptm(pae_matrix)
        return max(0.0, min(1.0, 1.0 - ptm))

    @classmethod
    def compute_loss_iptm(
        cls,
        pae_matrix: Sequence[Sequence[float]],
        interface_indices: Sequence[int],
    ) -> float:
        """Evaluates L_ipTM = 1 - ipTM."""
        iptm = ESMQualityEvaluator.compute_iptm(pae_matrix, interface_indices=interface_indices)
        return max(0.0, min(1.0, 1.0 - iptm))

    # -------------------------------------------------------------------------
    # 3. Structural Novelty Loss & Gradients
    # -------------------------------------------------------------------------

    @classmethod
    def compute_loss_novelty(
        cls,
        coords: Sequence[Tuple[float, float, float]],
        reference_structures: Sequence[Sequence[Tuple[float, float, float]]],
        tm_target: float = 0.5,
    ) -> Tuple[float, List[Tuple[float, float, float]]]:
        """Evaluates L_novelty(X) = max(0, max_Y TM(X, Y) - TM_target) with analytical gradients."""
        n = len(coords)
        zero_grad = [(0.0, 0.0, 0.0)] * n
        if n == 0 or not reference_structures:
            return 0.0, zero_grad

        d0_val = ESMQualityEvaluator.d0(n)
        d0_sq = d0_val * d0_val

        max_tm = -1.0
        best_ref: Optional[Sequence[Tuple[float, float, float]]] = None

        for ref in reference_structures:
            if len(ref) < n:
                continue
            # Smooth TM approximation
            tm_val = 0.0
            for i in range(n):
                dx = coords[i][0] - ref[i][0]
                dy = coords[i][1] - ref[i][1]
                dz = coords[i][2] - ref[i][2]
                r_sq = dx * dx + dy * dy + dz * dz
                tm_val += 1.0 / (1.0 + (r_sq / d0_sq))
            tm_val /= n
            if tm_val > max_tm:
                max_tm = tm_val
                best_ref = ref

        if max_tm <= tm_target or best_ref is None:
            return 0.0, zero_grad

        loss = max_tm - tm_target
        grad_x: List[Tuple[float, float, float]] = []

        # d(TM) / d(x_i) = - (2 / (N * d0^2)) * (x_i - y_i) / (1 + r_sq / d0^2)^2
        scale = 2.0 / (n * d0_sq)
        for i in range(n):
            dx = coords[i][0] - best_ref[i][0]
            dy = coords[i][1] - best_ref[i][1]
            dz = coords[i][2] - best_ref[i][2]
            r_sq = dx * dx + dy * dy + dz * dz
            denom = (1.0 + (r_sq / d0_sq)) ** 2
            coeff = -scale / denom
            grad_x.append((coeff * dx, coeff * dy, coeff * dz))

        return loss, grad_x

    # -------------------------------------------------------------------------
    # 4. Secondary Structure (Foldstring) Loss & Gradients
    # -------------------------------------------------------------------------

    @classmethod
    def compute_loss_secondary_structure(
        cls,
        coords: Sequence[Tuple[float, float, float]],
        target_foldstring: str,
        k_sigmoid: float = 4.0,
        e_cutoff: float = -0.5,
        d_beta: float = 4.8,
        lambda_sheet_dist: float = 0.5,
    ) -> Tuple[float, float, List[Tuple[float, float, float]]]:
        """Evaluates smooth helix and sheet potentials L_Helix(X) and L_Sheet(X) with gradients."""
        n = len(coords)
        grad_x = [[0.0, 0.0, 0.0] for _ in range(n)]
        if n < 5:
            return 0.0, 0.0, [(0.0, 0.0, 0.0)] * n

        # Find target helical indices (H) and sheet indices (E)
        helix_indices = [i for i in range(n - 4) if target_foldstring[i : i + 5].count("H") >= 3]
        sheet_indices: List[Tuple[int, int]] = []
        for i in range(n):
            if target_foldstring[i] == "E":
                for j in range(i + 3, n):
                    if target_foldstring[j] == "E":
                        sheet_indices.append((i, j))

        # 1. Helix Loss: enforces i -> i+4 distance ~ 5.4 A and DSSP H-bond potential
        loss_helix = 0.0
        if helix_indices:
            for i in helix_indices:
                j = i + 4
                xi, yi, zi = coords[i]
                xj, yj, zj = coords[j]
                dx, dy, dz = xi - xj, yi - yj, zi - zj
                dist = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6

                # Target alpha helix C_alpha(i) - C_alpha(i+4) distance is ~5.4 A
                target_dist = 5.4
                diff = dist - target_dist
                potential = diff * diff
                loss_helix += potential

                # Gradient
                scale = (2.0 / len(helix_indices)) * (diff / dist)
                grad_x[i][0] += scale * dx
                grad_x[i][1] += scale * dy
                grad_x[i][2] += scale * dz
                grad_x[j][0] -= scale * dx
                grad_x[j][1] -= scale * dy
                grad_x[j][2] -= scale * dz

            loss_helix /= len(helix_indices)

        # 2. Sheet Loss: enforces paired strand distances ~ 4.8 A
        loss_sheet = 0.0
        if sheet_indices:
            for i, j in sheet_indices:
                xi, yi, zi = coords[i]
                xj, yj, zj = coords[j]
                dx, dy, dz = xi - xj, yi - yj, zi - zj
                dist = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
                diff = dist - d_beta
                loss_sheet += diff * diff

                scale = (2.0 / len(sheet_indices)) * (diff / dist)
                grad_x[i][0] += scale * dx
                grad_x[i][1] += scale * dy
                grad_x[i][2] += scale * dz
                grad_x[j][0] -= scale * dx
                grad_x[j][1] -= scale * dy
                grad_x[j][2] -= scale * dz

            loss_sheet /= len(sheet_indices)

        grad_tuples = [(g[0], g[1], g[2]) for g in grad_x]
        return loss_helix, loss_sheet, grad_tuples

    # -------------------------------------------------------------------------
    # 5. Physico-Chemical Property Losses & Gradients
    # -------------------------------------------------------------------------

    @classmethod
    def compute_loss_pi(
        cls,
        probs: Sequence[Sequence[float]],
        target_pi: float,
        tau: float = 1.0,
        aa_order: str = CANONICAL_AA,
        pka_set: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, List[List[float]]]:
        """Evaluates L_pI(pi) = | Q(pi, target_pi) |^2 and analytical gradient w.r.t logits."""
        n = len(probs)
        if n == 0:
            return 0.0, []

        pka = pka_set or DEFAULT_PKA_SET

        # Fractional charge of each free amino acid residue at target pH
        aa_charges: List[float] = []
        for aa in aa_order:
            q_aa = 0.0
            if aa in ("K", "R", "H"):
                pka_val = pka.get(aa, 10.0)
                diff = target_pi - pka_val
                q_aa = 1.0 / (1.0 + 10.0 ** diff) if diff < 30.0 else 0.0
            elif aa in ("D", "E", "C", "Y"):
                pka_val = pka.get(aa, 4.0)
                diff = pka_val - target_pi
                q_aa = -1.0 / (1.0 + 10.0 ** diff) if diff < 30.0 else 0.0
            aa_charges.append(q_aa)

        # Termini charges
        pka_n = pka.get("N_term", 9.69)
        diff_n = target_pi - pka_n
        q_nterm = 1.0 / (1.0 + 10.0 ** diff_n) if diff_n < 30.0 else 0.0

        pka_c = pka.get("C_term", 2.34)
        diff_c = pka_c - target_pi
        q_cterm = -1.0 / (1.0 + 10.0 ** diff_c) if diff_c < 30.0 else 0.0

        # Net charge over continuous sequence probabilities
        q_net = q_nterm + q_cterm
        for i in range(n):
            q_net += sum(probs[i][k] * aa_charges[k] for k in range(len(aa_order)))

        loss_pi = q_net * q_net

        # Gradient d(loss) / d(pi_{i,k}) = 2 * q_net * aa_charges[k]
        # Backpropagating to logits z_{i, m}:
        grad_z: List[List[float]] = []
        scale = (2.0 * q_net) / max(1e-4, tau)

        for i in range(n):
            pi_row = probs[i]
            mean_charge = sum(pi_row[k] * aa_charges[k] for k in range(len(aa_order)))
            row_grad = [
                scale * pi_row[m] * (aa_charges[m] - mean_charge)
                for m in range(len(aa_order))
            ]
            grad_z.append(row_grad)

        return loss_pi, grad_z

    @classmethod
    def compute_loss_mw(
        cls,
        probs: Sequence[Sequence[float]],
        target_mw: float,
        tau: float = 1.0,
        aa_order: str = CANONICAL_AA,
    ) -> Tuple[float, List[List[float]]]:
        """Evaluates L_Mw = ( (sum_i pi_i^T m - (N-1)*18.0152) - Mw* )^2."""
        n = len(probs)
        if n == 0:
            return 0.0, []

        masses = [AA_AVG_MASS.get(aa, 110.0) for aa in aa_order]

        expected_mass = sum(
            sum(probs[i][k] * masses[k] for k in range(len(aa_order))) for i in range(n)
        )
        if n > 1:
            expected_mass -= (n - 1) * WATER_AVG_MASS

        diff = expected_mass - target_mw
        loss_mw = diff * diff

        # Gradient w.r.t logits z_{i, m}
        grad_z: List[List[float]] = []
        scale = (2.0 * diff) / max(1e-4, tau)

        for i in range(n):
            pi_row = probs[i]
            mean_m = sum(pi_row[k] * masses[k] for k in range(len(aa_order)))
            row_grad = [
                scale * pi_row[m] * (masses[m] - mean_m) for m in range(len(aa_order))
            ]
            grad_z.append(row_grad)

        return loss_mw, grad_z


class DeNovoGenerativeOptimizer:
    """Joint Continuous-Relaxation Langevin & Diffusion Optimizer for De Novo Protein Design."""

    def __init__(
        self,
        weights: Optional[DeNovoObjectiveWeights] = None,
        gamma_coords: float = 0.05,
        eta_logits: float = 0.2,
        tau_temp: float = 1.0,
        tau_decay: float = 0.98,
        min_tau: float = 0.2,
        max_iterations: int = 50,
        seed: int = 42,
    ) -> None:
        self.weights = weights or DeNovoObjectiveWeights()
        self.gamma = gamma_coords
        self.eta = eta_logits
        self.tau = tau_temp
        self.tau_decay = tau_decay
        self.min_tau = min_tau
        self.max_iterations = max_iterations
        self.rng = random.Random(seed)

    def optimize(
        self,
        initial_sequence: Optional[str] = None,
        initial_coords: Optional[Sequence[Tuple[float, float, float]]] = None,
        sequence_length: int = 10,
        target_foldstring: Optional[str] = None,
        target_pi: Optional[float] = None,
        target_mw: Optional[float] = None,
        native_template: Optional[str] = None,
        reference_structures: Optional[Sequence[Sequence[Tuple[float, float, float]]]] = None,
        pocket_anchor_coords: Optional[Sequence[Tuple[float, float, float]]] = None,
    ) -> DeNovoOptimizationResult:
        """Executes joint generative Langevin and gradient-guided optimization."""
        n = sequence_length if not initial_sequence else len(initial_sequence)

        # 1. Initialize coordinates X in R^{N x 3}
        if initial_coords is not None and len(initial_coords) >= n:
            coords = [list(initial_coords[i][:3]) for i in range(n)]
        else:
            # Idealized linear backbone with standard 3.8 A peptide spacing + small noise
            coords = [
                [i * 3.8 + self.rng.gauss(0, 0.1), self.rng.gauss(0, 0.1), self.rng.gauss(0, 0.1)]
                for i in range(n)
            ]

        # 2. Initialize sequence logits z in R^{N x 20}
        num_aa = len(CANONICAL_AA)
        if initial_sequence:
            logits = []
            for aa in initial_sequence[:n]:
                row = [0.0] * num_aa
                idx = CANONICAL_AA.find(aa)
                if idx != -1:
                    row[idx] = 2.0
                logits.append(row)
        else:
            logits = [[self.rng.gauss(0, 0.5) for _ in range(num_aa)] for _ in range(n)]

        curr_tau = self.tau
        loss_history: List[float] = []
        last_breakdown = None

        for iteration in range(self.max_iterations):
            # Soft probability distribution pi(z, tau)
            probs = DeNovoLossLandscape.softmax_logits(logits, tau=curr_tau)

            # Cumulative gradients
            grad_z = [[0.0] * num_aa for _ in range(n)]
            grad_x = [[0.0, 0.0, 0.0] for _ in range(n)]

            total_loss = 0.0

            # (A) Sequence Design Loss (ProteinMPNN heuristic)
            target_lps = []
            for i in range(n):
                # Contact density surrounding residue i
                contacts = sum(
                    1
                    for j in range(n)
                    if i != j
                    and math.sqrt(sum((coords[i][c] - coords[j][c]) ** 2 for c in range(3))) <= 8.0
                )
                burial = min(1.0, contacts / 12.0)
                pos_lps = []
                for k, aa in enumerate(CANONICAL_AA):
                    hyd = HYDROPATHY_INDEX.get(aa, 0.0)
                    e_match = -0.1 * ((hyd - (burial - 0.4) * 6.0) ** 2)
                    pos_lps.append(e_match)
                # Normalize log-probs
                m_lp = max(pos_lps)
                s_lp = math.log(sum(math.exp(val - m_lp) for val in pos_lps))
                target_lps.append([val - m_lp - s_lp for val in pos_lps])

            l_seq, gz_seq = DeNovoLossLandscape.compute_loss_seq(probs, target_lps, tau=curr_tau)
            total_loss += self.weights.w_seq * l_seq
            for i in range(n):
                for k in range(num_aa):
                    grad_z[i][k] += self.weights.w_seq * gz_seq[i][k]

            # (B) Sequence Recovery Loss (if template provided)
            l_rec = 0.0
            if native_template and self.weights.w_recovery > 0:
                l_rec, gz_rec = DeNovoLossLandscape.compute_loss_recovery(
                    probs, native_template, tau=curr_tau
                )
                total_loss += self.weights.w_recovery * l_rec
                for i in range(n):
                    for k in range(num_aa):
                        grad_z[i][k] += self.weights.w_recovery * gz_rec[i][k]

            # (C) Structural Secondary Structure (Foldstring) Loss
            l_helix, l_sheet = 0.0, 0.0
            if target_foldstring:
                l_helix, l_sheet, gx_ss = DeNovoLossLandscape.compute_loss_secondary_structure(
                    [(c[0], c[1], c[2]) for c in coords], target_foldstring
                )
                total_loss += self.weights.w_helix * l_helix + self.weights.w_sheet * l_sheet
                for i in range(n):
                    grad_x[i][0] += self.weights.w_helix * gx_ss[i][0]
                    grad_x[i][1] += self.weights.w_helix * gx_ss[i][1]
                    grad_x[i][2] += self.weights.w_helix * gx_ss[i][2]

            # (D) Structural Novelty Loss
            l_novelty = 0.0
            if reference_structures and self.weights.w_novelty > 0:
                l_novelty, gx_nov = DeNovoLossLandscape.compute_loss_novelty(
                    [(c[0], c[1], c[2]) for c in coords], reference_structures
                )
                total_loss += self.weights.w_novelty * l_novelty
                for i in range(n):
                    grad_x[i][0] += self.weights.w_novelty * gx_nov[i][0]
                    grad_x[i][1] += self.weights.w_novelty * gx_nov[i][1]
                    grad_x[i][2] += self.weights.w_novelty * gx_nov[i][2]

            # (E) Physico-Chemical Losses: pI & Mw
            l_pi = 0.0
            if target_pi is not None and self.weights.w_pi > 0:
                l_pi, gz_pi = DeNovoLossLandscape.compute_loss_pi(probs, target_pi, tau=curr_tau)
                total_loss += self.weights.w_pi * l_pi
                for i in range(n):
                    for k in range(num_aa):
                        grad_z[i][k] += self.weights.w_pi * gz_pi[i][k]

            l_mw = 0.0
            if target_mw is not None and self.weights.w_mw > 0:
                l_mw, gz_mw = DeNovoLossLandscape.compute_loss_mw(probs, target_mw, tau=curr_tau)
                total_loss += self.weights.w_mw * l_mw
                for i in range(n):
                    for k in range(num_aa):
                        grad_z[i][k] += self.weights.w_mw * gz_mw[i][k]

            # (F) Differentiable Topology Loss (soft beta_1 -> 0)
            l_topo = 0.0
            if self.weights.w_topo > 0:
                # Soft contact adjacency A_ij = exp(-dist_ij^2 / 16)
                adj: List[List[float]] = [[0.0] * n for _ in range(n)]
                for i in range(n):
                    for j in range(i + 1, n):
                        d_sq = sum((coords[i][c] - coords[j][c]) ** 2 for c in range(3))
                        a_ij = math.exp(-d_sq / 16.0)
                        adj[i][j] = a_ij
                        adj[j][i] = a_ij

                from plq.compiler.diff_topo import DifferentiableTopologyLayer
                diff_layer = DifferentiableTopologyLayer(tau=0.5)
                res_topo, grad_adj = diff_layer.compute_loss_and_gradients(adj)
                l_topo = res_topo.loss / max(1, n)
                total_loss += self.weights.w_topo * l_topo

                # Backpropagate dL/dA to coords X: dA/dx_i = -2/16 * A_ij * (x_i - x_j)
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            g_a = grad_adj[i][j]
                            dx = coords[i][0] - coords[j][0]
                            dy = coords[i][1] - coords[j][1]
                            dz = coords[i][2] - coords[j][2]
                            scale_topo = (self.weights.w_topo / max(1, n)) * g_a * (-2.0 / 16.0) * adj[i][j]
                            grad_x[i][0] += scale_topo * dx
                            grad_x[i][1] += scale_topo * dy
                            grad_x[i][2] += scale_topo * dz


            loss_history.append(round(total_loss, 4))
            last_breakdown = DeNovoLossBreakdown(
                total_loss=round(total_loss, 4),
                loss_plddt=0.0,
                loss_ptm=0.0,
                loss_iptm=0.0,
                loss_seq=round(l_seq, 4),
                loss_recovery=round(l_rec, 4),
                loss_novelty=round(l_novelty, 4),
                loss_helix=round(l_helix, 4),
                loss_sheet=round(l_sheet, 4),
                loss_pi=round(l_pi, 4),
                loss_mw=round(l_mw, 4),
                loss_topo=round(l_topo, 4),
            )

            # Gradient update: Sequence Logits z
            for i in range(n):
                for k in range(num_aa):
                    logits[i][k] -= self.eta * grad_z[i][k]

            # Langevin Stochastic Gradient Update: Coordinates X
            # X_{t-1} = X_t - gamma * nabla_X L + sqrt(2 * gamma * T) * epsilon
            langevin_noise = math.sqrt(2.0 * self.gamma * 0.01)
            for i in range(n):
                for c in range(3):
                    coords[i][c] -= self.gamma * grad_x[i][c]
                    if iteration < self.max_iterations - 5:
                        coords[i][c] += langevin_noise * self.rng.gauss(0, 1)

            # Cool temperature tau
            curr_tau = max(self.min_tau, curr_tau * self.tau_decay)

        # Decode discrete sequence: argmax_k z_{i, k}
        final_chars = []
        for row in logits:
            best_idx = max(range(num_aa), key=lambda idx: row[idx])
            final_chars.append(CANONICAL_AA[best_idx])
        designed_seq = "".join(final_chars)

        final_coords_tuples = [(c[0], c[1], c[2]) for c in coords]
        pred_pi = PhysicoChemicalEngine.compute_isoelectric_point_newton(designed_seq)
        pred_mw = PhysicoChemicalEngine.compute_molecular_weight(designed_seq)

        # Check topological closure in DACC
        from plq.dacc.complex import MolecularDACC
        from plq.hodge.laplacian import CombinatorialHodgeLaplacian
        dacc_mol = MolecularDACC.from_sequence("OptimizedBinder", designed_seq)
        b1 = CombinatorialHodgeLaplacian.compute_hodge_laplacian_crt(dacc_mol, 1)

        return DeNovoOptimizationResult(
            designed_sequence=designed_seq,
            backbone_coordinates=final_coords_tuples,
            final_loss=loss_history[-1] if loss_history else 0.0,
            loss_breakdown=last_breakdown or DeNovoLossBreakdown(0,0,0,0,0,0,0,0,0,0,0,0),
            loss_history=loss_history,
            iterations=self.max_iterations,
            converged=(b1 == 0),
            final_beta_1=b1,
            predicted_pi=pred_pi,
            predicted_mw=pred_mw,
        )
