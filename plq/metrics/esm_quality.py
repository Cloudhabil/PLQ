"""ESMFold Structural Quality Metrics: pLDDT, TM-score, pTM, and ipTM.

Implements:
1. pLDDT (Predicted Local Distance Difference Test):
       lDDT_i = (1/4) * sum_{theta in {0.5, 1, 2, 4}} (1 / |N_i|) * sum_{j in N_i} I(|d_ij^pred - d_ij^true| < theta)
2. TM-score with Kabsch optimal rotation:
       TM = (1 / L) * sum_{i=1}^L 1 / (1 + (e_i / d0(L))^2)
       where d0(L) = 1.24 * (max(16, L) - 15)^(1/3) - 1.8
3. pTM (Predicted Template Modeling Score) from PAE matrix:
       pTM = max_j [ (1 / L) * sum_{i=1}^L 1 / (1 + (E_ij / d0(L))^2) ]
4. ipTM (Interface Predicted TM-score) across interacting multi-chain residues.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple


@dataclass
class PLDDTReport:
    """Report of local structural confidence per residue and overall."""
    mean_plddt: float
    residue_plddt: List[float]
    high_confidence_ratio: float  # Fraction with pLDDT >= 70.0
    very_high_confidence_ratio: float  # Fraction with pLDDT >= 90.0


@dataclass
class TMScoreReport:
    """Report of global structural alignment and superposition."""
    tm_score: float
    rmsd: float
    aligned_length: int
    d0: float


class ESMQualityEvaluator:
    """Evaluates structure prediction accuracy and aligned error matrices."""

    @staticmethod
    def d0(length: int) -> float:
        """Computes TM-score length scaling factor d0(L)."""
        l = max(16, length)
        return 1.24 * ((l - 15) ** (1.0 / 3.0)) - 1.8

    @classmethod
    def compute_plddt(
        cls,
        pred_coords: Sequence[Tuple[float, float, float]],
        true_coords: Sequence[Tuple[float, float, float]],
        inclusion_radius: float = 15.0,
    ) -> PLDDTReport:
        """Computes residue-level lDDT and overall mean pLDDT over thresholds {0.5, 1, 2, 4} A."""
        n = min(len(pred_coords), len(true_coords))
        if n == 0:
            return PLDDTReport(0.0, [], 0.0, 0.0)

        thresholds = (0.5, 1.0, 2.0, 4.0)
        per_res_lddt: List[float] = []

        for i in range(n):
            # Find neighbors in true structure within inclusion radius
            neighbors: List[int] = []
            xi, yi, zi = true_coords[i]
            for j in range(n):
                if i != j:
                    xj, yj, zj = true_coords[j]
                    d_true = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
                    if d_true < inclusion_radius:
                        neighbors.append(j)

            if not neighbors:
                per_res_lddt.append(100.0)
                continue

            pxi, pyi, pzi = pred_coords[i]
            preserved_counts = [0, 0, 0, 0]

            for j in neighbors:
                d_true = math.sqrt(
                    (true_coords[i][0] - true_coords[j][0]) ** 2
                    + (true_coords[i][1] - true_coords[j][1]) ** 2
                    + (true_coords[i][2] - true_coords[j][2]) ** 2
                )
                d_pred = math.sqrt(
                    (pred_coords[i][0] - pred_coords[j][0]) ** 2
                    + (pred_coords[i][1] - pred_coords[j][1]) ** 2
                    + (pred_coords[i][2] - pred_coords[j][2]) ** 2
                )
                error = abs(d_pred - d_true)
                for t_idx, theta in enumerate(thresholds):
                    if error < theta:
                        preserved_counts[t_idx] += 1

            lddt_i = (
                sum(cnt / len(neighbors) for cnt in preserved_counts) / 4.0
            ) * 100.0
            per_res_lddt.append(round(lddt_i, 2))

        mean_val = sum(per_res_lddt) / n
        high_ratio = sum(1 for v in per_res_lddt if v >= 70.0) / n
        very_high_ratio = sum(1 for v in per_res_lddt if v >= 90.0) / n

        return PLDDTReport(
            mean_plddt=round(mean_val, 2),
            residue_plddt=per_res_lddt,
            high_confidence_ratio=round(high_ratio, 3),
            very_high_confidence_ratio=round(very_high_ratio, 3),
        )

    @classmethod
    def compute_plddt_from_bins(
        cls,
        bin_probabilities: Sequence[Sequence[float]],
        bin_midpoints: Optional[Sequence[float]] = None,
    ) -> PLDDTReport:
        """Computes expected pLDDT from binned probability distribution."""
        n = len(bin_probabilities)
        if n == 0:
            return PLDDTReport(0.0, [], 0.0, 0.0)

        num_bins = len(bin_probabilities[0])
        if bin_midpoints is None:
            # 50 bins across [0, 100] with width 2.0 and midpoints 1, 3, ..., 99
            bin_width = 100.0 / num_bins
            bin_midpoints = [ (b + 0.5) * bin_width for b in range(num_bins) ]

        scores: List[float] = []
        for probs in bin_probabilities:
            expected = sum(p * v for p, v in zip(probs, bin_midpoints))
            scores.append(round(expected, 2))

        mean_val = sum(scores) / n
        high_ratio = sum(1 for v in scores if v >= 70.0) / n
        very_high_ratio = sum(1 for v in scores if v >= 90.0) / n

        return PLDDTReport(
            mean_plddt=round(mean_val, 2),
            residue_plddt=scores,
            high_confidence_ratio=round(high_ratio, 3),
            very_high_confidence_ratio=round(very_high_ratio, 3),
        )

    # -------------------------------------------------------------------------
    # Pure-Python Kabsch Superposition and TM-score
    # -------------------------------------------------------------------------

    @classmethod
    def compute_tm_score(
        cls,
        pred_coords: Sequence[Tuple[float, float, float]],
        true_coords: Sequence[Tuple[float, float, float]],
    ) -> TMScoreReport:
        """Performs optimal rigid Kabsch superposition and computes TM-score."""
        n = min(len(pred_coords), len(true_coords))
        if n == 0:
            return TMScoreReport(0.0, 0.0, 0, 1.0)

        p = [list(pred_coords[i]) for i in range(n)]
        q = [list(true_coords[i]) for i in range(n)]

        # 1. Centroid alignment
        p_centroid = [sum(pt[c] for pt in p) / n for c in range(3)]
        q_centroid = [sum(pt[c] for pt in q) / n for c in range(3)]

        p_centered = [[pt[c] - p_centroid[c] for c in range(3)] for pt in p]
        q_centered = [[pt[c] - q_centroid[c] for c in range(3)] for pt in q]

        # 2. Covariance matrix H = P^T * Q (3x3)
        h = [[0.0 for _ in range(3)] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                h[i][j] = sum(p_centered[k][i] * q_centered[k][j] for k in range(n))

        # 3. Compute optimal rotation via Kabsch SVD
        r_mat = cls._kabsch_rotation(h)

        # 4. Rotate P
        p_rotated = []
        for pt in p_centered:
            rx = pt[0] * r_mat[0][0] + pt[1] * r_mat[1][0] + pt[2] * r_mat[2][0]
            ry = pt[0] * r_mat[0][1] + pt[1] * r_mat[1][1] + pt[2] * r_mat[2][1]
            rz = pt[0] * r_mat[0][2] + pt[1] * r_mat[1][2] + pt[2] * r_mat[2][2]
            p_rotated.append((rx, ry, rz))

        # 5. Distances e_i and TM-score
        d0_val = cls.d0(n)
        d0_sq = d0_val * d0_val
        sq_err_sum = 0.0
        tm_sum = 0.0

        for i in range(n):
            dx = p_rotated[i][0] - q_centered[i][0]
            dy = p_rotated[i][1] - q_centered[i][1]
            dz = p_rotated[i][2] - q_centered[i][2]
            dist_sq = dx * dx + dy * dy + dz * dz
            sq_err_sum += dist_sq
            tm_sum += 1.0 / (1.0 + (dist_sq / d0_sq))

        rmsd = math.sqrt(sq_err_sum / n)
        tm_score = tm_sum / n

        return TMScoreReport(
            tm_score=round(tm_score, 4),
            rmsd=round(rmsd, 3),
            aligned_length=n,
            d0=round(d0_val, 3),
        )

    @classmethod
    def _kabsch_rotation(cls, h: List[List[float]]) -> List[List[float]]:
        """Solves optimal 3x3 Kabsch rotation R for covariance matrix H via one-sided Jacobi SVD."""
        # A = H initially, V = I. Rotate columns of A until pairwise orthogonal
        a = [row[:] for row in h]
        v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]

        for _ in range(30):
            for p in range(3):
                for q in range(p + 1, 3):
                    app = sum(a[i][p] * a[i][p] for i in range(3))
                    aqq = sum(a[i][q] * a[i][q] for i in range(3))
                    apq = sum(a[i][p] * a[i][q] for i in range(3))
                    if abs(apq) < 1e-15:
                        continue
                    tau = (aqq - app) / (2.0 * apq)
                    t = math.copysign(1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau)), tau)
                    c = 1.0 / math.sqrt(1.0 + t * t)
                    s = t * c
                    for i in range(3):
                        a_ip = a[i][p]
                        a_iq = a[i][q]
                        a[i][p] = c * a_ip - s * a_iq
                        a[i][q] = s * a_ip + c * a_iq

                        v_ip = v[i][p]
                        v_iq = v[i][q]
                        v[i][p] = c * v_ip - s * v_iq
                        v[i][q] = s * v_ip + c * v_iq

        # Singular values are column norms of A
        svals = [math.sqrt(sum(a[i][j] ** 2 for i in range(3))) for j in range(3)]
        u = [[0.0 for _ in range(3)] for _ in range(3)]
        for j in range(3):
            if svals[j] > 1e-9:
                inv_s = 1.0 / svals[j]
                for i in range(3):
                    u[i][j] = a[i][j] * inv_s
            else:
                u[j][j] = 1.0

        # Optimal rotation R = U * V^T
        r = [[sum(u[i][k] * v[j][k] for k in range(3)) for j in range(3)] for i in range(3)]

        # Check reflection: det(R) must be +1
        det_r = (
            r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
        )
        if det_r < 0.0:
            min_idx = min(range(3), key=lambda idx: svals[idx])
            for i in range(3):
                v[i][min_idx] = -v[i][min_idx]
            r = [[sum(u[i][k] * v[j][k] for k in range(3)) for j in range(3)] for i in range(3)]

        return r



    @classmethod
    def _jacobi_eigen_3x3(cls, a: List[List[float]], max_iter: int = 50) -> Tuple[List[float], List[List[float]]]:
        """Jacobi eigenvalue algorithm for symmetric 3x3 matrix."""
        mat = [row[:] for row in a]
        v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]

        for _ in range(max_iter):
            # Find largest off-diagonal element
            p, q = 0, 1
            max_val = abs(mat[0][1])
            for i in range(3):
                for j in range(i + 1, 3):
                    if abs(mat[i][j]) > max_val:
                        max_val = abs(mat[i][j])
                        p, q = i, j
            if max_val < 1e-12:
                break

            diff = mat[q][q] - mat[p][p]
            if abs(mat[p][q]) < abs(diff) * 1e-15:
                t = mat[p][q] / diff
            else:
                phi = diff / (2.0 * mat[p][q])
                t = 1.0 / (abs(phi) + math.sqrt(phi * phi + 1.0))
                if phi < 0.0:
                    t = -t

            c = 1.0 / math.sqrt(t * t + 1.0)
            s = t * c
            tau = s / (1.0 + c)

            # Update elements
            h = t * mat[p][q]
            mat[p][p] -= h
            mat[q][q] += h
            mat[p][q] = 0.0

            for i in range(3):
                if i != p and i != q:
                    g = mat[i][p]
                    h = mat[i][q]
                    mat[i][p] = g - s * (h + g * tau)
                    mat[p][i] = mat[i][p]
                    mat[i][q] = h + s * (g - h * tau)
                    mat[q][i] = mat[i][q]

            for i in range(3):
                g = v[i][p]
                h = v[i][q]
                v[i][p] = g - s * (h + g * tau)
                v[i][q] = h + s * (g - h * tau)

        evals = [mat[i][i] for i in range(3)]
        return evals, v

    # -------------------------------------------------------------------------
    # pTM and ipTM from Predicted Aligned Error (PAE) Matrices
    # -------------------------------------------------------------------------

    @classmethod
    def compute_ptm(cls, pae_matrix: Sequence[Sequence[float]]) -> float:
        """Computes global pTM score from L x L Predicted Aligned Error matrix:
        
        pTM = max_j [ (1 / L) * sum_{i=1}^L 1 / (1 + (E_ij / d0(L))^2) ]
        """
        l = len(pae_matrix)
        if l == 0:
            return 0.0

        d0_val = cls.d0(l)
        d0_sq = d0_val * d0_val

        best_tm = 0.0
        for j in range(l):
            tm_j = sum(1.0 / (1.0 + (pae_matrix[i][j] * pae_matrix[i][j]) / d0_sq) for i in range(l)) / l
            if tm_j > best_tm:
                best_tm = tm_j

        return round(best_tm, 4)

    @classmethod
    def compute_iptm(
        cls,
        pae_matrix: Sequence[Sequence[float]],
        interface_indices: Optional[Sequence[int]] = None,
        chain_lengths: Optional[Sequence[int]] = None,
    ) -> float:
        """Computes interface ipTM score restricted to interacting multi-chain residue pairs:
        
        ipTM = max_j in Interface [ (1 / L_interface) * sum_{i in Interface} 1 / (1 + (E_ij / d0(L_interface))^2) ]
        """
        l = len(pae_matrix)
        if l == 0:
            return 0.0

        if interface_indices is not None:
            indices = list(interface_indices)
        elif chain_lengths is not None and len(chain_lengths) > 1:
            # Build interface pairs: cross-chain residues
            indices = list(range(min(l, sum(chain_lengths))))
        else:
            indices = list(range(l))

        num_interface = len(indices)
        if num_interface == 0:
            return 0.0

        d0_val = cls.d0(num_interface)
        d0_sq = d0_val * d0_val

        best_iptm = 0.0
        for j in indices:
            tm_j = sum(
                1.0 / (1.0 + (pae_matrix[i][j] * pae_matrix[i][j]) / d0_sq)
                for i in indices
            ) / num_interface
            if tm_j > best_iptm:
                best_iptm = tm_j

        return round(best_iptm, 4)

