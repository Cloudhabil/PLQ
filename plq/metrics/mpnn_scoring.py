"""ProteinMPNN Sequence Design & Compatibility Scoring Suite.

Implements:
1. ProteinMPNN Negative Log-Likelihood (NLL) Score:
       Score_MPNN = - (1 / N) * sum_{i=1}^N log P_theta(s_i | X, S_{<i})
2. Redesigned Score restricted to mutable paratope/CDR positions M:
       Score_redesigned = - (1 / |M|) * sum_{i in M} log P_theta(s_i | X, S_{<i})
3. Sequence Recovery Percentage against wild-type / native template:
       Seq_Recovery = (1 / N) * sum_{i=1}^N I(s_i^designed == s_i^native) * 100%
4. Geometric Backbone Compatibility Engine:
       Computes pseudo-log-probabilities from 3D backbone coordinates and DACC
       topological environment (Ramachandran dihedrals and contact densities).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Tuple, Union


# Standard 20 canonical amino acids
CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"

# Kyte-Doolittle hydropathy scale
HYDROPATHY_INDEX: Dict[str, float] = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}

# Preferred secondary structure / dihedral propensities
# (Helix favorability vs Sheet favorability, Chou-Fasman derived)
HELIX_PROPENSITY: Dict[str, float] = {
    "E": 1.51, "M": 1.45, "A": 1.42, "L": 1.21, "K": 1.16, "F": 1.13, "Q": 1.11,
    "I": 1.08, "W": 1.08, "V": 1.06, "D": 1.01, "H": 1.00, "R": 0.98, "T": 0.83,
    "S": 0.77, "C": 0.70, "Y": 0.69, "N": 0.67, "P": 0.57, "G": 0.57,
}

SHEET_PROPENSITY: Dict[str, float] = {
    "V": 1.70, "I": 1.60, "Y": 1.47, "F": 1.38, "W": 1.37, "L": 1.30, "C": 1.19,
    "T": 1.19, "Q": 1.10, "M": 1.05, "R": 0.93, "N": 0.89, "H": 0.87, "A": 0.83,
    "S": 0.75, "G": 0.75, "K": 0.74, "P": 0.55, "D": 0.54, "E": 0.37,
}


@dataclass
class MPNNScoreReport:
    """Report of ProteinMPNN sequence compatibility and recovery metrics."""
    nll_score: float
    redesigned_score: Optional[float]
    sequence_recovery: Optional[float]
    per_position_nll: List[float]
    sequence_length: int
    mean_perplexity: float


class ProteinMPNNScorer:
    """Evaluates sequence log-likelihood, paratope redesigned score, and sequence recovery."""

    @staticmethod
    def _extract_log_prob(
        pos_entry: Union[Mapping[str, float], Sequence[float]],
        aa: str,
        aa_order: Sequence[str] = CANONICAL_AA,
    ) -> float:
        """Safely extracts log P(aa) from either a dictionary or probability array."""
        if isinstance(pos_entry, Mapping):
            if aa in pos_entry:
                val = float(pos_entry[aa])
            else:
                val = -15.0  # Safe floor for unobserved amino acid
        else:
            try:
                idx = aa_order.index(aa)
                val = float(pos_entry[idx])
            except (ValueError, IndexError):
                val = -15.0
        # Guard against -inf or NaN
        if math.isinf(val) and val < 0:
            return -15.0
        if math.isnan(val):
            return -15.0
        return val

    @classmethod
    def compute_nll_score(
        cls,
        sequence: str,
        log_probs: Sequence[Union[Mapping[str, float], Sequence[float]]],
        aa_order: Sequence[str] = CANONICAL_AA,
    ) -> float:
        """Computes global average Negative Log-Likelihood (NLL).

        Score_MPNN = - (1 / N) * sum_{i=1}^N log P(s_i)
        """
        n = min(len(sequence), len(log_probs))
        if n == 0:
            return 0.0

        total_nll = 0.0
        for i in range(n):
            lp = cls._extract_log_prob(log_probs[i], sequence[i], aa_order)
            total_nll += -lp

        return round(total_nll / n, 4)

    @classmethod
    def compute_redesigned_score(
        cls,
        sequence: str,
        log_probs: Sequence[Union[Mapping[str, float], Sequence[float]]],
        redesigned_indices: Collection[int],
        aa_order: Sequence[str] = CANONICAL_AA,
    ) -> float:
        """Computes NLL score strictly over mutable paratope / redesigned indices M.

        Score_redesigned = - (1 / |M|) * sum_{i in M} log P(s_i)
        """
        valid_indices = [
            i for i in redesigned_indices if 0 <= i < min(len(sequence), len(log_probs))
        ]
        if not valid_indices:
            return 0.0

        total_nll = 0.0
        for i in valid_indices:
            lp = cls._extract_log_prob(log_probs[i], sequence[i], aa_order)
            total_nll += -lp

        return round(total_nll / len(valid_indices), 4)

    @classmethod
    def compute_sequence_recovery(
        cls,
        designed_seq: str,
        native_seq: str,
        mask_indices: Optional[Collection[int]] = None,
    ) -> float:
        """Measures percentage sequence identity match relative to native template.

        Seq Recovery = (1 / N) * sum_{i=1}^N I(s_i^designed == s_i^native) * 100%
        """
        n = min(len(designed_seq), len(native_seq))
        if n == 0:
            return 0.0

        if mask_indices is not None:
            indices = [i for i in mask_indices if 0 <= i < n]
            if not indices:
                return 0.0
        else:
            indices = list(range(n))

        matches = sum(1 for i in indices if designed_seq[i].upper() == native_seq[i].upper())
        return round((matches / len(indices)) * 100.0, 2)

    @classmethod
    def score_sequence(
        cls,
        sequence: str,
        log_probs: Sequence[Union[Mapping[str, float], Sequence[float]]],
        redesigned_indices: Optional[Collection[int]] = None,
        native_sequence: Optional[str] = None,
        aa_order: Sequence[str] = CANONICAL_AA,
    ) -> MPNNScoreReport:
        """Generates a complete MPNNScoreReport from sequence and log-probabilities."""
        n = min(len(sequence), len(log_probs))
        if n == 0:
            return MPNNScoreReport(
                nll_score=0.0,
                redesigned_score=None,
                sequence_recovery=None,
                per_position_nll=[],
                sequence_length=0,
                mean_perplexity=1.0,
            )

        per_pos: List[float] = []
        for i in range(n):
            lp = cls._extract_log_prob(log_probs[i], sequence[i], aa_order)
            per_pos.append(round(-lp, 4))

        mean_nll = sum(per_pos) / n
        perplexity = math.exp(min(20.0, mean_nll))

        redesigned_score = None
        if redesigned_indices is not None:
            redesigned_score = cls.compute_redesigned_score(
                sequence, log_probs, redesigned_indices, aa_order
            )

        recovery = None
        if native_sequence is not None:
            recovery = cls.compute_sequence_recovery(
                sequence, native_sequence, redesigned_indices
            )

        return MPNNScoreReport(
            nll_score=round(mean_nll, 4),
            redesigned_score=redesigned_score,
            sequence_recovery=recovery,
            per_position_nll=per_pos,
            sequence_length=n,
            mean_perplexity=round(perplexity, 3),
        )

    # -------------------------------------------------------------------------
    # Geometric Backbone Compatibility (Pseudo-NLL Estimation)
    # -------------------------------------------------------------------------

    @classmethod
    def estimate_backbone_compatibility(
        cls,
        sequence: str,
        dacc_or_coords: Any,
        redesigned_indices: Optional[Collection[int]] = None,
        native_sequence: Optional[str] = None,
    ) -> MPNNScoreReport:
        """Estimates pseudo-MPNN log-probabilities from 3D coordinates or DACC complex.

        Evaluates environment compatibility:
        - Contact density (core vs surface burial) matched to residue hydropathy.
        - Dihedral angles (phi, psi) matched to helix/sheet/turn propensities.
        - Softmax temperature T=1.0 normalized over all 20 canonical amino acids.
        """
        n = len(sequence)
        coords: List[Tuple[float, float, float]] = []
        dihedrals: List[Tuple[float, float]] = []

        # Extract coordinates and dihedral angles from DACC or list of coords
        if hasattr(dacc_or_coords, "get_skeleton"):
            k0 = dacc_or_coords.get_skeleton(0)
            for cell_id, cell in k0.items():
                c = getattr(cell, "coords", None)
                if c is not None and len(c) >= 3:
                    coords.append((float(c[0]), float(c[1]), float(c[2])))
                    phi = float(c[3]) if len(c) > 3 else -1.0
                    psi = float(c[4]) if len(c) > 4 else -0.8
                    dihedrals.append((phi, psi))
                else:
                    coords.append((0.0, 0.0, 0.0))
                    dihedrals.append((-1.0, -0.8))
        elif isinstance(dacc_or_coords, Sequence):
            for item in dacc_or_coords:
                if isinstance(item, (tuple, list)) and len(item) >= 3:
                    coords.append((float(item[0]), float(item[1]), float(item[2])))
                    phi = float(item[3]) if len(item) > 3 else -1.0
                    psi = float(item[4]) if len(item) > 4 else -0.8
                    dihedrals.append((phi, psi))
                else:
                    coords.append((0.0, 0.0, 0.0))
                    dihedrals.append((-1.0, -0.8))

        # Pad or truncate to sequence length
        if len(coords) < n:
            last = coords[-1] if coords else (0.0, 0.0, 0.0)
            for i in range(len(coords), n):
                coords.append((last[0] + (i * 3.8), last[1], last[2]))
                dihedrals.append((-1.0, -0.8))
        coords = coords[:n]
        dihedrals = dihedrals[:n]

        # Compute neighbor contacts within 8.0 A
        contact_counts: List[int] = []
        for i in range(n):
            xi, yi, zi = coords[i]
            cnt = 0
            for j in range(n):
                if i != j:
                    xj, yj, zj = coords[j]
                    d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
                    if d <= 8.0:
                        cnt += 1
            contact_counts.append(cnt)

        # Generate pseudo log-probabilities for each position
        synthetic_log_probs: List[Dict[str, float]] = []
        for i in range(n):
            contacts = contact_counts[i]
            # Normalized burial score: 0 = fully exposed, 1 = buried core
            burial = min(1.0, contacts / 14.0)
            phi, psi = dihedrals[i]
            is_helical = (-1.6 < phi < -0.6) and (-1.2 < psi < -0.2)

            energies: Dict[str, float] = {}
            for aa in CANONICAL_AA:
                # Hydropathy match energy
                hyd = HYDROPATHY_INDEX.get(aa, 0.0)
                # Buried residues favor high hydropathy; exposed favor low
                target_hyd = (burial - 0.4) * 8.0
                e_burial = 0.15 * (hyd - target_hyd) ** 2

                # Secondary structure match energy
                if is_helical:
                    e_ss = -0.8 * math.log(max(0.2, HELIX_PROPENSITY.get(aa, 1.0)))
                else:
                    e_ss = -0.8 * math.log(max(0.2, SHEET_PROPENSITY.get(aa, 1.0)))

                energies[aa] = -(e_burial + e_ss)

            # Softmax to log-probabilities
            max_e = max(energies.values())
            sum_exp = sum(math.exp(e - max_e) for e in energies.values())
            log_sum_exp = max_e + math.log(sum_exp)

            pos_lps: Dict[str, float] = {
                aa: round(energies[aa] - log_sum_exp, 4) for aa in CANONICAL_AA
            }
            synthetic_log_probs.append(pos_lps)

        return cls.score_sequence(
            sequence=sequence,
            log_probs=synthetic_log_probs,
            redesigned_indices=redesigned_indices,
            native_sequence=native_sequence,
        )
