"""Physico-Chemical Properties & Exact Numerical Root-Finding Suite.

Implements:
1. Exact Molecular Weight (Mw):
       Mw = sum_{i=1}^N M(a_i) - (N - 1) * 18.0152 Da
       Accounting for isotopic residue weights and condensation water loss.
2. Continuous Net Charge Q(pH):
       Q(pH) = 1 / (1 + 10^(pH - pKa_Nterm)) - 1 / (1 + 10^(pKa_Cterm - pH))
             + sum_{pos} 1 / (1 + 10^(pH - pKa_pos))
             - sum_{neg} 1 / (1 + 10^(pKa_neg - pH))
3. Exact Isoelectric Point (pI) via Newton-Raphson:
       pH^{(t+1)} = pH^{(t)} - Q(pH^{(t)}) / Q'(pH^{(t)})
       with analytical negative derivative Q'(pH) < 0 ensuring quadratic convergence.
4. Extinction Coefficient & GRAVY Hydropathy Index.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from plq.metrics.mpnn_scoring import HYDROPATHY_INDEX


# Standard average isotopic molecular weights of free amino acids (g/mol or Da)
AA_AVG_MASS: Dict[str, float] = {
    "A": 89.094, "R": 174.203, "N": 132.119, "D": 133.104, "C": 121.154,
    "E": 147.131, "Q": 146.146, "G": 75.067, "H": 155.156, "I": 131.175,
    "L": 131.175, "K": 146.189, "M": 149.208, "F": 165.192, "P": 115.132,
    "S": 105.093, "T": 119.119, "W": 204.228, "Y": 181.191, "V": 117.148,
}

# Monoisotopic molecular weights of free amino acids (using most abundant isotopes)
AA_MONO_MASS: Dict[str, float] = {
    "A": 89.04768, "R": 174.11168, "N": 132.05349, "D": 133.03751, "C": 121.01975,
    "E": 147.05316, "Q": 146.06914, "G": 75.03203, "H": 155.06948, "I": 131.09463,
    "L": 131.09463, "K": 146.10553, "M": 149.05105, "F": 165.07898, "P": 115.06333,
    "S": 105.04259, "T": 119.05824, "W": 204.08988, "Y": 181.07390, "V": 117.07898,
}

# Water condensation mass loss
WATER_AVG_MASS = 18.01524
WATER_MONO_MASS = 18.010565

# Standard pKa values (Lehninger consensus scale)
DEFAULT_PKA_SET: Dict[str, float] = {
    "N_term": 9.69,
    "C_term": 2.34,
    "K": 10.50,  # Lysine (positive)
    "R": 12.50,  # Arginine (positive)
    "H": 6.04,   # Histidine (positive)
    "D": 3.86,   # Aspartic acid (negative)
    "E": 4.25,   # Glutamic acid (negative)
    "C": 8.33,   # Cysteine (negative)
    "Y": 10.00,  # Tyrosine (negative)
}


@dataclass
class PhysicoChemicalReport:
    """Report of exact physico-chemical properties of a peptide/protein sequence."""
    sequence: str
    length: int
    molecular_weight: float
    monoisotopic_mw: float
    isoelectric_point: float
    charge_at_ph_7_4: float
    charge_at_ph_7_0: float
    net_charge_at_pi: float
    extinction_coefficient: float
    gravy: float


class PhysicoChemicalEngine:
    """Computes exact isotopic molecular weight and continuous isoelectric point via Newton-Raphson."""

    # -------------------------------------------------------------------------
    # 1. Molecular Weight
    # -------------------------------------------------------------------------

    @classmethod
    def compute_molecular_weight(
        cls,
        sequence: str,
        monoisotopic: bool = False,
    ) -> float:
        """Computes exact peptide molecular weight minus (N - 1) * H2O condensation loss.

        Mw = sum_{i=1}^N M(a_i) - (N - 1) * 18.0152 Da
        """
        seq = sequence.strip().upper()
        n = len(seq)
        if n == 0:
            return 0.0

        mass_table = AA_MONO_MASS if monoisotopic else AA_AVG_MASS
        water_loss = WATER_MONO_MASS if monoisotopic else WATER_AVG_MASS

        total_mass = sum(mass_table.get(aa, 110.0) for aa in seq)
        # Subtract condensation water loss for N-1 peptide bonds
        if n > 1:
            total_mass -= (n - 1) * water_loss

        return round(total_mass, 4)

    # -------------------------------------------------------------------------
    # 2. Continuous Net Charge Q(pH) & Analytical Derivative Q'(pH)
    # -------------------------------------------------------------------------

    @classmethod
    def compute_charge_and_derivative(
        cls,
        sequence: str,
        ph: float,
        pka_set: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float]:
        """Calculates total net charge Q(pH) and analytical derivative dQ/dpH.

        Analytical derivative:
        dQ/dpH = - ln(10) * [ sum_{pos} 10^(pH - pKa) / (1 + 10^(pH - pKa))^2
                            + sum_{neg} 10^(pKa - pH) / (1 + 10^(pKa - pH))^2 ] < 0
        """
        seq = sequence.strip().upper()
        if not seq:
            return 0.0, 0.0

        pka = pka_set or DEFAULT_PKA_SET
        ln10 = math.log(10.0)

        q_total = 0.0
        dq_total = 0.0

        # N-terminus (positive at acidic pH)
        pka_n = pka.get("N_term", 9.69)
        diff_n = ph - pka_n
        if diff_n > 30.0:
            frac_n = 0.0
            dfrac_n = 0.0
        elif diff_n < -30.0:
            frac_n = 1.0
            dfrac_n = 0.0
        else:
            pow_n = 10.0 ** diff_n
            denom_n = 1.0 + pow_n
            frac_n = 1.0 / denom_n
            dfrac_n = -ln10 * pow_n / (denom_n * denom_n)
        q_total += frac_n
        dq_total += dfrac_n

        # C-terminus (negative at basic pH)
        pka_c = pka.get("C_term", 2.34)
        diff_c = pka_c - ph
        if diff_c > 30.0:
            frac_c = 0.0
            dfrac_c = 0.0
        elif diff_c < -30.0:
            frac_c = 1.0
            dfrac_c = 0.0
        else:
            pow_c = 10.0 ** diff_c
            denom_c = 1.0 + pow_c
            frac_c = 1.0 / denom_c
            dfrac_c = -ln10 * pow_c / (denom_c * denom_c)
        q_total -= frac_c
        dq_total += dfrac_c

        # Count ionizable residues
        counts: Dict[str, int] = {}
        for aa in seq:
            counts[aa] = counts.get(aa, 0) + 1

        # Positive side chains: K, R, H
        for aa, pka_key in (("K", "K"), ("R", "R"), ("H", "H")):
            cnt = counts.get(aa, 0)
            if cnt > 0 and pka_key in pka:
                pka_val = pka[pka_key]
                diff = ph - pka_val
                if diff > 30.0:
                    frac = 0.0
                    dfrac = 0.0
                elif diff < -30.0:
                    frac = 1.0
                    dfrac = 0.0
                else:
                    pow_val = 10.0 ** diff
                    denom = 1.0 + pow_val
                    frac = 1.0 / denom
                    dfrac = -ln10 * pow_val / (denom * denom)
                q_total += cnt * frac
                dq_total += cnt * dfrac

        # Negative side chains: D, E, C, Y
        for aa, pka_key in (("D", "D"), ("E", "E"), ("C", "C"), ("Y", "Y")):
            cnt = counts.get(aa, 0)
            if cnt > 0 and pka_key in pka:
                pka_val = pka[pka_key]
                diff = pka_val - ph
                if diff > 30.0:
                    frac = 0.0
                    dfrac = 0.0
                elif diff < -30.0:
                    frac = 1.0
                    dfrac = 0.0
                else:
                    pow_val = 10.0 ** diff
                    denom = 1.0 + pow_val
                    frac = 1.0 / denom
                    dfrac = -ln10 * pow_val / (denom * denom)
                q_total -= cnt * frac
                dq_total += cnt * dfrac

        return q_total, dq_total

    @classmethod
    def compute_charge_at_ph(
        cls,
        sequence: str,
        ph: float,
        pka_set: Optional[Dict[str, float]] = None,
    ) -> float:
        """Returns the net charge of the peptide at a specific pH."""
        q, _ = cls.compute_charge_and_derivative(sequence, ph, pka_set)
        return round(q, 4)

    # -------------------------------------------------------------------------
    # 3. Exact Isoelectric Point (pI) via Newton-Raphson
    # -------------------------------------------------------------------------

    @classmethod
    def compute_isoelectric_point_newton(
        cls,
        sequence: str,
        max_iter: int = 100,
        tol: float = 1e-5,
        pka_set: Optional[Dict[str, float]] = None,
    ) -> float:
        """Solves Q(pH) = 0 via Newton-Raphson iteration with guaranteed quadratic convergence.

        pH^{(t+1)} = pH^{(t)} - Q(pH^{(t)}) / Q'(pH^{(t)})
        """
        seq = sequence.strip().upper()
        if not seq:
            return 7.0

        # Start search at physiological neutral pH 7.0
        ph = 7.0

        for _ in range(max_iter):
            q, dq = cls.compute_charge_and_derivative(seq, ph, pka_set)

            if abs(q) < tol:
                break

            if abs(dq) < 1e-12:
                # Derivative flattened (extreme pH); clamp step
                step = -math.copysign(0.5, q)
            else:
                step = -q / dq

            # Damped step to prevent overshooting outside physical range [0, 14]
            if abs(step) > 2.0:
                step = math.copysign(2.0, step)

            ph_next = ph + step

            # Clamp within [0.0, 14.0]
            if ph_next < 0.0:
                ph_next = 0.0
            elif ph_next > 14.0:
                ph_next = 14.0

            if abs(ph_next - ph) < tol:
                ph = ph_next
                break

            ph = ph_next

        return round(ph, 3)

    # -------------------------------------------------------------------------
    # 4. Extinction Coefficient & GRAVY
    # -------------------------------------------------------------------------

    @classmethod
    def compute_extinction_coefficient(cls, sequence: str) -> float:
        """Computes molar extinction coefficient at 280 nm in water (M^-1 cm^-1).

        epsilon = 5500 * n_Trp + 1490 * n_Tyr + 125 * (n_Cys / 2)
        """
        seq = sequence.strip().upper()
        nw = seq.count("W")
        ny = seq.count("Y")
        nc = seq.count("C")
        eps = (5500 * nw) + (1490 * ny) + (125 * (nc // 2))
        return float(eps)

    @classmethod
    def compute_gravy(cls, sequence: str) -> float:
        """Grand Average of Hydropathicity (GRAVY) based on Kyte-Doolittle scale."""
        seq = sequence.strip().upper()
        if not seq:
            return 0.0
        tot = sum(HYDROPATHY_INDEX.get(aa, -0.4) for aa in seq)
        return round(tot / len(seq), 3)

    # -------------------------------------------------------------------------
    # Comprehensive Sequence Evaluation
    # -------------------------------------------------------------------------

    @classmethod
    def evaluate_sequence(
        cls,
        sequence: str,
        pka_set: Optional[Dict[str, float]] = None,
    ) -> PhysicoChemicalReport:
        """Evaluates complete physico-chemical report for a given amino acid sequence."""
        seq = sequence.strip().upper()
        mw = cls.compute_molecular_weight(seq, monoisotopic=False)
        mono_mw = cls.compute_molecular_weight(seq, monoisotopic=True)
        pi = cls.compute_isoelectric_point_newton(seq, pka_set=pka_set)
        q_74 = cls.compute_charge_at_ph(seq, 7.4, pka_set=pka_set)
        q_70 = cls.compute_charge_at_ph(seq, 7.0, pka_set=pka_set)
        q_pi = cls.compute_charge_at_ph(seq, pi, pka_set=pka_set)
        eps = cls.compute_extinction_coefficient(seq)
        gravy = cls.compute_gravy(seq)

        return PhysicoChemicalReport(
            sequence=seq,
            length=len(seq),
            molecular_weight=mw,
            monoisotopic_mw=mono_mw,
            isoelectric_point=pi,
            charge_at_ph_7_4=q_74,
            charge_at_ph_7_0=q_70,
            net_charge_at_pi=q_pi,
            extinction_coefficient=eps,
            gravy=gravy,
        )
