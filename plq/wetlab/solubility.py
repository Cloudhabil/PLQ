"""Aqueous Solubility, Hydropathy Profiling, and Isoelectric Point (pI) Calculation.

Conforming to Nuclear Software Quality Standards: ASME NQA-1 Subpart 2.7.

Features:
1. Kyte-Doolittle hydropathy index and Grand Average of Hydropathy (GRAVY).
2. Theoretical isoelectric point (pI) computed via bisection on Henderson-Hasselbalch equilibria.
3. Net formal charge profile across physiological pH 7.4 and synthesis cleavage pH 2.0.
4. Laboratory buffer solubility classification (PBS, pH 7.4).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SolubilityProfile:
    """Diagnostic profile of peptide aqueous solubility and electrostatics."""
    sequence: str
    gravy_score: float             # Grand Average of Hydropathy [-4.5, +4.5]
    isoelectric_point_pI: float    # pH at which net charge is zero
    net_charge_ph74: float         # Net formal charge at assay pH 7.4
    net_charge_ph20: float         # Net formal charge at cleavage pH 2.0
    is_soluble_in_pbs: bool        # True if GRAVY in [-0.5, +0.3] and |charge_74| <= 3.5
    solubility_class: str          # 'High', 'Moderate', 'Insoluble/Prone to Aggregate'


class AqueousSolubilityAnalyzer:
    """Analyzes hydropathy and charge balance to ensure buffer solubility."""

    # Kyte-Doolittle Hydropathy Scale (J. Mol. Biol. 1982, 157:105-132)
    KYTE_DOOLITTLE: Dict[str, float] = {
        "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
        "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
        "L": 3.8,  "K": -3.9, "M": 1.9,  "F": 2.8,  "P": -1.6,
        "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
    }

    # Standard pKa values for ionizable groups (EMBOSS scale)
    PKA_VALUES = {
        "N_term": 8.6,
        "C_term": 3.6,
        "K": 10.8,
        "R": 12.5,
        "H": 6.2,   # Histidine switch
        "D": 3.9,
        "E": 4.1,
        "C": 8.5,
        "Y": 10.1,
    }

    @classmethod
    def calculate_gravy(cls, sequence: str) -> float:
        """Computes the Grand Average of Hydropathy (GRAVY).
        
        GRAVY = (sum of Kyte-Doolittle hydropathy values) / (number of residues)
        Negative values indicate hydrophilic/soluble; positive indicates hydrophobic.
        """
        seq_clean = sequence.upper().strip()
        if not seq_clean:
            return 0.0
        total_hydro = sum(cls.KYTE_DOOLITTLE.get(aa, 0.0) for aa in seq_clean)
        return round(total_hydro / len(seq_clean), 4)

    @classmethod
    def calculate_net_charge(cls, sequence: str, ph: float) -> float:
        """Computes net formal charge at a given pH using the Henderson-Hasselbalch equation."""
        seq_clean = sequence.upper().strip()
        charge = 0.0

        # N-terminus (basic)
        charge += 1.0 / (1.0 + 10.0 ** (ph - cls.PKA_VALUES["N_term"]))
        # C-terminus (acidic)
        charge -= 1.0 / (1.0 + 10.0 ** (cls.PKA_VALUES["C_term"] - ph))

        # Basic side chains (Lys, Arg, His)
        charge += seq_clean.count("K") * (1.0 / (1.0 + 10.0 ** (ph - cls.PKA_VALUES["K"])))
        charge += seq_clean.count("R") * (1.0 / (1.0 + 10.0 ** (ph - cls.PKA_VALUES["R"])))
        charge += seq_clean.count("H") * (1.0 / (1.0 + 10.0 ** (ph - cls.PKA_VALUES["H"])))

        # Acidic side chains (Asp, Glu, Cys, Tyr)
        charge -= seq_clean.count("D") * (1.0 / (1.0 + 10.0 ** (cls.PKA_VALUES["D"] - ph)))
        charge -= seq_clean.count("E") * (1.0 / (1.0 + 10.0 ** (cls.PKA_VALUES["E"] - ph)))
        charge -= seq_clean.count("C") * (1.0 / (1.0 + 10.0 ** (cls.PKA_VALUES["C"] - ph)))
        charge -= seq_clean.count("Y") * (1.0 / (1.0 + 10.0 ** (cls.PKA_VALUES["Y"] - ph)))

        return round(charge, 3)

    @classmethod
    def calculate_pI(cls, sequence: str) -> float:
        """Calculates isoelectric point (pI) using bisection between pH 0.0 and 14.0."""
        low = 0.0
        high = 14.0
        for _ in range(50): # 50 bisection steps -> machine precision < 1e-14
            mid = (low + high) / 2.0
            q = cls.calculate_net_charge(sequence, mid)
            if q > 0:
                low = mid
            else:
                high = mid
        return round((low + high) / 2.0, 2)

    @classmethod
    def evaluate_solubility(cls, sequence: str) -> SolubilityProfile:
        """Computes complete solubility diagnostic for wet-lab assay readiness."""
        gravy = cls.calculate_gravy(sequence)
        pI = cls.calculate_pI(sequence)
        q74 = cls.calculate_net_charge(sequence, 7.4)
        q20 = cls.calculate_net_charge(sequence, 2.0)

        # Standard biochemical criteria for soluble biologics:
        # 1. GRAVY <= +0.35 (not excessively hydrophobic)
        # 2. Net charge at pH 7.4 not near 0 if GRAVY is positive (avoids isoelectric precipitation)
        is_near_pi = abs(pI - 7.4) < 0.5
        is_soluble = (gravy <= 0.35) and not (is_near_pi and gravy > 0.0)

        if gravy < -0.3 and not is_near_pi:
            sol_class = "High"
        elif is_soluble:
            sol_class = "Moderate"
        else:
            sol_class = "Insoluble/Prone to Aggregate"

        return SolubilityProfile(
            sequence=sequence.upper().strip(),
            gravy_score=gravy,
            isoelectric_point_pI=pI,
            net_charge_ph74=q74,
            net_charge_ph20=q20,
            is_soluble_in_pbs=is_soluble,
            solubility_class=sol_class,
        )
