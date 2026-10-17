"""PLQ Ternary Lattice Quantization and State Matrix Representation.

Encodes electrostatic and pharmacophoric profiles into a ternary state space
T in {-1, 0, +1} (Protonation donor, Neutral/hydrophobic, Anionic acceptor).
"""

from __future__ import annotations
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


class TernaryState(IntEnum):
    """PLQ Ternary state code for amino acid residues and pharmacophores."""
    ACCEPTOR = -1  # Anionic acceptor anchor: Asp (-), Glu (-)
    NEUTRAL = 0    # Hydrophobic / neutral packing: Leu, Val, Ala, Ile, Phe, Trp, Pro, Gly
    DONOR = +1     # Protonation-driven donor: His (at pH 6.5), Lys (+), Arg (+)


class TernaryLattice:
    """Manages ternary state matrices, Lucas substates, and discrete energy matrices."""

    # Interaction energy table (kcal/mol) between ternary states
    # (Donor +1, Acceptor -1) -> Strongly attractive salt-bridge (-2.5 kcal/mol)
    # (Donor +1, Donor +1) -> Repulsive (+2.0 kcal/mol)
    # (Acceptor -1, Acceptor -1) -> Repulsive (+2.0 kcal/mol)
    # (Neutral 0, Neutral 0) -> Favorable hydrophobic packing (-0.8 kcal/mol)
    # (Neutral 0, Charged +/-1) -> Desolvation / uncompensated charge penalty (+0.4 kcal/mol)
    ENERGY_LOOKUP: Dict[Tuple[int, int], float] = {
        (TernaryState.DONOR, TernaryState.ACCEPTOR): -2.5,
        (TernaryState.ACCEPTOR, TernaryState.DONOR): -2.5,
        (TernaryState.DONOR, TernaryState.DONOR): +2.0,
        (TernaryState.ACCEPTOR, TernaryState.ACCEPTOR): +2.0,
        (TernaryState.NEUTRAL, TernaryState.NEUTRAL): -0.8,
        (TernaryState.NEUTRAL, TernaryState.DONOR): +0.4,
        (TernaryState.DONOR, TernaryState.NEUTRAL): +0.4,
        (TernaryState.NEUTRAL, TernaryState.ACCEPTOR): +0.4,
        (TernaryState.ACCEPTOR, TernaryState.NEUTRAL): +0.4,
    }

    @classmethod
    def assign_residue_state(cls, aa: str, ph: float = 7.4, his_pka: float = 6.2) -> int:
        """Assigns ternary state T in {-1, 0, +1} to an amino acid at given pH."""
        aa_upper = aa.upper()
        if aa_upper in ("D", "E"):
            return TernaryState.ACCEPTOR
        elif aa_upper in ("K", "R"):
            return TernaryState.DONOR
        elif aa_upper == "H":
            # pH-conditional switch: protonated positive at pH < pKa
            return TernaryState.DONOR if ph < his_pka else TernaryState.NEUTRAL
        else:
            return TernaryState.NEUTRAL

    @classmethod
    def sequence_to_ternary(cls, sequence: str, ph: float = 7.4) -> List[int]:
        """Maps a sequence string to a ternary state vector."""
        return [cls.assign_residue_state(aa, ph=ph) for aa in sequence]

    @classmethod
    def pair_energy(cls, state_a: int, state_b: int) -> float:
        """Returns the discrete pair interaction energy in kcal/mol."""
        return cls.ENERGY_LOOKUP.get((state_a, state_b), 0.0)

    @classmethod
    def evaluate_interface(
        cls, binder_states: List[int], pocket_states: List[int], contact_pairs: List[Tuple[int, int]]
    ) -> float:
        """Evaluates total ternary lattice binding score over interface contact pairs."""
        total_e = 0.0
        for b_idx, p_idx in contact_pairs:
            if b_idx < len(binder_states) and p_idx < len(pocket_states):
                total_e += cls.pair_energy(binder_states[b_idx], pocket_states[p_idx])
        return round(total_e, 4)
