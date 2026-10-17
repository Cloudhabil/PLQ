"""Histidine Protonation Switch Network & Bifold Protonation State Manifold.

Models the pH-conditional conformational and electrostatic transition between
acidic tumor microenvironments (pH 6.5, ON) and physiological serum (pH 7.4, OFF).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from plq.gnc.clifford import Multivector, CGA3D
from plq.ternary.lattice import TernaryLattice, TernaryState


@dataclass
class BifoldEvaluation:
    """Evaluation results across the bifold protonation state manifold."""
    delta_g_65: float      # Free energy of binding at pH 6.5 (kcal/mol)
    delta_g_74: float      # Free energy of binding at pH 7.4 (kcal/mol)
    delta_delta_g: float   # Switch delta: Delta G(7.4) - Delta G(6.5) (kcal/mol)
    his_protonation_65: float
    his_protonation_74: float
    is_switch_active: bool # True if delta_delta_g >= threshold and delta_g_65 < 0


class HistidineSwitchNetwork:
    """Governs the Histidine Protonation Switch Network at the binder-target interface."""

    def __init__(self, pka: float = 6.2, switch_threshold_ddg: float = 3.5) -> None:
        self.pka = pka
        self.switch_threshold_ddg = switch_threshold_ddg

    def compute_microenvironment_pka(
        self,
        local_anionic_contacts: int = 0,
        mean_contact_distance: float = 4.0,
    ) -> float:
        """Computes site-specific effective pKa shifted by local electrostatic environment.
        
        Delta pKa = sum_j (q_j / (epsilon * r_ij))
        Anionic carboxylates (Asp/Glu) within 3.5-5.0 A shift His pKa upward, stabilizing His+.
        """
        if local_anionic_contacts <= 0 or mean_contact_distance <= 0:
            return self.pka
        delta_pka = (local_anionic_contacts * 1.8) / max(2.5, mean_contact_distance)
        effective_pka = min(8.2, self.pka + delta_pka)
        return round(effective_pka, 2)

    def protonation_fraction(self, ph: float, effective_pka: Optional[float] = None) -> float:
        """Computes fraction of protonated positively charged His+ via Henderson-Hasselbalch:
        
        f(His+) = 1 / (1 + 10^(pH - pKa_eff))
        """
        target_pka = effective_pka if effective_pka is not None else self.pka
        exponent = ph - target_pka
        # Clamp exponent to prevent overflow
        exponent = max(-10.0, min(10.0, exponent))
        return 1.0 / (1.0 + 10.0 ** exponent)

    def evaluate_bifold_manifold(
        self,
        binder_seq: str,
        pocket_anionic_count: int = 3,
        pocket_hydrophobic_count: int = 4,
        cga_alignment_bonus: float = 1.0,
        interface_residues: Optional[str] = None,
    ) -> BifoldEvaluation:
        """Evaluates binding free energy across dual pH states (pH 6.5 and pH 7.4).
        
        At pH 6.5 (Tumor Microenvironment: ON):
        - Histidine residues in the paratope are protonated (His+, ternary code +1).
        - Direct salt-bridge / hydrogen-bond network with pocket Asp355/Glu stabilizes binding.
        - CGA wedge product S_binder ^ S_pocket vanishes, driving Delta G_bind << 0.
        
        At pH 7.4 (Healthy Serum: OFF):
        - Histidines deprotonate to neutral His0 (ternary code 0).
        - Salt-bridge network collapses. Buried neutral His facing unpartnered anionic pocket
          anchors incurs a heavy desolvation penalty and steric clash, driving Delta G_bind >= 0.
        """
        f65 = self.protonation_fraction(6.5)
        f74 = self.protonation_fraction(7.4)

        # Focus on interface contact residues (if not explicitly given, extract from sequence)
        target_seq = interface_residues if interface_residues is not None else binder_seq

        # Interface histidine count
        num_his = target_seq.count("H")
        # Interface hydrophobic packing residues
        num_hydro = sum(target_seq.count(aa) for aa in "LVIFAPWM")
        effective_hydro = min(float(num_hydro), float(pocket_hydrophobic_count * 2))

        # Baseline hydrophobic contribution
        e_hydrophobic = -0.5 * effective_hydro

        # At pH 6.5 (His+):
        # Protonated histidines form salt-bridges with pocket anionic anchors
        his_donors_65 = num_his * f65
        salt_bridges_65 = min(his_donors_65, float(pocket_anionic_count))
        e_salt_bridge_65 = -3.4 * salt_bridges_65 * cga_alignment_bonus
        # Secondary electrostatic pull
        e_pull_65 = -0.6 * max(0.0, his_donors_65 - salt_bridges_65)
        dg_65 = round(e_hydrophobic + e_salt_bridge_65 + e_pull_65, 4)

        # At pH 7.4 (His0):
        # His is neutral; salt-bridge network collapses
        his_donors_74 = num_his * f74
        salt_bridges_74 = min(his_donors_74, float(pocket_anionic_count))
        e_salt_bridge_74 = -3.4 * salt_bridges_74 * cga_alignment_bonus
        
        # Desolvation penalty of buried neutral His and unpartnered pocket anionic anchors
        neutral_his = num_his * (1.0 - f74)
        unpartnered_anionic = max(0.0, float(pocket_anionic_count) - salt_bridges_74)
        e_desolvation_clash = 2.2 * min(neutral_his, unpartnered_anionic) + 1.2 * neutral_his
        dg_74 = round(e_hydrophobic + e_salt_bridge_74 + e_desolvation_clash, 4)

        ddg = round(dg_74 - dg_65, 4)
        is_active = (ddg >= self.switch_threshold_ddg) and (dg_65 <= -3.0)

        return BifoldEvaluation(
            delta_g_65=dg_65,
            delta_g_74=dg_74,
            delta_delta_g=ddg,
            his_protonation_65=round(f65, 3),
            his_protonation_74=round(f74, 3),
            is_switch_active=is_active,
        )


class BifoldManifold:
    """Manages the bifold protonation manifold across multiple candidate designs."""

    def __init__(self, his_network: Optional[HistidineSwitchNetwork] = None) -> None:
        self.network = his_network or HistidineSwitchNetwork()

    def rank_candidates(
        self,
        candidates: List[Tuple[str, str]],  # (name, sequence)
        pocket_anionic: int,
        pocket_hydrophobic: int,
    ) -> List[Tuple[str, str, BifoldEvaluation]]:
        """Ranks candidates by pH switch delta Delta Delta G descending."""
        scored = []
        for name, seq in candidates:
            eval_res = self.network.evaluate_bifold_manifold(seq, pocket_anionic, pocket_hydrophobic)
            scored.append((name, seq, eval_res))
        # Sort by Delta Delta G descending, secondary sort by Delta G(6.5) ascending (more negative = stronger)
        scored.sort(key=lambda item: (item[2].delta_delta_g, -item[2].delta_g_65), reverse=True)
        return scored
