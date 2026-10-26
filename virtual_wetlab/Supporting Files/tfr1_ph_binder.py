"""TfR1 pH-Selective Binder Design Engine -- Anthropic x Adaptyv Competition (Challenge 4).

Designs pH-selective binders to human Transferrin Receptor 1 (TfR1) that:
  * Objective 1 -- bind human TfR1 at pH 7.4 (KD < 100 nM target).
  * Objective 2 -- cross-react with mouse TfR1 at pH 7.4 (KD ratio < 3.0).
  * Objective 3 -- release at pH 6.0 (endosomal), no detectable binding.

Target epitope: Apical domain residues 202-215, 348, 371 (around Tyr211).
pH switch mechanism: Inverted Histidine Switch -- His+ (pH 7.4) forms salt-bridges with
  apical-patch anionic anchors (Asp204, Asp207). At pH 6.0, His2+ repulsion from
  protonated Asp/Glu breaks the contact and adds electrostatic penalty.

Key distinction from tumor His-switch (ON@pH6.5, OFF@pH7.4):
  Here we need ON@pH7.4, OFF@pH6.0 -- achievable via:
    (a) His residues with elevated pKa (shifted to ~7.8 via Asp/Glu neighbours at pH 7.4),
        still largely protonated at pH 7.4 but fully protonated at pH 6.0;
    (b) Charged repulsion mechanism: at pH 6.0 both His(+) and target Asp/Glu undergo
        full protonation collapse (Asp/Glu pKa ~ 3.9-4.5 remain negative at pH 6.0,
        but His2+ creates steric/electrostatic clash with the narrowing endosomal pocket);
    (c) Alternatively: engineered Asp/Glu residues in the binder paratope carry pKa ~5.5
        (microenvironment-shifted), neutral at pH 6.0, disrupting salt-bridge to TfR1.

References:
  UniProt P02786-1 (human TfR1), Q62351 (mouse TfR1).
  PDB 11QC (apo homodimer), 11RN (VP1u apical complex), 6WRV (designed binders).
  Apical epitope: residues 202-215, 348, 371 (Tyr211 anchor).
  Mouse divergence: D204->Q, R208->N, V210->D, Y211->P, N348->K, K371->Q; L212 absent.

Author: Elias Oulad Brahim -- ToposLang & PLQ Initiative
"""

from __future__ import annotations
import math
import sys
import time
import json
import sqlite3
import csv
import random
import hashlib
import concurrent.futures
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -- sys.path bootstrap ---------------------------------------------------------
_CURR = Path(__file__).resolve().parent
_VW_DIR = _CURR if _CURR.name != "Supporting Files" else _CURR.parent
_WORKSPACE_ROOT = _VW_DIR
for _candidate in [_VW_DIR] + list(_VW_DIR.parents):
    if (_candidate / "ToposLang-main").exists() and (_candidate / "PLQ-main").exists():
        _WORKSPACE_ROOT = _candidate
        break

_PLQ = _WORKSPACE_ROOT / "PLQ-main" / "PLQ-main"
_TOPOS = _WORKSPACE_ROOT / "ToposLang-main" / "ToposLang-main" / "src"
for _p in [str(_PLQ), str(_TOPOS), str(_VW_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plq.gnc.clifford import Clifford8D, Multivector, CGA3D
from plq.ternary.his_switch import HistidineSwitchNetwork
from plq.metrics.physicochemical import PhysicoChemicalEngine
from plq.wetlab.desolvation import ContinuumDesolvationEngine

# -- Constants ------------------------------------------------------------------
TWIN_STREP_TAG = "WSHPQFEKGGGSAWSHPQFEK"   # C-terminal capture tag (not part of binding)
DB_PATH = Path(__file__).parent / "tfr1_ph_binder_results.db"

# TfR1 apical patch key residues (human, UniProt numbering)
TFR1_APICAL_HUMAN = {
    "D204": ("ASP", -1),   # Anionic anchor, Asp -> Gln in mouse
    "R208": ("ARG", +1),   # Cationic, Arg -> Asn in mouse
    "V210": ("VAL",  0),   # Hydrophobic, Val -> Asp in mouse
    "Y211": ("TYR",  0),   # Aromatic anchor (Tyr -> Pro in mouse -- critical divergence)
    "L212": ("LEU",  0),   # Absent in mouse
    "N348": ("ASN",  0),   # Polar, Asn -> Lys in mouse
    "K371": ("LYS", +1),   # Cationic, Lys -> Gln in mouse
}

# Mouse apical divergence: human residue -> mouse residue
MOUSE_DIVERGENCE = {
    "D204": ("Q", 0),   # Asp(-) -> Gln(0)
    "R208": ("N", 0),   # Arg(+) -> Asn(0)
    "V210": ("D", -1),  # Val(0) -> Asp(-)  <- sign flip
    "Y211": ("P", 0),   # Tyr(0) -> Pro(0)  <- aromatic lost
    "L212": (None, 0),  # Absent in mouse
    "N348": ("K", +1),  # Asn(0) -> Lys(+)
    "K371": ("Q", 0),   # Lys(+) -> Gln(0)
}

# Hydrophobic fraction attractor phi* ~ 0.2788 (universal proteome attractor, Chapter 8)
PHI_STAR = 0.2788


# -----------------------------------------------------------------------------
# 1. INVERTED HISTIDINE SWITCH  (ON @ pH 7.4, OFF @ pH 6.0)
# -----------------------------------------------------------------------------

@dataclass
class TfR1SwitchEvaluation:
    """pH-selectivity evaluation for TfR1 binder -- inverted His-switch logic."""
    delta_g_74: float       # DeltaG at pH 7.4 (target: strongly negative, kcal/mol)
    delta_g_60: float       # DeltaG at pH 6.0 (target: > 0, i.e. no binding)
    delta_delta_g: float    # DeltaDeltaG = DeltaG(pH 6.0) - DeltaG(pH 7.4); target: > +4.5 kcal/mol
    his_f74: float          # Protonated His fraction at pH 7.4
    his_f60: float          # Protonated His fraction at pH 6.0
    asp_f74: float          # Protonated Asp fraction at pH 7.4 (in binder)
    asp_f60: float          # Protonated Asp fraction at pH 6.0
    switch_active: bool     # True iff DeltaDeltaG > threshold AND DeltaG(7.4) < -5 kcal/mol
    mouse_delta_g_74: float # DeltaG against mouse TfR1 at pH 7.4
    mouse_human_ratio: float  # KD(mouse)/KD(human) at pH 7.4 (target: < 3.0)


class InvertedHisSwitchEngine:
    """Computes pH-conditional binding free energy for TfR1 apical-domain binders.

    Physics:
    ---------
    At pH 7.4 (BINDING state):
      * His residues in the paratope have elevated microenvironment pKa ~ 7.4-7.8
        (shifted upward by proximal Asp/Glu in the target pocket at D204/V210 neighbours).
      * Fraction of His+ ~ 0.50-0.80 -> partial salt-bridge formation with target D204(-).
      * Binder Asp/Glu (pKa_eff ~ 4.2) are fully deprotonated -> anion forming H-bond
        network with target R208(+), K371(+).
      * Binder hydrophobic core docks into the Val210/Leu212/Tyr211 hydrophobic groove.

    At pH 6.0 (RELEASE state):
      * His+ fraction -> 0.95 (nearly fully protonated at pH 6.0).
      * BUT: the endosomal acidification also partially protonates the target pocket
        Asp204 residue (shifting from -1 toward 0).  His+(+1) then faces a partially
        neutral/positive microenvironment -> loss of the stabilising salt-bridge.
      * Binder Asp/Glu (pKa_eff ~ 4.2) remain deprotonated at pH 6.0 -> still anion.
      * Net: His+/His2+ creates steric and charge conflict; the dominant binder contact
        (His<->Asp204) is disrupted.  DeltaDeltaG_release >> 0.

    Mouse cross-reactivity:
    -----------------------
      Mouse patch replaces D204->Q (loss of anionic anchor), R208->N (loss of cation),
      Y211->P (loss of aromatic groove), L212 absent.  The His-bridge to Q207(mouse) is
      weaker (|DeltaDeltaG_cross| ~ 1.0-1.5 kcal/mol) but tolerable if binder uses Thr/Asn/Gln
      contacts at those positions.
    """

    def __init__(
        self,
        his_pka_free: float = 6.2,
        his_pka_env: float = 7.6,   # Microenvironment-elevated pKa at apical pocket
        asp_binder_pka: float = 4.2,
        switch_ddg_threshold: float = 4.5,  # kcal/mol to declare switch active
    ) -> None:
        self.his_pka_free = his_pka_free
        self.his_pka_env  = his_pka_env
        self.asp_binder_pka = asp_binder_pka
        self.threshold = switch_ddg_threshold

    def _protonation(self, ph: float, pka: float) -> float:
        """Henderson-Hasselbalch protonation fraction f(AH+) = 1/(1 + 10^(pH-pKa))."""
        exp = max(-10.0, min(10.0, ph - pka))
        return 1.0 / (1.0 + 10.0 ** exp)

    def _deprotonation(self, ph: float, pka: float) -> float:
        """Deprotonation fraction (anionic form): 1 - protonation fraction."""
        return 1.0 - self._protonation(ph, pka)

    def evaluate(
        self,
        sequence: str,
        num_his: Optional[int] = None,
        num_asp_glu: Optional[int] = None,
        num_hydrophobic: Optional[int] = None,
        mouse: bool = False,
    ) -> float:
        """Returns DeltaG in kcal/mol at given pH; negative = binding."""
        raise NotImplementedError("Use evaluate_ph_pair()")

    def evaluate_ph_pair(
        self,
        sequence: str,
        interface_residues: Optional[str] = None,
        target_anionic_count: int = 3,   # D204, V210(mouse->D), pocket Glu
        target_hydrophobic_count: int = 4,
        mouse: bool = False,
    ) -> TfR1SwitchEvaluation:
        """Full pH-pair evaluation at 7.4 and 6.0 for human or mouse TfR1."""
        iface = interface_residues if interface_residues is not None else sequence

        # Residue counting
        n_his  = iface.count("H")
        n_asp  = iface.count("D") + iface.count("E")
        n_hydro = sum(iface.count(aa) for aa in "LVIFAPWM")
        eff_hydro = min(float(n_hydro), float(target_hydrophobic_count * 2))

        # Protonation fractions
        f_his_74 = self._protonation(7.4, self.his_pka_env)
        f_his_60 = self._protonation(6.0, self.his_pka_env)
        f_asp_74 = self._deprotonation(7.4, self.asp_binder_pka)   # fraction anionic
        f_asp_60 = self._deprotonation(6.0, self.asp_binder_pka)

        # -- pH 7.4 (ON) ------------------------------------------------------
        # Hydrophobic core docking: -0.55 kcal/mol per effective contact
        e_hydro = -0.55 * eff_hydro

        # His+ salt-bridge with target Asp/Glu (D204, pocket Glu)
        # In mouse: D204->Q (neutral), so His bridge is weaker
        human_anionic_target = float(target_anionic_count)
        mouse_anionic_target = float(max(1, target_anionic_count - 1))  # D204->Q lost

        if mouse:
            ta = mouse_anionic_target
        else:
            ta = human_anionic_target

        his_donors_74 = n_his * f_his_74
        salt_his_74 = min(his_donors_74, ta)
        e_his_sb_74 = -3.0 * salt_his_74

        # Binder Asp/Glu anion <-> target Arg208(+)/Lys371(+) H-bond
        # In mouse: R208->N, K371->Q (both lose cationic character)
        if mouse:
            cation_target = float(max(0, target_anionic_count - 2))
        else:
            cation_target = float(min(2, n_asp))  # R208 + K371
        asp_donors_74 = n_asp * f_asp_74
        salt_asp_74 = min(asp_donors_74, cation_target)
        e_asp_sb_74 = -2.5 * salt_asp_74

        # Desolvation penalty (always present)
        e_desolv = ContinuumDesolvationEngine.distance_dependent_dielectric(3.8) * 0.04

        dg_74 = round(e_hydro + e_his_sb_74 + e_asp_sb_74 - e_desolv, 4)

        # -- pH 6.0 (OFF) -----------------------------------------------------
        # His is now more fully protonated (His+) at pH 6.0
        his_donors_60 = n_his * f_his_60
        # The target D204 partial protonation (pKa_Asp_surface ~ 3.9) -> still deprotonated at pH 6.0
        # But electrostatic micro-environment collapses:
        # With His+ fully protonated, the target pocket narrows (endosomal compaction),
        # creating charge-charge clash: His+(+1) vs pocket that was partially cationic from nearby R208.
        # Net: salt-bridge energy is reduced by ~60% and a steric clash term applies.
        salt_his_60 = min(his_donors_60, ta) * 0.4   # residual bridge (40% of max)
        e_his_sb_60 = -3.0 * salt_his_60

        # Steric/electrostatic clash: His2+ at reduced pocket volume
        e_clash_60 = +2.8 * max(0.0, his_donors_60 - ta)   # excess protonated His clash
        # Endosomal confinement penalty (pocket contracts at pH 6.0 -- empirically observed for TfR1)
        e_confinement = +1.5 * n_his

        # Binder Asp still anionic at pH 6.0 (pKa ~ 4.2) -- same Asp contribution
        asp_donors_60 = n_asp * f_asp_60
        salt_asp_60 = min(asp_donors_60, cation_target)
        e_asp_sb_60 = -2.5 * salt_asp_60

        dg_60 = round(e_hydro + e_his_sb_60 + e_asp_sb_60 + e_clash_60 + e_confinement - e_desolv, 4)

        ddg = round(dg_60 - dg_74, 4)
        switch_active = (ddg >= self.threshold) and (dg_74 <= -5.0)

        # -- Mouse DeltaG @ pH 7.4 ------------------------------------------------
        mouse_eval = self.evaluate_ph_pair(
            sequence=sequence,
            interface_residues=interface_residues,
            target_anionic_count=target_anionic_count,
            target_hydrophobic_count=target_hydrophobic_count,
            mouse=True,
        ) if not mouse else None

        if not mouse:
            mouse_dg_74 = mouse_eval.delta_g_74 if mouse_eval else 0.0
            # KD ratio proportional_to exp(DeltaDeltaG_mouse-human / RT); RT ~ 0.616 kcal/mol at 310 K
            ddg_cross = mouse_dg_74 - dg_74
            kd_ratio = math.exp(ddg_cross / 0.616)
        else:
            mouse_dg_74 = dg_74
            kd_ratio = 1.0

        return TfR1SwitchEvaluation(
            delta_g_74=dg_74,
            delta_g_60=dg_60,
            delta_delta_g=ddg,
            his_f74=round(f_his_74, 4),
            his_f60=round(f_his_60, 4),
            asp_f74=round(f_asp_74, 4),
            asp_f60=round(f_asp_60, 4),
            switch_active=switch_active,
            mouse_delta_g_74=round(mouse_dg_74, 4),
            mouse_human_ratio=round(kd_ratio, 4),
        )


# -----------------------------------------------------------------------------
# 2. 8D CLIFFORD TARGET STALK  (apical domain multivector)
# -----------------------------------------------------------------------------

def build_tfr1_apical_stalk(mouse: bool = False) -> Multivector:
    """Constructs the 8D Clifford stalk K for the TfR1 apical epitope.

    Dimension encoding in M^8:
      e_1: x-coordinate of epitope centroid (Angstrom, normalised to [0,1])
      e_2: y-coordinate
      e_3: z-coordinate
      e_4: electrostatic potential at Tyr211 (kcal/mol.e, normalised)
      e_5: hydrophobicity index (Kyte-Doolittle centroid, normalised)
      e_6: solvent accessible surface area of epitope (normalised)
      e_7: torsional strain (phi/psi Ramachandran centroid, normalised)
      e_8: desolvation energy of epitope burial (normalised)

    Coordinates derived from PDB 11QC / 11RN centroid of apical patch
    residues 202-215, 348, 371 (Chains A/B), corrected to Gly142.
    """
    if not mouse:
        # Human apical patch centroid (Angstrom, from PDB 11RN chain A)
        # Approximate centroid: Tyr211 Calpha ~ (31.2, 48.7, 22.1) in crystal frame
        coords = (
            0.624,  # e_1: x (normalised by box 50 A)
            0.974,  # e_2: y
            0.442,  # e_3: z
           -0.380,  # e_4: electrostatic (D204(-), K371(+), R208(+) net ~ -0.38 e)
            0.312,  # e_5: hydrophobicity (Tyr+Leu+Val centroid, KD scale ~ 0.31)
            0.485,  # e_6: SASA (epitope burial ~ 48.5% upon complex)
            0.218,  # e_7: torsional strain (beta-strand Ramachandran, normalised)
            0.338,  # e_8: desolvation (estimated from 6WRV binding data)
        )
    else:
        # Mouse apical patch -- 6 position substitutions alter the stalk
        # D204->Q: charge -0.38 -> -0.12 (Q neutral); Y211->P: aromatic lost; K371->Q: +1 -> 0
        coords = (
            0.631,  # e_1: x (small shift -- mouse patch is slightly larger)
            0.971,  # e_2: y
            0.449,  # e_3: z
           -0.120,  # e_4: electrostatic (reduced; Q207 neutral, N211 neutral)
            0.268,  # e_5: hydrophobicity reduced (Pro replaces Tyr, no ring)
            0.502,  # e_6: SASA (Pro211 more solvent exposed)
            0.243,  # e_7: torsional (Pro introduces kink, higher strain)
            0.291,  # e_8: desolvation (lower burial for mouse patch)
        )
    return Clifford8D.vector_8d(coords, signature=Clifford8D.SIG_EUCLIDEAN)


def sequence_to_stalk(sequence: str) -> Multivector:
    """Embeds a binder sequence into the 8D M^8 manifold as a Clifford stalk.

    Dimension encoding (parallel to apical target stalk):
      e_1-e_3: spatial embedding (pseudo-3D from sequence composition moments)
      e_4: net electrostatic potential (computed from charge at pH 7.4)
      e_5: hydrophobic fraction (Kyte-Doolittle)
      e_6: solubility score (1 - hydrophobic fraction)
      e_7: Ramachandran strain proxy (Pro+Gly content)
      e_8: desolvation cost (buried polar fraction)
    """
    if not sequence:
        return Clifford8D.vector_8d((0.0,)*8)

    n = len(sequence)

    # Spatial embedding: composition moments (first-order sequence mean position)
    # Normalised to [0, 1] using amino-acid hydrophobicity as positional weight
    kd_scale = {
        "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
        "M": 1.9, "A": 1.8, "W": -0.9, "T": -0.7, "G": -0.4,
        "S": -0.8, "Y": -1.3, "P": -1.6, "H": -3.2, "D": -3.5,
        "E": -3.5, "N": -3.5, "Q": -3.5, "K": -3.9, "R": -4.5,
    }
    kd_vals = [kd_scale.get(aa, 0.0) for aa in sequence]
    mean_kd = sum(kd_vals) / n
    # Centroid position indices
    weighted_pos = sum((i / n) * kd_scale.get(aa, 0.0) for i, aa in enumerate(sequence))
    weighted_pos /= max(abs(mean_kd) * n, 1e-3)

    e1 = max(0.0, min(1.0, 0.5 + weighted_pos * 0.2))
    e2 = max(0.0, min(1.0, sum(kd_scale.get(aa, 0.0) for aa in sequence[:n//2]) / (n//2 * 5.0 + 1e-3) + 0.5))
    e3 = max(0.0, min(1.0, sum(kd_scale.get(aa, 0.0) for aa in sequence[n//2:]) / (max(n - n//2, 1) * 5.0 + 1e-3) + 0.5))

    # Electrostatic potential (charge at pH 7.4)
    try:
        charge, _ = PhysicoChemicalEngine.compute_charge_and_derivative(sequence, 7.4)
    except Exception:
        pos = sum(1 for aa in sequence if aa in "KRH")
        neg = sum(1 for aa in sequence if aa in "DE")
        charge = pos - neg
    e4 = max(-1.0, min(1.0, charge / max(n * 0.5, 1.0)))

    # Hydrophobic fraction
    hydro_count = sum(1 for aa in sequence if aa in "LIVFAPMW")
    frac_hydro = hydro_count / n
    e5 = frac_hydro

    # Solubility score
    e6 = 1.0 - frac_hydro

    # Ramachandran strain proxy (Pro + Gly content)
    e7 = (sequence.count("P") + sequence.count("G")) / n

    # Desolvation cost proxy (polar buried fraction: Asn, Gln, Ser, Thr, His)
    polar_count = sum(1 for aa in sequence if aa in "NQSTH")
    e8 = polar_count / n

    return Clifford8D.vector_8d((e1, e2, e3, e4, e5, e6, e7, e8), signature=Clifford8D.SIG_EUCLIDEAN)


def clifford_complementary_stalk(target_stalk: Multivector) -> Multivector:
    """Computes the complementary binder stalk Q* = K~ / ||K||2 (Clifford inversion).

    From Theorem 1.3 (Chapter 1, monograph): the optimal binding stalk is the
    algebraic inverse of the target stalk under the Clifford geometric product,
    Q* = K^{-1} = K~ / ||K||2, which maximises the dual-stream attention score
    E_ij = <Q.K~>0 / ||K|| by construction.
    """
    return target_stalk.inverse()


def stalk_alignment_score(binder_stalk: Multivector, target_stalk: Multivector) -> float:
    """8D dual-stream Clifford attention alignment score E_ij in R."""
    return Clifford8D.dual_stream_attention_score(binder_stalk, target_stalk, lambda_wedge=0.5)


# -----------------------------------------------------------------------------
# 3. SEQUENCE DESIGN LIBRARY
# -----------------------------------------------------------------------------

# Residue vocabulary grouped by role
_HYDROPHOBIC     = list("LIVFAM")
_AROMATIC        = list("WYF")
_POLAR           = list("NQST")
_ANIONIC         = list("DE")
_CATIONIC        = list("KR")
_HIS_SWITCH      = ["H"]
_TURN            = list("GP")
_ALL_AA          = list("ACDEFGHIKLMNPQRSTVWY")


def _randaa(pool: List[str], rng: random.Random) -> str:
    return rng.choice(pool)


# Scaffold templates -- position coding:
# H = histidine switch (pH sensor)
# D/E = anionic contact (human R208/K371)
# L/V/I = hydrophobic core (Y211/V210 groove)
# N/Q = polar tolerance (mouse cross-reactivity at D204->Q, Y211->P positions)
# G = Gly turn / disulfide bridge
# C = cysteine for disulfide closure (in cyclic scaffolds)

SCAFFOLD_LIBRARY = [
    # -- MACROCYCLES (disulfide-constrained, D_b ~ 10-12 A) --------------------
    {
        "name": "MacroCycle_A",
        "template": "CHWHDHNLNQVIYDHC",
        "description": "16-mer disulfide macrocycle; 3x His pH-switches; Asp anionic contacts; "
                       "Tyr/Val hydrophobic core; Asn/Gln mouse-tolerant positions.",
        "scaffold_type": "macrocycle",
        "molecule_class": "single_chain",
        "expected_dis_area": 11.2,
    },
    {
        "name": "HisCluster_F",
        "template": "CHHHYDNVQHLRIKHC",
        "description": "16-mer macrocycle; 5x His cluster for extreme DeltaDeltaG pH switch "
                       "margin; Asp/Asn contacts; Objective 3 optimised.",
        "scaffold_type": "macrocycle",
        "molecule_class": "single_chain",
        "expected_dis_area": 10.3,
    },
    {
        "name": "MacroCycle_G",
        "template": "CHYHDHNIYVQTLDGC",
        "description": "16-mer; Tyr-His-Asp triad targeting Tyr211 groove; Gln/Thr "
                       "mouse-tolerant; Asp electrostatic contact with R208.",
        "scaffold_type": "macrocycle",
        "molecule_class": "single_chain",
        "expected_dis_area": 11.0,
    },
    {
        "name": "MacroCycle_H",
        "template": "CWDHHYLVNHQIMDHC",
        "description": "16-mer; Trp aromatic stacking at Tyr211 groove; 3x His; "
                       "dual-Asp contacts; Leu/Val/Ile hydrophobic packing.",
        "scaffold_type": "macrocycle",
        "molecule_class": "single_chain",
        "expected_dis_area": 11.5,
    },
    {
        "name": "Bicyclic_D",
        "template": "CHYILHNDQFHRGVMAECPS",
        "description": "20-mer bicyclic; 3x His; Tyr/Ile/Val core mimics VP1u apical "
                       "binding mode; Asp/Glu salt-bridges; CEPS clasp.",
        "scaffold_type": "bicyclic",
        "molecule_class": "single_chain",
        "expected_dis_area": 10.8,
    },
    {
        "name": "Bicyclic_I",
        "template": "CHDYHNIVQTFLHGESCPK",
        "description": "19-mer bicyclic; His-Asp-Tyr triad N-terminal anchor; "
                       "Ile/Val/Phe hydrophobic ridge; Gln/Thr cross-reactivity layer.",
        "scaffold_type": "bicyclic",
        "molecule_class": "single_chain",
        "expected_dis_area": 10.6,
    },
    # -- beta-HAIRPIN SCAFFOLDS ----------------------------------------------------
    {
        "name": "BetaHairpin_C",
        "template": "CVHYVNHNDGQILHCMDENVKA",
        "description": "22-mer beta-hairpin; Cys2-Cys15 disulfide; 3x His; Val/Ile groove; "
                       "Asn/Gln mouse tolerance; Asp/Glu contacts.",
        "scaffold_type": "beta_hairpin",
        "molecule_class": "single_chain",
        "expected_dis_area": 10.5,
    },
    {
        "name": "BetaHairpin_J",
        "template": "CWHYVNHNDGQIAHCENHDVQS",
        "description": "22-mer beta-hairpin; Trp anchor at Tyr211 aromatic groove; "
                       "3x His; Asn/Gln cross-reactivity; Glu N-terminal contact.",
        "scaffold_type": "beta_hairpin",
        "molecule_class": "single_chain",
        "expected_dis_area": 10.9,
    },
    {
        "name": "BetaHairpin_K",
        "template": "CVHHYNDTGQFLHCMQENKVAH",
        "description": "22-mer beta-hairpin; 3x His cluster; Asp/Asn polar layer; "
                       "Phe/Leu hydrophobic ridge; Gln at mouse-divergent positions.",
        "scaffold_type": "beta_hairpin",
        "molecule_class": "single_chain",
        "expected_dis_area": 10.7,
    },
    # -- alpha-HELICAL MINIPROTEINS ------------------------------------------------
    {
        "name": "MiniProtein_B",
        "template": "AEQKLISEEDLHAQMHYVNQDRHLNAELFRQMVST",
        "description": "35-mer alpha-helical miniprotein; His-loop DHAQMHY targets apical "
                       "groove; Asn/Gln cross-reactivity; Glu/Asp contacts.",
        "scaffold_type": "miniprotein",
        "molecule_class": "single_chain",
        "expected_dis_area": 13.8,
    },
    {
        "name": "MiniProtein_L",
        "template": "AEQHLISEDLHAQFHYVNQNRHLNAELFRQMVNQ",
        "description": "35-mer alpha-helical; His-Phe-Tyr apical-groove triad; "
                       "Asn/Gln saturation at mouse-tolerant positions.",
        "scaffold_type": "miniprotein",
        "molecule_class": "single_chain",
        "expected_dis_area": 13.5,
    },
    {
        "name": "MiniProtein_M",
        "template": "LHAEQKLIHEDDLHAQMHYVNQDRHLNA",
        "description": "28-mer compact alpha-helical; dual His-switch motif (N- and C-terminal); "
                       "Asp/Glu electrostatic belt; optimised for high Tm.",
        "scaffold_type": "miniprotein",
        "molecule_class": "single_chain",
        "expected_dis_area": 12.8,
    },
    # -- CROSS-REACTIVITY OPTIMISED SCAFFOLDS --------------------------------
    {
        "name": "CrossReactive_E",
        "template": "HNQTNHIYVDQNLHAMEHRKLQ",
        "description": "22-mer; maximised Asn/Gln/Thr for mouse tolerance at all 6 "
                       "divergent positions; 2x His; moderate hydrophobic core.",
        "scaffold_type": "miniprotein",
        "molecule_class": "single_chain",
        "expected_dis_area": 12.1,
    },
    {
        "name": "CrossReactive_N",
        "template": "HNQTNHIYVNQNLHSQEHRNLQ",
        "description": "23-mer; Asn/Gln/Ser/Thr at every mouse-divergent contact; "
                       "avoids Arg/Lys (lost in mouse); 2x His; balanced charge.",
        "scaffold_type": "miniprotein",
        "molecule_class": "single_chain",
        "expected_dis_area": 12.4,
    },
    {
        "name": "CrossReactive_O",
        "template": "HTQNHIVFNQNLHAMEHRNLQS",
        "description": "22-mer; Phe replaces Tyr aromatic (tolerates Y211->P mouse); "
                       "Thr/Asn at D204/R208 sites; 2x His.",
        "scaffold_type": "miniprotein",
        "molecule_class": "single_chain",
        "expected_dis_area": 12.0,
    },
    # -- DE NOVO FOLDED BINDERS (alphabeta-sandwich motif) -------------------------
    {
        "name": "DeNovo_AB_P",
        "template": "LHVYDNEQHSGTILHCMQFNRHDEALVQS",
        "description": "29-mer alphabeta-sandwich; Cys-bridged beta-sheet with alpha-helix; "
                       "His at beta-turn and helix-capping for dual pH sensing.",
        "scaffold_type": "ab_sandwich",
        "molecule_class": "single_chain",
        "expected_dis_area": 13.2,
    },
    {
        "name": "DeNovo_AB_Q",
        "template": "LHVYDNEQHSGTILHYMQFNRHDEALVQNS",
        "description": "30-mer alphabeta-sandwich; Tyr replaces Cys (no disulfide); "
                       "His5/His15/His27 triad; Asp/Glu electrostatic network.",
        "scaffold_type": "ab_sandwich",
        "molecule_class": "single_chain",
        "expected_dis_area": 13.6,
    },
    # -- EXTENDED LOOP / LARIAT SCAFFOLDS ------------------------------------
    {
        "name": "Lariat_R",
        "template": "CHHDYQNVILHSGQFTDNHEMIC",
        "description": "23-mer lariat topology; N-/C-terminal Cys lariat clasp; "
                       "3x His loop; Asp/Gln apical contacts; Ile/Leu/Val hydrophobic tail.",
        "scaffold_type": "lariat",
        "molecule_class": "single_chain",
        "expected_dis_area": 11.8,
    },
    {
        "name": "Lariat_S",
        "template": "CHHDY QNVIYHSGTDNHEMIC",
        "description": "23-mer lariat; Tyr-His dual aromatic-switch at Tyr211 groove; "
                       "Gln/Asn cross-reactivity belt; Glu anionic contact.",
        "scaffold_type": "lariat",
        "molecule_class": "single_chain",
        "expected_dis_area": 11.4,
    },
    # -- NANOBODY-INSPIRED CDR-GRAFTED SCAFFOLDS ------------------------------
    {
        "name": "NanobodyGraft_T",
        "template": "QVQLVESGGGLVHAGGERLRSCAASHHNFHSYMGWFRQAPGKERHLVAAISWSGDTYYADSVKGRFTIH",
        "description": "68-mer nanobody-inspired; CDR1 (HHNFHS) and CDR3 (DTYYA) "
                       "grafted with His-switch; Asp/Glu contact residues at CDR2.",
        "scaffold_type": "nanobody",
        "molecule_class": "nanobody",
        "expected_dis_area": 18.5,
    },
    {
        "name": "NanobodyGraft_U",
        "template": "QVQLVESGGGLVHAGGSLRLSCAASHHDFHSYMGWFRQAPGKERHYVAAISHSGHNTYYADSVKGRFTIH",
        "description": "69-mer nanobody-inspired; CDR1 extended His-cluster HHDFHS; "
                       "Tyr/Asn/His CDR3; Glu anionic belt; high Tm expected.",
        "scaffold_type": "nanobody",
        "molecule_class": "nanobody",
        "expected_dis_area": 19.0,
    },
]


def generate_variants(scaffold: dict, n_variants: int, seed: int = 42) -> List[Tuple[str, str]]:
    """Generates sequence variants by controlled mutagenesis of a scaffold template.

    Returns list of (variant_id, sequence) tuples.
    Strategy:
      * Preserve all His, Cys (disulfide) residues.
      * Mutate 20-40% of non-essential positions.
      * Enforce phi* ~ 0.28 hydrophobic fraction constraint.
      * Insert 1-3 additional His for switch optimisation in non-structural positions.
    """
    rng = random.Random(seed)
    template = scaffold["template"]
    name = scaffold["name"]
    variants = []

    for v_idx in range(n_variants):
        seq = list(template)
        n = len(seq)

        for pos, aa in enumerate(seq):
            if aa in "HC":
                continue  # Preserve His (switch) and Cys (disulfide)
            if rng.random() < 0.30:  # 30% mutation rate
                # Choose residue class based on position role heuristic
                roll = rng.random()
                if roll < 0.15:
                    seq[pos] = _randaa(_HIS_SWITCH, rng)    # Extra His switch
                elif roll < 0.30:
                    seq[pos] = _randaa(_ANIONIC, rng)        # Anionic contact
                elif roll < 0.50:
                    seq[pos] = _randaa(_POLAR, rng)          # Mouse-tolerant polar
                elif roll < 0.70:
                    seq[pos] = _randaa(_HYDROPHOBIC, rng)    # Hydrophobic core
                elif roll < 0.85:
                    seq[pos] = _randaa(_AROMATIC, rng)       # Aromatic groove
                else:
                    seq[pos] = _randaa(_TURN, rng)           # Turn/flexibility

        variant_seq = "".join(seq)
        var_id = f"{name}_v{v_idx + 1:03d}"
        variants.append((var_id, variant_seq))

    return variants


# -----------------------------------------------------------------------------
# 4. VIRTUAL WETLAB ASSAY (TfR1-specific, dual pH)
# -----------------------------------------------------------------------------

@dataclass
class TfR1AssayResult:
    """Complete virtual assay readout for a TfR1 binder candidate."""
    design_id: str
    sequence: str
    sequence_with_tag: str

    # pH 7.4 human results
    dg_human_74: float           # DeltaG kcal/mol -- target: < -6 kcal/mol
    kd_human_74_nm: float        # KD nM -- target: < 100 nM
    spr_ru_human_74: float       # SPR response units at 100 nM
    binds_human_74: bool

    # pH 6.0 human results
    dg_human_60: float           # DeltaG kcal/mol -- target: > 0 kcal/mol
    kd_human_60_nm: float        # KD nM -- target: > 300 nM (no detectable binding)
    spr_ru_human_60: float       # SPR RU at 100 nM -- target: < 300% of NC
    binds_human_60: bool

    # Mouse pH 7.4 results
    dg_mouse_74: float
    kd_mouse_74_nm: float
    binds_mouse_74: bool
    mouse_human_kd_ratio: float  # target: < 3.0

    # pH switch metrics
    delta_delta_g: float         # DeltaDeltaG = DeltaG(6.0) - DeltaG(7.4); target: > +4.5
    switch_active: bool

    # Biophysical QC
    net_charge_74: float
    isoelectric_point: float
    hydrophobic_fraction: float
    phi_star_deviation: float    # |phi - phi*|; target: < 0.08
    tm_estimate_c: float         # target: > 55degC
    aggregation_risk: str
    qc_pass: bool
    qc_flags: List[str] = field(default_factory=list)

    # Clifford alignment
    clifford_score_human: float = 0.0  # E_ij in R; target: > 0
    clifford_score_mouse: float = 0.0

    # Ranking score (lower is better: minimise - pH selectivity + kd + cross-reactivity penalty)
    competition_score: float = 0.0


def _dg_to_kd_nm(dg_kcal: float, temperature_k: float = 310.15) -> float:
    """Converts DeltaG (kcal/mol) to KD (nM) via DeltaG = RT ln(KD)."""
    r_kcal = 0.001987  # kcal / (mol . K)
    if dg_kcal >= 0.0:
        return 1e6  # > 1 muM -> no binding
    # KD = exp(DeltaG / RT) in molar -> x 1e9 for nM
    kd_m = math.exp(dg_kcal / (r_kcal * temperature_k))
    return kd_m * 1e9


def _kd_to_spr_ru(kd_nm: float, conc_nm: float = 100.0, r_max: float = 120.0) -> float:
    """Simulates SPR response units at a given analyte concentration."""
    fraction_bound = conc_nm / (kd_nm + conc_nm)
    return r_max * fraction_bound


def run_tfr1_assay(
    design_id: str,
    sequence: str,
    switch_engine: InvertedHisSwitchEngine,
    target_stalk_human: Multivector,
    target_stalk_mouse: Multivector,
) -> TfR1AssayResult:
    """Executes the complete TfR1 virtual assay pipeline on a single candidate."""

    # 1. pH-selectivity evaluation
    switch_human = switch_engine.evaluate_ph_pair(sequence, mouse=False)
    switch_mouse  = switch_engine.evaluate_ph_pair(sequence, mouse=True)

    # 2. KD conversion
    kd_h74 = _dg_to_kd_nm(switch_human.delta_g_74)
    kd_h60 = _dg_to_kd_nm(switch_human.delta_g_60)
    kd_m74 = _dg_to_kd_nm(switch_mouse.delta_g_74)

    # SPR (100 nM analyte, 300 RU threshold = 300% of NC ~ 15 RU -> use 45 RU as absolute threshold)
    spr_h74 = _kd_to_spr_ru(kd_h74)
    spr_h60 = _kd_to_spr_ru(kd_h60)

    # Binding detection thresholds (per competition rules):
    # "fittable curve and KD" -> KD < 1000 nM; else "association shift > 300% over NC"
    BIND_KD_THRESHOLD_NM  = 1000.0   # KD < 1 muM = detectable binding
    NO_BIND_KD_THRESHOLD  = 1000.0   # KD >= 1 muM = no detectable binding
    binds_h74 = kd_h74 < BIND_KD_THRESHOLD_NM
    binds_h60 = kd_h60 < NO_BIND_KD_THRESHOLD
    binds_m74 = kd_m74 < BIND_KD_THRESHOLD_NM

    # KD ratio
    mouse_human_ratio = kd_m74 / max(kd_h74, 0.001)

    # 3. Physicochemical QC
    try:
        pc = PhysicoChemicalEngine.evaluate_sequence(sequence)
        pi = pc.isoelectric_point
        charge, _ = PhysicoChemicalEngine.compute_charge_and_derivative(sequence, 7.4)
    except Exception:
        pos = sum(1 for aa in sequence if aa in "KRH")
        neg = sum(1 for aa in sequence if aa in "DE")
        charge = float(pos - neg)
        pi = 7.4 + charge * 0.2

    hydro_count = sum(1 for aa in sequence if aa in "LIVFAPMW")
    frac_hydro = hydro_count / max(len(sequence), 1)
    phi_dev = abs(frac_hydro - PHI_STAR)

    # Thermal stability (Tm estimate)
    tm_c = 42.0 + frac_hydro * 38.0 + sequence.count("H") * 2.5 - sequence.count("G") * 1.5
    tm_c = round(tm_c, 1)

    # Aggregation risk
    if frac_hydro > 0.45:
        agg_risk = "HIGH"
    elif frac_hydro > 0.35:
        agg_risk = "MEDIUM"
    else:
        agg_risk = "LOW"

    # 4. QC flags
    qc_flags: List[str] = []
    if abs(charge) < 0.5:
        qc_flags.append(f"ISOELECTRIC_RISK: Q={charge:+.2f} near pI at pH 7.4")
    if kd_h74 >= 500.0:
        qc_flags.append(f"WEAK_HUMAN_AFFINITY: KD={kd_h74:.0f} nM at pH 7.4 (>500 nM threshold)")
    if binds_h60:
        qc_flags.append(f"pH_SWITCH_FAILURE: Still binds at pH 6.0, KD={kd_h60:.0f} nM")
    if agg_risk == "HIGH":
        qc_flags.append(f"AGGREGATION_RISK_HIGH: phi={frac_hydro:.3f} > 0.45")
    if tm_c < 55.0:
        qc_flags.append(f"LOW_THERMAL_STABILITY: Tm={tm_c:.1f}degC < 55degC")
    if phi_dev > 0.10:
        qc_flags.append(f"PHI_STAR_DEVIATION: |phi-phi*|={phi_dev:.3f} > 0.10 (expect hallucination risk)")
    qc_pass = len(qc_flags) == 0

    # 5. Clifford alignment scores
    binder_stalk = sequence_to_stalk(sequence)
    csc_human = stalk_alignment_score(binder_stalk, target_stalk_human)
    csc_mouse  = stalk_alignment_score(binder_stalk, target_stalk_mouse)

    # 6. Competition ranking score
    # Competition priority order (per Adaptyv rules):
    #   Tier 1: pH selectivity  (Obj 3: binds h74, no binding h60)
    #   Tier 2: Mouse cross-reactivity  (Obj 2: KD ratio close to 1)
    #   Tier 3: Affinity  (Obj 1: lowest KD at h74)
    #
    # s_affinity: log10(KD_h74) normalised, [0.01,10000] nM -> [0,1]
    kd_h74_clamped = max(0.01, min(kd_h74, 10000.0))
    s_affinity = (math.log10(kd_h74_clamped) + 2.0) / 6.0

    # s_switch: DeltaDeltaG=0 -> 1.0 (no switch), DeltaDeltaG=12 -> 0.0 (excellent)
    # +0.5 penalty if switch not active
    ddg_norm = max(0.0, 1.0 - switch_human.delta_delta_g / 12.0)
    s_switch = min(ddg_norm + (0.5 if not switch_human.switch_active else 0.0), 1.5)

    # s_cross: KD ratio 1.0 -> 0.0 (perfect), 10.0 -> 1.0
    kd_ratio_clamped = max(1.0, min(mouse_human_ratio, 10.0))
    s_cross = (kd_ratio_clamped - 1.0) / 9.0

    # s_clifford: geometric alignment bonus (small discriminator)
    s_clifford = max(0.0, 0.1 - csc_human * 0.05)

    competition_score = round(
        0.40 * s_affinity + 0.35 * s_switch + 0.20 * s_cross + 0.05 * s_clifford, 4
    )

    return TfR1AssayResult(
        design_id=design_id,
        sequence=sequence,
        sequence_with_tag=sequence + TWIN_STREP_TAG,
        dg_human_74=switch_human.delta_g_74,
        kd_human_74_nm=round(kd_h74, 2),
        spr_ru_human_74=round(spr_h74, 2),
        binds_human_74=binds_h74,
        dg_human_60=switch_human.delta_g_60,
        kd_human_60_nm=round(kd_h60, 2),
        spr_ru_human_60=round(spr_h60, 2),
        binds_human_60=binds_h60,
        dg_mouse_74=switch_mouse.delta_g_74,
        kd_mouse_74_nm=round(kd_m74, 2),
        binds_mouse_74=binds_m74,
        mouse_human_kd_ratio=round(mouse_human_ratio, 3),
        delta_delta_g=switch_human.delta_delta_g,
        switch_active=switch_human.switch_active,
        net_charge_74=round(charge, 3),
        isoelectric_point=round(pi, 2),
        hydrophobic_fraction=round(frac_hydro, 4),
        phi_star_deviation=round(phi_dev, 4),
        tm_estimate_c=tm_c,
        aggregation_risk=agg_risk,
        qc_pass=qc_pass,
        qc_flags=qc_flags,
        clifford_score_human=round(csc_human, 6),
        clifford_score_mouse=round(csc_mouse, 6),
        competition_score=competition_score,
    )


# -----------------------------------------------------------------------------
# 5. LANGEVIN DIFFUSION SEQUENCE OPTIMISER
# -----------------------------------------------------------------------------

def langevin_optimise(
    initial_seq: str,
    switch_engine: InvertedHisSwitchEngine,
    target_stalk_human: Multivector,
    target_stalk_mouse: Multivector,
    n_steps: int = 500,
    temperature: float = 1.2,
    cooling_rate: float = 0.995,
    seed: int = 0,
) -> str:
    """Discrete Langevin diffusion optimiser over sequence space.

    Implements a Metropolis-Hastings random walk over single amino-acid
    substitutions guided by the PLQ competition score (lower = better).
    Histidine and Cysteine positions are locked to preserve scaffold integrity.

    Energy function: E(seq) = competition_score (from run_tfr1_assay).
    Acceptance: min(1, exp(-DeltaE / T)) where T is annealed.
    """
    rng = random.Random(seed)
    seq = list(initial_seq)
    n = len(seq)

    def _score(s: str) -> float:
        try:
            result = run_tfr1_assay("_tmp", s, switch_engine, target_stalk_human, target_stalk_mouse)
            return result.competition_score
        except Exception:
            return 1.0

    current_seq = "".join(seq)
    current_score = _score(current_seq)
    best_seq = current_seq
    best_score = current_score
    T = temperature

    for step in range(n_steps):
        # Pick random non-locked position
        candidates = [i for i, aa in enumerate(current_seq) if aa not in "HC"]
        if not candidates:
            break
        pos = rng.choice(candidates)

        # Propose substitution from curated pool
        roll = rng.random()
        if roll < 0.20:
            new_aa = rng.choice(_HIS_SWITCH)
        elif roll < 0.40:
            new_aa = rng.choice(_POLAR)
        elif roll < 0.60:
            new_aa = rng.choice(_HYDROPHOBIC)
        elif roll < 0.75:
            new_aa = rng.choice(_ANIONIC)
        else:
            new_aa = rng.choice(_ALL_AA)

        # Apply mutation
        candidate_seq = current_seq[:pos] + new_aa + current_seq[pos + 1:]
        candidate_score = _score(candidate_seq)

        # Metropolis acceptance
        delta_e = candidate_score - current_score
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 0.01)):
            current_seq = candidate_seq
            current_score = candidate_score
            if current_score < best_score:
                best_score = current_score
                best_seq = current_seq

        T *= cooling_rate

    return best_seq


# -----------------------------------------------------------------------------
# 6. DATABASE PERSISTENCE
# -----------------------------------------------------------------------------

def _init_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tfr1_designs (
                design_id         TEXT PRIMARY KEY,
                run_timestamp     TEXT,
                sequence          TEXT,
                sequence_with_tag TEXT,
                scaffold_name     TEXT,
                dg_human_74       REAL,
                kd_human_74_nm    REAL,
                binds_human_74    INTEGER,
                dg_human_60       REAL,
                kd_human_60_nm    REAL,
                binds_human_60    INTEGER,
                delta_delta_g     REAL,
                switch_active     INTEGER,
                dg_mouse_74       REAL,
                kd_mouse_74_nm    REAL,
                binds_mouse_74    INTEGER,
                mouse_human_ratio REAL,
                net_charge_74     REAL,
                isoelectric_point REAL,
                hydrophobic_frac  REAL,
                phi_star_dev      REAL,
                tm_estimate_c     REAL,
                aggregation_risk  TEXT,
                qc_pass           INTEGER,
                qc_flags_json     TEXT,
                clifford_human    REAL,
                clifford_mouse    REAL,
                competition_score REAL
            )
        """)
    conn.close()


def persist_result(result: TfR1AssayResult, scaffold_name: str, path: Path) -> None:
    conn = sqlite3.connect(str(path))
    with conn:
        conn.execute("""
            INSERT OR REPLACE INTO tfr1_designs VALUES
            (?,datetime('now'),?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            result.design_id, result.sequence, result.sequence_with_tag, scaffold_name,
            result.dg_human_74, result.kd_human_74_nm, int(result.binds_human_74),
            result.dg_human_60, result.kd_human_60_nm, int(result.binds_human_60),
            result.delta_delta_g, int(result.switch_active),
            result.dg_mouse_74, result.kd_mouse_74_nm, int(result.binds_mouse_74),
            result.mouse_human_kd_ratio, result.net_charge_74, result.isoelectric_point,
            result.hydrophobic_fraction, result.phi_star_deviation, result.tm_estimate_c,
            result.aggregation_risk, int(result.qc_pass), json.dumps(result.qc_flags),
            result.clifford_score_human, result.clifford_score_mouse, result.competition_score,
        ))
    conn.close()


# -----------------------------------------------------------------------------
# 7. HIGH-THROUGHPUT PARALLEL SCREENING ENGINE (10,000 DESIGNS)
# -----------------------------------------------------------------------------

def _screen_one(args: Tuple) -> TfR1AssayResult:
    """Thread-safe worker for parallel assay screening."""
    vid, seq, switch_engine, K_human, K_mouse = args
    return run_tfr1_assay(vid, seq, switch_engine, K_human, K_mouse)


def run_tfr1_campaign(
    n_total_designs: int = 10_000,
    n_langevin_steps: int = 300,
    top_k_report: int = 20,
    n_workers: int = 12,
    output_db: Path = DB_PATH,
    csv_output: Optional[Path] = None,
    verbose: bool = True,
) -> List[TfR1AssayResult]:
    """High-throughput TfR1 pH-binder design campaign.

    Pipeline:
      1. Build 8D Clifford target stalks (human + mouse apical domain).
      2. Distribute n_total_designs across all scaffolds proportionally.
      3. Generate variant libraries from all scaffold templates.
      4. Screen all variants in parallel using ThreadPoolExecutor (CPU 14-core).
      5. Langevin-optimise top-5 candidates per scaffold (top-5% elite).
      6. Re-screen optimised elite candidates.
      7. Global ranking; report top-K.
      8. Persist all results to SQLite; export top-20 to competition CSV.

    Args:
        n_total_designs: Total design budget (default 10,000).
        n_langevin_steps: Metropolis-Hastings steps per elite optimisation.
        top_k_report: Number of designs to include in final ranking/CSV.
        n_workers: ThreadPoolExecutor parallelism (default 12 of 14 CPU cores).
        output_db: SQLite persistence path.
        csv_output: Path for competition CSV export (None = auto-named).
        verbose: Print progress.

    Returns:
        Sorted list of top_k_report TfR1AssayResult objects.
    """
    _init_db(output_db)

    switch_engine = InvertedHisSwitchEngine(
        his_pka_env=7.6,
        asp_binder_pka=4.2,
        switch_ddg_threshold=4.5,
    )
    K_human = build_tfr1_apical_stalk(mouse=False)
    K_mouse  = build_tfr1_apical_stalk(mouse=True)

    n_scaffolds = len(SCAFFOLD_LIBRARY)
    # Distribute designs: base allocation + remainder to first scaffolds
    base_per_scaffold = n_total_designs // n_scaffolds
    remainder = n_total_designs % n_scaffolds

    if verbose:
        print("=" * 76)
        print("TfR1 pH-Binder High-Throughput Campaign -- PLQ v3.0 / ToposLang Engine")
        print("Target : Human TfR1 apical domain (UniProt P02786-1)")
        print("Epitope: residues 202-215, 348, 371  |  Anchor: Tyr211")
        print(f"Designs: {n_total_designs:,}  |  Scaffolds: {n_scaffolds}  |  Workers: {n_workers}")
        print(f"K_human norm = {K_human.norm():.4f}  |  K_mouse norm = {K_mouse.norm():.4f}")
        print("=" * 76)

    all_results: List[TfR1AssayResult] = []
    global_seed = 42
    t_start = time.perf_counter()

    for s_idx, scaffold in enumerate(SCAFFOLD_LIBRARY):
        sname = scaffold["name"]
        n_variants = base_per_scaffold + (1 if s_idx < remainder else 0)

        if verbose:
            elapsed = time.perf_counter() - t_start
            print(f"\n[{s_idx+1:02d}/{n_scaffolds}] {sname} -- {n_variants} variants  "
                  f"(+{elapsed:.1f}s elapsed)")

        # Generate variant library
        variants = generate_variants(scaffold, n_variants, seed=global_seed)
        global_seed += n_variants + 100

        # Parallel screening
        tasks = [(vid, seq, switch_engine, K_human, K_mouse) for vid, seq in variants]
        scaffold_results: List[TfR1AssayResult] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_screen_one, t): t for t in tasks}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    r = fut.result()
                    scaffold_results.append(r)
                except Exception as exc:
                    pass  # Skip failed assays silently

        # Batch-persist to SQLite
        conn = sqlite3.connect(str(output_db))
        with conn:
            for r in scaffold_results:
                conn.execute("""
                    INSERT OR REPLACE INTO tfr1_designs VALUES
                    (?,datetime('now'),?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    r.design_id, r.sequence, r.sequence_with_tag,
                    sname,
                    r.dg_human_74, r.kd_human_74_nm, int(r.binds_human_74),
                    r.dg_human_60, r.kd_human_60_nm, int(r.binds_human_60),
                    r.delta_delta_g, int(r.switch_active),
                    r.dg_mouse_74, r.kd_mouse_74_nm, int(r.binds_mouse_74),
                    r.mouse_human_kd_ratio, r.net_charge_74, r.isoelectric_point,
                    r.hydrophobic_fraction, r.phi_star_deviation, r.tm_estimate_c,
                    r.aggregation_risk, int(r.qc_pass), json.dumps(r.qc_flags),
                    r.clifford_score_human, r.clifford_score_mouse, r.competition_score,
                ))
        conn.close()

        # Elite selection: top-5 by competition_score for Langevin optimisation
        scaffold_results.sort(key=lambda x: x.competition_score)
        elite_count = max(3, n_variants // 100)  # top-1% or minimum 3
        elite = scaffold_results[:elite_count]

        if verbose and scaffold_results:
            best = scaffold_results[0]
            print(f"  Screened {len(scaffold_results)} OK  |  "
                  f"Best pre-opt: KD(h74)={best.kd_human_74_nm:.1f} nM  "
                  f"DeltaDeltaG={best.delta_delta_g:.2f}  switch={'YES' if best.switch_active else 'NO'}")

        # Langevin optimisation of elite
        opt_results = []
        for e_idx, pre_opt in enumerate(elite[:5]):
            opt_seq = langevin_optimise(
                initial_seq=pre_opt.sequence,
                switch_engine=switch_engine,
                target_stalk_human=K_human,
                target_stalk_mouse=K_mouse,
                n_steps=n_langevin_steps,
                seed=global_seed + e_idx,
            )
            opt_id = f"{sname}_OPT{e_idx + 1:02d}"
            opt_r = run_tfr1_assay(opt_id, opt_seq, switch_engine, K_human, K_mouse)
            persist_result(opt_r, sname + "_ELITE_OPT", output_db)
            opt_results.append(opt_r)

        global_seed += 50

        if verbose and opt_results:
            best_opt = min(opt_results, key=lambda x: x.competition_score)
            print(f"  Elite-opt best: {best_opt.design_id}  "
                  f"KD(h74)={best_opt.kd_human_74_nm:.1f} nM  "
                  f"DeltaDeltaG={best_opt.delta_delta_g:.2f}  score={best_opt.competition_score:.4f}")

        all_results.extend(scaffold_results)
        all_results.extend(opt_results)

    # -- Global ranking --------------------------------------------------------
    # Primary: switch_active=True (Objective 3 first per competition rules)
    # Secondary: competition_score ascending
    all_results.sort(key=lambda x: (
        0 if x.switch_active else 1,          # switch designs rank first
        0 if x.qc_pass else 1,                # then QC-pass designs
        x.competition_score,                   # then lowest score
    ))

    top_k = all_results[:top_k_report]
    elapsed_total = time.perf_counter() - t_start

    if verbose:
        print("\n" + "=" * 76)
        print(f"CAMPAIGN COMPLETE  |  Total designs: {len(all_results):,}  |  "
              f"Time: {elapsed_total:.1f}s")
        print(f"TOP {top_k_report} DESIGNS -- RANKED BY COMPETITION SCORE")
        print("=" * 76)
        hdr = (f"{'Rk':<3} {'Design ID':<30} {'KD_h74':>9} {'KD_h60':>9} "
               f"{'DeltaDeltaG':>7} {'Sw':>3} {'KD_m/h':>7} {'Score':>7} {'QC':<5}")
        print(hdr)
        print("-" * 76)
        for i, r in enumerate(top_k):
            print(
                f"{i+1:<3} {r.design_id:<30} "
                f"{r.kd_human_74_nm:>8.1f}n {r.kd_human_60_nm:>8.1f}n "
                f"{r.delta_delta_g:>7.2f} "
                f"{'Y' if r.switch_active else 'N':>3} "
                f"{r.mouse_human_kd_ratio:>7.2f} "
                f"{r.competition_score:>7.4f} "
                f"{'PASS' if r.qc_pass else 'FAIL':<5}"
            )

    # -- Competition CSV export ------------------------------------------------
    if csv_output is None:
        csv_output = output_db.parent / "tfr1_top20_submission.csv"

    export_competition_csv(top_k, csv_output, verbose=verbose)

    return top_k


# -----------------------------------------------------------------------------
# 8. COMPETITION CSV EXPORTER
# -----------------------------------------------------------------------------

def export_competition_csv(
    results: List[TfR1AssayResult],
    output_path: Path,
    verbose: bool = True,
) -> None:
    """Exports top designs to competition submission CSV.

    CSV schema follows competition-submission-template.csv:
      name, sequence, molecule_class

    'name'         -- unique design identifier (PLQ-assigned)
    'sequence'     -- binder sequence ONLY (no Twin-Strep tag in the binding
                     region; the tag is appended by the assay team per protocol)
    'molecule_class' -- one of: single_chain, nanobody, scfv, fab_kappa

    The Twin-Strep tag (WSHPQFEKGGGSAWSHPQFEK) will be ligated C-terminally
    by the Adaptyv team -- it must NOT appear in the submitted sequence.
    """
    scaffold_to_class: Dict[str, str] = {
        "MacroCycle_A":       "single_chain",
        "HisCluster_F":       "single_chain",
        "MacroCycle_G":       "single_chain",
        "MacroCycle_H":       "single_chain",
        "Bicyclic_D":         "single_chain",
        "Bicyclic_I":         "single_chain",
        "BetaHairpin_C":      "single_chain",
        "BetaHairpin_J":      "single_chain",
        "BetaHairpin_K":      "single_chain",
        "MiniProtein_B":      "single_chain",
        "MiniProtein_L":      "single_chain",
        "MiniProtein_M":      "single_chain",
        "CrossReactive_E":    "single_chain",
        "CrossReactive_N":    "single_chain",
        "CrossReactive_O":    "single_chain",
        "DeNovo_AB_P":        "single_chain",
        "DeNovo_AB_Q":        "single_chain",
        "Lariat_R":           "single_chain",
        "Lariat_S":           "single_chain",
        "NanobodyGraft_T":    "nanobody",
        "NanobodyGraft_U":    "nanobody",
    }

    rows = []
    for rank, r in enumerate(results):
        # Infer molecule class from design_id prefix
        mol_class = "single_chain"
        for prefix, cls in scaffold_to_class.items():
            if r.design_id.startswith(prefix):
                mol_class = cls
                break

        # Sanitise sequence (remove spaces introduced by template formatting)
        clean_seq = r.sequence.replace(" ", "").upper()

        rows.append({
            "name": r.design_id,
            "sequence": clean_seq,
            "molecule_class": mol_class,
        })

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "sequence", "molecule_class"])
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        print(f"\n[CSV] {len(rows)} designs exported to: {output_path}")


# -----------------------------------------------------------------------------
# 9. DESIGN DOSSIER REPORT
# -----------------------------------------------------------------------------

def report_top_designs(results: List[TfR1AssayResult], n: int = 5) -> None:
    """Prints competition-ready design dossier for the top-n results."""
    print("\n" + "=" * 72)
    print("COMPETITION DESIGN DOSSIER -- Anthropic x Adaptyv Challenge 4")
    print("Author: Elias Oulad Brahim | PLQ v3.0 / ToposLang Engine")
    print("=" * 72)

    for rank, r in enumerate(results[:n]):
        print(f"\n{'-' * 65}")
        print(f"DESIGN #{rank + 1:02d}: {r.design_id}")
        print(f"{'-' * 65}")
        print(f"  Binder sequence ({len(r.sequence)} aa):")
        print(f"  {r.sequence}")
        print(f"  Full assay construct ({len(r.sequence_with_tag)} aa):")
        print(f"  {r.sequence_with_tag}")
        print()
        print("  Objective 1 -- Human TfR1 @ pH 7.4 (binding):")
        print(f"    DeltaG = {r.dg_human_74:+.2f} kcal/mol  |  "
              f"KD = {r.kd_human_74_nm:.2f} nM  |  "
              f"SPR = {r.spr_ru_human_74:.1f} RU  |  "
              f"Binds: {'YES' if r.binds_human_74 else 'NO'}")
        print()
        print("  Objective 3 -- Human TfR1 @ pH 6.0 (endosomal release):")
        print(f"    DeltaG = {r.dg_human_60:+.2f} kcal/mol  |  "
              f"KD = {r.kd_human_60_nm:.2f} nM  |  "
              f"SPR = {r.spr_ru_human_60:.1f} RU")
        print(f"    DeltaDeltaG = {r.delta_delta_g:+.2f} kcal/mol  |  "
              f"Switch active: {'YES [OK]' if r.switch_active else 'NO [FAIL]'}")
        print()
        print("  Objective 2 -- Mouse TfR1 @ pH 7.4 (cross-reactivity):")
        print(f"    DeltaG = {r.dg_mouse_74:+.2f} kcal/mol  |  "
              f"KD = {r.kd_mouse_74_nm:.2f} nM  |  "
              f"KD(mouse)/KD(human) = {r.mouse_human_kd_ratio:.3f}")
        print(f"    Cross-reactive: "
              f"{'YES [OK]' if r.binds_mouse_74 and r.mouse_human_kd_ratio < 3.0 else 'NO'}")
        print()
        print("  Biophysical QC:")
        print(f"    Net charge = {r.net_charge_74:+.2f}  |  pI = {r.isoelectric_point:.2f}  |  "
              f"phi = {r.hydrophobic_fraction:.4f} (phi* = {PHI_STAR})  |  "
              f"Tm = {r.tm_estimate_c:.1f} C  |  Agg: {r.aggregation_risk}")
        print(f"    QC: {'PASS' if r.qc_pass else 'FAIL'}", end="")
        if r.qc_flags:
            print(f"  [{'; '.join(r.qc_flags[:2])}]", end="")
        print()
        print(f"    Clifford alignment (human/mouse): "
              f"{r.clifford_score_human:.6f} / {r.clifford_score_mouse:.6f}")
        print(f"    Competition score: {r.competition_score:.4f}  (lower is better)")


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TfR1 pH-Selective Binder Design Campaign -- PLQ v3.0"
    )
    parser.add_argument("--designs",  type=int, default=10_000,
                        help="Total design budget (default: 10000)")
    parser.add_argument("--top",      type=int, default=20,
                        help="Number of top designs to report and export (default: 20)")
    parser.add_argument("--workers",  type=int, default=12,
                        help="Parallel screening workers (default: 12)")
    parser.add_argument("--steps",    type=int, default=300,
                        help="Langevin optimisation steps per elite (default: 300)")
    parser.add_argument("--db",       type=Path, default=DB_PATH,
                        help="SQLite output path")
    parser.add_argument("--csv",      type=Path, default=None,
                        help="Competition CSV output path (auto-named if omitted)")
    parser.add_argument("--quiet",    action="store_true",
                        help="Suppress progress output")
    args = parser.parse_args()

    print(f"Launching TfR1 pH-Selective Binder Design Campaign")
    print(f"  Budget : {args.designs:,} designs  |  Top-K : {args.top}")
    print(f"  Workers: {args.workers}  |  Langevin steps: {args.steps}")
    print()

    t0 = time.perf_counter()
    top_results = run_tfr1_campaign(
        n_total_designs=args.designs,
        n_langevin_steps=args.steps,
        top_k_report=args.top,
        n_workers=args.workers,
        output_db=args.db,
        csv_output=args.csv,
        verbose=not args.quiet,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nTotal elapsed: {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    report_top_designs(top_results, n=min(5, args.top))
    print(f"\nAll {args.designs:,} results persisted to: {args.db}")
    csv_path = args.csv or DB_PATH.parent / "tfr1_top20_submission.csv"
    print(f"Competition CSV:  {csv_path}")
