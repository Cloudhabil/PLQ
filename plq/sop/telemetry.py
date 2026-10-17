"""Standard Operating Procedure (SOP 1-4) Automated Wet-Lab Telemetry Loopback.

Implements automated ingestion, fitting, and hypothesis testing for the four
experimental validation protocols detailed in Section 7.3 of the monograph:
- SOP-1: Surface Plasmon Resonance (SPR) pH switching kinetics (Theorem T4)
- SOP-2: Circular Dichroism (CD) secondary structure & thermal melt (Theorem T1)
- SOP-3: Differential Scanning Fluorimetry (DSF) high-throughput pH stability
- SOP-4: Cryo-EM coordinate extraction and empirical DACC reconstruction (--from-pdb)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from plq.dacc.complex import DACCComplex
from plq.dacc.fingerprint import TopologicalDACCInvariant


@dataclass
class SOP1Report:
    candidate_id: str
    kd_74_nm: float
    kd_60_nm: float
    ratio_kd: float
    his_count: int
    t4_switch_confirmed: bool
    chi_square_ru2: float
    acceptance_passed: bool


@dataclass
class SOP2Report:
    candidate_id: str
    scaffold_class: str
    f_alpha_helix: float
    f_beta_sheet: float
    tm_celsius: float
    t1_foldability_confirmed: bool
    acceptance_passed: bool


@dataclass
class SOP3Report:
    candidate_id: str
    tm_74_celsius: float
    tm_60_celsius: float
    delta_tm_celsius: float
    his_count: int
    percolation_destabilization_confirmed: bool


@dataclass
class SOP4Report:
    candidate_id: str
    resolution_angstrom: float
    predicted_beta1: int
    experimental_beta1: int
    topological_match: bool
    falsification_detected: bool


class ExperimentalTelemetryLoopback:
    """Processes wet-lab instrument data and validates topological predictions."""

    @staticmethod
    def evaluate_sop1_spr(
        candidate_id: str,
        kd_74_nm: float,
        kd_60_nm: float,
        his_count: int,
        chi_square: float = 1.2,
    ) -> SOP1Report:
        """Evaluates Biacore SPR kinetics according to SOP-1 (Section 7.3.1).
        
        Hypothesis T4: N_His >= 4 must exhibit KD(6.0) / KD(7.4) >= 100.
        """
        ratio = round(kd_60_nm / max(1e-9, kd_74_nm), 2)
        if his_count >= 4:
            switch_ok = (ratio >= 100.0)
        else:
            switch_ok = (ratio < 10.0)

        quality_ok = (chi_square < 2.0)
        passed = switch_ok and quality_ok

        return SOP1Report(
            candidate_id=candidate_id,
            kd_74_nm=kd_74_nm,
            kd_60_nm=kd_60_nm,
            ratio_kd=ratio,
            his_count=his_count,
            t4_switch_confirmed=switch_ok,
            chi_square_ru2=chi_square,
            acceptance_passed=passed,
        )

    @staticmethod
    def evaluate_sop2_cd(
        candidate_id: str,
        scaffold_class: str,
        f_helix: float,
        f_sheet: float,
        tm_celsius: float,
    ) -> SOP2Report:
        """Evaluates Far-UV CD and thermal denaturation according to SOP-2 (Section 7.3.2).
        
        Hypothesis T1: Nilpotency ensures folded secondary structure & Tm >= 60 C.
        """
        tm_ok = (tm_celsius >= 60.0)
        if scaffold_class in ("de_novo_minibinder", "long_helical_bundle", "single_chain"):
            sec_ok = (f_helix >= 0.35)
        elif scaffold_class == "nanobody":
            sec_ok = (f_sheet >= 0.30)
        else:
            sec_ok = (f_helix + f_sheet >= 0.40)

        passed = tm_ok and sec_ok
        return SOP2Report(
            candidate_id=candidate_id,
            scaffold_class=scaffold_class,
            f_alpha_helix=f_helix,
            f_beta_sheet=f_sheet,
            tm_celsius=tm_celsius,
            t1_foldability_confirmed=passed,
            acceptance_passed=passed,
        )

    @staticmethod
    def evaluate_sop3_dsf(
        candidate_id: str,
        tm_74: float,
        tm_60: float,
        his_count: int,
    ) -> SOP3Report:
        """Evaluates high-throughput DSF melt curves according to SOP-3 (Section 7.3.3).
        
        Delta Tm = Tm(6.0) - Tm(7.4) <= -10 C confirms percolation-driven pH switch.
        """
        delta_tm = round(tm_60 - tm_74, 2)
        confirmed = (delta_tm <= -10.0) if his_count >= 4 else (abs(delta_tm) < 5.0)

        return SOP3Report(
            candidate_id=candidate_id,
            tm_74_celsius=tm_74,
            tm_60_celsius=tm_60,
            delta_tm_celsius=delta_tm,
            his_count=his_count,
            percolation_destabilization_confirmed=confirmed,
        )

    @staticmethod
    def evaluate_sop4_cryoem(
        candidate_id: str,
        resolution_angstrom: float,
        predicted_beta1: int,
        interface_coords: Dict[str, Tuple[float, float, float]],
        contact_cutoff: float = 5.0,
    ) -> SOP4Report:
        """Reconstructs empirical DACC complex from refined Cryo-EM coordinates (SOP-4)."""
        dacc_exp = DACCComplex(f"CryoEM_{candidate_id}")
        nodes = sorted(interface_coords.keys())
        for n in nodes:
            dacc_exp.add_0cell(n)

        # Build contact edges
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                cu, cv = interface_coords[u], interface_coords[v]
                d = math.sqrt(sum((a - b)**2 for a, b in zip(cu, cv)))
                if d <= contact_cutoff:
                    dacc_exp.add_1cell(f"e_{u}_{v}", source=u, target=v)

        sig = TopologicalDACCInvariant.compute_signature(dacc_exp)
        exp_beta1 = sig.beta_1
        match = (exp_beta1 == predicted_beta1)

        return SOP4Report(
            candidate_id=candidate_id,
            resolution_angstrom=resolution_angstrom,
            predicted_beta1=predicted_beta1,
            experimental_beta1=exp_beta1,
            topological_match=match,
            falsification_detected=not match,
        )
