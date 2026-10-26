"""Demonstration of In Silico Virtual Wet-Lab with Dynamic-Well Screening.

Simulates a therapeutic design candidate across an adaptive dynamic-well plate,
extracts an SPR dose-response binding curve (KD), and passes through automated QC Gating.
"""

from __future__ import annotations
import sys
import json
from pathlib import Path

_VW_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _VW_DIR
for _candidate in [_VW_DIR] + list(_VW_DIR.parents):
    if (_candidate / "ToposLang-main").exists() and (_candidate / "PLQ-main").exists():
        _WORKSPACE_ROOT = _candidate
        break

WORKSPACE_DIR = _WORKSPACE_ROOT
PLQ_PATH = _WORKSPACE_ROOT / "PLQ-main" / "PLQ-main"
TOPOS_PATH = _WORKSPACE_ROOT / "ToposLang-main" / "ToposLang-main" / "src"

for _p in [str(PLQ_PATH), str(TOPOS_PATH), str(_VW_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine import DynamicPlate, WellCoordinates

def run_demonstration():
    print("=" * 80)
    print(" VIRTUAL WET-LAB: DYNAMIC-WELL REACTION SCREENING & ASSAY SUITE")
    print("=" * 80)

    # Example candidate sequence: Dual-action Histidine-switch therapeutic binder
    candidate_id = "CANDIDATE_HIS_SWITCH_V1"
    sequence = "ACDEFGHIKLMNPQRSTVWY"  # 20-aa synthetic peptide containing Histidine switch

    print(f"\n1. Initializing Dynamic-Well Plate for Candidate: {candidate_id}")
    print(f"Sequence: {sequence} (Length: {len(sequence)} aa)")
    plate = DynamicPlate(plate_id=candidate_id)

    # 1. Coarse Grid Exploration
    print("\n2. Deploying Initial Coarse Multi-Parametric Grid (pH 5.0 to 8.5, Salt 50 to 300 mM)...")
    plate.build_coarse_grid(ph_range=(5.0, 8.5, 5), salt_range=(50.0, 300.0, 3))
    print(f"Coarse wells initialized: {len(plate.wells)} wells")

    # 2. Adaptive Sub-Well Fission
    print("\n3. Running Assays & Triggering Dynamic Sub-Well Fission at Phase Transitions...")
    all_wells = plate.adapt_and_subdivide(sequence=sequence, base_kd_nm=18.5, subdivision_depth=3)
    print(f"Total Wells after Dynamic Sub-Division: {len(plate.wells)} wells")

    # Display some wells
    print("\nSample Readouts from Dynamic Wells:")
    print("-" * 80)
    print(f"{'Well ID':<30} | {'pH':>4} | {'Salt':>5} | {'Net Q':>6} | {'KD (nM)':>7} | {'Tm (°C)':>7} | {'QC'}")
    print("-" * 80)
    for w in list(plate.wells.values())[:8]:
        r = w.readout
        status = "PASS" if r.qc_pass else "FLAG"
        print(f"{r.well_id:<30} | {r.coords.ph:>4.2f} | {r.coords.salt_conc_mm:>5.0f} | {r.net_charge_q:>+6.2f} | {r.estimated_kd_nm:>7.1f} | {r.thermal_stability_tm_c:>7.1f} | {status}")

    # 3. Simulated SPR Dose-Response Curve
    print("\n4. Generating Simulated Surface Plasmon Resonance (SPR) Dose-Response Curve...")
    dose_curve = plate.generate_dose_response_curve(sequence=sequence, base_kd_nm=18.5)
    print(f"SPR Sensorgram Fitted KD: {dose_curve.fitted_kd_nm} nM (R_max: {dose_curve.r_max_ru} RU, R²: {dose_curve.r_squared})")
    print(f"Concentrations (nM): {dose_curve.concentrations_nm[:5]}...")
    print(f"Responses (RU):      {dose_curve.response_ru[:5]}...")

    # 4. Automated QC Gating
    print("\n5. Running Automated Laboratory QC Gating Pipeline...")
    qc_result = plate.evaluate_qc_gate(sequence=sequence, target_ph=7.4)
    print(f"Overall QC Status: {qc_result.overall_status}")
    print(f"- Isoelectric Precipitation Risk: {qc_result.isoelectric_risk}")
    print(f"- Solubility Risk:                {qc_result.solubility_risk}")
    print(f"- Aggregation Propensity:         {qc_result.aggregation_risk}")
    print(f"- Affinity Viability (KD < 500nM):{qc_result.affinity_viable}")
    print(f"- Thermal Stability (Tm >= 55°C): {qc_result.thermal_viable}")
    if qc_result.failure_reasons:
        print("Flagged Reasons:")
        for reason in qc_result.failure_reasons:
            print(f"  * {reason}")

    print("\n" + "=" * 80)
    print(f" All experimental readouts safely recorded to SQLite:")
    print(f" File: virtual_wetlab/virtual_wetlab_results.db")
    print("=" * 80)

if __name__ == "__main__":
    run_demonstration()
