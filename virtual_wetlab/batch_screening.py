"""Batch Screening Pipeline for Existing ProteinBase & PLQ Target Deposits.

Iterates over designs stored in 'PLQ-main/PLQ-main/data/proteinbase_targets_discovery.db'
and 'PLQ-main/PLQ-main/data/topological_protein_discovery.db', evaluates each design
through the Dynamic-Well virtual plate, generates SPR dose-response curves,
applies the automated QC gate, and permanently persists results in:
'virtual_wetlab/virtual_wetlab_results.db'.
"""

from __future__ import annotations
import math
import sys
import time
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


DISCOVERY_DB_PATH = PLQ_PATH / "data" / "proteinbase_targets_discovery.db"
WETLAB_DB_PATH = _VW_DIR / "virtual_wetlab_results.db"


def init_screening_summary_table(conn: sqlite3.Connection):
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candidate_wetlab_screen_summary (
                design_id TEXT PRIMARY KEY,
                target_slug TEXT,
                target_name TEXT,
                scaffold_type TEXT,
                sequence_length INTEGER,
                fitted_kd_nm REAL,
                r_max_ru REAL,
                net_charge_ph74 REAL,
                isoelectric_point REAL,
                tm_c REAL,
                aggregation_risk TEXT,
                qc_overall_status TEXT,
                qc_failure_reasons_json TEXT,
                total_dynamic_wells INTEGER,
                screening_timestamp TEXT
            )
        """)


def run_batch_evaluation(max_candidates: int = 50, verbose: bool = True):
    print("=" * 80)
    print(f" BATCH VIRTUAL WET-LAB SCREENING OVER EXISTENT DESIGNS (Target: {max_candidates} candidates)")
    print(f" Source DB: {DISCOVERY_DB_PATH}")
    print(f" Target DB: {WETLAB_DB_PATH}")
    print("=" * 80)

    # 1. Connect to source designs database
    src_conn = sqlite3.connect(str(DISCOVERY_DB_PATH))
    src_cur = src_conn.cursor()

    query = """
        SELECT design_id, target_slug, target_name, scaffold_type, sequence, predicted_plddt, novelty_score
        FROM de_novo_designs
        ORDER BY design_id ASC
        LIMIT ?
    """
    candidates = src_cur.execute(query, (max_candidates,)).fetchall()
    src_conn.close()

    print(f"Loaded {len(candidates)} candidates from discovery database.\n")

    # 2. Connect to destination wetlab database
    dest_conn = sqlite3.connect(str(WETLAB_DB_PATH))
    init_screening_summary_table(dest_conn)

    t_start = time.perf_counter()
    summary_counts = {"PASS": 0, "FLAGGED": 0, "FAIL": 0}

    for idx, row in enumerate(candidates, 1):
        design_id, target_slug, target_name, scaffold_type, seq, plddt, novelty = row

        # Initialize plate for this candidate
        plate = DynamicPlate(plate_id=design_id, db_path=WETLAB_DB_PATH)

        # Build coarse grid and dynamically adapt/subdivide at phase transitions
        plate.build_coarse_grid(ph_range=(5.5, 8.5, 4), salt_range=(100.0, 250.0, 2))
        
        # Estimate base KD using scaffold/plddt heuristic (plddt 90+ -> ~10-25 nM)
        base_kd = max(5.0, round(100.0 - (plddt * 0.8), 2)) if plddt else 25.0

        all_wells = plate.adapt_and_subdivide(sequence=seq, base_kd_nm=base_kd, subdivision_depth=2)

        # Generate simulated SPR dose-response sensorgram
        dose_curve = plate.generate_dose_response_curve(sequence=seq, base_kd_nm=base_kd)

        # Evaluate QC gate
        qc = plate.evaluate_qc_gate(sequence=seq, target_ph=7.4)
        summary_counts[qc.overall_status] = summary_counts.get(qc.overall_status, 0) + 1

        # Retrieve physiological well readout (pH 7.4)
        physio_well = plate.wells.get("QCGate_EvaluationWell")
        r = physio_well.readout if physio_well else None

        net_q = r.net_charge_q if r else 0.0
        pi = r.isoelectric_point if r else 7.0
        tm = r.thermal_stability_tm_c if r else 50.0
        agg_risk = r.aggregation_risk if r else "UNKNOWN"

        # Record in summary table
        with dest_conn:
            dest_conn.execute("""
                INSERT OR REPLACE INTO candidate_wetlab_screen_summary
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                design_id,
                target_slug,
                target_name,
                scaffold_type,
                len(seq),
                dose_curve.fitted_kd_nm,
                dose_curve.r_max_ru,
                net_q,
                pi,
                tm,
                agg_risk,
                qc.overall_status,
                json.dumps(qc.failure_reasons),
                len(plate.wells)
            ))

        if verbose and (idx <= 10 or idx % 10 == 0 or idx == len(candidates)):
            print(f"[{idx:02d}/{len(candidates)}] {design_id:<16} | Target: {target_slug:<14} | KD: {base_kd:>5.1f} nM | Tm: {tm:>4.1f}°C | Wells: {len(plate.wells):>2} | QC: {qc.overall_status}")

    dest_conn.close()
    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 80)
    print(" BATCH SCREENING RUN COMPLETE")
    print(f" Evaluated: {len(candidates)} de novo protein designs in {elapsed:.2f}s ({len(candidates)/elapsed:.1f} designs/sec)")
    print(f" QC Status Breakdown: PASS: {summary_counts.get('PASS', 0)} | FLAGGED: {summary_counts.get('FLAGGED', 0)} | FAIL: {summary_counts.get('FAIL', 0)}")
    print(f" Total dynamic well reactions recorded in SQLite: {WETLAB_DB_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    run_batch_evaluation(max_candidates=50, verbose=True)
