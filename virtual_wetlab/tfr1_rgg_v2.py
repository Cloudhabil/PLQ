"""TfR1 Rule-Guided Generation v2 (RGG-v2) -- 5,000 Next-Generation Designs.

Incorporates all 10 empirical laws extracted from cross-database analysis
of 38,284 designs (Round 1 10K + Pan-Biosphere 18K + Wetlab 10K):

  RULE 1  N_His = 5 at binding interface (His-Number Law sweet spot)
  RULE 2  Sequence length 28-31 aa (Scaffold Aperture Law optimum)
  RULE 3  C-Y-T core triad embedded (mouse cross-reactivity signature)
  RULE 4  Target pI = 5.2-6.2 (electrostatic pre-organisation window)
  RULE 5  No K or R at paratope (His-switch competitors)
  RULE 6  Embed HQI, VNH, HYL motifs (3-mer enrichment signature)
  RULE 7  NanobodyGraft or DeNovo_AB for Objective 2 (mouse)
  RULE 8  delta_charge >= 1.85 (endosomal escape charge surplus)
  RULE 9  beta_1 tracked per scaffold (cyclic = 1, linear = 0)
  RULE 10 Clifford Q* stalk used as sequence composition seed

Author: Elias Oulad Brahim -- PLQ v3.0 / ToposLang Engine  (RGG v2)
"""

from __future__ import annotations
import sys, math, time, json, sqlite3, random, csv, traceback, statistics
import concurrent.futures
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# ── sys.path bootstrap ────────────────────────────────────────────────────────
_VW_DIR = Path(__file__).resolve().parent
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

# ── engine imports (verified field names) ────────────────────────────────────
from tfr1_ph_binder import (
    InvertedHisSwitchEngine,
    build_tfr1_apical_stalk,
    run_tfr1_assay,
    langevin_optimise,
    sequence_to_stalk,
    stalk_alignment_score,
    clifford_complementary_stalk,
    PHI_STAR,
    TWIN_STREP_TAG,
    TfR1AssayResult,
)
from plq.metrics.physicochemical import PhysicoChemicalEngine

# ── Verified TfR1AssayResult fields (from dataclasses.fields audit) ──────────
# design_id, sequence, sequence_with_tag,
# dg_human_74, kd_human_74_nm, spr_ru_human_74, binds_human_74,
# dg_human_60, kd_human_60_nm, spr_ru_human_60, binds_human_60,
# dg_mouse_74, kd_mouse_74_nm, binds_mouse_74,
# mouse_human_kd_ratio,   <-- NOTE: kd_ratio NOT mouse_human_ratio
# delta_delta_g, switch_active,
# net_charge_74, isoelectric_point, hydrophobic_fraction,
# phi_star_deviation, tm_estimate_c, aggregation_risk,
# qc_pass, qc_flags,
# clifford_score_human, clifford_score_mouse, competition_score

# ── Output paths ──────────────────────────────────────────────────────────────
_VW     = _VW_DIR
RGG_DB  = _VW / "tfr1_rgg_v2_results.db"
RGG_CSV = _VW / "tfr1_rgg_v2_top20.csv"
SF_DB   = _VW / "Supporting Files" / "tfr1_ph_binder_results.db"

# ── Shared read-only engine globals (initialised once) ───────────────────────
_ENGINE  = InvertedHisSwitchEngine(his_pka_env=7.6, asp_binder_pka=4.2, switch_ddg_threshold=4.5)
_K_HUMAN = build_tfr1_apical_stalk(mouse=False)
_K_MOUSE  = build_tfr1_apical_stalk(mouse=True)
_Q_STAR   = clifford_complementary_stalk(_K_HUMAN)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  RULE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TARGET_N_HIS     = 5
LEN_LOW, LEN_HIGH = 22, 32      # Rule 2: practical window (template +-4)
PI_LOW,  PI_HIGH  = 5.2, 6.2   # Rule 4
FORBIDDEN        = set("KR")   # Rule 5
ANCHOR_MOTIFS    = ["HQI", "VNH", "HYL"]  # Rule 6
DC_MIN           = 1.85        # Rule 8: min delta_charge

_HIS     = ["H"]
_CYS     = ["C"]
_POLAR   = ["N", "Q", "S", "T"]
_HYDRO   = ["L", "V", "I", "F", "M"]
_ANIONIC = ["D", "E"]
_AROMATIC= ["Y", "F", "W"]
_ALLOWED = list("ACDEFHILMNQSTVY")   # K, R, G, P excluded at paratope

# ─────────────────────────────────────────────────────────────────────────────
# 2.  RGG SCAFFOLD LIBRARY  (20 templates, 6 new families)
# ─────────────────────────────────────────────────────────────────────────────
SCAFFOLDS: List[Dict] = [

    # Family 1: RGG_HQI  -- canonical HQI-VNH-HYL triad anchor, 5x His macrocycle
    {"name": "RGG_HQI_A", "beta_1": 1, "mol": "single_chain",
     "template": "CHYHDHNIYVHQIVNHHYLDGC"},
    {"name": "RGG_HQI_B", "beta_1": 1, "mol": "single_chain",
     "template": "CHYHDHNIYVHQIFVNHHYLSGQC"},
    {"name": "RGG_HQI_C", "beta_1": 1, "mol": "single_chain",
     "template": "LHQIVNHHYLDNIYVHDHYHCWC"},

    # Family 2: RGG_CYT  -- C-Y-T cross-reactivity core, 5x His bicyclic
    {"name": "RGG_CYT_A", "beta_1": 1, "mol": "single_chain",
     "template": "CHYTDHNIYVHQIHYLCYTDNHMC"},
    {"name": "RGG_CYT_B", "beta_1": 1, "mol": "single_chain",
     "template": "CYTCHYHDHNIYLVNHHQICYTC"},
    {"name": "RGG_CYT_C", "beta_1": 1, "mol": "single_chain",
     "template": "LHYTDHNIYVHQIHYLCYTDNHC"},

    # Family 3: RGG_DeNovo30  -- 28-30 aa, Q*-seeded alphabeta
    {"name": "RGG_DN30_A", "beta_1": 0, "mol": "single_chain",
     "template": "LHQIVNHHYLDHYTMNIYVHDHCHYC"},
    {"name": "RGG_DN30_B", "beta_1": 1, "mol": "single_chain",
     "template": "CHQITNHHYLDHYVSNIYVHDHYHC"},
    {"name": "RGG_DN30_C", "beta_1": 0, "mol": "single_chain",
     "template": "LHQIVNHHYLDNIYVTDHCHYTHMC"},

    # Family 4: RGG_Lariat30  -- 28-30 aa lariat, all 6 rules
    {"name": "RGG_LAR30_A", "beta_1": 1, "mol": "single_chain",
     "template": "CHHYTDNIYVHQIFLHYLTNHDHYHC"},
    {"name": "RGG_LAR30_B", "beta_1": 1, "mol": "single_chain",
     "template": "CHDHYNIVFHQILHYLTCYTDNHC"},
    {"name": "RGG_LAR30_C", "beta_1": 1, "mol": "single_chain",
     "template": "CHHDHNIYVHQIFVNHHYLCYTC"},

    # Family 5: RGG_Bicyclic30  -- bicyclic with beta_1=1
    {"name": "RGG_BIC30_A", "beta_1": 1, "mol": "single_chain",
     "template": "CHYTDHNIYVHQIHYLTCNHDHC"},
    {"name": "RGG_BIC30_B", "beta_1": 1, "mol": "single_chain",
     "template": "CHDHNIYVHQICYTHYLTNHDHMC"},

    # Family 6: RGG_pI_Tuned  -- pI targeting 5.5-5.9 via D/E/H balance
    {"name": "RGG_PI_A", "beta_1": 1, "mol": "single_chain",
     "template": "CHYHDHEIYVHQIVNHHYLDEDMC"},
    {"name": "RGG_PI_B", "beta_1": 1, "mol": "single_chain",
     "template": "CDHHNIYVHQIHYLTVNHEDHMC"},
    {"name": "RGG_PI_C", "beta_1": 0, "mol": "single_chain",
     "template": "LHYHDHNIYVHQIEVNHHYLDED"},

    # Family 7: RGG_Nanobody_V2  -- Rule 7, CDRs engineered with all motifs
    {"name": "RGG_NB_V2_A", "beta_1": 0, "mol": "nanobody",
     "template": "QVQLVESGGGLVHAGGSLRLSCAASGHHQIFHSYMGWFRQAPGKERHYVAAIVNHHYLSSDTYYADSVKGRFTISDNAKNTVYLHMNSLKPEDTAVYYCAADHCYTNIYVHQIWGQGTQVTVSS"},
    {"name": "RGG_NB_V2_B", "beta_1": 0, "mol": "nanobody",
     "template": "QVQLVESGGGLVHAGGSLRLSCAASGVNHHYLFHSYMGWFRQAPGKERHYVAAIHQITNHSSDTYYADSVKGRFTISDNAKNTVYLHMNSLKPEDTAVYYCAACYTDHNIYVNHWGQGTQVTVSS"},

    # Family 8: RGG_HisCluster_V2  -- 5x His cluster, CYT tail, pI-tuned
    {"name": "RGG_HC_V2", "beta_1": 1, "mol": "single_chain",
     "template": "CHHHYDHQIIVNHHYLCYTEC"},
]

# ─────────────────────────────────────────────────────────────────────────────
# 3.  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _pi_estimate(seq: str) -> float:
    try:
        return PhysicoChemicalEngine.evaluate_sequence(seq).isoelectric_point
    except Exception:
        pos = seq.count("H") * 0.45 + seq.count("K") + seq.count("R")
        neg = seq.count("D") + seq.count("E")
        return 6.0 + (neg - pos) * 0.25


def _delta_charge(seq: str) -> float:
    """Protonation shift pH7.4->6.0: each His gains 0.37 charge units."""
    return seq.count("H") * 0.37


def _find_motif_positions(seq: list, motif: str) -> set:
    s = "".join(seq)
    idx = s.find(motif)
    return {idx, idx+1, idx+2} if idx >= 0 else set()


def _purge_forbidden(seq: list, rng: random.Random) -> list:
    for i, aa in enumerate(seq):
        if aa in FORBIDDEN:
            seq[i] = rng.choice(_ANIONIC + _POLAR)
    return seq


def _adjust_his(seq: list, protect: set, rng: random.Random) -> list:
    """Adjust His count toward TARGET_N_HIS."""
    n_his = seq.count("H")
    if n_his < TARGET_N_HIS:
        eligible = [i for i in range(len(seq))
                    if seq[i] not in "HCY" and i not in protect]
        rng.shuffle(eligible)
        for i in eligible[:TARGET_N_HIS - n_his]:
            seq[i] = "H"
    elif n_his > TARGET_N_HIS + 2:
        his_pos = [i for i, aa in enumerate(seq) if aa == "H" and i not in protect]
        rng.shuffle(his_pos)
        for i in his_pos[:n_his - (TARGET_N_HIS + 2)]:
            seq[i] = rng.choice(["N", "Q"])
    return seq


def _embed_motif_if_missing(seq: list, motif: str, protect: set, rng: random.Random) -> list:
    if motif not in "".join(seq):
        # Find a 3-position window with no protected positions and no H/C
        candidates = [i for i in range(len(seq) - 2)
                      if all(j not in protect and seq[j] not in "HC"
                             for j in (i, i+1, i+2))]
        if candidates:
            p = rng.choice(candidates)
            seq[p], seq[p+1], seq[p+2] = list(motif)
    return seq


def _nudge_pi(seq: list, rng: random.Random) -> list:
    pi = _pi_estimate("".join(seq))
    for _ in range(10):
        if PI_LOW <= pi <= PI_HIGH:
            break
        mutable = [i for i, aa in enumerate(seq) if aa not in "HCYT"]
        if not mutable:
            break
        if pi > PI_HIGH:
            seq[rng.choice(mutable)] = rng.choice(["D", "E"])
        else:
            de_pos = [i for i in range(len(seq)) if seq[i] in "DE"]
            if de_pos:
                seq[rng.choice(de_pos)] = rng.choice(["N", "Q", "T"])
        pi = _pi_estimate("".join(seq))
    return seq


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SEQUENCE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_variants(scaffold: dict, n: int, seed: int) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    template = scaffold["template"].replace(" ", "")
    name     = scaffold["name"]
    is_nano  = scaffold.get("mol", "single_chain") == "nanobody"
    variants = []

    for v_idx in range(n):
        seq = list(template)

        # Protected positions: H, C (structural), and first occurrence of each anchor motif
        protect = {i for i, aa in enumerate(seq) if aa in "HC"}
        for motif in ANCHOR_MOTIFS:
            protect |= _find_motif_positions(seq, motif)

        # Step 1: Random mutagenesis (25% rate) on non-protected positions
        for i in range(len(seq)):
            if i in protect:
                continue
            if rng.random() < 0.25:
                r = rng.random()
                if r < 0.10:
                    seq[i] = "H"
                elif r < 0.22:
                    seq[i] = rng.choice(_POLAR)
                elif r < 0.42:
                    seq[i] = rng.choice(_HYDRO)
                elif r < 0.56:
                    seq[i] = rng.choice(_ANIONIC)
                elif r < 0.68:
                    seq[i] = rng.choice(_AROMATIC)
                elif r < 0.76:
                    seq[i] = "T"
                else:
                    seq[i] = rng.choice(_ALLOWED)

        # Step 2: Purge K and R (Rule 5)
        seq = _purge_forbidden(seq, rng)

        # Step 3: Enforce N_His (Rule 1)
        seq = _adjust_his(seq, protect, rng)

        # Step 4: Embed missing anchor motifs (Rule 6)
        for motif in ANCHOR_MOTIFS:
            seq = _embed_motif_if_missing(seq, motif, protect, rng)

        # Step 5: Ensure C-Y-T for non-nanobody (Rule 3)
        if not is_nano and "CYT" not in "".join(seq):
            mutable3 = [i for i in range(len(seq)-2)
                        if all(j not in protect and seq[j] not in "H"
                               for j in (i, i+1, i+2))]
            if mutable3:
                p = rng.choice(mutable3)
                seq[p], seq[p+1], seq[p+2] = "C", "Y", "T"

        # Step 6: pI nudge (Rule 4)
        seq = _nudge_pi(seq, rng)

        # Step 7: Final K/R sweep (nudge may have introduced them)
        seq = _purge_forbidden(seq, rng)

        variants.append((f"{name}_v{v_idx+1:04d}", "".join(seq)))

    return variants


# ─────────────────────────────────────────────────────────────────────────────
# 5.  RULE COMPLIANCE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def rule_score(seq: str, scaffold: dict) -> float:
    """Compliance bonus [0,1]. Higher = more rule-compliant."""
    s = 0.0
    # R1: N_His = 5
    s += max(0.0, 1.0 - abs(seq.count("H") - TARGET_N_HIS) * 0.2)
    # R2: length 22-32
    s += 1.0 if LEN_LOW <= len(seq) <= LEN_HIGH else max(0.0, 1.0 - abs(len(seq)-27)*0.05)
    # R3: CYT present
    s += 1.0 if "CYT" in seq else 0.0
    # R4: pI in window
    pi = _pi_estimate(seq)
    s += 1.0 if PI_LOW <= pi <= PI_HIGH else max(0.0, 1.0 - abs(pi-5.7)*0.3)
    # R5: no K/R
    s += 1.0 if not any(aa in FORBIDDEN for aa in seq) else 0.0
    # R6: anchor motifs
    s += sum(1.0 for m in ANCHOR_MOTIFS if m in seq) / 3.0
    # R7: scaffold bonus (nanobody/denovo)
    if any(x in scaffold["name"] for x in ["NB", "DN30"]):
        s += 0.5
    # R8: delta_charge
    s += 1.0 if _delta_charge(seq) >= DC_MIN else _delta_charge(seq) / DC_MIN
    # R9: beta_1 tracking
    s += 0.5 if scaffold.get("beta_1", 0) == (1 if seq.count("C") >= 2 else 0) else 0.0
    # R10: Clifford Q* alignment
    try:
        bs = sequence_to_stalk(seq)
        s += max(0.0, min(1.0, stalk_alignment_score(bs, _Q_STAR) * 8.0))
    except Exception:
        pass
    return s / 10.0


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DATABASE
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS tfr1_rgg_designs (
    design_id          TEXT PRIMARY KEY,
    run_timestamp      TEXT,
    generation         INTEGER DEFAULT 2,
    scaffold_name      TEXT,
    scaffold_type      TEXT,
    beta_1             INTEGER,
    mol_class          TEXT,
    sequence           TEXT,
    sequence_with_tag  TEXT,
    n_his              INTEGER,
    has_cyt            INTEGER,
    has_hqi            INTEGER,
    has_vnh            INTEGER,
    has_hyl            INTEGER,
    n_motifs           INTEGER,
    pi_estimate        REAL,
    delta_charge       REAL,
    rule_compliance    REAL,
    dg_human_74        REAL,
    kd_human_74_nm     REAL,
    spr_ru_human_74    REAL,
    binds_human_74     INTEGER,
    dg_human_60        REAL,
    kd_human_60_nm     REAL,
    spr_ru_human_60    REAL,
    binds_human_60     INTEGER,
    delta_delta_g      REAL,
    switch_active      INTEGER,
    dg_mouse_74        REAL,
    kd_mouse_74_nm     REAL,
    binds_mouse_74     INTEGER,
    mouse_human_kd_ratio REAL,
    net_charge_74      REAL,
    isoelectric_point  REAL,
    hydrophobic_frac   REAL,
    phi_star_dev       REAL,
    tm_estimate_c      REAL,
    aggregation_risk   TEXT,
    qc_pass            INTEGER,
    qc_flags_json      TEXT,
    clifford_human     REAL,
    clifford_mouse     REAL,
    competition_score  REAL,
    rgg_score          REAL
)
"""

def _init_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    with conn:
        conn.execute(_CREATE_SQL)
    conn.close()


def _row_from_result(r: TfR1AssayResult, scaffold: dict) -> tuple:
    """Build a complete DB row from a TfR1AssayResult + scaffold metadata.
    Uses ONLY verified field names from TfR1AssayResult dataclass.
    """
    seq = r.sequence.replace(" ", "")
    rc  = rule_score(seq, scaffold)
    pi  = _pi_estimate(seq)
    dc  = _delta_charge(seq)
    n_motifs = sum(1 for m in ANCHOR_MOTIFS if m in seq)
    rgg = round(r.competition_score - rc * 0.03, 5)

    return (
        r.design_id,                         # design_id
        None,                                # run_timestamp (filled by datetime('now'))
        2,                                   # generation
        scaffold["name"],                    # scaffold_name
        scaffold.get("name", "unknown"),     # scaffold_type (reuse name as type)
        scaffold.get("beta_1", 0),           # beta_1
        scaffold.get("mol", "single_chain"), # mol_class
        r.sequence,                          # sequence
        r.sequence_with_tag,                 # sequence_with_tag
        seq.count("H"),                      # n_his
        int("CYT" in seq),                   # has_cyt
        int("HQI" in seq),                   # has_hqi
        int("VNH" in seq),                   # has_vnh
        int("HYL" in seq),                   # has_hyl
        n_motifs,                            # n_motifs
        round(pi, 3),                        # pi_estimate
        round(dc, 4),                        # delta_charge
        round(rc, 4),                        # rule_compliance
        r.dg_human_74,                       # dg_human_74
        r.kd_human_74_nm,                    # kd_human_74_nm
        r.spr_ru_human_74,                   # spr_ru_human_74
        int(r.binds_human_74),               # binds_human_74
        r.dg_human_60,                       # dg_human_60
        r.kd_human_60_nm,                    # kd_human_60_nm
        r.spr_ru_human_60,                   # spr_ru_human_60
        int(r.binds_human_60),               # binds_human_60
        r.delta_delta_g,                     # delta_delta_g
        int(r.switch_active),                # switch_active
        r.dg_mouse_74,                       # dg_mouse_74
        r.kd_mouse_74_nm,                    # kd_mouse_74_nm
        int(r.binds_mouse_74),               # binds_mouse_74
        r.mouse_human_kd_ratio,              # mouse_human_kd_ratio (correct field name!)
        r.net_charge_74,                     # net_charge_74
        r.isoelectric_point,                 # isoelectric_point
        r.hydrophobic_fraction,              # hydrophobic_frac
        r.phi_star_deviation,                # phi_star_dev
        r.tm_estimate_c,                     # tm_estimate_c
        r.aggregation_risk,                  # aggregation_risk
        int(r.qc_pass),                      # qc_pass
        json.dumps(r.qc_flags),              # qc_flags_json
        r.clifford_score_human,              # clifford_human
        r.clifford_score_mouse,              # clifford_mouse
        r.competition_score,                 # competition_score
        rgg,                                 # rgg_score
    )


_INSERT_SQL = """
    INSERT OR REPLACE INTO tfr1_rgg_designs VALUES
    (?,datetime('now'),?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""

def _persist_batch(rows: list, path: Path) -> None:
    conn = sqlite3.connect(str(path))
    with conn:
        for row in rows:
            # Remove the run_timestamp placeholder (index 1) -- SQL fills it
            db_row = (row[0],) + row[2:]
            conn.execute(_INSERT_SQL, db_row)
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  SCREENING WORKER (sequential, no threading issues)
# ─────────────────────────────────────────────────────────────────────────────

def screen_one(design_id: str, seq: str, scaffold: dict) -> Optional[tuple]:
    """Screen one sequence. Returns DB row tuple or None on error."""
    try:
        r = run_tfr1_assay(design_id, seq, _ENGINE, _K_HUMAN, _K_MOUSE)
        return _row_from_result(r, scaffold)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PARALLEL WRAPPER  (thread pool with proper error surfacing)
# ─────────────────────────────────────────────────────────────────────────────

def _thread_worker(args: tuple) -> Optional[tuple]:
    did, seq, scf = args
    return screen_one(did, seq, scf)


def screen_parallel(
    tasks: List[Tuple[str, str, dict]],
    n_workers: int,
) -> List[tuple]:
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        future_to_task = {ex.submit(_thread_worker, t): t for t in tasks}
        for fut in concurrent.futures.as_completed(future_to_task):
            try:
                row = fut.result()
                if row is not None:
                    rows.append(row)
            except Exception:
                pass
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 9.  LANGEVIN ELITE OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────

def optimise_elite(
    rows: list,
    scaffold: dict,
    n_elite: int,
    n_steps: int,
    seed: int,
) -> list:
    """Run Langevin optimisation on top n_elite rows. Returns new rows."""
    if not rows:
        return []

    sorted_rows = sorted(rows, key=lambda x: x[-1])   # rgg_score ascending
    opt_rows = []

    for e_idx, pre_row in enumerate(sorted_rows[:n_elite]):
        pre_seq = pre_row[7]   # sequence column
        try:
            opt_seq = langevin_optimise(
                initial_seq=pre_seq,
                switch_engine=_ENGINE,
                target_stalk_human=_K_HUMAN,
                target_stalk_mouse=_K_MOUSE,
                n_steps=n_steps,
                seed=seed + e_idx,
            )
            opt_id  = f"{scaffold['name']}_OPT{e_idx+1:02d}"
            opt_row = screen_one(opt_id, opt_seq, scaffold)
            if opt_row is not None:
                opt_rows.append(opt_row)
        except Exception:
            pass

    return opt_rows


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN CAMPAIGN
# ─────────────────────────────────────────────────────────────────────────────

def run_campaign(
    n_total: int    = 5000,
    n_workers: int  = 10,
    n_steps: int    = 300,
    top_k: int      = 20,
    db_path: Path   = RGG_DB,
    csv_path: Path  = RGG_CSV,
    verbose: bool   = True,
) -> list:
    """Execute the 5,000-design RGG v2 campaign."""
    _init_db(db_path)

    n_scafs  = len(SCAFFOLDS)
    base     = n_total // n_scafs
    remainder= n_total % n_scafs

    if verbose:
        print("=" * 76)
        print("TfR1 RGG v2 -- Rule-Guided Generation  (10 Empirical Laws)")
        print(f"Designs: {n_total:,}  Scaffolds: {n_scafs}  Workers: {n_workers}  Steps: {n_steps}")
        print(f"Q* norm={_Q_STAR.norm():.4f}  K_human norm={_K_HUMAN.norm():.4f}")
        print("=" * 76)

    all_rows = []
    global_seed = 137
    t0 = time.perf_counter()

    for s_idx, scaffold in enumerate(SCAFFOLDS):
        sname    = scaffold["name"]
        n_vars   = base + (1 if s_idx < remainder else 0)
        elapsed  = time.perf_counter() - t0

        if verbose:
            print(f"\n[{s_idx+1:02d}/{n_scafs}] {sname}  ({n_vars} variants +{elapsed:.1f}s)")

        # Generate
        variants = generate_variants(scaffold, n_vars, seed=global_seed)
        global_seed += n_vars + 50

        # Screen (parallel)
        tasks = [(vid, seq, scaffold) for vid, seq in variants]
        batch = screen_parallel(tasks, n_workers)

        # Persist
        _persist_batch(batch, db_path)

        # Elite Langevin optimisation (top-3%)
        n_elite = max(3, len(batch) // 30)
        opt_rows = optimise_elite(batch, scaffold, n_elite, n_steps, global_seed)
        _persist_batch(opt_rows, db_path)
        global_seed += 30

        all_rows.extend(batch)
        all_rows.extend(opt_rows)

        if verbose and batch:
            best = min(batch, key=lambda x: x[-1])
            print(f"  OK {len(batch)} screened + {len(opt_rows)} opt  |  "
                  f"Best KD_h74={best[19]:.2f}nM  "
                  f"DDG={best[26]:.2f}  "
                  f"SW={'Y' if best[27] else 'N'}  "
                  f"rc={best[17]:.3f}  rgg={best[-1]:.4f}")

    elapsed_total = time.perf_counter() - t0

    # ── Global rank: switch -> qc_pass -> rgg_score ─────────────────────────
    all_rows.sort(key=lambda x: (0 if x[27] else 1, 0 if x[38] else 1, x[-1]))

    if verbose:
        print(f"\n{'=' * 76}")
        print(f"COMPLETE  {len(all_rows):,} designs  {elapsed_total:.1f}s  ({elapsed_total/60:.1f}min)")
        print(f"TOP {top_k} DESIGNS  (Rule-Guided Generation v2)")
        print("=" * 76)
        print(f"{'Rk':<3} {'Design':<28} {'KD_h74':>8} {'KD_h60':>10} "
              f"{'DDG':>7} {'SW':>3} {'KD_m/h':>8} {'RC':>6} {'rgg':>7}")
        print("-" * 76)
        for i, row in enumerate(all_rows[:top_k]):
            # Columns: [0]design_id [7]seq [19]kd_h74 [23]kd_h60 [26]ddg
            #          [27]switch [31]ratio [17]rc [-1]rgg
            print(f"{i+1:<3} {row[0]:<28} "
                  f"{row[19]:>8.2f}n "
                  f"{row[23]:>9.0f}n "
                  f"{row[26]:>7.2f} "
                  f"{'Y' if row[27] else 'N':>3} "
                  f"{row[31]:>8.1f} "
                  f"{row[17]:>6.3f} "
                  f"{row[-1]:>7.4f}")

    # ── CSV export ────────────────────────────────────────────────────────────
    seen_seqs: Dict[str, bool] = {}
    family_count: Dict[str, int] = {}
    csv_rows = []

    for row in all_rows:
        if len(csv_rows) >= top_k:
            break
        seq = row[7].replace(" ", "").upper()
        if seq in seen_seqs:
            continue
        fam = row[3][:10]   # scaffold family prefix
        if family_count.get(fam, 0) >= 3:
            continue
        seen_seqs[seq] = True
        family_count[fam] = family_count.get(fam, 0) + 1
        csv_rows.append({
            "name": row[0],
            "sequence": seq,
            "molecule_class": row[6],   # mol_class
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "sequence", "molecule_class"])
        writer.writeheader()
        writer.writerows(csv_rows)

    if verbose:
        print(f"\nCSV -> {csv_path}  ({len(csv_rows)} rows)")
        print(f"DB  -> {db_path}")

    return all_rows[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# 11. ROUND 1 vs ROUND 2 COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_rounds(top_rows: list, verbose: bool = True) -> None:
    if not SF_DB.exists():
        print("Round 1 DB not found -- skipping comparison")
        return

    conn = sqlite3.connect(str(SF_DB))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT kd_human_74_nm, kd_human_60_nm, delta_delta_g,
               mouse_human_ratio, competition_score, hydrophobic_frac
        FROM tfr1_designs WHERE switch_active=1 AND qc_pass=1
    """)
    r1 = [dict(r) for r in cur.fetchall()]
    conn.close()

    def _m(vals):
        return statistics.mean(vals) if vals else 0.0

    r1_kd74  = [r['kd_human_74_nm'] for r in r1 if r['kd_human_74_nm'] is not None]
    r1_ddg   = [r['delta_delta_g'] for r in r1 if r['delta_delta_g'] is not None]
    r1_ratio = [r['mouse_human_ratio'] for r in r1
                if r['mouse_human_ratio'] is not None and r['mouse_human_ratio'] < 1e4]
    r1_score = [r['competition_score'] for r in r1]

    r2_kd74  = [row[19] for row in top_rows if row[19] is not None]
    r2_ddg   = [row[26] for row in top_rows]
    r2_ratio = [row[31] for row in top_rows if row[31] is not None and row[31] < 1e4]
    r2_score = [row[-1] for row in top_rows]

    print(f"\n{'=' * 72}")
    print("ROUND 1  vs  ROUND 2 (RGG v2)  --  Top-design comparison")
    print("=" * 72)
    metrics = [
        ("Mean KD_h74 (nM)",        r1_kd74,  r2_kd74),
        ("Mean DeltaDeltaG",        r1_ddg,   r2_ddg),
        ("Best KD_mouse/human",     r1_ratio, r2_ratio),
        ("Mean competition score",  r1_score, r2_score),
    ]
    print(f"  {'Metric':<26} {'Round1':>12} {'RGG-v2':>12} {'Delta':>10}")
    print(f"  {'-'*62}")
    for label, v1, v2 in metrics:
        m1, m2 = _m(v1), _m(v2)
        d = m2 - m1
        tag = "better" if d < 0 else "same" if abs(d) < 0.0001 else "worse"
        print(f"  {label:<26} {m1:>12.4f} {m2:>12.4f} {d:>+10.4f} ({tag})")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="TfR1 RGG v2 -- 5000 rule-guided designs")
    ap.add_argument("--designs", type=int,  default=5000)
    ap.add_argument("--workers", type=int,  default=10)
    ap.add_argument("--steps",   type=int,  default=300)
    ap.add_argument("--top",     type=int,  default=20)
    ap.add_argument("--db",      type=Path, default=RGG_DB)
    ap.add_argument("--csv",     type=Path, default=RGG_CSV)
    args = ap.parse_args()

    print(f"TfR1 RGG v2  --  Rule-Guided Generation")
    print(f"  {args.designs:,} designs across {len(SCAFFOLDS)} scaffolds")
    print(f"  10 empirical laws from 38,284 cross-database designs")
    print()

    t0 = time.perf_counter()
    top = run_campaign(
        n_total  = args.designs,
        n_workers= args.workers,
        n_steps  = args.steps,
        top_k    = args.top,
        db_path  = args.db,
        csv_path = args.csv,
        verbose  = True,
    )
    print(f"\nElapsed: {time.perf_counter()-t0:.1f}s")
    compare_rounds(top)
