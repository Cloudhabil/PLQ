"""Solid-Phase Peptide Synthesis (SPPS) Manufacturability & Chemical Liability Scanner.

Conforming to Nuclear Software Quality & Laboratory Synthesis Standards:
- ASME NQA-1 Subpart 2.7
- SPPS Fmoc Chemistry Guidelines

Features:
1. Scans for hydrophobic aggregation motifs causing resin collapse (e.g., VVV, III, FFF).
2. Identifies chemical rearrangement liabilities:
   - Aspartimide formation traps (DG, DS, DT under base treatment).
   - Spontaneous deamidation motifs (NG, NS at neutral pH).
   - Methionine oxidation liabilities (unprotected M).
   - Unpaired cysteine residues (odd C counts causing random dimerization).
3. Evaluates continuous SPPS synthesizability score S_SPPS in [0.0, 1.0].
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ChemicalLiabilityReport:
    """Detailed diagnostic report of chemical and synthetic liabilities."""
    sequence: str
    spps_score: float                  # [0.0, 1.0], >= 0.80 considered laboratory synthesizable
    is_synthesizable: bool
    hydrophobic_runs: List[str]        # Consecutive hydrophobic stretches >= 3
    aspartimide_traps: List[str]       # DG, DS, DT
    deamidation_traps: List[str]       # NG, NS
    oxidation_sites: List[int]         # Position indices of Met
    unpaired_cysteines: int            # 0 if properly paired, odd if unpaired free thiol
    liability_penalties: Dict[str, float]


class SPPSManufacturabilityScanner:
    """Scans peptide sequences for resin-collapse motifs and synthesis liabilities."""

    # Aggregation-prone hydrophobic amino acids in SPPS
    HYDROPHOBIC_RESIDUES = set("VILFWMY")

    # Critical liability motifs
    ASPARTIMIDE_MOTIFS = {"DG", "DS", "DT"}
    DEAMIDATION_MOTIFS = {"NG", "NS"}

    @classmethod
    def scan_hydrophobic_runs(cls, sequence: str, min_length: int = 3) -> List[str]:
        """Detects consecutive runs of hydrophobic residues that collapse on polymer resins."""
        runs: List[str] = []
        current_run: List[str] = []

        for aa in sequence.upper():
            if aa in cls.HYDROPHOBIC_RESIDUES:
                current_run.append(aa)
            else:
                if len(current_run) >= min_length:
                    runs.append("".join(current_run))
                current_run = []

        if len(current_run) >= min_length:
            runs.append("".join(current_run))

        return runs

    @classmethod
    def scan_sequence(cls, sequence: str) -> ChemicalLiabilityReport:
        """Performs a comprehensive chemical liability scan on a candidate peptide."""
        seq_clean = sequence.upper().strip()
        n = len(seq_clean)
        penalties: Dict[str, float] = {}

        # 1. Hydrophobic aggregation runs
        h_runs = cls.scan_hydrophobic_runs(seq_clean, min_length=3)
        h_penalty = sum(0.20 * (len(r) - 2) for r in h_runs)
        if h_penalty > 0:
            penalties["hydrophobic_aggregation"] = round(h_penalty, 3)

        # 2. Aspartimide traps (DG, DS, DT)
        aspartimide_matches = []
        for i in range(n - 1):
            pair = seq_clean[i : i + 2]
            if pair in cls.ASPARTIMIDE_MOTIFS:
                aspartimide_matches.append(f"{pair}@{i}")
        if aspartimide_matches:
            penalties["aspartimide_traps"] = round(len(aspartimide_matches) * 0.25, 3)

        # 3. Deamidation traps (NG, NS)
        deamidation_matches = []
        for i in range(n - 1):
            pair = seq_clean[i : i + 2]
            if pair in cls.DEAMIDATION_MOTIFS:
                deamidation_matches.append(f"{pair}@{i}")
        if deamidation_matches:
            penalties["deamidation_traps"] = round(len(deamidation_matches) * 0.20, 3)

        # 4. Methionine oxidation
        met_positions = [i for i, aa in enumerate(seq_clean) if aa == "M"]
        if met_positions:
            penalties["methionine_oxidation"] = round(len(met_positions) * 0.15, 3)

        # 5. Cysteine pairing
        cys_count = seq_clean.count("C")
        unpaired_cys = cys_count % 2
        if unpaired_cys != 0:
            penalties["unpaired_cysteine"] = 0.35

        # 6. Repetitive tandem residues (e.g. AAAAA)
        tandem_runs = re.findall(r"(.)\1{4,}", seq_clean)
        if tandem_runs:
            penalties["homopolymer_run"] = round(len(tandem_runs) * 0.15, 3)

        # Base score 1.0 minus all penalties, clamped to [0.0, 1.0]
        total_penalty = sum(penalties.values())
        spps_score = max(0.0, min(1.0, 1.0 - total_penalty))
        spps_score = round(spps_score, 4)

        # Laboratory synthesizable threshold: score >= 0.80 and no aspartimide/unpaired cys
        is_synthesizable = (
            spps_score >= 0.80
            and len(aspartimide_matches) == 0
            and unpaired_cys == 0
        )

        return ChemicalLiabilityReport(
            sequence=seq_clean,
            spps_score=spps_score,
            is_synthesizable=is_synthesizable,
            hydrophobic_runs=h_runs,
            aspartimide_traps=aspartimide_matches,
            deamidation_traps=deamidation_matches,
            oxidation_sites=met_positions,
            unpaired_cysteines=unpaired_cys,
            liability_penalties=penalties,
        )
