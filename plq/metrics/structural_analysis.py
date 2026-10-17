"""Structural Analysis & Foldstring Classification Suite.

Implements:
1. Novelty:
       Novelty = 1 - max_{S_k in D_PDB} TM-score(X_query, X_{S_k})
2. TED Confidence (Domain Segmentation Modularity):
       S_cut = Cut(A, B) / (Vol(A) * Vol(B))
       Evaluates normalized graph-cut modularity on the C_alpha contact graph.
3. DSSP Foldstring (Kabsch-Sander Electrostatic Hydrogen-Bond Model):
       E = q1 * q2 * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN) * 332 kcal/mol
       with q1 = 0.42e, q2 = 0.20e (threshold E < -0.5 kcal/mol).
       Emits secondary structure strings with ('H', 'E', 'T', 'C').
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from plq.metrics.esm_quality import ESMQualityEvaluator


@dataclass
class DSSPReport:
    """Report of secondary structure foldstring and electrostatic H-bond networks."""
    foldstring: str
    helix_fraction: float
    sheet_fraction: float
    turn_fraction: float
    coil_fraction: float
    hbond_pairs: List[Tuple[int, int, float]]  # (donor_idx, acceptor_idx, energy_kcal_mol)
    sequence_length: int


@dataclass
class TEDReport:
    """Report of domain segmentation modularity via normalized graph cuts."""
    normalized_cut: float
    best_split_index: Optional[int]
    domain_a_range: Optional[Tuple[int, int]]
    domain_b_range: Optional[Tuple[int, int]]
    modularity: float
    is_multi_domain: bool


class StructuralClassifier:
    """Evaluates fold novelty, domain boundaries, and secondary structure foldstrings."""

    # -------------------------------------------------------------------------
    # 1. Structural Novelty vs Reference Database
    # -------------------------------------------------------------------------

    @classmethod
    def compute_novelty(
        cls,
        query_coords: Sequence[Tuple[float, float, float]],
        reference_coords_list: Sequence[Sequence[Tuple[float, float, float]]],
    ) -> float:
        """Computes structural novelty against reference structures.

        Novelty = 1.0 - max_{ref} TM-score(query, ref)
        """
        if not reference_coords_list or not query_coords:
            return 1.0  # Fully novel if no reference database exists

        max_tm = 0.0
        for ref_coords in reference_coords_list:
            if not ref_coords:
                continue
            report = ESMQualityEvaluator.compute_tm_score(query_coords, ref_coords)
            if report.tm_score > max_tm:
                max_tm = report.tm_score

        novelty = max(0.0, min(1.0, 1.0 - max_tm))
        return round(novelty, 4)

    # -------------------------------------------------------------------------
    # 2. TED Domain Segmentation Modularity (Normalized Graph Cut)
    # -------------------------------------------------------------------------

    @classmethod
    def compute_ted_confidence(
        cls,
        coords: Sequence[Tuple[float, float, float]],
        cut_threshold: float = 8.0,
        min_domain_length: int = 15,
    ) -> TEDReport:
        """Evaluates domain modularity via normalized graph cuts on C_alpha contact graph.

        S_cut = Cut(A, B) / (Vol(A) * Vol(B))
        """
        n = len(coords)
        if n < 2 * min_domain_length:
            # Single compact domain, too short to split into two independent domains
            return TEDReport(
                normalized_cut=0.0,
                best_split_index=None,
                domain_a_range=(0, n),
                domain_b_range=None,
                modularity=1.0,
                is_multi_domain=False,
            )

        # Build adjacency matrix and vertex degrees
        adj: List[List[float]] = [[0.0] * n for _ in range(n)]
        degree: List[float] = [0.0] * n

        cut_thresh_sq = cut_threshold * cut_threshold
        for i in range(n):
            xi, yi, zi = coords[i][:3]
            for j in range(i + 1, n):
                xj, yj, zj = coords[j][:3]
                d_sq = (xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2
                if d_sq <= cut_thresh_sq:
                    adj[i][j] = 1.0
                    adj[j][i] = 1.0
                    degree[i] += 1.0
                    degree[j] += 1.0

        total_vol = sum(degree)
        if total_vol <= 1e-6:
            return TEDReport(
                normalized_cut=0.0,
                best_split_index=None,
                domain_a_range=(0, n),
                domain_b_range=None,
                modularity=0.0,
                is_multi_domain=False,
            )

        best_score = float("inf")
        best_split: Optional[int] = None
        best_ncut = 1.0
        best_modularity = 0.0

        # Scan possible hinge partition points k between [min_domain_length, n - min_domain_length]
        for k in range(min_domain_length, n - min_domain_length + 1):
            # Domain A: 0 .. k-1, Domain B: k .. n-1
            vol_a = sum(degree[i] for i in range(k))
            vol_b = sum(degree[i] for i in range(k, n))

            if vol_a <= 1e-6 or vol_b <= 1e-6:
                continue

            cut_ab = 0.0
            for i in range(k):
                for j in range(k, n):
                    if adj[i][j] > 0.0:
                        cut_ab += 1.0

            s_cut = cut_ab / (vol_a * vol_b)
            ncut = (cut_ab / vol_a) + (cut_ab / vol_b)
            # Modularity: 1 - inter-domain cut ratio
            min_vol = min(vol_a, vol_b)
            modularity = 1.0 - (cut_ab / min_vol) if min_vol > 0 else 0.0

            if s_cut < best_score:
                best_score = s_cut
                best_split = k
                best_ncut = ncut
                best_modularity = modularity

        is_multi = (best_ncut < 0.25 and best_modularity > 0.75) if best_split is not None else False

        domain_a = (0, best_split) if best_split is not None else (0, n)
        domain_b = (best_split, n) if best_split is not None else None

        return TEDReport(
            normalized_cut=round(best_ncut, 4),
            best_split_index=best_split,
            domain_a_range=domain_a,
            domain_b_range=domain_b,
            modularity=round(max(0.0, best_modularity), 4),
            is_multi_domain=is_multi,
        )

    # -------------------------------------------------------------------------
    # 3. DSSP Electrostatic Hydrogen-Bond Model & Foldstring Assignment
    # -------------------------------------------------------------------------

    @classmethod
    def compute_foldstring_dssp(
        cls,
        backbone_atoms: Sequence[Any],
    ) -> DSSPReport:
        """Computes secondary structure foldstring ('H', 'E', 'T', 'C') via DSSP H-bonds.

        Accepts either:
        - A sequence of residue dictionaries: [{'N': (x,y,z), 'CA': (x,y,z), 'C': (x,y,z), 'O': (x,y,z)}, ...]
        - Or a sequence of (x,y,z) C_alpha coordinates (idealized backbone will be reconstructed).
        """
        n = len(backbone_atoms)
        if n == 0:
            return DSSPReport(
                foldstring="",
                helix_fraction=0.0,
                sheet_fraction=0.0,
                turn_fraction=0.0,
                coil_fraction=0.0,
                hbond_pairs=[],
                sequence_length=0,
            )

        # Parse or reconstruct N, CA, C, O, H for each residue
        res_atoms = cls._standardize_backbone(backbone_atoms)

        # Constant: 332 * 0.42 * 0.20 = 27.888 kcal * A / mol
        c_dssp = 332.0 * 0.42 * 0.20
        hbond_pairs: List[Tuple[int, int, float]] = []

        # Find all H-bonds between donor residue j (N_j - H_j) and acceptor residue i (C_i = O_i)
        # DSSP cutoff: E < -0.5 kcal/mol
        hbond_matrix: Set[Tuple[int, int]] = set()

        for j in range(len(res_atoms)):  # donor j (amide N-H)
            nj = res_atoms[j].get("N")
            hj = res_atoms[j].get("H")
            if nj is None or hj is None:
                continue

            for i in range(len(res_atoms)):  # acceptor i (carbonyl C=O)
                if abs(i - j) < 2:
                    continue  # Peptide bond neighbors cannot form intra-backbone H-bond

                ci = res_atoms[i].get("C")
                oi = res_atoms[i].get("O")
                if ci is None or oi is None:
                    continue

                r_on = cls._dist(oi, nj)
                r_ch = cls._dist(ci, hj)
                r_oh = cls._dist(oi, hj)
                r_cn = cls._dist(ci, nj)

                if min(r_on, r_ch, r_oh, r_cn) < 0.5:
                    continue  # Steric clash or unphysical overlap

                energy = c_dssp * ((1.0 / r_on) + (1.0 / r_ch) - (1.0 / r_oh) - (1.0 / r_cn))
                if energy < -0.5:
                    hbond_pairs.append((j, i, round(energy, 3)))
                    hbond_matrix.add((j, i))

        # Classify secondary structure per residue
        ss_types = ["C"] * n

        # Detect beta-bridges (parallel or antiparallel between i and j)
        for i in range(n):
            for j in range(i + 3, n):
                # Antiparallel bridge: (i->j and j->i) or (i-1->j+1 and j-1->i+1)
                antiparallel = (
                    ((i, j) in hbond_matrix and (j, i) in hbond_matrix)
                    or (
                        i > 0
                        and j < n - 1
                        and (i - 1, j + 1) in hbond_matrix
                        and (j - 1, i + 1) in hbond_matrix
                    )
                )
                # Parallel bridge: (i-1->j and j->i+1) or (j-1->i and i->j+1)
                parallel = (
                    (i > 0 and (i - 1, j) in hbond_matrix and (j, i + 1) in hbond_matrix)
                    or (j > 0 and (j - 1, i) in hbond_matrix and (i, j + 1) in hbond_matrix)
                )

                if antiparallel or parallel:
                    ss_types[i] = "E"
                    ss_types[j] = "E"

        # Detect turns (i+3 or i+4 H-bond)
        for i in range(n - 3):
            if (i + 3, i) in hbond_matrix or (i + 4, i) in hbond_matrix:
                for t in range(i, min(n, i + 4)):
                    if ss_types[t] == "C":
                        ss_types[t] = "T"

        # Detect alpha-helices: consecutive i -> i+4 H-bonds
        for i in range(n - 4):
            if (i + 4, i) in hbond_matrix:
                # Mark i through i+4 as helix if part of consecutive turn or turn pair
                if i > 0 and (i + 3, i - 1) in hbond_matrix:
                    for h_idx in range(i, i + 5):
                        ss_types[h_idx] = "H"
                elif i + 5 < n and (i + 5, i + 1) in hbond_matrix:
                    for h_idx in range(i, i + 5):
                        ss_types[h_idx] = "H"
                elif ss_types[i + 1] == "H" or ss_types[i + 2] == "H":
                    for h_idx in range(i, i + 5):
                        ss_types[h_idx] = "H"

        foldstring = "".join(ss_types)
        num_h = foldstring.count("H")
        num_e = foldstring.count("E")
        num_t = foldstring.count("T")
        num_c = foldstring.count("C")

        return DSSPReport(
            foldstring=foldstring,
            helix_fraction=round(num_h / n, 4),
            sheet_fraction=round(num_e / n, 4),
            turn_fraction=round(num_t / n, 4),
            coil_fraction=round(num_c / n, 4),
            hbond_pairs=hbond_pairs,
            sequence_length=n,
        )

    # -------------------------------------------------------------------------
    # Helper: Backbone Geometry & Reconstruction
    # -------------------------------------------------------------------------

    @staticmethod
    def _dist(p1: Sequence[float], p2: Sequence[float]) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    @classmethod
    def _standardize_backbone(
        cls,
        backbone_items: Sequence[Any],
    ) -> List[Dict[str, Tuple[float, float, float]]]:
        """Standardizes input into explicit N, CA, C, O, H atom coordinates per residue."""
        res_atoms: List[Dict[str, Tuple[float, float, float]]] = []

        for idx, item in enumerate(backbone_items):
            if isinstance(item, dict):
                entry: Dict[str, Tuple[float, float, float]] = {}
                for k in ("N", "CA", "C", "O", "H"):
                    if k in item:
                        c = item[k]
                        entry[k] = (float(c[0]), float(c[1]), float(c[2]))
                # If H is missing, compute from N and predecessor C
                if "H" not in entry and "N" in entry and idx > 0:
                    prev_c = res_atoms[idx - 1].get("C")
                    if prev_c is not None:
                        # Vector N - prev_C normalized
                        nx, ny, nz = entry["N"]
                        px, py, pz = prev_c
                        vx, vy, vz = nx - px, ny - py, nz - pz
                        vlen = math.sqrt(vx * vx + vy * vy + vz * vz)
                        if vlen > 1e-4:
                            entry["H"] = (nx + vx / vlen, ny + vy / vlen, nz + vz / vlen)
                res_atoms.append(entry)
            elif isinstance(item, (tuple, list)):
                # Given C_alpha coordinate: reconstruct approximate N, CA, C, O, H
                ca = (float(item[0]), float(item[1]), float(item[2]))
                res_atoms.append(cls._reconstruct_ideal_residue(ca, idx))
            else:
                res_atoms.append({})

        return res_atoms

    @staticmethod
    def _reconstruct_ideal_residue(
        ca: Tuple[float, float, float],
        index: int,
    ) -> Dict[str, Tuple[float, float, float]]:
        """Reconstructs idealized N, CA, C, O, H from a C_alpha trace."""
        x, y, z = ca
        # Standard peptide geometry offsets along virtual peptide chain
        n = (x - 0.8, y + 0.9, z - 0.5)
        c = (x + 1.2, y - 0.7, z + 0.3)
        o = (x + 1.4, y - 1.8, z + 0.8)
        h = (x - 1.5, y + 1.2, z - 0.7)
        return {"N": n, "CA": ca, "C": c, "O": o, "H": h}
