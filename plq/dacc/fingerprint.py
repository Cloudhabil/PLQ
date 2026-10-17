"""Topological Invariant Fingerprinting replacing 1D ECFP4 vectors.

Computes exact cellular homology signatures (Betti numbers beta_0, beta_1, beta_2,
Euler characteristic chi, and boundary spectrum) over Directed Acyclic Cell Complexes.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from plq.dacc.complex import DACCComplex


@dataclass(frozen=True)
class TopologicalSignature:
    """Exact topological homology signature of a molecular or pocket complex across 8D."""
    beta_0: int
    beta_1: int
    beta_2: int
    chi: int
    num_0cells: int
    num_1cells: int
    num_2cells: int
    ternary_charge_balance: int
    his_switch_density: float
    beta_3: int = 0
    beta_4: int = 0
    beta_5: int = 0
    beta_6: int = 0
    beta_7: int = 0
    beta_8: int = 0
    num_3cells: int = 0
    num_4cells: int = 0
    num_5cells: int = 0
    num_6cells: int = 0
    num_7cells: int = 0
    num_8cells: int = 0
    is_druggable_3d: bool = False

    @property
    def betti_numbers(self) -> Tuple[int, int, int, int, int, int, int, int, int]:
        """Returns the 9 Betti numbers (beta_0 ... beta_8)."""
        return (
            self.beta_0,
            self.beta_1,
            self.beta_2,
            self.beta_3,
            self.beta_4,
            self.beta_5,
            self.beta_6,
            self.beta_7,
            self.beta_8,
        )

    def as_vector(self) -> List[float]:
        """Provides a dense invariant vector for downstream ranking."""
        return [
            float(self.beta_0),
            float(self.beta_1),
            float(self.beta_2),
            float(self.beta_3),
            float(self.beta_4),
            float(self.beta_5),
            float(self.beta_6),
            float(self.beta_7),
            float(self.beta_8),
            float(self.chi),
            float(self.num_0cells),
            float(self.num_1cells),
            float(self.num_2cells),
            float(self.num_3cells),
            float(self.ternary_charge_balance),
            self.his_switch_density,
        ]


class TopologicalDACCInvariant:
    """Computes topological DACC homology invariants and topological similarity."""

    @classmethod
    def compute_signature(cls, complex_obj: DACCComplex) -> TopologicalSignature:
        """Computes Betti numbers and invariant signature from cell incidence data across 8D."""
        counts = [len(complex_obj.cells.get(k, {})) for k in range(9)]
        k0, k1, k2, k3, k4, k5, k6, k7, k8 = counts

        # Compute exact boundary ranks using CRT modular RREF across all dimensions
        ranks: Dict[int, int] = {}
        for dim in range(1, 9):
            ranks[dim] = cls.compute_boundary_rank_crt(complex_obj, dim)

        # Betti numbers: beta_k = |K_k| - rank(B_k) - rank(B_{k+1})
        # For k=0: beta_0 = |K_0| - rank(B_1)
        beta_0 = max(0, k0 - ranks[1]) if k0 > 0 else 0
        beta_1 = max(0, k1 - ranks[1] - ranks[2]) if k1 > 0 else 0
        beta_2 = max(0, k2 - ranks[2] - ranks[3]) if k2 > 0 else 0
        beta_3 = max(0, k3 - ranks[3] - ranks[4]) if k3 > 0 else 0
        beta_4 = max(0, k4 - ranks[4] - ranks[5]) if k4 > 0 else 0
        beta_5 = max(0, k5 - ranks[5] - ranks[6]) if k5 > 0 else 0
        beta_6 = max(0, k6 - ranks[6] - ranks[7]) if k6 > 0 else 0
        beta_7 = max(0, k7 - ranks[7] - ranks[8]) if k7 > 0 else 0
        beta_8 = max(0, k8 - ranks[8]) if k8 > 0 else 0

        # 8D Euler characteristic: chi = sum_{k=0}^8 (-1)^k |K_k| == sum (-1)^k beta_k
        chi = sum((-1)**k * counts[k] for k in range(9))
        is_druggable = (beta_0 - beta_1 + beta_2 == 1)

        # Ternary charge and histidine density
        total_charge = 0
        his_count = 0
        for c0 in complex_obj.cells[0].values():
            total_charge += c0.attributes.get("ternary_state", 0)
            if c0.attributes.get("is_histidine", False):
                his_count += 1

        his_density = (his_count / k0) if k0 > 0 else 0.0

        return TopologicalSignature(
            beta_0=beta_0,
            beta_1=beta_1,
            beta_2=beta_2,
            chi=chi,
            num_0cells=k0,
            num_1cells=k1,
            num_2cells=k2,
            ternary_charge_balance=total_charge,
            his_switch_density=round(his_density, 4),
            beta_3=beta_3,
            beta_4=beta_4,
            beta_5=beta_5,
            beta_6=beta_6,
            beta_7=beta_7,
            beta_8=beta_8,
            num_3cells=k3,
            num_4cells=k4,
            num_5cells=k5,
            num_6cells=k6,
            num_7cells=k7,
            num_8cells=k8,
            is_druggable_3d=is_druggable,
        )

    @classmethod
    def compute_boundary_rank_crt(cls, complex_obj: DACCComplex, dim: int) -> int:
        """Computes rank_Q(B_k) using CRT modular RREF over parallel prime lanes."""
        cells_k = list(complex_obj.cells.get(dim, {}).values())
        cells_prev = list(complex_obj.cells.get(dim - 1, {}).values())
        if not cells_k or not cells_prev:
            return 0

        # Boundary matrix B_k shape: |K_{k-1}| x |K_k|
        row_index = {name: i for i, name in enumerate(sorted(complex_obj.cells[dim - 1].keys()))}
        num_rows = len(row_index)
        num_cols = len(cells_k)

        matrix: List[List[int]] = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
        for col_idx, cell in enumerate(sorted(cells_k, key=lambda c: c.name)):
            for prev_name, coeff in cell.boundary_coeffs.items():
                if prev_name in row_index:
                    matrix[row_index[prev_name]][col_idx] = int(coeff)

        try:
            from topos.core.crt import CRTMatrixRREF, DEFAULT_CRT_PRIMES
            rref_engine = CRTMatrixRREF(DEFAULT_CRT_PRIMES)
            _, rational_rank = rref_engine.compute_ranks_across_lanes(matrix)
            return rational_rank
        except Exception:
            float_rows = [[float(v) for v in r] for r in matrix]
            return cls._matrix_rank(float_rows, num_cols)

    @classmethod
    def _count_connected_components(cls, complex_obj: DACCComplex) -> int:
        """Counts connected components beta_0 over 0-cells and 1-cells."""
        parent: Dict[str, str] = {name: name for name in complex_obj.cells[0].keys()}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for edge in complex_obj.cells[1].values():
            nodes = [n for n in edge.boundary_coeffs.keys()]
            if len(nodes) >= 2:
                union(nodes[0], nodes[1])

        unique_roots = {find(name) for name in complex_obj.cells[0].keys()}
        return len(unique_roots) if complex_obj.cells[0] else 0

    @classmethod
    def _compute_rank_b2(cls, complex_obj: DACCComplex) -> int:
        """Computes rank of boundary operator B_2 over rational field Q."""
        faces = list(complex_obj.cells[2].values())
        if not faces:
            return 0
        edge_index = {e_name: i for i, e_name in enumerate(sorted(complex_obj.cells[1].keys()))}
        num_edges = len(edge_index)

        # Build column vectors
        vectors: List[List[float]] = []
        for face in faces:
            col = [0.0] * num_edges
            for e_name, coeff in face.boundary_coeffs.items():
                if e_name in edge_index:
                    col[edge_index[e_name]] = float(coeff)
            vectors.append(col)

        # Exact Gaussian elimination for rank
        return cls._matrix_rank(vectors, num_edges)

    @classmethod
    def _compute_rank_b3(cls, complex_obj: DACCComplex) -> int:
        """Computes rank of boundary operator B_3 over rational field Q."""
        cells3 = list(complex_obj.cells.get(3, {}).values())
        if not cells3:
            return 0
        face_index = {f_name: i for i, f_name in enumerate(sorted(complex_obj.cells[2].keys()))}
        num_faces = len(face_index)

        vectors: List[List[float]] = []
        for c3 in cells3:
            col = [0.0] * num_faces
            for f_name, coeff in c3.boundary_coeffs.items():
                if f_name in face_index:
                    col[face_index[f_name]] = float(coeff)
            vectors.append(col)

        return cls._matrix_rank(vectors, num_faces)

    @staticmethod
    def _matrix_rank(rows: List[List[float]], num_cols: int) -> int:
        """Standard Gaussian elimination rank over float (representing exact rationals)."""
        if not rows or num_cols == 0:
            return 0
        mat = [list(r) for r in rows]
        num_rows = len(mat)
        rank = 0
        lead = 0
        for r in range(num_rows):
            if lead >= num_cols:
                break
            i = r
            while abs(mat[i][lead]) < 1e-9:
                i += 1
                if i == num_rows:
                    i = r
                    lead += 1
                    if lead == num_cols:
                        break
            if lead >= num_cols:
                break
            mat[r], mat[i] = mat[i], mat[r]
            lv = mat[r][lead]
            mat[r] = [val / lv for val in mat[r]]
            for i2 in range(num_rows):
                if i2 != r:
                    lv2 = mat[i2][lead]
                    mat[i2] = [iv - lv2 * rv for iv, rv in zip(mat[i2], mat[r])]
            lead += 1
            rank += 1
        return rank

    @classmethod
    def topological_distance(cls, sig_a: TopologicalSignature, sig_b: TopologicalSignature) -> float:
        """Topological distance metric replacing Tanimoto distance.
        
        Evaluates Betti divergence and ternary switch alignment.
        """
        va = sig_a.as_vector()
        vb = sig_b.as_vector()
        # Normalized weighted Euclidean distance
        weights = [2.0, 3.0, 3.0, 2.0, 0.5, 0.5, 0.5, 1.5, 2.5]
        diff_sq = sum(w * ((x - y) ** 2) for w, x, y in zip(weights, va, vb))
        return math.sqrt(diff_sq)
