"""Combinatorial Hodge Laplacian and Exact Betti Invariant Computation.

Computes L_k = B_k^T B_k + B_{k+1} B_{k+1}^T in Q^{|K_k| x |K_k|} and exact Betti
numbers beta_k = dim(ker L_k) to govern generative loop convergence deterministically.
"""

from __future__ import annotations
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from plq.dacc.complex import DACCComplex


class CombinatorialHodgeLaplacian:
    """Computes exact combinatorial Hodge Laplacians and kernels over rational fields."""

    @classmethod
    def build_boundary_matrix(cls, complex_obj: DACCComplex, dim: int) -> List[List[Fraction]]:
        """Constructs boundary incidence matrix B_k : C_k -> C_{k-1} with exact Fraction entries."""
        if dim <= 0:
            return []

        cells_k = sorted(complex_obj.cells[dim].keys())
        cells_prev = sorted(complex_obj.cells[dim - 1].keys())
        if not cells_k or not cells_prev:
            return []

        row_map = {name: i for i, name in enumerate(cells_prev)}
        num_rows = len(cells_prev)
        num_cols = len(cells_k)

        matrix = [[Fraction(0, 1) for _ in range(num_cols)] for _ in range(num_rows)]

        for col_idx, cell_name in enumerate(cells_k):
            cell = complex_obj.cells[dim][cell_name]
            for target_name, coeff in cell.boundary_coeffs.items():
                if target_name in row_map:
                    row_idx = row_map[target_name]
                    matrix[row_idx][col_idx] = Fraction(coeff, 1)

        return matrix

    @classmethod
    def matrix_transpose(cls, mat: List[List[Fraction]]) -> List[List[Fraction]]:
        if not mat or not mat[0]:
            return []
        rows, cols = len(mat), len(mat[0])
        return [[mat[r][c] for r in range(rows)] for c in range(cols)]

    @classmethod
    def matrix_multiply(
        cls, a: List[List[Fraction]], b: List[List[Fraction]]
    ) -> List[List[Fraction]]:
        if not a or not b:
            return []
        r_a, c_a = len(a), len(a[0])
        r_b, c_b = len(b), len(b[0])
        if c_a != r_b:
            raise ValueError(f"Matrix dimensions incompatible for multiply: {r_a}x{c_a} and {r_b}x{c_b}")

        res = [[Fraction(0, 1) for _ in range(c_b)] for _ in range(r_a)]
        for i in range(r_a):
            for k in range(c_a):
                if a[i][k] != 0:
                    for j in range(c_b):
                        res[i][j] += a[i][k] * b[k][j]
        return res

    @classmethod
    def compute_hodge_laplacian_1(cls, complex_obj: DACCComplex) -> Tuple[List[List[Fraction]], int]:
        """Computes 1-st Hodge Laplacian: L_1 = B_1^T B_1 + B_2 B_2^T.
        
        Returns the matrix L_1 and the exact 1st Betti number beta_1 = dim(ker L_1).
        """
        k1_count = len(complex_obj.cells.get(1, {}))
        if k1_count == 0:
            return [], 0

        b1 = cls.build_boundary_matrix(complex_obj, dim=1)  # shape: |K_0| x |K_1|
        b2 = cls.build_boundary_matrix(complex_obj, dim=2)  # shape: |K_1| x |K_2|

        # Term 1: B_1^T B_1 (shape: |K_1| x |K_1|)
        if b1 and b1[0]:
            b1_t = cls.matrix_transpose(b1)
            term1 = cls.matrix_multiply(b1_t, b1)
        else:
            term1 = [[Fraction(0, 1) for _ in range(k1_count)] for _ in range(k1_count)]

        # Term 2: B_2 B_2^T (shape: |K_1| x |K_1|)
        if b2 and b2[0]:
            b2_t = cls.matrix_transpose(b2)
            term2 = cls.matrix_multiply(b2, b2_t)
        else:
            term2 = [[Fraction(0, 1) for _ in range(k1_count)] for _ in range(k1_count)]

        # Sum: L_1 = term1 + term2
        l1 = [
            [term1[i][j] + term2[i][j] for j in range(k1_count)]
            for i in range(k1_count)
        ]

        # Accelerate rank calculation using CRT modular RREF when available
        try:
            from topos.core.crt import CRTMatrixRREF, DEFAULT_CRT_PRIMES
            int_mat = [[int(val.numerator) if val.denominator == 1 else int(val) for val in row] for row in l1]
            rref = CRTMatrixRREF(DEFAULT_CRT_PRIMES)
            _, rank_l1 = rref.compute_ranks_across_lanes(int_mat)
        except Exception:
            rank_l1 = cls.exact_matrix_rank(l1)

        beta_1 = max(0, k1_count - rank_l1)
        return l1, beta_1

    @classmethod
    def compute_hodge_laplacian_k(
        cls, complex_obj: DACCComplex, k: int
    ) -> Tuple[List[List[Fraction]], int]:
        """Computes k-th Hodge Laplacian in 8D: L_k = B_k^T B_k + B_{k+1} B_{k+1}^T.
        
        Returns the matrix L_k and exact Betti number beta_k = dim(ker L_k).
        """
        kk_count = len(complex_obj.cells.get(k, {}))
        if kk_count == 0:
            return [], 0

        bk = cls.build_boundary_matrix(complex_obj, dim=k)
        bk1 = cls.build_boundary_matrix(complex_obj, dim=k + 1)

        term1 = (
            cls.matrix_multiply(cls.matrix_transpose(bk), bk)
            if (bk and bk[0])
            else [[Fraction(0, 1) for _ in range(kk_count)] for _ in range(kk_count)]
        )

        term2 = (
            cls.matrix_multiply(bk1, cls.matrix_transpose(bk1))
            if (bk1 and bk1[0])
            else [[Fraction(0, 1) for _ in range(kk_count)] for _ in range(kk_count)]
        )

        lk = [
            [term1[i][j] + term2[i][j] for j in range(kk_count)]
            for i in range(kk_count)
        ]

        try:
            from topos.core.crt import CRTMatrixRREF, DEFAULT_CRT_PRIMES
            int_mat = [[int(val.numerator) if val.denominator == 1 else int(val) for val in row] for row in lk]
            rref = CRTMatrixRREF(DEFAULT_CRT_PRIMES)
            _, rank_lk = rref.compute_ranks_across_lanes(int_mat)
        except Exception:
            rank_lk = cls.exact_matrix_rank(lk)

        beta_k = max(0, kk_count - rank_lk)
        return lk, beta_k

    @classmethod
    def compute_hodge_laplacian_crt(
        cls, complex_obj: DACCComplex, k: int
    ) -> int:
        """Fast Betti number calculation beta_k = dim(ker L_k) directly using CRT modular reduction."""
        from plq.dacc.fingerprint import TopologicalDACCInvariant
        sig = TopologicalDACCInvariant.compute_signature(complex_obj)
        betti_vals = sig.betti_numbers
        return betti_vals[k] if 0 <= k < len(betti_vals) else 0

    @classmethod
    def verify_roofline_machine_balance(cls, complex_obj: DACCComplex, dim: int = 1) -> Any:
        """Statically verifies the ToposLang compiler invariant L * d >= 19 on boundary B_k."""
        from topos.compiler.roofline import RooflineCompilerVerifier
        verifier = RooflineCompilerVerifier()

        num_cells = len(complex_obj.cells.get(dim, {}))
        if num_cells == 0:
            return None

        nnz = sum(len(c.boundary_coeffs) for c in complex_obj.cells.get(dim, {}).values())
        return verifier.analyze_complex_boundary(num_cells_N=num_cells, nnz_boundary=max(1, nnz))

    @classmethod
    def exact_matrix_rank(cls, matrix: List[List[Fraction]]) -> int:
        """Computes exact matrix rank over the field of rational numbers Q using Gaussian elimination."""
        if not matrix or not matrix[0]:
            return 0
        mat = [list(row) for row in matrix]
        num_rows = len(mat)
        num_cols = len(mat[0])
        rank = 0
        lead = 0

        for r in range(num_rows):
            if lead >= num_cols:
                break
            i = r
            while mat[i][lead] == 0:
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

