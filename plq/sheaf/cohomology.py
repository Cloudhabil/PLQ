"""Cellular Sheaf Cohomology & Binding Obstruction Engine.

Conforming to Nuclear Software Quality Standards: ASME NQA-1 Subpart 2.7.

Features:
1. Implements cellular sheaves F over Directed Acyclic Cell Complexes K = (K_0, K_1, K_2).
2. Computes the 0-th sheaf cohomology H^0(K; F) = ker(delta^0), representing
   the linear space of global, mutually compatible binding sections.
3. Computes the 1-st sheaf cohomology H^1(K; F) = ker(delta^1) / im(delta^0),
   representing the topological obstruction space.
4. Identifies geometric/electrostatic frustration: dim H^1(K; F) == 0 guarantees
   that local paratope contacts extend globally across the pocket without strain.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Set, Tuple

from plq.dacc.complex import DACCComplex
from topos.topology.homology import ExactMatrix


@dataclass
class SheafCohomologyReport:
    """Diagnostic report of cellular sheaf cohomology invariants."""
    complex_name: str
    stalk_dimension: int
    dim_H0_global_sections: int
    dim_H1_obstructions: int
    is_globally_unobstructed: bool # True if dim H^1 == 0
    frustration_index: float


class CellularSheafEngine:
    """Computes sheaf coboundary operators and obstruction cohomology over Q."""

    @classmethod
    def compute_obstruction_cohomology(
        cls,
        complex_obj: DACCComplex,
        stalk_dim: int = 1,
    ) -> SheafCohomologyReport:
        r"""Evaluates H^0(K; F) and H^1(K; F) over rational field Q.
        
        The 0-cochain space C^0 has dimension |K_0| * stalk_dim.
        The 1-cochain space C^1 has dimension |K_1| * stalk_dim.
        The coboundary operator delta^0: C^0 -> C^1 is the signed incidence
        map tensored with the stalk restriction maps.
        """
        nodes = sorted(complex_obj.cells[0].keys())
        edges = sorted(complex_obj.cells[1].keys())
        faces = sorted(complex_obj.cells[2].keys())

        n0 = len(nodes)
        n1 = len(edges)
        n2 = len(faces)

        if n0 == 0:
            return SheafCohomologyReport(
                complex_name=complex_obj.name,
                stalk_dimension=stalk_dim,
                dim_H0_global_sections=0,
                dim_H1_obstructions=0,
                is_globally_unobstructed=True,
                frustration_index=0.0,
            )

        node_to_idx = {name: i for i, name in enumerate(nodes)}
        edge_to_idx = {name: i for i, name in enumerate(edges)}

        # Build coboundary matrix delta^0 of shape (n1 * stalk_dim, n0 * stalk_dim)
        # delta^0(e) = rho_{tgt <- e}(v_tgt) - rho_{src <- e}(v_src)
        dim_c0 = n0 * stalk_dim
        dim_c1 = n1 * stalk_dim
        dim_c2 = n2 * stalk_dim

        # Cochain coboundary 0: (delta^0)_{e, v} = B_1(v, e)
        d0_entries: List[List[Fraction]] = [[Fraction(0, 1) for _ in range(dim_c0)] for _ in range(dim_c1)]

        for e_idx, e_name in enumerate(edges):
            edge = complex_obj.cells[1][e_name]
            for v_name, sign in edge.boundary_coeffs.items():
                if v_name in node_to_idx:
                    v_idx = node_to_idx[v_name]
                    # Assign signed identity restriction map
                    for d in range(stalk_dim):
                        row = e_idx * stalk_dim + d
                        col = v_idx * stalk_dim + d
                        d0_entries[row][col] = Fraction(sign, 1)

        mat_d0 = ExactMatrix(dim_c1, dim_c0, d0_entries)
        rank_d0 = mat_d0.rank()

        # dim H^0 = dim(ker delta^0) = dim(C^0) - rank(delta^0)
        dim_h0 = max(0, dim_c0 - rank_d0)

        # Build coboundary matrix delta^1 of shape (n2 * stalk_dim, n1 * stalk_dim)
        # delta^1(f) = sum signed edges around face
        dim_h1 = 0
        if n2 > 0 and n1 > 0:
            d1_entries: List[List[Fraction]] = [[Fraction(0, 1) for _ in range(dim_c1)] for _ in range(dim_c2)]
            for f_idx, f_name in enumerate(faces):
                face = complex_obj.cells[2][f_name]
                for e_name, sign in face.boundary_coeffs.items():
                    if e_name in edge_to_idx:
                        e_idx = edge_to_idx[e_name]
                        for d in range(stalk_dim):
                            row = f_idx * stalk_dim + d
                            col = e_idx * stalk_dim + d
                            d1_entries[row][col] = Fraction(sign, 1)

            mat_d1 = ExactMatrix(dim_c2, dim_c1, d1_entries)
            rank_d1 = mat_d1.rank()

            # dim(ker delta^1) = dim(C^1) - rank(delta^1)
            dim_ker_d1 = max(0, dim_c1 - rank_d1)
            # dim H^1 = dim(ker delta^1) - rank(delta^0)
            dim_h1 = max(0, dim_ker_d1 - rank_d0)
        else:
            # If no 2-cells exist, ker(delta^1) = C^1, so H^1 = dim(C^1) - rank(delta^0)
            dim_h1 = max(0, dim_c1 - rank_d0)

        is_unobstructed = (dim_h1 == 0)
        frustration = float(dim_h1) / float(dim_c1) if dim_c1 > 0 else 0.0

        return SheafCohomologyReport(
            complex_name=complex_obj.name,
            stalk_dimension=stalk_dim,
            dim_H0_global_sections=dim_h0,
            dim_H1_obstructions=dim_h1,
            is_globally_unobstructed=is_unobstructed,
            frustration_index=round(frustration, 4),
        )
