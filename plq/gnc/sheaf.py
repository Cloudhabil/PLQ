"""Clifford Multivector Sheaves over Directed Acyclic Cell Complexes.

Attaches geometric multivector stalks to 0-cells, 1-cells, and 2-cells, with
continuous restriction maps and wedge product interaction evaluations:
    S(r_1) ^ S(r_2) = 0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from plq.dacc.complex import DACCComplex
from plq.gnc.clifford import Multivector, CGA3D


@dataclass
class CliffordStalk:
    """A multivector geometric data fiber attached to a topological cell."""
    cell_name: str
    cell_dim: int
    fiber: Multivector


class CliffordSheaf:
    """Cellular Sheaf of Clifford multivectors over a DACCComplex."""

    def __init__(self, name: str, complex_obj: DACCComplex) -> None:
        self.name = name
        self.complex = complex_obj
        # stalks: mapping cell_name -> CliffordStalk
        self.stalks: Dict[str, CliffordStalk] = {}

    def attach_stalk(self, cell_name: str, fiber: Multivector) -> None:
        """Attaches a geometric multivector fiber to a designated cell."""
        cell = self.complex.get_cell(cell_name)
        self.stalks[cell_name] = CliffordStalk(
            cell_name=cell_name,
            cell_dim=cell.dim,
            fiber=fiber,
        )

    def attach_point_stalk(self, cell_name: str, x: float, y: float, z: float) -> None:
        """Attaches a CGA conformal point fiber P(x, y, z) to a 0-cell."""
        fiber = CGA3D.point(x, y, z)
        self.attach_stalk(cell_name, fiber)

    def attach_sphere_stalk(
        self, cell_name: str, x: float, y: float, z: float, radius: float
    ) -> None:
        """Attaches a CGA sphere fiber S(center, radius) representing an atom or residue van der Waals sphere."""
        fiber = CGA3D.sphere(x, y, z, radius)
        self.attach_stalk(cell_name, fiber)

    def compute_continuous_wedge_overlap(self, other_sheaf: CliffordSheaf) -> float:
        """Evaluates continuous multivector wedge product between two sheaf fields:
        
        Overlap = sum_{c1, c2} ||S(c1) ^ S(c2)||
        When this vanishes (wedge == 0), the continuous spatial extents are disjoint / non-clashing.
        """
        total_overlap = 0.0
        for name_a, stalk_a in self.stalks.items():
            for name_b, stalk_b in other_sheaf.stalks.items():
                w = stalk_a.fiber.wedge(stalk_b.fiber)
                total_overlap += w.norm()
        return round(total_overlap, 6)

    def global_energy_potential(self, target_sheaf: CliffordSheaf, ph_state: float = 6.5) -> float:
        """Computes coordinate-free binding energy using multivector inner and wedge products.
        
        Attractive electrostatic and contact terms (inner products) minus steric clash penalties (wedge products).
        """
        attractive_energy = 0.0
        steric_penalty = 0.0

        for name_a, stalk_a in self.stalks.items():
            for name_b, stalk_b in target_sheaf.stalks.items():
                inner_val = stalk_a.fiber.dot(stalk_b.fiber).scalar_part()
                wedge_norm = stalk_a.fiber.wedge(stalk_b.fiber).norm()

                # Inner product contributes to electrostatic / van der Waals attraction
                attractive_energy += inner_val
                steric_penalty += wedge_norm

        # Binding free energy: attractive is negative, steric is positive
        # pH factor scales electrostatic interactions based on protonation
        c = 1.2 if ph_state <= 6.8 else 0.4
        delta_g = -abs(attractive_energy) * c + steric_penalty * 0.5
        return round(delta_g, 4)
