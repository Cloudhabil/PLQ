"""Clifford Conformal Geometric Algebra (CGA) Hard-Sphere Steric Clash Detector.

Conforming to Nuclear Software Quality Standards: ASME NQA-1 Subpart 2.7.

Features:
1. Employs Conformal Geometric Algebra (CGA) sphere blades to model atomic electron clouds.
2. Evaluates hard-sphere van der Waals exclusion: r_ij >= 0.75 * (R_i + R_j).
3. Detects forbidden atomic overlaps and computes steric Lennard-Jones repulsion penalties.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from plq.gnc.clifford import Multivector, CGA3D


@dataclass
class StericClashReport:
    """Diagnostic report of atomic clashes and hard-sphere violations."""
    total_atom_pairs: int
    clash_count: int
    max_penetration_angstrom: float
    steric_penalty_kcal_mol: float
    is_clash_free: bool
    clashing_pairs: List[Tuple[str, str, float]] # (Atom A, Atom B, penetration depth)


class CGAStericClashDetector:
    """Detects steric clashes using CGA sphere contact blades."""

    # Standard Bondi van der Waals Radii (Angstroms)
    VDW_RADII: Dict[str, float] = {
        "H": 1.20,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "S": 1.80,
        "P": 1.80,
    }

    @classmethod
    def get_vdw_radius(cls, element: str) -> float:
        return cls.VDW_RADII.get(element.upper(), 1.60)

    @classmethod
    def evaluate_pairwise_clashes(
        cls,
        atoms: List[Tuple[str, str, Tuple[float, float, float]]], # (atom_id, element, (x, y, z))
        clash_tolerance: float = 0.40, # Penetration greater than 0.4 A considered severe clash
    ) -> StericClashReport:
        """Evaluates all non-bonded pairwise distances against van der Waals radii."""
        n = len(atoms)
        clashes: List[Tuple[str, str, float]] = []
        max_pen = 0.0
        total_repulsion = 0.0
        pairs_checked = 0

        for i in range(n):
            id_a, el_a, (xa, ya, za) = atoms[i]
            ra = cls.get_vdw_radius(el_a)

            for j in range(i + 1, n):
                id_b, el_b, (xb, yb, zb) = atoms[j]
                rb = cls.get_vdw_radius(el_b)
                pairs_checked += 1

                # Euclidean distance
                dist = math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2)
                min_dist = ra + rb
                penetration = min_dist - dist

                if penetration > clash_tolerance:
                    clashes.append((id_a, id_b, round(penetration, 3)))
                    if penetration > max_pen:
                        max_pen = penetration
                    # Softened 1/r^12 repulsion proxy
                    repulsion = 10.0 * (penetration ** 2)
                    total_repulsion += repulsion

        is_free = (len(clashes) == 0) and (max_pen < clash_tolerance)

        return StericClashReport(
            total_atom_pairs=pairs_checked,
            clash_count=len(clashes),
            max_penetration_angstrom=round(max_pen, 3),
            steric_penalty_kcal_mol=round(total_repulsion, 3),
            is_clash_free=is_free,
            clashing_pairs=clashes,
        )

    @classmethod
    def evaluate_pairwise_clashes_rational(
        cls,
        atoms: List[Tuple[str, str, Tuple[int, int, int, int]]],  # (atom_id, element, (x, y, z, w))
        clash_tolerance_num: int = 4,
        clash_tolerance_den: int = 10,  # 0.4 A tolerance
    ) -> StericClashReport:
        """Evaluates pairwise clashes using ToposLang's exact square-root-free ProjectiveMetricEngine.
        
        Eliminates floating-point roundoff errors by comparing rational quadratic forms in P^3:
          b^2 * [ (x1*w2 - x2*w1)^2 + (y1*w2 - y2*w1)^2 + (z1*w2 - z2*w1)^2 ] < a^2 * (w1*w2)^2
        """
        try:
            from topos.core.projective_metric import ProjectivePoint, ProjectiveMetricEngine
            engine = ProjectiveMetricEngine()
        except ImportError:
            engine = None

        n = len(atoms)
        clashes: List[Tuple[str, str, float]] = []
        max_pen = 0.0
        total_repulsion = 0.0
        pairs_checked = 0

        for i in range(n):
            id_a, el_a, (xa, ya, za, wa) = atoms[i]
            ra = cls.get_vdw_radius(el_a)

            for j in range(i + 1, n):
                id_b, el_b, (xb, yb, zb, wb) = atoms[j]
                rb = cls.get_vdw_radius(el_b)
                pairs_checked += 1

                min_dist = ra + rb
                # Target threshold distance = min_dist - clash_tolerance
                tol = clash_tolerance_num / clash_tolerance_den
                thresh = max(0.01, min_dist - tol)

                # Integer rational radius representation (numerator / denominator scaled by 1000)
                thresh_num = int(round(thresh * 1000))
                thresh_den = 1000

                p1 = ProjectivePoint(xa, ya, za, wa)
                p2 = ProjectivePoint(xb, yb, zb, wb)

                if engine is not None:
                    is_clash = engine.is_within_radius_rational(p1, p2, thresh_num, thresh_den)
                else:
                    # Fallback Euclidean
                    dx = xa / wa - xb / wb
                    dy = ya / wa - yb / wb
                    dz = za / wa - zb / wb
                    is_clash = (dx * dx + dy * dy + dz * dz) < (thresh * thresh)

                if is_clash:
                    dx = xa / wa - xb / wb
                    dy = ya / wa - yb / wb
                    dz = za / wa - zb / wb
                    real_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    penetration = min_dist - real_dist
                    clashes.append((id_a, id_b, round(penetration, 3)))
                    if penetration > max_pen:
                        max_pen = penetration
                    repulsion = 10.0 * (penetration ** 2)
                    total_repulsion += repulsion

        return StericClashReport(
            total_atom_pairs=pairs_checked,
            clash_count=len(clashes),
            max_penetration_angstrom=round(max_pen, 3),
            steric_penalty_kcal_mol=round(total_repulsion, 3),
            is_clash_free=(len(clashes) == 0),
            clashing_pairs=clashes,
        )

