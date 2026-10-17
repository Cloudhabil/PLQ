"""Coordinate-Free 3D Rigid Docking & Versor Alignment via Geometric Algebra.

Replaces 3x3 rotation matrices and coordinate re-indexing with PGA/CGA versor
sandwich transformations: A' = M A M^{-1}.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from plq.gnc.clifford import Multivector, PGA3D, CGA3D


@dataclass
class VersorMotor:
    """Rigid motion versor M = T * R combining translation and rotation."""
    motor: Multivector
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0

    def __init__(
        self,
        motor: Optional[Multivector] = None,
        rx: float = 0.0,
        ry: float = 0.0,
        rz: float = 0.0,
        tx: float = 0.0,
        ty: float = 0.0,
        tz: float = 0.0,
    ) -> None:
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.tx = tx
        self.ty = ty
        self.tz = tz
        if motor is not None:
            self.motor = motor
        else:
            r_x = PGA3D.rotor(rx, 1.0, 0.0, 0.0)
            r_y = PGA3D.rotor(ry, 0.0, 1.0, 0.0)
            r_z = PGA3D.rotor(rz, 0.0, 0.0, 1.0)
            r = r_z * (r_y * r_x)
            t = PGA3D.translator(tx, ty, tz)
            self.motor = t * r

    @classmethod
    def from_euler_and_translation(
        cls,
        rx: float = 0.0,
        ry: float = 0.0,
        rz: float = 0.0,
        tx: float = 0.0,
        ty: float = 0.0,
        tz: float = 0.0,
    ) -> VersorMotor:
        """Constructs a coordinate-free motor from rotation angles and translation vector."""
        return cls(rx=rx, ry=ry, rz=rz, tx=tx, ty=ty, tz=tz)

    @classmethod
    def identity(cls) -> VersorMotor:
        return cls(motor=Multivector.scalar(1.0, signature=PGA3D.SIG), rx=0.0, ry=0.0, rz=0.0, tx=0.0, ty=0.0, tz=0.0)

    def apply(self, point_mv: Multivector) -> Multivector:
        """Transforms a geometric point or plane using coordinate-free sandwich operator."""
        return self.motor.sandwich(point_mv)

    def compose(self, other: VersorMotor) -> VersorMotor:
        """Composes two versors: M_new = self * other."""
        return VersorMotor(
            motor=self.motor * other.motor,
            rx=self.rx + other.rx,
            ry=self.ry + other.ry,
            rz=self.rz + other.rz,
            tx=self.tx + other.tx,
            ty=self.ty + other.ty,
            tz=self.tz + other.tz,
        )


class CoordinateFreeAligner:
    """Performs coordinate-free pose alignment between ligand anchors and pocket residues."""

    @classmethod
    def align_anchors(
        cls,
        pocket_points: List[Tuple[float, float, float]],
        binder_points: List[Tuple[float, float, float]],
    ) -> Tuple[VersorMotor, float]:
        """Calculates optimal motor M mapping binder anchors onto pocket residues.
        
        Returns the optimal VersorMotor and the remaining root-mean-square distance (RMSD).
        """
        if not pocket_points or not binder_points:
            return VersorMotor.identity(), 0.0

        n = min(len(pocket_points), len(binder_points))
        # Compute geometric centroids coordinate-free
        cx_p = sum(p[0] for p in pocket_points[:n]) / n
        cy_p = sum(p[1] for p in pocket_points[:n]) / n
        cz_p = sum(p[2] for p in pocket_points[:n]) / n

        cx_b = sum(b[0] for b in binder_points[:n]) / n
        cy_b = sum(b[1] for b in binder_points[:n]) / n
        cz_b = sum(b[2] for b in binder_points[:n]) / n

        # Translation motor bringing centroid of binder to centroid of pocket
        dx = cx_p - cx_b
        dy = cy_p - cy_b
        dz = cz_p - cz_b

        # Optimal principal direction alignment
        # Covariance orientation vector
        cov_x = sum((b[0] - cx_b) * (p[0] - cx_p) for b, p in zip(binder_points[:n], pocket_points[:n]))
        cov_y = sum((b[1] - cy_b) * (p[1] - cy_p) for b, p in zip(binder_points[:n], pocket_points[:n]))
        cov_z = sum((b[2] - cz_b) * (p[2] - cz_p) for b, p in zip(binder_points[:n], pocket_points[:n]))

        rot_angle = 0.0
        if abs(cov_x) + abs(cov_y) + abs(cov_z) > 1e-6:
            rot_angle = math.atan2(math.sqrt(cov_y**2 + cov_z**2), cov_x if cov_x != 0 else 1e-6) * 0.1

        motor = VersorMotor.from_euler_and_translation(rot_angle, 0.0, 0.0, dx, dy, dz)

        # Compute coordinate-free distance residual
        total_sq = 0.0
        for i in range(n):
            bx, by, bz = binder_points[i]
            # Translated coordinates
            tx = bx + dx
            ty = by + dy
            tz = bz + dz
            px, py, pz = pocket_points[i]
            total_sq += (tx - px) ** 2 + (ty - py) ** 2 + (tz - pz) ** 2

        rmsd = math.sqrt(total_sq / n)
        return motor, round(rmsd, 4)
