"""Clifford Geometric Algebra for Projective (PGA) and Conformal (CGA) 3D Spaces.

Provides coordinate-free multivectors, basis blade multiplication, versor sandwich
operators (A' = M A M^{-1}), and geometric intersection tests via wedge products.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class Multivector:
    """Clifford Multivector over an arbitrary signature (p, q, r).
    
    Blades are represented as tuples of 1-based generator indices, e.g.:
    () -> scalar (grade 0)
    (1,) -> e_1 (grade 1)
    (1, 2) -> e_1 ^ e_2 (grade 2)
    (1, 2, 3) -> e_1 ^ e_2 ^ e_3 (grade 3)
    Stored as a dictionary {blade_tuple: coefficient}.
    """
    signature: Tuple[int, int, int]  # (p, q, r): p positive, q negative, r zero
    components: Dict[Tuple[int, ...], float]

    def __post_init__(self) -> None:
        # Filter near-zero coefficients
        self.components = {
            blade: coeff for blade, coeff in self.components.items() if abs(coeff) > 1e-12
        }

    @classmethod
    def scalar(cls, value: float, signature: Tuple[int, int, int] = (3, 0, 1)) -> Multivector:
        return cls(signature=signature, components={(): float(value)})

    @classmethod
    def basis(cls, index: int, signature: Tuple[int, int, int] = (3, 0, 1), coeff: float = 1.0) -> Multivector:
        return cls(signature=signature, components={(index,): coeff})

    def grade(self, k: int) -> Multivector:
        """Projects to k-blade components."""
        return Multivector(
            signature=self.signature,
            components={b: c for b, c in self.components.items() if len(b) == k},
        )

    def scalar_part(self) -> float:
        return self.components.get((), 0.0)

    def __add__(self, other: Union[Multivector, float, int]) -> Multivector:
        if isinstance(other, (int, float)):
            other = Multivector.scalar(float(other), signature=self.signature)
        new_comp = dict(self.components)
        for b, c in other.components.items():
            new_comp[b] = new_comp.get(b, 0.0) + c
        return Multivector(signature=self.signature, components=new_comp)

    def __sub__(self, other: Union[Multivector, float, int]) -> Multivector:
        if isinstance(other, (int, float)):
            other = Multivector.scalar(float(other), signature=self.signature)
        new_comp = dict(self.components)
        for b, c in other.components.items():
            new_comp[b] = new_comp.get(b, 0.0) - c
        return Multivector(signature=self.signature, components=new_comp)

    def __mul__(self, other: Union[Multivector, float, int]) -> Multivector:
        if isinstance(other, (int, float)):
            return Multivector(
                signature=self.signature,
                components={b: c * float(other) for b, c in self.components.items()},
            )
        return self._geometric_product(other)

    def __rmul__(self, other: Union[float, int]) -> Multivector:
        return self.__mul__(other)

    def __neg__(self) -> Multivector:
        return Multivector(
            signature=self.signature,
            components={b: -c for b, c in self.components.items()},
        )

    def reverse(self) -> Multivector:
        """Reversion operator ~A: reverses order of basis vectors in each blade."""
        new_comp = {}
        for b, c in self.components.items():
            k = len(b)
            sign = -1 if (k * (k - 1) // 2) % 2 == 1 else 1
            new_comp[b] = c * sign
        return Multivector(signature=self.signature, components=new_comp)

    def wedge(self, other: Multivector) -> Multivector:
        """Outer (wedge) product A ^ B."""
        return self._outer_product(other)

    def __xor__(self, other: Multivector) -> Multivector:
        return self.wedge(other)

    def dot(self, other: Multivector) -> Multivector:
        """Inner (dot) product A . B."""
        return self._inner_product(other)

    def norm_sq(self) -> float:
        """Squared norm ||A||^2 = (A * ~A)_0."""
        return (self * self.reverse()).scalar_part()

    def norm(self) -> float:
        val = self.norm_sq()
        return math.sqrt(max(0.0, val))

    def inverse(self) -> Multivector:
        """Versor inverse M^{-1} = ~M / ||M||^2."""
        ns = self.norm_sq()
        if abs(ns) < 1e-12:
            raise ZeroDivisionError("Cannot invert multivector with zero norm.")
        return self.reverse() * (1.0 / ns)

    def sandwich(self, target: Multivector) -> Multivector:
        """Coordinate-free sandwich operator: target' = M * target * M^{-1}."""
        inv = self.inverse()
        return (self * target) * inv

    # Internal Clifford Algebra Multiplication Engine
    def _multiply_blades(
        self, b1: Tuple[int, ...], b2: Tuple[int, ...]
    ) -> Tuple[float, Tuple[int, ...]]:
        """Multiplies two basis blades according to signature (p, q, r).
        
        Generator indexing:
        1 .. p: positive (+1)
        p+1 .. p+q: negative (-1)
        p+q+1 .. p+q+r: zero (0)
        """
        p, q, r = self.signature
        sign = 1.0
        vecs = list(b1) + list(b2)
        # Sort vecs via adjacent swaps (each swap flips sign)
        n = len(vecs)
        for i in range(n):
            for j in range(0, n - i - 1):
                if vecs[j] > vecs[j + 1]:
                    vecs[j], vecs[j + 1] = vecs[j + 1], vecs[j]
                    sign = -sign

        # Contract identical adjacent generators
        final_vecs: List[int] = []
        idx = 0
        while idx < len(vecs):
            if idx + 1 < len(vecs) and vecs[idx] == vecs[idx + 1]:
                gen = vecs[idx]
                if 1 <= gen <= p:
                    sign *= 1.0
                elif p < gen <= p + q:
                    sign *= -1.0
                else:  # Zero generator
                    return 0.0, ()
                idx += 2
            else:
                final_vecs.append(vecs[idx])
                idx += 1

        return sign, tuple(final_vecs)

    def _geometric_product(self, other: Multivector) -> Multivector:
        res: Dict[Tuple[int, ...], float] = {}
        for b1, c1 in self.components.items():
            for b2, c2 in other.components.items():
                sign, b_res = self._multiply_blades(b1, b2)
                if abs(sign) > 1e-12:
                    val = c1 * c2 * sign
                    res[b_res] = res.get(b_res, 0.0) + val
        return Multivector(signature=self.signature, components=res)

    def _outer_product(self, other: Multivector) -> Multivector:
        """Grade-preserving outer product: only keep terms where grade = grade(A) + grade(B)."""
        res: Dict[Tuple[int, ...], float] = {}
        for b1, c1 in self.components.items():
            k1 = len(b1)
            for b2, c2 in other.components.items():
                k2 = len(b2)
                expected_grade = k1 + k2
                sign, b_res = self._multiply_blades(b1, b2)
                if len(b_res) == expected_grade and abs(sign) > 1e-12:
                    val = c1 * c2 * sign
                    res[b_res] = res.get(b_res, 0.0) + val
        return Multivector(signature=self.signature, components=res)

    def _inner_product(self, other: Multivector) -> Multivector:
        """Left contraction / inner product: grade = |grade(A) - grade(B)|."""
        res: Dict[Tuple[int, ...], float] = {}
        for b1, c1 in self.components.items():
            k1 = len(b1)
            for b2, c2 in other.components.items():
                k2 = len(b2)
                expected_grade = abs(k1 - k2)
                sign, b_res = self._multiply_blades(b1, b2)
                if len(b_res) == expected_grade and abs(sign) > 1e-12:
                    val = c1 * c2 * sign
                    res[b_res] = res.get(b_res, 0.0) + val
        return Multivector(signature=self.signature, components=res)


class PGA3D:
    """3D Projective Geometric Algebra (R^{3,0,1}).
    
    Generators:
    e_1, e_2, e_3 : Euclidean basis (e_i^2 = +1)
    e_0 (index 4) : Ideal / degenerate basis (e_0^2 = 0)
    """
    SIG = (3, 0, 1)

    @classmethod
    def point(cls, x: float, y: float, z: float) -> Multivector:
        """Creates a dual PGA point P = e_123 + x e_032 + y e_013 + z e_021."""
        return Multivector(
            signature=cls.SIG,
            components={
                (1, 2, 3): 1.0,
                (4, 2, 3): -x,
                (4, 1, 3): y,
                (4, 1, 2): -z,
            },
        )

    @classmethod
    def rotor(cls, angle: float, axis_x: float, axis_y: float, axis_z: float) -> Multivector:
        """Creates a PGA rotation rotor: R = cos(theta/2) + sin(theta/2) (u_x e_23 + u_y e_31 + u_z e_12)."""
        half = angle / 2.0
        c = math.cos(half)
        s = math.sin(half)
        length = math.sqrt(axis_x**2 + axis_y**2 + axis_z**2) or 1.0
        ux, uy, uz = axis_x / length, axis_y / length, axis_z / length
        return Multivector(
            signature=cls.SIG,
            components={
                (): c,
                (2, 3): s * ux,
                (1, 3): -s * uy,
                (1, 2): s * uz,
            },
        )

    @classmethod
    def translator(cls, dx: float, dy: float, dz: float) -> Multivector:
        """Creates a PGA translation motor: T = 1 - 0.5 (dx e_01 + dy e_02 + dz e_03)."""
        return Multivector(
            signature=cls.SIG,
            components={
                (): 1.0,
                (1, 4): 0.5 * dx,
                (2, 4): 0.5 * dy,
                (3, 4): 0.5 * dz,
            },
        )


class CGA3D:
    """3D Conformal Geometric Algebra (R^{4,1,0}).
    
    Generators:
    e_1, e_2, e_3 : Euclidean basis (e_i^2 = +1)
    e_4 : e_+ (e_+^2 = +1)
    e_5 : e_- (e_-^2 = -1)
    Null basis:
    e_0 = 0.5 * (e_- - e_+) (origin)
    e_inf = e_- + e_+ (infinity)
    """
    SIG = (4, 1, 0)

    @classmethod
    def e0(cls) -> Multivector:
        """Origin null vector e_0 = 0.5 * (e_5 - e_4)."""
        return Multivector(cls.SIG, {(5,): 0.5, (4,): -0.5})

    @classmethod
    def einf(cls) -> Multivector:
        """Infinity null vector e_inf = e_5 + e_4."""
        return Multivector(cls.SIG, {(5,): 1.0, (4,): 1.0})

    @classmethod
    def point(cls, x: float, y: float, z: float) -> Multivector:
        """Embeds a 3D Euclidean point into CGA: P = x e_1 + y e_2 + z e_3 + 0.5 |x|^2 e_inf + e_0."""
        r2 = x * x + y * y + z * z
        return (
            Multivector(cls.SIG, {(1,): x, (2,): y, (3,): z})
            + (cls.einf() * (0.5 * r2))
            + cls.e0()
        )

    @classmethod
    def sphere(cls, x: float, y: float, z: float, radius: float) -> Multivector:
        """Dual sphere: S = P(center) - 0.5 * r^2 e_inf."""
        p = cls.point(x, y, z)
        return p - (cls.einf() * (0.5 * radius * radius))

    @classmethod
    def test_intersection(cls, s1: Multivector, s2: Multivector) -> float:
        """Calculates geometric overlap between two conformal spheres or points.
        
        When wedge product s1 ^ s2 vanishes, their spatial boundaries intersect or coincide.
        """
        wedge_prod = s1.wedge(s2)
        return wedge_prod.norm()


class Clifford8D:
    """8-Dimensional Real Clifford Algebra Cl(p,q,r) with 2^8 = 256 basis blades.
    
    Generators: e_1, e_2, e_3, e_4, e_5, e_6, e_7, e_8.
    Binomial grade distribution binom(8, g): [1, 8, 28, 56, 70, 56, 28, 8, 1].
    """
    SIG_EUCLIDEAN: Tuple[int, int, int] = (8, 0, 0)
    SIG_PROJECTIVE: Tuple[int, int, int] = (7, 0, 1)
    SIG_SPACETIME: Tuple[int, int, int] = (6, 2, 0)
    SIG_CONFORMAL: Tuple[int, int, int] = (7, 1, 0)

    # Binomial grade counts for grades 0..8
    GRADE_DIMENSIONS: Tuple[int, ...] = (1, 8, 28, 56, 70, 56, 28, 8, 1)

    @classmethod
    def vector_8d(
        cls,
        coords: Tuple[float, float, float, float, float, float, float, float],
        signature: Tuple[int, int, int] = (8, 0, 0),
    ) -> Multivector:
        """Embeds an 8D coordinate vector x in M^8 into a grade-1 multivector."""
        comps = {(i + 1,): float(coords[i]) for i in range(8) if abs(coords[i]) > 1e-12}
        return Multivector(signature=signature, components=comps)

    @classmethod
    def bivector_28(
        cls,
        bivector_dict: Dict[Tuple[int, int], float],
        signature: Tuple[int, int, int] = (8, 0, 0),
    ) -> Multivector:
        """Constructs a grade-2 multivector from up to binom(8,2) = 28 bivector generators."""
        comps: Dict[Tuple[int, ...], float] = {}
        for (i, j), val in bivector_dict.items():
            if abs(val) > 1e-12:
                blade = tuple(sorted((i, j)))
                sign = 1.0 if (i < j) else -1.0
                comps[blade] = comps.get(blade, 0.0) + val * sign
        return Multivector(signature=signature, components=comps)

    @classmethod
    def dual_stream_attention_score(
        cls,
        q: Multivector,
        k: Multivector,
        lambda_wedge: float = 0.5,
    ) -> float:
        """Evaluates 8D Dual-Stream Clifford Attention score:
        
        E_ij = (1 / sqrt(256)) * ( <Q * ~K>_0 - lambda_wedge * ||Q ^ K||_Cl^2 )
             = (1 / 16) * ( <Q * ~K>_0 - lambda_wedge * ||Q ^ K||_Cl^2 ).
        """
        # Metric alignment: scalar part of Q * ~K
        metric_alignment = (q * k.reverse()).scalar_part()

        # Exterior wedge meet: squared norm of Q ^ K
        q_wedge_k = q.wedge(k)
        wedge_dist_sq = sum(c * c for c in q_wedge_k.components.values())

        score = (metric_alignment - lambda_wedge * wedge_dist_sq) / 16.0
        return float(score)

