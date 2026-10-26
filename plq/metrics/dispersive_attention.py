"""Dispersive Hodge-Clifford Spatial Transformer Attention Kernel for PLQ.

Ateneia Dispersiva Cliffordiana (Attentio Dispersiva):
    Full Theorem 3.3 implementation of the dispersive dual-stream attention energy:

        E_ij(t) = (1/16)(⟨Qᵢ·K̃ⱼ⟩₀ - λ‖Qᵢ∧Kⱼ‖²_Cl) · Φ_disp(‖xᵢ-xⱼ‖, t)

    where:
        ⟨Q·K̃⟩₀  — scalar meet (grade-0 inner product): geometric alignment
        ‖Q∧K‖²   — bivector wedge norm: steric clash penalty
        Φ_disp   — capillary-gravity dispersive envelope:
                   Φ_disp(r,t) = ∫₀^∞ S(k)·cos(kr - ω(k)t)dk

Multi-scale routing:
    High-k (capillary):  c_g ≈ 3/2·c_p → local side-chain / rotamer contacts (fast)
    Low-k  (gravity):    c_g ≈ 1/2·c_p → global allosteric / domain shifts  (slow)

Pure Python — zero external dependencies (math module only).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# 8D stalk helpers (pure Python)
# ---------------------------------------------------------------------------

def _scalar_meet_8d(u: List[float], v: List[float]) -> float:
    """⟨u·v⟩₀ = Σᵢ uᵢvᵢ — grade-0 Clifford inner product."""
    return sum(u[i] * v[i] for i in range(8))


def _bivector_wedge_norm2_8d(u: List[float], v: List[float]) -> float:
    """‖u∧v‖²_Cl = Σ_{i<j}(uᵢvⱼ - uⱼvᵢ)² — 28-component bivector clash norm."""
    norm2 = 0.0
    for i in range(8):
        for j in range(i + 1, 8):
            bij = u[i] * v[j] - u[j] * v[i]
            norm2 += bij * bij
    return norm2


def _distance_3d(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


# ---------------------------------------------------------------------------
# Dispersive envelope (inline capillary-gravity dispersion)
# ---------------------------------------------------------------------------

def _dispersive_envelope(
    r: float,
    t: float,
    g: float = 9.81,
    gamma: float = 0.0728,
    rho: float = 1000.0,
    h: float = 1.0,
    n_modes: int = 64,
    k_min: float = 0.1,
    k_max: float = 100.0,
) -> float:
    """Compute Φ_disp(r, t) = ∫ S(k)·cos(kr-ω(k)t)dk via Riemann sum.

    Self-contained inline implementation to avoid circular imports.
    For full CapillaryGravityDispersion, use topos.physics.dispersion.
    """
    if t < 0.0:
        return 0.0
    dk = (k_max - k_min) / n_modes
    total = 0.0
    for i in range(n_modes):
        k = k_min + (i + 0.5) * dk
        # ω²(k) = (gk + γk³/ρ)·tanh(kh)
        om2 = (g * k + gamma * k**3 / rho) * math.tanh(h * k)
        om = math.sqrt(max(0.0, om2))
        S_k = math.exp(-(k / k_max)**2)
        total += S_k * math.cos(k * r - om * t) * dk
    return total


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AttentionMatrixResult:
    """Output of DispersiveHodgeCliffordAttention.compute_attention_matrix().

    Attributes
    ----------
    energy_matrix : List[List[float]]
        n×m matrix E_ij(t) of dispersive attention energies.
    meet_matrix : List[List[float]]
        Scalar meet values ⟨Qᵢ·K̃ⱼ⟩₀.
    clash_matrix : List[List[float]]
        Bivector clash norms ‖Qᵢ∧Kⱼ‖²_Cl.
    disp_matrix : List[List[float]]
        Dispersive envelope values Φ_disp(rᵢⱼ, t).
    n_queries : int
    n_keys : int
    """
    energy_matrix: List[List[float]]
    meet_matrix: List[List[float]]
    clash_matrix: List[List[float]]
    disp_matrix: List[List[float]]
    n_queries: int
    n_keys: int

    def best_match(self) -> Tuple[int, int, float]:
        """Return (query_idx, key_idx, energy) of the highest-energy pair."""
        best_i, best_j, best_e = 0, 0, float("-inf")
        for i, row in enumerate(self.energy_matrix):
            for j, e in enumerate(row):
                if e > best_e:
                    best_e = e
                    best_i, best_j = i, j
        return best_i, best_j, best_e

    def favorable_count(self, threshold: float = 0.0) -> int:
        """Count pairs with E_ij > threshold (positive / attractive)."""
        return sum(
            1 for row in self.energy_matrix for e in row if e > threshold
        )


# ---------------------------------------------------------------------------
# Dispersive Hodge-Clifford Attention
# ---------------------------------------------------------------------------

class DispersiveHodgeCliffordAttention:
    """Dispersive Hodge-Clifford dual-stream attention kernel.

    Theorem 3.3 (from Topological Mechanics paper):
        E_ij(t) = (1/16)(⟨Qᵢ·K̃ⱼ⟩₀ - λ‖Qᵢ∧Kⱼ‖²_Cl) · Φ_disp(‖xᵢ-xⱼ‖, t)

    The dispersive envelope Φ_disp(r, t) physically separates:
        - High-frequency capillary modes: fast local side-chain / lattice contacts.
        - Low-frequency gravity modes:    slow global allosteric domain shifts.

    This replaces instantaneous distance-based attention with causal,
    wave-speed-limited feature propagation.

    Parameters
    ----------
    lam : float
        Bivector clash penalty weight λ (Theorem 3.3 default: 0.5).
    use_dispersive_envelope : bool
        If True, modulate attention by Φ_disp(r,t). If False, Φ_disp = 1 (static).
    g : float
        Effective gravity for capillary-gravity dispersion.
    gamma : float
        Surface tension / stiffness.
    rho : float
        Medium density.
    h : float
        Medium depth / lattice thickness.
    n_modes : int
        Number of integration modes for Φ_disp quadrature.
    """

    def __init__(
        self,
        lam: float = 0.5,
        use_dispersive_envelope: bool = True,
        g: float = 9.81,
        gamma: float = 0.0728,
        rho: float = 1000.0,
        h: float = 1.0,
        n_modes: int = 64,
    ) -> None:
        self.lam = lam
        self.use_dispersive = use_dispersive_envelope
        self.g = g
        self.gamma = gamma
        self.rho = rho
        self.h = h
        self.n_modes = n_modes

    def _envelope(self, r: float, t: float) -> float:
        if not self.use_dispersive or t <= 0.0:
            return 1.0
        return _dispersive_envelope(
            r, t, self.g, self.gamma, self.rho, self.h, self.n_modes
        )

    def single_energy(
        self,
        query: List[float],
        key: List[float],
        r: float = 0.0,
        t: float = 0.0,
    ) -> Tuple[float, float, float, float]:
        """Compute E_ij(t) for a single (query, key) stalk pair.

        Parameters
        ----------
        query : List[float]
            Query stalk Qᵢ ∈ ℝ⁸.
        key : List[float]
            Key stalk Kⱼ ∈ ℝ⁸.
        r : float
            Spatial distance ‖xᵢ - xⱼ‖.
        t : float
            Elapsed time.

        Returns
        -------
        Tuple[float, float, float, float]
            (E_ij, meet, clash, disp_envelope).
        """
        meet = _scalar_meet_8d(query, key)
        clash = _bivector_wedge_norm2_8d(query, key)
        phi = self._envelope(r, t)
        energy = ((meet - self.lam * clash) / 16.0) * phi
        return energy, meet, clash, phi

    def compute_attention_matrix(
        self,
        query_stalks: List[List[float]],
        key_stalks: List[List[float]],
        query_positions: Optional[List[Tuple[float, float, float]]] = None,
        key_positions: Optional[List[Tuple[float, float, float]]] = None,
        t: float = 0.0,
    ) -> AttentionMatrixResult:
        """Compute the full dispersive attention matrix E(t) of shape n×m.

        Parameters
        ----------
        query_stalks : List[List[float]]
            n query stalks Qᵢ ∈ ℝ⁸.
        key_stalks : List[List[float]]
            m key stalks Kⱼ ∈ ℝ⁸.
        query_positions : List[Tuple[float,float,float]], optional
            3D coordinates of query cells. If None, r=0 (no dispersive envelope).
        key_positions : List[Tuple[float,float,float]], optional
            3D coordinates of key cells.
        t : float
            Elapsed simulation time.

        Returns
        -------
        AttentionMatrixResult
        """
        n = len(query_stalks)
        m = len(key_stalks)

        energy_mat: List[List[float]] = []
        meet_mat: List[List[float]] = []
        clash_mat: List[List[float]] = []
        disp_mat: List[List[float]] = []

        for i in range(n):
            row_e, row_m, row_c, row_d = [], [], [], []
            for j in range(m):
                # Spatial distance
                if query_positions and key_positions:
                    r = _distance_3d(query_positions[i], key_positions[j])
                else:
                    r = 0.0
                e, meet, clash, phi = self.single_energy(
                    query_stalks[i], key_stalks[j], r, t
                )
                row_e.append(round(e, 8))
                row_m.append(round(meet, 8))
                row_c.append(round(clash, 8))
                row_d.append(round(phi, 8))
            energy_mat.append(row_e)
            meet_mat.append(row_m)
            clash_mat.append(row_c)
            disp_mat.append(row_d)

        return AttentionMatrixResult(
            energy_matrix=energy_mat,
            meet_matrix=meet_mat,
            clash_matrix=clash_mat,
            disp_matrix=disp_mat,
            n_queries=n,
            n_keys=m,
        )

    def __repr__(self) -> str:
        return (
            f"DispersiveHodgeCliffordAttention("
            f"λ={self.lam}, dispersive={self.use_dispersive})"
        )
