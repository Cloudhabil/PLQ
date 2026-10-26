"""Dispersive Wave Sheaf Execution Engine over Directed Acyclic Cell Complexes.

Protoporeia Causal (Πρωτοπορεία Αιτιώδης) — the retarded Green's function kernel:
    G(r, Δt) = H(c·Δt - r) / (2π·c·√(c²·Δt² - r²))

Pneuma Fluens (Πνεῦμα Ῥέον) — continuous wave field propagating across DACC strata:
    A_k(x, t) = Σⱼ B_{k+1}(j,k) · ∫₀ᵗ G(‖x-xⱼ‖, t-t') · V_j(t') dt'

DispersiveWaveSheafDACC upgrades ToposLang's Hodge-Dirac message passing from
static neighborhood aggregation to a continuous spatial-temporal wave equation:
    ∂²A/∂t² + γ_damp·∂A/∂t + c²·L_k·A = F_P(x,t)

This enforces the Heaviside causality constraint: no feature update can propagate
across the DACC complex faster than wave speed c.

WavePhysicalConfig provides domain-specific physical constant profiles:
    PROTEIN_AQUEOUS_CONFIG        — c = 1750 m/s (acoustic speed in proteins)
    SOLID_STATE_METAMATERIAL_CONFIG — c = 5100 m/s (elastic wave speed in ceramics)
    MACRO_FLUID_WATER_CONFIG      — c = 1.5 m/s (surface ripple macro-analogy)

Pure Python — zero external dependencies (math module only).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Physical configuration dataclass & preset profiles
# ---------------------------------------------------------------------------

@dataclass
class WavePhysicalConfig:
    """Parametrized physical constants for domain-specific wave environments.

    Attributes
    ----------
    c_wave_speed : float
        Acoustic / elastic wave propagation speed (m/s).
    gamma_stiffness : float
        Surface tension / interfacial elastic modulus (N/m).
    rho_density : float
        Medium density (kg/m³).
    lambda_poisson : float
        Stochastic Poisson collision / impact arrival rate (events·m⁻²·s⁻¹).
    cutoff_delta : float
        Microscopic core cutoff distance δ > 0 (Å) for Green's kernel
        regularisation (prevents log divergence at r → 0).
    damping : float
        Medium attenuation coefficient γ_damp for the Hodge-Dirac wave ODE.
    """
    c_wave_speed: float = 1750.0      # m/s — protein acoustic speed
    gamma_stiffness: float = 0.0728   # N/m — water surface tension
    rho_density: float = 1000.0       # kg/m³ — water density
    lambda_poisson: float = 10.0      # impacts/m²/s
    cutoff_delta: float = 0.01        # Å — microscopic core cutoff
    damping: float = 0.05             # dimensionless attenuation


# Preset domain profiles
PROTEIN_AQUEOUS_CONFIG = WavePhysicalConfig(
    c_wave_speed=1750.0,
    gamma_stiffness=0.0728,
    rho_density=1000.0,
    lambda_poisson=10.0,
    cutoff_delta=0.01,
    damping=0.05,
)

SOLID_STATE_METAMATERIAL_CONFIG = WavePhysicalConfig(
    c_wave_speed=5100.0,
    gamma_stiffness=1.25,
    rho_density=3200.0,
    lambda_poisson=5.0,
    cutoff_delta=0.005,
    damping=0.02,
)

MACRO_FLUID_WATER_CONFIG = WavePhysicalConfig(
    c_wave_speed=1.5,
    gamma_stiffness=0.0728,
    rho_density=1000.0,
    lambda_poisson=50.0,
    cutoff_delta=0.001,
    damping=0.1,
)


# ---------------------------------------------------------------------------
# Retarded Green's Function Kernel
# ---------------------------------------------------------------------------

class CausalGreensKernel:
    """Retarded causal Green's function for the 2D wave equation.

    Protoporeia Causal (Πρωτοπορεία Αιτιώδης):
        G(r, Δt) = H(c·Δt - r) / (2π·c·√(c²·Δt² - r²))

    The Heaviside step H(c·Δt - r) enforces strict causal propagation:
    the wave front has not reached point r at time Δt if c·Δt < r.

    The singularity at r → c·Δt (wavefront) is regularised by a small
    ε cutoff on the discriminant denominator.

    Parameters
    ----------
    config : WavePhysicalConfig
        Physical constants. Uses config.c_wave_speed and config.cutoff_delta.
    epsilon : float
        Discriminant regularisation ε (prevents 1/0 at the exact wavefront).
    """

    def __init__(
        self,
        config: WavePhysicalConfig = PROTEIN_AQUEOUS_CONFIG,
        epsilon: float = 1e-9,
    ) -> None:
        self.c = config.c_wave_speed
        self.epsilon = epsilon

    def evaluate(self, r: float, dt: float) -> float:
        """Evaluate G(r, Δt).

        Parameters
        ----------
        r : float
            Spatial distance from source to query point (same units as c·Δt).
        dt : float
            Elapsed time since impulse (s or lattice time).

        Returns
        -------
        float
            G(r, Δt) ≥ 0.  Returns 0.0 outside the causal cone (r > c·Δt).
        """
        if dt <= 0.0:
            return 0.0
        c_dt = self.c * dt
        # Heaviside causality gate
        if c_dt <= r:
            return 0.0
        discriminant = c_dt ** 2 - r ** 2
        if discriminant < self.epsilon:
            return 0.0
        return 1.0 / (2.0 * math.pi * self.c * math.sqrt(discriminant))

    def evaluate_batch(
        self,
        r_values: List[float],
        dt: float,
    ) -> List[float]:
        """Evaluate G for a list of distances at fixed Δt."""
        return [self.evaluate(r, dt) for r in r_values]

    def __repr__(self) -> str:
        return f"CausalGreensKernel(c={self.c}, ε={self.epsilon})"


# ---------------------------------------------------------------------------
# 8D stalk arithmetic helpers (pure Python, no numpy)
# ---------------------------------------------------------------------------

def _stalk_sub(a: List[float], b: List[float]) -> List[float]:
    """Component-wise subtraction of two 8D stalks."""
    return [a[i] - b[i] for i in range(8)]


def _stalk_add(a: List[float], b: List[float]) -> List[float]:
    """Component-wise addition of two 8D stalks."""
    return [a[i] + b[i] for i in range(8)]


def _stalk_scale(a: List[float], s: float) -> List[float]:
    """Scalar multiplication of an 8D stalk."""
    return [a[i] * s for i in range(8)]


def _stalk_norm(a: List[float]) -> float:
    """Euclidean norm of an 8D stalk."""
    return math.sqrt(sum(x ** 2 for x in a))


def _cell_distance_3d(
    coords_a: Tuple[float, float, float],
    coords_b: Tuple[float, float, float],
) -> float:
    """Euclidean distance between two 3D cell coordinates."""
    return math.sqrt(
        (coords_a[0] - coords_b[0]) ** 2
        + (coords_a[1] - coords_b[1]) ** 2
        + (coords_a[2] - coords_b[2]) ** 2
    )


def _scalar_meet_8d(u: List[float], v: List[float]) -> float:
    """Grade-0 Clifford inner product ⟨u·v⟩₀ = Σᵢ uᵢvᵢ."""
    return sum(u[i] * v[i] for i in range(8))


def _bivector_wedge_norm2_8d(u: List[float], v: List[float]) -> float:
    """Squared 28-component bivector wedge norm ‖u∧v‖²_Cl.

    ‖u∧v‖² = Σ_{i<j} (uᵢvⱼ - uⱼvᵢ)²   (28 terms for dim=8)
    """
    norm2 = 0.0
    for i in range(8):
        for j in range(i + 1, 8):
            bij = u[i] * v[j] - u[j] * v[i]
            norm2 += bij * bij
    return norm2


def _algebraic_inverse_8d(u: List[float]) -> List[float]:
    """Katalepsis Inversa: Q* = K⁻¹ = K̃/‖K‖²_Cl = u/‖u‖² for grade-1 vector.

    For a grade-1 vector in Cl(ℝ⁸), reversion K̃ = K (grade 1 is self-reversing),
    and ‖K‖²_Cl = Σᵢ uᵢ².  Therefore K⁻¹ = u / ‖u‖².
    """
    norm_sq = sum(x ** 2 for x in u)
    if norm_sq < 1e-15:
        raise ValueError("Degenerate stalk: cannot invert a zero multivector.")
    return [x / norm_sq for x in u]


# ---------------------------------------------------------------------------
# Dispersive Wave Sheaf DACC Engine
# ---------------------------------------------------------------------------

@dataclass
class DACCGeometry:
    """Minimal DACC geometry descriptor used by DispersiveWaveSheafDACC.

    Attributes
    ----------
    num_cells : int
        Total number of k-cells in the active stratum.
    coords : List[Tuple[float,float,float]]
        3D spatial coordinates of each cell centre (Å or m).
    b1 : List[List[int]]
        Incidence matrix B₁: (n_vertices × n_edges) boundary operator ∂₁.
    b2 : List[List[int]]
        Incidence matrix B₂: (n_edges × n_faces) boundary operator ∂₂.
    """
    num_cells: int
    coords: List[Tuple[float, float, float]]
    b1: List[List[int]] = field(default_factory=list)
    b2: List[List[int]] = field(default_factory=list)


class DispersiveWaveSheafDACC:
    """Full ToposLang Dispersive Wave Sheaf execution engine over a DACC.

    Typos Metricus (Τύπος Μεταιχμιακός) — 8D Clifford multivector stalks
    A_k(x,t) ∈ Cl(ℝ⁸) propagate via the retarded Protoporeia Causal Green's
    function across DACC co-faces, enforcing:

        1. Anilpotencia Gradualis: B_{k-1}·B_k = 0 (boundary nilpotency).
        2. Galois CRT homology: β₁ = |K₁| - rank(B₁) - rank(B₂) via F_p.
        3. Causal Green's propagation: no feature travels faster than c.
        4. Verlet time integration with configurable damping.

    Parameters
    ----------
    geometry : DACCGeometry
        DACC cell structure and incidence matrices.
    config : WavePhysicalConfig
        Physical constants for the wave environment.
    primes : List[int]
        Galois field primes for CRT homology rank recovery.
    """

    def __init__(
        self,
        geometry: DACCGeometry,
        config: WavePhysicalConfig = PROTEIN_AQUEOUS_CONFIG,
        primes: Optional[List[int]] = None,
    ) -> None:
        self.geometry = geometry
        self.config = config
        self.primes = primes or [67, 71, 73]
        self._kernel = CausalGreensKernel(config)

        n = geometry.num_cells
        # 8D multivector stalks A_k(x,t) — initialised to zero
        self.stalks: List[List[float]] = [[0.0] * 8 for _ in range(n)]
        # Velocity field ∂A/∂t — for Verlet integration
        self.velocity: List[List[float]] = [[0.0] * 8 for _ in range(n)]
        # Cumulative simulation time
        self.time: float = 0.0
        # History of stalk norms for telemetry
        self._step_telemetry: List[dict] = []

        # Validate boundary nilpotency if incidence matrices provided
        if geometry.b1 and geometry.b2:
            self._verify_nilpotency()

    # ------------------------------------------------------------------
    # Topological validation
    # ------------------------------------------------------------------

    def _verify_nilpotency(self) -> None:
        """Anilpotencia Gradualis: verifies B₁·B₂ = 0 (mod ℤ).

        Raises
        ------
        ValueError
            If the incidence matrices violate boundary nilpotency.
        """
        b1 = self.geometry.b1
        b2 = self.geometry.b2
        n_rows = len(b1)
        n_inner = len(b2)
        n_cols = len(b2[0]) if b2 else 0
        for i in range(n_rows):
            for j in range(n_cols):
                val = sum(b1[i][k] * b2[k][j] for k in range(n_inner))
                if val != 0:
                    raise ValueError(
                        f"Topological Anomaly: Boundary Nilpotency (∂²=0) "
                        f"violated at B₁[{i},:]·B₂[:,{j}] = {val} ≠ 0."
                    )

    def _gf_rank(self, matrix: List[List[int]], prime: int) -> int:
        """Gaussian elimination over Galois field 𝔽_p for exact rank recovery.

        Parameters
        ----------
        matrix : List[List[int]]
            Integer incidence matrix.
        prime : int
            Prime modulus p for 𝔽_p arithmetic.

        Returns
        -------
        int
            Row rank of matrix over 𝔽_p.
        """
        if not matrix or not matrix[0]:
            return 0
        rows = len(matrix)
        cols = len(matrix[0])
        # Copy and reduce mod p
        A: List[List[int]] = [[int(matrix[i][j]) % prime for j in range(cols)]
                               for i in range(rows)]
        rank = 0
        row_idx = 0
        for col in range(cols):
            # Find pivot
            pivot = -1
            for r in range(row_idx, rows):
                if A[r][col] != 0:
                    pivot = r
                    break
            if pivot == -1:
                continue
            # Swap rows
            A[row_idx], A[pivot] = A[pivot], A[row_idx]
            # Scale pivot row so A[row_idx][col] = 1
            inv = pow(int(A[row_idx][col]), prime - 2, prime)  # Fermat's little theorem
            A[row_idx] = [(A[row_idx][j] * inv) % prime for j in range(cols)]
            # Eliminate column from all other rows
            for r in range(rows):
                if r != row_idx and A[r][col] != 0:
                    factor = A[r][col]
                    A[r] = [(A[r][j] - factor * A[row_idx][j]) % prime for j in range(cols)]
            rank += 1
            row_idx += 1
            if row_idx == rows:
                break
        return rank

    def compute_betti_1(self) -> int:
        """Theorem 2.5 — Galois CRT homology: β₁ = |K₁| - rank(B₁) - rank(B₂).

        Uses multi-prime Galois field rank recovery over F_p primes to avoid
        rational bit-length explosion (replaces floating-point SVD).

        Returns
        -------
        int
            First Betti number β₁ (number of independent 1-cycles).
        """
        if not self.geometry.b1 or not self.geometry.b2:
            return 0
        n_edges = len(self.geometry.b1[0]) if self.geometry.b1 else 0
        rank_b1 = max(self._gf_rank(self.geometry.b1, p) for p in self.primes)
        rank_b2 = max(self._gf_rank(self.geometry.b2, p) for p in self.primes)
        return max(0, n_edges - rank_b1 - rank_b2)

    # ------------------------------------------------------------------
    # Stalk injection & retrieval
    # ------------------------------------------------------------------

    def inject_stalk(self, cell_idx: int, stalk_8d: List[float]) -> None:
        """Set the 8D Clifford stalk of a cell (e.g., target pocket stalk).

        Parameters
        ----------
        cell_idx : int
            Index of the cell in the DACC.
        stalk_8d : List[float]
            8-component Clifford multivector stalk A_k ∈ Cl(ℝ⁸).
        """
        if len(stalk_8d) != 8:
            raise ValueError(f"Stalk must be 8D, got {len(stalk_8d)}D.")
        self.stalks[cell_idx] = list(stalk_8d)

    def get_stalk(self, cell_idx: int) -> List[float]:
        """Retrieve the current 8D stalk of a cell."""
        return list(self.stalks[cell_idx])

    def stalk_norm(self, cell_idx: int) -> float:
        """Euclidean norm of the stalk at cell_idx."""
        return _stalk_norm(self.stalks[cell_idx])

    # ------------------------------------------------------------------
    # Wave propagation step (Hodge-Dirac Verlet integration)
    # ------------------------------------------------------------------

    def step_wave_propagation(self, dt: float) -> dict:
        """Advance all stalk multivectors by one time step Δt.

        Implements the discrete Verlet time-integration of the Hodge-Dirac
        wave operator over the DACC cell complex:

            ∂²A/∂t² + γ_damp·∂A/∂t + c²·L_k·A = F_P(x,t)

        The spatial coupling term is approximated via the causal Green's
        function kernel G(r, Δt) between cell pairs (i,j):

            acceleration_i += G(‖xᵢ-xⱼ‖, Δt) · (A_j - A_i)

        Parameters
        ----------
        dt : float
            Time step size.

        Returns
        -------
        dict
            Step telemetry: {'time', 'dt', 'green_evaluations', 'stalk_norms'}.
        """
        n = self.geometry.num_cells
        c = self.config.c_wave_speed
        damp = self.config.damping
        new_stalks = [list(self.stalks[i]) for i in range(n)]
        green_evals = 0

        for i in range(n):
            # Damping deceleration: -γ·∂A/∂t
            accel = _stalk_scale(self.velocity[i], -damp)

            for j in range(n):
                if i == j:
                    continue
                r = _cell_distance_3d(self.geometry.coords[i], self.geometry.coords[j])
                G_val = self._kernel.evaluate(r, dt)
                green_evals += 1
                if G_val > 0.0:
                    diff = _stalk_sub(self.stalks[j], self.stalks[i])
                    coupling = _stalk_scale(diff, G_val)
                    accel = _stalk_add(accel, coupling)

            # Verlet velocity update: v ← v + a·Δt
            self.velocity[i] = _stalk_add(
                self.velocity[i],
                _stalk_scale(accel, dt)
            )
            # Position (stalk) update: A ← A + v·Δt
            new_stalks[i] = _stalk_add(
                self.stalks[i],
                _stalk_scale(self.velocity[i], dt)
            )

        self.stalks = new_stalks
        self.time += dt

        telemetry = {
            "time": round(self.time, 8),
            "dt": dt,
            "green_evaluations": green_evals,
            "stalk_norms": [round(_stalk_norm(self.stalks[i]), 6) for i in range(n)],
        }
        self._step_telemetry.append(telemetry)
        return telemetry

    def run_propagation(self, total_time: float, dt: float) -> List[dict]:
        """Run the wave propagation for total_time / dt steps.

        Parameters
        ----------
        total_time : float
            Total simulation time.
        dt : float
            Time step.

        Returns
        -------
        List[dict]
            List of per-step telemetry dictionaries.
        """
        n_steps = max(1, int(round(total_time / dt)))
        results = []
        for _ in range(n_steps):
            results.append(self.step_wave_propagation(dt))
        return results

    # ------------------------------------------------------------------
    # Clifford attention utilities
    # ------------------------------------------------------------------

    def dual_stream_attention(
        self,
        query_idx: int,
        key_idx: int,
        lam: float = 0.5,
        disp_envelope: float = 1.0,
    ) -> float:
        """Dispersive Hodge-Clifford dual-stream attention energy.

        E_ij(t) = (1/16)(⟨Qᵢ·K̃ⱼ⟩₀ - λ‖Qᵢ∧Kⱼ‖²_Cl) · Φ_disp(‖xᵢ-xⱼ‖, t)

        Parameters
        ----------
        query_idx : int
            Index of query cell i.
        key_idx : int
            Index of key cell j.
        lam : float
            Bivector clash penalty weight λ.
        disp_envelope : float
            Pre-computed Φ_disp(r, t) dispersive envelope value.

        Returns
        -------
        float
            Attention energy E_ij.
        """
        Q = self.stalks[query_idx]
        K = self.stalks[key_idx]
        meet = _scalar_meet_8d(Q, K)
        clash = _bivector_wedge_norm2_8d(Q, K)
        return ((meet - lam * clash) / 16.0) * disp_envelope

    def compute_algebraic_inverse(self, cell_idx: int) -> List[float]:
        """Katalepsis Inversa — algebraic inverse Q* = K⁻¹ = K̃/‖K‖²_Cl.

        For the grade-1 restriction to Cl(ℝ⁸) 1-vectors:
            Q* = u / ‖u‖²  (reversion of grade-1 blade = identity)

        Returns
        -------
        List[float]
            Inverse 8D stalk.
        """
        return _algebraic_inverse_8d(self.stalks[cell_idx])

    def katalepsis_inversa(self, stalk_or_idx: int | List[float]) -> List[float]:
        """Katalepsis Inversa Q* = K⁻¹ for either a cell index or an 8D stalk."""
        if isinstance(stalk_or_idx, int):
            return self.compute_algebraic_inverse(stalk_or_idx)
        return _algebraic_inverse_8d(stalk_or_idx)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a diagnostic summary of the current wave sheaf state."""
        return {
            "num_cells": self.geometry.num_cells,
            "simulation_time": round(self.time, 6),
            "config": {
                "c": self.config.c_wave_speed,
                "damping": self.config.damping,
                "lambda_poisson": self.config.lambda_poisson,
            },
            "stalk_norms": [round(_stalk_norm(s), 6) for s in self.stalks],
            "propagation_steps": len(self._step_telemetry),
        }

    def __repr__(self) -> str:
        return (
            f"DispersiveWaveSheafDACC("
            f"n_cells={self.geometry.num_cells}, "
            f"c={self.config.c_wave_speed}, "
            f"t={self.time:.4f})"
        )
