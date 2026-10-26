"""Capillary-Gravity Dispersion Relation for the ToposLang Wave Sheaf Engine.

Symploke Dispersiva (Συμπλοκὴ Διασκεδαστική):
    Governs multi-scale group velocity routing in the Hodge-Clifford Spatial
    Transformer, separating high-frequency local capillary modes (side-chain
    rotamers, lattice defects) from low-frequency global gravity modes (domain
    movements, allosteric hinge bending, acoustic phonons).

Dispersion Relation (Section 2.2 of Topological Mechanics paper):
    ω²(k) = (gk + γk³/ρ) · tanh(kh)

Group Velocity Asymptotes:
    Capillary regime (k ≫ ρg/γ):  c_g ≈ (3/2)·c_p  — local, high-frequency
    Gravity regime  (k ≪ ρg/γ):  c_g ≈ (1/2)·c_p  — global, low-frequency

Pure Python — zero external dependencies (math module only).
"""

from __future__ import annotations
import math
from typing import Literal


class CapillaryGravityDispersion:
    """Closed-form capillary-gravity dispersion relation ω²(k) = (gk + γk³/ρ)·tanh(kh).

    Parameters
    ----------
    g : float
        Effective gravitational acceleration / long-range restoring force (m/s²).
        For proteins: acoustic restoring modulus / effective lattice gravity.
    gamma : float
        Surface tension / interfacial elastic stiffness (N/m).
    rho : float
        Medium density (kg/m³).
    h : float
        Medium depth / lattice thickness (m or Å, consistent with other params).
    """

    def __init__(
        self,
        g: float = 9.81,
        gamma: float = 0.0728,
        rho: float = 1000.0,
        h: float = 1.0,
    ) -> None:
        if rho <= 0.0:
            raise ValueError("Density rho must be strictly positive.")
        if h <= 0.0:
            raise ValueError("Medium depth h must be strictly positive.")
        self.g = g
        self.gamma = gamma
        self.rho = rho
        self.h = h
        # Crossover wavenumber between capillary and gravity regimes
        # k_cross = sqrt(rho * g / gamma)  — derived from ω²_gravity = ω²_capillary
        self._k_cross: float = math.sqrt(rho * g / gamma) if gamma > 0.0 else float("inf")

    # ------------------------------------------------------------------
    # Core dispersion quantities
    # ------------------------------------------------------------------

    def omega_squared(self, k: float) -> float:
        """Squared angular frequency ω²(k) = (gk + γk³/ρ)·tanh(kh).

        Parameters
        ----------
        k : float
            Wavenumber (rad/m or rad/Å).

        Returns
        -------
        float
            ω²(k) ≥ 0.
        """
        if k <= 0.0:
            return 0.0
        capillary_term = self.gamma * k ** 3 / self.rho
        gravity_term = self.g * k
        return (gravity_term + capillary_term) * math.tanh(self.h * k)

    def omega(self, k: float) -> float:
        """Angular frequency ω(k) = sqrt(ω²(k))."""
        return math.sqrt(max(0.0, self.omega_squared(k)))

    def phase_velocity(self, k: float) -> float:
        """Phase velocity c_p(k) = ω(k)/k.

        Resolves the local propagation speed of a single Fourier component.
        """
        if k <= 0.0:
            return 0.0
        return self.omega(k) / k

    def group_velocity(self, k: float, dk: float = 1e-5) -> float:
        """Group velocity c_g(k) = dω/dk via central finite difference.

        The group velocity governs energy / information transport speed and
        provides the multi-scale routing in dispersive Hodge-Clifford attention.

        Parameters
        ----------
        k : float
            Wavenumber (rad/m).
        dk : float
            Finite-difference step (default 1e-5 rad/m).
        """
        if k <= dk:
            return self._group_velocity_analytic(dk)
        return (self.omega(k + dk) - self.omega(k - dk)) / (2.0 * dk)

    def _group_velocity_analytic(self, k: float) -> float:
        """Analytic derivative c_g = dω/dk = (1/2ω)·d(ω²)/dk.

        d(ω²)/dk = [g + 3γk²/ρ]·tanh(kh) + (gk + γk³/ρ)·h·sech²(kh)
        """
        if k <= 0.0:
            return 0.0
        om = self.omega(k)
        if om < 1e-15:
            return 0.0
        tanh_kh = math.tanh(self.h * k)
        sech2_kh = 1.0 - tanh_kh ** 2  # sech²(kh) = 1 - tanh²(kh)
        d_om2_dk = (
            (self.g + 3.0 * self.gamma * k ** 2 / self.rho) * tanh_kh
            + (self.g * k + self.gamma * k ** 3 / self.rho) * self.h * sech2_kh
        )
        return d_om2_dk / (2.0 * om)

    # ------------------------------------------------------------------
    # Regime classification & asymptotes
    # ------------------------------------------------------------------

    def regime(self, k: float) -> Literal["capillary", "gravity", "mixed"]:
        """Classify wavenumber into capillary, gravity, or mixed regime.

        Returns
        -------
        str
            'capillary' if k ≫ k_cross (c_g ≈ 3/2·c_p)
            'gravity'   if k ≪ k_cross (c_g ≈ 1/2·c_p)
            'mixed'     if within one decade of k_cross
        """
        if self._k_cross == float("inf"):
            return "gravity"
        ratio = k / self._k_cross
        if ratio > 10.0:
            return "capillary"
        if ratio < 0.1:
            return "gravity"
        return "mixed"

    def group_to_phase_ratio(self, k: float) -> float:
        """Returns c_g(k) / c_p(k) — the dispersive group/phase velocity ratio.

        Asymptotes:
            Capillary: → 3/2
            Gravity:   → 1/2
        """
        cp = self.phase_velocity(k)
        if cp < 1e-15:
            return 0.0
        return self.group_velocity(k) / cp

    # ------------------------------------------------------------------
    # Dispersive spatial envelope Φ_disp(r, t)
    # ------------------------------------------------------------------

    def dispersive_envelope(
        self,
        r: float,
        t: float,
        n_modes: int = 64,
        k_min: float = 0.1,
        k_max: float = 100.0,
    ) -> float:
        """Dispersive spatial envelope Φ_disp(r, t) = ∫S(k)·cos(kr - ω(k)t)dk.

        Discretised Riemann sum approximation over [k_min, k_max].
        Used to modulate dispersive Hodge-Clifford attention energies:
            E_ij(t) = (1/16)(⟨Q·K̃⟩₀ - λ‖Q∧K‖²_Cl) · Φ_disp(‖xᵢ-xⱼ‖, t)

        Parameters
        ----------
        r : float
            Spatial separation ‖xᵢ - xⱼ‖ (Å or m).
        t : float
            Elapsed time (s or dimensionless lattice time).
        n_modes : int
            Number of quadrature modes.
        k_min, k_max : float
            Wavenumber integration bounds.

        Returns
        -------
        float
            Φ_disp(r, t) ∈ ℝ  (dimensionless envelope).
        """
        if t < 0.0:
            return 0.0
        dk = (k_max - k_min) / n_modes
        total = 0.0
        # Gaussian spectral weight S(k) = exp(-k²/k_max²) peaks at low k
        for i in range(n_modes):
            k = k_min + (i + 0.5) * dk
            S_k = math.exp(-(k / k_max) ** 2)
            phase = k * r - self.omega(k) * t
            total += S_k * math.cos(phase) * dk
        return total

    def __repr__(self) -> str:
        return (
            f"CapillaryGravityDispersion("
            f"g={self.g}, γ={self.gamma}, ρ={self.rho}, h={self.h}, "
            f"k_cross={self._k_cross:.4f})"
        )
