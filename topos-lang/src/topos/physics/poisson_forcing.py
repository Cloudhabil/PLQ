"""Stochastic Poisson Point-Process Forcing Field for the ToposLang Wave Sheaf.

Stochasmos Poissonialis (Στοχασμὸς Πουασόνιος):
    Models thermal solvent collisions, raindrop impacts, ion hops, and
    microenvironmental perturbations as a 3D Spatial-Temporal Poisson Point
    Process P(λ) driving the Hodge-Dirac wave operator over the DACC manifold.

    F_P(x, t) = Σᵢ ξᵢ · δ(x - xᵢ) · δ(t - tᵢ),   (xᵢ, tᵢ) ∈ P(λ)

Campbell's Theorem Variance (closed-form integral):
    Var(η(x,t)) = λ(μ²_A + σ²_A) · ∫₀ᵗ∫_Ω G²(‖x-x'‖, t-t') d²x' dt'

    Closed-form result after Hankel kernel integration:
        Var(η) = λ(μ²_A + σ²_A)/(4πc²) · [t·ln(ct/(2δ)) - t]

    Entropic desolvation penalty:
        TΔS_solv ∝ Var(η_water)

Pure Python — zero external dependencies (math, random modules only).
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Impact event dataclass
# ---------------------------------------------------------------------------

@dataclass
class ImpactEvent:
    """A single Poisson impulse impact at position (x, y) and time t.

    Attributes
    ----------
    x, y : float
        2D spatial position of the impact (Å or m).
    t : float
        Arrival time (s or lattice time units).
    amplitude : float
        Random amplitude ξᵢ drawn from N(mu_A, sigma_A).
    """
    x: float
    y: float
    t: float
    amplitude: float


# ---------------------------------------------------------------------------
# Poisson Impulse Field
# ---------------------------------------------------------------------------

class PoissonImpulseField:
    """Stochastic Poisson Point-Process forcing field F_P(x,t).

    Generates spatial-temporal raindrop / thermal-collision impact sequences
    following a homogeneous Poisson process with rate λ (impacts·m⁻²·s⁻¹).

    The field drives the Hodge-Dirac wave operator:
        ∂²A/∂t² + γ_damp·∂A/∂t + c²·L_k·A = F_P(x,t)

    Parameters
    ----------
    lambda_rate : float
        Poisson arrival rate λ (impacts per unit area per unit time).
    mu_amplitude : float
        Mean impact amplitude μ_A.
    sigma_amplitude : float
        Std-dev of impact amplitude σ_A.
    seed : int, optional
        Random seed for reproducible sampling.
    """

    def __init__(
        self,
        lambda_rate: float = 10.0,
        mu_amplitude: float = 1.0,
        sigma_amplitude: float = 0.2,
        seed: Optional[int] = None,
    ) -> None:
        if lambda_rate <= 0.0:
            raise ValueError("Poisson rate lambda_rate must be strictly positive.")
        self.lambda_rate = lambda_rate
        self.mu_amplitude = mu_amplitude
        self.sigma_amplitude = sigma_amplitude
        self._rng = random.Random(seed)

    def _gauss_amplitude(self) -> float:
        """Draws amplitude ξ ~ N(μ_A, σ_A) using Box-Muller transform."""
        # Python's random.gauss is pure-Python based on Box-Muller
        return self._rng.gauss(self.mu_amplitude, self.sigma_amplitude)

    def sample_impacts(
        self,
        area: float,
        dt: float,
        x_range: Tuple[float, float] = (0.0, 1.0),
        y_range: Tuple[float, float] = (0.0, 1.0),
        t_offset: float = 0.0,
    ) -> List[ImpactEvent]:
        """Sample a Poisson batch of impact events over area × time-interval.

        Number of impacts N ~ Poisson(λ · area · dt):
            P(N=k) = (λ·|A|·Δt)^k · exp(-λ·|A|·Δt) / k!

        Positions (xᵢ, yᵢ) are uniform over [x_range] × [y_range].
        Times tᵢ are uniform over [t_offset, t_offset + dt].

        Parameters
        ----------
        area : float
            Total area |A| over which impacts are sampled (m² or Å²).
        dt : float
            Time interval Δt (s or lattice time).
        x_range, y_range : tuple
            Spatial bounds for impact locations.
        t_offset : float
            Starting time of the interval.

        Returns
        -------
        List[ImpactEvent]
            Sampled impact events in this interval.
        """
        mean_count = self.lambda_rate * area * dt
        # Draw N ~ Poisson(mean_count) via Knuth's algorithm (pure Python)
        n_impacts = self._knuth_poisson(mean_count)
        events: List[ImpactEvent] = []
        for _ in range(n_impacts):
            x = self._rng.uniform(x_range[0], x_range[1])
            y = self._rng.uniform(y_range[0], y_range[1])
            t = self._rng.uniform(t_offset, t_offset + dt)
            amp = self._gauss_amplitude()
            events.append(ImpactEvent(x=x, y=y, t=t, amplitude=amp))
        return events

    def _knuth_poisson(self, lam: float) -> int:
        """Knuth's algorithm for exact Poisson random variate generation.

        Works by simulating the inter-arrival times of exponential waiting
        events: draw U ~ Uniform(0,1) until product < exp(-λ).
        """
        if lam <= 0.0:
            return 0
        # For large λ, split to avoid underflow
        if lam > 500.0:
            # Normal approximation: N ≈ round(N(λ, √λ))
            n = int(self._rng.gauss(lam, math.sqrt(lam)))
            return max(0, n)
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= self._rng.random()
        return k - 1

    def evaluate_field_at(
        self,
        x: float,
        y: float,
        t: float,
        events: List[ImpactEvent],
        c: float = 1.5,
        shell_thickness: float = 0.05,
    ) -> float:
        """Evaluate superposed wave field η(x,y,t) from a list of impact events.

        Each event contributes when the causal wavefront passes (x, y):
            η ≈ Aᵢ · cos(2π·dr/shell_thickness)   if |r - c(t - tᵢ)| < shell_thickness
            η = 0                                    otherwise

        Parameters
        ----------
        x, y : float
            Query point coordinates.
        t : float
            Query time.
        events : List[ImpactEvent]
            Active impact events.
        c : float
            Wave speed.
        shell_thickness : float
            Wavefront shell half-width (for smooth cosine interpolation).

        Returns
        -------
        float
            Superposed surface displacement η(x, y, t).
        """
        eta = 0.0
        for ev in events:
            dt_ev = t - ev.t
            if dt_ev <= 0.0:
                continue
            r = math.sqrt((x - ev.x) ** 2 + (y - ev.y) ** 2)
            r_front = c * dt_ev
            dr = abs(r - r_front)
            if dr < shell_thickness and r_front > 1e-9:
                # Amplitude decays as 1/√r (radial flux conservation)
                amp_decay = ev.amplitude / math.sqrt(r_front)
                eta += amp_decay * math.cos(math.pi * dr / shell_thickness)
        return eta

    def __repr__(self) -> str:
        return (
            f"PoissonImpulseField(λ={self.lambda_rate}, "
            f"μ_A={self.mu_amplitude}, σ_A={self.sigma_amplitude})"
        )


# ---------------------------------------------------------------------------
# Campbell's Theorem Variance (closed-form)
# ---------------------------------------------------------------------------

class CampbellTheoremVariance:
    """Closed-form Campbell's Theorem variance for Poisson-driven wave fields.

    Campbell's Theorem states that for a Poisson-driven integral field:
        η(x,t) = Σᵢ Aᵢ · G(‖x - xᵢ‖, t - tᵢ)

    The variance is:
        Var(η) = λ · E[A²] · ∫₀ᵗ ∫_Ω G²(‖x-x'‖, t-t') d²x' dt'

    Substituting the 2D retarded Green's kernel G(r,τ) = H(cτ-r)/(2πc√(c²τ²-r²))
    and evaluating the double integral analytically (Hankel transform identity):

        ∫_Ω G²(r,τ) d²x' = (1/(4πc²)) · ln(c²τ²/δ(2cτ-δ))
                          ≈ (1/(4πc²)) · ln(cτ/(2δ))   for cτ ≫ δ

    Integrating over time [0, t]:
        Var(η) = λ(μ²_A + σ²_A)/(4πc²) · [t·ln(ct/(2δ)) - t]

    This converts empirical solvation heuristics into a closed-form thermodynamic
    variance equation (see Proof 4 / Theorem 4.3 in Topological Mechanics paper).

    Parameters
    ----------
    lambda_rate : float
        Poisson collision rate λ.
    mu_amplitude : float
        Mean amplitude μ_A.
    sigma_amplitude : float
        Std-dev of amplitude σ_A.
    c : float
        Wave propagation speed (m/s or consistent units).
    cutoff_delta : float
        Microscopic core cutoff δ > 0 (Å or m) — prevents log divergence at r=0.
    """

    def __init__(
        self,
        lambda_rate: float = 10.0,
        mu_amplitude: float = 1.0,
        sigma_amplitude: float = 0.2,
        c: float = 1750.0,
        cutoff_delta: float = 0.01,
    ) -> None:
        if cutoff_delta <= 0.0:
            raise ValueError("cutoff_delta must be > 0 to regularise log singularity.")
        if c <= 0.0:
            raise ValueError("Wave speed c must be strictly positive.")
        self.lambda_rate = lambda_rate
        self.mu_amplitude = mu_amplitude
        self.sigma_amplitude = sigma_amplitude
        self.c = c
        self.delta = cutoff_delta

    def variance_at(self, t: float) -> float:
        """Closed-form spatial variance Var(η(x,t)) at time t.

        Var(η) = λ(μ²_A + σ²_A)/(4πc²) · [t·ln(ct/(2δ)) - t]

        Parameters
        ----------
        t : float
            Elapsed time.

        Returns
        -------
        float
            Var(η(x,t)) ≥ 0.
        """
        if t <= 0.0:
            return 0.0
        ct = self.c * t
        argument = ct / (2.0 * self.delta)
        if argument <= 1.0:
            # Wavefront hasn't propagated beyond cutoff yet
            return 0.0
        E_A2 = self.mu_amplitude ** 2 + self.sigma_amplitude ** 2
        prefactor = self.lambda_rate * E_A2 / (4.0 * math.pi * self.c ** 2)
        log_term = math.log(argument)
        return prefactor * t * (log_term - 1.0)

    def variance_at_interface(
        self,
        r_interface: List[float],
        t: float,
    ) -> List[float]:
        """Point-wise variance at a list of interface distances.

        Since the Green's function is translation-invariant and the closed-form
        integrates over all space, the spatial variance is homogeneous (independent
        of query position for a spatially uniform Poisson process).  This method
        returns the same scalar Var(η(t)) for each interface point but weighted
        by proximity as (1/r²)·Var(η):

            Var_local(r, t) ≈ Var(η(t)) / max(r², δ²)

        Parameters
        ----------
        r_interface : List[float]
            Interface distances (Å or m).
        t : float
            Elapsed time.

        Returns
        -------
        List[float]
            Local variance estimate at each interface distance.
        """
        base_var = self.variance_at(t)
        result: List[float] = []
        for r in r_interface:
            denom = max(r ** 2, self.delta ** 2)
            result.append(base_var / denom)
        return result

    def __repr__(self) -> str:
        return (
            f"CampbellTheoremVariance("
            f"λ={self.lambda_rate}, c={self.c}, δ={self.delta})"
        )
