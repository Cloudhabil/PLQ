"""PocketWaveField — Green's Function Wave Propagator for PLQ Binding Pockets.

Applies the retarded causal Green's function kernel G(r, Δt) to a molecular
binding pocket represented as a list of residue contact distances, computing:

    1. Dynamic wave-field η(r, t) at each residue position.
    2. Ligand-induced solvent shockwave propagation (k_on kinetics analogue).
    3. Hydration-shell desolvation waves (k_off kinetics analogue).

Pneuma Fluens (Πνεῦμα Ῥέον) over the pocket manifold:
    η_pocket(rᵢ, t) = Σⱼ Aⱼ·G(‖rᵢ-rⱼ‖, t-tⱼ)

Protoporeia Causal:
    G(r, Δt) = H(c·Δt - r) / (2π·c·√(c²Δt² - r²))

This module operates entirely in PLQ's biological domain (Å, ps, kcal/mol)
and bridges to the ToposLang wave sheaf via shared physical constants.

Pure Python — zero external dependencies.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Physical config (inline, mirrors topos.physics.wave_sheaf profiles)
# ---------------------------------------------------------------------------

# Speed of acoustic wave propagation in an aqueous protein environment (Å/ps)
# 1750 m/s = 1750 Å/ps  (1 m/s = 1 Å/ps in consistent Å-ps units)
PROTEIN_C_WAVE: float = 1750.0
SOLID_STATE_C_WAVE: float = 5100.0
MACRO_FLUID_C_WAVE: float = 1.5


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PocketWaveResult:
    """Results from PocketWaveField.propagate().

    Attributes
    ----------
    eta_field : List[float]
        Wave-field displacement η(rᵢ, t) at each residue contact site.
    active_wavefront_radius : float
        Current causal wavefront radius c·t (Å).
    residues_reached : int
        Number of residue sites within the causal wavefront.
    desolvation_wave_amplitude : float
        Peak |η| amplitude (proxy for hydration-shell displacement intensity).
    kon_proxy : float
        Ligand approach wave energy: Σ max(η_i, 0) (positive phase → k_on).
    koff_proxy : float
        Dissociation wave energy: Σ max(-η_i, 0) (negative phase → k_off).
    simulation_time : float
        Elapsed simulation time t (ps).
    """
    eta_field: List[float]
    active_wavefront_radius: float
    residues_reached: int
    desolvation_wave_amplitude: float
    kon_proxy: float
    koff_proxy: float
    simulation_time: float


# ---------------------------------------------------------------------------
# Impact source dataclass
# ---------------------------------------------------------------------------

@dataclass
class WaveImpact:
    """A ligand-approach or solvent-collision impulse event.

    Attributes
    ----------
    r_source : float
        Radial distance of the impact from the pocket centre (Å).
    t_impact : float
        Time of impact (ps).
    amplitude : float
        Wave amplitude (positive = approach, negative = departure).
    """
    r_source: float
    t_impact: float
    amplitude: float = 1.0


# ---------------------------------------------------------------------------
# PocketWaveField
# ---------------------------------------------------------------------------

class PocketWaveField:
    """Green's function wave propagator for PLQ binding pocket kinetics.

    Models the pocket as a 1D radial manifold with residue contact sites at
    distances r₁, r₂, … from the pocket centre (Å).

    Ligand approach and solvent-collision events are injected as WaveImpact
    sources. The retarded Green's kernel propagates the resulting displacement
    field η(rᵢ, t) causally outward from each source.

    From this field we extract:
        - k_on proxy: net inward wave energy (ligand binding front)
        - k_off proxy: net outward wave energy (desolvation / escape front)
        - Desolvation amplitude: peak hydration-shell displacement

    Parameters
    ----------
    residue_distances : List[float]
        Radial contact distances of pocket residues from pocket centre (Å).
    c_wave : float
        Acoustic wave speed (Å/ps). Default: PROTEIN_C_WAVE = 1750 Å/ps.
    shell_thickness : float
        Wavefront shell half-width for smooth cosine interpolation (Å).
    epsilon : float
        Green's kernel discriminant regularisation ε.
    """

    def __init__(
        self,
        residue_distances: List[float],
        c_wave: float = PROTEIN_C_WAVE,
        shell_thickness: float = 0.5,
        epsilon: float = 1e-9,
    ) -> None:
        if not residue_distances:
            raise ValueError("PocketWaveField requires at least one residue distance.")
        if c_wave <= 0.0:
            raise ValueError("Wave speed c must be positive.")
        self.residues = list(residue_distances)
        self.c = c_wave
        self.shell_thickness = shell_thickness
        self.epsilon = epsilon
        self._impacts: List[WaveImpact] = []
        self._time: float = 0.0

    # ------------------------------------------------------------------
    # Impact injection
    # ------------------------------------------------------------------

    def inject_ligand_approach(
        self,
        r_source: float,
        t_impact: float,
        amplitude: float = 1.0,
    ) -> None:
        """Register a ligand-approach impulse (positive wave, inward).

        Parameters
        ----------
        r_source : float
            Distance from pocket centre at which ligand impulse originates (Å).
        t_impact : float
            Arrival time of the impulse (ps).
        amplitude : float
            Wave amplitude. Positive → inward binding wave.
        """
        self._impacts.append(WaveImpact(r_source, t_impact, amplitude))

    def inject_solvent_collision(
        self,
        r_source: float,
        t_impact: float,
        amplitude: float = 0.3,
    ) -> None:
        """Register a thermal solvent-collision impulse (Poisson P(λ) event).

        Parameters
        ----------
        r_source : float
            Distance of the colliding water molecule from pocket centre (Å).
        t_impact : float
            Collision time (ps).
        amplitude : float
            Displacement amplitude (smaller than ligand, typically 0.1–0.5).
        """
        self._impacts.append(WaveImpact(r_source, t_impact, amplitude))

    def inject_dissociation_wave(
        self,
        r_source: float = 0.0,
        t_impact: float = 0.0,
        amplitude: float = -1.0,
    ) -> None:
        """Register a dissociation / k_off impulse (negative wave, outward).

        A negative amplitude represents the departure of a bound ligand,
        generating an outward water re-entry wave (desolvation reversal).
        """
        self._impacts.append(WaveImpact(r_source, t_impact, amplitude))

    # ------------------------------------------------------------------
    # Green's kernel
    # ------------------------------------------------------------------

    def _greens_kernel(self, r: float, dt: float) -> float:
        """Retarded 2D Green's function G(r, Δt).

        G(r, Δt) = H(c·Δt - r) / (2π·c·√(c²Δt² - r²))
        """
        if dt <= 0.0:
            return 0.0
        c_dt = self.c * dt
        if c_dt <= r:
            return 0.0  # Outside causal cone
        disc = c_dt ** 2 - r ** 2
        if disc < self.epsilon:
            return 0.0
        return 1.0 / (2.0 * math.pi * self.c * math.sqrt(disc))

    def _wavefront_eta(
        self, r_query: float, impact: WaveImpact, t: float
    ) -> float:
        """Evaluate wave field contribution of one impact at (r_query, t).

        Combines the wavefront shell cosine interpolation with the retarded 2D
        Green's function wake G(|r_query - r_source|, Δt) inside the causal cone.
        """
        dt = t - impact.t_impact
        if dt <= 0.0:
            return 0.0
        dist = abs(r_query - impact.r_source)
        r_front = self.c * dt
        if r_front < 1e-9:
            return 0.0
        dr = abs(dist - r_front)
        shell_val = 0.0
        if dr < self.shell_thickness:
            # Amplitude decays as 1/√(r_front) — radial energy flux conservation
            amp_decay = impact.amplitude / math.sqrt(r_front)
            shell_val = amp_decay * math.cos(0.5 * math.pi * dr / self.shell_thickness)
        wake_val = impact.amplitude * self._greens_kernel(dist, dt)
        return shell_val + wake_val

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def propagate(self, t: float) -> PocketWaveResult:
        """Evaluate the wave field at time t across all residue sites.

        Superposes all registered WaveImpact sources via the causal
        Green's function kernel.

        Parameters
        ----------
        t : float
            Simulation time (ps).

        Returns
        -------
        PocketWaveResult
            Full pocket wave-field telemetry at time t.
        """
        self._time = t
        eta_field: List[float] = []

        for r_res in self.residues:
            eta_r = sum(
                self._wavefront_eta(r_res, imp, t)
                for imp in self._impacts
            )
            eta_field.append(round(eta_r, 8))

        # Kinetics proxies
        kon_proxy = sum(e for e in eta_field if e > 0.0)
        koff_proxy = sum(-e for e in eta_field if e < 0.0)
        peak_amp = max((abs(e) for e in eta_field), default=0.0)

        # Causal wavefront radius at time t (from earliest impact)
        earliest_dt = max(
            (t - imp.t_impact for imp in self._impacts if t > imp.t_impact),
            default=0.0,
        )
        wavefront_r = self.c * earliest_dt

        residues_reached = sum(
            1 for r in self.residues if r <= wavefront_r
        )

        return PocketWaveResult(
            eta_field=eta_field,
            active_wavefront_radius=round(wavefront_r, 4),
            residues_reached=residues_reached,
            desolvation_wave_amplitude=round(peak_amp, 8),
            kon_proxy=round(kon_proxy, 8),
            koff_proxy=round(koff_proxy, 8),
            simulation_time=round(t, 6),
        )

    def run_time_series(
        self,
        t_start: float,
        t_end: float,
        n_steps: int,
    ) -> List[PocketWaveResult]:
        """Run a time series of pocket wave propagations.

        Parameters
        ----------
        t_start, t_end : float
            Time range (ps).
        n_steps : int
            Number of evenly-spaced time points.

        Returns
        -------
        List[PocketWaveResult]
            One result per time step.
        """
        if n_steps < 1:
            return []
        dt_series = (t_end - t_start) / n_steps
        results = []
        for i in range(n_steps + 1):
            t_i = t_start + i * dt_series
            results.append(self.propagate(t_i))
        return results

    def __repr__(self) -> str:
        return (
            f"PocketWaveField("
            f"n_residues={len(self.residues)}, "
            f"c={self.c} Å/ps, "
            f"n_impacts={len(self._impacts)})"
        )
