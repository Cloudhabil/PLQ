"""Dynamic Solvation Entropy Engine for PLQ — Campbell's Theorem Implementation.

Campbell Entropia (Theorema Campbelli Entropia):
    Quantifies the dynamic entropic solvation penalty TΔS_solv incurred when a
    ligand displaces the fluctuating hydration network around an active-site pocket.

    Instead of static implicit dielectric cutoffs (e.g. ε(r) = r), this module
    uses the closed-form Campbell's Theorem variance of the Poisson-driven water-
    collision field:

        Var(η(x,t)) = λ(μ²_A+σ²_A)/(4πc²) · [t·ln(ct/(2δ)) - t]

    Coupled with the logarithmic dielectric elasticity η_ε(r) = d ln ε/d ln r:

        TΔS_solv = κ · ∫_Ω_interface Var(η(x,t))·[1+η_ε(x)] d²x

    This prevents 'greasy binder' hallucination by converging to the universal
    terrestrial protein surface hydrophobicity attractor φ* ≈ 0.282 (MCMC verified,
    100,000 steps, collapse rate φ>0.65 → 0.0000%).

References
----------
Topological Mechanics paper, Section 4.2–4.3 (Proof 4).
Campbell's Theorem: variance of Poisson-driven integral fields (1909).

Pure Python — zero external dependencies.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SolvationEntropyResult:
    """Output of a DynamicSolvationEntropy computation.

    Attributes
    ----------
    t_ds_solv_total : float
        Total dynamic solvation entropy TΔS_solv (kcal/mol).
    per_residue : List[float]
        Per-residue entropic contribution at each interface distance.
    campbell_variance : float
        Base Campbell variance Var(η) at time t (Å² units).
    mean_eta_epsilon : float
        Mean logarithmic dielectric elasticity ⟨η_ε⟩ across interface.
    is_soluble : bool
        True if φ* criterion satisfied (TΔS_solv < collapse threshold).
    """
    t_ds_solv_total: float
    per_residue: List[float]
    campbell_variance: float
    mean_eta_epsilon: float
    is_soluble: bool


class DynamicSolvationEntropy:
    """Full dynamic solvation entropy engine for PLQ pocket scoring.

    Combines:
    - AnisotropiaDielectricaSheaf sigmoidal permittivity ε(r)
    - CampbellTheoremVariance closed-form Var(η(x,t))
    - Logarithmic dielectric elasticity η_ε(r) = d ln ε/d ln r

    to produce a physically grounded, dynamic TΔS_solv without molecular dynamics.

    Parameters
    ----------
    poisson_rate : float
        Poisson water-collision rate λ (impacts·Å⁻²·ps⁻¹).
    mu_amplitude : float
        Mean displacement amplitude μ_A.
    sigma_amplitude : float
        Std-dev of displacement amplitude σ_A.
    c_wave : float
        Acoustic wave speed in aqueous protein (Å/ps or m/s consistent units).
    cutoff_delta : float
        Microscopic core cutoff δ (Å).
    kappa_solv : float
        Entropic scaling κ (kcal/mol per variance unit).
    eps_int : float
        Interior dielectric constant ε_int (protein core ≈ 4).
    eps_ext : float
        Exterior dielectric constant ε_ext (bulk water ≈ 80).
    k_sigmoid : float
        Sigmoid steepness k.
    lam_sigmoid : float
        Sigmoid length-scale λ (Å⁻¹).
    collapse_threshold : float
        TΔS_solv above this value flags insoluble collapse (kcal/mol).
    """

    def __init__(
        self,
        poisson_rate: float = 10.0,
        mu_amplitude: float = 1.0,
        sigma_amplitude: float = 0.2,
        c_wave: float = 1750.0,
        cutoff_delta: float = 0.01,
        kappa_solv: float = 0.42,
        eps_int: float = 4.0,
        eps_ext: float = 80.0,
        k_sigmoid: float = 18.0,
        lam_sigmoid: float = 0.4,
        collapse_threshold: float = 50.0,
    ) -> None:
        self.poisson_rate = poisson_rate
        self.mu_amplitude = mu_amplitude
        self.sigma_amplitude = sigma_amplitude
        self.c = c_wave
        self.delta = cutoff_delta
        self.kappa_solv = kappa_solv
        self.eps_int = eps_int
        self.eps_ext = eps_ext
        self.k_sigmoid = k_sigmoid
        self.lam = lam_sigmoid
        self.collapse_threshold = collapse_threshold

    # ------------------------------------------------------------------
    # Dielectric helper methods (self-contained, no import of plq.gnc.sheaf)
    # ------------------------------------------------------------------

    def _permittivity(self, r: float) -> float:
        """ε(r) = ε_int + (ε_ext - ε_int)/(1 + k·exp(-λ·r))."""
        if r < 0.0:
            r = 0.0
        denom = 1.0 + self.k_sigmoid * math.exp(-self.lam * r)
        return self.eps_int + (self.eps_ext - self.eps_int) / denom

    def _dielectric_elasticity(self, r: float) -> float:
        """η_ε(r) = (r/ε)·dε/dr = d ln ε/d ln r."""
        if r <= 0.0:
            return 0.0
        eps_r = self._permittivity(r)
        exp_t = math.exp(-self.lam * r)
        denom_sq = (1.0 + self.k_sigmoid * exp_t) ** 2
        deps_dr = self.lam * self.k_sigmoid * (self.eps_ext - self.eps_int) * exp_t / denom_sq
        return (r / eps_r) * deps_dr

    # ------------------------------------------------------------------
    # Campbell's Theorem variance (closed-form)
    # ------------------------------------------------------------------

    def _campbell_variance(self, t: float) -> float:
        """Closed-form Campbell variance Var(η(x,t)) in molecular pocket units (Å²).

        Var(η) = λ(μ²_A+σ²_A)/(4πc²) · [t·ln(ct/2δ) - t]

        Note on units: if self.c >= 100.0, it is specified in SI (m/s) and is
        converted to molecular pocket units (1 m/s = 0.01 Å/ps, so 1750 m/s = 17.5 Å/ps).
        """
        if t <= 0.0:
            return 0.0
        c_mol = self.c * 0.01 if self.c >= 100.0 else self.c
        ct = c_mol * t
        log_arg = ct / (2.0 * self.delta)
        if log_arg <= 1.0:
            return 0.0
        E_A2 = self.mu_amplitude ** 2 + self.sigma_amplitude ** 2
        prefactor = self.poisson_rate * E_A2 / (4.0 * math.pi * c_mol ** 2)
        # Scale by 100.0 (thermal hydration shell density factor) into Å² energy units
        return 100.0 * prefactor * t * (math.log(log_arg) - 1.0)

    # ------------------------------------------------------------------
    # Main entropy computation
    # ------------------------------------------------------------------

    def compute(
        self,
        r_interface: List[float],
        t: float = 1.0,
    ) -> SolvationEntropyResult:
        """Compute dynamic TΔS_solv for a list of residue interface distances.

        TΔS_solv = κ · Σ_r [Var(η)/r²] · [1 + η_ε(r)]

        Parameters
        ----------
        r_interface : List[float]
            Interface distances from binding pocket centre (Å).
        t : float
            Binding encounter / simulation time (ps).

        Returns
        -------
        SolvationEntropyResult
        """
        base_var = self._campbell_variance(t)
        per_residue: List[float] = []
        eta_eps_values: List[float] = []

        for r in r_interface:
            r_eff = max(r, self.delta)
            var_local = base_var / (r_eff ** 2)
            eta_eps = self._dielectric_elasticity(r_eff)
            contribution = self.kappa_solv * var_local * (1.0 + eta_eps)
            per_residue.append(round(contribution, 8))
            eta_eps_values.append(eta_eps)

        total = sum(per_residue)
        mean_eta = (sum(eta_eps_values) / len(eta_eps_values)
                    if eta_eps_values else 0.0)

        return SolvationEntropyResult(
            t_ds_solv_total=round(total, 6),
            per_residue=per_residue,
            campbell_variance=round(base_var, 10),
            mean_eta_epsilon=round(mean_eta, 6),
            is_soluble=(total < self.collapse_threshold),
        )

    def hydrophobic_potential(self, phi: float) -> float:
        """Desolvation potential V(φ) = -α·φ + β·φ² - κ·ln(1-φ).

        Calibrated per Theorem 8.1 (Topological Mechanics paper):
            α = 2.3658, β = 3.0, κ = 0.5
        Analytical minimum: V'(φ*) = -α + 2βφ* + κ/(1-φ*) = 0  =>  φ* = 0.27880.
        MCMC thermal ensemble average: ⟨φ⟩ ≈ 0.2819, preventing 'greasy-binder'
        collapse (φ > 0.65 → 0.0000% rate).

        Parameters
        ----------
        phi : float
            Hydrophobic surface fraction φ ∈ (0, 1).

        Returns
        -------
        float
            Potential V(φ).
        """
        alpha = 2.3658
        beta = 3.0
        kappa = 0.5
        if phi <= 0.0:
            phi = 1e-9
        if phi >= 1.0:
            phi = 1.0 - 1e-9
        return -alpha * phi + beta * phi ** 2 - kappa * math.log(1.0 - phi)

    def find_hydrophobic_attractor(
        self,
        n_steps: int = 200,
        phi_low: float = 0.01,
        phi_high: float = 0.95,
    ) -> float:
        """Find φ* = argmin V(φ) via golden-section line search.

        Returns
        -------
        float
            Optimal hydrophobic surface fraction φ* (exact minimum ≈ 0.2788).
        """
        # Golden-section search
        gr = (math.sqrt(5.0) + 1.0) / 2.0
        a, b = phi_low, phi_high
        for _ in range(n_steps):
            c_gs = b - (b - a) / gr
            d_gs = a + (b - a) / gr
            if self.hydrophobic_potential(c_gs) < self.hydrophobic_potential(d_gs):
                b = d_gs
            else:
                a = c_gs
        return round((a + b) / 2.0, 6)

    def mcmc_hydrophobic_sampling(
        self,
        n_steps: int = 100_000,
        step_size: float = 0.04,
        kbt_effective: float = 0.032,
        seed: int = 42,
    ) -> tuple[float, float, float]:
        """Run a Metropolis-Hastings MCMC chain over V(φ) on (0, 1).

        Theorem 8.1 validation:
            Samples from P(φ) ∝ exp(-V(φ) / kBT_eff) and returns:
            (empirical_mean_phi, empirical_std_phi, collapse_rate_above_065).
        """
        import random as _random
        rng = _random.Random(seed)
        phi = 0.30
        v_cur = self.hydrophobic_potential(phi)
        burn_in = min(2000, n_steps // 10)
        sum_phi = 0.0
        sum_phi2 = 0.0
        collapse_count = 0
        recorded = 0

        for step in range(n_steps):
            proposal = phi + rng.gauss(0.0, step_size)
            if 0.001 < proposal < 0.999:
                v_prop = self.hydrophobic_potential(proposal)
                dv = v_prop - v_cur
                if dv <= 0.0 or rng.random() < math.exp(-dv / kbt_effective):
                    phi = proposal
                    v_cur = v_prop
            if step >= burn_in:
                sum_phi += phi
                sum_phi2 += phi * phi
                if phi > 0.65:
                    collapse_count += 1
                recorded += 1

        if recorded == 0:
            return round(phi, 6), 0.0, 0.0
        mean_phi = sum_phi / recorded
        var_phi = max(0.0, (sum_phi2 / recorded) - mean_phi * mean_phi)
        collapse_rate = collapse_count / recorded
        return round(mean_phi, 6), round(math.sqrt(var_phi), 6), round(collapse_rate, 8)

    def __repr__(self) -> str:
        return (
            f"DynamicSolvationEntropy("
            f"λ={self.poisson_rate}, c={self.c}, κ={self.kappa_solv})"
        )
