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


# ==============================================================================
# Anisotropia Dielectrica Sheaf — Dynamic Wave-Enhanced Dielectric Continuum
# (Upgrade alongside existing CliffordSheaf — backward compatible)
# ==============================================================================

class AnisotropiaDielectricaSheaf:
    """Sigmoidal Dielectric Continuum Sheaf with Logarithmic Elasticity & Wave Entropy.

    Anisotropia Dielectrica (Sheaf Dielectrica):
        Replaces static spatial dielectric heuristics with:
        1. Sigmoidal permittivity ε(r) modelling the hydration-shell transition.
        2. Logarithmic dielectric elasticity η_ε(r) = d ln ε / d ln r — the
           outward thermodynamic pressure opposing uncompensated desolvation.
        3. Dynamic solvation entropy TΔS_solv derived from Campbell's Theorem
           variance of the Poisson-driven water-collision field.

    Section 4.2 of the Topological Mechanics paper (Eq. 85–89):
        ε(r) = ε_int + (ε_ext - ε_int) / (1 + k·exp(-λ·r))
        η_ε(r) = (r/ε(r)) · dε/dr  =  d ln ε(r) / d ln r

    Designed to be used alongside (not replacing) the existing CliffordSheaf
    class. Backward-compatible addition — no existing code is changed.

    Parameters
    ----------
    eps_int : float
        Interior (protein core) dielectric constant ε_int (≈ 4).
    eps_ext : float
        Exterior (bulk solvent) dielectric constant ε_ext (≈ 80).
    k_sigmoid : float
        Sigmoid steepness parameter k (controls transition sharpness).
    lam_sigmoid : float
        Sigmoid length-scale parameter λ (Å⁻¹).
    kappa_solv : float
        Entropic scaling factor κ_solv for TΔS_solv (kcal/mol / variance unit).
    ke_coulomb : float
        Coulomb constant in molecular units (eV·Å/e²) — default 14.40.
    """

    def __init__(
        self,
        eps_int: float = 4.0,
        eps_ext: float = 80.0,
        k_sigmoid: float = 18.0,
        lam_sigmoid: float = 0.4,
        kappa_solv: float = 0.42,
        ke_coulomb: float = 14.40,
    ) -> None:
        import math as _math
        self._math = _math
        if eps_int <= 0.0 or eps_ext <= eps_int:
            raise ValueError("Require 0 < ε_int < ε_ext.")
        self.eps_int = eps_int
        self.eps_ext = eps_ext
        self.k_sigmoid = k_sigmoid
        self.lam = lam_sigmoid
        self.kappa_solv = kappa_solv
        self.ke = ke_coulomb

    # ------------------------------------------------------------------
    # Core dielectric functions (Eq. 85–89)
    # ------------------------------------------------------------------

    def permittivity(self, r: float) -> float:
        """Sigmoidal dielectric permittivity ε(r).

        ε(r) = ε_int + (ε_ext - ε_int) / (1 + k·exp(-λ·r))

        Parameters
        ----------
        r : float
            Distance from protein core surface (Å).
        """
        if r < 0.0:
            r = 0.0
        denom = 1.0 + self.k_sigmoid * self._math.exp(-self.lam * r)
        return self.eps_int + (self.eps_ext - self.eps_int) / denom

    def dielectric_elasticity(self, r: float) -> float:
        """Logarithmic dielectric elasticity η_ε(r) = d ln ε / d ln r.

        η_ε(r) = (r / ε(r)) · dε/dr

        dε/dr = λ·k·(ε_ext - ε_int)·exp(-λ·r) / (1 + k·exp(-λ·r))²

        A positive η_ε indicates increasing dielectric with distance (favourable
        solvation); the force correction factor (1 + η_ε) amplifies Coulomb
        attraction at short range and screens it at long range.
        """
        if r <= 0.0:
            return 0.0
        eps_r = self.permittivity(r)
        exp_term = self._math.exp(-self.lam * r)
        denom_sq = (1.0 + self.k_sigmoid * exp_term) ** 2
        deps_dr = (self.lam * self.k_sigmoid * (self.eps_ext - self.eps_int)
                   * exp_term / denom_sq)
        return (r / eps_r) * deps_dr

    def solvated_electrostatic_force(
        self, q1: float, q2: float, r: float
    ) -> float:
        """Solvated Coulomb force with dielectric elasticity correction (Eq. 87).

        F_elec(r) = [q₁·q₂ / (4π·ε₀·r²·ε(r))] · [1 + η_ε(r)]

        In molecular units (ke = 14.40 eV·Å·e⁻²):
            F = ke·q₁·q₂ / (r²·ε(r)) · (1 + η_ε(r))

        Positive = repulsive; Negative = attractive.
        """
        import math as _m
        if r < 1e-9:
            return 0.0
        eps_r = self.permittivity(r)
        eta_eps = self.dielectric_elasticity(r)
        f_screened = (self.ke * q1 * q2) / (r ** 2 * eps_r)
        return f_screened * (1.0 + eta_eps)

    # ------------------------------------------------------------------
    # Dynamic Solvation Entropy (Campbell's Theorem)
    # ------------------------------------------------------------------

    def compute_dynamic_solvation_entropy(
        self,
        r_interface: list,
        poisson_rate: float = 10.0,
        t: float = 1.0,
        c: float = 1750.0,
        cutoff_delta: float = 0.01,
        mu_amplitude: float = 1.0,
        sigma_amplitude: float = 0.2,
    ) -> float:
        """Dynamic desolvation entropy TΔS_solv using Campbell's Theorem.

        TΔS_solv = κ_solv · Σ_r Var(η(r,t)) · [1 + η_ε(r)]

        where:
            Var(η(r,t)) ≈ λ(μ²_A+σ²_A)/(4πc²) · [t·ln(ct/2δ)-t] / r²

        This quantifies the entropy penalty incurred when a ligand displaces
        the fluctuating hydration network around the active-site interface.

        Parameters
        ----------
        r_interface : list[float]
            Interface distances for each residue contact (Å).
        poisson_rate : float
            Thermal water-collision Poisson rate λ.
        t : float
            Binding encounter time (ps or lattice time).
        c : float
            Acoustic wave speed in aqueous protein (Å/ps or m/s consistent units).
        cutoff_delta : float
            Microscopic core cutoff δ (Å).
        mu_amplitude, sigma_amplitude : float
            Amplitude distribution parameters for Aᵢ ~ N(μ_A, σ_A).

        Returns
        -------
        float
            TΔS_solv (kcal/mol, approximated units).
        """
        import math as _m
        if t <= 0.0 or not r_interface:
            return 0.0
        c_mol = c * 0.01 if c >= 100.0 else c
        ct = c_mol * t
        E_A2 = mu_amplitude ** 2 + sigma_amplitude ** 2
        log_arg = ct / (2.0 * cutoff_delta)
        if log_arg <= 1.0:
            return 0.0
        # Base variance from Campbell integral (homogeneous Poisson, full space)
        base_var_prefactor = (poisson_rate * E_A2
                              / (4.0 * _m.pi * c_mol ** 2))
        base_var = 100.0 * base_var_prefactor * t * (_m.log(log_arg) - 1.0)

        total_entropy = 0.0
        for r in r_interface:
            if r <= 0.0:
                r = cutoff_delta
            # Local variance ∝ 1/r² (radial Green's function decay)
            var_local = base_var / max(r ** 2, cutoff_delta ** 2)
            eta_eps = self.dielectric_elasticity(r)
            total_entropy += var_local * (1.0 + eta_eps)

        return round(self.kappa_solv * total_entropy, 6)

    def __repr__(self) -> str:
        return (
            f"AnisotropiaDielectricaSheaf("
            f"ε_int={self.eps_int}, ε_ext={self.eps_ext}, "
            f"k={self.k_sigmoid}, λ={self.lam})"
        )
