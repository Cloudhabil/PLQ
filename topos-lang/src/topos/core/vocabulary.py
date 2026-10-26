"""Pontic-Greek and Latin Architectural Vocabulary Index for ToposLang & PLQ.

This module is the machine-readable canonical glossary of all formal terminology
established in the Topological Mechanics paper and the Dynamic Wave Sheaf
architecture.  It serves as:

    1. An in-code reference for developers navigating the formalism.
    2. A documentation anchor that links Greek/Latin term names to their
       exact mathematical objects, Python implementations, and PLQ roles.
    3. A searchable index for future Lean 4 formalization emissions.

Usage
-----
    from topos.core.vocabulary import VOCABULARY, lookup, print_index
    print(lookup("Protoporeia Causal"))
    print_index()
"""

from __future__ import annotations
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Vocabulary Registry
# ---------------------------------------------------------------------------

VOCABULARY: Dict[str, Dict[str, str]] = {
    # -----------------------------------------------------------------------
    # Wave Field (Continuous Domain)
    # -----------------------------------------------------------------------
    "Pneuma Fluens": {
        "greek_latin": "Πνεῦμα Ῥέον",
        "math": "η(x,t) ∈ C∞(M⁸ × ℝ⁺)",
        "role": (
            "Continuous wave field propagating across the 8D pharmacological "
            "manifold M⁸ and its spatial-temporal cell strata.  Drives the "
            "Hodge-Dirac operator ∂²A/∂t² + γ·∂A/∂t + c²·L_k·A = F_P(x,t)."
        ),
        "python": "topos.physics.wave_sheaf.DispersiveWaveSheafDACC.stalks",
        "equation": "∂²A/∂t² + γ_damp·∂A/∂t + c²·L_k·A = Σᵢ ξᵢδ(x-xᵢ)δ(t-tᵢ)",
    },

    # -----------------------------------------------------------------------
    # Clifford Multivector Stalk
    # -----------------------------------------------------------------------
    "Typos Metricus": {
        "greek_latin": "Τύπος Μεταιχμιακός",
        "math": "A_k ∈ Cl(ℝ⁸) ≅ ⨁_{g=0}^{8} ∧^g ℝ⁸",
        "role": (
            "256-blade Clifford algebra multivector stalk encoding local cell "
            "geometry, electrostatic fields, torsion angles, and solvation state.  "
            "The geometric product split uv = u·v + u∧v isolates scalar alignment "
            "(u·v, attractive) from oriented steric clash (u∧v, repulsive)."
        ),
        "python": "topos.core.clifford.Multivector / topos.physics.wave_sheaf (8D list[float])",
        "equation": "uv = ⟨uv⟩₀ + ⟨uv⟩₂ + … = u·v + u∧v",
    },

    # -----------------------------------------------------------------------
    # Capillary-Gravity Dispersion Relation
    # -----------------------------------------------------------------------
    "Symploke Dispersiva": {
        "greek_latin": "Συμπλοκὴ Διασκεδαστική",
        "math": "ω²(k) = (gk + γk³/ρ)·tanh(kh)",
        "role": (
            "Governs group velocity cg(k) = dω/dk for multi-scale local vs. global "
            "information routing in the Hodge-Clifford Spatial Transformer. "
            "Capillary regime (k≫k_cross): cg ≈ 3/2·cp — resolves side-chain rotamers fast. "
            "Gravity regime  (k≪k_cross): cg ≈ 1/2·cp — global allosteric domain shifts."
        ),
        "python": "topos.physics.dispersion.CapillaryGravityDispersion",
        "equation": "ω²(k) = (gk + γk³/ρ)·tanh(kh),  k_cross = √(ρg/γ)",
    },

    # -----------------------------------------------------------------------
    # Stochastic Poisson Forcing
    # -----------------------------------------------------------------------
    "Stochasmos Poissonialis": {
        "greek_latin": "Στοχασμὸς Πουασόνιος",
        "math": "P(λ): F_P(x,t) = Σᵢ ξᵢ·δ(x-xᵢ)·δ(t-tᵢ)",
        "role": (
            "Stochastic arrival model for thermal solvent collisions, raindrop impacts, "
            "ion hops, and microenvironmental perturbations.  Arrival count N ~ Poisson(λ·|A|·Δt). "
            "Provides the Campbell variance driving TΔS_solv computation."
        ),
        "python": "topos.physics.poisson_forcing.PoissonImpulseField",
        "equation": "P(N=k) = (λ·|A|·Δt)^k·exp(-λ·|A|·Δt)/k!",
    },

    # -----------------------------------------------------------------------
    # Retarded Green's Function
    # -----------------------------------------------------------------------
    "Protoporeia Causal": {
        "greek_latin": "Πρωτοπορεία Αιτιώδης",
        "math": "G(r,Δt) = H(c·Δt-r) / (2π·c·√(c²·Δt²-r²))",
        "role": (
            "Retarded causal wave operator — the fundamental solution to the 2D "
            "wave equation ∂²η/∂t² = c²∇²η.  The Heaviside factor H(c·Δt-r) "
            "enforces strict relativistic/wave-speed spatial causality bounds: "
            "no feature update can propagate across DACC cells faster than c."
        ),
        "python": "topos.physics.wave_sheaf.CausalGreensKernel",
        "equation": "G(r,Δt) = H(c·Δt-r)/(2πc·√(c²Δt²-r²)),  A(r)∝1/√r",
    },

    # -----------------------------------------------------------------------
    # Boundary Nilpotency
    # -----------------------------------------------------------------------
    "Anilpotencia Gradualis": {
        "greek_latin": "Nilpotentia Gradualis",
        "math": "∂_{k-1} ∘ ∂_k ≡ 0 ⟺ B_{k-1}·B_k = 0",
        "role": (
            "Universal topological conservation law.  Enforces that force distributions, "
            "dielectric fluxes, and wave energy flowing across DACC cell strata satisfy "
            "exact boundary conservation without generating monopoles or torn boundaries. "
            "Verified at compile / instantiation time over ℤ."
        ),
        "python": "topos.physics.wave_sheaf.DispersiveWaveSheafDACC._verify_nilpotency",
        "equation": "B_{k-1}·B_k = 0 mod ℤ  ∀k ∈ {1,…,4}",
    },

    # -----------------------------------------------------------------------
    # Dielectric Elasticity (Logarithmic)
    # -----------------------------------------------------------------------
    "Anisotropia Dielectrica": {
        "greek_latin": "Sheaf Dielectrica",
        "math": "η_ε(r) = d ln ε(r) / d ln r",
        "role": (
            "Logarithmic dielectric elasticity — the dynamic reaction pressure of the "
            "hydration shell opposing uncompensated desolvation.  Acts as an outward "
            "thermodynamic restoring force that penalises exposed hydrophobic surfaces. "
            "Coupled to Campbell variance for dynamic TΔS_solv computation."
        ),
        "python": "plq.gnc.sheaf.AnisotropiaDielectricaSheaf.dielectric_elasticity",
        "equation": "ε(r) = εᵢₙₜ + (εₑₓₜ-εᵢₙₜ)/(1+k·exp(-λr)),  η_ε = d ln ε/d ln r",
    },

    # -----------------------------------------------------------------------
    # Bifold Cooperative Partition Function
    # -----------------------------------------------------------------------
    "Synergeia Bifolda": {
        "greek_latin": "Συνέργεια Δίπτυχος",
        "math": "Ξ(pH/E) = Σ_{s∈{0,1}³} exp(-G(s)/k_BT),  n_Hill = 3.0",
        "role": (
            "8-state bifold grand canonical partition function over coupled domain triads "
            "(histidine-carboxylate triads in biology; ferroelectric/superconducting grain "
            "domains in materials science).  At the critical switching threshold E*, "
            "intermediate microstates collapse → sharp Heaviside step: ΔQ ≥ +3.5e "
            "between pH 7.4 (serum) and pH 6.0 (endosome)."
        ),
        "python": "plq.gnc.allostery.SynergeiaBifoldaCascade",
        "equation": "G(s,pH) = ln(10)kT Σᵢ sᵢ(pH-pKaᵢ) + Σᵢ<ⱼ sᵢsⱼ·J - Σᵢ sᵢ·Uᵢ",
    },

    # -----------------------------------------------------------------------
    # Algebraic Inverse Binder
    # -----------------------------------------------------------------------
    "Katalepsis Inversa": {
        "greek_latin": "Completio Inverse",
        "math": "Q* = K⁻¹ = K̃/‖K‖²_Cl",
        "role": (
            "Exact coordinate-free complementary binder derivation via multivector inversion. "
            "Monte Carlo validated (N=50,000): ‖Q*∧K‖²_Cl = 0.0000 (exact steric cancellation), "
            "⟨Q*·K⟩₀ = 1.0000 (perfect scalar alignment).  Replaces stochastic diffusion search "
            "with a single closed-form matrix inversion step."
        ),
        "python": "topos.physics.wave_sheaf.DispersiveWaveSheafDACC.compute_algebraic_inverse",
        "equation": "Q*·K = K̃·K/‖K‖²_Cl = ‖K‖²_Cl·1/‖K‖²_Cl = 1",
    },

    # -----------------------------------------------------------------------
    # Campbell's Theorem Entropy
    # -----------------------------------------------------------------------
    "Campbell Entropia": {
        "greek_latin": "Theorema Campbelli Entropia",
        "math": "Var(η) = λ(μ²_A+σ²_A)/(4πc²)·[t·ln(ct/2δ)-t]",
        "role": (
            "Closed-form thermodynamic variance of the Poisson-driven hydration field. "
            "Directly quantifies dynamic entropic solvation penalty: TΔS_solv ∝ Var(η_water). "
            "Replaces static implicit dielectric continuum with exact temporal fluctuation "
            "statistics.  Prevents greasy-binder collapse (φ>0.65) via φ* ≈ 0.282 attractor."
        ),
        "python": "topos.physics.poisson_forcing.CampbellTheoremVariance",
        "equation": "TΔS_solv = κ·∫_Ω Var(η(x,t))·[1+η_ε(x)] d²x",
    },

    # -----------------------------------------------------------------------
    # Dispersive Hodge-Clifford Attention
    # -----------------------------------------------------------------------
    "Ateneia Dispersiva": {
        "greek_latin": "Attentio Dispersiva Cliffordiana",
        "math": "E_ij(t) = (1/16)(⟨Qᵢ·K̃ⱼ⟩₀ - λ‖Qᵢ∧Kⱼ‖²_Cl)·Φ_disp(‖xᵢ-xⱼ‖,t)",
        "role": (
            "Full dispersive Hodge-Clifford dual-stream attention kernel.  The scalar meet "
            "⟨Q·K̃⟩₀ encodes complementary alignment; the bivector clash ‖Q∧K‖² penalises "
            "steric overlap; Φ_disp(r,t) separates capillary (local, fast) from gravity "
            "(global, slow) information channels."
        ),
        "python": "plq.metrics.dispersive_attention.DispersiveHodgeCliffordAttention",
        "equation": "Φ_disp(r,t) = ∫₀^∞ S(k)·cos(kr-ω(k)t)dk",
    },
}


# ---------------------------------------------------------------------------
# Lookup & Display Utilities
# ---------------------------------------------------------------------------

def lookup(term: str) -> Optional[Dict[str, str]]:
    """Look up a vocabulary term by name (case-sensitive).

    Parameters
    ----------
    term : str
        Pontic-Greek or Latin term name (e.g. 'Protoporeia Causal').

    Returns
    -------
    dict or None
        Vocabulary entry dict with keys: greek_latin, math, role, python, equation.
    """
    return VOCABULARY.get(term)


def search(keyword: str) -> Dict[str, Dict[str, str]]:
    """Search vocabulary entries whose name, math, or role contain keyword.

    Parameters
    ----------
    keyword : str
        Case-insensitive search string.

    Returns
    -------
    dict
        Matching vocabulary entries.
    """
    kw = keyword.lower()
    return {
        k: v for k, v in VOCABULARY.items()
        if kw in k.lower()
        or kw in v.get("math", "").lower()
        or kw in v.get("role", "").lower()
        or kw in v.get("greek_latin", "").lower()
    }


def print_index() -> None:
    """Pretty-print the full Pontic-Greek/Latin vocabulary index."""
    width = 78
    print("=" * width)
    print("  TOPOSLANG & PLQ — PONTIC-GREEK/LATIN ARCHITECTURAL VOCABULARY INDEX")
    print("=" * width)
    for term, entry in VOCABULARY.items():
        print(f"\n▸ {term}  ({entry['greek_latin']})")
        print(f"  Math:    {entry['math']}")
        print(f"  Python:  {entry['python']}")
        print(f"  Eq:      {entry['equation']}")
        print(f"  Role:    {entry['role'][:120]}{'…' if len(entry['role'])>120 else ''}")
    print("\n" + "=" * width)


if __name__ == "__main__":
    print_index()
