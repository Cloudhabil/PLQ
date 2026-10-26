"""ToposLang Physics Module: Continuous Scale-Invariance, Turbulence, and Constructive Yang-Mills."""

from topos.physics.navier_stokes import (
    NavierStokesProSolver,
    TurbulenceState,
    VortexFilament,
)
from topos.physics.orbit_space import (
    OrbitSpaceManifold,
    OrbitState,
    WilsonLoopObservable,
)
from topos.physics.peierls_vortex import (
    CenterVortexSurface,
    PeierlsHandoverCertificate,
    PeierlsVortexAnalyzer,
)
from topos.physics.bakry_emery import (
    BakryEmeryCertificate,
    BakryEmeryCurvatureEvaluator,
)

__all__ = [
    "NavierStokesProSolver",
    "TurbulenceState",
    "VortexFilament",
    "OrbitSpaceManifold",
    "OrbitState",
    "WilsonLoopObservable",
    "CenterVortexSurface",
    "PeierlsHandoverCertificate",
    "PeierlsVortexAnalyzer",
    "BakryEmeryCertificate",
    "BakryEmeryCurvatureEvaluator",
    # --- Wave Sheaf Extension (Dynamic Spatial-Temporal Clifford Wave Sheaf S_W) ---
    "CapillaryGravityDispersion",
    "PoissonImpulseField",
    "CampbellTheoremVariance",
    "WavePhysicalConfig",
    "PROTEIN_AQUEOUS_CONFIG",
    "SOLID_STATE_METAMATERIAL_CONFIG",
    "MACRO_FLUID_WATER_CONFIG",
    "CausalGreensKernel",
    "DispersiveWaveSheafDACC",
]

# --- Wave Sheaf Extension imports (lazy to avoid circular deps) ---
from topos.physics.dispersion import CapillaryGravityDispersion
from topos.physics.poisson_forcing import PoissonImpulseField, CampbellTheoremVariance
from topos.physics.wave_sheaf import (
    WavePhysicalConfig,
    PROTEIN_AQUEOUS_CONFIG,
    SOLID_STATE_METAMATERIAL_CONFIG,
    MACRO_FLUID_WATER_CONFIG,
    CausalGreensKernel,
    DispersiveWaveSheafDACC,
)
