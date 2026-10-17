"""Pharmacological Lattice Quantization (PLQ) with ToposLang (tau-Lang) and Geometric Native Coding (GNC).

Transforms conventional chemical graph and ligand-target pipelines into
a lock-free, coordinate-free, and topologically verified execution system.
"""

__version__ = "2.0.0"

from plq.dacc.complex import MolecularDACC, PocketDACC, DACCCell, DACCComplex
from plq.dacc.fingerprint import TopologicalDACCInvariant
from plq.gnc.clifford import Multivector, PGA3D, CGA3D
from plq.gnc.versor import VersorMotor, CoordinateFreeAligner
from plq.gnc.sheaf import CliffordSheaf
from plq.ternary.lattice import TernaryLattice, TernaryState
from plq.ternary.his_switch import HistidineSwitchNetwork, BifoldManifold
from plq.polypharmacology.dual_tier import DualTierScheduler, TargetScreeningWavefront
from plq.hodge.laplacian import CombinatorialHodgeLaplacian
from plq.hodge.engine import HodgeGuidedGenerator
from plq.compiler.symbolic_gnc import SymbolicGNCPass

__all__ = [
    "MolecularDACC",
    "PocketDACC",
    "DACCCell",
    "DACCComplex",
    "TopologicalDACCInvariant",
    "Multivector",
    "PGA3D",
    "CGA3D",
    "VersorMotor",
    "CoordinateFreeAligner",
    "CliffordSheaf",
    "TernaryLattice",
    "TernaryState",
    "HistidineSwitchNetwork",
    "BifoldManifold",
    "DualTierScheduler",
    "TargetScreeningWavefront",
    "CombinatorialHodgeLaplacian",
    "HodgeGuidedGenerator",
    "SymbolicGNCPass",
]
