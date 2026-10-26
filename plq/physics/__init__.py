"""PLQ Physics Sub-package — Wave Mechanics Bridge for Binding Pocket Dynamics.

Re-exports ToposLang wave sheaf primitives and provides the PLQ-specific
PocketWaveField for binding kinetics simulation.

The PLQ physics layer connects the continuous Green's function wave propagator
(Protoporeia Causal) to molecular binding pockets via:
    - PocketWaveField: applies wave impulse forcing to residue-level DACC stalks
    - Preset configs: PROTEIN_AQUEOUS_CONFIG, SOLID_STATE_METAMATERIAL_CONFIG

All modules are pure Python (math module only).
"""

from plq.physics.wave_pocket import PocketWaveField, PocketWaveResult

__all__ = [
    "PocketWaveField",
    "PocketWaveResult",
]
