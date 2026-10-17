"""PLQ Neural Module.

Provides native 8D Hodge-Clifford Spatial Transformer (HCST) architectures
for molecular pocket scoring, equivariant multivector message passing,
and homology-preserving topological pooling.
"""

from plq.neural.hcst_pocket import (
    PLQHCST8DTransformer,
    HCSTAttentionScoreReport,
    MultivectorStalk8D,
)

__all__ = [
    "PLQHCST8DTransformer",
    "HCSTAttentionScoreReport",
    "MultivectorStalk8D",
]
