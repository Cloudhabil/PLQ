"""Virtual Wet-Lab: Dynamic-Well Reaction Chamber & Screening Manifold.

Maintains complete independence outside PLQ-main and ToposLang-main while importing
and leveraging both as algebraic and physical engines.
"""

from __future__ import annotations
import sys
from pathlib import Path

_VW_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _VW_DIR
for _candidate in [_VW_DIR] + list(_VW_DIR.parents):
    if (_candidate / "ToposLang-main").exists() and (_candidate / "PLQ-main").exists():
        _WORKSPACE_ROOT = _candidate
        break

_PLQ_PATH = _WORKSPACE_ROOT / "PLQ-main" / "PLQ-main"
_TOPOS_PATH = _WORKSPACE_ROOT / "ToposLang-main" / "ToposLang-main" / "src"

for _p in [str(_PLQ_PATH), str(_TOPOS_PATH), str(_VW_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine import DynamicPlate, DynamicWell, WellCoordinates, WellAssayReadout
