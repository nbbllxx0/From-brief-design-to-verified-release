"""The three FEAD benchmark problems restated as specs (nondimensional: E = 1, domain 2 x 1 x 1).

`fead(name, k)` must reproduce voxfem.core.PROBLEMS[name](k) exactly; tests/test_identity.py checks it.
"""
from __future__ import annotations

from .spec import Spec

_COMMON = {
    "units": {"length": "mm", "force": "N"},
    "domain": {"size": [2.0, 1.0, 1.0]},
    "material": {"name": "nondimensional", "E": 1.0, "nu": 0.3},
    "volume_fraction": 0.30,
}


def cantilever(k: int = 80, patch: float = 0.1, total: float = -1.0) -> Spec:
    return Spec.model_validate({
        **_COMMON, "name": "cantilever",
        "description": "FEAD cantilever: x=0 face clamped, -y load on a square patch centred on the x=2 face",
        "supports": [{"id": "clamp", "region": {"kind": "face", "face": "x-"}}],
        "loads": [{"id": "tip", "force": [0.0, total, 0.0],
                   "region": {"kind": "patch", "face": "x+", "center": [0.5, 0.5], "size": [patch, patch]}}],
        "resolution": {"cells_short_edge": k},
    })


def bridge(k: int = 80, patch: float = 0.1, total: float = -1.0) -> Spec:
    return Spec.model_validate({
        **_COMMON, "name": "bridge",
        "description": "FEAD bridge: bottom edge at x=0 pinned, bottom edge at x=2 roller (x free), "
                       "-y load on a full-depth strip at the top centre",
        "supports": [
            {"id": "pin", "region": {"kind": "edge", "axis": "z", "at": [0.0, 0.0]}},
            {"id": "roller", "fix": ["y", "z"], "region": {"kind": "edge", "axis": "z", "at": [2.0, 0.0]}},
        ],
        "loads": [{"id": "deck", "force": [0.0, total, 0.0],
                   "region": {"kind": "patch", "face": "y+", "center": [1.0, 0.5], "size": [patch, 1.0]}}],
        "resolution": {"cells_short_edge": k},
    })


def torsion(k: int = 80, patch: float = 0.15, total: float = 1.0) -> Spec:
    return Spec.model_validate({
        **_COMMON, "name": "torsion",
        "description": "FEAD torsion: x=0 face clamped, torque about x on the x=2 face from two patches at "
                       "y=1 and y=0 carrying opposite (y, z) forces",
        "supports": [{"id": "clamp", "region": {"kind": "face", "face": "x-"}}],
        "loads": [
            {"id": "top", "force": [0.0, -0.5 * total, 0.5 * total],
             "region": {"kind": "patch", "face": "x+", "center": [1.0, 0.5], "size": [patch, patch]}},
            {"id": "bottom", "force": [0.0, 0.5 * total, -0.5 * total],
             "region": {"kind": "patch", "face": "x+", "center": [0.0, 0.5], "size": [patch, patch]}},
        ],
        "resolution": {"cells_short_edge": k},
    })


FEAD = {"cantilever": cantilever, "bridge": bridge, "torsion": torsion}


def fead(name: str, k: int = 80) -> Spec:
    return FEAD[name](k)
