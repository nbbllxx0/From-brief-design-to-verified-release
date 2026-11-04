"""Access to the voxfem solver package.

voxfem (the GPU finite-element and SIMP solver this project builds on) is distributed separately and is not part of
this repository. agentto uses it read-only and extends it only by composition or subclassing.

agentto looks for voxfem in this order:
1. the source directory named by the environment variable VOXFEM_SRC (or its older name AGENTTO_VOXFEM_SRC): the
   folder that contains the `voxfem/` package directory;
2. an installed package, found by a plain `import voxfem`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ENV_VARS = ("VOXFEM_SRC", "AGENTTO_VOXFEM_SRC")

_MISSING = (
    "agentto needs the voxfem solver, which is distributed separately and is not included in this repository. "
    "Either install voxfem so that `import voxfem` works, or set the environment variable VOXFEM_SRC to the "
    "directory that contains the voxfem package (the folder holding voxfem/__init__.py). "
    "See README.md, section 'The voxfem dependency'."
)


def _source_dir() -> Path | None:
    for name in _ENV_VARS:
        value = os.environ.get(name)
        if value:
            path = Path(value).expanduser()
            if not (path / "voxfem" / "__init__.py").is_file():
                raise ImportError(f"{name}={value!r} does not contain the voxfem package "
                                  f"(expected {path / 'voxfem' / '__init__.py'}). {_MISSING}")
            return path
    return None


VOXFEM_SRC = _source_dir()

if VOXFEM_SRC is not None:
    sys.dont_write_bytecode = True          # never write __pycache__ into the solver's source tree
    if str(VOXFEM_SRC) not in sys.path:
        sys.path.insert(0, str(VOXFEM_SRC))

try:
    import voxfem  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "voxfem":
        raise                               # voxfem was found but one of its own dependencies is missing
    raise ImportError(_MISSING) from exc

from voxfem.core import Grid, Problem, SparseLoad  # noqa: E402,F401
