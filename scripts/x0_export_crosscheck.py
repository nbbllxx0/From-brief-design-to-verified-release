"""X0 independent cross-check, export side: optimise two problems with passive regions, then save the final
design's moduli, supports and nodal loads so another FE code can recompute the compliance at that fixed design.

  passive_cantilever  FEAD cantilever k=20 (16k elements, nondimensional) + keep-out cylinder + keep-in box
  b05_l_bracket       brief b05 at h = 2 mm (50 x 50 x 20 = 50k elements), steel, keep-out corner

Each npz holds: shape (nx,ny,nz), size (Lx,Ly,Lz), h, E_phys (per element, x-major ravel, = E_material * E32),
nu, fixed (uint8 per node, bit0 x, bit1 y, bit2 z; node ids (ix*(ny+1)+iy)*(nz+1)+iz), load_idx (DOF 3*node+c),
load_val (force), and the voxfem compliance in physical units (compliance_phys = c * 2 / (E h)).
Writes results/X0/crosscheck_<name>.npz and .json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import cupy as cp  # noqa: E402
from agentto import library  # noqa: E402
from agentto.problem import build  # noqa: E402
from agentto.solver import SessionSIMP  # noqa: E402
from agentto.spec import Spec  # noqa: E402
from voxfem.core import unpad_elements  # noqa: E402
from voxfem.simp import Config  # noqa: E402

OUT = ROOT / "results" / "X0"
OUT.mkdir(parents=True, exist_ok=True)


def passive_cantilever():
    d = library.cantilever(20).model_dump()
    d["passive"] = [
        {"id": "hole", "kind": "void", "region": {"kind": "cylinder", "axis": "z", "center": [0.8, 0.5],
                                                   "radius": 0.2, "span": [0.0, 1.0]}},
        {"id": "boss", "kind": "solid", "region": {"kind": "box", "min": [1.8, 0.3, 0.3], "max": [2.0, 0.7, 0.7]}},
    ]
    return Spec.model_validate(d)


def b05():
    t = json.loads((ROOT / "briefs" / "intended" / "b05_l_bracket.json").read_text())
    d = t["spec"]
    d["resolution"] = {"h": 2.0}
    return Spec.model_validate(d)


for name, spec in (("passive_cantilever", passive_cantilever()), ("b05_l_bracket", b05())):
    sp = build(spec)
    s = SessionSIMP(sp, Config(volfrac=spec.volume_fraction, solver="mg64", op32="stencil", tol=1e-8,
                               spill_check_every=0, log_every=0))
    r = s.run()
    E32 = cp.asnumpy(unpad_elements(s.E32, sp.grid, s.grid)).astype(np.float64)
    g = sp.grid
    np.savez_compressed(OUT / f"crosscheck_{name}.npz", shape=np.array(g.shape), size=np.array(spec.domain.size),
                        h=sp.h, E_phys=E32 * spec.material.E, nu=spec.material.nu, fixed=sp.problem.fixed,
                        load_idx=sp.problem.load.idx, load_val=sp.problem.load.val)
    meta = {"name": name, "grid": list(g.shape), "n_elem": g.n_elem, "h": sp.h, "status": r["status"],
            "iterations": r["iterations"], "final_rel_true": r["final_solve"]["rel_true"], "tol": 1e-8,
            "compliance_nd": r["compliance"], "compliance_phys": r["compliance"] * sp.compliance_scale,
            "units": spec.units.model_dump(), "passive": s.passive_check()}
    (OUT / f"crosscheck_{name}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))
