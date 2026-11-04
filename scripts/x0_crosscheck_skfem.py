"""X0 independent cross-check, solve side: recompute the compliance of each exported design with scikit-fem.

scikit-fem has its own trilinear hexahedron, quadrature, assembly and node numbering; nothing is shared with
voxfem except the exported data (per-element moduli, fixed DOFs, nodal loads, all in physical units). A match
checks: the element stiffness and operator, the spec -> problem node selection, passive moduli (Emin, not 0), and
agentto's physical compliance scaling 2/(E h). Direct sparse solve (SuperLU).
Reads results/X0/crosscheck_<name>.npz/.json, writes results/X0/crosscheck_skfem.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementHex0, ElementHex1, ElementVector, MeshHex, asm, condense
from skfem.helpers import ddot, sym_grad, trace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "X0"
names = sys.argv[1:] or ["passive_cantilever", "b05_l_bracket"]
rows = []
for name in names:
    d = np.load(OUT / f"crosscheck_{name}.npz")
    meta = json.loads((OUT / f"crosscheck_{name}.json").read_text())
    nx, ny, nz = (int(v) for v in d["shape"])
    Lx, Ly, Lz = (float(v) for v in d["size"])
    h, nu = float(d["h"]), float(d["nu"])
    t0 = time.perf_counter()
    mesh = MeshHex.init_tensor(np.linspace(0, Lx, nx + 1), np.linspace(0, Ly, ny + 1), np.linspace(0, Lz, nz + 1))
    basis = Basis(mesh, ElementVector(ElementHex1()), intorder=3)
    # element moduli in scikit-fem's element order, located by centroid
    cen = mesh.p[:, mesh.t].mean(axis=1)
    ex, ey, ez = (np.floor(cen[i] / h).astype(int) for i in range(3))
    E = d["E_phys"][(ex * ny + ey) * nz + ez]
    b0 = basis.with_element(ElementHex0())
    Ef = b0.interpolate(E)

    @BilinearForm
    def stiffness(u, v, w):
        lam = w.E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = w.E / (2 * (1 + nu))
        eu, ev = sym_grad(u), sym_grad(v)
        return 2 * mu * ddot(eu, ev) + lam * trace(eu) * trace(ev)

    K = asm(stiffness, basis, E=Ef)
    # voxfem node id of every scikit-fem node
    ix, iy, iz = (np.rint(mesh.p[i] / h).astype(int) for i in range(3))
    vox = (ix * (ny + 1) + iy) * (nz + 1) + iz
    sk_of_vox = np.empty(vox.size, np.int64)
    sk_of_vox[vox] = np.arange(vox.size)
    dofs = basis.nodal_dofs                      # (3, n_nodes)
    f = np.zeros(basis.N)
    node, comp = np.divmod(d["load_idx"], 3)
    np.add.at(f, dofs[comp, sk_of_vox[node]], d["load_val"])
    fixed = d["fixed"]
    D = np.concatenate([dofs[c, sk_of_vox[np.nonzero(fixed & (1 << c))[0]]] for c in range(3)])
    A, b, x, I = condense(K, f, D=D)
    u = np.zeros(basis.N)
    u[I] = spsolve(A.tocsc(), b)
    c_sk = float(f @ u)
    res = float(np.linalg.norm(A @ u[I] - b) / np.linalg.norm(b))
    row = {"name": name, "n_elem": nx * ny * nz, "ndof": int(basis.N), "compliance_skfem": c_sk,
           "compliance_voxfem": meta["compliance_phys"],
           "rel_diff": abs(c_sk - meta["compliance_phys"]) / abs(c_sk), "skfem_rel_residual": res,
           "voxfem_rel_true": meta["final_rel_true"], "wall_s": time.perf_counter() - t0}
    print(json.dumps(row), flush=True)
    rows.append(row)
(OUT / "crosscheck_skfem.json").write_text(json.dumps({
    "criterion": "relative compliance difference <= 1e-6 (voxfem solve accepted at 1e-8 true residual)",
    "all_pass": all(r["rel_diff"] <= 1e-6 for r in rows), "rows": rows}, indent=1))
