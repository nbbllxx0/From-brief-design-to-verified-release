"""X0 identity (GPU part): voxfem.SIMP on the spec-built problem vs on voxfem's own problem.

For each FEAD problem at k (default 80, 1.02M elements): run both, then compare iteration count, compliance and
max |d rho_phys|. Same Config as the FEAD runs (mg64, volfrac 0.3, R2 schedule), guards on.
Writes results/X0/identity_gpu_k<k>.json.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentto import library  # noqa: E402
from agentto.problem import build  # noqa: E402
import cupy as cp  # noqa: E402
from voxfem.core import PROBLEMS  # noqa: E402
from voxfem.simp import SIMP, Config  # noqa: E402

k = int(sys.argv[1]) if len(sys.argv) > 1 else 80
out = ROOT / "results" / "X0" / f"identity_gpu_k{k}.json"
out.parent.mkdir(parents=True, exist_ok=True)


def run(prob):
    s = SIMP(prob, Config(volfrac=0.30, solver="mg64", op32="stencil", log_every=0), log=lambda *_: None)
    t = time.perf_counter()
    r = s.run()
    rho = s.final_density().copy()
    r = {key: r[key] for key in ("converged", "valid", "iterations", "compliance", "volume_phys", "mnd")}
    r["wall_s"] = time.perf_counter() - t
    del s
    cp.get_default_memory_pool().free_all_blocks()
    return r, rho


rows = []
for name in ("cantilever", "bridge", "torsion"):
    ra, rho_a = run(PROBLEMS[name](k))
    rb, rho_b = run(build(library.fead(name, k)).problem)
    row = {"problem": name, "k": k, "voxfem": ra, "spec": rb,
           "same_iterations": ra["iterations"] == rb["iterations"],
           "compliance_rel_diff": abs(ra["compliance"] - rb["compliance"]) / abs(ra["compliance"]),
           "max_abs_drho": float(cp.max(cp.abs(rho_a - rho_b)))}
    row["pass"] = row["same_iterations"] and row["compliance_rel_diff"] <= 1e-9 and row["max_abs_drho"] == 0.0
    print(json.dumps(row), flush=True)
    rows.append(row)
    del rho_a, rho_b

out.write_text(json.dumps({"device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                           "criterion": "same iterations, compliance within 1e-9, max|d rho| = 0",
                           "all_pass": all(r["pass"] for r in rows), "rows": rows}, indent=1))
