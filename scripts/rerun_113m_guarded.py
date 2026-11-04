"""Clean re-run of the 113M escalation chain (pilot chain B) with voxfem's guards ON.

Cantilever, volfrac 0.3: cold k=96 (1.8M, mg64) -> warm k=192 (14.2M, mg64) -> warm k=384 (113.2M, mg_ir), each
warm rung starting from the 2x piecewise-constant upsampling of the previous design and running only the final
continuation phase. Spill checks every 10 iterations and at the end (default Config); a run that spills is
invalid. Compute time = sum of per-iteration times (excludes guard sampling). Records the VRAM held by other
processes before and after, since the desktop shares the GPU.
Writes results/X1/chainB_guarded.json.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentto import _voxfem  # noqa: E402,F401
import numpy as np  # noqa: E402
import cupy as cp  # noqa: E402
from voxfem import guard  # noqa: E402
from voxfem.core import PROBLEMS  # noqa: E402
from voxfem.simp import SIMP, Config  # noqa: E402

OUT = ROOT / "results" / "X1" / "chainB_guarded.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
TAIL = [[1, 4.5, 32.0, 0.05, 1.20], [1000, 4.5, 32.0, 0.05, 1.20]]
rows = []
env = {"device_before": guard.device_state(), "free_before": guard.effective_free_bytes()[1]}


def save():
    OUT.write_text(json.dumps({"note": "guards on (spill_check_every=10 + end check); desktop shares the GPU",
                               "env": env, "rows": rows}, indent=1))


def run(k, solver, warm=None):
    prob = PROBLEMS["cantilever"](k)
    kw = dict(volfrac=0.30, solver=solver, op32="stencil", log_every=0)
    if warm is not None:
        kw["schedule"] = [list(s) for s in TAIL]
    t0 = time.perf_counter()
    s = SIMP(prob, Config(**kw), log=lambda *_: None)
    t_setup = time.perf_counter() - t0
    if warm is not None:
        s.rho[...] = cp.asarray(warm.ravel(), dtype=cp.float32)
        del warm
    t1 = time.perf_counter()
    r = s.run()
    cp.cuda.Device().synchronize()
    row = {"k": k, "n_elem": prob.grid.n_elem, "solver": solver, "warm": "schedule" in kw,
           "setup_s": round(t_setup, 2), "run_wall_s": round(time.perf_counter() - t1, 2),
           "compute_s": round(sum(h["t_iter_s"] for h in s.history), 2), "iterations": r["iterations"],
           "compliance": r["compliance"], "converged": r["converged"], "valid": r["valid"],
           "spill": {k2: r["spill"][k2] for k2 in ("max_shared_growth_bytes", "spilled")},
           "others_dedicated_GiB": [round(x["others_dedicated"] / 2**30, 2) for x in r["spill"]["samples"]],
           "memory": r["memory"], "final_rel_true": r["final_solve"].get("rel_true")}
    print(json.dumps(row), flush=True)
    rows.append(row)
    save()
    rho = cp.asnumpy(s.final_density()).reshape(prob.grid.shape).astype(np.float32)
    del s
    import gc
    gc.collect()                                 # free the previous rung's arrays even if a reference cycle holds them
    cp.get_default_memory_pool().free_all_blocks()
    return rho


def up2(a):
    return np.repeat(np.repeat(np.repeat(a, 2, 0), 2, 1), 2, 2)


run(20, "mg64")
rows.clear()
d = run(96, "mg64")
d = run(192, "mg64", up2(d))
d = run(384, "mg_ir", up2(d))
del d
env["device_after"] = guard.device_state()
save()
