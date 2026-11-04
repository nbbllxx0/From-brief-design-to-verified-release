"""X1 follow-up: re-run a failed cold reference at the release rung with a more robust solver, and re-score its arms.

On 2026-10-30 the cold bridge at 65.5M (mg64) stopped at iteration 15 with
SolveFailed('ir_lean: correction rejected with inner tolerance 1e-07'), where the schedule moves from p 1.5 to p 3.5.
The warm arms were fine, but their gaps were computed against the failed run. This script runs the cold arm again
with solver mg_ir (then, if that fails too, with the dilated coarse-support rule), keeps the failed row as
"cold_failed", and recomputes gap_to_cold and the shape metrics of every arm from the saved pooled fields.

    python -B scripts/x1_fix_cold.py bridge
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
from agentto.escalation import arm_schedule  # noqa: E402
from agentto.problem import build  # noqa: E402
from agentto.shape import betti, compare, pool  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from x1_escalation import FIELDS, OUT, cfg, problems  # noqa: E402


def run_cold(spec, h, solver, coarse_bc):
    import gc
    import time
    import cupy as cp
    from voxfem.simp import Config
    from agentto.solver import SessionSIMP
    sp = build(spec, h=h)
    sch, _ = arm_schedule("cold")
    kw = {**cfg(True, sp.grid.n_elem), "solver": solver, "schedule": sch}
    s = SessionSIMP(sp, Config(volfrac=spec.volume_fraction, **kw), coarse_bc=coarse_bc)
    t0 = time.perf_counter()
    r = s.run()
    from agentto.escalation import _unpad
    rho = _unpad(s, s.rho_phys)
    row = {"arm": "cold", "status": r["status"], "iterations": r["iterations"], "compute_s": r["compute_s"],
           "wall_s": time.perf_counter() - t0, "s_per_iter": r["compute_s"] / max(1, r["iterations"]),
           "compliance": r["compliance"], "volume": r["volume_phys"], "certified": r["certificate"]["certified"],
           "rmin_final": sch[-1][4], "error": r.get("error"), "grid": list(sp.grid.shape), "n_elem": int(sp.grid.n_elem),
           "solver": solver, "coarse_bc": s.coarse_bc, "rerun": True}
    del s
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return row, rho


def main(name="bridge"):
    data = json.loads(OUT.read_text())
    row = next(r for r in data["rows"] if r["problem"] == name)
    spec = problems()[name]
    h = row["rungs"]["release"]["h"]
    rel = row["release"]
    tries = [("mg_ir", "auto"), ("mg_ir", "dilate"), ("mg64", "dilate")]
    for solver, cbc in tries:
        cold, rho = run_cold(spec, h, solver, cbc)
        print(name, solver, cbc, cold["status"], cold["iterations"], f"{cold['compute_s']:.1f}s", cold.get("error") or "",
              flush=True)
        if cold["status"] == "converged":
            break
    if cold["status"] != "converged":
        print("no robust cold reference; leaving the file unchanged")
        return
    np.save(FIELDS / f"{name}_cold_release_p2.npy", pool(rho, 2).astype(np.float16))
    f = rel["tail"]["pool"]
    mc = (pool(rho, f) > 0.5) if f > 1 else rho > 0.5
    cold.update(pool=f, betti_end=betti(mc))
    rel["cold_failed"] = rel["cold"]
    rel["cold"] = cold
    for arm in ("cold", "tail", "reopen", "reopen_long"):
        v = rel[arm]
        v["gap_to_cold"] = v["compliance"] / cold["compliance"] - 1
        p2 = np.load(FIELDS / f"{name}_{arm}_release_p2.npy").astype(np.float32)
        m = (pool(p2, f // 2) > 0.5) if f > 2 else p2 > 0.5        # p2 is already pooled by 2
        v["shape_vs_cold"] = compare(m, mc)
        print(f"  {arm:12s} gap {v['gap_to_cold']:+.3%} IoU {v['shape_vs_cold']['iou']:.3f}")
    rel["cold_failed"]["note"] = "failed run kept for the record; gaps are to the re-run cold arm"
    OUT.write_text(json.dumps(data, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
