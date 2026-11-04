"""Diagnose warm-start convergence and iteration cost on brief b06 (motor mount, passive bosses).

Sketch rung h=2 cold, then h=1 warm (tail schedule), logging per iteration: compliance, max |d rho|, number of
elements still moving, PCG iterations, solve and total iteration time. Writes results/diag/warm_b06.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cupy as cp  # noqa: E402
from agentto.problem import build  # noqa: E402
from agentto.solver import SessionSIMP, tail_schedule  # noqa: E402
from agentto.spec import Spec  # noqa: E402
from voxfem.simp import Config  # noqa: E402

spec = Spec.model_validate(json.loads((ROOT / "briefs" / "intended" / "b06_motor_mount.json").read_text())["spec"])
out = {}


def run(h, warm=None, max_outer=150):
    sp = build(spec, h=h)
    kw = dict(volfrac=spec.volume_fraction, solver="mg64", op32="stencil", spill_check_every=0, log_every=0,
              max_outer=max_outer, conv_moving_ppm=float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)
    if warm is not None:
        kw["schedule"] = tail_schedule()
    s = SessionSIMP(sp, Config(**kw))
    if warm is not None:
        s.warm_start(warm)
    r = s.run()
    hist = [{"it": x["it"], "c": x["c"], "drho": x["drho_phys"], "moving": x["n_elem_drho_gt_tol"],
             "pcg": x["solve"].get("iters"), "solve_s": x["solve"]["wall_s"], "setup_s": x["solve"]["setup_s"],
             "t_iter_s": x["t_iter_s"], "oc_probes": x["oc_probes"]} for x in s.history]
    rho = cp.asnumpy(s.final_density()).reshape(sp.grid.shape)
    print(h, r["status"], r["iterations"], r["coarse_bc"], f"compute {r['compute_s']:.1f}s",
          "levels", s.mg_levels, s.grid.shape, r.get("error"), flush=True)
    return rho, {"h": h, "n_elem": sp.grid.n_elem, "status": r["status"], "iterations": r["iterations"],
                 "coarse_bc": r["coarse_bc"], "compute_s": r["compute_s"], "history": hist}


rho2, out["h2_cold"] = run(2.0)
_, out["h1_warm"] = run(1.0, warm=rho2)
dst = ROOT / "results" / "diag"
dst.mkdir(parents=True, exist_ok=True)
(dst / "warm_b06.json").write_text(json.dumps(out, indent=1))
for key in out:
    hs = out[key]["history"]
    print(key, "mean t_iter", sum(x["t_iter_s"] for x in hs) / len(hs), "mean pcg", sum(x["pcg"] or 0 for x in hs) / len(hs),
          "last moving", [x["moving"] for x in hs[-8:]], "last drho", [round(x["drho"], 3) for x in hs[-8:]])
