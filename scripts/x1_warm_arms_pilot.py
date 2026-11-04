"""X1 pilot: warm-start arms on one coarse -> fine step, against a cold fine run.

Arms (fine rung, all starting from the same converged coarse design):
  tail_raw       upsample only, final phase (p 4.5, beta 32, move 0.05)    -- what the 2026-10-29 pilot did
  tail_matched   upsample + one filter pass + exact volume match, final phase
  reopen         upsample + filter pass + volume match, beta 8 -> 16 -> 32 (moves 0.10/0.08/0.05)
  cold           full R2 schedule from uniform density (reference)
Metrics: status, iterations, compute s, compliance and its gap to cold, volume, and how much the design changed
from its starting point (fraction of elements with |rho_end - rho_start| > 0.5).
Cases: FEAD cantilever k 40 -> 80, brief b06 (passive bosses) h 2 -> 1. Sessions use conv_moving_ppm = 100.
Writes results/X1/warm_arms_pilot.json.
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
from agentto.solver import SessionSIMP, reopen_schedule, tail_schedule  # noqa: E402
from agentto.spec import Spec  # noqa: E402
from voxfem.simp import Config  # noqa: E402


def cfg(vf, **kw):
    base = dict(volfrac=vf, solver="mg64", op32="stencil", spill_check_every=0, log_every=0, conv_moving_ppm=100.0,
                max_outer=300)
    base.update(kw)
    return Config(**base)


def b06():
    return Spec.model_validate(json.loads((ROOT / "briefs" / "intended" / "b06_motor_mount.json").read_text())["spec"])


cases = [("cantilever_k40_to_k80", library.cantilever(40), 1.0 / 40, 1.0 / 80),
         ("b06_h2_to_h1", b06(), 2.0, 1.0)]
rows = []
for name, spec, hc, hf in cases:
    vf = spec.volume_fraction
    s = SessionSIMP(build(spec, h=hc), cfg(vf))
    rc = s.run()
    coarse = cp.asnumpy(s.final_density()).reshape(s.sp.grid.shape)
    del s
    arms = {"cold": None, "tail_raw": (tail_schedule(), False, False), "tail_matched": (tail_schedule(), True, True),
            "reopen": (reopen_schedule(), True, True)}
    res = {}
    for arm, a in arms.items():
        kw = {} if a is None else {"schedule": a[0]}
        s = SessionSIMP(build(spec, h=hf), cfg(vf, **kw))
        info = {}
        if a is not None:
            info = s.warm_start(coarse, smooth=a[1], match_volume=a[2])
        start = cp.asnumpy(s.rho).copy()
        r = s.run()
        end = cp.asnumpy(s.rho)
        res[arm] = {"status": r["status"], "iterations": r["iterations"], "compute_s": r["compute_s"],
                    "compliance": r["compliance"], "volume": r["volume_phys"],
                    "changed_frac": float(np.mean(np.abs(end - start) > 0.5)), "warm_info": info,
                    "levels_overridden": r["levels_overridden"], "error": r.get("error")}
        del s
        cp.get_default_memory_pool().free_all_blocks()
    c0 = res["cold"]["compliance"]
    for arm in res:
        if res[arm]["compliance"] is not None and c0:
            res[arm]["gap_to_cold"] = res[arm]["compliance"] / c0 - 1
    row = {"case": name, "coarse": {"iterations": rc["iterations"], "compute_s": rc["compute_s"]}, "arms": res}
    rows.append(row)
    for arm, v in res.items():
        print(name, f"{arm:13s}", v["status"], v["iterations"], f"{v['compute_s']:.1f}s",
              f"gap {v.get('gap_to_cold', float('nan')):+.4%}", f"changed {v['changed_frac']:.3%}", v.get("error") or "",
              flush=True)
(ROOT / "results" / "X1" / "warm_arms_pilot.json").write_text(json.dumps({"rows": rows}, indent=1))
