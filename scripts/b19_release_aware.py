"""b19's 36M release under the pad-aware coarse rule (results/solver/NOTES.md), warm from the design X3's own refine
rung produced, with the same reopen_long schedule X3's release used. X3's release under voxfem's rule:
61 iterations, 1507.6 s compute, C 121.26051525896641, certified.

    python -B scripts/b19_release_aware.py [aware|voxfem]      (GPU; about 4 min for aware, 25 min for voxfem)
Writes results/solver/b19_release_<rule>.json.
"""
import json
import gc
import sys
from pathlib import Path

import numpy as np
import cupy as cp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentto.escalation import reopen_long_schedule  # noqa: E402
from agentto.problem import build  # noqa: E402
from agentto.solver import SessionSIMP  # noqa: E402
from agentto.spec import Spec  # noqa: E402
from agentto.tools import Session, read_log  # noqa: E402
from voxfem.simp import Config  # noqa: E402

SESSION = ROOT / "results" / "X3" / "b19_engine_bracket" / "session"


def main(rules):
    s = Session(None, profile="spec")
    for e in read_log(SESSION / "session.jsonl"):
        if e.get("type") == "call" and e["tool"] in ("new_spec", "set_spec", "set_params", "add_support", "add_load",
                                                      "add_passive", "update_item", "remove_item", "undo"):
            s.call(e["tool"], e["args"], actor=e["actor"])
    spec = Spec.model_validate(s.draft)
    sp = build(spec, h=0.0125)
    coarse = np.load(SESSION / "fields" / "rho_3e1401110c4c.npy").astype(np.float32)
    for mode in rules:
        kw = dict(volfrac=spec.volume_fraction, solver="mg64", op32="stencil", log_every=0, conv_moving_ppm=100.0,
                  spill_check_every=10, max_outer=300, schedule=reopen_long_schedule())
        m = SessionSIMP(sp, Config(**kw), coarse_pad=mode)
        m.warm_start(coarse)
        r = m.run()
        its = [h["solve"]["iters"] for h in m.history]
        out = {"rule": mode, "grid": list(sp.grid.shape), "padded": list(m.grid.shape), "levels": m.mg_levels,
               "status": r["status"], "iterations": r["iterations"], "compute_s": r["compute_s"],
               "pcg_iters_mean": float(np.mean(its)), "pcg_iters_max": int(max(its)),
               "compliance": r["compliance"] * sp.compliance_scale if r["compliance"] is not None else None,
               "certified": r["certificate"]["certified"] if r.get("certificate") else None}
        (ROOT / "results" / "solver" / f"b19_release_{mode}.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out), flush=True)
        m = None
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main(sys.argv[1:] or ["aware"])
