"""X0 general BCs: voxfem's injection coarse rule vs RobustGMG's dilation on briefs whose supports injection loses.

For each case: 25 optimisation iterations with each rule (mg64, deterministic coarse assembly), recording PCG
iterations per state solve, whether every solve was accepted (FP64 true residual <= 1e-5), and the coarsest
level's free DOFs. Cases: b05 (L-bracket, ceiling patch on an odd line at the coarsest stride), b09 (duct bridge,
right bearing lost), and the FEAD cantilever (injection safe) as a control.
Writes results/X0/coarse_bc.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
from agentto import library  # noqa: E402
from agentto.problem import build  # noqa: E402
from agentto.rules import coarse_check  # noqa: E402
from agentto.solver import SessionSIMP  # noqa: E402
from agentto.spec import Spec  # noqa: E402
from voxfem.simp import Config  # noqa: E402


def task(slug, h):
    d = json.loads((ROOT / "briefs" / "intended" / f"{slug}.json").read_text())["spec"]
    d["resolution"] = {"h": h}
    return Spec.model_validate(d)


def both_odd_bridge(k=40):
    """Both bearings on the first odd node line: injection keeps no support at all on the coarse levels."""
    d = library.bridge(k).model_dump()
    for s in d["supports"]:
        s["region"]["at"] = [s["region"]["at"][0], 1.0 / k]
    return Spec.model_validate(d)


cases = [("b05_l_bracket_h1", task("b05_l_bracket", 1.0)), ("b09_duct_bridge_h2", task("b09_duct_bridge", 2.0)),
         ("bridge_k40_both_odd", both_odd_bridge(40)), ("cantilever_k40_control", library.cantilever(40))]
if len(sys.argv) > 1:
    cases = [c for c in cases if c[0] in sys.argv[1:]]
rows = []
for name, spec in cases:
    sp = build(spec)
    cc = coarse_check(sp)
    for rule in ("inject", "dilate"):
        s = SessionSIMP(sp, Config(volfrac=spec.volume_fraction, solver="mg64", op32="stencil", max_outer=25,
                                   spill_check_every=0, log_every=0), coarse_bc=rule)
        r = s.run()
        its = [h["solve"].get("iters") for h in s.history]
        acc = [h["solve"]["rel_true"] <= 1e-5 for h in s.history]
        row = {"case": name, "n_elem": sp.grid.n_elem, "rule": rule, "status": r["status"],
               "lost_on_injection": cc["lost"], "free_modes_on_injection": cc["free_modes"],
               "mg_levels": s.mg_levels, "coarse_free_dofs": s.mg.coarse_free_dofs,
               "pcg_iters_mean": float(np.mean(its)) if its else None, "pcg_iters_max": max(its) if its else None,
               "all_solves_accepted": all(acc), "n_solves": len(acc), "error": r.get("error")}
        print(json.dumps(row), flush=True)
        rows.append(row)
(ROOT / "results" / "X0" / "coarse_bc.json").write_text(json.dumps({"rows": rows}, indent=1))
