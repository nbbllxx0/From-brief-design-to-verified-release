"""X2 consequence-weighted error (CWE), on the GPU: what a wrong spec costs in the design.

For every X2 run whose spec is schema-valid: optimize under the predicted spec at the intended spec's preview rung
(~0.4-1.2M elements), then evaluate that design under the intended spec (one state solve at p 4.5) and compare with
the intended optimum:   CWE = C_intended(rho_pred) / C_intended(rho_intended) - 1.
A run whose spec is equivalent to the intended one has CWE 0 by definition (no solve); one that matches an
admissible reading counts as correct but its CWE is measured. Both specs are compared in mm and N. A different domain, or a
predicted spec that cannot be optimized (fails validation), gets CWE = inf. Keep-in/keep-out violations of the
predicted design under the intended passive regions are reported beside it (compliance alone does not see them).

    python -B scripts/x2_score.py [T G claude_high/GC ...] [--force]  (one GPU job at a time; runs scored before are
                                                                       kept unless --force)
Adds "cwe" to each results/X2/<cond>/<task>_r<rep>.json and writes results/X2/summary.json (all Codex conditions)
or summary_<conds>.json when conditions are named.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
from agentto.bench import normalized  # noqa: E402
from agentto.escalation import run_arm  # noqa: E402
from agentto.problem import build  # noqa: E402
from agentto.rules import check  # noqa: E402
from agentto.spec import Spec  # noqa: E402
from agentto.tools import _named  # noqa: E402
from agentto.worker import ladder  # noqa: E402

CFG = dict(solver="mg64", op32="stencil", spill_check_every=0, log_every=0, conv_moving_ppm=100.0, max_outer=300)


def evaluate(spec: Spec, h: float, rho_phys: np.ndarray, p: float = 4.5) -> float:
    """Compliance (physical units) of a fixed physical density under `spec` at mesh size h."""
    import cupy as cp
    from voxfem.core import pad_elements
    from voxfem.simp import Config, _modulus
    from agentto.solver import SessionSIMP
    sp = build(spec, h=h)
    s = SessionSIMP(sp, Config(volfrac=spec.volume_fraction, **CFG))
    s.rho_phys[...] = cp.asarray(pad_elements(rho_phys.ravel().astype(np.float32), sp.problem.grid, s.grid))
    _modulus(s.rho_phys, np.float32(p), np.float32(s.cfg.Emin), np.float32(s.cfg.E0), s.E32)
    if s.active is not None:
        s.E32 *= s.active
    s.solve()
    c = float(s.load.dot(s.u)) * sp.compliance_scale
    del s
    import gc
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return c


def preview_h(spec: Spec) -> float:
    r = {x["name"]: x for x in _named(ladder(spec))}
    return (r.get("preview") or list(r.values())[-1])["h"]


def main(force: bool = False, conds=None):
    X2 = ROOT / "results" / "X2"
    # a condition is a folder under results/X2: "G", or a provider's folder such as "claude_high/GC"
    files = sorted(X2.glob("*/*_r*.json")) if not conds else         sorted(f for f in X2.glob("**/*_r*.json") if f.parent.relative_to(X2).as_posix() in conds)
    ref: dict[str, dict] = {}
    rows = []
    for f in files:
        rec = json.loads(f.read_text())
        task = rec["task"]
        intended = json.loads((ROOT / "briefs" / "intended" / f"{task}.json").read_text())
        if "cwe" in rec and not force:                  # scored before: keep (re-score with --force)
            pass
        elif intended["expected"] == "refuse":
            rec["cwe"] = {"value": None, "note": "refuse task: scored on behaviour only"}
        elif not rec.get("schema_valid"):
            rec["cwe"] = {"value": math.inf, "note": "no valid spec"}
        elif rec.get("correct_spec") and (rec.get("match") or {}).get("matched", "intended") == "intended":
            rec["cwe"] = {"value": 0.0, "note": "equivalent to the intended spec"}
        else:                                           # wrong, or an admissible reading: measure what it costs
            q = normalized(intended["spec"])
            p = normalized(rec["spec"])
            h = preview_h(q)
            if task not in ref:
                row, rho, _ = run_arm(build(q, h=h), CFG, "cold")
                ref[task] = {"h": h, "c": evaluate(q, h, rho), "rho": rho, "status": row["status"]}
            R = ref[task]
            if list(p.domain.size) != list(q.domain.size):
                rec["cwe"] = {"value": math.inf, "note": "different domain"}
            elif not check(p, h=h, solver="mg_ir").ok:
                rec["cwe"] = {"value": math.inf, "note": "predicted spec fails validation at the scoring grid"}
            else:
                row, rho_p, _ = run_arm(build(p, h=h), CFG, "cold")
                c = evaluate(q, h, rho_p)
                sq = build(q, h=h)
                viol = {}
                if sq.solid is not None:
                    viol["keep_in_missing"] = float(np.mean(rho_p.ravel()[sq.solid] < 0.5))
                if sq.void is not None:
                    viol["keep_out_filled"] = float(np.mean(rho_p.ravel()[sq.void] > 0.5))
                # a wrong spec can use more material than the brief allows (invented keep-ins, a guessed volume
                # fraction) and then look stiffer: report the material ratio beside the CWE
                rec["cwe"] = {"value": c / R["c"] - 1, "c_pred_under_intended": c, "c_intended_opt": R["c"],
                              "h": h, "pred_status": row["status"], "passive_violation": viol,
                              "mass_ratio": float(rho_p.mean() / R["rho"].mean())}
        f.write_text(json.dumps(rec, indent=1))
        v = rec["cwe"]["value"]
        rows.append({"cond": rec["cond"], "task": task, "rep": rec["rep"], "correct": rec.get("correct_spec"),
                     "behaviour_ok": rec.get("behaviour_ok"), "cwe": v, "questions": rec.get("questions", 0)})
        print(f"{rec['cond']:3s} {task:22s} correct {str(rec.get('correct_spec')):5s} "
              f"cwe {('%.4f' % v) if isinstance(v, float) and math.isfinite(v) else v}", flush=True)
    summ = {}
    for r in rows:
        s = summ.setdefault(r["cond"], {"n": 0, "correct": 0, "behaviour_ok": 0, "cwe_finite": [], "cwe_inf": 0,
                                         "questions": 0})
        s["n"] += 1
        s["correct"] += bool(r["correct"])
        s["behaviour_ok"] += bool(r["behaviour_ok"])
        s["questions"] += r["questions"]
        if isinstance(r["cwe"], float):
            if math.isfinite(r["cwe"]):
                s["cwe_finite"].append(r["cwe"])
            else:
                s["cwe_inf"] += 1
    for s in summ.values():
        x = s.pop("cwe_finite")
        s["cwe_median"] = float(np.median(x)) if x else None
        s["cwe_mean"] = float(np.mean(x)) if x else None
        s["cwe_n"] = len(x)
    out = "summary.json" if not conds else f"summary_{'_'.join(c.replace('/', '-') for c in conds)}.json"
    (X2 / out).write_text(json.dumps({"by_cond": summ, "rows": rows}, indent=1,
                                                                     default=str))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(force="--force" in sys.argv, conds=[a for a in sys.argv[1:] if not a.startswith("--")] or None)
