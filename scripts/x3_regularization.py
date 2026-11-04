"""Certificate regularization block reconstructed from the logs of the complete agent sessions (X3).

The reported sessions ran before the certificate recorded its regularization block (src/agentto/solver.py,
SessionSIMP.regularization). The same fields follow from each session log: the release job's element size, iteration
count and warm start, the solve call's `optimize` flag (long re-opening) and the stated resolution of the released
specification, which equals the intended reading in every released session. Filter radius 1.2 elements on the final
phase of every schedule. Writes results/X3/regularization.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentto.problem import grid_for          # noqa: E402
from agentto.spec import Spec                  # noqa: E402

RMIN = 1.2
X3 = ROOT / "results/X3"


def main():
    rows = []
    for d in sorted(p for p in X3.iterdir() if p.is_dir() and (p / "session/session.jsonl").is_file()):
        if "_run" in d.name:                     # superseded cantilever-block runs
            continue
        log = [json.loads(l) for l in (d / "session/session.jsonl").open(encoding="utf-8")]
        calls = {x["result"]["job"]: x for x in log if x.get("type") == "call" and x.get("tool") == "solve"
                 and isinstance(x.get("result"), dict) and x["result"].get("job")}
        rel = [x["result"] for x in log if x.get("type") == "result" and x["result"].get("release_rung")
               and x["result"].get("certified")]
        if not rel:
            rows.append({"task": d.name, "released": False})
            continue
        r = rel[-1]
        call = calls.get(r["job"], {})
        intended = json.loads((ROOT / "briefs/intended" / f"{d.name}.json").read_text(encoding="utf-8"))
        spec = Spec.model_validate(intended.get("spec", intended))
        h_stated = float(grid_for(spec)[0])
        h = float(r["h"])
        warm = bool(r.get("warm_used"))
        sched = ("reopen_long" if call.get("args", {}).get("optimize") else "tail") if warm else "cold"
        rows.append({"task": d.name, "released": True, "release_h": h, "stated_h": h_stated,
                     "refinement_factor": h_stated / h, "filter_rmin_elements": RMIN, "filter_length": RMIN * h,
                     "filter_length_at_stated_h": RMIN * h_stated, "schedule": sched,
                     "warm_from_h": 2 * h if warm else None, "iterations_on_grid": r["iterations"],
                     "n_elem": r["n_elem"], "units": spec.units.length})
    out = {"source": "results/X3/*/session/session.jsonl; briefs/intended/*.json", "rows": rows,
           "released": sum(r["released"] for r in rows),
           "refinement_factors": sorted({r["refinement_factor"] for r in rows if r["released"]}),
           "schedules": {s: sum(r.get("schedule") == s for r in rows) for s in ("cold", "tail", "reopen_long")}}
    (X3 / "regularization.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for r in rows:
        print(r["task"], r.get("refinement_factor"), r.get("schedule"), r.get("iterations_on_grid"))
    print(out["released"], out["refinement_factors"], out["schedules"])


if __name__ == "__main__":
    main()
