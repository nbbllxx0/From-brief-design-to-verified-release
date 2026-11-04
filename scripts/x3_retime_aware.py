"""Re-time the X3 sessions under the pad-aware coarse rule (results/solver/NOTES.md), without any LLM.

X3 ran with voxfem's coarse rule. Each session log is replayed with every solve switched to `coarse_pad="aware"`;
the agent's own time (LLM turns, tool calls) is taken from the original record, so
    session time (aware) = X3 total - X3 solve wall + replayed solve wall.
Compliance is not expected to be bit-identical across rules (b19: -6.5e-8 relative); the relative change and the
certificate are recorded.

    python -B scripts/x3_retime_aware.py [b01 ...]        (GPU, one job at a time)
Writes results/X3_aware/<task>.json and results/X3_aware/summary.md.
"""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentto.tools import read_log, replay  # noqa: E402

OUT = ROOT / "results" / "X3_aware"


def tasks(sel):
    out = sorted(f.stem for f in (ROOT / "results" / "X3").glob("b*.json") if "_run" not in f.stem)
    return [t for t in out if not sel or any(t.startswith(s) for s in sel)]


def run(task):
    x3 = json.loads((ROOT / "results" / "X3" / f"{task}.json").read_text(encoding="utf-8"))
    log = ROOT / "results" / "X3" / task / "session" / "session.jsonl"
    recs = read_log(log)
    orig = {}
    for r in recs:
        if r.get("type") == "result" and r["result"].get("status") in ("converged", "not_converged"):
            r["result"]["coarse_pad"] = "aware"
            orig[r["seq"]] = r["result"]
    for r in recs:                                 # solves whose result rides on the call record
        if r.get("type") == "call" and r["tool"] == "solve" and r["seq"] not in orig and \
                (r.get("result") or {}).get("status") in ("converged", "not_converged"):
            r["result"]["coarse_pad"] = "aware"
            orig[r["seq"]] = r["result"]
    base = OUT / task
    t = time.perf_counter()
    rep = replay(recs, root=base)
    wall = time.perf_counter() - t
    new = [r["result"] for r in read_log(base / "session.jsonl") if r.get("type") == "result"
           and r["result"].get("status") in ("converged", "not_converged")]
    old = [orig[k] for k in sorted(orig)]
    solves = []
    for o, n in zip(old, new):
        solves.append({"h": o.get("h"), "n_elem": o.get("n_elem"), "status": [o.get("status"), n.get("status")],
                       "iterations": [o.get("iterations"), n.get("iterations")],
                       "compute_s": [o.get("compute_s"), n.get("compute_s")],
                       "wall_s": [o.get("wall_s"), n.get("wall_s")],
                       "certified": [o.get("certified"), n.get("certified")],
                       "rel_dC": None if not (o.get("compliance_phys") and n.get("compliance_phys")) else
                       n["compliance_phys"] / o["compliance_phys"] - 1})
    solve_old = sum(s["wall_s"][0] or 0 for s in solves)
    solve_new = sum(s["wall_s"][1] or 0 for s in solves)
    total = x3["stages"]["total_s"] if "stages" in x3 else x3.get("wall_s")
    out = {"task": task, "n_solves": len(solves), "replay_calls_ok": not [m for m in rep["mismatches"]
                                                                       if m["field"] in ("ok", "spec_hash")],
           "x3_total_s": total, "solve_wall_voxfem_s": solve_old, "solve_wall_aware_s": solve_new,
           "total_aware_s": None if total is None else total - solve_old + solve_new,
           "release": solves[-1] if solves else None, "solves": solves, "replay_wall_s": wall,
           "mismatches": rep["mismatches"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{task}.json").write_text(json.dumps(out, indent=1))
    print(task, f"solves {solve_old:.0f} -> {solve_new:.0f} s; session {total:.0f} -> {out['total_aware_s']:.0f} s; "
          f"release certified {solves[-1]['certified'] if solves else None}", flush=True)
    return out


def summarize():
    rows = [json.loads(f.read_text()) for f in sorted(OUT.glob("b*.json"))]
    med = statistics.median
    lines = ["# X3 sessions re-timed under the pad-aware coarse rule (replay, no LLM)", "",
             "| Task | Release elements | Solve wall, voxfem rule → aware (s) | Session, X3 → aware (s) | "
             "Release compute (s) | Release ΔC | Certified |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        rel = r["release"] or {}
        lines.append(f"| {r['task']} | {(rel.get('n_elem') or 0) / 1e6:.1f}M | {r['solve_wall_voxfem_s']:.0f} → "
                     f"{r['solve_wall_aware_s']:.0f} | {r['x3_total_s']:.0f} → {r['total_aware_s']:.0f} | "
                     f"{(rel.get('compute_s') or [0, 0])[0]:.0f} → {(rel.get('compute_s') or [0, 0])[1]:.0f} | "
                     f"{'–' if rel.get('rel_dC') is None else format(rel['rel_dC'], '+.1e')} | "
                     f"{(rel.get('certified') or [None, None])[1]} |")
    rows = [r for r in rows if r["n_solves"]]          # the refused brief has no solve to re-time
    if rows:
        lines += ["", f"Median session: {med(r['x3_total_s'] for r in rows):.0f} s (X3) → "
                  f"{med(r['total_aware_s'] for r in rows):.0f} s (aware); range "
                  f"{min(r['total_aware_s'] for r in rows):.0f}–{max(r['total_aware_s'] for r in rows):.0f} s ({len(rows)} released "
                  "sessions). Unpadded grids are rule-independent, so their changes are run-to-run timing noise."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8")
    if "--summary" not in argv:
        for t in tasks(argv):
            if (OUT / f"{t}.json").exists():
                continue
            run(t)
    summarize()


if __name__ == "__main__":
    main([a for a in sys.argv[1:]])
