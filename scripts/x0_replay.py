"""X0 replay test: a scripted session through the tool contract, then a replay of its log with no LLM.

For each task: load the intended spec of a draft brief, edit it through the contract (as the GUI or agent would),
validate, render, solve the sketch rung cold and the preview rung warm, read field_stats, then replay the log in a
fresh session. Pass = every call reproduces its ok flag and spec hash, every render its PNG hash, and every solve
its status, iteration count, certificate and compliance bit for bit.

    python -B scripts/x0_replay.py [b01_cantilever_block b06_motor_mount ...]
Writes results/X0/replay.json and keeps the session logs in results/X0/replay_sessions/.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(slugs):
    from agentto.tools import Session, replay
    out_dir = ROOT / "results" / "X0" / "replay_sessions"
    rows = []
    for slug in slugs:
        spec = json.loads((ROOT / "briefs" / "intended" / f"{slug}.json").read_text(encoding="utf-8"))["spec"]
        root = out_dir / slug
        shutil.rmtree(root, ignore_errors=True)
        s = Session(root)
        t0 = time.perf_counter()
        calls = [("set_spec", {"spec": spec}, "user"), ("validate_spec", {}, "agent"), ("render", {}, "agent"),
                 ("set_params", {"volume_fraction": round(spec["volume_fraction"] + 0.05, 4)}, "user"),
                 ("undo", {}, "user"), ("list_regions", {}, "agent"), ("ladder", {}, "agent"),
                 ("solve", {"rung": "sketch"}, "user"), ("solve", {"rung": "preview"}, "user"),
                 ("field_stats", {}, "agent")]
        summary = []
        for name, args, actor in calls:
            r = s.call(name, args, actor)
            res = r.get("result") or {}
            summary.append({"tool": name, "ok": r["ok"], "error": r.get("error"),
                            **({k: res.get(k) for k in ("status", "iterations", "compliance_phys", "certified",
                                                         "warm_used", "compute_s")} if name == "solve" else {}),
                            **({k: res.get(k) for k in ("solid_components", "load_connected_to_support",
                                                         "material_fraction")} if name == "field_stats" else {})})
            print(slug, name, r["ok"], r.get("error") or "", {k: v for k, v in summary[-1].items()
                                                               if k not in ("tool", "ok", "error")})
        s.runner.close()
        t_run = time.perf_counter() - t0
        t0 = time.perf_counter()
        rep = replay(root / "session.jsonl", root=root / "replay")
        t_rep = time.perf_counter() - t0
        print(slug, "replay", rep["ok"], rep["mismatches"])
        rows.append({"task": slug, "session_s": round(t_run, 2), "replay_s": round(t_rep, 2), "calls": summary,
                     "replay": rep})
    res = {"study": "X0-replay", "date": time.strftime("%Y-%m-%d"), "pass": all(r["replay"]["ok"] for r in rows),
           "tasks": rows}
    (ROOT / "results" / "X0" / "replay.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("PASS" if res["pass"] else "FAIL")


if __name__ == "__main__":
    main(sys.argv[1:] or ["b01_cantilever_block", "b06_motor_mount"])
