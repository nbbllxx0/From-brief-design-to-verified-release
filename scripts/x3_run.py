"""X3: end-to-end sessions. An agent takes a brief to a certified release and its STL + report, alone.

Codex (subscription CLI) gets the brief and the agentto MCP server in a stand-alone session with profile "full" and
the scripted oracle (clarification answered from the intended reading). The session runs its own GPU worker, so run
this with nothing else on the GPU. From the session log: wall time per stage (spec, each rung, export), GPU compute,
tool calls and failures, questions, tokens; whether the released spec is the intended one; whether the release is
certified; and a full replay of the log without the LLM (solves bit-exact).

    python -B scripts/x3_run.py b01_cantilever_block [b06_motor_mount ...] [--provider codex|claude]
Writes results/X3/<task>.json and keeps the session (log, fields, exports) in results/X3/<task>/.
"""
import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentto.bench import match_task  # noqa: E402
from agentto.llm.cli import codex_mcp_config, mcp_server_spec, run_claude, run_codex  # noqa: E402
from agentto.tools import read_log, replay  # noqa: E402

PROMPT = """You are an engineer's design assistant with the agentto topology-optimization tools. Take this brief all
the way to a released design, on your own:
1. Build the spec with the tools; ground it with list_regions, check it with validate_spec, and look at render once.
   Where the brief is ambiguous in a way that changes the design, call ask_clarification with 2-4 concrete readings.
2. Solve the ladder in order: sketch, preview, refine, release (solve with rung=...). After the preview, check with
   field_stats that every load is connected to a support; fix the spec and start again if not.
3. Release only if the release solve is certified. Then export_stl and export_report for the release job.
Finish with one line: DONE, the release job id, and its compliance; or STOPPED and why.

Brief:
{brief}"""


def stage_times(recs):
    """Wall-clock stages from the log: spec (first call to the first solve), each solve (call to result), export."""
    t = lambda r: datetime.fromisoformat(r["t"]).timestamp()          # noqa: E731
    calls = [r for r in recs if r.get("type") == "call"]
    results = {r["seq"]: r for r in recs if r.get("type") == "result"}
    out, solves = {}, [c for c in calls if c["tool"] == "solve" and c["ok"]]
    if calls:
        out["spec_s"] = (t(solves[0]) if solves else t(calls[-1])) - t(calls[0])
    for c in solves:
        r = results.get(c["seq"])
        if r:
            key = f"solve_h{c['result'].get('h')}"
            out[key] = {"wall_s": t(r) - t(c), "compute_s": r["result"].get("compute_s"),
                        "iterations": r["result"].get("iterations"), "n_elem": r["result"].get("n_elem"),
                        "warm": r["result"].get("warm_used"), "status": r["result"].get("status"),
                        "certified": r["result"].get("certified")}
    ex = [c for c in calls if c["tool"].startswith("export_")]
    if ex:
        out["export_s"] = sum(c.get("dt_s", 0) for c in ex)
    if calls:
        out["total_s"] = t(recs[-1]) - t(calls[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="+")
    ap.add_argument("--provider", default="codex", choices=["codex", "claude"])
    a = ap.parse_args()
    for task in a.tasks:
        brief = (ROOT / "briefs" / f"{task}.md").read_text(encoding="utf-8")
        intended = json.loads((ROOT / "briefs" / "intended" / f"{task}.json").read_text(encoding="utf-8"))
        base = ROOT / "results" / "X3" / (task if a.provider == "codex" else f"claude_{task}")
        shutil.rmtree(base, ignore_errors=True)
        sess = base / "session"
        kw = dict(session_dir=sess, profile="full", oracle=ROOT / "briefs" / "intended" / f"{task}.json")
        t0 = time.perf_counter()
        if a.provider == "claude":
            r = run_claude(PROMPT.format(brief=brief), base / "work", mcp=mcp_server_spec(**kw), timeout=3600)
        else:
            r = run_codex(PROMPT.format(brief=brief), base / "work", extra_config=codex_mcp_config(**kw, tool_timeout_s=1800),
                          timeout=3600)
        wall = time.perf_counter() - t0
        recs = read_log(sess / "session.jsonl") if (sess / "session.jsonl").exists() else []
        calls = [x for x in recs if x.get("type") == "call"]
        rel = [x["result"] for x in recs if x.get("type") == "result" and x["result"].get("release_rung")]
        out = {"task": task, "provider": r.provider, "model": r.model, "wall_s": wall, "usage": r.usage,
               "returncode": r.returncode, "final": r.final[-2000:], "n_tool_calls": len(calls),
               "n_failed_calls": sum(not x["ok"] for x in calls), "tools_used": [x["tool"] for x in calls],
               "questions": sum(1 for x in recs if x.get("type") == "answer"), "stages": stage_times(recs),
               "released": bool(rel and rel[-1].get("certified")),
               "exports": [x["result"] for x in calls if x["tool"].startswith("export_") and x["ok"]]}
        if rel:
            job_spec = next((c["args"] for c in calls if c["tool"] == "solve"), None)
            spec_at = [x for x in recs if x.get("type") == "call" and x["tool"] == "solve" and x["ok"]
                       and x["result"].get("job") == rel[-1]["job"]]
            if spec_at:
                from agentto.tools import Session
                s = Session(None, profile="full")
                for c in calls:
                    if c["seq"] > spec_at[0]["seq"]:
                        break
                    if c["tool"] not in ("solve", "status", "cancel", "field_stats", "export_stl", "export_report",
                                         "ask_clarification"):
                        s.call(c["tool"], c["args"], actor=c["actor"])
                out["release_spec_match"] = match_task(s.draft, intended)
            del job_spec
        out["replay"] = replay(sess / "session.jsonl", root=base / "replay") if calls else None
        (base.parent / f"{base.name}.json").write_text(json.dumps(out, indent=1, default=str))
        print(task, "released", out["released"], "match", (out.get("release_spec_match") or {}).get("same"),
              "calls", out["n_tool_calls"], f"{wall:.0f}s", "replay", (out["replay"] or {}).get("ok"), flush=True)


if __name__ == "__main__":
    main()
