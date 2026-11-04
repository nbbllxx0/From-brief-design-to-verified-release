"""Week-3 probe: one grounded (condition G) Codex session over MCP, to check the tool contract end to end and
measure tokens and tool calls.

The agent gets the brief and the agentto MCP server (profile "spec": editing, list_regions, validate_spec,
explain_spec, render); the spec it leaves in the session is compared with the intended reading on the solver's own
terms (bench.equivalent). Subscription CLI only (llm/cli.py refuses API keys).

    python -B scripts/codex_mcp_probe.py [b01_cantilever_block]
Writes results/probe/codex_G_<brief>.json and keeps the session log beside it.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentto.bench import equivalent  # noqa: E402
from agentto.llm.cli import codex_mcp_config, run_codex  # noqa: E402
from agentto.tools import read_log  # noqa: E402

PROMPT = """You set up a 3D topology-optimization problem from an engineering brief, using the agentto tools only.
Build the complete spec with the tools, check it with list_regions and validate_spec and fix any errors, and look
at render once to confirm the supports and loads are where the brief says. Do not ask questions: where the brief is
silent, choose a reasonable value. Finish with one line: DONE, then any assumptions you made.

Brief:
{brief}"""


def main(slug="b01_cantilever_block"):
    brief = (ROOT / "briefs" / f"{slug}.md").read_text(encoding="utf-8")
    intended = json.loads((ROOT / "briefs" / "intended" / f"{slug}.json").read_text(encoding="utf-8"))
    base = ROOT / "results" / "probe" / f"codex_G_{slug}"
    shutil.rmtree(base, ignore_errors=True)
    sess, work = base / "session", base / "work"
    r = run_codex(PROMPT.format(brief=brief), work, extra_config=codex_mcp_config(sess, "spec"))
    log = read_log(sess / "session.jsonl") if (sess / "session.jsonl").exists() else []
    calls = [e for e in log if e.get("type") == "call"]
    from agentto.tools import replay
    rep = replay(sess / "session.jsonl", solve=False) if calls else None
    out = {"brief": slug, "condition": "G", "provider": r.provider, "model": r.model, "effort": r.effort,
           "cli_version": r.cli_version, "wall_s": r.wall_s, "usage": r.usage, "returncode": r.returncode,
           "final": r.final.strip(), "n_tool_calls": len(calls), "n_failed_calls": sum(not e["ok"] for e in calls),
           "tools_used": [e["tool"] for e in calls], "replay_ok": rep["ok"] if rep else None,
           "stderr_tail": r.stderr[-1500:]}
    # final spec = state after replaying the log (identical to the live session by the replay check)
    if calls:
        from agentto.tools import Session
        s = Session(None, profile="full")
        for e in calls:
            s.call(e["tool"], e["args"])
        final = s.draft
        out["spec"] = final
        try:
            out["match"] = equivalent(final, intended["spec"])
        except Exception as ex:                  # noqa: BLE001 -- an incomplete spec is a result
            out["match"] = {"same": False, "error": repr(ex)[:500]}
    (ROOT / "results" / "probe" / f"codex_G_{slug}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out.get(k) for k in ("wall_s", "usage", "n_tool_calls", "n_failed_calls", "tools_used",
                                              "replay_ok", "match", "final")}, indent=1)[:4000])


if __name__ == "__main__":
    main(*sys.argv[1:])
