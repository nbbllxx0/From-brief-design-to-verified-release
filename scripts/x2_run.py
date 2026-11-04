"""X2 harness: brief -> spec under one condition, on the subscription Codex CLI only (llm/cli.py refuses API keys).

Conditions
    T    text only: the brief and the JSON Schema; one JSON spec back, no tools.
    G    agent with the agentto tools over MCP, profile "spec" (editing, list_regions, validate_spec, explain_spec,
         render); no clarification.
    GNR  ablation of G: the same prompt, tools and provider, but validate_spec checks the schema only and returns
         no rule findings (mcp_server --no-rules); grounding by list_regions and render is unchanged.
    GC  G plus ask_clarification (profile "clarify"), answered by the scripted oracle that holds the intended
         reading (bench.Oracle; matches readings on the solver's grid, never an LLM).
    GP   G with typed value provenance (profile "spec_prov"): the agent tags every spec item stated / derived /
         assumed with set_source; validate_spec adds the advisory R014 listing items tagged assumed or untagged.
         Everything else as G (assume, not ask). The first condition that varies the representation.
    GPC  GC with the same provenance tags (profile "clarify_prov"); ask_clarification is accepted only about items
         the agent tagged assumed, and the scripted oracle answers as in GC.
    TS   T plus schema feedback: when the reply is not a schema-valid spec (or a refusal), a new call gets the
         previous reply and the validator's errors, at most twice. No grounding tools.
    GCV  GC plus the confirm-the-picture step: the GC run's final spec is rendered (iso, front, top) and a fresh
         Codex call that sees only the brief and the pictures says whether they match; if it lists mismatches, one
         repair session (seeded with that spec, oracle still available) gets the list. Paired with GC: it reuses
         the GC run of the same task and repeat, so V's effect is isolated.
Each run is scored against briefs/intended/<task>.json on what the solver sees (bench.match_task: the intended spec or an admissible reading that counts) and on the
expected behaviour (solve / clarify / refuse). The consequence-weighted error needs solves and is computed later
by scripts/x2_score.py on the GPU.

    python -B scripts/x2_run.py --cond G [--briefs b01 b06] [--rep 1] [--force] [--provider codex|claude]
Writes results/X2/<cond>/<task>_r<rep>.json (Claude Code: results/X2/claude/<cond>/...), with the session log for
G/GC beside it; finished runs are skipped.
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentto.bench import match_task  # noqa: E402
from agentto.llm.cli import codex_mcp_config, mcp_server_spec, run_claude, run_codex  # noqa: E402
from agentto.rules import check  # noqa: E402
from agentto.spec import Spec  # noqa: E402
from agentto.tools import Session, read_log, replay  # noqa: E402

FRAME = """Region coordinates are absolute in the domain frame, origin at the domain's min corner. A face patch gives
its centre and full size in the face's two in-plane axes in increasing axis order: (y, z) on x faces, (x, z) on y faces,
(x, y) on z faces. Loads are total forces on the region."""

PROMPT_T = """You translate an engineering design brief into a topology-optimization spec.
Return ONLY one JSON object that validates against this JSON Schema (agentto.spec/v0). No prose, no code fences.
{frame}
If the brief cannot be solved as stated, return {{"refuse": "<one sentence why>"}} instead.

JSON Schema:
{schema}

Brief:
{brief}"""

PROMPT_G = """You set up a 3D topology-optimization problem from an engineering brief, using the agentto tools only.
Build the complete spec with the tools, check it with list_regions and validate_spec and fix any errors, and look at
render once to confirm the supports and loads are where the brief says. {clarify}
If the brief cannot be solved as stated (for example the part cannot be held still by the given supports), say so:
finish with REFUSE and one sentence why. Otherwise finish with one line: DONE, then any assumptions you made.

Brief:
{brief}"""
CLARIFY = {"G": "Do not ask questions: where the brief is silent, choose a reasonable value and state it.",
           "GC": "Where the brief is ambiguous or silent on something that changes the design, call ask_clarification "
                 "with 2-4 concrete readings (the full item or parameter values each would give); the user answers. "
                 "Do not ask about things the brief states."}
CLARIFY["GNR"] = CLARIFY["G"]
PROVENANCE = ("For every spec item (units, domain, material, volume_fraction, resolution, and each support, load and "
              "passive region) record where its value comes from with set_source: stated (the brief gives it), derived "
              "(computed from numbers the brief gives), or assumed (your own choice where the brief is silent or "
              "ambiguous). Call list_sources before you finish; every item must be tagged.")
CLARIFY["GP"] = CLARIFY["G"] + " " + PROVENANCE
CLARIFY["GPC"] = (PROVENANCE + " Where an item you tagged assumed changes the design, you may call ask_clarification "
                  "with 2-4 concrete readings (the full item or parameter values each would give) and its key in about; "
                  "questions about items tagged stated or derived are rejected. After an answer, apply it and tag the "
                  "item answered.")
PROFILE = {"G": "spec", "GNR": "spec", "GP": "spec_prov", "GC": "clarify", "GPC": "clarify_prov"}
ORACLE_CONDS = ("GC", "GPC")
RETRY_TS = """Your previous reply was not a valid spec. The validator reported:
{errors}

Previous reply:
{previous}

Return ONLY the corrected JSON object (or the refusal object), nothing else."""


def tasks(sel):
    out = sorted(p.stem for p in (ROOT / "briefs" / "intended").glob("*.json"))
    return [t for t in out if not sel or any(t.startswith(s) for s in sel)]


def final_session(log_path: Path) -> Session:
    s = Session(None, profile="full")
    for e in read_log(log_path):
        if e.get("type") == "call":
            s.call(e["tool"], e["args"], actor=e["actor"])
    return s


def final_spec(log_path: Path):
    return final_session(log_path).draft


def parse_t(text: str):
    """(spec, refused, error) from a text-only reply."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, False, "no JSON object in the reply"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as ex:
        return None, False, f"invalid JSON: {ex}"
    if "refuse" in obj and len(obj) == 1:
        return None, True, None
    try:
        Spec.model_validate(obj)
    except Exception as ex:                              # noqa: BLE001
        return obj, False, str(ex)[:1500]
    return obj, False, None


def score(spec: dict | None, intended: dict, refused: bool) -> dict:
    exp = intended["expected"]
    out = {"expected": exp, "refused": refused}
    if spec is not None:
        try:
            Spec.model_validate(spec)
            out["schema_valid"] = True
        except Exception as ex:                          # noqa: BLE001
            out.update(schema_valid=False, schema_error=str(ex)[:400])
    else:
        out["schema_valid"] = False
    if out["schema_valid"]:
        out["rules_ok"] = check(Spec.model_validate(spec)).ok
        try:
            out["match"] = match_task(spec, intended)
        except Exception as ex:                          # noqa: BLE001
            out["match"] = {"same": False, "error": repr(ex)[:300]}
    same = bool(out.get("match", {}).get("same"))
    out["correct_spec"] = same
    out["behaviour_ok"] = refused if exp == "refuse" else (same and not refused)
    return out


VERIFY = """You check a topology-optimization setup against its engineering brief. The pictures show the design
domain (box), supports (blue, with the fixed directions), loads (red arrows, labelled with force) and keep-in (green
solid) / keep-out (dashed) regions, from three views. Compare them with the brief: is every support, load and
keep-in/keep-out region where the brief says, with the right direction and size? Ignore colours and styling.
Return JSON: {{"match": true|false, "issues": ["<one concrete mismatch per item>", ...]}}.

Brief:
{brief}"""
VERIFY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["match", "issues"],
                 "properties": {"match": {"type": "boolean"}, "issues": {"type": "array", "items": {"type": "string"}}}}
REPAIR = """You set up a 3D topology-optimization problem from an engineering brief, using the agentto tools only. The
session already holds a spec built from the brief. A reviewer compared pictures of it with the brief and reported:
{issues}
Check each point against the brief with get_spec and list_regions. Fix the spec where the reviewer is right (keep
what is already correct), validate it, and where the brief is ambiguous call ask_clarification. Finish with one line:
DONE, then what you changed.

Brief:
{brief}"""


def run_gcv(task: str, rep: int, intended: dict, brief: str, base: Path) -> dict:
    from agentto.render import render_spec
    gc = run_one("GC", task, rep, False)
    rec = {"gc_correct": gc.get("correct_spec"), "gc_usage": gc.get("usage"), "gc_wall_s": gc.get("wall_s")}
    spec = gc.get("spec")
    if spec is None:
        rec.update(spec=None, verify=None, repaired=False, usage=gc.get("usage", {}), wall_s=gc.get("wall_s", 0),
                   questions=gc.get("questions", 0), n_tool_calls=gc.get("n_tool_calls", 0),
                   n_failed_calls=gc.get("n_failed_calls", 0))
        return rec
    base.mkdir(parents=True, exist_ok=True)
    imgs = []
    for v in ("iso", "front", "top"):
        p = base / f"render_{v}.png"
        p.write_bytes(render_spec(spec, view=v))
        imgs.append(p)
    sch = base / "verify_schema.json"
    sch.write_text(json.dumps(VERIFY_SCHEMA))
    if PROVIDER["name"] == "claude":
        rv = run_claude(VERIFY.format(brief=brief) + "\nReturn only the JSON object.", base / "verify", images=imgs,
                        effort=PROVIDER.get("effort"))
    else:
        rv = run_codex(VERIFY.format(brief=brief), base / "verify", output_schema=sch, images=imgs)
    try:
        ver = json.loads(re.search(r"\{.*\}", rv.final, re.S).group(0))
    except Exception:                                    # noqa: BLE001
        ver = {"match": True, "issues": [], "unparsed": rv.final[-500:]}
    rec.update(verify=ver, verify_usage=rv.usage, verify_wall_s=rv.wall_s)
    usage = {k: gc["usage"].get(k, 0) + rv.usage.get(k, 0) for k in set(gc["usage"]) | set(rv.usage)}
    wall = gc["wall_s"] + rv.wall_s
    if ver.get("match") or not ver.get("issues"):
        rec.update(spec=spec, repaired=False, usage=usage, wall_s=wall, questions=gc.get("questions", 0),
                   n_tool_calls=gc.get("n_tool_calls", 0), n_failed_calls=gc.get("n_failed_calls", 0))
        return rec
    seed = base / "seed.json"
    seed.write_text(json.dumps(spec))
    sess = base / "session"
    rr = llm(REPAIR.format(issues="\n".join(f"- {x}" for x in ver["issues"]), brief=brief), base / "work",
             dict(session_dir=sess, profile="clarify", seed=seed,
                  oracle=ROOT / "briefs" / "intended" / f"{task}.json"))
    log = sess / "session.jsonl"
    recs = read_log(log) if log.exists() else []
    calls = [e for e in recs if e.get("type") == "call"]
    answers = [e for e in recs if e.get("type") == "answer"]
    usage = {k: usage.get(k, 0) + rr.usage.get(k, 0) for k in set(usage) | set(rr.usage)}
    rec.update(spec=final_spec(log) if calls else spec, repaired=True, repair_usage=rr.usage,
               repair_final=rr.final[-1500:], usage=usage, wall_s=wall + rr.wall_s,
               questions=gc.get("questions", 0) + len(answers),
               n_tool_calls=gc.get("n_tool_calls", 0) + len(calls),
               n_failed_calls=gc.get("n_failed_calls", 0) + sum(not e["ok"] for e in calls),
               session_log=str(log.relative_to(ROOT)))
    return rec


PROVIDER = {"name": "codex", "effort": None}   # effort: Claude Code only; results go to X2/claude_<effort>/


def llm(prompt, workdir, mcp_kw=None):
    """One session on the chosen subscription CLI; mcp_kw are mcp_server_spec / codex_mcp_config arguments."""
    if PROVIDER["name"] == "claude":
        return run_claude(prompt, workdir, mcp=mcp_server_spec(**mcp_kw) if mcp_kw else None,
                          effort=PROVIDER.get("effort"))
    return run_codex(prompt, workdir, extra_config=codex_mcp_config(**mcp_kw) if mcp_kw else None)


def run_one(cond: str, task: str, rep: int, force: bool):
    sub = PROVIDER["name"] + (f"_{PROVIDER['effort']}" if PROVIDER.get("effort") else "")
    out_dir = ROOT / "results" / "X2" / (cond if PROVIDER["name"] == "codex" else f"{sub}/{cond}")
    dst = out_dir / f"{task}_r{rep}.json"
    if dst.exists() and not force:
        return json.loads(dst.read_text())
    brief = (ROOT / "briefs" / f"{task}.md").read_text(encoding="utf-8")
    intended = json.loads((ROOT / "briefs" / "intended" / f"{task}.json").read_text(encoding="utf-8"))
    base = out_dir / f"{task}_r{rep}"
    shutil.rmtree(base, ignore_errors=True)
    t0 = time.perf_counter()
    rec = {"task": task, "cond": cond, "rep": rep, "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if cond == "GCV":
        g = run_gcv(task, rep, intended, brief, base)
        spec, text = g.pop("spec"), ""
        refused = bool(re.search(r"(^|\n)\s*REFUSE", json.loads(
            (out_dir.parent / "GC" / f"{task}_r{rep}.json").read_text()).get("final", "")))
        rec.update(g)
        rec.update(provider=PROVIDER["name"], model=None, effort=PROVIDER.get("effort"), cli_version=None, returncode=0,
                   final=g.get("repair_final", ""), spec=spec, stderr_tail="")
        rec.update(score(spec, intended, refused))
        rec["asked_when_expected"] = (rec.get("questions", 0) > 0) if intended["expected"] == "clarify" else None
        rec["harness_s"] = time.perf_counter() - t0
        out_dir.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(rec, indent=1))
        return rec
    if cond == "T":
        schema = (ROOT / "schemas" / "spec_v0.schema.json").read_text()
        r = llm(PROMPT_T.format(frame=FRAME, schema=schema, brief=brief), base / "work")
        text = r.final.strip()
        spec, refused = None, False
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                refused = "refuse" in obj and len(obj) == 1
                spec = None if refused else obj
            except json.JSONDecodeError:
                pass
        rec.update(n_tool_calls=0, n_failed_calls=0, questions=0)
    elif cond == "TS":
        schema = (ROOT / "schemas" / "spec_v0.schema.json").read_text()
        prompt = PROMPT_T.format(frame=FRAME, schema=schema, brief=brief)
        r = llm(prompt, base / "work")
        usage, wall, tries = dict(r.usage), r.wall_s, []
        text = r.final.strip()
        spec, refused, err = parse_t(text)
        for k in range(2):                               # at most two retries with the validator's errors
            if err is None:
                break
            tries.append({"error": err[:600]})
            r = llm(prompt + "\n\n" + RETRY_TS.format(errors=err, previous=text[-6000:]), base / f"retry{k + 1}")
            usage = {x: usage.get(x, 0) + r.usage.get(x, 0) for x in set(usage) | set(r.usage)}
            wall += r.wall_s
            text = r.final.strip()
            spec, refused, err = parse_t(text)
        r.usage, r.wall_s = usage, wall
        rec.update(n_tool_calls=0, n_failed_calls=0, questions=0, retries=len(tries), retry_errors=tries)
    else:
        sess = base / "session"
        kw = dict(session_dir=sess, profile=PROFILE[cond],
                  oracle=(ROOT / "briefs" / "intended" / f"{task}.json") if cond in ORACLE_CONDS else None)
        if cond == "GNR":
            kw["no_rules"] = True
        prompt = PROMPT_G.format(clarify=CLARIFY[cond], brief=brief)
        r = llm(prompt, base / "work", kw)
        text = r.final.strip()
        log = sess / "session.jsonl"
        recs = read_log(log) if log.exists() else []
        calls = [e for e in recs if e.get("type") == "call"]
        answers = [e["answer"] for e in recs if e.get("type") == "answer"]
        fs = final_session(log) if calls else None
        spec = fs.draft if fs else None
        if cond in ("GP", "GPC") and fs is not None and spec is not None:
            try:
                rec["sources"] = fs._source_table()
            except Exception as ex:                      # noqa: BLE001 -- an incomplete draft
                rec["sources"] = {"error": repr(ex)[:300]}
        refused = bool(re.search(r"(^|\n)\s*REFUSE", text))
        rep_ = replay(log, solve=False) if calls else None
        rec.update(n_tool_calls=len(calls), n_failed_calls=sum(not e["ok"] for e in calls),
                   tools_used=[e["tool"] for e in calls], questions=len(answers),
                   oracle_hits=sum(a.get("choice") is not None for a in answers),
                   replay_ok=rep_["ok"] if rep_ else None, session_log=str(log.relative_to(ROOT)))
    rec.update(provider=r.provider, model=r.model, effort=r.effort, cli_version=r.cli_version, wall_s=r.wall_s,
               usage=r.usage, returncode=r.returncode, final=text[-3000:], spec=spec,
               stderr_tail=r.stderr[-800:] if r.returncode else "")
    rec.update(score(spec, intended, refused))
    rec["asked_when_expected"] = (rec.get("questions", 0) > 0) if intended["expected"] == "clarify" else None
    rec["harness_s"] = time.perf_counter() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(rec, indent=1))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=["T", "TS", "G", "GNR", "GP", "GC", "GPC", "GCV"])
    ap.add_argument("--briefs", nargs="*")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--provider", default="codex", choices=["codex", "claude"])
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"],
                    help="Claude Code effort (Codex runs at DEFAULT_CODEX_EFFORT)")
    a = ap.parse_args()
    PROVIDER["name"] = a.provider
    PROVIDER["effort"] = a.effort if a.provider == "claude" else None

    for t in tasks(a.briefs):
        r = run_one(a.cond, t, a.rep, a.force)
        u = r.get("usage", {})
        print(f"{a.cond} {t:22s} correct {str(r['correct_spec']):5s} behaviour {str(r['behaviour_ok']):5s} "
              f"q {r.get('questions', 0)} calls {r.get('n_tool_calls', 0)} fail {r.get('n_failed_calls', 0)} "
              f"in {u.get('input_tokens', 0) / 1e3:.0f}k ({u.get('cached_input_tokens', 0) / 1e3:.0f}k cached) "
              f"{r.get('wall_s', 0):.0f}s", flush=True)


if __name__ == "__main__":
    main()
