"""X2 representation and schema-feedback arms, first repeat, scored like every other condition against the
adjudicated references. No LLM calls, no solves.

    GP  vs G   does typed value provenance change outcomes, and do the agent's own "assumed" tags single out wrong
               readings? (compare: post hoc value-source tag, free-text assumption notes)
    GPC vs GC  does asking only about items tagged assumed keep right outcomes with fewer questions on stated briefs?
    TS  vs T   is the T-to-G gain just output validity? (schema feedback, at most two retries, no grounding tools)
Optional cells are included when present: repeat 2 of GP on the 20-brief repeat subset, and Claude Code GPC
(paired with Claude Code GC; the Claude runs have no G condition to pair a GP arm with).

Output: results/X2/provenance_summary.json
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_tables import mcnemar, silent, wilson  # noqa: E402

X2 = ROOT / "results" / "X2"
DESIGN = ("domain", "material", "volume_fraction", "support:", "load:", "passive:")
FIELD_OF = {"support": "supports", "load": "loads", "passive": "passive"}


def load(sub, rep=1):
    return {d["task"]: d for d in (json.loads(f.read_text(encoding="utf-8"))
                                   for f in sorted((X2 / sub).glob(f"b*_r{rep}.json")))}


def counts(runs):
    n = len(runs)
    right = sum(bool(d.get("behaviour_ok")) for d in runs.values())
    return {"n": n, "right": right, "right_ci95": wilson(right, n) if n else None,
            "delivered_wrong": sorted(t for t, d in runs.items() if silent(d)),
            "schema_invalid": sorted(t for t, d in runs.items() if not d.get("schema_valid") and not d.get("refused")),
            "questions": sum(d.get("questions", 0) for d in runs.values()),
            "input_tokens_median": statistics.median(d.get("usage", {}).get("input_tokens", 0)
                                                     for d in runs.values()) if n else None}


def paired(a, b):
    common = sorted(set(a) & set(b))
    fixed = [t for t in common if b[t].get("behaviour_ok") and not a[t].get("behaviour_ok")]
    broken = [t for t in common if a[t].get("behaviour_ok") and not b[t].get("behaviour_ok")]
    return {"n": len(common), "corrected": fixed, "broken": broken, "p_mcnemar": mcnemar(len(fixed), len(broken))}


def assumed_keys(d, material=True):
    """Items tagged assumed that change the stated problem. material=False drops the material: agents often tag a
    modulus taken from an alloy name as assumed, and for one linear material it scales compliance, not topology."""
    s = d.get("sources") or {}
    return [k for k in s.get("assumed", []) if k.startswith(DESIGN) and (material or k != "material")]


def flags(runs, material=True):
    """Delivered (schema-valid, not refused) specifications on non-refusal briefs: share flagged by at least one
    design-changing item tagged assumed, split by right and wrong; for wrong ones, whether a flag lands on a field
    the comparator found wrong."""
    deliv = {t: d for t, d in runs.items() if d.get("schema_valid") and not d.get("refused")
             and d.get("expected") != "refuse"}
    right = {t: d for t, d in deliv.items() if d.get("correct_spec")}
    wrong = {t: d for t, d in deliv.items() if not d.get("correct_spec")}

    def on_failed(d):
        failed = {k for k, v in ((d.get("match") or {}).get("fields") or {}).items() if v is False}
        return any(FIELD_OF.get(k.split(":")[0], k) in failed for k in assumed_keys(d, material))
    untagged = sum(bool((d.get("sources") or {}).get("untagged")) for d in deliv.values())
    return {"delivered": len(deliv), "untagged_any": untagged,
            "wrong": len(wrong), "wrong_flagged": sorted(t for t, d in wrong.items() if assumed_keys(d, material)),
            "wrong_flag_on_failed_field": sorted(t for t, d in wrong.items() if on_failed(d)),
            "right": len(right), "right_flagged": sorted(t for t, d in right.items() if assumed_keys(d, material))}


def consequences(runs):
    """What each delivered-wrong specification costs under the intended loading (scripts/x2_score.py, GPU): relative
    compliance of its design under the intended problem, material ratio, and keep-in/keep-out violations."""
    out = {}
    for t, d in sorted(runs.items()):
        c = d.get("cwe")
        if silent(d) and c:
            v = c.get("value")
            out[t] = {"cwe": v if v is None or v == v and abs(v) != float("inf") else str(v),
                      "mass_ratio": c.get("mass_ratio"), "passive_violation": c.get("passive_violation") or {}}
    return out


def stated_questions(runs):
    st = {t: d for t, d in runs.items() if d.get("expected") == "solve"}
    return {"stated_briefs": len(st), "asked_on": sum(d.get("questions", 0) > 0 for d in st.values()),
            "questions": sum(d.get("questions", 0) for d in st.values())}


def main():
    out = {}
    T, TS, G, GP, GC, GPC = (load(c) for c in ("T", "TS", "G", "GP", "GC", "GPC"))
    for name, runs in (("T", T), ("TS", TS), ("G", G), ("GP", GP), ("GC", GC), ("GPC", GPC)):
        out[name] = counts(runs)
    out["TS"]["retries"] = {t: d.get("retries", 0) for t, d in TS.items() if d.get("retries")}
    out["T_to_TS"], out["G_to_GP"], out["GC_to_GPC"] = paired(T, TS), paired(G, GP), paired(GC, GPC)
    out["TS_to_G"] = paired(TS, G)
    out["GP_flags"], out["GPC_flags"] = flags(GP), flags(GPC)
    out["GP_flags_without_material"], out["GPC_flags_without_material"] = flags(GP, False), flags(GPC, False)
    out["questions_on_stated_briefs"] = {"GC": stated_questions(GC), "GPC": stated_questions(GPC)}
    out["consequences"] = {name: consequences(runs) for name, runs in (("T", T), ("TS", TS), ("G", G), ("GP", GP),
                                                                       ("GPC", GPC))}
    for label, sub, rep in (("GP_r2", "GP", 2), ("claude_GPC", "claude_high/GPC", 1)):
        runs = load(sub, rep)
        if runs:
            out[label] = counts(runs) | {"flags": flags(runs), "flags_without_material": flags(runs, False),
                                         "consequences": consequences(runs)}
            base = load("G", rep) if label == "GP_r2" else load("claude_high/GC", 1)
            out[label]["paired_vs_" + ("G_r2" if label == "GP_r2" else "claude_GC")] = paired(base, runs)
            if label == "claude_GPC":
                out[label]["questions_on_stated_briefs"] = {"claude_GC": stated_questions(base),
                                                            "claude_GPC": stated_questions(runs)}
    (X2 / "provenance_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k in ("T", "TS", "G", "GP", "GC", "GPC"):
        print(k, out[k]["n"], "right", out[k]["right"], "wrong", len(out[k]["delivered_wrong"]))
    print("GP flags", out["GP_flags"])


if __name__ == "__main__":
    main()
