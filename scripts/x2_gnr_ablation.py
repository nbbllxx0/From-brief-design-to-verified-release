"""X2 ablation GNR: the grounding protocol G with the rule findings removed from validate_spec (same prompt, tools,
provider and CLI), first repeat, scored like every other X2 condition against the adjudicated references. Compares
GNR with G task by task. No LLM calls, no solves.

Output: results/X2/GNR/summary.json
"""
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from make_tables import mcnemar, silent, wilson, x2  # noqa: E402
from x2_failure_categories import cat  # noqa: E402
from agentto.rules import check  # noqa: E402
from agentto.spec import Spec  # noqa: E402


def rule_errors(d):
    if not d.get("schema_valid"):
        return None
    return sorted({f.code for f in check(Spec.model_validate(d["spec"])).findings if f.severity == "error"})


def condition(runs):
    by_exp = collections.defaultdict(lambda: collections.Counter())
    for d in runs.values():
        by_exp[d["expected"]][cat(d)] += 1
    val = [d.get("tools_used", []).count("validate_spec") for d in runs.values()]
    return {"n": len(runs), "right": sum(bool(d.get("behaviour_ok")) for d in runs.values()),
            "right_ci95": wilson(sum(bool(d.get("behaviour_ok")) for d in runs.values()), len(runs)),
            "delivered_wrong": sorted(t for t, d in runs.items() if silent(d)),
            "delivered_with_rule_error": sorted(t for t, d in runs.items()
                                                if d.get("schema_valid") and not d.get("refused")
                                                and d.get("rules_ok") is False),
            "categories": {k: dict(v) for k, v in sorted(by_exp.items())},
            "validate_calls_median": statistics.median(val),
            "input_tokens_median": statistics.median(d.get("usage", {}).get("input_tokens", 0) for d in runs.values()),
            "wall_s_median": statistics.median(d.get("wall_s", 0) for d in runs.values())}


G, N = x2("G"), x2("GNR")
assert set(G) == set(N), (len(G), len(N))
fixed = sorted(t for t in G if N[t].get("behaviour_ok") and not G[t].get("behaviour_ok"))
broken = sorted(t for t in G if G[t].get("behaviour_ok") and not N[t].get("behaviour_ok"))
out = {"G": condition(G), "GNR": condition(N),
       "G_to_GNR": {"corrected": fixed, "broken": broken, "p_mcnemar": mcnemar(len(fixed), len(broken))},
       "discordant": {t: {"expected": G[t]["expected"], "G": cat(G[t]), "GNR": cat(N[t]),
                          "GNR_fields_wrong": [k for k, v in (N[t].get("match", {}).get("fields") or {}).items()
                                               if v is False],
                          "GNR_rule_errors": rule_errors(N[t]), "GNR_final": N[t].get("final", "")[-600:]}
                      for t in fixed + broken},
       "GNR_delivered_wrong_detail": {t: {"expected": N[t]["expected"], "rule_errors": rule_errors(N[t]),
                                          "fields_wrong": [k for k, v in (N[t].get("match", {}).get("fields") or {})
                                                           .items() if v is False]}
                                      for t in sorted(t for t, d in N.items() if silent(d))}}
for c in ("G", "GNR"):
    s = out[c]
    print(f"{c}: right {s['right']}/{s['n']}, delivered wrong {len(s['delivered_wrong'])} {s['delivered_wrong']}, "
          f"delivered with a rule error {s['delivered_with_rule_error']}, validate calls median "
          f"{s['validate_calls_median']}")
    print("   ", s["categories"])
print("G->GNR corrected", fixed, "broken", broken, "p", round(out["G_to_GNR"]["p_mcnemar"], 4))
for t, d in out["GNR_delivered_wrong_detail"].items():
    print("  GNR wrong", t, d)
dst = ROOT / "results" / "X2" / "GNR" / "summary.json"
dst.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("->", dst)
