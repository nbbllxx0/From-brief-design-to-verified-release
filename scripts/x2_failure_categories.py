"""X2: failure categories of the first-repeat text-only (T) and grounding (G) outputs, and of the T outputs that G
corrected or broke. Categories, in order: right, did not refuse (refusal brief), refused a solvable brief, no
specification, schema-invalid, rule error, wrong reading (schema-valid, rule-clean, not the reference).

Output: results/X2/failure_categories.json
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_tables import x2  # noqa: E402


def cat(d):
    if d.get("behaviour_ok"):
        return "right"
    if d.get("expected") == "refuse":
        return "did not refuse"
    if d.get("refused"):
        return "refused solvable"
    if not d.get("spec"):
        return "no spec"
    if not d.get("schema_valid"):
        return "schema-invalid"
    if not d.get("rules_ok"):
        return "rule error"
    return "wrong reading"


T, G = x2("T"), x2("G")
fixed = [t for t in T if G[t].get("behaviour_ok") and not T[t].get("behaviour_ok")]
broken = [t for t in T if T[t].get("behaviour_ok") and not G[t].get("behaviour_ok")]
out = {"T": dict(collections.Counter(cat(d) for d in T.values())),
       "G": dict(collections.Counter(cat(d) for d in G.values())),
       "T_to_G_fixed": {t: [T[t]["expected"], cat(T[t])] for t in fixed},
       "T_to_G_broken": {t: [T[t]["expected"], cat(G[t])] for t in broken}}
print(json.dumps(out, indent=1))
(ROOT / "results/X2/failure_categories.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
