"""C4d: does the stored specification itself expose a wrong reading without the reference?

Reads the first-repeat X2 records (no LLM calls, no solves). For every delivered specification of a solve or
clarify brief, flags the specification when its own free-text fields (description and the note of any support,
load or passive region) state an assumption (the stem "assum"). The same test is applied to the agent's closing
message. Reports, per condition, how many wrong and how many right specifications carry the flag.

Output: results/C4/assumption_flags.json
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_tables import x2  # noqa: E402

STEM = re.compile(r"assum", re.I)


def spec_text(s):
    parts = [s.get("description") or ""]
    for k in ("supports", "loads", "passive"):
        parts += [r.get("note") or "" for r in s.get(k) or []]
    return " ".join(parts)


out = {"rule": "specification free text (description, region notes) contains the stem 'assum'", "conditions": {}}
for cond in ("T", "G", "GC", "GCV"):
    rows = []
    for task, d in sorted(x2(cond).items()):
        if d.get("expected") == "refuse" or d.get("refused") or not d.get("spec"):
            continue
        rows.append({"task": task, "expected": d["expected"], "correct": bool(d.get("correct_spec")),
                     "spec_flag": bool(STEM.search(spec_text(d["spec"]))),
                     "final_flag": bool(STEM.search(str(d.get("final") or "")))})
    wrong = [r for r in rows if not r["correct"]]
    right = [r for r in rows if r["correct"]]
    out["conditions"][cond] = {
        "delivered": len(rows), "wrong": len(wrong), "right": len(right),
        "wrong_spec_flag": sum(r["spec_flag"] for r in wrong), "right_spec_flag": sum(r["spec_flag"] for r in right),
        "wrong_final_flag": sum(r["final_flag"] for r in wrong), "right_final_flag": sum(r["final_flag"] for r in right),
        "wrong_tasks": [r["task"] for r in wrong], "unflagged_wrong": [r["task"] for r in wrong if not r["spec_flag"]],
        "rows": rows}
    c = out["conditions"][cond]
    print(f"{cond}: delivered {c['delivered']}; spec flag on wrong {c['wrong_spec_flag']}/{c['wrong']}, "
          f"on right {c['right_spec_flag']}/{c['right']}; closing-message flag on wrong {c['wrong_final_flag']}/{c['wrong']}, "
          f"on right {c['right_final_flag']}/{c['right']}")
dst = ROOT / "results/C4/assumption_flags.json"
dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
print("->", dst)
