"""Read the author's filled adjudication sheet back and apply what can be applied mechanically.

    python -B scripts/apply_adjudication.py <filled ADJUDICATION_SHEET.xlsx> [--dry]

- Contested rows marked A: the agent's spec becomes an admissible reading of that brief
  (briefs/intended/<task>.json, "admissible", with "adjudicated": true and the author's words as the reading).
- Brief rows marked N (intended reading or source), and extra admissible readings written in words, need a spec edit:
  they are listed for Claude to encode, together with the author's text.
- Every decision is written to paper/adjudication/adjudication_decisions.json (the record for the supplement).
Then: x2_rescore.py, make_ledger.py, make_tables.py, make_figures.py, make_adjudication.py, and x2_score.py for runs
whose verdict changed.
"""
import json
import sys
import time
from pathlib import Path

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_adjudication import CONDS  # noqa: E402

SUB = {lab: sub for sub, lab in CONDS}


def rows(ws):
    it = ws.iter_rows(values_only=True)
    head = [str(h) for h in next(it)]
    for r in it:
        if any(v not in (None, "") for v in r):
            yield {h: ("" if v is None else str(v).strip()) for h, v in zip(head, r)}


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8")
    dry = "--dry" in argv
    path = Path(next(a for a in argv if not a.startswith("--")))
    wb = load_workbook(path, data_only=True)
    briefs, contested = list(rows(wb["Briefs"])), list(rows(wb["Contested"]))
    empty_b = [b["Task"] for b in briefs if b["Intended OK? (Y/N)"] not in ("Y", "N")]
    empty_c = [f"#{c['#']} {c['Task']} {c['Condition']}" for c in contested if c["Verdict (W/A)"] not in ("W", "A")]
    if empty_b or empty_c:
        print(f"not filled: {len(empty_b)} briefs {empty_b[:8]}..., {len(empty_c)} contested {empty_c[:8]}...")

    todo, added = [], []
    for b in briefs:
        if b["Intended OK? (Y/N)"] == "N":
            todo.append({"task": b["Task"], "what": "intended reading marked N", "author": b["If N: correct reading"],
                         "comment": b["Comment"]})
        if b["Source OK? (Y/N/n.a.)"] == "N":
            todo.append({"task": b["Task"], "what": "source marked N", "source": b["Source"], "comment": b["Comment"]})
        if b["Extra admissible reading"]:
            todo.append({"task": b["Task"], "what": "extra admissible reading (encode as a spec)",
                         "author": b["Extra admissible reading"], "comment": b["Comment"]})

    stamp = time.strftime("%Y-%m-%d")
    for c in contested:
        if c["Verdict (W/A)"] != "A":
            continue
        task, cond = c["Task"], c["Condition"]
        rec = json.loads((ROOT / "results" / "X2" / SUB[cond] / f"{task}_r1.json").read_text(encoding="utf-8"))
        f = ROOT / "briefs" / "intended" / f"{task}.json"
        intended = json.loads(f.read_text(encoding="utf-8"))
        adm = intended.setdefault("admissible", [])
        if any(a.get("spec") == rec["spec"] for a in adm):
            continue
        adm.append({"reading": c["If A: the reading"] or f"as {cond} read it (contested #{c['#']})",
                    "why": c["Comment"] or "judged acceptable by the author in the adjudication",
                    "counts": True, "adjudicated": stamp, "from_run": f"{SUB[cond]}/{task}_r1", "spec": rec["spec"]})
        added.append(f"{task} <- {cond}")
        if not dry:
            f.write_text(json.dumps(intended, indent=1, ensure_ascii=False), encoding="utf-8")

    decisions = {"sheet": str(path), "applied": stamp, "dry": dry,
                 "briefs": {"Y": sum(b["Intended OK? (Y/N)"] == "Y" for b in briefs),
                            "N": sum(b["Intended OK? (Y/N)"] == "N" for b in briefs), "empty": len(empty_b)},
                 "contested": {"W": sum(c["Verdict (W/A)"] == "W" for c in contested),
                               "A": sum(c["Verdict (W/A)"] == "A" for c in contested), "empty": len(empty_c)},
                 "admissible_added": added, "needs_spec_edit": todo,
                 "rows": {"briefs": briefs, "contested": contested}}
    if not dry:
        (ROOT / "paper" / "adjudication" / "adjudication_decisions.json").write_text(
            json.dumps(decisions, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: decisions[k] for k in ("briefs", "contested")}))
    print(f"admissible readings added{' (dry)' if dry else ''}: {len(added)}", *added, sep="\n  ")
    print(f"need a spec edit by hand: {len(todo)}")
    for t in todo:
        print(f"  {t['task']}: {t['what']}: {t.get('author') or t.get('source', '')} {t.get('comment', '')}")


if __name__ == "__main__":
    main(sys.argv[1:])
