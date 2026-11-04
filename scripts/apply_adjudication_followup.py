"""Read the filled follow-up adjudication sheet back (scripts/make_adjudication_pack_followup.py) and apply it.

    python -B scripts/apply_adjudication_followup.py <filled ADJUDICATION_SHEET_FOLLOWUP.xlsx> [--dry]

- Rows marked A: the run's spec becomes an admissible reading of that brief (briefs/intended/<task>.json,
  "admissible", with "adjudicated": <date> and the adjudicator's words), exactly as apply_adjudication.py does for the
  suite; like every admissible reading it then counts for any run that produced an equivalent spec.
- Rows marked A whose reading was judged W on 31 October are listed as revisions of that earlier verdict.
- Rows with the same reading as an earlier row but a different verdict are listed as inconsistent and not applied.
- Every decision goes to paper/adjudication/followup/followup_decisions.json; the 31 October record
  (paper/adjudication/adjudication_decisions.json) is left as it is.
Then: x2_rescore.py, x2_provenance.py, make_tables.py, and the paper tables (paper/V4/scripts/prepare_assets.py, X2 tables only).
"""
import json
import sys
import time
from pathlib import Path

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "adjudication" / "followup"


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
    contested = list(rows(load_workbook(path, data_only=True)["Contested"]))
    verdict = {c["#"]: c["Verdict (W/A)"] for c in contested}
    empty = [f"#{c['#']} {c['Task']} {c['Condition']}" for c in contested if c["Verdict (W/A)"] not in ("W", "A")]
    if empty:
        print(f"not filled: {len(empty)} rows {empty[:8]}...")
    inconsistent = [f"#{c['#']} {c['Verdict (W/A)']} vs #{c['Same reading as row']} {verdict[c['Same reading as row']]}"
                    for c in contested if c["Same reading as row"] and c["Verdict (W/A)"] in ("W", "A")
                    and verdict.get(c["Same reading as row"]) in ("W", "A")
                    and c["Verdict (W/A)"] != verdict[c["Same reading as row"]]]
    revised = [f"#{c['#']} {c['Task']}: now A, judged {c['Judged on 31 Oct']} on 31 October" for c in contested
               if c["Verdict (W/A)"] == "A" and c["Judged on 31 Oct"].startswith("W")]
    bad = {c.split()[0] for c in inconsistent}

    stamp = time.strftime("%Y-%m-%d")
    added = []
    for c in contested:
        if c["Verdict (W/A)"] != "A" or f"#{c['#']}" in bad:
            continue
        task, record = c["Task"], c["Record"]
        rec = json.loads((ROOT / record).read_text(encoding="utf-8"))
        f = ROOT / "briefs" / "intended" / f"{task}.json"
        intended = json.loads(f.read_text(encoding="utf-8"))
        adm = intended.setdefault("admissible", [])
        if any(a.get("spec") == rec["spec"] for a in adm):
            continue
        adm.append({"reading": c["If A: the reading"] or f"as {c['Condition']} read it (follow-up contested #{c['#']})",
                    "why": c["Comment"] or "judged acceptable in the follow-up adjudication",
                    "counts": True, "adjudicated": stamp,
                    "from_run": record.removeprefix("results/X2/").removesuffix(".json"), "spec": rec["spec"]})
        added.append(f"{task} <- {record}")
        if not dry:
            f.write_text(json.dumps(intended, indent=1, ensure_ascii=False), encoding="utf-8")

    decisions = {"sheet": str(path), "applied": stamp, "dry": dry,
                 "contested": {"W": sum(c["Verdict (W/A)"] == "W" for c in contested),
                               "A": sum(c["Verdict (W/A)"] == "A" for c in contested), "empty": len(empty)},
                 "admissible_added": added, "revisions_of_31_october": revised, "inconsistent_not_applied": inconsistent,
                 "rows": contested}
    if not dry:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "followup_decisions.json").write_text(json.dumps(decisions, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(decisions["contested"]))
    print(f"admissible readings added{' (dry)' if dry else ''}: {len(added)}", *added, sep="\n  ")
    print(f"revisions of a 31 October verdict: {len(revised)}", *revised, sep="\n  ")
    print(f"inconsistent with an equivalent row (not applied): {len(inconsistent)}", *inconsistent, sep="\n  ")


if __name__ == "__main__":
    main(sys.argv[1:])
