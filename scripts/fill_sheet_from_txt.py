"""Copy the adjudicator's plain-text results into a copy of ADJUDICATION_SHEET.xlsx, for apply_adjudication.py.

    python -B scripts/fill_sheet_from_txt.py <results.txt> <out.xlsx>

Two formats are read:
- final (paper/adjudication/ADJUDICATION_RESULTS_<date>.txt): a BRIEFS block ("<task>: intended Y; source N", comment
  lines, optional "Additional reading:" and "Correction:") and a CONTESTED RUNS block ("<nn> <task> / <cond>: W|A",
  rationale, optional "Accepted reading:", "Evidence:");
- preliminary: contested rows only, one per line as "<nn> | W|A | rationale".
"""
import re
import sys
from pathlib import Path

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]


def parse(txt):
    lines = txt.splitlines()
    briefs, contested, cur = {}, {}, None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(\d{2}) \| ([WA]) \| (.+)$", line)
        if m:
            contested[int(m.group(1))] = {"verdict": m.group(2), "comment": m.group(3), "reading": ""}
            cur = None
            continue
        m = re.match(r"^(b\d\d_\w+): intended ([YN]); source (Y|N|n\.a\.)$", line)
        if m:
            cur = briefs.setdefault(m.group(1), {"intended": m.group(2), "source": m.group(3), "comment": [],
                                                 "extra": "", "correction": ""})
            continue
        m = re.match(r"^(\d{2}) (b\d\d_\w+) / (.+): ([WA])$", line)
        if m:
            cur = contested.setdefault(int(m.group(1)), {"task": m.group(2), "cond": m.group(3),
                                                         "verdict": m.group(4), "comment": [], "reading": ""})
            continue
        if cur is None:
            continue
        for key, tag in (("extra", "Additional reading:"), ("correction", "Correction:"), ("reading", "Accepted reading:")):
            if line.startswith(tag):
                cur[key] = line[len(tag):].strip()
                break
        else:
            if not line.startswith("Evidence:") and isinstance(cur.get("comment"), list):
                cur["comment"].append(line)
    for d in list(briefs.values()) + list(contested.values()):
        if isinstance(d.get("comment"), list):
            d["comment"] = " ".join(d["comment"])
    return briefs, contested


def main(argv):
    txt, out = Path(argv[0]), Path(argv[1])
    briefs, contested = parse(txt.read_text(encoding="utf-8"))
    wb = load_workbook(ROOT / "paper" / "adjudication" / "ADJUDICATION_SHEET.xlsx")

    ws = wb["Contested"]
    ci = {c.value: i + 1 for i, c in enumerate(ws[1])}
    nc = 0
    for r in range(2, ws.max_row + 1):
        num = ws.cell(r, ci["#"]).value
        c = contested.get(int(num)) if num is not None else None
        if not c:
            continue
        if "task" in c:
            assert c["task"] == ws.cell(r, ci["Task"]).value, (num, c["task"])
            assert c["cond"] == ws.cell(r, ci["Condition"]).value, (num, c["cond"])
        ws.cell(r, ci["Verdict (W/A)"]).value = c["verdict"]
        if c["verdict"] == "A":
            ws.cell(r, ci["If A: the reading"]).value = c["reading"] or re.split(r"(?<=[.;])\s", c["comment"], 1)[0]
        ws.cell(r, ci["Comment"]).value = c["comment"]
        nc += 1

    ws = wb["Briefs"]
    ci = {c.value: i + 1 for i, c in enumerate(ws[1])}
    nb = 0
    for r in range(2, ws.max_row + 1):
        b = briefs.get(ws.cell(r, ci["Task"]).value)
        if not b:
            continue
        ws.cell(r, ci["Intended OK? (Y/N)"]).value = b["intended"]
        ws.cell(r, ci["Source OK? (Y/N/n.a.)"]).value = b["source"]
        ws.cell(r, ci["If N: correct reading"]).value = b["correction"] or None
        ws.cell(r, ci["Extra admissible reading"]).value = b["extra"] or None
        ws.cell(r, ci["Comment"]).value = b["comment"]
        nb += 1
    wb.save(out)
    print(f"{nb} briefs and {nc} contested verdicts written to {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
