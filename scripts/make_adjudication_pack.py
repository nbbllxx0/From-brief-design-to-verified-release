"""Adjudication package for the author: one Excel fill sheet plus the evidence, zipped.

    python -B scripts/make_adjudication_pack.py      (needs openpyxl: base Anaconda)

Writes paper/adjudication/AgenticTO_adjudication_pack_<date>.zip with
  ADJUDICATION_SHEET.xlsx   Start here / Briefs (50) / Contested (41) / Progress; the yellow columns are for the author
  README.md                 what to check and how to return the sheet
  briefs/<task>.md          the brief texts the agents saw
  intended/<task>.json      the intended reading, admissible readings and oracle answers
  contested/<nn>_<task>_<cond>.json   each contested run: the agent's spec and final message, what differs, consequence
The filled sheet is read back by scripts/apply_adjudication.py.
"""
import json
import sys
import time
import zipfile
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_adjudication import CONDS, R, failed_fields, runs, short  # noqa: E402

OUT = ROOT / "paper" / "adjudication"
INPUT = PatternFill("solid", fgColor="FFF4C2")
HEAD = PatternFill("solid", fgColor="DDE4EE")
HIGH = PatternFill("solid", fgColor="FBE3E3")
WRAP = Alignment(wrap_text=True, vertical="top")
# conditions whose silent-wrong counts are headline numbers (C2.3: Codex G; C2.6: Claude G+C)
PRIORITY = {"G", "Claude GC"}

README = """# Adjudication package (AgenticTO-100M, X2)

You are the adjudicator. Every X2 number in the paper rests on the intended readings of the 50 briefs and on whether
the "silent wrong" specs are really wrong. Fill the **yellow columns** of `ADJUDICATION_SHEET.xlsx`, save it under the
same name, and send it back. No code is needed.

## Sheet "Briefs" (50 rows)

Read the brief as an engineer would. Then check the intended reading (and, for clarify tasks, the oracle's answers).
- **Intended OK?** Y if the intended reading follows from the brief. N if not: then write the right reading in
  "If N: correct reading".
- **Source OK?** For restated benchmarks: is the cited source right? Use n.a. for engineering briefs.
- **Extra admissible reading:** another reading that should also count as right. Leave it empty if there is none.
- What to watch: units, the up axis, which face, load direction and sense, whether a pad or plate is a passive solid,
  volume fraction of the design (non-passive) volume, and whether a "solve" brief is fully stated.
- Clarify tasks: is the ambiguity real, and would the brief's author answer as the oracle does?
- Refuse tasks (b08, b49, b50): is the brief really unsolvable as stated?

The full intended spec is in `intended/<task>.json`; the brief text is in `briefs/<task>.md`.

## Sheet "Contested" (41 rows)

Each row is a run whose spec is valid, not refused, and matches neither the intended nor an admissible reading.
These rows set the paper's silent-wrong counts.
- **Verdict:** W = truly wrong (an engineer would get a different part); A = acceptable.
- For A, describe the reading in a few words in "If A: the reading"; it becomes an admissible reading for that brief.
- The "compliance excess" column gives the consequence: the agent's design, loaded and supported as intended, against
  the intended optimum (0% = no loss; a huge value means no load path). A negative value means a stiffer part
  bought with more material; check the mass ratio. A different spec can still give the same part, for example
  when E is off by a constant factor.
- Rows marked **high** feed the headline numbers: Codex G's 8 (C2.3) and Claude Code G+C's 8 (C2.6).
- Two cases need your judgment:
  - **b13 and b37 (Claude):** it worked in kN and mm but left E in MPa, so the material is 1000x too stiff. The design
    is the same, but the spec is wrong. Claude's scoring says W.
  - **b09 and b30 (Claude):** the roller is left free across the width (only y is fixed), while the brief frees only
    the span direction. Claude's scoring says W.

The evidence for each row is in `contested/<nn>_<task>_<condition>.json`: the agent's spec, the intended spec, the
fields that differ, the consequence, and the agent's final message.

## After you return the sheet

Claude runs `scripts/apply_adjudication.py`. It adds every A reading as an admissible reading, lists the N rows for
correction, and rescores all 672 saved runs without an LLM. It then regenerates the ledger, tables and figures.
Only the X2-derived numbers can change. The supplement records who adjudicated and the before/after counts.

Sheet "Progress" counts what is still empty.
"""


def write_sheet(ws, header, rows, widths, inputs, dv_specs, priority_col=None):
    ws.append(header)
    for c, _ in enumerate(header, 1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True)
        cell.fill = INPUT if header[c - 1] in inputs else HEAD
        cell.alignment = WRAP
    for r in rows:
        ws.append(r)
    n = len(rows)
    for c, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(c)].width = w
    for row in ws.iter_rows(min_row=2, max_row=n + 1):
        high = priority_col is not None and row[priority_col].value == "high"
        for cell in row:
            cell.alignment = WRAP
            if header[cell.column - 1] in inputs:
                cell.fill = INPUT
            elif high:
                cell.fill = HIGH
    for col_name, values in dv_specs.items():
        col = get_column_letter(header.index(col_name) + 1)
        dv = DataValidation(type="list", formula1='"' + ",".join(values) + '"', allow_blank=True,
                            showErrorMessage=True, errorTitle="Choose from the list", error="Allowed: " + ", ".join(values))
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{n + 1}")
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(header))}{n + 1}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    all_runs = {lab: (sub, runs(sub)) for sub, lab in CONDS}
    ints = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in sorted((ROOT / "briefs" / "intended").glob("*.json"))}

    # ---- briefs
    labs = [lab for _, lab in CONDS]
    b_head = ["Task", "Type", "Source", "Brief (what the agents saw)", "Intended reading", "Oracle answers (clarify tasks)",
              "Admissible readings now", "Notes"] + labs + \
             ["Intended OK? (Y/N)", "Source OK? (Y/N/n.a.)", "If N: correct reading", "Extra admissible reading", "Comment"]
    b_rows = []
    for t, i in ints.items():
        brief = (ROOT / "briefs" / f"{t}.md").read_text(encoding="utf-8").strip()
        outs = []
        for lab in labs:
            d = all_runs[lab][1].get(t)
            outs.append("" if d is None else "right" if d.get("behaviour_ok") else "refused" if d.get("refused")
                        else "invalid" if not d.get("schema_valid") else "WRONG")
        b_rows.append([t, i["expected"], i.get("source", ""), brief, short(i.get("spec")),
                       "\n".join(f"{o['about']}: {o['answer']}" for o in i.get("oracle", [])),
                       "\n".join(a["reading"] for a in i.get("admissible", [])), i.get("notes", "")] + outs + ["", "", "", "", ""])

    # ---- contested (same selection as make_adjudication.py)
    c_head = ["#", "Task", "Type", "Condition", "Priority", "Fields that differ", "Intended reading", "Agent's spec",
              "Compliance excess under intended BCs (0% = no loss)", "Mass ratio (agent / intended)", "Agent's stated assumptions (end of final message)", "Evidence file",
              "Verdict (W/A)", "If A: the reading", "Comment"]
    c_rows, evidence = [], {}
    n = 0
    for lab, (sub, rs) in all_runs.items():
        for t, d in sorted(rs.items()):
            if d.get("behaviour_ok") or d.get("refused") or not d.get("schema_valid") or d.get("expected") == "refuse":
                continue
            n += 1
            cwe = d.get("cwe") or {}
            cr = cwe.get("value")
            name = f"contested/{n:02d}_{t.split('_')[0]}_{lab.replace(' ', '')}.json"
            evidence[name] = {"row": n, "task": t, "condition": lab, "record": f"results/X2/{sub}/{t}_r1.json",
                              "fields_that_differ": failed_fields(d), "match": d.get("match"), "consequence": cwe,
                              "agent_final_message": d.get("final"), "agent_spec": d.get("spec"),
                              "intended_spec": ints[t].get("spec")}
            c_rows.append([n, t, d.get("expected"), lab, "high" if lab in PRIORITY else "", ", ".join(failed_fields(d)) or "-",
                           short(ints[t].get("spec")), short(d.get("spec")),
                           "" if cr is None else (f"{100 * cr:+.1f}%" if cr < 10 else f"{cr:.1e}x (no load path)"),
                           "" if cwe.get("mass_ratio") is None else f"{cwe['mass_ratio']:.3f}",
                           " ".join((d.get("final") or "").split())[-600:], name, "", "", ""])

    wb = Workbook()
    ws = wb.active
    ws.title = "Start here"
    ws.column_dimensions["A"].width = 120
    for line in README.splitlines():
        ws.append([line.replace("**", "").replace("`", "")])
        if line.startswith("#"):
            ws.cell(row=ws.max_row, column=1).font = Font(bold=True, size=13 if line.startswith("# ") else 11)
    write_sheet(wb.create_sheet("Briefs"), b_head, b_rows,
                [20, 9, 18, 60, 55, 45, 30, 35] + [9] * len(labs) + [11, 11, 40, 40, 35],
                {"Intended OK? (Y/N)", "Source OK? (Y/N/n.a.)", "If N: correct reading", "Extra admissible reading", "Comment"},
                {"Intended OK? (Y/N)": ["Y", "N"], "Source OK? (Y/N/n.a.)": ["Y", "N", "n.a."]})
    write_sheet(wb.create_sheet("Contested"), c_head, c_rows,
                [5, 22, 9, 11, 8, 18, 50, 50, 16, 9, 60, 30, 10, 40, 35],
                {"Verdict (W/A)", "If A: the reading", "Comment"}, {"Verdict (W/A)": ["W", "A"]}, priority_col=4)
    p = wb.create_sheet("Progress")
    p.column_dimensions["A"].width = 44
    p.column_dimensions["B"].width = 10
    nb, nc = len(b_rows) + 1, len(c_rows) + 1
    ib = get_column_letter(b_head.index("Intended OK? (Y/N)") + 1)
    sb = get_column_letter(b_head.index("Source OK? (Y/N/n.a.)") + 1)
    vc = get_column_letter(c_head.index("Verdict (W/A)") + 1)
    for label, f in [("Briefs checked (Intended OK filled)", f'=COUNTA(Briefs!{ib}2:{ib}{nb})&" / {len(b_rows)}"'),
                     ("  intended reading marked N", f'=COUNTIF(Briefs!{ib}2:{ib}{nb},"N")'),
                     ("  source marked N", f'=COUNTIF(Briefs!{sb}2:{sb}{nb},"N")'),
                     ("Contested runs judged", f'=COUNTA(Contested!{vc}2:{vc}{nc})&" / {len(c_rows)}"'),
                     ("  W (truly wrong)", f'=COUNTIF(Contested!{vc}2:{vc}{nc},"W")'),
                     ("  A (acceptable)", f'=COUNTIF(Contested!{vc}2:{vc}{nc},"A")')]:
        p.append([label, f])
    p.cell(row=1, column=1).font = Font(bold=True)

    OUT.mkdir(parents=True, exist_ok=True)
    xlsx = OUT / "ADJUDICATION_SHEET.xlsx"
    wb.save(xlsx)
    zp = OUT / f"AgenticTO_adjudication_pack_{time.strftime('%Y-%m-%d')}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(xlsx, "ADJUDICATION_SHEET.xlsx")
        z.writestr("README.md", README)
        for t in ints:
            z.write(ROOT / "briefs" / f"{t}.md", f"briefs/{t}.md")
            z.write(ROOT / "briefs" / "intended" / f"{t}.json", f"intended/{t}.json")
        for name, ev in evidence.items():
            z.writestr(name, json.dumps(ev, indent=1, ensure_ascii=False))
    print(f"{len(b_rows)} briefs, {len(c_rows)} contested runs -> {zp} ({zp.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
