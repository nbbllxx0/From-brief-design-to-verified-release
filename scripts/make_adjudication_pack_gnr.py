"""Adjudication package for the rule-free ablation (GNR), so its outputs get the same adjudication as G's.

    python -B scripts/make_adjudication_pack_gnr.py      (needs openpyxl: base Anaconda)

Same contested selection and sheet layout as make_adjudication_pack.py, restricted to GNR's first repeat: every
schema-valid, non-refused GNR specification that matches neither the intended nor an admissible reading on a
non-refusal task. Each row also names the G outcome on the same brief and, where the G reading was credited through an
admissible alternative taken from a G run, that alternative, so the adjudicator judges both conditions by one rule.
Writes paper/adjudication/gnr/AgenticTO_adjudication_GNR_<date>.zip; the filled sheet is read back by
scripts/apply_adjudication.py (Contested sheet, same columns).
"""
import json
import sys
import time
import zipfile
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_adjudication import failed_fields, runs, short  # noqa: E402
from make_adjudication_pack import write_sheet  # noqa: E402

OUT = ROOT / "paper" / "adjudication" / "gnr"

README = """# Adjudication of the rule-free ablation (GNR)

GNR repeated the grounding condition G with the same prompt, tools, model and CLI, but validation returned schema
errors only (no rule findings). The paper scores GNR automatically: 40 of 50 right, 10 delivered wrong. G's outputs
were adjudicated (the round of 31 October 2026); GNR's were not. This sheet asks for the same judgement on GNR.

For each row in **Contested**, read the brief (`briefs/<task>.md`), the intended and admissible readings
(`intended/<task>.json`) and the evidence file, then enter in the yellow columns:

- **W**: the specification changes the stated problem (loads, supports, material, regions or volume the brief fixes),
  even where the optimal topology would not change;
- **A**: the brief leaves the difference open, so the specification is an acceptable reading. Describe the reading.

Use the same rule as in the first adjudication. The column "G on this brief" shows how G was scored, and "G credited
via" names the admissible reading that credited G when it came from a G run, so both conditions are judged alike.
Return the filled ADJUDICATION_SHEET_GNR.xlsx.
"""


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    gnr, g = runs("GNR"), runs("G")
    ints = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in sorted((ROOT / "briefs" / "intended").glob("*.json"))}
    head = ["#", "Task", "Type", "Condition", "G on this brief", "G credited via", "Fields that differ", "Intended reading",
            "Agent's spec", "Compliance excess under intended BCs (0% = no loss)", "Mass ratio (agent / intended)",
            "Agent's stated assumptions (end of final message)", "Evidence file", "Verdict (W/A)", "If A: the reading",
            "Comment"]
    rows, evidence, n = [], {}, 0
    for t, d in sorted(gnr.items()):
        if d.get("behaviour_ok") or d.get("refused") or not d.get("schema_valid") or d.get("expected") == "refuse":
            continue
        n += 1
        gd = g.get(t, {})
        g_out = "right" if gd.get("behaviour_ok") else "refused" if gd.get("refused") else "WRONG"
        via = ""
        m = gd.get("match") or {}
        if str(m.get("matched", "")).startswith("admissible:"):
            a = ints[t].get("admissible", [])[int(m["matched"].split(":")[1])]
            if a.get("from_run", "").startswith("G/"):
                via = f"{a['reading']} (from {a['from_run']})"
        cwe = d.get("cwe") or {}
        cr = cwe.get("value")
        name = f"contested/{n:02d}_{t.split('_')[0]}_GNR.json"
        evidence[name] = {"row": n, "task": t, "condition": "GNR", "record": f"results/X2/GNR/{t}_r1.json",
                          "fields_that_differ": failed_fields(d), "match": d.get("match"), "consequence": cwe,
                          "agent_final_message": d.get("final"), "agent_spec": d.get("spec"),
                          "intended_spec": ints[t].get("spec"), "G_outcome": g_out, "G_record": f"results/X2/G/{t}_r1.json",
                          "G_spec": gd.get("spec")}
        rows.append([n, t, d.get("expected"), "GNR", g_out, via, ", ".join(failed_fields(d)) or "-",
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
            ws.cell(row=ws.max_row, column=1).font = Font(bold=True, size=13)
    write_sheet(wb.create_sheet("Contested"), head, rows, [5, 22, 9, 8, 10, 40, 18, 50, 50, 16, 9, 60, 30, 10, 40, 35],
                {"Verdict (W/A)", "If A: the reading", "Comment"}, {"Verdict (W/A)": ["W", "A"]})
    OUT.mkdir(parents=True, exist_ok=True)
    xlsx = OUT / "ADJUDICATION_SHEET_GNR.xlsx"
    wb.save(xlsx)
    zp = OUT / f"AgenticTO_adjudication_GNR_{time.strftime('%Y-%m-%d')}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(xlsx, "ADJUDICATION_SHEET_GNR.xlsx")
        z.writestr("README.md", README)
        for t in sorted({r[1] for r in rows}):
            z.write(ROOT / "briefs" / f"{t}.md", f"briefs/{t}.md")
            z.write(ROOT / "briefs" / "intended" / f"{t}.json", f"intended/{t}.json")
        for nm, ev in evidence.items():
            z.writestr(nm, json.dumps(ev, indent=1, ensure_ascii=False))
    print(f"{len(rows)} contested GNR runs -> {zp} ({zp.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
