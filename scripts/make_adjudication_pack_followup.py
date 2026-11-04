"""Adjudication package for the follow-up X2 runs, so their contested outputs get the same judgement as the suite's.

    python -B scripts/make_adjudication_pack_followup.py   (needs openpyxl: base Anaconda)

Same contested selection as make_adjudication_pack.py and make_adjudication_pack_gnr.py: every schema-valid,
non-refused specification that matches neither the intended nor an admissible reading on a non-refusal task, here
from GNR (rule-free ablation), T+S (schema feedback), G+P and its second repeat (value-source tags), G+C+P
(provenance-gated questions) and the second provider's G+C+P. It contains the 10 GNR rows of the GNR-only pack and
replaces it. Each row names the G outcome on the same brief, the admissible reading that credited G when it came from
a G run, and the earlier row whose specification is identical, so one reading is judged once and by one rule.
Writes paper/adjudication/followup/AgenticTO_adjudication_followup_<date>.zip; the filled sheet is read back by
scripts/apply_adjudication_followup.py.
"""
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_adjudication import failed_fields, short  # noqa: E402

X2 = ROOT / "results" / "X2"
OUT = ROOT / "paper" / "adjudication" / "followup"
# (label shown to the adjudicator, results folder, repeat, what the condition changed)
ARMS = [("GNR", "GNR", 1, "G with schema-only validation (no rule findings)"),
        ("T+S", "TS", 1, "text only, with up to two retries on the schema validator's errors"),
        ("G+P", "GP", 1, "G, plus a source tag (stated, derived, assumed) on every specification item"),
        ("G+P", "GP", 2, "second repeat of G+P on the 20-brief subset"),
        ("G+C+P", "GPC", 1, "clarification allowed only about items the agent tagged assumed (oracle answers)"),
        ("Claude G+C+P", "claude_high/GPC", 1, "G+C+P with the second provider (Claude Code, Opus 5.5)")]

README = """# Adjudication of the follow-up runs (1 November 2026)

After the suite was adjudicated (31 October 2026), five follow-up conditions were run on the same 50 briefs; the
paper scores them automatically against the adjudicated references and says they were not adjudicated. This sheet
asks for the same judgement on their contested outputs: every valid specification, delivered as finished on a
brief that should be solved or clarified, that matches neither the intended nor an admissible reading.

| Condition | What it changed |
|---|---|
{arms}

It contains the 10 rows of the earlier GNR-only sheet (ADJUDICATION_SHEET_GNR.xlsx); if you have started that sheet,
copy your verdicts across (same task and condition) and return only this one.

For each row in **Contested**, read the brief (`briefs/<task>.md`), the intended and admissible readings
(`intended/<task>.json`) and the evidence file, then enter in the yellow columns:

- **W**: the specification changes the stated problem (loads, supports, material, regions or volume the brief fixes),
  even where the optimal topology would not change;
- **A**: the brief leaves the difference open, so the specification is an acceptable reading. Describe the reading.

Use the same rule as in the first adjudication. "G on this brief" shows how G was scored, and "G credited via" names
the admissible reading that credited G when it came from a G run. "Same reading as row" points to an earlier row whose
specification the benchmark comparator finds equivalent (same problem up to names, units and wording): it normally
takes the same verdict, and it is listed so that every run is accounted for. "Judged on 31 Oct" names a contested
record of the first adjudication with an equivalent specification and your verdict on it: confirm it, or revise it
and say why in the comment (a revision would change that earlier record too). The compliance
and mass columns show what the reading costs under the intended problem; they do not decide the verdict.
Return the filled ADJUDICATION_SHEET_FOLLOWUP.xlsx.
"""


def runs(sub, rep):
    return {d["task"]: d for d in (json.loads(f.read_text(encoding="utf-8"))
                                   for f in sorted((X2 / sub).glob(f"b*_r{rep}.json")))}


def candidates():
    """Contested runs of every follow-up arm, ordered by brief so that repeats of a reading sit together."""
    cand = []
    for lab, sub, rep, _ in ARMS:
        for t, d in runs(sub, rep).items():
            if d.get("behaviour_ok") or d.get("refused") or not d.get("schema_valid") or d.get("expected") == "refuse":
                continue
            cand.append((t, lab, sub, rep, d))
    cand.sort(key=lambda c: (c[0], [a[0] for a in ARMS].index(c[1]), c[3]))
    return cand


def main():
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from make_adjudication_pack import write_sheet
    sys.stdout.reconfigure(encoding="utf-8")
    g = runs("G", 1)
    ints = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in sorted((ROOT / "briefs" / "intended").glob("*.json"))}
    eq = json.loads((OUT / "equivalence.json").read_text(encoding="utf-8"))   # scripts/followup_equivalence.py
    before = json.loads((OUT / "earlier_verdicts.json").read_text(encoding="utf-8"))
    head = ["#", "Task", "Type", "Condition", "Repeat", "Record", "Same reading as row", "Judged on 31 Oct",
            "G on this brief", "G credited via",
            "Fields that differ", "Intended reading", "Agent's spec",
            "Compliance excess under intended BCs (0% = no loss)", "Mass ratio (agent / intended)",
            "Agent's stated assumptions (end of final message)", "Evidence file", "Verdict (W/A)", "If A: the reading",
            "Comment"]
    rows, evidence, row_of = [], {}, {}
    for n, (t, lab, sub, rep, d) in enumerate(candidates(), 1):
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
        record = f"results/X2/{sub}/{t}_r{rep}.json"
        row_of[record] = n
        same = row_of[eq[record]] if record in eq else n
        b = before.get(record)
        judged = f"{b['verdict']} ({b['record'].split('/X2/')[1].rsplit('/', 1)[0]})" if b else ""
        name = f"contested/{n:02d}_{t.split('_')[0]}_{lab.replace(' ', '_').replace('+', '')}_r{rep}.json"
        evidence[name] = {"row": n, "task": t, "condition": lab, "repeat": rep, "record": record,
                          "fields_that_differ": failed_fields(d), "match": d.get("match"), "consequence": cwe,
                          "questions_asked": d.get("questions", 0), "agent_final_message": d.get("final"),
                          "agent_spec": d.get("spec"), "intended_spec": ints[t].get("spec"),
                          "agent_source_tags": d.get("sources"), "G_outcome": g_out,
                          "G_record": f"results/X2/G/{t}_r1.json", "G_spec": gd.get("spec"),
                          "same_reading_as_row": None if same == n else same, "judged_31_oct": b}
        rows.append([n, t, d.get("expected"), lab, rep, record, "" if same == n else same, judged, g_out, via,
                     ", ".join(failed_fields(d)) or "-", short(ints[t].get("spec")), short(d.get("spec")),
                     "" if cr is None else (f"{100 * cr:+.1f}%" if cr < 10 else f"{cr:.1e}x (no load path)"),
                     "" if cwe.get("mass_ratio") is None else f"{cwe['mass_ratio']:.3f}",
                     " ".join((d.get("final") or "").split())[-600:], name, "", "", ""])

    readme = README.format(arms="\n".join(f"| {lab}{'' if rep == 1 else ' (repeat 2)'} | {what} |"
                                          for lab, _, rep, what in ARMS))
    wb = Workbook()
    ws = wb.active
    ws.title = "Start here"
    ws.column_dimensions["A"].width = 120
    for line in readme.splitlines():
        ws.append([line.replace("**", "").replace("`", "")])
        if line.startswith("#"):
            ws.cell(row=ws.max_row, column=1).font = Font(bold=True, size=13)
    write_sheet(wb.create_sheet("Contested"), head, rows,
                [5, 22, 9, 12, 7, 30, 9, 14, 10, 40, 18, 50, 50, 16, 9, 60, 30, 10, 40, 35],
                {"Verdict (W/A)", "If A: the reading", "Comment"}, {"Verdict (W/A)": ["W", "A"]})
    OUT.mkdir(parents=True, exist_ok=True)
    xlsx = OUT / "ADJUDICATION_SHEET_FOLLOWUP.xlsx"
    wb.save(xlsx)
    zp = OUT / f"AgenticTO_adjudication_followup_{time.strftime('%Y-%m-%d')}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(xlsx, "ADJUDICATION_SHEET_FOLLOWUP.xlsx")
        z.writestr("README.md", readme)
        for t in sorted({r[1] for r in rows}):
            z.write(ROOT / "briefs" / f"{t}.md", f"briefs/{t}.md")
            z.write(ROOT / "briefs" / "intended" / f"{t}.json", f"intended/{t}.json")
        for nm, ev in evidence.items():
            z.writestr(nm, json.dumps(ev, indent=1, ensure_ascii=False))
    uniq = [r for r in rows if r[6] == ""]
    by = {}
    for r in rows:
        by[r[3] + ("" if r[4] == 1 else " r2")] = by.get(r[3] + ("" if r[4] == 1 else " r2"), 0) + 1
    print(f"{len(rows)} contested runs ({len(uniq)} distinct readings, {sum(r[7] == '' for r in uniq)} not judged "
          f"before) on {len({r[1] for r in rows})} briefs: {by}")
    print(f"-> {zp} ({zp.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
