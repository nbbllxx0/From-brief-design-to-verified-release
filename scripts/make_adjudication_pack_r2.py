"""Blinded second-round adjudication package (second adjudicator: S.Y.; first round: J.W.).

    python -B scripts/make_adjudication_pack_r2.py

The second adjudicator judges the same items as the first round, on the same reference readings:
  Part A  the 50 intended readings as drafted before adjudication (from the archived scoring inputs)
  Part B  the 41 contested records, shuffled, under neutral ids C01..C41
Blinding: no condition, provider, agent message, run outcome, first-round verdict or comment; candidate specs are
re-written in one neutral form (no region ids, names lower-cased, no metadata), so writing style does not reveal the
provider. The id -> record key is written outside the zip, to paper/adjudication/round2/key_r2.json.
The filled sheet is compared with the first round by scripts/compare_adjudications.py.
"""
import json
import random
import sys
import tarfile
import time
import zipfile
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_adjudication_pack import write_sheet  # noqa: E402

FIRST = ROOT / "paper" / "adjudication" / "ADJUDICATION_SHEET_2026-10-31.xlsx"
FIRST_PACK = ROOT / "paper" / "adjudication" / "AgenticTO_adjudication_pack_2026-10-30.zip"
PRE = ROOT / "paper" / "archives" / "pre_adjudication_20261031" / "scoring_inputs.tgz"
OUT = ROOT / "paper" / "adjudication" / "round2"
SEED = 20261031

README = """# Second-round adjudication (AgenticTO, 50-brief benchmark)

You are the second adjudicator. The first adjudication was done independently; its verdicts are not in this package.
Please do not consult them, the results folder, the paper's tables, or any earlier adjudication file until you have
returned this sheet. Fill the **yellow columns** of `ADJUDICATION_SHEET_R2.xlsx` and send it back.

What is hidden from you: which model, provider or condition produced each candidate, the agents' messages, the run
outcomes and the first adjudicator's verdicts. Candidate specifications are re-written in one neutral form, and the
contested rows are in random order.

## Sheet "A_References" (50 rows)

Read each brief as an engineer would, then the reference reading as drafted before adjudication.
- **Reference OK? (Y/N)**: Y if the reading follows from the brief. If N, write the right reading.
- **Source OK? (Y/N/n.a.)**: for restated benchmarks, is the cited source right? n.a. for engineering briefs.
- **Extra admissible reading**: another reading that should also count as right, in words. Leave empty if none.
- Watch units, the up axis, which face, load direction and sense, whether a pad or plate is a retained solid, the
  volume-fraction base (design volume, excluding retained solids), and whether a "solve" brief is fully stated.
- Clarify tasks: is the ambiguity real, and would the brief's author answer as the oracle does?
- Refuse tasks: is the brief really unsolvable as stated?

## Sheet "B_Candidates" (41 rows)

Each row is a valid specification that matched neither the reference nor an admissible reading listed in sheet A.
- **Verdict (W/A)**: W = wrong (an engineer would get a different part, or a different problem); A = acceptable reading
  of the brief.
- For A, describe the reading in a few words.
- "Compliance excess" is the candidate's design loaded and supported as the reference states, against the reference
  optimum (0% = no loss; "no load path" = the design cannot carry the reference load). A negative value means a stiffer
  part bought with more material; check the mass ratio. A different specification can give the same part (for
  example when E differs by a constant factor); judge the specification, and say in the comment if you would judge
  the part differently.

Evidence: `briefs/<task>.md` (the text the agents saw), `reference/<task>.json` (reference, admissible readings and
oracle answers as drafted), `candidates/<id>.json` (the candidate specification in neutral form and its consequence).

Sheet "Progress" counts what is still empty.
"""


def g(v):
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def gl(vs):
    return "[" + ", ".join(g(x) for x in vs or []) + "]"


def region(r):
    k = r["kind"]
    if k == "face":
        return f"face {r['face']}"
    if k == "patch":
        return f"patch on {r['face']} centre {gl(r['center'])} size {gl(r['size'])}"
    if k == "edge":
        return f"edge along {r['axis']} at {gl(r['at'])}"
    if k == "box":
        return f"box {gl(r['min'])} to {gl(r['max'])}"
    if k == "cylinder":
        return f"cylinder axis {r.get('axis')} centre {gl(r.get('center'))} radius {g(r.get('radius'))}"
    return k


def neutral(spec):
    """One fixed wording for any spec: no ids, names lower-cased, numbers in %g, no metadata."""
    if not spec:
        return ""
    m = spec.get("material", {})
    s = [f"domain {gl(spec['domain']['size'])} {spec['units']['length']}, force {spec['units'].get('force', 'N')}",
         f"volume fraction {g(spec.get('volume_fraction'))}",
         f"material {(m.get('name') or '').lower()} E {g(m.get('E'))} nu {g(m.get('nu'))}"]
    for it in spec.get("supports", []):
        s.append(f"support: {region(it['region'])}, fixes {''.join(it.get('fix', ['x', 'y', 'z']))}")
    for it in spec.get("loads", []):
        s.append(f"load: {region(it['region'])}, force {gl(it.get('force'))}")
    for it in spec.get("passive", []):
        s.append(f"{it['kind']}: {region(it['region'])}")
    res = spec.get("resolution") or {}
    if res:
        s.append("resolution " + ", ".join(f"{k} {g(v)}" for k, v in res.items()))
    if spec.get("min_feature") is not None:
        s.append(f"minimum feature {g(spec['min_feature'])}")
    return "; ".join(s)


def num(o):
    """Same number format for every candidate: integral floats become ints."""
    if isinstance(o, dict):
        return {k: num(v) for k, v in o.items()}
    if isinstance(o, list):
        return [num(v) for v in o]
    if isinstance(o, float) and o.is_integer():
        return int(o)
    return o


def strip_spec(spec):
    if not spec:
        return spec
    spec = num(spec)
    out = {k: v for k, v in spec.items() if k not in ("meta", "name", "id", "notes", "description", "schema_version")}
    out["material"] = {k: (v.lower() if k == "name" and isinstance(v, str) else v)
                       for k, v in spec.get("material", {}).items()}
    for key, tag in (("supports", "s"), ("loads", "l"), ("passive", "p")):
        out[key] = [{**{k: v for k, v in it.items() if k not in ("id", "name", "label", "note", "description", "distribution")},
                     "id": f"{tag}{i + 1}"}
                    for i, it in enumerate(spec.get(key, []))]
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    wb1 = load_workbook(FIRST, data_only=True)
    b1 = list(wb1["Briefs"].iter_rows(values_only=True))
    c1 = list(wb1["Contested"].iter_rows(values_only=True))
    bh, ch = b1[0], c1[0]
    with tarfile.open(PRE) as tf:
        pre = {Path(m.name).stem: json.load(tf.extractfile(m)) for m in tf.getmembers()
               if m.isfile() and m.name.startswith("briefs/intended/") and m.name.endswith(".json")}
    with zipfile.ZipFile(FIRST_PACK) as z:
        ev1 = {n: json.loads(z.read(n)) for n in z.namelist() if n.startswith("contested/")}

    # ---- Part A: same 50 references as the first round (pre-adjudication drafts)
    a_head = ["Task", "Type", "Source", "Brief (what the agents saw)", "Reference reading (as drafted)",
              "Oracle answers (clarify tasks)", "Admissible readings (as drafted)",
              "Reference OK? (Y/N)", "Source OK? (Y/N/n.a.)", "If N: correct reading", "Extra admissible reading", "Comment"]
    a_rows = []
    for r in b1[1:]:
        row = dict(zip(bh, r))
        t = row["Task"]
        i = pre[t]
        a_rows.append([t, i["expected"], i.get("source", ""), (ROOT / "briefs" / f"{t}.md").read_text(encoding="utf-8").strip(),
                       neutral(i.get("spec")), "\n".join(f"{o['about']}: {o['answer']}" for o in i.get("oracle", [])),
                       "\n".join(a["reading"] for a in i.get("admissible", [])), "", "", "", "", ""])

    # ---- Part B: the same 41 contested records, shuffled, neutral ids
    recs = [dict(zip(ch, r)) for r in c1[1:]]
    order = list(range(len(recs)))
    random.Random(SEED).shuffle(order)
    b_head = ["ID", "Task", "Type", "Brief (what the agents saw)", "Reference reading (as drafted)", "Fields that differ",
              "Candidate specification (neutral form)", "Compliance excess under reference BCs (0% = no loss)",
              "Mass ratio (candidate / reference)", "Verdict (W/A)", "If A: the reading", "Comment"]
    b_rows, key, cands = [], {}, {}
    for k, j in enumerate(order, 1):
        rec = recs[j]
        cid = f"C{k:02d}"
        ev = ev1[rec["Evidence file"]]
        t = rec["Task"]
        key[cid] = {"first_round_row": int(rec["#"]), "task": t, "condition": rec["Condition"],
                    "record": ev["record"], "first_round_verdict": rec["Verdict (W/A)"]}
        cons = ev.get("consequence") or {}
        cands[f"candidates/{cid}.json"] = {"id": cid, "task": t, "fields_that_differ": ev["fields_that_differ"],
                                           "candidate_spec": strip_spec(ev["agent_spec"]),
                                           "reference_spec": strip_spec(pre[t].get("spec")),
                                           "consequence": {"compliance_excess": cons.get("value"),
                                                           "mass_ratio": cons.get("mass_ratio")}}
        b_rows.append([cid, t, rec["Type"], (ROOT / "briefs" / f"{t}.md").read_text(encoding="utf-8").strip(),
                       neutral(pre[t].get("spec")), rec["Fields that differ"], neutral(ev["agent_spec"]),
                       rec["Compliance excess under intended BCs (0% = no loss)"] or "",
                       rec["Mass ratio (agent / intended)"] or "", "", "", ""])

    wb = Workbook()
    ws = wb.active
    ws.title = "Start here"
    ws.column_dimensions["A"].width = 120
    for line in README.splitlines():
        ws.append([line.replace("**", "").replace("`", "")])
        if line.startswith("#"):
            ws.cell(row=ws.max_row, column=1).font = Font(bold=True, size=13 if line.startswith("# ") else 11)
    write_sheet(wb.create_sheet("A_References"), a_head, a_rows, [20, 9, 18, 60, 60, 45, 35, 11, 11, 40, 40, 35],
                {"Reference OK? (Y/N)", "Source OK? (Y/N/n.a.)", "If N: correct reading", "Extra admissible reading", "Comment"},
                {"Reference OK? (Y/N)": ["Y", "N"], "Source OK? (Y/N/n.a.)": ["Y", "N", "n.a."]})
    write_sheet(wb.create_sheet("B_Candidates"), b_head, b_rows, [6, 22, 9, 55, 55, 18, 55, 16, 10, 10, 40, 35],
                {"Verdict (W/A)", "If A: the reading", "Comment"}, {"Verdict (W/A)": ["W", "A"]})
    p = wb.create_sheet("Progress")
    p.column_dimensions["A"].width = 44
    na, nb = len(a_rows) + 1, len(b_rows) + 1
    for label, f in [("References checked", f'=COUNTA(A_References!H2:H{na})&" / {len(a_rows)}"'),
                     ("  marked N", f'=COUNTIF(A_References!H2:H{na},"N")'),
                     ("Candidates judged", f'=COUNTA(B_Candidates!J2:J{nb})&" / {len(b_rows)}"'),
                     ("  W", f'=COUNTIF(B_Candidates!J2:J{nb},"W")'),
                     ("  A", f'=COUNTIF(B_Candidates!J2:J{nb},"A")')]:
        p.append([label, f])

    OUT.mkdir(parents=True, exist_ok=True)
    xlsx = OUT / "ADJUDICATION_SHEET_R2.xlsx"
    wb.save(xlsx)
    (OUT / "key_r2.json").write_text(json.dumps({"seed": SEED, "first_round_sheet": FIRST.name, "items": key},
                                                indent=1, ensure_ascii=False), encoding="utf-8")
    zp = OUT / f"AgenticTO_adjudication_round2_{time.strftime('%Y-%m-%d')}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(xlsx, "ADJUDICATION_SHEET_R2.xlsx")
        z.writestr("README.md", README)
        for t in pre:
            z.write(ROOT / "briefs" / f"{t}.md", f"briefs/{t}.md")
            z.writestr(f"reference/{t}.json", json.dumps({k: v for k, v in pre[t].items()
                                                          if k in ("task", "expected", "source", "spec", "oracle", "admissible")},
                                                         indent=1, ensure_ascii=False))
        for name, c in cands.items():
            z.writestr(name, json.dumps(c, indent=1, ensure_ascii=False))
    print(f"{len(a_rows)} references, {len(b_rows)} candidates -> {zp} ({zp.stat().st_size / 1e3:.0f} kB); key -> {OUT / 'key_r2.json'}")


if __name__ == "__main__":
    main()
