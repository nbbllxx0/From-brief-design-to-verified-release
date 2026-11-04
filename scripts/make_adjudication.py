"""Human adjudication pack: everything the author should check by hand before the numbers go into the paper.

1. adjudication_briefs.csv: one row per brief. The brief, the intended reading in short form, the oracle's
   answers, the admissible alternatives, and every condition's outcome. Columns `verdict_*` are for the author.
2. adjudication_contested.md: every silent wrong spec (valid, not refused, scored wrong) in repeat 1 of every
   condition and provider, with the field that differs and the agent's own closing note. These cases decide the
   "silent wrong" counts; the author marks each as truly wrong or acceptable.

    python -B scripts/make_adjudication.py      writes paper/adjudication/
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "X2"
OUT = ROOT / "paper" / "adjudication"
CONDS = [("R", "R"), ("T", "T"), ("G", "G"), ("GC", "GC"), ("GCV", "GCV"),
         ("claude_high/T", "Claude T"), ("claude_high/GC", "Claude GC"), ("claude_high/GCV", "Claude GCV")]


def runs(sub):
    return {d["task"]: d for d in (json.loads(f.read_text(encoding="utf-8")) for f in sorted((R / sub).glob("b*_r1.json")))}


def short(spec):
    if not spec:
        return ""
    s = [f"{spec['domain']['size']} {spec['units']['length']}", f"vf {spec.get('volume_fraction')}",
         f"{spec['material'].get('name', '')} E {spec['material'].get('E')}"]
    for it in spec.get("supports", []):
        s.append(f"support {it['id']}: {region(it['region'])} fix {it.get('fix', ['x', 'y', 'z'])}")
    for it in spec.get("loads", []):
        s.append(f"load {it['id']}: {region(it['region'])} F {it.get('force')}")
    for it in spec.get("passive", []):
        s.append(f"{it['kind']} {it['id']}: {region(it['region'])}")
    return "; ".join(s)


def region(r):
    k = r["kind"]
    if k == "face":
        return f"face {r['face']}"
    if k == "patch":
        return f"patch {r['face']} c{r['center']} s{r['size']}"
    if k == "edge":
        return f"edge along {r['axis']} at {r['at']}"
    if k == "box":
        return f"box {r['min']}–{r['max']}"
    if k == "cylinder":
        return f"cylinder {r.get('axis')} c{r.get('center')} r{r.get('radius')}"
    return k


def failed_fields(d):
    m = d.get("match") or {}
    f = [k for k, v in (m.get("fields") or {}).items() if v is False]
    if m.get("units_same") is False:
        f.append("units")
    return f


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    all_runs = {lab: runs(sub) for sub, lab in CONDS}
    ints = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in sorted((ROOT / "briefs" / "intended").glob("*.json"))}
    rows = []
    for t, i in ints.items():
        brief = (ROOT / "briefs" / f"{t}.md").read_text(encoding="utf-8").strip()
        row = {"task": t, "expected": i["expected"], "source": i.get("source", ""),
               "brief": " ".join(brief.split()),
               "intended": short(i.get("spec")),
               "oracle": " | ".join(f"{o['about']}: {o['answer']}" for o in i.get("oracle", [])),
               "admissible": " | ".join(a["reading"] for a in i.get("admissible", [])),
               "notes": i.get("notes", "")}
        for lab, rs in all_runs.items():
            d = rs.get(t)
            row[lab] = "" if d is None else ("right" if d.get("behaviour_ok") else
                                              "refused" if d.get("refused") else
                                              "invalid" if not d.get("schema_valid") else "WRONG")
        row.update(verdict_intended_ok="", verdict_source_ok="", verdict_comment="")
        rows.append(row)
    with (OUT / "adjudication_briefs.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    md = ["# Contested cases: every silent wrong spec in repeat 1", "",
          "A silent wrong spec is valid, not refused, and not the intended or an admissible reading. These cases "
          "set the paper's silent-wrong counts (C2.3, C2.6). For each, mark **W** (truly wrong: an engineer would "
          "get a different part) or **A** (acceptable: add it to `admissible` in `briefs/intended/<task>.json`, "
          "then rerun `scripts/x2_score.py` and `scripts/make_ledger.py`).", "",
          "| # | Task | Condition | Fields that differ | Agent's stated assumptions (end of its final message) | Verdict W/A |",
          "|---|---|---|---|---|---|"]
    n = 0
    for lab, rs in all_runs.items():
        for t, d in sorted(rs.items()):
            if d.get("behaviour_ok") or d.get("refused") or not d.get("schema_valid") or d.get("expected") == "refuse":
                continue
            n += 1
            fin = " ".join((d.get("final") or "").split())[-380:].replace("|", "/")
            md.append(f"| {n} | {t} ({d.get('expected')}) | {lab} | {', '.join(failed_fields(d)) or '–'} | {fin} | |")
    md += ["", "Wrong refusals and invalid specs are not listed: they are not silent (the user sees them)."]
    (OUT / "adjudication_contested.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} brief rows and {n} contested cases to {OUT}")


if __name__ == "__main__":
    main()
