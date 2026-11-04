"""Compare the second-round adjudication (S.Y.) with the first (J.W.), and bound the scores.

    python -B scripts/compare_adjudications.py <filled ADJUDICATION_SHEET_R2.xlsx>

Writes paper/adjudication/round2/agreement.md and agreement.json:
  Part A  agreement on the 50 reference readings (Y/N), Cohen's kappa, disagreements
  Part B  agreement on the 41 contested records (W/A), Cohen's kappa, disagreements with their record
  Bounds  right counts per condition under J.W. (the paper), S.Y. alone, both-must-accept and either-accepts,
          and the exact paired (McNemar) p for T->G, G->G+C, T->G+C and the two provider pairs under each rule.
Part A disagreements change references, so they are listed for re-encoding, not scored here.
"""
import json
import sys
from math import comb
from pathlib import Path

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_adjudication import CONDS, runs  # noqa: E402

FIRST = ROOT / "paper" / "adjudication" / "ADJUDICATION_SHEET_2026-10-31.xlsx"
OUT = ROOT / "paper" / "adjudication" / "round2"
SUB = {lab: sub for sub, lab in CONDS}
PAIRS = (("T", "G"), ("G", "GC"), ("T", "GC"), ("T", "Claude T"), ("GC", "Claude GC"))


def kappa(a, b, cats):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def mcnemar(x_ok, y_ok):
    """Exact two-sided paired test on two {task: bool} maps over the same tasks."""
    b = sum(1 for t in x_ok if x_ok[t] and not y_ok[t])
    c = sum(1 for t in x_ok if y_ok[t] and not x_ok[t])
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n
    return b, c, min(1.0, p)


def sheet(path, name):
    rows = list(load_workbook(path, data_only=True)[name].iter_rows(values_only=True))
    return [dict(zip(rows[0], r)) for r in rows[1:]]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    filled = Path(sys.argv[1])
    key = json.loads((OUT / "key_r2.json").read_text(encoding="utf-8"))["items"]
    a1 = {r["Task"]: (r["Intended OK? (Y/N)"] or "").strip().upper() for r in sheet(FIRST, "Briefs")}
    a2 = {r["Task"]: ((r["Reference OK? (Y/N)"] or "").strip().upper(), r.get("If N: correct reading"), r.get("Comment"))
          for r in sheet(filled, "A_References")}
    b2 = {r["ID"]: ((r["Verdict (W/A)"] or "").strip().upper(), r.get("If A: the reading"), r.get("Comment"))
          for r in sheet(filled, "B_Candidates")}
    missing = [t for t, v in a2.items() if v[0] not in ("Y", "N")] + [i for i, v in b2.items() if v[0] not in ("W", "A")]
    if missing:
        sys.exit(f"{len(missing)} rows not filled yet, e.g. {missing[:5]}")

    tasks = sorted(a2)
    pa, ka = kappa([a1[t] for t in tasks], [a2[t][0] for t in tasks], "YN")
    dis_a = [{"task": t, "first": a1[t], "second": a2[t][0], "second_reading": a2[t][1], "comment": a2[t][2]}
             for t in tasks if a1[t] != a2[t][0]]
    ids = sorted(b2)
    v1 = [key[i]["first_round_verdict"].strip().upper() for i in ids]
    v2 = [b2[i][0] for i in ids]
    pb, kb = kappa(v1, v2, "WA")
    dis_b = [{"id": i, **key[i], "second": b2[i][0], "second_reading": b2[i][1], "comment": b2[i][2]}
             for i, x, y in zip(ids, v1, v2) if x != y]

    # current outcomes (J.W.-adjudicated references) and the contested records' verdicts by rule
    ok = {lab: {t: bool(d.get("behaviour_ok")) for t, d in runs(sub).items()} for sub, lab in CONDS}
    rules = {"J.W. (paper)": lambda x, y: x == "A", "S.Y. alone": lambda x, y: y == "A",
             "both must accept": lambda x, y: x == "A" and y == "A", "either accepts": lambda x, y: x == "A" or y == "A"}
    res = {}
    for name, rule in rules.items():
        o = {lab: dict(m) for lab, m in ok.items()}
        for i, x, y in zip(ids, v1, v2):
            k = key[i]
            o[k["condition"]][k["task"]] = rule(x, y)
        res[name] = {"right": {lab: sum(m.values()) for lab, m in o.items()},
                     "paired": {f"{a}->{b}": mcnemar(o[a], o[b]) for a, b in PAIRS}}

    out = {"part_a": {"n": len(tasks), "agreement": pa, "kappa": ka, "disagreements": dis_a},
           "part_b": {"n": len(ids), "agreement": pb, "kappa": kb, "disagreements": dis_b}, "bounds": res}
    (OUT / "agreement.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    L = ["# Second adjudication: agreement and score bounds", "",
         f"Part A references: {len(tasks) - len(dis_a)}/{len(tasks)} agree ({100 * pa:.0f}%), kappa {ka:.2f}",
         f"Part B contested records: {len(ids) - len(dis_b)}/{len(ids)} agree ({100 * pb:.0f}%), kappa {kb:.2f}", "",
         "| Rule | " + " | ".join(ok) + " | " + " | ".join(f"{x}->{y} (b/c, p)" for x, y in PAIRS) + " |",
         "|---|" + "---|" * (len(ok) + len(PAIRS))]
    for name, r in res.items():
        L.append(f"| {name} | " + " | ".join(str(r["right"][lab]) for lab in ok) + " | " +
                 " | ".join(f"{b}/{c}, {p:.3g}" for b, c, p in r["paired"].values()) + " |")
    L += ["", "## Part A disagreements (need re-encoding)", ""] + \
         [f"- {d['task']}: first {d['first']}, second {d['second']}; {d['second_reading'] or ''} {d['comment'] or ''}" for d in dis_a]
    L += ["", "## Part B disagreements", ""] + \
         [f"- {d['id']} {d['task']} ({d['condition']}): first {d['first_round_verdict']}, second {d['second']}; "
          f"{d['second_reading'] or ''} {d['comment'] or ''}" for d in dis_b]
    (OUT / "agreement.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:12 + len(res)]))


if __name__ == "__main__":
    main()
