"""C4f: a back-filled 'stated in the brief' tag per value, computed without the reference reading.

For every delivered specification of a solve or clarify brief, each numeric value that places a support, load or
passive region (centres, sizes, spans, radii, box corners) and each load force component is tagged 'traceable' when it
can be formed from the numbers written in the brief: a brief number (also x1000 and /1000 for unit changes), zero, a
domain size, half of either, or a sum or difference of two of these (each optionally halved). Anything else is tagged
'untraced', i.e. assumed by the agent. A specification is flagged when it carries any untraced value. The rule was
fixed before the wrong/right split was examined; it is a post hoc proxy for typed field provenance, not part of the
evaluated system. As a control it is applied to the reference readings too. No LLM calls, no solves.

Output: results/C4/value_traceability.json
"""
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_tables import x2  # noqa: E402

NUM = re.compile(r"(?<![A-Za-z])-?\d+(?:\.\d+)?")
TOL = 1e-6


def brief_numbers(task):
    text = (ROOT / "briefs" / f"{task}.md").read_text(encoding="utf-8")
    vals = {abs(float(m)) for m in NUM.findall(text)}
    return vals | {v * 1000 for v in vals} | {v / 1000 for v in vals}


def allowed(task, spec):
    base = brief_numbers(task) | {0.0} | {abs(float(x)) for x in (spec.get("domain") or {}).get("size", [])}
    base |= {v / 2 for v in base}
    out = set(base)
    for a, b in itertools.product(base, repeat=2):
        out |= {a + b, abs(a - b), a + b / 2, abs(a - b / 2)}
    return out


def region_values(r):
    vals = []
    for k in ("center", "size", "span", "min", "max", "point", "start", "end"):
        v = r.get(k)
        if isinstance(v, (list, tuple)):
            vals += [float(x) for x in v if isinstance(x, (int, float))]
    for k in ("radius", "offset", "width", "height", "thickness"):
        if isinstance(r.get(k), (int, float)):
            vals.append(float(r[k]))
    return vals


def spec_values(spec):
    vals = []
    for key in ("supports", "loads", "passive"):
        for item in spec.get(key) or []:
            vals += [("%s.%s" % (key, item.get("id")), v) for v in region_values(item.get("region") or {})]
            if key == "loads":
                vals += [("loads.%s.force" % item.get("id"), float(f)) for f in item.get("force") or [] if f]
    return vals


def untraced(task, spec):
    ok = sorted(allowed(task, spec))
    import bisect

    def near(v):
        v, tol = abs(v), TOL * max(1.0, abs(v))
        i = bisect.bisect_left(ok, v - tol)
        return i < len(ok) and ok[i] <= v + tol
    return [(f, v) for f, v in spec_values(spec) if not near(v)]


def summarize(rows):
    wrong = [r for r in rows if not r["correct"]]
    right = [r for r in rows if r["correct"]]
    return {"delivered": len(rows), "wrong": len(wrong), "right": len(right),
            "flag_wrong": sum(r["flag"] for r in wrong), "flag_right": sum(r["flag"] for r in right),
            "flag_right_solve": sum(r["flag"] for r in right if r["expected"] == "solve"),
            "right_solve": sum(r["expected"] == "solve" for r in right),
            "flag_right_clarify": sum(r["flag"] for r in right if r["expected"] == "clarify"),
            "right_clarify": sum(r["expected"] == "clarify" for r in right)}


out = {"rule": __doc__.split("\n\n")[1].replace("\n", " "), "conditions": {}, "references": {}}
for cond in ("T", "G", "GNR", "GC", "GCV", "claude_high/T", "claude_high/GC", "claude_high/GCV"):
    rows = []
    for task, d in sorted(x2(cond).items()):
        if d.get("expected") == "refuse" or d.get("refused") or not d.get("spec") or not d.get("schema_valid"):
            continue
        u = untraced(task, d["spec"])
        rows.append({"task": task, "expected": d["expected"], "correct": bool(d.get("correct_spec")),
                     "flag": bool(u), "untraced": u[:12]})
    s = summarize(rows)
    s["rows"] = rows
    out["conditions"][cond] = s
    print(f"{cond}: delivered schema-valid {s['delivered']}; flagged wrong {s['flag_wrong']}/{s['wrong']}, "
          f"right {s['flag_right']}/{s['right']} (solve {s['flag_right_solve']}/{s['right_solve']}, "
          f"clarify {s['flag_right_clarify']}/{s['right_clarify']})")
    if cond == "G":
        for r in rows:
            if not r["correct"] or (r["flag"] and r["expected"] == "solve"):
                print("   ", r["task"], r["expected"], "correct" if r["correct"] else "WRONG", r["untraced"][:4])

pool = [r for c, s in out["conditions"].items() for r in s["rows"]]
out["pooled_all_conditions"] = {k: v for k, v in summarize(pool).items()}
print("pooled over all conditions and providers:", out["pooled_all_conditions"])

ref_rows = []
for f in sorted((ROOT / "briefs/intended").glob("b*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    if d.get("expected") == "refuse":
        continue
    u = untraced(f.stem, d["spec"])
    ref_rows.append({"task": f.stem, "expected": d.get("expected"), "flag": bool(u), "untraced": u[:12]})
out["references"] = {"solve": [sum(r["flag"] for r in ref_rows if r["expected"] == "solve"),
                               sum(r["expected"] == "solve" for r in ref_rows)],
                     "clarify": [sum(r["flag"] for r in ref_rows if r["expected"] == "clarify"),
                                 sum(r["expected"] == "clarify" for r in ref_rows)], "rows": ref_rows}
print("references flagged: solve %d/%d, clarify %d/%d" % tuple(out["references"]["solve"] + out["references"]["clarify"]))
for r in ref_rows:
    if r["flag"] and r["expected"] == "solve":
        print("   ref", r["task"], r["untraced"][:4])
dst = ROOT / "results/C4/value_traceability.json"
dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
print("->", dst)
