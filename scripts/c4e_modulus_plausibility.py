"""C4e: would a deterministic modulus-plausibility check, which the rule set does not contain, flag wrong readings?

Converts each delivered specification's Young's modulus to MPa using its declared length and force units and flags
it when the result lies outside 0.1 GPa to 1000 GPa (1e2 to 1e6 MPa), a range that covers engineering polymers to
diamond. Reads the first-repeat X2 records of both providers; no LLM calls, no solves.

Output: results/C4/modulus_plausibility.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_tables import x2  # noqa: E402

LEN_M = {"mm": 1e-3, "m": 1.0, "cm": 1e-2, "in": 0.0254, "ft": 0.3048}
FORCE_N = {"N": 1.0, "kN": 1e3, "MN": 1e6, "lbf": 4.4482216152605}
LO, HI = 1e2, 1e6


def e_mpa(spec):
    u = spec.get("units") or {}
    lf, ff = LEN_M.get(u.get("length", "mm")), FORCE_N.get(u.get("force", "N"))
    E = (spec.get("material") or {}).get("E")
    if lf is None or ff is None or E is None:
        return None
    return float(E) * ff / lf ** 2 / 1e6   # spec units force/length^2 -> Pa -> MPa


def provider_runs():
    yield "primary", "T", x2("T")
    yield "primary", "G", x2("G")
    yield "primary", "GC", x2("GC")
    sec = ROOT / "results/X2/claude_high"
    for cond in sorted(p.name for p in sec.iterdir() if p.is_dir()):
        runs = {}
        for f in sorted((sec / cond).glob("b*_r1.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            runs[d.get("task", f.stem[:-3])] = d
        yield "second", cond, runs


out = {"rule": f"E converted to MPa outside [{LO:g}, {HI:g}]", "groups": []}
for prov, cond, runs in provider_runs():
    rows = []
    for task, d in sorted(runs.items()):
        if d.get("expected") == "refuse" or d.get("refused") or not d.get("spec"):
            continue
        e = e_mpa(d["spec"])
        rows.append({"task": task, "correct": bool(d.get("correct_spec")), "E_MPa": e,
                     "flag": e is not None and not (LO <= e <= HI)})
    g = {"provider": prov, "condition": cond, "delivered": len(rows),
         "wrong": sum(not r["correct"] for r in rows), "right": sum(r["correct"] for r in rows),
         "flag_wrong": sum(r["flag"] and not r["correct"] for r in rows),
         "flag_right": sum(r["flag"] and r["correct"] for r in rows),
         "flagged": [(r["task"], r["E_MPa"], r["correct"]) for r in rows if r["flag"]], "rows": rows}
    out["groups"].append(g)
    print(f"{prov} {cond}: delivered {g['delivered']}; flagged wrong {g['flag_wrong']}/{g['wrong']}, "
          f"right {g['flag_right']}/{g['right']}; {g['flagged']}")
dst = ROOT / "results/C4/modulus_plausibility.json"
dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
print("->", dst)
