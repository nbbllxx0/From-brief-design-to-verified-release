"""C4c: which specification faults do the deterministic rules catch?

Each reference specification in the suite is checked unchanged and then with one injected fault. Admissibility faults
make the problem ill-posed (zero force, load on the supported nodes, load inside a keep-out region, supports that leave
rigid-body motion free); two defects the paper names as outside the rule set are injected too (a load straddling a
keep-out face, a modulus slip of 1000); intent faults keep it well posed but change what is solved (force sign flipped, a kN/N unit
slip, an invented keep-in pad, a changed volume fraction, a support moved to the opposite face). A fault counts as
caught when the faulted spec has an error-severity finding that the unfaulted spec does not. Checked on the grid the
C4 solves use (shortest side / 20, or the spec's own h where that does not divide every side). CPU only.

    python -B scripts/c4c_fault_matrix.py
"""
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "C4"
OPP = {"x-": "x+", "x+": "x-", "y-": "y+", "y+": "y-", "z-": "z+", "z+": "z-"}


def zero_force(d):
    for ld in d["loads"]:
        ld["force"] = [0.0, 0.0, 0.0]
    return d


def load_on_support(d):
    s = next((s for s in d["supports"] if len(s.get("fix", "xyz")) == 3), None)
    if s is None:
        return None
    d["loads"][0]["region"] = copy.deepcopy(s["region"])
    return d


def load_in_void(d):
    v = next((p for p in d.get("passive", []) if p["kind"] == "void"), None)
    if v is None:
        return None
    r = copy.deepcopy(v["region"])
    if r["kind"] == "box":
        c = [(a + b) / 2 for a, b in zip(r["min"], r["max"])]
        w = [(b - a) / 4 for a, b in zip(r["min"], r["max"])]
        r["min"] = [ci - wi for ci, wi in zip(c, w)]
        r["max"] = [ci + wi for ci, wi in zip(c, w)]
    else:
        r["radius"] /= 2
        a, b = r["span"]
        r["span"] = [a + (b - a) / 4, b - (b - a) / 4]
    d["loads"][0]["region"] = r
    return d


def load_partly_in_void(d):
    v = next((p for p in d.get("passive", []) if p["kind"] == "void"), None)
    if v is None:
        return None
    r = copy.deepcopy(v["region"])
    L = d["domain"]["size"]
    if r["kind"] == "box":
        c = [(a + b) / 2 for a, b in zip(r["min"], r["max"])]
        w = [(b - a) / 4 for a, b in zip(r["min"], r["max"])]
        k = max(range(3), key=lambda i: min(r["max"][i] + w[i], L[i]) - r["max"][i])  # axis with room outside
        if r["max"][k] >= L[k]:
            return None                         # the keep-out reaches the domain boundary: nothing to straddle
        c[k] = r["max"][k]                      # straddle the keep-out face: half inside, half outside
        r["min"] = [max(ci - wi, 0.0) for ci, wi in zip(c, w)]
        r["max"] = [min(ci + wi, Li) for ci, wi, Li in zip(c, w, L)]
    else:
        a, b = r["span"]
        if b >= L["xyz".index(r["axis"])]:
            return None                         # the keep-out runs through the domain: nothing to straddle
        r["span"] = [b - (b - a) / 4, min(b + (b - a) / 4, L["xyz".index(r["axis"])])]
        r["radius"] /= 2
    d["loads"][0]["region"] = r
    return d


def modulus_slip(d):
    d.setdefault("material", {})
    d["material"]["E"] = d["material"].get("E", 210e3) / 1000   # MPa value entered as GPa
    return d


def free_rigid(d):
    s = copy.deepcopy(d["supports"][0])
    s["fix"] = [s.get("fix", ["x", "y", "z"])[0]]
    d["supports"] = [s]
    return d


def flip_sign(d):
    for ld in d["loads"]:
        ld["force"] = [-f for f in ld["force"]]
    return d


def unit_slip(d):
    for ld in d["loads"]:
        ld["force"] = [1000.0 * f for f in ld["force"]]
    return d


def invented_pad(d):
    L = d["domain"]["size"]
    c = [x / 2 for x in L]
    w = [x / 10 for x in L]
    d.setdefault("passive", []).append({"id": "fault_pad", "kind": "solid",
                                        "region": {"kind": "box", "min": [ci - wi for ci, wi in zip(c, w)],
                                                   "max": [ci + wi for ci, wi in zip(c, w)]}})
    return d


def volume_change(d):
    v = d["volume_fraction"]
    d["volume_fraction"] = round(v * 1.5, 4) if v * 1.5 < 0.9 else round(v / 2, 4)
    return d


def support_opposite(d):
    r = d["supports"][0]["region"]
    if r["kind"] not in ("face", "patch"):
        return None
    r["face"] = OPP[r["face"]]
    return d


FAULTS = [("admissibility", "zero force", zero_force), ("admissibility", "load on supported nodes", load_on_support),
          ("admissibility", "load inside keep-out", load_in_void), ("admissibility", "rigid-body motion free", free_rigid),
          ("outside rule set", "load straddling a keep-out face", load_partly_in_void),
          ("outside rule set", "modulus divided by 1000 (GPa/MPa slip)", modulus_slip),
          ("intent", "force sign flipped", flip_sign), ("intent", "kN/N unit slip (x1000)", unit_slip),
          ("intent", "invented keep-in pad", invented_pad), ("intent", "volume fraction changed", volume_change),
          ("intent", "support on opposite face", support_opposite)]


def errors(spec, h):
    from agentto.rules import check
    return {f.code for f in check(spec, h=h).findings if f.severity == "error"}


def main():
    from agentto.spec import Spec

    rows = []
    for p in sorted((ROOT / "briefs/intended").glob("b*.json")):
        d0 = json.loads(p.read_text(encoding="utf-8"))
        sd = d0["spec"]
        spec = Spec.model_validate(sd)
        h = min(spec.domain.size) / 20
        if any(abs(L / h - round(L / h)) > 1e-9 for L in spec.domain.size):
            h = spec.resolution.h or min(spec.domain.size) / spec.resolution.cells_short_edge
        base = errors(spec, h)
        for cls, name, f in FAULTS:
            d = f(copy.deepcopy(sd))
            if d is None:
                continue
            try:
                new = errors(Spec.model_validate(d), h) - base
                row = {"task": p.stem, "expected": d0.get("expected"), "class": cls, "fault": name,
                       "base_errors": sorted(base), "new_errors": sorted(new), "caught": bool(new)}
            except Exception as ex:                     # noqa: BLE001  (schema rejection also catches the fault)
                row = {"task": p.stem, "expected": d0.get("expected"), "class": cls, "fault": name,
                       "base_errors": sorted(base), "new_errors": [type(ex).__name__], "caught": True}
            rows.append(row)
        print(p.stem, sorted(base), flush=True)

    summary = defaultdict(lambda: {"applied": 0, "caught": 0, "applied_base_clean": 0, "caught_base_clean": 0,
                                   "codes": defaultdict(int)})
    for r in rows:
        s = summary[(r["class"], r["fault"])]
        s["applied"] += 1
        s["caught"] += r["caught"]
        if not r["base_errors"]:                        # a refuse brief's reference already fails the rules
            s["applied_base_clean"] += 1
            s["caught_base_clean"] += r["caught"]
        for c in r["new_errors"]:
            s["codes"][c] += 1
    table = [{"class": k[0], "fault": k[1], **{n: v[n] for n in ("applied", "caught", "applied_base_clean",
                                                                  "caught_base_clean")}, "codes": dict(v["codes"])}
             for k, v in summary.items()]
    for t in table:
        print(json.dumps(t))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fault_matrix.json").write_text(json.dumps({"summary": table, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
