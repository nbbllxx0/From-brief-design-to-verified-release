"""X2 condition R: a deterministic rule-based brief parser (no LLM), the baseline the plan names.

It reads the brief language with regular expressions: envelope sizes and axes, units, material (named alloy or
E / nu), volume fraction, mesh size; supports on whole faces ("the x = 0 face is fully fixed", "bottom face (y = 0)
bolted ... over its full area", "both end faces"); loads as a force with a direction on a pad at the centre of a face,
on a whole face, or along a bottom edge; keep-in / keep-out boxes given as "x from a to b, y from c to d". Anything it
cannot read (washers, cylinders, edges at bearings, symmetry, corner pads, ambiguity) makes it refuse: it never
guesses. It was written after the suite existed, so its score is an optimistic bound for a rule parser.

    python -B scripts/x2_rules_baseline.py
Writes results/X2/R/<task>_r1.json, scored like the LLM conditions (bench.match_task, expected behaviour).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentto.bench import match_task  # noqa: E402
from agentto.rules import check  # noqa: E402
from agentto.spec import Spec  # noqa: E402

NUM = r"(\d+(?:\.\d+)?)"
LEN = {"mm": 1.0, "m": 1000.0, "in": 25.4}
MATERIALS = [("7075", 71700.0, 0.33), ("6061", 69000.0, 0.33), ("6082", 70000.0, 0.33), ("aluminium", 69000.0, 0.33),
             ("ti-6al-4v", 114000.0, 0.34), ("cast iron", 110000.0, 0.26), ("peek", 3600.0, 0.38),
             ("pa12", 1700.0, 0.4), ("steel", 210000.0, 0.3)]
AX = "xyz"


class Unreadable(Exception):
    pass


def units(t):
    s = 1.0
    if re.search(r"work in metres", t):
        s = 1000.0
    elif re.search(r"work in inches", t):
        s = 25.4
    f = 1000.0 if re.search(r"kilonewtons", t) else 4.4482216152605 if "pounds-force" in t else 1.0
    return s, f


def envelope(t, s):
    m = re.search(NUM + r"\s*x\s*" + NUM + r"\s*x\s*" + NUM + r"\s*(mm|m|in)\b", t)
    if m:
        return [float(m.group(i)) * LEN[m.group(4)] for i in (1, 2, 3)]
    m = re.search(r"(?:a|an) " + NUM + r"\s*(mm|m) cube", t)
    if m:
        return [float(m.group(1)) * LEN[m.group(2)]] * 3
    sizes = {}
    for v, u, a in re.findall(NUM + r"\s*(mm|m|in)\b[a-z ]{0,30}?\(([xyz])\b", t):
        sizes.setdefault(a, float(v) * LEN[u])
    m = re.search(NUM + r"\s*(mm|m) long \(x\) with an? " + NUM + r" x " + NUM + r" mm square cross-section", t)
    if m:
        sizes = {"x": float(m.group(1)) * LEN[m.group(2)], "y": float(m.group(3)), "z": float(m.group(4))}
    if set(sizes) != set(AX):
        raise Unreadable("envelope")
    return [sizes[a] for a in AX]


def up_axis(t):
    m = re.search(r"\b([xyz]) (?:pointing )?up\b", t) or re.search(r"\(([xyz]),? up\)", t) or \
        re.search(r"\(negative ([xyz])\)", t)
    if not m:
        raise Unreadable("no up axis")
    return m.group(1)


def material(t):
    E = re.search(r"e = " + NUM + r" (gpa|mpsi)", t)
    nu = re.search(r"nu = " + NUM, t)
    for key, e, n in MATERIALS:
        if key in t:
            out = {"name": key, "E": e, "nu": n}
            break
    else:
        if not E:
            raise Unreadable("material")
        out = {"name": "material", "E": 0.0, "nu": 0.3}
    if E:
        out["E"] = float(E.group(1)) * (1000.0 if E.group(2) == "gpa" else 6894.757 * 1.0)
        if E.group(2) == "mpsi":
            out["E"] = float(E.group(1)) * 1e6 * 0.006894757
    if nu:
        out["nu"] = float(nu.group(1))
    return out


def resolution(t, s):
    m = re.search(NUM + r"\s*(mm|m|in) elements", t)
    if m:
        return {"h": float(m.group(1)) * LEN[m.group(2)]}
    m = re.search(r"(\d+) elements across the shortest side", t)
    if m:
        return {"cells_short_edge": int(m.group(1))}
    raise Unreadable("mesh size")


def face_of(phrase, L, up):
    """'x = 0', 'x = 160', 'bottom', 'top' -> face id."""
    m = re.search(r"([xyz]) = " + NUM + r"( m\b)?", phrase)
    if m:
        a = m.group(1)
        v = float(m.group(2)) * (1000.0 if m.group(3) else 1.0)
        i = AX.index(a)
        if abs(v) < 1e-9:
            return a + "-"
        if abs(v - L[i]) < 1e-6 * max(1, L[i]):
            return a + "+"
        raise Unreadable("face position inside the domain")
    if "bottom" in phrase:
        return up + "-"
    if "top" in phrase:
        return up + "+"
    raise Unreadable("face")


def supports(t, L, up):
    out = []
    m = re.search(r"both end faces \(x = 0 and x = " + NUM + r"\)[^.]*?(fixed|clamped)", t)
    if m:
        return [{"id": "end_0", "region": {"kind": "face", "face": "x-"}},
                {"id": "end_1", "region": {"kind": "face", "face": "x+"}}]
    for sent in re.split(r"(?<=\.)\s", t):
        if re.search(r"washer|bolts? (?:on|hold|fix)|screws|pads?\b|corner|column|feet|pin|shaft|spindle|edge|rail|"
                     r"roller|bearing|hinge|symmetry|patch", sent) and \
                re.search(r"fix|bolt|pin|rest|ride|hang|key|stand", sent):
            raise Unreadable("support other than a whole face")
        if re.search(r"fully fixed|fixed in all|welded|bolted|bolts to|clamped|bonded", sent) and \
                re.search(r"face|end\b|base", sent) and not re.search(r"load|push|pull|carr", sent):
            ph = re.search(r"(\(?[xyz] = " + NUM + r"( m)?\)?|bottom|top|back|base)", sent)
            if not ph:
                raise Unreadable("support face")
            p = ph.group(0)
            if p in ("back",):
                raise Unreadable("back face")
            if p == "base":
                p = "bottom"
            out.append({"id": f"support_{len(out) + 1}", "region": {"kind": "face", "face": face_of(p, L, up)}})
    if not out:
        raise Unreadable("no support")
    return out


DIR = {"downward": (1, -1), "down": (1, -1), "upward": (1, 1), "up": (1, 1)}


def direction(phrase, up):
    for w, (_, sgn) in DIR.items():
        if re.search(r"\b" + w + r"\b", phrase):
            v = [0.0, 0.0, 0.0]
            v[AX.index(up)] = sgn
            return v
    m = re.search(r"along ([+-])([xyz])", phrase) or re.search(r"\(([+-])([xyz])\)", phrase)
    if m:
        v = [0.0, 0.0, 0.0]
        v[AX.index(m.group(2))] = 1.0 if m.group(1) == "+" else -1.0
        return v
    raise Unreadable("load direction")


def loads(t, L, up, fs):
    out = []
    for sent in re.split(r"(?<=\.)\s", t):
        fm = re.findall(NUM + r"\s*(kn|n)\b", sent)
        if not fm or not re.search(r"load|push|pull|carr|hang|press|thrust|weight", sent):
            continue
        if re.search(r" and " + NUM + r"\s*(kn|n)\b", sent) or "torque" in sent or "angle" in sent or "rises" in sent:
            raise Unreadable("combined or derived load")
        F = float(fm[0][0]) * (1000.0 if fm[0][1] == "kn" else 1.0)          # each force carries its unit
        d = direction(sent, up)
        force = [F * x for x in d]
        pm = re.search(NUM + r" ?(mm|m)? x " + NUM + r" ?(mm|m) (?:pad|plate|bearing plate) (?:in the middle|at the "
                       r"centre) of the (?:free )?(?:end )?face \(?([xyz] = " + NUM + r"(?: m)?)", sent) or \
            re.search(NUM + r" ?(mm|m)? x " + NUM + r" ?(mm|m) (?:pad|plate) (?:in the middle|at the centre) of the "
                      r"(top|bottom) face", sent)
        if pm:
            g = pm.groups()
            un = LEN[g[3]]
            g = (float(g[0]) * un, float(g[2]) * un, g[4])
            face = face_of(g[2], L, up)
            a = AX.index(face[0])
            b, c = [i for i in range(3) if i != a]
            # sizes are given in the face's increasing axis order
            out.append({"id": f"load_{len(out) + 1}", "region": {"kind": "patch", "face": face,
                        "center": [L[b] / 2, L[c] / 2], "size": [float(g[0]), float(g[1])]}, "force": force})
            continue
        if re.search(r"(whole|entire|full) (top|deck|seat) (face|surface)|over the whole top", sent) or \
                re.search(r"spread uniformly over the whole top", sent):
            out.append({"id": f"load_{len(out) + 1}", "region": {"kind": "face", "face": up + "+"}, "force": force})
            continue
        em = re.search(r"along the (?:bottom|top) edge of the free end \(([xyz]) = " + NUM + r", ([xyz]) = " + NUM
                       + r"\)", sent)
        if em:
            a1, a2 = em.group(1), em.group(3)
            axis = next(a for a in AX if a not in (a1, a2))
            at = {a1: float(em.group(2)), a2: float(em.group(4))}
            out.append({"id": f"load_{len(out) + 1}", "region": {"kind": "edge", "axis": axis,
                        "at": [at[a] for a in AX if a != axis]}, "force": force})
            continue
        m = re.search(r"whole (?:free )?end face \(x = " + NUM, sent)
        if m:
            out.append({"id": f"load_{len(out) + 1}", "region": {"kind": "face", "face": face_of("x = " + m.group(1), L, up)},
                        "force": force})
            continue
        raise Unreadable("load region")
    if not out:
        raise Unreadable("no load")
    return out


def passive(t):
    out = []
    for sent in re.split(r"(?<=\.)\s", t):
        if not re.search(r"must (stay|be kept) (empty|solid|clear)|must be kept|must stay", sent):
            continue
        kind = "void" if re.search(r"empty|clear", sent) else "solid"
        rng = dict((a, (float(lo), float(hi))) for a, lo, hi in re.findall(r"([xyz]) from " + NUM + r" to " + NUM, sent))
        for a, lo in re.findall(r"([xyz]) > " + NUM, sent):
            rng[a] = (float(lo), None)
        if not rng or "diameter" in sent or "cylinder" in sent:
            raise Unreadable("passive region")
        out.append({"kind": kind, "rng": rng})
    return out


def parse(text: str) -> dict:
    t = " ".join(text.lower().split())
    s, fs = units(t)
    L = envelope(t, s)
    up = up_axis(t)
    vf = re.search(r"volume fraction " + NUM, t)
    if not vf:
        raise Unreadable("volume fraction")
    pas = []
    for i, p in enumerate(passive(t)):
        lo = [p["rng"].get(a, (0, L[k]))[0] for k, a in enumerate(AX)]
        hi = [L[k] if p["rng"].get(a, (0, L[k]))[1] is None else p["rng"].get(a, (0, L[k]))[1]
              for k, a in enumerate(AX)]
        pas.append({"id": f"passive_{i + 1}", "kind": p["kind"], "region": {"kind": "box", "min": lo, "max": hi}})
    spec = {"schema_version": "agentto.spec/v0", "name": "R", "units": {"length": "mm", "force": "N"},
            "domain": {"size": L}, "material": material(t), "volume_fraction": float(vf.group(1)),
            "supports": supports(t, L, up), "loads": loads(t, L, up, fs), "passive": pas,
            "resolution": resolution(t, s)}
    if "thinner than" in t:
        m = re.search(r"thinner than " + NUM + r" mm", t)
        spec["min_feature"] = float(m.group(1))
    Spec.model_validate(spec)
    return spec


def main():
    out = ROOT / "results" / "X2" / "R"
    out.mkdir(parents=True, exist_ok=True)
    n = ok = 0
    for p in sorted((ROOT / "briefs" / "intended").glob("*.json")):
        task = json.loads(p.read_text(encoding="utf-8"))
        brief = (ROOT / "briefs" / f"{p.stem}.md").read_text(encoding="utf-8")
        rec = {"task": p.stem, "cond": "R", "rep": 1, "expected": task["expected"], "questions": 0}
        try:
            spec = parse(brief)
            rec.update(spec=spec, refused=False, schema_valid=True, rules_ok=check(Spec.model_validate(spec)).ok)
            if not rec["rules_ok"]:                     # a rule-based tool stops on a failed check: a refusal
                rec["refused"] = True
            rec["match"] = match_task(spec, task)
        except Unreadable as ex:
            rec.update(spec=None, refused=True, schema_valid=False, reason=str(ex))
        except Exception as ex:                          # noqa: BLE001 -- a parse that builds an invalid spec
            rec.update(spec=None, refused=True, schema_valid=False, reason=f"invalid: {ex!r}"[:300])
        same = bool((rec.get("match") or {}).get("same"))
        rec["correct_spec"] = same
        rec["behaviour_ok"] = rec["refused"] if task["expected"] == "refuse" else (same and not rec["refused"])
        (out / f"{p.stem}_r1.json").write_text(json.dumps(rec, indent=1))
        n += 1
        ok += rec["correct_spec"]
        bad = [k for k, v in ((rec.get("match") or {}).get("fields") or {}).items() if not v]
        print(f"R {p.stem:28s} correct {str(rec['correct_spec']):5s} behaviour {str(rec['behaviour_ok']):5s} "
              f"{rec.get('reason', '')} {bad if bad else ''}")
    print(f"{ok}/{n} correct")


if __name__ == "__main__":
    main()
