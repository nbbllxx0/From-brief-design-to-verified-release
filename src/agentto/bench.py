"""Spec comparison for the agent benchmarks (X2): is a predicted spec the same problem as the intended one?

Two specs are compared on what the solver sees, not on how they are written: both are built on one common grid
and the fixed-DOF masks, nodal load vectors and passive masks are compared exactly, plus the scalar fields. So a
support written as a face or as a box covering that face counts as the same support, and ids or names do not
matter. Field-level results feed the X2 metrics; the consequence-weighted error is computed separately by solving.
"""
from __future__ import annotations

import numpy as np

from .problem import ResolutionError, build
from .spec import Spec


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


LENGTH_MM = {"mm": 1.0, "m": 1000.0, "in": 25.4}
FORCE_N = {"N": 1.0, "kN": 1000.0, "lbf": 4.4482216152605}
_LEN_KEYS = ("center", "size", "at", "span", "min", "max", "radius")


def _scale_region(r: dict, s: float) -> dict:
    """Lengths times s. Agents sometimes write numbers as strings ("400"), which the schema accepts; so do we."""
    return {k: (None if v is None else [float(x) * s for x in v] if isinstance(v, (list, tuple)) else float(v) * s)
            if k in _LEN_KEYS else v for k, v in r.items()}


def scale_item(item: dict, key: str, units: dict) -> dict:
    """One support / load / passive item, from `units` to mm and N."""
    s, f = LENGTH_MM[units.get("length", "mm")], FORCE_N[units.get("force", "N")]
    out = dict(item, region=_scale_region(item["region"], s)) if "region" in item else dict(item)
    if key == "loads" and "force" in out:
        out["force"] = [float(x) * f for x in out["force"]]
    return out


def normalized(spec: dict | Spec) -> Spec:
    """The same problem in mm and N: lengths, forces and E are converted; nothing else changes. Two specs written in
    different units are compared through this."""
    d = (spec if isinstance(spec, Spec) else Spec.model_validate(spec)).model_dump(mode="json")
    u = d["units"]
    s, f = LENGTH_MM[u["length"]], FORCE_N[u["force"]]
    if (s, f) == (1.0, 1.0):
        return Spec.model_validate(d)
    d["units"] = {"length": "mm", "force": "N"}
    d["domain"] = dict(d["domain"], size=[x * s for x in d["domain"]["size"]])
    d["material"] = dict(d["material"], E=d["material"]["E"] * f / s ** 2)
    for key in ("supports", "loads", "passive"):
        d[key] = [scale_item(x, key, u) for x in d[key]]
    if d["resolution"].get("h") is not None:
        d["resolution"] = dict(d["resolution"], h=d["resolution"]["h"] * s)
    if d.get("min_feature") is not None:
        d["min_feature"] *= s
    return Spec.model_validate(d)


def _same_grid(p: Spec, q: Spec) -> bool:
    from .problem import grid_for
    try:
        (hp, gp), (hq, gq) = grid_for(p), grid_for(q)
    except ResolutionError:
        return False
    return gp.shape == gq.shape and _rel(hp, hq) < 1e-6


def equivalent(pred: dict | Spec, intended: dict | Spec, h: float | None = None, e_rtol: float = 0.03) -> dict:
    """Field-by-field comparison on a common grid (default: the intended spec's resolution, coarsened if needed so
    that it fits both domains). Returns {"same": bool, "fields": {name: bool}, "notes": [...]}.

    Both specs are first converted to mm and N, so a spec written in other units is the same problem if it converts
    to it (`units_same` reports whether the units were written the same and does not count towards `same`); the
    resolution matches when both give the same grid. E matches within `e_rtol` (default 3%): a brief that names an
    alloy without a modulus admits any handbook value (6061: 68.9-69 GPa). `material_exact` reports the strict
    comparison and does not count towards `same`."""
    p0 = pred if isinstance(pred, Spec) else Spec.model_validate(pred)
    q0 = intended if isinstance(intended, Spec) else Spec.model_validate(intended)
    p, q = normalized(p0), normalized(q0)
    if h is not None:
        h *= LENGTH_MM[q0.units.length]
    f, notes = {}, []
    f["domain"] = all(_rel(a, b) < 1e-9 for a, b in zip(p.domain.size, q.domain.size))
    f["material"] = _rel(p.material.E, q.material.E) <= e_rtol and abs(p.material.nu - q.material.nu) < 0.02
    exact = _rel(p.material.E, q.material.E) < 1e-6 and abs(p.material.nu - q.material.nu) < 1e-9
    f["volume_fraction"] = abs(p.volume_fraction - q.volume_fraction) < 1e-6
    f["resolution"] = _same_grid(p, q)
    f["min_feature"] = (p.min_feature is None) == (q.min_feature is None) and \
        (q.min_feature is None or _rel(p.min_feature, q.min_feature) < 1e-6)
    if not f["domain"]:
        f.update(supports=False, loads=False, passive=False)
        return {"same": False, "fields": f, "notes": ["domains differ; regions not compared"]}
    from .problem import grid_for
    h0, _ = grid_for(q)
    bp = bq = None
    for hh in ([h] if h is not None else []) + [h0 * m for m in (1, 2, 4, 8, 16)]:
        try:
            bq, bp = build(q, h=hh), build(p, h=hh)
        except ResolutionError:
            continue
        if h is not None or bq.grid.n_elem <= 4e6:
            break
    if bp is None:
        return {"same": False, "fields": f, "notes": ["no common grid"]}
    notes.append(f"compared on h={bq.h:g}, grid {bq.grid.shape}")
    f["supports"] = bool(np.array_equal(bp.problem.fixed, bq.problem.fixed))
    lp, lq = bp.problem.load, bq.problem.load
    f["loads"] = bool(lp.idx.size == lq.idx.size and np.array_equal(lp.idx, lq.idx) and
                      np.allclose(lp.val, lq.val, rtol=1e-6, atol=1e-12 * max(1.0, float(np.abs(lq.val).max(
                          initial=0)))))
    z = np.zeros(bq.grid.n_elem, bool)
    f["passive"] = bool(np.array_equal(z if bp.solid is None else bp.solid, z if bq.solid is None else bq.solid) and
                        np.array_equal(z if bp.void is None else bp.void, z if bq.void is None else bq.void))
    return {"same": all(f.values()), "fields": f, "material_exact": exact, "units_same": p0.units == q0.units,
            "notes": notes}


def match_task(pred: dict | Spec, task: dict, h: float | None = None) -> dict:
    """`equivalent` against the task's intended spec, then against each admissible reading that counts (each carries
    its own `spec`). Returns the intended comparison plus "same" (any match) and "matched": "intended",
    "admissible:<i>" or None."""
    r = equivalent(pred, task["spec"], h)
    if r["same"]:
        return {**r, "matched": "intended"}
    for i, a in enumerate(task.get("admissible") or []):
        if a.get("counts") and a.get("spec") is not None and equivalent(pred, a["spec"], h)["same"]:
            return {**r, "same": True, "matched": f"admissible:{i}"}
    return {**r, "matched": None}


# ---- clarification oracle (X2 condition G+C) ----------------------------------------------------------------
# The oracle is a script that holds the intended reading; it is never an LLM. Given one question with 2-4 options,
# each a concrete reading (an item the spec would contain, and/or parameter values), it picks the option that agrees
# with the intended spec on what the solver sees, or answers "none of these" without revealing the intended value.

LIST_KEY = {"support": "supports", "load": "loads", "passive": "passive"}


def _grid_h(q: Spec, h: float | None):
    from .problem import grid_for
    h0, _ = grid_for(q)
    for hh in ([h] if h is not None else []) + [h0 * m for m in (1, 2, 4, 8, 16)]:
        try:
            if build(q, h=hh).grid.n_elem <= 2e6 or h is not None:
                return hh
        except ResolutionError:
            continue
    return None


def _item_signature(base: dict, key: str, item: dict, h: float):
    """What the solver sees of one item, built inside the intended spec's domain: fixed-DOF mask (supports),
    nodal load vector (loads), or element mask and kind (passive). None if the item does not build."""
    d = {**base, key: [item]}
    if key != "supports" and not d.get("supports"):
        d["supports"] = base["supports"]
    try:
        sp = build(Spec.model_validate(d), h=h)
    except Exception:                                   # noqa: BLE001 -- an unbuildable reading matches nothing
        return None
    if key == "supports":
        return ("s", sp.problem.fixed.tobytes())
    if key == "loads":
        ld = sp.problem.load
        return ("l", ld.idx.tobytes(), np.round(ld.val, 9).tobytes())
    m = sp.solid if item.get("kind") == "solid" else sp.void
    return ("p", item.get("kind"), None if m is None else np.packbits(m).tobytes())


def item_match(item: dict, kind: str, intended: dict | Spec, h: float | None = None) -> int | None:
    """Index of the intended item (of this kind) that is the same on the solver's grid as `item`, else None."""
    q = intended if isinstance(intended, Spec) else Spec.model_validate(intended)
    base = q.model_dump(mode="json")
    key = LIST_KEY[kind]
    hh = _grid_h(q, h)
    if hh is None:
        return None
    sig = _item_signature(base, key, item, hh)
    if sig is None:
        return None
    for i, x in enumerate(base[key]):
        if _item_signature(base, key, x, hh) == sig:
            return i
    return None


def params_match(params: dict, intended: dict | Spec, e_rtol: float = 0.03) -> bool:
    from .tools import spec_params
    q = (intended if isinstance(intended, Spec) else Spec.model_validate(intended)).model_dump(mode="json")
    for k, v in spec_params(params, q).items():
        t = q.get(k)
        if k == "material":
            if not isinstance(v, dict) or _rel(float(v.get("E", -1)), t["E"]) > e_rtol or \
                    abs(float(v.get("nu", t["nu"])) - t["nu"]) >= 0.02:
                return False
        elif k == "volume_fraction":
            if abs(float(v) - t) > 1e-6:
                return False
        elif k == "domain":
            size = v.get("size") if isinstance(v, dict) else v
            if any(_rel(float(a), b) > 1e-9 for a, b in zip(size, t["size"])):
                return False
        elif k == "min_feature":
            if (v is None) != (t is None) or (t is not None and _rel(float(v), t) > 1e-6):
                return False
        elif k == "units":
            if any(t.get(kk) != vv for kk, vv in v.items()):
                return False
        elif k == "resolution":
            a = {kk: vv for kk, vv in t.items() if vv is not None}
            b = {kk: vv for kk, vv in (v or {}).items() if vv is not None}
            if a.keys() != b.keys() or any(_rel(float(b[kk]), float(a[kk])) > 1e-6 for kk in a):
                return False
        elif v != t:
            return False
    return True


class Oracle:
    """Scripted answerer: `oracle(question, options)` -> {"choice": index | None, "label", "comment"}."""

    def __init__(self, intended: dict | Spec, h: float | None = None, script: list[dict] | None = None):
        """script: the task's scripted answers ([{about, answer}]); when no option matches, the oracle says so and
        then states these facts verbatim, as a user would (recorded as a miss)."""
        q = intended if isinstance(intended, Spec) else Spec.model_validate(intended)
        self.intended = normalized(q)                  # options are compared in mm and N, whatever units they use
        self.h = None if h is None else h * LENGTH_MM[q.units.length]
        self.script = script or []
        self.asked = 0

    def matches(self, opt: dict, cand: dict | None = None) -> bool:
        """`cand`, the draft with this option applied, says which units the option is written in (default mm, N)."""
        units = (cand or {}).get("units") or {}
        touched = False
        if opt.get("item") is not None:
            touched = True
            kind = opt.get("item_kind") or "load"
            item = scale_item(opt["item"], LIST_KEY.get(kind, "loads"), units)
            if item_match(item, kind, self.intended, self.h) is None:
                return False
        if opt.get("params"):
            touched = True
            try:
                c = normalized(cand).model_dump(mode="json") if cand is not None else None
            except Exception:                           # noqa: BLE001 -- an incomplete draft: compare as written
                c = None
            if c is not None:
                from .tools import spec_params
                keys = set(spec_params(opt["params"], c))
                if not params_match({k: c.get(k) for k in keys}, self.intended):
                    return False
            elif not params_match(opt["params"], self.intended):
                return False
        return touched

    def __call__(self, question: str, options: list[dict], previews=None, candidates=None) -> dict:
        self.asked += 1
        for i, o in enumerate(options):
            if self.matches(o, (candidates or [None] * len(options))[i]):
                return {"choice": i, "label": o.get("label"), "comment": "", "by": "oracle"}
        facts = " ".join(x["answer"] for x in self.script)
        return {"choice": None, "label": None, "by": "oracle",
                "comment": "None of these is what the brief means." + (f" {facts}" if facts else "")}
