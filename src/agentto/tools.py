"""Tool contract v0: one typed, logged call surface for the GUI, the LLM agent and scripts.

A `Session` holds a draft spec (a dict that may be incomplete while it is being built) and the jobs solved from it.
Every call goes through `Session.call(name, args, actor)`, which validates the arguments against the tool's
signature, runs the tool, and appends one record to the provenance log (`session.jsonl`):

    {"type": "call", "seq", "t", "actor", "tool", "args", "ok", "result" | "error", "spec_hash", "dt_s"}

Solves that finish after their call returns append {"type": "result", "seq", "job", "result"}. `replay` re-runs a
log without any LLM and checks that every spec hash and every solve result is reproduced (sessions use the
deterministic multigrid, so compliance must match to the last bit).

The agent can read certificates but nothing in this contract can write one: `certified` comes from the solver.
"""
from __future__ import annotations

import base64
import copy
import datetime as _dt
import hashlib
import inspect
import json
import threading
import time
import typing
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from . import SCHEMA_VERSION, __version__
from .spec import (Axis, BoxDomain, Load, Material, NodeRegion, Passive, Resolution, Spec, Support, Units, Vec3,
                   VolumeRegion)

class ClarifyOption(BaseModel):
    """One concrete reading of an ambiguous brief: the item the spec would contain under it (a full support, load,
    or keep-in/keep-out item, as for add_support / add_load / add_passive) and/or parameter values."""
    model_config = ConfigDict(extra="forbid")
    label: str
    item_kind: Literal["support", "load", "passive"] | None = None
    item: dict | None = None
    params: dict | None = None


RUNG_NAMES = ["sketch", "preview", "refine", "release"]
# X2 provenance conditions: spec_prov = spec + set_source/list_sources (GP); clarify_prov = clarify + the same, and
# ask_clarification only about items tagged assumed (GPC). Every spec/clarify tool joins its *_prov profile below.
PROV_PROFILES = ("spec_prov", "clarify_prov")
SKETCH_CELLS = 40           # resolution used for grounding when a draft has none yet
COARSE_PAD = "aware"        # coarse-level rule on padded grids (solver._PadAware; "voxfem" in logs before 2026-10-30)


# --------------------------------------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------------------------------------

TOOLS: dict[str, dict] = {}


def tool(*, mutates: bool = False, profiles=("spec", "clarify", "full"), replay: str = "run"):
    """Register a Session method as a contract tool. `replay`: "run" (re-run and compare), "solve" (re-run and
    compare the job result), "skip" (timing-dependent, e.g. cancel/status)."""
    def deco(fn):
        params = list(inspect.signature(fn).parameters.values())[1:]
        hints = typing.get_type_hints(fn, include_extras=True)
        fields = {p.name: (hints.get(p.name, Any), ... if p.default is inspect.Parameter.empty else p.default)
                  for p in params}
        model = create_model(f"{fn.__name__}_args", __config__=ConfigDict(extra="forbid"), **fields)
        TOOLS[fn.__name__] = {"fn": fn, "model": model, "mutates": mutates, "profiles": set(profiles),
                              "replay": replay, "doc": inspect.getdoc(fn) or "", "params": params}
        return fn
    return deco


class ToolError(ValueError):
    pass


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def spec_hash(draft: dict | None) -> str | None:
    """Hash of the normalized spec (schema-valid drafts are normalized through the model, so defaults count)."""
    if draft is None:
        return None
    try:
        norm = Spec.model_validate(draft).model_dump(mode="json")
    except ValidationError:
        norm = draft
    return hashlib.sha256(_canon(norm).encode()).hexdigest()[:16]


def _dump(x):
    if isinstance(x, BaseModel):
        return x.model_dump(mode="json")
    if isinstance(x, (list, tuple)):
        return [_dump(v) for v in x]
    return x


def _compact(result):
    """Log form of a result: binary payloads are dropped (their hash is kept)."""
    if isinstance(result, dict):
        return {k: _compact(v) for k, v in result.items() if not k.endswith("_b64")}
    if isinstance(result, list):
        return [_compact(v) for v in result]
    return result


def _now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


# --------------------------------------------------------------------------------------------------------
# job runners: where solves execute. LocalRunner owns a GPU worker (scripts, MCP stand-alone); the web
# backend supplies its own runner around the app's worker so GUI and agent share one GPU queue.
# --------------------------------------------------------------------------------------------------------

class LocalRunner:
    def __init__(self):
        self.worker = None
        self.session = None
        self._pump = None

    def _ensure(self):
        if self.worker is None or not self.worker.alive():
            from .worker import GPUWorker
            self.worker = GPUWorker()
        if self._pump is None or not self._pump.is_alive():
            # Events reach the session in the background, so status() between calls is current (X3: without
            # this, a wait=False solve stayed "queued" until some later call blocked in wait()).
            self._pump = threading.Thread(target=self._run_pump, args=(self.worker,), daemon=True)
            self._pump.start()

    def _run_pump(self, worker):
        while self.worker is worker:
            ev = worker.get(timeout=0.5)
            if ev is not None:
                self.session.on_event(ev)
            elif not worker.alive():             # drained and dead
                break

    def submit(self, job: dict) -> str:
        self._ensure()
        return self.worker.submit(job)

    def cancel(self):
        if self.worker is not None:
            self.worker.cancel()

    def wait(self, done: threading.Event, timeout: float):
        """Block until `done` is set (the pump delivers the events) or the timeout passes."""
        t_end = time.perf_counter() + timeout
        while not done.wait(min(max(t_end - time.perf_counter(), 0.0), 0.5)):
            if time.perf_counter() >= t_end:
                return False
            if self.worker is None or not self.worker.alive():
                raise ToolError("GPU worker died")
        return True

    def close(self):
        if self.worker is not None:
            w, self.worker = self.worker, None
            w.close()
            if self._pump is not None:
                self._pump.join(timeout=5)
            self._pump = None


# --------------------------------------------------------------------------------------------------------
# session
# --------------------------------------------------------------------------------------------------------

class Session:
    def __init__(self, root: str | Path | None = None, runner=None, profile: str = "full"):
        self.root = Path(root) if root is not None else None
        self.draft: dict | None = None
        self.undo_stack: list = []
        self.redo_stack: list = []                 # filled by undo, emptied by any other spec change
        self.seq = 0
        self.jobs: dict[str, dict] = {}            # job id -> record
        self.profile = profile
        self.rules_feedback = True                 # False: validate_spec checks the schema only (X2 ablation GNR)
        self.sources: dict[str, dict] = {}         # spec item -> {"source", "note"} (profiles *_prov; X2 GP, GPC)
        self.runner = runner or LocalRunner()
        self.runner.session = self
        self.listeners = []                        # callables(record) for GUI broadcast
        self.replay_warm: dict | None = None       # set by replay(): original solve seq -> replayed job id
        self._lock = threading.RLock()
        self._done: dict[str, threading.Event] = {}
        self._pending_wait = None
        self._pending_ask = None
        self.answerer = None                       # callable(question, options, previews) -> answer dict
        self.replay_answer: dict | None = None     # set by replay(): the recorded answer to reuse
        self._cur_seq = 0
        self.log_path = None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            self.log_path = self.root / "session.jsonl"
            self._append({"type": "session", "t": _now(), "agentto": __version__, "schema": SCHEMA_VERSION,
                          "id": uuid.uuid4().hex[:12], "profile": profile})

    # ---- plumbing -------------------------------------------------------------------------------------
    def _append(self, rec: dict):
        if self.log_path is not None:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        for fn in list(self.listeners):
            try:
                fn(rec)
            except Exception:                    # noqa: BLE001 -- a GUI listener must not break the log
                pass

    def tools(self, profile: str | None = None) -> list[str]:
        p = profile or self.profile
        return [n for n, t in TOOLS.items() if p in t["profiles"]]

    def call(self, name: str, args: dict | None = None, actor: str = "script", meta: dict | None = None) -> dict:
        """Validate, run and log one tool call. Returns {"ok", "seq", "result" | "error", "spec_hash"}."""
        args = dict(args or {})
        with self._lock:
            if name not in TOOLS or self.profile not in TOOLS[name]["profiles"]:
                return {"ok": False, "error": f"unknown tool {name!r}"}
            t = TOOLS[name]
            self.seq += 1
            seq = self.seq
            t0 = time.perf_counter()
            before = copy.deepcopy(self.draft)
            self._cur_seq = seq
            try:
                parsed = t["model"].model_validate(args)
                kw = {p.name: getattr(parsed, p.name) for p in t["params"]}
                result = t["fn"](self, **kw)
                ok, err = True, None
            except ValidationError as ex:
                ok, err, result = False, _short_errors(ex), None
            except (ToolError, ValueError) as ex:
                ok, err, result = False, str(ex), None
            except Exception as ex:              # noqa: BLE001 -- a tool bug is reported, never kills the server
                ok, err, result = False, f"internal error: {type(ex).__name__}: {ex}", None
            if not ok and t["mutates"]:
                self.draft = before
            elif ok and name == "undo":
                self.redo_stack.append(before)
            elif ok and name == "redo":
                self.undo_stack.append(before)
            elif ok and t["mutates"] and before != self.draft:
                self.undo_stack.append(before)
                self.redo_stack.clear()
            rec = {"type": "call", "seq": seq, "t": _now(), "actor": actor, "tool": name, "args": args, "ok": ok,
                   "spec_hash": spec_hash(self.draft), "dt_s": round(time.perf_counter() - t0, 4)}
            if meta:
                rec["meta"] = meta               # e.g. GUI origin and a human label; not used by replay
            if ok:
                rec["result"] = _compact(result)
            else:
                rec["error"] = err
            self._append(rec)
            out = {"ok": ok, "seq": seq, "spec_hash": rec["spec_hash"]}
            out.update({"result": result} if ok else {"error": err})
            pending, self._pending_wait = getattr(self, "_pending_wait", None), None
            ask, self._pending_ask = self._pending_ask, None
        if ask is not None:                      # clarification: previews and the answer, outside the lock
            out["result"] = self._answer(*ask)
        if pending is not None:                  # blocking solve: wait outside the lock so events can arrive
            jid, timeout = pending
            try:
                if self.runner.wait(self._done[jid], timeout):
                    out["result"] = dict(self.jobs[jid]["result"])
                else:
                    out["result"] = {"job": jid, "status": self.jobs[jid]["status"], "it": self.jobs[jid].get("it"),
                                     "timeout": True, "hint": "still running: call wait_job to keep waiting; "
                                     "cancelling and resubmitting starts the run again from zero"}
            except Exception as ex:              # noqa: BLE001 -- e.g. the worker process died
                out["result"] = {"job": jid, "status": "error", "error": f"{type(ex).__name__}: {ex}"}
        return out

    def _need_draft(self) -> dict:
        if self.draft is None:
            raise ToolError("no spec yet: call new_spec or set_spec first")
        return self.draft

    def _items(self):
        d = self._need_draft()
        return [("supports", x) for x in d.get("supports", [])] + [("loads", x) for x in d.get("loads", [])] + \
               [("passive", x) for x in d.get("passive", [])]

    def _check_new_id(self, id_: str):
        if any(x.get("id") == id_ for _, x in self._items()):
            raise ToolError(f"id {id_!r} already exists")

    def _parts(self):
        """The draft as a Spec whose components are validated individually, so a partial draft (no loads yet,
        no resolution) can still be resolved on a grid for grounding. Returns (spec, problems)."""
        d = self._need_draft()
        probs = []

        def v(model, x, what):
            try:
                return model.model_validate(x)
            except ValidationError as ex:
                probs.append(f"{what}: {_short_errors(ex)}")
                return None
        dom = v(BoxDomain, d.get("domain"), "domain")
        if dom is None:
            raise ToolError("; ".join(probs))
        res = d.get("resolution")
        res = v(Resolution, res, "resolution") if res else None
        if res is None:
            res = Resolution(cells_short_edge=SKETCH_CELLS)
            probs.append(f"no resolution set; grounding uses {SKETCH_CELLS} elements on the short edge")
        sp = Spec.model_construct(
            name=d.get("name", "draft"), units=v(Units, d.get("units", {}), "units") or Units(), domain=dom,
            material=v(Material, d.get("material", {}), "material") or Material(),
            volume_fraction=d.get("volume_fraction"),
            supports=[s for s in (v(Support, x, f"support {x.get('id')}") for x in d.get("supports", [])) if s],
            loads=[s for s in (v(Load, x, f"load {x.get('id')}") for x in d.get("loads", [])) if s],
            passive=[s for s in (v(Passive, x, f"passive {x.get('id')}") for x in d.get("passive", [])) if s],
            resolution=res, min_feature=d.get("min_feature"), meta={}, description="",
            schema_version=SCHEMA_VERSION)
        return sp, probs

    def _spec(self) -> Spec:
        try:
            return Spec.model_validate(self._need_draft())
        except ValidationError as ex:
            raise ToolError(f"spec is not complete/valid: {_short_errors(ex)}") from None

    # ---- spec editing ---------------------------------------------------------------------------------
    @tool(mutates=True)
    def new_spec(self, name: str, size: Vec3, length_unit: Literal["mm", "m", "in"] = "mm",
                 force_unit: Literal["N", "kN", "lbf"] = "N") -> dict:
        """Start a new spec: a box design domain of the given size (x, y, z) with its min corner at the origin.
        Replaces any current spec. Material defaults to steel (E 210e3, nu 0.3) until set_params changes it."""
        self.draft = {"schema_version": SCHEMA_VERSION, "name": name, "units": {"length": length_unit, "force":
                      force_unit}, "domain": BoxDomain(size=size).model_dump(mode="json"),
                      "material": Material().model_dump(mode="json"), "supports": [], "loads": [], "passive": []}
        return self._domain_info()

    @tool(mutates=True)
    def set_spec(self, spec: dict) -> dict:
        """Replace the whole spec with an agentto.spec/v0 document. A document that does not pass the schema yet
        (e.g. no load) is kept as a draft and its schema errors are returned."""
        try:
            self.draft = Spec.model_validate(spec).model_dump(mode="json")
            return {"name": self.draft["name"], "valid": True, "items": [x["id"] for _, x in self._items()]}
        except ValidationError as ex:
            try:
                BoxDomain.model_validate(spec.get("domain"))
            except (ValidationError, AttributeError):
                raise ToolError(_short_errors(ex)) from None
            self.draft = copy.deepcopy(spec)
            return {"name": spec.get("name"), "valid": False, "schema_errors": _short_errors(ex, as_list=True)}

    @tool(mutates=True, profiles=("full",))
    def load_template(self, name: Literal["cantilever", "bridge", "torsion"], k: int = 40) -> dict:
        """Load a FEAD benchmark problem (2:1:1 box, E = 1) at k elements on the short edge."""
        from . import library
        self.draft = library.fead(name, k).model_dump(mode="json")
        return {"name": self.draft["name"], "items": [x["id"] for _, x in self._items()]}

    @tool(mutates=True)
    def set_params(self, name: str | None = None, description: str | None = None, size: Vec3 | None = None,
                   length_unit: Literal["mm", "m", "in"] | None = None,
                   force_unit: Literal["N", "kN", "lbf"] | None = None, E: float | None = None,
                   nu: float | None = None, material_name: str | None = None,
                   volume_fraction: float | None = None, h: float | None = None,
                   cells_short_edge: int | None = None, min_feature: float | None = None,
                   clear_min_feature: bool = False) -> dict:
        """Set scalar spec fields; only the arguments given change. E is in force/length^2 of the spec units
        (210 GPa = 210e3 N/mm^2). volume_fraction is of the design (non-passive) volume, 0-1. Resolution: give
        h (element edge, length units) or cells_short_edge, not both."""
        changed = [k for k, v in locals().items() if k != "self" and v not in (None, False)]
        d = self._need_draft()
        if name is not None:
            d["name"] = name
        if description is not None:
            d["description"] = description
        if size is not None:
            d["domain"] = BoxDomain(size=size).model_dump(mode="json")
        if length_unit or force_unit:
            u = dict(d.get("units", {}))
            u.update({k: v for k, v in (("length", length_unit), ("force", force_unit)) if v})
            d["units"] = Units.model_validate(u).model_dump(mode="json")
        if E is not None or nu is not None or material_name is not None:
            m = dict(d.get("material", {}))
            m.update({k: v for k, v in (("E", E), ("nu", nu), ("name", material_name)) if v is not None})
            d["material"] = Material.model_validate(m).model_dump(mode="json")
        if volume_fraction is not None:
            if not 0 < volume_fraction < 1:
                raise ToolError("volume_fraction must be between 0 and 1 (e.g. 0.3 for 30%)")
            d["volume_fraction"] = volume_fraction
        if h is not None or cells_short_edge is not None:
            d["resolution"] = Resolution(h=h, cells_short_edge=cells_short_edge).model_dump(mode="json")
        if clear_min_feature:
            d.pop("min_feature", None)
        elif min_feature is not None:
            if min_feature <= 0:
                raise ToolError("min_feature must be positive")
            d["min_feature"] = min_feature
        return {"changed": changed}

    @tool(mutates=True)
    def add_support(self, id: str, region: NodeRegion, fix: list[Axis] = ["x", "y", "z"], note: str = "") -> dict:
        """Fix the nodes of a region in the given directions. Regions: face {face}; patch {face, center, size,
        shape rect|ellipse} with center/size in the face's two in-plane axes in increasing order ((y,z) on x
        faces, (x,z) on y faces, (x,y) on z faces); edge {axis, at, span?}; box {min, max}; cylinder {axis,
        center, radius, span}. Coordinates are absolute, origin at the domain's min corner."""
        self._check_new_id(id)
        s = Support(id=id, region=region, fix=fix, note=note)
        self._need_draft().setdefault("supports", []).append(s.model_dump(mode="json"))
        return self._item_info(s.model_dump(mode="json"))

    @tool(mutates=True)
    def add_load(self, id: str, region: NodeRegion, force: Vec3, note: str = "") -> dict:
        """Apply a total force vector (spec force units, x y z components) shared equally by the region's nodes.
        Region forms as in add_support."""
        self._check_new_id(id)
        ld = Load(id=id, region=region, force=force, note=note)
        self._need_draft().setdefault("loads", []).append(ld.model_dump(mode="json"))
        return self._item_info(ld.model_dump(mode="json"))

    @tool(mutates=True)
    def add_passive(self, id: str, kind: Literal["solid", "void"], region: VolumeRegion, note: str = "") -> dict:
        """Keep-in (solid) or keep-out (void) volume, a box or cylinder: elements whose centres fall inside are
        held at that state and excluded from the volume fraction."""
        self._check_new_id(id)
        p = Passive(id=id, kind=kind, region=region, note=note)
        self._need_draft().setdefault("passive", []).append(p.model_dump(mode="json"))
        return self._item_info(p.model_dump(mode="json"))

    @tool(mutates=True)
    def update_item(self, id: str, region: dict | None = None, fix: list[Axis] | None = None,
                    force: Vec3 | None = None, kind: Literal["solid", "void"] | None = None,
                    note: str | None = None, new_id: str | None = None) -> dict:
        """Change fields of an existing support, load or passive region (only the arguments given)."""
        for key, x in self._items():
            if x["id"] == id:
                y = dict(x)
                for k, v in (("region", region), ("fix", fix), ("force", force), ("kind", kind), ("note", note)):
                    if v is not None:
                        y[k] = list(v) if isinstance(v, tuple) else v
                if new_id is not None and new_id != id:
                    self._check_new_id(new_id)
                    y["id"] = new_id
                model = {"supports": Support, "loads": Load, "passive": Passive}[key]
                try:
                    y = model.model_validate(y).model_dump(mode="json")
                except ValidationError as ex:
                    raise ToolError(_short_errors(ex)) from None
                lst = self.draft[key]
                lst[lst.index(x)] = y
                return self._item_info(y)
        raise ToolError(f"no item with id {id!r}")

    @tool(mutates=True)
    def remove_item(self, id: str) -> dict:
        """Delete a support, load or passive region by id."""
        for key, x in self._items():
            if x["id"] == id:
                self.draft[key].remove(x)
                return {"removed": id}
        raise ToolError(f"no item with id {id!r}")

    @tool(mutates=True)
    def undo(self) -> dict:
        """Revert the last spec change."""
        if not self.undo_stack:
            raise ToolError("nothing to undo")
        self.draft = self.undo_stack.pop()
        return {"spec_hash": spec_hash(self.draft)}

    @tool(mutates=True)
    def redo(self) -> dict:
        """Re-apply the last spec change that undo reverted."""
        if not self.redo_stack:
            raise ToolError("nothing to redo")
        self.draft = self.redo_stack.pop()
        return {"spec_hash": spec_hash(self.draft)}

    # ---- grounding and checks -------------------------------------------------------------------------
    @tool()
    def get_spec(self) -> dict:
        """The current spec document (possibly incomplete)."""
        return {"spec": copy.deepcopy(self._need_draft())}

    def _domain_info(self) -> dict:
        d = self._need_draft()
        L = d["domain"]["size"]
        faces = {}
        for a, ax in enumerate("xyz"):
            b, c = [i for i in range(3) if i != a]
            for sgn, val in (("-", 0.0), ("+", L[a])):
                faces[f"{ax}{sgn}"] = (f"{ax}={val:g}; in-plane axes ({'xyz'[b]},{'xyz'[c]}) spanning "
                                       f"[0,{L[b]:g}]x[0,{L[c]:g}]")
        u = d.get("units", {})
        return {"domain": {"size": L, "units": u, "origin": "min corner at (0,0,0)"}, "faces": faces}

    def _item_info(self, x: dict) -> dict:
        from .render import region_geometry
        g = region_geometry(x["region"], self._need_draft()["domain"]["size"])
        info = {"id": x["id"], "kind": x["region"]["kind"], "bbox": [np.round(g["bbox"][0], 6).tolist(),
                np.round(g["bbox"][1], 6).tolist()], g["measure"][0]: round(g["measure"][1], 6)}
        if "force" in x:
            info["force"] = x["force"]
        if "fix" in x:
            info["fix"] = x["fix"]
        if "kind" in x and x["region"]["kind"] in ("box", "cylinder") and x.get("kind") in ("solid", "void"):
            info["passive"] = x["kind"]
        return info

    @tool()
    def list_regions(self, h: float | None = None) -> dict:
        """Ground the spec on the mesh: domain faces and their in-plane axes, then every support/load/passive
        item with its physical bounding box and size, and how many nodes (or elements) it selects at mesh size
        h (default: the spec's resolution), with any snapping. An item that selects 0 nodes does nothing."""
        from .problem import ResolutionError, build
        sp, probs = self._parts()
        out = self._domain_info()
        try:
            pb = build(sp, h=h)
            out["mesh"] = {"h": pb.h, "grid": list(pb.grid.shape), "n_elem": pb.grid.n_elem}
            regions = pb.regions
        except ResolutionError as ex:
            out["mesh"] = {"error": str(ex)}
            regions = {}
        items = []
        for key, x in self._items():
            info = {"role": {"supports": "support", "loads": "load", "passive": "passive"}[key]}
            try:
                info.update(self._item_info(x))
            except Exception as ex:              # noqa: BLE001 -- report a malformed item, keep the rest
                info.update({"id": x.get("id"), "error": str(ex)})
            r = regions.get(x.get("id"))
            if r:
                info.update({k: v for k, v in r.items() if v or k in ("n_nodes", "n_elem")})
            items.append(info)
        out["items"] = items
        if probs:
            out["notes"] = probs
        return out

    @tool()
    def validate_spec(self, h: float | None = None) -> dict:
        """Run the schema and the deterministic rule base (well-posedness, rigid-body modes, empty regions, loads
        on fixed nodes, overlaps, resolvable minimum feature, memory) at mesh size h (default: the spec's).
        Returns ok plus findings with stable codes and suggested fixes. Only an ok spec can be solved."""
        from .rules import check
        try:
            s = Spec.model_validate(self._need_draft())
        except ValidationError as ex:
            return {"ok": False, "schema_errors": _short_errors(ex, as_list=True)}
        if not self.rules_feedback:
            return {"ok": True, "findings": []}
        out = check(s, h=h).to_dict()
        if self.profile in PROV_PROFILES:          # R014, advisory: values the agent chose itself, and untagged ones
            src = self._source_table()
            if src["assumed"]:
                out["findings"].append({"code": "R014", "severity": "info", "ids": src["assumed"], "fix": None,
                                        "message": "values tagged assumed (not given by the brief): "
                                                   + ", ".join(src["assumed"])})
            if src["untagged"]:
                out["findings"].append({"code": "R014", "severity": "info", "ids": src["untagged"], "fix": None,
                                        "message": "items without a source tag (call set_source): "
                                                   + ", ".join(src["untagged"])})
        return out

    # ---- value provenance (profiles spec_prov, clarify_prov) -------------------------------------------
    def _source_keys(self) -> list[str]:
        d = self._need_draft()
        keys = [k for k in ("units", "domain", "material", "volume_fraction", "resolution") if k in d]
        role = {"supports": "support", "loads": "load", "passive": "passive"}
        return keys + [f"{role[k]}:{x.get('id')}" for k, x in self._items()]

    def _source_table(self) -> dict:
        keys = self._source_keys()
        rows = [{"key": k, **self.sources[k]} for k in keys if k in self.sources]
        return {"sources": rows, "untagged": [k for k in keys if k not in self.sources],
                "assumed": [r["key"] for r in rows if r["source"] == "assumed"]}

    @tool(profiles=("spec_prov", "clarify_prov", "full"))
    def set_source(self, key: str, source: Literal["stated", "derived", "assumed", "answered"], note: str = "") -> dict:
        """Record where one spec item's value comes from. key: units, domain, material, volume_fraction,
        resolution, or support:<id>, load:<id>, passive:<id>. source: stated (the brief gives it), derived
        (computed from numbers the brief gives), assumed (your choice where the brief is silent or ambiguous),
        answered (fixed by an answer to ask_clarification). note: what was stated, derived or assumed."""
        if key not in self._source_keys():
            raise ToolError(f"no spec item {key!r}; items: {', '.join(self._source_keys())}")
        self.sources[key] = {"source": source, "note": note}
        t = self._source_table()
        return {"key": key, "source": source, "untagged": t["untagged"]}

    @tool(profiles=("spec_prov", "clarify_prov", "full"))
    def list_sources(self) -> dict:
        """Every spec item with its source tag (stated, derived, assumed, answered), the untagged items, and the
        items tagged assumed."""
        return self._source_table()

    @tool()
    def explain_spec(self) -> dict:
        """Plain-language summary of the spec, for confirming intent with the user."""
        d = self._need_draft()
        u = d.get("units", {})
        lu, fu = u.get("length", "mm"), u.get("force", "N")
        L = d["domain"]["size"]
        m = d.get("material", {})
        lines = [f"Design domain: box {L[0]:g} x {L[1]:g} x {L[2]:g} {lu} (x, y, z), origin at a corner.",
                 f"Material: {m.get('name', '?')}, E = {m.get('E', '?'):g} {fu}/{lu}^2, nu = {m.get('nu', '?')}."]
        for s in d.get("supports", []):
            lines.append(f"Support {s['id']}: {_region_words(s['region'], lu)} fixed in {''.join(s['fix'])}.")
        for ld in d.get("loads", []):
            f = np.asarray(ld["force"], float)
            lines.append(f"Load {ld['id']}: total {np.linalg.norm(f):g} {fu} along "
                         f"({', '.join(f'{c:g}' for c in f)}) on {_region_words(ld['region'], lu)}.")
        for p in d.get("passive", []):
            lines.append(f"{'Keep-in' if p['kind'] == 'solid' else 'Keep-out'} {p['id']}: "
                         f"{_region_words(p['region'], lu)}.")
        vf = d.get("volume_fraction")
        lines.append(f"Use {vf * 100:g}% of the design volume." if vf else "Volume fraction: not set.")
        r = d.get("resolution")
        if r:
            lines.append(f"Mesh: element edge {r['h']:g} {lu}." if r.get("h") else
                         f"Mesh: {r['cells_short_edge']} elements on the shortest edge.")
        else:
            lines.append("Mesh: not set.")
        if d.get("min_feature"):
            lines.append(f"Smallest member needed: {d['min_feature']:g} {lu}.")
        return {"text": "\n".join(lines)}

    @tool()
    def render(self, view: Literal["iso", "front", "top", "side", "back_iso"] = "iso") -> dict:
        """PNG picture of the domain with supports (blue), load arrows (red) and keep-in/keep-out regions, drawn
        from the spec geometry. Use it to confirm the setup with the user before solving."""
        from .render import png_sha256, render_spec
        png = render_spec(self._need_draft(), view=view)
        out = {"view": view, "png_sha256": png_sha256(png), "bytes": len(png), "png_b64":
               base64.b64encode(png).decode()}
        if self.root is not None:
            p = self.root / "renders" / f"{self._cur_seq:04d}_{view}.png"
            p.parent.mkdir(exist_ok=True)
            p.write_bytes(png)
            out["path"] = str(p)
        return out

    @tool()
    def ladder(self) -> dict:
        """Resolution rungs for this spec (sketch, preview, refine, release; each 2x finer per axis). The spec's
        resolution h is the base mesh the ladder is built from, not the release mesh: release is the finest rung
        that fits on the GPU, often finer than the spec's h. Only a solve at the release rung can be released."""
        from .worker import ladder
        rungs = _named(ladder(self._spec()))
        h0 = (self.draft.get("resolution") or {}).get("h") if self.draft else None
        out = {"rungs": rungs}
        if rungs and h0 and rungs[-1]["h"] < h0:
            out["note"] = (f"the spec's h = {h0:g} is a base mesh; the release rung (h = {rungs[-1]['h']:g}) is finer, "
                           "which satisfies a brief that asks for elements of at most h")
        return out

    # ---- solving ----------------------------------------------------------------------------------------
    @tool(profiles=("full",), replay="solve")
    def solve(self, rung: Literal["sketch", "preview", "refine", "release"] | None = None, h: float | None = None,
              warm: bool = True, optimize: bool = False, wait: bool = True, timeout_s: float = 900) -> dict:
        """Optimize the current spec on a ladder rung (or mesh size h). Warm-starts from this spec's design on the
        2x coarser grid when there is one. By default that warm start only certifies the upsampled coarse design
        (about 12 iterations); optimize=True re-opens it (about 60 iterations, ~60% of a cold run, compliance
        within +0.1% of cold or better on the X1 problems). wait=False returns the job id at once. The result
        carries the solver's certificate; only a certified release is a released design."""
        from .rules import check
        from .worker import ladder
        s = self._spec()
        rungs = _named(ladder(s))
        if h is not None and rung is not None:
            hit = [r for r in rungs if r["name"] == rung]
            if not hit or abs(hit[0]["h"] - h) > 1e-12 * h:
                raise ToolError(f"give rung or h, not both: rung {rung!r} is "
                                f"{'h = %g' % hit[0]['h'] if hit else 'not available'}; h = {h:g} is "
                                f"{'not a rung' if not hit else 'a different rung'}. Rungs: "
                                f"{[(r['name'], r['h']) for r in rungs]}")
        if h is None:
            if rung is None:
                raise ToolError("give rung or h")
            hit = [r for r in rungs if r["name"] == rung]
            if not hit:
                raise ToolError(f"rung {rung!r} not available; rungs: {[r['name'] for r in rungs]}")
            h = hit[0]["h"]
        rep = check(s, h=h, solver="mg_ir")
        if not rep.ok:
            raise ToolError(f"spec does not pass validation at h={h:g}: {rep.codes('error')}")
        sh = spec_hash(self.draft)
        release = bool(rungs) and abs(h - rungs[-1]["h"]) < 1e-12 * h
        warm_from, preempt = None, True
        if warm:                                 # a design of this exact spec on the 2x coarser grid
            hc = 2.0 * h
            cands = [j for j in self.jobs.values() if j["spec_hash"] == sh and abs(j["h"] - hc) < 1e-9 * hc
                     and j["status"] in ("queued", "running", "converged")]
            if cands:
                src = cands[-1]
                warm_from = src["id"]
                preempt = src["status"] == "converged"      # queue behind a coarser rung that is still running
            elif not release:                    # after an edit: the latest design on this same grid
                grid = list(build_grid_shape(s, h))
                same = [j for j in self.jobs.values() if abs(j["h"] - h) < 1e-9 * h and j["status"] == "converged"
                        and j.get("grid") == grid]
                if same:
                    warm_from = same[-1]["id"]
        if self.replay_warm is not None:                    # replay: reproduce the recorded warm source exactly
            warm_from = self.replay_warm.get("warm_from")
            preempt = self.replay_warm.get("preempt", True)
        src_h = self.jobs[warm_from]["h"] if warm_from in self.jobs else None
        warm_kind = None if warm_from is None else ("same_grid" if abs(src_h - h) < 1e-9 * h else "coarser")
        jid = uuid.uuid4().hex[:12]
        job = {"id": jid, "spec": Spec.model_validate(self.draft).model_dump(mode="json"), "h": h,
               "warm_from": warm_from, "preempt": preempt, "spill_check_every": 10 if release else 0,
               "coarse_pad": (self.replay_warm or {}).get("coarse_pad", COARSE_PAD)}
        if warm_kind == "same_grid":             # the spec changed: re-open the projection so the design can move
            job["warm_schedule"] = "reopen"
        elif warm_kind == "coarser" and optimize:
            job["warm_schedule"] = "reopen_long"
        if self.root is not None:
            job["save_rho"] = str(self.root / "fields" / f"rho_{jid}.npy")
        rec = {"id": jid, "seq": self._cur_seq, "h": h, "spec_hash": sh, "spec": job["spec"], "status": "queued",
               "warm_from": warm_from, "warm_kind": warm_kind, "preempt": preempt, "release_rung": release,
               "warm_from_seq": self.jobs[warm_from]["seq"] if warm_from in self.jobs else None,
               "rho_path": job.get("save_rho")}
        self.jobs[jid] = rec
        self._done[jid] = threading.Event()
        self.runner.submit(job)
        if wait:
            self._pending_wait = (jid, timeout_s)
        return {"job": jid, "status": "queued", "h": h, "warm_from": warm_from, "warm_kind": warm_kind}

    def on_event(self, ev: dict):
        """Worker event -> job record (called by the runner or by the web backend's event pump)."""
        jid = ev.get("job")
        with self._lock:
            rec = self.jobs.get(jid)
            if rec is None:
                return
            kind = ev["type"]
            if kind == "started":
                rec.update(status="running", grid=ev["grid"], n_elem=ev["n_elem"], solver=ev["solver"],
                           warm_used=ev["warm"], coarse_bc=ev["coarse_bc"], mg_levels=ev["mg_levels"],
                           coarse_pad=ev.get("coarse_pad", "voxfem"))
            elif kind == "progress":
                rec["it"] = ev["it"]
            elif kind == "retry":                    # mg_ir solve failure, rerun on mg64
                rec["retry"] = {"solver": ev["solver"], "error": ev.get("error")}
            elif kind in ("done", "error", "skipped"):
                cert = ev.get("certificate") or {}
                res = {"job": jid, "status": ev.get("status", kind), "h": rec["h"], "grid": rec.get("grid"),
                       "n_elem": rec.get("n_elem"), "solver": rec.get("solver"),
                       "warm_used": rec.get("warm_used", False), "warm_from_seq": rec["warm_from_seq"],
                       "warm_kind": rec.get("warm_kind") if rec.get("warm_used") else None,
                       "iterations": ev.get("iterations"), "compliance_phys": ev.get("compliance_phys"),
                       "volume": ev.get("volume"), "compute_s": ev.get("compute_s"), "wall_s": ev.get("wall_s"),
                       "certified": bool(cert.get("certified")), "checks": cert.get("checks"),
                       "release_rung": rec["release_rung"], "spec_hash": rec["spec_hash"],
                       "preview": bool(rec.get("preview"))}
                if cert.get("regularization"):
                    res["regularization"] = cert["regularization"]
                if rec.get("retry"):
                    res["retry"] = rec["retry"]
                if rec.get("coarse_pad"):
                    res["coarse_pad"] = rec["coarse_pad"]
                if kind == "error":
                    res.update(status="error", error=ev.get("message") or ev.get("kind"))
                rec["status"], rec["result"] = res["status"], res
                self._append({"type": "result", "seq": rec["seq"], "t": _now(), "job": jid, "result": res})
                ev_done = self._done.get(jid)
                if ev_done is not None:
                    ev_done.set()

    @tool(profiles=("full",), replay="skip")
    def status(self) -> dict:
        """Jobs of this session, newest last: rung h, status, iteration, certified."""
        return {"jobs": [{"job": j["id"], "h": j["h"], "status": j["status"], "it": j.get("it"),
                          "certified": (j.get("result") or {}).get("certified")} for j in self.jobs.values()]}

    @tool(profiles=("full",), replay="skip")
    def wait_job(self, job: str | None = None, timeout_s: float = 900) -> dict:
        """Wait for a queued or running solve (default: the newest job) and return its result, as solve(wait=True)
        would. Use this after a solve timed out instead of cancelling and resubmitting."""
        if job is None:
            if not self.jobs:
                raise ToolError("no jobs in this session")
            job = list(self.jobs)[-1]
        rec = self.jobs.get(job)
        if rec is None:
            raise ToolError(f"no job with id {job!r}")
        if rec.get("result") is not None:
            return dict(rec["result"])
        self._pending_wait = (job, timeout_s)
        return {"job": job, "status": rec["status"]}

    @tool(profiles=("full",), replay="skip")
    def cancel(self) -> dict:
        """Stop the running solve (it ends with status cancelled at the next iteration)."""
        self.runner.cancel()
        return {"cancelled": True}

    @tool(profiles=("full",))
    def field_stats(self, job: str | None = None, region: str | None = None,
                    box_min: Vec3 | None = None, box_max: Vec3 | None = None) -> dict:
        """Numbers on a finished design (default: the latest finished job): material fraction, grey fraction,
        solid components, and whether each load is connected through solid to a support. With region (an item id)
        or box_min/box_max, also the density inside that region."""
        from scipy import ndimage
        from .problem import build
        from .rules import build_nodes
        j = self._finished(job)
        rho = np.load(j["rho_path"], mmap_mode="r")
        spec = Spec.model_validate(j["spec"])
        sp = build(spec, h=j["h"])
        solid = np.asarray(rho > 0.5)
        lab, ncomp = ndimage.label(solid)
        sizes = np.bincount(lab.ravel())[1:] if ncomp else np.zeros(0)
        g = sp.grid

        def touching(item):
            nd = build_nodes(sp, item)
            ix, rem = np.divmod(nd, (g.ny + 1) * (g.nz + 1))
            iy, iz = np.divmod(rem, g.nz + 1)
            labs = set()
            for dx in (-1, 0):
                for dy in (-1, 0):
                    for dz in (-1, 0):
                        ex, ey, ez = ix + dx, iy + dy, iz + dz
                        ok = (ex >= 0) & (ex < g.nx) & (ey >= 0) & (ey < g.ny) & (ez >= 0) & (ez < g.nz)
                        labs |= set(np.unique(lab[ex[ok], ey[ok], ez[ok]]).tolist())
            labs.discard(0)
            return labs
        sup = set().union(*[touching(s) for s in spec.supports]) if spec.supports else set()
        loads = {ld.id: bool(touching(ld) & sup) for ld in spec.loads}
        out = {"job": j["id"], "h": j["h"], "grid": list(g.shape),
               "material_fraction": round(float(rho.mean(dtype=np.float64)), 6),
               "grey_fraction": round(float(((rho > 0.1) & (rho < 0.9)).mean()), 6),
               "solid_components": int(ncomp),
               "largest_component_share": round(float(sizes.max() / max(sizes.sum(), 1)), 6) if ncomp else 0.0,
               "load_connected_to_support": loads}
        if region is not None or box_min is not None:
            from .render import region_geometry
            if region is not None:
                sd = j["spec"]
                items = [x for x in sd["supports"] + sd["loads"] + sd["passive"] if x["id"] == region]
                if not items:
                    raise ToolError(f"no item {region!r}")
                lo, hi = region_geometry(items[0]["region"], spec.domain.size)["bbox"]
                pad = sp.h                       # node regions are thin: take one element layer around them
                lo, hi = np.asarray(lo) - pad, np.asarray(hi) + pad
            else:
                if box_max is None:
                    raise ToolError("give box_min and box_max")
                lo, hi = np.asarray(box_min, float), np.asarray(box_max, float)
            sl = tuple(slice(max(0, int(np.floor(a / sp.h + 1e-9))), min(n, int(np.ceil(b / sp.h - 1e-9))))
                       for a, b, n in zip(lo, hi, g.shape))
            sub = np.asarray(rho[sl], dtype=np.float32)
            out["region"] = {"elements": int(sub.size), "mean_density": round(float(sub.mean()), 6) if sub.size
                             else None, "solid_fraction": round(float((sub > 0.5).mean()), 6) if sub.size else None}
        return out

    # ---- clarification --------------------------------------------------------------------------------
    @tool(profiles=("clarify", "full"))
    def ask_clarification(self, question: str, options: list[ClarifyOption], previews: bool = False,
                          about: list[str] | None = None) -> dict:
        """Ask the user ONE question when the brief is ambiguous in a way that changes the design (which face,
        how large, what force or material). Give 2-4 options, each a concrete reading: the full item the spec would
        contain (item_kind + item, as you would pass to add_support / add_load / add_passive) and/or parameter
        values (params, e.g. {"volume_fraction": 0.3}). Returns the chosen option, which you then apply with the
        editing tools, or choice null with the user's comment. previews=true (full profile) first solves each
        reading at the sketch rung, so the user sees what it does to the design. about: the spec items the question
        concerns (keys as for set_source); in a provenance session every one must be tagged assumed."""
        if not 2 <= len(options) <= 4:
            raise ToolError("give 2 to 4 options")
        if self.profile == "clarify_prov":
            if not about:
                raise ToolError("give about: the spec items this question concerns (keys as for set_source)")
            not_assumed = [k for k in about if self.sources.get(k, {}).get("source") != "assumed"]
            if not_assumed:
                raise ToolError("ask only about items you tagged assumed; not assumed: " + ", ".join(not_assumed))
        opts = [o.model_dump(exclude_none=True) for o in options]
        cands = []
        for o in opts:
            if o.get("item") is not None and o.get("item_kind") is None:
                raise ToolError(f"option {o['label']!r}: item needs item_kind (support, load or passive)")
            if o.get("item") is None and not o.get("params"):
                raise ToolError(f"option {o['label']!r}: give an item or params")
            cands.append(_apply_option(self.draft, o))
        self._pending_ask = (self._cur_seq, question, opts, cands, previews and self.profile == "full")
        return {"question": question, "status": "asked"}

    def _answer(self, seq, question, opts, cands, previews) -> dict:
        prev = None
        if self.replay_answer is not None:       # replay: the recorded answer, no previews, nobody asked
            ans, self.replay_answer = self.replay_answer, None
        else:
            if previews:
                prev = [self._preview(c) for c in cands]
            if self.answerer is None:
                ans = {"choice": None, "label": None, "by": "none",
                       "comment": "Nobody can answer questions in this session: choose the most reasonable reading "
                                  "and state it as an assumption."}
            else:
                try:
                    ans = dict(self.answerer(question, opts, prev, cands))
                    if prev is None and ans.get("previews") is not None:    # computed by the answerer (GUI)
                        prev = ans.pop("previews")
                except Exception as ex:          # noqa: BLE001
                    ans = {"choice": None, "label": None, "by": "error", "comment": f"no answer ({ex})"}
        c = ans.get("choice")
        if c is not None and not (isinstance(c, int) and 0 <= c < len(opts)):
            ans = {**ans, "choice": None}
        rec = {"type": "answer", "seq": seq, "t": _now(), "answer": ans}
        if prev is not None:
            rec["previews"] = [_compact(p) for p in prev]
        self._append(rec)
        out = {"question": question, **ans, "chosen": opts[ans["choice"]] if ans.get("choice") is not None else None}
        if prev is not None:
            out["previews"] = [{k: v for k, v in p.items() if k != "png_b64"} for p in prev]
        return out

    def _preview(self, cand: dict | None) -> dict:
        """Solve one candidate reading at the sketch rung (warm from the latest design on that grid)."""
        from .render import density_thumbnail
        from .rules import check
        from .worker import ladder
        try:
            s = Spec.model_validate(cand)
        except (ValidationError, TypeError):
            return {"ok": False, "error": "not a complete spec under this reading"}
        rungs = _named(ladder(s))
        hit = [r for r in rungs if r["name"] == "sketch"] or rungs[:1]
        h = hit[0]["h"]
        if not check(s, h=h, solver="mg_ir").ok:
            return {"ok": False, "error": "this reading does not pass validation"}
        grid = list(build_grid_shape(s, h))
        with self._lock:
            # warm from the latest design of the session itself, never from another reading's preview, so every
            # reading of one question starts from the same design and their compliances compare fairly
            same = [j for j in self.jobs.values() if abs(j["h"] - h) < 1e-9 * h and j["status"] == "converged"
                    and j.get("grid") == grid and not j.get("preview")]
            warm_from = same[-1]["id"] if same else None
            jid = uuid.uuid4().hex[:12]
            job = {"id": jid, "spec": s.model_dump(mode="json"), "h": h, "warm_from": warm_from, "preempt": False,
                   "spill_check_every": 0, "coarse_pad": COARSE_PAD}
            if warm_from:
                job["warm_schedule"] = "reopen"
            if self.root is not None:
                job["save_rho"] = str(self.root / "fields" / f"rho_{jid}.npy")
            self.jobs[jid] = {"id": jid, "seq": self._cur_seq, "h": h, "spec_hash": spec_hash(job["spec"]),
                              "spec": job["spec"], "status": "queued", "warm_from": warm_from,
                              "warm_kind": "same_grid" if warm_from else None, "preempt": False,
                              "release_rung": False, "preview": True,
                              "warm_from_seq": self.jobs[warm_from]["seq"] if warm_from else None,
                              "rho_path": job.get("save_rho")}
            self._done[jid] = threading.Event()
        self.runner.submit(job)
        if not self.runner.wait(self._done[jid], 300):
            return {"ok": False, "job": jid, "error": "preview timed out"}
        r = self.jobs[jid]["result"]
        out = {"ok": r["status"] == "converged", "job": jid, "h": h, "compliance_phys": r.get("compliance_phys"),
               "iterations": r.get("iterations"), "compute_s": r.get("compute_s")}
        rp = self.jobs[jid].get("rho_path")
        if rp and Path(rp).is_file():
            png = density_thumbnail(np.load(rp), s.domain.size)
            out["png_b64"] = base64.b64encode(png).decode()
        return out

    def _finished(self, job: str | None) -> dict:
        done = [j for j in self.jobs.values() if j.get("result") and j["status"] in ("converged", "not_converged")]
        if job is not None:
            done = [j for j in done if j["id"] == job]
        if not done:
            raise ToolError("no finished design" + (f" with id {job!r}" if job else ""))
        j = done[-1]
        if not j.get("rho_path") or not Path(j["rho_path"]).is_file():
            raise ToolError("this session keeps no density fields (it has no session directory)")
        return j

    @tool(profiles=("full",))
    def export_stl(self, job: str | None = None, pool: int = 1) -> dict:
        """Write the design's rho = 0.5 surface as a binary STL (default: the latest finished job; pool > 1
        averages pool^3 element blocks first). Only a certified release is a released design: anything else is
        exported as a draft and says so."""
        from .export import export_stl
        j = self._finished(job)
        if pool not in (1, 2, 4, 8):
            raise ToolError("pool must be 1, 2, 4 or 8")
        rho = np.load(j["rho_path"])
        cert = bool(j["result"].get("certified"))
        name = f"{_slug(j['spec']['name'])}_{j['id']}{'' if cert else '_draft'}{'' if pool == 1 else f'_p{pool}'}.stl"
        out = export_stl(rho, j["h"], self.root / "exports" / name, pool=pool)
        out.update(job=j["id"], certified=cert, released=cert and bool(j["result"].get("release_rung")))
        j.setdefault("exports", {})[pool] = out
        return out

    @tool(profiles=("full",))
    def export_report(self, job: str | None = None) -> dict:
        """Write the design report (JSON and Markdown): spec and hash, rung and solver, iterations and time,
        compliance, every certificate check, the STL if one was exported, and the hash of this session's log."""
        from .export import dump, report, report_markdown, sha256_file
        j = self._finished(job)
        stl = (j.get("exports") or {}).get(1) or next(iter((j.get("exports") or {}).values()), None)
        rep = report(j, j["spec"], stl, sha256_file(self.log_path) if self.log_path else None)
        base = self.root / "exports" / f"{_slug(j['spec']['name'])}_{j['id']}_report"
        base.parent.mkdir(parents=True, exist_ok=True)
        Path(str(base) + ".json").write_text(dump(rep), encoding="utf-8")
        Path(str(base) + ".md").write_text(report_markdown(rep), encoding="utf-8")
        return {"job": j["id"], "certified": rep["certified"], "json": str(base) + ".json",
                "markdown": str(base) + ".md", "stl": stl["path"] if stl else None}


# --------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------

def spec_params(params: dict, base: dict | None = None) -> dict:
    """Clarification params as spec fields. Options may use spec field names (volume_fraction, material, ...) or
    set_params argument names (E, nu, material_name, size, length_unit, force_unit, h, cells_short_edge,
    clear_min_feature); both mean the same change. `base` supplies the fields a partial change keeps."""
    base = base or {}
    out = {}
    for k, v in params.items():
        if k in ("E", "nu", "material_name"):
            m = dict(out.get("material") or base.get("material") or {})
            m[{"material_name": "name"}.get(k, k)] = v
            out["material"] = m
        elif k in ("length_unit", "force_unit"):
            u = dict(out.get("units") or base.get("units") or {})
            u[k.split("_")[0]] = v
            out["units"] = u
        elif k == "size":
            out["domain"] = {"kind": "box", "size": v}
        elif k in ("h", "cells_short_edge"):
            out["resolution"] = {k: v}
        elif k == "clear_min_feature":
            if v:
                out["min_feature"] = None
        else:
            out[k] = v
    return out


def _apply_option(draft: dict | None, opt: dict) -> dict | None:
    """The draft as it would be under one clarification option (None if there is no draft yet)."""
    if draft is None:
        return None
    d = copy.deepcopy(draft)
    if opt.get("item") is not None:
        key = {"support": "supports", "load": "loads", "passive": "passive"}[opt["item_kind"]]
        item = opt["item"]
        lst = [x for x in d.get(key) or [] if x.get("id") != item.get("id")]
        d[key] = lst + [item]
    for k, v in spec_params(opt.get("params") or {}, d).items():
        if v is None:
            d.pop(k, None)
        else:
            d[k] = v
    return d


for _t in TOOLS.values():
    _t["profiles"] |= {p + "_prov" for p in ("spec", "clarify") if p in _t["profiles"]}


def _slug(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "design"


def build_grid_shape(spec: Spec, h: float) -> tuple:
    from .problem import grid_for
    return tuple(grid_for(spec, h)[1].shape)


def _named(rungs: list) -> list:
    shown = rungs[-4:]
    names = RUNG_NAMES[4 - len(shown):]
    out = [dict(r, name=f"coarse{i}") for i, r in enumerate(rungs[:-4])]
    return out + [dict(r, name=n) for r, n in zip(shown, names)]


def _short_errors(ex: ValidationError, as_list: bool = False):
    errs = [{"loc": ".".join(str(p) for p in e["loc"]), "msg": e["msg"]} for e in ex.errors()][:12]
    if as_list:
        return errs
    return "; ".join(f"{e['loc']}: {e['msg']}" if e["loc"] else e["msg"] for e in errs)


def _region_words(r: dict, lu: str) -> str:
    k = r["kind"]
    if k == "face":
        return f"the whole {r['face']} face"
    if k == "patch":
        shape = "elliptical" if r.get("shape") == "ellipse" else "rectangular"
        return (f"a {r['size'][0]:g} x {r['size'][1]:g} {lu} {shape} patch on the {r['face']} face centred at "
                f"({r['center'][0]:g}, {r['center'][1]:g})")
    if k == "edge":
        span = f" from {r['span'][0]:g} to {r['span'][1]:g}" if r.get("span") else ""
        return f"the line along {r['axis']} through ({r['at'][0]:g}, {r['at'][1]:g}){span}"
    if k == "box":
        return f"the box from {tuple(r['min'])} to {tuple(r['max'])} {lu}"
    if k == "cylinder":
        return (f"a cylinder along {r['axis']}, radius {r['radius']:g} {lu}, axis at ({r['center'][0]:g}, "
                f"{r['center'][1]:g}), from {r['span'][0]:g} to {r['span'][1]:g}")
    return k


def read_log(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def replay(log: str | Path | list, root: str | Path | None = None, solve: bool = True, runner=None) -> dict:
    """Re-run a session log without any LLM. Checks every call's ok flag and spec hash, and (solve=True) that every
    solve reproduces its recorded status, iteration count, compliance (bit-exact) and certificate."""
    recs = read_log(log) if not isinstance(log, list) else log
    calls = [r for r in recs if r.get("type") == "call"]
    results = {r["seq"]: r["result"] for r in recs if r.get("type") == "result"}
    answers = {r["seq"]: r["answer"] for r in recs if r.get("type") == "answer"}
    s = Session(root, runner=runner, profile="full")
    seq_job: dict[int, str] = {}
    job_map: dict[str, str] = {}
    mism, solved, skipped = [], 0, 0
    try:
        for e in calls:
            t = TOOLS.get(e["tool"])
            if t is None or t["replay"] == "skip":
                skipped += 1
                continue
            orig = results.get(e["seq"]) or (e.get("result") if e["tool"] == "solve" else None)
            if t["replay"] == "solve":
                if not solve or not e["ok"] or orig is None or orig["status"] not in ("converged", "not_converged"):
                    skipped += 1                 # cancelled / superseded / timed-out runs are timing artefacts
                    continue
                ws = orig.get("warm_from_seq") if orig.get("warm_used") else None
                # the coarse padding rule the run used (logs from before 2026-10-30 record none: voxfem's)
                s.replay_warm = {"warm_from": seq_job.get(ws), "preempt": True,
                                 "coarse_pad": orig.get("coarse_pad", "voxfem")}
                args = dict(e["args"], wait=True, timeout_s=86400.0)    # the run finished in the log: wait for it
                out = s.call("solve", args, actor=e["actor"])
                s.replay_warm = None
                solved += 1
                if out["ok"]:
                    seq_job[e["seq"]] = out["result"]["job"]
                    job_map[orig["job"]] = out["result"]["job"]
                    r = out["result"]
                    for k in ("status", "iterations", "compliance_phys", "certified", "warm_used"):
                        if r.get(k) != orig.get(k):
                            mism.append({"seq": e["seq"], "tool": "solve", "field": k, "log": orig.get(k),
                                         "replay": r.get(k)})
                else:
                    mism.append({"seq": e["seq"], "tool": "solve", "field": "ok", "log": True,
                                 "replay": out.get("error")})
                continue
            args = dict(e["args"])
            if args.get("job") in job_map:               # job ids are fresh in a replay
                args["job"] = job_map[args["job"]]
            if e["tool"] == "ask_clarification" and e["ok"]:
                s.replay_answer = answers.get(e["seq"]) or {"choice": None, "label": None, "comment": "(not logged)"}
            out = s.call(e["tool"], args, actor=e["actor"])
            s.replay_answer = None
            if e["tool"] == "ask_clarification" and e["ok"] and out["ok"] and \
                    out["result"].get("choice") != (answers.get(e["seq"]) or {}).get("choice"):
                mism.append({"seq": e["seq"], "tool": "ask_clarification", "field": "choice", "log": None,
                             "replay": out["result"].get("choice")})
            if out["ok"] != e["ok"]:
                mism.append({"seq": e["seq"], "tool": e["tool"], "field": "ok", "log": e["ok"], "replay": out["ok"]})
            if out["spec_hash"] != e["spec_hash"]:
                mism.append({"seq": e["seq"], "tool": e["tool"], "field": "spec_hash", "log": e["spec_hash"],
                             "replay": out["spec_hash"]})
            if e["tool"] == "render" and e["ok"] and out["ok"] and \
                    out["result"]["png_sha256"] != e["result"]["png_sha256"]:
                mism.append({"seq": e["seq"], "tool": "render", "field": "png_sha256", "log": "differs",
                             "replay": "differs"})
            if e["tool"] == "export_stl" and e["ok"] and out["ok"] and out["result"]["sha256"] != e["result"]["sha256"]:
                mism.append({"seq": e["seq"], "tool": "export_stl", "field": "sha256", "log": e["result"]["sha256"],
                             "replay": out["result"]["sha256"]})
    finally:
        if runner is None:
            s.runner.close()
    return {"ok": not mism, "calls": len(calls), "solves_replayed": solved, "skipped": skipped,
            "mismatches": mism, "final_spec_hash": spec_hash(s.draft)}
