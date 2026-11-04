"""Persistent GPU worker: one process owns CUDA; the backend submits jobs and receives events.

Keeping CUDA out of the web process means a solver failure or a VRAM spill never takes the server down, and
compiled kernels stay warm between jobs. One job runs at a time. Submitting a job cancels the running one (the
live sketch is latest-wins); a job marked `supersede` is skipped if another is already queued behind it.

Job (dict):  id, spec (agentto.spec/v0 dict), h (optional rung override), warm_from (id of a finished job whose
             design seeds this one: 2x upsampling + tail schedule), solver ("auto" | "mg64" | "mg_ir"),
             spill_check_every (0 for sketches; > 0 is required for a certified release), max_outer,
             frame_every_s (minimum seconds between density frames), frame_max_dim, supersede,
             save_rho (path: the final density is saved there as float16 .npy).
Events (dict, via queue): started, progress (every iteration), frame (uint8 density, downsampled, as bytes),
             done (summary + certificate), error.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
import traceback
import uuid
from collections import OrderedDict

# Largest rung offered. 1.2e8 fits the 24 GB card alone, but beside a desktop holding 2-5 GB of VRAM rungs near 1e8
# spill (113M twice, a 59M mg64 solve once); 74M ran certified on mg_ir (X1, X5). Releases stop below 8e7 (2026-10-30).
# AGENTTO_LADDER_MAX lowers it for fast GUI tests on small meshes (scripts/gui_test.py); never raises it.
LADDER_MAX = min(8.0e7, float(os.environ.get("AGENTTO_LADDER_MAX", 8.0e7)))


def ladder(spec, max_elems: float = LADDER_MAX, min_elems: float = 5e3):
    """Rungs h_spec * 2^k (k integer) whose grids are whole and within [min_elems, max_elems], coarse to fine.
    Consecutive rungs differ by exactly 2x per axis, so every rung can warm-start the next."""
    from .problem import ResolutionError, grid_for
    h0, g0 = grid_for(spec)
    rungs = []
    for k in range(-6, 7):
        h = h0 * 2.0 ** k
        try:
            h, g = grid_for(spec, h=h)
        except ResolutionError:
            continue
        if min_elems <= g.n_elem <= max_elems:
            rungs.append({"h": h, "grid": list(g.shape), "n_elem": g.n_elem})
    rungs.sort(key=lambda r: r["n_elem"])
    out = [rungs[0]] if rungs else []
    for r in rungs[1:]:
        if all(b == 2 * a for a, b in zip(out[-1]["grid"], r["grid"])):
            out.append(r)
        else:
            out = [r]
    return out


# --------------------------------------------------------------------------------------------------------
# worker process
# --------------------------------------------------------------------------------------------------------

def _frame(s, max_dim):
    import cupy as cp
    import numpy as np
    g, go = s.grid, s.sp.grid
    rp = s.rho_phys.reshape(g.shape)[:go.nx, :go.ny, :go.nz]
    f = max(1, int(np.ceil(max(go.shape) / max_dim)))
    n = [int(np.ceil(a / f)) for a in go.shape]
    if f > 1:
        pad = cp.zeros((n[0] * f, n[1] * f, n[2] * f), cp.float32)
        pad[:go.nx, :go.ny, :go.nz] = rp
        rp = pad.reshape(n[0], f, n[1], f, n[2], f).mean(axis=(1, 3, 5))
    u8 = cp.clip(rp * 255.0 + 0.5, 0, 255).astype(cp.uint8)
    return cp.asnumpy(u8).tobytes(), n, f


def _run_job(job, evt, cancel, cache):
    import cupy as cp
    import numpy as np
    from .problem import build
    from .rules import check
    from .solver import SessionSIMP, reopen_schedule, tail_schedule
    from .spec import Spec
    from voxfem.simp import Config

    jid = job["id"]
    spec = Spec.model_validate(job["spec"])
    rep = check(spec, h=job.get("h"), solver="mg_ir")
    if not rep.ok:
        evt.put({"type": "error", "job": jid, "kind": "invalid_spec", "report": rep.to_dict()})
        return
    sp = build(spec, h=job.get("h"))
    solver = job.get("solver", "auto")
    if solver == "auto":
        solver = "mg64" if sp.grid.n_elem <= 6.0e7 else "mg_ir"
    warm = cache.get(job.get("warm_from")) if job.get("warm_from") else None
    if warm is not None and warm.shape != tuple(sp.grid.shape) and             any(2 * a != b for a, b in zip(warm.shape, sp.grid.shape)):
        warm = None                              # neither the same grid nor one doubling coarser
    # voxfem's bounded-oscillation stop allows 1 ppm of elements still moving; sessions allow 100 ppm (a limit cycle
    # of a few boundary elements with compliance flat to 1e-4 is common after a warm start). Recorded in the
    # certificate's config.
    kw = dict(volfrac=spec.volume_fraction, solver=solver, op32="stencil", log_every=0,
              conv_moving_ppm=float(job.get("conv_moving_ppm", 100.0)),
              spill_check_every=int(job.get("spill_check_every", 0)), max_outer=int(job.get("max_outer", 300)))
    if warm is not None:
        from .escalation import reopen_long_schedule
        kw["schedule"] = {"reopen": reopen_schedule, "reopen_long": reopen_long_schedule}.get(
            job.get("warm_schedule"), tail_schedule)()
    every_s, max_dim = float(job.get("frame_every_s", 0.4)), int(job.get("frame_max_dim", 96))
    state = {"last_frame": 0.0}

    def on_iter(s, rec):
        drho = rec["drho_phys"]
        evt.put({"type": "progress", "job": jid, "it": rec["it"], "c": rec["c"],
                 "c_phys": rec["c"] * sp.compliance_scale, "vol": rec["vol"],
                 "drho": drho if np.isfinite(drho) else None,       # NaN on iteration 1; JSON has no NaN
                 "p": rec["p"], "beta": rec["beta"], "t_iter_s": rec["t_iter_s"],
                 "rel_true": rec["solve"]["rel_true"]})
        now = time.perf_counter()
        if now - state["last_frame"] >= every_s:
            data, shape, f = _frame(s, max_dim)
            evt.put({"type": "frame", "job": jid, "it": rec["it"], "shape": shape, "factor": f, "data": data})
            state["last_frame"] = now

    t0 = time.perf_counter()
    while True:
        s = SessionSIMP(sp, Config(**kw), on_iter=on_iter, cancel=cancel,
                        coarse_pad=job.get("coarse_pad", "voxfem"))
        warm_info = s.warm_start(warm) if warm is not None else None     # smoothed + exact volume match
        if warm is not None:                     # recorded in the certificate's regularization block
            s.schedule_name = job.get("warm_schedule") if job.get("warm_schedule") in ("reopen", "reopen_long") else "tail"
            s.warm_from_h = sp.h * sp.grid.shape[0] / warm.shape[0]
        evt.put({"type": "started", "job": jid, "grid": list(sp.grid.shape), "n_elem": sp.grid.n_elem, "h": sp.h,
                 "solver": kw["solver"], "warm": warm is not None, "warm_info": warm_info, "coarse_bc": s.coarse_bc,
                 "coarse_pad": s.coarse_pad,
                 "mg_levels": s.mg_levels, "levels_overridden": s.levels_overridden,
                 "setup_s": time.perf_counter() - t0})
        r = s.run()
        # X1: mg_ir's lean iterative refinement can reject a correction on the gray early iterates (cold bridge at
        # 65.5M); mg64 solved the same run. Retry once on mg64 when the grid fits its memory (65.5M ran in X1).
        if (r["status"] == "solve_failed" and kw["solver"] == "mg_ir"
                and sp.grid.n_elem <= 6.6e7):
            evt.put({"type": "retry", "job": jid, "solver": "mg64", "error": r.get("error")})
            del s
            import gc
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            kw["solver"] = "mg64"
            continue
        break
    rho = cp.asnumpy(s.final_density()).reshape(sp.grid.shape).astype(np.float32)
    if r["status"] in ("converged", "not_converged", "cancelled"):
        data, shape, f = _frame(s, max_dim)
        evt.put({"type": "frame", "job": jid, "it": r["iterations"], "shape": shape, "factor": f, "data": data,
                 "final": True})
    if r["status"] in ("converged", "not_converged"):
        cache[jid] = rho
        if job.get("save_rho"):                  # session field store (float16: 2 B/element, for field_stats)
            from pathlib import Path
            p = Path(job["save_rho"])
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, rho.astype(np.float16))
        while len(cache) > 4 or sum(a.nbytes for a in cache.values()) > 2.5e9:
            cache.popitem(last=False)
    cert = r.get("certificate", {})
    evt.put({"type": "done", "job": jid, "status": r["status"], "iterations": r["iterations"],
             "compliance": r.get("compliance"), "compliance_phys": cert.get("compliance_phys"),
             "volume": r.get("volume_phys"), "compute_s": r.get("compute_s"), "wall_s": time.perf_counter() - t0,
             "certificate": cert, "error": r.get("error")})
    del s
    import gc
    gc.collect()                                 # SIMP holds a bound method of itself (a cycle): free it now
    cp.get_default_memory_pool().free_all_blocks()


def _main(cmd, evt, cancel):
    import sys
    sys.dont_write_bytecode = True
    from . import _voxfem  # noqa: F401  (path + no-bytecode guard before any voxfem import)
    cache: OrderedDict = OrderedDict()
    evt.put({"type": "ready"})
    while True:
        job = cmd.get()
        if job is None:
            break
        if job.get("supersede") and not cmd.empty():
            evt.put({"type": "skipped", "job": job["id"]})
            continue
        cancel.clear()
        try:
            _run_job(job, evt, cancel, cache)
        except Exception as ex:                  # noqa: BLE001 -- report every failure to the client
            evt.put({"type": "error", "job": job["id"], "kind": type(ex).__name__, "message": str(ex),
                     "trace": traceback.format_exc()[-3000:]})
            try:
                import gc
                import cupy as cp
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:                    # noqa: BLE001
                pass


class GPUWorker:
    def __init__(self):
        ctx = mp.get_context("spawn")
        self.cmd, self.evt, self.cancel_ev = ctx.Queue(), ctx.Queue(), ctx.Event()
        self.proc = ctx.Process(target=_main, args=(self.cmd, self.evt, self.cancel_ev), daemon=True,
                                name="agentto-gpu")
        self.proc.start()

    def submit(self, job: dict) -> str:
        job = dict(job)
        job.setdefault("id", uuid.uuid4().hex[:12])
        if job.pop("preempt", True):
            self.cancel_ev.set()                 # stop whatever is running; the next job clears it
        self.cmd.put(job)
        return job["id"]

    def cancel(self):
        self.cancel_ev.set()

    def get(self, timeout: float | None = None):
        try:
            return self.evt.get(timeout=timeout)
        except queue.Empty:
            return None

    def alive(self) -> bool:
        return self.proc.is_alive()

    def close(self, timeout: float = 10):
        self.cancel_ev.set()
        self.cmd.put(None)
        self.proc.join(timeout)
        if self.proc.is_alive():
            self.proc.terminate()
