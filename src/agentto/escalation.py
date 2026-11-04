"""Coarse-to-fine escalation arms (X1) and one helper that runs an arm on a spec.

Every warm arm starts from the converged design of the rung one doubling coarser (2x upsampling, one filter pass,
exact volume match: `SessionSIMP.warm_start`). The arms differ only in the continuation schedule run afterwards:

    cold              full R2 schedule from uniform density (the reference)
    tail              final phase only (p 4.5, beta 32, move 0.05): certifies the upsampled design
    reopen            beta 8 -> 16 -> 32 with moves 0.10 / 0.08 / 0.05 (short re-opening)
    reopen_long       the cold schedule from its second phase on (p 3.5, beta 4, move 0.15, then as cold):
                      projection far from saturated, so boundaries move and members finer than the coarse rung
                      can form
    phys_*            the same schedules with rmin scaled by h_coarse / h_fine, i.e. the filter radius constant
                      in physical units across the doubling (a mesh-independent formulation; the reference for
                      these arms is cold_phys)
"""
from __future__ import annotations

import time

import numpy as np

from .solver import TAIL_PHASE, reopen_schedule, tail_schedule

COLD = [[15, 1.5, 1.0, 0.2, 1.5], [40, 3.5, 4.0, 0.15, 1.35], [65, 4.5, 16.0, 0.08, 1.25],
        [120, 4.5, 32.0, 0.05, 1.2]]                      # voxfem.simp.Config().schedule (R2)


def reopen_long_schedule():
    p, beta, move, r = TAIL_PHASE
    return [[25, 3.5, 4.0, 0.15, 1.35], [50, 4.5, 16.0, 0.08, 1.25], [1000, p, beta, move, r]]


def scale_rmin(schedule, f: float):
    return [[row[0], row[1], row[2], row[3], row[4] * f] for row in schedule]


BASE = {"cold": lambda: [list(r) for r in COLD], "tail": tail_schedule, "reopen": reopen_schedule,
        "reopen_long": reopen_long_schedule}
ARMS = ("cold", "tail", "reopen", "reopen_long", "cold_phys", "phys_tail", "phys_reopen_long")


def arm_schedule(arm: str, ratio: float = 2.0):
    """(schedule, warm) for an arm; ratio = h_coarse / h_fine for the physical-filter arms."""
    phys = arm.endswith("_phys") or arm.startswith("phys_")
    base = arm.replace("_phys", "").replace("phys_", "")
    sch = BASE[base]()
    return (scale_rmin(sch, ratio) if phys else sch), base != "cold"


def reference_for(arm: str) -> str:
    return "cold_phys" if (arm.endswith("_phys") or arm.startswith("phys_")) else "cold"


def run_arm(sp, cfg_kw: dict, arm: str, coarse=None, ratio: float = 2.0, keep_start: bool = True):
    """Run one arm on a built SpecProblem. Returns (result row, final physical density on the original grid,
    start density or None). `cfg_kw` are voxfem Config fields (volfrac is taken from the problem)."""
    import cupy as cp
    from voxfem.simp import Config

    from .solver import SessionSIMP
    sch, warm = arm_schedule(arm, ratio)
    cfg = Config(**{"volfrac": sp.spec.volume_fraction, **cfg_kw, "schedule": sch})
    s = SessionSIMP(sp, cfg)
    info, start = {}, None
    t0 = time.perf_counter()
    if warm:
        if coarse is None:
            raise ValueError(f"arm {arm} needs a coarse design")
        info = s.warm_start(coarse)
        if keep_start:
            st = cp.empty_like(s.rho)
            p0, beta0 = cfg.schedule[0][1], cfg.schedule[0][2]
            s.physical(s.rho, beta0, st)
            start = _unpad(s, st)
    r = s.run()
    wall = time.perf_counter() - t0
    rho = _unpad(s, s.rho_phys)
    hist = s.history
    row = {"arm": arm, "status": r["status"], "iterations": r["iterations"], "compute_s": r["compute_s"],
           "wall_s": wall, "s_per_iter": r["compute_s"] / max(1, r["iterations"]),
           "compliance": r["compliance"], "volume": r["volume_phys"], "certified": r["certificate"]["certified"],
           "rmin_final": sch[-1][4], "warm_info": info, "error": r.get("error"),
           "grid": list(sp.grid.shape), "n_elem": int(sp.grid.n_elem),
           "c_history": [h["c"] for h in hist][::max(1, len(hist) // 60)]}
    del s
    import gc
    gc.collect()                                     # SIMP references itself through its log hook
    cp.get_default_memory_pool().free_all_blocks()
    return row, rho, start


def _unpad(s, x) -> np.ndarray:
    """Physical density on the original (unpadded) grid as a float32 host array."""
    import cupy as cp
    from voxfem.core import unpad_elements
    if s.padded:
        x = unpad_elements(x, s.orig.grid, s.grid)
    return cp.asnumpy(x).reshape(s.orig.grid.shape).astype(np.float32)
