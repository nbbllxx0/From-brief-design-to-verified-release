"""voxfem extensions for interactive sessions (subclassing only; voxfem itself is never modified).

SessionSIMP(SIMP)
  * passive regions: keep-in elements have rho_phys = 1 and keep-out elements rho_phys = 0 at every iteration.
    Their raw densities are fixed (1 / 0) and enter the filter, so the design sees them; they are not design
    variables, carry zero sensitivity, and the volume constraint is over the design elements only. Keep-out
    elements keep modulus Emin (not 0), so nodes inside a keep-out stay nonsingular.
  * progress and cancel: a per-iteration callback receives the history record; setting the cancel event stops
    the run after the current iteration, keeping the current design (status "cancelled").
  * warm start from a coarser design (2x piecewise-constant upsampling) and the tail schedule.
  * a release certificate (what may be released, decided only from solver evidence).
  * the multigrid coarse support rule: voxfem's injection, or RobustGMG's dilation when injection would lose
    supports (rules.coarse_check), chosen automatically.

With no passive regions and the injection rule, every operation reduces to voxfem.SIMP's exactly (X0 identity:
same iterations, compliance and densities, tests/test_session_identity.py and results/X0).
"""
from __future__ import annotations

import math
import threading
import time
from contextlib import contextmanager

import numpy as np
import cupy as cp

from ._voxfem import voxfem  # noqa: F401
import voxfem.simp as _vsimp
from voxfem.core import ke_unit, pad_elements
from voxfem.krylov import SolveFailed
from voxfem.mg import GMG
from voxfem.simp import SIMP, Config, _dc_phys, _dproj, _oc_step, _sum64  # noqa: F401

from .problem import SpecProblem

TAIL_PHASE = (4.5, 32.0, 0.05, 1.20)          # final phase of the R2 schedule: p, beta, move, rmin


def tail_schedule(rmin: float | None = None):
    """Final continuation phase only (simp.py needs at least two rows)."""
    p, beta, move, r = TAIL_PHASE
    r = r if rmin is None else rmin
    return [[1, p, beta, move, r], [1000, p, beta, move, r]]


def reopen_schedule(rmin: float | None = None):
    """Short re-opening continuation for a warm start (X1 arm c): beta 8 -> 16 -> 32 with larger moves, so
    elements near the upsampled boundary can still move and members finer than the coarse rung can form."""
    p, _, _, r = TAIL_PHASE
    r = r if rmin is None else rmin
    return [[4, p, 8.0, 0.10, r], [10, p, 16.0, 0.08, r], [1000, p, 32.0, 0.05, r]]


def physical_rmin(schedule, r_phys: float, h: float):
    """Schedule with the filter radius fixed in physical length units (X1 arm d)."""
    return [[row[0], row[1], row[2], row[3], r_phys / h] for row in schedule]


def upsample2(rho: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(np.repeat(rho, 2, 0), 2, 1), 2, 2)


class Cancelled(Exception):
    pass


class RobustGMG(GMG):
    """GMG whose coarse levels fix a DOF when ANY fine node in the coarse node's interpolation support
    (index distance <= 1) has it fixed. voxfem injects instead (fixed only if the coincident even node is), which
    drops supports lying on odd grid lines and can leave the coarsest operator singular. Dilation over-constrains
    the coarse correction near supports; the fine-level operator, and with it every accepted solution (FP64 true
    residual), is unchanged: only the preconditioner differs."""

    @staticmethod
    def _coarse_fixed(fine_lv, gc):
        ff = fine_lv.fixed.reshape(fine_lv.grid.node_shape)
        nx, ny, nz = gc.node_shape
        pad = cp.pad(ff, 1)
        out = cp.zeros((nx, ny, nz), dtype=cp.uint8)
        for dx in range(3):
            for dy in range(3):
                for dz in range(3):
                    out |= pad[dx:dx + 2 * nx - 1:2, dy:dy + 2 * ny - 1:2, dz:dz + 2 * nz - 1:2]
        return out.ravel()


class _DeterministicCoarsest:
    """Mixin: assemble the coarsest dense matrix with a fixed-order reduction instead of cupyx.scatter_add
    (float64 atomics, order varies run to run). voxfem's runs differ at ~1e-12 in compliance because of that
    scatter, and at beta = 32 the difference can flip single elements of the final design; with this mixin a
    session is bit-reproducible, which the replay contract (C1) needs.

    Each dense entry receives at most 8 element contributions. A gather table T (n_entries x max_count, padded
    with an index of a zero) is built once per grid; the assembly is then vals = M_ext[T].sum(axis=1) on the GPU,
    a reduction with a fixed order and no atomics."""

    def _det_table(self, g, n):
        if getattr(self, "_det", None) is not None and self._det[0] == n:
            return self._det[1], self._det[2]
        edof = g.edof()
        flat = (edof[:, :, None] * n + edof[:, None, :]).ravel()
        order = np.argsort(flat, kind="stable")
        uniq, start, counts = np.unique(flat[order], return_index=True, return_counts=True)
        T = np.full((uniq.size, int(counts.max())), flat.size, dtype=np.int64)
        for k in range(T.shape[1]):
            m = counts > k
            T[m, k] = order[start[m] + k]
        self._det = (n, cp.asarray(uniq), cp.asarray(T))
        return self._det[1], self._det[2]

    def _coarsest(self):
        from voxfem.mg import UNPACK_IDX
        from cupyx.scipy.linalg import solve_triangular
        lv = self.levels[-1]
        g = lv.grid
        Ms = getattr(lv, "M", None)
        if Ms is None:
            l = self.L - 1
            s = 2 ** l
            Eg = self.E.reshape(g.nx, s, g.ny, s, g.nz, s).transpose(1, 3, 5, 0, 2, 4)
            Eg = cp.ascontiguousarray(Eg).reshape(s ** 3, g.n_elem)
            Ms = self._ke_table(l) @ Eg
        M = Ms[cp.asarray(UNPACK_IDX)].T.reshape(g.n_elem, 24, 24).astype(cp.float64)
        n = g.ndof
        uniq, T = self._det_table(g, n)
        Mext = cp.concatenate([M.ravel(), cp.zeros(1, cp.float64)])
        A = cp.zeros(n * n, dtype=cp.float64)
        A[uniq] = Mext[T].sum(axis=1)
        del Mext
        A = A.reshape(n, n)
        fx = lv.fixed
        fd = cp.stack([(fx & 1) > 0, (fx & 2) > 0, (fx & 4) > 0], axis=1).ravel()
        free = (~fd).astype(cp.float64)
        A = A * free[:, None] * free[None, :] + cp.diag(1.0 - free)
        Lc = cp.linalg.cholesky(A)
        Linv = solve_triangular(Lc, cp.eye(n), lower=True)
        self.coarse_Linv = Linv * free[None, :]
        self.coarse_inv = self.coarse_Linv
        self.coarse_n = n
        self.coarse_free_dofs = int(free.sum())


class DeterministicGMG(_DeterministicCoarsest, GMG):
    pass


class DeterministicRobustGMG(_DeterministicCoarsest, RobustGMG):
    pass


class _PadAware:
    """Mixin for padded grids. voxfem pads with zero-modulus elements and fixes every node outside the original node
    box, which leaves the fine operator exact. On the coarse levels the base rule then fixes the coarse nodes just
    past the real boundary as well, and the coarse problems see a clamp along a free surface: one padded side costs
    4-6x the PCG iterations (scripts/mg_padding_study.py). Here a coarse node is fixed by padding only if its whole
    interpolation support lies in the padding (those nodes have no stiffness). Real supports are coarsened by the
    base rule with the padding nodes cleared first. Only the preconditioner changes; on an unpadded grid nothing does."""

    orig_elems: tuple = ()                          # element counts of the unpadded grid, set by gmg_class()

    def _coarse_fixed(self, fine_lv, gc):
        finest = self.levels[0].grid
        sf = finest.nx // fine_lv.grid.nx               # stride of the fine level, in finest elements
        sc = 2 * sf
        ff = fine_lv.fixed.reshape(fine_lv.grid.node_shape).copy()
        inside_f = [cp.arange(n) * sf <= m for n, m in zip(fine_lv.grid.node_shape, self.orig_elems)]
        ff *= (inside_f[0][:, None, None] & inside_f[1][None, :, None] & inside_f[2][None, None, :]).astype(cp.uint8)
        real = type("FineLevel", (), {"fixed": ff.ravel(), "grid": fine_lv.grid})
        out = super()._coarse_fixed(real, gc).reshape(gc.node_shape)
        outside_c = [(cp.arange(n) - 1) * sc >= m for n, m in zip(gc.node_shape, self.orig_elems)]
        dead = outside_c[0][:, None, None] | outside_c[1][None, :, None] | outside_c[2][None, None, :]
        out[dead] = 7
        return out.ravel()


def gmg_class(coarse_bc: str, deterministic: bool, orig_elems: tuple | None = None):
    base = {("inject", False): GMG, ("dilate", False): RobustGMG,
            ("inject", True): DeterministicGMG, ("dilate", True): DeterministicRobustGMG}[(coarse_bc, deterministic)]
    if orig_elems is None:
        return base
    return type("PadAware" + base.__name__, (_PadAware, base), {"orig_elems": tuple(int(n) for n in orig_elems)})


class FastSpillWatch:
    """voxfem.guard.SpillWatch on the same WDDM "GPU Process Memory" counters, read in-process through PDH
    (pywin32) instead of a PowerShell subprocess: about 0.1 ms per sample instead of about 2.4 s. voxfem samples at
    the start and end of every run (even with spill checks off) and every `spill_check_every` iterations, so the
    subprocess cost 5 s per sketch and about 2.4 s per 10 release iterations. Same samples, same spill rule
    (shared usage growth > voxfem.guard.SPILL_LIMIT_BYTES); falls back to voxfem's sampler without pywin32."""
    _query = None

    def __init__(self, limit=None):
        import os
        from voxfem import guard
        self.pid = os.getpid()
        self.limit = guard.SPILL_LIMIT_BYTES if limit is None else limit
        self.samples = []
        self.base = self._sample("start")

    @classmethod
    def _memory(cls):
        try:
            import win32pdh
        except ImportError:
            from voxfem import guard
            return guard.gpu_process_memory()
        if cls._query is None:
            q = win32pdh.OpenQuery()
            cls._query = (q, win32pdh.AddEnglishCounter(q, r"\GPU Process Memory(*)\Dedicated Usage"),
                          win32pdh.AddEnglishCounter(q, r"\GPU Process Memory(*)\Shared Usage"))
        q, hd, hs = cls._query
        win32pdh.CollectQueryData(q)
        out: dict = {}
        for h, key in ((hd, "dedicated"), (hs, "shared")):
            for name, v in win32pdh.GetFormattedCounterArray(h, win32pdh.PDH_FMT_LARGE).items():
                try:
                    pid = int(name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                out.setdefault(pid, {"dedicated": 0, "shared": 0})[key] += int(v)
        return out

    def _sample(self, tag):
        m = self._memory()
        mine = m.get(self.pid, {"dedicated": 0, "shared": 0})
        rec = {"tag": tag, "t": time.time(), "dedicated": mine["dedicated"], "shared": mine["shared"],
               "others_dedicated": sum(v["dedicated"] for p, v in m.items() if p != self.pid), "sampler": "pdh"}
        self.samples.append(rec)
        return rec

    def check(self, tag="check"):
        return self._sample(tag)

    @property
    def max_shared_growth(self):
        return max(s["shared"] for s in self.samples) - self.base["shared"]

    @property
    def spilled(self):
        return self.max_shared_growth > self.limit

    def report(self):
        return {"pid": self.pid, "limit_bytes": self.limit, "max_shared_growth_bytes": int(self.max_shared_growth),
                "spilled": bool(self.spilled), "samples": self.samples, "sampler": "pdh"}


@contextmanager
def _fast_spill():
    """SIMP.__init__ constructs voxfem.guard.SpillWatch; use FastSpillWatch for that construction."""
    from voxfem import guard
    old = guard.SpillWatch
    guard.SpillWatch = FastSpillWatch
    try:
        yield
    finally:
        guard.SpillWatch = old


@contextmanager
def _gmg_class(cls):
    """SIMP.__init__ constructs voxfem.simp.GMG; swap the class for the duration of one construction."""
    old = _vsimp.GMG
    _vsimp.GMG = cls
    try:
        yield
    finally:
        _vsimp.GMG = old


class SessionSIMP(SIMP):
    @staticmethod
    def _will_pad(sp, cfg) -> bool:
        """Whether voxfem.SIMP will pad this grid for multigrid (mirrors its level choice)."""
        if not cfg.solver.startswith("mg"):
            return False
        from voxfem.mg import choose_levels
        g = sp.grid
        L = cfg.mg_levels or choose_levels(g.padded(4) if not g.coarsenable(4) else g, cfg.max_coarse_dofs)
        return not g.coarsenable(L)

    def __init__(self, sp: SpecProblem, cfg: Config, coarse_bc: str = "auto", deterministic: bool = True,
                 levels: str = "auto", on_iter=None, cancel: threading.Event | None = None, log=None,
                 coarse_pad: str = "voxfem"):
        self.sp = sp
        self.on_iter, self.cancel = on_iter, cancel
        self._user_log = log
        self.levels_overridden = False
        if levels == "auto" and cfg.solver.startswith("mg") and cfg.mg_levels is None:
            from .rules import session_levels
            _, L, over = session_levels(sp.grid, cfg.max_coarse_dofs)
            if over:
                cfg.mg_levels, self.levels_overridden = L, True
        if coarse_bc == "auto":
            from .rules import coarse_check
            coarse_bc = "inject" if coarse_check(sp).get("injection_ok", True) else "dilate"
        if coarse_bc not in ("inject", "dilate"):
            raise ValueError(coarse_bc)
        self.coarse_bc, self.deterministic = coarse_bc, deterministic
        self.schedule_name, self.warm_from_h = "cold", None   # set by the caller for a warm start; certificate field
        if on_iter is not None or cancel is not None:
            cfg.log_every = 1                       # the per-iteration hook rides on SIMP's log call
        ke = ke_unit(nu=sp.spec.material.nu)
        if coarse_pad not in ("voxfem", "aware"):
            raise ValueError(coarse_pad)
        self.coarse_pad = coarse_pad
        orig = sp.grid.shape if coarse_pad == "aware" else None
        if orig is not None and coarse_bc == "inject" and self._will_pad(sp, cfg):
            # without the padding clamp, small supports (washers, pads) vanish under injection on the coarse levels
            # and the coarsest operator is singular (b19); dilation keeps every support on every level
            coarse_bc = "dilate"
            self.coarse_bc = coarse_bc
        with _gmg_class(gmg_class(coarse_bc, deterministic, orig)), _fast_spill():
            super().__init__(sp.problem, cfg, ke=ke, log=self._hook)
        # passive masks on the (possibly padded) solver grid, uint8 to keep memory at 1 B/element each
        orig, g = sp.problem.grid, self.grid
        self.has_passive = sp.passive_mask is not None
        self.design = self.solid8 = self.void8 = None
        self.n_solid = 0
        if self.has_passive:
            design = np.ones(orig.n_elem, np.uint8)
            design[sp.passive_mask] = 0
            self.design = cp.asarray(pad_elements(design, orig, g))
            if sp.solid is not None:
                self.solid8 = cp.asarray(pad_elements(sp.solid.astype(np.uint8), orig, g))
                self.n_solid = int(sp.solid.sum())
            if sp.void is not None:
                void = sp.void & ~(sp.solid if sp.solid is not None else False)
                self.void8 = cp.asarray(pad_elements(void.astype(np.uint8), orig, g))
            self.n_active = sp.n_design
            self.rho *= self.design
            if self.solid8 is not None:
                self.rho += self.solid8
        self.status = "ready"

    # ---- the hook: progress and cancellation ------------------------------
    def _hook(self, msg):
        if self._user_log is not None:
            self._user_log(msg)
        if self.on_iter is not None and self.history:
            self.on_iter(self, self.history[-1])
        if self.cancel is not None and self.cancel.is_set():
            raise Cancelled()

    # ---- passive-aware pieces (identical to SIMP's when there are no passive regions) ---------------
    def _apply_passive(self, x):
        x *= self.design
        if self.solid8 is not None:
            x += self.solid8
        return x

    def physical(self, rho, beta, out):
        super().physical(rho, beta, out)
        if self.has_passive:
            self._apply_passive(out)
        return out

    def volume(self, rho_phys):
        if not self.has_passive:
            return super().volume(rho_phys)
        return (float(_sum64(rho_phys)) - self.n_solid) / self.n_active

    def sensitivities(self, p, beta):
        if not self.has_passive:
            return super().sensitivities(p, beta)
        g = self.grid
        ce = self.w1
        self.k_energy(((g.n_elem + 127) // 128,), (128,),
                      (self.KE64, self.u, ce, np.int32(g.nx), np.int32(g.ny), np.int32(g.nz)))
        self.filt.forward(self.rho, self.w2)
        tbe = math.tanh(beta * self.cfg.eta)
        den = tbe + math.tanh(beta * (1 - self.cfg.eta))
        _dproj(self.w2, np.float32(beta), np.float32(self.cfg.eta), np.float32(den), self.w2)
        _dc_phys(self.rho_phys, ce, np.float32(p), np.float32(self.cfg.Emin), np.float32(self.cfg.E0), self.w2, ce)
        ce *= self.design                         # passive rho_phys is overridden: zero derivative
        self.w2 *= self.design
        self.filt.transpose(ce, self.dc)
        self.w2 *= np.float32(1.0 / self.n_active)
        self.filt.transpose(self.w2, self.dv)

    def oc(self, move, beta):
        if not self.has_passive:
            return super().oc(move, beta)
        vf = self.cfg.volfrac
        lb = np.float32(self.cfg.rho_lb)

        def probe(lam):
            _oc_step(self.rho, self.dc, self.dv, np.float32(lam), np.float32(move), lb, self.w1)
            self._apply_passive(self.w1)          # passive raw densities stay 1 / 0
            return self.volume(self.physical(self.w1, beta, self.w2))

        probes = 0
        if self.lam:
            l1, l2 = self.lam / 4, self.lam * 4
            for _ in range(30):
                v1, v2 = probe(l1), probe(l2)
                probes += 2
                if v1 > vf >= v2:
                    break
                if v1 <= vf:
                    l1 /= 8
                if v2 > vf:
                    l2 *= 8
        else:
            l1, l2 = 1e-12, 1e9
        for _ in range(100):
            lam = math.sqrt(l1 * l2) if l1 > 0 else 0.5 * (l1 + l2)
            v = probe(lam)
            probes += 1
            if v > vf:
                l1 = lam
            else:
                l2 = lam
            if (l2 - l1) <= 1e-7 * (l1 + l2):
                break
        lam = l2
        probe(lam)
        self.rho[...] = self.w1
        self.lam = lam
        return probes

    # ---- warm start -----------------------------------------------------
    def _mask(self, x):
        if self.active is not None:
            x *= self.active
        if self.has_passive:
            self._apply_passive(x)
        return x

    def warm_start(self, rho_coarse: np.ndarray, smooth: bool = True, match_volume: bool = True) -> dict:
        """Start from a coarser physical design: 2x piecewise-constant upsampling to this grid, then

        smooth         one pass of this rung's filter, so boundary elements take intermediate values. An
                       upsampled 0/1 design is otherwise saturated under the final-phase projection (beta 32,
                       rmin 1.2): no filtered density is near 0.5 and a 0.05 move changes nothing.
        match_volume   shift the densities (bisection on a constant offset) so that the design volume after
                       filtering and projection at the schedule's first beta equals volfrac. Passive regions
                       voxelise differently at the two resolutions, so the upsampled volume is not exactly the
                       target (b06: 0.2014 vs 0.2), and OC cannot repair a saturated design.
        """
        orig = self.sp.problem.grid
        up = rho_coarse
        while up.shape != orig.shape:
            if any(2 * a > b for a, b in zip(up.shape, orig.shape)):
                raise ValueError(f"cannot upsample {rho_coarse.shape} to {orig.shape} by factors of 2")
            up = upsample2(up)
        self.rho[...] = cp.asarray(pad_elements(up.ravel().astype(np.float32), orig, self.grid))
        self._mask(self.rho)
        p0, beta0, _, r0, _ = _vsimp.schedule_at(1, self.cfg.schedule)
        self.filt.set_rmin(r0)
        if smooth:
            self.filt.forward(self.rho, self.w2)
            self.rho[...] = self.w2
            self._mask(self.rho)
        info = {"smooth": smooth, "beta0": beta0}
        if match_volume:
            vf = self.cfg.volfrac

            def vol(delta):
                cp.clip(self.rho + np.float32(delta), 0.0, 1.0, out=self.w1)
                self._mask(self.w1)
                return self.volume(self.physical(self.w1, beta0, self.w2))

            info["volume_before"] = vol(0.0)
            lo, hi = -1.0, 1.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if vol(mid) > vf:
                    hi = mid
                else:
                    lo = mid
            delta = 0.5 * (lo + hi)
            info["volume_after"] = vol(delta)
            info["delta"] = delta
            self.rho[...] = self.w1
        return info

    # ---- run + certificate ----------------------------------------------
    def run(self):
        t0 = time.perf_counter()
        self.status = "running"
        try:
            res = super().run()
            self.status = "converged" if res["converged"] else "not_converged"
        except Cancelled:
            self.status = "cancelled"
            res = self._partial("cancelled")
        except SolveFailed as ex:
            self.status = "solve_failed"
            res = self._partial("solve_failed", repr(ex))
        except RuntimeError as ex:               # voxfem raises RuntimeError on a VRAM spill
            self.status = "failed"
            res = self._partial("failed", repr(ex))
        res["status"] = self.status
        res["coarse_bc"] = self.coarse_bc
        res["deterministic"] = self.deterministic
        res["levels_overridden"] = self.levels_overridden
        res["session_wall_s"] = time.perf_counter() - t0
        res["compute_s"] = sum(h["t_iter_s"] for h in self.history)
        res["certificate"] = self.certificate(res)
        return res

    def _partial(self, status, error=None):
        h = self.history[-1] if self.history else {}
        p, beta = (h.get("p"), h.get("beta")) if h else (None, None)
        if beta is not None:
            self.physical(self.rho, beta, self.rho_phys)
        return {"converged": False, "valid": False, "termination": status, "iterations": len(self.history),
                "compliance": h.get("c"), "volume_phys": h.get("vol"), "error": error,
                "grid": list(self.orig.grid.shape), "config": None}

    def passive_check(self) -> dict:
        out = {"keep_in_ok": True, "keep_out_ok": True}
        if self.solid8 is not None:
            m = self.solid8.astype(bool)
            out["keep_in_min"] = float(cp.min(self.rho_phys[m]))
            out["keep_in_ok"] = out["keep_in_min"] == 1.0
        if self.void8 is not None:
            m = self.void8.astype(bool)
            out["keep_out_max"] = float(cp.max(self.rho_phys[m]))
            out["keep_out_ok"] = out["keep_out_max"] == 0.0
        return out

    def certificate(self, res) -> dict:
        """Release evidence. `certified` is True only if the run converged, the final state solve was accepted on
        the FP64 true residual at the configured tolerance, no VRAM spill was seen, the volume constraint holds
        and the passive regions are exact. The agent can read this; nothing outside the solver can set it."""
        fs = res.get("final_solve") or {}
        rel = fs.get("rel_true")
        pc = self.passive_check() if self.status in ("converged", "not_converged") else {}
        checks = {
            "converged": bool(res.get("converged")),
            "state_solve_accepted": rel is not None and rel <= self.cfg.tol,
            "no_vram_spill": not self.spill.spilled,
            "spill_monitored": self.cfg.spill_check_every > 0,
            "volume_ok": res.get("max_vol_error") is not None and res["max_vol_error"] <= 1e-3,
            "passive_exact": bool(pc.get("keep_in_ok", True) and pc.get("keep_out_ok", True)),
        }
        c = res.get("compliance")
        return {
            "certified": all(checks.values()),
            "checks": checks,
            "rel_true": rel, "tol": self.cfg.tol, "solver": self.cfg.solver, "coarse_bc": self.coarse_bc,
            "grid": list(self.sp.grid.shape), "n_elem": self.sp.grid.n_elem, "h": self.sp.h,
            "units": self.sp.spec.units.model_dump(),
            "compliance": c, "compliance_phys": None if c is None else c * self.sp.compliance_scale,
            "volume_phys": res.get("volume_phys"), "passive": pc, "iterations": res.get("iterations"),
            "termination": res.get("termination"),
            "stopping": {"conv_drho": self.cfg.conv_drho, "conv_dc_rel": self.cfg.conv_dc_rel,
                         "conv_window": self.cfg.conv_window, "conv_moving_ppm": self.cfg.conv_moving_ppm},
            "regularization": self.regularization(res),
            "issued_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def regularization(self, res) -> dict:
        """What the released problem's regularization is, next to the one the engineer accepted: the filter radius is
        set in element units, so a release grid finer than the specification's stated resolution has a shorter
        physical filter. Also records the continuation schedule, so a tail release (transferred design, final
        phase only) can be told from a re-optimization."""
        from .problem import grid_for
        rmin = float(_vsimp.schedule_at(10 ** 9, self.cfg.schedule)[3])
        h = float(self.sp.h)
        try:
            h_stated = float(grid_for(self.sp.spec)[0])
        except Exception:                        # the stated resolution does not fit the domain
            h_stated = None
        return {"release_h": h, "stated_h": h_stated,
                "refinement_factor": None if h_stated is None else h_stated / h,
                "filter_rmin_elements": rmin, "filter_length": rmin * h,
                "filter_length_at_stated_h": None if h_stated is None else rmin * h_stated,
                "schedule": self.schedule_name, "warm_from_h": self.warm_from_h,
                "iterations_on_grid": res.get("iterations")}
