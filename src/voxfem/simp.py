"""SIMP compliance minimisation driver, memory-lean, fail-closed.

Formulation (identical across every solver configuration):
    min_rho  b . u   s.t.  K(E(rho_phys)) u = b,  mean(rho_phys) = V_f,  0 <= rho <= 1
    rho_bar  = cone filter of rho (radius rmin, element units)
    rho_phys = smoothed Heaviside projection of rho_bar (beta, eta = 0.5)
    E        = Emin + (E0 - Emin) rho_phys^p,  E0 = 1, Emin = 1e-9
The raw design variable has lower bound 0 (no separate raw-density floor);
Emin alone keeps K nonsingular.

Continuation (penalty p, projection beta, move limit, filter radius) follows
the R2 schedule.  After its last phase the run continues until BOTH the
physical design change max|d rho_phys| <= 0.01 and the relative spread of the
compliance over the last 10 iterates <= 1e-4 hold; `max_outer` is a guard and
a run that reaches it is reported as not converged.  The reported design is
the final iterate, re-solved once more after the last update.

Every state solve is accepted only on the FP64 true residual (krylov.py).
Solver configurations:
    jacobi64   FP64 PCG, Jacobi                      (R2 FP64 reference)
    jacobi_ir  FP64 iterative refinement, FP32 Jacobi-PCG corrections (R2 best)
    mg64       FP64 PCG, FP32 Galerkin V-cycle preconditioner
    mg_ir      FP64 iterative refinement, FP32 V-cycle-PCG corrections
Control (NOT an accepted solver; used only to measure what a recursive
single-precision stopping rule does to a complete optimization):
    mg32_rec   FP32 V-cycle-PCG, FP64 iterate, stop on the recursive FP32
               residual at tol; the true residual is recorded, never tested.
               After the loop the final design is additionally solved by
               mg_ir to 1e-7 and that certified compliance is recorded.
    mg32_rec64 as mg32_rec, but the operator inside the FP32 iteration is
               evaluated in FP64 arithmetic with the FP64 element matrix
               (MixedOperator); the V-cycle is unchanged.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import cupy as cp

from .core import Problem, ke_unit, pad_elements, unpad_elements
from .operators import Operator, Diagonal, Residual32, MixedOperator, _get_kernel, _ctype
from .filters import ConeFilter
from .krylov import (DeviceLoad, pcg64, ir_lean, pcg32_recursive, LeanIRWorkspace, JacobiPrecond,
                     CastPrecond, SolveFailed)
from .mg import GMG, choose_levels
from . import guard

SCHEDULE = [  # (until_iter, penal, beta, move, rmin)  -- R2 schedule
    (15, 1.5, 1.0, 0.20, 1.50),
    (40, 3.5, 4.0, 0.15, 1.35),
    (65, 4.5, 16.0, 0.08, 1.25),
    (120, 4.5, 32.0, 0.05, 1.20),
]


@dataclass
class Config:
    volfrac: float = 0.30
    solver: str = "mg_ir"
    op64: str = "node_hoist"
    op32: str = "stencil"
    op32_inner: str | None = None     # FP32 operator of the IR corrections; None: same as op32 (V-cycle fine level)
    tol: float = 1e-5
    inner_tol: float = 1e-1
    inner_maxit: int = 300
    pcg_maxit: int = 20000
    mg_nu: int = 2
    mg_omega: float | None = None   # None: per-level 1.2/lambda_max(D^-1 A)
    mg_levels: int | None = None
    max_coarse_dofs: int = 2500
    Emin: float = 1e-9
    E0: float = 1.0
    rho_lb: float = 0.0
    eta: float = 0.5
    max_outer: int = 300
    min_outer: int = 0                # no stopping test may end the run before this iteration (0: off)
    conv_drho: float = 0.01
    conv_dc_rel: float = 1e-4
    conv_window: int = 10
    conv_moving_ppm: float = 1.0      # bounded-oscillation rule: <= 1 ppm of elements still moving
    schedule: list = field(default_factory=lambda: [list(s) for s in SCHEDULE])
    spill_check_every: int = 10
    log_every: int = 1


def schedule_at(it, sch):
    for until, p, beta, move, rmin in sch:
        if it <= until:
            return p, beta, move, rmin, until == sch[-1][0]
    p, beta, move, rmin = sch[-1][1:]
    return p, beta, move, rmin, True


_proj = cp.ElementwiseKernel(
    "float32 rb, float32 beta, float32 eta, float32 tbe, float32 den", "float32 rp",
    "rp = fminf(fmaxf((tbe + tanhf(beta * (rb - eta))) / den, 0.0f), 1.0f)", "vox_proj")
_dproj = cp.ElementwiseKernel(
    "float32 rb, float32 beta, float32 eta, float32 den", "float32 d",
    "float c = coshf(beta * (rb - eta)); d = beta / (den * c * c)", "vox_dproj")
_modulus = cp.ElementwiseKernel(
    "float32 rp, float32 p, float32 emin, float32 e0", "float32 E32",
    "double v = (double)emin + ((double)e0 - (double)emin) * pow(fmax((double)rp, 0.0), (double)p); E32 = (float)v;",
    "vox_modulus")
_dc_phys = cp.ElementwiseKernel(
    "float32 rp, float32 ce, float32 p, float32 emin, float32 e0, float32 dpr", "float32 g",
    "g = -p * (e0 - emin) * powf(fmaxf(rp, 0.0f), p - 1.0f) * ce * dpr", "vox_dcphys")
_oc_step = cp.ElementwiseKernel(
    "float32 rho, float32 dc, float32 dv, float32 lam, float32 move, float32 lb", "float32 rn",
    """float B = sqrtf(fmaxf(-dc, 1e-30f) / (lam * fmaxf(dv, 1e-30f)));
       float v = rho * B;
       v = fminf(fmaxf(v, rho - move), rho + move);
       rn = fminf(fmaxf(v, lb), 1.0f);""", "vox_oc")

_sum64 = cp.ReductionKernel("float32 x", "float64 y", "(double)x", "a + b", "y = a", "0", "vox_sum64")
_gray64 = cp.ReductionKernel("float32 x", "float64 y", "(double)x * (1.0 - (double)x)", "a + b", "y = a", "0",
                             "vox_gray64")
_maxabsdiff = cp.ReductionKernel("float32 a, float16 b", "float32 y", "fabsf(a - (float)b)", "fmaxf(a, b)", "y = a",
                                 "0", "vox_maxabsdiff")

_count_gt = cp.ReductionKernel("float32 a, float16 b, float32 thr", "int64 y", "(fabsf(a - (float)b) > thr) ? 1 : 0",
                              "a + b", "y = a", "0", "vox_count_gt")

_SRC_ENERGY = r"""
extern "C" __global__ void vox_energy(const double* __restrict__ KE, const double* __restrict__ u,
    float* __restrict__ ce, const int nx, const int ny, const int nz)
{
    __shared__ double Ks[576];
    for (int i = threadIdx.x; i < 576; i += blockDim.x) Ks[i] = KE[i];
    __syncthreads();
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nx * ny * nz) return;
    const int ez = e % nz, t = e / nz, ey = t % ny, ex = t / ny;
    const int sy = nz + 1, sx = (ny + 1) * (nz + 1);
    const int nb = ex * sx + ey * sy + ez;
    int nd[8];
    nd[0] = nb;           nd[1] = nb + sx;
    nd[2] = nb + sx + sy; nd[3] = nb + sy;
    nd[4] = nb + 1;       nd[5] = nb + sx + 1;
    nd[6] = nb + sx+sy+1; nd[7] = nb + sy + 1;
    double ue[24];
    #pragma unroll
    for (int m = 0; m < 8; ++m) { const int d = 3 * nd[m]; ue[3*m] = u[d]; ue[3*m+1] = u[d+1]; ue[3*m+2] = u[d+2]; }
    double acc = 0;
    #pragma unroll 4
    for (int i = 0; i < 24; ++i) {
        double row = 0;
        #pragma unroll
        for (int j = 0; j < 24; ++j) row += Ks[i * 24 + j] * ue[j];
        acc += row * ue[i];
    }
    ce[e] = (float)acc;
}
"""


class SIMP:
    def __init__(self, problem: Problem, cfg: Config, ke=None, log=print):
        self.cfg, self.log = cfg, log
        self.ke = ke_unit() if ke is None else np.asarray(ke)
        self.orig = problem
        uses_mg = cfg.solver.startswith("mg")
        if uses_mg:
            L = cfg.mg_levels or choose_levels(problem.grid.padded(4) if not problem.grid.coarsenable(4)
                                               else problem.grid, cfg.max_coarse_dofs)
            self.problem = problem.padded(L) if not problem.grid.coarsenable(L) else problem
            L = cfg.mg_levels or choose_levels(self.problem.grid, cfg.max_coarse_dofs)
            self.mg_levels = L
        else:
            self.problem = problem
            self.mg_levels = None
        P = self.problem
        g = self.grid = P.grid
        self.padded = P.grid != problem.grid
        self.fixed = cp.asarray(P.fixed, dtype=cp.uint8)
        self.load = DeviceLoad(P.load)
        self.mem = guard.PeakMemory()
        n, nd = g.n_elem, g.ndof
        f32 = cp.float32
        # design fields (FP32)
        self.rho = cp.full(n, cfg.volfrac, dtype=f32)
        self.active = None
        if self.padded:
            self.active = cp.asarray(pad_elements(np.ones(problem.grid.n_elem, np.float32), problem.grid, g))
            self.rho *= self.active
        self.rho_phys = cp.empty(n, f32)
        self.rho_prev = cp.zeros(n, cp.float16)     # previous physical field, for max|d rho| only
        self.E32 = cp.empty(n, f32)          # the moduli of the system, used by every operator
        self.dc = cp.empty(n, f32)
        self.dv = cp.empty(n, f32)
        self.w1 = cp.empty(n, f32)
        self.w2 = cp.empty(n, f32)
        self.u = cp.zeros(nd, cp.float64)
        self.filt = ConeFilter(g, cfg.schedule[0][4], np.float32)
        self.op64 = Operator(g, self.ke, cfg.op64, np.float64, e_dtype=np.float32)
        self.KE64 = cp.asarray(self.ke)
        self.k_energy = _get_kernel(_SRC_ENERGY, "vox_energy")
        self.n_active = problem.grid.n_elem
        # solver
        self.mg = None
        if uses_mg:
            self.mg = GMG(g, self.fixed, self.ke, fine_variant=cfg.op32, levels=self.mg_levels,
                          nu=cfg.mg_nu, omega=cfg.mg_omega, max_coarse_dofs=cfg.max_coarse_dofs)
            self.fixed = self.mg.fixed0
        if cfg.solver.endswith("_ir") or cfg.solver in ("mg32_rec", "mg32_rec64"):
            self.op32 = Operator(g, self.ke, cfg.op32_inner or cfg.op32, np.float32)
            if cfg.solver == "mg32_rec64":
                self.opmix = MixedOperator(g, self.ke)
            self.ws = LeanIRWorkspace(nd)
            self.res32 = Residual32(g, self.ke, P.load.idx, P.load.val)
            if self.mg is not None:
                self.mg.scratch0 = self.ws.p32       # power-iteration scratch, free during setup
        else:
            self.pcg_work = [cp.empty(nd, cp.float64) for _ in range(3)]
        if cfg.solver.startswith("jacobi"):
            self.diag = Diagonal(g, self.ke, np.float64 if cfg.solver == "jacobi64" else np.float32,
                                 e_dtype=np.float32)
            self.dinv = cp.empty(nd, np.float64 if cfg.solver == "jacobi64" else np.float32)
        if cfg.solver == "mg64":
            self.castM = CastPrecond(self.mg, nd)
            self.mg.scratch0 = self.castM.z32
        self.history = []
        self.lam = None
        self.spill = guard.SpillWatch()
        self.mem.sample()

    # ---- pieces ----------------------------------------------------------
    def physical(self, rho, beta, out):
        self.filt.forward(rho, out)
        tbe = math.tanh(beta * self.cfg.eta)
        den = tbe + math.tanh(beta * (1 - self.cfg.eta))
        _proj(out, np.float32(beta), np.float32(self.cfg.eta), np.float32(tbe), np.float32(den), out)
        if self.active is not None:
            out *= self.active
        return out

    def volume(self, rho_phys):
        return float(_sum64(rho_phys)) / self.n_active

    def A64(self, x, y):
        return self.op64.apply(x, self.E32, y, fixed=self.fixed)

    def A32(self, x, y):
        return self.op32.apply(x, self.E32, y, fixed=self.fixed)

    def Amix(self, x, y):
        return self.opmix.apply(x, self.E32, y, fixed=self.fixed)

    def solve(self):
        cfg = self.cfg
        t0 = time.perf_counter()
        setup = 0.0
        if self.mg is not None:
            ts = time.perf_counter()
            self.mg.update(self.E32)
            cp.cuda.Stream.null.synchronize()
            setup = time.perf_counter() - ts
        if cfg.solver == "jacobi64":
            self.diag(self.E32, fixed=self.fixed, out=self.dinv)
            cp.reciprocal(self.dinv, out=self.dinv)
            st = pcg64(self.A64, self.load, self.u, JacobiPrecond(self.dinv), cfg.tol, cfg.pcg_maxit,
                       work=self.pcg_work, reset_if_worse=True)
        elif cfg.solver == "mg64":
            st = pcg64(self.A64, self.load, self.u, self.castM, cfg.tol, cfg.pcg_maxit, work=self.pcg_work,
                       reset_if_worse=True)
        elif cfg.solver == "jacobi_ir":
            self.diag(self.E32, fixed=self.fixed, out=self.dinv)
            cp.reciprocal(self.dinv, out=self.dinv)
            st = ir_lean(self.res32, self.E32, self.fixed, self.load.norm, self.A32, self.u,
                         JacobiPrecond(self.dinv), cfg.tol, cfg.inner_tol, cfg.inner_maxit * 20, ws=self.ws,
                         reset_if_worse=True)
        elif cfg.solver == "mg_ir":
            st = ir_lean(self.res32, self.E32, self.fixed, self.load.norm, self.A32, self.u,
                         self.mg, cfg.tol, cfg.inner_tol, cfg.inner_maxit, ws=self.ws, reset_if_worse=True)
        elif cfg.solver in ("mg32_rec", "mg32_rec64"):
            A = self.A32 if cfg.solver == "mg32_rec" else self.Amix
            st = pcg32_recursive(self.res32, self.E32, self.fixed, self.load.norm, A, self.u,
                                 self.mg, cfg.tol, cfg.pcg_maxit, ws=self.ws, reset_if_worse=True)
        else:
            raise ValueError(cfg.solver)
        cp.cuda.Stream.null.synchronize()
        st["setup_s"] = setup
        st["wall_s"] = time.perf_counter() - t0
        st.pop("history", None)
        return st

    def sensitivities(self, p, beta):
        g = self.grid
        ce = self.w1
        self.k_energy(((g.n_elem + 127) // 128,), (128,),
                      (self.KE64, self.u, ce, np.int32(g.nx), np.int32(g.ny), np.int32(g.nz)))
        # rho_bar is needed for the projection derivative: recompute into w2
        self.filt.forward(self.rho, self.w2)
        tbe = math.tanh(beta * self.cfg.eta)
        den = tbe + math.tanh(beta * (1 - self.cfg.eta))
        _dproj(self.w2, np.float32(beta), np.float32(self.cfg.eta), np.float32(den), self.w2)  # w2 = dproj
        _dc_phys(self.rho_phys, ce, np.float32(p), np.float32(self.cfg.Emin), np.float32(self.cfg.E0),
                 self.w2, ce)                                                            # w1 = dc/drho_bar
        if self.active is not None:
            ce *= self.active
            self.w2 *= self.active
        self.filt.transpose(ce, self.dc)
        self.w2 *= np.float32(1.0 / self.n_active)
        self.filt.transpose(self.w2, self.dv)

    def oc(self, move, beta):
        vf = self.cfg.volfrac
        lb = np.float32(self.cfg.rho_lb)

        def probe(lam):
            _oc_step(self.rho, self.dc, self.dv, np.float32(lam), np.float32(move), lb, self.w1)
            if self.active is not None:
                self.w1 *= self.active
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
        lam = math.sqrt(l1 * l2)
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

    # ---- loop ------------------------------------------------------------
    def run(self):
        cfg = self.cfg
        t_start = time.perf_counter()
        comp_hist = []
        converged = False
        termination = "guard"
        it = 0
        last_pb = None
        totals = {"solve_s": 0.0, "setup_s": 0.0, "sens_s": 0.0, "oc_s": 0.0, "filter_s": 0.0}
        sync = cp.cuda.Stream.null.synchronize
        while it < cfg.max_outer:
            it += 1
            p, beta, move, rmin, last_phase = schedule_at(it, cfg.schedule)
            self.filt.set_rmin(rmin)
            t0 = time.perf_counter()
            self.physical(self.rho, beta, self.rho_phys)
            _modulus(self.rho_phys, np.float32(p), np.float32(cfg.Emin), np.float32(cfg.E0), self.E32)
            if self.active is not None:
                self.E32 *= self.active
            sync(); t1 = time.perf_counter()
            if self.mg is not None and last_pb is not None and (p, beta) != last_pb:
                self.mg.force_weights = True            # continuation step: refresh smoother weights
            last_pb = (p, beta)
            st = self.solve()
            t2 = time.perf_counter()
            c = self.load.dot(self.u)
            self.sensitivities(p, beta)
            sync(); t3 = time.perf_counter()
            drho = float(_maxabsdiff(self.rho_phys, self.rho_prev)) if it > 1 else float("nan")
            n_moving = int(_count_gt(self.rho_phys, self.rho_prev, np.float32(cfg.conv_drho))) if it > 1 else -1
            self.rho_prev[...] = self.rho_phys
            vol = self.volume(self.rho_phys)
            mnd = 4.0 * float(_gray64(self.rho_phys)) / self.n_active
            probes = self.oc(move, beta)
            sync(); t4 = time.perf_counter()
            self.mem.sample()
            totals["solve_s"] += st["wall_s"] - st["setup_s"]
            totals["setup_s"] += st["setup_s"]
            totals["sens_s"] += t3 - t2
            totals["oc_s"] += t4 - t3
            totals["filter_s"] += t1 - t0
            comp_hist.append(c)
            rec = {"it": it, "c": c, "vol": vol, "mnd": mnd, "drho_phys": drho, "n_elem_drho_gt_tol": n_moving,
                   "p": p, "beta": beta,
                   "move": move, "rmin": rmin, "oc_probes": probes, "t_iter_s": t4 - t0, "solve": st}
            self.history.append(rec)
            if cfg.log_every and it % cfg.log_every == 0:
                it_s = st.get("iters", st.get("inner_iters"))
                self.log(f"it {it:3d} c={c:.6e} vol={vol:.5f} mnd={mnd:.3f} drho={drho:.3e} p={p} b={beta} "
                         f"solve {st['wall_s']:.2f}s (setup {st['setup_s']:.2f}) it={it_s} "
                         f"outer={st.get('outer', '-')} res={st['rel_true']:.2e} iter {t4 - t0:.2f}s")
            if cfg.spill_check_every and it % cfg.spill_check_every == 0:
                self.spill.check(f"it{it}")
                if self.spill.spilled:
                    raise RuntimeError("VRAM oversubscription: shared-memory spill detected")
            if last_phase and it > cfg.schedule[-2][0] + cfg.conv_window and it >= cfg.min_outer:
                w = comp_hist[-cfg.conv_window:]
                spread = (max(w) - min(w)) / abs(np.mean(w))
                bound = max(1, int(cfg.conv_moving_ppm * 1e-6 * self.n_active))
                if spread <= cfg.conv_dc_rel and drho <= cfg.conv_drho:
                    converged, termination = True, "max_change"
                    break
                if spread <= cfg.conv_dc_rel and 0 <= n_moving <= bound:
                    converged, termination = True, "bounded_oscillation"
                    break
        # final iterate: physical field of the retained design, one more solve
        p, beta, move, rmin, _ = schedule_at(it, cfg.schedule)
        self.physical(self.rho, beta, self.rho_phys)
        _modulus(self.rho_phys, np.float32(p), np.float32(cfg.Emin), np.float32(cfg.E0), self.E32)
        if self.active is not None:
            self.E32 *= self.active
        st = self.solve()
        c_final = self.load.dot(self.u)
        certified = None
        if cfg.solver in ("mg32_rec", "mg32_rec64"):   # control: certify the returned design independently
            tc = time.perf_counter()
            stc = ir_lean(self.res32, self.E32, self.fixed, self.load.norm, self.A32, self.u,
                          self.mg, 1e-7, cfg.inner_tol, cfg.inner_maxit, ws=self.ws, max_outer=200)
            stc.pop("history", None)
            certified = {"tol": 1e-7, "compliance": self.load.dot(self.u), "solve": stc,
                         "wall_s": time.perf_counter() - tc}
        self.spill.check("end")
        self.mem.sample()
        vol = self.volume(self.rho_phys)
        mnd = 4.0 * float(_gray64(self.rho_phys)) / self.n_active
        wall = time.perf_counter() - t_start
        res = {
            "converged": converged, "termination": termination, "iterations": it, "compliance": c_final, "volume_phys": vol, "mnd": mnd,
            "final_solve": st, "wall_s": wall, "totals": totals, "memory": self.mem.report(),
            "spill": self.spill.report(), "valid": converged and not self.spill.spilled,
            "grid": list(self.orig.grid.shape), "solver_grid": list(self.grid.shape),
            "padded": self.padded, "mg_levels": self.mg_levels,
            "mg_storage": self.mg.storage_bytes() if self.mg else None,
            "mg_coarse_free_dofs": self.mg.coarse_free_dofs if self.mg else None,
            "config": asdict(cfg), "min_rho_phys": float(cp.min(self.rho_phys)),
            "max_vol_error": abs(vol - cfg.volfrac),
        }
        if certified is not None:
            res["certified_final"] = certified
        return res

    def final_density(self):
        rp = self.rho_phys
        if self.padded:
            rp = unpad_elements(rp, self.orig.grid, self.grid)
        return rp
