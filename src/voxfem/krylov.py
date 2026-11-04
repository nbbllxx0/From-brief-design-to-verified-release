"""Fail-closed Krylov solvers.  Every accepted solve is accepted on the TRUE
residual ||b - K u|| / ||b|| computed in FP64, never on a recursively updated
Krylov residual.

pcg64        FP64 preconditioned CG.  The recursive residual is replaced by
             the true one every `replace_every` iterations and whenever it
             claims convergence; convergence is declared only on the true
             residual.  Four FP64 vectors (x, r, p, z/q shared).
ir           mixed-precision iterative refinement: FP64 residual and update,
             correction K d = r solved by FP32 PCG with any preconditioner to
             a relative inner tolerance.  Two FP64 + four FP32 vectors.  A
             correction that does not reduce the FP64 residual is undone, the
             inner tolerance is tightened tenfold, and the solve fails closed
             once the inner tolerance would fall below `inner_floor`.
All vector updates are fused elementwise kernels (no full-size temporaries).
The load b is sparse (DOF indices and values on the device).
"""
from __future__ import annotations

import math
import time

import numpy as np
import cupy as cp


class SolveFailed(RuntimeError):
    pass


_axpy = cp.ElementwiseKernel("T a, T x", "T y", "y += a * x", "vox_axpy")
_xpby = cp.ElementwiseKernel("T z, T b", "T p", "p = z + b * p", "vox_xpby")
_axpy_mixed = cp.ElementwiseKernel("float64 a, float32 d", "float64 x", "x += a * (double)d",
                                   "vox_axpy_mixed")
_scale_cast = cp.ElementwiseKernel("float64 r, float64 s", "float32 y", "y = (float)(r * s)",
                                   "vox_scale_cast")


class DeviceLoad:
    def __init__(self, load):
        self.idx = cp.asarray(load.idx, dtype=cp.int64)
        self.val = cp.asarray(load.val, dtype=cp.float64)
        self.norm = float(np.sqrt(np.sum(np.asarray(load.val, dtype=np.float64) ** 2)))

    def residual(self, Ax, out):
        """out = b - Ax (out may alias Ax)."""
        cp.negative(Ax, out=out)
        out[self.idx] += self.val.astype(out.dtype, copy=False)
        return out

    def dot(self, x):
        """b . x  (compliance when x solves K x = b)."""
        return float(cp.dot(x[self.idx].astype(cp.float64), self.val))


def _norm(v):
    return math.sqrt(float(cp.dot(v, v)))


def pcg64(A, load: DeviceLoad, x, M, tol=1e-5, maxit=5000, replace_every=50, work=None, reset_if_worse=False):
    """A(v, out) applies the operator; M(r, out) the preconditioner (FP64 in/out)."""
    t0 = time.perf_counter()
    nd = x.size
    if work is None:
        work = [cp.empty(nd, cp.float64) for _ in range(3)]
    r, p, zq = work
    bn = load.norm
    A(x, r); load.residual(r, r)
    rel = _norm(r) / bn
    if not math.isfinite(rel):
        raise SolveFailed(f"pcg64: non-finite initial residual {rel}")
    reset = False
    if reset_if_worse and rel > 1.0:          # the warm start is worse than zero
        x.fill(0)
        A(x, r); load.residual(r, r)
        rel, reset = _norm(r) / bn, True
    stats = {"iters": 0, "matvecs": 1, "precond": 0, "true_checks": 1,
             "rel_initial": rel, "warm_start_reset": reset, "history": [(0, rel)]}
    if rel <= tol:
        stats.update(rel_true=rel, converged=True, time_s=time.perf_counter() - t0)
        return stats
    M(r, zq); stats["precond"] += 1
    rz = float(cp.dot(r, zq))
    p[...] = zq
    it = 0
    while it < maxit:
        it += 1
        A(p, zq); stats["matvecs"] += 1
        pq = float(cp.dot(p, zq))
        if not (pq > 0) or not math.isfinite(pq):
            raise SolveFailed(f"pcg64: non-positive curvature {pq} at iteration {it}")
        alpha = rz / pq
        _axpy(alpha, p, x)
        _axpy(-alpha, zq, r)
        relr = _norm(r) / bn
        if relr <= tol or it % replace_every == 0:
            A(x, r); load.residual(r, r); stats["matvecs"] += 1; stats["true_checks"] += 1
            rel = _norm(r) / bn
            stats["history"].append((it, rel))
            if not math.isfinite(rel):
                raise SolveFailed(f"pcg64: non-finite true residual at iteration {it}")
            if rel <= tol:
                stats.update(iters=it, rel_true=rel, rel_recursive_at_stop=relr, converged=True,
                             time_s=time.perf_counter() - t0)
                return stats
        M(r, zq); stats["precond"] += 1
        rz_new = float(cp.dot(r, zq))
        beta = rz_new / rz
        rz = rz_new
        _xpby(zq, beta, p)
    A(x, r); load.residual(r, r)
    rel = _norm(r) / bn
    stats.update(iters=it, rel_true=rel, converged=False, time_s=time.perf_counter() - t0)
    raise SolveFailed(f"pcg64: true residual {rel:.3e} > {tol:.1e} after {it} iterations")


class IRWorkspace:
    def __init__(self, ndof):
        self.r64 = cp.empty(ndof, cp.float64)
        self.r32 = cp.empty(ndof, cp.float32)
        self.d32 = cp.empty(ndof, cp.float32)
        self.p32 = cp.empty(ndof, cp.float32)
        self.zq32 = cp.empty(ndof, cp.float32)

    @property
    def nbytes(self):
        return sum(int(v.nbytes) for v in (self.r64, self.r32, self.d32, self.p32, self.zq32))


def _inner(A32, ws: IRWorkspace, M32, tol, maxit):
    """FP32 PCG for K d = r32 (r32 holds the RHS on entry and is consumed as
    the inner residual); d32 starts at zero.  Stops on the recursive FP32
    residual -- acceptance happens on the outer FP64 true residual."""
    d, r, p, zq = ws.d32, ws.r32, ws.p32, ws.zq32
    d.fill(0)
    bn = _norm(r)
    if bn == 0:
        return 0
    M32(r, zq)
    rz = float(cp.dot(r, zq))
    p[...] = zq
    it = 0
    while it < maxit:
        it += 1
        A32(p, zq)
        pq = float(cp.dot(p, zq))
        if not (pq > 0) or not math.isfinite(pq):
            break
        alpha = np.float32(rz / pq)
        _axpy(alpha, p, d)
        _axpy(-alpha, zq, r)
        if _norm(r) / bn <= tol:
            break
        M32(r, zq)
        rz_new = float(cp.dot(r, zq))
        beta = np.float32(rz_new / rz)
        rz = rz_new
        _xpby(zq, beta, p)
    return it


def ir(A64, A32, load: DeviceLoad, x, M32, tol=1e-5, inner_tol=1e-1, inner_maxit=500,
       max_outer=60, inner_floor=1e-6, ws: IRWorkspace | None = None):
    t0 = time.perf_counter()
    ws = ws or IRWorkspace(x.size)
    bn = load.norm
    A64(x, ws.r64); load.residual(ws.r64, ws.r64)
    rel = _norm(ws.r64) / bn
    if not math.isfinite(rel):
        raise SolveFailed(f"ir: non-finite initial residual {rel}")
    stats = {"outer": 0, "inner_iters": 0, "matvecs64": 1, "matvecs32": 0, "precond": 0,
             "rel_initial": rel, "history": [(0, 0, rel)], "rejected_steps": 0}
    tin = inner_tol
    while not (rel <= tol):          # a NaN residual can never pass
        if stats["outer"] >= max_outer:
            raise SolveFailed(f"ir: true residual {rel:.3e} after {stats['outer']} outer steps")
        stats["outer"] += 1
        scale = rel * bn                                    # = ||r||
        _scale_cast(ws.r64, 1.0 / scale, ws.r32)            # normalised FP32 RHS
        it = _inner(A32, ws, M32, tin, inner_maxit)
        stats["inner_iters"] += it
        stats["matvecs32"] += it
        stats["precond"] += it + 1 if it else 0
        _axpy_mixed(scale, ws.d32, x)
        A64(x, ws.r64); load.residual(ws.r64, ws.r64); stats["matvecs64"] += 1
        new = _norm(ws.r64) / bn
        stats["history"].append((stats["outer"], it, new))
        if not (new < rel) or not math.isfinite(new):
            _axpy_mixed(-scale, ws.d32, x)                  # undo
            stats["rejected_steps"] += 1
            tin *= 0.1
            if tin < inner_floor:
                raise SolveFailed(f"ir: correction rejected with inner tolerance {tin:.0e}")
            A64(x, ws.r64); load.residual(ws.r64, ws.r64); stats["matvecs64"] += 1
            rel = _norm(ws.r64) / bn
            if not math.isfinite(rel):
                raise SolveFailed("ir: non-finite residual after undo")
            continue
        rel = new
    if not (rel <= tol):
        raise SolveFailed(f"ir: not converged ({rel})")
    stats.update(rel_true=rel, converged=True, inner_tol_final=tin, time_s=time.perf_counter() - t0)
    return stats


class LeanIRWorkspace:
    """Four FP32 vectors; no FP64 residual vector (see ir_lean)."""

    def __init__(self, ndof):
        self.r32 = cp.empty(ndof, cp.float32)
        self.d32 = cp.empty(ndof, cp.float32)
        self.p32 = cp.empty(ndof, cp.float32)
        self.zq32 = cp.empty(ndof, cp.float32)

    @property
    def nbytes(self):
        return sum(int(v.nbytes) for v in (self.r32, self.d32, self.p32, self.zq32))


def ir_lean(R, E32, fixed, bnorm, A32, x, M32, tol=1e-5, inner_tol=1e-1, inner_maxit=500,
            max_outer=60, inner_floor=1e-6, ws: LeanIRWorkspace | None = None, reset_if_worse=False):
    """Iterative refinement whose FP64 residual b - K u is formed by the fused
    kernel R (voxfem.operators.Residual32: FP64 arithmetic, result rounded to
    FP32 once).  Its norm is accumulated in FP64 and is the acceptance test;
    the rounding of the residual entries perturbs that norm by O(1e-7)
    relative, far below the 1e-5 tolerance."""
    t0 = time.perf_counter()
    ws = ws or LeanIRWorkspace(x.size)
    R(E32, x, ws.r32, fixed)
    rel = R.norm(ws.r32) / bnorm
    if not math.isfinite(rel):
        raise SolveFailed(f"ir_lean: non-finite initial residual {rel}")
    reset = False
    if reset_if_worse and rel > 1.0:          # the warm start is worse than zero
        x.fill(0)
        R(E32, x, ws.r32, fixed)
        rel, reset = R.norm(ws.r32) / bnorm, True
    stats = {"outer": 0, "inner_iters": 0, "matvecs64": 1, "matvecs32": 0, "precond": 0,
             "rel_initial": rel, "warm_start_reset": reset, "history": [(0, 0, rel)], "rejected_steps": 0}
    tin = inner_tol
    while not (rel <= tol):
        if stats["outer"] >= max_outer:
            raise SolveFailed(f"ir_lean: true residual {rel:.3e} after {stats['outer']} outer steps")
        stats["outer"] += 1
        scale = rel * bnorm
        ws.r32 *= np.float32(1.0 / scale)
        it = _inner(A32, ws, M32, tin, inner_maxit)
        stats["inner_iters"] += it
        stats["matvecs32"] += it
        stats["precond"] += it + 1 if it else 0
        _axpy_mixed(scale, ws.d32, x)
        R(E32, x, ws.r32, fixed); stats["matvecs64"] += 1
        new = R.norm(ws.r32) / bnorm
        stats["history"].append((stats["outer"], it, new))
        if not (new < rel) or not math.isfinite(new):
            _axpy_mixed(-scale, ws.d32, x)
            stats["rejected_steps"] += 1
            tin *= 0.1
            if tin < inner_floor:
                raise SolveFailed(f"ir_lean: correction rejected with inner tolerance {tin:.0e}")
            R(E32, x, ws.r32, fixed); stats["matvecs64"] += 1
            rel = R.norm(ws.r32) / bnorm
            if not math.isfinite(rel):
                raise SolveFailed("ir_lean: non-finite residual after undo")
            continue
        rel = new
    stats.update(rel_true=rel, converged=True, inner_tol_final=tin, time_s=time.perf_counter() - t0)
    return stats


def pcg32_recursive(R, E32, fixed, bnorm, A32, x, M32, tol=1e-5, maxit=5000,
                    ws: LeanIRWorkspace | None = None, reset_if_worse=False):
    """CONTROL, not an accepted solver: FP32 PCG whose iterate is accumulated
    in FP64 and which stops when its recursively updated FP32 residual
    satisfies ||r||/||b|| <= tol.  The initial residual is formed by R (FP64
    arithmetic, rounded to FP32).  The true FP64 residual of the returned
    iterate is measured once and recorded (rel_true) but never tested, so
    this function does not fail closed on accuracy; it fails only on a
    non-finite or non-positive curvature, or when maxit is reached."""
    t0 = time.perf_counter()
    ws = ws or LeanIRWorkspace(x.size)
    r, p, zq = ws.r32, ws.p32, ws.zq32
    R(E32, x, r, fixed)
    rel0 = R.norm(r) / bnorm
    reset = False
    if reset_if_worse and rel0 > 1.0:
        x.fill(0)
        R(E32, x, r, fixed)
        rel0, reset = R.norm(r) / bnorm, True
    stats = {"iters": 0, "matvecs32": 0, "precond": 0, "rel_initial": rel0, "warm_start_reset": reset}
    mon = rel0
    if mon > tol:
        M32(r, zq); stats["precond"] += 1
        rz = float(cp.dot(r, zq))
        p[...] = zq
        it = 0
        while True:
            if it >= maxit:
                raise SolveFailed(f"pcg32_recursive: no monitored pass after {it} iterations ({mon:.3e})")
            it += 1
            A32(p, zq)
            pq = float(cp.dot(p, zq))
            if not (pq > 0) or not math.isfinite(pq):
                raise SolveFailed(f"pcg32_recursive: curvature {pq}")
            alpha = rz / pq
            _axpy_mixed(alpha, p, x)
            _axpy(np.float32(-alpha), zq, r)
            mon = _norm(r) / bnorm
            if mon <= tol:
                break
            M32(r, zq); stats["precond"] += 1
            rz_new = float(cp.dot(r, zq))
            _xpby(zq, np.float32(rz_new / rz), p)
            rz = rz_new
        stats["iters"] = it
        stats["matvecs32"] = it
    R(E32, x, r, fixed)                       # measured only; not an acceptance test
    stats.update(rel_recursive_at_stop=mon, rel_true=R.norm(r) / bnorm, converged=True,
                 time_s=time.perf_counter() - t0)
    return stats


class JacobiPrecond:
    """z = D^{-1} r with D the operator diagonal (fixed DOFs: identity)."""

    def __init__(self, dinv):
        self.dinv = dinv

    def __call__(self, r, out):
        cp.multiply(r, self.dinv, out=out)
        return out


class CastPrecond:
    """Wrap an FP32 preconditioner for an FP64 Krylov method."""

    def __init__(self, M32, ndof):
        self.M32 = M32
        self.r32 = cp.empty(ndof, cp.float32)
        self.z32 = cp.empty(ndof, cp.float32)

    def __call__(self, r, out):
        self.r32[...] = r
        self.M32(self.r32, self.z32)
        out[...] = self.z32
        return out
