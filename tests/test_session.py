"""X0 gate, GPU part: SessionSIMP extensions against voxfem.SIMP and against first principles.

  identity        no passive regions + injection rule: same iterations, compliance and densities as voxfem.SIMP
  passive         keep-in = 1 and keep-out = 0 at every iteration; volume constraint over design elements
  sensitivities   central finite differences of compliance and volume w.r.t. design densities, with passive
                  regions present (16k elements)
  rigid           K(E) applied to a rigid translation is zero with keep-out (Emin) elements present
  coarse rule     dilated coarse supports contain the injected ones; a support on an odd grid line solves with
                  dilation; cancel stops a run and keeps the design
"""
import threading

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from agentto import library  # noqa: E402
from agentto.problem import build  # noqa: E402
from agentto.solver import RobustGMG, SessionSIMP, tail_schedule  # noqa: E402
from agentto.spec import Spec  # noqa: E402
from voxfem.core import PROBLEMS, ke_unit  # noqa: E402
from voxfem.operators import Operator  # noqa: E402
from voxfem.simp import SIMP, Config, _modulus  # noqa: E402


def cfg(**kw):
    base = dict(volfrac=0.30, solver="mg64", op32="stencil", spill_check_every=0, log_every=0)
    base.update(kw)
    return Config(**base)


def passive_spec(k=20):
    d = library.cantilever(k).model_dump()
    d["passive"] = [
        {"id": "hole", "kind": "void", "region": {"kind": "cylinder", "axis": "z", "center": [0.8, 0.5],
                                                   "radius": 0.2, "span": [0.0, 1.0]}},
        {"id": "boss", "kind": "solid", "region": {"kind": "box", "min": [1.8, 0.3, 0.3], "max": [2.0, 0.7, 0.7]}},
    ]
    return Spec.model_validate(d)


@pytest.mark.parametrize("name", ["cantilever", "bridge", "torsion"])
def test_identity_with_voxfem_jacobi(name):
    """jacobi64 has no atomics, so voxfem is deterministic and the session must match it bit for bit."""
    k = 8
    c = dict(solver="jacobi64", pcg_maxit=100000)
    a = SIMP(PROBLEMS[name](k), cfg(**c), log=lambda *_: None)
    ra = a.run()
    b = SessionSIMP(build(library.fead(name, k)), cfg(**c), coarse_bc="inject")
    rb = b.run()
    assert ra["iterations"] == rb["iterations"]
    assert ra["compliance"] == rb["compliance"]
    assert float(cp.max(cp.abs(a.final_density() - b.final_density()))) == 0.0
    assert rb["status"] == "converged"


@pytest.mark.parametrize("name", ["cantilever", "bridge"])
def test_deterministic_session_is_bit_reproducible(name):
    out = []
    for _ in range(3):
        s = SessionSIMP(build(library.fead(name, 16)), cfg(), coarse_bc="inject", deterministic=True)
        r = s.run()
        out.append((r["iterations"], r["compliance"], cp.asnumpy(s.final_density())))
    for it, c, d in out[1:]:
        assert it == out[0][0] and c == out[0][1] and np.array_equal(d, out[0][2])


def test_passive_invariants_every_iteration():
    sp = build(passive_spec(20))
    seen = []

    def on_iter(s, rec):
        seen.append(s.passive_check() | {"vol": rec["vol"]})

    s = SessionSIMP(sp, cfg(), on_iter=on_iter)
    r = s.run()
    assert r["status"] == "converged", r
    assert len(seen) == r["iterations"]
    assert all(x["keep_in_ok"] and x["keep_out_ok"] for x in seen)
    assert abs(r["volume_phys"] - 0.30) < 1e-3
    # volume over design elements only, computed independently on the host
    rp = cp.asnumpy(s.final_density())
    design = ~sp.passive_mask
    assert abs(rp[design].mean() - 0.30) < 1e-3
    assert r["certificate"]["checks"]["passive_exact"]


def test_finite_difference_sensitivities_with_passive():
    sp = build(passive_spec(20))                  # 40 x 20 x 20 = 16k elements
    s = SessionSIMP(sp, cfg(solver="jacobi64", tol=1e-11, pcg_maxit=200000))
    rng = np.random.default_rng(0)
    design = cp.asarray(~sp.passive_mask)
    p, beta = 3.0, 4.0
    s.filt.set_rmin(1.5)
    rho0 = cp.asarray(rng.uniform(0.2, 0.8, sp.grid.n_elem).astype(np.float32))
    s.rho[...] = rho0
    s._apply_passive(s.rho)
    base = s.rho.copy()

    def evaluate(rho):
        s.rho[...] = rho
        s.physical(s.rho, beta, s.rho_phys)
        _modulus(s.rho_phys, np.float32(p), np.float32(s.cfg.Emin), np.float32(s.cfg.E0), s.E32)
        st = s.solve()
        assert st["rel_true"] <= 1e-10
        return s.load.dot(s.u), s.volume(s.rho_phys)

    evaluate(base)
    s.sensitivities(p, beta)
    dc, dv = cp.asnumpy(s.dc).copy(), cp.asnumpy(s.dv).copy()
    idx = rng.choice(np.nonzero(cp.asnumpy(design))[0], 12, replace=False)
    scale = np.abs(dc[cp.asnumpy(design)]).max()
    for j in idx:
        up, dn = base.copy(), base.copy()
        up[j] += np.float32(1e-2)
        dn[j] -= np.float32(1e-2)
        step = float(up[j] - dn[j])
        (cu, vu), (cd, vd) = evaluate(up), evaluate(dn)
        fd_c, fd_v = (cu - cd) / step, (vu - vd) / step
        assert abs(fd_c - dc[j]) <= 2e-3 * scale + 2e-3 * abs(dc[j]), (j, fd_c, dc[j])
        assert abs(fd_v - dv[j]) <= 1e-3 * abs(dv[j]) + 1e-9, (j, fd_v, dv[j])


@pytest.mark.parametrize("variant", ["stencil", "stencil_diff", "elem_mean", "node_hoist"])
def test_rigid_translation_zero_force_with_keepout(variant):
    sp = build(passive_spec(16))
    g = sp.grid
    E = np.full(g.n_elem, 1.0, np.float32)
    E[sp.void] = 1e-9
    E[np.random.default_rng(1).random(g.n_elem) < 0.3] = 0.37
    dtype = np.float64 if variant == "node_hoist" else np.float32
    op = Operator(g, ke_unit(), variant, dtype)
    for c in range(3):
        u = np.zeros(g.ndof, dtype)
        u[c::3] = 1.0
        y = op.apply(cp.asarray(u), cp.asarray(E.astype(np.float32 if dtype == np.float32 else dtype)))
        r = cp.asarray(np.random.default_rng(2).standard_normal(g.ndof).astype(dtype))
        ref = float(cp.linalg.norm(op.apply(r, cp.asarray(E.astype(dtype)))))
        assert float(cp.linalg.norm(y)) <= 1e-6 * ref, (variant, c)


def odd_line_spec(k=20):
    d = library.bridge(k).model_dump()
    d["supports"][0]["region"]["at"] = [0.0, 1.0 / k]        # pin on the first odd node line above the bottom
    return Spec.model_validate(d)


def test_dilated_coarse_fixed_contains_injected():
    sp = build(odd_line_spec(20))
    s = SessionSIMP(sp, cfg(), coarse_bc="dilate")
    lv = s.mg.levels
    for l in range(1, len(lv)):
        inj = lv[l - 1].fixed.reshape(lv[l - 1].grid.node_shape)[::2, ::2, ::2].ravel()
        assert bool(cp.all((lv[l].fixed & inj) == inj))
    assert isinstance(s.mg, RobustGMG)


def test_odd_line_support_auto_dilates_and_converges():
    sp = build(odd_line_spec(20))
    s = SessionSIMP(sp, cfg(max_outer=40))
    assert s.coarse_bc == "dilate"
    r = s.run()
    assert all(h["solve"]["rel_true"] <= 1e-5 for h in s.history)
    assert r["final_solve"]["rel_true"] <= 1e-5


def test_cancel_keeps_design():
    ev = threading.Event()

    def on_iter(s, rec):
        if rec["it"] == 5:
            ev.set()

    s = SessionSIMP(build(library.cantilever(16)), cfg(), on_iter=on_iter, cancel=ev)
    r = s.run()
    assert r["status"] == "cancelled" and r["iterations"] == 5
    assert not r["certificate"]["certified"]
    assert float(cp.max(s.final_density())) > 0.5


def test_warm_start_tail():
    a = SessionSIMP(build(library.cantilever(10)), cfg())
    a.run()
    coarse = cp.asnumpy(a.final_density()).reshape(a.sp.grid.shape)
    b = SessionSIMP(build(library.cantilever(10), h=a.sp.h / 2), cfg(schedule=tail_schedule()))
    b.warm_start(coarse)
    b.schedule_name, b.warm_from_h = "tail", a.sp.h          # what the worker records for a warm start
    r = b.run()
    assert r["status"] == "converged" and r["iterations"] < 40
    g = r["certificate"]["regularization"]                  # release grid twice as fine as the stated resolution
    assert g["stated_h"] == a.sp.h and g["release_h"] == a.sp.h / 2 and g["refinement_factor"] == 2.0
    assert g["filter_rmin_elements"] == 1.2 and g["filter_length_at_stated_h"] == 2 * g["filter_length"]
    assert g["schedule"] == "tail" and g["warm_from_h"] == a.sp.h and g["iterations_on_grid"] == r["iterations"]


def _pad_spec(ny):
    return Spec.model_validate({
        "name": "pad", "domain": {"size": [64, ny, 32]}, "volume_fraction": 0.3,
        "supports": [{"id": "wall", "region": {"kind": "face", "face": "x-"}}],
        "loads": [{"id": "tip", "region": {"kind": "patch", "face": "x+", "center": [ny / 2, 16], "size": [4, 4]},
                   "force": [0, -100, 0]}],
        "resolution": {"h": 1.0}})


def test_pad_aware_coarse_rule_cuts_pcg_iterations_and_leaves_unpadded_grids_alone():
    def run(ny, mode):
        s = SessionSIMP(build(_pad_spec(ny), h=1.0), cfg(max_outer=8, mg_levels=4), levels="manual", coarse_pad=mode)
        r = s.run()
        return r["compliance"], np.mean([h["solve"]["iters"] for h in s.history]), s.padded
    c0, it0, padded = run(32, "voxfem")
    c1, it1, _ = run(32, "aware")
    assert not padded and c0 == c1 and it0 == it1                 # unpadded: bit-identical
    c0, it0, padded = run(25, "voxfem")                           # 25 -> 32: 28% padding
    c1, it1, _ = run(25, "aware")
    assert padded and it1 < 0.6 * it0, (it0, it1)
    assert abs(c1 / c0 - 1) < 1e-5                                # same solution within the solve tolerance
