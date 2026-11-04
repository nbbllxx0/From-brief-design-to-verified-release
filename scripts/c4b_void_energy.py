"""C4b: would a certificate predicate on the solved field catch a load inside a keep-out region?

The candidate predicate named in the paper (Section 6.4) is the share of strain energy carried by keep-out elements.
A keep-out element keeps the ersatz modulus Emin, so its strain energy is Emin * u_e^T k0 u_e; the share is that sum over
keep-out elements divided by the compliance f^T u. Solved with the rules bypassed on the same coarse grid as C4
(h = shortest side / 20, or the spec's own h where that does not divide every side; same solver settings): the b50 spec the G agent built (pad inside the keep-out), the
corrected b50 reference (pad at x = 40), and every reference reading in the suite that has a keep-out region
(well-posed problems, to see whether the predicate would raise false alarms).

    python -B scripts/c4b_void_energy.py
"""
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "C4"
VOIDS = ["b05_l_bracket", "b09_duct_bridge", "b18_cantilever_hole", "b28_stool", "b30_bridge_clearance",
         "b41_hinge_bracket", "b43_press_riser", "b46_sensor_housing", "b50_valve_bracket"]


def main():
    import cupy as cp
    import numpy as np
    from agentto.problem import build
    from agentto.rules import check
    from agentto.solver import SessionSIMP
    from agentto.spec import Spec
    from voxfem.simp import Config

    g_rec = json.loads((ROOT / "results/X2/G/b50_valve_bracket_r1.json").read_text(encoding="utf-8"))["spec"]
    cases = [("b50_valve_bracket", "G spec, pad inside keep-out (rules off)", g_rec)]
    ref = json.loads(json.dumps(g_rec))
    ref["loads"][0]["region"]["center"][0] = 40.0
    cases.append(("b50_valve_bracket", "G spec, pad moved to x = 40 (C4 reference)", ref))
    for t in VOIDS:
        d = json.loads((ROOT / "briefs/intended" / f"{t}.json").read_text(encoding="utf-8"))
        cases.append((t, f"reference reading ({d['expected']})", d["spec"]))

    rows = []
    for task, label, sd in cases:
        spec = Spec.model_validate(sd)
        rep = check(spec)
        h = min(spec.domain.size) / 20
        if any(abs(L / h - round(L / h)) > 1e-9 for L in spec.domain.size):
            h = spec.resolution.h               # b41: the shortest side / 20 does not divide every side
        row = {"task": task, "case": label, "rule_errors": sorted({f.code for f in rep.findings if f.severity == "error"}),
               "h": h}
        t0 = time.perf_counter()
        try:
            sp = build(spec, h=h)
            kw = dict(volfrac=spec.volume_fraction, solver="mg64", op32="stencil", log_every=0,
                      conv_moving_ppm=100.0, spill_check_every=10, max_outer=150)
            m = SessionSIMP(sp, Config(**kw), coarse_pad="aware")
            r = m.run()
            cert = r.get("certificate") or {}
            gr = m.grid
            ce = cp.zeros(gr.n_elem, dtype=m.w1.dtype)
            m.k_energy(((gr.n_elem + 127) // 128,), (128,),
                       (m.KE64, m.u, ce, np.int32(gr.nx), np.int32(gr.ny), np.int32(gr.nz)))
            void_energy = float(m.cfg.Emin * cp.sum(ce * m.void8)) if m.void8 is not None else 0.0
            c_solver = float(r["compliance"])
            row.update(grid=list(sp.grid.shape), n_elem=sp.grid.n_elem, status=r["status"], iterations=r["iterations"],
                       certified=cert.get("certified"), compliance=c_solver * sp.compliance_scale,
                       void_energy_share=void_energy / c_solver if c_solver else None,
                       n_void=int(cp.sum(m.void8)) if m.void8 is not None else 0)
            m = None
        except Exception as ex:                         # noqa: BLE001
            row.update(status="exception", error=f"{type(ex).__name__}: {ex}")
        row["wall_s"] = time.perf_counter() - t0
        print(json.dumps({k: row.get(k) for k in ("task", "case", "rule_errors", "status", "certified",
                                                  "void_energy_share")}), flush=True)
        rows.append(row)
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "void_energy.json").write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
