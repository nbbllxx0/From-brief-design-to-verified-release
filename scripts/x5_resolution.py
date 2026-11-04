"""X5: why 10^8 elements. Does a coarser optimum stay competitive when the part is judged at fine resolution?

For each problem, a cold optimum at every rung of a ladder of halving element size (filter radius in element units,
as the GUI runs it, so finer rungs can form thinner members). Every design is then turned into a part:
    upsampled (nearest) to the finest grid, binarized at the threshold that gives exactly the volume fraction of the
    design elements (passive solids stay solid), and evaluated at the finest grid (one state solve; solid/void).
So every design is judged by the same fine model of the same manufactured volume. Reported per rung:
    C_part      compliance of that part at the finest grid (physical units), and its gap to the finest rung's part
    C_own       the optimizer's compliance at its own rung (grey, p 4.5), for reference
    thickness   member thickness in mm: 2 x the distance-to-void at ridge points of the solid (local maxima of the
                distance transform), p5 / median, on the design's own grid
    betti       components, tunnels, cavities of the part (pooled to <= 9M elements)
    stl         triangles, export time and file size of the rho = 0.5 surface at the design's own grid

Problems:
    span_bridge   1280 x 160 x 160 mm (L/h up to 1280; 160 = 20 x 8 halves cleanly for multigrid; the finest rung,
                  33M, fits mg64 beside the desktop's VRAM, 59M did not), solid 8 mm deck, pinned bottom end edges, uniform deck
                  load, volume fraction 0.15 (at 0.10 the 8M rung loses its load path at the first continuation
                  step and the solve fails)
    b06           the motor mount brief (keep-in bosses and pad), 0.14M -> 74M

    python -B scripts/x5_resolution.py [problem ...]     (one GPU job at a time; resumes per problem)
Writes results/X5/resolution.json; the finest designs are kept as float16 in results/X5/fields/.
"""
import gc
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from agentto.escalation import run_arm  # noqa: E402
from agentto.export import export_stl  # noqa: E402
from agentto.problem import build  # noqa: E402
from agentto.shape import betti, pool, pool_factor  # noqa: E402
from agentto.spec import Spec  # noqa: E402

OUT = ROOT / "results" / "X5" / "resolution.json"
FIELDS = ROOT / "results" / "X5" / "fields"


def cfg(n_elem: int, guarded: bool) -> dict:
    return dict(solver="mg64" if n_elem <= 6.6e7 else "mg_ir", op32="stencil", spill_check_every=10 if guarded else 0,
                log_every=0, conv_moving_ppm=100.0, max_outer=300)


def problems():
    bridge = {"schema_version": "agentto.spec/v0", "name": "X5 span bridge", "units": {"length": "mm", "force": "N"},
              "domain": {"size": [1280, 160, 160]}, "material": {"name": "steel", "E": 210000.0, "nu": 0.3},
              "volume_fraction": 0.15,
              "supports": [{"id": "abutment_left", "region": {"kind": "edge", "axis": "z", "at": [0, 0]}},
                           {"id": "abutment_right", "region": {"kind": "edge", "axis": "z", "at": [1280, 0]}}],
              "loads": [{"id": "deck_load", "region": {"kind": "face", "face": "y+"}, "force": [0, -32000, 0]}],
              "passive": [{"id": "deck", "kind": "solid", "region": {"kind": "box", "min": [0, 152, 0],
                                                                       "max": [1280, 160, 160]}}],
              "resolution": {"h": 1.0}}
    b06 = json.loads((ROOT / "briefs" / "intended" / "b06_motor_mount.json").read_text())["spec"]
    return {"span_bridge": (Spec.model_validate(bridge), [8.0, 4.0, 2.0, 1.0]),
            "b06": (Spec.model_validate(b06), [2.0, 1.0, 0.5, 0.25])}


def part(rho: np.ndarray, sp, vf: float) -> np.ndarray:
    """Solid/void part with exactly vf of the design elements solid (passive solids solid, voids void)."""
    r = rho.ravel()
    design = np.ones(r.size, bool)
    for m in (sp.solid, sp.void):
        if m is not None:
            design &= ~m
    k = int(round(vf * design.sum()))
    out = np.zeros(r.size, np.float32)
    idx = np.flatnonzero(design)
    order = np.argpartition(-r[idx], k - 1)[:k]
    out[idx[order]] = 1.0
    if sp.solid is not None:
        out[sp.solid] = 1.0
    return out.reshape(rho.shape)


def upsample(x: np.ndarray, f: int) -> np.ndarray:
    for ax in range(3):
        x = np.repeat(x, f, axis=ax)
    return x


def evaluate(sp, rho_phys: np.ndarray, kw: dict, p: float = 4.5) -> float:
    """Physical compliance of a fixed density at sp's grid (one state solve)."""
    import cupy as cp
    from voxfem.core import pad_elements
    from voxfem.simp import Config, _modulus
    from agentto.solver import SessionSIMP
    s = SessionSIMP(sp, Config(volfrac=sp.spec.volume_fraction, **kw))
    s.rho_phys[...] = cp.asarray(pad_elements(rho_phys.ravel().astype(np.float32), sp.problem.grid, s.grid))
    _modulus(s.rho_phys, np.float32(p), np.float32(s.cfg.Emin), np.float32(s.cfg.E0), s.E32)
    if s.active is not None:
        s.E32 *= s.active
    s.solve()
    if kw.get("spill_check_every"):
        s.spill.check("evaluate")
        if s.spill.spilled:
            raise RuntimeError("VRAM oversubscription during the evaluation solve")
    c = float(s.load.dot(s.u)) * sp.compliance_scale
    if not np.isfinite(c):
        raise RuntimeError("non-finite compliance in the evaluation solve")
    del s
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return c


def thickness(m: np.ndarray, h: float) -> dict:
    d = ndimage.distance_transform_edt(m).astype(np.float32)
    ridge = m & (d >= 1) & (d == ndimage.maximum_filter(d, size=3))
    t = 2.0 * d[ridge] * h
    del d
    if t.size == 0:
        return {"p5_mm": None, "median_mm": None, "n_ridge": 0}
    return {"p5_mm": float(np.percentile(t, 5)), "median_mm": float(np.median(t)), "n_ridge": int(t.size),
            "min_resolvable_mm": 2.4 * h}


def main(only=None):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    FIELDS.mkdir(parents=True, exist_ok=True)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, (spec, hs) in problems().items():
        if name in res or (only and name not in only):
            continue
        fine = build(spec, h=hs[-1])
        # evaluations at the finest grid run guarded; mg_ir rejects its correction on these solid/void parts, so
        # they stay on mg64 up to 6.6e7 elements as the optimization does
        kw_fine = cfg(fine.grid.n_elem, True)
        rows = []
        for h in hs:
            sp = build(spec, h=h)
            row, rho, _ = run_arm(sp, cfg(sp.grid.n_elem, h == hs[-1]), "cold")
            if row["status"] not in ("converged", "not_converged"):
                raise RuntimeError(f"{name} h {h}: optimization {row['status']}: {row['error']}")
            row.pop("c_history", None)
            row["h"] = h
            row["c_own_phys"] = row["compliance"] * sp.compliance_scale if row["compliance"] else None
            m = part(rho, sp, spec.volume_fraction)
            f = int(round(h / hs[-1]))
            t0 = time.perf_counter()
            row["c_part"] = evaluate(fine, upsample(m, f) if f > 1 else m, kw_fine)
            row["eval_s"] = time.perf_counter() - t0
            mb = m > 0.5
            row["thickness"] = thickness(mb, h)
            row["betti"] = betti(pool(m, pf) > 0.5 if (pf := pool_factor(m.shape)) > 1 else mb)
            row["betti_pool"] = pf
            with tempfile.TemporaryDirectory() as td:
                st = export_stl(rho, h, Path(td) / "part.stl", check=False)
            row["stl"] = {k: v for k, v in st.items() if k not in ("path", "mesh")}
            if h == hs[-1]:
                np.save(FIELDS / f"{name}_cold_h{h:g}.npy", rho.astype(np.float16))
            rows.append(row)
            print(name, f"h {h:g}", sp.grid.shape, row["status"], row["iterations"], f"{row['compute_s']:.0f}s",
                  f"C_part {row['c_part']:.6g}", row["thickness"], row["betti"], flush=True)
            del rho, m, mb
            gc.collect()
        ref = rows[-1]["c_part"]
        for r in rows:
            r["c_part_gap"] = r["c_part"] / ref - 1
        res[name] = {"spec": spec.model_dump(mode="json"), "fine_h": hs[-1], "fine_grid": list(fine.grid.shape),
                     "rows": rows}
        OUT.write_text(json.dumps(res, indent=1, default=str))
    for name, r in res.items():
        print(name)
        for x in r["rows"]:
            print(f"  h {x['h']:<5g} {x['n_elem'] / 1e6:6.2f}M {x['status']:13s} it {x['iterations']:3d} "
                  f"{x['compute_s']:6.0f}s  C_part {x['c_part']:.6g} ({x['c_part_gap']:+.2%})  "
                  f"t p5 {x['thickness']['p5_mm']} med {x['thickness']['median_mm']}  b1 {x['betti']['b1']}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
