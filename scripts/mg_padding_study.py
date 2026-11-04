"""How void padding of the multigrid grid costs PCG iterations (finding of X5 and X3 b19, 2026-10-30).

voxfem's GMG needs every grid side divisible by 2^(levels-1); `rules.session_levels` pads the grid with void elements
to allow more levels and picks the candidate with the lowest modelled cost per iteration (coarsest dense solve plus
work per element), which ignores that padding slows PCG down. Here one cantilever (x- face fixed, a patch load on
x+) is run for 20 design iterations on grids whose short side n is varied, and for each grid every level count the
padding cap allows is tried: PCG iterations, solve time and the padded fraction are recorded.

    python -B scripts/mg_padding_study.py [voxfem|aware]     (GPU; a few minutes)
Writes results/solver/mg_padding.json (voxfem's coarse rule) or mg_padding_aware.json (`solver._PadAware`).
"""
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentto.problem import build  # noqa: E402
from agentto.rules import iter_cost  # noqa: E402
from agentto.solver import SessionSIMP  # noqa: E402
from agentto.spec import Spec  # noqa: E402

OUT = ROOT / "results" / "solver"


def spec(n):
    return Spec.model_validate({
        "name": f"pad study n={n}", "domain": {"size": [256, n, 128]}, "volume_fraction": 0.3,
        "supports": [{"id": "wall", "region": {"kind": "face", "face": "x-"}}],
        "loads": [{"id": "tip", "region": {"kind": "patch", "face": "x+", "center": [n / 2, 64], "size": [8, 8]},
                   "force": [0, -1000, 0]}],
        "resolution": {"h": 1.0}})


def main(rule="voxfem"):
    import cupy as cp
    from voxfem.core import Grid
    from voxfem.simp import Config
    rows = []
    for n in (64, 60, 56, 50, 48, 40, 36, 33):
        sp = build(spec(n), h=1.0)
        g = sp.grid
        for L in range(3, 8):
            f = 2 ** (L - 1)
            gp = Grid(*[-(-a // f) * f for a in g.shape])
            pad = gp.n_elem / g.n_elem - 1
            if pad > 0.6:
                continue
            try:
                s = SessionSIMP(sp, Config(volfrac=0.3, solver="mg64", op32="stencil", spill_check_every=0, log_every=0,
                                           max_outer=20, mg_levels=L), levels="manual", coarse_pad=rule)
                s.run()                                  # 20 design iterations: uniform start, then a forming design
                its = [h["solve"]["iters"] for h in s.history]
                walls = [h["solve"]["wall_s"] for h in s.history]
                rows.append({"n": n, "grid": list(g.shape), "levels": L, "padded": list(s.grid.shape),
                             "pad_frac": round(pad, 4), "pcg_iters_first": its[0], "pcg_iters_last": its[-1],
                             "pcg_iters_mean": sum(its) / len(its), "solve_s_mean": sum(walls) / len(walls),
                             "model_cost": iter_cost(gp, L)})
                print(rows[-1], flush=True)
            except Exception as ex:                      # noqa: BLE001 -- e.g. a coarsest level that is too large
                print(n, L, "skipped:", repr(ex)[:120], flush=True)
            finally:
                s = None
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("mg_padding.json" if rule == "voxfem" else f"mg_padding_{rule}.json")).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:2])
