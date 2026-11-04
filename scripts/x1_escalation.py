"""X1: certified coarse-to-fine escalation on the GUI's own ladder (preview -> refine -> release).

Problems: the FEAD cantilever, bridge and torsion, and briefs b05 (L bracket, keep-out), b06 (motor mount, keep-in
bosses), b09 (duct bridge, keep-out duct). For each: a cold preview rung (~1M elements); then at the refine (~8M)
and release (26-74M) rungs the arms cold / tail / reopen / reopen_long (agentto.escalation), each warm arm chained
from the same arm's design one rung coarser (cold is the reference at every rung). Release rungs run with the spill
guard on (spill_check_every 10), as a certified release must.

Per run: status, certificate, iterations, compute and wall time, s/iteration, compliance, gap to cold at the same
rung, the fraction of elements that changed solid/void state from the warm start, shape agreement with cold (IoU,
surface Hausdorff in elements of the pooled grid) and Betti numbers (pooled to <= 9M elements).
Release designs are saved pooled 2x as float16 (results/X1/fields/); the cold and reopen_long release fields of b06
and b09 are also kept at full resolution for X5 and the STL export timing.

    python -B scripts/x1_escalation.py [problem ...]       (resumes: finished problems are skipped)
Writes results/X1/escalation.json after every problem.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
from agentto import library  # noqa: E402
from agentto.escalation import run_arm  # noqa: E402
from agentto.problem import build  # noqa: E402
from agentto.shape import betti, compare, pool, pool_factor  # noqa: E402
from agentto.spec import Spec  # noqa: E402
from agentto.tools import _named  # noqa: E402
from agentto.worker import ladder  # noqa: E402

OUT = ROOT / "results" / "X1" / "escalation.json"
FIELDS = ROOT / "results" / "X1" / "fields"
ARMS = ("cold", "tail", "reopen", "reopen_long")
KEEP_FULL = {("b06_motor_mount", "cold"), ("b06_motor_mount", "reopen_long"),
             ("b09_duct_bridge", "cold"), ("b09_duct_bridge", "reopen_long")}


def cfg(release: bool, n_elem: int = 0):
    return dict(solver="mg64" if n_elem <= 6.0e7 else "mg_ir", op32="stencil", spill_check_every=10 if release else 0, log_every=0,
                conv_moving_ppm=100.0, max_outer=300)


def problems():
    out = {"cantilever": library.cantilever(40), "bridge": library.bridge(40), "torsion": library.torsion(40)}
    for b in ("b05_l_bracket", "b06_motor_mount", "b09_duct_bridge"):
        out[b] = Spec.model_validate(json.loads((ROOT / "briefs" / "intended" / f"{b}.json").read_text())["spec"])
    return out


def masks(rho, f):
    return (pool(rho, f) > 0.5) if f > 1 else rho > 0.5


def main(only=None):
    rows = json.loads(OUT.read_text())["rows"] if OUT.exists() else []
    done = {r["problem"] for r in rows}
    FIELDS.mkdir(parents=True, exist_ok=True)
    for name, spec in problems().items():
        if name in done or (only and name not in only):
            continue
        rungs = {r["name"]: r for r in _named(ladder(spec))}
        t0 = time.perf_counter()
        prow, prev, _ = run_arm(build(spec, h=rungs["preview"]["h"]), cfg(False), "cold")
        prev = {a: prev for a in ARMS}                    # every chain starts from the cold preview design
        out = {"problem": name, "rungs": {k: {kk: rungs[k][kk] for kk in ("h", "grid", "n_elem")} for k in
                                          ("preview", "refine", "release")},
               "preview": prow, "refine": {}, "release": {}}
        print(name, "preview", prow["status"], prow["iterations"], f"{prow['compute_s']:.1f}s", flush=True)
        for rung in ("refine", "release"):
            sp = build(spec, h=rungs[rung]["h"])
            rel = rung == "release"
            f = pool_factor(tuple(rungs[rung]["grid"]))
            res, m, cur = {}, {}, {}
            for arm in ARMS:
                row, rho, start = run_arm(sp, cfg(rel, sp.grid.n_elem), arm, coarse=None if arm == "cold" else prev[arm])
                m[arm] = masks(rho, f)
                row["pool"] = f
                row["betti_end"] = betti(m[arm])
                if start is not None:
                    row["changed_frac"] = float(np.mean((start > 0.5) != (rho > 0.5)))
                    row["betti_start"] = betti(masks(start, f))
                cur[arm] = rho
                if rel:
                    np.save(FIELDS / f"{name}_{arm}_release_p2.npy", pool(rho, 2).astype(np.float16))
                    if (name, arm) in KEEP_FULL:
                        np.save(FIELDS / f"{name}_{arm}_release.npy", rho.astype(np.float16))
                res[arm] = row
                print(name, rung, f"{arm:12s}", row["status"], row["certified"], row["iterations"],
                      f"{row['compute_s']:.1f}s", f"{row['s_per_iter']:.2f}s/it", f"C {row['compliance'] or float('nan'):.6g}",
                      row["error"] or "",
                      f"chg {row.get('changed_frac', float('nan')):.3%}", row["betti_end"], flush=True)
                del rho, start
            c0 = res["cold"]["compliance"]
            for arm, row in res.items():
                row["gap_to_cold"] = row["compliance"] / c0 - 1 if row["compliance"] and c0 else None
                row["shape_vs_cold"] = compare(m[arm], m["cold"])
            out[rung] = res
            prev = cur
        out["wall_s"] = time.perf_counter() - t0
        rows.append(out)
        OUT.write_text(json.dumps({"rows": rows, "arms": ARMS}, indent=1))
    for r in rows:
        print(r["problem"])
        for rung in ("refine", "release"):
            for arm, v in r[rung].items():
                print(f"  {rung:7s} {arm:12s} {v['status']:13s} cert {str(v['certified']):5s} it {v['iterations']:3d} "
                      f"{v['compute_s']:6.1f}s gap {v['gap_to_cold'] if v['gap_to_cold'] is not None else float('nan'):+.3%} IoU {v['shape_vs_cold']['iou']:.3f} "
                      f"chg {v.get('changed_frac', float('nan')):.3%} b1 {v['betti_end']['b1']}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
