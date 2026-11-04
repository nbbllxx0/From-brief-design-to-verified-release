"""X1 pilot 2: do any warm arms actually refine? One coarse -> fine doubling, all arms in agentto.escalation.

For each arm: iterations, compute time and s/iteration, compliance gap to its reference (cold, or cold_phys for the
physical-filter arms), the fraction of elements whose solid/void state changed from the warm start, shape agreement
with the reference (IoU, surface Hausdorff in fine elements) and Betti numbers of the start and end designs (new
members show up as b1 changes: an upsampled design that only gets certified keeps the coarse topology).
Cases: FEAD cantilever and bridge k 40 -> 80, brief b06 (passive bosses) h 2 -> 1.
Writes results/X1/refine_pilot.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
from agentto import library  # noqa: E402
from agentto.escalation import ARMS, reference_for, run_arm  # noqa: E402
from agentto.problem import build  # noqa: E402
from agentto.shape import betti, compare  # noqa: E402
from agentto.spec import Spec  # noqa: E402

CFG = dict(solver="mg64", op32="stencil", spill_check_every=0, log_every=0, conv_moving_ppm=100.0, max_outer=300)
OUT = ROOT / "results" / "X1" / "refine_pilot.json"


def brief(slug):
    return Spec.model_validate(json.loads((ROOT / "briefs" / "intended" / f"{slug}.json").read_text())["spec"])


def main(only=None):
    cases = [("cantilever_k40_to_k80", library.cantilever(40), 1 / 40, 1 / 80),
             ("bridge_k40_to_k80", library.bridge(40), 1 / 40, 1 / 80),
             ("b06_h2_to_h1", brief("b06_motor_mount"), 2.0, 1.0)]
    rows = json.loads(OUT.read_text())["rows"] if OUT.exists() else []
    done = {r["case"] for r in rows}
    for name, spec, hc, hf in cases:
        if name in done or (only and name not in only):
            continue
        crow, coarse, _ = run_arm(build(spec, h=hc), CFG, "cold")
        sp = build(spec, h=hf)
        res, dens = {}, {}
        for arm in ARMS:
            row, rho, start = run_arm(sp, CFG, arm, coarse=coarse)
            dens[arm] = rho > 0.5
            row["betti_end"] = betti(dens[arm])
            if start is not None:
                s = start > 0.5
                row["changed_frac"] = float(np.mean(s != dens[arm]))
                row["betti_start"] = betti(s)
            res[arm] = row
            print(name, f"{arm:17s}", row["status"], row["iterations"], f"{row['compute_s']:.1f}s",
                  f"{row['s_per_iter']*1e3:.0f}ms/it", f"C {row['compliance']:.5g}",
                  f"chg {row.get('changed_frac', float('nan')):.3%}", row["betti_end"], flush=True)
        for arm, row in res.items():
            ref = reference_for(arm)
            row["reference"] = ref
            row["gap_to_ref"] = row["compliance"] / res[ref]["compliance"] - 1
            row["gap_to_cold"] = row["compliance"] / res["cold"]["compliance"] - 1
            row["shape_vs_ref"] = compare(dens[arm], dens[ref])
        rows.append({"case": name, "h_coarse": hc, "h_fine": hf, "coarse": crow,
                     "coarse_betti": betti(coarse > 0.5), "arms": res})
        OUT.write_text(json.dumps({"rows": rows}, indent=1))
    for r in rows:
        print(r["case"], "coarse", r["coarse_betti"])
        for arm, v in r["arms"].items():
            print(f"  {arm:17s} it {v['iterations']:3d}  gap_ref {v['gap_to_ref']:+.3%}  gap_cold {v['gap_to_cold']:+.3%}"
                  f"  IoU {v['shape_vs_ref']['iou']:.3f}  H95 {v['shape_vs_ref']['hausdorff']['p95']:.1f}"
                  f"  chg {v.get('changed_frac', float('nan')):.3%}  b {v['betti_end']['b0']}/{v['betti_end']['b1']}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
