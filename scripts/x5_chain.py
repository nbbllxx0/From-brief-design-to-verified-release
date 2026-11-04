"""X5 addendum: the ladder's own release path against a cold run at the finest rung.

x5_resolution.py showed that on the span bridge a cold run at the finest rung (33M) ends 26% worse, judged as a part,
than the 4 mm rung's design judged by the same fine model, and that every coarse rung over-predicts its own
stiffness. The release path the system offers ("Optimize release", X1's reopen_long arm) chains the rungs instead:
cold at the coarsest rung, then reopen_long at each finer rung, warm from the rung above. This script runs that
chain on the X5 problems and scores every rung's design exactly as x5_resolution.py does (volume-matched solid/void
part, evaluated at the finest grid).

    python -B scripts/x5_chain.py [problem ...]          (one GPU job at a time; resumes per problem)
Adds "chain_reopen_long" to results/X5/resolution.json.
"""
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
from agentto.escalation import run_arm  # noqa: E402
from agentto.problem import build  # noqa: E402
from agentto.shape import betti, compare, pool, pool_factor  # noqa: E402
from x5_resolution import FIELDS, OUT, cfg, evaluate, part, problems, thickness, upsample  # noqa: E402


def main(only=None):
    res = json.loads(OUT.read_text())
    for name, (spec, hs) in problems().items():
        if name not in res or "chain_reopen_long" in res[name] or (only and name not in only):
            continue
        fine = build(spec, h=hs[-1])
        kw_fine = cfg(fine.grid.n_elem, True)
        rows, prev = [], None
        for h in hs:
            sp = build(spec, h=h)
            arm = "cold" if prev is None else "reopen_long"
            row, rho, _ = run_arm(sp, cfg(sp.grid.n_elem, h == hs[-1]), arm, coarse=prev, keep_start=False)
            if row["status"] not in ("converged", "not_converged"):
                raise RuntimeError(f"{name} h {h}: {arm} {row['status']}: {row['error']}")
            row.pop("c_history", None)
            row["h"] = h
            row["c_own_phys"] = row["compliance"] * sp.compliance_scale
            m = part(rho, sp, spec.volume_fraction)
            f = int(round(h / hs[-1]))
            t0 = time.perf_counter()
            row["c_part"] = evaluate(fine, upsample(m, f) if f > 1 else m, kw_fine)
            row["eval_s"] = time.perf_counter() - t0
            row["thickness"] = thickness(m > 0.5, h)
            pf = pool_factor(m.shape)
            row["betti"] = betti(pool(m, pf) > 0.5 if pf > 1 else m > 0.5)
            if h == hs[-1]:
                cold = np.load(FIELDS / f"{name}_cold_h{h:g}.npy").astype(np.float32)
                mc = part(cold, sp, spec.volume_fraction)
                row["shape_vs_cold_fine"] = compare(pool(m, pf) > 0.5, pool(mc, pf) > 0.5)
                np.save(FIELDS / f"{name}_reopen_long_h{h:g}.npy", rho.astype(np.float16))
                del cold, mc
            rows.append(row)
            print(name, f"h {h:g}", arm, row["status"], row["iterations"], f"{row['compute_s']:.0f}s",
                  f"C_part {row['c_part']:.6g}", row["betti"], flush=True)
            prev = rho
            del m
            gc.collect()
        ref = res[name]["rows"][-1]["c_part"]                 # the cold finest rung
        for r in rows:
            r["c_part_gap_to_cold_fine"] = r["c_part"] / ref - 1
        res[name]["chain_reopen_long"] = rows
        OUT.write_text(json.dumps(res, indent=1, default=str))
    for name, r in res.items():
        if "chain_reopen_long" not in r:
            continue
        print(name, "chain: cold at the coarsest rung, then reopen_long")
        for x in r["chain_reopen_long"]:
            print(f"  h {x['h']:<5g} {x['arm']:12s} it {x['iterations']:4d} {x['compute_s']:6.0f}s  "
                  f"C_part {x['c_part']:.6g} ({x['c_part_gap_to_cold_fine']:+.2%} vs cold fine)  b1 {x['betti']['b1']}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
