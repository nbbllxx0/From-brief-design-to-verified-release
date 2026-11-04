"""Task-suite check on the GPU: every solve/clarify intended spec is optimized cold at its preview rung (the rung
x2_score uses for the consequence-weighted error), in mm and N. A task whose preview does not converge, or does
not certify, cannot give a reliable CWE reference and is flagged.

    python -B scripts/validate_suite.py [b11 b12 ...]     (one GPU job at a time; nothing else on the GPU)
Writes results/X2/suite_check.json (merged with earlier runs).
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from agentto.bench import normalized  # noqa: E402
from agentto.escalation import run_arm  # noqa: E402
from agentto.problem import build  # noqa: E402
from x2_score import CFG, preview_h  # noqa: E402


def main(sel):
    out_p = ROOT / "results" / "X2" / "suite_check.json"
    out = json.loads(out_p.read_text()) if out_p.exists() else {}
    for p in sorted((ROOT / "briefs" / "intended").glob("*.json")):
        if sel and not any(p.stem.startswith(s) for s in sel):
            continue
        t = json.loads(p.read_text(encoding="utf-8"))
        if t["expected"] == "refuse":
            continue
        q = normalized(t["spec"])
        h = preview_h(q)
        sp = build(q, h=h)
        t0 = time.perf_counter()
        row, rho, _ = run_arm(sp, CFG, "cold")
        rec = {"h": h, "grid": list(sp.grid.shape), "n_elem": sp.grid.n_elem, "wall_s": time.perf_counter() - t0,
               **{k: row.get(k) for k in ("status", "iterations", "compute_s", "compliance", "volume",
                                         "certified", "error")}}
        rec["solid_fraction"] = float((rho > 0.5).mean())
        tail = row.get("c_history", [])[-6:]          # a capped run whose compliance has settled is a usable reference
        rec["c_tail_spread"] = (max(tail) - min(tail)) / abs(tail[-1]) if tail and tail[-1] else None
        out[p.stem] = rec
        out_p.write_text(json.dumps(out, indent=1, default=str))
        print(f"{p.stem:28s} h {h:<6g} {sp.grid.n_elem / 1e6:5.2f}M {rec['status']:>12s} it {rec['iterations']} "
              f"C {rec['compliance']} cert {rec['certified']} {rec['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
