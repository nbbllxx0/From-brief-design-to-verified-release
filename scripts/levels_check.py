"""Per-iteration cost of b01 at its refine (1 mm, 2M) and release (0.5 mm, 16M) rungs with the session's
multigrid levels (rules.session_levels): 10 cold iterations each, s/iteration printed. Checks the cost model."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentto.escalation import run_arm  # noqa: E402
from agentto.problem import build  # noqa: E402
from agentto.rules import _coarsest_dofs, iter_cost, session_levels  # noqa: E402
from agentto.spec import Spec  # noqa: E402

spec = Spec.model_validate(json.loads((ROOT / "briefs" / "intended" / "b01_cantilever_block.json").read_text())["spec"])
for h in (1.0, 0.5):
    sp = build(spec, h=h)
    gp, L, over = session_levels(sp.grid)
    row, _, _ = run_arm(sp, dict(solver="mg64", op32="stencil", spill_check_every=0, log_every=0, max_outer=10), "cold")
    print(f"h {h}: grid {sp.grid.shape} -> {gp.shape} L {L} coarsest {_coarsest_dofs(gp, L)} "
          f"predicted {iter_cost(gp, L):.2f} s/it, measured {row['s_per_iter']:.2f} s/it", flush=True)
