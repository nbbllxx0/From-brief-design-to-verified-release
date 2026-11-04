"""Task suite: every intended spec validates, and the rule base agrees with the expected behaviour."""
import json
from pathlib import Path

import pytest

from agentto.rules import check
from agentto.spec import Spec

INTENDED = sorted((Path(__file__).resolve().parents[1] / "briefs" / "intended").glob("*.json"))


@pytest.mark.parametrize("path", INTENDED, ids=[p.stem for p in INTENDED])
def test_intended(path):
    t = json.loads(path.read_text(encoding="utf-8"))
    spec = Spec.model_validate(t["spec"])
    r = check(spec, solver="mg_ir")
    assert (path.parent.parent / f"{path.stem}.md").exists()
    if t["expected"] == "refuse":
        assert not r.ok, "an infeasible brief must fail the rule base"
    else:
        assert r.ok, r.to_dict()
        # R012 (supports lost on coarse multigrid levels) is expected on general geometry until RobustGMG
        # lands (week 2); it costs solver iterations, not correctness. Anything else is a defect in the task.
        # A task may accept other warnings it explains in its notes (b22: a face load sharing an edge with the wall).
        assert set(r.codes("warning")) <= {"R012", *t.get("accept_warnings", [])}, r.to_dict()
    for p in spec.passive:
        assert r.resolution["n_design"] < r.resolution["n_elem"]
