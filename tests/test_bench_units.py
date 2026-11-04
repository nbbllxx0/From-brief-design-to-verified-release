"""Scoring across unit systems and admissible readings (bench.normalized / match_task / Oracle)."""
import json
from pathlib import Path

import pytest

from agentto.bench import Oracle, equivalent, match_task, normalized
from agentto.rules import check
from agentto.spec import Spec

BRIEFS = Path(__file__).resolve().parents[1] / "briefs" / "intended"


def _task(stem):
    return json.loads((BRIEFS / f"{stem}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("stem", ["b19_engine_bracket", "b25_cantilever_metres", "b34_antenna_bracket",
                                  "b37_battery_crossmember"])
def test_a_spec_converted_to_mm_and_n_is_the_same_problem(stem):
    q = _task(stem)["spec"]
    mm = normalized(q).model_dump(mode="json")
    assert mm["units"] == {"length": "mm", "force": "N"} != q["units"]
    r = equivalent(mm, q)
    assert r["same"], r
    assert not r["units_same"]
    wrong = json.loads(json.dumps(mm))
    wrong["loads"][0]["force"] = [x * 2 for x in wrong["loads"][0]["force"]]
    assert not equivalent(wrong, q)["same"]


def test_resolution_given_as_cells_or_h_is_the_same_grid():
    q = _task("b43_press_riser")["spec"]
    p = dict(q, resolution={"h": 2.0})
    assert equivalent(p, q)["fields"]["resolution"]
    assert not equivalent(dict(q, resolution={"h": 4.0}), q)["fields"]["resolution"]


def test_admissible_readings_are_credited_and_named():
    for stem in ("b06_motor_mount", "b25_cantilever_metres", "b44_gearbox_bracket"):
        t = _task(stem)
        alts = [a for a in t["admissible"] if a.get("counts")]
        assert alts and all("spec" in a for a in alts), stem
        assert not equivalent(alts[0]["spec"], t["spec"])["same"], stem
        r = match_task(alts[0]["spec"], t)
        assert r["same"] and r["matched"].startswith("admissible:"), stem
        assert match_task(t["spec"], t)["matched"] == "intended"


def test_withdrawn_reading_is_not_credited():
    t = _task("b02_mbb_half")                                     # adjudication 2026-10-31: x-fixed end withdrawn
    withdrawn = [a for a in t["admissible"] if "fixes x" in a["reading"]]
    assert withdrawn and not withdrawn[0]["counts"]
    assert not match_task(withdrawn[0]["spec"], t)["same"]


def test_refuse_tasks_fail_for_the_stated_reason():
    for stem, code in (("b08_slider_bracket", "R003"), ("b49_hinged_arm", "R003"), ("b50_valve_bracket", "R008")):
        r = check(Spec.model_validate(_task(stem)["spec"]))
        assert code in r.codes("error"), (stem, r.to_dict())


def test_oracle_matches_an_option_written_in_other_units():
    t = _task("b25_cantilever_metres")                           # intended in m and kN
    o = Oracle(t["spec"], script=t["oracle"])
    mm = normalized(t["spec"]).model_dump(mode="json")
    item = mm["loads"][0]                                         # the same load, in mm and N
    other = dict(item, force=[0, -50000, 0])
    ans = o("which load?", [{"label": "half", "item_kind": "load", "item": other},
                            {"label": "full", "item_kind": "load", "item": item}],
            None, [dict(mm, loads=[other]), mm])
    assert ans["choice"] == 1
    vf = o("how much material?", [{"label": "a", "params": {"volume_fraction": 0.15}},
                                  {"label": "b", "params": {"volume_fraction": 0.3}}],
           None, [dict(mm, volume_fraction=0.15), dict(mm, volume_fraction=0.3)])
    assert vf["choice"] == 0


def test_oracle_accepts_numbers_written_as_strings():
    t = _task("b03_bridge_deck")
    o = Oracle(t["spec"], script=t["oracle"])
    item = {"id": "r", "fix": ["y", "z"], "region": {"kind": "edge", "axis": "z", "at": ["400", "0"],
                                                     "span": ["0", "100"]}}
    ans = o("right bearing?", [{"label": "y only", "item_kind": "support", "item": dict(item, fix=["y"])},
                               {"label": "y and z", "item_kind": "support", "item": item}])
    assert ans["choice"] == 1 and ans["by"] == "oracle"
