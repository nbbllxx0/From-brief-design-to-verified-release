"""ask_clarification and the scripted oracle (X2 condition G+C): matching readings on the solver's grid, replay."""
import json

from agentto import library
from agentto.bench import Oracle, item_match, params_match
from agentto.tools import Session, read_log, replay


def test_item_match_ignores_ids_and_region_wording():
    spec = library.cantilever(10).model_dump(mode="json")
    face_as_box = {"id": "wall", "fix": ["x", "y", "z"], "region": {"kind": "box", "min": [-0.01, 0, 0], "max": [0.01, 1, 1]}}
    assert item_match(face_as_box, "support", spec) == 0
    roller = dict(face_as_box, fix=["x"])
    assert item_match(roller, "support", spec) is None
    tip = dict(spec["loads"][0], id="P")
    assert item_match(tip, "load", spec) == 0
    assert item_match(dict(tip, force=[0, -2.0, 0]), "load", spec) is None
    assert params_match({"volume_fraction": 0.3}, spec) and not params_match({"volume_fraction": 0.25}, spec)
    assert params_match({"material": {"name": "x", "E": 1.02, "nu": 0.3}}, spec)       # within the 3% E tolerance
    assert params_match({"E": 1.0}, spec) and not params_match({"E": 2.0}, spec)       # set_params-style names too
    assert params_match({"clear_min_feature": True}, spec) and not params_match({"min_feature": 2.0}, spec)


def test_oracle_answers_and_replay_reuses_the_answer(tmp_path):
    spec = library.cantilever(10).model_dump(mode="json")
    s = Session(tmp_path, profile="clarify")
    s.answerer = Oracle(spec)
    s.call("set_spec", {"spec": spec})
    top = {"id": "tip", "force": [0, -1, 0], "region": {"kind": "patch", "face": "x+", "center": [1.0, 0.5], "size": [0.1, 0.1]}}
    opts = [{"label": "at the top edge", "item_kind": "load", "item": top},
            {"label": "centred", "item_kind": "load", "item": spec["loads"][0]}]
    r = s.call("ask_clarification", {"question": "Where on the end face?", "options": opts})
    assert r["ok"] and r["result"]["choice"] == 1 and r["result"]["chosen"]["label"] == "centred"
    r = s.call("ask_clarification", {"question": "Fraction?", "options": [{"label": "a", "params": {"volume_fraction": 0.2}},
                                                                          {"label": "b", "params": {"volume_fraction": 0.4}}]})
    assert r["result"]["choice"] is None and "None of these" in r["result"]["comment"]
    bad = s.call("ask_clarification", {"question": "x", "options": [{"label": "a", "item": top}, {"label": "b", "params": {}}]})
    assert not bad["ok"] and "item_kind" in bad["error"]
    recs = read_log(tmp_path / "session.jsonl")
    assert [x["answer"]["choice"] for x in recs if x["type"] == "answer"] == [1, None]
    rep = replay(tmp_path / "session.jsonl", solve=False)      # no oracle in the replay: answers come from the log
    assert rep["ok"], rep["mismatches"]


def test_no_answerer_tells_the_agent_to_assume():
    s = Session(None, profile="clarify")
    s.call("set_spec", {"spec": library.cantilever(10).model_dump(mode="json")})
    assert "ask_clarification" not in Session(None, profile="spec").tools()     # condition G has no clarification
    r = s.call("ask_clarification", {"question": "q", "options": [{"label": "a", "params": {"volume_fraction": 0.2}},
                                                                  {"label": "b", "params": {"volume_fraction": 0.3}}]})
    assert r["ok"] and r["result"]["choice"] is None and r["result"]["by"] == "none"
