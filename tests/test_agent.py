"""Assistant proposals: spec diff -> action cards -> partial acceptance (no LLM involved)."""
import json

from agentto import library
from agentto.app.agent import apply_cards, build_prompt, spec_diff
from agentto.tools import spec_hash


def test_cards_cover_every_change_and_accept_all_reproduces_the_proposal():
    base = library.cantilever(10).model_dump(mode="json")
    prop = json.loads(json.dumps(base))
    prop["loads"][0]["force"] = [0, -2.0, 0]
    prop["supports"].append({"id": "pin", "fix": ["y"], "region": {"kind": "edge", "axis": "z", "at": [2.0, 0.0]},
                             "note": ""})
    prop["passive"] = [{"id": "hole", "kind": "void", "region": {"kind": "box", "min": [0.8, 0.3, 0], "max": [1.2, 0.7, 1]},
                        "note": ""}]
    prop["volume_fraction"] = 0.25
    cards = spec_diff(base, prop)
    assert sorted(c["op"] for c in cards) == ["add", "add", "change", "set"]
    assert spec_hash(apply_cards(base, cards, prop)) == spec_hash(prop)
    only = [c for c in cards if c["op"] == "change"]
    part = apply_cards(base, only, prop)
    assert part["loads"][0]["force"] == [0, -2.0, 0] and part["volume_fraction"] == base["volume_fraction"]
    assert len(part["supports"]) == 1


def test_remove_card_and_empty_base():
    base = library.bridge(10).model_dump(mode="json")
    prop = json.loads(json.dumps(base))
    prop["supports"] = prop["supports"][:1]
    cards = spec_diff(base, prop)
    assert [c["op"] for c in cards] == ["remove"] and cards[0]["item_id"] == "roller"
    assert spec_hash(apply_cards(base, cards, prop)) == spec_hash(prop)
    assert spec_diff(base, base) == []
    assert apply_cards(None, spec_diff(None, prop), prop) == prop


def test_prompt_carries_mode_selection_and_history():
    p = build_prompt("make it lighter", "act", library.cantilever(10).model_dump(mode="json"),
                     [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hello"}],
                     {"selection": {"kind": "load", "id": "tip"}, "findings": ["R004 warning: x"]})
    assert "Mode ACT" in p and '"id": "tip"' in p and "assistant: hello" in p and p.endswith("user: make it lighter")


def test_act_run_keeps_a_copy_of_the_spec_for_undo(tmp_path):
    # Act-mode tool calls edit the live draft in place; the run's undo base must not follow them
    from agentto.app.agent import AgentRun
    live = library.cantilever(10).model_dump(mode="json")
    run = AgentRun("x", "act", live, [], {}, tmp_path, "http://127.0.0.1:1", lambda m: None)
    live["volume_fraction"] = 0.25
    assert run.base["volume_fraction"] != 0.25


def test_late_clarification_answer_starts_a_follow_up_run():
    import threading
    from types import SimpleNamespace
    from agentto.app import server
    done = threading.Thread(target=lambda: None)
    done.start()
    done.join()
    opts = [{"label": "10 mm", "item_kind": "load", "item": {"id": "load_2"}}, {"label": "20 mm", "params": {"volume_fraction": 0.2}}]
    ended = SimpleNamespace(thread=done, mode="act")
    server.CLARIFY["q1"] = {"event": threading.Event(), "answer": None, "question": "How large?", "options": opts, "run": ended}
    msg, mode = server._clarify_answer({"qid": "q1", "choice": 1, "label": "20 mm"})
    assert mode == "act" and "How large?" in msg and "20 mm" in msg and '"volume_fraction": 0.2' in msg
    assert server.CLARIFY["q1"]["event"].is_set() and server.CLARIFY["q1"]["answer"]["choice"] == 1
    assert server._clarify_answer({"qid": "q1", "choice": 0}) is None          # answered once only
    alive = SimpleNamespace(thread=SimpleNamespace(is_alive=lambda: True), mode="act")
    server.CLARIFY["q2"] = {"event": threading.Event(), "answer": None, "question": "Which face?", "options": opts, "run": alive}
    assert server._clarify_answer({"qid": "q2", "choice": 0}) is None          # the asking run gets it itself
    server.CLARIFY.pop("q1"), server.CLARIFY.pop("q2")
    import shutil
    if len(server.live.log_path.read_text(encoding="utf-8").splitlines()) <= 1:   # importing the app opened an empty session
        shutil.rmtree(server.live.log_path.parent, ignore_errors=True)
