"""Tool contract, provenance log and replay, MCP exposure, spec equivalence."""
import asyncio
import json
import time

import pytest

from agentto import library
from agentto.bench import equivalent
from agentto.tools import TOOLS, Session, read_log, replay, spec_hash

PATCH = {"kind": "patch", "face": "x+", "center": [50, 50], "size": [10, 10]}


def build_b01(s, actor="agent"):
    calls = [("new_spec", {"name": "b01", "size": [200, 100, 100]}),
             ("add_support", {"id": "wall", "region": {"kind": "face", "face": "x-"}}),
             ("add_load", {"id": "pad", "region": PATCH, "force": [0, -1000, 0]}),
             ("set_params", {"volume_fraction": 0.3, "h": 5.0})]
    return [s.call(n, a, actor) for n, a in calls]


def test_build_validate_and_log(tmp_path):
    s = Session(tmp_path)
    outs = build_b01(s)
    assert all(o["ok"] for o in outs)
    v = s.call("validate_spec")
    assert v["ok"] and v["result"]["ok"] and v["result"]["resolution"]["grid"] == [40, 20, 20]
    lr = s.call("list_regions")["result"]
    pad = [i for i in lr["items"] if i["id"] == "pad"][0]
    assert pad["n_nodes"] == 9 and pad["area"] == 100.0      # 10 mm patch at h = 5: 3 x 3 nodes
    recs = read_log(tmp_path / "session.jsonl")
    assert recs[0]["type"] == "session" and [r["seq"] for r in recs[1:]] == list(range(1, 7))
    assert recs[-1]["spec_hash"] == spec_hash(s.draft)


def test_failed_call_is_logged_and_leaves_spec(tmp_path):
    s = Session(tmp_path)
    build_b01(s)
    h0 = spec_hash(s.draft)
    bad = [s.call("add_support", {"id": "wall", "region": {"kind": "face", "face": "x+"}}),     # duplicate id
           s.call("add_load", {"id": "L2", "region": {"kind": "blob"}, "force": [0, 1, 0]}),     # bad region
           s.call("set_params", {"volume_fraction": 30}),                                       # percent, not fraction
           s.call("remove_item", {"id": "nope"}), s.call("no_such_tool", {})]
    assert not any(b["ok"] for b in bad) and spec_hash(s.draft) == h0
    errs = [r for r in read_log(tmp_path / "session.jsonl") if r.get("type") == "call" and not r["ok"]]
    assert len(errs) == 4                           # unknown tools are refused without a record


def test_undo_and_partial_drafts():
    s = Session(None)
    build_b01(s)
    h0 = spec_hash(s.draft)
    s.call("remove_item", {"id": "pad"})
    assert spec_hash(s.draft) != h0
    s.call("undo")
    assert spec_hash(s.draft) == h0
    h1 = spec_hash(s.draft)
    s.call("redo")                                   # the removal again
    assert spec_hash(s.draft) != h1 and not any(x["id"] == "pad" for x in s.draft["loads"] + s.draft["supports"]
                                                + s.draft.get("passive", []))
    s.call("undo")
    assert spec_hash(s.draft) == h1
    s.call("set_params", {"volume_fraction": 0.25})  # a new edit empties the redo stack
    assert not s.call("redo")["ok"]
    s.call("undo")
    d = dict(s.draft, loads=[])                      # schema-invalid (no load): kept as a draft, errors returned
    r = s.call("set_spec", {"spec": d})
    assert r["ok"] and r["result"]["valid"] is False and r["result"]["schema_errors"]
    assert s.call("validate_spec")["result"]["ok"] is False
    assert s.call("list_regions")["ok"]              # grounding still works on a partial draft


def test_spec_hash_normalizes_defaults():
    s = library.cantilever(10)
    d = s.model_dump(mode="json")
    d2 = json.loads(json.dumps(d))
    d2.pop("passive")
    assert spec_hash(d) == spec_hash(d2)


def test_render_is_deterministic_png():
    s = Session(None)
    build_b01(s)
    a, b = s.call("render")["result"], s.call("render", {"view": "iso"})["result"]
    assert a["png_sha256"] == b["png_sha256"] and a["png_b64"].startswith("iVBOR")
    assert s.call("render", {"view": "top"})["result"]["png_sha256"] != a["png_sha256"]


def test_replay_without_solves(tmp_path):
    s = Session(tmp_path)
    build_b01(s, actor="user")
    s.call("add_passive", {"id": "hole", "kind": "void", "region": {"kind": "box", "min": [90, 40, 0],
                                                                      "max": [110, 60, 100]}})
    s.call("update_item", {"id": "pad", "force": [0, -2000, 0]})
    s.call("add_support", {"id": "wall", "region": {"kind": "face", "face": "x+"}})        # fails, replays failing
    s.call("undo")
    s.call("render")
    r = replay(tmp_path / "session.jsonl", solve=False)
    assert r["ok"], r["mismatches"]
    assert r["final_spec_hash"] == spec_hash(s.draft)


class RecordingRunner:
    def __init__(self):
        self.jobs = []

    def submit(self, job):
        self.jobs.append(job)
        return job["id"]


def test_optimize_reopens_a_coarser_warm_start():
    rr = RecordingRunner()
    s = Session(None, runner=rr)
    build_b01(s)
    sh = spec_hash(s.draft)
    s.jobs["c0"] = {"id": "c0", "seq": 0, "h": 10.0, "spec_hash": sh, "status": "converged", "grid": [20, 10, 10]}
    for opt in (False, True):
        r = s.call("solve", {"h": 5.0, "wait": False, "optimize": opt})
        assert r["ok"] and r["result"]["warm_kind"] == "coarser"
    assert [j.get("warm_schedule") for j in rr.jobs] == [None, "reopen_long"]


def test_profiles():
    spec_tools = set(Session(None, profile="spec").tools())
    assert {"new_spec", "add_load", "list_regions", "validate_spec", "render"} <= spec_tools
    assert not {"solve", "field_stats", "load_template", "cancel"} & spec_tools
    assert set(Session(None).tools()) == set(TOOLS)


def test_equivalent_compares_what_the_solver_sees():
    base = library.cantilever(10).model_dump(mode="json")
    same = json.loads(json.dumps(base))
    s0 = same["supports"][0]
    assert s0["region"]["kind"] == "face" and s0["region"]["face"] == "x-"
    s0["region"] = {"kind": "box", "min": [-0.01, 0, 0], "max": [0.01, 1, 1]}  # the same nodes, written differently
    s0["id"] = "renamed"
    r = equivalent(same, base)
    assert r["same"], r
    other = json.loads(json.dumps(base))
    other["loads"][0]["force"] = [0, 0, -2.0]
    assert not equivalent(other, base)["fields"]["loads"]
    e = json.loads(json.dumps(base))
    e["material"]["E"] = base["material"]["E"] * 1.02
    r = equivalent(e, base)
    assert r["fields"]["material"] and not r["material_exact"]


def test_mcp_exposes_profile_and_logs_bad_arguments():
    from mcp.client.client import Client
    from agentto.mcp_server import build_server, local_call

    async def go():
        s = Session(None, profile="spec")
        async with Client(build_server(local_call(s), "spec")) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert names == set(s.tools())
            r = await c.call_tool("new_spec", {"name": "b01", "size": [200, 100, 100]})
            assert not r.is_error
            r = await c.call_tool("add_load", {"id": "L", "region": {"kind": "blob"}, "force": [0, -1, 0]})
            assert r.is_error and "region" in r.content[0].text
            await c.call_tool("add_support", {"id": "wall", "region": {"kind": "face", "face": "x-"}})
            await c.call_tool("add_load", {"id": "pad", "region": PATCH, "force": [0, -1000, 0]})
            r = await c.call_tool("render", {})
            assert [type(x).__name__ for x in r.content] == ["TextContent", "ImageContent"]
        return s
    s = asyncio.run(go())
    assert s.seq == 5                                # the rejected call reached the session and was counted


def test_session_solve_warm_and_replay_bit_exact(tmp_path):
    s = Session(tmp_path)
    try:
        s.call("set_spec", {"spec": library.cantilever(10).model_dump(mode="json")})
        a = s.call("solve", {"h": 0.1})["result"]
        b = s.call("solve", {"h": 0.05})["result"]
        fs = s.call("field_stats", {"region": "tip"})
        stl = s.call("export_stl", {"job": b["job"]})
        rep = s.call("export_report", {})
        s.call("update_item", {"id": "tip", "force": [0.0, -1.1, 0.0]})
        c = s.call("solve", {"h": 0.1})["result"]       # edited spec: warm from the last design on this grid
        d = s.call("solve", {"h": 0.1, "wait": False})["result"]
        t_end = time.perf_counter() + 120                # status alone must see it finish (background event pump)
        while time.perf_counter() < t_end and \
                s.call("status")["result"]["jobs"][-1]["status"] in ("queued", "running"):
            time.sleep(0.2)
        dw = s.call("wait_job", {"job": d["job"]})["result"]
    finally:
        s.runner.close()
    assert dw["status"] == "converged" and dw["job"] == d["job"]
    assert a["status"] == b["status"] == "converged"
    assert not a["warm_used"] and b["warm_used"] and b["iterations"] < a["iterations"]
    assert fs["ok"] and fs["result"]["solid_components"] >= 1
    assert all(fs["result"]["load_connected_to_support"].values())
    st = stl["result"]
    assert stl["ok"] and st["triangles"] > 1000 and st["mesh"]["closed"] and not st["released"]
    assert "_draft" in st["path"]                    # h = 0.05 is not the release rung of this spec
    md = open(rep["result"]["markdown"], encoding="utf-8").read()
    assert rep["ok"] and "## Certificate checks" in md and st["sha256"][:16] in md
    assert c["status"] == "converged" and c["warm_used"] and c["warm_kind"] == "same_grid"
    assert c["iterations"] < a["iterations"]
    r = replay(tmp_path / "session.jsonl", root=tmp_path / "replay")
    assert r["ok"] and r["solves_replayed"] == 4, r["mismatches"]
