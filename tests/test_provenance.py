"""Value provenance (profiles spec_prov, clarify_prov; X2 conditions GP and GPC). CPU only."""
from agentto import library
from agentto.tools import TOOLS, Session


def session(profile):
    s = Session(None, profile=profile)
    s.draft = library.cantilever(10).model_dump(mode="json", exclude_none=True)
    return s


def test_prov_profiles_extend_spec_and_clarify():
    for name, t in TOOLS.items():
        if "spec" in t["profiles"]:
            assert "spec_prov" in t["profiles"], name
        if "clarify" in t["profiles"]:
            assert "clarify_prov" in t["profiles"], name
    assert "set_source" not in Session(None, profile="spec").tools()
    assert {"set_source", "list_sources"} <= set(Session(None, profile="spec_prov").tools())


def test_set_and_list_sources():
    s = session("spec_prov")
    keys = s._source_keys()
    assert keys[:2] == ["units", "domain"] and any(k.startswith("load:") for k in keys)
    assert not s.call("set_source", {"key": "load:nope", "source": "stated"})["ok"]
    load = next(k for k in keys if k.startswith("load:"))
    r = s.call("set_source", {"key": load, "source": "assumed", "note": "direction not stated"})
    assert r["ok"] and load not in r["result"]["untagged"]
    t = s.call("list_sources")["result"]
    assert t["assumed"] == [load] and len(t["untagged"]) == len(keys) - 1


def test_validate_adds_advisory_r014_only_with_provenance():
    s = session("spec_prov")
    load = next(k for k in s._source_keys() if k.startswith("load:"))
    s.call("set_source", {"key": load, "source": "assumed"})
    out = s.call("validate_spec")["result"]
    r014 = [f for f in out["findings"] if f["code"] == "R014"]
    assert len(r014) == 2 and all(f["severity"] == "info" for f in r014) and out["ok"]
    g = session("spec").call("validate_spec")["result"]
    assert not any(f["code"] == "R014" for f in g["findings"])


def test_clarify_prov_accepts_questions_only_about_assumed_items():
    s = session("clarify_prov")
    opts = [{"label": "a", "params": {"volume_fraction": 0.3}}, {"label": "b", "params": {"volume_fraction": 0.4}}]
    q = {"question": "how much material?", "options": opts}
    assert "about" in s.call("ask_clarification", q)["error"]
    s.call("set_source", {"key": "volume_fraction", "source": "stated"})
    assert "not assumed" in s.call("ask_clarification", q | {"about": ["volume_fraction"]})["error"]
    s.call("set_source", {"key": "volume_fraction", "source": "assumed"})
    r = s.call("ask_clarification", q | {"about": ["volume_fraction"]})
    assert r["ok"], r
    c = session("clarify").call("ask_clarification", q)          # GC: no gate
    assert c["ok"], c
