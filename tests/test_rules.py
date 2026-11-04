"""Rule base v0: each rule fires on a spec built to trigger it and stays quiet on the FEAD problems."""
import pytest

from agentto import library
from agentto.rules import check
from agentto.spec import Spec


def edit(spec: Spec, **changes) -> Spec:
    d = spec.model_dump()
    d.update(changes)
    return Spec.model_validate(d)


@pytest.mark.parametrize("name", ["cantilever", "bridge", "torsion"])
def test_fead_clean(name):
    r = check(library.fead(name, 40))
    assert r.ok, r.to_dict()
    assert r.codes(severity="error") == []


def test_r001_resolution():
    s = edit(library.cantilever(40), resolution={"h": 0.3})
    assert check(s).codes() == ["R001"]


def test_r003_roller_only_is_unstable():
    s = library.bridge(40)
    d = s.model_dump()
    d["supports"] = [x for x in d["supports"] if x["id"] == "roller"]
    r = check(Spec.model_validate(d))
    assert "R003" in r.codes("error")


def test_r003_single_edge_pin_leaves_rotation():
    s = library.bridge(40)
    d = s.model_dump()
    d["supports"] = [x for x in d["supports"] if x["id"] == "pin"]
    r = check(Spec.model_validate(d))
    f = next(f for f in r.findings if f.code == "R003")
    assert "rotation about z" in f.message


def test_r005_load_on_clamped_face():
    s = library.cantilever(40)
    d = s.model_dump()
    d["loads"][0]["region"]["face"] = "x-"
    assert "R005" in check(Spec.model_validate(d)).codes("error")


def test_r004_zero_force():
    s = library.cantilever(40)
    d = s.model_dump()
    d["loads"][0]["force"] = [0, 0, 0]
    assert "R004" in check(Spec.model_validate(d)).codes("error")


def test_r007_r008_passive():
    s = library.cantilever(40)
    d = s.model_dump()
    d["passive"] = [
        {"id": "hole", "kind": "void", "region": {"kind": "box", "min": [1.8, 0.0, 0.0], "max": [2.0, 1.0, 1.0]}},
        {"id": "boss", "kind": "solid", "region": {"kind": "box", "min": [1.9, 0.4, 0.4], "max": [2.0, 0.6, 0.6]}},
    ]
    codes = check(Spec.model_validate(d)).codes("error")
    assert "R007" in codes and "R008" in codes


def test_r009_min_feature():
    s = edit(library.cantilever(20), min_feature=0.05)
    f = next(f for f in check(s).findings if f.code == "R009")
    assert f.fix["set"]["resolution"]["h"] == pytest.approx(0.05 / 2.4)


def test_r010_snapped_patch_reported():
    s = library.cantilever(10)
    d = s.model_dump()
    d["loads"][0]["region"]["size"] = [0.01, 0.01]
    d["loads"][0]["region"]["center"] = [0.55, 0.55]
    assert "R010" in check(Spec.model_validate(d)).codes("warning")


def test_r012_odd_line_support_lost_on_coarse_levels():
    s = library.bridge(40)
    d = s.model_dump()
    d["supports"][0]["region"]["at"] = [0.0, 0.025]       # y = 1 element: an odd node line
    r = check(Spec.model_validate(d))
    assert "R012" in r.codes("warning")


def test_r013_memory():
    r = check(library.cantilever(400), solver="mg64")
    assert "R013" in r.codes("error")


def test_session_levels_pads_when_no_padding_reaches_the_target():
    # b05 refine: voxfem's default leaves a 22,308-DOF coarsest level, which its GMG refuses (> 8 x 2500)
    from agentto import _voxfem  # noqa: F401
    from voxfem.core import Grid
    from agentto.rules import session_levels
    gp, L, over = session_levels(Grid(200, 200, 80))
    assert over and gp.shape == (208, 208, 80) and L == 5


def test_session_levels_pads_past_25_percent_when_it_pays():
    # b01 at 1 mm (X3): 15,288 coarsest DOFs cost 4.7 s/iteration; +30% elements and one more level: 2,688
    from agentto import _voxfem  # noqa: F401
    from voxfem.core import Grid
    from agentto.rules import session_levels
    gp, L, over = session_levels(Grid(200, 100, 100))
    assert over and gp.shape == (208, 112, 112) and L == 5
    gp, L, over = session_levels(Grid(640, 240, 480))         # b06 release (74M): unchanged, +14%
    assert gp.shape == (640, 256, 512) and L == 7
