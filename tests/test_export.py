"""Released-design export: closed isosurface at the element boundaries, binary STL, report; the PDH spill sampler."""
import struct

import numpy as np

from agentto.export import isosurface, mesh_stats, preview_mesh, report_markdown, write_stl


def _box():
    r = np.zeros((20, 10, 10), np.float32)
    r[2:12, 2:8, 0:10] = 1.0                               # touches both z faces: the void padding closes it
    return r


def test_isosurface_is_closed_and_sits_on_element_boundaries(tmp_path):
    v, f = isosurface(_box(), 0.5)
    st = mesh_stats(v, f)
    assert st["closed"] and st["nonmanifold_edges"] == 0
    assert np.allclose(v.min(0), [1.0, 1.0, 0.0]) and np.allclose(v.max(0), [6.0, 4.0, 5.0])
    assert 0.95 * 75 < st["volume"] <= 75                  # marching cubes chamfers the box edges
    out = write_stl(tmp_path / "b.stl", v, f)
    raw = (tmp_path / "b.stl").read_bytes()
    assert out["bytes"] == 84 + 50 * len(f) and struct.unpack("<I", raw[80:84])[0] == len(f)
    assert write_stl(tmp_path / "c.stl", v, f)["sha256"] == out["sha256"]          # deterministic bytes


def test_preview_mesh_and_report():
    data, info = preview_mesh(_box(), 0.5, max_triangles=10_000)
    nv, nt = struct.unpack("<II", data[:8])
    assert (nv, nt) == (info["vertices"], info["triangles"]) and len(data) == 12 + 12 * nv + 12 * nt
    rep = {"spec": {"name": "box", "units": {"length": "mm", "force": "N"}, "domain": {"size": [10, 5, 5]},
                    "material": {"name": "steel", "E": 210000.0, "nu": 0.3}, "volume_fraction": 0.3,
                    "supports": [{"id": "s"}], "loads": [{"id": "p", "force": [0, -1000.0000000001, 0]}], "passive": []},
           "certified": True, "status": "converged", "job": "j", "issued_at": "t", "spec_hash": "h", "grid": [2, 1, 1],
           "n_elem": 2, "h": 5, "solver": "mg64", "warm_used": False, "iterations": 3, "compute_s": 1.0,
           "compliance_phys": 1.5, "volume": 0.3, "checks": {"converged": True}, "stl": None,
           "session_log_sha256": "x"}
    md = report_markdown(rep)
    assert "**CERTIFIED**" in md and "p [0, -1000, 0] N" in md


def test_fast_spill_watch_reads_counters():
    from agentto.solver import FastSpillWatch
    w = FastSpillWatch()
    w.check("x")
    assert len(w.samples) == 2 and not w.spilled and w.report()["sampler"] == "pdh"
