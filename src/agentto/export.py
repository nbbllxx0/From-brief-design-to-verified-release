"""Released-design export: isosurface, binary STL, preview mesh for the GUI, and the design report.

The surface is the rho = 0.5 isosurface of the element densities, taken as samples at element centres and padded
with one void layer so the surface is closed. VTK's Flying Edges (multi-threaded marching cubes) extracts it; at
10^8 elements that takes seconds, where scikit-image needs minutes. Positions are in the spec's length unit.
"""
from __future__ import annotations

import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np


def isosurface(rho: np.ndarray, h: float, level: float = 0.5, pool: int = 1):
    """(vertices float32 (n, 3), triangles int64 (m, 3)) of the closed rho = level surface; `pool` > 1 averages
    pool^3 element blocks first (a coarser preview surface)."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    from .shape import pool as _pool
    a = _pool(np.asarray(rho, dtype=np.float32), pool) if pool > 1 else np.asarray(rho, dtype=np.float32)
    hp = h * pool
    a = np.pad(a, 1)                                       # void layer: closed surface at the domain faces
    img = vtk.vtkImageData()
    img.SetDimensions(*a.shape)
    img.SetSpacing(hp, hp, hp)
    img.SetOrigin(-0.5 * hp, -0.5 * hp, -0.5 * hp)         # sample i sits at the centre of element i - 1
    # VTK point order is x fastest; our arrays are C-ordered (x, y, z) with z fastest
    img.GetPointData().SetScalars(numpy_to_vtk(np.ascontiguousarray(a.transpose(2, 1, 0)).ravel(), deep=1))
    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(img)
    fe.SetValue(0, level)
    fe.ComputeNormalsOff()
    fe.ComputeGradientsOff()
    fe.ComputeScalarsOff()
    fe.Update()
    out = fe.GetOutput()
    if out.GetNumberOfPoints() == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)
    v = vtk_to_numpy(out.GetPoints().GetData()).astype(np.float32)
    polys = out.GetPolys()
    conn = vtk_to_numpy(polys.GetConnectivityArray()).astype(np.int64)
    return v, conn.reshape(-1, 3)


def write_stl(path: Path, v: np.ndarray, f: np.ndarray, header: str = "agentto release") -> dict:
    """Binary STL (facet normals from the vertex order). Returns {triangles, bytes, sha256}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = v[f]                                               # (m, 3, 3)
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("p", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["p"] = n, p
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii", "replace")[:80].ljust(80, b" "))
        fh.write(struct.pack("<I", len(f)))
        fh.write(rec.tobytes())
    return {"triangles": int(len(f)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def mesh_stats(v: np.ndarray, f: np.ndarray) -> dict:
    """Closedness check: every edge shared by exactly two triangles; enclosed volume by the divergence theorem."""
    if len(f) == 0:
        return {"closed": False, "volume": 0.0, "boundary_edges": 0}
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    key = e[:, 0] * (int(f.max()) + 1) + e[:, 1]
    _, cnt = np.unique(key, return_counts=True)
    p = v[f].astype(np.float64)
    vol = float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0)
    return {"closed": bool((cnt == 2).all()), "boundary_edges": int((cnt == 1).sum()),
            "nonmanifold_edges": int((cnt > 2).sum()), "volume": abs(vol)}


def export_stl(rho: np.ndarray, h: float, path: Path, pool: int = 1, check: bool = True) -> dict:
    t0 = time.perf_counter()
    v, f = isosurface(rho, h, pool=pool)
    t1 = time.perf_counter()
    out = write_stl(path, v, f)
    t2 = time.perf_counter()
    out.update(path=str(path), pool=pool, vertices=int(len(v)), t_surface_s=round(t1 - t0, 3),
               t_write_s=round(t2 - t1, 3))
    if check and len(f) <= 30_000_000:
        out["mesh"] = mesh_stats(v, f)
    return out


def preview_mesh(rho: np.ndarray, h: float, max_triangles: int = 1_500_000) -> tuple[bytes, dict]:
    """Binary preview surface for the GUI: header (uint32 n_vertices, uint32 n_triangles, float32 h_eff) then
    float32 positions and uint32 indices. Pools the field until the surface fits the triangle budget."""
    from .shape import pool_factor
    f0 = pool_factor(rho.shape, max_elem=4_000_000)
    while True:
        v, f = isosurface(rho, h, pool=f0)
        if len(f) <= max_triangles or any(n % (2 * f0) for n in rho.shape):
            break
        f0 *= 2
    head = struct.pack("<IIf", len(v), len(f), h * f0)
    return head + v.astype("<f4").tobytes() + f.astype("<u4").tobytes(), {"pool": f0, "vertices": len(v),
                                                                        "triangles": len(f)}


def report(job: dict, spec: dict, stl: dict | None, log_hash: str | None) -> dict:
    """The design report: what was asked (spec + hash), what was solved (rung, grid, solver, iterations, time), the
    certificate checks, and the exported geometry. Markdown is rendered from the same dict."""
    from .tools import spec_hash
    r = job.get("result") or {}
    return {"report": "agentto design report v0", "issued_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "spec_hash": spec_hash(spec), "spec": spec, "job": job["id"], "h": job["h"],
            "grid": r.get("grid"), "n_elem": r.get("n_elem"), "solver": r.get("solver"),
            "warm_used": r.get("warm_used"), "iterations": r.get("iterations"), "compute_s": r.get("compute_s"),
            "compliance_phys": r.get("compliance_phys"), "volume": r.get("volume"), "status": r.get("status"),
            "certified": r.get("certified"), "checks": r.get("checks"), "release_rung": r.get("release_rung"),
            "regularization": r.get("regularization"), "stl": stl, "session_log_sha256": log_hash}


def report_markdown(rep: dict) -> str:
    s = rep["spec"]
    lu, fu = s["units"]["length"], s["units"]["force"]
    ok = "CERTIFIED" if rep["certified"] else "NOT CERTIFIED"
    lines = [f"# {s['name']}: design report", "", f"**{ok}** — {rep['status']}, job `{rep['job']}`, "
             f"issued {rep['issued_at']}", "", "## Problem", "",
             f"- Domain {s['domain']['size']} {lu}; material {s['material']['name']} (E = {s['material']['E']:g}, "
             f"ν = {s['material']['nu']:g}); volume fraction {s['volume_fraction']:g}",
             f"- Supports: {', '.join(x['id'] for x in s['supports'])}; loads: "
             + ", ".join(f"{x['id']} [{', '.join(f'{v:g}' for v in x['force'])}] {fu}" for x in s["loads"]),
             f"- Passive regions: {', '.join(x['id'] + ' (' + x['kind'] + ')' for x in s.get('passive', [])) or 'none'}",
             f"- Spec hash `{rep['spec_hash']}`", "", "## Solve", "",
             f"- Grid {rep['grid']} ({rep['n_elem']:,} elements), h = {rep['h']:g} {lu}, solver {rep['solver']}, "
             f"warm start {'yes' if rep['warm_used'] else 'no'}",
             f"- {rep['iterations']} iterations, {rep['compute_s']:.1f} s compute" if rep.get("compute_s") else
             f"- {rep['iterations']} iterations",
             f"- Compliance {rep['compliance_phys']:.6g} {fu}·{lu}; volume {rep['volume']:.4f}", "",
             "## Certificate checks", ""]
    lines += [f"- {k.replace('_', ' ')}: {'pass' if v else 'FAIL'}" for k, v in (rep.get("checks") or {}).items()]
    g = rep.get("regularization")
    if g:
        lines += ["", "## Regularization of the released problem", "",
                  f"- Schedule {g['schedule']}" + (f", warm start from h = {g['warm_from_h']:g} {lu}" if g.get("warm_from_h")
                  else "") + f", {g['iterations_on_grid']} iterations on this grid",
                  f"- Filter radius {g['filter_rmin_elements']:g} elements: {g['filter_length']:g} {lu} on the release grid"
                  + (f", {g['filter_length_at_stated_h']:g} {lu} at the specification's stated h = {g['stated_h']:g} {lu}"
                     f" (release grid {g['refinement_factor']:g} times finer)" if g.get("stated_h") else "")]
    if rep.get("stl"):
        st = rep["stl"]
        lines += ["", "## Geometry", "", f"- STL `{Path(st['path']).name}`: {st['triangles']:,} triangles, "
                  f"{st['bytes'] / 1e6:.1f} MB, sha256 `{st['sha256'][:16]}…`"]
        if st.get("mesh"):
            lines.append(f"- Closed surface: {'yes' if st['mesh']['closed'] else 'no'}; enclosed volume "
                         f"{st['mesh']['volume']:.6g} {lu}³")
    lines += ["", f"Session log sha256 `{(rep.get('session_log_sha256') or '')[:16]}…`", ""]
    return "\n".join(lines)


def dump(obj) -> str:
    return json.dumps(obj, indent=1, default=str)
