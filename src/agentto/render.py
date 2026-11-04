"""Deterministic pictures of a spec: the BC/load figure the user (or a VLM) confirms before solving.

Everything is drawn from the spec's own geometry in physical units, not from the mesh, so the picture shows what
the spec says. Mesh effects (empty or snapped regions) are reported by list_regions and the rule base instead.
Colours carry meaning only: supports blue, loads red arrows, keep-in green, keep-out grey-red hatched.
"""
from __future__ import annotations

import hashlib
import io

import numpy as np

from .problem import AX

VIEWS = {"iso": (22, -30), "front": (0, 0), "top": (90, 0), "side": (0, 90), "back_iso": (22, 150)}  # y up; front looks along -z
C_SUP, C_LOAD, C_SOLID, C_VOID, C_DOM = "#1f6fd1", "#d1342f", "#2e9e4f", "#8a5a5a", "#555555"


def _other(a):
    return [b for b in range(3) if b != a]


def region_geometry(region: dict, L) -> dict:
    """Physical-unit geometry of a region dict: {"poly": [Nx3 arrays], "lines": [...], "centroid": xyz,
    "bbox": [min, max], "measure": (name, value)}."""
    k = region["kind"]
    L = np.asarray(L, float)
    if k in ("face", "patch"):
        a = AX[region["face"][0]]
        b, c = _other(a)
        x0 = 0.0 if region["face"][1] == "-" else L[a]
        if k == "face":
            cb, cc, sb, sc, ell = L[b] / 2, L[c] / 2, L[b], L[c], False
        else:
            (cb, cc), (sb, sc) = region["center"], region["size"]
            ell = region.get("shape", "rect") == "ellipse"
        if ell:
            t = np.linspace(0, 2 * np.pi, 49)
            ub, uc = cb + sb / 2 * np.cos(t), cc + sc / 2 * np.sin(t)
            area = np.pi * sb * sc / 4
        else:
            ub = np.array([cb - sb / 2, cb + sb / 2, cb + sb / 2, cb - sb / 2])
            uc = np.array([cc - sc / 2, cc - sc / 2, cc + sc / 2, cc + sc / 2])
            area = sb * sc
        P = np.zeros((ub.size, 3))
        P[:, a], P[:, b], P[:, c] = x0, ub, uc
        cen = np.zeros(3)
        cen[a], cen[b], cen[c] = x0, cb, cc
        lo, hi = P.min(0), P.max(0)
        return {"poly": [P], "lines": [], "centroid": cen, "bbox": [lo, hi], "measure": ("area", float(area)),
                "normal_axis": a, "normal_sign": -1 if region["face"][1] == "-" else 1}
    if k == "edge":
        a = AX[region["axis"]]
        b, c = _other(a)
        s0, s1 = region.get("span") or (0.0, L[a])
        P = np.zeros((2, 3))
        P[:, a] = [s0, s1]
        P[:, b], P[:, c] = region["at"]
        return {"poly": [], "lines": [P], "centroid": P.mean(0), "bbox": [P.min(0), P.max(0)],
                "measure": ("length", float(s1 - s0))}
    if k == "box":
        lo, hi = np.asarray(region["min"], float), np.asarray(region["max"], float)
        return {"poly": [], "lines": _box_lines(lo, hi), "centroid": (lo + hi) / 2, "bbox": [lo, hi],
                "measure": ("volume", float(np.prod(hi - lo)))}
    if k == "cylinder":
        a = AX[region["axis"]]
        b, c = _other(a)
        (cb, cc), r, (s0, s1) = region["center"], region["radius"], region["span"]
        t = np.linspace(0, 2 * np.pi, 49)
        lines = []
        for s in (s0, s1):
            P = np.zeros((t.size, 3))
            P[:, a], P[:, b], P[:, c] = s, cb + r * np.cos(t), cc + r * np.sin(t)
            lines.append(P)
        for th in np.linspace(0, 2 * np.pi, 4, endpoint=False):
            P = np.zeros((2, 3))
            P[:, a], P[:, b], P[:, c] = [s0, s1], cb + r * np.cos(th), cc + r * np.sin(th)
            lines.append(P)
        lo, hi = np.zeros(3), np.zeros(3)
        lo[a], hi[a] = s0, s1
        lo[b], hi[b], lo[c], hi[c] = cb - r, cb + r, cc - r, cc + r
        return {"poly": [], "lines": lines, "centroid": (lo + hi) / 2, "bbox": [lo, hi],
                "measure": ("volume", float(np.pi * r * r * (s1 - s0)))}
    raise ValueError(f"unknown region kind {k!r}")


def _box_lines(lo, hi):
    c = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    E = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [c[[i, j]] for i, j in E]


def render_spec(draft: dict, view: str = "iso", size_px: tuple[int, int] = (640, 480), labels: bool = True) -> bytes:
    """PNG of the domain, supports, loads and passive regions of a (possibly incomplete) spec dict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    L = np.asarray(draft["domain"]["size"], float)
    u = draft.get("units", {}) or {}
    lu, fu = u.get("length", "mm"), u.get("force", "N")
    dpi = 100
    fig = plt.figure(figsize=(size_px[0] / dpi, size_px[1] / dpi), dpi=dpi)
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    for P in _box_lines(np.zeros(3), L):
        ax.plot(*P.T, color=C_DOM, lw=0.8)

    def label(p, text, color):
        if labels:
            ax.text(*p, text, color=color, fontsize=8, weight="bold")

    for p in draft.get("passive", []) or []:
        g = region_geometry(p["region"], L)
        col = C_SOLID if p["kind"] == "solid" else C_VOID
        for P in g["lines"]:
            ax.plot(*P.T, color=col, lw=1.4, ls="-" if p["kind"] == "solid" else "--")
        label(g["centroid"], f'{p["id"]} {"keep-in" if p["kind"] == "solid" else "keep-out"}', col)
    for s in draft.get("supports", []) or []:
        g = region_geometry(s["region"], L)
        for P in g["poly"]:
            ax.add_collection3d(Poly3DCollection([P], facecolor=C_SUP, alpha=0.35, edgecolor=C_SUP, lw=1.0))
        for P in g["lines"]:
            ax.plot(*P.T, color=C_SUP, lw=2.5)
        label(g["centroid"], f'{s["id"]} fix {"".join(s.get("fix", ["x", "y", "z"]))}', C_SUP)
    span = float(L.max())
    for ld in draft.get("loads", []) or []:
        g = region_geometry(ld["region"], L)
        for P in g["poly"]:
            ax.add_collection3d(Poly3DCollection([P], facecolor=C_LOAD, alpha=0.5, edgecolor=C_LOAD, lw=1.0))
        for P in g["lines"]:
            ax.plot(*P.T, color=C_LOAD, lw=2.5)
        f = np.asarray(ld["force"], float)
        n = float(np.linalg.norm(f))
        if n > 0:
            d = f / n * 0.28 * span
            tip = g["centroid"]
            ax.quiver(*(tip - d), *d, color=C_LOAD, lw=2.0, arrow_length_ratio=0.25)
            label(tip - d, f'{ld["id"]} {n:g} {fu}', C_LOAD)
    ax.set_xlabel(f"x ({lu})", fontsize=8)
    ax.set_ylabel(f"y ({lu})", fontsize=8)
    ax.set_zlabel(f"z ({lu})", fontsize=8)
    ax.tick_params(labelsize=7)
    pad = 0.18 * span
    ax.set_xlim(-pad, L[0] + pad)
    ax.set_ylim(-pad, L[1] + pad)
    ax.set_zlim(-pad, L[2] + pad)
    ax.set_box_aspect(tuple(L + 2 * pad))
    elev, azim = VIEWS.get(view, VIEWS["iso"])
    ax.view_init(elev=elev, azim=azim, vertical_axis="y")
    ax.set_title(f'{draft.get("name", "")}  ({view})', fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", metadata={"Software": None})
    plt.close(fig)
    return buf.getvalue()


def png_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def density_thumbnail(rho, L, size_px=(200, 120)) -> bytes:
    """Small deterministic PNG of a design for clarification chips: the mean density seen along the domain's
    shortest axis (dark = material), on the domain's aspect ratio."""
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = int(np.argmin(L))
    img = np.asarray(rho, dtype=np.float32).mean(axis=a)
    rest = [i for i in range(3) if i != a]
    w, h = L[rest[0]], L[rest[1]]
    fig = plt.figure(figsize=(size_px[0] / 100, size_px[1] / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img.T, origin="lower", cmap="Greys", vmin=0, vmax=1, extent=(0, w, 0, h), interpolation="nearest")
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", metadata={"Software": None})
    plt.close(fig)
    return buf.getvalue()
