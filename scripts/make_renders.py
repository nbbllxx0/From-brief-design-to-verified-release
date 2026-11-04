"""3D renders of released designs for the paper (pyvista, off screen).

- X3 gallery: the exported STL of every released X3 session (the certified release rung), one PNG each, plus a
  labelled gallery figure.
- X1 arms: cold / tail / reopen_long release designs of the same problem (density fields, 2x pooled), as one row each.
- X5: the span bridge at 33M, cold fine run against the release path.

    python -B scripts/make_renders.py [gallery|arms|x5]      (all by default)
Writes results/figures/renders/*.png and results/figures/{x3_gallery, x1_arms_3d, x5_bridge_3d}.{png,pdf}.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
REN = FIG / "renders"
INK, MUTED = "#1f1e1c", "#6b6a66"
PART = "#b9b7b0"
CUT = {"b04", "b31", "b35", "b44"}   # designs that are closed shells from outside          # neutral part colour; identity is never carried by colour here
pv.global_theme.font.family = "arial"
import matplotlib
matplotlib.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42})   # same type as the other figures


def _plotter(size=(1400, 1000)):
    p = pv.Plotter(off_screen=True, window_size=list(size))
    p.set_background("white")
    p.enable_anti_aliasing("ssaa")
    return p


def _shot(mesh, out: Path, box=None, view="iso", zoom=1.0, size=(1400, 1000)):
    p = _plotter(size)
    p.add_mesh(mesh, color=PART, smooth_shading=True, specular=0.25, specular_power=20, ambient=0.15)
    if box is not None:
        p.add_mesh(pv.Box(bounds=box).outline(), color="#9b9a95", line_width=1.5)
    # the suite's y is up: an engineering three-quarter view from +x, +y, +z with y vertical on screen
    c = np.array(mesh.center)
    d = np.array({"iso": (0.75, 0.55, 1.25), "elev": (0.22, 0.14, 1.0), "front": (0.0, 0.0, 1.0), "span": (0.3, 0.32, 1.0)}[view], dtype=float)
    p.camera.focal_point = c
    p.camera.position = c + d / np.linalg.norm(d) * 3.2 * mesh.length
    p.camera.up = (0.0, 1.0, 0.0)
    p.reset_camera()
    p.camera.zoom(zoom)
    out.parent.mkdir(parents=True, exist_ok=True)
    p.screenshot(str(out), transparent_background=False)
    p.close()


def _iso(field: np.ndarray, h: float = 1.0, level: float = 0.5):
    """rho = level surface, closed: a one-cell void layer around the domain closes the surface where material lies
    on a domain wall (without it, wall-bound webs render as loose one-sided sheets)."""
    field = np.pad(np.asarray(field, dtype=np.float32), 1)
    img = pv.ImageData(dimensions=np.array(field.shape) + 1, spacing=(h, h, h), origin=(-h, -h, -h))
    img.cell_data["rho"] = field.ravel(order="F")
    pts = img.cell_data_to_point_data()
    return pts.contour([level], scalars="rho", method="flying_edges")


def _y_up(mesh):
    """Light Taubin smoothing of the voxel steps (the camera keeps y up; no rotation is needed)."""
    return mesh.smooth_taubin(n_iter=30, pass_band=0.08)


# ---- X3 gallery ----------------------------------------------------------------------------------------------

def x3_tasks():
    out = []
    for f in sorted((ROOT / "results" / "X3").glob("b*.json")):
        if "_run" in f.stem:
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        stl = sorted((ROOT / "results" / "X3" / f.stem / "session" / "exports").glob("*.stl"))
        if not d.get("released") or not stl:
            continue
        rel = max((v for k, v in d["stages"].items() if k.startswith("solve_")), key=lambda v: v["n_elem"])
        out.append({"task": f.stem, "stl": stl[-1], "n_elem": rel["n_elem"], "certified": rel["certified"]})
    return out


def gallery():
    import matplotlib.pyplot as plt
    rows = x3_tasks()
    for r in rows:
        png = REN / f"x3_{r['task']}.png"
        if not png.exists():
            m = _y_up(pv.read(str(r["stl"])))
            if r["task"][:3] in CUT:                 # closed shells: cut the near half away to show the inside
                m = m.clip(normal="z", origin=m.center, invert=True)
            _shot(m, png, box=None, zoom=1.15)
        r["png"] = png
        print("rendered", r["task"], flush=True)
    n = len(rows)
    cols = 5
    k = -(-n // cols)
    fig, axs = plt.subplots(k, cols, figsize=(6.3, 1.4 * k), dpi=300)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, r in zip(axs.ravel(), rows):
        im = plt.imread(r["png"])
        ys, xs = np.where(im[..., :3].mean(axis=2) < 0.985)
        pad = 10
        im = im[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]
        ax.imshow(im)
        name = r["task"].split("_", 1)
        cut = " (half cut away)" if r["task"][:3] in CUT else ""
        ax.set_title(f"{name[0]} {name[1].replace('_', ' ')}{cut}\n{r['n_elem'] / 1e6:.1f}M elements",
                     fontsize=6.5, color=INK, pad=2)
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"x3_gallery.{ext}", bbox_inches="tight")
    plt.close(fig)


# ---- X1 arms and X5 ------------------------------------------------------------------------------------------

def _row_figure(items, out_stem, ncols=None, title_size=7):
    import matplotlib.pyplot as plt
    ncols = ncols or len(items)
    nrows = -(-len(items) // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6.3, 1.8 * nrows), dpi=300, squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (png, label) in zip(axs.ravel(), items):
        im = plt.imread(png)
        ys, xs = np.where(im[..., :3].mean(axis=2) < 0.985)
        im = im[max(ys.min() - 10, 0):ys.max() + 10, max(xs.min() - 10, 0):xs.max() + 10]
        ax.imshow(im)
        ax.set_title(label, fontsize=title_size, color=INK, pad=3)
    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{out_stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def arms():
    esc = json.loads((ROOT / "results" / "X1" / "escalation.json").read_text())
    items = []
    for prob in ("bridge", "b06_motor_mount"):
        row = next(r for r in esc["rows"] if r["problem"] in (prob, prob.split("_")[0]))
        for arm in ("cold", "tail", "reopen_long"):
            f = ROOT / "results" / "X1" / "fields" / f"{prob}_{arm}_release_p2.npy"
            png = REN / f"x1_{prob}_{arm}.png"
            if not png.exists():
                m = _y_up(_iso(np.load(f)))
                if prob == "bridge":                 # full-depth side webs hide the struts: cut the near half away
                    m = m.clip(normal="z", origin=m.center, invert=True)
                _shot(m, png, zoom=1.15, view="iso")
            rel = row["release"].get(arm) or {}
            c, c0 = rel.get("compliance"), (row["release"].get("cold") or {}).get("compliance")
            gap = "" if arm == "cold" or not (c and c0) else f", {100 * (c / c0 - 1):+.2f}% vs cold"
            label = {"cold": "cold", "tail": "warm, tail only", "reopen_long": "warm, long re-opening"}[arm]
            items.append((png, f"{prob.replace('_', ' ')}: {label}{gap}"))   # the bridge cutaway is stated in the caption
            print("rendered", prob, arm, flush=True)
    _row_figure(items, "x1_arms_3d", ncols=3, title_size=6)


def x5():
    res = json.loads((ROOT / "results" / "X5" / "resolution.json").read_text())["span_bridge"]
    items = []
    for arm, label in (("cold", "cold fine run, 33M (near half cut away)"),
                       ("reopen_long", "release path (cold coarse, then long re-opening), 33M (near half cut away)")):
        f = ROOT / "results" / "X5" / "fields" / f"span_bridge_{arm}_h1.npy"
        png = REN / f"x5_span_bridge_{arm}.png"
        if not png.exists():
            m = _y_up(_iso(np.load(f)))              # full resolution: 2x pooling thinned the joints below rho = 0.5
            m = m.clip(normal="z", origin=m.center, invert=True)
            _shot(m, png, zoom=1.3, size=(2000, 700), view="span")
        items.append((png, label))
        print("rendered x5", arm, flush=True)
    _row_figure(items, "x5_bridge_3d", ncols=1)


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8")
    REN.mkdir(parents=True, exist_ok=True)
    for what in argv or ["gallery", "arms", "x5"]:
        {"gallery": gallery, "arms": arms, "x5": x5}[what]()


if __name__ == "__main__":
    main(sys.argv[1:])
