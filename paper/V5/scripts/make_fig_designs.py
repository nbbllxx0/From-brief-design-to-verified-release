"""Main-text design figure: released parts from complete agent sessions, each with the supports and loads its
specification states (drawn from the release report's own spec, in the spec's units and frame).

    python -B scripts/make_fig_designs.py

Reads results/X3/<task>.json and results/X3/<task>/session/exports/*.stl + *_report.json (the certified release
rung). Writes figures/fig_designs.{pdf,png} at print width (162 mm) and the per-part renders to
figures/fig_designs_renders/. Colours keep the project's meanings: supports blue, loads vermillion,
keep-out regions grey dashed, part neutral grey; nothing else is encoded by colour.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pyvista as pv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentto.render import region_geometry  # noqa: E402

OUT = PACKAGE / "figures"
REN = OUT / "fig_designs_renders"
MM = 1 / 25.4
INK, MUTED = "#1f1e1c", "#5f5e5a"
PART, SUP, LOAD, VOID, BOX = "#bdbab2", "#0072B2", "#D55E00", "#8c8a85", "#a9a7a1"
GREEN = "#009E73"
plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "font.size": 6.5})

# Eight released sessions chosen to span the suite: brackets, a bridge, a mount, a beam, an imperial-unit part and a
# clarification task. Order: reading order of the figure.
TASKS = [("b06_motor_mount", "Motor mount"), ("b19_engine_bracket", "Engine bracket"),
         ("b05_l_bracket", "L-bracket"), ("b17_arch_bridge", "Arch bridge"),
         ("b47_camera_mount", "Camera mount"), ("b40_crank_arm", "Crank arm"),
         ("b02_mbb_half", "MBB beam, half model"), ("b11_cantilever_line_load", "Cantilever, line load")]
VIEW = {"b17_arch_bridge": (0.35, 0.35, 1.0), "b02_mbb_half": (0.45, 0.45, 1.0)}
DEFAULT_VIEW = (0.85, 0.6, 1.2)


def release(task):
    d = json.loads((ROOT / "results" / "X3" / f"{task}.json").read_text(encoding="utf-8"))
    stl = sorted((ROOT / "results" / "X3" / task / "session" / "exports").glob("*.stl"))[-1]
    rep = json.loads(stl.with_name(stl.stem + "_report.json").read_text(encoding="utf-8"))
    assert d["released"] and rep["certified"] and rep["release_rung"], task
    return stl, rep


def cone(p, tip, outward, h):
    """A support glyph: a small cone whose tip touches the restrained surface, base outside the part."""
    outward = np.asarray(outward, float) / np.linalg.norm(outward)
    p.add_mesh(pv.Cone(center=np.asarray(tip) + outward * h / 2, direction=-outward, height=h, radius=0.45 * h,
                       resolution=24), color=SUP, smooth_shading=True)


def exit_point(c, d, lo, hi):
    """Where the ray from c along d leaves the box [lo, hi] (c inside); c itself if the box is flat along d."""
    t = np.inf
    for k in range(3):
        if abs(d[k]) > 1e-12 and hi[k] - lo[k] > 1e-9:
            t = min(t, ((hi[k] if d[k] > 0 else lo[k]) - c[k]) / d[k])
    return c if not np.isfinite(t) else c + d * t


def overlay(p, spec):
    L = np.asarray(spec["domain"]["size"], float)
    diag = float(np.linalg.norm(L))
    h = 0.045 * diag
    p.add_mesh(pv.Box(bounds=(0, L[0], 0, L[1], 0, L[2])).outline(), color=BOX, line_width=1.2)
    for s in spec.get("supports", []):
        reg = s["region"]
        g = region_geometry(reg, L)
        for P in g["poly"]:
            a, sgn = g["normal_axis"], g["normal_sign"]
            out = np.zeros(3)
            out[a] = sgn
            Q = P.copy()
            Q[:, a] += sgn * 0.004 * diag                     # just outside the part so the patch is not hidden
            faces = np.hstack([[len(Q)], np.arange(len(Q))])
            p.add_mesh(pv.PolyData(Q, faces), color=SUP, opacity=0.5, show_edges=False)
            p.add_mesh(pv.lines_from_points(np.vstack([Q, Q[:1]])), color=SUP, line_width=2.0)
            lo, hi = g["bbox"]
            b, c = [k for k in range(3) if k != a]
            nb = 1 if hi[b] - lo[b] < 3 * h else 3
            nc = 1 if hi[c] - lo[c] < 3 * h else 2
            for ub in (np.linspace(lo[b], hi[b], nb + 2)[1:-1] if nb > 1 else [g["centroid"][b]]):
                for uc in (np.linspace(lo[c], hi[c], nc + 2)[1:-1] if nc > 1 else [g["centroid"][c]]):
                    tip = np.zeros(3)
                    tip[a], tip[b], tip[c] = Q[0, a], ub, uc
                    cone(p, tip, out, h)
        if reg["kind"] == "edge":
            P = g["lines"][0]
            ax = np.argmax(np.abs(P[1] - P[0]))
            p.add_mesh(pv.Tube(pointa=P[0], pointb=P[1], radius=0.006 * diag), color=SUP)
            for t in (0.2, 0.5, 0.8):
                q = P[0] + t * (P[1] - P[0])
                out = q - L / 2
                out[ax] = 0
                if np.linalg.norm(out) == 0:
                    out[1] = -1
                cone(p, q, out, h)
        elif reg["kind"] == "cylinder":
            k = "xyz".index(reg["axis"])
            (cb, cc), r, (s0, s1) = reg["center"], reg["radius"], reg["span"]
            ctr = np.zeros(3)
            o = [j for j in range(3) if j != k]
            ctr[k], ctr[o[0]], ctr[o[1]] = (s0 + s1) / 2, cb, cc
            ax3 = np.zeros(3)
            ax3[k] = 1
            p.add_mesh(pv.Cylinder(center=ctr, direction=ax3, radius=r * 1.06, height=(s1 - s0) * 1.02,
                                   resolution=60), color=SUP, opacity=0.55, smooth_shading=True)
        elif reg["kind"] == "box":
            for P in g["lines"]:
                p.add_mesh(pv.lines_from_points(P), color=SUP, line_width=2.5)
    loads = spec.get("loads", [])
    fmax = max((np.linalg.norm(ld["force"]) for ld in loads), default=1.0) or 1.0
    for ld in loads:
        g = region_geometry(ld["region"], L)
        f = np.asarray(ld["force"], float)
        n = np.linalg.norm(f)
        if n == 0:
            continue
        length = 0.22 * diag * (0.45 + 0.55 * n / fmax)
        lo, hi = (np.asarray(v, float) for v in g["bbox"])
        tip = np.asarray(g["centroid"], float)
        if ld["region"]["kind"] in ("box", "cylinder"):          # volume regions: put the tip on their surface
            tip = exit_point(tip, -f / n, lo, hi)
        start = tip - f / n * length
        p.add_mesh(pv.Arrow(start=start, direction=f / n, scale=length, tip_length=0.28, tip_radius=0.09,
                            shaft_radius=0.035), color=LOAD)
        for P in g["poly"]:
            p.add_mesh(pv.lines_from_points(np.vstack([P, P[:1]])), color=LOAD, line_width=1.6)
    for pr in spec.get("passive", []):
        if pr["kind"] == "void":
            for P in region_geometry(pr["region"], L)["lines"]:
                p.add_mesh(pv.lines_from_points(P), color=VOID, line_width=1.2, opacity=0.9)


def shot(task, stl, spec, out):
    m = pv.read(str(stl)).smooth_taubin(n_iter=30, pass_band=0.08)
    p = pv.Plotter(off_screen=True, window_size=[1500, 1200])
    p.set_background("white")
    p.enable_anti_aliasing("ssaa")
    p.add_mesh(m, color=PART, smooth_shading=True, specular=0.25, specular_power=20, ambient=0.18, diffuse=0.85)
    overlay(p, spec)
    L = np.asarray(spec["domain"]["size"], float)
    c = L / 2
    d = np.asarray(VIEW.get(task, DEFAULT_VIEW), float)
    p.camera.focal_point = c
    p.camera.position = c + d / np.linalg.norm(d) * 3.2 * np.linalg.norm(L)
    p.camera.up = (0.0, 1.0, 0.0)
    p.reset_camera()
    p.camera.zoom(1.25)
    out.parent.mkdir(parents=True, exist_ok=True)
    p.screenshot(str(out))
    p.close()


def crop(im, pad=8):
    ys, xs = np.where(im[..., :3].mean(axis=2) < 0.985)
    return im[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]


def fmt_force(f, unit):
    n = float(np.linalg.norm(f))
    if unit == "N" and n >= 1000:
        return f"{n / 1000:.3g} kN"
    return f"{n:,.0f} {unit}" if n >= 1000 else f"{n:.3g} {unit}"


def main():
    cards = []
    for task, title in TASKS:
        stl, rep = release(task)
        spec = rep["spec"]
        png = REN / f"{task}.png"
        shot(task, stl, spec, png)
        u = spec["units"]
        loads = ", ".join(fmt_force(ld["force"], u["force"]) for ld in spec.get("loads", []))
        L = spec["domain"]["size"]
        line2 = f"{L[0]:g} × {L[1]:g} × {L[2]:g} {u['length']}  ·  load {loads}"
        line3 = f"{rep['n_elem'] / 1e6:.1f} M elements  ·  certified"
        cards.append((title, task.split("_")[0], line2, line3, png))
        print("rendered", task, flush=True)

    fig = plt.figure(figsize=(162 * MM, 92 * MM), dpi=300)
    cols, rows = 4, 2
    gw, gh = 1 / cols, 1 / rows
    for i, (title, tid, line2, line3, png) in enumerate(cards):
        r, c = divmod(i, cols)
        x0, y0 = c * gw, 1 - (r + 1) * gh
        ax = fig.add_axes([x0 + 0.006, y0 + 0.215 * gh, gw - 0.012, 0.76 * gh])
        ax.imshow(crop(plt.imread(png)))
        ax.set_axis_off()
        fig.text(x0 + 0.012, y0 + 0.165 * gh, f"{chr(97 + i)}", fontsize=8, fontweight="bold", color=INK, va="baseline")
        fig.text(x0 + 0.045, y0 + 0.165 * gh, f"{title}  ({tid})", fontsize=6.8, fontweight="bold", color=INK, va="baseline")
        fig.text(x0 + 0.045, y0 + 0.085 * gh, line2, fontsize=6.0, color=MUTED, va="baseline")
        fig.text(x0 + 0.045, y0 + 0.012 * gh, line3.replace("certified", ""), fontsize=6.0, color=MUTED, va="baseline")
        tw = fig.text(x0 + 0.045, y0 + 0.012 * gh, line3.replace("certified", ""), fontsize=6.0, alpha=0)
        fig.canvas.draw()
        bb = tw.get_window_extent().transformed(fig.transFigure.inverted())
        fig.text(bb.x1, y0 + 0.012 * gh, "certified", fontsize=6.0, color=GREEN, fontweight="bold", va="baseline")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_designs.{ext}")
    plt.close(fig)
    print("wrote", OUT / "fig_designs.pdf")


if __name__ == "__main__":
    main()
