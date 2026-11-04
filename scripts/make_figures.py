"""Paper figures, generated from the result files only (no GPU, no LLM).

    python -B scripts/make_figures.py        writes results/figures/<name>.pdf and .png

Style: the same system as the paper's Figure 1: Arial, type 6.5-7.5 pt
at the printed size, each figure drawn at the width it is printed at (the LaTeX includes them at natural size).
Colour: Okabe-Ito (colour-blind safe), with the meanings Figure 1 uses: bluish green = right / certified,
vermillion = wrong / missed target, blue = agent or the main arm, orange = intermediate or alternative arm,
dark grey = GUI, light grey = other or reference marks. Every series also has its own marker or a direct label.
One y-scale per panel.
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "figures"
BLUE, ORANGE, SKY, PASS, FAIL = "#0072B2", "#E69F00", "#56B4E9", "#009E73", "#D55E00"
GREY, INK, INK2, DARK = "#9b9a95", "#1d1d1b", "#55534e", "#4a4944"
MM = 1 / 25.4
W_FULL, W_MID = 160 * MM, 99 * MM          # printed widths in the V1 text block (162 mm)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "legend.frameon": False,
    "legend.fontsize": 6.5, "lines.linewidth": 1.2, "lines.markersize": 4, "savefig.dpi": 600,
    "figure.facecolor": "white", "axes.facecolor": "white", "pdf.fonttype": 42,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def load(p):
    return json.loads((R / p).read_text(encoding="utf-8"))


# ---- X2: outcomes per condition ---------------------------------------------------------------------------

def fig_x2():
    conds = [("R", "R  rule parser"), ("T", "T  text only"), ("G", "G  grounding tools"), ("GC", "G+C  + clarification"),
             ("GCV", "G+C+V  + picture check")]
    rows = []
    for c, label in conds:
        runs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((R / "X2" / c).glob("b*_r1.json"))]
        right = sum(bool(d.get("behaviour_ok")) for d in runs)
        silent = sum(1 for d in runs if not d.get("behaviour_ok") and d.get("schema_valid") and not d.get("refused")
                     and d.get("expected") != "refuse")
        rows.append((label, right, silent, len(runs) - right - silent))
    fig, ax = plt.subplots(figsize=(W_FULL * 0.9, 1.9))
    y = range(len(rows))[::-1]
    for yi, (label, right, silent, other) in zip(y, rows):
        left = 0
        for v, col in ((right, PASS), (silent, FAIL), (other, GREY)):
            if v:
                ax.barh(yi, v - 0.15, left=left, height=0.62, color=col, edgecolor="white", linewidth=0)
            left += v
        ax.text(right / 2, yi, f"{right}", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
        if silent:
            ax.text(right + silent / 2, yi, f"{silent}", ha="center", va="center", color="white", fontsize=7)
    ax.set_yticks(list(y), [r[0] for r in rows])
    ax.set_xlim(0, 50)
    ax.set_xlabel("briefs (of 50, repeat 1)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PASS, label="right behavior"), Patch(color=FAIL, label="delivered wrong spec"),
                       Patch(color=GREY, label="other failure (invalid, wrong refusal)")],
              loc="upper center", bbox_to_anchor=(0.45, 1.22), ncol=3, handlelength=1, columnspacing=1.2)
    save(fig, "x2_outcomes")


# ---- X2: two providers at matched effort ------------------------------------------------------------------

def fig_providers():
    """Codex and Claude Code, both at effort high, all 50 briefs, repeat 1: outcome bars per condition."""
    rows = []
    for c, cl in (("T", "T  text only"), ("GC", "G+C"), ("GCV", "G+C+V")):
        for sub, prov in ((c, "Codex"), (f"claude_high/{c}", "Claude Code")):
            runs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((R / "X2" / sub).glob("b*_r1.json"))]
            right = sum(bool(d.get("behaviour_ok")) for d in runs)
            silent = sum(1 for d in runs if not d.get("behaviour_ok") and d.get("schema_valid")
                         and not d.get("refused") and d.get("expected") != "refuse")
            q = sum(d.get("questions", 0) for d in runs)
            rows.append((f"{cl}, {prov}", right, silent, len(runs) - right - silent, q, c != "T"))
    fig, ax = plt.subplots(figsize=(W_FULL * 0.9, 2.3))
    ys, y = [], 0.0
    for i in range(len(rows)):
        ys.append(-y)
        y += 0.8 if i % 2 == 0 else 1.25          # pairs of rows per condition
    for yi, (label, right, silent, other, q, show_q) in zip(ys, rows):
        left = 0
        for v, col in ((right, PASS), (silent, FAIL), (other, GREY)):
            if v:
                ax.barh(yi, v - 0.15, left=left, height=0.62, color=col, edgecolor="white", linewidth=0)
            left += v
        ax.text(right / 2, yi, f"{right}", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
        if silent:
            ax.text(right + silent / 2, yi, f"{silent}", ha="center", va="center", color="white", fontsize=7)
        if show_q:
            ax.text(50.8, yi, f"{q} questions", ha="left", va="center", color=INK2, fontsize=7)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlim(0, 50)
    ax.set_xlabel("briefs (of 50, repeat 1; both providers at effort high)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PASS, label="right behavior"), Patch(color=FAIL, label="delivered wrong spec"),
                       Patch(color=GREY, label="other failure")],
              loc="upper center", bbox_to_anchor=(0.45, 1.16), ncol=3, handlelength=1, columnspacing=1.2)
    save(fig, "x2_providers")


# ---- X1: warm release arms against cold -------------------------------------------------------------------

def fig_x1():
    d = load("X1/escalation.json")
    arms = [("tail", "tail (certifies the upsampled design)", ORANGE, "s"), ("reopen", "short re-opening", SKY, "o"),
            ("reopen_long", "long re-opening (Optimize release)", BLUE, "D")]
    fig, ax = plt.subplots(figsize=(W_MID, 2.5))
    ax.axhline(0, color=GREY, linewidth=0.8, zorder=0)
    ax.axvline(100, color=GREY, linewidth=0.8, zorder=0)
    for arm, label, col, mk in arms:
        xs, ys = [], []
        for row in d["rows"]:
            rel = row["release"]
            a, c = rel.get(arm), rel.get("cold")
            if a and c and a.get("gap_to_cold") is not None and c.get("compute_s"):
                xs.append(100 * a["compute_s"] / c["compute_s"])
                ys.append(100 * a["gap_to_cold"])
        ax.scatter(xs, ys, s=22, marker=mk, color=col, edgecolor="white", linewidth=0.6, label=label, zorder=3)
    ax.set_xlabel("release compute (% of cold at the same rung)")
    ax.set_ylabel("compliance vs cold (%)")
    ax.text(100, 1.55, " cold", color=INK2, fontsize=7, va="top")
    ax.set_xlim(0, 110)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), handletextpad=0.3)
    save(fig, "x1_release_arms")


# ---- X5: part compliance along the resolution path --------------------------------------------------------

def fig_x5():
    x = load("X5/resolution.json")["span_bridge"]
    fine = x["fine_grid"]
    cold = sorted((r for r in x["rows"] if r["arm"] == "cold" and r.get("c_part")), key=lambda r: -r["h"])
    path = sorted((r for r in x["chain_reopen_long"] if r.get("c_part")), key=lambda r: -r["h"])
    fig, ax = plt.subplots(figsize=(W_MID, 2.4))
    ax.plot([r["h"] for r in cold], [r["c_part"] for r in cold], color=ORANGE, marker="s", label="cold run at each h")
    ax.plot([r["h"] for r in path], [r["c_part"] for r in path], color=BLUE, marker="o",
            label="release path (cold 8 mm, then long re-opening)")
    own = [(r["h"], r["c_own_phys"]) for r in cold if r.get("c_own_phys") and r["grid"] != fine]
    ax.plot([h for h, _ in own], [c for _, c in own], color=GREY, marker="^", linewidth=1, linestyle="--",
            label="coarse rung's own prediction")
    ax.set_xscale("log", base=2)
    ax.invert_xaxis()
    ax.set_xticks([8, 4, 2, 1], ["8", "4", "2", "1"])
    ax.set_xlabel("element size h (mm); finer to the right")
    ax.set_ylabel("part compliance at 33M (N·mm)")
    ax.legend(loc="upper left")
    ax.annotate(f"{path[-1]['c_part']:.0f}", (path[-1]["h"], path[-1]["c_part"]), textcoords="offset points",
                xytext=(4, -10), fontsize=7, color=INK)
    ax.annotate(f"{cold[-1]['c_part']:.0f}", (cold[-1]["h"], cold[-1]["c_part"]), textcoords="offset points",
                xytext=(4, 2), fontsize=7, color=INK)
    save(fig, "x5_resolution_path")


# ---- solver: padding and PCG iterations -------------------------------------------------------------------

def fig_padding():
    a = load("solver/mg_padding_aware.json")
    v = load("solver/mg_padding.json")
    fig, ax = plt.subplots(figsize=(W_MID, 2.4))
    for rows, col, mk, label in ((v, ORANGE, "s", "default coarse rule"), (a, BLUE, "o", "pad-aware coarse rule")):
        ax.scatter([100 * r["pad_frac"] for r in rows], [r["pcg_iters_mean"] for r in rows], s=22, marker=mk,
                   color=col, edgecolor="white", linewidth=0.7, label=label, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("padded elements added (% of the grid)")
    ax.set_ylabel("PCG iterations per solve")
    ax.legend(loc="upper left")
    save(fig, "solver_padding")


# ---- X6b: KLM user time per task --------------------------------------------------------------------------

def fig_klm():
    k = [r for r in load("X6/klm.json") if r["expected"] != "refuse"]
    k.sort(key=lambda r: r["B"]["s"])
    fig, ax = plt.subplots(figsize=(W_FULL * 0.9, 3.0))
    y = list(range(len(k)))
    series = [("A1", "solver script", ORANGE, "s", True), ("A2", "typed structured specification", ORANGE, "s", False),
              ("B", "GUI alone", DARK, "o", True), ("C_typed", "GUI + agent, brief typed", BLUE, "D", False),
              ("C_pasted", "GUI + agent, brief pasted", BLUE, "D", True)]
    for key, label, col, mk, filled in series:
        ax.scatter([r[key]["s"] for r in k], y, s=20, marker=mk, label=label, zorder=3,
                   color=col if filled else "white", edgecolor=col, linewidth=1.1)
    for yi, r in zip(y, k):
        xs = [r[s[0]]["s"] for s in series]
        ax.plot([min(xs), max(xs)], [yi, yi], color="#e6e5e1", linewidth=1, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([20, 50, 100, 200, 500, 1000], ["20", "50", "100", "200", "500", "1000"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(y, [r["task"].split("_", 1)[0] for r in k])
    ax.set_xlabel("estimated user time to set up, release and export (s, KLM; log scale)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, handletextpad=0.2, columnspacing=1)
    save(fig, "x6b_klm")


# ---- X3: where the session time goes ----------------------------------------------------------------------

def fig_x3():
    recs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((R / "X3").glob("b*.json"))
            if "_run" not in f.stem]
    recs = [d for d in recs if d.get("released")]
    rows = []
    for d in recs:
        st = d["stages"]
        solves = {k: v for k, v in st.items() if k.startswith("solve_h") and isinstance(v, dict)}
        rel = max(solves.values(), key=lambda v: v["n_elem"])
        coarse = sum(v["wall_s"] for v in solves.values() if v is not rel)
        base = st.get("total_s", d["wall_s"])   # logged session: first tool call to final export, CLI startup excluded
        other = max(0.0, base - st.get("spec_s", 0) - coarse - rel["wall_s"])
        aw = R / "X3_aware" / f"{d['task']}.json"
        aware = json.loads(aw.read_text()).get("total_aware_s") if aw.exists() else None
        code, name = d["task"].split("_", 1)
        rows.append((f"{name.replace('_', ' ')} ({code})", st.get("spec_s", 0), coarse, rel["wall_s"], other, rel["n_elem"], aware))
    rows.sort(key=lambda r: sum(r[1:5]))
    fig, ax = plt.subplots(figsize=(W_FULL, 2.9))
    y = list(range(len(rows)))
    parts = [(1, "agent writes the spec", BLUE), (2, "sketch, preview, refine", ORANGE), (3, "certified release", PASS),
             (4, "other agent time (checks, export, report)", GREY)]
    for yi, r in zip(y, rows):
        left = 0
        for i, _, col in parts:
            if r[i] > 0:
                ax.barh(yi, max(r[i] - 1.5, 0.5), left=left, height=0.62, color=col, linewidth=0)
            left += r[i]
        ax.text(max(left, r[6] or 0) + 22, yi, f"{r[5] / 1e6:.0f}M", va="center", fontsize=6.5, color=INK2)
        if r[6] is not None:
            ax.plot([r[6]], [yi], marker="D", ms=4.5, mfc="white", mec=INK, mew=1.0, ls="none", zorder=5)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("logged session time (s); label: release elements")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    hs = [Patch(color=c, label=l) for _, l, c in parts]
    if any(r[6] is not None for r in rows):
        hs.append(Line2D([], [], marker="D", ms=4.5, mfc="white", mec=INK, ls="none",
                         label="whole session, solves replayed with the pad-aware coarse rule"))
    ax.legend(handles=hs, loc="lower right")
    save(fig, "x3_session_time")


def fig_latency():
    files = sorted((R / "X6" / "latency").glob("b*.json"))
    if not files:
        print("skip latency: X6a not run")
        return
    rows = [json.loads(f.read_text()) for f in files]
    rung = lambda r, n: next(g["wall_s"] for g in r["rungs"] if g["rung"] == n)  # noqa: E731
    items = [("validation (p95)", [r["validate"]["p95_ms"] / 1e3 for r in rows], 0.2),
             ("edit → warm sketch (p95)", [r["edit_to_sketch"]["wall_p95_s"] for r in rows], 3),
             ("cold sketch (new spec)", [rung(r, "sketch") for r in rows], 3),
             ("preview (warm)", [rung(r, "preview") for r in rows], 5),
             ("release surface, first fetch (backend)", [r["surface"]["first_s"] for r in rows if r.get("surface")], 3)][::-1]
    fig, ax = plt.subplots(figsize=(W_MID, 2.4))
    for i, (name, xs, hi) in enumerate(items):
        ax.add_patch(plt.Rectangle((5e-4, i - 0.32), hi - 5e-4, 0.64, color="#e6e5e1", zorder=0, linewidth=0))
        ok = [x for x in xs if x <= hi]
        over = [x for x in xs if x > hi]
        ax.scatter(ok, [i] * len(ok), s=14, color=PASS, edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter(over, [i] * len(over), s=16, color=FAIL, marker="s", edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(30, i, f"{len(ok)}/{len(xs)}", va="center", fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(5e-4, 60)
    ax.set_xticks([1e-3, 1e-2, 0.1, 1, 10], ["1 ms", "10 ms", "0.1 s", "1 s", "10 s"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(range(len(items)), [n for n, _, _ in items])
    ax.set_xlabel("one dot per task; grey band: target")
    ax.grid(axis="y", visible=False)
    save(fig, "x6a_latency")


if __name__ == "__main__":
    for f in (fig_x2, fig_providers, fig_x1, fig_x5, fig_padding, fig_klm, fig_x3, fig_latency):
        f()
