"""Figure 2: the running GUI (a) with its result card enlarged to readable print size (b).

The screenshot is 1680 x 1000 px; at full text width its 11-12 px interface text prints near 3 pt. Panel b crops
the result card at native pixels and places it at about 0.18 mm per pixel, so that text prints near 6 pt.
Images are embedded at native resolution (no resampling). Run from the manuscript package directory.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
SRC = HERE / "figures" / "gui_overview.png"
OUT = HERE / "figures" / "fig2_gui"

MM = 1 / 25.4
W = 162.0
GAP = 5.0
CARD = (1362, 44, 1680, 352)              # result card: certificate checks, grid, compliance, export
AMBER, INK = "#E69F00", "#1d1d1b"

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42})
im = Image.open(SRC).convert("RGB")
crop = im.crop(CARD)
wb = 56.0                                   # mm for panel b
hb = wb * crop.height / crop.width
wa = W - wb - GAP
ha = wa * im.height / im.width
H = max(ha, hb) + 5.0

fig = plt.figure(figsize=(W * MM, H * MM))
ax_a = fig.add_axes([0, 0, wa / W, ha / H])
ax_b = fig.add_axes([(wa + GAP) / W, (ha - hb) / H if ha > hb else 0, wb / W, hb / H])
for ax, img in ((ax_a, im), (ax_b, crop)):
    ax.imshow(img, interpolation="none")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5); s.set_color("#9b9a95")
x0, y0, x1, y1 = CARD
ax_a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.0, ec=AMBER))
ax_b.spines[:].set_color(AMBER); ax_b.spines[:].set_linewidth(1.0)
for (xa_, ya_), (xb_, yb_) in (((x1, y0), (0, 0)), ((x1, y1), (0, crop.height))):
    fig.add_artist(ConnectionPatch((xa_, ya_), (xb_, yb_), "data", "data", axesA=ax_a, axesB=ax_b,
                                   lw=0.6, color=AMBER))
top = max(ha, hb)
for x, lab, txt in ((0, "a", "Running interface, motor-mount task"),
                    ((wa + GAP), "b", "Result card enlarged: refine rung, not certified")):
    fig.text(x / W, (top + 1.4) / H, lab, fontsize=8, fontweight="bold", color=INK, va="bottom")
    fig.text((x + 3.2) / W, (top + 1.4) / H, txt, fontsize=7, color=INK, va="bottom")
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=600)
print(f"wrote {OUT.name}.pdf/png: {W:.0f} x {H:.1f} mm; panel b {wb:.0f} mm = {wb / crop.width:.3f} mm/px")
