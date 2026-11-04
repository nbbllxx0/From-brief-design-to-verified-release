"""Build the user guide: the numbered overview figure, docs/USER_GUIDE.html and docs/AgenticTO_User_Guide.pdf.

    python -B scripts/make_user_guide.py      (base Anaconda: markdown and Pillow)
Run scripts/guide_shots.py first; it writes the screenshots and rects.json to docs/user_guide/img/.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import markdown
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
IMG = DOCS / "user_guide" / "img"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
CALLOUTS = ["topbar", "tree", "checks", "inspector", "toolbar", "viewport", "result", "assistant", "ladder", "timeline"]
# where each number sits on its panel box: fraction of width and height, then an offset in CSS pixels
ANCHOR = {"topbar": (0.30, 0.5, 0, 0), "tree": (0.62, 0.06, 0, 0), "checks": (0.62, 0.15, 0, 0),
          "inspector": (0.62, 0.12, 0, 0), "toolbar": (1.0, 0.5, 20, 0), "viewport": (0.12, 0.13, 0, 0),
          "result": (0.90, 0.0, 0, 80), "assistant": (0.26, 0.05, 0, 0), "ladder": (0.80, 0.5, 0, 0),
          "timeline": (0.0, 0.5, 70, 0)}


def overview():
    meta = json.loads((IMG / "rects.json").read_text(encoding="utf-8"))
    s = meta["dpr"]
    im = Image.open(IMG / "09_release_surface.png").convert("RGB")
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("segoeuib.ttf", 15 * s)
    except OSError:
        font = ImageFont.load_default()
    r = 13 * s
    for n, key in enumerate(CALLOUTS, 1):
        box = meta["rects"].get(key)
        if not box:
            continue
        fx, fy, dx, dy = ANCHOR[key]
        x, y = (box[0] + fx * box[2] + dx) * s, (box[1] + fy * box[3] + dy) * s
        x = min(max(x, r + 2), im.width - r - 2)
        d.ellipse([x - r - 2 * s, y - r - 2 * s, x + r + 2 * s, y + r + 2 * s], fill="white")
        d.ellipse([x - r, y - r, x + r, y + r], fill="#d9480f")
        d.text((x, y), str(n), fill="white", font=font, anchor="mm")
    im.save(IMG / "00_overview.png", optimize=True)
    v = meta["rects"]["viewport"]                    # cover: the certified part, cropped from the same screenshot
    src = Image.open(IMG / "09_release_surface.png").convert("RGB")
    x0, y0, w, h = (v[0] + 0.14 * v[2]) * s, (v[1] + 0.25 * v[3]) * s, 0.76 * v[2] * s, 0.58 * v[3] * s
    src.crop((int(x0), int(y0), int(x0 + w), int(y0 + h))).save(IMG / "00_cover.png", optimize=True)


CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
:root { --ink: #1d2433; --ink2: #4a5468; --rule: #d9dee7; --accent: #2f55d4; --soft: #f3f5f9; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; color: var(--ink); font-size: 10.2pt;
  line-height: 1.5; max-width: 182mm; margin: 0 auto; padding: 0 4mm; background: #fff; }
h1 { font-size: 30pt; line-height: 1.1; margin: 30mm 0 4mm; letter-spacing: -0.5pt; }
h1 + p strong { font-size: 14pt; color: var(--ink2); font-weight: 600; }
h1 + p + p { color: var(--ink2); }
.cover { margin-top: 14mm; }
.cover img { width: 100%; border-radius: 6px; }
h2 { font-size: 17pt; margin: 0 0 3mm; padding-top: 2mm; border-bottom: 2px solid var(--accent); padding-bottom: 1.5mm;
  break-before: page; }
h3 { font-size: 12pt; margin: 6mm 0 2mm; color: var(--accent); }
p, li { orphans: 3; widows: 3; }
a { color: var(--accent); text-decoration: none; }
hr { display: none; }
code, kbd, pre { font-family: "Cascadia Mono", Consolas, monospace; }
code { font-size: 8.8pt; background: var(--soft); padding: 0.5pt 3pt; border-radius: 3px; }
pre { background: #1f2533; color: #e8ecf4; padding: 3mm 4mm; border-radius: 5px; font-size: 8.4pt; line-height: 1.45;
  overflow-wrap: anywhere; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; color: inherit; font-size: inherit; }
kbd { font-size: 8.4pt; border: 1px solid #b9c1cf; border-bottom-width: 2px; border-radius: 4px; padding: 0 4pt;
  background: #fff; white-space: nowrap; }
table { border-collapse: collapse; width: 100%; margin: 3mm 0 4mm; font-size: 9pt; break-inside: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th { text-align: left; background: var(--soft); border-bottom: 1.5px solid var(--ink2); padding: 1.6mm 2mm; }
td { border-bottom: 1px solid var(--rule); padding: 1.4mm 2mm; vertical-align: top; }
blockquote { margin: 3mm 0; padding: 2mm 4mm; background: #fff7e6; border-left: 3px solid #e8a200; border-radius: 0 4px 4px 0;
  break-inside: avoid; }
blockquote p { margin: 0; }
figure { margin: 4mm 0 5mm; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 120mm; border: 1px solid var(--rule); border-radius: 4px; }
figure.narrow img { max-height: 95mm; }
figure.short img { max-height: 84mm; }
figcaption { font-size: 8.8pt; color: var(--ink2); margin-top: 1.5mm; text-align: left; }
figcaption b { color: var(--ink); }
.pair { display: flex; gap: 5mm; align-items: flex-start; break-inside: avoid; }
.pair figure { flex: 1; margin-top: 0; }
.toc ol { columns: 2; column-gap: 10mm; padding-left: 5mm; }
"""


def figures(html: str) -> str:
    """<p><img alt="Figure N. text" src=...></p> becomes a figure with a caption."""
    def fig(m):
        src, alt = m.group("src"), m.group("alt")
        cap = re.sub(r"^(Figure \d+\.)", r"<b>\1</b>", alt)
        narrow = " class=\"narrow\"" if re.search(r"(06_rule|11_result|13_export|14_stale|16_keymap|17_palette|21_clarify|25_act)", src) else ""
        narrow = " class=\"short\"" if "00_overview" in src else narrow
        return f'<figure{narrow}><img src="{src}" alt=""><figcaption>{cap}</figcaption></figure>'
    pat = r'<p>\s*<img alt="(?P<alt>[^"]*)" src="(?P<src>[^"]+)"\s*/?>\s*</p>'
    html = re.sub(pat, fig, html)
    return re.sub(r'<img alt="(?P<alt>[^"]*)" src="(?P<src>[^"]+)"\s*/?>', fig, html)


def build():
    md = (DOCS / "USER_GUIDE.md").read_text(encoding="utf-8")
    md = md.replace("**Contents**", '<div class="toc" markdown="1">\n\n**Contents**', 1)
    md = md.replace("12. Troubleshooting and limits\n", "12. Troubleshooting and limits\n\n</div>\n", 1)
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "md_in_html", "sane_lists"])
    body = figures(body)
    body = body.replace("<hr />", '<div class="cover"><img src="user_guide/img/00_cover.png" alt=""></div>', 1)
    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>AgenticTO User Guide</title>'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><style>{CSS}</style></head>'
            f"<body>{body}</body></html>")
    out = DOCS / "USER_GUIDE.html"
    out.write_text(html, encoding="utf-8")
    pdf = DOCS / "AgenticTO_User_Guide.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    out.resolve().as_uri()], check=True, capture_output=True, timeout=180)
    print("wrote", out, pdf, f"{pdf.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    overview()
    build()
