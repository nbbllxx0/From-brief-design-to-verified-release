"""Screenshots for the user guide (docs/USER_GUIDE.md): a real session driven through headless Chrome with real
clicks and keys, on the same harness as scripts/gui_test.py.

Needs the backend and the Vite front end running and the GPU free:
    python -B scripts/serve.py --ladder-max 1.2e7       (small rungs, so the release solves in about a minute)
    node web/node_modules/vite/bin/vite.js web
    python -B scripts/guide_shots.py [build] [solve] [overlays] [assistant] [attach]
`assistant` runs Codex sessions and `attach` one Claude Code session, both on the subscription CLIs.
Writes docs/user_guide/img/<name>.png at 2x device pixels, plus rects.json (panel boxes for the callout figure).
"""
import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gui_test import CDP, CHROME, ST, URL, fresh, new_design  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "user_guide" / "img"
W, H, DPR = 1600, 960, 2
PANELS = {"topbar": ".topbar", "tree": ".tree", "checks": ".findings", "inspector": ".inspector", "toolbar": ".toolbar",
          "viewport": ".view", "result": "[aria-label='Latest result']", "assistant": ".assistant-panel",
          "ladder": ".ladder", "timeline": ".timeline"}
RUNG = "[...document.querySelectorAll('.ladder .rung')][%d]"


async def shot(c: CDP, name, sel=None, pad=8):
    """Full window, or the box of `sel` plus `pad` CSS pixels."""
    clip = None
    if sel:
        r = await c.js(f"""(() => {{ const e = document.querySelector({json.dumps(sel)}); if (!e) return null;
          const r = e.getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; }})()""")
        if r:
            x, y = max(0, r[0] - pad), max(0, r[1] - pad)
            clip = {"x": x, "y": y, "width": min(W - x, r[2] + 2 * pad), "height": min(H - y, r[3] + 2 * pad), "scale": 1}
    kw = {"clip": clip} if clip else {}
    res = await c.call("Page.captureScreenshot", format="png", **kw)
    (IMG / f"{name}.png").write_bytes(base64.b64decode(res["data"]))
    print("wrote", name, flush=True)


async def rects(c: CDP):
    out = {}
    for k, sel in PANELS.items():
        out[k] = await c.js(f"""(() => {{ const e = document.querySelector({json.dumps(sel)}); if (!e) return null;
          const r = e.getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; }})()""")
    (IMG / "rects.json").write_text(json.dumps({"dpr": DPR, "rects": out}, indent=1), encoding="utf-8")


async def ready(c: CDP):
    await c.until("[...document.querySelectorAll('.pill')].some(p => /GPU ready/.test(p.innerText))", 240, what="GPU ready")


async def rung_done(c: CDP, i, timeout=900):
    await c.until(f"/\\d+\\.\\ds/.test({RUNG % i}.innerText)", timeout, 1.0, what=f"rung {i}")


# ------------------------------------------------------------------------------------------------------------
async def build(c: CDP):
    """A cantilever bracket built by hand on a blank box: support, load, a rule error and its fix, keep-in/out."""
    await fresh(c)
    await new_design(c)
    await asyncio.sleep(1.0)
    await shot(c, "02_new_design")

    await c.click(800, 480)
    await c.key("s")
    await asyncio.sleep(0.3)
    await c.js("window.__agentto.view([-1, 0.45, 0.35]); true")
    await asyncio.sleep(0.5)
    await shot(c, "03_support_tool")
    await c.click_face("x-")

    await c.key("l")
    await c.click_face("x+", (0.5, 0.5))
    await c.type_into('[aria-label="force x"]', "0")
    await c.type_into('[aria-label="force y"]', "-2000")
    await c.type_into('[aria-label="size y"]', "10")
    await c.type_into('[aria-label="size z"]', "20")
    await asyncio.sleep(1.5)
    await shot(c, "04_load_inspector")

    # a support that fixes x only: the rule base reports the free rigid-body modes (R003)
    await c.click_sel(".tree-item", "support_1")
    for a in ("y", "z"):
        await c.js(f"(() => {{ const l = [...document.querySelectorAll('.checks label')].find(x => x.innerText.trim() === '{a}'); "
                   "l.querySelector('input').click(); return true; })()")
        await asyncio.sleep(0.2)
    await asyncio.sleep(1.2)
    await shot(c, "06_rule_error", ".left", pad=0)
    for a in ("y", "z"):
        await c.js(f"(() => {{ const l = [...document.querySelectorAll('.checks label')].find(x => x.innerText.trim() === '{a}'); "
                   "l.querySelector('input').click(); return true; })()")
        await asyncio.sleep(0.2)

    # keep-in plate under the load (K on x+), keep-out hole through the middle (O on z+, then a cylinder)
    await c.key("k")
    await c.click_face("x+", (0.5, 0.5))
    for f, v in (("min x", 92), ("min y", 15), ("max y", 35), ("min z", 10), ("max z", 40)):
        await c.type_into(f'[aria-label="{f}"]', str(v))
    await c.click_sel(".tree-item", "Domain")
    await c.type_into('[aria-label="volume %"]', "20")
    await c.key("o")
    await c.click_face("z+", (0.5, 0.5))
    await c.choose('select[aria-label="region kind"]', "cylinder")
    await asyncio.sleep(0.4)
    await c.type_into('[aria-label="radius"]', "8")
    await c.type_into('[aria-label="centre a"]', "40")
    await c.js("window.__agentto.view([0.45, 0.4, 1.0]); true")
    await asyncio.sleep(1.5)
    await shot(c, "05_keepin_keepout")


async def solve(c: CDP):
    """The ladder on the design from `build`: sketch, preview, refine, then Optimize release, surface, export."""
    await fresh(c)
    await ready(c)
    if not await c.js(f"{ST}.spec?.passive?.length === 2"):
        await build(c)
    await c.until(f"document.querySelectorAll('.ladder .rung').length === 4 && !{RUNG % 0}.disabled", 30, what="rungs")
    await c.js(f"{ST}.select(null); true")
    await c.js("window.__agentto.view([0.45, 0.4, 1.0]); true")
    await c.click_sel(".ladder .rung", "Sketch")
    await rung_done(c, 0, 120)
    await asyncio.sleep(1.0)
    await shot(c, "07_sketch")
    await c.click_sel(".ladder .rung", "Preview")
    await rung_done(c, 1, 300)
    await c.click_sel(".ladder .rung", "Refine")
    await c.until(f"/it \\d+/.test({RUNG % 2}.innerText)", 120, 0.3, what="refine running")
    await asyncio.sleep(2.0)
    await shot(c, "08_refine_running")
    await rung_done(c, 2, 600)
    await asyncio.sleep(1.0)
    await c.click_sel(".ladder button", "Optimize release")
    await rung_done(c, 3, 1200)
    await c.until(f"!!{ST}.surface", 120, what="surface")
    await asyncio.sleep(2.0)
    await shot(c, "09_release_surface")
    await rects(c)
    await shot(c, "10_ladder", ".bottom", pad=0)
    await shot(c, "11_result_card", "[aria-label='Latest result']")
    await c.click_sel(".result-actions button", "Voxels")
    await asyncio.sleep(1.5)
    await shot(c, "12_voxels", ".view", pad=0)
    await c.click_sel(".result-actions button", "Surface")
    await c.until(f"!!{ST}.surface", 60, what="surface back")
    await c.click_sel(".result-actions button", "Export STL")
    await c.until("!!document.querySelector('[aria-label=\"Latest result\"] .files a[href$=\".stl\"]')", 300, what="export")
    await asyncio.sleep(0.5)
    await shot(c, "13_export", "[aria-label='Latest result']")
    await c.click_sel(".tree-item", "Domain")
    await c.type_into('[aria-label="volume %"]', "25")
    await c.until("!!document.querySelector('[aria-label=\"Latest result\"] .badge.err')", 10, what="stale")
    await asyncio.sleep(0.5)
    await shot(c, "14_stale", "[aria-label='Latest result']")
    await c.click(800, 160)
    await c.key("z", mods=2)
    await c.until("!document.querySelector('[aria-label=\"Latest result\"] .badge.err')", 10, what="undo")
    await asyncio.sleep(0.5)
    await shot(c, "15_timeline", ".timeline", pad=2)


async def overlays(c: CDP):
    await c.click(800, 160)
    await c.key("?", code="Slash", vk=191, mods=8, text="?")
    await c.until("!!document.querySelector('.modal')", 5)
    await shot(c, "16_keymap", ".modal .sheet")
    await c.key("Escape", code="Escape", vk=27, text="")
    await c.key("k", mods=2)
    await c.until("!!document.querySelector('.palette input')", 5)
    await asyncio.sleep(0.3)
    await shot(c, "17_palette", ".modal .sheet")
    await c.key("Escape", code="Escape", vk=27, text="")
    await c.click_sel('button[aria-label="Toggle theme"]')
    await asyncio.sleep(1.0)
    await shot(c, "18_dark")
    await c.click_sel('button[aria-label="Toggle theme"]')
    await fresh(c, tips=True)
    await asyncio.sleep(1.5)
    await shot(c, "01_first_run")
    await c.js("(() => { const b = [...document.querySelectorAll('.tips button')].find(x => /got it/i.test(x.innerText)); "
               "b && b.click(); return true; })()")


async def send(c: CDP, text):
    await c.js("document.querySelector('textarea[aria-label=\"Message the assistant\"]').focus(); true")
    await c.call("Input.insertText", text=text)
    await c.key("Enter", code="Enter", vk=13, text="\r")
    await c.until("!!document.querySelector('.working')", 20, what="run started")


async def assistant(c: CDP):
    """Propose on the ambiguous b44 brief (question with consequence previews, then cards), then one Act run."""
    brief = (ROOT / "briefs" / "b44_gearbox_bracket.md").read_text(encoding="utf-8")
    await fresh(c)
    await ready(c)
    await new_design(c)
    await c.js("(() => { const b = document.querySelector('.result-toggle'); "
               "if (b && b.getAttribute('aria-expanded') === 'true') b.click(); return true; })()")
    await c.click_sel('.dial button', "Propose")
    await send(c, "Set up this part from the brief. Build a complete draft first; then, for anything the brief leaves "
                  "open, ask me with each reading as a concrete option.\n\n" + brief)
    await c.until("document.querySelectorAll('[aria-label=\"Clarification question\"] .clar-chip').length > 0", 900, 2.0,
                  what="question")
    await c.until("[...document.querySelectorAll('[aria-label=\"Clarification question\"] .clar-chip img')].length > 0", 300, 1.0,
                  what="previews")
    await asyncio.sleep(1.5)
    await shot(c, "20_clarify")
    await shot(c, "21_clarify_chips", "[aria-label='Clarification question']")
    for _ in range(6):                                   # this and any further question: take the first reading
        qid = await c.js(f"{ST}.clarify?.qid ?? null")
        if qid is None:
            break
        await c.js("(() => { const b = document.querySelector('[aria-label=\"Clarification question\"] .clar-chip'); "
                   "b && b.click(); return true; })()")
        await c.until(f"{ST}.clarify?.qid !== {json.dumps(qid)}", 30, 0.5, what="answer taken")
        await c.until(f"!!{ST}.clarify || !document.querySelector('.working')", 900, 2.0, what="next question or end")
    await c.until("!document.querySelector('.working')", 900, 2.0, what="run end")
    await asyncio.sleep(1.0)
    xy = await c.js("(() => { const cs = [...document.querySelectorAll('[aria-label=\"Proposed changes\"] .card')]; "
                    "const k = cs.find(x => /load/i.test(x.innerText)) || cs[0]; if (!k) return null; k.scrollIntoView({block: 'center'}); "
                    "const r = k.getBoundingClientRect(); return [r.left + 30, r.top + r.height / 2]; })()")
    if xy:
        await c.call("Input.dispatchMouseEvent", type="mouseMoved", x=xy[0], y=xy[1])
        await asyncio.sleep(1.0)
    await shot(c, "22_cards")
    await c.call("Input.dispatchMouseEvent", type="mouseMoved", x=800, y=500)
    await c.click_sel("[aria-label='Proposed changes'] .card-actions button", "Accept all")
    await asyncio.sleep(1.5)
    await shot(c, "23_accepted")

    await c.click_sel('.dial button', "Act")
    await send(c, "Lower the volume fraction by 0.05, then run a sketch solve to check that the design still carries the load.")
    await c.until("!document.querySelector('.working')", 900, 2.0, what="act run end")
    await asyncio.sleep(2.0)
    await shot(c, "24_act")
    await shot(c, "25_act_panel", ".assistant-panel", pad=0)


async def attach(c: CDP):
    """Claude Code, a second subscription CLI, drives the same live session over MCP (--attach); the GUI shows it."""
    sys.path.insert(0, str(ROOT / "src"))
    from agentto.llm.cli import mcp_server_spec, run_claude
    await c.js(f"{ST}.select(null); true")
    work = Path(tempfile.mkdtemp(prefix="agentto_claude_"))
    prompt = ("You are connected to a live agentto design session through the agentto tools. Call get_spec, then add a "
              "second load with id load_claude: 300 N in the -z direction on a 10 x 10 patch centred on the top face "
              "(z+). Validate the spec with validate_spec and fix anything it reports. Finish with one sentence saying "
              "what you changed.")
    spec = mcp_server_spec(attach="http://127.0.0.1:8765", profile="full",
                           python=sys.executable)
    res = await asyncio.to_thread(run_claude, prompt, work, spec, None, 600, None, "high")
    print("claude:", res.returncode, res.final[:300], flush=True)
    (IMG / "attach_claude.json").write_text(json.dumps({"final": res.final, "model": res.model, "cli": res.cli_version,
                                                        "wall_s": res.wall_s}, indent=1), encoding="utf-8")
    await c.until(f"{ST}.spec.loads.some(l => l.id === 'load_claude')", 60, what="claude edit visible")
    await c.click_sel(".tree-item", "load_claude")
    await asyncio.sleep(1.5)
    await shot(c, "26_attach")
    await shot(c, "27_attach_timeline", ".timeline", pad=2)


async def main(what):
    IMG.mkdir(parents=True, exist_ok=True)
    prof = tempfile.mkdtemp(prefix="agentto_chrome_")
    port = 9335
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={prof}",
                             f"--window-size={W},{H}", "--hide-scrollbars", f"--force-device-scale-factor={DPR}",
                             "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json").read())
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:                       # noqa: BLE001
                await asyncio.sleep(0.2)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
            c = CDP(ws)
            await c.call("Page.enable")
            await c.call("Runtime.enable")
            await c.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=DPR, mobile=False)
            await c.call("Page.navigate", url=URL)
            await asyncio.sleep(2)
            for name, fn in (("build", build), ("solve", solve), ("overlays", overlays), ("assistant", assistant),
                             ("attach", attach)):
                if name in what:
                    print(f"== {name}", flush=True)
                    await fn(c)
            if c.console:
                print("console:", c.console[:5], flush=True)
    finally:
        proc.terminate()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main(sys.argv[1:] or ["build", "solve", "overlays", "assistant", "attach"]))
