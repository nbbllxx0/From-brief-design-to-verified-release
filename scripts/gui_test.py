"""End-to-end GUI test: headless Chrome over the DevTools protocol, real mouse and keyboard input, small meshes.

Needs the backend started with a low ladder cap and the Vite front end:
    python -B scripts/serve.py --ladder-max 1.2e7        (rungs 16k / 128k / 1M / 8M on a 100 x 50 x 50 box)
    node web/node_modules/vite/bin/vite.js web
    python -B scripts/gui_test.py [setup] [ladder] [overlays] [assistant]
The page is opened with ?test, which exposes window.__agentto (the store, a camera turn and a spec-point -> screen
projection) so the script can click faces of the domain the way a user does. `assistant` runs two Codex sessions on
the subscription. Writes results/gui_test/<step>.png and results/gui_test/report_<groups>.json.
"""
import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "gui_test"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
URL = "http://localhost:5173/?test"
W, H = 1680, 1000
RESULTS: list[dict] = []
ST = "window.__agentto.store.getState()"


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "ok": bool(ok), "detail": str(detail)[:400]})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""), flush=True)
    return ok


class CDP:
    def __init__(self, ws):
        self.ws, self.n, self.console = ws, 0, []
        self.waiters: dict[int, asyncio.Future] = {}
        self.reader = asyncio.create_task(self._read())

    async def _read(self):
        async for raw in self.ws:
            m = json.loads(raw)
            if "id" in m and m["id"] in self.waiters:
                self.waiters.pop(m["id"]).set_result(m)
            elif m.get("method") == "Runtime.exceptionThrown":
                self.console.append(("exception", m["params"]["exceptionDetails"].get("exception", {}).get("description", "")))
            elif m.get("method") == "Runtime.consoleAPICalled" and m["params"]["type"] in ("error", "assert"):
                self.console.append(("error", " ".join(str(a.get("value", a.get("description", ""))) for a in m["params"]["args"])))

    async def call(self, method, **params):
        self.n += 1
        fut = asyncio.get_running_loop().create_future()
        self.waiters[self.n] = fut
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        m = await fut
        if "error" in m:
            raise RuntimeError(f"{method}: {m['error']}")
        return m.get("result", {})

    async def js(self, expr, timeout=60):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True, timeout=timeout * 1000)
        if r.get("exceptionDetails"):
            raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description", r["exceptionDetails"]))
        return r.get("result", {}).get("value")

    async def until(self, expr, timeout=120, every=0.5, what=""):
        t = time.time()
        while time.time() - t < timeout:
            if await self.js(expr):
                return True
            await asyncio.sleep(every)
        raise TimeoutError(what or expr)

    async def soon(self, expr, timeout=10.0):
        """True once `expr` holds within `timeout` s (no exception)."""
        try:
            return await self.until(expr, timeout, 0.25)
        except TimeoutError:
            return False

    async def shot(self, name):
        r = await self.call("Page.captureScreenshot", format="png")
        (OUT / f"{name}.png").write_bytes(base64.b64decode(r["data"]))

    async def click(self, x, y):
        await self.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
        await self.call("Input.dispatchMouseEvent", type="mousePressed", x=x, y=y, button="left", clickCount=1)
        await self.call("Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y, button="left", clickCount=1)
        await asyncio.sleep(0.25)

    async def click_sel(self, sel, text=None):
        """Real click at the centre of the first element matching `sel` (and containing `text`)."""
        xy = await self.js(f"""(() => {{ const e = [...document.querySelectorAll({json.dumps(sel)})]
          .find(x => {json.dumps(text)} === null || x.innerText.trim().startsWith({json.dumps(text)}));
          if (!e) return null; e.scrollIntoView({{block: 'nearest'}}); const r = e.getBoundingClientRect();
          return [r.left + r.width / 2, r.top + r.height / 2]; }})()""")
        if not xy:
            return False
        await self.click(*xy)
        return True

    async def key(self, key, code=None, vk=None, mods=0, text=None):
        code = code or (f"Key{key.upper()}" if len(key) == 1 and key.isalpha() else key)
        vk = vk or (ord(key.upper()) if len(key) == 1 else 0)
        base = dict(key=key, code=code, windowsVirtualKeyCode=vk, nativeVirtualKeyCode=vk, modifiers=mods)
        t = text if text is not None else (key if len(key) == 1 and not mods & 2 else "")
        await self.call("Input.dispatchKeyEvent", type="keyDown" if t else "rawKeyDown", text=t, **base)
        await self.call("Input.dispatchKeyEvent", type="keyUp", **base)
        await asyncio.sleep(0.15)

    async def type_into(self, sel, value):
        """Focus a field, select its text and type `value` as a user would, then Tab away."""
        ok = await self.js(f"""(() => {{ const e = document.querySelector({json.dumps(sel)}); if (!e) return false;
          e.scrollIntoView({{block: 'nearest'}}); e.focus(); e.select?.(); return true; }})()""")
        if not ok:
            return False
        await self.call("Input.insertText", text=str(value))
        await asyncio.sleep(0.1)
        await self.js("document.activeElement?.blur(); true")
        await asyncio.sleep(0.1)
        return True

    async def choose(self, sel, value):
        return await self.js(f"""(() => {{ const s = document.querySelector({json.dumps(sel)}); if (!s) return false;
          const set = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set;
          set.call(s, {json.dumps(value)}); s.dispatchEvent(new Event('change', {{bubbles: true}})); return true; }})()""")

    async def spec(self):
        return await self.js(f"{ST}.spec")

    async def click_face(self, face, uv=(0.5, 0.5)):
        """Turn the camera towards `face` and click the face at in-plane fraction uv (real pointer events)."""
        L = (await self.spec())["domain"]["size"]
        a = "xyz".index(face[0])
        n = [0.0, 0.0, 0.0]
        n[a] = -1.0 if face[1] == "-" else 1.0
        d = [n[i] if i == a else (0.45 if i == 1 else 0.35) * (1 if face != "y-" else -1) for i in range(3)]
        if face == "y-":
            d = [0.35, -1.0, 0.45]
        await self.js(f"window.__agentto.view({json.dumps(d)}); true")
        await asyncio.sleep(0.4)
        p = [0.0, 0.0, 0.0]
        o = [i for i in range(3) if i != a]
        p[a] = 0.0 if face[1] == "-" else L[a]
        p[o[0]], p[o[1]] = uv[0] * L[o[0]], uv[1] * L[o[1]]
        xy = await self.js(f"window.__agentto.project({json.dumps(p)})")
        await self.click(*xy)
        return p


async def fresh(c: CDP, tips=False):
    await c.js("localStorage.setItem('agentto.theme','light');" + ("localStorage.removeItem('agentto.tips.v1');" if tips
               else "localStorage.setItem('agentto.tips.v1','done');") + "true")
    await c.call("Page.navigate", url=URL)
    await c.until("!!window.__agentto && document.querySelectorAll('.picker option').length > 3", 60, what="app up")
    await c.until(f"{ST}.conn === 'open'", 30, what="socket")


async def new_design(c: CDP):
    await c.choose('select[aria-label="Problem"]', "new:")
    await c.until(f"{ST}.spec?.name === 'untitled'", 10, what="blank design")
    await asyncio.sleep(0.8)


# ------------------------------------------------------------------------------------------------------------
async def t_setup(c: CDP):
    await fresh(c)
    await new_design(c)
    s = await c.spec()
    check("new design: blank 100x50x50 box, no BCs", s["domain"]["size"] == [100, 50, 50] and not s["supports"] and not s["loads"])
    await asyncio.sleep(0.8)
    txt = await c.js("document.querySelector('.findings')?.innerText ?? ''")
    check("checks: missing support and load are explained", "No support yet" in txt and "No load yet" in txt, txt)
    check("ladder: no rungs while the spec is incomplete",
          await c.js("document.querySelectorAll('.ladder .rung').length") == 0)
    check("inspector shows the domain on a new design", await c.js("!!document.querySelector('[aria-label=\"size x\"]')"))

    # support on x- with the S key and a real click on the face
    await c.click(840, 500)                                   # focus the page (viewport), no tool yet: selects domain
    await c.key("s")
    check("S key arms the support tool", await c.js(f"{ST}.tool") == "support")
    check("toolbar hint while a tool is armed", "Click a face" in (await c.js("document.querySelector('.toolbar').innerText")))
    await c.click_face("x-")
    s = await c.spec()
    check("click on x- adds a fixed face support", len(s["supports"]) == 1 and s["supports"][0]["region"] == {"kind": "face", "face": "x-"}
          and s["supports"][0]["fix"] == ["x", "y", "z"], s["supports"])
    check("tool returns to select after placing", await c.js(f"{ST}.tool") == "select")
    check("new support is selected in the inspector", await c.js(f"{ST}.selection?.id") == "support_1")
    await c.shot("01_support")

    # load on x+ with the toolbar button, at a point off-centre
    await c.click_sel(".toolbar button", "Load")
    p = await c.click_face("x+", (0.3, 0.7))
    s = await c.spec()
    ld = s["loads"][0] if s["loads"] else {}
    ok = ld.get("region", {}).get("face") == "x+" and ld.get("force") == [-1000, 0, 0]
    cen = ld.get("region", {}).get("center", [0, 0])
    check("click on x+ adds a 1 kN patch load into the face", ok, ld)
    check("load patch is centred where the face was clicked", abs(cen[0] - p[1]) < 3 and abs(cen[1] - p[2]) < 3, (cen, p))

    # inspector: retype the force as -y; the field keeps focus semantics (type, Tab away)
    await c.type_into('[aria-label="force x"]', "0")
    await c.type_into('[aria-label="force y"]', "-2000")
    s = await c.spec()
    check("inspector: force edited in place", s["loads"][0]["force"] == [0, -2000, 0], s["loads"][0]["force"])
    await asyncio.sleep(1.0)
    check("valid spec once a support and a load exist", await c.soon("/valid spec|valid ·/.test(document.querySelector('.topbar').innerText)"),
          await c.js("document.querySelector('.topbar').innerText"))
    check("ladder: four rungs on the small ladder", await c.soon("document.querySelectorAll('.ladder .rung').length === 4"),
          await c.js("document.querySelectorAll('.ladder .rung').length"))

    # keep-in (K) on y+ and keep-out (O) on z+
    await c.key("k")
    await c.click_face("y+")
    await c.key("o")
    await c.click_face("z+")
    s = await c.spec()
    kinds = [(x["kind"], x["region"]["min"], x["region"]["max"]) for x in s["passive"]]
    check("K and O add a keep-in slab on y+ and a keep-out slab on z+", len(kinds) == 2 and kinds[0][0] == "solid"
          and kinds[0][1][1] == 45 and kinds[1][0] == "void" and kinds[1][1][2] == 45, kinds)
    tree = await c.js("document.querySelector('.tree').innerText")
    check("intent tree lists every item", all(k in tree for k in ("support_1", "load_1", "keep_in_1", "keep_out_1")), tree)
    await asyncio.sleep(1.0)
    check("timeline: the newest record is in view (not clipped)", await c.js(
        "(() => { const t = document.querySelector('.timeline').getBoundingClientRect(); const it = [...document.querySelectorAll('.tl-item')]; "
        "const r = it[it.length - 1].getBoundingClientRect(); return r.right <= t.right + 1 && /keep_out_1/.test(it[it.length - 1].innerText); })()"),
        await c.js("[...document.querySelectorAll('.tl-item')].slice(-2).map(x => x.innerText)"))
    check("composer keeps its width with a selection chip", await c.js(
        "document.querySelector('textarea[aria-label=\"Message the assistant\"]').getBoundingClientRect().width > 200"))
    fx = await c.js("document.querySelector('.findings').innerText")
    check("keep-in and keep-out meeting at the y+/z+ edge are flagged (R007)", "R007" in fx, fx)
    await c.shot("02_bcs")

    # pick an item in the viewport (select tool): click the support face
    await c.js(f"{ST}.select(null); true")
    await c.click_face("x-", (0.5, 0.5))
    check("viewport pick selects the support under the pointer", await c.js(f"{ST}.selection?.id") == "support_1",
          await c.js(f"JSON.stringify({ST}.selection)"))
    # tree pick
    await c.click_sel(".tree-item", "load_1")
    check("tree pick selects the load", await c.js(f"{ST}.selection?.id") == "load_1")

    # support DOF checkboxes: free z (a roller)
    await c.click_sel(".tree-item", "support_1")
    await c.js("(() => { const l = [...document.querySelectorAll('.checks label')].find(x => x.innerText.trim() === 'z'); l.querySelector('input').click(); return true; })()")
    await asyncio.sleep(0.3)
    check("support DOF checkbox frees z", (await c.spec())["supports"][0]["fix"] == ["x", "y"])
    await c.js("(() => { const l = [...document.querySelectorAll('.checks label')].find(x => x.innerText.trim() === 'z'); l.querySelector('input').click(); return true; })()")
    await asyncio.sleep(0.3)

    # region kind conversion: support face -> patch
    has_kind = await c.js("!!document.querySelector('select[aria-label=\"region kind\"]')")
    if has_kind:
        await c.choose('select[aria-label="region kind"]', "patch")
        await asyncio.sleep(0.3)
        r = (await c.spec())["supports"][0]["region"]
        check("region kind converts face -> patch on the same face", r["kind"] == "patch" and r["face"] == "x-", r)
        await c.choose('select[aria-label="region kind"]', "face")
        await asyncio.sleep(0.3)
    else:
        check("inspector offers a region-kind picker", False, "no select[aria-label=region kind]")

    # domain edits
    await c.click_sel(".tree-item", "Domain")
    await c.type_into('[aria-label="size x"]', "120")
    await c.type_into('[aria-label="volume %"]', "25")
    s = await c.spec()
    check("domain size and volume edited", s["domain"]["size"][0] == 120 and abs(s["volume_fraction"] - 0.25) < 1e-9,
          (s["domain"]["size"], s["volume_fraction"]))
    await asyncio.sleep(1.0)
    rung0 = await c.js("[...document.querySelectorAll('.ladder .rung')].map(b => b.title.split(' ')[0])")
    check("ladder follows the new domain (120 mm long)", bool(rung0) and rung0[0].startswith("48"), rung0)

    # delete an item: clears the overlap
    await c.click_sel(".tree-item", "keep_out_1")
    await c.click_sel(".linkbtn.danger", "Delete")
    await asyncio.sleep(0.5)
    check("Delete removes the selected keep-out", len((await c.spec())["passive"]) == 1)

    check("overlap error clears with the keep-out gone", await c.soon("!/R007/.test(document.querySelector('.findings').innerText)", 5))

    # validation feedback: an impossible volume
    await c.click_sel(".tree-item", "Domain")
    await c.type_into('[aria-label="volume %"]', "150")
    await asyncio.sleep(1.0)
    fx = await c.js("document.querySelector('.findings').innerText")
    check("volume 150% is flagged", "error" in fx.lower() or "volume" in fx.lower(), fx)
    check("ladder offers no runnable rung while an error stands",
          await c.js("[...document.querySelectorAll('.ladder .rung')].every(b => b.disabled)"))
    await c.type_into('[aria-label="volume %"]', "25")
    await asyncio.sleep(1.0)
    check("error clears when fixed", await c.soon("[...document.querySelectorAll('.ladder .rung')].some(b => !b.disabled)"),
          [await c.js(f"{ST}.spec.volume_fraction"), await c.js("document.querySelector('.findings').innerText"),
           await c.js("document.querySelectorAll('.ladder .rung').length")])

    # mesh-by switch
    await c.choose('select[aria-label="mesh by"]', "h")
    await asyncio.sleep(0.3)
    r = (await c.spec())["resolution"]
    check("mesh-by switches cells -> h", "h" in r and r["h"] > 0, r)
    await c.choose('select[aria-label="mesh by"]', "cells")
    await asyncio.sleep(0.3)

    # undo / redo through the session (Ctrl+Z, Ctrl+Y)
    await asyncio.sleep(0.8)                                   # let the 300 ms sync commit the last edit
    before = (await c.spec())["volume_fraction"]
    await c.click_sel(".tree-item", "Domain")
    await c.type_into('[aria-label="volume %"]', "40")
    await asyncio.sleep(0.9)
    await c.click(840, 160)                                    # a neutral click; keys go to the page, not a field
    await c.key("z", mods=2)
    ok = await c.soon(f"Math.abs({ST}.spec.volume_fraction - {before}) < 1e-9", 5)
    check("Ctrl+Z undoes the last edit", ok, await c.js(f"{ST}.spec.volume_fraction"))
    await c.key("y", mods=2)
    ok = await c.soon(f"Math.abs({ST}.spec.volume_fraction - 0.4) < 1e-9", 5)
    check("Ctrl+Y redoes it", ok, await c.js(f"{ST}.spec.volume_fraction"))
    await c.key("z", mods=2, code="KeyZ")
    await c.soon(f"Math.abs({ST}.spec.volume_fraction - {before}) < 1e-9", 5)

    # timeline shows the edits as labelled user records
    await asyncio.sleep(0.8)
    tl = await c.js("[...document.querySelectorAll('.tl-item')].map(x => x.innerText)")
    check("timeline logs labelled user edits", any("support_1" in x for x in tl) and any("delete keep_out_1" in x for x in tl), tl)
    await c.shot("03_setup_done")


# ------------------------------------------------------------------------------------------------------------
RUNG = "[...document.querySelectorAll('.ladder .rung')][%d]"


async def solve_small(c: CDP):
    """b-like cantilever on the blank box: support x-, 1 kN down at the free end."""
    await new_design(c)
    await c.js(f"""{ST}.editSpec(s => ({{...s, supports: [{{id: 'support_1', region: {{kind: 'face', face: 'x-'}}, fix: ['x','y','z']}}],
      loads: [{{id: 'load_1', force: [0, -1000, 0], region: {{kind: 'patch', face: 'x+', center: [25, 25], size: [10, 50], shape: 'rect'}}}}]}}), 'test setup'); true""")
    await c.until("document.querySelectorAll('.ladder .rung').length === 4 && ![...document.querySelectorAll('.ladder .rung')][0].disabled", 30, what="rungs")


async def t_ladder(c: CDP):
    await fresh(c)
    await c.until("[...document.querySelectorAll('.pill')].some(p => /GPU ready/.test(p.innerText))", 240, what="GPU ready")
    await solve_small(c)
    note = await c.js("document.querySelector('.ladder .note')?.innerText ?? ''")
    check("release note shown before the first solve", "Release = the finest rung" in note, note)
    names = await c.js("[...document.querySelectorAll('.ladder .rung')].map(b => b.innerText.split('\\n')[0])")
    check("rungs are Sketch / Preview / Refine / Release", [n.strip() for n in names][:4] == ["Sketch", "Preview", "Refine", "Release"], names)

    t0 = time.time()
    await c.click_sel(".ladder .rung", "Sketch")
    ok = await c.soon(f"/\\d+\\.\\ds/.test({RUNG % 0}.innerText)", 60)
    check("sketch (16k) solves from the ladder button", ok, await c.js(f"{RUNG % 0}.innerText"))
    check("sketch wall time under 10 s", time.time() - t0 < 10, f"{time.time() - t0:.1f} s")
    check("density frames stream to the viewport", await c.js(f"!!{ST}.frame"))
    check("result card appears with a certificate list", await c.soon("!!document.querySelector('[aria-label=\"Latest result\"] .checklist li')"))
    await c.shot("10_sketch")

    for i, name in ((1, "Preview"), (2, "Refine")):
        await c.click_sel(".ladder .rung", name)
        ok = await c.soon(f"/\\d+\\.\\ds/.test({RUNG % i}.innerText)", 300)
        txt = await c.js(f"{RUNG % i}.innerText")
        check(f"{name.lower()} solves and warm-starts (↑)", ok and "↑" in txt, txt)
    check("Optimize release appears once Refine has converged", await c.soon("[...document.querySelectorAll('.ladder button')].some(b => b.innerText === 'Optimize release')"))

    # Stop: start the 8M release and cancel it while it runs
    await c.click_sel(".ladder .rung", "Release")
    running = await c.soon("[...document.querySelectorAll('.ladder button')].some(b => b.innerText === 'Stop')", 30)
    check("Stop button while a solve runs", running)
    setup_txt = await c.js(f"{RUNG % 3}.innerText")
    check("release rung shows queued / setting up / it N while running", any(k in setup_txt for k in ("queued", "setting up", "it ")), setup_txt)
    await asyncio.sleep(1.0)
    clicked = await c.click_sel(".ladder button", "Stop")
    check("Stop clicked while the release still ran", clicked)
    ok = await c.soon(f"(() => {{ const j = Object.values({ST}.jobs).find(j => j.h === {ST}.rungs[3].h); return j && j.status === 'cancelled'; }})()", 60)
    check("Stop cancels the running release", ok, await c.js(f"JSON.stringify(Object.values({ST}.jobs).map(j => [j.h, j.status]))"))
    check("Stop button gone after cancel", await c.soon("![...document.querySelectorAll('.ladder button')].some(b => b.innerText === 'Stop')", 10))

    # full release with Optimize release (reopen_long), certified
    t0 = time.time()
    await c.click_sel(".ladder button", "Optimize release")
    ok = await c.soon(f"/\\d+\\.\\ds/.test({RUNG % 3}.innerText)", 900)
    txt = await c.js(f"{RUNG % 3}.innerText")
    check("Optimize release (8M) converges", ok, txt)
    check("release is certified (seal on the rung)", "certified" in txt, txt)
    rc = await c.js("document.querySelector('[aria-label=\"Latest result\"]').innerText")
    check("result card: certified, release rung, all checks ticked", "certified" in rc and "✗" not in rc and "Only a certified Release" not in rc, rc)
    print(f"   release wall {time.time() - t0:.0f} s", flush=True)
    # marching-cubes surface: shown on its own for a certified release; Voxels / Surface toggle it
    ok = await c.soon(f"!!{ST}.surface", 60)
    tri = await c.js(f"{ST}.surface?.triangles ?? 0")
    check("certified release appears as its marching-cubes surface", ok and tri > 1000, tri)
    check("surface note reports triangles and pooling", await c.soon("/surface: [\d,]+ triangles/.test(document.querySelector('[aria-label=\"Latest result\"]').innerText)"))
    await asyncio.sleep(1)
    await c.shot("12_release_surface")
    await c.click_sel(".result-actions button", "Voxels")
    check("Voxels shows the density voxels", await c.soon(f"{ST}.surface === null && !!{ST}.frame"))
    await asyncio.sleep(0.5)
    await c.shot("11_release_voxels")
    await c.click_sel(".result-actions button", "Surface")
    check("Surface brings the isosurface back", await c.soon(f"!!{ST}.surface", 60))

    # export STL + report, and the links serve the files
    await c.click_sel(".result-actions button", "Export STL")
    ok = await c.soon("!!document.querySelector('[aria-label=\"Latest result\"] .files a[href$=\".stl\"]')", 180)
    check("Export writes an STL and a report", ok)
    if ok:
        links = await c.js("[...document.querySelectorAll('[aria-label=\"Latest result\"] .files a')].map(a => a.getAttribute('href'))")
        codes = []
        for href in links:
            try:
                with urllib.request.urlopen("http://localhost:5173" + href) as r:
                    body = r.read()
                    codes.append((href, r.status, len(body)))
            except Exception as e:                  # noqa: BLE001
                codes.append((href, str(e), 0))
        check("STL / markdown / json links download (200, non-empty)", all(x[1] == 200 and x[2] > 100 for x in codes), codes)
        stl = next((x for x in codes if x[0].endswith(".stl")), None)
        if stl:
            with urllib.request.urlopen("http://localhost:5173" + stl[0]) as r:
                head = r.read(84)
            n = int.from_bytes(head[80:84], "little")
            check("STL is binary with the triangle count in its header", stl[2] == 84 + 50 * n, (stl[2], n))
    await c.shot("13_export")

    # stale badge after an edit; undo clears it
    await c.click_sel(".tree-item", "Domain")
    await c.type_into('[aria-label="volume %"]', "28")
    ok = await c.soon("!!document.querySelector('[aria-label=\"Latest result\"] .badge.err')", 5)
    check("result card turns stale after an edit", ok)
    await c.click(840, 160)
    await c.key("z", mods=2)
    check("undo restores the solved spec: stale badge clears",
          await c.soon("!document.querySelector('[aria-label=\"Latest result\"] .badge.err')", 5))
    check("rung states survive the undo (same spec hash)", await c.soon(f"/\\d+\\.\\ds/.test({RUNG % 0}.innerText)", 5),
          await c.js(f"{RUNG % 0}.innerText"))
    await c.shot("14_after_undo")


# ------------------------------------------------------------------------------------------------------------
async def t_overlays(c: CDP):
    await fresh(c, tips=True)
    check("first-run tips shown on a fresh browser", await c.soon("!!document.querySelector('.tips')"))
    await c.shot("20_tips")
    await c.js("(() => { const b = [...document.querySelectorAll('.tips button')].find(x => /got it|dismiss|×/i.test(x.innerText || x.getAttribute('aria-label') || '')); b && b.click(); return !!b; })()")
    await asyncio.sleep(0.3)
    gone = await c.js("!document.querySelector('.tips')")
    check("tips dismiss", gone)
    await c.call("Page.reload")
    await c.until("!!window.__agentto", 30)
    await asyncio.sleep(1)
    check("tips stay dismissed after a reload", await c.js("!document.querySelector('.tips')"))

    await c.click(840, 160)
    await c.key("?", code="Slash", vk=191, mods=8, text="?")
    check("? opens the keyboard map", await c.soon("!!document.querySelector('.modal')", 3))
    await c.shot("21_keymap")
    await c.key("Escape", code="Escape", vk=27, text="")
    check("Esc closes it", await c.soon("!document.querySelector('.modal')", 3))
    check("legend overlay is not a dim layer (regression)", await c.js(
        "(() => { const o = document.querySelector('.view .overlay'); const r = o.getBoundingClientRect(); return r.height < 80; })()"))

    await c.key("k", mods=2)
    check("Ctrl+K opens the command palette", await c.soon("!!document.querySelector('.palette input')", 3))
    focused = await c.js("document.activeElement === document.querySelector('.palette input')")
    check("palette input has focus", focused)
    await c.call("Input.insertText", text="theme")
    await asyncio.sleep(0.3)
    items = await c.js("[...document.querySelectorAll('.palette li, .palette button')].map(x => x.innerText)")
    check("palette filters commands", 0 < len(items) <= 3 and all("heme" in x for x in items), items)
    theme0 = await c.js(f"{ST}.theme")
    await c.key("Enter", code="Enter", vk=13, text="\r")
    check("palette runs the command (theme toggles)", await c.soon(f"{ST}.theme !== {json.dumps(theme0)}", 3))
    check("palette closes after running", await c.soon("!document.querySelector('.palette')", 3))
    await c.shot("22_dark")
    await c.click_sel('button[aria-label="Toggle theme"]')
    check("theme button toggles back", await c.soon(f"{ST}.theme === {json.dumps(theme0)}", 3))
    check("theme persists in localStorage", await c.js(f"localStorage.getItem('agentto.theme') === {json.dumps(theme0)}"))

    # tool keys must not fire while typing in a field
    await c.click_sel(".tree-item", "Material")
    await c.js("document.querySelector('[aria-label=\"material name\"]').focus(); true")
    await c.key("s")
    check("letter keys in a text field do not arm tools", await c.js(f"{ST}.tool") == "select")
    await c.js("document.activeElement.blur(); true")

    # problem picker: a suite task (spec + brief) and a library problem
    await c.choose('select[aria-label="Problem"]', "task:b06_motor_mount")
    ok = await c.soon(f"{ST}.source === 'task:b06_motor_mount' && !!{ST}.brief", 10)
    check("picker loads a suite task with its brief", ok and await c.soon("!!document.querySelector('.chat .brief')", 5))
    check("a loaded task validates", await c.soon("/valid spec|valid ·/.test(document.querySelector('.topbar').innerText)", 10),
          await c.js("document.querySelector('.topbar').innerText"))
    lib = await c.js("[...document.querySelectorAll('select[aria-label=\"Problem\"] option')].map(o => o.value).find(v => v.startsWith('lib:'))")
    if lib:
        await c.choose('select[aria-label="Problem"]', lib)
        check(f"picker loads a library problem ({lib})", await c.soon(f"{ST}.source === {json.dumps(lib)} && {ST}.brief === null", 10))
    await c.shot("23_task_loaded")

    # a proposal that resizes the box: its ghosts sit on the proposed box, outlined (store-injected run, no agent)
    await new_design(c)
    await c.js(f"""(() => {{ const st = {ST}; const p = structuredClone(st.spec); p.domain.size = [130, 50, 70];
      p.supports = [{{id: 'support_1', region: {{kind: 'face', face: 'x+'}}, fix: ['x','y','z']}}];
      st.set({{agentRun: {{id: 'fake', mode: 'propose', running: false, t0: Date.now(), steps: [], messages: [], proposal: p, decided: {{}},
        cards: [{{id: 'c1', op: 'set', field: 'domain', before: st.spec.domain, after: p.domain, text: 'set domain to 130 x 50 x 70'}},
                {{id: 'c2', op: 'add', list: 'supports', item_id: 'support_1', after: p.supports[0], text: 'add support support_1'}}]}},
        log: [...st.log, {{type: 'answer', seq: 999, answer: {{choice: 0, label: '10 x 10 mm', by: 'user'}}}}]}}); return true; }})()""")
    await asyncio.sleep(0.8)
    await c.shot("24_ghost_new_domain")
    check("chat bubbles are not squashed (text fits its box)", await c.js(
        "[...document.querySelectorAll('.chat .msg, .chat .brief')].every(m => m.scrollHeight <= m.clientHeight + 2)"),
        await c.js("[...document.querySelectorAll('.chat .msg, .chat .brief')].filter(m => m.scrollHeight > m.clientHeight + 2)"
                   ".map(m => [m.className, m.scrollHeight, m.clientHeight, getComputedStyle(m).flexShrink])"))
    tl = await c.js("[...document.querySelectorAll('.tl-item')].map(x => x.innerText)")
    check("timeline labels a clarification answer", any("answer: 10 x 10 mm" in x for x in tl), tl[-3:])
    await c.js(f"{ST}.set({{agentRun: null}}); true")



# ------------------------------------------------------------------------------------------------------------
async def t_assistant(c: CDP):
    await fresh(c)
    await solve_small(c)
    await asyncio.sleep(1.0)
    n0 = len((await c.spec())["loads"])
    # Act: a direct edit
    await c.click_sel('.dial button', "Act")
    await c.type_into('textarea[aria-label="Message the assistant"]', "")
    await c.js("document.querySelector('textarea[aria-label=\"Message the assistant\"]').focus(); true")
    await c.call("Input.insertText", text="Add a second load: 500 N in -z on the top face (y+), centred. Before editing, ask me "
                                          "how large the patch should be, offering 10 x 10 mm or 20 x 20 mm.")
    await c.key("Enter", code="Enter", vk=13, text="\r")
    check("Enter sends; the run shows 'working'", await c.soon("!!document.querySelector('.working')", 10))
    chips = await c.soon("document.querySelectorAll('[aria-label=\"Clarification question\"] .clar-chip').length >= 2", 300)
    check("Act run asks its question as chips", chips)
    if chips:
        await c.shot("30_act_question")
        await asyncio.sleep(80)                 # a user who thinks it over: past the ~60 s after which Codex hands back
        await c.js("document.querySelector('[aria-label=\"Clarification question\"] .clar-chip').click(); true")
    want = "l => l.region.face === 'y+' && l.force[0] === 0 && l.force[1] === 0 && l.force[2] === -500"
    ok = await c.soon(f"{ST}.spec.loads.some({want}) && !document.querySelector('.working')", 600)
    s = await c.spec()
    new = [l for l in s["loads"] if l["id"] != "load_1"]
    follow = await c.js(f"{ST}.chat.filter(m => m.role === 'user' && m.text.startsWith('My answer')).length")
    check("a late answer (80 s) still gets applied", ok and len(s["loads"]) == n0 + 1, (new, f"follow-up runs: {follow}"))
    print(f"   follow-up runs: {follow}", flush=True)
    await c.shot("30_act")
    if await c.click_sel(".run .linkbtn", "Undo this run"):
        check("'Undo this run's edits' reverts the load", await c.soon(f"{ST}.spec.loads.length === {n0}", 10))
    else:
        check("'Undo this run's edits' offered after an Act run", False)

    # Propose: cards, ghost, accept one
    await c.click_sel('.dial button', "Propose")
    await c.js("document.querySelector('textarea[aria-label=\"Message the assistant\"]').focus(); true")
    await c.call("Input.insertText", text="Propose two changes: set the volume fraction to 0.2, and add a keep-out cylinder of radius 8 mm "
                                          "along z through the middle of the box (x 50, y 25), full depth.")
    await c.click_sel(".composer button", "Send")
    await c.soon("!!document.querySelector('.working')", 10)
    ok = await c.soon("!document.querySelector('.working')", 600)
    ncards = await c.js("document.querySelectorAll('[aria-label=\"Proposed changes\"] .card').length")
    check("Propose returns cards and leaves the spec untouched", ok and ncards >= 2 and abs((await c.spec())["volume_fraction"] - 0.3) < 1e-9, ncards)
    await c.js("(() => { const c = [...document.querySelectorAll('[aria-label=\"Proposed changes\"] .card')].find(x => /keep|passive|cylinder/i.test(x.innerText)); "
               "c && c.scrollIntoView({block: 'center'}); return true; })()")
    xy = await c.js("(() => { const c = [...document.querySelectorAll('[aria-label=\"Proposed changes\"] .card')].find(x => /keep|passive|cylinder/i.test(x.innerText)); "
                    "if (!c) return null; const r = c.getBoundingClientRect(); return [r.left + 20, r.top + r.height / 2]; })()")
    if xy:
        await c.call("Input.dispatchMouseEvent", type="mouseMoved", x=xy[0], y=xy[1])
        check("hovering a card draws its ghost", await c.soon(f"!!{ST}.ghostCard", 3))
        await asyncio.sleep(0.5)
        await c.shot("31_propose_ghost")
    ok = await c.js("(() => { const c = [...document.querySelectorAll('[aria-label=\"Proposed changes\"] .card')].find(x => /volume/i.test(x.innerText)); "
                    "const b = c && [...c.querySelectorAll('button')].find(b => b.innerText === 'Accept'); b && b.click(); return !!b; })()")
    ok = ok and await c.soon(f"Math.abs({ST}.spec.volume_fraction - 0.2) < 1e-9", 10)
    check("Accept applies only that card", ok and len((await c.spec())["passive"]) == 0, await c.js(f"{ST}.spec.volume_fraction"))
    await c.js("(() => { const b = [...document.querySelectorAll('[aria-label=\"Proposed changes\"] .linkbtn')].find(b => b.innerText === 'Reject'); b && b.click(); return true; })()")
    await asyncio.sleep(0.3)
    check("Reject marks the other card rejected", await c.js("[...document.querySelectorAll('[aria-label=\"Proposed changes\"] .decided')].some(x => x.innerText === 'rejected')"))
    await c.shot("32_propose_decided")


async def main(what):
    OUT.mkdir(parents=True, exist_ok=True)
    prof = tempfile.mkdtemp(prefix="agentto_chrome_")
    port = 9334
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={prof}",
                             f"--window-size={W},{H}", "--hide-scrollbars", "--force-device-scale-factor=1",
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
            await c.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
            await c.call("Page.navigate", url=URL)
            await asyncio.sleep(2)
            for name, fn in (("setup", t_setup), ("ladder", t_ladder), ("overlays", t_overlays), ("assistant", t_assistant)):
                if name not in what:
                    continue
                print(f"== {name}", flush=True)
                try:
                    await fn(c)
                except Exception as e:              # noqa: BLE001
                    check(f"{name}: ran to the end", False, f"{type(e).__name__}: {e}")
                    await c.shot(f"error_{name}")
            check("no uncaught exceptions or console errors", not c.console, c.console[:5])
    finally:
        proc.terminate()
    fails = [r for r in RESULTS if not r["ok"]]
    (OUT / f"report_{'_'.join(what)}.json").write_text(json.dumps({"what": what, "passed": len(RESULTS) - len(fails), "failed": len(fails),
                                                 "results": RESULTS}, indent=1), encoding="utf-8")
    print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} passed", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main(sys.argv[1:] or ["setup", "ladder", "overlays"]))
