"""Paper screenshots of the GUI, driven through headless Chrome over the DevTools protocol (no extra packages).

Needs the backend (scripts/serve.py, port 8765) and the Vite front end (port 5173) running, and the GPU free.
    python -B scripts/gui_screenshots.py [overview] [assistant]
- overview: b06 motor mount, sketch → preview → refine solved in the GUI, the refine design on screen.
- assistant: the b44 brief (ambiguous) sent to the assistant in Propose mode; one screenshot while its clarification
  question is open (chips with consequence previews), one after the oracle-free choice of the first chip, with the
  proposed cards. This runs one Codex session on the subscription.
Writes results/figures/gui_<name>.png.
"""
import asyncio
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
URL = "http://localhost:5173/"
W, H = 1680, 1000


class CDP:
    def __init__(self, ws):
        self.ws, self.n, self.pending = ws, 0, {}

    async def call(self, method, **params):
        self.n += 1
        mid = self.n
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            m = json.loads(await self.ws.recv())
            if m.get("id") == mid:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})

    async def js(self, expr, timeout=60):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True,
                            timeout=timeout * 1000)
        if r.get("exceptionDetails"):
            raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description", r["exceptionDetails"]))
        return r.get("result", {}).get("value")

    async def until(self, expr, timeout=600, every=1.0, what=""):
        t = time.time()
        while time.time() - t < timeout:
            if await self.js(expr):
                return True
            await asyncio.sleep(every)
        raise TimeoutError(what or expr)

    async def shot(self, name):
        r = await self.call("Page.captureScreenshot", format="png", captureBeyondViewport=False)
        out = FIG / f"gui_{name}.png"
        out.write_bytes(base64.b64decode(r["data"]))
        print("wrote", out, flush=True)


CLICK_RUNG = """(() => { const b = [...document.querySelectorAll('.ladder .rung')].find(x => x.innerText.startsWith('%s'));
  if (!b || b.disabled) return false; b.click(); return true; })()"""
RUNG_DONE = """(() => { const b = [...document.querySelectorAll('.ladder .rung')].find(x => x.innerText.startsWith('%s'));
  return !!b && /\\d+\\.\\ds/.test(b.innerText); })()"""
TYPE = """(() => { const t = document.querySelector('textarea[aria-label="Message the assistant"]');
  const set = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
  set.call(t, %s); t.dispatchEvent(new Event('input', {bubbles: true}));
  return true; })()"""
COLLAPSE_RESULT = """(() => { const b = document.querySelector('.result-toggle');
  if (b && b.getAttribute('aria-expanded') === 'true') b.click(); return true; })()"""
SEND = """(() => { const b = [...document.querySelectorAll('.composer button')].find(x => x.innerText === 'Send');
  if (!b || b.disabled) return false; b.click(); return true; })()"""


async def overview(c: CDP):
    await c.js("localStorage.setItem('agentto.tips.v1','done'); localStorage.setItem('agentto.theme','light'); true")
    await c.call("Page.navigate", url=URL)
    await c.until("document.querySelectorAll('.ladder .rung').length >= 4", 60, what="ladder")
    await c.until("[...document.querySelectorAll('.pill')].some(p => /GPU ready/.test(p.innerText))", 180, what="GPU worker")
    for rung in ("Sketch", "Preview", "Refine"):
        await c.until(CLICK_RUNG % rung, 60, what=f"click {rung}")
        await c.until(RUNG_DONE % rung, 900, 2.0, what=f"{rung} done")
        await asyncio.sleep(2)
    await asyncio.sleep(3)
    await c.shot("overview")


async def assistant(c: CDP):
    brief = (ROOT / "briefs" / "b44_gearbox_bracket.md").read_text(encoding="utf-8")
    await c.js("localStorage.setItem('agentto.tips.v1','done'); localStorage.setItem('agentto.theme','light'); true")
    await c.call("Page.navigate", url=URL)
    await c.until("!!document.querySelector('textarea[aria-label=\"Message the assistant\"]')", 60, what="composer")
    await c.js("(() => { const o = [...document.querySelectorAll('.picker option')].find(x => x.value === 'new:'); "
               "const s = o && o.parentElement; if (!s) return false; "
               "const set = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set; "
               "set.call(s, 'new:'); s.dispatchEvent(new Event('change', {bubbles: true})); return true; })()")
    await asyncio.sleep(2)
    await c.js(COLLAPSE_RESULT)
    await c.js(TYPE % json.dumps("Set up this part from the brief. Build a complete draft first; then, for anything the "
                                 "brief leaves open, ask me with each reading as a concrete option.\n\n" + brief))
    await c.until(SEND, 30, what="send")
    await c.until("!!document.querySelector('[aria-label=\"Clarification question\"]')", 900, 2.0, what="question")
    await c.until("document.querySelectorAll('[aria-label=\"Clarification question\"] .clar-chip').length > 0", 600, 2.0,
                  what="chips")
    await asyncio.sleep(2)
    await c.js(COLLAPSE_RESULT)
    await c.js("document.querySelector('[aria-label=\"Clarification question\"]').scrollIntoView({block: 'center'}); true")
    await asyncio.sleep(1)
    await c.shot("clarify")
    await c.js("(() => { const b = document.querySelector('[aria-label=\"Clarification question\"] button'); "
               "if (!b) return false; b.click(); return true; })()")
    await c.until("!!document.querySelector('[aria-label=\"Proposed changes\"]') || "
                  "!!document.querySelector('[aria-label=\"Clarification question\"]')", 900, 2.0, what="cards")
    # further questions: take the first chip each time until the cards arrive
    for _ in range(4):
        if await c.js("!!document.querySelector('[aria-label=\"Proposed changes\"]')"):
            break
        await c.js("(() => { const b = document.querySelector('[aria-label=\"Clarification question\"] button'); "
                   "if (b) b.click(); return true; })()")
        await c.until("!!document.querySelector('[aria-label=\"Proposed changes\"]') || "
                      "!!document.querySelector('[aria-label=\"Clarification question\"] button')", 900, 2.0, what="cards")
        await asyncio.sleep(1)
    await c.until("!document.querySelector('.working')", 900, 2.0, what="run end")
    await asyncio.sleep(2)
    await c.js(COLLAPSE_RESULT)
    await c.js("(() => { const k = document.querySelector('[aria-label=\"Proposed changes\"]'); if (!k) return false; "
               "k.scrollIntoView({block: 'start'}); const c = k.querySelector('.card'); "
               "if (c) c.dispatchEvent(new MouseEvent('mouseover', {bubbles: true})); return true; })()")
    await asyncio.sleep(1.5)
    await c.shot("assistant")


async def main(what):
    FIG.mkdir(parents=True, exist_ok=True)
    prof = tempfile.mkdtemp(prefix="agentto_chrome_")
    port = 9333
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
            await asyncio.sleep(3)
            if "overview" in what:
                await overview(c)
            if "assistant" in what:
                await assistant(c)
    finally:
        proc.terminate()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main(sys.argv[1:] or ["overview", "assistant"]))
