"""Week-4 latency check against the GUI latency budget (GUI_INTERACTION_PLAN.md section 6), through the running
backend exactly as the GUI uses it (WebSocket tool calls and events, HTTP for meshes).

    python -B scripts/serve.py            (in another terminal; nothing else on the GPU)
    python -B scripts/latency_check.py [b06_motor_mount]

Measures: validation round trip, set_spec round trip, each ladder rung from the solve call to its done event (first
frame time, compute, iterations, warm), an edit -> sketch re-solve, the release preview surface over HTTP, and the
full-resolution STL export. Writes results/latency/latency_<task>.json.
"""
import asyncio
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
BASE = "127.0.0.1:8765"


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def summary(xs):
    return {"n": len(xs), "p50_ms": round(1e3 * statistics.median(xs), 2), "p95_ms": round(1e3 * pct(xs, 0.95), 2),
            "max_ms": round(1e3 * max(xs), 2)}


def get(path):
    t = time.perf_counter()
    with urllib.request.urlopen(f"http://{BASE}{path}", timeout=600) as r:
        body = r.read()
        hdr = dict(r.headers)
    return body, hdr, time.perf_counter() - t


async def main(task="b06_motor_mount"):
    spec = json.loads(get(f"/api/tasks/{task}")[0])["spec"]
    out = {"task": task, "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    async with websockets.connect(f"ws://{BASE}/ws/session", max_size=None) as ws:
        inbox: asyncio.Queue = asyncio.Queue()

        async def reader():
            async for m in ws:
                if isinstance(m, bytes):
                    continue
                d = json.loads(m)
                d["_t"] = time.perf_counter()
                await inbox.put(d)
        rt = asyncio.create_task(reader())

        async def until(pred, timeout=900):
            while True:
                d = await asyncio.wait_for(inbox.get(), timeout)
                if pred(d):
                    return d

        await until(lambda d: d["type"] == "hello")
        await ws.send(json.dumps({"type": "worker_start"}))
        ref = 0

        async def tool(name, args):
            nonlocal ref
            ref += 1
            r = f"lat{ref}"
            t = time.perf_counter()
            await ws.send(json.dumps({"type": "tool", "name": name, "args": args, "ref": r,
                                      "meta": {"label": f"latency {name}"}}))
            d = await until(lambda d: d["type"] == "tool_result" and d.get("ref") == r)
            return d, d["_t"] - t, t

        # validation and edit round trips
        vs = []
        for i in range(30):
            t = time.perf_counter()
            await ws.send(json.dumps({"type": "validate", "id": i, "spec": spec}))
            d = await until(lambda d: d["type"] == "report" and d.get("id") == i)
            vs.append(d["_t"] - t)
        out["validate"] = summary(vs)
        ss = []
        for _ in range(10):
            d, dt, _ = await tool("set_spec", {"spec": spec})
            ss.append(dt)
        out["set_spec"] = summary(ss)

        lad = (await tool("ladder", {}))[0]["result"]["rungs"]
        rungs = []
        for r in lad[-4:]:
            d, dt, t0 = await tool("solve", {"h": r["h"], "wait": False})
            assert d["ok"], d
            jid = d["result"]["job"]
            first_frame = None
            while True:
                e = await until(lambda e: e.get("job") == jid and e["type"] in ("frame", "done", "error", "skipped"))
                if e["type"] == "frame" and first_frame is None:
                    first_frame = e["_t"] - t0
                if e["type"] != "frame":
                    break
            row = {"rung": r["name"], "h": r["h"], "n_elem": r["n_elem"], "call_ms": round(1e3 * dt, 1),
                   "first_frame_s": first_frame, "wall_s": e["_t"] - t0, "status": e.get("status", e["type"]),
                   "compute_s": e.get("compute_s"), "iterations": e.get("iterations"), "warm": None,
                   "certified": (e.get("certificate") or {}).get("certified"), "job": jid}
            rungs.append(row)
            print(json.dumps(row), flush=True)
        log = json.loads(get("/api/session?tail=400")[0])["log"]
        warm = {x["result"]["job"]: x["result"].get("warm_used") for x in log if x.get("type") == "result"}
        for row in rungs:
            row["warm"] = warm.get(row["job"])
        out["rungs"] = rungs

        # surface and export of the release design
        rel = rungs[-1]
        if rel["status"] == "converged":
            body, hdr, t1 = get(f"/api/mesh/{rel['job']}")
            _, _, t2 = get(f"/api/mesh/{rel['job']}")
            out["surface"] = {"first_s": t1, "cached_s": t2, "bytes": len(body), **json.loads(hdr.get("X-Mesh-Info", "{}"))}
            d, dt, _ = await tool("export_stl", {"job": rel["job"]})
            out["export_stl"] = {"call_s": dt, **({k: d["result"][k] for k in
                                 ("triangles", "bytes", "t_surface_s", "t_write_s", "certified", "released")}
                                 if d["ok"] else {"error": d.get("error")})}
            if d["ok"] and d["result"].get("mesh"):
                out["export_stl"]["closed"] = d["result"]["mesh"]["closed"]
            d, dt, _ = await tool("export_report", {"job": rel["job"]})
            out["export_report"] = {"call_s": dt, "ok": d["ok"]}

        # edit -> sketch: a 10% load change, then the sketch rung again (cold: the spec changed)
        edited = json.loads(json.dumps(spec))
        edited["loads"][0]["force"] = [f * 1.1 for f in edited["loads"][0]["force"]]
        await tool("set_spec", {"spec": edited})
        sk = lad[-4]
        d, dt, t0 = await tool("solve", {"h": sk["h"], "wait": False})
        jid = d["result"]["job"]
        e = await until(lambda e: e.get("job") == jid and e["type"] in ("done", "error", "skipped"))
        out["edit_to_sketch"] = {"wall_s": e["_t"] - t0, "compute_s": e.get("compute_s"),
                                 "iterations": e.get("iterations")}
        rt.cancel()
    p = ROOT / "results" / "latency" / f"latency_{task}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rungs"}, indent=1))


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
