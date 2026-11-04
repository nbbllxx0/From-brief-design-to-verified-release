"""X6a: the GUI latency budget (GUI_INTERACTION_PLAN.md section 6) over the 14 released X3 tasks, through the running
backend exactly as the GUI drives it (WebSocket tool calls and events, HTTP for meshes).

    python -B scripts/serve.py            (in another terminal; nothing else on the GPU)
    python -B scripts/x6a_latency.py [--repeats 20] [b01_cantilever_block ...]

Per task:
- spec edit -> validation and the set_spec commit, `repeats` times each;
- the ladder once (sketch cold, then preview, refine and release, each warm from the rung below, as the GUI runs them);
- the release surface over HTTP (first and cached) and the STL export;
- edit -> sketch `repeats` times: a 10% load change, alternating up and down, then the sketch rung warm from the
  last sketch.
The front-end frame times (three.js) are not measured here.
Writes results/X6/latency/<task>.json and results/X6/latency_summary.md.
"""
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latency_check import BASE, get, summary  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "X6" / "latency"
TASKS = ["b01_cantilever_block", "b02_mbb_half", "b04_torsion_shaft", "b05_l_bracket", "b06_motor_mount",
         "b11_cantilever_line_load", "b17_arch_bridge", "b19_engine_bracket", "b25_cantilever_metres",
         "b31_pillow_block", "b35_padeye", "b40_crank_arm", "b44_gearbox_bracket", "b47_camera_mount"]


async def run_task(task: str, repeats: int) -> dict:
    spec = json.loads(get(f"/api/tasks/{task}")[0])["spec"]
    out = {"task": task, "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "repeats": repeats}
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

        async def until(pred, timeout=3600):
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
            r = f"x6a{ref}"
            t = time.perf_counter()
            await ws.send(json.dumps({"type": "tool", "name": name, "args": args, "ref": r,
                                      "meta": {"label": f"X6a {name}"}}))
            d = await until(lambda d: d["type"] == "tool_result" and d.get("ref") == r)
            return d, d["_t"] - t, t

        async def solve(h):
            d, dt, t0 = await tool("solve", {"h": h, "wait": False})
            assert d["ok"], d
            jid = d["result"]["job"]
            first = None
            while True:
                e = await until(lambda e: e.get("job") == jid and e["type"] in ("frame", "done", "error", "skipped"))
                if e["type"] == "frame" and first is None:
                    first = e["_t"] - t0
                if e["type"] != "frame":
                    break
            return {"h": h, "call_ms": round(1e3 * dt, 1), "first_frame_s": first, "wall_s": e["_t"] - t0,
                    "status": e.get("status", e["type"]), "compute_s": e.get("compute_s"),
                    "iterations": e.get("iterations"), "certified": (e.get("certificate") or {}).get("certified"),
                    "job": jid}

        vs = []
        for i in range(repeats):
            t = time.perf_counter()
            await ws.send(json.dumps({"type": "validate", "id": i, "spec": spec}))
            d = await until(lambda d: d["type"] == "report" and d.get("id") == i)
            vs.append(d["_t"] - t)
        out["validate"] = summary(vs)
        ss = []
        for _ in range(repeats):
            ss.append((await tool("set_spec", {"spec": spec}))[1])
        out["set_spec"] = summary(ss)

        lad = (await tool("ladder", {}))[0]["result"]["rungs"]
        rungs = []
        for r in lad[-4:]:
            row = {"rung": r["name"], "n_elem": r["n_elem"], **await solve(r["h"])}
            rungs.append(row)
            print(task, json.dumps({k: row[k] for k in ("rung", "n_elem", "wall_s", "compute_s", "status")}),
                  flush=True)
        out["rungs"] = rungs

        rel = rungs[-1]
        if rel["status"] == "converged":
            body, hdr, t1 = get(f"/api/mesh/{rel['job']}")
            _, _, t2 = get(f"/api/mesh/{rel['job']}")
            out["surface"] = {"first_s": t1, "cached_s": t2, "bytes": len(body)}
            d, dt, _ = await tool("export_stl", {"job": rel["job"]})
            out["export_stl"] = {"call_s": dt, **({k: d["result"].get(k) for k in ("triangles", "bytes", "certified")}
                                                  if d["ok"] else {"error": d.get("error")})}

        sk = lad[-4]
        edits = []
        for i in range(repeats):
            edited = json.loads(json.dumps(spec))
            f = 1.1 if i % 2 == 0 else 1 / 1.1
            edited["loads"][0]["force"] = [x * f for x in edited["loads"][0]["force"]]
            spec = edited
            await tool("set_spec", {"spec": edited})
            edits.append(await solve(sk["h"]))
        out["edit_to_sketch"] = {"n": len(edits), "n_elem": sk["n_elem"],
                                 "wall_p50_s": statistics.median(e["wall_s"] for e in edits),
                                 "wall_p95_s": sorted(e["wall_s"] for e in edits)[int(round(0.95 * (len(edits) - 1)))],
                                 "first_frame_p50_s": statistics.median(e["first_frame_s"] or e["wall_s"] for e in edits),
                                 "runs": edits}
        rt.cancel()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{task}.json").write_text(json.dumps(out, indent=1))
    return out


def summarize():
    rows = [json.loads((OUT / f"{t}.json").read_text()) for t in TASKS if (OUT / f"{t}.json").exists()]
    if not rows:
        return
    med = statistics.median

    def rung(r, name, k):
        x = next((g for g in r["rungs"] if g["rung"] == name), None)
        return None if x is None else x[k]

    lines = [f"# X6a: latency over {len(rows)} X3 tasks (backend, {rows[0]['repeats']} repeats of the fast interactions)",
             "", "| Interaction | Target | Median over tasks | Range |", "|---|---|---|---|"]

    def add(name, target, xs, unit="s", f=1.0):
        xs = [x * f for x in xs if x is not None]
        if xs:
            lines.append(f"| {name} | {target} | {med(xs):.3g} {unit} | {min(xs):.3g}–{max(xs):.3g} {unit} |")
    add("Spec edit → validation (p50 per task)", "< 200 ms", [r["validate"]["p50_ms"] for r in rows], "ms")
    add("Spec edit → validation (p95 per task)", "< 200 ms", [r["validate"]["p95_ms"] for r in rows], "ms")
    add("Edit commit to the specification (p50)", "–", [r["set_spec"]["p50_ms"] for r in rows], "ms")
    add("Edit → sketch, warm (p50 per task)", "1–3 s", [r["edit_to_sketch"]["wall_p50_s"] for r in rows])
    add("Edit → sketch, warm (p95 per task)", "1–3 s", [r["edit_to_sketch"]["wall_p95_s"] for r in rows])
    add("Edit → first sketch frame (p50)", "–", [r["edit_to_sketch"]["first_frame_p50_s"] for r in rows])
    add("Sketch, cold (new spec)", "1–3 s", [rung(r, "sketch", "wall_s") for r in rows])
    add("Preview, warm", "≈ 5 s", [rung(r, "preview", "wall_s") for r in rows])
    add("Refine, warm", "2–5 s compute", [rung(r, "refine", "compute_s") for r in rows])
    add("Release, warm, compute", "≈ 10–25 s", [rung(r, "release", "compute_s") for r in rows])
    add("Release surface on screen (first fetch)", "< 3 s", [r.get("surface", {}).get("first_s") for r in rows])
    add("STL export at release", "–", [r.get("export_stl", {}).get("call_s") for r in rows])
    lines += ["", "| Task | Release elements | Sketch cold s | Edit→sketch p50 / p95 s | Preview s | Refine compute s | "
              "Release compute s (status) | Surface s |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rel = next((g for g in r["rungs"] if g["rung"] == "release"), {})
        lines.append(f"| {r['task']} | {rel.get('n_elem', 0) / 1e6:.1f}M | {rung(r, 'sketch', 'wall_s'):.1f} | "
                     f"{r['edit_to_sketch']['wall_p50_s']:.2f} / {r['edit_to_sketch']['wall_p95_s']:.2f} | "
                     f"{rung(r, 'preview', 'wall_s'):.1f} | {rung(r, 'refine', 'compute_s') or 0:.1f} | "
                     f"{rel.get('compute_s') or 0:.1f} ({rel.get('status')}) | "
                     f"{r.get('surface', {}).get('first_s', float('nan')):.2f} |")
    (ROOT / "results" / "X6" / "latency_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8")
    repeats = 20
    if "--repeats" in argv:
        i = argv.index("--repeats")
        repeats = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if "--summary" in argv:
        return summarize()
    for t in argv or TASKS:
        if (OUT / f"{t}.json").exists():
            print(t, "done already", flush=True)
            continue
        asyncio.run(run_task(t, repeats))
    summarize()


if __name__ == "__main__":
    main(sys.argv[1:])
