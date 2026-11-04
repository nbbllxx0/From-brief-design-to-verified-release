"""FastAPI backend: spec library, validation, the live design session and its WebSocket.

Run:  python -B scripts/serve.py        (listens on 127.0.0.1:8765; the Vite dev server proxies /api, /ws)

The app holds one `tools.Session` (the tool contract, logged to sessions/<time>/session.jsonl). GUI edits and solves
arrive as tool calls over the socket (actor "user"); an agent attached through `agentto.mcp_server --attach` calls
POST /api/tools/{name} (actor "agent"). Every log record and spec change is broadcast to all sockets, and solves
run on the one persistent GPU worker, whose progress and density frames are streamed as before.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from pydantic import ValidationError

from .. import SCHEMA_VERSION, __version__, library
from ..rules import check
from ..spec import Spec, json_schema
from ..tools import TOOLS, Session

app = FastAPI(title="agentto", version=__version__)
TASKS = Path(__file__).resolve().parents[3] / "briefs" / "intended"
SESSIONS = Path(__file__).resolve().parents[3] / "sessions"


@app.get("/api/health")
def health():
    return {"ok": True, "version": __version__, "schema": SCHEMA_VERSION}


@app.get("/api/schema")
def schema():
    return json_schema()


@app.get("/api/library")
def library_index():
    return sorted(library.FEAD)


@app.get("/api/library/{name}")
def library_spec(name: str, k: int = 40):
    if name not in library.FEAD:
        raise HTTPException(404, f"unknown problem {name!r}")
    return library.fead(name, k).model_dump()


@app.get("/api/tasks")
def tasks_index():
    """Draft X2 tasks (briefs/intended/*.json), for development: their intended specs load like library items."""
    return sorted(p.stem for p in TASKS.glob("*.json"))


@app.get("/api/tasks/{slug}")
def task_spec(slug: str):
    p = TASKS / f"{slug}.json"
    if not p.is_file() or p.parent != TASKS:
        raise HTTPException(404, f"unknown task {slug!r}")
    t = json.loads(p.read_text(encoding="utf-8"))
    brief = (TASKS.parent / f"{slug}.md").read_text(encoding="utf-8")
    return {"spec": Spec.model_validate(t["spec"]).model_dump(), "expected": t["expected"], "brief": brief}


@app.post("/api/validate")
def validate(spec: dict):
    try:
        s = Spec.model_validate(spec)
    except ValidationError as ex:
        return {"ok": False, "schema_errors": json.loads(ex.json())}
    return check(s).to_dict()


@app.post("/api/ladder")
def ladder_for(spec: dict):
    from ..worker import ladder
    try:
        s = Spec.model_validate(spec)
    except ValidationError:                 # a design still being built (no load yet): no rungs, validate says why
        return []
    return ladder(s)


# ---- GPU worker and session sockets -----------------------------------------------------------------------
# One worker process for the app; its events are broadcast to every connected session socket. Density frames go
# as a JSON header ({type: "frame", ..., bytes: n}) followed by one binary message with the uint8 volume.

class _Hub:
    def __init__(self):
        self.worker = None
        self.sockets: set[WebSocket] = set()
        self.pump = None
        self.ready = False

    def ensure(self):
        if self.worker is None or not self.worker.alive():
            from ..worker import GPUWorker
            self.worker = GPUWorker()
            self.ready = False
        if self.pump is None or self.pump.done():
            self.pump = asyncio.create_task(self._pump())

    async def _pump(self):
        loop = asyncio.get_running_loop()
        while self.worker is not None:
            ev = await loop.run_in_executor(None, self.worker.get, 0.5)
            if ev is None:
                if not self.worker.alive():
                    await self.broadcast({"type": "worker", "state": "dead"})
                    self.worker = None
                    return
                continue
            if ev["type"] == "ready":
                self.ready = True
                await self.broadcast({"type": "worker", "state": "ready"})
                continue
            data = ev.pop("data", None)
            if data is not None:
                ev["bytes"] = len(data)
            else:
                live.on_event(ev)             # job records + "result" log entries
            await self.broadcast(ev, data)

    async def broadcast(self, msg, data=None):
        for ws in list(self.sockets):
            try:
                await ws.send_json(msg)
                if data is not None:
                    await ws.send_bytes(data)
            except Exception:                    # noqa: BLE001 -- drop dead sockets
                self.sockets.discard(ws)


hub = _Hub()


class _HubRunner:
    """Session runner on the app's worker: the event pump feeds job events back into the session."""
    session = None

    def submit(self, job):
        if hub.worker is None or not hub.worker.alive():
            raise RuntimeError("GPU worker is not running")
        return hub.worker.submit(job)

    def cancel(self):
        if hub.worker is not None:
            hub.worker.cancel()

    def wait(self, done: threading.Event, timeout: float):
        return done.wait(timeout)

    def close(self):
        pass


live = Session(SESSIONS / time.strftime("%Y%m%d-%H%M%S"), runner=_HubRunner(), profile="full")
_loop: asyncio.AbstractEventLoop | None = None


def _on_record(rec):
    """Session log listener: broadcast each record, and the new spec after a successful edit."""
    if _loop is None:
        return
    msgs = [{"type": "log", "record": rec}]
    if rec.get("type") == "call" and rec["ok"] and TOOLS[rec["tool"]]["mutates"]:
        msgs.append({"type": "spec", "spec": live.draft, "seq": rec["seq"], "actor": rec["actor"],
                     "tool": rec["tool"], "origin": (rec.get("meta") or {}).get("origin")})
    for m in msgs:
        asyncio.run_coroutine_threadsafe(hub.broadcast(m), _loop)


live.listeners.append(_on_record)


async def _tool(name: str, args: dict, actor: str, meta: dict | None = None) -> dict:
    global _loop
    _loop = asyncio.get_running_loop()
    if name == "solve":
        hub.ensure()
    return await _loop.run_in_executor(None, lambda: live.call(name, args, actor, meta))


@app.post("/api/tools/{name}")
async def call_tool(name: str, body: dict):
    """Tool contract over HTTP (used by the MCP server in --attach mode). Body: {args, actor}."""
    return await _tool(name, body.get("args") or {}, body.get("actor", "agent"))


@app.get("/api/session")
def session_state(tail: int = 50):
    recs = []
    if live.log_path is not None and live.log_path.exists():
        recs = [json.loads(x) for x in live.log_path.read_text(encoding="utf-8").splitlines()[-tail:]]
    return {"spec": live.draft, "log": recs, "path": str(live.log_path), "chat": CHAT[-40:]}


# ---- assistant (G4): one Codex session per message, subscription CLI only ---------------------------------
CHAT: list[dict] = []                    # {role: user | assistant, text, run}
AGENT: dict = {"run": None, "runs": {}}


def _emit(msg):
    if _loop is not None:
        asyncio.run_coroutine_threadsafe(hub.broadcast(msg), _loop)


def _agent_finished(run, final):
    CHAT.append({"role": "assistant", "text": final or "(no reply)", "run": run.id})
    if AGENT["run"] is run:
        AGENT["run"] = None


def _agent_start(message: str, mode: str, context: dict):
    from .agent import AgentRun
    cur = AGENT["run"]
    if cur is not None and cur.thread.is_alive():
        raise RuntimeError("the assistant is still working on the previous message")
    history = [dict(h) for h in CHAT]
    CHAT.append({"role": "user", "text": message})
    run = AgentRun(message, mode, live.draft, history, context, live.root, "http://127.0.0.1:8765", _emit,
                   _agent_finished)
    AGENT["run"] = run
    AGENT["runs"][run.id] = run
    return run.start()


# ---- clarification (G5): the user answers ask_clarification from chips with consequence previews -----------
CLARIFY: dict[str, dict] = {}


async def _ensure_worker():
    hub.ensure()


def gui_answerer(question: str, options: list, previews=None, candidates=None) -> dict:
    """Put one clarification question to the GUI user. The previews (each reading solved at the sketch rung on the
    app's worker, warm from the latest sketch) are computed here, so they exist whichever session asked."""
    t0 = time.perf_counter()
    if previews is None and candidates:
        if _loop is not None:
            asyncio.run_coroutine_threadsafe(_ensure_worker(), _loop).result(10)
        previews = [live._preview(c) if c else {"ok": False, "error": "no draft yet"} for c in candidates]
    t_prev = time.perf_counter() - t0
    qid = uuid.uuid4().hex[:8]
    ev = threading.Event()
    # the asking run is kept: if it ends before the user answers, the answer starts a follow-up run (_clarify_answer)
    CLARIFY[qid] = {"event": ev, "answer": None, "question": question, "options": options, "run": AGENT["run"]}
    _emit({"type": "clarify", "qid": qid, "question": question, "options": options, "previews": previews,
           "previews_s": round(t_prev, 2)})
    ev.wait(600)
    got = CLARIFY.pop(qid)["answer"]
    ans = got or {"choice": None, "label": None, "comment": "no answer within 10 minutes"}
    ans.update(by="user", previews=previews, previews_s=round(t_prev, 2))
    _emit({"type": "clarify_done", "qid": qid, "choice": ans.get("choice")})
    return ans


live.answerer = gui_answerer


def _clarify_answer(msg: dict) -> tuple[str, str] | None:
    """Record the user's answer. If the run that asked has already ended (an agent can end its turn while the
    question is open), return (message, mode) for a follow-up run that carries the answer, so it is not lost."""
    q = CLARIFY.get(msg.get("qid", ""))
    if q is None or q["answer"] is not None:
        return None
    c = msg.get("choice")
    c = c if isinstance(c, int) and 0 <= c < len(q["options"]) else None
    q["answer"] = {"choice": c, "label": msg.get("label"), "comment": str(msg.get("comment") or "")}
    q["event"].set()
    run = q.get("run")
    if run is None or run.thread.is_alive():
        return None
    if c is not None:
        o = q["options"][c]
        reading = {k: o[k] for k in ("item_kind", "item", "params") if o.get(k) is not None}
        text = f"My answer to your question \"{q['question']}\": {o.get('label')} ({json.dumps(reading)})."
    else:
        text = f"My answer to your question \"{q['question']}\": none of these. {q['answer']['comment']}"
    return text + " Apply it.", run.mode


@app.post("/api/clarify")
async def clarify(body: dict):
    """ask_clarification from a stand-alone (propose-mode) agent session: answered by this app's user."""
    global _loop
    _loop = asyncio.get_running_loop()
    ans = await _loop.run_in_executor(None, lambda: gui_answerer(body.get("question", ""), body.get("options") or [],
                                                                  None, body.get("candidates")))
    return {k: v for k, v in ans.items() if k != "previews"} | {
        "previews": [{k: v for k, v in p.items() if k != "png_b64"} for p in ans.get("previews") or []]}


def _agent_accept(run_id: str, card_ids: list[str]) -> dict:
    from .agent import apply_cards
    run = AGENT["runs"].get(run_id)
    if run is None or run.proposal is None:
        return {"ok": False, "error": f"no proposal for run {run_id!r}"}
    chosen = [c for c in run.cards if c["id"] in set(card_ids)]
    if not chosen:
        return {"ok": False, "error": "no cards chosen"}
    new = apply_cards(live.draft, chosen, run.proposal)
    return live.call("set_spec", {"spec": new}, "user",
                     {"label": f"accept {len(chosen)} agent card{'s' if len(chosen) > 1 else ''}", "agent_run": run_id,
                      "cards": [c["id"] for c in chosen]})


_MESH: dict[str, tuple[bytes, dict]] = {}


@app.get("/api/mesh/{job}")
def mesh(job: str):
    """Preview surface of a finished design (pooled to at most 1.5M triangles): uint32 n_vertices, uint32
    n_triangles, float32 h_eff, then float32 positions (spec units) and uint32 indices."""
    import numpy as np
    from ..export import preview_mesh
    j = live.jobs.get(job)
    if j is None or not j.get("rho_path") or not Path(j["rho_path"]).is_file():
        raise HTTPException(404, f"no saved design for job {job!r}")
    if job not in _MESH:
        _MESH[job] = preview_mesh(np.load(j["rho_path"]), j["h"])
        while len(_MESH) > 6:
            _MESH.pop(next(iter(_MESH)))
    data, info = _MESH[job]
    return Response(data, media_type="application/octet-stream", headers={"X-Mesh-Info": json.dumps(info)})


@app.get("/api/files/{name}")
def export_file(name: str):
    """Download an exported file (STL, report) of the live session."""
    d = (live.root / "exports").resolve()
    p = (d / name).resolve()
    if p.parent != d or not p.is_file():
        raise HTTPException(404, f"no export {name!r}")
    return FileResponse(p, filename=name)


@app.websocket("/ws/session")
async def session(ws: WebSocket):
    """JSON messages {type, ...}: ping, validate, tool {name, args, meta, ref}, worker_start."""
    global _loop
    _loop = asyncio.get_running_loop()
    await ws.accept()
    hub.sockets.add(ws)
    cid = uuid.uuid4().hex[:8]
    await ws.send_json({"type": "hello", "version": __version__, "t": time.time(), "client": cid,
                        "worker": "ready" if hub.ready else ("starting" if hub.worker else "off")})
    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            if kind == "ping":
                await ws.send_json({"type": "pong", "t": time.time(), "echo": msg.get("t")})
            elif kind == "validate":
                await ws.send_json({"type": "report", "id": msg.get("id"), "report": validate(msg.get("spec", {}))})
            elif kind == "tool":
                meta = dict(msg.get("meta") or {}, origin=cid)
                out = await _tool(msg.get("name", ""), msg.get("args") or {}, "user", meta)
                await ws.send_json({"type": "tool_result", "ref": msg.get("ref"), **out})
            elif kind == "worker_start":
                hub.ensure()
            elif kind == "agent":
                try:
                    run = await _loop.run_in_executor(None, lambda: _agent_start(
                        str(msg.get("message", "")), msg.get("mode", "propose"), msg.get("context") or {}))
                    await hub.broadcast({"type": "chat", "role": "user", "text": msg.get("message", ""), "run": run.id})
                except Exception as ex:          # noqa: BLE001 -- policy refusal, busy, CLI missing
                    await ws.send_json({"type": "agent_event", "kind": "error", "run": None, "message": str(ex)})
            elif kind == "clarify_answer":
                follow = _clarify_answer(msg)
                if follow is not None:
                    try:
                        run = await _loop.run_in_executor(None, lambda: _agent_start(follow[0], follow[1], {}))
                        await hub.broadcast({"type": "chat", "role": "user", "text": follow[0], "run": run.id})
                    except Exception as ex:      # noqa: BLE001
                        await ws.send_json({"type": "agent_event", "kind": "error", "run": None, "message": str(ex)})
            elif kind == "agent_revert":             # undo a whole Act-mode run in one step
                run = AGENT["runs"].get(msg.get("run", ""))
                if run is not None and run.base is not None and not run.thread.is_alive():
                    await _loop.run_in_executor(None, lambda: live.call(
                        "set_spec", {"spec": run.base}, "user", {"label": "revert agent run", "agent_run": run.id}))
            elif kind == "agent_stop":
                if AGENT["run"] is not None:
                    AGENT["run"].stop()
            elif kind == "agent_accept":
                out = await _loop.run_in_executor(None, lambda: _agent_accept(msg.get("run", ""), msg.get("cards") or []))
                await hub.broadcast({"type": "agent_accepted", "run": msg.get("run"), "cards": msg.get("cards"),
                                     "ok": out.get("ok"), "error": out.get("error")})
            else:
                await ws.send_json({"type": "error", "message": f"unknown message type {kind!r}"})
    except WebSocketDisconnect:
        hub.sockets.discard(ws)


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")


if __name__ == "__main__":
    main()
