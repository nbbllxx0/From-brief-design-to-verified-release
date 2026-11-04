"""GUI assistant (G4): one headless Codex session per user message, on the subscription CLI only (llm/cli.py).

Two autonomy levels (the dial in the assistant panel):
    propose   the agent works on a private copy of the live spec (a stand-alone MCP session seeded with it, profile
              "clarify": editing, grounding and ask_clarification, no solving). When it finishes, the difference to the live spec becomes action cards; the
              user accepts cards, which are applied to the live session as one set_spec call by the user.
    act       the agent is attached to the live session (profile "full"): its edits apply at once and are
              undoable, it may solve the sketch and preview rungs to check a design; releasing stays the user's.
Codex events are translated to {"kind": start | message | tool | tool_done | proposal | done | error, ...} and passed
to `emit`; the backend broadcasts them to every socket, and the tool calls also land in the session log (actor
"agent") like any other call.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from ..llm.cli import CodexStream, codex_mcp_config

LISTS = ("supports", "loads", "passive")
FIELDS = ("units", "domain", "material", "volume_fraction", "resolution", "min_feature", "name", "description")

INSTRUCTIONS = """You are the design assistant inside agentto, an interactive 3D topology-optimization workspace. You
change the user's design only through the agentto tools. {mode_text}
Ground every region in the domain's coordinates: use list_regions to see faces and what each item covers, and
validate_spec after a change; fix what it reports. Keep ids that already exist. If the request is ambiguous in a way
that changes the design (which face, how large, what force), do not guess: call ask_clarification once with 2-4
concrete readings (the full item or parameter values each would give), then apply the reading the user chooses.
ask_clarification returns only once the user has answered, which can take minutes: if you are told the call is still
running, keep waiting for its result; never end your turn while your question is open.
Finish with 1-3 plain sentences: what you changed and anything you assumed."""

MODE_TEXT = {
    "propose": "Mode PROPOSE: your edits go to a private copy of the design; the user sees them as cards and "
               "accepts or rejects each one. Make the complete change asked for, then stop. You cannot solve.",
    "act": "Mode ACT: your edits apply to the live design at once (the user can undo them). You may solve the "
           "sketch or preview rung to check a design (solve with rung='sketch' or 'preview'); never solve the "
           "refine or release rungs, which the user starts.",
}


def build_prompt(message: str, mode: str, spec: dict | None, history: list[dict], context: dict) -> str:
    parts = [INSTRUCTIONS.format(mode_text=MODE_TEXT[mode]), ""]
    if spec:
        parts += ["Current design (agentto spec v0):", json.dumps(spec, separators=(",", ":")), ""]
    else:
        parts += ["There is no design yet: start one with new_spec.", ""]
    if context.get("selection"):
        parts.append(f"The user has selected: {json.dumps(context['selection'])}. 'this' / 'it' refers to it.")
    if context.get("findings"):
        parts.append("Current validation findings: " + "; ".join(context["findings"][:8]))
    if context.get("result"):
        parts.append("Latest solve: " + json.dumps(context["result"], separators=(",", ":")))
    if history:
        parts += ["", "Conversation so far:"] + [f"{h['role']}: {h['text']}" for h in history[-8:]]
    parts += ["", f"user: {message}"]
    return "\n".join(parts)


def _items(spec: dict, key: str) -> dict:
    return {x["id"]: x for x in (spec or {}).get(key) or []}


def _say(op: str, key: str, item: dict) -> str:
    kind = {"supports": "support", "loads": "load", "passive": "keep-in" if item.get("kind") == "solid" else "keep-out"}
    r = item.get("region", {})
    where = r.get("face") or r.get("kind", "")
    extra = f" force {item['force']}" if key == "loads" else ""
    return f"{op} {kind[key]} {item['id']} ({where}){extra}"


def spec_diff(before: dict | None, after: dict | None) -> list[dict]:
    """Action cards that turn `before` into `after`: one per added / removed / changed item and per changed field."""
    cards = []
    if after is None:
        return cards
    for key in LISTS:
        b, a = _items(before, key), _items(after, key)
        for i, x in a.items():
            if i not in b:
                cards.append({"op": "add", "list": key, "item_id": i, "after": x, "text": _say("add", key, x)})
            elif json.dumps(b[i], sort_keys=True) != json.dumps(x, sort_keys=True):
                cards.append({"op": "change", "list": key, "item_id": i, "before": b[i], "after": x,
                              "text": _say("change", key, x)})
        for i, x in b.items():
            if i not in a:
                cards.append({"op": "remove", "list": key, "item_id": i, "before": x, "text": _say("remove", key, x)})
    for f in FIELDS:
        bv, av = (before or {}).get(f), after.get(f)
        if json.dumps(bv, sort_keys=True) != json.dumps(av, sort_keys=True):
            cards.append({"op": "set", "field": f, "before": bv, "after": av,
                          "text": f"set {f.replace('_', ' ')} to {json.dumps(av)[:80]}"})
    for k, c in enumerate(cards):
        c["id"] = f"c{k + 1}"
    return cards


def apply_cards(base: dict | None, cards: list[dict], proposal: dict) -> dict:
    """The live spec with the accepted cards applied (cards not chosen are left as they were)."""
    if base is None:                                   # nothing live yet: the proposal is the design
        return json.loads(json.dumps(proposal))
    out = json.loads(json.dumps(base))
    for c in cards:
        if c["op"] == "set":
            out[c["field"]] = json.loads(json.dumps(c["after"]))
            continue
        lst = out.setdefault(c["list"], [])
        if c["op"] == "add":
            lst.append(json.loads(json.dumps(c["after"])))
        elif c["op"] == "remove":
            out[c["list"]] = [x for x in lst if x["id"] != c["item_id"]]
        else:
            out[c["list"]] = [json.loads(json.dumps(c["after"])) if x["id"] == c["item_id"] else x for x in lst]
    return out


def proposal_spec(session_dir: Path) -> dict | None:
    """Final draft of a stand-alone proposal session, rebuilt from its log (no LLM)."""
    from ..tools import Session, read_log
    log = Path(session_dir) / "session.jsonl"
    if not log.exists():
        return None
    s = Session(None, profile="full")
    for e in read_log(log):
        if e.get("type") == "call":
            s.call(e["tool"], e["args"], actor=e["actor"])
    return s.draft


class AgentRun:
    def __init__(self, message: str, mode: str, spec: dict | None, history: list[dict], context: dict,
                 root: Path, attach: str, emit, on_finish=None):
        if mode not in MODE_TEXT:
            raise ValueError(f"mode must be one of {sorted(MODE_TEXT)}")
        self.id = uuid.uuid4().hex[:8]
        # a copy: Act-mode edits change the live draft in place, and undo must restore the spec as it was here
        self.mode, self.message, self.base = mode, message, json.loads(json.dumps(spec)) if spec is not None else None
        self.dir = Path(root) / "agent" / self.id
        self.emit, self.on_finish = emit, on_finish
        self.cards: list[dict] = []
        self.proposal: dict | None = None
        cfg = (codex_mcp_config(attach=attach, profile="full") if mode == "act" else
               codex_mcp_config(self.dir / "session", "clarify", seed=self._seed(spec), answer_url=attach))
        self.stream = CodexStream(build_prompt(message, mode, spec, history, context), self.dir / "work",
                                  extra_config=cfg)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.timing: dict = {}

    def _seed(self, spec):
        if spec is None:
            return None
        p = self.dir / "seed.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(spec), encoding="utf-8")
        return p

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stream.stop()

    def _send(self, kind, **kw):
        self.emit({"type": "agent_event", "run": self.id, "kind": kind, "t": time.time(), **kw})

    def _run(self):
        t0 = time.perf_counter()
        self._send("start", mode=self.mode, message=self.message)
        text = []
        try:
            for ev in self.stream.events():
                et, item = ev.get("type"), ev.get("item") or {}
                dt = time.perf_counter() - t0
                if et in ("item.started", "item.completed"):
                    self.timing.setdefault("first_event_s", dt)
                if et == "item.started" and item.get("type") == "mcp_tool_call":
                    self.timing.setdefault("first_tool_s", dt)
                    self._send("tool", id=item.get("id"), tool=item.get("tool"), args=item.get("arguments"))
                elif et == "item.completed" and item.get("type") == "mcp_tool_call":
                    res = item.get("result") or {}
                    ok = item.get("error") is None and not (res.get("isError") or res.get("is_error"))
                    first = next((c.get("text", "") for c in res.get("content") or [] if c.get("type") == "text"), "")
                    self._send("tool_done", id=item.get("id"), tool=item.get("tool"), ok=ok,
                               summary=(first or json.dumps(item.get("error")))[:240])
                elif et == "item.completed" and item.get("type") == "agent_message":
                    self.timing.setdefault("first_message_s", dt)
                    text.append(item.get("text", ""))
                    self._send("message", text=item.get("text", ""))
                elif et in ("error", "turn.failed"):
                    self._send("error", message=json.dumps(ev)[:500])
            if self.mode == "propose" and not self.stream.stopped:
                self.proposal = proposal_spec(self.dir / "session")
                self.cards = spec_diff(self.base, self.proposal)
                self.timing["proposal_s"] = time.perf_counter() - t0
                self._send("proposal", cards=self.cards, spec=self.proposal)
        except Exception as ex:                       # noqa: BLE001 -- reported to the panel, not raised
            self._send("error", message=repr(ex)[:500])
        self.timing["wall_s"] = time.perf_counter() - t0
        final = (self.stream.final or (text[-1] if text else "")).strip()
        self._send("done", final=final, usage=self.stream.usage, timing=self.timing,
                   stopped=self.stream.stopped, returncode=self.stream.returncode,
                   stderr=None if self.stream.returncode in (0, None) else self.stream.stderr[-800:])
        if self.on_finish:
            self.on_finish(self, final)
