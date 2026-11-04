"""MCP server for the tool contract (stdio). Any MCP client (Codex CLI, Claude Code) drives the same logged tools
the GUI uses.

    python -B -m agentto.mcp_server --session DIR [--profile spec|full]      stand-alone session (benchmarks)
    python -B -m agentto.mcp_server --attach http://127.0.0.1:8765           the GUI's live session (actor "agent")

Profiles: "spec" = spec editing, grounding (list_regions, render), validate/explain (the X2 G condition); "full"
adds templates, solving and field_stats; "spec_prov" and "clarify_prov" add set_source/list_sources (X2 GP,
GPC), and in "clarify_prov" questions may concern only items tagged assumed. Tool results are compact JSON text; render returns the PNG as an image.
"""
from __future__ import annotations

import argparse
import base64
import inspect
import json
import sys
import typing
import urllib.request

INSTRUCTIONS = ("agentto: tools for a 3D topology-optimization design spec. Build it with new_spec, set_params and "
                "add_support/add_load/add_passive; ground it with list_regions; check it with validate_spec; show "
                "it with render. Coordinates are absolute, origin at the domain's min corner.")


def _raw_args_model():
    from pydantic import ConfigDict
    from mcp.server.mcpserver.utilities.func_metadata import ArgModelBase

    class RawArgs(ArgModelBase):
        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

        def model_dump_one_level(self):
            return {"__raw__": dict(self.model_extra or {})}
    return RawArgs


def _content(result: dict):
    from mcp.server.mcpserver import Image
    if isinstance(result, dict) and "png_b64" in result:
        meta = {k: v for k, v in result.items() if k != "png_b64"}
        return [json.dumps(meta, separators=(",", ":")), Image(data=base64.b64decode(result["png_b64"]),
                                                               format="png")]
    return json.dumps(result, separators=(",", ":"), ensure_ascii=False)


def build_server(call, profile: str):
    """MCP server whose tools forward to `call(name, args) -> Session.call-style dict`."""
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError as McpToolError
    from .tools import TOOLS
    srv = MCPServer("agentto", instructions=INSTRUCTIONS)
    _RawArgs = _raw_args_model()
    for name, t in TOOLS.items():
        if profile not in t["profiles"]:
            continue
        hints = typing.get_type_hints(t["fn"], include_extras=True)
        params = [p.replace(annotation=hints.get(p.name, typing.Any)) for p in t["params"]]

        def make(name=name, params=params):
            def fn(**kw):
                out = call(name, kw.get("__raw__", kw))
                if not out["ok"]:
                    raise McpToolError(out["error"])
                return _content(out["result"])
            fn.__name__ = name
            fn.__signature__ = inspect.Signature(params)
            fn.__annotations__ = {p.name: p.annotation for p in params}
            return fn
        srv.add_tool(make(), name=name, description=t["doc"], structured_output=False)
        # The published schema comes from the signature, but arguments reach the session unvalidated, so that
        # malformed calls are rejected (and logged) by Session.call like any other failure.
        srv._tool_manager.get_tool(name).fn_metadata.arg_model = _RawArgs
    return srv


def local_call(session):
    def call(name, args):
        return session.call(name, args, actor="agent")
    return call


def attach_call(base: str):
    def call(name, args):
        body = json.dumps({"args": args, "actor": "agent"}).encode()
        req = urllib.request.Request(f"{base.rstrip('/')}/api/tools/{name}", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.loads(r.read())
    return call


def url_answerer(base: str):
    """ask_clarification answered by the GUI backend's user (long poll; the backend renders the previews)."""
    def answer(question, options, previews=None, candidates=None):
        body = json.dumps({"question": question, "options": options, "candidates": candidates}).encode()
        req = urllib.request.Request(f"{base.rstrip('/')}/api/clarify", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read())
    return answer


def main(argv=None):
    sys.dont_write_bytecode = True
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", help="session directory (stand-alone mode)")
    ap.add_argument("--attach", help="backend URL of a running GUI session")
    ap.add_argument("--profile", default="spec", choices=["spec", "clarify", "spec_prov", "clarify_prov", "full"])
    ap.add_argument("--seed", help="stand-alone mode: JSON spec the session starts from (logged as actor system)")
    ap.add_argument("--oracle", help="stand-alone mode: intended-reading JSON; a scripted oracle answers "
                                     "ask_clarification (X2 condition G+C)")
    ap.add_argument("--answer-url", help="stand-alone mode: a GUI backend that puts ask_clarification to its user")
    ap.add_argument("--no-rules", action="store_true",
                    help="stand-alone mode: validate_spec checks the schema only, no rule findings (X2 ablation GNR)")
    a = ap.parse_args(argv)
    if a.attach:
        call = attach_call(a.attach)
    else:
        from .tools import Session
        s = Session(a.session, profile=a.profile)
        if a.no_rules:
            s.rules_feedback = False
            s._append({"type": "config", "rules_feedback": False})
        if a.seed:
            s.call("set_spec", {"spec": json.loads(open(a.seed, encoding="utf-8").read())}, actor="system")
        if a.oracle:
            from .bench import Oracle
            t = json.loads(open(a.oracle, encoding="utf-8").read())
            s.answerer = Oracle(t.get("spec", t), script=t.get("oracle"))
        elif a.answer_url:
            s.answerer = url_answerer(a.answer_url)
        call = local_call(s)
    build_server(call, a.profile).run("stdio")


if __name__ == "__main__":
    main()
