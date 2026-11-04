"""LLM access through subscription CLIs only.

Policy (project rule, 2026-10-29): no pay-as-you-go API. Every LLM call goes through a CLI logged in with the
author's subscription: Codex CLI (ChatGPT login) first, Claude Code optional. `assert_subscription_only` runs
before every call and refuses if an API key is present in the environment or Codex is not on a ChatGPT login,
because either would silently switch billing to metered API use.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

API_KEY_VARS = ("OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX")
DEFAULT_CODEX_MODEL = "gpt-6.1-sol"
DEFAULT_CODEX_EFFORT = "high"


class PolicyError(RuntimeError):
    pass


def codex_bin() -> str:
    exe = os.environ.get("AGENTTO_CODEX") or shutil.which("codex")
    if not exe:
        raise FileNotFoundError("codex CLI not found (set AGENTTO_CODEX)")
    return exe


def assert_subscription_only(provider: str = "codex"):
    present = [v for v in API_KEY_VARS if os.environ.get(v)]
    if present:
        raise PolicyError(f"API-key variables set ({present}); refusing: pay-as-you-go API use is prohibited")
    if provider == "codex":
        out = subprocess.run([codex_bin(), "login", "status"], capture_output=True, text=True, timeout=60)
        text = (out.stdout + out.stderr).strip()
        if "ChatGPT" not in text:
            raise PolicyError(f"codex is not logged in with a ChatGPT subscription: {text!r}")
    elif provider == "claude":
        out = subprocess.run([claude_bin(), "auth", "status"], capture_output=True, text=True, timeout=60)
        try:
            st = json.loads(out.stdout)
        except json.JSONDecodeError:
            st = {}
        if not st.get("loggedIn") or st.get("authMethod") != "claude.ai":
            raise PolicyError(f"claude is not logged in with a claude.ai subscription: {out.stdout.strip()[:200]!r}")


def claude_bin() -> str:
    exe = os.environ.get("AGENTTO_CLAUDE") or shutil.which("claude")
    if not exe:
        raise FileNotFoundError("claude CLI not found (set AGENTTO_CLAUDE)")
    return exe


def _clean_env():
    env = dict(os.environ)
    for v in API_KEY_VARS:
        env.pop(v, None)
    return env


def codex_mcp_config(session_dir: Path | None = None, profile: str = "spec", attach: str | None = None,
                     python: str | None = None, tool_timeout_s: int = 900, seed: Path | None = None,
                     oracle: Path | None = None, answer_url: str | None = None, no_rules: bool = False) -> list[str]:
    """`-c` overrides that give a Codex session exactly one MCP server: the agentto tool contract (stand-alone on
    `session_dir`, or attached to the GUI backend). With --ignore-user-config no other server is loaded."""
    import sys
    src = Path(__file__).resolve().parents[2]
    args = ["-B", "-m", "agentto.mcp_server", "--profile", profile]
    args += ["--attach", attach] if attach else ["--session", str(Path(session_dir).resolve())]
    if no_rules and not attach:
        args += ["--no-rules"]
    if seed is not None and not attach:
        args += ["--seed", str(Path(seed).resolve())]
    if oracle is not None and not attach:
        args += ["--oracle", str(Path(oracle).resolve())]
    if answer_url is not None and not attach:
        args += ["--answer-url", answer_url]
    py = (python or sys.executable).replace("\\", "/")
    return [f'mcp_servers.agentto.command="{py}"',
            "mcp_servers.agentto.args=" + json.dumps([a.replace("\\", "/") for a in args]),
            f'mcp_servers.agentto.env={{PYTHONPATH="{str(src).replace(chr(92), "/")}"}}',
            "mcp_servers.agentto.startup_timeout_sec=60",
            # exec mode cannot prompt; agentto tools only touch the session directory, so approve them
            'mcp_servers.agentto.default_tools_approval_mode="approve"',
            f"mcp_servers.agentto.tool_timeout_sec={tool_timeout_s}"]


def mcp_server_spec(session_dir: Path | None = None, profile: str = "spec", attach: str | None = None,
                    python: str | None = None, seed: Path | None = None, oracle: Path | None = None,
                    answer_url: str | None = None) -> dict:
    """{command, args, env} of the agentto MCP server, for any CLI's MCP configuration."""
    import sys
    src = Path(__file__).resolve().parents[2]
    args = ["-B", "-m", "agentto.mcp_server", "--profile", profile]
    args += ["--attach", attach] if attach else ["--session", str(Path(session_dir).resolve())]
    if not attach:
        if seed is not None:
            args += ["--seed", str(Path(seed).resolve())]
        if oracle is not None:
            args += ["--oracle", str(Path(oracle).resolve())]
        if answer_url is not None:
            args += ["--answer-url", answer_url]
    return {"command": (python or sys.executable).replace("\\", "/"),
            "args": [a.replace("\\", "/") for a in args], "env": {"PYTHONPATH": str(src).replace("\\", "/")}}


@dataclass
class CliResult:
    provider: str
    model: str
    effort: str
    cli_version: str
    final: str
    usage: dict
    events: list = field(default_factory=list)
    wall_s: float = 0.0
    returncode: int = 0
    stderr: str = ""


def _exec_cmd(exe: str, workdir: Path, model: str, effort: str, last: Path, extra_config, output_schema=None,
              images=None):
    cmd = [exe, "exec", "--json", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check",
           "-s", "read-only", "-C", str(workdir), "-m", model, "-c", f'model_reasoning_effort="{effort}"',
           "-o", str(last)]
    for c in extra_config or []:
        cmd += ["-c", c]
    if output_schema is not None:
        cmd += ["--output-schema", str(output_schema)]
    for im in images or []:
        cmd += ["-i", str(im)]
    return cmd + ["-"]


class CodexStream:
    """One headless Codex session whose JSONL events are delivered as they happen (the GUI assistant).

        st = CodexStream(prompt, workdir, extra_config=codex_mcp_config(attach=...))
        for ev in st.events():  ...           # dicts, as `codex exec --json` prints them
        st.final, st.usage, st.returncode
    `stop()` (from another thread) terminates the run. Same policy checks as run_codex."""

    def __init__(self, prompt: str, workdir: Path, model: str = DEFAULT_CODEX_MODEL,
                 effort: str = DEFAULT_CODEX_EFFORT, extra_config: list[str] | None = None):
        assert_subscription_only("codex")
        self.exe = codex_bin()
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.last = self.workdir / "last_message.txt"
        self.last.unlink(missing_ok=True)
        self.model, self.effort = model, effort
        self.cmd = _exec_cmd(self.exe, self.workdir, model, effort, self.last, extra_config)
        self.prompt = prompt
        self.proc = None
        self.final, self.usage, self.returncode, self.stderr = "", {}, None, ""
        self.stopped = False
        self.t0 = None

    def events(self):
        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, encoding="utf-8", env=_clean_env())
        import threading
        err: list[str] = []
        threading.Thread(target=lambda: err.extend(self.proc.stderr), daemon=True).start()
        self.proc.stdin.write(self.prompt)
        self.proc.stdin.close()
        raw = (self.workdir / "events.jsonl").open("w", encoding="utf-8")     # the run's event stream, kept
        for line in self.proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            raw.write(line + "\n")
            raw.flush()
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = ev.get("usage") if isinstance(ev, dict) else None
            if isinstance(u, dict):
                for k, v in u.items():
                    if isinstance(v, (int, float)):
                        self.usage[k] = self.usage.get(k, 0) + v
            yield ev
        raw.close()
        self.returncode = self.proc.wait()
        self.stderr = "".join(err)[-4000:]
        (self.workdir / "stderr.txt").write_text("".join(err), encoding="utf-8")
        self.final = self.last.read_text(encoding="utf-8") if self.last.exists() else ""
        self.wall_s = time.perf_counter() - self.t0

    def stop(self):
        self.stopped = True
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()


def run_claude(prompt: str, workdir: Path, mcp: dict | None = None, model: str | None = None,
               timeout: float = 900, images: list[Path] | None = None, effort: str | None = None) -> CliResult:
    """One headless Claude Code session on the claude.ai subscription: only the given MCP server
    (--strict-mcp-config), no built-in tools, user/project settings not loaded, nothing persisted. `mcp` is
    mcp_server_spec(...); its tools are pre-approved (mcp__agentto)."""
    assert_subscription_only("claude")
    exe = claude_bin()
    version = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    tools = ""
    if images:                                   # Claude Code has no image flag: copy them in, allow Read only
        import shutil as _sh
        names = []
        for im in images:
            dst = workdir / Path(im).name
            if Path(im).resolve() != dst.resolve():
                _sh.copyfile(im, dst)
            names.append(dst.name)
        prompt = prompt + "\n\nThe pictures are these files in the working directory; open each with Read: " + \
            ", ".join(names)
        tools = "Read"
    cmd = [exe, "-p", "--output-format", "stream-json", "--verbose", "--no-session-persistence",
           "--setting-sources", "", "--tools", tools]
    if images:
        cmd += ["--allowedTools", "Read"]
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--effort", effort]
    if mcp is not None:
        cfg = workdir / "mcp.json"
        cfg.write_text(json.dumps({"mcpServers": {"agentto": mcp}}), encoding="utf-8")
        cmd += ["--strict-mcp-config", "--mcp-config", str(cfg), "--allowedTools", "mcp__agentto"]
    t = time.perf_counter()
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=timeout,
                       env=_clean_env(), cwd=str(workdir))
    wall = time.perf_counter() - t
    events = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    res = next((e for e in reversed(events) if e.get("type") == "result"), {})
    u = res.get("usage") or {}
    usage = {"input_tokens": u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
             + u.get("cache_creation_input_tokens", 0),
             "cached_input_tokens": u.get("cache_read_input_tokens", 0),
             "cache_write_input_tokens": u.get("cache_creation_input_tokens", 0),
             "output_tokens": u.get("output_tokens", 0)}
    model_used = next((e.get("model") for e in events if e.get("type") == "system" and e.get("model")), model or "")
    return CliResult("claude", model_used, effort or "", version, res.get("result", "") or "", usage, events, wall,
                     p.returncode, p.stderr[-4000:])


def run_codex(prompt: str, workdir: Path, model: str = DEFAULT_CODEX_MODEL, effort: str = DEFAULT_CODEX_EFFORT,
              output_schema: Path | None = None, extra_config: list[str] | None = None,
              timeout: float = 900, images: list[Path] | None = None) -> CliResult:
    """One headless Codex session: user config ignored (no plugins or user MCP servers), read-only sandbox,
    nothing persisted. Tool servers for an experiment are passed explicitly through `extra_config`."""
    assert_subscription_only("codex")
    exe = codex_bin()
    version = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    last = workdir / "last_message.txt"
    cmd = _exec_cmd(exe, workdir, model, effort, last, extra_config, output_schema, images)
    t = time.perf_counter()
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=timeout,
                       env=_clean_env())
    wall = time.perf_counter() - t
    events = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    usage = {}
    for e in events:
        u = e.get("usage") if isinstance(e, dict) else None
        if isinstance(u, dict):
            for key, val in u.items():
                if isinstance(val, (int, float)):
                    usage[key] = usage.get(key, 0) + val
    final = last.read_text(encoding="utf-8") if last.exists() else ""
    return CliResult("codex", model, effort, version, final, usage, events, wall, p.returncode, p.stderr[-4000:])
