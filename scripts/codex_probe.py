"""Week-1 probe: one text-only (condition T) session on Codex CLI, to check headless use and measure tokens.

Brief -> spec JSON with the v0 schema in the prompt; no tools. The reply is validated with pydantic and the
rule base. Writes results/probe/codex_T_<brief>.json (usage, wall time, CLI version, validation).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentto.llm.cli import run_codex  # noqa: E402
from agentto.rules import check  # noqa: E402
from agentto.spec import Spec  # noqa: E402

BRIEF = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "briefs" / "b01_cantilever_block.md")
brief_path = Path(BRIEF)
schema = (ROOT / "schemas" / "spec_v0.schema.json").read_text()
prompt = f"""You translate an engineering design brief into a topology-optimization spec.
Return ONLY one JSON object that validates against this JSON Schema (agentto.spec/v0). No prose, no code fences.
Region coordinates are absolute in the domain frame, origin at the domain's min corner. A face patch gives its
centre and full size in the face's two in-plane axes in increasing axis order: (y, z) on x faces, (x, z) on y
faces, (x, y) on z faces. Loads are total forces on the region.

JSON Schema:
{schema}

Brief:
{brief_path.read_text(encoding='utf-8')}
"""
work = ROOT / "results" / "probe" / "work"
r = run_codex(prompt, work)
text = r.final.strip()
m = re.search(r"\{.*\}", text, re.S)
out = {"brief": brief_path.name, "provider": r.provider, "model": r.model, "effort": r.effort,
       "cli_version": r.cli_version, "wall_s": r.wall_s, "usage": r.usage, "returncode": r.returncode,
       "n_events": len(r.events), "prompt_chars": len(prompt), "final": text}
try:
    spec = Spec.model_validate(json.loads(m.group(0)))
    rep = check(spec)
    out["schema_valid"] = True
    out["rules"] = rep.to_dict()
    out["spec"] = spec.model_dump()
except Exception as ex:                       # noqa: BLE001 -- record any failure, it is a result
    out["schema_valid"] = False
    out["error"] = repr(ex)[:2000]
    out["stderr"] = r.stderr
dst = ROOT / "results" / "probe" / f"codex_T_{brief_path.stem}.json"
dst.write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("model", "effort", "cli_version", "wall_s", "usage", "schema_valid")}))
