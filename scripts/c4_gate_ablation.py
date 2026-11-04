"""C4.3: the release contract with each gate switched off.

Part A, the certificate gate (no new runs): every solve recorded in X1, X3, X5 and X6a, and which of them a system
without the certificate would have shown as a finished design although a check failed.
Part B, the rule gate (GPU, a few minutes): the three ill-posed suite briefs (b08 and b49 leave a rigid-body mode
free, R003; b50 loads a pad inside its own keep-out, R008), solved directly with the rules bypassed, using the spec
the G agent built before validation stopped it. Does the solver, or its certificate, notice?

    python -B scripts/c4_gate_ablation.py [census|rules_off]      (both by default)
Writes results/C4/census.json, results/C4/rules_off.json and results/C4/summary.md.
"""
import gc
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "C4"
REFUSE = {"b08_slider_bracket": "R003", "b49_hinged_arm": "R003", "b50_valve_bracket": "R008"}


def _walk(x, path, out):
    if isinstance(x, dict):
        cert = x.get("certificate") if isinstance(x.get("certificate"), dict) else None
        if "status" in x and ("certified" in x or cert):
            checks = x.get("checks") or (cert or {}).get("checks")
            out.append({"path": path, "status": x.get("status"), "n_elem": x.get("n_elem"),
                        "certified": bool(x["certified"] if "certified" in x else cert.get("certified")),
                        "checks": checks, "error": x.get("error"),
                        "release": bool(x.get("release_rung")) or x.get("rung") == "release" or ".release" in path
                        or bool((checks or {}).get("spill_monitored"))})
            return
        for k, v in x.items():
            _walk(v, f"{path}.{k}", out)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            _walk(v, f"{path}[{i}]", out)


def census():
    files = {"X1": sorted((ROOT / "results" / "X1").glob("*.json")),
             "X3": [f for f in sorted((ROOT / "results" / "X3").glob("b*.json"))],
             "X5": sorted((ROOT / "results" / "X5").glob("*.json")),
             "X6a": sorted((ROOT / "results" / "X6" / "latency").glob("*.json")),
             "solver": sorted((ROOT / "results" / "solver").glob("*.json"))}
    rows = []
    for study, fs in files.items():
        for f in fs:
            found = []
            _walk(json.loads(f.read_text(encoding="utf-8")), f.name, found)
            top = max((r["n_elem"] or 0 for r in found), default=0)
            for r in found:
                r["study"] = study
                # X3 and X5 records name no rung: their finest solve per file is the release (spill guard on)
                if study in ("X3", "X5", "solver") and (r["n_elem"] or 0) == top:
                    r["release"] = True
            rows += found
    uncert = [r for r in rows if not r["certified"]]
    finished = [r for r in uncert if r["status"] in ("converged", "not_converged")]
    # the gate's own catches recorded in the study notes (the runs stopped, so no result record exists)
    notes = [{"study": "X1", "what": "113.2M warm release (mg_ir), twice", "count": 2, "stopped_by": "spill guard",
              "source": "results/X1/NOTES.md"},
             {"study": "X5", "what": "59M mg64 evaluation beside the desktop", "count": 1, "stopped_by": "spill guard",
              "source": "results/X5/NOTES.md"}]

    def failed(r):
        return sorted(k for k, v in (r["checks"] or {}).items() if not v)
    out = {
        "solves": len(rows), "certified": sum(r["certified"] for r in rows),
        "by_study": {s: {"solves": sum(r["study"] == s for r in rows),
                         "certified": sum(r["certified"] for r in rows if r["study"] == s)} for s in files},
        "uncertified_by_status": dict(Counter(r["status"] for r in uncert)),
        # without the gate, a finished but uncertified solve is indistinguishable from a released design
        "would_pass_without_gate": len(finished),
        "not_converged_finished": [{"path": r["path"], "study": r["study"], "n_elem": r["n_elem"]}
                                   for r in finished if r["status"] == "not_converged"],
        "gate_catches_in_notes": notes,
        "would_pass_by_failed_check": dict(Counter(",".join(failed(r)) or "none" for r in finished)),
        "release_attempts": sum(r["release"] for r in rows),
        "release_attempts_uncertified": [{"path": r["path"], "study": r["study"], "status": r["status"],
                                          "failed": failed(r), "error": r["error"]}
                                         for r in rows if r["release"] and not r["certified"]],
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "census.json").write_text(json.dumps(out, indent=1))
    return out


def rules_off():
    import cupy as cp
    from agentto.problem import build
    from agentto.rules import check
    from agentto.solver import SessionSIMP
    from agentto.spec import Spec
    from voxfem.simp import Config
    rows = []
    cases = [(t, c, None) for t, c in REFUSE.items()]
    # scale reference: b50 with its pad moved out of the keep-out (x = 40), a well-posed problem of the same size
    cases.append(("b50_valve_bracket", "none (reference: pad at x = 40)", 40.0))
    for task, code, move_x in cases:
        rec = json.loads((ROOT / "results" / "X2" / "G" / f"{task}_r1.json").read_text(encoding="utf-8"))
        if move_x is not None:
            rec["spec"]["loads"][0]["region"]["center"][0] = move_x
        spec = Spec.model_validate(rec["spec"])
        rep = check(spec)
        codes = sorted({f.code for f in rep.findings if f.severity == "error"}) if hasattr(rep, "findings") else None
        h = min(spec.domain.size) / 20
        row = {"task": task, "expected_rule": code, "rule_errors": codes, "h": h}
        t = time.perf_counter()
        try:
            sp = build(spec, h=h)
            row.update(grid=list(sp.grid.shape), n_elem=sp.grid.n_elem)
            kw = dict(volfrac=spec.volume_fraction, solver="mg64", op32="stencil", log_every=0,
                      conv_moving_ppm=100.0, spill_check_every=10, max_outer=150)
            m = SessionSIMP(sp, Config(**kw), coarse_pad="aware")
            r = m.run()
            cert = r.get("certificate") or {}
            its = [hh["solve"]["iters"] for hh in m.history if hh.get("solve")]
            row.update(status=r["status"], iterations=r["iterations"], compute_s=r["compute_s"],
                       compliance=None if r["compliance"] is None else r["compliance"] * sp.compliance_scale,
                       certified=cert.get("certified"), checks=cert.get("checks"), rel_true=cert.get("rel_true"),
                       pcg_iters_max=max(its) if its else None, error=r.get("error"))
            m = None
        except Exception as ex:                         # noqa: BLE001
            row.update(status="exception", error=f"{type(ex).__name__}: {ex}",
                       trace=traceback.format_exc()[-1500:])
        row["wall_s"] = time.perf_counter() - t
        print(json.dumps({k: row.get(k) for k in ("task", "rule_errors", "status", "certified", "compliance",
                                                  "error")}), flush=True)
        rows.append(row)
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rules_off.json").write_text(json.dumps(rows, indent=1, default=str))
    return rows


def summary():
    c = json.loads((OUT / "census.json").read_text())
    lines = ["# C4.3: the release contract with each gate off", "",
             "## Certificate gate off (census of every recorded solve, no new runs)", "",
             f"- Solves recorded: {c['solves']} (certified {c['certified']}).",
             "- By study: " + "; ".join(f"{s} {v['certified']}/{v['solves']}" for s, v in c["by_study"].items()) + ".",
             f"- Uncertified, by status: {c['uncertified_by_status']}. Sketch, preview and refine solves are "
             "uncertified by design (no spill guard), so only release attempts are counted below.",
             f"- Finished without converging (a system without the gate would show these as designs): "
             f"{len(c['not_converged_finished'])}: " + ", ".join(f"{r['study']} `{r['path']}`"
                                                                for r in c["not_converged_finished"]) + ".",
             f"- Release attempts (spill-monitored or release rung): {c['release_attempts']}; uncertified: "
             f"{len(c['release_attempts_uncertified'])}."]
    for r in c["release_attempts_uncertified"]:
        lines.append(f"  - {r['study']} `{r['path']}`: {r['status']}, failed {r['failed']}, {r['error'] or ''}")
    lines.append("- Release attempts the spill guard stopped (recorded in the notes; no design was produced): "
                 + "; ".join(f"{n['study']} {n['what']}" for n in c["gate_catches_in_notes"]) + ".")
    p = OUT / "rules_off.json"
    if p.exists():
        lines += ["", "## Rule gate off (ill-posed briefs solved with validation bypassed)", "",
                  "| Task | Rule the gate raises | Solver status | Certified | Compliance | Max PCG its | Note |",
                  "|---|---|---|---|---|---|---|"]
        for r in json.loads(p.read_text()):
            comp = r.get("compliance")
            lines.append(f"| {r['task']} | {r['expected_rule']} ({r.get('rule_errors')}) | {r.get('status')} | "
                         f"{r.get('certified')} | {'–' if comp is None else f'{comp:.4g}'} | {r.get('pcg_iters_max')} | "
                         f"{(r.get('error') or '')[:120]} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8")
    what = argv or ["census", "rules_off"]
    if "census" in what:
        census()
    if "rules_off" in what:
        rules_off()
    summary()


if __name__ == "__main__":
    main(sys.argv[1:])
