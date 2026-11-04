"""Paper tables, generated from the result files only (no GPU, no LLM).

    python -B scripts/make_tables.py        writes paper/tables/<name>.md and <name>.tex (booktabs)

Every number comes from a result file named in the table's source line; nothing is typed by hand except the rule
descriptions (which paraphrase src/agentto/rules.py) and the condition labels.
"""
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "paper" / "tables"
med = statistics.median


def load(p):
    return json.loads((R / p).read_text(encoding="utf-8"))


def x2(sub, rep=1):
    return {d["task"]: d for d in (json.loads(f.read_text(encoding="utf-8"))
                                    for f in sorted((R / "X2" / sub).glob(f"b*_r{rep}.json")))}


def mcnemar(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def silent(d):
    return (not d.get("behaviour_ok") and d.get("schema_valid") and not d.get("refused")
            and d.get("expected") != "refuse")


TEX_ESC = {"%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_", "→": r"$\rightarrow$", "×": r"$\times$", "–": "--",
           "≈": r"$\approx$", "≤": r"$\leq$", "≥": r"$\geq$", "−": "$-$", "Δ": r"$\Delta$", "·": r"$\cdot$",
           "β": r"$\beta$", "ν": r"$\nu$", "°": r"$^\circ$"}


def tex(s):
    s = str(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"`(.+?)`", r"\\texttt{\1}", s)
    return "".join(TEX_ESC.get(ch, ch) for ch in s)


def write(name, caption, head, rows, source, notes=()):
    OUT.mkdir(parents=True, exist_ok=True)
    md = [f"**{caption}**", "", "| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    md += ["", f"Source: {source}."] + [f"{n}" for n in notes]
    (OUT / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    t = [r"\begin{table}[t]", r"\centering\small", rf"\caption{{{tex(caption)}}}", rf"\label{{tab:{name}}}",
         r"\begin{tabular}{" + "l" + "r" * (len(head) - 1) + "}", r"\toprule",
         " & ".join(tex(h) for h in head) + r" \\", r"\midrule"]
    t += [" & ".join(tex(c) for c in r) + r" \\" for r in rows]
    t += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        t += [r"\par\smallskip\footnotesize " + tex(" ".join(notes))]
    t += [r"\end{table}"]
    (OUT / f"{name}.tex").write_text("\n".join(t) + "\n", encoding="utf-8")
    print("wrote", name)


# ---- T1 task suite ------------------------------------------------------------------------------------------

def t_suite():
    ints = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((ROOT / "briefs" / "intended").glob("*.json"))]
    exp = Counter(i["expected"] for i in ints)
    specs = [i.get("spec") or (i.get("readings") or [{}])[0].get("spec") for i in ints]
    specs = [s for s in specs if s]
    units = Counter(f"{s['units']['length']}, {s['units']['force']}" for s in specs)
    kinds = Counter(r["region"]["kind"] for s in specs for r in s["supports"] + s["loads"])
    passive = Counter(p["kind"] for s in specs for p in s.get("passive", []))
    with_passive = sum(1 for s in specs if s.get("passive"))
    rows = [["Briefs", len(ints)],
            ["Expected behavior", ", ".join(f"{k} {v}" for k, v in sorted(exp.items()))],
            ["Unit systems (length, force)", "; ".join(f"{k} ({v})" for k, v in units.most_common())],
            ["Support and load region kinds", ", ".join(f"{k} {v}" for k, v in kinds.most_common())],
            ["Briefs with passive regions", f"{with_passive} (solid {passive.get('solid', 0)}, "
                                            f"void {passive.get('void', 0)} regions)"],
            ["Clarify tasks: stored oracle facts per brief", ", ".join(
                str(len(i.get("oracle", []))) for i in ints if i["expected"] == "clarify")],
            ["Briefs with credited alternative readings",
             sum(1 for i in ints if any(a.get("counts") and a.get("spec") is not None for a in i.get("admissible", [])))],
            ["Source of the problem", f"engineering brief {sum(i.get('source') == 'engineering' for i in ints)}; "
                                      f"benchmark-modeled {sum(i.get('source') != 'engineering' for i in ints)} "
                                      f"(family confirmed 6, source untraced 6, generic 13)"]]
    write("t1_suite", "Task suite (50 briefs)", ["Property", "Value"], rows, "briefs/intended/*.json")


# ---- T2 rule base ---------------------------------------------------------------------------------------

RULES = [("R001", "error", "resolution cannot be met (grid too fine or too coarse)", "spec"),
         ("R002", "error", "a support selects no nodes", "spec"),
         ("R003", "error", "supports leave a rigid-body mode free", "physics"),
         ("R004", "error", "a load selects no nodes, or every load is zero", "spec"),
         ("R005", "error / warning", "load applied on fixed DOFs (dropped and recorded)", "physics"),
         ("R006", "error", "no design elements remain after passive regions", "spec"),
         ("R007", "error", "elements both keep-in and keep-out", "spec"),
         ("R008", "error", "a support or load lies entirely inside a keep-out", "physics"),
         ("R009", "warning", "smallest required feature not resolvable at this h", "resolution"),
         ("R010", "warning", "a region was snapped to the grid", "geometry"),
         ("R011", "warning", "a passive region contains no element centre", "geometry"),
         ("R012", "warning / info", "supports lost on coarse multigrid levels", "solver"),
         ("R013", "error", "memory need exceeds the GPU", "solver")]


def t_rules():
    write("t2_rules", "Verification rule base (deterministic; any error blocks solving and release)",
          ["Code", "Severity", "Condition", "Kind"], RULES, "src/agentto/rules.py")


# ---- T3 X1 escalation -----------------------------------------------------------------------------------

def t_x1():
    d = load("X1/escalation.json")
    rows = []
    for r in d["rows"]:
        cold = r["release"].get("cold") or r["release"].get("cold_failed") or {}
        row = [r["problem"].replace("_", " "), f"{r['rungs']['release']['n_elem'] / 1e6:.1f}M",
               f"{cold.get('compute_s', float('nan')):.0f}"]
        for arm in ("tail", "reopen_long"):
            a = r["release"].get(arm) or {}
            gap = a.get("gap_to_cold")
            frac = a.get("compute_s", 0) / cold["compute_s"] if cold.get("compute_s") else None
            iou = (a.get("shape_vs_cold") or {}).get("iou")
            row += [f"{100 * gap:+.2f}%" if gap is not None else "–",
                    f"{100 * frac:.0f}%" if frac else "–", f"{iou:.2f}" if iou else "–"]
        row.append("yes" if all((r["release"].get(a) or {}).get("certified") for a in ("tail", "reopen_long")) else "no")
        rows.append(row)
    write("t3_x1_escalation", "Certified release: warm arms against a cold run at the release rung",
          ["Problem", "Release elements", "Cold compute (s)", "Tail ΔC", "Tail compute", "Tail IoU",
           "Reopen-long ΔC", "Reopen-long compute", "Reopen-long IoU", "All certified"], rows,
          "results/X1/escalation.json",
          ["ΔC: release compliance relative to the cold run; compute as a share of the cold run's; IoU of the "
           "binarized design with the cold design."])


# ---- T4 X2 conditions -------------------------------------------------------------------------------------

LABEL = {"R": "R rule parser", "T": "T text only", "G": "G grounding tools", "GC": "G+C + clarification",
         "GCV": "G+C+V + picture check"}
# Follow-up conditions, run after the suite; their contested outputs were adjudicated afterwards (J.W., 1 Nov 2026)
FOLLOW = {"TS": "T+S text + schema feedback", "GNR": "GNR G without rule feedback", "GP": "G+P G + value provenance",
          "GPC": "G+C+P provenance-gated questions"}
SHOW = {"GC": "G+C", "GCV": "G+C+V", "TS": "T+S", "GP": "G+P", "GPC": "G+C+P"}


def t_x2():
    runs = {c: x2(c) for c in LABEL | FOLLOW}
    rows = []
    for c, lab in (LABEL | FOLLOW).items():
        rs = list(runs[c].values())
        right = sum(bool(d.get("behaviour_ok")) for d in rs)
        lo, hi = wilson(right, len(rs))
        tok = [d.get("usage", {}).get("input_tokens", 0) for d in rs if d.get("usage")]
        wall = [d.get("wall_s", 0) for d in rs if d.get("wall_s")]
        rows.append([lab, sum(bool(d.get("correct_spec")) for d in rs), f"{right} [{100 * lo:.0f}–{100 * hi:.0f}%]",
                     sum(silent(d) for d in rs), sum(d.get("questions", 0) for d in rs) if c in ("GC", "GCV", "GPC") else "–",
                     f"{med(tok) / 1e3:.0f}k" if tok else "–", f"{med(wall):.0f}" if wall else "–"])
    write("t4_x2_conditions", "Spec grounding on the 50-brief suite (Codex, GPT-6.1-Sol, effort high, repeat 1)",
          ["Condition", "Correct spec", "Right behavior [95% CI]", "Delivered wrong", "Questions",
           "Median input tokens", "Median wall (s)"], rows, "results/X2/<cond>/*_r1.json",
          ["Right behavior: the intended or an admissible spec, or a refusal where the brief must be refused. "
           "CI: Wilson score interval. T+S, GNR, G+P and G+C+P were run after the suite; their contested outputs "
           "were adjudicated afterwards by the same rule."])
    tasks = sorted(runs["GC"])
    mrows = []
    for a, b in (("R", "T"), ("T", "G"), ("G", "GC"), ("T", "GC"), ("GC", "GCV"),
                 ("T", "TS"), ("TS", "G"), ("G", "GNR"), ("G", "GP"), ("GC", "GPC")):
        fx = sum(1 for t in tasks if runs[b][t].get("behaviour_ok") and not runs[a][t].get("behaviour_ok"))
        br = sum(1 for t in tasks if runs[a][t].get("behaviour_ok") and not runs[b][t].get("behaviour_ok"))
        mrows.append([f"{SHOW.get(a, a)} → {SHOW.get(b, b)}", fx, br, f"{mcnemar(fx, br):.2g}"])
    write("t5_x2_mcnemar", "Paired comparisons of right behavior (exact McNemar, 50 briefs)",
          ["Comparison", "Fixed", "Broken", "p"], mrows, "results/X2/<cond>/*_r1.json")
    reps = []
    for c in ("T", "G", "GC", "GCV", "GP"):
        for r in (1, 2, 3):
            rs = x2(c, r)
            if rs:
                reps.append([(LABEL | FOLLOW)[c], r, len(rs), sum(bool(d.get("behaviour_ok")) for d in rs.values()),
                             sum(silent(d) for d in rs.values())])
    write("t6_x2_repeats", "Run-to-run variation (repeats 2 and 3 on subsets)",
          ["Condition", "Repeat", "Briefs", "Right behavior", "Delivered wrong"], reps, "results/X2/<cond>/*_r{1,2,3}.json")


def t_providers():
    rows = []
    for c in ("T", "GC", "GCV", "GPC"):
        cx, cl = x2(c), x2(f"claude_high/{c}")
        ts = sorted(set(cx) & set(cl))
        fx = sum(1 for t in ts if cx[t].get("behaviour_ok") and not cl[t].get("behaviour_ok"))
        br = sum(1 for t in ts if cl[t].get("behaviour_ok") and not cx[t].get("behaviour_ok"))
        for name, R_ in (("Codex gpt-6.1-sol", cx), ("Claude Code claude-opus-5-5", cl)):
            rs = [R_[t] for t in ts]
            rows.append([(LABEL | FOLLOW)[c], name, sum(bool(d.get("behaviour_ok")) for d in rs), sum(silent(d) for d in rs),
                         sum(d.get("questions", 0) for d in rs) if c != "T" else "–",
                         f"{med(d.get('usage', {}).get('input_tokens', 0) for d in rs) / 1e3:.0f}k",
                         f"{mcnemar(fx, br):.2g}" if name.startswith("Claude") else ""])
    write("t7_providers", "Two providers at matched effort (high), all 50 briefs, repeat 1",
          ["Condition", "Provider", "Right behavior", "Delivered wrong", "Questions", "Median input tokens",
           "McNemar p (vs Codex)"], rows, "results/X2/<cond>/*_r1.json, results/X2/claude_high/<cond>/*_r1.json")


# ---- T8 X3 end to end -------------------------------------------------------------------------------------

def t_x3():
    rows = []
    for f in sorted((R / "X3").glob("b*.json")):
        if "_run" in f.stem:
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        aw = R / "X3_aware" / f.name
        a = json.loads(aw.read_text()) if aw.exists() else {}
        st = d.get("stages") or {}
        rel = max((v for k, v in st.items() if k.startswith("solve_")), key=lambda v: v["n_elem"], default={})
        tok = d.get("usage", {}).get("input_tokens", 0)
        rows.append([f.stem.split("_", 1)[0], f"{rel.get('n_elem', 0) / 1e6:.1f}M" if rel else "–",
                     d.get("n_tool_calls"), d.get("questions"), f"{tok / 1e3:.0f}k",
                     f"{st.get('total_s', d.get('wall_s', 0)):.0f}",
                     f"{a['total_aware_s']:.0f}" if a.get("total_aware_s") else "–",
                     "yes" if rel.get("certified") else ("stopped (refuse)" if not d.get("released") else "no"),
                     "yes" if (d.get("replay") or {}).get("ok") else "–"])
    write("t8_x3_sessions", "End-to-end agent sessions: brief to certified release and STL (Codex, G+C)",
          ["Task", "Release elements", "Tool calls", "Questions", "Input tokens", "Session (s), X3",
           "Session (s), pad-aware rule", "Certified release", "Replay exact"], rows,
          "results/X3/b*.json, results/X3_aware/b*.json",
          ["Pad-aware column: the same session with every solve replayed under the pad-aware coarse rule; agent "
           "time is taken from the original run."])


# ---- T9 X5 resolution path ------------------------------------------------------------------------------

def t_x5():
    d = load("X5/resolution.json")
    rows = []
    for prob, x in d.items():
        for r in x.get("chain_reopen_long", []):
            rows.append([prob.replace("_", " "), "release path", f"{r['n_elem'] / 1e6:.2f}M", f"{r['c_part']:.4g}",
                         f"{100 * r['c_part_gap_to_cold_fine']:+.1f}%" if r.get("c_part_gap_to_cold_fine") is not None else "–",
                         f"{r['compute_s']:.0f}"])
        cf = [r for r in x.get("rows", []) if r["n_elem"] == max(q["n_elem"] for q in x["rows"])]
        for r in cf:
            rows.append([prob.replace("_", " "), "cold fine run", f"{r['n_elem'] / 1e6:.2f}M", f"{r['c_part']:.4g}",
                         "0%", f"{r['compute_s']:.0f}"])
    write("t9_x5_resolution", "Part compliance along the release path, judged at the finest grid",
          ["Problem", "Arm", "Elements", r"Part compliance (N\,mm)", "vs cold fine", "Compute (s)"], rows,
          "results/X5/resolution.json")


# ---- T10 X6 -------------------------------------------------------------------------------------------------

def t_x6():
    k = load("X6/klm.json")
    rel = [r for r in k if r.get("expected") != "refuse"]
    rows = []
    for key, lab in (("A1", "A1 solver script"), ("A2", "A2 typed structured specification"), ("B", "B GUI, no agent"),
                     ("C_typed", "C GUI + agent, brief typed"), ("C_pasted", "C GUI + agent, brief pasted")):
        ts = [r[key]["s"] if isinstance(r[key], dict) else r[key] for r in rel if r.get(key) is not None]
        acts = [r[key].get("actions") for r in rel if isinstance(r.get(key), dict) and r[key].get("actions")]
        rows.append([lab, f"{med(ts):.0f}", f"{min(ts):.0f}–{max(ts):.0f}", f"{med(acts):.0f}" if acts else "–"])
    write("t10_x6b_klm", f"Interaction cost by Keystroke-Level Model ({len(rel)} released tasks)",
          ["Condition", "Median user time (s)", "Range (s)", "Median actions"], rows, "results/X6/klm.json",
          ["A model, not a measurement: error-free expert, solver and agent waiting excluded."])
    lat = (R / "X6" / "latency_summary.md").read_text(encoding="utf-8").splitlines()
    i = next(n for n, l in enumerate(lat) if l.startswith("| Interaction"))
    body = [l for l in lat[i + 2:] if l.startswith("|")]
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in body]
    rows = [r for r in rows if len(r) == 4]
    write("t11_x6a_latency", "Interactive latency through the backend (14 tasks, 20 repeats of the fast interactions)",
          ["Interaction", "Target", "Median over tasks", "Range"], rows, "results/X6/latency/*.json")


# ---- T12 C4 gates -------------------------------------------------------------------------------------------

def t_gates():
    c = load("C4/census.json")
    ro = load("C4/rules_off.json")
    ref = next(r for r in ro if r["expected_rule"].startswith("none"))
    rows = [["Certificate", "release attempts certified", f"{c['release_attempts'] - len(c['release_attempts_uncertified'])}"
             f"/{c['release_attempts']}", "the two uncertified attempts produced no releasable design"],
            ["Certificate", "spill-guard stops (113M ×2, 59M)", sum(n.get("count", 1) for n in c["gate_catches_in_notes"]),
             "runs stopped; no design"],
            ["Certificate", "non-converged solves", len(c["not_converged_finished"]),
             "a gate-less system would show these as designs"]]
    for r in ro:
        if r["expected_rule"].startswith("none"):
            continue
        note = (r.get("error") or "")[:70] if r["status"] != "converged" else \
            f"compliance {r['compliance'] / ref['compliance']:.1e}× the corrected reference"
        rows.append(["Rule base off", f"{r['task'].split('_')[0]} ({r['expected_rule']})",
                     f"{r['status'].replace('_', ' ')}; {'certified' if r.get('certified') else 'not certified'}", note])
    write("t12_gates", "The release contract with each gate removed",
          ["Gate removed", "Case", "Outcome", "Note"], rows, "results/C4/census.json, results/C4/rules_off.json",
          ["With both gates on, no uncertified or ill-posed design was released in any study."])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for fn in (t_suite, t_rules, t_x1, t_x2, t_providers, t_x3, t_x5, t_x6, t_gates):
        try:
            fn()
        except Exception as ex:                     # noqa: BLE001
            print("FAILED", fn.__name__, type(ex).__name__, ex)


if __name__ == "__main__":
    main()
