"""Generated claims ledger: every number the paper may state, read from the result files, with its source and status.

    python -B scripts/make_ledger.py        writes results/claims_ledger.md (no GPU, no LLM)

Status: supported (the evidence says what the claim says), partial (the evidence is narrower than the planned claim;
the note says how), restate (the planned wording goes beyond the evidence; use the wording in the row), pending
(the study has not run), missing (a source file is absent). Contribution numbers follow plan/PLAN.md section 3.
"""
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
BRIEFS = ROOT / "briefs" / "intended"


def load(p):
    return json.loads((R / p).read_text(encoding="utf-8"))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def x2_runs(cond, rep=1):
    out = {}
    for f in sorted((R / "X2" / cond).glob(f"b*_r{rep}.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out[d["task"]] = d
    return out


ROWS = []


def claim(cid, text, value, source, status="supported", note=""):
    ROWS.append((cid, text, value, source, status, note))


def guard(fn):
    try:
        fn()
    except FileNotFoundError as ex:
        claim(fn.__name__, "–", "–", str(ex.filename), "missing")


# ---- C1: knowledge representation and provenance -----------------------------------------------------------

def c1():
    x = load("X0/identity_gpu_k80.json")
    claim("C1.1", "With no passive regions, SessionSIMP reproduces voxfem's SIMP exactly", f"{len(x['rows'])} cases, "
          f"{x['criterion']}, all pass: {x['all_pass']}", "results/X0/identity_gpu_k80.json")
    s = load("X0/crosscheck_skfem.json")
    worst = max(r["rel_diff"] for r in s["rows"])
    claim("C1.2", "Spec-built problems (passive regions included) match an independent FE code (scikit-fem)",
          f"{len(s['rows'])} cases, max relative compliance difference {worst:.1e}", "results/X0/crosscheck_skfem.json")
    rp = load("X0/replay.json")
    claim("C1.3", "Sessions replay from the provenance log without an LLM", f"X0: {len(rp['tasks'])} sessions, "
          f"pass {rp['pass']}", "results/X0/replay.json")
    x3 = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((R / "X3").glob("b*.json"))
          if "_run" not in f.stem]
    ok = sum(1 for d in x3 if (d.get("replay") or {}).get("ok"))
    claim("C1.4", "Every X3 end-to-end session (brief to STL) replays exactly", f"{ok}/{len(x3)}", "results/X3/b*.json")
    rr = x2_runs("R")
    claim("C1.5", "A rule-based parser (no LLM) cannot read the suite", f"R correct {sum(bool(d.get('correct_spec')) for d in rr.values())}/{len(rr)}",
          "results/X2/R/*_r1.json")


# ---- C2: grounding and clarification (X2) -----------------------------------------------------------------

def c2():
    runs = {c: x2_runs(c) for c in ("R", "T", "G", "GC", "GCV")}
    tasks = sorted(runs["GC"])
    cor = {c: {t: bool(runs[c].get(t, {}).get("correct_spec")) for t in tasks} for c in runs}
    beh = {c: {t: bool(runs[c].get(t, {}).get("behaviour_ok")) for t in tasks} for c in runs}
    claim("C2.1", "Correct specs on the 50-brief suite (repeat 1): T / G / G+C / G+C+V",
          " / ".join(f"{sum(cor[c].values())}" for c in ("T", "G", "GC", "GCV")) + " of 50; right behaviour "
          + " / ".join(str(sum(beh[c].values())) for c in ("T", "G", "GC", "GCV")), "results/X2/<cond>/*_r1.json",
          "supported", "right behaviour: the intended (or an admissible) spec, or a refusal where the task must be refused")

    def mc(m, a, b):
        bb = sum(1 for t in tasks if m[b][t] and not m[a][t])     # fixed by b
        cc = sum(1 for t in tasks if m[a][t] and not m[b][t])     # broken by b
        return bb, cc, mcnemar_exact(bb, cc)
    for a, b in (("R", "T"), ("T", "G"), ("G", "GC"), ("T", "GC")):
        fb, bb, p = mc(beh, a, b)
        fs, bs, ps = mc(cor, a, b)
        claim(f"C2.2 {a}→{b}", f"Paired exact McNemar, {a} → {b} (primary: right behaviour; secondary: correct spec)",
              f"behaviour: fixed {fb}, broken {bb}, p = {p:.2g}; spec: fixed {fs}, broken {bs}, p = {ps:.2g}",
              "results/X2/<cond>/*_r1.json", "supported",
              "" if p < 0.05 else "behaviour difference not significant: report as no evidence of a difference")
    silent = {c: [t for t in tasks if runs[c].get(t, {}).get("expected") != "refuse" and not cor[c][t]
                  and not runs[c][t].get("refused") and runs[c][t].get("schema_valid")] for c in ("T", "G", "GC")}
    claim("C2.3", "Silent wrong specs (valid, not refused, not the intended or an admissible reading)",
          "; ".join(f"{c} {len(v)}" for c, v in silent.items()), "results/X2/<cond>/*_r1.json")
    reps = []
    for r in (1, 2, 3):
        rs = x2_runs("GC", r)
        if rs:
            reps.append((r, len(rs), sum(bool(d.get("correct_spec")) for d in rs.values())))
    claim("C2.4", "G+C is stable over repeats", ", ".join(f"r{r}: {c}/{n}" for r, n, c in reps),
          "results/X2/GC/*_r{1,2,3}.json")
    q = sum(runs["GC"][t].get("questions", 0) for t in tasks)
    qs = sum(runs["GC"][t].get("questions", 0) for t in tasks if runs["GC"][t].get("expected") == "solve")
    claim("C2.5", "Cost of clarification: questions asked by G+C", f"{q} questions on 50 briefs, {qs} of them on fully "
          "stated (solve) tasks", "results/X2/GC/*_r1.json", "supported",
          "the over-asking is a limitation to state")
    def claude(sub_dir, label, cid, subset=None):
        cl = x2_runs(f"{sub_dir}/GC")
        if not cl:
            return False
        sub = [t for t in tasks if t in cl and (subset is None or t[:3] in subset)]
        cx_ok = sum(beh["GC"][t] for t in sub)
        cl_ok = sum(bool(cl[t].get("behaviour_ok")) for t in sub)
        fb = sum(1 for t in sub if beh["GC"][t] and not cl[t].get("behaviour_ok"))
        bb = sum(1 for t in sub if cl[t].get("behaviour_ok") and not beh["GC"][t])
        qx = sum(runs["GC"][t].get("questions", 0) for t in sub)
        ql = sum(cl[t].get("questions", 0) for t in sub)
        sil = sum(1 for t in sub if cl[t].get("expected") != "refuse" and not cl[t].get("correct_spec")
                  and not cl[t].get("refused") and cl[t].get("schema_valid"))
        other = {c: x2_runs(f"{sub_dir}/{c}") for c in ("T", "GCV")}
        extra = "; ".join(f"{c} {sum(bool(other[c][t].get('behaviour_ok')) for t in sub if t in other[c])}/"
                          f"{sum(t in other[c] for t in sub)}" for c in other if other[c])
        p = mcnemar_exact(fb, bb)
        claim(cid, f"Second provider (Claude Code, claude-opus-5-5, {label}), G+C on {len(sub)} briefs",
              f"right behaviour Codex {cx_ok}/{len(sub)}, Claude {cl_ok}/{len(sub)}; McNemar p = {p:.2g}; "
              f"Claude silent wrong {sil}; questions {qx} vs {ql}" + (f"; Claude {extra}" if extra else ""),
              f"results/X2/{sub_dir}/GC/*_r1.json", "supported" if p >= 0.05 else "restate",
              "no evidence of a provider difference" if p >= 0.05 else
              "the tools carry over; the benefit depends on the model asking. Plan said 3 providers: report 2")
        return True
    lean = set("b01 b03 b04 b06 b07 b08 b09 b10 b19 b22 b25 b31 b39 b44 b45 b46 b47 b48 b49 b50".split())
    hi = claude("claude_high", "effort high, as Codex", "C2.6")
    lo = claude("claude", "default effort", "C2.7" if hi else "C2.6", lean)
    if not (hi or lo):
        claim("C2.6", "Planned: 3 LLM providers", "Codex gpt-6.1-sol only so far", "plan/experiments.md", "partial",
              "Claude Code generality runs pending; otherwise state one provider")


# ---- C3: certified escalation (X1, X5, solver) -----------------------------------------------------------

def c3():
    d = load("X1/escalation.json")
    gaps, ratios, n_el, cert = [], [], [], True
    for row in d["rows"]:
        rel = row["release"]
        rl, cold = rel.get("reopen_long"), rel.get("cold")
        if not rl or not cold:
            continue
        gaps.append(rl["gap_to_cold"])
        if cold.get("compute_s"):
            ratios.append(rl["compute_s"] / cold["compute_s"])
        n_el.append(rl["n_elem"])
        cert &= bool(rl.get("certified"))
    claim("C3.1", "Warm reopen_long release vs a cold run at the release rung", f"{len(gaps)} problems, "
          f"{min(n_el) / 1e6:.1f}–{max(n_el) / 1e6:.1f}M elements; gap {100 * min(gaps):+.2f}% to {100 * max(gaps):+.2f}%; "
          f"compute {100 * min(ratios):.0f}–{100 * max(ratios):.0f}% of cold; all certified {cert}",
          "results/X1/escalation.json", "supported",
          "the bridge's cold reference needed mg64 (mg_ir SolveFailed); see X1 NOTES")
    claim("C3.2", "Planned: release at 6.6×10^7–1.1×10^8 elements", f"largest certified release {max(n_el) / 1e6:.1f}M "
          "(X1); 113M skipped (VRAM spill beside the desktop)", "results/X1/NOTES.md", "restate",
          "say: certified releases of 2.6×10^7–7.4×10^7 elements on one 24 GB GPU")
    x5 = load("X5/resolution.json")
    br = x5["span_bridge"]["chain_reopen_long"]
    parts = [c["c_part"] for c in br if c.get("c_part")]
    cold_fine = next(r for r in x5["span_bridge"]["rows"] if r["arm"] == "cold" and r["grid"] == x5["span_bridge"]["fine_grid"])
    claim("C3.3", "Release path (cold coarse, then reopen_long per doubling) vs a cold fine run, as parts at 33M",
          f"part compliance {' → '.join(f'{p:.0f}' for p in parts)}; final vs cold fine {100 * (parts[-1] / cold_fine['c_part'] - 1):+.1f}%",
          "results/X5/resolution.json", "supported", "frame on the path, not on 'finer optima are better'")
    own = [r["c_own_phys"] / r["c_part"] - 1 for r in x5["span_bridge"]["rows"] + x5["b06"]["rows"]
           if r["arm"] == "cold" and r.get("c_part") and r.get("c_own_phys") and r["grid"] not in
           (x5["span_bridge"]["fine_grid"], x5["b06"]["fine_grid"])]
    claim("C3.4", "A coarse rung's own compliance under-predicts its part's", f"{100 * min(own):+.1f}% to {100 * max(own):+.1f}%",
          "results/X5/resolution.json")
    a = load("solver/b19_release_aware.json")
    claim("C3.5", "Pad-aware coarse levels: b19's 36M release, same warm start and schedule as X3",
          f"{a['x3_voxfem_rule']['compute_s']:.0f} → {a['compute_s']:.0f} s compute; compliance "
          f"{(a['compliance'] / a['x3_voxfem_rule']['compliance'] - 1):+.1e} relative; certified {a['certified']}",
          "results/solver/b19_release_aware.json")
    pa = load("solver/mg_padding_aware.json")
    pv = {(r["n"], r["levels"]): r for r in load("solver/mg_padding.json")}
    worst = max(pv[(r["n"], r["levels"])]["pcg_iters_mean"] for r in pa if r["pad_frac"] > 0)
    worst_a = max(r["pcg_iters_mean"] for r in pa if r["pad_frac"] > 0)
    claim("C3.6", "PCG iterations per solve on padded grids, voxfem's coarse rule → pad-aware", f"worst case "
          f"{worst:.0f} → {worst_a:.0f} (unpadded: 11–22)", "results/solver/mg_padding*.json")


# ---- C4: fail-closed release ---------------------------------------------------------------------------

def c4():
    cards = load("X6/cards.json")
    g = [r for r in cards if r["cond"] == "G" and r["rep"] == 1 and "validation" in r]
    stopped = sum(1 for r in g if r["validation"]["stopped"])
    claim("C4.1", "Validation stops every ill-posed brief before a solve (G)", f"{stopped}/3 refuse tasks "
          "(R003 ×2, R008)", "results/X6/cards.json")
    wrong = [r for r in g if r["expected"] != "refuse" and r["rules_ok"] and not r["correct_spec"] and not r["refused"]]
    claim("C4.2", "Validation cannot see misreadings: G's wrong specs that passed every rule", f"{len(wrong)}",
          "results/X6/cards.json", "supported", "clarification removes them: G+C 0")
    cen = load("C4/census.json")
    nc = len(cen["not_converged_finished"])
    claim("C4.3", "Certificate gate: release attempts, and what it held back", f"{cen['release_attempts']} release "
          f"attempts, {cen['release_attempts'] - len(cen['release_attempts_uncertified'])} certified; held back: "
          f"{len(cen['release_attempts_uncertified'])} uncertified plus "
          f"{sum(n.get('count', 1) for n in cen['gate_catches_in_notes'])} spill "
          f"stops; {nc} non-converged solves that a gate-less system would show as designs",
          "results/C4/census.json", "supported", "false acceptance with the gate: 0 by construction (every released "
          "design carries a passing certificate); without it, every non-converged or spilled run would pass")
    ro = {(r["task"], r["expected_rule"]): r for r in load("C4/rules_off.json")}
    b50 = ro[("b50_valve_bracket", "R008")]
    ref = next(r for (t, c), r in ro.items() if t == "b50_valve_bracket" and c.startswith("none"))
    fails = [t for (t, c), r in ro.items() if c == "R003" and r["status"] != "converged"]
    claim("C4.4", "Rule gate off: the ill-posed briefs solved with validation bypassed",
          f"R003 (rigid-body mode): {len(fails)}/2 end in SolveFailed (b49 after 20,000 PCG iterations); R008 (load "
          f"inside its keep-out, b50): converged and certified {b50['certified']}, compliance "
          f"{b50['compliance'] / ref['compliance']:.1e}× the corrected reference",
          "results/C4/rules_off.json", "supported",
          "the certificate alone would release b50: the rule base and the certificate catch different failures")


# ---- C5: end-to-end (X3) ---------------------------------------------------------------------------------

def c5():
    x3 = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((R / "X3").glob("b*.json")) if "_run" not in f.stem]
    rel = [d for d in x3 if d.get("released")]
    walls = [d["wall_s"] for d in rel if d.get("wall_s")]
    claim("C5.1", "Brief to certified release and STL, agent-driven (X3)", f"{len(rel)}/{len(x3)} released "
          f"(the other is a refuse task, stopped); median session {statistics.median(walls):.0f} s, range "
          f"{min(walls):.0f}–{max(walls):.0f} s", "results/X3/b*.json", "supported",
          "timings use voxfem's coarse rule; see C5.2 for the pad-aware re-time")
    aw = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((R / "X3_aware").glob("b*.json"))]
    aw = [a for a in aw if a.get("n_solves")]
    if aw:
        x3t = [a["x3_total_s"] for a in aw]
        awt = [a["total_aware_s"] for a in aw]
        cert = sum(bool((a.get("release") or {}).get("certified", [None, None])[1]) for a in aw)
        dC = [abs(a["release"]["rel_dC"]) for a in aw if (a.get("release") or {}).get("rel_dC") is not None]
        claim("C5.2", "X3 sessions with every solve replayed under the pad-aware coarse rule (agent time from the run)",
              f"{len(aw)} sessions: median {statistics.median(x3t):.0f} → {statistics.median(awt):.0f} s, range "
              f"{min(awt):.0f}–{max(awt):.0f} s; releases certified {cert}/{len(aw)}; max |ΔC| of the release "
              f"{max(dC):.1e}", "results/X3_aware/b*.json", "supported",
              "unpadded grids are rule-independent: their changes (up to ±15%) are run-to-run timing noise")


# ---- C6: interaction (X6) --------------------------------------------------------------------------------

def c6():
    k = load("X6/klm.json")
    rs = [r for r in k if r["expected"] != "refuse"]
    med = lambda c: statistics.median(r[c]["s"] for r in rs)  # noqa: E731
    claim("C6.1", "User setup-and-release time by KLM, median of the released X3 tasks",
          f"voxfem script {med('A1'):.0f} s; agentto spec {med('A2'):.0f} s; GUI alone {med('B'):.0f} s; GUI + agent "
          f"{med('C_pasted'):.0f} s (brief pasted) / {med('C_typed'):.0f} s (typed)", "results/X6/klm.json", "supported",
          "a model, not a measurement; claim only 'removes the translation of an existing text'")
    claim("C6.2", "A1 problem setups execute and match problem.build", f"{sum(r['A1']['setup_verified'] for r in k)}/{len(k)}",
          "results/X6/klm.json")
    c = load("X6/cards.json")
    gc = [r for r in c if r["cond"] == "GC" and r["rep"] == 1 and "n" in r]
    n = sum(sum(r["n"].values()) for r in gc)
    acc = sum(r["n"].get("accept", 0) for r in gc)
    claim("C6.3", "G+C proposals accepted as-is by the scripted oracle", f"{acc}/{n} cards", "results/X6/cards.json")
    p = R / "X6" / "latency_summary.md"
    if p.exists():
        rows = [json.loads(f.read_text()) for f in sorted((R / "X6" / "latency").glob("b*.json"))]
        e2s = [r["edit_to_sketch"]["wall_p95_s"] for r in rows]
        val = [r["validate"]["p95_ms"] for r in rows]
        cold = [next(g for g in r["rungs"] if g["rung"] == "sketch")["wall_s"] for r in rows]
        cert = sum(bool(r["rungs"][-1]["certified"]) for r in rows)
        claim("C6.4", "Interactive loop within budget (X6a, backend, 20 repeats)", f"{len(rows)} tasks: validation p95 "
              f"{min(val):.1f}–{max(val):.1f} ms; edit → warm sketch p95 {min(e2s):.2f}–{max(e2s):.2f} s; releases "
              f"certified {cert}/{len(rows)}", "results/X6/latency/*.json")
        claim("C6.5", "Cold sketch (new spec) within 1–3 s", f"{sum(c <= 3 for c in cold)}/{len(rows)} within; "
              f"{min(cold):.1f}–{max(cold):.1f} s", "results/X6/latency/*.json", "restate",
              "target missed for cold sketches; the first frame arrives in about 0.1 s")
    else:
        claim("C6.4", "Latency budget over the X3 tasks (X6a)", "–", "results/X6/latency_summary.md", "pending")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for fn in (c1, c2, c3, c4, c5, c6):
        guard(fn)
    lines = ["# Claims ledger (generated by `scripts/make_ledger.py`; do not edit by hand)", "",
             "| # | Claim | Value | Source | Status | Note |", "|---|---|---|---|---|---|"]
    for cid, text, value, src, status, note in ROWS:
        lines.append(f"| {cid} | {text} | {value} | `{src}` | {status} | {note} |")
    (R / "claims_ledger.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
