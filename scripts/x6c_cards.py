"""X6c: agent action cards and validation catches under the scripted oracle (no LLM, no GPU; reads the X2 runs).

Each X2 run's final spec is replayed as one Propose-mode run on an empty workspace: `app.agent.spec_diff` turns it
into the cards the GUI shows (one per item, one per changed field; the cosmetic name and description cards are left
out). The scripted oracle then decides each card the way the brief's author would:
    accept   the card is already right: the item builds to the same fixed DOFs / nodal loads / element mask as an item
             of the reference spec, or the field equals the reference's;
    edit     the card is wrong but stands for something the reference has (an item of the same list that no card has
             claimed yet, or a field): the user corrects its values;
    reject   nothing in the reference corresponds to it.
Reference items that no card claimed are counted as missing (the user adds them by hand). The reference is the spec
the run was credited with (the intended reading or the admissible one it matched), else the intended spec.

Validation catches come from the G and GC tool logs: a session in which `validate_spec` reported an error before any
solve either fixed it (a later validation is clean) or stopped on it (the run ends unsolved and reports the problem,
which is the right outcome on the refuse tasks). The table also counts wrong specs that validation passed: errors of
intent, which no rule can see.

    python -B scripts/x6c_cards.py           writes results/X6/cards.json and results/X6/cards_summary.md
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentto.app.agent import spec_diff  # noqa: E402
from agentto.bench import equivalent, item_match, normalized  # noqa: E402

X2 = ROOT / "results" / "X2"
BRIEFS = ROOT / "briefs" / "intended"
OUT = ROOT / "results" / "X6"
CONDS = ("T", "G", "GC", "GCV")
KIND = {"supports": "support", "loads": "load", "passive": "passive"}
FIELD_OF = {"domain": "domain", "material": "material", "volume_fraction": "volume_fraction",
            "resolution": "resolution", "min_feature": "min_feature"}


def reference(run: dict, task: dict) -> dict:
    m = (run.get("match") or {}).get("matched")
    if m and m.startswith("admissible:"):
        return task["admissible"][int(m.split(":")[1])]["spec"]
    return task["spec"]


def decide(pred: dict, ref: dict) -> dict:
    """Oracle decision for every card of an empty-workspace proposal of `pred` against `ref`."""
    p = normalized(pred).model_dump(mode="json")
    q = normalized(ref).model_dump(mode="json")
    fields = equivalent(pred, ref)["fields"]
    cards = [c for c in spec_diff(None, pred) if c.get("field") not in ("name", "description")]
    pn = {key: {x["id"]: x for x in p.get(key) or []} for key in KIND}
    claimed = {key: set() for key in KIND}
    out = []
    for c in cards:
        if c["op"] == "set":
            ok = fields.get(FIELD_OF.get(c["field"], ""), True)      # units: a way of writing, judged via the values
            out.append({"card": c["text"], "decision": "accept" if ok else "edit"})
            continue
        key = c["list"]
        i = item_match(pn[key][c["item_id"]], KIND[key], q)
        if i is not None and i not in claimed[key]:
            claimed[key].add(i)
            out.append({"card": c["text"], "decision": "accept"})
            continue
        free = [j for j in range(len(q.get(key) or [])) if j not in claimed[key]]
        if free:
            claimed[key].add(free[0])
            out.append({"card": c["text"], "decision": "edit"})
        else:
            out.append({"card": c["text"], "decision": "reject"})
    missing = sum(len(q.get(key) or []) - len(claimed[key]) for key in KIND)
    return {"cards": out, "missing": missing}


def validation_catches(log: Path) -> dict:
    """Validation errors reported during the session, and whether the last validation was clean."""
    seen, last_ok, n_val, before_solve = set(), None, 0, True
    for line in log.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e.get("type") != "call":
            continue
        if e["tool"] == "solve":
            before_solve = False
        if e["tool"] == "validate_spec" and e.get("ok"):
            n_val += 1
            r = e.get("result") or {}
            errs = {f["code"] for f in r.get("findings", []) if f.get("severity") == "error"}
            if errs and before_solve:
                seen |= errs
            last_ok = not errs
    return {"validations": n_val, "error_codes": sorted(seen), "fixed": bool(seen) and last_ok is True,
            "stopped": bool(seen) and last_ok is False and before_solve, "ended_clean": last_ok}


def main():
    rows = []
    for cond in CONDS:
        for f in sorted((X2 / cond).glob("b*_r*.json")):
            run = json.loads(f.read_text(encoding="utf-8"))
            task = json.loads((BRIEFS / f"{run['task']}.json").read_text(encoding="utf-8"))
            row = {"cond": cond, "task": run["task"], "rep": run.get("rep", 1), "expected": run.get("expected"),
                   "correct_spec": run.get("correct_spec"), "refused": run.get("refused"),
                   "rules_ok": run.get("rules_ok"), "behaviour_ok": run.get("behaviour_ok")}
            spec = run.get("spec")
            if isinstance(spec, dict) and run.get("schema_valid"):
                try:
                    d = decide(spec, reference(run, task))
                except Exception as ex:                    # noqa: BLE001 -- a spec that does not build: no cards
                    d = {"cards": [], "missing": None, "error": repr(ex)[:200]}
                row.update(d)
                row["n"] = Counter(c["decision"] for c in d["cards"])
            log = f.with_suffix("") / "session" / "session.jsonl"
            if cond in ("G", "GC") and log.exists():
                row["validation"] = validation_catches(log)
            rows.append(row)
            print(cond, run["task"], row.get("n"), row.get("missing"), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cards.json").write_text(json.dumps(rows, indent=1, default=dict), encoding="utf-8")
    summarize(rows)


def summarize(rows):
    lines = ["# X6c: action cards and validation catches (scripted oracle, X2 repeat 1, 50 briefs)", "",
             "| Condition | Runs with cards | Cards | Accepted | Edited | Rejected | Missing items | All cards accepted, nothing missing |",
             "|---|---|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in rows:
        if r["rep"] == 1:                     # repeat 1: all 50 briefs in every condition (cards.json keeps all)
            by[r["cond"]].append(r)
    for cond in CONDS:
        rs = [r for r in by[cond] if "cards" in r]
        tot = Counter()
        for r in rs:
            tot.update(r["n"])
        n = sum(tot.values())
        miss = sum(r["missing"] or 0 for r in rs)
        clean = sum(1 for r in rs if r["missing"] == 0 and not r["n"].get("edit") and not r["n"].get("reject"))
        pct = lambda k: f"{tot[k]} ({100 * tot[k] / n:.1f}%)" if n else "–"  # noqa: E731
        lines.append(f"| {cond} | {len(rs)} | {n} | {pct('accept')} | {pct('edit')} | {pct('reject')} | {miss} | "
                     f"{clean}/{len(rs)} |")
    lines += ["", "Validation during the agent sessions (G and GC tool logs):", "",
              "| Condition | Sessions | Validated at least once | Error caught before a solve: fixed / stopped on | Codes | "
              "Wrong spec that validation passed (solve and clarify tasks) |",
              "|---|---|---|---|---|---|"]
    for cond in ("G", "GC"):
        rs = [r for r in by[cond] if "validation" in r]
        v = [r["validation"] for r in rs]
        codes = Counter(c for x in v for c in x["error_codes"])
        passed_wrong = [r["task"] for r in rs if r["expected"] != "refuse" and r["rules_ok"] and not r["correct_spec"]
                        and not r["refused"]]
        lines.append(f"| {cond} | {len(rs)} | {sum(1 for x in v if x['validations'])} | "
                     f"{sum(1 for x in v if x['fixed'])} / {sum(1 for x in v if x['stopped'])} | "
                     f"{', '.join(f'{k} ×{n}' for k, n in codes.most_common())} | {len(passed_wrong)} "
                     f"({', '.join(t[:3] for t in passed_wrong)}) |")
    (OUT / "cards_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
