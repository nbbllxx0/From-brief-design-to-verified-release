"""Rescore every saved X2 run against the current intended readings (after the author's adjudication).

No LLM and no GPU: each record's final spec and refusal flag are kept; only the scoring fields that depend on
briefs/intended/<task>.json are recomputed (`x2_run.score`). Prints every run whose verdict changed.

    python -B scripts/x2_rescore.py [--dry]        (all conditions, all repeats, both providers)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from x2_run import score  # noqa: E402

KEYS = ("expected", "refused", "schema_valid", "schema_error", "rules_ok", "match", "correct_spec", "behaviour_ok")


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8")
    dry = "--dry" in argv
    ints = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in (ROOT / "briefs" / "intended").glob("*.json")}
    changed = n = 0
    for f in sorted((ROOT / "results" / "X2").glob("**/b*_r[0-9].json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if rec.get("task") not in ints:
            continue
        n += 1
        new = score(rec.get("spec"), ints[rec["task"]], bool(rec.get("refused")))
        before = (rec.get("correct_spec"), rec.get("behaviour_ok"))
        after = (new["correct_spec"], new["behaviour_ok"])
        if before != after:
            changed += 1
            print(f"{f.relative_to(ROOT)}: correct/behaviour {before} -> {after}")
        if not dry:
            for k in KEYS:
                rec.pop(k, None)
            rec.update(new)
            f.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"{n} runs rescored, {changed} changed{' (dry run, nothing written)' if dry else ''}")


if __name__ == "__main__":
    main(sys.argv[1:])
