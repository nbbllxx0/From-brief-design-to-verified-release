"""Which contested follow-up specifications state the same problem (benchmark comparator, agentto.bench.equivalent),
so the adjudication sheet can point a repeated reading to its first row. No GPU, no LLM.

    ~/anaconda3/envs/agentto/python.exe -B scripts/followup_equivalence.py

Writes paper/adjudication/followup/equivalence.json: {"<record>": "<record of the first equivalent row>"} and
earlier_verdicts.json: {"<record>": {"record", "verdict", "comment"}} for readings equivalent to a contested record of the
31 October adjudication; both read by scripts/make_adjudication_pack_followup.py (same candidate order).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from agentto.bench import equivalent  # noqa: E402
from make_adjudication import CONDS  # noqa: E402
from make_adjudication_pack_followup import OUT, candidates  # noqa: E402

SUB = {lab: sub for sub, lab in CONDS}


def same(a, b, what):
    try:
        return bool(equivalent(a, b).get("same"))
    except Exception as ex:                              # noqa: BLE001
        print("compare failed", what, repr(ex)[:120])
        return False


def main():
    dec = json.loads((ROOT / "paper" / "adjudication" / "adjudication_decisions.json").read_text(encoding="utf-8"))
    judged = {}
    for c in dec["rows"]["contested"]:
        rec = f"results/X2/{SUB[c['Condition']]}/{c['Task']}_r1.json"
        spec = json.loads((ROOT / rec).read_text(encoding="utf-8")).get("spec")
        judged.setdefault(c["Task"], []).append((rec, spec, c["Verdict (W/A)"], c.get("Comment", "")))
    first, out, earlier = {}, {}, {}
    for t, lab, sub, rep, d in candidates():
        rec = f"results/X2/{sub}/{t}_r{rep}.json"
        for jrec, jspec, verdict, comment in judged.get(t, []):
            if jspec and same(d["spec"], jspec, f"{rec} vs {jrec}"):
                earlier[rec] = {"record": jrec, "verdict": verdict, "comment": comment}
                break
        for prev_rec, prev_spec in first.get(t, []):
            if same(d["spec"], prev_spec, f"{rec} vs {prev_rec}"):
                out[rec] = prev_rec
                break
        else:
            first.setdefault(t, []).append((rec, d["spec"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "equivalence.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (OUT / "earlier_verdicts.json").write_text(json.dumps(earlier, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} rows repeat an earlier row's reading")
    for k, v in out.items():
        print(" ", k, "=", v)
    print(f"{len(earlier)} rows repeat a reading judged on 31 October")
    for k, v in earlier.items():
        print(" ", k, "=", v["record"], v["verdict"])


if __name__ == "__main__":
    main()
