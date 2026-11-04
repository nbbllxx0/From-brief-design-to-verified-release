# Human adjudication: what to check, and how the results update

**Package to fill:** `adjudication/AgenticTO_adjudication_pack_<date>.zip` (built by
`scripts/make_adjudication_pack.py`). It holds one Excel sheet, `ADJUDICATION_SHEET.xlsx`, with dropdowns in the yellow
columns, plus the briefs, the intended specs and one evidence file per contested run. Return the filled sheet; it is
read back by `scripts/apply_adjudication.py`. The sections below explain the checks. The CSV and Markdown files in
`adjudication/` are the same content in plain form.

Every X2 number rests on the intended readings of the 50 briefs, which were drafted by Claude from the brief texts and
the benchmark sources. You are the adjudicator. Nothing here needs code: mark the sheets, and I rerun the scoring.

## 1. The briefs and their intended readings (`adjudication/adjudication_briefs.csv`)

One row per brief, opened in Excel. Columns:
- `brief`: the text the agents saw.
- `intended`: the intended reading in short form (domain, volume fraction, material, every support, load and passive
  region). The full spec is `briefs/intended/<task>.json`.
- `oracle`: for clarify tasks, the scripted answers to the questions an agent may ask.
- `admissible`: other readings that also count as right.
- `source`: "engineering" or the published benchmark the brief restates.
- `R` … `Claude GCV`: each condition's outcome (right / WRONG / refused / invalid).
- **For you:** `verdict_intended_ok` (Y/N), `verdict_source_ok` (Y/N for benchmark citations), `verdict_comment`.

What to look for:
1. Does the intended reading follow from the brief, as an engineer would read it? Watch units, the up axis, which
   face, load direction and sense, whether a "pad" or "plate" is a passive solid, and whether the volume fraction is
   of the design volume (our convention).
2. For solve tasks: is the brief fully stated, so that one reading is right? If not, it should be a clarify task or
   gain an admissible reading.
3. For clarify tasks: is the ambiguity real, and are the oracle's answers what the brief's author would say?
4. For refuse tasks (b08, b49, b50): is the brief really unsolvable as stated?
5. Benchmark rows: is the cited source right (`source` column)?

## 2. Contested cases (`adjudication/adjudication_contested.md`)

41 runs scored as silent wrong specs, across all conditions and both providers. Each row shows the fields that differ
from the intended reading and the agent's own closing note. Mark **W** (truly wrong: an engineer would get a different
part) or **A** (acceptable). The ones that most affect the paper:
- Codex G's 8 (b07, b25, b39, b44–b48): C2.3 "G 8 silent wrong".
- Claude Code G+C's 8 (b07, b09, b10, b13, b30, b37, b44, b48): C2.6. In particular:
  - b13 and b37: Claude worked in kN and mm but left E in MPa (1000× too stiff). The design is identical (compliance
    scales with 1/E) but the spec is wrong. My scoring says W; you may disagree.
  - b09 and b30: the roller is left free across the width (fixes y only); the briefs free only the span direction.
    My scoring says W.
- Claude T's 17: mostly the same kinds.

## 3. After your verdicts

1. `scripts/apply_adjudication.py <filled sheet>` adds each **A** reading to `admissible` in
   `briefs/intended/<task>.json` and records every decision in `adjudication/adjudication_decisions.json`.
2. It lists the briefs marked N and the extra admissible readings written in words; Claude encodes those as specs.
3. I rerun: `scripts/x2_rescore.py` (rescores all 672 saved runs against the edited readings; no LLM, no GPU;
   it reproduces today's verdicts exactly), then `scripts/make_ledger.py`, `scripts/make_tables.py`,
   `scripts/make_figures.py`, `scripts/make_adjudication.py`, and `scripts/x2_score.py` for the consequence-weighted
   error of any run whose verdict changed (GPU). Only the X2-derived numbers change (C1.5, C2.x, C4.2, C6.3, the X2
   tables and figures).
4. Record the adjudication in the supplement: who adjudicated, how many readings changed, and the before/after counts.
