# Adjudication and rescore (2026-10-31, final)

Adjudicator: J. Wang (author). Input: `ADJUDICATION_RESULTS_2026-10-31.txt` (all 50 briefs, all 41 contested runs).
Encoded into `ADJUDICATION_SHEET_2026-10-31.xlsx` by `scripts/fill_sheet_from_txt.py`; applied by
`scripts/apply_adjudication.py` (record: `adjudication_decisions.json`); every brief's verdict is also stored as
`"adjudication"` in `briefs/intended/<task>.json` and printed with the brief texts in the supplement. All 672 saved runs
rescored by `scripts/x2_rescore.py`; `rescore_2026-10-31.log` lists the 27 runs that changed relative to before
adjudication (all wrong to right). Ledger, tables, figures, X2 summary, card replay (`x6c_cards.py`), Figure 1 and the
graphical abstract regenerated. Backup of every input and output before adjudication:
`paper/archives/pre_adjudication_20261031/`.

The preliminary verdicts (`ADJUDICATION_RESULTS_PRELIMINARY_2026-10-31.txt`) were applied first and then fully replaced
by the final ones; they differ on two runs (#04 b39 G and #18 b36 Claude T, both A -> W).

## Applied

- Contested: 19 A became admissible readings; 22 W stand.
- Briefs: 49 intended readings confirmed. b43 intended support corrected to the two bottom strips beside the coolant
  channel; by author decision (2026-10-31) the full-face reading stays admissible, because the brief says "bolted ...
  over its full area" and the extra fixed nodes touch only void elements. No b43 run changed.
- b02: the admissible reading that also fixes x withdrawn (counts false). No run changed.
- Extra readings written in words (b03, b07, b09, b19, b25, b30, b31, b33, b39, b41, b48): each is the spec of an
  accepted contested run, so no hand encoding was needed.
- Sources: family confirmed for b02, b11, b12, b19, b21, b29 (b19 and b29 as adaptations); not traced for b01, b03,
  b04, b05, b13, b25; 13 generic family descriptions treated as engineering examples. b12's source label restored to
  its original wording (the adjudicator confirmed it); b19 labelled "adapted from".
- Complete-session (X3) release specs all matched the unchanged intended readings; the 14/15 claim stands.

## Before / after (repeat 1 unless stated)

| Claim | Before | After |
|---|---|---|
| Briefs with credited alternative readings | 5 | 17 |
| Codex right T / G / G+C / G+C+V | 37 / 42 / 50 / 50 | 37 / 44 / 50 / 50 |
| Codex silent wrong T / G / G+C | 1 / 8 / 0 | 1 / 6 / 0 |
| McNemar T -> G (behaviour) | 7 / 2, p = 0.18 | 8 / 1, p = 0.039 |
| McNemar T -> G (spec) | 10 / 2, p = 0.039 | 11 / 1, p = 0.0063 |
| McNemar G -> G+C | 8 / 0, p = 0.0078 | 6 / 0, p = 0.031 |
| Codex G repeat 2 / T repeat 2 | 40 / 35 | 42 / 37 |
| Claude (high) T / G+C / G+C+V | 33 / 42 / 43 | 42 / 46 / 47 |
| Claude G+C silent wrong | 8 | 4 |
| Provider McNemar G+C / G+C+V / T | 0.0078 / 0.016 / 0.29 | 0.12 / 0.25 / 0.18 |
| Claude (default effort, 20 briefs) G+C | 14/20, p = 0.031 | 15/20, p = 0.062 |
| G wrong specs that passed every rule (C4.2) | 8 | 6 |
| G+C cards accepted unchanged (C6.3) | 423/423 | 423/423 |

## Effect on the paper's claims

- Clarification still fixes every remaining grounding-only misreading (6 fixed, 0 broken, p = 0.031) and reaches
  50/50. Grounding over text becomes significant (p = 0.039).
- The provider difference in right outcomes is not significant (G+C p = 0.12): the second provider asked 7 questions
  against 59 and kept 4 silent wrong specs; the paper states this descriptively.
- The "pending human adjudication" hedges are replaced by the statement that one author adjudicated; the threats
  section names single-author, non-independent adjudication and the six untraced sources, and the next steps ask for an
  independent second adjudication.
- Not rerun: consequence-weighted error for the b43 runs against the corrected intended support (the adjudicator's
  estimate of the consequence is about 1e-10).
