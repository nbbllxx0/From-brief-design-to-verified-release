# X2 notes

## 2026-10-30: draft run on b01–b10 (one repeat, Codex `gpt-6.1-sol` high, subscription CLI)

`scripts/x2_run.py`, results in `results/X2/<cond>/`. Scored on what the solver sees (`bench.equivalent`), against the
**unadjudicated** intended readings. This is a shakedown of the harness, not the X2 result: 10 tasks, one repeat.

| Condition | Correct spec | Right behaviour | Wrong | Questions | Tool calls (failed) | Input tokens, median (cached) | Wall, median |
|---|---|---|---|---|---|---|---|
| T (text only) | 5/10 | 6/10 | b04 b06 b07 b10 (+ invalid JSON on b06) | – | – | 21k (53%) | 10 s |
| G (tools) | 6/10 | 6/10 | b04 b06 b07 b10 | 0 | 92 (0) | 87k (84%) | 27 s |
| G+C (tools + clarification, oracle) | **9/10** | **9/10** | b04 | 18 | 103 (1) | 125k (86%) | 48 s |
| G+C+V (+ picture check) | 9/10 | 9/10 | b04 | 20 | 109 (1) | 155k (78%) | 66 s |

- "Right behaviour" means a correct spec on solve/clarify tasks, and a refusal on b08 (an infeasible rail bracket).
- All G and G+C session logs replay with no LLM.
- Clarification fixes:
  - the two "clarify" briefs, b07 (sensor arm) and b10 (hook plate);
  - b06, where G spent the 20% budget on the whole envelope; the agent asked about it under G+C.
- The picture check (V) flagged only b04 and triggered one repair. That repair did not reach the intended reading, so V
  added nothing on this set.
- The first G+C pass ran before an oracle fix (`results/X2_GC_preoraclefix.log`): options written with `set_params`
  names, such as `clear_min_feature`, were not recognised. It also scored 9/10. The table is the rerun.

### For the author's adjudication (`briefs/intended/`)

1. **b04 (torsion) is ambiguous; consider changing it to expected = "clarify".** The brief does not say which axis is
   "up", or whether the 12 mm lugs sit on the edges or just inside them.
   - Every condition took z as up and kept the lugs inside the face: lug centres 6 mm in, so 588 N on a 68 mm arm.
   - The intended reading has y up and the lugs centred on the edges: 500 N on 80 mm.
2. **"20% material" when there are keep-in solids.** The schema defines `volume_fraction` over the design (non-passive)
   volume.
   - G on b06 and b04 converted the brief's percentage as a share of the whole envelope (0.20 → 0.18).
   - The intended reading applies it to the design volume.
   - Decide which reading the briefs mean, and either say it in the briefs or admit both.
3. **Admissible alternatives that count** (b04 sign swapped, b06 square washers) are written as text. Turn them into
   alternative specs so the scorer can credit them.

The consequence-weighted error is in the next section.

## 2026-10-30: consequence-weighted error (`scripts/x2_score.py`, `summary.json`)

What each wrong spec costs: the predicted spec is optimized at the intended spec's preview rung, and the resulting
design is evaluated under the intended spec.

CWE = C_intended(ρ_pred) / C_intended(ρ_intended) − 1. Equivalent specs score 0; a run with no valid spec scores ∞.
b08 is a refuse task, scored on behaviour only.

| Condition | CWE = 0 | Wrong runs and their CWE |
|---|---|---|
| T | 5/9 | b04 **+0.27%**, b07 +99%, b06 ∞ (invalid JSON), b10 ∞ (no valid spec) |
| G | 5/9 | b06 +5.5%, b10 +7.3%, b07 +173%, b04 **2.2×10⁸** (no load path: the design is void where the intended loads act) |
| G+C | 8/9 | b04 ∞: the agent **refused** after the oracle rejected all 3–5 readings it offered |
| G+C+V | 8/9 | b04 ∞: same refusal |

Findings:
- **Spec accuracy and consequence are different scales.**
  - T's wrong b04 is almost harmless (+0.27%).
  - G's wrong b04 is a design that does not carry the intended load at all.
  - So X2 should report CWE in bands, not as a mean; the G mean is 2.4×10⁷. Suggested bands:
    - 0;
    - up to 1%;
    - 1–10%;
    - above 10%;
    - no load path (CWE above 10³);
    - no spec.
- **Clarification turns silent errors into safe failures.**
  - Under G+C the agent's b04 readings did not include the intended one (y up, lugs centred on the edges).
  - The oracle answered "none of these", and the agent refused instead of guessing.
  - G guessed and produced a design with no load path.
  - Count a refusal as a non-silent failure, separately from a wrong design.
- **The b04 oracle script** gives no fact that would let the agent recover the intended reading. That is part of
  adjudication item 1 above.
- G b07 printed an OC overflow warning (λ overflow) while it optimized the predicted spec; the run still converged.

## 2026-10-30: adjudication items resolved, suite draft 2 (b01–b50)

The author decided item 2 and delegated items 1 and 3 ("pick the optimal for the paper").
1. **b04 is now fully stated, and its old wording became the clarify task b44.**
   - b04 now says y is up, the 16 mm lugs are flush with the top and bottom edges, and the torque is about +x. The
     intended spec is a 625 N couple on a 64 mm arm.
   - b44 keeps the draft-1 wording, with oracle facts that recover the intended reading. The two tasks pair the same
     problem stated fully and stated ambiguously, which isolates the effect of clarification on one geometry.
   - The draft-1 b04 results above refer to the old wording.
2. **Volume fraction = share of the design (non-passive) volume**, the common convention of commercial TO tools and the
   schema's meaning. Every brief now says "volume fraction 0.3" instead of "30% material".
3. **Admissible readings that count now carry specs** (b02 pinned end support, b06 square washers, b44 opposite torque
   sense). `bench.match_task` credits them; their CWE is measured, not set to 0.
   - b06's "bolt holes as keep-out cylinders" reading was dropped: a keep-out inside a keep-in boss conflicts (R007).

Scorer changes:
- Specs are compared in mm and N (`bench.normalized`). The resolution matches on the resulting grid, so `h` and
  `cells_short_edge` can both give the same one.
- The oracle reads option items and params in the units of the session's draft.
- `scripts/x2_score.py` compares in mm and N, and measures CWE for admissible matches.

Suite draft 2 (`briefs/README.md`):
- 25 restated benchmark families and 25 engineering briefs;
- 40 solve, 7 clarify, 3 refuse;
- five unit systems, four z-up tasks, half and quarter symmetry models, cells-based resolution, and min-feature limits.

The draft-1 X2 numbers above are a pilot on the old wording. The X2 batch reruns all 50 tasks.

## 2026-10-30: GPU check of suite draft 2 (`scripts/validate_suite.py`, `suite_check.json`)

Every non-refuse intended spec (47) was optimized cold at its preview rung (0.25–1.7M elements), in mm and N. That is
the reference x2_score uses.
- **41 converge** in 76–82 iterations.
- **6 reach the 300-iteration cap with a settled compliance:**

  | Task | Spread of the last compliance samples |
  |---|---|
  | b03 | 6×10⁻⁷ |
  | b09 | 5×10⁻⁸ |
  | b22 | 5×10⁻⁷ |
  | b24 | 1.9×10⁻⁶ |
  | b26 | 1×10⁻⁸ |

  - All carry a uniform load over a whole design face. The elements under the loaded face settle into a period-2
    oscillation that the change criterion never accepts, while the compliance moves by less than 2×10⁻⁶.
  - Their references are good to well below the 1% CWE band.
  - b24 with a one-element solid skin under the load converges in 82 iterations.
- **b13 and b30 now carry a solid load plate.** Their capped compliance had not settled (spread 0.15% and 1.2%). The
  plate (a 4 mm end plate on b13, a 4 mm deck on b30) is the usual engineering form, and both now converge, in 79 and
  76 iterations.
- **System fix found by b22.** b22's uniform top load shares its wall edge with the support, so 1.3% of the load sits
  on fixed DOFs.
  - voxfem kept that share in the right-hand side, and neither linear solver could reduce it: mg64 stopped with a
    true residual of 8×10⁻² after 20,000 iterations, and mg_ir's correction was rejected.
  - `problem.build` now drops load components on fixed DOFs and records the dropped share, which rule R005 reports.
    b22 then solves.
  - X0 identity still holds (137 tests pass).
- **Possible rule** (not added): R014 info, "a uniform load over design elements: consider a solid skin". Agents would
  then propose the plate instead of meeting the oscillation.

## 2026-10-30: X2 batch on suite draft 2 (50 tasks × 4 conditions, Codex gpt-6.1-sol high, one repeat)

Runs: `results/X2/<cond>/`, logs `run_<cond>.log`; consequence scores `score.log`, `summary.json`. Every G, GC and GCV
session log replays with no LLM. The draft-1 pilot is in `results/X2_draft1/`.

| Condition | Correct spec | Right behaviour | solve | clarify | refuse | Questions (on solve tasks) | Tool calls (failed) | Input tokens, median | Wall, median |
|---|---|---|---|---|---|---|---|---|---|
| T, text only | 34/50 | 37/50 | 33/40 | 1/7 | 3/3 | – | – | 20k | 10 s |
| G, tools | 42/50 | 42/50 | 38/40 | 1/7 | 3/3 | 0 | 432 (1) | 86k | 28 s |
| G+C, + clarification | **50/50** | **50/50** | 40/40 | 7/7 | 3/3 | 59 (45) | 520 (2) | 104k | 37 s |
| G+C+V, + picture check | 50/50 | 50/50 | 40/40 | 7/7 | 3/3 | 59 (45) | 524 (2) | 135k | 49 s |

Consequence of the wrong specs (CWE bands, over the 47 non-refuse tasks):

| Condition | CWE 0 | ≤ 1% | > 10% | No load path | No valid spec | Stiffer, but uses more material |
|---|---|---|---|---|---|---|
| T | 34 | 0 | 0 | 0 | 12 | 1 (b47: guessed volume fraction, mass +33%, CWE −26%) |
| G | 39 | 1 (b45) | 1 (b07, +164%) | 1 (b44, CWE 1.1×10⁸) | 0 | 5 (invented keep-ins: mass +0.2–10%, CWE −0.2% to −2.5%) |
| G+C | 47 | 0 | 0 | 0 | 0 | 0 |
| G+C+V | 47 | 0 | 0 | 0 | 0 | 0 |

Findings:
- **Tools fix the representation errors; clarification fixes the reading errors.**
  - T's misses on solve tasks are all invalid JSON: washers written as zero-length cylinders on 7 tasks, and a
    zero-length cylinder load on b40. The tools remove these.
  - G's two solve misses (b25, b39) are **over-specification**: the agent invented solid pads (a bearing plate, washer
    bosses) that the brief never asked for. They count as keep-in regions outside the material budget.
  - G+C gets every task right.
- **CWE needs a material column.** A wrong spec can look stiffer than the intended design when it spends more
  material. All six negative CWEs have a mass ratio above 1 (`cwe.mass_ratio` in each record), so CWE < 0 is a
  material-budget violation, not an improvement.
- **The b04/b44 pair works as designed.**
  - Every condition, T included, solves b04 (fully stated).
  - Under G, b44 (the same problem with the ambiguous wording) is read the wrong way, and the design has no load path.
  - Under G+C, the agent asks and recovers the intended reading.
- **Safe and silent failures.** T, which cannot ask, refused 5 of the 7 clarify tasks and named what was missing. G
  never refused a clarify task: it guessed. The table counts both as wrong, but a refusal is a safe failure; report it
  separately.
- **b45's gap is in the direction only.** G guessed the pump load's size wrong and its direction right, and CWE is 0:
  a compliance design depends on the load's direction, not its size.
- **Clarification costs questions.**
  - G+C asked 59 questions: 14 on the 7 clarify tasks (each was asked about), and 45 on 26 of the 40 fully stated
    tasks (up to 6 on b31).
  - This is about one question per fully stated task. The paper should report it as user burden, beside the accuracy.
- **The picture check (V)** flagged one task (b32). The repair kept the correct spec, so V adds nothing on this suite
  beyond G+C, at 30% more tokens.
- **Scoring bug found and fixed during the run.** Agents sometimes write numbers as strings ("400"). The new
  unit-aware oracle multiplied those before converting them, so 22 G+C sessions met "no answer".
  - The agents carried on with assumptions and still scored 49/50.
  - Both clarification conditions were rerun after the fix; the invalid runs are kept in `results/X2_oraclebug/`.
  - A test covers the case now.

Still to do for X2:
- repeats (3 per cell) for confidence intervals;
- the Claude Code provider run;
- the rule-parser baseline named in experiments.md.

## 2026-10-30: rule-parser baseline R and silent failures (`scripts/x2_rules_baseline.py`, `results/X2/R/`)

R is a deterministic regex parser with no LLM, the plan's condition R.
- It reads envelopes, axes, units, material, volume fraction, mesh size, whole-face supports, pad, face and edge loads,
  and box keep-in/keep-out regions.
- It refuses anything else and never guesses.
- It was written after the suite existed, so 6/50 is an optimistic bound for a rule parser.
- Its three right "refusals" on b08, b49 and b50 come from unreadable phrasing, not from seeing the infeasibility.

Repeat 1, all conditions:

| Condition | Correct spec | Silent wrong spec | Refused a solvable task | Invalid output |
|---|---|---|---|---|
| R | 6 | 0 | 41 | 0 |
| T | 34 | 1 (b47) | 5 | 7 |
| G | 42 | **8** (b07 b25 b39 b44 b45 b46 b47 b48) | 0 | 0 |
| G+C | 50 | 0 | 0 | 0 |
| G+C+V | 50 | 0 | 0 | 0 |

A silent failure is a schema-valid spec that differs from the intended reading, delivered without a refusal.
- **Tools without clarification raise the silent-failure rate from 1 to 8.** The tools make every output valid and
  confident, so the errors that remain are guesses on the clarify tasks and invented keep-ins.
- **Clarification removes them all.**
- **R is safe but nearly useless:** it refuses 41 solvable tasks.

This is the paper's X2 argument: grounding tools and clarification are complementary, and either one alone fails in a
different way.

## 2026-10-30: repeats and paired tests

Repeats, on the plan's lean design:
- 20 briefs (all 7 clarify, all 3 refuse, 10 solve, including G's misses);
- T: 3 repeats; G+C and G+C+V: 3 repeats each;
- T and G also have a second repeat on all 50 (it ran before the batch was cut to the plan).

| Condition | Right behaviour, 20-brief subset, per repeat | Questions per repeat | Tasks that change between repeats | All 50, repeat 2 |
|---|---|---|---|---|
| T | 10, 9, 9 | – | b44 | 35/50 (r1 37) |
| G | 12, 11 | – | b10 | 40/50 (r1 42) |
| G+C | 20, 20, 20 | 34, 32, 34 | none | – |
| G+C+V | 20, 20, 20 | 34, 32, 34 | none | – |

Run-to-run variation:
- T and G vary by one or two tasks, always on clarify tasks, where they guess.
- G+C is 60/60 over three repeats, and its question count varies by 2.

Paired exact McNemar tests on right behaviour (repeat 1, 50 tasks; "fixed" = wrong under the first condition and right
under the second):

| From → to | Fixed | Broken | p |
|---|---|---|---|
| R → T | 28 | 0 | 7.5×10⁻⁹ |
| T → G | 7 | 2 | 0.18 |
| G → G+C | 8 | 0 | 0.0078 |
| T → G+C | 13 | 0 | 0.00024 |
| G+C → G+C+V | 0 | 0 | 1 |

- Tools alone are not a significant gain over text: they fix 7 tasks and break 2. On b25, G added an invented bearing
  plate. On b44, T happened to pick the intended reading and G the other.
- Clarification is: it fixes 8 of G's errors and breaks none.
- The picture check adds nothing measurable on this suite.

## 2026-10-30: generality check on a second provider, Claude Code (`results/X2/claude/`)

Setup:
- the plan's lean design: the same 20 briefs as the repeats (all 7 clarify, all 3 refuse, 10 solve);
- conditions T, G+C and G+C+V;
- repeat 1;
- the Claude Code CLI on the subscription (model `claude-opus-5-5`, the CLI default);
- the same tools, oracle and scoring as Codex.

The 50-brief × {T, G+C+V} plan was cut to these 20 to spare the user's plan quota.

| Condition | Codex gpt-6.1-sol: right / silent wrong | Claude Code: right / silent wrong | Questions (Codex / Claude) | Median input tokens | Median wall |
|---|---|---|---|---|---|
| T | 10 / 1 | 8 / 11 | – | 20k / 9k | 10 / 11 s |
| G+C | **20 / 0** | **14 / 6** | 34 / 6 | 121k / 44k | 40 / 22 s |
| G+C+V | 20 / 0 | 14 / 6 | 34 / 7 | 141k / 64k | 53 / 35 s |

Paired exact McNemar, G+C, Codex vs Claude on the same 20 briefs: 6 discordant, all in Codex's favour (p = 0.031).

**Claude's six G+C misses** (b04, b07, b10, b22, b44, b48):
- The harness worked. Claude called `ask_clarification` twice and the oracle answered both times.
- It asked about one thing and then guessed the rest, stating each assumption in its final message:
  - b07: weight along −z, where the intended reading is −y;
  - b10: the load direction;
  - b04, b44, b48: keep-in lugs or a boss "my addition; the brief doesn't ask for them";
  - b22: the load moved off the wall row of nodes (to avoid R005), which changes its distribution.
- G+C+V's picture check caught none of them: the renders match what Claude built.
- Four of the six are clarify tasks, so these are the silent errors clarification exists to prevent. The other two
  (b04, b22) are over-specification of fully stated briefs.

**What this means for the paper.**
- The grounding and clarification tools are provider-independent. Their benefit is not: it depends on the model
  being willing to ask.
- Codex asked 34 questions on these 20 briefs, 1.7 per brief (over-asking on stated tasks). Claude asked 6 and guessed
  instead.
- So C2 should be stated as "clarification with an oracle removes silent errors when the agent asks", and the
  provider difference reported as a finding.
- An obvious follow-up, not run: an instruction that makes asking mandatory before adding unrequested items or
  choosing a gravity direction.

## 2026-10-31: Claude Code at effort high on all 50 briefs (`results/X2/claude_high/`)

The default-effort run above left two differences from Codex: Codex runs at effort high, Claude at the CLI default,
and Claude saw only 20 briefs. This run removes both.

Setup:
- `claude-opus-5-5` with `--effort high` (recorded in every record as `effort: high`), Claude Code 2.1.312;
- all 50 briefs × T, G+C, G+C+V, repeat 1;
- the same tools, oracle and scoring.

Harness fixes:
- `run_claude` and `x2_run.py --effort` were added.
- The first G+C+V pass ran its picture check at the default effort (a patch that did not apply) and crashed on b49,
  where G+C refused. Both are fixed, and G+C+V was rerun; the flawed pass is in
  `results/X2_claude_high_gcv_verifydefault/`.

| Condition | Codex (high): right / silent wrong | Claude (high): right / silent wrong | Questions (Codex / Claude) | Median input tokens | Median wall |
|---|---|---|---|---|---|
| T | 37 / 1 | 33 / 17 | – | 20k / 9k | 10 / 11 s |
| G+C | **50 / 0** | **42 / 8** | 59 / 7 | 104k / 44k | 37 / 20 s |
| G+C+V | 50 / 0 | 43 / 7 | 59 / 8 | 135k / 59k | 49 / 33 s |

Paired exact McNemar, Codex vs Claude on the same 50 briefs:
- G+C: 8 discordant, all in Codex's favour (p = 0.0078);
- G+C+V: 7 discordant, all in Codex's favour (p = 0.016);
- T: 6 / 2 (p = 0.29), no evidence of a difference.

**Effort is not the cause.**
- On the earlier 20 briefs, G+C went from 14 to 15 right (b04 and b22 now right, b09 now wrong).
- Claude asked 7 questions in 50 briefs at high effort, against Codex's 59.

**Claude's eight G+C misses:**
- 4 clarify tasks where it guessed instead of asking, and said so:
  - b07: gravity along −y without asking which axis is up;
  - b10: the load at one of two mirror corners;
  - b44: the torque sign;
  - b48: the pull direction, plus a solid boss of its own.
- 2 unit slips: it chose kN and mm but left E in MPa, so E is 1000× too stiff (b13, b37). The design is unchanged,
  because compliance scales linearly with E, but the spec is wrong.
- 2 rollers left free across the width (fix y only, where the brief only frees x), on b09 and b30. b09 was right at
  default effort, so this is run-to-run variation.

The picture check (G+C+V):
- flagged 2 specs and repaired both: b44 became right, and b45 stayed right;
- caught none of the others, because their renders match what Claude built.

**Consequence for the paper:** unchanged from the default-effort run, now on the full suite with matched effort.
- The tools and protocol carry over to a second provider.
- The benefit of clarification depends on the model asking.
- Report C2.6 as a provider finding, not as generality of the headline number.
