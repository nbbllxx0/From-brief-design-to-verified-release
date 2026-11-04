# Solver notes (agentto extensions to voxfem, which itself is unchanged)

## 2026-10-30: the coarse levels of padded grids (`solver._PadAware`, `scripts/mg_padding_study.py`)

**Problem.**
- voxfem's multigrid needs every grid side divisible by 2^(levels−1). It pads the grid with zero-modulus elements and
  fixes every node outside the original node box. That keeps the fine operator exact.
- On the coarse levels, though, the coarse nodes just past the real boundary inherit the fixed state. The coarse
  problems then see a clamp along what is really a free surface, and the coarse correction is wrong there.
- Measured on a 256 × n × 128 cantilever, 20 design iterations (`mg_padding.json`):
  - unpadded grids take 11–22 PCG iterations per solve;
  - any padding takes 49–517: 60 → 64 (7% more elements) already costs about 85.
- In the sessions:
  - the X5 bridge, 25 → 32: about 450 iterations;
  - X3 b19's 36M release (200 → 256 and 560 → 576): 1508 s of compute;
  - across the suite check, the slowest tasks are the padded thin parts.

**Fix: `coarse_pad="aware"`, the default for new jobs since 2026-10-30.**
- A coarse node is fixed by padding only if its whole interpolation support lies in the padding; those nodes have no
  stiffness.
- Real supports are coarsened by the base rule with the padding nodes cleared first.
- On a padded grid, the real supports are coarsened by dilation, not injection. Without the padding clamp, small
  supports such as b19's washers vanish under injection, and the coarsest operator is singular: the first try failed
  with NaN curvature, and the coarsest level had no fixed node.
- Only the preconditioner changes: the fine operator, the FP64 true-residual acceptance and the certificate do not. On
  an unpadded grid it is bit-identical.

| Case | Padding | PCG iterations per solve, voxfem rule → aware | Effect |
|---|---|---|---|
| 256 × 60 × 128 | 7% | 83 → 22 | solve 0.67 → 0.21 s |
| 256 × 50 × 128 | 28% | 120 → 36 | 1.00 → 0.33 s |
| 256 × 33 × 128 | 94% | 517 → 43 | 4.26 → 0.36 s |
| 256 × 64 × 128 | none | 14 → 14 | bit-identical |
| b19 (70 × 25 × 40 → 280 × 100 × 160) | 34% | 60–155, growing → 21–28, flat | – |
| **b19 release, 36M, reopen_long warm from X3's own refine design** | 34% | – | **1508 → 212 s compute**, certified; compliance 121.260515 → 121.260507 (−6.5×10⁻⁸) |

- Compliance changes stay within the solve tolerance (at most 1.6×10⁻⁶ relative).
- **Replay contract:**
  - every job records the rule it used (`coarse_pad` in the result);
  - replay re-runs a job with its recorded rule, and logs from before this change record none, so they replay with
    voxfem's rule;
  - X3 b01's log (padded 21%) replays with all 4 solves bit-exact.
- Tests:
  - `tests/test_session.py::test_pad_aware_coarse_rule_cuts_pcg_iterations_and_leaves_unpadded_grids_alone`;
  - 139 pass.
- **The X1, X2 scoring, X3 and X5 timings were measured with voxfem's rule.** Session times on padded grids will
  be shorter now. b19's X3 session (1898 s) would take about 600 s.

### The padding study under both rules (`mg_padding.json`, `mg_padding_aware.json`)

Same setup as the first study:
- a 256 × n × 128 cantilever, 20 design iterations, mg64;
- every level count the 60% padding cap allows;
- the run: `python -B scripts/mg_padding_study.py aware`.

| n | Padding of the level `session_levels` picks | PCG iterations per solve, voxfem → aware | Solve time, voxfem → aware |
|---|---|---|---|
| 64 | none | 14.1 → 14.1 | 0.16 → 0.16 s |
| 60 | 7% | 83.0 → 15.8 | 0.68 → 0.21 s |
| 56 | 14% | 56.9 → 15.8 | 0.48 → 0.20 s |
| 50 | 28% | 128.0 → 19.1 | 1.10 → 0.19 s |
| 48 | none | 16.4 → 16.4 | 0.14 → 0.13 s |
| 40 | 20% | 71.8 → 17.9 | 0.50 → 0.14 s |
| 36 | 33% | 122.9 → 25.1 | 0.86 → 0.19 s |
| 33 | 45% | 428.9 → 39.4 | 2.86 → 0.28 s |

**`session_levels` needs no change.**
- Under the aware rule, its choice (cost model, then the fewest levels within 10% of the cheapest) is the measured
  fastest on all 8 grids.
- The modelled cost now tracks the measured time: for example 0.183 against 0.19–0.21 s at L5.
- Under voxfem's rule the model could not see the padding penalty. Its picks were still the fastest available on 7
  of 8 grids, but those were 2–10× slower.
