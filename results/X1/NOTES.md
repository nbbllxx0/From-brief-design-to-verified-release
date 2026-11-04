# X1 notes

## 2026-10-29: guarded re-run of chain B (113M), `chainB_guarded.json`

With guards on, the guarded rungs were valid:
- k=96 (1.77M elements, cold, mg64): 78 iterations, 7.6 s of compute;
- k=192 (14.2M, warm, mg64): 15 iterations, 5.3 s of compute.

The k=384 rung (113.2M, warm, mg_ir) was **stopped by voxfem's spill guard** ("VRAM oversubscription:
shared-memory spill detected"). At that point other processes held **5.1 GB** of the 24 GB card: the desktop
compositor, the Claude and Codex desktop apps, Chrome, Teams and others. The rung needs about 16.5 GB (146 B per element) plus
workspace. The guard behaved as designed (fail-closed); no invalid timing was recorded.

What a valid 113M timing needs: other processes holding no more than about 2.5 GB. In practice that means closing the
desktop apps that use the GPU, or running from a session with nothing else on the GPU. Until then the release rung defaults
to 65.5M (`mg64`) and 113M stays the "max" rung (PLAN.md §6). The pilot's 31 s for 113M remains an unguarded
feasibility number.

## 2026-10-29: the tail-only warm start does not refine (`warm_arms_pilot.json`). This changes the C3 claim.

What the final phase alone does to an upsampled design (p 4.5, β 32, move 0.05, rmin 1.2 elements): **nothing**.
- With rmin 1.2 the cone filter reaches only face neighbours. Every element of an upsampled 0/1 design then has a filtered
  density of about 0.92 or 0.08, never near 0.5, so the β = 32 projection is saturated everywhere.
- A 0.05 move changes no projected density.
- On a problem without passive regions the volume is already exactly on target, so the run "converges" in 12 iterations
  with **0.000% of elements changed**.
- So the original pilot's "warm 65.5M in 16.6 s, +0.28%" result was the coarse design upsampled and then certified at the
  fine resolution. That is still a valid certified design, but nothing was optimized at the fine level.

On brief b06 (keep-in bosses) the same arm **collapses**:
- The bosses voxelize differently at the two resolutions, so the upsampled design volume is 0.2014 instead of 0.2.
- OC cannot fix a saturated design, so λ runs to infinity.
- Material then drains from every element, compliance grows 17×, and the coarsest factorization hits a NaN.
- This is what the GUI showed on 2026-10-29.

The fix, now the default in `SessionSIMP.warm_start` and the worker:
- one pass of the fine filter on the upsampled design;
- an exact volume match by bisection on a density offset, at the schedule's first β.
- With it, b06 converges in 12 iterations at +1.43% to cold.

The re-opening arm (β 8 → 16 → 32, moves 0.10/0.08/0.05) changes 0.05–0.07% of elements and narrows the gap slightly:
- cantilever: +0.94% vs +1.15%;
- b06: +1.42% vs +1.43%.

What this means for the paper:
- The honest claim is **"certified 10^8 release of a design optimised at a coarser rung, with a measured gap"**. It is not
  "optimised at 10^8 in seconds".
- X1 must report how much each arm changes the design, not only the compliance gap.
- X1 must test arms that really refine:
  - a longer re-opening phase;
  - the physical-units filter (arm d), which keeps the filter footprint and allows finer members;
  - a final full-continuation run at the fine rung as the reference.
- X5 (why 10^8) needs designs in which fine members actually appear. Otherwise the scale argument rests on the
  cold fine runs only.

## 2026-10-30: second guarded try at 113M (`chainB_guarded_run2.log`)

Spilled again at the k=384 rung.
- Others' dedicated VRAM (WDDM counters) was 4.96 GB, although nvidia-smi showed 2.4 GB used just before the run. The guard's
  count is the one that matters, and the desktop apps (Claude, Codex/ChatGPT, Chrome, Teams, Docker, dwm) hold it.
- k=96 cold 78 it / 7.4 s and k=192 warm 16 it / 4.9 s were valid.
- 113M still needs those apps closed.

**Decision 2026-10-30 (author): 113M is skipped.** The paper's release rung is 65.5M (`mg64`; X1 used
26–74M releases, all certified). The pilot's unguarded 113M number is not reported as a result; at most it is mentioned
as a feasibility note with that caveat.

## 2026-10-30: do any warm arms refine? (`refine_pilot.json`, `scripts/x1_refine_pilot.py`)

One doubling each: cantilever and bridge at k40→80, b06 at h 2→1.
- Arms: `src/agentto/escalation.py`.
- "chg" is the fraction of elements whose solid/void state changed from the warm start; b1 = tunnels (Betti number).

| case | arm | it | gap to cold | IoU vs cold | chg | b1 (cold) |
|---|---|---|---|---|---|---|
| cantilever | tail | 12 | +1.15% | 0.944 | 0.000% | 0 (0) |
| cantilever | reopen | 21 | +0.94% | 0.945 | 0.048% | 0 |
| cantilever | **reopen_long** | 61 | **−0.56%** | 0.949 | 0.67% | 0 |
| bridge | tail | 12 | +5.08% | 0.773 | 0.001% | 1 (5) |
| bridge | reopen | 21 | +5.06% | 0.773 | 0.007% | 1 |
| bridge | **reopen_long** | 61 | **+2.24%** | 0.764 | 1.16% | 1 |
| b06 | tail | 12 | +1.43% | 0.956 | 0.033% | 0 (0) |
| b06 | reopen | 21 | +1.43% | 0.956 | 0.137% | 0 |
| b06 | **reopen_long** | 61 | **+0.28%** | 0.970 | 0.73% | 0 |

Findings:
- **`reopen_long` refines.** It restarts at p 3.5, β 4, move 0.15 and then follows the cold schedule. It removes most of the gap:
  - on the cantilever it ends 0.56% below cold;
  - on b06 it ends 0.28% above cold;
  - on the bridge it halves the gap to 2.2%.
  - It does this in 61 iterations instead of 76, starting from a design that is already feasible.
- **`tail` and `reopen` only certify.** They change at most 0.14% of elements, so the coarse topology and its gap stay.
- **New members do not form from a warm start.**
  - The fine cold bridge has 5 tunnels.
  - The coarse design has 1, and every warm arm keeps 1.
  - Members finer than the coarse rung appear only in cold fine runs. This is the X5 argument ("resolution changes the
    answer"), and it bounds what escalation can claim.
- **Physical-units filter arms (rmin × 2) did not converge** in 300 iterations on any case.
  - They cost 80–190 ms per iteration against 50–170 ms, because the filter transpose is about 21× more work at R = 2.
  - Their cold reference differs from the element-unit cold, so they answer a different question (the mesh-independent
    problem). They are left out of the scale runs.
- **Consequence for the GUI policy:**
  - Refine and Release keep the fast tail warm start for interactive use (certifies in about 12 iterations).
  - "Refine properly" (`reopen_long`) is the option that buys compliance. The X1 scale run (`escalation.json`) measures
    what it costs at 8M and 65–74M.

## 2026-10-30: the X1 scale run (`escalation.json`, `scripts/x1_escalation.py`)

Ladder preview (about 1M elements, cold) → refine (8M) → release (26–74M, spill guard on). Each warm arm is chained
from the same arm one rung coarser. Gaps and IoU are against cold at the same rung. IoU is computed on solid masks pooled
to at most 9M elements.

Release rung:

| problem (n) | cold: it / compute | tail: gap, IoU, compute | reopen: gap, IoU, compute | **reopen_long**: gap, IoU, it / compute |
|---|---|---|---|---|
| cantilever (65.5M) | 76 / 118 s | +0.28%, 0.964, 8 s | −0.15%, 0.964, 31 s | **−1.20%**, 0.960, 61 / 81 s |
| bridge (65.5M)\* | 76 / 249 s | +1.39%, 0.888, 27 s | +0.85%, 0.888, 50 s | **−1.19%**, 0.882, 61 / 127 s |
| torsion (65.5M) | 76 / 121 s | +0.86%, 0.955, 8 s | +0.79%, 0.955, 29 s | **+0.07%**, 0.965, 61 / 72 s |
| b05 L bracket (25.6M) | 76 / 251 s | +0.005%, 0.863, 8 s | −0.83%, 0.864, 58 s | **−3.06%**, 0.859, 61 / 128 s |
| b06 motor mount (73.7M) | 76 / 424 s | +1.22%, 0.959, 33 s | +0.95%, 0.959, 107 s | **+0.03%**, 0.955, 61 / 258 s |

Every release run converged and is certified.

\*The bridge cold reference at 65.5M failed twice with `mg_ir`: `SolveFailed` "ir_lean: correction rejected" at
iteration 15, with both the auto and dilate settings. It converged with `mg64` + dilate (`x1_fix_cold.py`,
`fix_cold_bridge.log`). The warm bridge arms ran on `mg_ir` without trouble. So the failure is in `mg_ir` on the gray
early iterations of a cold run.

Findings:
- **`reopen_long` matches or beats cold on all five problems.** It ends between −3.1% and +0.07% of cold, against the
  tail warm start's +0.005% to +1.39%.
  - It costs 51–69% of the cold compute at the release rung.
  - Counting the refine-rung `reopen_long` as well, it costs 59–80%.
  - The designs differ from cold: IoU 0.86–0.97, and the cold runs have more tunnels at release (bridge b1 18 against
    5–9; b05 6 against 3).
  - So escalation reaches a different local optimum that is as good or better, not cold's design faster.
- **`tail` remains certify-only** at 8–74M (0.000% changed). Its gap is whatever the coarse rung's gap was, up to +1.4%.
- **The certified 74M release (b06)** takes 424 s cold, 258 s with `reopen_long` and 33 s with the tail warm start.
- **b09 (duct bridge) is excluded** (`b09_partial.json`):
  - its cold preview did not converge in 300 iterations;
  - at the refine rung the cold design is a lattice (b1 953) and costs 9 s per iteration at 8.6M (b06: 0.8 s);
  - tail diverged (compliance ×400, 300 iterations), and `reopen_long` did not converge;
  - the 69M cold release failed. The crash lost its error message; the script now records failures instead of crashing.
  - A pressure load over the whole top face at 25% volume, with 3 mm members, is a lattice problem. Treat b09 as a
    stress case for the solver, not an escalation benchmark.
- **Consequences for the system:**
  - The Release button should offer `reopen_long` ("Release, optimised") next to the fast certified tail release.
  - A release `SolveFailed` on `mg_ir` should retry on `mg64` when the grid fits.
