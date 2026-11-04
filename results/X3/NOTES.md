# X3 notes: brief → certified release, by the agent alone

`scripts/x3_run.py`: Codex `gpt-6.1-sol` (high, subscription CLI), MCP profile `full`, scripted oracle for questions,
own GPU worker. Each result is `results/X3/<task>.json`, and the session (log, fields, exports) is in `results/X3/<task>/`.

## 2026-10-30, first runs

**b06 motor mount: released.**
- 176 s from the first tool call to the STL and report: 201 s wall including CLI start-up.
- 36 tool calls, 0 failed, 4 clarification questions.
- Released spec equals the intended reading on every field.
- Ladder:
  - sketch 144k, cold: 80 iterations, 6.4 s;
  - preview 1.15M: 12 iterations, 2.0 s;
  - refine 9.2M: 12 iterations, 10.4 s;
  - release 73.7M: 12 iterations, 39.4 s wall, **certified**.
- Export: STL with 1.6M triangles, closed and manifold, in 1.8 s with the report.
- Replay without the LLM: ok, 5 solves bit-exact.
- 549k input tokens (97% cached).

**b01 cantilever block, run 1: not released** (kept as `b01_cantilever_block_run1*`). The run found three system
problems, now fixed:
1. **The refine rung (2M elements) cost 4.7 s per iteration.**
   - Cause: the 200×100×100 grid stops coarsening at a 15,288-DOF dense coarsest level.
   - The padding that fixes it adds 30% more elements, more than `session_levels` allowed (25%).
   - The agent's 45 s waits timed out, so it polled `status` 20 times and cancelled twice.
   - Fix: `rules.session_levels` now picks the padding by a measured cost model (`iter_cost`), with padding up to 60%.
   - Measured after the fix: 0.45 s/iteration at 2M and 2.6 s/iteration at 16M (`scripts/levels_check.py`).
   - Grids that already met the target choose the same levels as before.
2. **Release rung vs the brief's "mesh at 1 mm".**
   - The ladder's release rung is 0.5 mm, which is finer than the spec's h.
   - The agent called `solve(rung="release", h=1)`. The h won, so the result was not a release.
   - The agent then stopped, asking which mesh governs.
   - Fix: `ladder` now explains that the spec's h is a base mesh and the release rung may be finer.
   - Fix: `solve` refuses a rung and h that disagree.
3. **Replay compared a solve that outlived its recorded `timeout_s`.**
   - Fix: replay now waits for any solve that finished in the log.

**b01 runs 2 and 3, after the fixes: released.**

| run | wall (first call → report) | calls | cancels | status polls | input tokens | release (16M, certified) |
|---|---|---|---|---|---|---|
| 1 | 957 s, not released | 46 | 4 | 20 | 1.96M | – |
| 2 (levels, rung, replay fixes) | 344 s | 29 | 2 | 7 | 0.90M | 61 iterations, 62 s |
| 3 (+ event pump, `wait_job`) | **125 s** | 23 | 0 | 0 | 0.48M | 61 iterations, 63 s |

- Run 2 still cancelled and resubmitted because of a fourth bug: `LocalRunner` delivered worker events only while a
  call was blocked in `wait()`.
  - A `solve(wait=False)` therefore showed "queued" through every later `status` call.
  - A release whose 60 s wait had timed out showed "running" after it finished, so the agent cancelled and re-ran it.
  - Fix: events are pumped in a background thread. A timed-out solve now returns a hint and the new `wait_job` tool.
- In runs 2 and 3 the agent chose `optimize=True` (the re-opened warm start) at every rung.
- Both runs released the same design, compliance 4.434352810744401 N·mm, bit-identical.
- Both released specs equal the intended reading, and both replays are exact.
- Stage times, run 3:
  - spec 26 s, including one question (the agent asked whether the release may be finer than the brief's 1 mm);
  - sketch 6.2 s, preview 6.6 s, refine 12.8 s, release 62.7 s;
  - export 0.9 s.

## 2026-10-30: X3 on 15 briefs (`summary_table.md`, `run_batch2.log`)

Codex gpt-6.1-sol (high), the scripted oracle, profile full, one session per brief, alone on the GPU. The ladder is
capped at 8e7 elements (the 113M decision), so b02, b17 and b40 release at their 12–14M rung rather than at 98–115M.

| Task | Released | Spec = intended | Release elements | Release compute | Session wall | Tool calls (failed) | Questions | Input tokens (cached) | Replay |
|---|---|---|---|---|---|---|---|---|---|
| b01_cantilever_block | True | True | 16.0M | 62 s | 142 s | 23 (0) | 1 | 475k (452k) | True |
| b02_mbb_half | True | True | 12.3M | 32 s | 93 s | 22 (0) | 1 | 422k (410k) | True |
| b04_torsion_shaft | True | True | 65.5M | 85 s | 161 s | 21 (0) | 0 | 478k (451k) | True |
| b05_l_bracket | True | True | 25.6M | 133 s | 258 s | 26 (1) | 1 | 564k (538k) | True |
| b06_motor_mount | True | True | 73.7M | 35 s | 201 s | 36 (0) | 4 | 549k (530k) | True |
| b11_cantilever_line_load | True | True | 49.2M | 141 s | 286 s | 25 (0) | 1 | 567k (526k) | True |
| b17_arch_bridge | True | True | 14.4M | 208 s | 367 s | 27 (0) | 1 | 823k (794k) | True |
| b19_engine_bracket | True | True | 35.8M | 1508 s | 1898 s | 59 (1) | 1 | 2959k (2905k) | True |
| b25_cantilever_metres | True | True | 16.0M | 105 s | 197 s | 21 (0) | 0 | 611k (593k) | True |
| b31_pillow_block | True | True | 18.4M | 56 s | 150 s | 28 (1) | 2 | 584k (553k) | True |
| b35_padeye | True | True | 15.4M | 55 s | 141 s | 21 (0) | 0 | 419k (393k) | True |
| b40_crank_arm | True | True | 13.8M | 298 s | 477 s | 32 (0) | 0 | 1179k (1134k) | True |
| b44_gearbox_bracket | True | True | 65.5M | 82 s | 167 s | 25 (0) | 1 | 521k (508k) | True |
| b47_camera_mount | True | True | 27.6M | 199 s | 305 s | 27 (1) | 1 | 730k (701k) | True |
| b49_hinged_arm | False | – | – | – | 26 s | 7 (0) | 0 | 92k (79k) | True |

Released 14/15; session wall for released tasks: median 199 s, range 93-1898 s.

- **14/14 solvable briefs released certified with the intended spec, and every session replays exactly with no LLM.**
  - The 14 cover the restated benchmarks b01, b02, b04, b05, b11, b17 and b25, and the engineering parts b06, b19,
    b31, b35, b40, b44 and b47.
  - b44 and b47 are clarify tasks: the agent asked one question each and released the intended design.
  - b25 (m/kN) and b19 (in/lbf) are in other units.
- **b49 (free hinge) was stopped, correctly, in 26 s.** The agent's stop message says R003 leaves rotation about z
  free and the hinge moment unbalanced, so no certified static release exists.
- **The agents release through "Optimize release".** Every warm rung ran reopen_long (61 iterations), the arm X1 and X5
  found best.
- **Session wall times** (released tasks): median 199 s, range 93–477 s, apart from b19 at 1898 s.
  - b19's 36M release grid pads its 200-element side to 256 with void, and the release took 1508 s of compute.
  - It is the multigrid padding problem of X5 (`results/solver/mg_padding.json`).
- **Cost:** 0.4–1.2M input tokens per session (95% cached), and 2.96M for b19, which polled `wait_job` 30 times
  during its long release.
- One failed tool call in each of b05, b19, b31 and b47; each was retried correctly.
