# C4.3: the release contract with each gate off

## Certificate gate off (census of every recorded solve, no new runs)

- Solves recorded: 486 (certified 54).
- By study: X1 20/70; X3 15/63; X5 4/16; X6a 14/336; solver 1/1.
- Uncertified, by status: {'converged': 420, 'solve_failed': 1, 'not_converged': 11}. Sketch, preview and refine solves are uncertified by design (no spill guard), so only release attempts are counted below.
- Finished without converging (a system without the gate would show these as designs): 11: X1 `refine_pilot.json.rows[0].arms.cold_phys`, X1 `refine_pilot.json.rows[0].arms.phys_tail`, X1 `refine_pilot.json.rows[0].arms.phys_reopen_long`, X1 `refine_pilot.json.rows[1].arms.cold_phys`, X1 `refine_pilot.json.rows[1].arms.phys_tail`, X1 `refine_pilot.json.rows[1].arms.phys_reopen_long`, X1 `refine_pilot.json.rows[2].arms.cold_phys`, X1 `refine_pilot.json.rows[2].arms.phys_tail`, X1 `refine_pilot.json.rows[2].arms.phys_reopen_long`, X6a `b17_arch_bridge.json.rungs[1]`, X6a `b40_crank_arm.json.rungs[2]`.
- Release attempts (spill-monitored or release rung): 54; uncertified: 2.
  - X1 `escalation.json.rows[1].release.cold_failed`: solve_failed, failed [], SolveFailed('ir_lean: correction rejected with inner tolerance 1e-07')
  - X3 `b01_cantilever_block_run1.json.stages.solve_h1.0`: converged, failed [], 
- Release attempts the spill guard stopped (recorded in the notes; no design was produced): X1 113.2M warm release (mg_ir), twice; X5 59M mg64 evaluation beside the desktop.

## Rule gate off (ill-posed briefs solved with validation bypassed)

| Task | Rule the gate raises | Solver status | Certified | Compliance | Max PCG its | Note |
|---|---|---|---|---|---|---|
| b08_slider_bracket | R003 (['R003']) | solve_failed | False | – | None | SolveFailed('pcg64: non-positive curvature nan at iteration 1') |
| b49_hinged_arm | R003 (['R003']) | solve_failed | False | – | None | SolveFailed('pcg64: true residual 2.367e+04 > 1.0e-05 after 20000 iterations') |
| b50_valve_bracket | R008 (['R008']) | converged | True | 1.449e+08 | 23 |  |
| b50_valve_bracket | none (reference: pad at x = 40) ([]) | converged | True | 0.3817 | 9 |  |
