# X5 notes: why 10^8 elements

## 2026-10-30: resolution study (`scripts/x5_resolution.py`, `scripts/x5_chain.py`, `resolution.json`)

Design:
- Two problems on a ladder of halving element size, with the filter radius in element units, as the GUI runs it:
  - **span bridge**: 1280 × 160 × 160 mm, 8 mm solid deck, pinned abutments, uniform deck load, volume fraction 0.15;
    h = 8 → 1 mm, 0.06M → 33M elements, L/h up to 1280;
  - **b06 motor mount**: the brief's keep-in bosses and pad; h = 2 → 0.25 mm, 0.14M → 74M elements.
- Every design becomes a part: upsampled to the finest grid, binarized at the threshold that makes exactly the volume
  fraction of the design elements solid, then evaluated by one state solve at the finest grid. Every part is judged by
  the same fine model at the same manufactured volume.
- Runs at the finest rung, and every evaluation at the finest grid, have the spill guard on.

### Cold optimum at each rung, judged as a part at the finest grid

| Problem | h (mm) | Elements | Compute | C_part | vs cold finest | Own prediction vs part | b1 (tunnels) | p5 member (mm) |
|---|---|---|---|---|---|---|---|---|
| bridge | 8 | 0.06M | 36 s | 2715 | −18.6% | −12.3% | 24 | 16 |
| bridge | 4 | 0.51M | 57 s | 2647 | −20.6% | −6.7% | 50 | 8 |
| bridge | 2 | 4.1M | 204 s | 2940 | −11.8% | −2.7% | 82 | 4 |
| bridge | 1 | 33M | 466 s | 3333 | 0 | 0 | 146 (b0 5, b2 10) | 2.8 |
| b06 | 2 | 0.14M | 7 s | 4.0121 | +2.6% | −2.7% | 0 | 20 |
| b06 | 1 | 1.15M | 14 s | 3.9601 | +1.2% | −0.6% | 0 | 20 |
| b06 | 0.5 | 9.2M | 66 s | 3.9275 | +0.4% | −0.25% | 0 | 4.2 |
| b06 | 0.25 | 74M | 466 s | 3.9116 | 0 | 0 | 0 | 4.9 |

### The release path: cold at the coarsest rung, then reopen_long at every finer rung, warm from the rung above

| Problem | h (mm) | Arm | Iterations | Compute | C_part | vs cold finest |
|---|---|---|---|---|---|---|
| bridge | 8 | cold | 80 | 24 s | 2715 | −18.6% |
| bridge | 4 | reopen_long | 61 | 32 s | 2506 | −24.8% |
| bridge | 2 | reopen_long | 61 | 124 s | 2422 | −27.3% |
| bridge | 1 | reopen_long | 61 | 232 s | **2382** | **−28.5%** |
| b06 | 2 | cold | 80 | 6 s | 4.0121 | +2.6% |
| b06 | 1 | reopen_long | 61 | 8 s | 3.9695 | +1.5% |
| b06 | 0.5 | reopen_long | 61 | 40 s | 3.9334 | +0.6% |
| b06 | 0.25 | reopen_long | 61 | 260 s | 3.9151 | +0.09% |

Total compute: the bridge chain 412 s against 466 s for the cold 33M run; the b06 chain 314 s against 466 s.

### Findings (what X5 can claim)

1. **Coarse models over-predict their own stiffness.** A rung's own compliance is lower than its part's compliance at
   the finest grid.
   - Bridge: 12.3%, 6.7% and 2.7% at 8, 4 and 2 mm. b06: 2.7%, 0.6% and 0.25%.
   - The gap shrinks about linearly with h. Only a fine model gives the stiffness the part will have.
   - This is the clean argument for releasing at 10⁷–10⁸ elements: the certificate is about the part, not about a coarse
     surrogate.
2. **Along the release path, resolution pays monotonically.**
   - On the bridge, each doubling improves the part: 2715 → 2506 → 2422 → 2382 (−12% from 8 to 1 mm). b06: −2.4%.
   - Members get thinner, down to 2.8–4.9 mm (p5).
3. **A cold run at the finest rung is a poor route on slender problems.**
   - The cold 33M bridge ends 26% worse than the cold 4 mm design judged by the same fine model, and 40% worse than the
     release path.
   - It is intricate: 146 tunnels, 5 disconnected pieces and 10 enclosed cavities after binarization.
   - With the filter in element units, a fine cold run starts from a field of small-scale features and settles in a
     worse local optimum.
   - On compact b06 the cold fine run and the release path agree within 0.09%.
   - This matches X1 (reopen_long beat cold by up to 3.1%), and makes coarse-to-fine the better route, not only the
     faster one.
4. **STL at the finest rung:** 2.3M triangles (bridge, 114 MB) and 1.6M (b06, 81 MB). The surface takes 0.35–0.56 s.

### Problems met on the way (all fixed or worked around; worth a sentence in the paper)

- **Padded thin directions slow multigrid badly.**
  - At 1600 × 200 × 200 mm, the 8 mm rung (25 elements across) is padded to 32 with void.
  - PCG then needed about 450 iterations per solve, against 27 at 192 mm (24 elements, no padding): a 16× cost.
  - Across the suite, per-iteration cost varies 80× per million elements; the thin, padded parts are the slowest.
    `session_levels`' cost model counts only the coarsest level's size, not this effect.
  - Follow-up: penalise void padding in the cost model, or prefer fewer levels without padding.
- **Volume fraction 0.10 failed on the 8M bridge rung.** At the first continuation step (p 1.5 → 3.5, β 1 → 4) the
  volume dropped from 0.10 to 0.061 and compliance jumped 100×. The design lost its load path, and the next solve
  failed (NaN curvature in PCG). With 0.15 it converges. The coarser rungs survive the same step.
- **VRAM:** a 59M mg64 evaluation spilled beside the desktop's 2–5 GB (the guard caught it; an earlier unguarded try
  returned NaN). So the bridge was sized to 33M. mg_ir rejected its correction on the bridge's solid/void parts, but
  evaluated b06's 74M part (3.9116, 17 s).
