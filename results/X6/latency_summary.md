# X6a: latency over 14 X3 tasks (backend, 20 repeats of the fast interactions)

| Interaction | Target | Median over tasks | Range |
|---|---|---|---|
| Spec edit → validation (p50 per task) | < 200 ms | 2.15 ms | 1.01–10.2 ms |
| Spec edit → validation (p95 per task) | < 200 ms | 2.49 ms | 1.17–10.5 ms |
| Edit commit, set_spec (p50) | – | 0.925 ms | 0.88–1.14 ms |
| Edit → sketch, warm (p50 per task) | 1–3 s | 0.934 s | 0.577–2.13 s |
| Edit → sketch, warm (p95 per task) | 1–3 s | 0.951 s | 0.586–2.4 s |
| Edit → first sketch frame (p50) | – | 0.0943 s | 0.0779–0.221 s |
| Sketch, cold (new spec) | 1–3 s | 3.32 s | 2.48–6.75 s |
| Preview, warm | ≈ 5 s | 0.794 s | 0.41–18 s |
| Refine, warm | 2–5 s compute | 1.52 s | 0.721–17.5 s |
| Release, warm, compute | ≈ 10–25 s | 6.65 s | 3.14–56.4 s |
| Release surface on screen (first fetch) | < 3 s | 0.136 s | 0.0834–0.714 s |
| STL export at release | – | 0.593 s | 0.298–2.19 s |

| Task | Release elements | Sketch cold s | Edit→sketch p50 / p95 s | Preview s | Refine compute s | Release compute s (status) | Surface s |
|---|---|---|---|---|---|---|---|
| b01_cantilever_block | 16.0M | 3.5 | 0.64 / 0.77 | 0.9 | 1.1 | 4.0 (converged) | 0.71 |
| b02_mbb_half | 12.3M | 3.3 | 0.91 / 0.92 | 0.5 | 0.7 | 3.1 (converged) | 0.08 |
| b04_torsion_shaft | 65.5M | 2.6 | 0.76 / 0.82 | 0.4 | 1.4 | 8.4 (converged) | 0.23 |
| b05_l_bracket | 25.6M | 3.3 | 1.00 / 1.04 | 0.5 | 1.9 | 5.7 (converged) | 0.15 |
| b06_motor_mount | 73.7M | 3.4 | 0.96 / 0.98 | 1.0 | 4.4 | 18.2 (converged) | 0.23 |
| b11_cantilever_line_load | 49.2M | 6.7 | 2.13 / 2.40 | 0.7 | 2.8 | 13.6 (converged) | 0.19 |
| b17_arch_bridge | 14.4M | 5.0 | 1.02 / 1.39 | 18.0 | 17.5 | 4.9 (converged) | 0.10 |
| b19_engine_bracket | 35.8M | 4.1 | 1.02 / 1.16 | 1.0 | 3.1 | 19.4 (converged) | 0.12 |
| b25_cantilever_metres | 16.0M | 2.9 | 0.83 / 0.92 | 0.9 | 1.1 | 3.8 (converged) | 0.09 |
| b31_pillow_block | 18.4M | 2.6 | 0.58 / 0.59 | 0.7 | 1.7 | 6.3 (converged) | 0.11 |
| b35_padeye | 15.4M | 4.0 | 1.10 / 1.14 | 0.8 | 1.2 | 4.9 (converged) | 0.09 |
| b40_crank_arm | 13.8M | 3.0 | 0.75 / 0.91 | 2.4 | 16.2 | 56.4 (converged) | 0.09 |
| b44_gearbox_bracket | 65.5M | 2.5 | 0.70 / 0.81 | 0.4 | 1.3 | 7.7 (converged) | 0.22 |
| b47_camera_mount | 27.6M | 3.7 | 1.17 / 1.25 | 0.4 | 1.1 | 7.0 (converged) | 0.15 |
