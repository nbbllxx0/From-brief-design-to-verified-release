**Interactive latency through the backend (14 tasks, 20 repeats of the fast interactions)**

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

Source: results/X6/latency/*.json.
