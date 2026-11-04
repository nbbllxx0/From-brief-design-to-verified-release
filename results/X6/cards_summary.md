# X6c: action cards and validation catches (scripted oracle, X2 repeat 1, 50 briefs)

| Condition | Runs with cards | Cards | Accepted | Edited | Rejected | Missing items | All cards accepted, nothing missing |
|---|---|---|---|---|---|---|---|
| T | 35 | 292 | 291 (99.7%) | 1 (0.3%) | 0 (0.0%) | 0 | 34/35 |
| G | 50 | 436 | 420 (96.3%) | 6 (1.4%) | 10 (2.3%) | 0 | 44/50 |
| GC | 50 | 423 | 423 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 | 50/50 |
| GCV | 50 | 423 | 423 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 | 50/50 |

Validation during the agent sessions (G and GC tool logs):

| Condition | Sessions | Validated at least once | Error caught before a solve: fixed / stopped on | Codes | Wrong spec that validation passed (solve and clarify tasks) |
|---|---|---|---|---|---|
| G | 50 | 50 | 0 / 3 | R003 ×2, R008 ×1 | 6 (b07, b39, b44, b45, b46, b47) |
| GC | 50 | 50 | 0 / 3 | R003 ×2, R008 ×1 | 0 () |
