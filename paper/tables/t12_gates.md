**The release contract with each gate removed**

| Gate removed | Case | Outcome | Note |
|---|---|---|---|
| Certificate | release attempts certified | 52/54 | the two uncertified attempts produced no releasable design |
| Certificate | spill-guard stops (113M ×2, 59M) | 3 | runs stopped; no design |
| Certificate | non-converged solves | 11 | a gate-less system would show these as designs |
| Rule base off | b08 (R003) | solve failed; not certified | SolveFailed('pcg64: non-positive curvature nan at iteration 1') |
| Rule base off | b49 (R003) | solve failed; not certified | SolveFailed('pcg64: true residual 2.367e+04 > 1.0e-05 after 20000 iter |
| Rule base off | b50 (R008) | converged; certified | compliance 3.8e+08× the corrected reference |

Source: results/C4/census.json, results/C4/rules_off.json.
With both gates on, no uncertified or ill-posed design was released in any study.
