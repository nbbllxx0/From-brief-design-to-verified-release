**Certified release: warm arms against a cold run at the release rung**

| Problem | Release elements | Cold compute (s) | Tail ΔC | Tail compute | Tail IoU | Reopen-long ΔC | Reopen-long compute | Reopen-long IoU | All certified |
|---|---|---|---|---|---|---|---|---|---|
| cantilever | 65.5M | 118 | +0.28% | 7% | 0.96 | -1.20% | 69% | 0.96 | yes |
| bridge | 65.5M | 249 | +1.39% | 11% | 0.89 | -1.19% | 51% | 0.88 | yes |
| torsion | 65.5M | 121 | +0.86% | 6% | 0.95 | +0.07% | 59% | 0.96 | yes |
| b05 l bracket | 25.6M | 251 | +0.00% | 3% | 0.86 | -3.06% | 51% | 0.86 | yes |
| b06 motor mount | 73.7M | 423 | +1.22% | 8% | 0.96 | +0.03% | 61% | 0.96 | yes |

Source: results/X1/escalation.json.
ΔC: release compliance relative to the cold run; compute as a share of the cold run's; IoU of the binarized design with the cold design.
