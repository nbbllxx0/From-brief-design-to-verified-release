# Data left out of this repository

The repository holds every result record (`results/**/*.json`), every session log (`session.jsonl`), the study notes,
the generated tables and figures, and the brief suite. It does not hold the large binary outputs of the runs:

| Kind | Where it was written | Why it is left out |
|---|---|---|
| Density fields (`*.npy`, float16) | `results/<study>/fields/`, `results/X3/<task>/session/fields/`, ... | large; regenerable by replay |
| Meshes (`*.stl`, `*.vtp`) and export reports' meshes | `.../exports/` | large; regenerable by replay or `export_stl` |
| Setup renders inside session folders (`*.png`) | `.../renders/` | regenerable by replay (render hashes are in the logs) |
| Cross-check arrays (`*.npz`) | `results/X0/` | regenerable by `scripts/x0_export_crosscheck.py` |
| Runs from discarded passes | `results/X2_draft1`, `results/X2_oraclebug`, `results/X2_claude_high_gcv_verifydefault` | superseded; not used by any table or figure |
| Logs of discarded or superseded passes (`*.log`) | `results/`, `results/<study>/` | console output; the JSON records carry the results |
| Local GUI sessions | `sessions/` | development sessions, not part of any study |

Measured at staging time (files under `results/` excluded by the rules above):

| Kind | Files | Size |
|---|---|---|
| density fields (*.npy) | 230 | 4.42 GB |
| meshes and exports | 132 | 3.11 GB |
| renders inside session folders (*.png) | 1244 | 64.1 MB |
| runs from discarded passes | 1007 | 22.1 MB |
| scratch folders (work/) | 690 | 0.3 MB |
| console logs (*.log) | 36 | 0.1 MB |
| cross-check arrays (*.npz) | 2 | 0.0 MB |
| **total** | 3341 | 7.61 GB |

## Regenerating the binary outputs

Every session log can be replayed without a language model. Replay re-runs every tool call and every solve, checks
that each solve reproduces its status, iteration count, compliance (bit for bit) and certificate, and writes the
density fields, renders and exports again:

```python
import sys; sys.path.insert(0, "src")
from agentto.tools import replay
print(replay("results/X3/b06_motor_mount/session/session.jsonl", root="replayed/b06"))
```

Replay needs the voxfem solver and a CUDA GPU with enough memory for the session's largest grid (the X3 releases are
2.6-7.4 x 10^7 elements; they were run on a 24 GB card). The study scripts in `scripts/` regenerate the fields of
X1 and X5 directly (see the README, section "Reproducing the paper").

## Archive

If the binary outputs are needed without re-running them, they can be archived separately (for example on Zenodo).
No such archive exists yet; if one is made, its DOI goes here and in `CITATION.cff`.
