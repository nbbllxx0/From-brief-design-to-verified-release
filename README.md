# From design brief to verified release

Code, task suite and result records for the paper

> Shaoliang Yang and Jun Wang. *From topology-optimization brief to solver-checked release: separating intent,
> admissibility and acceptance knowledge for language agents.* arXiv:2611.04077 (2026),
> <https://arxiv.org/abs/2611.04077>.

The software is called AgenticTO (Python package `agentto`). It turns a written design brief into a 3D topology
optimization problem, solves it on one GPU (certified releases of up to 7.4 x 10^7 voxel elements on a 24 GB card
in the paper), and releases a design only when a solver certificate passes. A language agent (Codex CLI or Claude Code) and a human in the web GUI edit the same
design through the same logged tools, exposed over the Model Context Protocol (MCP). Every session is written to a
log that replays without the language model.

What is in this repository:

- `src/agentto/`: the spec schema, the rule base (deterministic checks R001-R013), the tool contract, the session
  solver built on voxfem, the GPU worker, STL and report export, replay, the MCP server and the GUI backend;
- `web/`: the GUI front end (React, three.js, Vite);
- `briefs/`: the 50-brief task suite with intended readings;
- `scripts/`: every study, table and figure script;
- `results/`: every result record and session log of the paper, the study notes, tables' and figures' inputs;
- `paper/`: generated tables, the adjudication record, the script for the manuscript's Figure 2 and the frozen
  inputs and outputs of Figure 1 and the graphical abstract;
- `docs/`: user documentation.

## Guide book

A guide book for users is at [`docs/guidebook/AgenticTO_Guidebook.pdf`](docs/guidebook/AgenticTO_Guidebook.pdf).
The shorter user guide with GUI screenshots is [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)
(PDF: [`docs/AgenticTO_User_Guide.pdf`](docs/AgenticTO_User_Guide.pdf)).

## Requirements

| Item | Tested with | Notes |
|---|---|---|
| GPU | NVIDIA RTX 4090, 24 GB | Any CUDA 12 GPU. GPU memory bounds the largest grid; rule R013 estimates it before a solve. |
| OS | Windows 11 | Plain Python and TypeScript; other platforms are untested. The spill monitor uses `pywin32` (Windows). |
| Python | 3.10 | numpy, cupy (CUDA 12), scipy, pydantic 2, fastapi, uvicorn, mcp, vtk, Pillow, websockets, pytest |
| voxfem | 0.1.0 | The GPU finite-element and SIMP solver. Distributed separately; see below. |
| Node.js | 22 | Only for the web front end |
| Codex CLI (optional) | 0.165, ChatGPT login | The built-in assistant and the benchmark runs |
| Claude Code (optional) | claude.ai login | Second provider for MCP sessions and benchmarks |

Optional, for some scripts: matplotlib (figures), pyvista (3D renders), scikit-fem (independent compliance
cross-check), openpyxl (adjudication sheet), scikit-image and trimesh (shape metrics and meshes), markdown (user
guide build).

Language models are called only through the subscription command-line tools. Before every call the code checks
that no API key is set (`OPENAI_API_KEY`, `CODEX_API_KEY`, `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, Bedrock or
Vertex switches) and refuses to start an agent if one is.

## Install

```bash
conda create -n agentto python=3.10
conda activate agentto
pip install numpy scipy cupy-cuda12x "pydantic>=2.8,<3" fastapi uvicorn mcp vtk pillow websockets pytest
pip install matplotlib pyvista scikit-fem openpyxl scikit-image trimesh     # optional, for some scripts

cd web
npm install          # front-end dependencies, pinned in package-lock.json
cd ..
```

The scripts run from the repository root without installing the package (they add `src/` to the path). To install
it instead, after the dependencies above: `pip install -e . --no-deps` (the `cupy` entry in `pyproject.toml` would
otherwise make pip build CuPy from source).

## The voxfem dependency

The solver, `voxfem`, is not part of this repository. It is the GPU finite-element and SIMP code of the authors'
earlier work and is distributed separately. agentto uses it read-only and extends it only by subclassing and
composition (`src/agentto/solver.py`).

agentto finds voxfem in this order (`src/agentto/_voxfem.py`):

1. the directory named by the environment variable `VOXFEM_SRC` (older name: `AGENTTO_VOXFEM_SRC`), which must
   contain the `voxfem/` package folder;
2. an installed package (`import voxfem`).

```bash
export VOXFEM_SRC=/path/to/voxfem/src            # PowerShell: $env:VOXFEM_SRC = "D:/path/to/voxfem/src"
```

If neither works, importing `agentto.problem`, `agentto.rules` or anything that builds on them raises an
`ImportError` that says voxfem is distributed separately. Without voxfem you can still read every record, rebuild
the tables and most figures from the records (see below), and run the tests listed under "Tests".

## Quick start

### Run the GUI

```bash
python -B scripts/serve.py                        # backend and GPU worker on 127.0.0.1:8765
node web/node_modules/vite/bin/vite.js web        # front end on http://localhost:5173 (or: cd web; npm run dev)
```

Open http://localhost:5173. For smaller GPUs or quick tests, cap the resolution ladder:
`python -B scripts/serve.py --ladder-max 1.2e7`. The backend writes each session to `sessions/<time>/`.

### Connect an agent over MCP

The tools behind the GUI are an MCP server, `agentto.mcp_server`:

```bash
python -B -m agentto.mcp_server --session runs/demo --profile spec    # stand-alone session (as in the benchmarks)
python -B -m agentto.mcp_server --attach http://127.0.0.1:8765        # attach to the GUI's live session
```

Profiles: `spec` (editing, `list_regions`, `render`, `validate_spec`, `explain_spec`), `clarify` (adds
`ask_clarification`), `full` (adds templates, solving, `wait_job`, `field_stats`, export). For Claude Code, save
an `mcp.json` such as

```json
{"mcpServers": {"agentto": {
  "command": "python",
  "args": ["-B", "-m", "agentto.mcp_server", "--profile", "full", "--attach", "http://127.0.0.1:8765"],
  "env": {"PYTHONPATH": "/path/to/this/repository/src", "VOXFEM_SRC": "/path/to/voxfem/src"}}}}
```

and run `claude --mcp-config mcp.json`. Use the full path of the environment's Python as `command` if `python` is
not on the path. The Codex CLI setup is in the user guide, section 8.

### Use the tools from Python

```python
import sys; sys.path.insert(0, "src")
from agentto.tools import Session, replay

s = Session("runs/demo", profile="full")
s.call("new_spec", {"name": "bracket", "size": [100, 50, 50]})
s.call("add_support", {"id": "wall", "region": {"kind": "face", "face": "x-"}})
s.call("add_load", {"id": "tip", "force": [0, -2000, 0],
                    "region": {"kind": "patch", "face": "x+", "center": [25, 25], "size": [10, 20]}})
print(s.call("validate_spec"))
print(s.call("solve", {"rung": "sketch"}))          # GPU
print(replay("runs/demo/session.jsonl", root="runs/replayed"))
```

### Tests

```bash
python -B -m pytest -q
```

The full suite has 140 tests. `tests/test_session.py`, `tests/test_worker.py` and
`tests/test_tools.py::test_session_solve_warm_and_replay_bit_exact` run solves on the GPU and need voxfem and a free
CUDA GPU. To run everything except those:

```bash
python -B -m pytest -q --ignore=tests/test_session.py --ignore=tests/test_worker.py \
    --deselect tests/test_tools.py::test_session_solve_warm_and_replay_bit_exact
```

Most other tests also need voxfem (on the CPU), because the problem builder and the rule base use its grid and
problem classes. Without voxfem, only `tests/test_agent.py`, `tests/test_export.py` and `tests/test_shape.py` can be
collected, and two of their tests still stop with the voxfem `ImportError`:

```bash
python -B -m pytest -q tests/test_agent.py tests/test_export.py tests/test_shape.py
```

## Reproducing the paper

All tables, the claims ledger and the chart figures are generated from the records in `results/` alone, with no GPU
and no language model:

```bash
python -B scripts/make_ledger.py       # results/claims_ledger.md
python -B scripts/make_tables.py       # paper/tables/*.md and *.tex
python -B scripts/make_figures.py      # results/figures/*.pdf and *.png
```

Re-running a study regenerates its records. Marks: **GPU** = needs voxfem and a CUDA GPU (one job at a time; nothing
else on the GPU), **GPU-heavy** = grids of 10^7-10^8 elements, hours in total on a 24 GB card, **LLM** = runs Codex
or Claude Code on a subscription login, **CPU** = neither.

| Paper item | Records | Study script(s) | Table / figure script |
|---|---|---|---|
| Task suite (T1) | `briefs/`, `briefs/intended/` | `make_briefs.py` (CPU) | `make_tables.py` -> `t1_suite` |
| Rule base (T2) | `src/agentto/rules.py` | - | `make_tables.py` -> `t2_rules` |
| X0 verification: identity, replay, independent cross-check | `results/X0/` | `x0_identity_gpu.py` (GPU), `x0_replay.py` (GPU), `x0_export_crosscheck.py` (GPU), `x0_crosscheck_skfem.py` (CPU, scikit-fem), `x0_coarse_bc.py` (GPU) | ledger |
| X1 certified escalation (T3, Fig. x1_release_arms, x1_arms_3d) | `results/X1/` | `x1_escalation.py` (GPU-heavy), `x1_fix_cold.py` (GPU-heavy), pilots `x1_warm_arms_pilot.py`, `x1_refine_pilot.py` (GPU) | `make_tables.py` -> `t3_x1_escalation`; `make_figures.py`; `make_renders.py arms` |
| X2 spec grounding, 50 briefs (T4-T6, Fig. x2_outcomes) | `results/X2/{T,G,GC,GCV,R}/` | `x2_run.py` (LLM), `x2_score.py` (GPU), `x2_rules_baseline.py` (CPU), `x2_rescore.py` (CPU), `validate_suite.py` (GPU) | `make_tables.py` -> `t4`-`t6`; `make_figures.py` |
| Second provider (T7, Fig. x2_providers) | `results/X2/claude_high/`, `results/X2/claude/` | `x2_run.py --provider claude --effort high` (LLM), `x2_score.py claude_high/GC ...` (GPU) | `make_tables.py` -> `t7_providers`; `make_figures.py` |
| X3 brief to certified release (T8, Fig. x3_session_time, x3_gallery) | `results/X3/`, `results/X3_aware/` | `x3_run.py <task>` (LLM, GPU-heavy), `x3_retime_aware.py` (GPU-heavy) | `make_tables.py` -> `t8_x3_sessions`; `make_figures.py`; `make_renders.py gallery` |
| X5 why 10^8 (T9, Fig. x5_resolution_path, x5_bridge_3d) | `results/X5/` | `x5_resolution.py`, `x5_chain.py` (GPU-heavy) | `make_tables.py` -> `t9_x5_resolution`; `make_figures.py`; `make_renders.py x5` |
| Multigrid on padded grids (Fig. solver_padding) | `results/solver/` | `mg_padding_study.py` (GPU), `b19_release_aware.py` (GPU-heavy) | `make_figures.py` |
| X6a latency (T11, Fig. x6a_latency) | `results/X6/latency/` | `x6a_latency.py` with `serve.py` running (GPU) | `make_tables.py` -> `t11_x6a_latency`; `make_figures.py` |
| X6b interaction cost (T10, Fig. x6b_klm) | `results/X6/klm.json` | `x6b_klm.py` (CPU, needs voxfem) | `make_tables.py` -> `t10_x6b_klm`; `make_figures.py` |
| X6c action cards | `results/X6/cards.json` | `x6c_cards.py` (CPU, needs voxfem) | ledger |
| C4 gate ablation (T12) | `results/C4/` | `c4_gate_ablation.py` (census: CPU; rules_off: GPU, minutes) | `make_tables.py` -> `t12_gates` |
| GUI screenshots (gui_overview, gui_assistant, gui_clarify) | `results/figures/` | `gui_screenshots.py` with backend and front end running (GPU, LLM, Chrome) | - |
| Manuscript Figure 1 (workflow), graphical abstract | `paper/V5/figures/fig1_data/` | - | drawn figures; the PDFs are in `paper/V5/figures/` (the drawing scripts are not published) |
| Manuscript Figure 2 (GUI with enlarged result card) | `paper/V5/figures/gui_overview.png` | - | `paper/V5/scripts/make_fig2_gui.py` (CPU) |
| Manuscript released-designs figure | `results/X3/` | - | `paper/V5/scripts/make_fig_designs.py` (CPU; needs the exported STLs, which regenerate from the session logs) |
| GUI end-to-end test (90 checks) | `results/gui_test/` | `gui_test.py setup ladder overlays assistant` with backend and front end running (GPU, LLM, Chrome) | - |
| X2 follow-up conditions (T+S, GNR, G+P, G+C+P; Tables 5, 6, S2, S3) | `results/X2/TS/`, `GNR/`, `GP/`, `GPC/`, `claude_high/GPC/` | `x2_run.py` (LLM), `x2_provenance_batch.sh` (LLM), `x2_score.py TS GP GPC claude_high/GPC` (GPU) | `x2_provenance.py` -> `provenance_summary.json`; `x2_gnr_ablation.py` -> `summary_GNR.json`; `make_tables.py` |
| Adjudication and rescore | `paper/adjudication/`, `paper/ADJUDICATION.md` | `make_adjudication.py`, `fill_sheet_from_txt.py`, `apply_adjudication.py`, `x2_rescore.py` (CPU) | - |
| Second (blinded) adjudication | `paper/adjudication/round2/` | `make_adjudication_pack_r2.py` (CPU) | `compare_adjudications.py <sheet>` -> `agreement.md`, `agreement.json` |
| Follow-up adjudication | `paper/adjudication/followup/` | `make_adjudication_pack_followup.py`, `followup_equivalence.py` (CPU) | `apply_adjudication_followup.py <sheet>` -> `followup_decisions.json`; then `x2_rescore.py` |

All script paths are under `scripts/` unless given in full. Each script's docstring states its inputs, outputs and
run time. The 3D renders (`make_renders.py`, pyvista) read density fields and STL files, which are not in this
repository: regenerate them first by replaying the sessions or re-running the study (see `data/README.md`).

## Data layout

`briefs/`

- `bNN_<slug>.md`: the brief, the only text an agent sees;
- `intended/bNN_<slug>.json`: the intended spec, the expected behaviour (`solve`, `clarify` or `refuse`), the
  scripted oracle's answers, admissible readings, the source of the problem and the author's adjudication.

`results/`

| Folder | Contents |
|---|---|
| `X0/` | identity, replay and cross-check records; `replay_sessions/` holds the replayed session logs |
| `X1/` | `escalation.json` (every arm at every rung), pilots, `NOTES.md` |
| `X2/<cond>/<task>_r<rep>.json` | one record per run: prompt, final spec, tool calls, questions, tokens, score, consequence-weighted error; `<task>_r<rep>/session/session.jsonl` is the tool log. `<cond>` is `T`, `G`, `GC`, `GCV` or `R`; `claude_high/` and `claude/` hold the Claude Code runs. `summary*.json`, `suite_check.json`, `NOTES.md` |
| `X3/<task>.json`, `X3/<task>/session/session.jsonl`, `X3/<task>/replay/session.jsonl` | end-to-end sessions, their logs and their replays; `_run1`/`_run2` are earlier runs of b01 |
| `X3_aware/` | the X3 sessions re-timed under the pad-aware coarse rule |
| `X5/resolution.json` | resolution study and release-path chain |
| `X6/` | latency (`latency/`), interaction cost (`klm.json`, `scripts/` with the generated voxfem scripts), cards (`cards.json`) |
| `C4/`, `solver/`, `latency/`, `probe/`, `diag/`, `gui_test/` | gate ablation, multigrid padding study, latency check, first CLI probes, a warm-start diagnosis, GUI test report |
| `figures/` | every chart figure (PDF and PNG) |
| `claims_ledger.md` | every number the paper states, with its source file and status |

Records written on Windows keep Windows-style relative paths (for example `results\X3\...`).

Left out: density fields, meshes and session renders (several GB). They regenerate by replay; see
[`data/README.md`](data/README.md).

## License

Code: MIT License ([`LICENSE`](LICENSE)). Data, records and documentation: CC BY 4.0. The scope of each license and
what neither covers (third-party dependencies, the separately distributed voxfem solver) are set out in
[`LICENSE-DATA.md`](LICENSE-DATA.md).

## Citation

See [`CITATION.cff`](CITATION.cff). Contact: Jun Wang, Santa Clara University (jwang22@scu.edu).
