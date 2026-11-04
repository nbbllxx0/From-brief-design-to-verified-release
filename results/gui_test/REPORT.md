# GUI test pass (2026-10-31)

End-to-end test of every GUI function on small meshes. The script drives headless Chrome with real mouse and keyboard
input: faces are clicked on the 3D viewport, fields are typed into, and keys are pressed.
- Script: [`scripts/gui_test.py`](../../scripts/gui_test.py).
- Screenshots: `results/gui_test/*.png`.
- Raw results: `report_setup_ladder_overlays.json` and `report_assistant.json`.

## Result

| Group | Checks | Passed | What it covers |
|---|---|---|---|
| setup | 34 | 34 | New design; S/L/K/O tools and real face clicks; patch placed where clicked; inspector edits (force, DOF checkboxes, region-kind conversion, domain size, volume, mesh-by); validation messages (missing BCs, R007 keep-in/out overlap, volume > 100%); ladder lock; tree and viewport picking; Delete; Ctrl+Z / Ctrl+Y; timeline labels and clipping; composer layout |
| ladder | 27 | 27 | Release note; Sketch → Preview → Refine with warm starts (↑); streamed frames; result card; Optimize release appears only after Refine; Stop cancels a running release; an 8.2M Optimize release certified (33–40 s); marching-cubes surface shown automatically, Voxels/Surface toggle; STL + markdown + JSON export, links download, STL header consistent; stale badge after an edit; undo clears it and restores the rung states |
| overlays | 19 | 19 | First-run tips, dismissal kept after reload; `?` keyboard map and Esc; legend not a dim layer; Ctrl+K palette (focus, filter, Enter runs, closes); theme button and persistence; tool keys inert in text fields; problem picker (suite task with brief, library problem); proposal ghost on a resized box, framed with the current box; chat bubbles fit their text; timeline label for a clarification answer |
| assistant | 8 | 8 | Act run that asks a question, **answered after 80 s**, and the load applied; "Undo this run's edits"; Propose returns cards without touching the spec; hover draws the ghost; Accept applies one card only; Reject |
| no uncaught exception or console error | 2 | 2 | one check per run |
| **Total** | **90** | **90** | |

The unit tests also pass: `pytest tests` gives 140 passed, including a new test for late clarification answers.

Run it:
```
python -B scripts/serve.py --ladder-max 1.2e7      # rungs 16k / 128k / 1M / 8M on a 100 x 50 x 50 box
node web/node_modules/vite/bin/vite.js web
python -B scripts/gui_test.py setup ladder overlays  # about 3 min, GPU
python -B scripts/gui_test.py assistant              # about 5 min, two Codex sessions on the subscription
```
The launch configuration `agentto-backend-small` starts the backend with that cap.

## Bugs found and fixed

| # | Sev. | Bug | Fix |
|---|---|---|---|
| 1 | 4 | **A clarification answered after about a minute was lost.** Codex hands control back to the model about 60 s into a blocking tool call. In Act mode the model then ended its turn ("awaiting your choice") with the question still on screen. Clicking a chip afterwards reached no agent. Found by a run where the answer came late; a Codex probe confirmed that the call itself does keep waiting. | (a) The agent instructions now say that `ask_clarification` can take minutes and that the agent must not end its turn while its question is open. With this, the 80 s answer was applied in the same run. (b) Backstop: if the asking run has ended, the backend starts a follow-up run that carries the answer and the chosen reading (`server._clarify_answer`). A unit test covers it. |
| 2 | 2 | Undo back to a solved spec cleared the ladder: the rungs showed no state although their designs existed (the result card was already correct) | Jobs remember their spec hash; after a server spec change the ladder re-attaches the jobs of the current hash |
| 3 | 2 | The timeline row clipped the **newest** records at the right edge | Records are right-aligned, so the oldest are cut |
| 4 | 2 | The Setup / Explore / Release switch in the top bar changed nothing (its modes were never wired; Explore's variants are descoped) | Removed |
| 5 | 1 | A selection chip squeezed the assistant's text box to about 110 px | The chip has its own line |
| 6 | 1 | The timeline showed "undefined" for a clarification answer | Labelled "user · answer: …" |
| 7 | 1 | Proposal ghosts were drawn in the old domain (heuristic review 2, finding 9) | Ghosts use the proposal's domain; the proposed box is outlined, and the view frames both boxes while the proposal is open |
| 8 | 1 | A task's brief was hidden as soon as the chat held any message, for example after switching problems | The brief is shown whenever one is loaded |
| 9 | 3 | The assistant's reply bubbles were squashed to zero height, so long replies spilled over the cards below. Cause: the panel's CSS class `.assistant` also matched the bubbles (`.msg.assistant`) and gave them `flex: 1` (the same kind of clash as the earlier `.overlay` one). Found in the regenerated paper screenshot. | The panel class is now `.assistant-panel`; a test checks that every bubble fits its text |

The paper screenshots (`results/figures/gui_overview.png`, `gui_clarify.png`, `gui_assistant.png`) were regenerated on a
fresh backend after these fixes.

Improvement made on the way: a certified release now appears as its marching-cubes surface instead of voxels, and a late
density frame of the same job no longer drops the surface.

## Marching cubes

Yes. `src/agentto/export.py` extracts the ρ = 0.5 isosurface with VTK Flying Edges (a marching-cubes variant):
- `isosurface()` pads one void layer so the part is closed;
- `preview_mesh()` pools the volume to at most 4M elements for display;
- `export_stl()` writes the full-resolution binary STL.

The viewport shows instanced voxels while a solve streams. A certified release now switches to the surface by itself, and
the result card toggles Surface / Voxels for any finished solve. The 8.2M test release gave 166k display triangles at
pool ×2.

## Observed, not changed

- **K/O slabs on adjacent faces conflict.** Each slab is a full face 10% deep, so a keep-in on one face and a keep-out on
  an adjacent face overlap along the shared edge. R007 reports it clearly; the user edits the box.
- **The ladder ignores the spec's resolution.** It always offers rungs up to the cap (8e7); `resolution` sets h for agent
  solves and the rung spacing. This is by design, but the inspector's "cells short" can suggest otherwise.
- **Voxel frames are pooled for streaming**, so they look coarser than the solve. The surface does not have this issue.
- **Keyboard map and palette have no focus trap** (backlog, sev 1).

## Not covered in this pass

- Full-size releases (65–74M). They were covered in the week-5 GUI test and in X1/X5; this pass capped the ladder at 1.2e7
  on purpose.
- Reconnecting after a backend restart while a page is open.
- Two browser clients editing the same session.
- Narrow (mobile) widths.

## Test-only and provenance additions

- `?test` in the URL exposes `window.__agentto`: the store, a camera turn, and a spec-point → screen projection. Without the
  parameter nothing is exposed.
- `scripts/serve.py --ladder-max N` and the `AGENTTO_LADDER_MAX` variable can lower the ladder cap (never raise it).
- Each GUI agent run now keeps Codex's event stream and stderr (`sessions/<s>/agent/<run>/work/events.jsonl`,
  `stderr.txt`). These files are what showed bug 1.
