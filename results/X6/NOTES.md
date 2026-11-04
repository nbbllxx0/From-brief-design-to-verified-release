# X6: participant-free GUI evaluation

No participants. Every number here comes from logs or a stated model, and the paper makes no usability claim.

## G6, before X6b: the GUI without the agent can now build every spec in the suite (2026-10-30)

X6b compares the GUI with and without the agent. A fair comparison needs the GUI alone to be able to build every
task. Before this change it could not create:
- keep-ins or keep-outs;
- edge, box or cylinder regions;
- a units choice, a blank design, or the choice between the h and cells resolution forms.

These are now in the GUI:
- viewport tools K (keep-in) and O (keep-out): a click on a face adds a slab a tenth of the domain deep;
- a region-kind select in the inspector (face, patch, edge, box, cylinder; box and cylinder for passive regions),
  which converts the item and sizes it from the domain;
- face, axis and patch-shape selects; edge span ("limit" / "whole edge");
- the keep-in / keep-out type select;
- the units selects, the material name, and the mesh-by select (h or cells);
- "New design…" in the problem picker.

Keyboard: V select, S support, L load, K keep-in, O keep-out, Esc cancel, Ctrl+Z undo.

## X6b: interaction cost by Keystroke-Level Model (`scripts/x6b_klm.py`; `klm.json`, `klm_summary.md`)

The 15 X3 tasks. Each condition ends with a certified release and an STL.
- **A1, a hand-written voxfem script.**
  - The problem setup is generated per task as an expert would write it (numpy masks, fixed bits, nodal loads).
  - Every setup is **executed and checked identical to `problem.build`**: fixed DOFs, load vector, solid and void
    masks, 15/15.
  - Around the setup come imports, the run, an STL export and, for tasks with passive regions, a SIMP subclass:
    voxfem has none, so the expert must write one. The subclass and the run lines are counted but not executed.
  - Scripts: `scripts/<task>_voxfem.py`.
- **A2, the spec typed for the agentto library.** The spec as a Python dict, without default fields, then
  set_spec / solve / export. This is the lower bound for any scripting route. Scripts: `scripts/<task>_agentto.py`.
- **B, the GUI without the agent.**
  - New design; every item placed with S / L / K / O and a click on a face; the inspector then corrects what the
    GUI filled in, using the GUI's own defaults (`Viewport.onDomainClick`, `Panels.convert`).
  - Only values that differ from those defaults are typed. A clicked patch centre always counts as typed.
  - One orbit drag each time the next face is on the far side.
- **C, the GUI with the agent** (Propose mode, X2 G+C repeat 1, scripted oracle):
  - the brief is typed or pasted;
  - one chip click per clarification question;
  - one mental operator per card to review it, then Accept all;
  - X6c found no card to edit in G+C;
  - then the same release and export clicks as B.
- **AutoSiMP:** not run. It is 2D, and all the tasks are 3D.

**KLM operators** (Card, Moran and Newell 1983; Kieras 2001):
- K 0.28 s; P 1.1 s; BB 0.2 s; M 1.35 s;
- H 0.4 s, inserted at every switch between keyboard and mouse;
- M also before each chunk: a code line, a typed value, a choice.

Solver and agent time is waiting and is not counted.

| Medians over the 14 released tasks | User time | Actions (keystrokes + clicks) |
|---|---|---|
| A1 voxfem script (24–55 lines) | 480 s | ≈ 1,100–2,700 |
| A2 agentto spec typed | 208 s | – |
| B GUI, no agent | 104 s | 94 |
| C GUI + agent, brief typed | 141 s | ≈ 290–560 |
| **C GUI + agent, brief pasted** | **25 s** | **8** |
| (agent run, waiting, X2 G+C wall) | 47 s | – |

Per task (`klm_summary.md`):
- The GUI alone costs 67–374 s; the most is b06, with 4 keep-ins and holes, and b19.
- With a pasted brief, C costs 20–42 s.

**What this supports.**
- With a written brief (a requirements text, a ticket), the agent cuts user setup to a few clicks: about 4× less than
  the GUI alone, about 19× less than scripting.
- Typing the whole brief into the agent costs more than building simple tasks by hand (b01: 125 s against 68 s). The
  agent wins on user time only on the complex tasks (b06: 190 s against 374 s; b19: 188 s against 277 s).
- So the claim is about not having to translate an existing text, not about typing less.
- All conditions assume an error-free user, so they leave out debugging. For A1 and B the setup errors G made (X2:
  8 silent wrong specs) are the kind a person makes too; the cost model does not capture them.

## X6c: action cards and validation catches (`scripts/x6c_cards.py`; `cards.json`, `cards_summary.md`)

Setup:
- Each X2 run's final spec is replayed as one Propose run on an empty workspace: `spec_diff` gives the cards the GUI
  shows.
- The scripted oracle accepts a card that is already right (the same fixed DOFs, loads or mask as an item of the
  credited reading), edits a wrong card that stands for something the reading has, and rejects the rest.
- Repeat 1, 50 briefs.

| Condition | Cards | Accepted | Edited | Rejected | Runs with every card right |
|---|---|---|---|---|---|
| T (35 runs with a valid spec) | 292 | 99.7% | 1 | 0 | 34/35 |
| G | 436 | 95.9% | 6 | 12 | 42/50 |
| G+C | 423 | **100%** | 0 | 0 | **50/50** |
| G+C+V | 423 | 100% | 0 | 0 | 50/50 |

Validation, from the G and G+C tool logs:
- Every session called `validate_spec`.
- Validation reported an error in 3 sessions per condition: R003 on b08 and b49, R008 on b50. Those are exactly the
  three refuse tasks, and in each the agent stopped without solving.
- **None of G's 8 wrong specs failed validation** (b07, b25, b39, b44–b48). They are errors of intent, which no rule
  can see.
- Clarification removed all 8: G+C has none.

So the rule base catches ill-posed problems, clarification catches misreadings, and the two do not overlap on this
suite.

Consequence chips: under a scripted oracle, the chips (previews of each reading's consequences) cannot change which
reading is chosen, because the oracle answers from the brief's facts. Their effect on people is left to the user
study named as future work.

## X6a: latency (`scripts/x6a_latency.py`; `latency/<task>.json`, `latency_summary.md`)

Setup:
- the 14 released X3 tasks, through the running backend exactly as the GUI drives it (WebSocket tool calls and events,
  HTTP for meshes);
- the pad-aware coarse rule;
- 20 repeats of validation, the edit commit and edit → sketch; the ladder once per task.

Front-end frame times (three.js) are not measured.

| Interaction | Target | Median over tasks | Range | Met |
|---|---|---|---|---|
| Spec edit → validation, p95 | < 200 ms | 2.5 ms | 1.2–10.5 ms | 14/14 |
| **Edit → sketch, warm**, p95 | 1–3 s | **0.95 s** | 0.59–2.4 s | 14/14 |
| Edit → first sketch frame | – | 0.09 s | 0.08–0.22 s | – |
| Sketch, cold (new spec) | 1–3 s | 3.3 s | 2.5–6.7 s | **4/14** |
| Preview, warm | ≈ 5 s | 0.8 s | 0.4–18 s | 13/14 |
| Refine, warm (compute) | 2–5 s | 1.5 s | 0.7–17.5 s | 12/14 |
| Release, warm (compute; all 14 certified) | ≈ 10–25 s | 6.7 s | 3.1–56 s | 13/14 |
| Release surface on screen | < 3 s | 0.14 s | 0.08–0.71 s | 14/14 |
| STL export | – | 0.59 s | 0.30–2.2 s | – |

**What it shows.**
- The interactive loop meets its budget on every task: a spec edit validates in milliseconds, and the re-solved
  sketch is back in under 2.4 s (p95).
- **A cold sketch misses its target.** It is a new spec with no field to warm from, so it runs the full
  76–82-iteration schedule (2.4–6.7 s). The first density frame is on screen in about 0.1 s (b01: 1.4 s), so the
  user sees the design forming at once. Report the target as missed for cold sketches.
- The outliers are the two tasks whose rungs did not converge:
  - b17 arch bridge: the preview stopped at its cap (18 s);
  - b40 crank arm: the refine stopped at its cap (16 s), and the release then needed 56 s.
  Both releases are certified.
- Releases here use the GUI's default release, which certifies the warm design (the tail arm). The "Optimize
  release" button (reopen_long) costs 51–69% of a cold run (X1).
