"""X6b: interaction cost of setting up and releasing the 15 X3 tasks, by Keystroke-Level Model (no participants).

Conditions (each ends with a certified release and an STL):
    A1  a hand-written `voxfem` script. The problem setup (node and element masks, fixed bits, nodal loads) is
        generated per task the way an expert would write it with numpy, and is executed here and checked against
        `problem.build` on the task's grid (fixed DOFs, load vector, solid and void masks must be identical), so the
        count is for a script that works. Around it: imports, the solver run, an STL export and, for tasks with
        keep-in / keep-out regions, a SIMP subclass that imposes them (voxfem has no passive regions).
    A2  a hand-written spec for the agentto library (the spec as a Python dict without default fields, then
        set_spec / solve / export through `tools.Session`): the lower bound for scripting.
    B   the GUI without the agent: New design, then every item placed with the viewport tools (S, L, K, O + a click
        on a face) and corrected in the inspector, as the GUI builds them (`Viewport.onDomainClick`,
        `Panels.convert`): only values that differ from what the GUI fills in are typed.
    C   the GUI with the agent in Propose mode, answered by the scripted oracle (X2 condition G+C, repeat 1): the
        user gives the brief (typed, or pasted when it already exists as text), answers each clarification with one
        chip, reviews each card (one mental operator per card) and accepts all; X6c found no card to edit or reject
        in G+C. Then the same release and export clicks as B. The agent's own run time is waiting, not user work,
        and is reported separately.
AutoSiMP is not run: it is 2D, and every X3 task is 3D.

Operators (Card, Moran and Newell 1983; Kieras 2001): K 0.28 s (average skilled typist), P 1.1 s, B 0.1 s (a click
is BB), H 0.4 s (inserted automatically at every switch between keyboard and mouse), M 1.35 s (before each chunk:
each typed value, each choice, each code line). Waiting for the system is not counted.

    python -B scripts/x6b_klm.py        writes results/X6/klm.json and results/X6/klm_summary.md
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentto.problem import build, grid_for  # noqa: E402
from agentto.spec import Spec  # noqa: E402

BRIEFS = ROOT / "briefs" / "intended"
X2 = ROOT / "results" / "X2"
X6 = ROOT / "results" / "X6"
TASKS = ["b01_cantilever_block", "b02_mbb_half", "b04_torsion_shaft", "b05_l_bracket", "b06_motor_mount",
         "b11_cantilever_line_load", "b17_arch_bridge", "b19_engine_bracket", "b25_cantilever_metres",
         "b31_pillow_block", "b35_padeye", "b40_crank_arm", "b44_gearbox_bracket", "b47_camera_mount",
         "b49_hinged_arm"]
T = {"K": 0.28, "P": 1.1, "B": 0.1, "H": 0.4, "M": 1.35}
AX = {"x": 0, "y": 1, "z": 2}


def num(v) -> str:
    """A number as a person would type it."""
    v = float(v)
    return str(int(v)) if v == int(v) and abs(v) < 1e15 else f"{v:.6g}"


def klm(ops: list[str]) -> dict:
    """Operator counts and time; H is inserted at every keyboard/mouse switch."""
    seq, hand = [], None
    for o in ops:
        dev = "kbd" if o == "K" else "mouse" if o in "PB" else None
        if dev and hand and dev != hand:
            seq.append("H")
        if dev:
            hand = dev
        seq.append(o)
    n = {k: seq.count(k) for k in T}
    return {"ops": n, "actions": n["K"] + n["B"] // 2, "s": round(sum(T[k] * c for k, c in n.items()), 1)}


# ---- A1: a voxfem script -------------------------------------------------------------------------------

def _node_mask(r: dict, L) -> str:
    k = r["kind"]
    if k in ("face", "patch"):
        a = AX[r["face"][0]]
        A, (b, c) = "XYZ"[a], [i for i in range(3) if i != a]
        m = f"({A} <= t)" if r["face"][1] == "-" else f"({A} >= {num(L[a])} - t)"
        if k == "face":
            return m
        Bn, Cn = "XYZ"[b], "XYZ"[c]
        (cb, cc), (sb, sc) = r["center"], r["size"]
        if r.get("shape") == "ellipse":
            return (f"{m} & ((({Bn} - {num(cb)}) / {num(sb / 2)}) ** 2 + (({Cn} - {num(cc)}) / {num(sc / 2)}) ** 2 "
                    f"<= 1 + 1e-9)")
        return f"{m} & (abs({Bn} - {num(cb)}) <= {num(sb / 2)} + t) & (abs({Cn} - {num(cc)}) <= {num(sc / 2)} + t)"
    if k == "edge":
        a = AX[r["axis"]]
        b, c = [i for i in range(3) if i != a]
        m = f"(abs({'XYZ'[b]} - {num(r['at'][0])}) <= t) & (abs({'XYZ'[c]} - {num(r['at'][1])}) <= t)"
        if r.get("span"):
            m += f" & ({'XYZ'[a]} >= {num(r['span'][0])} - t) & ({'XYZ'[a]} <= {num(r['span'][1])} + t)"
        return m
    if k == "box":
        return " & ".join(f"({v} >= {num(lo)} - t) & ({v} <= {num(hi)} + t)"
                          for v, lo, hi in zip("XYZ", r["min"], r["max"]))
    return _cyl(r, "XYZ", "t")


def _cyl(r: dict, V: str, t: str) -> str:
    a = AX[r["axis"]]
    b, c = [i for i in range(3) if i != a]
    return (f"(({V[b]} - {num(r['center'][0])}) ** 2 + ({V[c]} - {num(r['center'][1])}) ** 2 <= ({num(r['radius'])} + "
            f"{t}) ** 2) & ({V[a]} >= {num(r['span'][0])} - {t}) & ({V[a]} <= {num(r['span'][1])} + {t})")


def _elem_mask(r: dict) -> str:
    if r["kind"] == "box":
        return " & ".join(f"({v} >= {num(lo)}) & ({v} <= {num(hi)})" for v, lo, hi in zip(("XC", "YC", "ZC"),
                                                                                     r["min"], r["max"]))
    return _cyl(r, ("XC", "YC", "ZC"), "t")


def setup_code(spec: dict) -> str:
    """The problem setup an expert writes with numpy and voxfem.core (executed and checked by `check_setup`)."""
    s = Spec.model_validate(spec)
    h, g = grid_for(s)
    L = s.domain.size
    out = [f"L = [{', '.join(num(v) for v in L)}]",
           f"N = [{g.nx}, {g.ny}, {g.nz}]",
           "g = Grid(*N)",
           "x, y, z = (np.arange(n + 1) * l / n for n, l in zip(N, L))",
           "X, Y, Z = np.meshgrid(x, y, z, indexing='ij')",
           "t = 1e-9 * max(l / n for n, l in zip(N, L))",
           "fixed = np.zeros(X.shape, np.uint8)"]
    for sp in spec["supports"]:
        bits = sum({"x": 1, "y": 2, "z": 4}[a] for a in sp.get("fix", ["x", "y", "z"]))
        out.append(f"fixed[{_node_mask(sp['region'], L)}] |= {bits}")
    out.append("idx, val = [], []")
    for ld in spec["loads"]:
        out.append(f"n = np.flatnonzero({_node_mask(ld['region'], L)})")
        for c, f in enumerate(ld["force"]):
            if f:
                out.append(f"idx.append(3 * n + {c}); val.append(np.full(n.size, {num(f)} / n.size))")
    out.append("load = SparseLoad(np.concatenate(idx), np.concatenate(val)).coalesce()")
    out.append("problem = Problem('task', g, fixed.ravel(), load)")
    if spec.get("passive"):
        out += ["xc, yc, zc = ((np.arange(n) + 0.5) * l / n for n, l in zip(N, L))",
                "XC, YC, ZC = np.meshgrid(xc, yc, zc, indexing='ij')",
                "solid = np.zeros(XC.shape, bool)", "void = np.zeros(XC.shape, bool)"]
        for p in spec["passive"]:
            out.append(f"{'solid' if p['kind'] == 'solid' else 'void'} |= {_elem_mask(p['region'])}")
        out.append("solid, void = solid.ravel(), void.ravel()")
    return "\n".join(out)


HEAD = """import numpy as np
import cupy as cp
from voxfem.core import Grid, Problem, SparseLoad, pad_elements
from voxfem.simp import SIMP, Config
from skimage.measure import marching_cubes
import trimesh"""

PASSIVE = """class PassiveSIMP(SIMP):
    def __init__(self, problem, cfg, solid, void):
        n_design = int((~(solid | void)).sum())
        cfg.volfrac = (cfg.volfrac * n_design + solid.sum()) / solid.size
        super().__init__(problem, cfg)
        g0, g = problem.grid, self.problem.grid
        self.keep = cp.asarray(pad_elements((~(solid | void)).astype(np.float32), g0, g))
        self.solid = cp.asarray(pad_elements(solid.astype(np.float32), g0, g))
    def physical(self, rho, beta, out):
        super().physical(rho, beta, out)
        out *= self.keep
        out += self.solid
        return out
    def sensitivities(self, p, beta):
        super().sensitivities(p, beta)
        self.dc *= self.keep
        self.dv *= self.keep"""

RUN = """cfg = Config(volfrac={vf}, solver="mg64")
{make}
rho = s.run()
v, f, _, _ = marching_cubes(rho.reshape(N), 0.5, spacing=[l / n for n, l in zip(N, L)])
trimesh.Trimesh(v, f).export("part.stl")"""


def voxfem_script(spec: dict) -> str:
    make = ("s = PassiveSIMP(problem, cfg, solid, void)" if spec.get("passive") else "s = SIMP(problem, cfg)")
    parts = [HEAD] + ([PASSIVE] if spec.get("passive") else []) + [setup_code(spec),
                                                                   RUN.format(vf=num(spec["volume_fraction"]), make=make)]
    return "\n".join(parts) + "\n"


def check_setup(spec: dict) -> bool:
    """Run the generated setup and compare it with problem.build on the task's own grid."""
    from agentto._voxfem import Grid, Problem, SparseLoad
    env = {"np": np, "Grid": Grid, "Problem": Problem, "SparseLoad": SparseLoad}
    exec(setup_code(spec), env)                                   # noqa: S102 -- our own generated code
    sp = build(Spec.model_validate(spec))
    p = env["problem"]
    ok = np.array_equal(p.fixed, sp.problem.fixed)
    a, b = p.load, sp.problem.load
    ok &= a.idx.size == b.idx.size and np.array_equal(a.idx, b.idx) and np.allclose(a.val, b.val, rtol=1e-9)
    z = np.zeros(sp.grid.n_elem, bool)
    if spec.get("passive"):
        ok &= np.array_equal(env["solid"], z if sp.solid is None else sp.solid)
        ok &= np.array_equal(env["void"], z if sp.void is None else sp.void)
    return bool(ok)


def code_ops(text: str) -> list[str]:
    """Typing a script: M before each non-blank line, K per character and one K for each newline."""
    ops = []
    for line in text.splitlines():
        s = line.strip()
        if s:
            ops += ["M"] + ["K"] * (len(s) + 1)           # indentation comes from the editor
    return ops


# ---- A2: the agentto library ------------------------------------------------------------------------

def library_script(spec: dict) -> str:
    lean = Spec.model_validate(spec).model_dump(mode="json", exclude_defaults=True)
    lean.pop("schema_version", None)
    lean.pop("description", None)
    body = json.dumps(lean, separators=(", ", ": "))
    return ("from agentto.tools import Session\n"
            "s = Session(None)\n"
            f"s.call('set_spec', {{'spec': {body}}})\n"
            "job = s.call('solve', {'rung': 'release', 'optimize': True})['job']\n"
            "s.call('wait_job', {'job': job})\n"
            "s.call('export_stl', {'job': job})\n")


def library_ops(text: str) -> list[str]:
    """The spec literal is typed as chunks: M before each item or field, K per character."""
    ops = []
    for line in text.splitlines():
        if "set_spec" in line:
            chunks = line.count('"id"') + line.count('"volume_fraction"') + line.count('"material"') + \
                line.count('"domain"') + line.count('"units"') + line.count('"resolution"') + 1
            ops += ["M"] * chunks + ["K"] * (len(line) + 1)
        else:
            ops += ["M"] + ["K"] * (len(line) + 1)
    return ops


# ---- B: the GUI without the agent --------------------------------------------------------------------

def click():
    return ["P", "B", "B"]


def field(v) -> list[str]:
    """Type one value into a number or text field: point, click, select all, type."""
    return ["M"] + click() + ["K", "K"] + ["K"] * len(v if isinstance(v, str) else num(v))


def choose():
    """Pick from a select: open it, click the option."""
    return ["M"] + click() + click()


def r4(v):
    return float(f"{v:.4g}")


def gui_default(kind: str, face: str, L) -> dict:
    """Region the GUI makes when its kind select is switched to `kind` (Panels.convert, from a region without axis)."""
    fa = AX[face[0]]
    fb, fc = [i for i in range(3) if i != fa]
    ia = AX["z"]
    ib, ic = [i for i in range(3) if i != ia]
    if kind == "face":
        return {"kind": "face", "face": face}
    if kind == "patch":
        return {"kind": "patch", "face": face, "center": [r4(L[fb] / 2), r4(L[fc] / 2)],
                "size": [r4(L[fb] / 10), r4(L[fc] / 10)], "shape": "rect"}
    if kind == "edge":
        return {"kind": "edge", "axis": "z", "at": [0, 0], "span": None}
    if kind == "box":
        return {"kind": "box", "min": [r4(v * 0.45) for v in L], "max": [r4(v * 0.55) for v in L]}
    return {"kind": "cylinder", "axis": "z", "center": [r4(L[ib] / 2), r4(L[ic] / 2)],
            "radius": r4(min(L[ib], L[ic]) / 10), "span": [0, L[ia]]}


def region_ops(want: dict, have: dict, L) -> list[str]:
    """Inspector edits that turn the GUI's region `have` into `want`."""
    ops = []
    if want["kind"] != have["kind"]:
        ops += choose()
        have = gui_default(want["kind"], have.get("face", "x+"), L)
    k = want["kind"]
    if k in ("face", "patch") and want["face"] != have["face"]:
        ops += choose()
    if k == "patch":
        for key in ("center", "size"):
            for a, b in zip(want[key], have[key]):
                if key == "center" or abs(a - b) > 1e-9:          # a clicked centre is never exact
                    ops += field(a)
        if want.get("shape", "rect") != have.get("shape", "rect"):
            ops += choose()
    if k in ("edge", "cylinder") and want["axis"] != have["axis"]:
        ops += choose()
    if k == "edge":
        ops += [o for a, b in zip(want["at"], have["at"]) if abs(a - b) > 1e-9 for o in field(a)]
        if want.get("span"):
            ops += ["M"] + click()                                  # "limit"
            full = [0, L[AX[want["axis"]]]]
            ops += [o for a, b in zip(want["span"], full) if abs(a - b) > 1e-9 for o in field(a)]
    if k == "box":
        for key in ("min", "max"):
            ops += [o for a, b in zip(want[key], have[key]) if abs(a - b) > 1e-9 for o in field(a)]
    if k == "cylinder":
        ops += [o for a, b in zip(want["center"], have["center"]) if abs(a - b) > 1e-9 for o in field(a)]
        if abs(want["radius"] - have["radius"]) > 1e-9:
            ops += field(want["radius"])
        ops += [o for a, b in zip(want["span"], have["span"]) if abs(a - b) > 1e-9 for o in field(a)]
    return ops


class View:
    """The default camera shows the x+, y+ and z+ faces; reaching a face on the other side takes one orbit drag."""

    def __init__(self):
        self.back = False

    def face(self, f: str) -> list[str]:
        back = f[1] == "-"
        if back == self.back:
            return []
        self.back = back
        return ["M", "P", "B", "P", "B"]


def place(tool_key: str, face: str, view: View) -> list[str]:
    return ["M", "K"] + view.face(face) + ["M"] + click()


def gui_ops(spec: dict) -> list[str]:
    L = spec["domain"]["size"]
    ops = choose()                                                  # problem picker → New design…
    u = spec.get("units", {"length": "mm", "force": "N"})
    ops += choose() if u.get("length", "mm") != "mm" else []
    ops += choose() if u.get("force", "N") != "N" else []
    ops += [o for a, b in zip(L, [100, 50, 50]) if abs(a - b) > 1e-9 for o in field(a)]
    if abs(spec["volume_fraction"] - 0.3) > 1e-9:
        ops += field(num(round(spec["volume_fraction"] * 100, 6)))
    res = spec.get("resolution") or {}
    if res.get("h") is not None:
        ops += choose() + field(res["h"])
    elif res.get("cells_short_edge") not in (None, 20):
        ops += field(res["cells_short_edge"])
    if spec.get("min_feature"):
        ops += field(spec["min_feature"])
    m = spec.get("material", {})
    if m and (m.get("E", 210e3) != 210e3 or m.get("nu", 0.3) != 0.3 or m.get("name", "steel") != "steel"):
        ops += ["M"] + click()                                      # tree: material
        ops += field(m["name"]) if m.get("name", "steel") != "steel" else []
        ops += field(m["E"]) if m.get("E", 210e3) != 210e3 else []
        ops += field(m["nu"]) if m.get("nu", 0.3) != 0.3 else []
    view = View()
    for s in spec["supports"]:
        r = s["region"]
        face = r.get("face", "x+")
        ops += place("S", face, view)
        ops += region_ops(r, {"kind": "face", "face": face}, L)
        missing = [a for a in "xyz" if a not in s.get("fix", ["x", "y", "z"])]
        ops += (["M"] + [o for _ in missing for o in click()]) if missing else []
    for ld in spec["loads"]:
        r = ld["region"]
        face = r.get("face", "x+")
        ops += place("L", face, view)
        a = AX[face[0]]
        b, c = [i for i in range(3) if i != a]
        have = {"kind": "patch", "face": face, "center": [None, None], "size": [r4(L[b] / 10), r4(L[c] / 10)],
                "shape": "rect"}
        ops += region_ops(r, have, L)
        n = [0.0, 0.0, 0.0]
        n[a] = -1000.0 if face[1] == "+" else 1000.0
        ops += [o for f, d in zip(ld["force"], n) if abs(f - d) > 1e-12 for o in field(f)]
    for p in spec.get("passive") or []:
        r = p["region"]
        face = "x+"
        ops += place("K" if p["kind"] == "solid" else "O", face, view)
        a = AX[face[0]]
        lo, hi = [0.0, 0.0, 0.0], [float(v) for v in L]
        lo[a] = r4(L[a] * 0.9)
        ops += region_ops(r, {"kind": "box", "min": lo, "max": hi}, L)
    return ops


def release_ops() -> list[str]:
    return ["M"] + click() + ["M"] + click()                        # Optimize release, then Export


# ---- C: the GUI with the agent -----------------------------------------------------------------------

def agent_ops(brief: str, questions: int, cards: int, typed: bool) -> list[str]:
    ops = ["M"] + click()                                           # focus the assistant box
    ops += ["M"] + ["K"] * len(brief) if typed else ["K", "K"]      # type, or paste the brief
    ops += ["K"]                                                    # Enter
    ops += [o for _ in range(questions) for o in ["M"] + click()]   # one chip per question
    ops += ["M"] * cards + click()                                  # review each card, Accept all
    return ops


def main():
    from agentto.app.agent import spec_diff
    rows = []
    for name in TASKS:
        task = json.loads((BRIEFS / f"{name}.json").read_text(encoding="utf-8"))
        spec, brief = task["spec"], (ROOT / "briefs" / f"{name}.md").read_text(encoding="utf-8").strip()
        run = json.loads((X2 / "GC" / f"{name}_r1.json").read_text(encoding="utf-8"))
        cards = [c for c in spec_diff(None, run["spec"]) if c.get("field") not in ("name", "description")] \
            if isinstance(run.get("spec"), dict) else []
        a1, a2 = voxfem_script(spec), library_script(spec)
        row = {"task": name, "expected": task.get("expected"),
               "A1": {**klm(code_ops(a1) + ["M"] + ["K"] * 14), "lines": len([x for x in a1.splitlines() if x.strip()]),
                      "chars": len(a1), "setup_verified": check_setup(spec)},
               "A2": {**klm(library_ops(a2) + ["M"] + ["K"] * 14), "chars": len(a2)},
               "B": klm(gui_ops(spec) + release_ops()),
               "C_typed": klm(agent_ops(brief, run.get("questions", 0), len(cards), True) + release_ops()),
               "C_pasted": klm(agent_ops(brief, run.get("questions", 0), len(cards), False) + release_ops()),
               "agent_wait_s": round(run.get("wall_s") or 0, 1), "questions": run.get("questions", 0),
               "cards": len(cards), "gc_correct": run.get("correct_spec")}
        if task.get("expected") == "refuse":                        # stops at validation: no release, no export
            for k in ("B", "C_typed", "C_pasted"):
                row[k]["note"] = "includes release and export, which a refused task never reaches"
        rows.append(row)
        print(name, {k: row[k]["s"] for k in ("A1", "A2", "B", "C_typed", "C_pasted")}, row["A1"]["setup_verified"])
        (X6 / "scripts").mkdir(parents=True, exist_ok=True)
        (X6 / "scripts" / f"{name}_voxfem.py").write_text(a1, encoding="utf-8")
        (X6 / "scripts" / f"{name}_agentto.py").write_text(a2, encoding="utf-8")
    (X6 / "klm.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    summarize(rows)


def summarize(rows):
    rs = [r for r in rows if r["expected"] != "refuse"]
    med = lambda k, f="s": statistics.median(r[k][f] for r in rs)  # noqa: E731
    lines = [f"# X6b: interaction cost by Keystroke-Level Model ({len(rs)} released X3 tasks; b49 is refused)", "",
             "Operators: K 0.28 s, P 1.1 s, BB 0.2 s, H 0.4 s, M 1.35 s. Agent and solver time is waiting and is not "
             "counted. Actions are keystrokes plus clicks.", "",
             "| Task | A1 voxfem script: lines, s | A2 agentto spec: s | B GUI, no agent: actions, s | "
             "C GUI + agent (brief typed / pasted): actions, s | Questions, cards | Agent run, s |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        tag = " (refused)" if r["expected"] == "refuse" else ""
        lines.append(f"| {r['task']}{tag} | {r['A1']['lines']}, {r['A1']['s']:.0f}"
                     f"{'' if r['A1']['setup_verified'] else ' (setup NOT verified)'} | {r['A2']['s']:.0f} | "
                     f"{r['B']['actions']}, {r['B']['s']:.0f} | {r['C_typed']['actions']}, {r['C_typed']['s']:.0f} / "
                     f"{r['C_pasted']['actions']}, {r['C_pasted']['s']:.0f} | {r['questions']}, {r['cards']} | "
                     f"{r['agent_wait_s']:.0f} |")
    lines += ["", f"Medians over the {len(rs)} released tasks: A1 {med('A1'):.0f} s, A2 {med('A2'):.0f} s, "
              f"B {med('B'):.0f} s ({med('B', 'actions'):.0f} actions), C typed {med('C_typed'):.0f} s, "
              f"C pasted {med('C_pasted'):.0f} s ({med('C_pasted', 'actions'):.0f} actions); agent run "
              f"{statistics.median(r['agent_wait_s'] for r in rs):.0f} s of waiting.",
              f"A1 setups verified against problem.build: {sum(r['A1']['setup_verified'] for r in rows)}/{len(rows)}."]
    (X6 / "klm_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
