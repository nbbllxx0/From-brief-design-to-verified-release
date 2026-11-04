"""Writes the X2 task suite: briefs b01-b50 and their intended readings. Re-run to regenerate.

The intended readings are Claude's drafts for the author to adjudicate (briefs/README.md). Conventions:
- "volume fraction" is the share of the design (non-passive) volume, as in commercial TO tools; every brief uses
  that term, so the briefs carry no unintended ambiguity about it;
- an admissible reading that counts carries its own spec, so the scorer can credit it (bench.match_task);
- b01-b05 and b11-b30 restate published benchmark families in physical units (the source names the family; the
  dimensions are ours); b06-b10 and b31-b50 are engineering briefs, of which the clarify and refuse tasks are
  deliberate.
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentto.spec import Spec  # noqa: E402

STEEL = {"name": "steel", "E": 210000.0, "nu": 0.3}
ALU = {"name": "aluminium 6061", "E": 69000.0, "nu": 0.33}
U = {"length": "mm", "force": "N"}


def patch(face, center, size, shape="rect"):
    return {"kind": "patch", "face": face, "center": list(center), "size": list(size), "shape": shape}


def edge(axis, at, span=None):
    return {"kind": "edge", "axis": axis, "at": list(at), "span": span}


def face(f):
    return {"kind": "face", "face": f}


def box(lo, hi):
    return {"kind": "box", "min": list(lo), "max": list(hi)}


def cyl(axis, center, radius, span):
    return {"kind": "cylinder", "axis": axis, "center": list(center), "radius": radius, "span": list(span)}


def solid(i, region, note=""):
    return {"id": i, "kind": "solid", "region": region, "note": note}


def void(i, region, note=""):
    return {"id": i, "kind": "void", "region": region, "note": note}


TASKS = []


def task(slug, brief, spec, expected="solve", source="engineering", oracle=None, admissible=None, notes="",
         accept_warnings=()):
    spec = {"schema_version": "agentto.spec/v0", "units": spec.pop("units", U), **spec}
    Spec.model_validate(spec)
    for a in admissible or []:
        if a.get("spec") is not None:
            Spec.model_validate(a["spec"])
    TASKS.append((slug, brief.strip() + "\n", {"spec": spec, "expected": expected, "source": source,
                                                "oracle": oracle or [], "admissible": admissible or [],
                                                "notes": notes,
                                                **({"accept_warnings": list(accept_warnings)} if accept_warnings
                                                   else {})}))
    return spec


def alt(spec, fn):
    """An admissible reading: a copy of `spec` changed by fn(copy)."""
    s = copy.deepcopy(spec)
    fn(s)
    return s


# ---- b01-b10 (draft 1, revised) ------------------------------------------------------------------------------

task("b01_cantilever_block", """
Bracket design, rev A

Envelope: a rectangular block 200 mm long (x), 100 mm tall (y) and 100 mm deep (z). Steel, E = 210 GPa,
nu = 0.3.

Mounting: the whole end face at x = 0 is welded to a wall (fully fixed).

Load: 1 kN pushing straight down (negative y) on a 10 mm x 10 mm pad in the middle of the free end face (x = 200).

Volume fraction 0.3. Mesh at 1 mm elements.
""", {
    "name": "b01 cantilever block", "domain": {"size": [200, 100, 100]}, "material": STEEL,
    "volume_fraction": 0.3,
    "supports": [{"id": "wall", "region": face("x-")}],
    "loads": [{"id": "pad", "region": patch("x+", (50, 50), (10, 10)), "force": [0, -1000, 0]}],
    "resolution": {"h": 1.0}},
    source="restates the voxfem solver's 3D cantilever benchmark (after Sigmund) in physical units")

b02 = task("b02_mbb_half", """
Simply supported beam, symmetric half model

The full beam is 240 mm long (x), 40 mm deep (y, up) and 40 mm wide (z), aluminium 6061. It sits on two
supports along the bottom edges at its two ends, across the full width; the supports carry vertical and
sideways load only. A 2 kN downward line load acts across the full width at mid-span on the top surface.

Model only the left half (x from 0 to 120 mm) and use symmetry at mid-span. Volume fraction 0.3, 1 mm elements.
""", {
    "name": "b02 MBB half", "domain": {"size": [120, 40, 40]}, "material": ALU, "volume_fraction": 0.3,
    "supports": [{"id": "end_support", "fix": ["y", "z"], "region": edge("z", (0, 0))},
                 {"id": "symmetry", "fix": ["x"], "region": face("x+")}],
    "loads": [{"id": "midspan", "region": edge("z", (120, 40)), "force": [0, -1000, 0]}],
    "resolution": {"h": 1.0}},
    source="3D MBB beam (half model), as in Liu & Tovar 2014, top3d",
    notes="Half of the 2 kN acts on the half model.")
TASKS[-1][2]["admissible"] = [
    {"reading": "full 240 mm model with both end supports and the full 2 kN",
     "why": "the brief asks for the half model; the full model is equivalent but not what was asked", "counts": False},
    {"reading": "the end support also fixes x (pinned)", "why": "over-constrains a roller; changes the design little "
     "because the symmetry face already restrains x", "counts": True,
     "spec": alt(b02, lambda s: s["supports"][0].update(fix=["x", "y", "z"]))}]

task("b03_bridge_deck", """
Bridge segment

A deck segment spanning 400 mm (x), 100 mm high (y, up) and 100 mm wide (z), steel. It rests on two bearings
along the bottom edges at each end, running across the full width. The left bearing is pinned; the right one
lets the deck expand along the span. A uniform 5 kN load acts downward on the entire top surface.

Volume fraction 0.25, 2 mm elements.
""", {
    "name": "b03 bridge deck", "domain": {"size": [400, 100, 100]}, "material": STEEL, "volume_fraction": 0.25,
    "supports": [{"id": "pin", "region": edge("z", (0, 0))},
                 {"id": "expansion", "fix": ["y", "z"], "region": edge("z", (400, 0))}],
    "loads": [{"id": "deck", "region": face("y+"), "force": [0, -5000, 0]}],
    "resolution": {"h": 2.0}},
    source="voxfem solver benchmark bridge with a distributed deck load")

# b04 was ambiguous in draft 1 (no up axis; lugs on or inside the edges): every X2 condition read it differently
# from the intended spec. It is now stated fully; the ambiguous wording became the clarify task b44.
task("b04_torsion_shaft", """
Shaft bracket in torsion

Envelope 160 mm long (x) with an 80 x 80 mm square cross-section (y up, z), steel. The x = 0 end is bolted
rigidly to a gearbox. At the far end (x = 160) a 40 N*m torque about +x (right-hand rule) is applied through two
16 x 16 mm drive lugs on the end face: one flush with the top edge (y = 80), one flush with the bottom edge
(y = 0), both centred at z = 40. Each lug carries an equal and opposite force along z.

Volume fraction 0.3, 2 mm elements.
""", {
    "name": "b04 torsion shaft", "domain": {"size": [160, 80, 80]}, "material": STEEL, "volume_fraction": 0.3,
    "supports": [{"id": "gearbox", "region": face("x-")}],
    "loads": [{"id": "lug_top", "region": patch("x+", (72, 40), (16, 16)), "force": [0, 0, 625]},
              {"id": "lug_bottom", "region": patch("x+", (8, 40), (16, 16)), "force": [0, 0, -625]}],
    "resolution": {"h": 2.0}},
    source="voxfem solver benchmark torsion in physical units",
    notes="40 N*m = 40000 N*mm; the lug centres are 64 mm apart, so a 625 N couple in z. M_x = r_y F_z: the top "
          "lug (r_y = +32) pushes +z.")

task("b05_l_bracket", """
L-bracket

Envelope 100 x 100 x 40 mm (x, y, z), y up, steel. The upper-right corner region, x > 40 mm and y > 40 mm, is
occupied by another part and must stay empty through the full depth. The top face of the vertical arm
(y = 100, x from 0 to 40) is bolted to the ceiling. A 500 N downward load acts on a 10 x 10 mm pad on the
right end face of the horizontal arm (x = 100), centred at y = 20 and mid-depth.

Volume fraction 0.3, 1 mm elements.
""", {
    "name": "b05 L-bracket", "domain": {"size": [100, 100, 40]}, "material": STEEL, "volume_fraction": 0.3,
    "supports": [{"id": "ceiling", "region": patch("y+", (20, 20), (40, 40))}],
    "loads": [{"id": "tip", "region": patch("x+", (20, 20), (10, 10)), "force": [0, -500, 0]}],
    "passive": [void("occupied", box((40, 40, 0), (100, 100, 40)))],
    "resolution": {"h": 1.0}},
    source="3D L-bracket benchmark (e.g. Le et al. 2010), keep-out as a passive void")

b06 = task("b06_motor_mount", """
Motor mount

Plate envelope 160 x 60 x 120 mm (x, y, z), y up, aluminium 6061. It bolts to the machine bed through four M10
bolts on its bottom face (y = 0) at x = 20 and 140 mm, z = 20 and 100 mm; each bolt clamps through a 24 mm
washer. Keep a solid boss of 24 mm diameter and 10 mm height around each bolt.

The motor sits on a 40 x 40 mm, 5 mm thick solid pad on the top face, centred at x = 80, z = 60. Its
mounting pad carries 2 kN downward and 0.8 kN sideways (+x) from belt tension.

Volume fraction 0.2, 1 mm elements.
""", {
    "name": "b06 motor mount", "domain": {"size": [160, 60, 120]}, "material": ALU, "volume_fraction": 0.2,
    "supports": [{"id": f"bolt_{i}", "region": patch("y-", (x, z), (24, 24), "ellipse")}
                 for i, (x, z) in enumerate([(20, 20), (20, 100), (140, 20), (140, 100)], 1)],
    "loads": [{"id": "motor", "region": patch("y+", (80, 60), (40, 40)), "force": [800, -2000, 0]}],
    "passive": [solid(f"boss_{i}", cyl("y", (x, z), 12, (0, 10)))
                for i, (x, z) in enumerate([(20, 20), (20, 100), (140, 20), (140, 100)], 1)]
               + [solid("pad", box((60, 55, 40), (100, 60, 80)))],
    "resolution": {"h": 1.0}},
    notes="Bolt holes inside the bosses are not modelled: the bolt fills the hole, and a keep-out hole inside a "
          "keep-in boss would conflict (R007).")
TASKS[-1][2]["admissible"] = [
    {"reading": "square 24 x 24 mm washer patches", "why": "small change in support area", "counts": True,
     "spec": alt(b06, lambda s: [x["region"].update(shape="rect") for x in s["supports"]])}]

task("b07_sensor_arm", """
Sensor arm

Envelope 150 x 30 x 30 mm, the arm running along x. It carries a 2 kg camera (about 20 N) at the far end,
mounted on a 20 x 20 mm plate in the middle of the end face at x = 150 mm. The arm attaches to the robot frame.

Steel, volume fraction 0.25, 1 mm elements.
""", {
    "name": "b07 sensor arm", "domain": {"size": [150, 30, 30]}, "material": STEEL, "volume_fraction": 0.25,
    "supports": [{"id": "frame", "region": face("x-")}],
    "loads": [{"id": "camera", "region": patch("x+", (15, 15), (20, 20)), "force": [0, -20, 0]}],
    "resolution": {"h": 1.0}},
    expected="clarify",
    oracle=[{"about": "where and how the arm attaches", "answer": "The x = 0 end face is bolted flat to the "
             "frame; treat it as fully fixed."},
            {"about": "direction of the camera load", "answer": "Gravity acts along -y."}],
    notes="Two decision-relevant gaps: the attachment and the load direction.")

task("b08_slider_bracket", """
Slider bracket

Envelope 120 x 40 x 40 mm (x, y, z), y up, steel. It rides on a linear rail along its whole bottom face
(y = 0): the rail holds it vertically and sideways but lets it slide freely along x. A 300 N push along +x acts
on a 10 x 10 mm pad at the centre of the left end face (x = 0).

Volume fraction 0.3, 1 mm elements.
""", {
    "name": "b08 slider bracket", "domain": {"size": [120, 40, 40]}, "material": STEEL, "volume_fraction": 0.3,
    "supports": [{"id": "rail", "fix": ["y", "z"], "region": face("y-")}],
    "loads": [{"id": "push", "region": patch("x-", (20, 20), (10, 10)), "force": [300, 0, 0]}],
    "resolution": {"h": 1.0}},
    expected="refuse",
    oracle=[{"about": "any other support or an end stop", "answer": "No, it is supported only by the rail."}],
    notes="As stated the part is a mechanism along x (rule R003): nothing reacts the push. A correct agent "
          "reports that and asks for an x restraint rather than inventing one.")

task("b09_duct_bridge", """
Pedestrian bridge model with cable duct

Span 300 mm (x), 60 mm high (y, up), 60 mm wide (z), steel. Both bottom end edges rest on bearings across the
full width: left pinned, right free to slide along the span. 3 kN acts downward on the full top surface.
A 16 mm diameter cable duct runs across the full width (along z) at x = 150 mm, y = 20 mm and must stay empty.
No member may be thinner than 3 mm.

Volume fraction 0.25, 1 mm elements.
""", {
    "name": "b09 duct bridge", "domain": {"size": [300, 60, 60]}, "material": STEEL, "volume_fraction": 0.25,
    "supports": [{"id": "pin", "region": edge("z", (0, 0))},
                 {"id": "slide", "fix": ["y", "z"], "region": edge("z", (300, 0))}],
    "loads": [{"id": "deck", "region": face("y+"), "force": [0, -3000, 0]}],
    "passive": [void("duct", cyl("z", (150, 20), 8, (0, 60)))],
    "min_feature": 3.0, "resolution": {"h": 1.0}})

task("b10_hook_plate", """
Hook plate

Plate envelope 120 x 80 x 20 mm (x, y, z), y up, steel. It is bolted along its top face (y = 80) to a beam.
A 1.5 kN load hangs from the lower corner.

Volume fraction 0.3, 1 mm elements.
""", {
    "name": "b10 hook plate", "domain": {"size": [120, 80, 20]}, "material": STEEL, "volume_fraction": 0.3,
    "supports": [{"id": "beam", "region": face("y+")}],
    "loads": [{"id": "hook", "region": patch("y-", (115, 10), (10, 20)), "force": [0, -1500, 0]}],
    "resolution": {"h": 1.0}},
    expected="clarify",
    oracle=[{"about": "which lower corner, and the attachment size",
             "answer": "The lower right corner (x = 120). The hook hangs from a 10 mm wide lug on the bottom face "
                       "at the right end (x from 110 to 120), across the full thickness."}],
    admissible=[{"reading": "the mirror image (lower left corner)",
                 "why": "the design is the mirror image; it is not the intended corner, but compliance is equal",
                 "counts": False}])

# ---- b11-b30: restated benchmark families --------------------------------------------------------------------

task("b11_cantilever_line_load", """
Cantilever with an end line load

Beam envelope 120 mm long (x), 40 mm tall (y, up) and 20 mm wide (z), steel. The x = 0 end face is fully
fixed. A 1 kN downward load is spread evenly along the bottom edge of the free end (x = 120, y = 0), across
the full width.

Volume fraction 0.3, 1 mm elements.
""", {
    "name": "b11 cantilever line load", "domain": {"size": [120, 40, 20]}, "material": STEEL,
    "volume_fraction": 0.3,
    "supports": [{"id": "wall", "region": face("x-")}],
    "loads": [{"id": "tip_edge", "region": edge("z", (120, 0)), "force": [0, -1000, 0]}],
    "resolution": {"h": 1.0}},
    source="3D cantilever with an end line load, the default problem of top3d (Liu & Tovar 2014)")

task("b12_wheel", """
Wheel problem

Envelope 100 x 50 x 100 mm (x, y, z), y up, steel. The part stands on four 5 x 5 mm pads, one in each corner
of its bottom face (y = 0); each pad is fixed in all directions. A 2 kN downward load hangs from a 10 x 10 mm
pad at the centre of the bottom face.

Volume fraction 0.2, 1 mm elements.
""", {
    "name": "b12 wheel", "domain": {"size": [100, 50, 100]}, "material": STEEL, "volume_fraction": 0.2,
    "supports": [{"id": f"corner_{i}", "region": patch("y-", (x, z), (5, 5))}
                 for i, (x, z) in enumerate([(2.5, 2.5), (97.5, 2.5), (2.5, 97.5), (97.5, 97.5)], 1)],
    "loads": [{"id": "centre", "region": patch("y-", (50, 50), (10, 10)), "force": [0, -2000, 0]}],
    "resolution": {"h": 1.0}},
    source="the 'wheel' problem of top3d (Liu & Tovar 2014): bottom corners fixed, load at the bottom centre")

task("b13_cantilever_end_traction", """
Cantilever with a distributed end load

Envelope 160 x 80 x 80 mm (x, y, z), y up, steel. The x = 0 face is fully fixed. The last 4 mm of the free end
(x from 156 to 160) is a solid end plate that must be kept. A 4 kN downward shear load is spread uniformly over
the whole free end face (x = 160).

Volume fraction 0.12, 2 mm elements.
""", {
    "name": "b13 cantilever end traction", "domain": {"size": [160, 80, 80]}, "material": STEEL,
    "volume_fraction": 0.12,
    "supports": [{"id": "wall", "region": face("x-")}],
    "loads": [{"id": "end", "region": face("x+"), "force": [0, -4000, 0]}],
    "passive": [solid("end_plate", box((156, 0, 0), (160, 80, 80)))],
    "resolution": {"h": 2.0}},
    source="2:1:1 cantilever with a distributed end load, the large-scale benchmark of the PETSc TopOpt framework "
           "(Aage et al. 2015)",
    notes="The solid end plate carries the distributed load; without it the loaded face oscillates and the preview "
          "reference does not settle (results/X2/suite_check.json, 2026-10-30).")

task("b14_cube_torsion", """
Block in torsion

A 100 mm cube (x, y, z), z up, steel. The bottom face (z = 0) is bolted down over its full area. An 18 N*m
torque about +z (right-hand rule, counter-clockwise seen from above) is applied to the top face through four
10 x 10 mm pads, one at the midpoint of each top edge and flush with that edge. Each pad carries an equal force
tangential to the edge it sits on.

Volume fraction 0.15, 1 mm elements.
""", {
    "name": "b14 cube torsion", "domain": {"size": [100, 100, 100]}, "material": STEEL, "volume_fraction": 0.15,
    "supports": [{"id": "base", "region": face("z-")}],
    "loads": [{"id": "pad_front", "region": patch("z+", (50, 5), (10, 10)), "force": [100, 0, 0]},
              {"id": "pad_right", "region": patch("z+", (95, 50), (10, 10)), "force": [0, 100, 0]},
              {"id": "pad_back", "region": patch("z+", (50, 95), (10, 10)), "force": [-100, 0, 0]},
              {"id": "pad_left", "region": patch("z+", (5, 50), (10, 10)), "force": [0, -100, 0]}],
    "resolution": {"h": 1.0}},
    source="3D torsion benchmark family (Michell-type torsion in a box domain)",
    notes="Pad centres are 45 mm from the axis: 4 x 45 mm x F = 18000 N*mm gives F = 100 N. M_z = r_x F_y - r_y F_x.")

task("b15_corner_plate", """
Plate on four corner supports

Square plate envelope 200 x 20 x 200 mm (x, y, z), y up, aluminium 6061. It is pinned (fixed in all
directions) at four 6 x 6 mm pads in the corners of its bottom face. A 3 kN downward load acts on a
20 x 20 mm pad at the centre of the top face.

Volume fraction 0.3, 2 mm elements.
""", {
    "name": "b15 corner plate", "domain": {"size": [200, 20, 200]}, "material": ALU, "volume_fraction": 0.3,
    "supports": [{"id": f"corner_{i}", "region": patch("y-", (x, z), (6, 6))}
                 for i, (x, z) in enumerate([(3, 3), (197, 3), (3, 197), (197, 197)], 1)],
    "loads": [{"id": "centre", "region": patch("y+", (100, 100), (20, 20)), "force": [0, -3000, 0]}],
    "resolution": {"h": 2.0}},
    source="point-loaded square plate on four corner supports (3D plate benchmark family)")

task("b16_column", """
Column under combined load

Column envelope 60 x 240 x 60 mm (x, y, z), y up, steel. It stands on four 10 x 10 mm footing pads in the
corners of its bottom face, each fixed in all directions. A 20 x 20 mm cap pad at the centre of the top face
carries 10 kN downward and 1 kN sideways along +x.

Volume fraction 0.2, 2 mm elements.
""", {
    "name": "b16 column", "domain": {"size": [60, 240, 60]}, "material": STEEL, "volume_fraction": 0.2,
    "supports": [{"id": f"foot_{i}", "region": patch("y-", (x, z), (10, 10))}
                 for i, (x, z) in enumerate([(5, 5), (55, 5), (5, 55), (55, 55)], 1)],
    "loads": [{"id": "cap", "region": patch("y+", (30, 30), (20, 20)), "force": [1000, -10000, 0]}],
    "resolution": {"h": 2.0}},
    source="tower / column under an axial and a lateral top load (3D benchmark family)")

task("b17_arch_bridge", """
Arch bridge with a solid deck

Envelope 300 mm span (x), 100 mm high (y, up), 60 mm wide (z), steel. The top 6 mm (y from 94 to 100) is the
deck and must stay solid. The bridge is pinned (fixed in all directions) along both bottom end edges, across
the full width. A 6 kN load acts downward uniformly over the whole deck surface (y = 100).

Volume fraction 0.15, 2 mm elements.
""", {
    "name": "b17 arch bridge", "domain": {"size": [300, 100, 60]}, "material": STEEL, "volume_fraction": 0.15,
    "supports": [{"id": "abutment_left", "region": edge("z", (0, 0))},
                 {"id": "abutment_right", "region": edge("z", (300, 0))}],
    "loads": [{"id": "deck_load", "region": face("y+"), "force": [0, -6000, 0]}],
    "passive": [solid("deck", box((0, 94, 0), (300, 100, 60)))],
    "resolution": {"h": 2.0}},
    source="bridge benchmark with a passive solid deck and pinned abutments (arch-forming)")

task("b18_cantilever_hole", """
Cantilever with a hole

Envelope 200 x 100 x 40 mm (x, y, z), y up, steel. The x = 0 face is fully fixed. A 2 kN downward load acts
on a 10 x 10 mm pad at the centre of the free end face (x = 200). A 40 mm diameter hole runs through the full
thickness (along z), centred at x = 100, y = 50; it must stay empty.

Volume fraction 0.3, 1 mm elements.
""", {
    "name": "b18 cantilever hole", "domain": {"size": [200, 100, 40]}, "material": STEEL, "volume_fraction": 0.3,
    "supports": [{"id": "wall", "region": face("x-")}],
    "loads": [{"id": "tip", "region": patch("x+", (50, 20), (10, 10)), "force": [0, -2000, 0]}],
    "passive": [void("hole", cyl("z", (100, 50), 20, (0, 40)))],
    "resolution": {"h": 1.0}},
    source="cantilever with a circular keep-out (passive-void benchmark family)")

TI_IN = {"name": "Ti-6Al-4V", "E": 16.5e6, "nu": 0.34}
task("b19_engine_bracket", """
Engine bracket (inch units)

Envelope 7.0 x 2.5 x 4.0 in (x, y, z), y up, Ti-6Al-4V (E = 16.5 Mpsi, nu = 0.34). Work in inches and pounds-force.
It bolts down through four bolts on its bottom face (y = 0) at x = 0.5 and 2.5 in, z = 0.5 and 3.5 in, each
clamping through a 0.5 in diameter washer. The clevis is a solid cylinder that must be kept: axis along z,
centred at x = 6.0 in, y = 1.75 in, radius 0.5 in, from z = 1.25 to 2.75 in. The design load is an 8000 lbf
vertical pull (+y) carried by the clevis.

Volume fraction 0.2, 0.05 in elements.
""", {
    "name": "b19 engine bracket", "units": {"length": "in", "force": "lbf"}, "domain": {"size": [7.0, 2.5, 4.0]},
    "material": TI_IN, "volume_fraction": 0.2,
    "supports": [{"id": f"bolt_{i}", "region": patch("y-", (x, z), (0.5, 0.5), "ellipse")}
                 for i, (x, z) in enumerate([(0.5, 0.5), (0.5, 3.5), (2.5, 0.5), (2.5, 3.5)], 1)],
    "loads": [{"id": "pull", "region": cyl("z", (6.0, 1.75), 0.5, (1.25, 2.75)), "force": [0, 8000, 0]}],
    "passive": [solid("clevis", cyl("z", (6.0, 1.75), 0.5, (1.25, 2.75)))],
    "resolution": {"h": 0.05}},
    source="after the GE jet-engine bracket challenge (GrabCAD 2013): box envelope, its vertical load case only",
    notes="A spec converted to mm and N is the same problem (the scorer compares in mm and N).")

b20 = task("b20_quarter_plate", """
Plate, quarter model

The full plate is 240 x 24 x 240 mm (x, y, z), y up, steel. It is pinned (fixed in all directions) at four
8 x 8 mm pads in the corners of its bottom face, and carries 4 kN downward on a 20 x 20 mm pad at the centre of
its top face.

Model one quarter only (x and z from 0 to 120 mm) and use the two symmetry planes at x = 120 and z = 120.
Volume fraction 0.25, 1 mm elements.
""", {
    "name": "b20 quarter plate", "domain": {"size": [120, 24, 120]}, "material": STEEL, "volume_fraction": 0.25,
    "supports": [{"id": "corner", "region": patch("y-", (4, 4), (8, 8))},
                 {"id": "sym_x", "fix": ["x"], "region": face("x+")},
                 {"id": "sym_z", "fix": ["z"], "region": face("z+")}],
    "loads": [{"id": "centre_quarter", "region": patch("y+", (115, 115), (10, 10)), "force": [0, -1000, 0]}],
    "resolution": {"h": 1.0}},
    source="point-loaded plate on corner supports, quarter model with two symmetry planes",
    admissible=[{"reading": "the full plate", "why": "equivalent, but the brief asks for the quarter model",
                 "counts": False}],
    notes="A quarter of the load (1 kN) on a quarter of the pad (10 x 10 mm at the symmetry corner).")

task("b21_mbb_z_up", """
Simply supported beam, z up

Beam envelope 320 mm long (x), 40 mm wide (y) and 40 mm tall (z), with z pointing up; steel. The left bottom
edge (x = 0, z = 0) is pinned across the full width. The right bottom edge (x = 320, z = 0) is a roller across
the full width: it carries vertical and sideways load but lets the beam move along x. A 3 kN downward line load
acts across the full width at mid-span on the top surface.

Volume fraction 0.3, 2 mm elements.
""", {
    "name": "b21 MBB z up", "domain": {"size": [320, 40, 40]}, "material": STEEL, "volume_fraction": 0.3,
    "supports": [{"id": "pin", "region": edge("y", (0, 0))},
                 {"id": "roller", "fix": ["y", "z"], "region": edge("y", (320, 0))}],
    "loads": [{"id": "midspan", "region": edge("y", (160, 40)), "force": [0, 0, -3000]}],
    "resolution": {"h": 2.0}},
    source="3D MBB beam, full model (Liu & Tovar 2014), stated with z up")

task("b22_cantilever_slab", """
Cantilevered slab

Slab envelope 150 mm out from the wall (x), 20 mm thick (y, up) and 200 mm wide (z), steel. The wall face
(x = 0) is fully fixed. A 6 kN load acts downward, spread uniformly over the whole top surface.

Volume fraction 0.3, 2 mm elements.
""", {
    "name": "b22 cantilever slab", "domain": {"size": [150, 20, 200]}, "material": STEEL, "volume_fraction": 0.3,
    "supports": [{"id": "wall", "region": face("x-")}],
    "loads": [{"id": "pressure", "region": face("y+"), "force": [0, -6000, 0]}],
    "resolution": {"h": 2.0}},
    source="cantilevered plate under a uniform load (distributed-load cantilever family)",
    accept_warnings=["R005"],
    notes="The top face shares its wall edge with the support, so 1.3% of the load lands on fixed nodes (R005 "
          "warning); that is what a uniform load on the whole surface means.")

task("b23_clamped_beam", """
Clamped-clamped beam

Beam envelope 240 x 40 x 40 mm (x, y, z), y up, steel. Both end faces (x = 0 and x = 240) are fully clamped.
A 3 kN downward load acts on a 10 mm long strip across the full width of the top face at mid-span
(x from 115 to 125).

Volume fraction 0.25, 1 mm elements.
""", {
    "name": "b23 clamped beam", "domain": {"size": [240, 40, 40]}, "material": STEEL, "volume_fraction": 0.25,
    "supports": [{"id": "left", "region": face("x-")}, {"id": "right", "region": face("x+")}],
    "loads": [{"id": "centre", "region": patch("y+", (120, 20), (10, 40)), "force": [0, -3000, 0]}],
    "resolution": {"h": 1.0}},
    source="clamped-clamped beam with a central load (3D benchmark family)")

task("b24_shear_block", """
Block in shear

An 80 mm cube (x, y, z), y up, steel. The bottom face (y = 0) is fully fixed. A 4 kN horizontal load along +x
is spread uniformly over the whole top face.

Volume fraction 0.2, 1 mm elements.
""", {
    "name": "b24 shear block", "domain": {"size": [80, 80, 80]}, "material": STEEL, "volume_fraction": 0.2,
    "supports": [{"id": "base", "region": face("y-")}],
    "loads": [{"id": "shear", "region": face("y+"), "force": [4000, 0, 0]}],
    "resolution": {"h": 1.0}},
    source="shear-loaded block (3D benchmark family)")

task("b25_cantilever_metres", """
Large cantilever (metres and kilonewtons)

Envelope 2 m long (x), 1 m tall (y, up) and 1 m wide (z), structural steel, E = 210 GPa, nu = 0.3. Work in
metres and kilonewtons. The x = 0 face is fully fixed. A 100 kN downward load acts on a 0.2 x 0.2 m bearing
plate at the centre of the free end face (x = 2 m).

Volume fraction 0.15, 0.02 m elements.
""", {
    "name": "b25 cantilever metres", "units": {"length": "m", "force": "kN"}, "domain": {"size": [2.0, 1.0, 1.0]},
    "material": {"name": "structural steel", "E": 2.1e8, "nu": 0.3}, "volume_fraction": 0.15,
    "supports": [{"id": "wall", "region": face("x-")}],
    "loads": [{"id": "bearing", "region": patch("x+", (0.5, 0.5), (0.2, 0.2)), "force": [0, -100, 0]}],
    "resolution": {"h": 0.02}},
    source="2:1:1 cantilever (Aage et al. 2015 family), in metres and kN",
    notes="E = 210 GPa = 2.1e8 kN/m^2.")

task("b26_canopy", """
Canopy on four columns

Plate envelope 300 x 40 x 200 mm (x, y, z), y up, aluminium 6061. It rests on four columns whose 20 x 20 mm
heads meet its bottom face (y = 0) centred at x = 40 and 260 mm, z = 40 and 160 mm; each head is fixed in all
directions. An 8 kN load acts downward, spread uniformly over the whole top surface.

Volume fraction 0.25, 2 mm elements.
""", {
    "name": "b26 canopy", "domain": {"size": [300, 40, 200]}, "material": ALU, "volume_fraction": 0.25,
    "supports": [{"id": f"column_{i}", "region": patch("y-", (x, z), (20, 20))}
                 for i, (x, z) in enumerate([(40, 40), (260, 40), (40, 160), (260, 160)], 1)],
    "loads": [{"id": "roof", "region": face("y+"), "force": [0, -8000, 0]}],
    "resolution": {"h": 2.0}},
    source="plate on four interior column supports under a uniform load (multi-support slab family)")

task("b27_eccentric_cantilever", """
Cantilever with an eccentric end load

Envelope 160 x 40 x 40 mm (x, y, z), y up, steel. The x = 0 face is fully fixed. A 1 kN downward load acts on
a 10 x 10 mm pad in the top back corner of the free end face (x = 160): y from 30 to 40 mm and z from 30 to
40 mm. The beam therefore carries bending and torsion.

Volume fraction 0.25, 1 mm elements.
""", {
    "name": "b27 eccentric cantilever", "domain": {"size": [160, 40, 40]}, "material": STEEL,
    "volume_fraction": 0.25,
    "supports": [{"id": "wall", "region": face("x-")}],
    "loads": [{"id": "corner", "region": patch("x+", (35, 35), (10, 10)), "force": [0, -1000, 0]}],
    "resolution": {"h": 1.0}},
    source="cantilever with an eccentric end load, bending plus torsion (3D benchmark family)")

task("b28_stool", """
Stool

Envelope 200 x 200 x 200 mm (x, y, z), y up, aluminium 6061. The seat, the top 10 mm (y from 190 to 200), must
stay solid. It stands on four 20 x 20 mm feet in the corners of its bottom face (y = 0), each fixed in all
directions. The space under the middle of the seat, x and z from 50 to 150 mm and y from 0 to 120 mm, must stay
empty for leg room. A 1.2 kN load acts downward, spread over the whole seat surface.

Volume fraction 0.1, 2 mm elements.
""", {
    "name": "b28 stool", "domain": {"size": [200, 200, 200]}, "material": ALU, "volume_fraction": 0.1,
    "supports": [{"id": f"foot_{i}", "region": patch("y-", (x, z), (20, 20))}
                 for i, (x, z) in enumerate([(10, 10), (190, 10), (10, 190), (190, 190)], 1)],
    "loads": [{"id": "sitter", "region": face("y+"), "force": [0, -1200, 0]}],
    "passive": [solid("seat", box((0, 190, 0), (200, 200, 200))),
                void("leg_room", box((50, 0, 50), (150, 120, 150)))],
    "resolution": {"h": 2.0}},
    source="stool / table problem: top load on a passive seat, four feet (3D benchmark family)")

task("b29_michell_cantilever", """
Michell cantilever on a pin

Slab envelope 200 x 100 x 20 mm (x, y, z), y up, steel. It is held by a rigid 30 mm diameter pin through the
full thickness (along z) at x = 30, y = 50: everything inside the pin's circle is fixed in all directions. A
500 N downward load acts on a 10 mm tall strip of the right end face (x = 200), centred at y = 50, across the
full thickness.

Volume fraction 0.2, 1 mm elements.
""", {
    "name": "b29 Michell cantilever", "domain": {"size": [200, 100, 20]}, "material": STEEL,
    "volume_fraction": 0.2,
    "supports": [{"id": "pin", "region": cyl("z", (30, 50), 15, (0, 20))}],
    "loads": [{"id": "tip", "region": patch("x+", (50, 10), (10, 20)), "force": [0, -500, 0]}],
    "resolution": {"h": 1.0}},
    source="Michell cantilever on a circular support (Michell 1904), as a 3D slab")

task("b30_bridge_clearance", """
Bridge over a clearance

Envelope 320 mm span (x), 80 mm high (y, up), 80 mm wide (z), steel. The left bottom edge (x = 0, y = 0) is
pinned across the full width; the right bottom edge (x = 320, y = 0) is a roller across the full width that
lets the span expand along x. The top 4 mm (y from 76 to 80) is the deck and must stay solid. A 5 kN load acts
downward over the whole deck surface. Under mid-span, a clearance
x from 100 to 220 mm, y from 0 to 40 mm, through the full width, must stay empty.

Volume fraction 0.2, 2 mm elements.
""", {
    "name": "b30 bridge clearance", "domain": {"size": [320, 80, 80]}, "material": STEEL, "volume_fraction": 0.2,
    "supports": [{"id": "pin", "region": edge("z", (0, 0))},
                 {"id": "roller", "fix": ["y", "z"], "region": edge("z", (320, 0))}],
    "loads": [{"id": "deck", "region": face("y+"), "force": [0, -5000, 0]}],
    "passive": [solid("deck_plate", box((0, 76, 0), (320, 80, 80))), void("clearance", box((100, 0, 0), (220, 40, 80)))],
    "resolution": {"h": 2.0}},
    source="bridge with a passive-void clearance under the span (bridge benchmark family)")

# ---- b31-b43: engineering briefs, fully stated ----------------------------------------------------------------

task("b31_pillow_block", """
Pillow-block bearing bracket

Envelope 120 x 60 x 40 mm (x, y, z), y up, grey cast iron (E = 110 GPa, nu = 0.26). Two M8 bolts hold its bottom
face (y = 0) down, at x = 15 and 105 mm, z = 20 mm, each through a 16 mm washer. The bearing seat is a solid
cylinder that must be kept: axis along z, centred at x = 60, y = 40, 30 mm diameter, through the full depth.
The shaft pushes 3 kN downward on the seat.

Volume fraction 0.3, 1 mm elements.
""", {
    "name": "b31 pillow block", "domain": {"size": [120, 60, 40]},
    "material": {"name": "grey cast iron", "E": 110000.0, "nu": 0.26}, "volume_fraction": 0.3,
    "supports": [{"id": "bolt_left", "region": patch("y-", (15, 20), (16, 16), "ellipse")},
                 {"id": "bolt_right", "region": patch("y-", (105, 20), (16, 16), "ellipse")}],
    "loads": [{"id": "shaft", "region": cyl("z", (60, 40), 15, (0, 40)), "force": [0, -3000, 0]}],
    "passive": [solid("seat", cyl("z", (60, 40), 15, (0, 40)))],
    "resolution": {"h": 1.0}})

task("b32_drone_arm", """
Drone arm

Arm envelope 150 mm long (x), 20 mm wide (y) and 20 mm tall (z), with z pointing up; aluminium 7075. The root
end face (x = 0) is clamped in the frame. The motor mounts on a 20 x 20 mm, 5 mm thick solid pad on the top of
the far end (x from 130 to 150, the full width, z from 15 to 20), which must be kept. Full thrust is 15 N
upward (+z), acting on the top face of that pad.

Volume fraction 0.2, 0.5 mm elements.
""", {
    "name": "b32 drone arm", "domain": {"size": [150, 20, 20]},
    "material": {"name": "aluminium 7075", "E": 71700.0, "nu": 0.33}, "volume_fraction": 0.2,
    "supports": [{"id": "frame", "region": face("x-")}],
    "loads": [{"id": "thrust", "region": patch("z+", (140, 10), (20, 20)), "force": [0, 0, 15]}],
    "passive": [solid("motor_pad", box((130, 0, 15), (150, 20, 20)))],
    "resolution": {"h": 0.5}})

task("b33_shelf_bracket", """
Shelf bracket

Envelope 150 mm out from the wall (x), 150 mm tall (y, up) and 20 mm thick (z), aluminium 6061. Two screws
fix it to the wall face (x = 0), at y = 20 and 130 mm, z = 10 mm, each through a 10 mm washer. The shelf rests
on the whole top face (y = 150) and puts 600 N downward on it. No member may be thinner than 3 mm.

Volume fraction 0.2, 1 mm elements.
""", {
    "name": "b33 shelf bracket", "domain": {"size": [150, 150, 20]}, "material": ALU, "volume_fraction": 0.2,
    "supports": [{"id": "screw_low", "region": patch("x-", (20, 10), (10, 10), "ellipse")},
                 {"id": "screw_high", "region": patch("x-", (130, 10), (10, 10), "ellipse")}],
    "loads": [{"id": "shelf", "region": face("y+"), "force": [0, -600, 0]}],
    "min_feature": 3.0, "resolution": {"h": 1.0}})

task("b34_antenna_bracket", """
Antenna wall bracket (SI units)

Envelope 0.40 m out from the wall (x), 0.30 m tall (y, up) and 0.20 m wide (z); S355 steel, E = 210 GPa,
nu = 0.3. Work in metres and newtons. The wall face (x = 0) is welded over its full area. The mast clamp is a
solid vertical cylinder that must be kept: axis along y, centred at x = 0.33 m, z = 0.10 m, radius 0.04 m, from
y = 0.05 m to the top. The clamp carries the antenna's 300 N weight (downward) and a 1200 N wind load along +z.

Volume fraction 0.15, 5 mm elements.
""", {
    "name": "b34 antenna bracket", "units": {"length": "m", "force": "N"}, "domain": {"size": [0.4, 0.3, 0.2]},
    "material": {"name": "S355 steel", "E": 2.1e11, "nu": 0.3}, "volume_fraction": 0.15,
    "supports": [{"id": "wall", "region": face("x-")}],
    "loads": [{"id": "antenna", "region": cyl("y", (0.33, 0.10), 0.04, (0.05, 0.3)), "force": [0, -300, 1200]}],
    "passive": [solid("clamp", cyl("y", (0.33, 0.10), 0.04, (0.05, 0.3)))],
    "resolution": {"h": 0.005}},
    notes="E = 2.1e11 N/m^2; 5 mm elements = 0.005 m.")

task("b35_padeye", """
Lifting padeye

Envelope 100 x 80 x 30 mm (x, y, z), y up, steel. The base (y = 0) is welded to the deck over its full area.
The shackle pin passes through a solid boss that must be kept: axis along z, centred at x = 50, y = 55, 30 mm
diameter, through the full thickness. The sling pulls the boss with 6 kN upward and 2.5 kN along +x.

Volume fraction 0.3, 1 mm elements.
""", {
    "name": "b35 padeye", "domain": {"size": [100, 80, 30]}, "material": STEEL, "volume_fraction": 0.3,
    "supports": [{"id": "weld", "region": face("y-")}],
    "loads": [{"id": "sling", "region": cyl("z", (50, 55), 15, (0, 30)), "force": [2500, 6000, 0]}],
    "passive": [solid("boss", cyl("z", (50, 55), 15, (0, 30)))],
    "resolution": {"h": 1.0}})

task("b36_pedal_arm", """
Pedal arm

Arm envelope 200 mm long (x), 40 mm tall (y, up) and 20 mm thick (z), aluminium 6061. It is keyed to a 16 mm
shaft through its thickness at x = 15, y = 20 mm, so treat everything inside the shaft's circle as fixed in all
directions. The foot pushes 1 kN downward on a 30 mm long pad on the top face at the far end (x from 170 to 200),
across the full thickness.

Volume fraction 0.25, 1 mm elements.
""", {
    "name": "b36 pedal arm", "domain": {"size": [200, 40, 20]}, "material": ALU, "volume_fraction": 0.25,
    "supports": [{"id": "shaft", "region": cyl("z", (15, 20), 8, (0, 20))}],
    "loads": [{"id": "foot", "region": patch("y+", (185, 10), (30, 20)), "force": [0, -1000, 0]}],
    "resolution": {"h": 1.0}})

task("b37_battery_crossmember", """
Battery-tray crossmember (kN and mm)

Envelope 600 mm long (x), 60 mm tall (y, up) and 80 mm deep (z), aluminium 6082-T6 (E = 70 GPa, nu = 0.33).
Work in millimetres and kilonewtons. Both end faces (x = 0 and x = 600) bolt to the side rails and are fully
fixed. The battery module stands on two 40 x 40 mm feet on the top face, centred at x = 200 and 400 mm,
z = 40 mm; each foot puts 1.5 kN downward.

Volume fraction 0.2, 2 mm elements.
""", {
    "name": "b37 battery crossmember", "units": {"length": "mm", "force": "kN"},
    "domain": {"size": [600, 60, 80]}, "material": {"name": "aluminium 6082-T6", "E": 70.0, "nu": 0.33},
    "volume_fraction": 0.2,
    "supports": [{"id": "rail_left", "region": face("x-")}, {"id": "rail_right", "region": face("x+")}],
    "loads": [{"id": "foot_1", "region": patch("y+", (200, 40), (40, 40)), "force": [0, -1.5, 0]},
              {"id": "foot_2", "region": patch("y+", (400, 40), (40, 40)), "force": [0, -1.5, 0]}],
    "resolution": {"h": 2.0}},
    notes="E = 70 GPa = 70 kN/mm^2.")

task("b38_peek_bracket", """
Satellite equipment bracket

Envelope 80 x 60 x 40 mm (x, y, z), z normal to the panel; PEEK (E = 3.6 GPa, nu = 0.38). The base (z = 0) is
bonded to the panel over its full area. The unit mounts on a 30 x 30 mm, 4 mm thick solid interface pad on the
top face (z = 40), centred at x = 40, y = 30, which must be kept. Design for the launch load on the pad: 300 N
along +x and 300 N along -z together.

Volume fraction 0.25, 1 mm elements.
""", {
    "name": "b38 PEEK bracket", "domain": {"size": [80, 60, 40]},
    "material": {"name": "PEEK", "E": 3600.0, "nu": 0.38}, "volume_fraction": 0.25,
    "supports": [{"id": "panel", "region": face("z-")}],
    "loads": [{"id": "launch", "region": patch("z+", (40, 30), (30, 30)), "force": [300, 0, -300]}],
    "passive": [solid("interface_pad", box((25, 15, 36), (55, 45, 40)))],
    "resolution": {"h": 1.0}})

task("b39_tv_bracket", """
TV wall bracket

Envelope 300 mm out from the wall (x), 200 mm tall (y, up) and 40 mm wide (z), steel. Four lag screws fix it to
the wall face (x = 0) at y = 30 and 170 mm, z = 10 and 30 mm, each through a 12 mm washer. The TV hangs on a
mounting plate that covers the far end face (x = 300) from y = 50 to 150 mm across the full width; it puts
400 N downward and 50 N pulling away from the wall (+x) on that plate. No member may be thinner than 5 mm.

Volume fraction 0.15, 2 mm elements.
""", {
    "name": "b39 TV bracket", "domain": {"size": [300, 200, 40]}, "material": STEEL, "volume_fraction": 0.15,
    "supports": [{"id": f"screw_{i}", "region": patch("x-", (y, z), (12, 12), "ellipse")}
                 for i, (y, z) in enumerate([(30, 10), (30, 30), (170, 10), (170, 30)], 1)],
    "loads": [{"id": "tv", "region": patch("x+", (100, 20), (100, 40)), "force": [50, -400, 0]}],
    "min_feature": 5.0, "resolution": {"h": 2.0}})

task("b40_crank_arm", """
Crank arm

Envelope 180 mm long (x), 40 mm tall (y, up) and 30 mm thick (z), aluminium 7075. The axle end is keyed to a
20 mm diameter spindle through the thickness at x = 20, y = 20 mm: everything inside that circle is fixed. The
pedal threads into a solid boss that must be kept: axis along z at x = 160, y = 20 mm, 16 mm diameter, through
the full thickness. The rider's 1.2 kN downward push arrives on the boss's outer face (z = 30), over the 16 mm
circle.

Volume fraction 0.3, 1 mm elements.
""", {
    "name": "b40 crank arm", "domain": {"size": [180, 40, 30]},
    "material": {"name": "aluminium 7075", "E": 71700.0, "nu": 0.33}, "volume_fraction": 0.3,
    "supports": [{"id": "spindle", "region": cyl("z", (20, 20), 10, (0, 30))}],
    "loads": [{"id": "pedal", "region": patch("z+", (160, 20), (16, 16), "ellipse"), "force": [0, -1200, 0]}],
    "passive": [solid("pedal_boss", cyl("z", (160, 20), 8, (0, 30)))],
    "resolution": {"h": 1.0}},
    notes="The load sits on the outer face, so the arm carries bending and torsion.")

task("b41_hinge_bracket", """
Door hinge bracket

Envelope 140 x 80 x 30 mm (x, y, z), y up, steel. Two bolts hold its bottom face (y = 0) at x = 20 and 120 mm,
z = 15 mm, each through a 12 mm washer. The hinge pin runs through a solid boss that must be kept: axis along
z, centred at x = 70, y = 65 mm, 20 mm diameter, full thickness. The door hangs 800 N downward and pulls 300 N
along +x on the boss. A cable slot, x from 40 to 100 mm and y from 20 to 35 mm, through the full thickness,
must stay empty.

Volume fraction 0.3, 1 mm elements.
""", {
    "name": "b41 hinge bracket", "domain": {"size": [140, 80, 30]}, "material": STEEL, "volume_fraction": 0.3,
    "supports": [{"id": "bolt_left", "region": patch("y-", (20, 15), (12, 12), "ellipse")},
                 {"id": "bolt_right", "region": patch("y-", (120, 15), (12, 12), "ellipse")}],
    "loads": [{"id": "door", "region": cyl("z", (70, 65), 10, (0, 30)), "force": [300, -800, 0]}],
    "passive": [solid("hinge_boss", cyl("z", (70, 65), 10, (0, 30))),
                void("cable_slot", box((40, 20, 0), (100, 35, 30)))],
    "resolution": {"h": 1.0}})

task("b42_snap_hook", """
Printed snap hook

Envelope 40 mm long (x), 10 mm tall (y, up) and 12 mm wide (z), printed in PA12 (E = 1.7 GPa, nu = 0.4). The
root end face (x = 0) is part of the housing and fully fixed. On assembly the tip is pushed 15 N upward (+y) on
a 4 mm long strip of its underside (y = 0, x from 36 to 40), across the full width. No member may be thinner
than 1.2 mm.

Volume fraction 0.35, 0.25 mm elements.
""", {
    "name": "b42 snap hook", "domain": {"size": [40, 10, 12]}, "material": {"name": "PA12", "E": 1700.0, "nu": 0.4},
    "volume_fraction": 0.35,
    "supports": [{"id": "housing", "region": face("x-")}],
    "loads": [{"id": "tip", "region": patch("y-", (38, 6), (4, 12)), "force": [0, 15, 0]}],
    "min_feature": 1.2, "resolution": {"h": 0.25}})

task("b43_press_riser", """
Press riser block

Envelope 300 x 100 x 150 mm (x, y, z), y up, steel. The bottom face (y = 0) is bolted to the press bed over its
full area. A coolant channel along the full length (x), z from 45 to 105 mm and y from 0 to 30 mm, must stay
empty. The tool presses on a 100 x 50 mm pad at the centre of the top face with 20 kN downward and 2 kN along +z.

Volume fraction 0.3. Mesh it with 50 elements across the shortest side.
""", {
    "name": "b43 press riser", "domain": {"size": [300, 100, 150]}, "material": STEEL, "volume_fraction": 0.3,
    "supports": [{"id": "bed", "region": face("y-")}],
    "loads": [{"id": "tool", "region": patch("y+", (150, 75), (100, 50)), "force": [0, -20000, 2000]}],
    "passive": [void("coolant", box((0, 0, 45), (300, 30, 105)))],
    "resolution": {"cells_short_edge": 50}})

# ---- b44-b48: engineering briefs with a deliberate gap (expected: clarify) -----------------------------------

b44 = task("b44_gearbox_bracket", """
Gearbox output bracket

Envelope 160 mm long (x) with an 80 x 80 mm square cross-section (y, z), steel. The x = 0 end is bolted
rigidly to a gearbox. At the far end a 40 N*m torque about the long axis is applied through two 16 x 16 mm
drive lugs on the end face, one at the top edge and one at the bottom edge, each centred in depth.

Volume fraction 0.3, 2 mm elements.
""", {
    "name": "b44 gearbox bracket", "domain": {"size": [160, 80, 80]}, "material": STEEL, "volume_fraction": 0.3,
    "supports": [{"id": "gearbox", "region": face("x-")}],
    "loads": [{"id": "lug_top", "region": patch("x+", (72, 40), (16, 16)), "force": [0, 0, 625]},
              {"id": "lug_bottom", "region": patch("x+", (8, 40), (16, 16)), "force": [0, 0, -625]}],
    "resolution": {"h": 2.0}},
    expected="clarify",
    oracle=[{"about": "which axis is up, and where the lugs sit",
             "answer": "y is up. Both lugs sit inside the end face, flush with its top edge (y = 80) and bottom edge "
                       "(y = 0), centred at z = 40, so their centres are 64 mm apart."},
            {"about": "the sense of the torque", "answer": "Either sense is fine; take +x by the right-hand rule."}],
    notes="The draft-1 wording of b04: no up axis, and 'at the top edge' may mean on or inside the edge. In the "
          "draft X2 run every condition read it differently from the intended spec (z up, or lugs centred on the "
          "edges). Same problem as b04, which states it fully.")
TASKS[-1][2]["admissible"] = [
    {"reading": "torque in the opposite sense (signs swapped)", "why": "the brief gives no sense; the user accepts "
     "either", "counts": True,
     "spec": alt(b44, lambda s: [x.update(force=[0, 0, -x["force"][2]]) for x in s["loads"]])}]

task("b45_pump_bracket", """
Pump bracket

Envelope 160 x 100 x 60 mm (x, y, z), y up, steel. The back face (z = 0) bolts to the wall over its full area.
The pump sits on the top face (y = 100) over a 60 x 40 mm footprint centred at x = 80, z = 35. Design it for the
pump.

Volume fraction 0.25, 2 mm elements.
""", {
    "name": "b45 pump bracket", "domain": {"size": [160, 100, 60]}, "material": STEEL, "volume_fraction": 0.25,
    "supports": [{"id": "wall", "region": face("z-")}],
    "loads": [{"id": "pump", "region": patch("y+", (80, 35), (60, 40)), "force": [0, -250, 0]}],
    "resolution": {"h": 2.0}},
    expected="clarify",
    oracle=[{"about": "the pump's load", "answer": "The pump weighs 25 kg: use 250 N downward (-y) spread over "
             "the footprint. Nothing else acts on it."}],
    notes="Gap: the load. Its size alone does not change a compliance design, but whether it has a sideways "
          "part (vibration, pipe pull) does; a correct agent asks rather than guessing a direction and size.")

task("b46_sensor_housing", """
Sensor housing bracket

Envelope 80 x 50 x 40 mm (x, y, z), y up, aluminium 6061. The bottom face (y = 0) is bolted to the chassis over
its full area. The sensor head pushes 200 N along -x on a 10 x 10 mm pad at the centre of the x = 80 end face.
Leave room for the connector plug at the front.

Volume fraction 0.3, 1 mm elements.
""", {
    "name": "b46 sensor housing", "domain": {"size": [80, 50, 40]}, "material": ALU, "volume_fraction": 0.3,
    "supports": [{"id": "chassis", "region": face("y-")}],
    "loads": [{"id": "head", "region": patch("x+", (25, 20), (10, 10)), "force": [-200, 0, 0]}],
    "passive": [void("plug", box((20, 10, 0), (50, 30, 15)))],
    "resolution": {"h": 1.0}},
    expected="clarify",
    oracle=[{"about": "the connector clearance", "answer": "The front is the z = 0 face. The plug needs an empty "
             "box from x = 20 to 50 mm, y = 10 to 30 mm, and from the front face 15 mm deep (z = 0 to 15)."}],
    notes="Gap: the size and place of a keep-out.")

task("b47_camera_mount", """
Inspection-camera mount

Envelope 120 x 60 x 60 mm (x, y, z), y up, aluminium 6061. The x = 0 face is bolted to the gantry over its full
area. The camera hangs 80 N downward on a 20 x 20 mm pad at the centre of the x = 120 end face. Make it as light
as practical.

1 mm elements.
""", {
    "name": "b47 camera mount", "domain": {"size": [120, 60, 60]}, "material": ALU, "volume_fraction": 0.15,
    "supports": [{"id": "gantry", "region": face("x-")}],
    "loads": [{"id": "camera", "region": patch("x+", (30, 30), (20, 20)), "force": [0, -80, 0]}],
    "resolution": {"h": 1.0}},
    expected="clarify",
    oracle=[{"about": "how much material", "answer": "Use volume fraction 0.15."}],
    notes="Gap: the volume fraction ('as light as practical' has no value).")

task("b48_cable_anchor", """
Guy-cable anchor

Envelope 100 x 60 x 60 mm (x, y, z), y up, steel. The bottom face (y = 0) is bolted down over its full area. A
guy cable pulls 5 kN on a 16 mm diameter anchor boss face at the centre of the top face; the cable rises 3 for
every 4 it runs horizontally.

Volume fraction 0.25, 1 mm elements.
""", {
    "name": "b48 cable anchor", "domain": {"size": [100, 60, 60]}, "material": STEEL, "volume_fraction": 0.25,
    "supports": [{"id": "base", "region": face("y-")}],
    "loads": [{"id": "cable", "region": patch("y+", (50, 30), (16, 16), "ellipse"), "force": [4000, 3000, 0]}],
    "resolution": {"h": 1.0}},
    expected="clarify",
    oracle=[{"about": "which way the cable runs", "answer": "The cable leaves towards +x: 4 kN along +x and 3 kN "
             "upward (+y)."}],
    notes="Gap: the horizontal direction of the pull (+x, -x, +z or -z).")

# ---- b49-b50: infeasible or contradictory briefs (expected: refuse) -------------------------------------------

task("b49_hinged_arm", """
Hinged support arm

Arm envelope 200 x 40 x 20 mm (x, y, z), y up, steel. The arm hangs on a single hinge pin that runs through its
thickness (along z) at x = 10, y = 20 mm, and turns freely on it; nothing else holds it. A 400 N downward load
acts on a 10 mm tall strip of the far end face (x = 200), centred at y = 20, across the full thickness.

Volume fraction 0.3, 1 mm elements.
""", {
    "name": "b49 hinged arm", "domain": {"size": [200, 40, 20]}, "material": STEEL, "volume_fraction": 0.3,
    "supports": [{"id": "hinge", "region": edge("z", (10, 20))}],
    "loads": [{"id": "tip", "region": patch("x+", (20, 10), (10, 20)), "force": [0, -400, 0]}],
    "resolution": {"h": 1.0}},
    expected="refuse",
    oracle=[{"about": "any other support or a stop", "answer": "No. The arm must stay free to turn on the pin."}],
    notes="A mechanism: the arm rotates freely about the pin axis (rule R003, rotation about z). A correct agent "
          "says that nothing reacts the moment and asks for a stop, rather than clamping the pin.")

task("b50_valve_bracket", """
Valve actuator bracket

Envelope 120 x 80 x 40 mm (x, y, z), y up, aluminium 6061. The bottom face (y = 0) is bolted down over its full
area. The actuator's 600 N downward load enters through a 20 x 20 mm pad on the top face centred at x = 90,
z = 20. The region x > 70 mm, y > 50 mm must stay clear for the valve handle, through the full depth.

Volume fraction 0.3, 1 mm elements.
""", {
    "name": "b50 valve bracket", "domain": {"size": [120, 80, 40]}, "material": ALU, "volume_fraction": 0.3,
    "supports": [{"id": "base", "region": face("y-")}],
    "loads": [{"id": "actuator", "region": patch("y+", (90, 20), (20, 20)), "force": [0, -600, 0]}],
    "passive": [void("handle", box((70, 50, 0), (120, 80, 40)))],
    "resolution": {"h": 1.0}},
    expected="refuse",
    oracle=[{"about": "moving the pad or the clearance", "answer": "Neither can move: both are set by the "
             "actuator and the valve."}],
    notes="Contradictory: the load pad lies inside the region that must stay empty (rule R008). A correct agent "
          "reports the conflict instead of moving either.")

if __name__ == "__main__":
    (ROOT / "briefs" / "intended").mkdir(parents=True, exist_ok=True)
    for slug, brief, intended in TASKS:
        (ROOT / "briefs" / f"{slug}.md").write_text(brief, encoding="utf-8")
        (ROOT / "briefs" / "intended" / f"{slug}.json").write_text(json.dumps(intended, indent=1), encoding="utf-8")
    from collections import Counter
    print(len(TASKS), "tasks written;", dict(Counter(t[2]["expected"] for t in TASKS)))
