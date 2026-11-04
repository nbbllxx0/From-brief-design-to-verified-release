"""Deterministic validation rule base, v0.

Each rule inspects a spec (and the problem it builds at the checked resolution) and returns findings with a
stable code, a severity and, where one exists, a concrete fix the GUI can offer as one click and the agent can
propose as an action card. A spec with any error is not solvable; the release gate also requires no errors.

Codes are stable: experiments and logs refer to them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ._voxfem import Grid
from .problem import AX, ResolutionError, SpecProblem, build, grid_for
from .spec import Spec

ERROR, WARNING, INFO = "error", "warning", "info"
MODE_NAMES = ["translation x", "translation y", "translation z", "rotation about x", "rotation about y",
              "rotation about z"]
BYTES_PER_ELEM = {"mg64": 24e9 / 65.5e6, "mg_ir": 146.0}     # measured in the FEAD paper / today's pilots


@dataclass
class Finding:
    code: str
    severity: str
    message: str
    ids: list = field(default_factory=list)
    fix: dict | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class Report:
    findings: list
    resolution: dict

    @property
    def ok(self) -> bool:
        return not any(f.severity == ERROR for f in self.findings)

    def codes(self, severity=None):
        return [f.code for f in self.findings if severity is None or f.severity == severity]

    def to_dict(self):
        return {"ok": self.ok, "resolution": self.resolution, "findings": [f.to_dict() for f in self.findings]}


# ---- helpers ---------------------------------------------------------------

def node_xyz(g: Grid, L, nodes):
    ix, rem = np.divmod(nodes, (g.ny + 1) * (g.nz + 1))
    iy, iz = np.divmod(rem, g.nz + 1)
    return np.stack([ix * (L[0] / g.nx), iy * (L[1] / g.ny), iz * (L[2] / g.nz)], axis=1)


def free_rigid_modes(xyz: np.ndarray, bits: np.ndarray, scale: float) -> list[int]:
    """Rigid-body modes not removed by the fixed DOFs (indices into MODE_NAMES)."""
    if xyz.shape[0] == 0:
        return list(range(6))
    X = (xyz - xyz.mean(axis=0)) / scale
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    one, zero = np.ones_like(x), np.zeros_like(x)
    # modes[j][c] = component c of mode j at each node
    modes = [(one, zero, zero), (zero, one, zero), (zero, zero, one),
             (zero, -z, y), (z, zero, -x), (-y, x, zero)]
    G = np.zeros((6, 6))
    for c in range(3):
        m = (bits & (1 << c)) > 0
        if not m.any():
            continue
        R = np.stack([modes[j][c][m] for j in range(6)], axis=1)
        G += R.T @ R
    w, V = np.linalg.eigh(G)
    free = []
    for i in np.where(w <= 1e-10 * max(w.max(), 1e-300))[0]:
        free.append(int(np.argmax(np.abs(V[:, i]))))
    return sorted(set(free))


def solver_levels(g: Grid, max_coarse_dofs=2500):
    """Grid and level count exactly as voxfem.SIMP chooses them for the multigrid solvers."""
    from voxfem.mg import choose_levels
    gp = g.padded(4) if not g.coarsenable(4) else g
    L = choose_levels(gp, max_coarse_dofs)
    gs = g if g.coarsenable(L) else g.padded(L)
    return gs, choose_levels(gs, max_coarse_dofs)


def _coarsest_dofs(gs: Grid, L: int) -> int:
    s = 2 ** (L - 1)
    return 3 * (gs.nx // s + 1) * (gs.ny // s + 1) * (gs.nz // s + 1)


# Per-iteration cost model on the 4090 (sessions, stencil operator): fine-grid work per element, and the dense
# coarsest factorisation + inverse (fp64) per DOF^3. Fitted to b06 refine (9.2M, 1,485 coarsest DOFs: 0.76 s/it)
# and b01 refine in X3 (2.0M, 15,288 coarsest DOFs: 4.7 s/it).
COST_PER_ELEM = 8e-8
COST_PER_COARSE_DOF3 = 1.3e-12


def iter_cost(gs: Grid, L: int) -> float:
    return COST_PER_ELEM * gs.n_elem + COST_PER_COARSE_DOF3 * _coarsest_dofs(gs, L) ** 3


def session_levels(g: Grid, max_coarse_dofs=2500, max_pad_frac=0.6):
    """(padded grid, levels, overridden) for a session. voxfem pads to a multiple of 8 and stops coarsening at the
    first odd dimension, which can leave a dense coarsest level of ~10^4 DOFs whose factorisation and inverse
    dominate every iteration (b06 at 1.2M: 1.3 s/iteration; b01 at 2M, 15,288 DOFs: 4.7 s/iteration). When the
    default coarsest level exceeds max_coarse_dofs, pad to a multiple of 2^(L-1) for the smallest L that reaches
    it, or else for the L with the lowest predicted iteration cost (`iter_cost`), padding at most max_pad_frac
    more elements. Padding elements have
    zero modulus and fixed nodes (core.Problem.padded): the operator on the original DOFs is unchanged, so only
    cost changes."""
    gs, L = solver_levels(g, max_coarse_dofs)
    base = _coarsest_dofs(gs, L)
    if base <= max_coarse_dofs:
        return gs, L, False
    cands = [(gs, L, iter_cost(gs, L), False)]
    for L2 in range(L + 1, 11):
        gp = g.padded(L2)
        if gp.n_elem > (1 + max_pad_frac) * g.n_elem:
            break
        if min(gp.shape) // 2 ** (L2 - 1) < 1:
            continue
        cands.append((gp, L2, iter_cost(gp, L2), True))
    # Among the candidates within 10% of the cheapest: one whose coarsest level meets the target if any, then the
    # fewest levels (deeper hierarchies on tiny grids buy little). b05 refine, 200x200x80: default coarsest 22,308
    # DOFs, which voxfem refuses; 208x208x80: 3,528. b01, 200x100x100: 15,288 DOFs; 208x112x112 (+30%
    # elements): 2,688, ~20x cheaper per iteration.
    c_min = min(c[2] for c in cands)
    near = [c for c in cands if c[2] <= 1.1 * c_min]
    gp, L2, _, over = min(near, key=lambda c: (_coarsest_dofs(c[0], c[1]) > max_coarse_dofs, c[1]))
    if over:
        return gp, L2, True
    return gs, L, False


def adjacent_live(g: Grid, nodes: np.ndarray, void: np.ndarray) -> np.ndarray:
    """Number of non-void elements around each node."""
    ix, rem = np.divmod(nodes, (g.ny + 1) * (g.nz + 1))
    iy, iz = np.divmod(rem, g.nz + 1)
    V = void.reshape(g.shape)
    live = np.zeros(nodes.size, np.int32)
    for dx in (-1, 0):
        for dy in (-1, 0):
            for dz in (-1, 0):
                ex, ey, ez = ix + dx, iy + dy, iz + dz
                ok = (ex >= 0) & (ex < g.nx) & (ey >= 0) & (ey < g.ny) & (ez >= 0) & (ez < g.nz)
                v = np.ones(nodes.size, bool)
                v[ok] = V[ex[ok], ey[ok], ez[ok]]
                live += (ok & ~v).astype(np.int32)
    return live


# ---- the rules -------------------------------------------------------------

def check(spec: Spec, h: float | None = None, cells_short_edge: int | None = None,
          solver: str = "mg64", vram_bytes: float = 24e9) -> Report:
    F: list[Finding] = []
    L = spec.domain.size
    try:
        h, g = grid_for(spec, h, cells_short_edge)
    except ResolutionError as ex:
        k = max(2, round(min(L) / (spec.resolution.h or min(L) / (spec.resolution.cells_short_edge or 2))))
        F.append(Finding("R001", ERROR, str(ex), fix={"set": {"resolution": {"cells_short_edge": k}},
                                                        "why": "an edge that divides every side"}))
        return Report(F, {})
    sp: SpecProblem = build(spec, h)
    res = {"h": h, "grid": list(g.shape), "n_elem": g.n_elem, "n_design": sp.n_design}
    fixed = sp.problem.fixed

    # R002 / R004: regions that select nothing
    for s in spec.supports:
        if sp.regions[s.id]["n_nodes"] == 0:
            F.append(Finding("R002", ERROR, f"support '{s.id}' selects no nodes", [s.id]))
    for ld in spec.loads:
        if sp.regions[ld.id]["n_nodes"] == 0:
            F.append(Finding("R004", ERROR, f"load '{ld.id}' selects no nodes", [ld.id]))
    if all(np.allclose(ld.force, 0) for ld in spec.loads):
        F.append(Finding("R004", ERROR, "every load has zero force", [ld.id for ld in spec.loads]))

    # R010: snapped regions are reported, never silent
    for n in sp.notes:
        F.append(Finding("R010", WARNING, n, [n.split(":")[0]]))

    # R003: rigid-body modes
    fn = np.nonzero(fixed)[0]
    free = free_rigid_modes(node_xyz(g, L, fn), fixed[fn], max(L))
    if free:
        F.append(Finding("R003", ERROR, "supports leave rigid-body motion free: "
                         + ", ".join(MODE_NAMES[i] for i in free),
                         [s.id for s in spec.supports],
                         fix={"suggest": "add a support that restrains " + ", ".join(MODE_NAMES[i] for i in free)}))

    # R005: load on fixed DOFs goes straight into the supports (build() drops that share and records it)
    frac = sp.problem.meta.get("load_on_fixed_frac", 0.0)
    if frac > 0:
        F.append(Finding("R005", ERROR if frac > 0.5 else WARNING,
                         f"{100 * frac:.1f}% of the applied load acts on fixed DOFs and is lost",
                         [x.id for x in spec.loads]))

    # R007 / R008: passive regions
    if sp.solid is not None and sp.void is not None:
        both = int((sp.solid & sp.void).sum())
        if both:
            F.append(Finding("R007", ERROR, f"{both} elements are both keep-in and keep-out",
                             [p.id for p in spec.passive]))
    if sp.void is not None:
        for item, kind in [(s, "support") for s in spec.supports] + [(x, "load") for x in spec.loads]:
            nd = build_nodes(sp, item)
            if nd.size and (adjacent_live(g, nd, sp.void) == 0).all():
                F.append(Finding("R008", ERROR, f"{kind} '{item.id}' lies entirely inside a keep-out region",
                                 [item.id]))
    for p in spec.passive:
        if sp.regions[p.id]["n_elem"] == 0:
            F.append(Finding("R011", WARNING, f"passive region '{p.id}' contains no element centre at h={h:g}",
                             [p.id], fix={"suggest": "refine the mesh or enlarge the region"}))
    if sp.n_design == 0:
        F.append(Finding("R006", ERROR, "no design elements remain after passive regions"))
    else:
        solid = 0 if sp.solid is None else int(sp.solid.sum())
        total = (solid + spec.volume_fraction * sp.n_design) / g.n_elem
        res["material_fraction_total"] = total

    # R009: resolvability of the smallest feature the user needs
    rmin = 1.2                                  # final-phase filter radius, element units (R2 schedule)
    if spec.min_feature is not None:
        cells = spec.min_feature / h
        if cells < 2 * rmin:
            need = spec.min_feature / (2 * rmin)
            F.append(Finding("R009", WARNING,
                             f"min feature {spec.min_feature:g} is {cells:.1f} elements; members below about "
                             f"{2 * rmin:.1f} elements are filtered out", fix={"set": {"resolution": {"h": need}},
                                                                              "why": "resolve the min feature"}))

    # R012: multigrid coarse levels keep enough supports (voxfem injects the fixed mask at even nodes)
    cc = coarse_check(sp)
    if cc["levels"] >= 2 and not free:          # with R003 open the fine problem itself is singular
        s, lost, free_c = cc["stride"], cc["lost"], cc["free_modes"]
        if free_c:
            F.append(Finding("R012", WARNING,
                             f"on the coarsest multigrid level (1/{s} resolution) the supports leave rigid modes free ("
                             + ", ".join(MODE_NAMES[i] for i in free_c)
                             + "); the solver switches to the dilated coarse support rule (RobustGMG)", lost))
        elif lost:
            F.append(Finding("R012", INFO, f"supports {lost} are not represented on the coarsest multigrid level "
                             f"(1/{s}); the solve stays correct but may need more iterations", lost))
        res["mg_levels"] = cc["levels"]

    # R013: memory
    need = BYTES_PER_ELEM.get(solver, 146.0) * g.n_elem
    res["est_bytes"] = need
    if need > vram_bytes:
        F.append(Finding("R013", ERROR, f"{g.n_elem / 1e6:.1f}M elements need about {need / 1e9:.1f} GB with "
                         f"{solver}; the GPU has {vram_bytes / 1e9:.0f} GB",
                         fix={"suggest": "use mg_ir or a coarser rung"}))
    return Report(F, res)


def coarse_check(sp: SpecProblem) -> dict:
    """What voxfem's injection rule leaves of the supports on the coarsest multigrid level.

    Returns levels, stride, the supports that vanish ("lost") and the rigid modes left free ("free_modes");
    `injection_ok` is True when neither happens, i.e. the plain voxfem GMG is safe."""
    g, h, L = sp.grid, sp.h, sp.spec.domain.size
    gs, lv, _ = session_levels(g)
    out = {"levels": lv, "stride": 2 ** (lv - 1), "lost": [], "free_modes": [], "injection_ok": True}
    if lv < 2:
        return out
    fz = np.full(gs.node_shape, 7 if gs != g else 0, np.uint8)     # padding nodes are fixed (core.py)
    fz[:g.nx + 1, :g.ny + 1, :g.nz + 1] = sp.problem.fixed.reshape(g.node_shape)
    s = out["stride"]
    fc = fz[::s, ::s, ::s]
    gc = Grid(gs.nx // s, gs.ny // s, gs.nz // s)
    cn = np.nonzero(fc.ravel())[0]
    out["free_modes"] = free_rigid_modes(node_xyz(gc, [gc.nx * s * h, gc.ny * s * h, gc.nz * s * h], cn),
                                         fc.ravel()[cn], max(L))
    out["lost"] = [x.id for x in sp.spec.supports
                   if sp.regions[x.id]["n_nodes"] and not _survives(sp, x.id, gs, s)]
    # X0 coarse_bc study (2026-10-29): when injection only loses some supports, the coarsest level stays
    # nonsingular and injection converges in fewer PCG iterations than dilation; dilation is needed only when
    # injection leaves rigid-body modes free.
    out["injection_ok"] = not out["free_modes"]
    return out


def build_nodes(sp: SpecProblem, item) -> np.ndarray:
    from .problem import _Builder
    return _Builder(sp.spec, sp.h, sp.grid).nodes(item.region, [])


def _survives(sp: SpecProblem, sid: str, gs: Grid, s: int) -> bool:
    item = next(x for x in sp.spec.supports if x.id == sid)
    nd = build_nodes(sp, item)
    g = sp.grid
    ix, rem = np.divmod(nd, (g.ny + 1) * (g.nz + 1))
    iy, iz = np.divmod(rem, g.nz + 1)
    return bool(((ix % s == 0) & (iy % s == 0) & (iz % s == 0)).any())
