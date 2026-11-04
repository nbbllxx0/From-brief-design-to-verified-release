"""Spec -> voxfem Problem, deterministically, at any resolution.

Node coordinates on axis a are i * (L_a / n_a), computed exactly as voxfem's problem builders do, so a spec that
restates a FEAD problem yields bit-identical fixed masks and loads (tests/test_identity.py). Regions select
nodes (supports, loads) or elements by centre (passive regions) with a rounding tolerance of 1e-9 element edges.
A face patch or edge that falls between grid lines snaps to the nearest line and is reported, as in voxfem's
patch rule; nothing is snapped silently.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ._voxfem import Grid, Problem, SparseLoad
from .spec import BoxRegion, CylinderRegion, EdgeRegion, FaceRegion, PatchRegion, Spec

AX = {"x": 0, "y": 1, "z": 2}
BIT = {"x": 1, "y": 2, "z": 4}
TOL = 1e-9          # in element edges


class ResolutionError(ValueError):
    pass


def grid_for(spec: Spec, h: float | None = None, cells_short_edge: int | None = None):
    """Element edge and grid for a spec; overrides take precedence over spec.resolution."""
    L = spec.domain.size
    if h is None and cells_short_edge is None:
        h, cells_short_edge = spec.resolution.h, spec.resolution.cells_short_edge
    if h is None:
        h = min(L) / cells_short_edge
    n = []
    for a, La in zip("xyz", L):
        r = La / h
        if abs(r - round(r)) > 1e-6 * max(1.0, r) or round(r) < 1:
            raise ResolutionError(f"domain {a} size {La} is not a whole number of elements of edge {h:g}")
        n.append(int(round(r)))
    return h, Grid(*n)


def _other(axis: int):
    return [a for a in range(3) if a != axis]


@dataclass
class SpecProblem:
    spec: Spec
    h: float
    grid: Grid
    problem: Problem
    solid: np.ndarray | None          # (n_elem,) bool, passive solid elements
    void: np.ndarray | None           # (n_elem,) bool, passive void elements
    regions: dict = field(default_factory=dict)   # id -> {"n_nodes"|"n_elem", "snapped": [...]}
    notes: list = field(default_factory=list)

    @property
    def compliance_scale(self) -> float:
        """Physical compliance = voxfem compliance * scale (force*length in spec units).

        voxfem's element matrix is for modulus 1 on a cube of edge 2; for edge h and modulus E the matrix is
        E*h/2 times larger, so the compliance f.K^-1 f is 2/(E h) times the voxfem value.
        """
        return 2.0 / (self.spec.material.E * self.h)

    @property
    def n_design(self) -> int:
        m = self.passive_mask
        return self.grid.n_elem - (0 if m is None else int(m.sum()))

    @property
    def passive_mask(self) -> np.ndarray | None:
        if self.solid is None and self.void is None:
            return None
        if self.solid is None or self.void is None:
            return self.solid if self.void is None else self.void
        return self.solid | self.void


class _Builder:
    def __init__(self, spec: Spec, h: float, g: Grid):
        self.spec, self.h, self.g = spec, h, g
        self.n = g.shape
        self.L = spec.domain.size
        self.coord = [np.arange(n + 1) * (L / n) for n, L in zip(self.n, self.L)]
        self.centre = [(np.arange(n) + 0.5) * (L / n) for n, L in zip(self.n, self.L)]
        self.tol = [TOL * L / n for n, L in zip(self.n, self.L)]

    # ---- node selection --------------------------------------------------
    def _near(self, a: int, c: float, half: float, snaps: list, what: str):
        x = self.coord[a]
        j = np.where(np.abs(x - c) <= half + self.tol[a])[0]
        if j.size == 0:
            j = np.array([int(min(max(round(c / (self.L[a] / self.n[a])), 0), self.n[a]))])
            snaps.append(f"{what} on {'xyz'[a]}: nothing within {half:g} of {c:g}; snapped to {x[j[0]]:g}")
        return j

    def _line(self, a: int, c: float, snaps: list, what: str):
        x = self.coord[a]
        j = int(np.argmin(np.abs(x - c)))
        if abs(x[j] - c) > self.tol[a]:
            snaps.append(f"{what} at {'xyz'[a]}={c:g} is between grid lines; snapped to {x[j]:g}")
        return np.array([j])

    def _span(self, a: int, s):
        x = self.coord[a]
        if s is None:
            return np.arange(x.size)
        return np.where((x >= s[0] - self.tol[a]) & (x <= s[1] + self.tol[a]))[0]

    def nodes(self, region, snaps: list) -> np.ndarray:
        g = self.g
        if isinstance(region, (FaceRegion, PatchRegion)):
            a = AX[region.face[0]]
            idx = [None, None, None]
            idx[a] = np.array([0 if region.face[1] == "-" else self.n[a]])
            b, c = _other(a)
            if isinstance(region, FaceRegion):
                idx[b], idx[c] = np.arange(self.n[b] + 1), np.arange(self.n[c] + 1)
            else:
                idx[b] = self._near(b, region.center[0], region.size[0] / 2, snaps, "patch")
                idx[c] = self._near(c, region.center[1], region.size[1] / 2, snaps, "patch")
                if region.shape == "ellipse" and idx[b].size > 1 and idx[c].size > 1:
                    B, C = np.meshgrid(idx[b], idx[c], indexing="ij")
                    rb = (self.coord[b][B] - region.center[0]) / (region.size[0] / 2)
                    rc = (self.coord[c][C] - region.center[1]) / (region.size[1] / 2)
                    keep = rb ** 2 + rc ** 2 <= 1 + 1e-9
                    sel = [None, None, None]
                    sel[a] = np.full(int(keep.sum()), idx[a][0])
                    sel[b], sel[c] = B[keep], C[keep]
                    return np.unique(g.node_id(*sel).astype(np.int64))
        elif isinstance(region, EdgeRegion):
            a = AX[region.axis]
            b, c = _other(a)
            idx = [None, None, None]
            idx[a] = self._span(a, region.span)
            idx[b] = self._line(b, region.at[0], snaps, "edge")
            idx[c] = self._line(c, region.at[1], snaps, "edge")
        elif isinstance(region, BoxRegion):
            idx = [np.where((x >= lo - t) & (x <= hi + t))[0]
                   for x, lo, hi, t in zip(self.coord, region.min, region.max, self.tol)]
        elif isinstance(region, CylinderRegion):
            return self._cyl(region, self.coord, lambda i: g.node_id(*i))
        else:
            raise TypeError(type(region))
        if any(i.size == 0 for i in idx):
            return np.zeros(0, np.int64)
        I, J, K = np.meshgrid(*idx, indexing="ij")
        return np.unique(g.node_id(I.ravel(), J.ravel(), K.ravel()).astype(np.int64))

    def _cyl(self, r: CylinderRegion, pts, ids):
        a = AX[r.axis]
        b, c = _other(a)
        t = max(self.tol)
        ia = np.where((pts[a] >= r.span[0] - t) & (pts[a] <= r.span[1] + t))[0]
        B, C = np.meshgrid(pts[b], pts[c], indexing="ij")
        jb, jc = np.nonzero((B - r.center[0]) ** 2 + (C - r.center[1]) ** 2 <= (r.radius + t) ** 2)
        if ia.size == 0 or jb.size == 0:
            return np.zeros(0, np.int64)
        idx = [None, None, None]
        idx[a] = np.repeat(ia, jb.size)
        idx[b] = np.tile(jb, ia.size)
        idx[c] = np.tile(jc, ia.size)
        return np.unique(np.asarray(ids(idx), dtype=np.int64))

    # ---- element selection (by centre) -----------------------------------
    def elements(self, region) -> np.ndarray:
        g = self.g
        mask = np.zeros(g.shape, dtype=bool)
        if isinstance(region, BoxRegion):
            sl = []
            for x, lo, hi in zip(self.centre, region.min, region.max):
                j = np.where((x >= lo) & (x <= hi))[0]
                if j.size == 0:
                    return mask.ravel()
                sl.append(slice(j[0], j[-1] + 1))
            mask[tuple(sl)] = True
        elif isinstance(region, CylinderRegion):
            e = self._cyl(region, self.centre, lambda i: g.elem_id(*i))
            mask.ravel()[e] = True
        else:
            raise TypeError(type(region))
        return mask.ravel()


def build(spec: Spec, h: float | None = None, cells_short_edge: int | None = None) -> SpecProblem:
    h, g = grid_for(spec, h, cells_short_edge)
    B = _Builder(spec, h, g)
    regions, notes = {}, []
    fixed = np.zeros(g.n_node, dtype=np.uint8)
    for s in spec.supports:
        snaps = []
        nd = B.nodes(s.region, snaps)
        bits = 0
        for a in s.fix:
            bits |= BIT[a]
        fixed[nd] |= np.uint8(bits)
        regions[s.id] = {"n_nodes": int(nd.size), "snapped": snaps}
    idx, val = [], []
    for ld in spec.loads:
        snaps = []
        nd = B.nodes(ld.region, snaps)
        regions[ld.id] = {"n_nodes": int(nd.size), "snapped": snaps}
        if nd.size == 0:
            continue
        for c in range(3):
            if ld.force[c] != 0:
                idx.append(3 * nd + c)
                val.append(np.full(nd.size, ld.force[c] / nd.size))
    lost = 0.0
    if idx:
        load = SparseLoad(np.concatenate(idx), np.concatenate(val)).coalesce()
        # a load component on a fixed DOF goes straight into the support; voxfem would keep it in the right-hand side,
        # where the Krylov solve cannot reduce it (b22 failed on mg64 and mg_ir). Drop it and report the share (R005).
        node, comp = np.divmod(load.idx, 3)
        on_fixed = (fixed[node] & (1 << comp).astype(np.uint8)) > 0
        if on_fixed.any():
            lost = float(np.abs(load.val[on_fixed]).sum() / max(np.abs(load.val).sum(), 1e-300))
            load = SparseLoad(load.idx[~on_fixed], load.val[~on_fixed])
    else:
        load = SparseLoad(np.zeros(0, np.int64), np.zeros(0))
    solid = void = None
    for p in spec.passive:
        m = B.elements(p.region)
        regions[p.id] = {"n_elem": int(m.sum()), "snapped": []}
        if p.kind == "solid":
            solid = m if solid is None else (solid | m)
        else:
            void = m if void is None else (void | m)
    for rid, r in regions.items():
        notes += [f"{rid}: {s}" for s in r["snapped"]]
    meta = {"spec": spec.name, "h": h, "domain": list(spec.domain.size), "regions": regions,
            "load_on_fixed_frac": lost}
    prob = Problem(spec.name, g, fixed, load, meta)
    return SpecProblem(spec, h, g, prob, solid, void, regions, notes)
