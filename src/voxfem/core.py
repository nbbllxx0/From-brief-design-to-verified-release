"""Grid indexing, element stiffness and boundary-value problems.

A problem is a structured hexahedral grid plus
  * a per-node fixed mask (uint8, bit c set when DOF 3n+c is fixed), and
  * a sparse load (DOF indices and values).
The load is kept sparse on purpose: at 10^8 elements a dense FP64 load vector
costs 24 B/node, and every load used here is a small surface patch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

OFFSETS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.int32)


def ke_unit(nu: float = 0.3, E: float = 1.0) -> np.ndarray:
    """24x24 trilinear hexahedron stiffness, 2x2x2 Gauss, unit modulus.

    Identical to gpu_fem.pub_simp_solver._build_ke_unit_3d (verified in
    tests): reference cube [-1,1]^3, so the element edge is 2 length units.
    Under uniform scaling K_e is proportional to the edge length, which only
    rescales compliance; every comparison in this project uses one element size
    per problem family.
    """
    gp = 1.0 / np.sqrt(3.0)
    signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    nc = signs.copy()
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
    D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = lam
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    KE = np.zeros((24, 24))
    for r, s, t in signs * gp:
        dN = 0.125 * np.array([
            [-(1 - s) * (1 - t), -(1 - r) * (1 - t), -(1 - r) * (1 - s)],
            [(1 - s) * (1 - t), -(1 + r) * (1 - t), -(1 + r) * (1 - s)],
            [(1 + s) * (1 - t), (1 + r) * (1 - t), -(1 + r) * (1 + s)],
            [-(1 + s) * (1 - t), (1 - r) * (1 - t), -(1 - r) * (1 + s)],
            [-(1 - s) * (1 + t), -(1 - r) * (1 + t), (1 - r) * (1 - s)],
            [(1 - s) * (1 + t), -(1 + r) * (1 + t), (1 + r) * (1 - s)],
            [(1 + s) * (1 + t), (1 + r) * (1 + t), (1 + r) * (1 + s)],
            [-(1 + s) * (1 + t), (1 - r) * (1 + t), (1 - r) * (1 + s)],
        ])
        J = dN.T @ nc
        dNdx = dN @ np.linalg.inv(J).T
        dx, dy, dz = dNdx[:, 0], dNdx[:, 1], dNdx[:, 2]
        B = np.zeros((6, 24))
        idx = np.arange(8)
        B[0, 3 * idx] = dx
        B[1, 3 * idx + 1] = dy
        B[2, 3 * idx + 2] = dz
        B[3, 3 * idx] = dy
        B[3, 3 * idx + 1] = dx
        B[4, 3 * idx + 1] = dz
        B[4, 3 * idx + 2] = dy
        B[5, 3 * idx] = dz
        B[5, 3 * idx + 2] = dx
        KE += B.T @ D @ B * abs(np.linalg.det(J))
    return KE


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    nz: int

    @property
    def shape(self):
        return (self.nx, self.ny, self.nz)

    @property
    def n_elem(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def node_shape(self):
        return (self.nx + 1, self.ny + 1, self.nz + 1)

    @property
    def n_node(self) -> int:
        return (self.nx + 1) * (self.ny + 1) * (self.nz + 1)

    @property
    def ndof(self) -> int:
        return 3 * self.n_node

    def node_id(self, ix, iy, iz):
        return (np.asarray(ix) * (self.ny + 1) + np.asarray(iy)) * (self.nz + 1) + np.asarray(iz)

    def elem_id(self, ex, ey, ez):
        return (np.asarray(ex) * self.ny + np.asarray(ey)) * self.nz + np.asarray(ez)

    def edof(self) -> np.ndarray:
        """(n_elem, 24) int64 DOF table -- reference/test use only."""
        ex, ey, ez = np.meshgrid(np.arange(self.nx), np.arange(self.ny),
                                 np.arange(self.nz), indexing="ij")
        ex, ey, ez = ex.ravel(), ey.ravel(), ez.ravel()
        nodes = np.stack([self.node_id(ex + o[0], ey + o[1], ez + o[2]) for o in OFFSETS], axis=1)
        return (3 * nodes[:, :, None] + np.arange(3)[None, None, :]).reshape(-1, 24)

    def coarsenable(self, levels: int) -> bool:
        f = 2 ** (levels - 1)
        return self.nx % f == 0 and self.ny % f == 0 and self.nz % f == 0

    def padded(self, levels: int) -> "Grid":
        f = 2 ** (levels - 1)
        up = lambda n: int(math.ceil(n / f) * f)
        return Grid(up(self.nx), up(self.ny), up(self.nz))


@dataclass
class SparseLoad:
    idx: np.ndarray          # int64 DOF indices
    val: np.ndarray          # float64 values

    def dense(self, ndof):
        b = np.zeros(ndof)
        np.add.at(b, self.idx, self.val)
        return b

    @property
    def norm(self) -> float:
        # indices are unique by construction (coalesce())
        return float(np.sqrt(np.sum(self.val ** 2)))

    def coalesce(self):
        u, inv = np.unique(self.idx, return_inverse=True)
        v = np.zeros(u.size)
        np.add.at(v, inv, self.val)
        keep = v != 0
        return SparseLoad(u[keep], v[keep])


@dataclass
class Problem:
    name: str
    grid: Grid
    fixed: np.ndarray        # (n_node,) uint8 bitmask
    load: SparseLoad
    meta: dict = field(default_factory=dict)

    @property
    def n_fixed_dofs(self):
        f = self.fixed
        return int(((f & 1) > 0).sum() + ((f & 2) > 0).sum() + ((f & 4) > 0).sum())

    def fixed_dof_mask(self) -> np.ndarray:
        f = self.fixed
        return np.stack([(f & 1) > 0, (f & 2) > 0, (f & 4) > 0], axis=1).ravel()

    def padded(self, levels: int) -> "Problem":
        """Embed in a grid divisible by 2^(levels-1).

        Padding elements get modulus zero (callers must pad E with zeros) and
        every node outside the original node box is fixed, so the operator on
        the original DOFs is unchanged exactly.
        """
        g = self.grid
        gp = g.padded(levels)
        if gp == g:
            return self
        fixed = np.full(gp.node_shape, 7, dtype=np.uint8)
        fixed[:g.nx + 1, :g.ny + 1, :g.nz + 1] = self.fixed.reshape(g.node_shape)
        n_old = np.arange(g.n_node)
        ix, rem = np.divmod(n_old, (g.ny + 1) * (g.nz + 1))
        iy, iz = np.divmod(rem, g.nz + 1)
        n_new = gp.node_id(ix, iy, iz)
        node_old, comp = np.divmod(self.load.idx, 3)
        idx = 3 * n_new[node_old] + comp
        meta = dict(self.meta)
        meta["padded_from"] = list(g.shape)
        return Problem(self.name, gp, fixed.ravel(), SparseLoad(idx, self.load.val.copy()), meta)


def pad_elements(E, g: Grid, gp: Grid, xp=np):
    if gp == g:
        return E
    out = xp.zeros(gp.shape, dtype=E.dtype)
    out[:g.nx, :g.ny, :g.nz] = E.reshape(g.shape)
    return out.ravel()


def unpad_elements(E, g: Grid, gp: Grid):
    if gp == g:
        return E
    return E.reshape(gp.shape)[:g.nx, :g.ny, :g.nz].ravel()


# ---------------------------------------------------------------------------
# Problem families used for complete optimizations.  Every support lies on a
# grid face or a grid line through even node indices, so it is representable
# on every multigrid level.  Loads are spread over patches of fixed physical
# size, so total force and traction are mesh independent (a point load is
# singular in 3D elasticity).
# ---------------------------------------------------------------------------

def _patch_nodes(g: Grid, ix: int, y_c: float, z_c: float, half: float, Ly, Lz):
    hy, hz = Ly / g.ny, Lz / g.nz
    ys = np.arange(g.ny + 1) * hy
    zs = np.arange(g.nz + 1) * hz
    jy = np.where(np.abs(ys - y_c) <= half + 1e-12)[0]
    jz = np.where(np.abs(zs - z_c) <= half + 1e-12)[0]
    if jy.size == 0:
        jy = np.array([int(round(y_c / hy))])
    if jz.size == 0:
        jz = np.array([int(round(z_c / hz))])
    gy, gz = np.meshgrid(jy, jz, indexing="ij")
    return g.node_id(np.full(gy.size, ix), gy.ravel(), gz.ravel())


def cantilever(k: int, patch=0.1, total=-1.0) -> Problem:
    """Grid 2k x k x k on [0,2]x[0,1]x[0,1]; x=0 clamped; y-load on a
    patch x patch square centred on the x=2 face."""
    g = Grid(2 * k, k, k)
    fixed = np.zeros(g.node_shape, dtype=np.uint8)
    fixed[0, :, :] = 7
    nodes = _patch_nodes(g, g.nx, 0.5, 0.5, patch / 2, 1.0, 1.0)
    load = SparseLoad(3 * nodes + 1, np.full(nodes.size, total / nodes.size)).coalesce()
    return Problem("cantilever", g, fixed.ravel(), load,
                   {"domain": [2.0, 1.0, 1.0], "patch": patch, "total_load": total,
                    "loaded_nodes": int(nodes.size), "supports": "x=0 face clamped"})


def bridge(k: int, patch=0.1, total=-1.0) -> Problem:
    """Grid 2k x k x k on [0,2]x[0,1]x[0,1].  Supports: the two bottom edges
    y=0 at x=0 (pinned: all DOFs) and x=2 (roller: y and z fixed, x free),
    each a full line along z.  Load: -y on a patch-wide strip at the top
    centre (x=1, y=1), spanning the full depth in z."""
    g = Grid(2 * k, k, k)
    fixed = np.zeros(g.node_shape, dtype=np.uint8)
    fixed[0, 0, :] = 7
    fixed[g.nx, 0, :] = 2 | 4
    hx = 2.0 / g.nx
    xs = np.arange(g.nx + 1) * hx
    jx = np.where(np.abs(xs - 1.0) <= patch / 2 + 1e-12)[0]
    gx, gz = np.meshgrid(jx, np.arange(g.nz + 1), indexing="ij")
    nodes = g.node_id(gx.ravel(), np.full(gx.size, g.ny), gz.ravel())
    load = SparseLoad(3 * nodes + 1, np.full(nodes.size, total / nodes.size)).coalesce()
    return Problem("bridge", g, fixed.ravel(), load,
                   {"domain": [2.0, 1.0, 1.0], "patch": patch, "total_load": total,
                    "loaded_nodes": int(nodes.size),
                    "supports": "bottom edge x=0 pinned, bottom edge x=2 roller (x free)"})


def torsion(k: int, patch=0.15, total=1.0) -> Problem:
    """Grid 2k x k x k on [0,2]x[0,1]x[0,1]; x=0 clamped; a torque about the x
    axis on the x=2 face from two fixed-area patches at y=1 and y=0 carrying
    opposite (y,z) forces, as in the R2 torsion definition."""
    g = Grid(2 * k, k, k)
    fixed = np.zeros(g.node_shape, dtype=np.uint8)
    fixed[0, :, :] = 7
    idx, val = [], []
    n_patch = 0
    for y_t, sy, sz in ((1.0, -0.5, 0.5), (0.0, 0.5, -0.5)):
        nodes = _patch_nodes(g, g.nx, y_t, 0.5, patch / 2, 1.0, 1.0)
        n_patch += nodes.size
        idx += [3 * nodes + 1, 3 * nodes + 2]
        val += [np.full(nodes.size, total * sy / nodes.size),
                np.full(nodes.size, total * sz / nodes.size)]
    load = SparseLoad(np.concatenate(idx), np.concatenate(val)).coalesce()
    return Problem("torsion", g, fixed.ravel(), load,
                   {"domain": [2.0, 1.0, 1.0], "patch": patch, "total_load": total,
                    "loaded_nodes": int(n_patch), "supports": "x=0 face clamped"})


PROBLEMS = {"cantilever": cantilever, "bridge": bridge, "torsion": torsion}
