import numpy as np
import cupy as cp
from voxfem.core import Grid, Problem, SparseLoad, pad_elements
from voxfem.simp import SIMP, Config
from skimage.measure import marching_cubes
import trimesh
class PassiveSIMP(SIMP):
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
        self.dv *= self.keep
L = [160, 60, 120]
N = [160, 60, 120]
g = Grid(*N)
x, y, z = (np.arange(n + 1) * l / n for n, l in zip(N, L))
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
t = 1e-9 * max(l / n for n, l in zip(N, L))
fixed = np.zeros(X.shape, np.uint8)
fixed[(Y <= t) & (((X - 20) / 12) ** 2 + ((Z - 20) / 12) ** 2 <= 1 + 1e-9)] |= 7
fixed[(Y <= t) & (((X - 20) / 12) ** 2 + ((Z - 100) / 12) ** 2 <= 1 + 1e-9)] |= 7
fixed[(Y <= t) & (((X - 140) / 12) ** 2 + ((Z - 20) / 12) ** 2 <= 1 + 1e-9)] |= 7
fixed[(Y <= t) & (((X - 140) / 12) ** 2 + ((Z - 100) / 12) ** 2 <= 1 + 1e-9)] |= 7
idx, val = [], []
n = np.flatnonzero((Y >= 60 - t) & (abs(X - 80) <= 20 + t) & (abs(Z - 60) <= 20 + t))
idx.append(3 * n + 0); val.append(np.full(n.size, 800 / n.size))
idx.append(3 * n + 1); val.append(np.full(n.size, -2000 / n.size))
load = SparseLoad(np.concatenate(idx), np.concatenate(val)).coalesce()
problem = Problem('task', g, fixed.ravel(), load)
xc, yc, zc = ((np.arange(n) + 0.5) * l / n for n, l in zip(N, L))
XC, YC, ZC = np.meshgrid(xc, yc, zc, indexing='ij')
solid = np.zeros(XC.shape, bool)
void = np.zeros(XC.shape, bool)
solid |= ((XC - 20) ** 2 + (ZC - 20) ** 2 <= (12 + t) ** 2) & (YC >= 0 - t) & (YC <= 10 + t)
solid |= ((XC - 20) ** 2 + (ZC - 100) ** 2 <= (12 + t) ** 2) & (YC >= 0 - t) & (YC <= 10 + t)
solid |= ((XC - 140) ** 2 + (ZC - 20) ** 2 <= (12 + t) ** 2) & (YC >= 0 - t) & (YC <= 10 + t)
solid |= ((XC - 140) ** 2 + (ZC - 100) ** 2 <= (12 + t) ** 2) & (YC >= 0 - t) & (YC <= 10 + t)
solid |= (XC >= 60) & (XC <= 100) & (YC >= 55) & (YC <= 60) & (ZC >= 40) & (ZC <= 80)
solid, void = solid.ravel(), void.ravel()
cfg = Config(volfrac=0.2, solver="mg64")
s = PassiveSIMP(problem, cfg, solid, void)
rho = s.run()
v, f, _, _ = marching_cubes(rho.reshape(N), 0.5, spacing=[l / n for n, l in zip(N, L)])
trimesh.Trimesh(v, f).export("part.stl")
