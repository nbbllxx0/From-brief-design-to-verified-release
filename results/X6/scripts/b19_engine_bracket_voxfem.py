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
L = [7, 2.5, 4]
N = [140, 50, 80]
g = Grid(*N)
x, y, z = (np.arange(n + 1) * l / n for n, l in zip(N, L))
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
t = 1e-9 * max(l / n for n, l in zip(N, L))
fixed = np.zeros(X.shape, np.uint8)
fixed[(Y <= t) & (((X - 0.5) / 0.25) ** 2 + ((Z - 0.5) / 0.25) ** 2 <= 1 + 1e-9)] |= 7
fixed[(Y <= t) & (((X - 0.5) / 0.25) ** 2 + ((Z - 3.5) / 0.25) ** 2 <= 1 + 1e-9)] |= 7
fixed[(Y <= t) & (((X - 2.5) / 0.25) ** 2 + ((Z - 0.5) / 0.25) ** 2 <= 1 + 1e-9)] |= 7
fixed[(Y <= t) & (((X - 2.5) / 0.25) ** 2 + ((Z - 3.5) / 0.25) ** 2 <= 1 + 1e-9)] |= 7
idx, val = [], []
n = np.flatnonzero(((X - 6) ** 2 + (Y - 1.75) ** 2 <= (0.5 + t) ** 2) & (Z >= 1.25 - t) & (Z <= 2.75 + t))
idx.append(3 * n + 1); val.append(np.full(n.size, 8000 / n.size))
load = SparseLoad(np.concatenate(idx), np.concatenate(val)).coalesce()
problem = Problem('task', g, fixed.ravel(), load)
xc, yc, zc = ((np.arange(n) + 0.5) * l / n for n, l in zip(N, L))
XC, YC, ZC = np.meshgrid(xc, yc, zc, indexing='ij')
solid = np.zeros(XC.shape, bool)
void = np.zeros(XC.shape, bool)
solid |= ((XC - 6) ** 2 + (YC - 1.75) ** 2 <= (0.5 + t) ** 2) & (ZC >= 1.25 - t) & (ZC <= 2.75 + t)
solid, void = solid.ravel(), void.ravel()
cfg = Config(volfrac=0.2, solver="mg64")
s = PassiveSIMP(problem, cfg, solid, void)
rho = s.run()
v, f, _, _ = marching_cubes(rho.reshape(N), 0.5, spacing=[l / n for n, l in zip(N, L)])
trimesh.Trimesh(v, f).export("part.stl")
