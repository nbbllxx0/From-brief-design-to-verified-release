import numpy as np
import cupy as cp
from voxfem.core import Grid, Problem, SparseLoad, pad_elements
from voxfem.simp import SIMP, Config
from skimage.measure import marching_cubes
import trimesh
L = [200, 100, 100]
N = [200, 100, 100]
g = Grid(*N)
x, y, z = (np.arange(n + 1) * l / n for n, l in zip(N, L))
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
t = 1e-9 * max(l / n for n, l in zip(N, L))
fixed = np.zeros(X.shape, np.uint8)
fixed[(X <= t)] |= 7
idx, val = [], []
n = np.flatnonzero((X >= 200 - t) & (abs(Y - 50) <= 5 + t) & (abs(Z - 50) <= 5 + t))
idx.append(3 * n + 1); val.append(np.full(n.size, -1000 / n.size))
load = SparseLoad(np.concatenate(idx), np.concatenate(val)).coalesce()
problem = Problem('task', g, fixed.ravel(), load)
cfg = Config(volfrac=0.3, solver="mg64")
s = SIMP(problem, cfg)
rho = s.run()
v, f, _, _ = marching_cubes(rho.reshape(N), 0.5, spacing=[l / n for n, l in zip(N, L)])
trimesh.Trimesh(v, f).export("part.stl")
