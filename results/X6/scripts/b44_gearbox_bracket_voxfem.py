import numpy as np
import cupy as cp
from voxfem.core import Grid, Problem, SparseLoad, pad_elements
from voxfem.simp import SIMP, Config
from skimage.measure import marching_cubes
import trimesh
L = [160, 80, 80]
N = [80, 40, 40]
g = Grid(*N)
x, y, z = (np.arange(n + 1) * l / n for n, l in zip(N, L))
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
t = 1e-9 * max(l / n for n, l in zip(N, L))
fixed = np.zeros(X.shape, np.uint8)
fixed[(X <= t)] |= 7
idx, val = [], []
n = np.flatnonzero((X >= 160 - t) & (abs(Y - 72) <= 8 + t) & (abs(Z - 40) <= 8 + t))
idx.append(3 * n + 2); val.append(np.full(n.size, 625 / n.size))
n = np.flatnonzero((X >= 160 - t) & (abs(Y - 8) <= 8 + t) & (abs(Z - 40) <= 8 + t))
idx.append(3 * n + 2); val.append(np.full(n.size, -625 / n.size))
load = SparseLoad(np.concatenate(idx), np.concatenate(val)).coalesce()
problem = Problem('task', g, fixed.ravel(), load)
cfg = Config(volfrac=0.3, solver="mg64")
s = SIMP(problem, cfg)
rho = s.run()
v, f, _, _ = marching_cubes(rho.reshape(N), 0.5, spacing=[l / n for n, l in zip(N, L)])
trimesh.Trimesh(v, f).export("part.stl")
