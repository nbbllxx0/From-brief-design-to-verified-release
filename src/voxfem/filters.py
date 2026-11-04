"""Matrix-free cone density filter on a structured grid (nothing stored).

    forward     rho_bar_i = (sum_j w_ij rho_j) / s_i,     s_i = sum_j w_ij
    transpose   g_rho_j   = sum_i w_ij g_i / s_i
with w_ij = max(0, rmin - |x_i - x_j|) (element units).  The row sums s_i
depend only on how much of the cone support falls inside the box, so they are
recomputed on the fly (in the transpose, once per neighbour) instead of being
stored: no per-element array is kept.  Ported from the R2 filter
(gpu_fem/filter_r2.py), which stored 1/s.
"""
from __future__ import annotations

import math

import numpy as np
import cupy as cp

from .operators import _get_kernel, _ctype

_SRC = r"""
__device__ __forceinline__ __T__ rowsum(const int ex, const int ey, const int ez,
    const int nx, const int ny, const int nz, const int R, const __T__ rmin)
{
    __T__ s = 0;
    for (int dx = -R; dx <= R; ++dx) { const int jx = ex + dx; if (jx < 0 || jx >= nx) continue;
    for (int dy = -R; dy <= R; ++dy) { const int jy = ey + dy; if (jy < 0 || jy >= ny) continue;
    for (int dz = -R; dz <= R; ++dz) { const int jz = ez + dz; if (jz < 0 || jz >= nz) continue;
        const __T__ w = rmin - sqrt((__T__)(dx * dx + dy * dy + dz * dz));
        if (w > (__T__)0) s += w;
    }}}
    return s;
}

extern "C" __global__ void __N___fwd(const __T__* __restrict__ x, __T__* __restrict__ y,
    const int nx, const int ny, const int nz, const int R, const __T__ rmin)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nx * ny * nz) return;
    const int ez = e % nz, t = e / nz, ey = t % ny, ex = t / ny;
    __T__ acc = 0, s = 0;
    for (int dx = -R; dx <= R; ++dx) { const int jx = ex + dx; if (jx < 0 || jx >= nx) continue;
    for (int dy = -R; dy <= R; ++dy) { const int jy = ey + dy; if (jy < 0 || jy >= ny) continue;
    for (int dz = -R; dz <= R; ++dz) { const int jz = ez + dz; if (jz < 0 || jz >= nz) continue;
        const __T__ w = rmin - sqrt((__T__)(dx * dx + dy * dy + dz * dz));
        if (w > (__T__)0) { acc += w * x[(jx * ny + jy) * nz + jz]; s += w; }
    }}}
    y[e] = acc / s;
}

extern "C" __global__ void __N___tr(const __T__* __restrict__ g, __T__* __restrict__ y,
    const int nx, const int ny, const int nz, const int R, const __T__ rmin)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nx * ny * nz) return;
    const int ez = e % nz, t = e / nz, ey = t % ny, ex = t / ny;
    __T__ acc = 0;
    for (int dx = -R; dx <= R; ++dx) { const int jx = ex + dx; if (jx < 0 || jx >= nx) continue;
    for (int dy = -R; dy <= R; ++dy) { const int jy = ey + dy; if (jy < 0 || jy >= ny) continue;
    for (int dz = -R; dz <= R; ++dz) { const int jz = ez + dz; if (jz < 0 || jz >= nz) continue;
        const __T__ w = rmin - sqrt((__T__)(dx * dx + dy * dy + dz * dz));
        if (w > (__T__)0)
            acc += w * g[(jx * ny + jy) * nz + jz] / rowsum(jx, jy, jz, nx, ny, nz, R, rmin);
    }}}
    y[e] = acc;
}
"""


class ConeFilter:
    def __init__(self, grid, rmin, dtype=np.float32):
        self.grid, self.dtype = grid, np.dtype(dtype)
        nm = "cone_" + _ctype(self.dtype)
        src = _SRC.replace("__N__", nm).replace("__T__", _ctype(self.dtype))
        mod = cp.RawModule(code=src, options=("-std=c++14",))
        self.k_fwd, self.k_tr = mod.get_function(nm + "_fwd"), mod.get_function(nm + "_tr")
        self.rmin = None
        self.set_rmin(rmin)

    def set_rmin(self, rmin):
        self.rmin = float(rmin)
        self.R = max(0, int(math.ceil(self.rmin)) - 1)

    def _run(self, k, x, y):
        g = self.grid
        k(((g.n_elem + 127) // 128,), (128,),
          (x, y, np.int32(g.nx), np.int32(g.ny), np.int32(g.nz), np.int32(self.R),
           self.dtype.type(self.rmin)))
        return y

    def forward(self, rho, out):
        return self._run(self.k_fwd, rho, out)

    def transpose(self, g, out):
        return self._run(self.k_tr, g, out)

    def neighbours(self):
        R, r = self.R, self.rmin
        return sum(1 for dx in range(-R, R + 1) for dy in range(-R, R + 1) for dz in range(-R, R + 1)
                   if r - math.sqrt(dx * dx + dy * dy + dz * dz) > 0)

    @property
    def inv_s(self):          # kept for memory accounting compatibility: nothing stored
        return None
