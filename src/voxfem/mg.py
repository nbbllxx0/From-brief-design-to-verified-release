"""Memory-lean Galerkin geometric multigrid for structured voxel elasticity.

Hierarchy (level 0 = fine grid; level l has (nx,ny,nz)/2^l elements):

  level 0      matrix-free, any mapping from voxfem.operators, fine moduli E
  level 1      matrix-free Galerkin: each coarse element matrix is
                   K1(e) = sum_{c=0..7} E_child(e,c) * KE1[c],
               KE1[c] = P_c^T K_unit P_c precomputed (8 x 24 x 24), applied
               on the fly from the fine moduli -- nothing stored
  level >= 2   element matrices stored (576 values per element, SoA layout);
               level 2 is one GEMM  K2 = KE2^T @ E_groups  with
               KE2[f] = P_f^T K_unit P_f for the 64 fine positions f; deeper
               levels by 2:1 Galerkin recursion of the stored matrices
  coarsest     dense matrix, explicitly inverted through Cholesky (FP64)

Because trilinear interpolation composes exactly on nested grids, every
coarse operator is the exact Galerkin product P^T K P of the unconstrained
fine operator; Dirichlet DOFs are then imposed on each level by masking
(identity rows), and any DOF whose Galerkin diagonal vanishes (e.g. inside
zero-modulus padding) is fixed as well.

Transfers are trilinear prolongation and its exact transpose (restriction),
both node-owned gathers.  Smoothing is damped Jacobi (nu pre- and nu post-
sweeps, same omega), so with the exact coarsest solve the V-cycle is a fixed
symmetric linear operator.  Every kernel is atomic-free (element-owned work
uses 8 parity colours), so applications are bitwise reproducible.

All multigrid arithmetic is in the precision `dtype` (FP32 by default); the
coarsest solve is FP64.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import cupy as cp

from .core import OFFSETS, Grid
from .operators import Operator, Diagonal, _get_kernel, _ctype

# ------------------------------------------------------------------ host math


def shape_values(xi):
    """Trilinear shape functions of the 8 reference nodes at local coords xi in [0,1]^3."""
    x, y, z = xi
    out = np.empty(8)
    for m, (ox, oy, oz) in enumerate(OFFSETS):
        out[m] = (x if ox else 1 - x) * (y if oy else 1 - y) * (z if oz else 1 - z)
    return out


def sub_prolongation(s, a, b, c):
    """24x24 map from a coarse element's DOFs to the DOFs of the fine element
    at position (a,b,c) of an s x s x s subdivision."""
    N = np.empty((8, 8))
    for i, (ox, oy, oz) in enumerate(OFFSETS):
        N[i] = shape_values(((a + ox) / s, (b + oy) / s, (c + oz) / s))
    return np.kron(N, np.eye(3))


_SUB_CACHE = {}


def sub_matrices(ke, s):
    """(s^3, 24, 24) array of P_f^T K P_f, f = (a*s + b)*s + c (vectorised)."""
    key = (s, hash(np.asarray(ke).tobytes()))
    if key in _SUB_CACHE:
        return _SUB_CACHE[key]
    a, b, c = np.meshgrid(np.arange(s), np.arange(s), np.arange(s), indexing="ij")
    a, b, c = a.ravel(), b.ravel(), c.ravel()
    N = np.empty((s ** 3, 8, 8))
    for i, (ox, oy, oz) in enumerate(OFFSETS):
        x, y, z = (a + ox) / s, (b + oy) / s, (c + oz) / s
        for m, (mx, my, mz) in enumerate(OFFSETS):
            N[:, i, m] = (x if mx else 1 - x) * (y if my else 1 - y) * (z if mz else 1 - z)
    P = np.einsum("fim,cd->ficmd", N, np.eye(3)).reshape(s ** 3, 24, 24)
    out = np.matmul(P.transpose(0, 2, 1), np.matmul(np.asarray(ke), P))
    _SUB_CACHE[key] = out
    return out


# symmetric packed storage of 24x24 element matrices: upper triangle, row-major
_PK_I, _PK_J = np.triu_indices(24)
PACK_IDX = (_PK_I * 24 + _PK_J).astype(np.int64)            # (300,) full -> packed order
UNPACK_IDX = np.empty(576, dtype=np.int64)                    # (576,) full position -> packed index
for _k, (_i, _j) in enumerate(zip(_PK_I, _PK_J)):
    UNPACK_IDX[_i * 24 + _j] = _k
    UNPACK_IDX[_j * 24 + _i] = _k
NPACK = 300

_PK_MACRO = r"""
#define PK(i, j) ((i) <= (j) ? 24 * (i) - ((i) * ((i) - 1)) / 2 + (j) - (i) \
                             : 24 * (j) - ((j) * ((j) - 1)) / 2 + (i) - (j))
"""

# ------------------------------------------------------------------ kernels

_SRC_L1 = r"""
extern "C" __global__ void __N__(const __T__* __restrict__ KE1, const __T__* __restrict__ Ef,
    const __T__* __restrict__ x, __T__* __restrict__ y,
    const int cx, const int cy, const int cz, const int px, const int py, const int pz)
{
    // level-1 matrix-free Galerkin, element-owned, one parity colour per launch
    extern __shared__ unsigned char raw[];
    __T__* K = (__T__*)raw;                              // 8*576
    for (int i = threadIdx.x; i < 8 * 576; i += blockDim.x) K[i] = KE1[i];
    __syncthreads();
    const int hx = (cx - px + 1) / 2, hy = (cy - py + 1) / 2, hz = (cz - pz + 1) / 2;
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= hx * hy * hz) return;
    const int kz = q % hz, t = q / hz, ky = t % hy, kx = t / hy;
    const int EX = 2 * kx + px, EY = 2 * ky + py, EZ = 2 * kz + pz;
    const int sy = cz + 1, sx = (cy + 1) * (cz + 1);
    const int nb = EX * sx + EY * sy + EZ;
    int nd[8];
    nd[0] = nb;           nd[1] = nb + sx;
    nd[2] = nb + sx + sy; nd[3] = nb + sy;
    nd[4] = nb + 1;       nd[5] = nb + sx + 1;
    nd[6] = nb + sx+sy+1; nd[7] = nb + sy + 1;
    __T__ xe[24], acc[24];
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        const int d = 3 * nd[m];
        xe[3*m] = x[d]; xe[3*m+1] = x[d+1]; xe[3*m+2] = x[d+2];
    }
    #pragma unroll
    for (int i = 0; i < 24; ++i) acc[i] = 0;
    const int fny = 2 * cy, fnz = 2 * cz;
    for (int c = 0; c < 8; ++c) {
        const int a = c >> 2, b = (c >> 1) & 1, d = c & 1;
        const __T__ Ec = Ef[((2 * EX + a) * fny + 2 * EY + b) * fnz + 2 * EZ + d];
        if (Ec == (__T__)0) continue;
        const __T__* Kc = K + c * 576;
        #pragma unroll
        for (int i = 0; i < 24; ++i) {
            __T__ s = 0;
            #pragma unroll
            for (int j = 0; j < 24; ++j) s = fma(Kc[i * 24 + j], xe[j], s);
            acc[i] = fma(Ec, s, acc[i]);
        }
    }
    #pragma unroll
    for (int i = 0; i < 24; ++i) y[3 * nd[i / 3] + (i % 3)] += acc[i];
}
"""

_SRC_STORED = _PK_MACRO + r"""
extern "C" __global__ void __N__(const __T__* __restrict__ M, const __T__* __restrict__ x,
    __T__* __restrict__ y, const int cx, const int cy, const int cz,
    const int px, const int py, const int pz)
{
    // stored symmetric-packed element matrices, SoA layout M[PK(i,j) * ne + e],
    // one parity colour per launch
    const int hx = (cx - px + 1) / 2, hy = (cy - py + 1) / 2, hz = (cz - pz + 1) / 2;
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= hx * hy * hz) return;
    const int kz = q % hz, t = q / hz, ky = t % hy, kx = t / hy;
    const int EX = 2 * kx + px, EY = 2 * ky + py, EZ = 2 * kz + pz;
    const int ne = cx * cy * cz;
    const int e = (EX * cy + EY) * cz + EZ;
    const int sy = cz + 1, sx = (cy + 1) * (cz + 1);
    const int nb = EX * sx + EY * sy + EZ;
    int nd[8];
    nd[0] = nb;           nd[1] = nb + sx;
    nd[2] = nb + sx + sy; nd[3] = nb + sy;
    nd[4] = nb + 1;       nd[5] = nb + sx + 1;
    nd[6] = nb + sx+sy+1; nd[7] = nb + sy + 1;
    __T__ xe[24];
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        const int d = 3 * nd[m];
        xe[3*m] = x[d]; xe[3*m+1] = x[d+1]; xe[3*m+2] = x[d+2];
    }
    #pragma unroll 4
    for (int i = 0; i < 24; ++i) {
        __T__ s = 0;
        #pragma unroll
        for (int j = 0; j < 24; ++j) s = fma(M[(size_t)PK(i, j) * ne + e], xe[j], s);
        y[3 * nd[i / 3] + (i % 3)] += s;
    }
}
"""

# diagonal of the level-1 Galerkin operator, node-owned (KE1 diag in global)
_SRC_DIAG_L1 = r"""
extern "C" __global__ void __N__(const __T__* __restrict__ KE1, const __T__* __restrict__ Ef,
    __T__* __restrict__ diag, const int cx, const int cy, const int cz)
{
    const int nnz = cz + 1, nny = cy + 1;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= (cx + 1) * nny * nnz) return;
    const int iz = n % nnz, tn = n / nnz, iy = tn % nny, ix = tn / nny;
    const int LUT[8] = {0, 4, 3, 7, 1, 5, 2, 6};
    const int fny = 2 * cy, fnz = 2 * cz;
    __T__ d0 = 0, d1 = 0, d2 = 0;
    for (int a = 0; a < 2; ++a) { const int ex = ix - 1 + a; if (ex < 0 || ex >= cx) continue;
    for (int b = 0; b < 2; ++b) { const int ey = iy - 1 + b; if (ey < 0 || ey >= cy) continue;
    for (int d = 0; d < 2; ++d) { const int ez = iz - 1 + d; if (ez < 0 || ez >= cz) continue;
        const int li = LUT[(1 - a) * 4 + (1 - b) * 2 + (1 - d)];
        for (int c = 0; c < 8; ++c) {
            const int ca = c >> 2, cb = (c >> 1) & 1, cd = c & 1;
            const __T__ Ec = Ef[((2 * ex + ca) * fny + 2 * ey + cb) * fnz + 2 * ez + cd];
            const __T__* Kc = KE1 + c * 576;
            d0 += Ec * Kc[(3*li) * 24 + 3*li];
            d1 += Ec * Kc[(3*li+1) * 24 + 3*li+1];
            d2 += Ec * Kc[(3*li+2) * 24 + 3*li+2];
        }
    }}}
    diag[3*n] = d0; diag[3*n+1] = d1; diag[3*n+2] = d2;
}
"""

_SRC_DIAG_STORED = _PK_MACRO + r"""
extern "C" __global__ void __N__(const __T__* __restrict__ M, __T__* __restrict__ diag,
    const int cx, const int cy, const int cz)
{
    const int nnz = cz + 1, nny = cy + 1;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= (cx + 1) * nny * nnz) return;
    const int iz = n % nnz, tn = n / nnz, iy = tn % nny, ix = tn / nny;
    const int LUT[8] = {0, 4, 3, 7, 1, 5, 2, 6};
    const int ne = cx * cy * cz;
    __T__ d0 = 0, d1 = 0, d2 = 0;
    for (int a = 0; a < 2; ++a) { const int ex = ix - 1 + a; if (ex < 0 || ex >= cx) continue;
    for (int b = 0; b < 2; ++b) { const int ey = iy - 1 + b; if (ey < 0 || ey >= cy) continue;
    for (int d = 0; d < 2; ++d) { const int ez = iz - 1 + d; if (ez < 0 || ez >= cz) continue;
        const int li = LUT[(1 - a) * 4 + (1 - b) * 2 + (1 - d)];
        const int e = (ex * cy + ey) * cz + ez;
        d0 += M[(size_t)PK(3*li, 3*li) * ne + e];
        d1 += M[(size_t)PK(3*li+1, 3*li+1) * ne + e];
        d2 += M[(size_t)PK(3*li+2, 3*li+2) * ne + e];
    }}}
    diag[3*n] = d0; diag[3*n+1] = d1; diag[3*n+2] = d2;
}
"""

# restriction fine(l) -> coarse(l+1): r_c = P^T r_f, node-owned on the coarse grid
_SRC_RESTRICT = r"""
extern "C" __global__ void __N__(const __T__* __restrict__ rf, __T__* __restrict__ rc,
    const int cx, const int cy, const int cz, const unsigned char* __restrict__ cfixed)
{
    const int nnz = cz + 1, nny = cy + 1;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= (cx + 1) * nny * nnz) return;
    const int K = n % nnz, tn = n / nnz, J = tn % nny, I = tn / nny;
    const int fx = 2 * cx, fy = 2 * cy, fz = 2 * cz;
    const int fsy = fz + 1, fsx = (fy + 1) * (fz + 1);
    __T__ s0 = 0, s1 = 0, s2 = 0;
    #pragma unroll
    for (int dx = -1; dx <= 1; ++dx) { const int i = 2 * I + dx; if (i < 0 || i > fx) continue;
      const __T__ wx = dx == 0 ? (__T__)1 : (__T__)0.5;
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) { const int j = 2 * J + dy; if (j < 0 || j > fy) continue;
      const __T__ wy = wx * (dy == 0 ? (__T__)1 : (__T__)0.5);
    #pragma unroll
    for (int dz = -1; dz <= 1; ++dz) { const int k = 2 * K + dz; if (k < 0 || k > fz) continue;
      const __T__ w = wy * (dz == 0 ? (__T__)1 : (__T__)0.5);
      const int f = 3 * (i * fsx + j * fsy + k);
      s0 = fma(w, rf[f], s0); s1 = fma(w, rf[f + 1], s1); s2 = fma(w, rf[f + 2], s2);
    }}}
    const unsigned char m = cfixed[n];
    rc[3*n] = (m & 1) ? (__T__)0 : s0;
    rc[3*n+1] = (m & 2) ? (__T__)0 : s1;
    rc[3*n+2] = (m & 4) ? (__T__)0 : s2;
}
"""

# prolongation coarse(l+1) -> fine(l), added to the fine vector: x_f += P x_c
_SRC_PROLONG = r"""
extern "C" __global__ void __N__(const __T__* __restrict__ xc, __T__* __restrict__ xf,
    const int fx, const int fy, const int fz, const unsigned char* __restrict__ ffixed)
{
    const int nnz = fz + 1, nny = fy + 1;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= (fx + 1) * nny * nnz) return;
    const unsigned char m = ffixed[n];
    if (m == 7) return;
    const int k = n % nnz, tn = n / nnz, j = tn % nny, i = tn / nny;
    const int cy = fy / 2, cz = fz / 2;
    const int csy = cz + 1, csx = (cy + 1) * (cz + 1);
    const int i0 = i >> 1, j0 = j >> 1, k0 = k >> 1;
    const int ni = (i & 1) ? 2 : 1, nj = (j & 1) ? 2 : 1, nk = (k & 1) ? 2 : 1;
    const __T__ wi = (i & 1) ? (__T__)0.5 : (__T__)1, wj = (j & 1) ? (__T__)0.5 : (__T__)1,
                wk = (k & 1) ? (__T__)0.5 : (__T__)1;
    __T__ s0 = 0, s1 = 0, s2 = 0;
    for (int a = 0; a < ni; ++a) for (int b = 0; b < nj; ++b) for (int c = 0; c < nk; ++c) {
        const __T__ w = wi * wj * wk;
        const int q = 3 * ((i0 + a) * csx + (j0 + b) * csy + k0 + c);
        s0 = fma(w, xc[q], s0); s1 = fma(w, xc[q + 1], s1); s2 = fma(w, xc[q + 2], s2);
    }
    if (!(m & 1)) xf[3*n] += s0;
    if (!(m & 2)) xf[3*n+1] += s1;
    if (!(m & 4)) xf[3*n+2] += s2;
}
"""

_SRC_MASKOUT = r"""
extern "C" __global__ void __N__(const __T__* __restrict__ x, __T__* __restrict__ y,
    const unsigned char* __restrict__ fixed, const int n_node)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= n_node) return;
    const unsigned char f = fixed[n];
    if (f & 1) y[3*n] = x[3*n];
    if (f & 2) y[3*n+1] = x[3*n+1];
    if (f & 4) y[3*n+2] = x[3*n+2];
}
"""


_SRC_L0 = r"""
__device__ __forceinline__ void diag3(const __T__* __restrict__ KE, const __T__* __restrict__ E,
    const int n, const int nx, const int ny, const int nz, __T__& d0, __T__& d1, __T__& d2)
{
    const int nnz = nz + 1, nny = ny + 1;
    const int iz = n % nnz, tn = n / nnz, iy = tn % nny, ix = tn / nny;
    const int LUT[8] = {0, 4, 3, 7, 1, 5, 2, 6};
    d0 = 0; d1 = 0; d2 = 0;
    #pragma unroll
    for (int a = 0; a < 2; ++a) { const int ex = ix - 1 + a; if (ex < 0 || ex >= nx) continue;
    #pragma unroll
    for (int b = 0; b < 2; ++b) { const int ey = iy - 1 + b; if (ey < 0 || ey >= ny) continue;
    #pragma unroll
    for (int d = 0; d < 2; ++d) { const int ez = iz - 1 + d; if (ez < 0 || ez >= nz) continue;
        const __T__ Ee = E[(ex * ny + ey) * nz + ez];
        const int li = LUT[(1 - a) * 4 + (1 - b) * 2 + (1 - d)];
        d0 += Ee * KE[(3*li) * 25]; d1 += Ee * KE[(3*li+1) * 25]; d2 += Ee * KE[(3*li+2) * 25];
    }}}
}

extern "C" __global__ void __N___jac(const __T__* __restrict__ KE, const __T__* __restrict__ E,
    const __T__* __restrict__ bv, const __T__* __restrict__ t, __T__* __restrict__ x,
    const unsigned char* __restrict__ fixed, const __T__ omega, const int mode,
    const int nx, const int ny, const int nz)
{
    // mode 0: x = omega D^-1 b;  mode 1: x += omega D^-1 (b - t);  mode 2: t = D^-1 t
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= (nx + 1) * (ny + 1) * (nz + 1)) return;
    __T__ d[3];
    diag3(KE, E, n, nx, ny, nz, d[0], d[1], d[2]);
    const unsigned char f = fixed[n];
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        const int q = 3 * n + c;
        const bool fx = (f & (1 << c)) || !(d[c] > (__T__)0);
        if (mode == 0) x[q] = fx ? (__T__)0 : omega * bv[q] / d[c];
        else if (mode == 1) { if (!fx) x[q] += omega * (bv[q] - t[q]) / d[c]; }
        else x[q] = fx ? (__T__)0 : x[q] / d[c];
    }
}

extern "C" __global__ void __N___fix(const __T__* __restrict__ KE, const __T__* __restrict__ E,
    unsigned char* __restrict__ fixed, const int nx, const int ny, const int nz)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= (nx + 1) * (ny + 1) * (nz + 1)) return;
    __T__ d0, d1, d2;
    diag3(KE, E, n, nx, ny, nz, d0, d1, d2);
    unsigned char f = fixed[n];
    if (!(d0 > (__T__)0)) f |= 1;
    if (!(d1 > (__T__)0)) f |= 2;
    if (!(d2 > (__T__)0)) f |= 4;
    fixed[n] = f;
}
"""

_SRC_FINISH = r"""
extern "C" __global__ void __N__(__T__* __restrict__ d, unsigned char* __restrict__ fixed, const int n_node)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= n_node) return;
    unsigned char f = fixed[n];
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        const __T__ v = d[3 * n + c];
        if (!(v > (__T__)0)) f |= (unsigned char)(1 << c);
        d[3 * n + c] = (f & (1 << c)) ? (__T__)0 : (__T__)1 / v;
    }
    fixed[n] = f;
}
"""


def _k(src, base, dtype):
    nm = f"{base}_{_ctype(dtype)}"
    return _get_kernel(src.replace("__N__", nm).replace("__T__", _ctype(dtype)), nm)


_jac_first = {}
_jac_sweep = {}


def _jacobi_kernels(dtype):
    key = np.dtype(dtype).name
    if key not in _jac_first:
        _jac_first[key] = cp.ElementwiseKernel(
            "T b, T dinv, T omega", "T x", "x = omega * dinv * b", f"jac_first_{key}")
        _jac_sweep[key] = cp.ElementwiseKernel(
            "T b, T ax, T dinv, T omega", "T x", "x += omega * dinv * (b - ax)", f"jac_sweep_{key}")
    return _jac_first[key], _jac_sweep[key]


# ------------------------------------------------------------------ hierarchy

class Level:
    pass


def choose_levels(grid: Grid, max_coarse_dofs=2500, max_levels=10):
    L = 1
    g = grid
    while L < max_levels:
        if g.nx % 2 or g.ny % 2 or g.nz % 2 or min(g.shape) < 4:
            break
        if g.ndof <= max_coarse_dofs:
            break
        g = Grid(g.nx // 2, g.ny // 2, g.nz // 2)
        L += 1
    return L


class GMG:
    """Galerkin V-cycle preconditioner z = M r on a (possibly padded) grid.

    grid, fixed        the fine grid and its per-node fixed mask (uint8, host
                       or device); grid dims must be divisible by 2^(levels-1)
    fine_variant       operator mapping used on level 0
    """

    def __init__(self, grid: Grid, fixed, ke, fine_variant="stencil", levels=None,
                 nu=2, omega=None, omega_factor=1.2, power_its=12, weights_every=5, dtype=np.float32,
                 max_coarse_dofs=2500):
        self.dtype = np.dtype(dtype)
        self.ke = np.asarray(ke, dtype=np.float64)
        self.nu = int(nu)
        self.fixed_omega = omega
        self.omega_factor, self.power_its = float(omega_factor), int(power_its)
        self.weights_every = int(weights_every)
        self._updates = 0
        self.L = levels or choose_levels(grid, max_coarse_dofs)
        if not grid.coarsenable(self.L):
            raise ValueError(f"grid {grid.shape} not divisible by 2^{self.L - 1}")
        s = 2 ** (self.L - 1)
        nc = 3 * (grid.nx // s + 1) * (grid.ny // s + 1) * (grid.nz // s + 1)
        if nc > 8 * max_coarse_dofs:      # the coarsest level is factorised densely: fail instead of exhausting VRAM
            raise ValueError(f"coarsest level of grid {grid.shape} has {nc} DoFs (> {8 * max_coarse_dofs}); "
                             "pad the grid (Problem.padded) so that it coarsens further")
        self.fine_variant = fine_variant
        self.levels = []
        fixed = cp.array(fixed, dtype=cp.uint8, copy=True)
        g = grid
        for l in range(self.L):
            lv = Level()
            lv.grid = g
            if l == 0:
                lv.fixed = fixed
            else:
                lv.fixed = self._coarse_fixed(self.levels[-1], g)
            lv.n_node = g.n_node
            self.levels.append(lv)
            g = Grid(g.nx // 2, g.ny // 2, g.nz // 2)
        dt = self.dtype
        self.op0 = Operator(grid, self.ke, fine_variant, dt)
        self.diag0 = Diagonal(grid, self.ke, dt)
        self.KE1 = cp.asarray(sub_matrices(self.ke, 2).reshape(8, 576), dtype=dt)
        self._ke_tables = {}
        Pc = np.stack([sub_prolongation(2, a, b, c) for a, b, c in product(range(2), repeat=3)])
        self.Pc = cp.asarray(Pc, dtype=dt)                                    # (8,24,24)
        self.k_l1 = _k(_SRC_L1, "mg_l1", dt)
        self.k_stored = _k(_SRC_STORED, "mg_stored", dt)
        self.k_diag_l1 = _k(_SRC_DIAG_L1, "mg_diag_l1", dt)
        self.k_diag_st = _k(_SRC_DIAG_STORED, "mg_diag_st", dt)
        self.k_restrict = _k(_SRC_RESTRICT, "mg_restrict", dt)
        self.k_prolong = _k(_SRC_PROLONG, "mg_prolong", dt)
        self.k_maskout = _k(_SRC_MASKOUT, "mg_maskout", dt)
        self.k_finish = _k(_SRC_FINISH, "mg_finish", dt)
        nm0 = f"mg_l0_{_ctype(dt)}"
        mod0 = cp.RawModule(code=_SRC_L0.replace("__N__", nm0).replace("__T__", _ctype(dt)),
                            options=("-std=c++14",))
        self.k_l0_jac = mod0.get_function(nm0 + "_jac")
        self.k_l0_fix = mod0.get_function(nm0 + "_fix")
        self.KE0 = cp.asarray(self.ke, dtype=dt)
        self.jac_first, self.jac_sweep = _jacobi_kernels(dt)
        # work vectors; level 0 uses caller's b/x plus one temporary
        for l, lv in enumerate(self.levels):
            nd = lv.grid.ndof
            lv.t = cp.empty(nd, dtype=dt)
            lv.dinv = cp.empty(nd, dtype=dt) if l > 0 else None    # level 0: diagonal on the fly
            if l > 0:
                lv.b = cp.empty(nd, dtype=dt)
                lv.x = cp.empty(nd, dtype=dt)
        self.E = None
        self.stats = {"vcycles": 0}

    DIRECT_MAX = 5

    def _ke_table(self, l):
        if l not in self._ke_tables:
            s = 2 ** l
            full = sub_matrices(self.ke, s).reshape(s ** 3, 576)
            self._ke_tables[l] = cp.asarray(full[:, PACK_IDX].T.copy(), dtype=self.dtype)  # (300, s^3)
        return self._ke_tables[l]

    # ---- setup -----------------------------------------------------------
    @staticmethod
    def _coarse_fixed(fine_lv, gc: Grid):
        ff = fine_lv.fixed.reshape(fine_lv.grid.node_shape)
        return cp.ascontiguousarray(ff[::2, ::2, ::2]).ravel()

    def update(self, E):
        """(Re)build every coarse operator for new fine moduli E (dtype)."""
        dt = self.dtype
        self.E = E.astype(dt, copy=False)
        g0 = self.levels[0].grid
        # level 0 diagonal
        lv0 = self.levels[0]
        g0 = lv0.grid
        self.k_l0_fix(((g0.n_node + 127) // 128,), (128,),
                      (self.KE0, self.E, lv0.fixed, np.int32(g0.nx), np.int32(g0.ny), np.int32(g0.nz)))
        if self.L == 1:
            raise ValueError("need at least two levels")
        # level 1 diagonal (matrix-free)
        lv1 = self.levels[1]
        g1 = lv1.grid
        d1 = lv1.dinv
        self.k_diag_l1(((g1.n_node + 127) // 128,), (128,),
                       (self.KE1, self.E, d1, np.int32(g1.nx), np.int32(g1.ny), np.int32(g1.nz)))
        self._finish_diag(lv1, d1)
        # stored levels: levels 2..DIRECT_MAX by one GEMM each from the fine
        # moduli (K_l = KE_l^T @ E grouped by level-l element), deeper levels
        # by 2:1 Galerkin recursion of the (by then tiny) stored matrices
        mats = None
        for l in range(2, self.L):
            gl = self.levels[l].grid
            if l <= self.DIRECT_MAX:
                s = 2 ** l
                KT = self._ke_table(l)                                  # (576, s^3)
                Eg = self.E.reshape(gl.nx, s, gl.ny, s, gl.nz, s).transpose(1, 3, 5, 0, 2, 4)
                Eg = cp.ascontiguousarray(Eg).reshape(s ** 3, gl.n_elem)
                mats = KT @ Eg                                          # (300, n_l) packed SoA
                del Eg
            else:
                mats = self._recurse(self.levels[l - 1].grid, gl, mats)
            self._set_stored(self.levels[l], mats)
        self._coarsest()
        if self._updates % self.weights_every == 0 or self.force_weights:
            self._smoother_weights()
            self.force_weights = False
        self._updates += 1
        cp.get_default_memory_pool().free_all_blocks()
        return self

    scratch0 = None
    force_weights = False

    def _v0(self):
        """Level-0 scratch vector for the power iteration: a caller-provided
        buffer that is free during setup (scratch0), else a temporary."""
        if self.scratch0 is not None:
            return self.scratch0
        return cp.empty(self.levels[0].grid.ndof, dtype=self.dtype)

    def _smoother_weights(self):
        """omega_l = omega_factor / lambda_max(D_l^-1 A_l), lambda_max from a
        fixed-seed power iteration (D^-1 A is similar to an SPD matrix, so its
        spectrum is real); a fixed omega overrides."""
        self.lambda_max = []
        for l, lv in enumerate(self.levels[:-1]):
            if self.fixed_omega is not None:
                lv.omega = self.dtype.type(self.fixed_omega)
                self.lambda_max.append(None)
                continue
            v = lv.x if l > 0 else self._v0()
            v[...] = 0
            v[:: 7] = 1.0                    # deterministic, allocation-free start vector
            v[:: 3] -= 0.5
            v *= self.dtype.type(1.0 / float(cp.linalg.norm(v)))
            lam = 0.0
            for _ in range(self.power_its):
                self._apply_level(l, v, lv.t)
                if l == 0:
                    self._l0(2, lv.t, lv.t, lv.t, np.float32(1))
                else:
                    lv.t *= lv.dinv
                nrm = float(cp.linalg.norm(lv.t))
                lam = nrm
                cp.multiply(lv.t, self.dtype.type(1.0 / nrm), out=v)
            self.lambda_max.append(lam)
            lv.omega = self.dtype.type(self.omega_factor / lam)

    def _finish_diag(self, lv, d):
        """In place: DOFs with a non-positive diagonal become fixed; dinv =
        1/diag on free DOFs and 0 on fixed ones (no full-size temporaries;
        d may alias lv.dinv)."""
        if d is not lv.dinv:
            lv.dinv[...] = d
        g = lv.grid
        self.k_finish(((g.n_node + 255) // 256,), (256,), (lv.dinv, lv.fixed, np.int32(g.n_node)))

    def _set_stored(self, lv, mats):
        lv.M = mats
        g = lv.grid
        d = lv.dinv
        self.k_diag_st(((g.n_node + 127) // 128,), (128,),
                       (mats, d, np.int32(g.nx), np.int32(g.ny), np.int32(g.nz)))
        self._finish_diag(lv, d)

    def _recurse(self, gf: Grid, gc: Grid, Mf):
        """Coarse element matrices from fine stored matrices (packed SoA (300, nf))."""
        Mf = Mf[cp.asarray(UNPACK_IDX)]
        A = Mf.T.reshape(gc.nx, 2, gc.ny, 2, gc.nz, 2, 24, 24)
        A = A.transpose(0, 2, 4, 1, 3, 5, 6, 7).reshape(gc.n_elem, 8, 24, 24)
        out = cp.zeros((gc.n_elem, 24, 24), dtype=self.dtype)
        for c in range(8):
            P = self.Pc[c]
            out += P.T @ A[:, c] @ P
        return cp.ascontiguousarray(out.reshape(gc.n_elem, 576).T[cp.asarray(PACK_IDX)])

    def _coarsest(self):
        lv = self.levels[-1]
        g = lv.grid
        Ms = getattr(lv, "M", None)
        if Ms is None:                       # two-level hierarchy: coarsest is level 1
            l = self.L - 1
            s = 2 ** l
            Eg = self.E.reshape(g.nx, s, g.ny, s, g.nz, s).transpose(1, 3, 5, 0, 2, 4)
            Eg = cp.ascontiguousarray(Eg).reshape(s ** 3, g.n_elem)
            Ms = self._ke_table(l) @ Eg
        M = Ms[cp.asarray(UNPACK_IDX)].T.reshape(g.n_elem, 24, 24).astype(cp.float64)
        n = g.ndof
        if getattr(self, "_coarse_index", None) is None or self._coarse_index[0] != n:
            edof = g.edof()
            flat = (edof[:, :, None] * n + edof[:, None, :]).ravel()
            self._coarse_index = (n, cp.asarray(flat, dtype=cp.int64))
        A = cp.zeros(n * n, dtype=cp.float64)
        import cupyx
        cupyx.scatter_add(A, self._coarse_index[1], M.ravel())
        A = A.reshape(n, n)
        fx = lv.fixed
        fd = cp.stack([(fx & 1) > 0, (fx & 2) > 0, (fx & 4) > 0], axis=1).ravel()
        free = (~fd).astype(cp.float64)
        A = A * free[:, None] * free[None, :] + cp.diag(1.0 - free)
        Lc = cp.linalg.cholesky(A)
        from cupyx.scipy.linalg import solve_triangular
        Linv = solve_triangular(Lc, cp.eye(n), lower=True)      # A^-1 = L^-T L^-1
        self.coarse_Linv = Linv * free[None, :]                  # zero fixed columns
        self.coarse_inv = self.coarse_Linv                       # (kept for storage accounting)
        self.coarse_n = n
        self.coarse_free_dofs = int(free.sum())

    # ---- application -----------------------------------------------------
    def _apply_level(self, l, x, y):
        lv = self.levels[l]
        g = lv.grid
        B = 128
        if l == 0:
            self.op0.apply(x, self.E, y, fixed=lv.fixed)
            return y
        y.fill(0)
        if l == 1:
            smem = 8 * 576 * self.dtype.itemsize
            for px, py, pz in product((0, 1), repeat=3):
                h = ((g.nx - px + 1) // 2) * ((g.ny - py + 1) // 2) * ((g.nz - pz + 1) // 2)
                if h:
                    self.k_l1(((h + B - 1) // B,), (B,),
                              (self.KE1, self.E, x, y, np.int32(g.nx), np.int32(g.ny), np.int32(g.nz),
                               np.int32(px), np.int32(py), np.int32(pz)), shared_mem=smem)
        else:
            for px, py, pz in product((0, 1), repeat=3):
                h = ((g.nx - px + 1) // 2) * ((g.ny - py + 1) // 2) * ((g.nz - pz + 1) // 2)
                if h:
                    self.k_stored(((h + B - 1) // B,), (B,),
                                  (lv.M, x, y, np.int32(g.nx), np.int32(g.ny), np.int32(g.nz),
                                   np.int32(px), np.int32(py), np.int32(pz)))
        self.k_maskout(((g.n_node + 255) // 256,), (256,), (x, y, lv.fixed, np.int32(g.n_node)))
        return y

    def apply_level(self, l, x, y=None):
        """Level-l operator application (testing)."""
        if y is None:
            y = cp.empty_like(x)
        return self._apply_level(l, x, y)

    def _vcycle(self, l, b, x):
        lv = self.levels[l]
        if l == self.L - 1:
            Li = self.coarse_Linv
            x[...] = (Li.T @ (Li @ b.astype(cp.float64))).astype(self.dtype)
            return
        om = lv.omega
        if l == 0:
            self._l0(0, b, b, x, om)
        else:
            self.jac_first(b, lv.dinv, om, x)
        for _ in range(self.nu - 1):
            self._apply_level(l, x, lv.t)
            if l == 0:
                self._l0(1, b, lv.t, x, om)
            else:
                self.jac_sweep(b, lv.t, lv.dinv, om, x)
        self._apply_level(l, x, lv.t)
        cp.subtract(b, lv.t, out=lv.t)                     # residual
        nxt = self.levels[l + 1]
        gc = nxt.grid
        self.k_restrict(((gc.n_node + 127) // 128,), (128,),
                        (lv.t, nxt.b, np.int32(gc.nx), np.int32(gc.ny), np.int32(gc.nz), nxt.fixed))
        self._vcycle(l + 1, nxt.b, nxt.x)
        g = lv.grid
        self.k_prolong(((g.n_node + 127) // 128,), (128,),
                       (nxt.x, x, np.int32(g.nx), np.int32(g.ny), np.int32(g.nz), lv.fixed))
        for _ in range(self.nu):
            self._apply_level(l, x, lv.t)
            if l == 0:
                self._l0(1, b, lv.t, x, om)
            else:
                self.jac_sweep(b, lv.t, lv.dinv, om, x)

    def _l0(self, mode, bv, t, x, omega):
        g = self.levels[0].grid
        self.k_l0_jac(((g.n_node + 127) // 128,), (128,),
                      (self.KE0, self.E, bv, t, x, self.levels[0].fixed, self.dtype.type(omega),
                       np.int32(mode), np.int32(g.nx), np.int32(g.ny), np.int32(g.nz)))

    def __call__(self, r, z):
        """z = M r on the fine grid (both dtype); z is overwritten."""
        self._vcycle(0, r, z)
        self.stats["vcycles"] += 1
        return z

    @property
    def fixed0(self):
        return self.levels[0].fixed

    def storage_bytes(self):
        tot = {"vectors": 0, "matrices": 0, "coarse_inverse": int(self.coarse_inv.nbytes)}
        for l, lv in enumerate(self.levels):
            tot["vectors"] += int(lv.t.nbytes + (lv.dinv.nbytes if lv.dinv is not None else 0)
                                  + lv.fixed.nbytes)
            if l > 0:
                tot["vectors"] += int(lv.b.nbytes + lv.x.nbytes)
            if getattr(lv, "M", None) is not None:
                tot["matrices"] += int(lv.M.nbytes)
        tot["KE_tables"] = int(self.KE1.nbytes + self.Pc.nbytes + sum(t.nbytes for t in self._ke_tables.values()))
        return tot
