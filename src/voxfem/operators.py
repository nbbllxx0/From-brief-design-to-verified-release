"""Matrix-free mappings of the structured-voxel stiffness action y = K(E) v.

Every mapping computes the same operator,

    y = sum_e E_e * B_e^T K_unit B_e v,

on a grid of nx*ny*nz trilinear hexahedra, and differs only in which thread
owns which output, what is re-read, and how many floating-point instructions
are issued per output:

  elem_atomic   one thread per element; gathers its 24 DOFs, applies K_unit
                from shared memory, scatters 24 results with atomicAdd.  The
                R2 "fused + analytic index" kernel.
  elem_colored  as elem_atomic, but launched in 8 parity colours so that no
                two threads of a launch share a node; plain read-modify-write,
                no atomics, deterministic.
  node          one thread per node, gathers from its <= 8 incident elements;
                the modulus multiplies every partial product (the R2 node-owned
                kernel, 4 FP instructions per row and neighbour node).
  node_hoist    node ownership with the modulus factored out of the inner
                loop: 3 FMA per row and neighbour node plus 3 per element,
                i.e. the same FP instruction count per output as elem_atomic
                has per element.
  dof           one thread per DOF (the DoF-by-DoF organisation), modulus
                inside the loop.
  tile          node ownership with the 3 x 3 x (B+2) displacement column
                staged in shared memory.
  spoke         27-neighbour node organisation: for each neighbour node, the
                contributions of the incident elements are added with
                compile-time constant weights (the organisation of the
                published OpenMP GPU code of Traeff et al.).
  stencil       generated 27-point stencil: for each neighbour, each 3x3
                block coefficient is formed once as a constant-weighted sum of
                incident moduli, grouped by equal weight, then applied.

All kernels accept an optional fixed-node mask (uint8 bitmask per node).
Node-owned kernels apply it in the epilogue: a fixed DOF returns its input,
which makes the operator SPD on the free space as long as callers keep their
iterates zero on fixed DOFs (the Krylov drivers do).  Element-owned kernels
apply it with a separate epilogue kernel.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import cupy as cp

from .core import OFFSETS, Grid

_OPTS = ("-std=c++14",)

# -------------------------------------------------------------------------
# hand-written kernels; __T__ is replaced by float/double, __N__ by the name
# -------------------------------------------------------------------------

_ELEM_COMMON = r"""
    const int sy = nz + 1, sx = (ny + 1) * (nz + 1);
    const int nb = ex * sx + ey * sy + ez;
    int nd[8];
    nd[0] = nb;           nd[1] = nb + sx;
    nd[2] = nb + sx + sy; nd[3] = nb + sy;
    nd[4] = nb + 1;       nd[5] = nb + sx + 1;
    nd[6] = nb + sx+sy+1; nd[7] = nb + sy + 1;
    __T__ ue[24];
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        const int d = 3 * nd[m];
        ue[3*m] = u[d]; ue[3*m+1] = u[d+1]; ue[3*m+2] = u[d+2];
    }
    const __T__ Ee = E[(ex * ny + ey) * nz + ez];
"""

_SRC_ELEM_ATOMIC = r"""
extern "C" __global__ void __N__(const __T__* __restrict__ KE,
    const __T__* __restrict__ E, const __T__* __restrict__ u,
    __T__* __restrict__ y, const int nx, const int ny, const int nz)
{
    __shared__ __T__ Ks[576];
    for (int i = threadIdx.x; i < 576; i += blockDim.x) Ks[i] = KE[i];
    __syncthreads();
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nx * ny * nz) return;
    const int ez = e % nz, t = e / nz, ey = t % ny, ex = t / ny;
""" + _ELEM_COMMON + r"""
    #pragma unroll
    for (int i = 0; i < 24; ++i) {
        __T__ acc = (__T__)0;
        #pragma unroll
        for (int j = 0; j < 24; ++j) acc += Ks[i * 24 + j] * ue[j];
        atomicAdd(&y[3 * nd[i / 3] + (i % 3)], Ee * acc);
    }
}
"""

# element-level difference form (review comparator): the element's first-node displacement is subtracted
# from all 24 components before the element product.  K_unit annihilates translations, so the result is the
# same operator in exact arithmetic, and a rigid translation gives exactly zero (all differences are zero).
_SRC_ELEM_REF = r"""
extern "C" __global__ void __N__(const __T__* __restrict__ KE,
    const __T__* __restrict__ E, const __T__* __restrict__ u,
    __T__* __restrict__ y, const int nx, const int ny, const int nz)
{
    __shared__ __T__ Ks[576];
    for (int i = threadIdx.x; i < 576; i += blockDim.x) Ks[i] = KE[i];
    __syncthreads();
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nx * ny * nz) return;
    const int ez = e % nz, t = e / nz, ey = t % ny, ex = t / ny;
""" + _ELEM_COMMON + r"""
    const __T__ r0 = ue[0], r1 = ue[1], r2 = ue[2];
    #pragma unroll
    for (int m = 0; m < 8; ++m) { ue[3*m] -= r0; ue[3*m+1] -= r1; ue[3*m+2] -= r2; }
    #pragma unroll
    for (int i = 0; i < 24; ++i) {
        __T__ acc = (__T__)0;
        #pragma unroll
        for (int j = 0; j < 24; ++j) acc += Ks[i * 24 + j] * ue[j];
        atomicAdd(&y[3 * nd[i / 3] + (i % 3)], Ee * acc);
    }
}
"""

# element-level difference form with a symmetric reference: the element mean, summed pairwise so that it is
# exact for a translation (2t, 4t, 8t are exact), then scaled by 1/8 (exact).
_SRC_ELEM_MEAN = _SRC_ELEM_REF.replace(
    "const __T__ r0 = ue[0], r1 = ue[1], r2 = ue[2];",
    "const __T__ r0 = (((ue[0]+ue[3])+(ue[6]+ue[9]))+((ue[12]+ue[15])+(ue[18]+ue[21]))) * (__T__)0.125;\n"
    "    const __T__ r1 = (((ue[1]+ue[4])+(ue[7]+ue[10]))+((ue[13]+ue[16])+(ue[19]+ue[22]))) * (__T__)0.125;\n"
    "    const __T__ r2 = (((ue[2]+ue[5])+(ue[8]+ue[11]))+((ue[14]+ue[17])+(ue[20]+ue[23]))) * (__T__)0.125;")

_SRC_ELEM_COLORED = r"""
extern "C" __global__ void __N__(const __T__* __restrict__ KE,
    const __T__* __restrict__ E, const __T__* __restrict__ u,
    __T__* __restrict__ y, const int nx, const int ny, const int nz,
    const int cx, const int cy, const int cz)
{
    __shared__ __T__ Ks[576];
    for (int i = threadIdx.x; i < 576; i += blockDim.x) Ks[i] = KE[i];
    __syncthreads();
    const int hx = (nx - cx + 1) / 2, hy = (ny - cy + 1) / 2, hz = (nz - cz + 1) / 2;
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= hx * hy * hz) return;
    const int kz = q % hz, t = q / hz, ky = t % hy, kx = t / hy;
    const int ex = 2 * kx + cx, ey = 2 * ky + cy, ez = 2 * kz + cz;
""" + _ELEM_COMMON + r"""
    #pragma unroll
    for (int i = 0; i < 24; ++i) {
        __T__ acc = (__T__)0;
        #pragma unroll
        for (int j = 0; j < 24; ++j) acc += Ks[i * 24 + j] * ue[j];
        y[3 * nd[i / 3] + (i % 3)] += Ee * acc;
    }
}
"""

_NODE_HEAD = r"""
extern "C" __global__ void __N__(const __T__* __restrict__ KE,
    const __T__* __restrict__ E, const __T__* __restrict__ u,
    __T__* __restrict__ y, const int nx, const int ny, const int nz,
    const unsigned char* __restrict__ fixed)
{
    __shared__ __T__ Ks[576];
    for (int i = threadIdx.x; i < 576; i += blockDim.x) Ks[i] = KE[i];
    __syncthreads();
    const int nnz = nz + 1, nny = ny + 1;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= (nx + 1) * nny * nnz) return;
    const int iz = n % nnz, tn = n / nnz, iy = tn % nny, ix = tn / nny;
    const int sx = nny * nnz, sy = nnz;
    const int LUT[8] = {0, 4, 3, 7, 1, 5, 2, 6};
    __T__ acc0 = (__T__)0, acc1 = (__T__)0, acc2 = (__T__)0;
    #pragma unroll
    for (int a = 0; a < 2; ++a) {
     const int ex = ix - 1 + a;
     if (ex < 0 || ex >= nx) continue;
     #pragma unroll
     for (int b = 0; b < 2; ++b) {
      const int ey = iy - 1 + b;
      if (ey < 0 || ey >= ny) continue;
      #pragma unroll
      for (int d = 0; d < 2; ++d) {
       const int ez = iz - 1 + d;
       if (ez < 0 || ez >= nz) continue;
       const __T__ Ee = E[(ex * ny + ey) * nz + ez];
       const int li = LUT[(1 - a) * 4 + (1 - b) * 2 + (1 - d)];
       const int r0 = (3 * li) * 24, r1 = r0 + 24, r2 = r1 + 24;
       const int nb = ex * sx + ey * sy + ez;
       int en[8];
       en[0] = nb;           en[1] = nb + sx;
       en[2] = nb + sx + sy; en[3] = nb + sy;
       en[4] = nb + 1;       en[5] = nb + sx + 1;
       en[6] = nb + sx+sy+1; en[7] = nb + sy + 1;
"""

_NODE_TAIL = r"""
      }
     }
    }
    if (fixed != 0) {
        const unsigned char f = fixed[n];
        if (f & 1) acc0 = u[3 * n];
        if (f & 2) acc1 = u[3 * n + 1];
        if (f & 4) acc2 = u[3 * n + 2];
    }
    y[3 * n] = acc0; y[3 * n + 1] = acc1; y[3 * n + 2] = acc2;
}
"""

_SRC_NODE = _NODE_HEAD + r"""
       #pragma unroll
       for (int m = 0; m < 8; ++m) {
         const int dd = 3 * en[m];
         const __T__ u0 = u[dd], u1 = u[dd + 1], u2 = u[dd + 2];
         const int j = 3 * m;
         acc0 += Ee * (Ks[r0+j] * u0 + Ks[r0+j+1] * u1 + Ks[r0+j+2] * u2);
         acc1 += Ee * (Ks[r1+j] * u0 + Ks[r1+j+1] * u1 + Ks[r1+j+2] * u2);
         acc2 += Ee * (Ks[r2+j] * u0 + Ks[r2+j+1] * u1 + Ks[r2+j+2] * u2);
       }
""" + _NODE_TAIL

_SRC_NODE_HOIST = _NODE_HEAD + r"""
       __T__ s0 = (__T__)0, s1 = (__T__)0, s2 = (__T__)0;
       #pragma unroll
       for (int m = 0; m < 8; ++m) {
         const int dd = 3 * en[m];
         const __T__ u0 = u[dd], u1 = u[dd + 1], u2 = u[dd + 2];
         const int j = 3 * m;
         s0 = fma(Ks[r0+j], u0, s0); s0 = fma(Ks[r0+j+1], u1, s0); s0 = fma(Ks[r0+j+2], u2, s0);
         s1 = fma(Ks[r1+j], u0, s1); s1 = fma(Ks[r1+j+1], u1, s1); s1 = fma(Ks[r1+j+2], u2, s1);
         s2 = fma(Ks[r2+j], u0, s2); s2 = fma(Ks[r2+j+1], u1, s2); s2 = fma(Ks[r2+j+2], u2, s2);
       }
       acc0 = fma(Ee, s0, acc0); acc1 = fma(Ee, s1, acc1); acc2 = fma(Ee, s2, acc2);
""" + _NODE_TAIL

_SRC_DOF = r"""
extern "C" __global__ void __N__(const __T__* __restrict__ KE,
    const __T__* __restrict__ E, const __T__* __restrict__ u,
    __T__* __restrict__ y, const int nx, const int ny, const int nz,
    const unsigned char* __restrict__ fixed)
{
    __shared__ __T__ Ks[576];
    for (int i = threadIdx.x; i < 576; i += blockDim.x) Ks[i] = KE[i];
    __syncthreads();
    const int nnz = nz + 1, nny = ny + 1;
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= 3 * (nx + 1) * nny * nnz) return;
    const int c = q % 3, n = q / 3;
    const int z = n % nnz, t = n / nnz, iy = t % nny, ix = t / nny;
    const int lut[8] = {0, 4, 3, 7, 1, 5, 2, 6};
    const int ox[8] = {0, 1, 1, 0, 0, 1, 1, 0};
    const int oy[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    const int oz[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    __T__ acc = (__T__)0;
    #pragma unroll
    for (int a = 0; a < 2; a++)
    #pragma unroll
    for (int b = 0; b < 2; b++)
    #pragma unroll
    for (int d = 0; d < 2; d++) {
        const int ex = ix - 1 + a, ey = iy - 1 + b, ez = z - 1 + d;
        if (ex < 0 || ex >= nx || ey < 0 || ey >= ny || ez < 0 || ez >= nz) continue;
        const int li = lut[(1 - a) * 4 + (1 - b) * 2 + 1 - d], row = (3 * li + c) * 24;
        const __T__ ee = E[(ex * ny + ey) * nz + ez];
        #pragma unroll
        for (int m = 0; m < 8; m++) {
            const int j = 3 * ((ex + ox[m]) * nny * nnz + (ey + oy[m]) * nnz + ez + oz[m]);
            acc += ee * (Ks[row + 3 * m] * u[j] + Ks[row + 3 * m + 1] * u[j + 1]
                         + Ks[row + 3 * m + 2] * u[j + 2]);
        }
    }
    if (fixed != 0 && (fixed[n] & (1 << c))) acc = u[q];
    y[q] = acc;
}
"""

_SRC_TILE = r"""
extern "C" __global__ void __N__(const __T__* __restrict__ KE,
    const __T__* __restrict__ E, const __T__* __restrict__ u,
    __T__* __restrict__ y, const int nx, const int ny, const int nz,
    const unsigned char* __restrict__ fixed)
{
    __shared__ __T__ Ks[576];
    extern __shared__ unsigned char raw[];
    __T__* us = (__T__*)raw;
    const int B = blockDim.x, nnz = nz + 1, nny = ny + 1;
    const int tiles = (nnz + B - 1) / B;
    const int col = blockIdx.x / tiles, z0 = (blockIdx.x % tiles) * B;
    const int iy = col % nny, ix = col / nny;
    for (int i = threadIdx.x; i < 576; i += B) Ks[i] = KE[i];
    for (int i = threadIdx.x; i < 9 * (B + 2) * 3; i += B) {
        const int c = i % 3, t = i / 3, k = t % (B + 2), p = t / (B + 2);
        const int x = ix + p / 3 - 1, yy = iy + p % 3 - 1, z = z0 + k - 1;
        us[i] = (x >= 0 && x <= nx && yy >= 0 && yy <= ny && z >= 0 && z <= nz)
                ? u[3 * ((x * nny + yy) * nnz + z) + c] : (__T__)0;
    }
    __syncthreads();
    const int z = z0 + threadIdx.x;
    if (z > nz) return;
    const int lut[8] = {0, 4, 3, 7, 1, 5, 2, 6};
    const int ox[8] = {0, 1, 1, 0, 0, 1, 1, 0};
    const int oy[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    const int oz[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    __T__ s0 = 0, s1 = 0, s2 = 0;
    #pragma unroll
    for (int a = 0; a < 2; a++)
    #pragma unroll
    for (int b = 0; b < 2; b++)
    #pragma unroll
    for (int d = 0; d < 2; d++) {
        const int ex = ix - 1 + a, ey = iy - 1 + b, ez = z - 1 + d;
        if (ex < 0 || ex >= nx || ey < 0 || ey >= ny || ez < 0 || ez >= nz) continue;
        const int li = lut[(1 - a) * 4 + (1 - b) * 2 + 1 - d], r0 = 3 * li * 24, r1 = r0 + 24, r2 = r1 + 24;
        const __T__ ee = E[(ex * ny + ey) * nz + ez];
        #pragma unroll
        for (int m = 0; m < 8; m++) {
            const int j = 3 * (((a + ox[m]) * 3 + b + oy[m]) * (B + 2) + threadIdx.x + d + oz[m]);
            const __T__ u0 = us[j], u1 = us[j + 1], u2 = us[j + 2];
            s0 += ee * (Ks[r0 + 3*m] * u0 + Ks[r0 + 3*m + 1] * u1 + Ks[r0 + 3*m + 2] * u2);
            s1 += ee * (Ks[r1 + 3*m] * u0 + Ks[r1 + 3*m + 1] * u1 + Ks[r1 + 3*m + 2] * u2);
            s2 += ee * (Ks[r2 + 3*m] * u0 + Ks[r2 + 3*m + 1] * u1 + Ks[r2 + 3*m + 2] * u2);
        }
    }
    const int n = (ix * nny + iy) * nnz + z;
    if (fixed != 0) {
        const unsigned char f = fixed[n];
        if (f & 1) s0 = u[3 * n];
        if (f & 2) s1 = u[3 * n + 1];
        if (f & 4) s2 = u[3 * n + 2];
    }
    y[3 * n] = s0; y[3 * n + 1] = s1; y[3 * n + 2] = s2;
}
"""

_SRC_MASK = r"""
extern "C" __global__ void __N__(const __T__* __restrict__ u, __T__* __restrict__ y,
    const unsigned char* __restrict__ fixed, const int n_node)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= n_node) return;
    const unsigned char f = fixed[n];
    if (f == 0) return;
    if (f & 1) y[3 * n] = u[3 * n];
    if (f & 2) y[3 * n + 1] = u[3 * n + 1];
    if (f & 4) y[3 * n + 2] = u[3 * n + 2];
}
"""

# Jacobi diagonal, node-owned, analytic indexing; fixed DOFs get 1.
_SRC_DIAG = r"""
extern "C" __global__ void __N__(const __T__* __restrict__ KE,
    const __T__* __restrict__ E, __T__* __restrict__ diag,
    const int nx, const int ny, const int nz, const unsigned char* __restrict__ fixed)
{
    const int nnz = nz + 1, nny = ny + 1;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= (nx + 1) * nny * nnz) return;
    const int iz = n % nnz, tn = n / nnz, iy = tn % nny, ix = tn / nny;
    const int LUT[8] = {0, 4, 3, 7, 1, 5, 2, 6};
    __T__ d0 = 0, d1 = 0, d2 = 0;
    for (int a = 0; a < 2; ++a) { const int ex = ix - 1 + a; if (ex < 0 || ex >= nx) continue;
    for (int b = 0; b < 2; ++b) { const int ey = iy - 1 + b; if (ey < 0 || ey >= ny) continue;
    for (int d = 0; d < 2; ++d) { const int ez = iz - 1 + d; if (ez < 0 || ez >= nz) continue;
        const __T__ Ee = E[(ex * ny + ey) * nz + ez];
        const int li = LUT[(1 - a) * 4 + (1 - b) * 2 + (1 - d)];
        d0 += Ee * KE[(3*li) * 24 + 3*li];
        d1 += Ee * KE[(3*li+1) * 24 + 3*li+1];
        d2 += Ee * KE[(3*li+2) * 24 + 3*li+2];
    }}}
    if (fixed != 0) {
        const unsigned char f = fixed[n];
        if (f & 1) d0 = 1; if (f & 2) d1 = 1; if (f & 4) d2 = 1;
    }
    diag[3*n] = d0; diag[3*n+1] = d1; diag[3*n+2] = d2;
}
"""

# element strain energy c_e = u_e^T K_unit u_e (unit modulus), analytic index
_SRC_ELEM_ENERGY = r"""
extern "C" __global__ void __N__(const __T__* __restrict__ KE,
    const __T__* __restrict__ u, __T__* __restrict__ ce,
    const int nx, const int ny, const int nz)
{
    __shared__ __T__ Ks[576];
    for (int i = threadIdx.x; i < 576; i += blockDim.x) Ks[i] = KE[i];
    __syncthreads();
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nx * ny * nz) return;
    const int ez = e % nz, t = e / nz, ey = t % ny, ex = t / ny;
    const int sy = nz + 1, sx = (ny + 1) * (nz + 1);
    const int nb = ex * sx + ey * sy + ez;
    int nd[8];
    nd[0] = nb;           nd[1] = nb + sx;
    nd[2] = nb + sx + sy; nd[3] = nb + sy;
    nd[4] = nb + 1;       nd[5] = nb + sx + 1;
    nd[6] = nb + sx+sy+1; nd[7] = nb + sy + 1;
    __T__ ue[24];
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        const int d = 3 * nd[m];
        ue[3*m] = u[d]; ue[3*m+1] = u[d+1]; ue[3*m+2] = u[d+2];
    }
    __T__ acc = 0;
    #pragma unroll
    for (int i = 0; i < 24; ++i) {
        __T__ row = 0;
        #pragma unroll
        for (int j = 0; j < 24; ++j) row += Ks[i * 24 + j] * ue[j];
        acc += row * ue[i];
    }
    ce[e] = acc;
}
"""


def _ctype(dtype):
    return "float" if np.dtype(dtype) == np.float32 else "double"


def _inst(src, name, dtype, edtype=None):
    """Instantiate a kernel template; the moduli array E may have its own
    precision (edtype), e.g. FP64 arithmetic on FP32-stored moduli."""
    src = src.replace("const __T__* __restrict__ E,", "const __E__* __restrict__ E,")
    return (src.replace("__N__", name).replace("__E__", _ctype(edtype or dtype))
            .replace("__T__", _ctype(dtype)))


# -------------------------------------------------------------------------
# generated 27-neighbour kernels (spoke / stencil)
# -------------------------------------------------------------------------

def stencil_weights(ke, dtype=np.float64):
    """For each neighbour offset r, w[r][:, :, a] is the 3x3 block that
    incident-element slot a contributes, so that the block coupling node n to
    node n+r equals sum_a E_{e(a)} * w[r][:, :, a]."""
    weights = {r: np.zeros((3, 3, 8), dtype=dtype) for r in product([-1, 0, 1], repeat=3)}
    k = np.asarray(ke, dtype=np.float64)
    for a, oa in enumerate(OFFSETS):
        for b, ob in enumerate(OFFSETS):
            rel = tuple(int(x) for x in (ob - oa))
            weights[rel][:, :, a] = k[3 * a:3 * a + 3, 3 * b:3 * b + 3]
    return weights


def _linear_expr(values, lit):
    groups = {}
    for i, v in enumerate(values):
        if v != 0:
            groups.setdefault(float(v), []).append("e" + str(i))
    terms = [f"{lit(v)}*(" + "+".join(names) + ")" for v, names in groups.items()]
    return "+".join(terms) or "(T)0"


def generate_27(ke, dtype, mode="stencil", name="gen27", edtype=None, coef32=False):
    """mode 'stencil' or 'spoke'; 'stencil_diff' is the stencil in difference
    form, y_n = sum_{r != 0} W_r (u_{n+r} - u_n): the self block is replaced by
    minus the sum of the other blocks, which is exact for the unconstrained
    operator (its blocks sum to zero because K_unit annihilates rigid
    translations) and keeps that property after the weights are rounded.
    'stencil_selfcorr' (review comparator) forms the self block as minus the sum
    of the rounded neighbour blocks but applies it in the standard order,
    y_n = W_0 u_n + sum_{r != 0} W_r u_{n+r}, so the coefficients have zero row
    sums up to the rounding of that sum while the application still cancels.
    coef32 (with dtype float64): the block coefficients are formed in FP32 exactly
    as the FP32 kernel forms them, then applied in FP64 arithmetic; this isolates
    coefficient rounding from rounding in the application."""
    diff = mode == "stencil_diff"
    selfc = mode == "stencil_selfcorr"
    ctype = _ctype(dtype)
    cctype = "float" if coef32 else ctype
    lit = (lambda v: f"{v:.9e}f") if cctype == "float" else (lambda v: f"{v:.17e}")
    w = stencil_weights(ke)
    new = coef32 or selfc          # existing variants keep their exact source text
    TCn = "TC" if new else "T"
    L = [f"typedef {ctype} T;", f"typedef {_ctype(edtype or dtype)} TE;"] + ([f"typedef {cctype} TC;"] if new else []) + [
         f'extern "C" __global__ void {name}(const TE* __restrict__ E, const T* __restrict__ u,',
         '    T* __restrict__ y, const int nx, const int ny, const int nz,',
         '    const unsigned char* __restrict__ fixed) {',
         'const int nnz = nz + 1, nny = ny + 1;',
         'const int n = blockIdx.x * blockDim.x + threadIdx.x;',
         'if (n >= (nx + 1) * nny * nnz) return;',
         'const int z = n % nnz, t = n / nnz, iy = t % nny, ix = t / nny;',
         'T acc0 = 0, acc1 = 0, acc2 = 0;']
    if diff or selfc:
        L.append("const T us0 = u[3 * n], us1 = u[3 * n + 1], us2 = u[3 * n + 2];")
    if selfc:
        L.append("TC s00 = 0, s01 = 0, s02 = 0, s10 = 0, s11 = 0, s12 = 0, s20 = 0, s21 = 0, s22 = 0;")
    # incident element slot a has node n at local position a; the element's
    # origin is n - OFFSETS[a]
    for i, (ox, oy, oz) in enumerate(OFFSETS):
        L.append(f"const int ex{i} = ix - {ox}, ey{i} = iy - {oy}, ez{i} = z - {oz};")
        if new:
            L.append(f"const TC e{i} = (ex{i} >= 0 && ex{i} < nx && ey{i} >= 0 && ey{i} < ny && "
                     f"ez{i} >= 0 && ez{i} < nz) ? (TC)E[(ex{i} * ny + ey{i}) * nz + ez{i}] : (TC)0;")
        else:
            L.append(f"const T e{i} = (ex{i} >= 0 && ex{i} < nx && ey{i} >= 0 && ey{i} < ny && "
                     f"ez{i} >= 0 && ez{i} < nz) ? E[(ex{i} * ny + ey{i}) * nz + ez{i}] : (T)0;")
    for (dx, dy, dz), wr in w.items():
        if (diff or selfc) and (dx, dy, dz) == (0, 0, 0):
            continue
        load = ("const T u0 = u[j] - us0, u1 = u[j + 1] - us1, u2 = u[j + 2] - us2;" if diff
                else "const T u0 = u[j], u1 = u[j + 1], u2 = u[j + 2];")
        L += ["{", f"const int x = ix + ({dx}), yy = iy + ({dy}), zz = z + ({dz});",
              "if (x >= 0 && x <= nx && yy >= 0 && yy <= ny && zz >= 0 && zz <= nz) {",
              "const int j = 3 * ((x * nny + yy) * nnz + zz);", load]
        if mode == "spoke":
            for a in range(8):
                if not np.any(wr[:, :, a]):
                    continue
                for r in range(3):
                    term = "+".join(f"{lit(float(wr[r, s, a]))}*u{s}" for s in range(3)
                                    if wr[r, s, a] != 0) or "(T)0"
                    L.append(f"acc{r} += e{a} * ({term});")
        else:
            names, coef = {}, {}
            for r in range(3):
                for s in range(3):
                    expr = _linear_expr(wr[r, s], lit)
                    if expr not in names:
                        nm = "w" + str(len(names))
                        names[expr] = nm
                        L.append(f"const {TCn} {nm} = {expr};")
                    coef[(r, s)] = names[expr]
            for r in range(3):
                terms = [f"{coef[(r, s)]}*u{s}" for s in range(3)]
                L.append(f"acc{r} += " + " + ".join(terms) + ";")
            if selfc:
                for r in range(3):
                    for s in range(3):
                        L.append(f"s{r}{s} -= {coef[(r, s)]};")
        L += ["}", "}"]
    if selfc:
        for r in range(3):
            L.append(f"acc{r} += s{r}0*us0 + s{r}1*us1 + s{r}2*us2;")
    L += ["if (fixed != 0) { const unsigned char f = fixed[n];",
          "  if (f & 1) acc0 = u[3 * n]; if (f & 2) acc1 = u[3 * n + 1]; if (f & 4) acc2 = u[3 * n + 2]; }",
          "y[3 * n] = acc0; y[3 * n + 1] = acc1; y[3 * n + 2] = acc2;", "}"]
    return "\n".join(L)


# -------------------------------------------------------------------------
# interface
# -------------------------------------------------------------------------

VARIANTS = ("elem_atomic", "elem_colored", "node", "node_hoist", "dof", "tile",
            "spoke", "stencil")

OWNERSHIP = {"elem_atomic": "element (atomic)", "elem_colored": "element (8 colours)",
             "node": "node", "node_hoist": "node", "dof": "DoF", "tile": "node",
             "spoke": "node (27-neighbour)", "stencil": "node (27-point stencil)",
             "stencil_diff": "node (27-point stencil, difference form)",
             "elem_ref": "element (atomic, nodal reference subtracted)",
             "elem_mean": "element (atomic, element mean subtracted)",
             "stencil_selfcorr": "node (27-point stencil, self block = -sum of neighbours)",
             "stencil_c32": "node (27-point stencil, FP32 coefficients, FP64 application)",
             "stencil_diff_c32": "node (difference form, FP32 coefficients, FP64 application)"}

# variants outside the operator survey: the difference-form stencil (precision control) and the review
# comparators: element-level difference form, self-block correction, and FP32-formed coefficients applied in FP64
EXTRA_VARIANTS = ("stencil_diff", "elem_ref", "elem_mean", "stencil_selfcorr", "stencil_c32", "stencil_diff_c32")

_SOURCES = {}            # (variant, dtype) -> (source, kernel name); for SASS analysis


def kernel_source(variant, dtype, ke, edtype=None):
    dt = np.dtype(dtype)
    edt = np.dtype(edtype or dtype)
    sfx = ("fp32" if dt == np.float32 else "fp64") + ("" if edt == dt else "_e" + edt.name[-2:])
    nm = f"{variant}_{sfx}"
    if variant in ("spoke", "stencil", "stencil_diff", "stencil_selfcorr"):
        return generate_27(ke, dt, variant, nm, edt), nm
    if variant in ("stencil_c32", "stencil_diff_c32"):
        assert dt == np.float64, "coefficient-level controls apply FP32 coefficients in FP64"
        return generate_27(ke, dt, variant[:-4], nm, np.dtype(np.float32), coef32=True), nm
    src = {"elem_atomic": _SRC_ELEM_ATOMIC, "elem_colored": _SRC_ELEM_COLORED, "elem_ref": _SRC_ELEM_REF, "elem_mean": _SRC_ELEM_MEAN,
           "node": _SRC_NODE, "node_hoist": _SRC_NODE_HOIST, "dof": _SRC_DOF,
           "tile": _SRC_TILE}[variant]
    return _inst(src, nm, dt, edt), nm


_MODULE_CACHE = {}


def _get_kernel(src, name):
    key = (name, hash(src))
    if key not in _MODULE_CACHE:
        mod = cp.RawModule(code=src, options=_OPTS)
        _MODULE_CACHE[key] = mod.get_function(name)
    return _MODULE_CACHE[key]


class Operator:
    """y = K(E) v for one mapping and one precision on one grid."""

    def __init__(self, grid: Grid, ke, variant: str, dtype=np.float32, block=128, e_dtype=None):
        if variant not in VARIANTS + EXTRA_VARIANTS:
            raise ValueError(variant)
        self.grid, self.variant = grid, variant
        self.dtype = np.dtype(dtype)
        self.e_dtype = np.dtype(e_dtype or dtype)
        self.block = int(block if variant != "tile" else min(block, 64))
        self.ke_host = np.asarray(ke, dtype=np.float64)
        self.KE = cp.asarray(self.ke_host, dtype=self.dtype)
        src, nm = kernel_source(variant, self.dtype, self.ke_host, self.e_dtype)
        self.source, self.kname = src, nm
        self.kernel = _get_kernel(src, nm)
        self._mask_k = _get_kernel(_inst(_SRC_MASK, "mask_" + _ctype(self.dtype), self.dtype),
                                   "mask_" + _ctype(self.dtype))
        self.atomic = variant in ("elem_atomic", "elem_ref", "elem_mean")
        self.needs_zero = variant in ("elem_atomic", "elem_colored", "elem_ref", "elem_mean")
        self.deterministic = not self.atomic

    def apply(self, u, E, y=None, fixed=None):
        g = self.grid
        nx, ny, nz = g.shape
        if y is None:
            y = cp.empty(g.ndof, dtype=self.dtype)
        B = self.block
        fx = fixed if fixed is not None else np.uintp(0)
        if self.variant in ("elem_atomic", "elem_ref", "elem_mean"):
            y.fill(0)
            self.kernel(((g.n_elem + B - 1) // B,), (B,),
                        (self.KE, E, u, y, np.int32(nx), np.int32(ny), np.int32(nz)))
        elif self.variant == "elem_colored":
            y.fill(0)
            for cx, cy, cz in product((0, 1), repeat=3):
                h = ((nx - cx + 1) // 2) * ((ny - cy + 1) // 2) * ((nz - cz + 1) // 2)
                if h == 0:
                    continue
                self.kernel(((h + B - 1) // B,), (B,),
                            (self.KE, E, u, y, np.int32(nx), np.int32(ny), np.int32(nz),
                             np.int32(cx), np.int32(cy), np.int32(cz)))
        elif self.variant == "dof":
            self.kernel(((g.ndof + B - 1) // B,), (B,),
                        (self.KE, E, u, y, np.int32(nx), np.int32(ny), np.int32(nz), fx))
        elif self.variant == "tile":
            tiles = (nz + 1 + B - 1) // B
            grid = (nx + 1) * (ny + 1) * tiles
            smem = 9 * (B + 2) * 3 * self.dtype.itemsize
            self.kernel((grid,), (B,),
                        (self.KE, E, u, y, np.int32(nx), np.int32(ny), np.int32(nz), fx),
                        shared_mem=smem)
        elif self.variant in ("spoke", "stencil", "stencil_diff", "stencil_selfcorr", "stencil_c32", "stencil_diff_c32"):
            self.kernel(((g.n_node + B - 1) // B,), (B,),
                        (E, u, y, np.int32(nx), np.int32(ny), np.int32(nz), fx))
        else:  # node, node_hoist
            self.kernel(((g.n_node + B - 1) // B,), (B,),
                        (self.KE, E, u, y, np.int32(nx), np.int32(ny), np.int32(nz), fx))
        if fixed is not None and self.needs_zero:
            self._mask_k(((g.n_node + 255) // 256,), (256,), (u, y, fixed, np.int32(g.n_node)))
        return y


class MixedOperator:
    """q32 = (float)(K(E32) (double)p32): the node-owned factored kernel with
    FP64 element matrix and FP64 arithmetic, reading and writing FP32 vectors.
    This is the operator organisation of a solver that keeps its Krylov
    vectors in FP32 but evaluates the element products in FP64 (the
    published OpenMP GPU code); used only by the precision controls."""

    def __init__(self, grid: Grid, ke, block=128):
        self.grid, self.block = grid, int(block)
        src = (_SRC_NODE_HOIST.replace("const __T__* __restrict__ u,", "const float* __restrict__ u,")
               .replace("__T__* __restrict__ y,", "float* __restrict__ y,"))
        nm = "node_hoist_mixed64"
        self.source = _inst(src, nm, np.float64, np.float32)
        self.kernel = _get_kernel(self.source, nm)
        self.KE = cp.asarray(np.asarray(ke, dtype=np.float64))
        self.dtype = np.dtype(np.float32)

    def apply(self, u, E, y=None, fixed=None):
        g = self.grid
        if y is None:
            y = cp.empty(g.ndof, dtype=cp.float32)
        fx = fixed if fixed is not None else np.uintp(0)
        B = self.block
        self.kernel(((g.n_node + B - 1) // B,), (B,),
                    (self.KE, E, u, y, np.int32(g.nx), np.int32(g.ny), np.int32(g.nz), fx))
        return y


class Diagonal:
    def __init__(self, grid: Grid, ke, dtype=np.float64, e_dtype=None):
        self.grid, self.dtype = grid, np.dtype(dtype)
        edt = np.dtype(e_dtype or dtype)
        self.KE = cp.asarray(ke, dtype=self.dtype)
        nm = "diag_" + _ctype(self.dtype) + "_e" + _ctype(edt)
        self.kernel = _get_kernel(_inst(_SRC_DIAG, nm, self.dtype, edt), nm)

    def __call__(self, E, fixed=None, out=None):
        g = self.grid
        if out is None:
            out = cp.empty(g.ndof, dtype=self.dtype)
        fx = fixed if fixed is not None else np.uintp(0)
        self.kernel(((g.n_node + 127) // 128,), (128,),
                    (self.KE, E, out, np.int32(g.nx), np.int32(g.ny), np.int32(g.nz), fx))
        return out


class ElementEnergy:
    """c_e = u_e^T K_unit u_e for every element (fused, no temporaries)."""

    def __init__(self, grid: Grid, ke, dtype=np.float64):
        self.grid, self.dtype = grid, np.dtype(dtype)
        self.KE = cp.asarray(ke, dtype=self.dtype)
        nm = "energy_" + _ctype(self.dtype)
        self.kernel = _get_kernel(_inst(_SRC_ELEM_ENERGY, nm, self.dtype), nm)

    def __call__(self, u, out=None):
        g = self.grid
        if out is None:
            out = cp.empty(g.n_elem, dtype=self.dtype)
        self.kernel(((g.n_elem + 127) // 128,), (128,),
                    (self.KE, u, out, np.int32(g.nx), np.int32(g.ny), np.int32(g.nz)))
        return out


_SRC_RESIDUAL = r"""
extern "C" __global__ void vox_residual32(const double* __restrict__ KE,
    const float* __restrict__ E, const double* __restrict__ u,
    float* __restrict__ r, const int nx, const int ny, const int nz,
    const unsigned char* __restrict__ fixed,
    const int* __restrict__ list, const double* __restrict__ bl, const int count)
{
    // r = (float)(b - K u) computed in FP64 (node-owned, factored modulus).
    // Full pass: list == 0, b taken as zero.  Fix-up pass: list holds the
    // loaded nodes and bl their 3 load components, so loaded DOFs are formed
    // as b - Ku in FP64 before rounding.
    __shared__ double Ks[576];
    for (int i = threadIdx.x; i < 576; i += blockDim.x) Ks[i] = KE[i];
    __syncthreads();
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const int n = list ? list[t] : t;
    const int nnz = nz + 1, nny = ny + 1;
    const int iz = n % nnz, tn = n / nnz, iy = tn % nny, ix = tn / nny;
    const int sx = nny * nnz, sy = nnz;
    const int LUT[8] = {0, 4, 3, 7, 1, 5, 2, 6};
    double acc0 = 0, acc1 = 0, acc2 = 0;
    #pragma unroll
    for (int a = 0; a < 2; ++a) {
     const int ex = ix - 1 + a;
     if (ex < 0 || ex >= nx) continue;
     #pragma unroll
     for (int b = 0; b < 2; ++b) {
      const int ey = iy - 1 + b;
      if (ey < 0 || ey >= ny) continue;
      #pragma unroll
      for (int d = 0; d < 2; ++d) {
       const int ez = iz - 1 + d;
       if (ez < 0 || ez >= nz) continue;
       const double Ee = (double)E[(ex * ny + ey) * nz + ez];
       const int li = LUT[(1 - a) * 4 + (1 - b) * 2 + (1 - d)];
       const int r0 = (3 * li) * 24, r1 = r0 + 24, r2 = r1 + 24;
       const int nb = ex * sx + ey * sy + ez;
       int en[8];
       en[0] = nb;           en[1] = nb + sx;
       en[2] = nb + sx + sy; en[3] = nb + sy;
       en[4] = nb + 1;       en[5] = nb + sx + 1;
       en[6] = nb + sx+sy+1; en[7] = nb + sy + 1;
       double s0 = 0, s1 = 0, s2 = 0;
       #pragma unroll
       for (int m = 0; m < 8; ++m) {
         const int dd = 3 * en[m];
         const double u0 = u[dd], u1 = u[dd + 1], u2 = u[dd + 2];
         const int j = 3 * m;
         s0 = fma(Ks[r0+j], u0, s0); s0 = fma(Ks[r0+j+1], u1, s0); s0 = fma(Ks[r0+j+2], u2, s0);
         s1 = fma(Ks[r1+j], u0, s1); s1 = fma(Ks[r1+j+1], u1, s1); s1 = fma(Ks[r1+j+2], u2, s1);
         s2 = fma(Ks[r2+j], u0, s2); s2 = fma(Ks[r2+j+1], u1, s2); s2 = fma(Ks[r2+j+2], u2, s2);
       }
       acc0 = fma(Ee, s0, acc0); acc1 = fma(Ee, s1, acc1); acc2 = fma(Ee, s2, acc2);
      }
     }
    }
    double b0 = 0, b1 = 0, b2 = 0;
    if (bl) { b0 = bl[3 * t]; b1 = bl[3 * t + 1]; b2 = bl[3 * t + 2]; }
    const unsigned char f = fixed ? fixed[n] : 0;
    r[3 * n]     = (f & 1) ? 0.f : (float)(b0 - acc0);
    r[3 * n + 1] = (f & 2) ? 0.f : (float)(b1 - acc1);
    r[3 * n + 2] = (f & 4) ? 0.f : (float)(b2 - acc2);
}
"""

_nrm2_f32 = cp.ReductionKernel("float32 x", "float64 y", "(double)x * (double)x", "a + b",
                               "y = sqrt(a)", "0", "vox_nrm2_f32_f64")


class Residual32:
    """r32 = (float)(b - K(E32) u64) with FP64 arithmetic, no FP64 residual
    vector: one full node-owned pass (b = 0) plus a fix-up pass over the
    loaded nodes that forms b - Ku in FP64 before rounding.  norm() returns
    ||r32|| accumulated in FP64."""

    def __init__(self, grid: Grid, ke, load_idx, load_val):
        self.grid = grid
        self.KE = cp.asarray(ke, dtype=cp.float64)
        self.k = _get_kernel(_SRC_RESIDUAL, "vox_residual32")
        idx = np.asarray(load_idx, dtype=np.int64)
        val = np.asarray(load_val, dtype=np.float64)
        nodes = np.unique(idx // 3)
        bl = np.zeros((nodes.size, 3))
        pos = {n: i for i, n in enumerate(nodes)}
        for i, v in zip(idx, val):
            bl[pos[i // 3], i % 3] += v
        self.nodes = cp.asarray(nodes, dtype=cp.int32)
        self.bl = cp.asarray(bl.ravel())

    def __call__(self, E32, u, r32, fixed):
        g = self.grid
        args = (np.int32(g.nx), np.int32(g.ny), np.int32(g.nz), fixed)
        self.k(((g.n_node + 127) // 128,), (128,),
               (self.KE, E32, u, r32, *args, np.uintp(0), np.uintp(0), np.int32(g.n_node)))
        m = int(self.nodes.size)
        self.k(((m + 127) // 128,), (128,), (self.KE, E32, u, r32, *args, self.nodes, self.bl, np.int32(m)))
        return r32

    @staticmethod
    def norm(r32):
        return float(_nrm2_f32(r32))


def reference_apply(grid: Grid, ke, E, u):
    """Independent CPU reference: assembled scipy CSR from the edof table."""
    import scipy.sparse as sp
    edof = grid.edof()
    ke = np.asarray(ke, dtype=np.float64)
    rows = np.repeat(edof, 24, axis=1).ravel()
    cols = np.tile(edof, (1, 24)).ravel()
    vals = (np.asarray(E, dtype=np.float64)[:, None, None] * ke[None]).ravel()
    K = sp.coo_matrix((vals, (rows, cols)), shape=(grid.ndof, grid.ndof)).tocsr()
    return K @ np.asarray(u, dtype=np.float64), K
