"""Assembled cuSPARSE baselines for the same operator.

Two storage formats of the identical matrix (built from the same element
matrix and moduli, boundary rows compacted, columns sorted):

  CSR   scalar compressed rows, applied by cuSPARSE SpMV through CuPy
        (cupyx.scipy.sparse.csr_matrix @ x).
  BSR   3x3 block rows (one block per coupled node pair), applied by the
        cuSPARSE legacy bsrmv routine called through ctypes on the same
        library instance CuPy has loaded.

Both expose apply(u), refresh(E) (re-assemble values for new moduli on the
fixed pattern, the per-design-iteration cost of an assembled method) and
storage_bytes.

refresh(E) is one fused kernel (vox_assemble): one thread per (node,
neighbour) pair forms its 3x3 block from the eight incident moduli and writes
it in place, at the position given by the boundary-clipped neighbour rank, so
no temporaries are allocated; the pattern build uses the same kernel to write
the column indices.  refresh_cupy(E) is the earlier array-level CuPy
assembly (dense block array + boolean compaction), kept as a cross-check and
as the "generic implementation" reference; it needs several times the
matrix's memory.
"""
from __future__ import annotations

import ctypes
from itertools import product

import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp

from .core import OFFSETS, Grid
from .operators import stencil_weights

_OFFS27 = np.array(list(product([-1, 0, 1], repeat=3)), dtype=np.int64)

_SRC_ASSEMBLE = r"""
extern "C" __global__ void __N__(const __T__* __restrict__ W, const __T__* __restrict__ E,
    const int* __restrict__ start, __T__* __restrict__ val, int* __restrict__ idx,
    const int csr, const int nx, const int ny, const int nz)
{
    // W: (27, 3, 3, 8) block weights per neighbour offset r and incident slot a.
    // start: BSR row pointer (blocks) or, for CSR, the scalar indptr (block start = indptr[3n] / 9).
    const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const long long nn = (long long)(nx + 1) * (ny + 1) * (nz + 1);
    if (t >= nn * 27) return;
    const int n = (int)(t / 27), r = (int)(t % 27);
    const int iz = n % (nz + 1), iy = (n / (nz + 1)) % (ny + 1), ix = n / ((nz + 1) * (ny + 1));
    const int dx = r / 9 - 1, dy = (r / 3) % 3 - 1, dz = r % 3 - 1;
    const int lox = ix > 0 ? -1 : 0, hix = ix < nx ? 1 : 0;
    const int loy = iy > 0 ? -1 : 0, hiy = iy < ny ? 1 : 0;
    const int loz = iz > 0 ? -1 : 0, hiz = iz < nz ? 1 : 0;
    if (dx < lox || dx > hix || dy < loy || dy > hiy || dz < loz || dz > hiz) return;
    const int cy = hiy - loy + 1, cz = hiz - loz + 1, nv = (hix - lox + 1) * cy * cz;
    const int rank = ((dx - lox) * cy + (dy - loy)) * cz + (dz - loz);
    const int ox[8] = {0, 1, 1, 0, 0, 1, 1, 0}, oy[8] = {0, 0, 1, 1, 0, 0, 1, 1}, oz[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    __T__ S[8];
    #pragma unroll
    for (int a = 0; a < 8; a++) {
        const int ex = ix - ox[a], ey = iy - oy[a], ez = iz - oz[a];
        S[a] = (ex >= 0 && ex < nx && ey >= 0 && ey < ny && ez >= 0 && ez < nz)
             ? E[((long long)ex * ny + ey) * nz + ez] : (__T__)0;
    }
    const __T__* w = W + r * 72;
    __T__ b[9];
    #pragma unroll
    for (int q = 0; q < 9; q++) {
        __T__ acc = 0;
        #pragma unroll
        for (int a = 0; a < 8; a++) acc += w[q * 8 + a] * S[a];
        b[q] = acc;
    }
    const int m = ((ix + dx) * (ny + 1) + (iy + dy)) * (nz + 1) + (iz + dz);
    if (!csr) {
        const long long j = (long long)start[n] + rank;
        #pragma unroll
        for (int q = 0; q < 9; q++) val[j * 9 + q] = b[q];
        if (idx) idx[j] = m;
    } else {
        const long long base = (long long)start[3LL * n];
        #pragma unroll
        for (int s = 0; s < 3; s++) {
            #pragma unroll
            for (int c = 0; c < 3; c++) {
                const long long p = base + (long long)s * 3 * nv + rank * 3 + c;
                val[p] = b[s * 3 + c];
                if (idx) idx[p] = 3 * m + c;
            }
        }
    }
}
"""
_ASM_CACHE = {}


def _assemble_kernel(dtype):
    nm = "vox_assemble_" + ("f32" if np.dtype(dtype) == np.float32 else "f64")
    if nm not in _ASM_CACHE:
        ct = "float" if np.dtype(dtype) == np.float32 else "double"
        _ASM_CACHE[nm] = cp.RawModule(code=_SRC_ASSEMBLE.replace("__N__", nm).replace("__T__", ct),
                                      options=("-std=c++14",)).get_function(nm)
    return _ASM_CACHE[nm]


def _block_weights(ke, dtype):
    W = stencil_weights(ke)
    return cp.asarray(np.stack([W[tuple(int(x) for x in r)] for r in _OFFS27]).ravel(), dtype=dtype)


def _lean_rowptr(grid: Grid):
    """Blocks per node row (boundary-clipped 27-neighbourhood) and the BSR row
    pointer, without the (n_node, 27) neighbour table."""
    nx, ny, nz = grid.shape
    n = cp.arange(grid.n_node, dtype=cp.int64)
    iz = n % (nz + 1); iy = (n // (nz + 1)) % (ny + 1); ix = n // ((nz + 1) * (ny + 1))
    del n
    c = lambda i, m: (1 + (i > 0) + (i < m)).astype(cp.int32)
    nv = c(ix, nx) * c(iy, ny) * c(iz, nz)
    del ix, iy, iz
    rp = cp.concatenate([cp.zeros(1, cp.int64), cp.cumsum(nv, dtype=cp.int64)])
    if int(rp[-1]) * 9 >= 2**31:
        raise MemoryError("matrix exceeds 32-bit CSR indexing")
    return nv, rp.astype(cp.int32)


def _launch_assemble(grid: Grid, W, E, start, val, idx, csr):
    nx, ny, nz = grid.shape
    total = grid.n_node * 27
    _assemble_kernel(val.dtype)(((total + 255) // 256,), (256,),
                                (W, E, start, val, idx if idx is not None else np.intp(0),
                                 np.int32(csr), np.int32(nx), np.int32(ny), np.int32(nz)))


def _slot_moduli(grid: Grid, E):
    """(8, n_node) moduli of the element in incident slot a of every node
    (zero outside the grid)."""
    nx, ny, nz = grid.shape
    Ep = cp.zeros((nx + 2, ny + 2, nz + 2), dtype=E.dtype)
    Ep[1:-1, 1:-1, 1:-1] = E.reshape(grid.shape)
    out = cp.empty((8, nx + 1, ny + 1, nz + 1), dtype=E.dtype)
    for a, (ox, oy, oz) in enumerate(OFFSETS):
        # element origin = node - offset; padded index = origin + 1
        out[a] = Ep[1 - ox:nx + 2 - ox, 1 - oy:ny + 2 - oy, 1 - oz:nz + 2 - oz]
    return out.reshape(8, -1)


def _pattern(grid: Grid):
    """Neighbour node ids (n_node, 27) and validity mask."""
    nx, ny, nz = grid.shape
    ix, iy, iz = cp.meshgrid(cp.arange(nx + 1), cp.arange(ny + 1), cp.arange(nz + 1), indexing="ij")
    ix, iy, iz = ix.ravel(), iy.ravel(), iz.ravel()
    offs = cp.asarray(_OFFS27)
    jx = ix[:, None] + offs[None, :, 0]
    jy = iy[:, None] + offs[None, :, 1]
    jz = iz[:, None] + offs[None, :, 2]
    valid = (jx >= 0) & (jx <= nx) & (jy >= 0) & (jy <= ny) & (jz >= 0) & (jz <= nz)
    nbr = (jx * (ny + 1) + jy) * (nz + 1) + jz
    return nbr.astype(cp.int32), valid


def _blocks(grid: Grid, ke, E):
    """(n_node, 27, 3, 3) block values in the precision of E."""
    W = stencil_weights(ke)
    Wm = np.stack([W[tuple(int(x) for x in r)] for r in _OFFS27])      # (27,3,3,8)
    Wg = cp.asarray(Wm.reshape(27 * 9, 8), dtype=E.dtype)
    S = _slot_moduli(grid, E)                                            # (8, n_node)
    return (Wg @ S).T.reshape(-1, 27, 3, 3)


class CSRBaseline:
    def __init__(self, grid: Grid, ke, E):
        self.grid, self.ke = grid, np.asarray(ke)
        self.dtype = E.dtype
        nv, rp = _lean_rowptr(grid)                                      # blocks per node row
        # row 3n+s starts at 9*rp[n] + s*3*nv[n]; entries: valid neighbours r (sorted), then t
        ip = 9 * rp[:-1].astype(cp.int64)[:, None] + cp.arange(3, dtype=cp.int64)[None, :] * (3 * nv[:, None])
        self.indptr = cp.concatenate([ip.ravel(), 9 * rp[-1:].astype(cp.int64)]).astype(cp.int32)
        del ip, nv, rp
        nnz = int(self.indptr[-1])
        self.indices = cp.empty(nnz, dtype=cp.int32)
        self.data = cp.empty(nnz, dtype=self.dtype)
        self._W = _block_weights(ke, self.dtype)
        _launch_assemble(grid, self._W, E, self.indptr, self.data, self.indices, 1)
        self.matrix = csp.csr_matrix((self.data, self.indices, self.indptr),
                                     shape=(grid.ndof, grid.ndof))
        self.matrix.has_canonical_format = True

    def refresh(self, E):
        """Fused in-place value refresh on the fixed pattern."""
        _launch_assemble(self.grid, self._W, E, self.indptr, self.data, None, 1)
        return self

    def refresh_cupy(self, E):
        """Array-level CuPy assembly (returns the values; does not touch the matrix)."""
        nbr, valid = _pattern(self.grid)
        del nbr
        B = _blocks(self.grid, self.ke, E)                               # (n,27,3,3) [r][s,t]
        Brow = B.transpose(0, 2, 1, 3)                                   # (n,3,27,3) row s
        vrow = cp.broadcast_to(valid[:, None, :, None], (self.grid.n_node, 3, 27, 3))
        return Brow[vrow]

    def apply(self, u):
        return self.matrix @ u

    @property
    def nnz(self):
        return int(self.indices.size)

    @property
    def storage_bytes(self):
        return int(self.data.nbytes + self.indices.nbytes + self.indptr.nbytes)


# ---- cuSPARSE bsrmv through ctypes ------------------------------------------

def _cusparse_lib():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetModuleHandleW.restype = ctypes.c_void_p
    k32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    k32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    for name in ("cusparse64_12.dll", "cusparse64_11.dll"):
        h = k32.GetModuleHandleW(name)
        if h:
            buf = ctypes.create_unicode_buffer(1024)
            k32.GetModuleFileNameW(h, buf, 1024)
            return ctypes.WinDLL(buf.value), buf.value
    raise RuntimeError("cuSPARSE is not loaded in this process")


class BSRBaseline:
    def __init__(self, grid: Grid, ke, E):
        from cupy_backends.cuda.libs import cusparse as _cs
        self.grid, self.ke, self.dtype = grid, np.asarray(ke), E.dtype
        cp.cuda.device.get_cusparse_handle()          # make CuPy load cuSPARSE
        self.lib, self.lib_path = _cusparse_lib()
        nv, self.rowptr = _lean_rowptr(grid)
        del nv
        nnzb = int(self.rowptr[-1])
        self.colind = cp.empty(nnzb, dtype=cp.int32)
        self.val = cp.empty((nnzb, 3, 3), dtype=self.dtype)
        self._W = _block_weights(ke, self.dtype)
        _launch_assemble(grid, self._W, E, self.rowptr, self.val, self.colind, 0)
        self.descr = _cs.createMatDescr()
        self.handle = cp.cuda.device.get_cusparse_handle()
        f = self.lib.cusparseSbsrmv if self.dtype == np.float32 else self.lib.cusparseDbsrmv
        f.restype = ctypes.c_int
        f.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                      ctypes.c_void_p, ctypes.c_void_p]
        self._f = f
        ct = ctypes.c_float if self.dtype == np.float32 else ctypes.c_double
        self._one, self._zero = ct(1.0), ct(0.0)

    def refresh(self, E):
        """Fused in-place value refresh on the fixed pattern."""
        _launch_assemble(self.grid, self._W, E, self.rowptr, self.val, None, 0)
        return self

    def refresh_cupy(self, E):
        """Array-level CuPy assembly (returns the blocks; does not touch the matrix)."""
        nbr, valid = _pattern(self.grid)
        del nbr
        B = _blocks(self.grid, self.ke, E)                               # (n,27,3,3) row-major blocks
        return B[valid]

    def apply(self, u, y=None):
        g = self.grid
        if y is None:
            y = cp.empty(g.ndof, dtype=self.dtype)
        st = self._f(ctypes.c_void_p(self.handle), 0, 0, g.n_node, g.n_node, int(self.colind.size),
                     ctypes.byref(self._one), ctypes.c_void_p(self.descr),
                     ctypes.c_void_p(self.val.data.ptr), ctypes.c_void_p(self.rowptr.data.ptr),
                     ctypes.c_void_p(self.colind.data.ptr), 3, ctypes.c_void_p(u.data.ptr),
                     ctypes.byref(self._zero), ctypes.c_void_p(y.data.ptr))
        if st != 0:
            raise RuntimeError(f"cusparse bsrmv status {st}")
        return y

    @property
    def nnzb(self):
        return int(self.colind.size)

    @property
    def storage_bytes(self):
        return int(self.val.nbytes + self.colind.nbytes + self.rowptr.nbytes)


def csr_bytes_estimate(grid: Grid, dtype):
    it = np.dtype(dtype).itemsize
    nnz = 81 * grid.ndof // 3 * 3          # upper bound 81 per scalar row
    return nnz * (it + 4) + (grid.ndof + 1) * 4


def bsr_bytes_estimate(grid: Grid, dtype):
    it = np.dtype(dtype).itemsize
    return 27 * grid.n_node * (9 * it + 4) + (grid.n_node + 1) * 4
