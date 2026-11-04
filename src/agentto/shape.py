"""Shape agreement between voxel designs (X1): IoU, surface Hausdorff distance and Betti numbers.

All functions take boolean solid masks (rho > 0.5) on the same grid; `pool` brings a fine design down to a common
grid first (mean over f^3 blocks, then threshold), so arms at different rungs can be compared.

Betti numbers of the solid as a union of closed unit cubes:
    b0  connected components (26-connectivity: cubes touching at a corner are connected)
    b2  enclosed cavities (6-connected components of the padded complement that do not reach the outside)
    b1  tunnels / handles, from the Euler characteristic chi = V - E + F - C of the cubical complex:
        b1 = b0 + b2 - chi
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def pool(rho: np.ndarray, f: int) -> np.ndarray:
    """Mean density over f x f x f blocks (shape must be divisible by f)."""
    if f == 1:
        return rho.astype(np.float32)
    a, b, c = (n // f for n in rho.shape)
    return rho.reshape(a, f, b, f, c, f).mean(axis=(1, 3, 5), dtype=np.float32)


def pool_factor(shape, max_elem: int = 9_000_000) -> int:
    f = 1
    while np.prod([n // f for n in shape]) > max_elem and all(n % (2 * f) == 0 for n in shape):
        f *= 2
    return f


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / u) if u else 1.0


def _surface(m: np.ndarray) -> np.ndarray:
    return m & ~ndimage.binary_erosion(m, border_value=0)


def hausdorff(a: np.ndarray, b: np.ndarray) -> dict:
    """Symmetric surface-to-surface distances in element units: max and 95th percentile."""
    sa, sb = _surface(a), _surface(b)
    if not sa.any() or not sb.any():
        return {"max": float("inf"), "p95": float("inf")}
    da = ndimage.distance_transform_edt(~sb)[sa]          # each surface voxel of a to the nearest of b
    db = ndimage.distance_transform_edt(~sa)[sb]
    d = np.concatenate([da, db])
    return {"max": float(d.max()), "p95": float(np.percentile(d, 95))}


def euler(m: np.ndarray) -> int:
    """Euler characteristic of the union of closed cubes (vertices - edges + faces - cubes)."""
    p = np.pad(m, 1)
    v = (p[:-1, :-1, :-1] | p[1:, :-1, :-1] | p[:-1, 1:, :-1] | p[:-1, :-1, 1:]
         | p[1:, 1:, :-1] | p[1:, :-1, 1:] | p[:-1, 1:, 1:] | p[1:, 1:, 1:])
    n_v = np.count_nonzero(v)
    n_e = 0
    for ax in range(3):                                    # edges along ax: OR of the 4 cubes around them
        q = np.moveaxis(p, ax, 0)[1:-1]
        n_e += np.count_nonzero(q[:, :-1, :-1] | q[:, 1:, :-1] | q[:, :-1, 1:] | q[:, 1:, 1:])
    n_f = 0
    for ax in range(3):                                    # faces normal to ax: OR of the 2 cubes beside them
        q = np.moveaxis(p, ax, 0)[:, 1:-1, 1:-1]
        n_f += np.count_nonzero(q[:-1] | q[1:])
    return int(n_v - n_e + n_f - np.count_nonzero(m))


def betti(m: np.ndarray) -> dict:
    b0 = int(ndimage.label(m, structure=np.ones((3, 3, 3)))[1])
    comp, n = ndimage.label(~np.pad(m, 1))                 # default structure: 6-connectivity
    b2 = int(n - 1) if n else 0                            # the component touching the padding is the outside
    chi = euler(m)
    return {"b0": b0, "b1": int(b0 + b2 - chi), "b2": b2, "chi": chi}


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: boolean masks on the same grid."""
    return {"iou": iou(a, b), "hausdorff": hausdorff(a, b)}
