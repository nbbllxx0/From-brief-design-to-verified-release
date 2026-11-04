"""Device-memory breakdown of a SIMP object by component (bytes and bytes
per element), from the arrays it actually holds."""
from __future__ import annotations

import cupy as cp


def _nb(*arrs):
    return int(sum(a.nbytes for a in arrs if isinstance(a, cp.ndarray)))


def breakdown(s) -> dict:
    g = s.grid
    out = {}
    out["design_fields"] = _nb(s.rho, s.rho_phys, s.rho_prev, s.dc, s.dv, s.w1, s.w2)
    out["moduli"] = _nb(s.E32)
    if getattr(s, "res32", None) is not None:
        out["residual_kernel_tables"] = _nb(s.res32.nodes, s.res32.bl, s.res32.KE)
    out["state_u_fp64"] = _nb(s.u)
    out["fixed_mask_and_load"] = _nb(s.fixed, s.load.idx, s.load.val)
    if hasattr(s, "ws"):
        out["ir_workspace"] = s.ws.nbytes
    if hasattr(s, "pcg_work"):
        out["pcg_workspace"] = _nb(*s.pcg_work)
    if hasattr(s, "dinv"):
        out["jacobi_diagonal"] = _nb(s.dinv)
    if getattr(s, "castM", None) is not None:
        out["cast_buffers"] = _nb(s.castM.r32, s.castM.z32)
    if s.mg is not None:
        st = s.mg.storage_bytes()
        out["mg_level_vectors"] = st["vectors"]
        out["mg_stored_matrices"] = st["matrices"]
        out["mg_coarse_inverse"] = st["coarse_inverse"]
        out["mg_tables"] = st["KE_tables"]
    out["total_accounted"] = sum(v for v in out.values())
    out["per_element"] = {k: v / g.n_elem for k, v in out.items() if isinstance(v, int)}
    out["n_elem"] = g.n_elem
    out["pool_used_bytes"] = int(cp.get_default_memory_pool().used_bytes())
    out["pool_reserved_bytes"] = int(cp.get_default_memory_pool().total_bytes())
    return out
