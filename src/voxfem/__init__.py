"""voxfem: matrix-free structured-voxel finite elements on one GPU.

Package layout
--------------
core        grid indexing, element stiffness, problem definitions
operators   every K*v mapping studied (element/node/DoF ownership, tiling,
            generated stencils) behind one interface, plus the Jacobi diagonal
sparse      assembled cuSPARSE baselines (scalar CSR and 3x3 BSR)
mg          memory-lean Galerkin geometric multigrid (matrix-free fine and
            first coarse level, stored element matrices below, dense coarsest)
krylov      fail-closed PCG on the true residual, mixed-precision variants
filters     matrix-free cone filter
simp        SIMP driver (OC update, projection continuation, design-convergence
            termination, final-iterate reporting)
guard       run-time guards: single GPU job, VRAM budget, Windows shared-memory
            spill detection, device state records
sass        static SASS instruction counts of the compiled kernels

Index conventions (identical to the R2 code in code/src/gpu_fem):
    element e = (ex*ny + ey)*nz + ez
    node    n = (ix*(ny+1) + iy)*(nz+1) + iz
    dof       = 3*n + c
    local node order (ox,oy,oz): 0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0)
                                 4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
"""

__version__ = "0.1.0"
