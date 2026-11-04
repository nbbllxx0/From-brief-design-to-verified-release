# X3 sessions re-timed under the pad-aware coarse rule (replay, no LLM)

| Task | Release elements | Solve wall, voxfem rule → aware (s) | Session, X3 → aware (s) | Release compute (s) | Release ΔC | Certified |
|---|---|---|---|---|---|---|
| b01_cantilever_block | 16.0M | 87 → 54 | 125 → 92 | 62 → 34 | -1.6e-07 | True |
| b02_mbb_half | 12.3M | 51 → 63 | 77 → 89 | 32 → 34 | +0.0e+00 | True |
| b04_torsion_shaft | 65.5M | 108 → 122 | 140 → 154 | 85 → 91 | +0.0e+00 | True |
| b05_l_bracket | 25.6M | 199 → 102 | 236 → 140 | 133 → 68 | -3.1e-06 | True |
| b06_motor_mount | 73.7M | 64 → 42 | 176 → 154 | 35 → 18 | +3.3e-06 | True |
| b11_cantilever_line_load | 49.2M | 244 → 204 | 273 → 232 | 141 → 140 | +3.5e-07 | True |
| b17_arch_bridge | 14.4M | 319 → 87 | 351 → 119 | 208 → 63 | -8.6e-07 | True |
| b19_engine_bracket | 35.8M | 1825 → 282 | 1877 → 334 | 1508 → 227 | +1.0e-05 | True |
| b25_cantilever_metres | 16.0M | 156 → 75 | 182 → 102 | 105 → 51 | -3.3e-07 | True |
| b31_pillow_block | 18.4M | 80 → 76 | 128 → 124 | 56 → 55 | +3.1e-06 | True |
| b35_padeye | 15.4M | 82 → 52 | 116 → 86 | 55 → 31 | +2.3e-08 | True |
| b40_crank_arm | 13.8M | 430 → 73 | 463 → 106 | 298 → 48 | -9.5e-06 | True |
| b44_gearbox_bracket | 65.5M | 105 → 112 | 138 → 145 | 82 → 88 | +0.0e+00 | True |
| b47_camera_mount | 27.6M | 260 → 120 | 287 → 148 | 199 → 94 | -6.7e-06 | True |
| b49_hinged_arm | 0.0M | 0 → 0 | 4 → 4 | 0 → 0 | – | None |

Median session: 179 s (X3) → 132 s (aware); range 86–334 s (14 released sessions). Unpadded grids are rule-independent, so their changes are run-to-run timing noise.
