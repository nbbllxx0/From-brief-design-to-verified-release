# X6b: interaction cost by Keystroke-Level Model (14 released X3 tasks; b49 is refused)

Operators: K 0.28 s, P 1.1 s, BB 0.2 s, H 0.4 s, M 1.35 s. Agent and solver time is waiting and is not counted. Actions are keystrokes plus clicks.

| Task | A1 voxfem script: lines, s | A2 agentto spec: s | B GUI, no agent: actions, s | C GUI + agent (brief typed / pasted): actions, s | Questions, cards | Agent run, s |
|---|---|---|---|---|---|---|
| b01_cantilever_block | 24, 304 | 168 | 56, 68 | 376, 125 / 7, 20 | 0, 7 | 20 |
| b02_mbb_half | 25, 314 | 209 | 95, 101 | 475, 154 / 7, 22 | 0, 8 | 37 |
| b04_torsion_shaft | 26, 347 | 204 | 94, 108 | 483, 159 / 8, 24 | 1, 8 | 47 |
| b05_l_bracket | 47, 618 | 213 | 85, 94 | 452, 150 / 8, 24 | 1, 8 | 51 |
| b06_motor_mount | 55, 823 | 479 | 347, 374 | 536, 190 / 11, 42 | 4, 15 | 81 |
| b11_cantilever_line_load | 24, 296 | 162 | 52, 67 | 304, 105 / 7, 20 | 0, 7 | 21 |
| b17_arch_bridge | 48, 612 | 217 | 69, 85 | 383, 132 / 8, 26 | 1, 9 | 33 |
| b19_engine_bracket | 50, 707 | 352 | 270, 277 | 557, 188 / 10, 34 | 3, 11 | 47 |
| b25_cantilever_metres | 24, 303 | 198 | 107, 107 | 355, 122 / 8, 23 | 1, 7 | 38 |
| b31_pillow_block | 48, 648 | 283 | 155, 168 | 452, 163 / 13, 39 | 6, 9 | 145 |
| b35_padeye | 48, 628 | 208 | 66, 84 | 379, 132 / 9, 27 | 2, 8 | 62 |
| b40_crank_arm | 47, 626 | 249 | 124, 124 | 503, 164 / 8, 24 | 1, 8 | 52 |
| b44_gearbox_bracket | 26, 347 | 204 | 94, 108 | 377, 131 / 9, 27 | 2, 8 | 73 |
| b47_camera_mount | 24, 304 | 187 | 100, 102 | 287, 107 / 10, 28 | 3, 7 | 44 |
| b49_hinged_arm (refused) | 24, 312 | 171 | 73, 81 | 388, 128 / 7, 20 | 0, 7 | 26 |

Medians over the 14 released tasks: A1 480 s, A2 208 s, B 104 s (94 actions), C typed 141 s, C pasted 25 s (8 actions); agent run 47 s of waiting.
A1 setups verified against problem.build: 15/15.
