# Second adjudication: agreement and score bounds

Part A references: 49/50 agree (98%), kappa 0.00
Part B contested records: 33/41 agree (80%), kappa 0.60

| Rule | R | T | G | GC | GCV | Claude T | Claude GC | Claude GCV | T->G (b/c, p) | G->GC (b/c, p) | T->GC (b/c, p) | T->Claude T (b/c, p) | GC->Claude GC (b/c, p) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| J.W. (paper) | 9 | 37 | 44 | 50 | 50 | 42 | 46 | 47 | 1/8, 0.0391 | 0/6, 0.0312 | 0/13, 0.000244 | 2/7, 0.18 | 4/0, 0.125 |
| S.Y. alone | 9 | 37 | 45 | 50 | 50 | 39 | 46 | 47 | 1/9, 0.0215 | 0/5, 0.0625 | 0/13, 0.000244 | 1/3, 0.625 | 4/0, 0.125 |
| both must accept | 9 | 37 | 44 | 50 | 50 | 37 | 46 | 47 | 1/8, 0.0391 | 0/6, 0.0312 | 0/13, 0.000244 | 2/2, 1 | 4/0, 0.125 |
| either accepts | 9 | 37 | 45 | 50 | 50 | 44 | 46 | 47 | 1/9, 0.0215 | 0/5, 0.0625 | 0/13, 0.000244 | 1/8, 0.0391 | 4/0, 0.125 |

## Part A disagreements (need re-encoding)

- b43_press_riser: first N, second Y;  Correct geometry and shortest-side mesh count (equivalent h = 2 mm). "Full bottom" means existing material outside the open channel. The split support patches in C31 implement that reading.

## Part B disagreements

- C06 b36_pedal_arm (Claude T): first W, second A; Retain the already fully fixed keyed-shaft cylinder as a solid non-design interface. The brief keys the arm to a shaft and fixes everything inside the 16 mm circle. Keeping that same cylinder solid reasonably represents the fixed interface. No free span is newly reinforced. This is an admissible interpretation, not a claim of identical material accounting.
- C13 b33_shelf_bracket (Claude T): first A, second W;  A nominal 10 mm washer conventionally gives a circular contact footprint. Candidate defaults to a 10 x 10 mm square, about 27% larger in area. This changes the support problem. Small compliance loss alone does not authorize a geometric approximation. Aluminium E rounding is acceptable.
- C25 b48_cable_anchor (Claude T): first W, second A; Represent the 16 mm boss face by a 1 mm retained terminal cylinder y = 59..60 mm. A one-element-deep retained boss is a reasonable finite interface for the named boss face. The same cable resultant acts at the same x,z and azimuth. Distribution through this thin terminal layer is accepted as that boss interface idealization, not general surface/body-load equivalence.
- C27 b39_tv_bracket (Claude T): first A, second W;  The four 12 mm washers imply circular bearing contacts. Candidate uses 12 x 12 mm square contacts at the correct centres. Area is about 27% larger per contact. It is an approximation to a different support region, with no stated approximation policy.
- C29 b39_tv_bracket (G): first W, second A; Keep the named mounting plate as a 6 mm end layer and the circular screw interfaces as 6 mm deep solids. A finite retained TV mounting plate and local circular screw interfaces reasonably represent the hardware named in the brief. Their thickness is unspecified. The added solids stay at those interfaces and preserve load/contact geometry. Higher total mass is not a performance improvement at equal mass.
- C35 b19_engine_bracket (Claude T): first A, second W;  The brief explicitly specifies 0.5 in diameter washers. Candidate uses 0.5 x 0.5 in square contacts rather than circles. Correct centres and small reported loss do not satisfy the diameter requirement.
- C40 b31_pillow_block (Claude T): first A, second W;  The 16 mm washers imply circular contacts. Candidate uses 16 x 16 mm squares, enlarging each support area by about 27%. This is a distinct boundary region, not an exact rewrite. The retained bearing seat and force are otherwise correct.
- C41 b41_hinge_bracket (Claude T): first A, second W;  The 12 mm washers imply circular contacts. Candidate uses 12 x 12 mm square contacts, enlarging each support area by about 27%. The correct boss, slot and load do not remove this support-region difference.
