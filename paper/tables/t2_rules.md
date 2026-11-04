**Verification rule base (deterministic; any error blocks solving and release)**

| Code | Severity | Condition | Kind |
|---|---|---|---|
| R001 | error | resolution cannot be met (grid too fine or too coarse) | spec |
| R002 | error | a support selects no nodes | spec |
| R003 | error | supports leave a rigid-body mode free | physics |
| R004 | error | a load selects no nodes, or every load is zero | spec |
| R005 | error / warning | load applied on fixed DOFs (dropped and recorded) | physics |
| R006 | error | no design elements remain after passive regions | spec |
| R007 | error | elements both keep-in and keep-out | spec |
| R008 | error | a support or load lies entirely inside a keep-out | physics |
| R009 | warning | smallest required feature not resolvable at this h | resolution |
| R010 | warning | a region was snapped to the grid | geometry |
| R011 | warning | a passive region contains no element centre | geometry |
| R012 | warning / info | supports lost on coarse multigrid levels | solver |
| R013 | error | memory need exceeds the GPU | solver |

Source: src/agentto/rules.py.
