# Refinement Interaction Report

This report uses saved adjacent-stage masks and GT metrics, not feature flags alone. Easy21 is the only dataset with complete V2/V3 chain execution in Phase 20C.

| Group | Stage transition | N | Mean ΔF1 | Min | Max | Positive | Negative | No change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V2 | Selected -> SPNP | 21 | 0.0030 | -0.0022 | 0.0098 | 20 | 1 | 0 |
| V2 | SPNP -> Residual | 21 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 21 |
| V2 | Residual -> GCFC | 21 | -0.0008 | -0.0035 | 0.0001 | 1 | 20 | 0 |
| V2 | GCFC -> Final | 21 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 21 |
| V3 | Selected -> SPNP | 21 | 0.0030 | -0.0022 | 0.0098 | 20 | 1 | 0 |
| V3 | SPNP -> Residual | 21 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 21 |
| V3 | Residual -> GCFC | 21 | -0.0008 | -0.0035 | 0.0001 | 1 | 20 | 0 |
| V3 | GCFC -> Final | 21 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 21 |

Interpretation:

- SPNP produces a measurable mean F1 gain and is accepted on all 21 Easy21 frames in V2/V3, with one frame showing a small negative change.
- Residual repair is called with `thin_repair_mode=adjacent`, but all saved residual masks equal the SPNP masks: `NO_CHANGE` on Easy21.
- GCFC is called and changes all 21 Easy21 masks, but its adjacent-stage F1 delta is negative on 20/21 frames. It remains a conditional module, not a default-locked component.
- Because GT10 V2/V3 is blocked, this interaction evidence cannot justify a complete hard-case refinement claim.
