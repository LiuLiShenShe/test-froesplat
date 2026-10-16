# Phase20G handoff

Phase20F stops after establishing the guarded measurement path. It does not
enter Phase20G and does not modify the paper.

## Ready artifacts

- All three frozen M0 raw/post pairs were read and hash-verified.
- Raw/post coordinate identity and largest-component face-count checks passed.
- Component deletion audit and fixed-camera RGB projections were generated.
- Guarded height, canopy, scale, and leaf measurement primitives were tested.
- Historical workbook structure and candidate variety matches were checked.
- Empty physical/error outputs preserve blocking states rather than zeros.

## Phase20G entry conditions

Do not start a physical validation experiment until the user supplies the data
listed in `USER_DATA_STILL_REQUIRED.md`. On entry, freeze a provenance manifest
with the specimen/time bridge, scale source, endpoint coordinates, reference
frame, trait definitions, and uncertainty fields before calculating any error.

## Suggested next execution

1. Verify the supplied scale and reference-frame records against the frozen
   camera/Mesh hashes.
2. Produce a plant-only point set without altering the protected Mesh; record the
   segmentation source and coverage limitations.
3. Compute raw and post traits under the predeclared definitions.
4. Run the same-plant pairing gate before joining any manual values.
5. Emit per-sample measurements, valid counts, MAE/RMSE/Bias/MAPE/R2 only when
   sample size and variation support them, and Bland-Altman data when repeated
   or paired values justify it.
6. Keep the old Table 8 statistics in a separate legacy provenance block until
   the new measurements are independently reconstructed.

## Current status

`NEXT_PHASE: USER_DATA_REQUIRED`. The current Phase20F directory is the input
package for that work; no training, model selection, Mesh editing, COLMAP edit,
or manuscript edit is required to satisfy this handoff.
