# ForeSplat Two-Stage Gaussian Artifact Suppression

## Scope and fixed evidence

Restore native GGGS color-Jacobian attribution, foreground/background comparison,
iterative deletion and reattribution from historical rounds 1, 2, 4 and 5.
Round 3 global scaling and the marker-specific spatial deletion are excluded.
Keep the 513,840-row Round 5 and 510,248-row plant+pot assets separately.
Existing C5 inputs/results and uncommitted work remain preserved.

## Implementation

1. Add `cleaner/historical/`: automatic camera sampling, cached external support,
   current-model attribution, adaptive contribution floors, historical deletion,
   disconnected-component discovery, optional V3 priors and bounded candidate-only
   anisotropic scale proposals with exact native filter/opacity compensation.
2. Add `--strategy historical_two_stage` to the public entry. Run 24-view quick
   evaluations, 64-view confirmation for promising whole-round proposals, and
   matched full-view evaluation of stage outputs. Never use full208 per small edit.
   Preserve original foreground contribution as an envelope after each edit.
3. Export native/viewer/removed PLY, row identities, actual field changes,
   acceptance/rejection records, timings, source/input hashes, and stage metrics.
   Keep automatic free views distinct from fixed historical eight-view evaluation.
4. Add a serial, restartable batch/comparison driver: B0 original, B1 historical
   best, B2 generalized stage1, B3 historical best→stage2, then existing models of
   ChangShouHua1, XiangPiShu1 and WangWenCao1. Freeze the shared config before
   heldout runs. Missing models are reported; no model training is permitted.
5. Generate contact sheets and real absolute-path indexes, and report all eleven
   requested conclusions with measured/pending status. A rendered contact sheet
   is not a completed interactive SuperSplat review; disclose that distinction.

## Verification

Unit checks target stale contribution cache rejection, camera sampling invariance,
foreground-envelope alignment through pruning, native covariance/opacity baking,
dark supported leaf protection and exact edit provenance. Native GPU smoke and
development runs verify contribution conservation, actual final whole-view
measurements and viewer parity. Completion markers are written only after export,
evaluation and unchanged-input/source verification.

## Runtime and risks

Only physical GPU1, existing uv environment. Development measurements determine
runtime; full-view evaluation is expected to dominate. External depth projections
are support proxies, never independent ground truth. Historical marker deletion
changes the RAP photometry and must not be conflated with artifact suppression.
Selection uses measured free-view artifacts and foreground preservation, followed
by visual review; deleted point count is not an objective. Bocha lookup returned
HTTP403 quota; algorithm evidence is local, not an unverified web claim.
