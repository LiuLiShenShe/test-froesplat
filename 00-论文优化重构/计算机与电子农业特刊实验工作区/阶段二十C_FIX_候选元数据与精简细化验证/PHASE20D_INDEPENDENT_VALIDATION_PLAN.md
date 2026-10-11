# Phase20D independent validation plan — preparation only

## Frozen data audit

Phase16 TEST32 contains **32 frames / 8 samples**; the sample-level DEV and TEST sets
are disjoint. All RGB files exist and match Phase17A TrackB input hashes. `U00_Control`
has **71 original P6 candidates**, complete identity/detail coverage on all 32 frames.
`TEST32_INPUT_AUDIT.csv`, `TEST32_CANDIDATE_AUDIT.csv` and the original manifest hashes
in `annotation_package/FROZEN_TEST_SPEC.json` provide the exact inventory.

No TEST32 human annotation JSON was found in the current GT annotation locations;
all frozen manifest GT statuses are pending. Plant/pot labels and target identity are
**NOT_MANUALLY_CONFIRMED**. No test accuracy, pseudo-GT, or parameter tuning occurs here.
The preselected D1/D11 difficulty labels are heuristic selection labels, not human truth.

## Work the user must complete

1. Use `annotation_package/rgb/<sample>_<frame>.jpg` at original resolution. Annotate
   exactly one actual target potted plant, excluding neighbor plants and background.
2. Draw separate polygons labelled `potted_plant`, `plant`, `pot`, and `blue_cube` if
   present. Keep coordinates on the original RGB grid. Define potted_clean as target
   potted_plant minus blue_cube; plant/pot are supplementary human-verified components.
3. Manually assign actual difficulty types (neighbor adhesion, pot/background confusion,
   thin structures, clutter, occlusion, etc.); do not copy D1/D11 by default. Record none
   if no difficult property is actually present.
4. A distinct second reviewer verifies target identity, plant/pot labels, cube presence,
   polygon geometry and difficulty. Fill `ANNOTATION_REVIEW_QUEUE.csv` with reviewer
   names, `YES` confirmations, cube YES/NO, human difficulty and `APPROVED`.
5. Save JSONs in `annotation_package/annotations/<sample>_<frame>.json`, record hashes,
   freeze the approved annotation version **before** running independent evaluation.

No annotation polygons or GT masks are generated from SAM predictions. Current RGB
links are a convenience; `scripts/audit_geometry_test32.py` recreates them without
changing the original manifest. No human review is represented as completed.

## Fixed evaluation

`scripts/evaluate_independent_test.py` requires full TEST32 coverage and separate
two-person human approval before accessing accuracy metrics. It rejects missing labels,
unknown labels, incorrect original dimensions, out-of-grid polygons, cube/review mismatch,
prediction dimension mismatch and overwriting a result file.

Metric definitions are frozen Phase20B F1 / foreground IoU / HD95 / Boundary-F1 (3px) /
Leakage = FP / image pixels. Report sample-equal means, frame-paired differences,
empty masks, severe F1<0.5 proxy and human target-identity audit separately.
Use the final method selected in `FINAL_FOREGROUND_METHOD_DECISION.md` with unchanged
P6, R1 and preprocessing; no test-driven threshold/weight/SPNP-policy adjustment.
If additional candidate inference is necessary, freeze a new source explicitly and do
not mix its candidates with U00_Control as though they were the same replay.

## Stop boundary

This stage prepares the annotation package and evaluator only. Independent evaluation
is **not ready to execute** until human GT review is complete. There is no Phase20D
GPU run, test metric or manuscript edit. The next phase must be explicitly authorized.

Sources: Phase16 DEV/TEST manifests, Phase17A U00_Control masks/detail CSVs, current
RGB/GT filesystem audit and this delivery's fixed evaluation gate tests.
