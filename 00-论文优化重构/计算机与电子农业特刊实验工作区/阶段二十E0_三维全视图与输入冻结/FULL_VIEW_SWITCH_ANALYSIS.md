# Full-view R0/R1 switch analysis

| sample | registered | candidate_views | new_instances | R0_R1_switches | train_mask_differences | ready | training_runs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WanNianQing1 | 213 | 213 | 339 | 3 | 3 | True | 3 |
| CaoMei2 | 106 | 106 | 111 | 1 | 1 | True | 3 |
| HongZhang | 213 | 213 | 472 | 0 | 0 | True | 2 |

Each RGB is inferred at most once for new candidates. Both policies consume the
identical cache, confidence precision, instance order, boxes, thresholds and full
Phase20D configuration. Only `ranking_policy` differs. Production scorer and
`ranking_policy.py` are loaded unchanged; every candidate decomposition is in
`reports/FULL_VIEW_SCORE_DECOMPOSITION.csv`.
The per-instance model/processor lineage is in
`reports/FULL_CANDIDATE_MODEL_LINEAGE.csv`: historical instances match archived
parameters and Phase20D frozen model hashes; a checkpoint tensor digest from the
original Phase17A generation was not independently archived. This limit is
retained, not fabricated. New inference records exact checkpoint and source hashes.

Switching IDs is not automatically improved segmentation. No GT is loaded by
generation or ranking. Empty candidates/masks, fallback and missing flags are
recorded. Pixel differences are recomputed both before and after undistortion.
If all registered **training** inputs match, B/C cannot test ranking effects on
training; execute only one and share that checkpoint. Validation-mask differences
alone do not justify duplicate optimization. Actual distinct runs: **8**.
