# ForeSplat Phase20C-FIX

New isolated verification after Phase20C commit `970082da`. Frozen R1 formula/weights,
original GT/candidate masks, historical stages, 2DGS/M2M3-GO/COLMAP core and manuscript
are not modified. Production cache metadata is the only functional repair.

## Start here

- `PHASE20C_FIX_FINAL_REPORT.md`: final status and actual metric evidence.
- `CANDIDATE_METADATA_PROVENANCE.md`: 55 identity/hash-matched archived confidences,
  boxes/order/threshold and explicit six-decimal versus unsaved tensor limitation.
- `CACHE_LOADER_FIX_REPORT.md`: backward-compatible strict metadata loader and tests.
- `R0_R1_REVALIDATION.md`: real-score selection replay and three hard-case rescues.
- `SPNP_ONLY_ABLATION_REPORT.md`: V4/V5 real GPU runs with all positive/negative metrics.
- `FINAL_FOREGROUND_METHOD_DECISION.md`: predeclared multi-metric decision, not GT switching.
- `GT10_COLMAP_OBSERVATION_DECISION.md`: true missing registration and coordinate caveat.
- `PHASE20D_INDEPENDENT_VALIDATION_PLAN.md`: TEST32 human work, no test tuning/accuracy.
- `ARTIFACT_RECOVERY.md`: exact commands, paths, hashes and ignored-source restoration.

## Evidence

`CANDIDATE_SCORE_REPLAY.csv`, `R0_R1_SELECTION_REPLAY.csv`,
`ROUNDING_SELECTION_CERTIFICATE.csv`, `V4_V5_METRICS.csv`, `V4_V5_PER_FRAME.csv`,
`FRAME_PAIRED_DIFFERENCES.csv`, `PAIRED_METRICS.csv`, `RUN_TIMES.csv`.
Actual small production outputs are in `small_run_outputs/`; original run stdout in
`logs/`; per-run source/config/artifact hashes in `execution_records/` and `artifact_hashes/`.

SCORE_EQUIVALENT is **numerical archived-six-decimal equivalence at 1e-7**. It never
claims byte-perfect reconstruction of discarded SAM3 tensor confidences. Exact selection
under all rounding-consistent original scores is independently certified for 62/62 pairs.

GPU masks/RGB/checkpoint stay local/ignored. No Phase20D evaluation is run; manual
annotations and separate human review are still pending. This is DEV/regression evidence.

Sources: frozen local repository/archive plus real new runs. Bocha web search returned
HTTP403 quota exhaustion; no internet snippet is used as experiment evidence.
