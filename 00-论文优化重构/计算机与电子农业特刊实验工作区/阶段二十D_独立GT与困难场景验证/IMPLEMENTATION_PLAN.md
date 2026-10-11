# Phase20D implementation plan — 2026-10-11

## Scope and dependencies

Baseline: `898cec2a`, clean `main`. Confirm the frozen P6/per_instance/single/R1
method, never develop or tune it. SPNP/Residual/GCFC/A6/A7 remain OFF. No changes
to existing algorithms, historical artifacts, 2DGS, M2M3-GO or manuscript text.
All scoring, prediction, accuracy evaluation and publication figures depend on
32 complete manual annotations and distinct human second review.

## Steps

1. Read Phase20C-FIX decisions/provenance/recovery, Phase16 manifests, Phase17A
   U00_Control, historical TEST32 changes, production strict loader and GT gate.
   Verify baseline and original source identities. Estimate: 10–20 minutes.
2. Add isolated read-only input audit, archived-score metadata snapshots, original
   RGB/context links, pending review/difficulty tables and hash inventory here.
   Verify 32/8/71 coverage, IDs, boxes, scores, dimensions, binary masks, duplicates
   and separation from both Phase16 DEV26 and Easy21/GT10. Estimate: 10–20 minutes.
3. Extend only the Phase20D preflight gate with annotation-hash binding, real-human
   attestations, two-person separation and explicit difficulty visual evidence.
   With incomplete GT, produce only BLOCKED result schemas/reports; do not launch
   R0/R1, re-score candidates, generate GT or invent human decisions. Estimate: 10 minutes.
4. Run focused synthetic gate tests and existing metadata/gate regression tests;
   audit original hashes and change boundary, commit only small public stage files,
   attempt a normal fast-forward push, and stop. Estimate: 5–10 minutes.

## Risks and verification

- Six-decimal historical confidence is the only recoverable precision: preserve it;
  do not infer extra digits or rerun SAM3. Full tensor precision remains NOT_SAVED.
- Existing D1/D11 labels are selection heuristics, never verified difficulties.
- Sequence IDs/hashes prove manifest separation, not independently confirmed
  physical target identity; manual confirmation remains required.
- Adjacent frames are correlated. Compare exact file and decoded-RGB duplicates;
  do not describe all 32 frames as independent plants or prove no near-duplicates.
- Never overwrite annotation/review files or historical data. Audit output creation
  is exclusive; a repeat audit must use a new output directory.
- No prediction/metric module runs during the blocked gate. Tests use synthetic
  in-memory fixtures, never write synthetic TEST32 GT.
- External Bocha retrieval returned HTTP403; conclusions use actual local files.
  `/home/test/.codex/RTK.md` is absent; `.codegraph/` and `graphify-out/` are absent.

Sources: user Phase20D specification; local Phase20C-FIX and Phase16–17C artifacts.
