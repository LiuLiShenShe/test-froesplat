# FINAL RAP-FSAM3 METHOD DECISION

## Decision

- Status: `PARTIAL`
- Recommended foreground method: `V1`
- R1 production validation: `PASS` for V0/V1 frozen-cache selection replay; score-metadata equivalence caveat remains.
- Refinement integration: `PARTIAL` — Easy21 full chain completed; GT10 V2/V3 blocked by three missing COLMAP observations.
- GT10 hard-case rescue: `PASS` for formal R0/R1 selected-mask comparison on `0039/0040/0041`.
- Easy21 regression: `NONE` for V0 vs V1; full-chain delta is +0.002003 F1 for V3 vs V1.
- Method status: `PARTIAL`.

## Frozen production configuration

- Prompt: `P6`; candidate mode: `per_instance`; prompt selection: `single`; prompt ensemble: OFF.
- Ranking: `r1`; weights: `area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5`; mask threshold: `0.5`.
- Refinement: SPNP `sam3_if_supported`, residual repair `adjacent`, corrective geometry `colmap_tracks` when geometry is available.
- A6/A7: OFF.
- GT: `potted_clean`; Easy21 and GT10 remain development/historical evaluation data.

## Observed effects

- Easy21 V1−V0 sample-equal F1: `0.0000`.
- Easy21 V2−V0 sample-equal F1: `0.0022`.
- Easy21 V3−V1 sample-equal F1: `0.0022`.
- Interaction `(V3−V2)−(V1−V0)` is reported in `metrics/paired_effects.csv`; no significance claim is made for these small development sets.
- Recommendation rationale: V3 的净 F1 增益来自 SPNP，但 GCFC 在 Easy21 21 帧中 20 帧降低 F1，且 GT10 完整链尚未验证；因此以 V1 为默认，refinement 仅条件启用。

## Production command

The exact commands are preserved in `configs/PHASE20C_COMMANDS.txt`; each run also stores `execution_command.txt`, `参数.json`, `运行日志.json`, and `artifact_hashes.json`.

## Blocking items

1. Supply and verify COLMAP sparse observations for `DouBanLv2_0037`, `DouBanLv2_0039`, and `DouBanLv2_0041`, then run GT10 V2/V3 from clean output directories.
2. Either make the production candidate-reuse path ingest frozen SAM3 score metadata or run fresh SAM3 candidate generation for a byte-equivalent replay; do not silently treat `sam_score=0.0` as equivalent.
3. Add an independent, non-development validation set before changing `PARTIAL` to `LOCKED_CANDIDATE`.

This decision does not authorize Phase 20D, 2DGS, M2M3-GO, or paper-text changes.
