# Table 8 reconstruction decision

## Decision

`TABLE8_STATUS: REPLACEMENT_REQUIRED` for a new Mesh-based validation table.
This phase did not find a verified physical scale, specimen/time bridge, leaf
identity record, or per-value Mesh/landmark provenance. Therefore the new raw or
post Mesh outputs cannot replace the old centimetre-level Table 8 values.

The original `.tex` manuscript was not modified.

## Legacy values found

The repository contains the following already-aggregated workbook statistics in
`ForeSplat_real_release/examples/paper_tables/table8_phenotype_summary.csv` and
`00-论文优化重构/计算机与电子农业特刊实验工作区/04-结果表格模板/实验五_人工虚拟表型汇总.csv`:

| Trait | n | MAE | RMSE | MAPE | Bias | R2 |
|---|---:|---:|---:|---:|---:|---:|
| Plant height | 21 | 0.9848 | 1.2073 | 6.9064% | 0.5771 | 0.9878 |
| Canopy width | 21 | 0.8648 | 0.9880 | 4.5000% | 0.6410 | 0.9879 |
| Leaf length | 63 | 0.5063 | 0.6370 | 7.4491% | 0.3133 | 0.9738 |
| Leaf width | 63 | 0.4502 | 0.6378 | 9.7256% | 0.3825 | 0.8999 |

These are **`LEGACY_RECORDED_STATISTICS`**, not results recomputed from this
phase's six M0 Meshes. The Phase20B audit states that the aggregate script reads
the workbook and does not read Mesh vertices, scale markers, landmarks, or
camera-to-physical transforms. The old `.tex` text therefore cannot be
relabelled as a newly reproducible 3D validation result.

## What can replace it now

Nothing in the current phase can populate a new physical Table 8. The new
outputs provide:

- verified Mesh identity and hashes for all three frozen scenes;
- raw/post component deletion and RGB projection evidence;
- reusable height, canopy, scale, and leaf-identity guards;
- 12 exploratory geometry rows in COLMAP arbitrary units;
- explicit empty physical results and an empty error-metrics table.

No cm, mm, cm2, cm3, MAE, RMSE, MAPE, R2, or Bland-Altman value is emitted for
the new Meshes.

## Conditions for a replacement table

Before replacing Table 8, obtain a specimen/time identity bridge, a verified
physical length anchor per applicable scene, the gravity/base reference, frozen
trait definitions, and independently recorded manual values for the same plant
and time. For leaf traits, provide persistent leaf IDs and base/tip/boundary
evidence. Re-run the phase runner or its measurement entry point with those
inputs, then report per-trait valid n and missingness rather than filling blocked
metrics with zero.

Until those conditions are met, the paper may discuss the historical table only
with its legacy provenance and limitations. The new Mesh analysis belongs in
Methods/Limitations or a replacement experiment, not as a silent update of the
old numerical claim.
