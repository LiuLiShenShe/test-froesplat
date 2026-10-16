# Phase20F phenotype measurement method

## Scope and evidence gate

This method operates on the frozen M0 raw and post Mesh files selected by the
Phase20E-3 manifest. It never edits Mesh vertices, source Gaussian PLY files,
camera poses, COLMAP outputs, or the historical workbook. Every physical value
must pass both gates below:

1. A verified real-world distance and two corresponding reconstructed endpoints
   must provide `scale_cm_per_unit = real_distance_cm / mesh_distance_units`.
2. The plant-only point set, vertical/reference plane, and trait definition must
   be recorded independently of the value being evaluated.

When either gate is absent, the result is blank with a blocking status. The
implementation does not infer a scale from a pot bounding box, PCA axis, or
manual plant height.

## Formal geometry

For a plant point set `P`, a base point `b`, and a unit vertical vector `v`, the
height candidate is:

```text
height_units = max((p - b) dot v for p in P)
height_cm = height_units * scale_cm_per_unit
```

The point set must exclude pot/background geometry. The base definition must be
frozen before comparing to manual height. A horizontal orthonormal basis
`(h_a, h_b)` with both axes perpendicular to `v` gives the two canopy spans:

```text
span_a = max(P dot h_a) - min(P dot h_a)
span_b = max(P dot h_b) - min(P dot h_b)
```

The maximum of these is only comparable to a manual maximum-width measurement
when that manual definition is explicitly documented. East-west and
north-south measurements require the corresponding recovered reference axes.

## Leaf measurements

Leaf length requires a stable `Leaf ID`, a base landmark, a tip landmark, and a
declared method. The current implementation exposes the straight base-to-tip
method. A curved centerline method must be implemented and declared separately;
the two methods must not be mixed.

Leaf width requires the same `Leaf ID`, a boundary point set, and a declared
cross-section direction or width plane. A visible leaf-like surface is not an
organ identity. If identity, boundary, or landmarks are missing, the output is
`LEAF_ID_NOT_AVAILABLE` or `LEAF_GEOMETRY_INCOMPLETE_AT_ORGAN_LEVEL`.

## Raw/post policy

Raw and post Meshes are measured independently. Post is the E3 largest
face-connected component, not a validated plant-only segmentation. The audit
therefore retains raw component IDs, triangle counts, centroid/bounding-box
summaries, source hashes, and RGB projection evidence. No component is removed
from the raw analysis merely because it is small.

## Outputs and statuses

- `PHENOTYPE_GEOMETRY_RESULTS.csv`: exploratory PCA extents in inherited COLMAP
  units. These values include pot or reconstruction artifacts and have
  `physical_trait_claim=False`.
- `PHENOTYPE_PHYSICAL_RESULTS.csv`: one row per scene, stage, and trait; values
  remain empty with `PHYSICAL_SCALE_NOT_AVAILABLE`.
- `LEAF_MEASURABILITY_REPORT.csv`: organ-level readiness and blocking reason.
- `PHENOTYPE_ERROR_METRICS.csv`: header only because no valid same-plant,
  same-time, calibrated pairs exist.

## Reproduction

```bash
/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python \
  '00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十F_植物表型与尺度恢复/scripts/run_phase20f.py'
```

The runner verifies input hashes before reading the six M0 Meshes, checks the
post vertex coordinate subset and largest-component face count, runs component
and frozen-camera projection analysis, reads the workbook through its OOXML
structure, and writes all phase outputs. `scripts/phenotype_core.py` contains
the guarded reusable geometry primitives.

## Mathematical tests

`tests/test_phenotype_core.py` covers known-box height, rotation-invariant PCA
extents, linear scale conversion, missing-scale rejection, invalid-axis
rejection, canopy-axis orthogonality, leaf identity enforcement, and raw/post
coordinate-subset checking. Synthetic geometry is used only for these unit
tests, never as a paper result.
