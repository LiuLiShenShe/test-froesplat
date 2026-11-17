# Manifest-wide metric phenotyping implementation

Scope: all 19 original manifest entries, preserving development / heldout labels.
Old source models, cameras, meshes and reports are immutable. GPU stages share the
existing physical-GPU1 lock. There is no algorithm sweep and no C1/C2 cleaning.

1. Audit actual SfM sources by registered feature observations / tracks, distorted
   camera model and source provenance. Never use manifest Omega training models as
   true-COLMAP models. Prepare common RGB/masks and camera trajectories. Reuse the
   existing KongQueZhuYu true-COLMAP model and 30K model. If no actual source exists,
   run installed COLMAP SIFT / sequential matching / mapper and preserve logs.
2. Implement one per-plant context and durable stage ledger with input signatures.
   Remove species names and fixed leaf counts from batch processing. Work order:
   DouBanLv1, CaoMei1, then remaining manifest entries; split labels stay frozen.
3. Use existing SAM3 checkpoint for independent cube masks and semantic/leaf
   proposals. Historical cube boundaries are optional and explicitly recorded.
   Reuse joint rigid-cube silhouette fitting, independent corner triangulation,
   adjacent-edge checks and reprojection validation. Uncalibrated plants retain
   model-unit geometry and blank cm measurements.
4. Uniform metric conversion: xyz and filter_3D *= s, log scales += log(s), camera
   centers and tvec *= s. Preserve other PLY fields bit-for-bit. Mark source/output
   units and one applied conversion. Render from metric Gaussian/cameras; scale
   existing depth observations only when they are mathematically equivalent.
5. Train only true-COLMAP GGGS at 30K / SH3 / SG0 / seed0 using existing native
   loop, then official bounded TSDF at scene diagonal/512, truncation 5 voxels,
   unchanged post policy (remove <50-triangle fragments, no smoothing/filling).
6. Fuse visible SAM3 semantic and instance masks into original triangles; validate
   temporal observations, retain ambiguity, no fixed maximum of 12 leaves. Extract
   stable leaf identities with source indices, pixel provenance, reference RGB,
   geometric endpoints and boundaries. Keep incomplete meshes.
7. Compute traits in cm meshes with scale multiplier exactly 1. Reuse the actual
   graph-path / single-side-area functions and historical phenotype_core. Exclude
   unreliable full-leaf values, preserve provisional observed measurements and
   actual pairing status. Provide per-leaf rotating/point-measurement viewer.
8. Single/all CLI with resume and equivalence checks; write accurate live all19
   results, index, RGB/Gaussian/Mesh/Plant/instance panels and cross-plant CSVs.
9. Validate uniform PLY scaling, camera projection invariance, metric depth and
   voxel equivalence, analytic curved-surface distances, untouched source hashes,
   source triangle/normal preservation, local links and stage completeness.

Execution duration depends on 18 native 30K trainings (existing reference run was
about 34 min). No lower-iteration results may be reported as completed 30K models.
Persistent batch jobs and live summaries must distinguish running / complete /
failed / uncalibrated rather than equating process launch with completion.
