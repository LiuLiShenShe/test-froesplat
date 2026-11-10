# ForeSplat execution plan (2026-11-09)

Status: active; no new experiment is claimed complete before its actual artifacts and validation exist.

Frozen input: KongQueZhuYu, source IDs 0–209 excluding 44/206 (208 frames), existing standard VGGT predictions, RAP applied after Point Map, corrected cameras, seed 0, physical GPU1, uv. No frontend or segmentation rerun. Historical artifacts are read-only.

1. Inventory E1–E11/M1–M5 (first): inspect PLY headers, checkpoints, actual sizes and resolved paths; retain original and unified metrics separately. Output artifact_index/* and history CSVs. Estimate 15–30 minutes.
2. P1: official GGGS tetrahedra; audit mask grid pixel convention and camera; preserve raw, 10-step binary-search, post mesh and extraction parameters. Validate topology, 208 projections, eight views. Runtime depends on tetrahedralization.
3. P2: E10 independent model; score multi-view mask support, visibility/depth, scale/anisotropy and local support; protect occluded leaf edges; save original/cleaned/removed native Gaussian PLY, renders and both mesh algorithms. Validate foreground loss separately from background alpha and mesh changes.
4. P3: official GS2Mesh stereo pipeline using GGGS-generated stereo views; preserve official stereo estimator/weights, confidence and fusion. Preparation can overlap CPU-only inventory. Record official commit, adaptation and weights. No substitute 3DGS training.
5. P4: same E11 checkpoint; expected/median depth × RAP/alpha validity × small shared scale variations; raw/post retained. Diagnose holes from depth through fusion through post-processing separately.
6. E12: independent GGGS source snapshot, only EO enabled; apply corrected RGB to L1 and SSIM with E10 weights. Same initialization/config 30K. Estimated 40–70 minutes plus evaluation and meshing. Record raw/corrected per-view exposure and geometry evidence.
7. E13: fixed official QGS source; retain quadratic representation and native optimization, adapt corrected cameras/COLMAP. Record depth_ratio/lambda_dist/densification; train to official terminal state, native mesh. Native vs approximate viewer export labelled explicitly. Runtime established from actual run.
8. E14: fixed official 2d-sugar; preserve each stage (priors, initialization, 2DGS, pruning, initial mesh, refinement), identify prior source/scale. Native stage artifacts and textured dependencies indexed. Runtime established from actual run.
9. Evaluate all complete historical/new models with 208 same-source views; report differing camera/frontend/background conditions; distinguish open leaf boundaries from enclosed projected missing areas (no boundary-edge count as hole count).
10. Produce fixed-eight-view RGB/alpha/depth/normal and gray/normal/color/wireframe panels plus spatial reflection analysis. Use actual renders, never generated illustrative substitutes.
11. Verify ALL_ARTIFACTS.csv paths/types/sizes; write analysis and final selection only after model execution/evaluation. Choose photo, geometry and default routes separately under proxy accuracy limits.

## Provenance and environment
- Required historical reports read. E10 GGGS source reference: 23b418c2f0f19b30b251acad6fe13445dae968c7.
- GPU1 currently shared with unrelated GWHD detection run; never terminate that process. Record competition for timing interpretation.
- Bocha search attempted: HTTP403 insufficient balance/package quota. Official repositories read directly. Context7 GGGS library resolution returned irrelevant GGRS matches; no unrelated library docs used.
- User-referenced /home/test/.codex/RTK.md absent.

## Validation / recovery
Every command/log/commit/patch and native completion state saved in new directory. Required checks: camera projection (noncentral principal point/half-pixel), native checkpoint restoration, deterministic input hashes, final iterations, model property headers, nonempty mesh, all 208 IDs, raw/post preservation, exact exposure objective. Failures get actual traceback and status; missing artifacts have empty paths, not fabricated files. Stages resume from verified completed output without overwriting historical files.

