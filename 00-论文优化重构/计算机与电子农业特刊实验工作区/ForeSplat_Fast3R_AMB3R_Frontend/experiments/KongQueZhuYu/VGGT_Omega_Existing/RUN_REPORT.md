# RUN REPORT: VGGT_Omega_Existing

- Saved views: 208; full sequence: True
- Scale: ambiguous; uncalibrated VGGT-Omega scene scale
- Geometry export: concatenation of original per-view predictions; no averaging, fusion or scale correction
- Weight SHA256: c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934
- Source commit: None
- Inference seconds: 59.61; peak GPU memory GiB: 21.68
- Inference timing scope: legacy recorded model inference; import/backprojection excluded; exact legacy stage boundaries not recorded
- Total frontend seconds: —; scope: unknown in legacy artifact; no invented combined timing
- Recorded bundle export / post-pipeline seconds: — / —; pipeline-local model / raw serialization seconds: — / —. These intervals have different scopes and are not summed as independent algorithm timings.
- Full model parameters and original model report: `run_parameters.json` and `MODEL_RUN_REPORT.md` when saved. Raw model outputs and original cameras are unchanged.

## Geometry diagnostics

- foreground_coverage: 1
- edge_coverage: 1
- edge_q85_coverage: 0.2703
- adjacent_depth_agreement_fraction: 0.9606
- adjacent_point_distance_p50_normalized: 0.02586
- rotation_step_p90_deg: 10.25
- pnp_success_rate: —
- inference_seconds: 59.61
- total_frontend_seconds: —
- peak_vram_gib: 21.68
- Whole-foreground q85 boundary coverage includes the plant, pot and blue object; shared rectangular leaf/edge ROI retention is reported separately.
- Finite foreground fraction: 1; camera count: 208.
- Registered / unmapped frames: — / —; final keyframes: —.
- PnP status scope: not applicable: pose-head predictions or SfM registration; eligible observations: 0.
- All returning local-call PnP rate: —; eligible observations: —; missing observations: —.
- Solver success and successful mapping do not establish accuracy of final fused cameras or thin-leaf geometry.
- Adjacent world-cloud NN, median / p90 normalized: 0.02014 / 0.04135. This proximity proxy is affected by density, occlusion and collapsed surfaces; it is not GT accuracy.

## Existing manual 2D GT support

- Original-view GT silhouette support recall: 0.9708; inner-edge recall: 0.209.
- This measures the original-pixel footprint of valid RAP foreground geometry. The GT masks are manual 2D references and are not measured 3D surfaces.

All per-frame metrics are in `geometry_per_frame.csv`, neighboring-view metrics in `geometry_adjacent_pairs.csv`, and aggregate values in `geometry_metrics.json`.

Coverage measures saved valid foreground pixels, not unseen surfaces. Local plane residual and depth jumps mix noise, curvature, thin surfaces and occlusion boundaries; they are not absolute reconstruction error. Confidence is not calibrated between models. This bundle has no claim of validated GGGS/COLMAP tracks and no training was started.