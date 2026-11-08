# RUN REPORT: Fast3R_FP32

- Saved views: 208; full sequence: True
- Scale: ambiguous; arbitrary learned scene scale, not metres
- Geometry export: concatenation of original per-view predictions; no averaging, fusion or scale correction
- Weight SHA256: 1357d42e74539ba03510ea1986c7464a7facf17157779e2b7ebd6284e79e0427
- Source commit: 33104d4b5b8df43795ecded236194958bbdac572
- Inference seconds: 23.15; peak GPU memory GiB: 19.23
- Inference timing scope: official multi-view inference call and GPU synchronization; excludes model loading, input preparation, PnP and output serialization
- Total frontend seconds: 38.39; scope: network inference plus official CPU PnP; excludes input preparation, model loading, geometry diagnostics and Bundle serialization
- Recorded bundle export / post-pipeline seconds: — / —; pipeline-local model / raw serialization seconds: — / —. These intervals have different scopes and are not summed as independent algorithm timings.
- Full model parameters and original model report: `run_parameters.json` and `MODEL_RUN_REPORT.md` when saved. Raw model outputs and original cameras are unchanged.

## Geometry diagnostics

- foreground_coverage: 1
- edge_coverage: 1
- edge_q85_coverage: 0.1183
- adjacent_depth_agreement_fraction: 0.7844
- adjacent_point_distance_p50_normalized: 0.05768
- rotation_step_p90_deg: 109.1
- pnp_success_rate: 1
- inference_seconds: 23.15
- total_frontend_seconds: 38.39
- peak_vram_gib: 19.23
- Whole-foreground q85 boundary coverage includes the plant, pot and blue object; shared rectangular leaf/edge ROI retention is reported separately.
- Finite foreground fraction: 1; camera count: 208.
- Registered / unmapped frames: — / —; final keyframes: —.
- PnP status scope: official direct Fast3R camera solver status; not registration quality; eligible observations: 208.
- All returning local-call PnP rate: —; eligible observations: —; missing observations: —.
- Solver success and successful mapping do not establish accuracy of final fused cameras or thin-leaf geometry.
- Adjacent world-cloud NN, median / p90 normalized: 0.02196 / 0.05667. This proximity proxy is affected by density, occlusion and collapsed surfaces; it is not GT accuracy.

## Existing manual 2D GT support

- Original-view GT silhouette support recall: 0.9708; inner-edge recall: 0.222.
- This measures the original-pixel footprint of valid RAP foreground geometry. The GT masks are manual 2D references and are not measured 3D surfaces.

All per-frame metrics are in `geometry_per_frame.csv`, neighboring-view metrics in `geometry_adjacent_pairs.csv`, and aggregate values in `geometry_metrics.json`.

Coverage measures saved valid foreground pixels, not unseen surfaces. Local plane residual and depth jumps mix noise, curvature, thin surfaces and occlusion boundaries; they are not absolute reconstruction error. Confidence is not calibrated between models. This bundle has no claim of validated GGGS/COLMAP tracks and no training was started.