# RUN REPORT: Fast3R_AMB3R_SfM

- Saved views: 208; full sequence: True
- Scale: ambiguous normalized pointmap/pose scale; not verified metres
- Geometry export: Official SfMemory confidence weighted fusion only; no additional frontend fusion
- Weight SHA256: 1357d42e74539ba03510ea1986c7464a7facf17157779e2b7ebd6284e79e0427
- Source commit: 33104d4b5b8df43795ecded236194958bbdac572
- Inference seconds: 2074; peak GPU memory GiB: 4.606
- Inference timing scope: wall time of features/clustering/mapping including observation and raw export
- Total frontend seconds: 2466; scope: recorded runner wall time from model loading/weight hashing through CPU bundle completion; excludes earlier shared input preparation, later persistent copy/verification and evaluation
- Recorded bundle export / post-pipeline seconds: 276.1 / 328.8; pipeline-local model / raw serialization seconds: 1261 / 99.18. These intervals have different scopes and are not summed as independent algorithm timings.
- Full model parameters and original model report: `run_parameters.json` and `MODEL_RUN_REPORT.md` when saved. Raw model outputs and original cameras are unchanged.

## Geometry diagnostics

- foreground_coverage: 1
- edge_coverage: 1
- edge_q85_coverage: 0.1749
- adjacent_depth_agreement_fraction: 0.6944
- adjacent_point_distance_p50_normalized: 0.1091
- rotation_step_p90_deg: 78.24
- pnp_success_rate: 1
- inference_seconds: 2074
- total_frontend_seconds: 2466
- peak_vram_gib: 4.606
- Whole-foreground q85 boundary coverage includes the plant, pot and blue object; shared rectangular leaf/edge ROI retention is reported separately.
- Finite foreground fraction: 1; camera count: 208.
- Registered / unmapped frames: 208 / 0; final keyframes: 35.
- PnP status scope: last accepted local Fast3R PnP contributor per frame; selected contributors only; not final fused-pose quality or all-local-call success rate; eligible observations: 208.
- All returning local-call PnP rate: 1; eligible observations: 9885; missing observations: 0.
- Solver success and successful mapping do not establish accuracy of final fused cameras or thin-leaf geometry.
- Adjacent world-cloud NN, median / p90 normalized: 0.04646 / 0.1051. This proximity proxy is affected by density, occlusion and collapsed surfaces; it is not GT accuracy.

## Existing manual 2D GT support

- Original-view GT silhouette support recall: 0.9708; inner-edge recall: 0.222.
- This measures the original-pixel footprint of valid RAP foreground geometry. The GT masks are manual 2D references and are not measured 3D surfaces.

All per-frame metrics are in `geometry_per_frame.csv`, neighboring-view metrics in `geometry_adjacent_pairs.csv`, and aggregate values in `geometry_metrics.json`.

Coverage measures saved valid foreground pixels, not unseen surfaces. Local plane residual and depth jumps mix noise, curvature, thin surfaces and occlusion boundaries; they are not absolute reconstruction error. Confidence is not calibrated between models. This bundle has no claim of validated GGGS/COLMAP tracks and no training was started.