# RUN REPORT: Fast3R_smoke_v2

- Saved views: 3; full sequence: False
- Scale: ambiguous; arbitrary learned scene scale, not metres
- Geometry export: concatenation of original per-view predictions; no averaging, fusion or scale correction
- Weight SHA256: 1357d42e74539ba03510ea1986c7464a7facf17157779e2b7ebd6284e79e0427
- Source commit: 33104d4b5b8df43795ecded236194958bbdac572
- Inference seconds: 7.267; peak GPU memory GiB: 4.102

## Geometry diagnostics

- foreground_coverage: 1
- edge_coverage: 1
- adjacent_depth_agreement_fraction: 0.9979
- adjacent_point_distance_p50_normalized: 0.01166
- rotation_step_p90_deg: 1.107
- pnp_success_rate: 1
- inference_seconds: 7.267
- peak_vram_gib: 4.102

All per-frame metrics are in `geometry_per_frame.csv`, neighboring-view metrics in `geometry_adjacent_pairs.csv`, and aggregate values in `geometry_metrics.json`.

Coverage measures saved valid foreground pixels, not unseen surfaces. Local plane residual and depth jumps mix noise, curvature, thin surfaces and occlusion boundaries; they are not absolute reconstruction error. Confidence is not calibrated between models. This bundle has no claim of validated GGGS/COLMAP tracks and no training was started.