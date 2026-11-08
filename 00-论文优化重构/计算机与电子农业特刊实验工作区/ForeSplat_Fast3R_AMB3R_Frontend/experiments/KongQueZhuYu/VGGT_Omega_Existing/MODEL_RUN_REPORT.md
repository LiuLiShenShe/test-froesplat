# RUN REPORT: VGGT_Omega_Existing

- Saved views: 208; full sequence: True
- Scale: ambiguous; uncalibrated VGGT-Omega scene scale
- Geometry export: concatenation of original per-view predictions; no averaging, fusion or scale correction
- Weight SHA256: c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934
- Source commit: None
- Inference seconds: 59.61; peak GPU memory GiB: 21.68

## Geometry diagnostics

- foreground_coverage: 1
- edge_coverage: 1
- edge_q85_coverage: 0.2703
- adjacent_depth_agreement_fraction: 0.9606
- adjacent_point_distance_p50_normalized: 0.02586
- rotation_step_p90_deg: 10.25
- pnp_success_rate: —
- inference_seconds: 59.61
- peak_vram_gib: 21.68

All per-frame metrics are in `geometry_per_frame.csv`, neighboring-view metrics in `geometry_adjacent_pairs.csv`, and aggregate values in `geometry_metrics.json`.

Coverage measures saved valid foreground pixels, not unseen surfaces. Local plane residual and depth jumps mix noise, curvature, thin surfaces and occlusion boundaries; they are not absolute reconstruction error. Confidence is not calibrated between models. This bundle has no claim of validated GGGS/COLMAP tracks and no training was started.