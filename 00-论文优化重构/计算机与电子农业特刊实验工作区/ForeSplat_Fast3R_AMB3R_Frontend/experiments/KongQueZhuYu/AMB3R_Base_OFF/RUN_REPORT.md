# RUN REPORT: AMB3R_Base_OFF

- Saved views: 208; full sequence: True
- Scale: normalized raw pointmaps and camera poses; separate predicted metric depth and explicit metric derivatives
- Geometry export: concatenation of original per-view predictions; no averaging, fusion or scale correction
- Weight SHA256: 2d9342282cce0a20e1074ad1513d9a18c3d100dc06c734a4a11f2d1baeea5239
- Source commit: 92c4081f910f98e683503092b85301861519175e
- Inference seconds: 43.65; peak GPU memory GiB: 23.67
- Inference timing scope: original AMB3R.forward including camera/intrinsic/depth/point heads and configured backend, plus CUDA synchronization; excludes model load, input preparation/H2D, CPU diagnostics and output serialization
- Total frontend seconds: —; scope: not measured as a combined frontend time
- Recorded bundle export / post-pipeline seconds: — / —; pipeline-local model / raw serialization seconds: — / —. These intervals have different scopes and are not summed as independent algorithm timings.
- Full model parameters and original model report: `run_parameters.json` and `MODEL_RUN_REPORT.md` when saved. Raw model outputs and original cameras are unchanged.

## Geometry diagnostics

- foreground_coverage: 0.2068
- edge_coverage: 0.08107
- edge_q85_coverage: 0.07394
- adjacent_depth_agreement_fraction: —
- adjacent_point_distance_p50_normalized: —
- rotation_step_p90_deg: 38.62
- pnp_success_rate: —
- inference_seconds: 43.65
- total_frontend_seconds: —
- peak_vram_gib: 23.67
- Whole-foreground q85 boundary coverage includes the plant, pot and blue object; shared rectangular leaf/edge ROI retention is reported separately.
- Finite foreground fraction: 1; camera count: 208.
- Registered / unmapped frames: — / —; final keyframes: —.
- PnP status scope: not applicable: pose-head predictions or SfM registration; eligible observations: 0.
- All returning local-call PnP rate: —; eligible observations: —; missing observations: —.
- Solver success and successful mapping do not establish accuracy of final fused cameras or thin-leaf geometry.
- Adjacent world-cloud NN, median / p90 normalized: 0.05705 / 0.1255. This proximity proxy is affected by density, occlusion and collapsed surfaces; it is not GT accuracy.

## Learned metric-scale prediction

- Normalized-to-metric factor predicted by the metric head: 0.07504
- Frame factors p10 / median / p90: 0.04244 / 0.07551 / 0.2102
- Official depth_metric is a learned metre prediction without measured verification. Direct point-head, depth and pose gauges are not demonstrated equal. Multiplying direct pointmaps by a depth-head factor is a conditional derivative, not validated metric geometry. Raw global pointmaps/camera translations retain their original normalized scale.

## Point-head / depth-head agreement

- All-foreground point-head/camera reprojection: 273.6px; depth-unprojection algebraic self-reprojection: 0.8722px.
- These are consistency checks between saved outputs. They do not isolate camera error or establish accuracy against real 3D geometry. See `head_consistency_diagnostic.json`.

## Original depth branch control

- See `depth_unprojection_diagnostic.json`, `depth_diagnostic_native_depth_confidence_pairs.csv` and the experiment-level `amb3r_depth_channel_comparison.csv` for the actual full208 OFF/ON depth-unprojection comparison.
- Native depth-confidence selection evaluates that original branch. The separate world-confidence selection is a geometry-control experiment, not a claim that the point-head confidence describes all AMB3R geometry.
- Depth-unprojected points are already world-frame points and are not a local point head. Algebraic self-reprojection does not validate real camera/depth accuracy. Official depth_metric metre predictions remain unverified by external length measurements.

## Existing manual 2D GT support

- Original-view GT silhouette support recall: 0.06541; inner-edge recall: 0.01054.
- This measures the original-pixel footprint of valid RAP foreground geometry. The GT masks are manual 2D references and are not measured 3D surfaces.

All per-frame metrics are in `geometry_per_frame.csv`, neighboring-view metrics in `geometry_adjacent_pairs.csv`, and aggregate values in `geometry_metrics.json`.

Coverage measures saved valid foreground pixels, not unseen surfaces. Local plane residual and depth jumps mix noise, curvature, thin surfaces and occlusion boundaries; they are not absolute reconstruction error. Confidence is not calibrated between models. This bundle has no claim of validated GGGS/COLMAP tracks and no training was started.