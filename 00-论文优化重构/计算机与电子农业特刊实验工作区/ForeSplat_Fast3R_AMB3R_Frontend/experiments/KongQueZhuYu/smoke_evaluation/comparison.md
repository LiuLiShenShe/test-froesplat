# KongQueZhuYu geometry comparison

These are diagnostics of predicted geometry. No scanned GT has been registered, no true scale correction has been applied, and no 3DGS training or PSNR evaluation has been run.

| method | status | views | foreground_coverage | edge_coverage | adjacent_depth_agreement_fraction | adjacent_point_distance_p50_normalized | rotation_step_p90_deg | pnp_success_rate | inference_seconds | peak_vram_gib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fast3R_smoke_v2 | complete | 3 | 1 | 1 | 0.9979 | 0.01166 | 1.107 | 1 | 7.267 | 4.102 |

Resolution and scope differ between the existing VGGT-Ω artifact, full-sequence frontends and any fixed-frame AMB3R OFF/ON test. Point counts alone are not a quality ranking. The confidence >1 baseline can retain almost every pixel because both confidence heads use 1+exp; geometry consistency and leaf diagnostics carry more information.

Self-reprojection is a pointmap/camera agreement diagnostic. It is algebraically zero for depth-unprojected VGGT-Ω points and must not be compared as a learned geometry accuracy score.

See `visualizations/render_parameters.json` and `docs/EVALUATION_PROTOCOL.md` for fixed plotting parameters, confidence semantics, source-pixel ROIs, visibility filtering and metric limitations.