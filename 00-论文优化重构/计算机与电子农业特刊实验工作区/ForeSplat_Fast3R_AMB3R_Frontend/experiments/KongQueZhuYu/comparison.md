# KongQueZhuYu geometry comparison

Measurements come from actual saved208-frame outputs. No3D scan registration, scale correction,3DGS training or PSNR evaluation was performed. The final metadata refresh reused all completed geometry measurements and loaded no prediction arrays.

| method | status | views | foreground_coverage | edge_coverage | edge_q85_coverage | adjacent_depth_agreement_fraction | adjacent_point_distance_p50_normalized | rotation_step_p90_deg | pnp_success_rate | inference_seconds | total_frontend_seconds | peak_vram_gib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VGGT_Omega_Existing | complete | 208 | 1 | 1 | 0.2703 | 0.9606 | 0.02586 | 10.25 | — | 59.61 | — | 21.68 |
| Fast3R | complete | 208 | 1 | 1 | 0.1191 | 0.8234 | 0.05513 | 99.25 | 1 | 20.09 | 29.28 | 13.04 |
| AMB3R_Base_OFF | complete | 208 | 0.2068 | 0.08107 | 0.07394 | — | — | 38.62 | — | 43.65 | — | 23.67 |
| AMB3R_Base_ON | complete | 208 | 0.9482 | 0.9063 | 0.2886 | 0.8834 | 0.4696 | 36.43 | — | 120.1 | — | 30.83 |
| AMB3R_SfM | complete | 208 | 1 | 1 | 0.3222 | 0.8367 | 0.1198 | 10.73 | — | 2645 | — | 11.62 |
| Fast3R_AMB3R_SfM | complete | 208 | 1 | 1 | 0.1749 | 0.6944 | 0.1091 | 78.24 | 1 | 2074 | 2466 | 4.606 |
| Fast3R_FP32 | complete | 208 | 1 | 1 | 0.1183 | 0.7844 | 0.05768 | 109.1 | 1 | 23.15 | 38.39 | 19.23 |

`edge_q85_coverage` is the complete foreground-mask boundary, including the plant, pot and blue object. The shared source0000 leaf and leaf-edge-neighborhood ROIs are rectangular windows; their q85 coverage describes retained foreground pixels, not annotated3D leaf-edge accuracy.

The existing VGGT-Ω depth reference retains its higher input resolution. Confidence >1, q85 ties and floating-point saturation at1 affect retained pixels; confidence heads are not calibrated between models. Point count and small WorldNN distance alone are not quality rankings.

Self-reprojection measures saved pointmap/camera agreement. It is algebraic for depth-unprojected VGGT-Ω points and does not establish measured geometry accuracy. AMB3R point/depth/pose head gauges are not assumed equal.

## Registration and observed camera solvers

| method | finite_camera_count | registered_frames | unmapped_frames | keyframe_count | pnp_success_rate | pnp_eligible_status_count | all_local_call_pnp_success_rate | all_local_call_pnp_eligible_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VGGT_Omega_Existing | 208 | — | — | — | — | 0 | — | — |
| Fast3R | 208 | — | — | — | 1 | 208 | — | — |
| AMB3R_Base_OFF | 208 | — | — | — | — | 0 | — | — |
| AMB3R_Base_ON | 208 | — | — | — | — | 0 | — | — |
| AMB3R_SfM | 208 | 208 | 0 | 73 | — | 0 | — | — |
| Fast3R_AMB3R_SfM | 208 | 208 | 0 | 35 | 1 | 208 | 1 | 9885 |
| Fast3R_FP32 | 208 | — | — | — | 1 | 208 | — | — |

Pure Fast3R PnP is the prescribed direct camera solver. Hybrid per-frame PnP is the last accepted local contributor; its all-local-call column covers every observed returning local call, including predictions not retained in final memory. Native AMB3R-SfM PnP is not applicable. Registered frames and solver success do not validate final fused cameras or thin-leaf surfaces. Final keyframe counts use saved keyframes.json, excluding temporary coarse-stage counts.

## Measured timing scopes

| method | inference_seconds | total_frontend_seconds | local_model_seconds | local_raw_export_seconds | bundle_export_seconds | peak_vram_gib |
| --- | --- | --- | --- | --- | --- | --- |
| VGGT_Omega_Existing | 59.61 | — | — | — | — | 21.68 |
| Fast3R | 20.09 | 29.28 | — | — | — | 13.04 |
| AMB3R_Base_OFF | 43.65 | — | — | — | — | 23.67 |
| AMB3R_Base_ON | 120.1 | — | — | — | — | 30.83 |
| AMB3R_SfM | 2645 | — | 1165 | 1023 | — | 11.62 |
| Fast3R_AMB3R_SfM | 2074 | 2466 | 1261 | 99.18 | 276.1 | 4.606 |
| Fast3R_FP32 | 23.15 | 38.39 | — | — | — | 19.23 |

Fast3R total is model+PnP and excludes model loading/export. AMB3R Base records official forward only. Native SfM pipeline time includes local serialization and registration; its older runner did not measure final Bundle export or a complete total. Hybrid total starts at model loading after input preparation and ends at CPU Bundle completion; later SSD-to-HDD copying is excluded. Local-model/raw-export intervals are components with distinct scopes and are not summed as separate algorithm timings. Shared HDD contention and native-HDD versus hybrid-SSD output prevent ranking algorithm speed solely by wall time.

- VGGT_Omega_Existing: inference scope: legacy recorded model inference; import/backprojection excluded; exact legacy stage boundaries not recorded; total scope: unknown in legacy artifact; no invented combined timing.
- Fast3R: inference scope: official multi-view inference call and GPU synchronization; excludes model loading, input preparation, PnP and output serialization; total scope: network inference plus official CPU PnP; excludes input preparation, model loading, geometry diagnostics and Bundle serialization.
- AMB3R_Base_OFF: inference scope: original AMB3R.forward including camera/intrinsic/depth/point heads and configured backend, plus CUDA synchronization; excludes model load, input preparation/H2D, CPU diagnostics and output serialization; total scope: not measured as a combined frontend time.
- AMB3R_Base_ON: inference scope: original AMB3R.forward including camera/intrinsic/depth/point heads and configured backend, plus CUDA synchronization; excludes model load, input preparation/H2D, CPU diagnostics and output serialization; total scope: not measured as a combined frontend time.
- AMB3R_SfM: inference scope: wall time of features/clustering/mapping including observation and raw export; total scope: not measured as a combined frontend time.
- Fast3R_AMB3R_SfM: inference scope: wall time of features/clustering/mapping including observation and raw export; total scope: Measured from model loading after inputs were prepared through CPU bundle export; includes official pipeline and observation/raw export; excludes original input reading/preparation and final SSD-to-HDD persistence copy.
- Fast3R_FP32: inference scope: official multi-view inference call and GPU synchronization; excludes model loading, input preparation, PnP and output serialization; total scope: network inference plus official CPU PnP; excludes input preparation, model loading, geometry diagnostics and Bundle serialization.

## Same source frames and same evaluation pairs

This completed control uses only208 source cameras registered and finite in every method. The existing bidirectional pair measurements are reused in original order, preserving source gaps0044/0206. Primary coverage continues to count every actual input; unmapped failures would remain visible rather than being removed.

| method | common_registered_frame_count | foreground_coverage | edge_q85_coverage | adjacent_depth_agreement_fraction | adjacent_point_distance_p50_normalized | projection_overlap_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| VGGT_Omega_Existing | 208 | 1 | 0.2703 | 0.9606 | 0.02586 | 0.9049 |
| Fast3R | 208 | 1 | 0.1191 | 0.8234 | 0.05513 | 0.8045 |
| AMB3R_Base_OFF | 208 | 0.2068 | 0.07394 | — | — | 0 |
| AMB3R_Base_ON | 208 | 0.9482 | 0.2886 | 0.8834 | 0.4696 | 0.1849 |
| AMB3R_SfM | 208 | 1 | 0.3222 | 0.8367 | 0.1198 | 0.8533 |
| Fast3R_AMB3R_SfM | 208 | 1 | 0.1749 | 0.6944 | 0.1091 | 0.5441 |
| Fast3R_FP32 | 208 | 1 | 0.1183 | 0.7844 | 0.05768 | 0.7805 |

The five manual2D GT masks/photos are pixel-support references, not registered3D surfaces. Full208 OFF/ON same-pixel controls and the separate original depth-unprojection branch remain in amb3r_off_on_geometry_controls.csv and amb3r_depth_channel_comparison.csv. No input/control evaluation was rerun during this refresh.

Sources: final local Bundle metadata, pose_status.json/keyframes.json, geometry_metrics.json and saved pair CSVs. See docs/EVALUATION_PROTOCOL.md, visualizations/render_parameters.json and final_analysis.md for definitions, fixed rendering and the reviewed technical decision.
