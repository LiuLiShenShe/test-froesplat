# E13 native QGS actual execution summary

Actual official native training completed at 30,000 iterations, seed 0, physical GPU1, producing 532,735 quadratic primitives. All 208 corrected cameras and training RGB hashes match the frozen E10 reference.

| Unified 208-view metric | Mean |
|---|---:|
| full_psnr | 22.8956023 |
| fg_mask_psnr | 18.2985798 |
| ssim | 0.858575275 |
| lpips_alex | 0.134406971 |
| fg_gradient_retention | 0.699031647 |
| foreground_alpha_mean | 0.99951904 |
| background_alpha_mean | 0.999343251 |
| alpha_mask05_iou | 0.310168997 |

RGB metrics use unclipped native float renders saved to NPZ, then clip RGB to [0,1] for PSNR, SSIM, LPIPS and gradients. Raw means no EO or appearance correction. SSIM and Alex LPIPS use exactly the historical E10/E11 evaluation recipe.

Recorded training wall time across attempts: 344.810 minutes. Saved prefix plus successful resumed training: 344.810 minutes. Successful segment: 344.810 minutes. Successful-process peak allocated VRAM: 3.459 GiB; peak reserved: 3.732 GiB.

Primary native preset: Mip; resume checkpoint: None. Archived base-preset attempts cost 342.196 actual training minutes. Two actual base attempts failed at11700 and13301; failed native models and original logs are separately retained. The13301 failed PTH is unreadable by Python ZIP and the native PyTorch archive reader; its separate native PLY has the complete expected payload length. The failed PTH is not a resume checkpoint. A fresh Mip run uses the same VGGT initialization, cameras and training RGB; no base checkpoint is promoted or resumed across different native presets. Native resumed captures, if any, lack RNG/viewpoint-stack state and precede the optimizer step; uninterrupted trajectory equivalence is never inferred.

Native terminal parameter audit: 3,456 nonfinite primitives (0.00648727791 of the model). The original native checkpoint and PLY retain them unchanged; exact float32 bit equality validates provenance, while any nonfinite count is a native numerical-stability defect and is not reported as an all-fields-finite success. Actual rendered outputs and metrics are validated separately. Derived shape categories exclude undefined parameters.

Wall time includes the frozen original-RGB adapter and upstream iterable get_mask IO. The separate native_mask_IO_diagnostic.json records the exact CPU mask getter on one representative source view (no GPU); this overhead is not intrinsic quadratic primitive compute and is not a pure-GPU algorithm speed measurement.

Actual native Mip: depth_ratio=1.0, lambda_dist=10.0, lambda_normal=0.015, lambda_dssim=0.2; densify threshold=1.0, percent_dense=0.0001, interval=100, from=500, until=15000, opacity reset=3000. Strict geometry/NCC branch: 30000<iteration<30000. No extra depth/normal prior or EO.

The raw training CSV preserves upstream conditional geo/NCC Python variables, which can hold prior-sample values when a branch contributes nothing. native_loss_log_audit.json records this distinction; terminal iteration30000 has zero actual geo/NCC contribution even if those raw logged variables remain nonzero. Actual total_loss is authoritative.

The native PLY has scale_0..5 and retains quadratic primitive intersections. The decoder is s=exp(scale0..2)*tanh(scale3..5); the third signed scale controls local quadratic curvature rather than ordinary 3DGS thickness. Native Gaussian/mean curvature diagnostics are saved separately; they do not establish ground-truth geometric accuracy.

Native bounded TSDF uses official SH0/kernel_size=-999/stop_z_gradient=True extraction maps, the corrected cameras, original RAP foreground fusion constraint and shared voxel/truncation. Original SH0 float RGB remains in NPZ; mesh RGB8 input is clipped to[0,1] to prevent out-of-range SH colors wrapping under the official uint8 cast. Depth and TSDF fusion are unchanged. The raw mesh, official top10-component post mesh, and comparable >=50-face-only post mesh are preserved. No smoothing or hole filling is applied. Exact counts and extraction time are in meshes/native_TSDF/complete.json.

| Actual artifact | Bytes |
|---|---:|
| `/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/checkpoints/point_cloud/iteration_30000/point_cloud.ply` | 138512700 |
| `/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/checkpoints/chkpnt30000.pth` | 402755781 |
| `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/meshes/native_TSDF/raw.ply` | 106381136 |
| `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/meshes/native_TSDF/post.ply` | 93814041 |
| `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/meshes/native_TSDF/official_top10_post.ply` | 86834501 |
| `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/viewer_exports/quadratic_centers_rgb_point_cloud.ply` | 7939365 |
| `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/native_geometry/quadratic_primitive_parameters.npz` | 67624783 |

The center XYZRGB PLY is a Point Cloud PLY for center inspection; it is not a Gaussian PLY or quadratic surface approximation. Inspect the native six-scale PLY with scripts/native_qgs_viewer.py using the original quadratic CUDA renderer. The actual terminal model-backed HTTP smoke verified RGB, alpha, unit camera-normal, camera-Z depth, gray primitive, orbit, zoom and 180° view-only responses. The test server was terminated before the following experiment. Optional 180° display transforms never modify original geometry.

This execution record makes no geometric superiority claim from PSNR alone. Final ordering must use the common 208-view silhouette/coverage/roughness/hole diagnostics and actual fixed/free visual comparisons with E10/E11/E12/E14. No same-plant scan reference exists.

Source: [official QGS pinned commit](https://github.com/will-zzy/QGS/tree/74d05c945e99fcaef7afe5a8831903be71ad9b55), local actual completion records and model files. Bocha search was unavailable with HTTP403; its reason was not established; no training-data-only claim is used.
