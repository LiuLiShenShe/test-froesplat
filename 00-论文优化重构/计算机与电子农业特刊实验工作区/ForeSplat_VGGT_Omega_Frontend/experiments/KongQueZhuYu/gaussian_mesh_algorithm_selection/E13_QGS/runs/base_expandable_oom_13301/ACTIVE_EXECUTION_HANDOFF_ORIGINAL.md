# E13 active native execution handoff

Snapshot: 2026-11-09T17:31:00.531935+08:00. Actual status `running`, native iteration 9200/30000, 325,616 quadratic primitives. Terminal30K model,208-view metrics and native mesh remain pending.

The initial native-base attempt genuinely failed with CUDA OOM at iteration11700. Actual verified7K checkpoint/PLY are retained; failed logs/config/metrics are archived under `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/runs/base_initial_oom_11700`. The parent launched explicit7K resume; active native driver PID is 1534953. Do not launch a competing GPU training/viewer.

Actual PyTorch2.5.1 expandable-segments validation passed. Native base losses, optimizer, densification, quadratic representation and frozen208 input tensors remain unchanged. Seed0 native restore recovers optimizer/Gaussian state but lacks saved RNG/viewpoint-stack state; it is not claimed as one uninterrupted fresh trajectory. One rolling native checkpoint is saved every1000 iterations; fixed7K/15K/30K checkpoint/PLY stages remain separate.

Initial noncentral CUDA forward/backward validation passed (centroid error.00147358456 pixels); all208 K/W2C/P and RGB hashes match E10. Initialization uses1,252,383 existing VGGT/RAP filtered points. Official source commit74d05c945e99fcaef7afe5a8831903be71ad9b55.

Observed native numerical quality defect: rolling8K checkpoint had4 invalid primitives;9K had82 (0.00851%). Original native parameters are retained unfiltered. Exact float32 bit provenance validates original PLY/checkpoint, including NaN payloads, and reports nonfinite counts separately. Curvature shape labels exclude undefined parameters. A provenance pass does not mean numerical stability.

Current metrics CSV contains the retained successful100–7000 prefix followed by the actual resumed7100+ trajectory in strict iteration order. Entire failed11700 attempt remains separately archived. Total attempt cost, successful saved prefix plus resumed cost, and resumed segment cost are reported separately. Successful-process peak VRAM is not presented as the initial failed attempt peak.

Remaining native stages: actual terminal30000 PLY/checkpoint → exact provenance and native loss-log audit → native primitive diagnostics → all208 floatRGB/alpha/depth/normal with separate exact official mesh flags → exact shared8 free cameras and fixed historical leaf windows → unified208 metrics → labelled center PointCloud PLY → official QGS bounded TSDF raw/upstreamtop10/comparablepost → shared top/oblique inspection poses → actual nativeHTTP RGB/alpha/normal/depth/gray/orbit/zoom/180° smoke → actual path/metrics summary.

Render/evaluation requires actual finite RGB/alpha/native sanitized depth. Native weighted normal/raw median defects are preserved and counted; native depth sanitization is the official renderer behavior. Viewer retains native invalid primitives and marks any visible nonfinite normal magenta instead of silent uint8 conversion. Native HTTP smoke saves actual responses and closes its own server before E14.

Authoritative existing files:

- `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/STATUS.json`
- `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/queue_status.json`
- `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/active_driver.json`
- `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/logs/train_qgs.log`
- `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/train_metrics.csv`
- `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/training_execution_records.json`
- `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/recovery/latest_native_state.pth`
- `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/recovery/latest_native_state.provenance.json`
- `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/recovery/native_numerical_observation_9000.json`
- `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/partial_checkpoint_provenance_validation.json`
- `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/checkpoints/chkpnt7000.pth`
- `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/checkpoints/point_cloud/iteration_7000/point_cloud.ply`

Native QGS PLY uses scale_0..5 with signed quadratic decoder; ordinary3DGS/SuperSplat cannot reproduce its primitive geometry. The optional center XYZRGB export is a PointCloud, never a Gaussian or surface. EO/appearance disabled.

No historical or shared environment files were modified. Own uv environment and patched copy are isolated. Exact CPU get_mask diagnostic records original-RGB IO overhead, which is not pure quadratic CUDA compute.

Sources: actual local records and [pinned official QGS](https://github.com/will-zzy/QGS/tree/74d05c945e99fcaef7afe5a8831903be71ad9b55); Bocha unavailable due HTTP403 quota.
