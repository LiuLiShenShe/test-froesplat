# E13 当前正式结果与接口适配

正式终态已完成：官方 **Mip** 预设、Seed0、统一208图像及修正相机、30,000迭代。实际配置以[actual_config.yaml](actual_config.yaml)为准：depth_ratio=1、lambda_dist=10、lambda_normal=.015、densify_grad_threshold=1、增密至15K。原生六尺度Quadratic Primitive及原生CUDA交点/优化/增密保留。原生模型含3,456个非有限基元，完整保留并单独报告，不用有限中心点云冒充模型。

最初base预设在11,700步OOM，从真实7K恢复的base尝试在13,301步再次OOM。随后主Mip运行从原VGGT初始化和Seed0重新开始30K；它没有恢复base权重。失败状态、原配置、日志及成本均在runs/归档。本页以下base“主方案”和“正在恢复”的文字是原准备记录，对应这些失败尝试，不能作为当前Mip终态配置。

[正式运行与真实路径](E13_RUN_SUMMARY.md) · [全部官方预设及实际差异](OFFICIAL_NATIVE_PRESETS.csv) · [预设适用范围](NATIVE_PRESET_SCOPE.md) · [恢复记录](RECOVERY_REPORT.md) · [实际30K来源/数值核验](checkpoint_provenance_validation.json)。

## 原准备记录（base失败尝试；接口修复仍适用于Mip）

# E13 QGS native run and recovery status

The initial official-base native run failed with CUDA OOM at iteration11,700 and11,712,536 primitives. The actual7K native PLY/checkpoint remain at their original absolute paths and passed CPU bitwise/finite validation. Failure metrics, configs and logs are retained under runs/base_initial_oom_11700. The parent launched an explicit native7K resume with PyTorch2.5.1 expandable_segments and passive rolling native checkpoints. Actual GPU allocator validation passed; current progress is recorded in STATUS.json. This is an independently seeded resumed native trajectory, not an uninterrupted fresh run. Native hyperparameters, quadratic representation, input RGB/K and optimization remain unchanged.


Official source: [will-zzy/QGS, commit 74d05c945e99fcaef7afe5a8831903be71ad9b55](https://github.com/will-zzy/QGS/tree/74d05c945e99fcaef7afe5a8831903be71ad9b55). The official working tree is clean. Full source, configuration, quadratic CUDA implementation, and native mesh extraction were read locally. Context7 resolution was attempted twice (Quadratic Gaussian Splatting and QGS), with no matching indexed official project. Bocha search is unavailable (parent observed HTTP403 insufficient quota); facts below derive from official source, not training data.

## Native configuration

Use the official generic config/base.yaml as the primary reproducible configuration, with 208 frozen COLMAP cameras and source RGB and no native alpha-training loss. Base optimizer remains unchanged: iterations 30000; depth_ratio=0; lambda_dssim=.2; lambda_dist=50000; lambda_normal=.05; percent_dense=.0001; densify_grad_threshold=.5; densification_interval100; densify_from_iter500; densify_until_iter15000; opacity reset3000. Depth-distortion starts after3000, normal after7000. Native multiview NCC(.15) and geometry(.03) losses apply from7000 until30000 with eight neighbors, preserving native strategy. External depth/normal priors are absent. EO/native appearance disabled as in base.

The TNT preset has different .1/1000/.3/.0005 depth_ratio/lambda_dist/densification/percent_dense; it is recorded as an alternative preset, not silently substituted or equalized with E11 weights. Official README recommends depth_ratio=1 when prioritizing rendering, while base uses0; this experiment explicitly reports actual0.

## Essential execution fixes

- Official train.py __main__ has training(...) commented out and instead runs DTU/TNT evaluation after printing Training complete. The experiment driver directly calls the **unchanged official numerical training function**. Completion is determined from actual iteration logs and native outputs.
- Original camera scales cx/cy without pixel-center terms and uses centered P; CUDA/rays/depth grids use+.5. Adapted source uniformly uses integer pixel centers, resized principal point (cx+.5)*scale-.5, noncentral P, matched CUDA forward/backward rays and NCC sampling. The quadratic intersection equations, signed curvature/scaling, cone bounding box, reordering, loss formulas, optimizer and densification remain unchanged.
- Original QGS preprocesses images using cv2 resize+recompression and writes into input tree. A runtime Scene.load_images adapter feeds original frozen images to the same PIL resize as E10/E11. All208 RGB tensor hashes, K/W2C/P and IDs are verified before training.
- QGS native checkpoint saving is added passively through Scene.save wrapper (native capture()+iteration), alongside native PLY. Native PLY contains scale_0..5; it is **not** an ordinary 3DGS Gaussian model. Optional center XYZRGB export is labelled Point Cloud PLY.
- Missing cstdint include fixed for CUDA12.6. First failed build log and retry retained; include-only change has no numerical effect.

## Environment and output stages

Own uv-created .venv, Python3.12, Torch2.5.1+cu121 reused read-only from historical E11 site-packages via .pth. QGS CUDA and native simple-knn extensions install only in E13_QGS/.venv. NumPy1.26.4 and OmegaConf2.3.0 installed locally. No shared environment mutation. Compiler CUDA12.6, architecture8.6, MAX_JOBS4. Actual environment import check passed. CPU-only validation also passed for all208 RGB float tensor SHA and canonical K (maximum K error4.55e-13;39.94s). Actual physical GPU forward/backward validation passed (noncentral alpha-centroid error .001474 pixels; depth-ray error2.98e-8; Open3D K error3.05e-6).

Queue entry: scripts/run_e13.py invokes validate_qgs_kernel.py → validate_allocator.py → train_qgs.py → audit_native_loss_log.py → audit_checkpoints.py → native_primitive_stats.py → render_qgs.py → render_qgs_free.py → evaluate_qgs.py → export_qgs_centers.py → extract_qgs_mesh.py → smoke_native_viewer.py → write_execution_summary.py. It uses actual completion JSON and logs, resuming verified successful stages. Formal launch occurred only after the parent GPU queue finished P1–P4 and E12.

The render stage saves all208 native float RGB/alpha/depth/normal/distortion/curvature maps and separate diffuse(SH0) RGB. Evaluation reuses exact historical region/gradient code, skimage win7/sample covariance SSIM and Alex LPIPS at1080×1920; GT/mask hashes verified. Native bounded TSDF uses official GaussianExtractor.extract_mesh_bounded with lazy stored native depth/color maps, same canonical cameras and shared RAP fusion constraint. Native mesh depth/color use the original extraction flags (SH degree 0, kernel_size=-999, stop_z_gradient=True), saved separately from the full-SH evaluation maps. Shared physical-scale voxel/truncation is read from the frozen standard scale. Raw, upstream top10-component post, and comparable >=50-face-only post all retained; no hole filling/smoothing.

Native model viewing: QGS's own renderer consumes six-scale PLY. A local interactive HTTP adapter supplies RGB/gray/alpha/depth/normal orbit and zoom using the actual original quadratic CUDA renderer; eight actual model-backed HTTP requests are scheduled at the queue tail and the server is terminated before E14. No released SuperSplat-compatible quadratic viewer is included in the upstream repo. XYZRGB center export can be opened with MeshLab/Open3D but is not a surface/primitive reconstruction. Mesh PLY works with Blender/MeshLab/Open3D. No raw coordinate transformation is applied.

Actual paths are indexed only after execution. Expected output templates in scripts are not claimed as artifacts.


## Raw-render metric convention

Native render NPZ retains unclipped float RGB. Raw PSNR means no EO/appearance correction, and uses RGB clipped to[0,1] before PSNR, SSIM, LPIPS and gradients, exactly as historical E10/E11. SSIM: skimage structural_similarity, channel_axis2/data_range1/win_size7/gaussian_weightsFalse/use_sample_covarianceTrue/fullTrue. LPIPS: Alex on clipped RGB mapped to[-1,1]. Original GT and RAP hashes are verified in all208 frames. Native QGS training SSIM/loss remains its official implementation and is not confused with evaluation SSIM.

The eight comparison free-camera poses are consumed verbatim from the parent `comparison/shared_free_views.json`, including world-Y yaw ±15°, pivot and gray RGB(.25,.25,.25). Additional inspection protocols can be rendered with `render_qgs_free.py --protocol <actual protocol JSON> --out <output directory> --skip-fixed-windows`; the original eight-camera protocol remains unchanged. Fixed texture and leaf-boundary windows use the same historical pixel coordinates.

The native geo/NCC Python loss variables are initialized outside the official loop and can retain old values when their conditional branch is inactive. Raw training metrics are preserved; native_loss_log_audit.json distinguishes eligible contributions, including zero actual geo/NCC at terminal iteration30000. The running queue's final CPU summary creates this audit if its earlier loaded stage list lacks the new optional CPU stage. Numerical training is unchanged.

Allocator diagnosis: the real error reported32.69 GiB live PyTorch allocation,8.82 GiB reserved-unused,2.46 GiB requested and2.06 GiB free; another process occupied3.46 GiB and was retained. Context7 resolved /pytorch/pytorch/v2.5.1 and the referenced official cuda.rst was read. Expandable segments targets fragmentation caused by changing Gaussian counts; it is an allocator change, not a numerical objective change. Its actual GPU validation runs in the locked resume queue before training. [Pinned PyTorch source](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/cuda.rst).

The driver now requires explicit --resume when partial checkpoints exist, holds a file lock across all native stages, archives native failure state when safely possible, and saves one rolling extra native checkpoint every1000 iterations while preserving the fixed7K/15K/30K native models. Resume restores actual optimizer/Gaussian state while resetting seed0 through native upstream behavior; capture does not store RNG/viewpoint-stack trajectory, and pre-step checkpoint semantics are documented. Total attempt wall time and successful segment duration are recorded separately.
