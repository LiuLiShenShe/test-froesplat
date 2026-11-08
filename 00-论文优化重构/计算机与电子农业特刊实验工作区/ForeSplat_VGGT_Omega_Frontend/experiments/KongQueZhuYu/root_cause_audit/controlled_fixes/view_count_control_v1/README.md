# 定向70/110/208输入视角数量对照

本实验因历史208相机中出现大幅突跳而触发：Ω source162→163约164.8°、164→165约152.7°；标准86→87约179°。突跳不自动证明帧冗余，本对照用于证伪“只要降低输入数量就能修复”的解释。只在 KongQueZhuYu 运行 Ω、standard 各70/110一次完整多视角推理；**208仅引用既有预测，没有重跑**。没有GGGS/2DGS训练、分组拼接、融合、尺度配准、权重下载或环境升级。

采样严格为 `np.rint(np.linspace(0,207,N)).astype(int)`，索引指历史208可用帧的顺序，不是原始文件名序号。选中的 RGB/Mask 按源名配对，保持顺序；两个模型用同一组选择。70和110共有38源帧，所有选择在各 `manifest.csv`。原图仍2160×3840。

| 实际完成目录 | 输入N | 原adapter调用时间(s) | peak allocated GiB |
|---|---:|---:|---:|
| Omega_70 | 70 | 18.733 | 11.227 |
| Omega_110 | 110 | 29.685 | 14.264 |
| Standard_70 | 70 | 27.639 | 13.325 |
| Standard_110 | 110 | 43.582 | 17.294 |

时间包含原接口的CPU输出转换（Ω还计算depth unprojection），不是纯网络kernel benchmark；模型每套只加载一次，首次/热缓存差异未做性能统计控制。完整实际记录在 [inference_runs.csv](inference_runs.csv) 和每目录 `metadata.json`。全部GPU作业用 physical `CUDA_VISIBLE_DEVICES=1`、uv，进程正常退出并已释放GPU1。

Ω调用未修改的 `foresplat_adapter.inference.run_inference()`：balanced512、patch16，实际W×H=384×688，原FP16 aggregator / FP32 heads与相机转换。Standard调用未修改的 `vggt_foreground.inference.run_vggt_heads()`：原图左右黑padding→1024 bicubic→518 bilinear，BF16 aggregator / FP32 camera-depth heads。输入实际float32，Mask最近邻。Standard历史nominal A保留，额外保存正确pixel-center A；两者仅用于坐标记录，不改变网络输入或预测。没有执行point head，两流程都保留depth/K/W2C支持。

权重只复用已有本地文件：Ω `weights/vggt_omega_1b_512.pt`、SHA `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`；standard `checkpoints/model.pt`、SHA `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`。本轮未重复multiGB全权重哈希。seed0、当前Python3.12.3/Torch2.10.0+cu128；官方standard commit=`a288dd0f14786c93483e45524328726ab7b1b4ce`，Ω不伪称有独立官方git commit，模型/adapter源码SHA另存。

## 输出与实际验证

每份包含 `manifest.csv`、`run_parameters.json`、`metadata.json`、`status.json`、`RUN_REPORT.md`、`cameras.json` / `cameras.npz`、`pixel_transforms.json`。`cameras.json['frames']` 保存 source_index、usable_index、image_name、模型K、OpenCV4×4 W2C、inverse C2W、历史A/正确pixel-center A、原始K两版本、实际float32输入SHA、model Mask SHA。没有额外world alignment或scene fusion。

`per_frame/` 每源名保存 `depth_crop_NNNN.npy`、`confidence_crop_NNNN.npy`、精确CHW `model_rgb_float32_crop_NNNN.npy` 和 `model_mask_crop_NNNN.npy`。原始depth/conf为float32；Mask是uint8 0/1。Ω原接口产生的额外小raw辅助输出保留NPZ；临时derived pointmap未在磁盘重复保存，不能冒充point-head结果。

[all_control_export_audit.json](all_control_export_audit.json) 实际通过全部8新控制（含背景控制）的456帧、1824 NPY头/类型/完整文件byte-length、相机小矩阵/逆方向、来源顺序、参数/seed/SHA/status/report检查。它不重算全depth/RGB payload checksum或宣称这些相机一定准确；独立评价读取实际几何值并使用真实RGB匹配。Ω、standard的38共同源帧 float32模型输入SHA、Mask SHA、A、尺寸全无差异，见两份 `*_N70_N110_input_invariants.json`。

## 固定共同源视角的独立几何结果

[evaluation/control_summary.json](evaluation/control_summary.json) 使用38共同source的相邻common-rank pair，共37尝试、32组共同接受的image-only RANSAC/SIFT集合；所有N使用同一真实foreground匹配与查询协议。以下为mask+finite基线，误差是shared384×688网格px；depth support是模型跨视图≤5%relative-depth支持，不是扫描GT表面精度。

| 模型/N | pair-median Sampson(px) | depth→matched RGB(px) | depth支持 |
|---|---:|---:|---:|
| Ω70 | 7.040 | 20.824 | 44.71% |
| Ω110 | 5.101 | 16.343 | 44.83% |
| Ω208历史 | 4.097 | 11.451 | 57.73% |
| Standard70 | 101.855 | 254.846 | 5.56% |
| Standard110 | 61.596 | 235.656 | 5.91% |
| Standard208历史 | 66.003 | 179.562 | 9.36% |

均匀减到70/110没有稳定修复几何，不能将208异常归为“帧太多/冗余”；也没有验证其他keyframe策略。不同N的上下文源集合改变，即使评价帧相同也不等同纯标量N的普遍规律。固定N24的背景单因素控制另见 [background_control_v1/README](../background_control_v1/README.md)，它避免了将输入上下文质量与相机数量混为原因。

首次Ω driver因两个仓库同名 `visual_util` 的组合sys.path在**模型加载/推理之前**退出；已隔离模型路径且保留 `Omega_attempt1_import_failed.log/json`。修复不修改网络、权重或预处理。完成目录拒绝重新GPU覆盖；复现需要新的独立输出版本。官方源与历史环境限制见 [01 provenance](../../01_experiment_provenance.md)。
