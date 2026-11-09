# E11：标准 VGGT + 修正相机 + 官方 2DGS

正式训练已经完成 30,000 iterations，Seed 0、GPU1、uv，最终 248,629 个 Surfel。全部 208 个训练视角的浮点 RGB、原生 expected/median Depth、Normal 和 Alpha 已保存；同一套 E10 评价实现计算的全帧 PSNR 为 **22.2615 dB**、FG-PSNR 为 **17.5925 dB**。训练耗时 2,911.28 s（48.52 min），Torch 峰值 allocated 2.6202 GiB、reserved 4.6836 GiB。

## 输入与相机

严格复用 `../geometry_rgb_dual_frontend/Standard_VGGT_Original_RGB/COLMAP_Corrected/`，训练 RGB 沿用原 RAP 前景 RGB；没有重新推理 VGGT 或 Mask。208 个有效 Frame ID 为 0–209，排除已有损坏帧 44、206，帧集合与 E10 一致。初始化使用相同的 1,252,383 点 PLY。运行时实际图像为 width=1080、height=1920，208 组 K、W2C、Projection 及浮点 GT/Mask 哈希均匹配 E10，并非仅以命令行 resolution=2 推断尺寸。

相机、像素中心、深度光线和 Mesh K 的完整修复及真实 CUDA 验证见 `../camera_interface/E11_INTERFACE_ANALYSIS.md`。保留原 E6 surfel rasterizer 核心数学和 GaussianModel。所有源码补丁、训练时源码快照、当前源码哈希分别保留，未覆盖 E1–E10。

## 原生训练配置

SH degree 3，lambda_normal=0.05，lambda_dist=0.2，depth_ratio=0.0。其余原生优化器、增密、剪枝、opacity reset 和正则启用时刻保持官方训练策略。使用 E6 已部署 Python3.12 / Torch2.5.1+cu121 环境，通过 uv 执行；E10 的 Torch 环境不同，但评价统一在 E10 环境进行。官方 train.py 与上游 commit `f3e3b9fa67bbd1c75e05167ff37391d8dab2a678` 完全匹配，E6 的其余源文件按哈希固定。精确解析参数和命令见 `actual_training_parameters.json`、`../configs/actual_commands.jsonl`。

EO、前景 RGB Loss、Alpha/Mask Loss、Mask Pruning 和额外 Depth Prior 均未启用。E6 相机实现将整套图像保留在 CPU，当前训练视角再传 GPU，因此不能将 2.6202 GiB 与 E10 的 8.5801 GiB 直接解释为算法固有显存差异。

## 全 208 帧照片评价

| 指标 | E10 GGGS | E11 2DGS |
|---|---:|---:|
| PSNR | 23.9005 | 22.2615 |
| FG-PSNR | 19.4605 | 17.5925 |
| 前景内侧 5px 边缘 PSNR | 13.8137 | 12.7367 |
| SSIM | 0.877556 | 0.863217 |
| LPIPS Alex ↓ | 0.117086 | 0.137349 |
| 前景梯度保留 | 0.731265 | 0.672860 |
| 背景 Alpha 均值 | 0.378039 | 0.993141 |
| 背景 Alpha>0.5 比例 | 0.382595 | 0.995759 |
| Gaussian / Surfel 数量 | 576,396 | 248,629 |
| 完整训练命令时间 s | 2,412.79 | 2,911.28 |

E11 在 172/208 视角的 PSNR 与 FG-PSNR 低于 E10，205/208 视角的 LPIPS 更差；个别局部纹理较好不能抵消侧向视角的条带与误差。配对分布记录在 `../comparisons/E11_E10_paired_render_deltas.json`。208 帧是单株、Seed 0 的相关训练视角，未进行错误的独立样本显著性检验。

浮点评价使用原生 1080×1920 的 clip[0,1] RGB、原 RAP Mask、SSIM win7/sample covariance、LPIPS Alex 及 E10 审计脚本的梯度/区域函数。PSNR 为逐视角 dB 的均值，不能混同全体像素 pooled PSNR。

## 真实产物索引

- `checkpoints/point_cloud/iteration_30000/point_cloud.ply` 和 `checkpoints/chkpnt30000.pth`：正式模型。
- `train_metrics.csv`、`training_complete.json`：训练过程和运行资源。
- `per_view_metrics.csv`、`evaluation_complete.json`：208 帧指标与实际评价版本。
- `renders/INDEX.json`、`depth/INDEX.json`、`normals/INDEX.json`、`alpha/INDEX.json`：指向 `../surface_inputs/e11_2dgs_surface/rendered/` 的全部浮点 NPZ，无重复大文件复制。
- `meshes/INDEX.json`：原生 M5 TSDF 的 raw/post、组件移除记录与 E11→NKSR 补充实验索引；完整 Mesh 评价及最终选择记录在总实验报告中。

来源：[官方 2DGS](https://github.com/hbb1/2d-gaussian-splatting)，本地实际执行和浮点评价；博查检索本轮额度不足不可用。
