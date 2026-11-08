# E1–E6 实际实验 Provenance 审计

本审计读实际日志、208 帧 Dataset、Camera JSON、PLY 头、NPZ 成员和小标量，以及运行器/官方源码。没有训练、GPU 探测、修改历史文件或重做前端。博查搜索本轮返回 403 配额不足；官方公开源码通过 web-access 规定的直接访问读取，本报告结论来自这些源码和本地实测，不依赖未检索的模型记忆。

完整证据：[experiment_provenance.json](gaussian_analysis/experiment_provenance.json)、[current_source_environment.json](gaussian_analysis/current_source_environment.json)、[optimization_count_timeline.json](gaussian_analysis/optimization_count_timeline.json)。脚本均在 `scripts/provenance_*.py`，使用 Ω 独立 `.venv` 的 Python 经 uv 执行，`CUDA_VISIBLE_DEVICES=1`；本审计实际只用 CPU。

## 实际图像和训练尺寸

Ω Dataset：`runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset`；标准 Dataset：`ForeSplat_Standard_VGGT_Frontend/experiments/KongQueZhuYu/standard_vggt_comparison/Standard_VGGT_Dataset`。两组均为 208 RGB / 208 Mask，文件名严格相同，全部 RGB 和 Mask 头尺寸 **2160×3840（W×H）**。标准 Dataset 的全部 208 RGB、全部 208 Mask 与 Ω Dataset 对应文件共享 dev+inode，即实际同一文件的 hardlink。

RAP 源位于 `/data/fj/F2DMAS/03-SAM/KongQueZhuYu`：RGB `crop_NNNN.png`，Mask `mask_NNNN.png`，Dataset 中 Mask 更名为 `crop_NNNN.png`。首/中/末 `crop_0000 / crop_0105 / crop_0209` 与 RAP 原文件的 RGB 字节 SHA 和解码像素均相同；三张 Mask 解码像素也相同且只有 0/255。全 208 Dataset 做了头/文件身份检查；RAP 全序列 payload 未重复扫描，不能将三帧内容抽检冒称全 208 的原文件 SHA 校验。

首、中帧 Mask 外 RGB 全为零；末帧 Mask 外全零比例 99.8448%，存在少量黑底残留。此事实不等于新发现了分割错误的原因，应结合输入边界审计。黑底与白底开关不同：全部实验 `white_background=False`，训练背景张量为 `[0,0,0]`。

全部六项实验 `resolution=2`，实际训练 **1080×1920**。保存的 `checkpoints/cameras.json` 是原始 CameraInfo 的 2160×3840，不能当作训练时尺寸。E1/E5 本轮实际加载检查点的 runtime Camera 也验证了 1080×1920，见 [02_camera_geometry_audit.md](02_camera_geometry_audit.md)。

前端网格与训练图像不能混淆：Ω balanced@512 本株实际前端网格 384×688；标准 VGGT 前端 518×518，先原尺寸左右黑 padding 再两次缩放。训练使用上述完整分辨率 RAP RGB hardlink，并非把 518 方图当训练 GT。

## 实际实验表

| 实验 | 训练表示/因素 | 实际训练 Mask | 初始 Gaussian | 最终导出 Gaussian | 保存状态/精确训练 CSV |
|---|---|---|---:|---:|---|
| E1_GGGS_Baseline | GGGS，EO off / FG off | 日志 `masks: None` | 1,252,383 | 600,458 | PLY 15k/30k/final；CSV 300 行 |
| E2_GGGS_EO | GGGS，EO on / FG off | 日志 `masks: None` | 1,252,383 | 607,422 | PLY 15k/30k/final；CSV 300 行 |
| E3_GGGS_Foreground | GGGS，EO off / FG on | Ω Dataset/masks | 1,252,383 | 1,036,619 | PLY 15k/30k/final；CSV 300 行、5 次 Mask pruning |
| E4_ForeSplat_GGGS | GGGS，EO on / FG on | Ω Dataset/masks | 1,252,383 | 814,161 | PLY 15k/30k/final；CSV 300 行、5 次 Mask pruning |
| E5_Standard_VGGT_GGGS_Baseline | GGGS，标准 VGGT 初始化，EO/FG off | 日志加载标准 Dataset/masks | 1,252,383 | 5,153,165 | PLY 7k/30k/final；**CSV 未存在** |
| E6_2DGS_Standard_VGGT | 2DGS 原生配方，标准 VGGT 初始化 | cfg 没有 FG/Mask 训练开关 | 1,252,383 | 119,836 | PLY 7k/30k；`chkpnt30000.pth`、`cfg_args` 存在 |

全部 30,000 步，seed=0，208 training views / 0 test views。seed 来自 `safe_state()` 与运行器，并不等于保存了 CUDA 全确定性设置或可精确恢复的优化器/RNG 状态。主相机采样是复制208相机stack后随机pop，**每208步无放回**；30k含144完整周期+48步，所以每帧访问144或145次，恰48帧145次。最后48帧身份和具体次序未保存，不能从稀疏CSV恢复，但不应声称访问次数完全不可得。[multiview_sampling.json](gaussian_analysis/multiview_sampling.json) 保存源码证明和实际邻视图。

Ω 初始 PLY 是 normal-aware、confidence-weighted、edge-preserving voxel fusion 的输出；标准初始 PLY 是 Mask/depth/conf p20 过滤后按 seed0、逐视角分层采样的原始单视角反投影。点数相等控制了数量，**没有控制几何、相机和点的观测支持**。GGGS 真正初始化读取 `points3D.ply`，不是旧 metadata 中一百万 COLMAP point/track 字段；PLY 头及第一段训练 CSV 均确认 1,252,383。

标准 PLY 中法线可为 camera-center−point 方向，但 GGGS `create_from_pcd()` 只读取 `pcd.points` / `pcd.colors`，从 KNN 距离初始化各向同性尺度，旋转为 identity quaternion；**不读取 pcd.normals**。因此不能将法线语义差异归因为 GGGS 初始旋转/尺度错误。[官方 GaussianModel](https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids/blob/23b418c2f0f19b30b251acad6fe13445dae968c7/scene/gaussian_model.py)。

## 实际参数与文档不一致

E1–E5 使用同一当前 `train_foresplat.py` 和 GGGS 官方优化核心。原生 defaults：位置学习率 0.00016→0.0000016（max steps 30000，乘 scene extent）、DC/rest 特征学习率 0.0013/0.00011、opacity .05、scaling .005、rotation .001；densification from500 / until15000 / interval100、normal-gradient threshold .0002、raw-opacity cull .05、opacity reset3000。损失 `.8 L1 + .2(1−SSIM)`；7000 起 depth-normal/NCC/geo 权重 .05/.6/.02。详细默认值由当前官方源码 AST 读取并保存，不能假装缺失的历史完整 argv 已存在。

E3/E4 额外启用前景 RGB（Mask 外权重0、crop padding12）、Mask/background-opacity loss（λ=.08/.02，l1_dice，boundary ignore2，500起/1500 warmup）；Mask pruning 从18000、每3000、最多12 views、最多移除3%，survival multi_cue 默认阈值.45。实际事件和移除数在 [loss_pruning_summary.json](gaussian_analysis/loss_pruning_summary.json)。E1/E2/E5 无该 pruning。E5 虽加载 `gt_mask`，但没有启用这三个 FG loss 或 pruning；原生 PatchMatch 不读取 `gt_mask`，因此加载 Mask 本身不等于对 E5 启用前景训练约束。

现有文档需要按证据解释，不修改历史文件：

1. `shared_config/experiment_config.yaml` 写 `sh_degree: 0`，但运行器未传 `--sh_degree`，实际默认 **3**。E1–E6 所有最终 PLY 均有 45 个 `f_rest` 字段（SH3 capacity），GGGS 每1000步提升 active degree，3000后达3；不是实际 SH0 实验。SG degree 默认0，未把空 SG 扩充成其他表示。
2. E1/E2 RUN_REPORT 复现命令含 `mask_dir`，而实际日志是 `masks: None`；运行器也只对 FG on 传 Mask。两者须区分。E5 报告“identical flags”也忽略了它与 E1 的 Mask 加载差别、save/test schedule 差别。
3. E5 报告列出 `metrics/train_metrics.csv`、`cfg`，实际 **没有该 CSV 或 cfg_args**；只有 tqdm EMA / count 和最终 PLY/NPZ。标准项目当前也没有 `gggs_community/`，报告相对 trainer 路径依赖未记录的执行 cwd；不能据此声称命令可在该项目根直接复现。
4. E1–E4 测试10k/20k/30k、保存15k/30k；E5 默认测试/保存7k/30k。训练配方主体相同，但应记录调度事实，而非说全部 CLI 完全相同。
5. E3 `iteration_30000` / CSV 最后为 **1,037,020**，最后一次 Mask pruning 后 `final_gaussians` 为 **1,036,619**；E4 对应 **814,255 →814,161**。`scene.save()` 在最后 Mask pruning **之前**，最终导出在循环结束之后。这是不同实际状态；用 `load_iteration=30000` 的历史评价不能标成 post-prune final。
6. E6 使用自身 2DGS 配方（λdist .2 / λnormal .05），不是 GGGS 的相同损失或增密/剪枝策略；其较小终点数量或 E6≈E5 图像指标不能单独排除训练表示/优化因素。全部历史指标是 training-view metrics，未留出独立 test。

## 场景 extent 与尺度自由度

E6既有Mesh已定位并实际读取原PLY头：[existing_mesh_inventory.json](gaussian_analysis/existing_mesh_inventory.json)。`E6_2DGS_Standard_VGGT/checkpoints/train/ours_30000/fuse.ply` 为8,413,786 vertices /13,590,516 faces，`fuse_post.ply` 为6,207,866 /11,836,007，与原mesh_metrics.json一致。E5目录只找到Gaussian/初始化点PLY，没有带face元素的既有surface mesh，不以E6或其他模型Mesh代替。这里仅读原头和路径，没有重新提取Mesh、读取全部大payload或将顶点数量当作几何精度；其voxel/depth_trunc仍是native场景单位，未校准公制尺度。

从完整 saved camera centers 重新计算的 Ω radius=1.008220800，标准=2.524666269（约2.504×）。原生 position learning rate 与 clone/split `.01*extent` 同步缩放，这是预期的场景尺度适配；**extent 数字变大本身不是异常**。只在各自坐标系用 `sigma/extent`、distance/extent 等归一化分布比较；没有扫描配准或跨世界坐标直接换点。真实标准相机/点与 RGB 的失配另见独立相机审计。

## 源码版本、环境和可恢复状态

GGGS 独立 git commit=`23b418c2f0f19b30b251acad6fe13445dae968c7`，standard VGGT 独立 commit=`a288dd0f14786c93483e45524328726ab7b1b4ce`。本轮下载的 GGGS `train.py` / `scene/gaussian_model.py` / `arguments/__init__.py` 与本地 **逐字节 SHA 相同**，快照 `source_snapshots/provenance_official_*.py`；另有 Camera/reader/graphics 比对在 `local_source_versions.json`。独立官方核心未被本审计修改；`train_foresplat.py` 是 untracked 自研 wrapper，SHA=`f1f811f2e5c014ad9b7e155352d8e71bd7b11cea8e03309976bb6fb4484c6c96`，不能宣称它本身是官方文件。

Ω adapter 和 `/data/fj/F2DMAS/2d-gaussian-splatting-main` 当前不是独立 git root。向上读出的 `f1a522d4d4a296a564394d20d735c6bcbfc87456` 属于 F2DMAS 工作区，**不是模型官方 commit**。E6 历史报告指向后者；本轮重渲染用实际源码位置和文件指纹补证，不能从父目录 commit 编造官方2DGS版本。

标准 VGGT checkpoint 的已保存 SHA=`d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`，来源旧 `model_metadata.json`；此轮没有重复读取5GB权重重算。Ω旧模型路径 `weights/vggt_omega_1b_512.pt`，已完成前端阶段的VGGT_Omega_Existing Bundle记录 SHA=`c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`，本轮定向输入对照复用该记录，不重复4.3GB仅为重算哈希。两者仍不提供足以重建训练期完整环境的所有锁文件；未补造。

当前 CPU audit 环境 Python3.12.3 / torch2.10.0+cu128 / torchvision0.25.0 / numpy2.5.4，通过 uv 执行；这是**当前审计环境**，不是 E1–E5 训练时 freeze 的证明。E6 历史报告写 Python3.12 / torch2.5.1+cu121，与本轮环境分开保留。没有升级全局环境或安装 conda。

E1–E5 未保存 `chkpnt*.pth`。其 `gaussians.npz` 只有 xyz、DC/rest、activated opacity/scales/rotation 与数量/时间/峰值内存，**没有 xyz_gradient_accum、absolute accum、denom、optimizer/RNG state**。PLY 更不保存这些状态。无法精确恢复7000→15000的 E5 梯度和增密事件。E6 `.pth` 属于另一个算法，不可当 E5 的缺失状态。E5 每事件 clone/split/prune/Q 历史及真实零分位触发保持 Unresolved，见 [04_gggs_optimization_audit.md](04_gggs_optimization_audit.md)。
