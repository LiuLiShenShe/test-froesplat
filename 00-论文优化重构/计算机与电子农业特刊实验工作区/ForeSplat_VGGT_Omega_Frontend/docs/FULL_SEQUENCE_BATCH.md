# 全序列批处理：VGGT-Ω 全帧联合推理 → 多视角融合 → GGGS 数据集

本阶段把前端的单场景导出升级为**按植物批处理**的完整系统：

```
03-SAM/<Plant>/{crop_XXXX.png, mask_XXXX.png}
        │  自动发现 + 文件完整性校验（损坏帧隔离，不静默丢弃）
        ▼
VGGT-Ω 全序列单次联合推理（不分帧、不抽帧、不分窗、不混植物）
        │  显存不足时沿阶梯降级（分辨率/预处理模式），绝不丢帧
        ▼
逐帧 dense depth / confidence / model RGB / model mask（.npy）
        │
        ▼
多视角融合（置信度加权 + 法线感知 + 边缘保持，体素哈希）
        │  记录每个融合点的观测来源（图、像素、置信度、权重）
        ▼
GGGS_Dataset/（全部视角 + 融合点云 points3D.ply + 逐帧几何 + 元数据）
```

---

## 1. 数据发现与清单

```bash
python tools/build_batch_manifest.py \
    --root /data/fj/F2DMAS/03-SAM \
    --output-root runs/full_sequence_batch \
    --manifest runs/full_sequence_batch/batch_manifest.csv
```

- 每株植物 = 一个独立场景；**不同植物绝不混合**。
- 配对规则：`crop_XXXX.png` ↔ `mask_XXXX.png`（按 4 位序号）。
- **完整性校验**：PNG magic + PIL 结构校验。校验失败的帧（连同缺失成对文件的帧）
  记入 `<Plant>/metadata/quarantined_frames.csv` 并从该场景剔除。
  这是唯一会减少帧数的情况；**可用帧一帧不丢**（无 stride / 无窗口 / 无抽样）。
- `batch_manifest.csv` 字段：`plant_id, images_dir, masks_dir, total_images,
  matched_images, quarantined_frames, image_resolution, prefix_mapping,
  output_dir, processing_status`。

### 03-SAM 实测（2026-11-05）

| 项 | 值 |
|---|---|
| 植物目录 | 19 个 |
| 每株帧数 | 210–215（配对后） |
| 可用帧 | 195–211 / 株，合计 **3878** |
| 分辨率 | 2160×3840（竖幅，aspect 1.778，无需裁切） |
| 损坏文件 | 每株 1–15 个（磁盘上内容被替换为无关字节，**重读稳定、不可修复**） |
| 原始源 | `02-FFT/<Plant>/XXXX.jpg`（JPEG 完好，损坏发生在 SAM3 输出写入阶段） |

> 说明：损坏是**数据源**问题（文件写入 2026-02-27，与本次流程无关）。
> 流水线对这些帧做隔离并逐一登记，不做任何"猜测性补帧"。

---

## 2. 显存阶梯（绝不丢帧）

`foresplat_adapter/full_sequence.py: OOM_LADDER`

```
balanced@512 → max_size@512 → balanced@448 → max_size@448 → balanced@384 → max_size@384
```

- 每次尝试都跑**同一条完整序列**；只有预处理档位变化。
- 每次尝试记录 `peak_vram_allocated_mib` / `peak_vram_reserved_mib` /
  `inference_time_s` / `model_input_hw`，写进 `metadata/fusion_metadata.json`
  与 `batch_summary.csv`。
- 每株植物保留自己**实际生效**的档位（不同植物可以不同）。
- 用 `--force-resolution-mode balanced@512` 可以钉死档位（跳过阶梯）。

本机约束：**只允许 GPU 1** —— 所有命令前置 `CUDA_VISIBLE_DEVICES=1`。

---

## 3. 融合模块 `foresplat_adapter/fusion/`

| 文件 | 内容 |
|---|---|
| `normal_estimation.py` | 由**单视角 point map 局部邻域**（5×5）估计法线；朝向该视角相机；输出平面度质量 `(λ2-λ3)/λ1`、邻域数、深度断续边缘标记 |
| `adaptive_voxel.py` | 场景自适应体素：`extent / cbrt(target)` 种子 + 中位最近邻尺度下限 + 按实际体素计数二分收敛 |
| `multiview_fusion.py` | 两遍体素哈希融合（核心 + 边缘），法线球面分桶，置信度加权平均，观测溯源 |
| `observation_mapping.py` | 融合点 → 源观测映射的构建、npz 存取、CSV 导出 |

### 3.1 候选点

每个视角所有通过 `foreground & finite_geometry & valid_depth & conf ≥ P20(前景)` 的
像素都进入候选（**融合前不做任何降采样**）。候选携带
`(XYZ, RGB, conf, normal, quality, edge, view, pixel)`。

### 3.2 法线感知：不合并薄叶两面 / 不同叶片

融合键 = `voxel_key × (azimuth_bins × elevation_bins) + normal_bin`，
法线按球面方位角/仰角分桶（默认 24×12）。因此：

- 同一体素内**法线近似相反**的两片表面（薄叶正反两面）落入不同键 → 永不合并；
- 同体素内**朝向不同**的两片叶子（叶片交叉/重叠）同理分离；
- 只有位置相邻且朝向一致的点才被融合成一个代表点。

### 3.3 置信度加权

观测权重 `w = conf × (0.25 + quality) × (edge ? edge_weight_boost : 1)`；
融合点位置/颜色/法线/置信度均为加权平均。高置信度视角对结果贡献更大。

### 3.4 边缘保持

深度断续像素（4 邻域相对深度跳变 > 3%）单独走**边缘通道**：

- 体素缩小为 `core_voxel / 2`（`--edge-voxel-divisor`）；
- 权重 ×2（`--edge-weight-boost`）；
- 结果附加到核心融合点之后，因此叶缘/轮廓的采样密度高于平滑区域。

### 3.5 点数预算

目标 **1.5M**（`--target-points`），范围 1M–2M：

- 核心通道目标 80%（≈1.2M），边缘通道补足；
- 若场景有效点本来就少于目标，**不做任何填充**（`below_min_points=true` 如实记录）；
- 超过 2M 时只对**边缘附加点**按固定种子抽稀，核心融合点永不删减。

### 3.6 观测溯源

`debug/observations.npz`（逐观测数组）：

| 字段 | 含义 |
|---|---|
| `fused_point_id` | 融合点索引（与 points3D.ply 行号一致） |
| `source_image_id` | 来源图像（COLMAP image_id，1 起） |
| `view_index` | 序列内下标 |
| `source_pixel_index` | 模型分辨率下的像素展平索引（`y*W+x`） |
| `source_confidence` / `source_weight` | 该观测的置信度与融合权重 |
| `source_normal` / `source_is_edge` | 该观测的法线与边缘标记 |

`observations_for_point()` / `export_observations_csv()` 可按点查询或导出 CSV。

---

## 4. 每株植物的输出

```
runs/full_sequence_batch/<Plant>/
├── GGGS_Dataset/
│   ├── images/           全部可用帧的前景 RGB（原始分辨率）
│   ├── masks/            全部可用帧的二值掩膜
│   ├── sparse/0/
│   │   ├── cameras.bin / images.bin / points3D.bin / *.txt
│   │   └── points3D.ply  ← **融合点云**（带法线，GGGS 高斯初始化用）
│   ├── per_frame/
│   │   ├── depth_XXXX.npy / confidence_XXXX.npy
│   │   ├── model_rgb_XXXX.npy / model_mask_XXXX.npy   （模型分辨率）
│   │   └── cameras.json  逐帧内参/外参/相机中心/像素仿射/对齐矩阵
│   ├── metadata/
│   │   ├── preprocess_transforms.json / image_manifest.json
│   │   ├── reconstruction_metadata.json
│   │   ├── fusion_metadata.json   融合参数、体素尺寸、阶梯尝试、峰值显存
│   │   └── quarantined_frames.csv  （仅有损坏帧时）
│   └── debug/
│       ├── raw_pointcloud.ply      逐视角原始单观测点云
│       ├── foreground_pointcloud.ply
│       ├── fused_pointcloud.ply    融合点云（带 quality/观测数）
│       └── observations.npz        观测溯源
├── visualizations/
│   ├── rgb_preview.png / mask_overlay.png
│   ├── depth_map.png / confidence_map.png
│   ├── camera_distribution.png
│   ├── pointcloud_rgb.png / raw_vs_fused.png
│   └── fused_pointcloud.html       （装有 plotly 时）
├── RUN_REPORT.md
└── STATUS.json     （--resume 标记）
```

COLMAP 的 `points3D.bin` 仍是**逐视角原始点**（每点 1 个观测、精确自洽）；
`points3D.ply` 是融合点云。GGGS `readColmapSceneInfo` 优先读 ply，因此训练初始化
使用融合点云；两者语义在 `reconstruction_metadata.json` 中写明。

---

## 5. 批处理运行

```bash
# 单株（先跑最大规模植物确认显存）
CUDA_VISIBLE_DEVICES=1 python tools/run_all_plant_datasets.py \
    --manifest runs/full_sequence_batch/batch_manifest.csv \
    --output-root runs/full_sequence_batch \
    --checkpoint weights/vggt_omega_1b_512.pt \
    --gggs-repo gggs_community \
    --plant ChangShouHua2

# 全部植物（最大规模优先；中断后 --resume 续跑）
CUDA_VISIBLE_DEVICES=1 python tools/run_all_plant_datasets.py \
    --manifest runs/full_sequence_batch/batch_manifest.csv \
    --output-root runs/full_sequence_batch \
    --checkpoint weights/vggt_omega_1b_512.pt \
    --gggs-repo gggs_community \
    --order largest_first --resume
```

默认顺序：**有效帧数从多到少**（先验证 48GB 显存上限）。每株处理完释放模型与显存。

输出汇总：

- `batch_summary.csv`：`plant_id, total_frames, processed_frames, image_resolution,
  preprocess_mode, inference_time, peak_vram_*, raw_points, fused_points,
  fusion_ratio, camera_count, mean_confidence, ggss_read_status,
  gaussian_init_status, quarantined_frames, status`
- `batch_report.md`：方法说明 + 汇总表。

历史 20 帧结果（`runs/KongQueZhuYu_20frames_512/`）不会被覆盖。

---

## 5.1 最大规模植物实测（ChangShouHua2，211 帧，RTX A6000 48GB）

| 项 | 值 |
|---|---|
| 生效档位 | **balanced@512**（首次尝试即成功，未动用降级阶梯） |
| 模型输入 | 211 × 384×688（24×43=1032 tokens/帧，共 217k tokens） |
| 推理耗时 | **85.3 s**（单次联合前向，全 211 帧） |
| 峰值显存 | **allocated 22,437 MiB / reserved 31,252 MiB** |
| 候选点 | 13,317,095（211 视角全部前景有效像素，融合前未降采样） |
| 融合点 | **1,289,801**（核心 1,226,843 @ voxel 0.00877 + 边缘 62,958 @ voxel 0.00438） |
| 平均观测数/融合点 | **10.3**（多视角溯源有效） |
| 融合耗时 | 94.9 s（CPU） |
| COLMAP 导出 | 1,000,000 原始单观测点（`--raw-max-points` 上限，165 s） |
| GGGS 回读 | **ok**：211 相机，`points3D.ply` 1,289,801 点且带法线 |
| 高斯初始化 | **ok**：`GaussianModel(3,3).create_from_pcd` → xyz `[1289801, 3]`，features `[1289801, 16, 3]` |

结论：48GB 显存下 211 帧全序列 @512 无需降档；剩余 18 株按同一档位顺序处理。

---

## 5.3 本地依赖位置（2026-11-05 起，不再放 /tmp）

| 依赖 | 位置 | 说明 |
|---|---|---|
| VGGT-Ω 代码 | `./vggt_omega/` | 项目内包，`foresplat_adapter` 直接导入 |
| 权重 | `./weights/vggt_omega_1b_512.pt` | 4.57 GB，sha256 `c02da418…7934`（见 `weights/sha256.txt`） |
| GGGS 仓库 | `./gggs_community/` | `git clone https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids`；`--gggs-repo` 默认自动探测 |

- 之前 GGGS 克隆在 `/tmp`，机器重启后被清空，导致验证静默降级为
  "not attempted"。现在改为项目内持久路径，两个工具默认值会自动探测
  `./gggs_community`，无需每次显式传参。
- `simple_knn` CUDA 扩展依赖 `libc10.so`（torch 包内）与 `libnvJitLink.so.12`
  （pip `nvidia-nvjitlink`），两者都不在默认加载路径。`gaussian_init.py`
  用 `ctypes.CDLL(..., RTLD_GLOBAL)` 预载（运行时改 `LD_LIBRARY_PATH`
  对 `dlopen` 无效），无需手工设环境变量。

---

## 5.4 全量批处理结果（2026-11-05，19 株全部完成）

| 项 | 值 |
|---|---|
| 植物数 | **19 / 19 done**（`--resume` 跳过 2 株首轮已完成的） |
| 处理帧数 | **3,878**（= 清单全部可用帧，一帧不丢；隔离 128 个损坏帧） |
| 生效档位 | 全部 **balanced@512**（无任何株触发降级阶梯） |
| 推理耗时 | 均值 55.9 s / 株（最大 85.3 s，ChangShouHua2 首轮含冷缓存） |
| 峰值显存 | allocated 21,204–22,437 MiB / reserved 29,452–31,252 MiB（均 <48GB） |
| 融合点数 | **1,152,834 – 1,369,270 / 株**，全部落在 1M–2M 目标区间，合计 23.98M |
| GGGS 验证 | 19 株全部 `readColmapSceneInfo ok`（相机数=帧数）+ `create_from_pcd ok` |
| 磁盘占用 | 28 GB（含逐帧 .npy、观测溯源 npz、8 张/株可视化） |

汇总表：`runs/full_sequence_batch/batch_summary.csv`，方法报告
`runs/full_sequence_batch/batch_report.md`；每株 `RUN_REPORT.md` 记录
档位、耗时、显存、融合统计与验证结果。

### 批处理期间修掉的两个坑

1. **报告写入 KeyError**（`inference_time` vs `inference_time_s`）曾令
   首轮批处理在 WanNianQing1 完成后崩溃——数据集无损，`finalize_plant.py`
   补齐了报告；报告写入现已包进 try/except 并修复键名。
2. **`verify_dataset_static` 误报**：COLMAP `images.txt` 的 POINTS2D 观测行
   被当图像行解析（真实导出每图 ~5,128 个观测点，观测行非空）。权威
   验证（GGGS 真实回读）一直通过；解析器已改为只认 10-token 图像行，
   19 株复检全绿。

---

## 5.2 内存与断点续跑（2026-11-05 OOM 教训）

- 批处理机是共享机：开跑前 swap 已 100% 占满。**严格串行**：主流程运行期间
  不得并发跑测试/其它重活；每个植物开跑前检查 `free -g`。
- 验证路径已瘦身：`verify_dataset_static`（只读文件头，不加载原图）+
  `check_dataset_and_init_gaussians`（`readColmapSceneInfo` + `create_from_pcd`，
  只读 PLY，不加载 211 张原图）。旧路径一次读入全部 2160×3840 原图，峰值
  RAM 额外 5–10 GB，是 OOM 的主要放大器之一。
- 若某株在验证/可视化阶段被打断而 `GGGS_Dataset/` 已完整，用
  `tools/finalize_plant.py` **只补验证 + 可视化 + 报告**，不重跑推理/融合/导出：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/finalize_plant.py \
    --plant ChangShouHua2 \
    --manifest runs/full_sequence_batch/batch_manifest.csv \
    --output-root runs/full_sequence_batch \
    --gggs-repo gggs_community
```

---

## 6. 验证

- 单元测试：`python -m pytest tests/ -q` —— 当前 **44 项全过**
  （预处理/导出 19 + 融合 17 + 融合-导出集成 6 + 边缘抽稀溯源回归 2）
- GGGS 回读：`foresplat_adapter/verify.py` 调用真实 `readColmapSceneInfo`
- 高斯初始化：`foresplat_adapter/gaussian_init.py` 调用
  `GaussianModel(3,3).create_from_pcd(...)`（`simple_knn` 由 ctypes 预载自动修复）

---

## 7. 语义边界（重要）

- 所有点仍是 **单视角深度反投影**、融合后为**多视角加权代表点**，
  **不是 SfM 三角化匹配点**；COLMAP 的 `error` 字段不构成几何精度指标。
- 深度是 VGGT-Ω 的**相对尺度**，非米制；GGGS 用相机中心计算 `cameras_extent`，不受影响。
- 未运行 GGGS 完整训练（`train.py`），只验证到数据读取 + 高斯初始化。