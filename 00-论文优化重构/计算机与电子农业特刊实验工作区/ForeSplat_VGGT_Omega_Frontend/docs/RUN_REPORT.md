# 真实运行报告：RAP-FSAM3 → VGGT-Ω → COLMAP → GGGS

日期：2026-11-05 ｜ GPU：NVIDIA RTX A6000 48GB ｜ torch 2.10.0+cu128 ｜ pycolmap 3.10.0

---

## 1. 权重

| 项 | 值 |
|---|---|
| 来源 | `https://modelscope.cn/datasets/hengyelyu/vggt-omega-weights` → `vggt_omega_1b_512.pt` |
| 大小 | 4,576,706,117 bytes（与镜像 README 一致） |
| SHA256 | `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`（**校验通过**） |
| 加载 | `VGGTOmega().load_state_dict(sd, strict=False)` → **missing 0 / unexpected 0**，1.1438 B 参数 |

## 2. 真实输入数据

RAP-FSAM3 已处理完成的多视角运行（**不重跑分割**）：

```
.../ForeSplat_Original_Method_Recovery/runs/KongQueZhuYu_RAP_FULL_210_R4bm05/
├── 前景图/crop_0000.png … crop_0209.png      RGB，2120×3791，黑底前景（210 帧）
└── 最终掩膜/mask_0000.png … mask_0209.png    二值 0/255，同尺寸（210 帧）
```

配对：文件名前缀不同（`crop_` / `mask_`），通过 `--image-prefix-to-strip crop_ --mask-prefix-to-strip mask_`
（或自动尾号回退）配对；**视角顺序与文件名完全保持**。

## 3. 单元/接口测试（不需要权重）

```
$ python -m pytest tests/ -q
19 passed, 3 warnings in 11.09s
```

覆盖内容包括：RGB 路径与社区 `load_and_preprocess_images` **逐元素完全一致**；
Mask 与 RGB 使用同一裁剪框/缩放/padding（padding 区域掩膜为 0）；
`A` 与 PIL 重采样质心的数值吻合（<0.75 px）；`K_original = A^-1 K_model`；
COLMAP 导出后**逐点重投影误差 < 1e-5 px**（对齐开/关两种模式）；
单观测 track；观测全部落在原始图像范围内；前缀配对与顺序保持。

## 4. 真实运行结果

| 运行 | 帧数 | 分辨率 | 点数 | 位姿一致性（fx max/min） | 跨视角深度中位误差 | 5% 内点率 | GGGS 回读 |
|---|---|---|---|---|---|---|---|
| smoke | 3（stride 70） | 256 | 38,488 | 1.79 | — | — | ✅ |
| diag A | 8（stride 26，跨 210 帧） | 512 | 500,000 | 1.238 | 11.6% | 26.8% | ✅ |
| diag B | 10（连续） | 512 | 481,521 | **1.047** | **0.65%** | **94.7%** | ✅ |
| **最终** | **20（连续）** | **512** | **1,137,200** | **1.046** | **1.04%** | **80.6%** | ✅ |

结论：

- 连续视角窗口下位姿与深度高度自洽（焦距跨帧差异 4.6%，跨视角深度中位误差 ~1%，5% 内点率 80–95%）。
- 宽基线子集（跨 210 帧取 stride 26，视角差可达 100°+）一致性明显下降（焦距差 23.8%，中位误差 11.6%）——
  这是 VGGT-Ω 在**大视角间隔 + 纯前景黑背景**下的正常表现，不是导出错误（几何自洽性仍逐点校验通过）。

### 4.1 最终运行日志摘录

```
[15:29:xx] collecting frames ... 20 frame(s) selected (order preserved)
[15:29:xx] foreground ratios (original resolution): crop_0000.png=0.2300, ...
[15:29:xx] Resizing image .../crop_0179.png from (2120, 3791) to (384, 688) (center-crop (0,0,2120,3791) from original (2120,3791))
[15:29:xx] model input tensor: (20, 3, 688, 384), mask tensor: (20, 688, 384), foreground pixel fraction: 0.3319
[15:29:xx] loading checkpoint weights/vggt_omega_1b_512.pt on cuda
[15:29:xx] running VGGT-Omega inference
[run_inference] finished in 9.18 s (depth: (20,688,384,1), conf range: 1.002..19.824, depth range: 0.453..1.468)
[15:29:34] exporting GGGS dataset (foreground-mask-constrained COLMAP)
[export_gggs_dataset] wrote 1137200 foreground 3D points for 20 frames to .../sparse/0 in 53.892 s
[15:30:28] verifying dataset with GGGS reader from /tmp/gggs_community
[15:30:34] GGGS verification all_checks_passed=True points=1137200
```

耗时：预处理 ~2 s，推理 9.18 s（20 帧 @384×688），COLMAP 导出 53.9 s（1.137M 点，约 21k 点/s），
GGGS 回读 6 s。

### 4.2 导出数值检查

```
# cameras.txt（第一台）
20 PINHOLE 2120 3791 2868.9365577697758 2904.3337676469669 1062.2604166666667 1897.755087209302

# reconstruction_metadata.json
num_frames=20  num_points3d=1,137,200
confidence_threshold_percentile=20  confidence_threshold_value=3.8026
scene_alignment det(R)=+1.0000000539   ← 纯旋转，无 Y/Z 重复翻转
```

- 相机尺寸为**原始分辨率** 2120×3791，与 `images/` 下文件一致；
- `cx=1062.26 ≈ 2120/2`、`cy=1897.76 ≈ 3791/2`（相差 <0.5 个模型像素，即 VGGT-Ω 像素索引约定经
  下采样放大后的固有半像素量级，见 `docs/VGGT_OMEGA_TO_GGGS.md` §3）；
- 每帧一个 PINHOLE 相机，参数 = `A^-1 K_model`，逐帧独立。

### 4.3 GGGS 官方代码读取验证

`foresplat_adapter/verify.py` 直接导入 GGGS 的 `scene.dataset_readers.readColmapSceneInfo`
与 `utils.camera_utils.cameraList_from_camInfos`：

```
Reading camera 1/20 ... 20/20
cameras extent: 0.5471
all_checks_passed: True | points: 1137200
checks: ply_exists ✅ num_points_positive ✅ camera_count_matches ✅
        image_size_matches_intrinsics ✅ masks_dir_exists ✅
        all_masks_present ✅ colmap_files_present ✅
```

再用 GGGS 的 `GaussianModel.create_from_pcd()` 真正跑高斯初始化（已编译 `simple-knn`）：

```
Number of points at initialisation :  1137200
gaussians created: xyz (1137200, 3) | features (1137200, 16, 3)
                 | scaling (1137200, 3) | opacity (1137200, 1)
```

✅ 即：**GGGS 能读取该数据集并把 113.7 万个前景点初始化为高斯**。

## 5. 可视化

| 文件 | 内容 |
|---|---|
| `runs/KongQueZhuYu_20frames_512/GGGS_Dataset/debug/preview_pointcloud.png` | 前景点云三视图（可辨认出叶片纹理与带条纹的花盆，只有目标植株，无背景） |
| `runs/KongQueZhuYu_20frames_512/GGGS_Dataset/debug/preview_mask_overlay.png` | 前景 RGB + 掩膜叠加 |
| `.../debug/model_inputs/crop_XXXX.png` | VGGT-Ω **实际输入**（384×688，含 padding 情况） |
| `.../debug/model_masks/crop_XXXX.png` | 模型分辨率掩膜（NEAREST + 零填充） |
| `.../debug/foreground_pointcloud.ply` | 前景点云（带 quality 字段） |
| `.../debug/diagnostics.json` | 逐对相机的基线、视角、跨视角深度一致性 |
| `.../debug/predictions.npz` | 原始 depth / depth_conf / extrinsic / intrinsic / world_points |

点云三视图显示：花盆圆筒结构闭合、叶片呈放射状分布、色彩与输入前景图一致 —— 
说明相机位姿、内参与深度反投影在**同一坐标系**下自洽。

## 6. 明确未验证的部分

1. **GGGS 完整训练（`train.py`）未运行**：需要编译 GGGS 的 `diff_gaussian_rasterization` 等 CUDA 扩展
   （`simple-knn` 已编译并用于验证初始化）、以及按其论文配置超参；本阶段按要求不修改 GGGS 优化算法，
   故只验证到"数据集被读取 + 高斯初始化成功"。
2. **全 210 帧**未一次性推理（显存/时间）：本次用 20 帧连续窗口验证；全量可用 `--frame-stride`/`--max-frames`
   或分批，但 VGGT-Ω 的 inter-frame attention 是全局的，帧数越多显存越大。
3. **绝对尺度**：VGGT-Ω 输出为相对尺度（depth ≈0.45–1.47），未做米制标定。
4. 只使用 CPU 路径的单元测试与 CUDA 推理；CPU 推理路径未实测。

## 7. 环境补充说明（本机）

- `simple-knn` 在 CUDA 12.6 + torch 2.10 下需要给 `simple_knn.cu` 增加 `#include <cfloat>`
  （上游已知问题）。为避免改动 GGGS 仓库，本机是从**仓库副本** `/tmp/simple-knn-fixed` 编译安装到适配层 venv，
  GGGS 源码未做任何修改。
- 只做数据读取验证时不需要该扩展（`verify.py` 会对缺失的 `open3d` / `simple_knn._C` 自动打桩）。
