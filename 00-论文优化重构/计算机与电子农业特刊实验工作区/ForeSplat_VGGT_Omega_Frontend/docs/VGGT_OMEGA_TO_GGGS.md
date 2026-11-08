# VGGT-Ω 原始输出 → GGGS 相机与点云：转换说明

本文档解释 ForeSplat 的 VGGT-Ω 前端如何把 **RAP-FSAM3 前景图像 + 二值掩膜** 转换为
GGGS（HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids）可读取的 COLMAP 数据集。
所有结论均来自对社区仓库 `tomhol/vggt-omega` 源码的实际阅读与实测，而不是假设。

---

## 1. VGGT-Ω 推理接口（源码事实）

### 1.1 模型与权重

- `vggt_omega/models/vggt_omega.py: VGGTOmega`，`aggregator`（24 层交替注意力 transformer）
  + `camera_head` + `dense_head`。
- 输入：`images: (B, N, 3, H_model, W_model)`，RGB，float，范围 `[0,1]`；
  `H_model`、`W_model` 必须是 `patch_size = 16` 的整数倍（`DenseHead` 用 `pixel_shuffle`
  解码到 patch 分辨率，因此需要整除）。
- 权重：`vggt_omega_1b_512.pt`（1.14B 参数），`strict=False` 加载——
  实测 **0 missing / 0 unexpected**，即权重完整匹配。

### 1.2 `predictions` 字典的真实结构（`VGGTOmega.forward`）

| key | 形状 | 含义 |
|---|---|---|
| `camera_and_register_tokens` | `(B, N, 17, 2048)` | camera token + 16 register token |
| `pose_enc` | `(B, N, 9)` | `[t(3), quat(4), fov_h, fov_w]` |
| `depth` | `(B, N, H, W, 1)` | 度量深度，`exp(logits)`，恒 `> 0` |
| `depth_conf` | `(B, N, H, W)` | 置信度，`1 + exp(logits)`，恒 `>= 1` |
| `images` | `(B, N, 3, H, W)` | **实际送进网络**的输入（`not training` 时才返回） |

### 1.3 `pose_enc` → 外参 / 内参（`utils/pose_enc.py: encoding_to_camera`）

```
T           = pose_enc[..., 0:3]      # 平移
q           = pose_enc[..., 3:7]      # 四元数 -> R
extrinsics  = [R | T]                 # (N, 3, 4)，world-to-camera（OpenCV 约定）
fy = (H/2) / tan(fov_h/2) ; fx = (W/2) / tan(fov_w/2)
intrinsics  = [[fx,0,W/2],[0,fy,H/2],[0,0,1]]      # (N, 3, 3)，**模型分辨率**
```

要点：

- 内参主点是 `(W/2, H/2)`（不是 `(W-1)/2`），这是 VGGT-Ω 自己的坐标约定，
  与它自己的反投影公式一致（见 1.4）。不要"修正"它，否则反投影与观测不再自洽。
  （后果：经 `A^-1` 映射回原始分辨率后，`cx` / `cy` 与图像中心会相差
  `(0.5/s - 0.5)` px ≈ 0.5 个模型像素 × 下采样倍数；本例 20 帧 @384×688 → 2120×3791，
  `cx=1062.26` vs 1060、`cy=1897.76` vs 1895.5。这是 VGGT-Ω 像素索引约定的忠实映射，
  对内参/观测/点三者**完全自洽**（逐点重投影误差 < 1e-5 px），占图像宽度 ~0.1%，可忽略。）
- `extrinsic` 是 **world→camera**，与 COLMAP 的 `cam_from_world` 同义，
  因此可以直接构造 `pycolmap.Rigid3d(R, t)`（社区实现正是如此）。

### 1.4 `world_points_from_depth` 的生成方式（`demo_gradio.py: unproject_depth_map_to_point_map`）

```python
camera_points = [ (x - cx)/fx * d,  (y - cy)/fy * d,  d ]      # x,y = 像素整数索引
world_points  = R^T (camera_points - T)                        # R,T 取自 extrinsic
```

即：**每个像素的深度单独反投影**，得到 `(N, H, W, 3)` 的世界坐标点。
这决定了下面第 5 节的核心语义：导出的 3D 点都是"单视角深度反投影点"，
而不是跨视角 SfM 三角化匹配点。

### 1.5 `depth_conf` 的含义

`1 + exp(logit)`，越大约可信；`proj_conf` 采用小初始化，均值约 1.05。
社区导出用**百分位阈值**（默认 20）对整幅图取分位数。ForeSplat 前端改为
**先应用前景掩膜、再在前景点上取分位数**（否则前景点容易被背景/边缘的高置信度挤压）。

### 1.6 图像预处理（`utils/load_fn.py: load_and_preprocess_images`）

对每张图依次执行：

1. **中心裁剪**到 `aspect_ratio = h/w ∈ [0.5, 2.0]`（超出才裁）；
2. **缩放**：`balanced` 模式保持 token 数 ≈ `(res/16)^2`；`max_size` 模式把长边缩到 `res`；
3. **Padding**：仅当各帧尺寸不一致时，padding 到共同最大尺寸，RGB 用 `1.0`（白）填充；
4. RGB 用 **BICUBIC**；无 patch 对齐之外的额外处理（目标尺寸本身已是 16 的倍数）。

ForeSplat 扩展（`load_and_preprocess_images_with_masks`）对 **RGB 与 Mask 使用完全相同的
裁剪框 / 缩放比例 / padding**，但 Mask 用 **NEAREST** 重采样、**0** 填充，
不使用双三次插值处理二值掩膜。

---

## 2. 像素坐标仿射变换 A

PIL 的重采样约定：输出像素中心 `i + 0.5` 采样输入坐标 `(i + 0.5) / s`（`s = 目标/裁剪后`）。
按 裁剪 → 缩放 → padding 的顺序，原始像素 `p_original` 到模型输入像素 `p_model` 的映射是仿射：

```
x_model = s_x * x_orig + (pad_left - s_x*crop_left + 0.5*s_x - 0.5)
y_model = s_y * y_orig + (pad_top  - s_y*crop_top  + 0.5*s_y - 0.5)

p_model = A @ p_original ,  A = [[s_x, 0, t_x],
                                 [0, s_y, t_y],
                                 [0,   0,   1 ]]
```

A 及其逆矩阵逐帧写入 `metadata/preprocess_transforms.json`
（字段 `pixel_transform` / `pixel_transform_inverse`），并用"单个亮像素 → 重采样后质心"的
数值实验验证过（误差 < 0.75 px，见 `tests/`）。

社区版 `save_colmap()` 只用 `s_x = W_orig / W_model` 缩放，**假定没有裁剪与 padding**。
当输入被中心裁剪（极端长宽比）或各帧尺寸不一致时，这种缩放会把内参与 2D 观测放错位置。
ForeSplat 前端改为严格的 `A^-1` 变换。

---

## 3. 内参与 2D 观测的逆变换

模型内参满足 `p_model = K_model X_cam`，而 `p_original = A^-1 p_model`，于是

```
K_original = A^-1 @ K_model
p_original = A^-1 @ p_model
```

实现（`visual_util.py: save_colmap`）：

- 相机：每帧一个 `PINHOLE` 相机，`params = [fx, fy, cx, cy]` 取自 `K_original`，
  `width/height = 原始分辨率`（不是模型分辨率）。
- 观测：每个保留下来的像素 `(x,y)` → `A^-1 [x,y,1]`，写入该帧的 `image.points2D`。
- 因此 **padding 区域永远不产生有效观测**：掩膜在 padding 处为 0 且几何/padding 外像素不属于任何图像。
- 不同分辨率图像各自使用自己的相机内参（camera_id = image_id = frame_index + 1）。

---

## 4. 前景约束的点云筛选

严格按以下顺序（`visual_util.py: _build_valid_mask`）：

```python
valid = foreground_mask            # RAP-FSAM3 掩膜，映射到模型分辨率
      & finite_geometry            # world_points 有限
      & valid_depth                # depth 有限且 > 0
      & (conf 有限)

conf_threshold = percentile(conf[valid], conf_thres)     # ← 只在前景点上统计！
valid &= conf >= conf_threshold
valid &= conf > min_conf
valid &= mask_black_bg / mask_white_bg  (可选，保持社区兼容)
valid = randomly_limit_trues(valid, max_points, rng=seed)  # 可复现的随机上限
```

- **深度边缘过滤默认关闭**：植物叶片轮廓本身就是强几何区域，
  社区默认开启 3×3 相对深度跳变检测（`rtol=0.03`）会把叶缘点成批删掉。
  需要时用 `--filter-depth-edges [--depth-edge-rtol ...]` 显式开启。
- 点颜色取自 `predictions["images"]`——即**实际送进 VGGT-Ω 的 RAP-FSAM3 前景 RGB**
  （RGBA 输入先在黑色背景上合成，与 RAP-FSAM3 输出形式一致）。
  **不使用"黑色背景"启发式替代掩膜**。

---

## 5. 点云语义：单视角深度反投影（重点）

每个保留下来的像素 `(x,y)` 在它自己的那一帧里产生 **一个** 3D 点：

- COLMAP `points3D.track` 长度恒为 **1**，即该点只被一帧观测到；
- 因此该点的 `error`（重投影误差）恒为 0，**不能**当作光束法平差残差；
- 这些点 **不是** 跨视角 SfM 三角化得到的匹配点，不能用于"匹配点数""三角化成功率"等指标。

这一点在 `reconstruction_metadata.json: point_semantics` 与 `points3D.ply` 的 PLY header
comment 中都有显式声明。

尽管如此，这些点仍然满足多视角几何一致性：把帧 i 的点投影到帧 j 的相机，
与其最近观测（≤3 px）的深度对比，实测中位相对误差见 `docs/RUN_REPORT.md`
（连续 10 帧窗口：中位 0.65%，p90 2.4%，5% 内点率 94.7%）。

---

## 6. 场景坐标对齐（只做一次，不重复 Y/Z 翻转）

社区 `save_colmap()` 与 GLB 导出共用同一个对齐矩阵（`visual_util.py: get_scene_alignment_transform`）：

```
T_align = inv(E_0) @ diag(1, -1, -1) @ rotY(180°)      # det = +1，是纯旋转
```

- 点：`X' = T_align X`
- 相机（world→camera）：`E' = E @ T_align^-1`

两者一起变换，投影关系不变（已用逐点重投影实验验证，误差 < 1e-5 px）。
ForeSplat 前端把这个变换 **在创建点/相机时一次性施加**，不再事后回写：
pycolmap ≥ 3.x 中 `reconstruction.points3D[id].xyz = ...` 返回的是副本，
社区版事后回写的做法在 3.x 上会**静默失效**（这正是本次实测中发现并修复的问题）。

`det(T_align[:3,:3]) = +1` 说明没有镜像翻转；代码里也只用这一处翻转，
不存在重复的 Y/Z 轴翻转。

---

## 7. COLMAP → GGGS

GGGS 的读取路径（`scene/dataset_readers.py: readColmapSceneInfo`）：

1. 先尝试 `sparse/0/images.bin` + `cameras.bin`（失败才回退 `.txt`）；
2. `readColmapCameras` 用 `R = qvec2rotmat(qvec).T`、`T = tvec`，只接受
   `PINHOLE` / `SIMPLE_PINHOLE`（我们写 `PINHOLE`）；
3. 图像路径 `images/<basename(image.name)>`，因此 COLMAP 里的 `image.name` 必须与
   `images/` 下的文件名一致（我们保留原始文件名）；
4. `points3D.ply`：若存在则直接 `fetchPly()`（**需要** `x,y,z,nx,ny,nz,red,green,blue` 字段），
   否则会从 `points3D.bin` 转换并覆盖生成——所以我们自己写出带法线的 `points3D.ply`；
5. `nerf_normalization` 由相机中心计算，`utils/camera_utils.loadCam` 按分辨率读取图像，
   相机内参来自 COLMAP，图像尺寸与 `camera.width/height` 必须一致（我们逐帧校验）。

法线：指向对应相机中心（`normalize(C_cam - X)`），与社区 GLB 导出使用的朝向一致。

---

## 8. 输出目录

```
GGGS_Dataset/
├── images/             RAP-FSAM3 前景 RGB，原始分辨率，文件名不变
├── masks/              二值掩膜（0/255），原始分辨率，同名
├── sparse/0/
│   ├── cameras.bin / images.bin / points3D.bin      pycolmap 写出
│   ├── cameras.txt / images.txt / points3D.txt
│   └── points3D.ply     前景约束点云（带法线，GGGS 初始化用）
├── metadata/
│   ├── preprocess_transforms.json   每帧 A / A^-1、裁剪、缩放、padding
│   ├── image_manifest.json          每帧原始尺寸、模型尺寸、前景比例、观测数
│   ├── reconstruction_metadata.json 阈值、相机参数、对齐矩阵、点语义说明
│   └── run_log.json                 本次运行的完整参数与日志
└── debug/
    ├── model_inputs/                VGGT-Ω 实际看到的图像（模型分辨率）
    ├── model_masks/                 模型分辨率掩膜
    ├── foreground_pointcloud.ply    前景点云（含 quality 字段）
    └── predictions.npz              （--save-predictions 时）原始 depth/conf/pose
```

---

## 9. 坐标尺度说明

VGGT-Ω 输出的深度是**归一化/相对尺度**（实测 depth 范围约 0.6–1.5），
不是米制。COLMAP 与 GGGS 都不要求绝对尺度（GGGS 用相机中心计算 `cameras_extent`
做场景归一化），因此不影响训练；但若后续需要真实尺度（如与 GT 尺寸对比），
必须在外部用已知尺度标定。
