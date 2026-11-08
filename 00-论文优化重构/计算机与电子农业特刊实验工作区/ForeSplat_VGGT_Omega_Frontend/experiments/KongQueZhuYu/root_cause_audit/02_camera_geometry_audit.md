# 相机与像素投影审计

本报告使用完整 208 帧的实际 COLMAP、原始预处理记录和训练时导出的 cameras.json。E1/E5/E6 最终运行时各208个 Camera 已由 Checkpoint 重渲染实际保存并核对；旧 cameras.json 的 2160×3840 是原始 CameraInfo，不能当作训练尺寸。

## 已确认的数值事实

| 项目 | VGGT-Ω | 标准 VGGT |
|---|---:|---:|
| 原始 RGB W×H | 2160×3840 | 2160×3840 |
| 实际前端网格 W×H | 384×688 | 518×518 |
| 后端 resolution=2 尺寸 W×H | 1080×1920 | 1080×1920 |
| COLMAP cx−W/2，所有帧 | +2.312500 | 0 |
| COLMAP cy−H/2，所有帧 | +2.290698 | 0 |
| fx/W 中位数 | 1.480948 | 2.624415 |
| fy/H 中位数 | 0.814800 | 1.451933 |
| 按真实像素变换计算，与 GGGS 中心主点的训练像素二维偏移 | 1.981045 px | 2.620936 px |

逐帧六项内参比值和偏移在 `camera_projection_comparison/intrinsic_chain_per_frame.csv`，完整 A、K、W2C 和训练缩放矩阵在 `camera_matrix_chain.json`，分布图 `intrinsics_distributions.png/pdf`。所有 208 帧，而非只检查首帧。

## 主点确实丢失，但不是大误差的充分解释

本地 GGGS commit `23b418c2f0f19b30b251acad6fe13445dae968c7` 的 Camera、Dataset Reader、Camera Utils、Graphics Utils 与本轮直接读取的官方源码逐字节相同。

- `gggs_community/scene/dataset_readers.py:141` / `:146` 只读取 focal 参数并转 FoV；CameraInfo 不保存 cx/cy。
- `gggs_community/utils/camera_utils.py:21` 根据实际 image.size 和 resolution 缩放图像，把 FoV 传给 Camera。
- `gggs_community/scene/cameras.py:48`–`:51` 由 FoV 计算 Fx/Fy，Cx/Cy 强制 `(W−1)/2,(H−1)/2`。
- `gggs_community/utils/graphics_utils.py:72` 的对称投影矩阵没有主点偏移。
- `gggs_community/gaussian_renderer/__init__.py:49` 实际传入 full_proj_transform；CUDA `auxiliary.h:38` 用 `((ndc+1)*size−1)/2` 转整数像素中心。审计没有仅检查 Python 的 Cx/Cy 属性。
- CUDA `render_forward.cu:420`、`render_backward.cu:755`、sample kernels 还硬编码中心射线。因此仅给 P 加偏移，不能宣称完整修复 GGGS 的深度、法向和几何约束路径。

使用 256 个已知相机坐标三维点，按实际 CUDA 矩阵方向和 ndc2Pix 重现投影，所得结果与 GGGS 的中心 K 最大差 <1e−12 px。这个测试验证投影方向和像素约定，不证明真实相机准确。

官方来源：[GGGS Camera](https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids/blob/main/scene/cameras.py)、[Dataset Reader](https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids/blob/main/scene/dataset_readers.py)、[Graphics Utils](https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids/blob/main/utils/graphics_utils.py)。本轮快照及 SHA 在 `source_snapshots/official_source_fetch.json` / `local_source_versions.json`，来源更新不代替实际历史配置。

## 像素中心变换

Ω balanced 对本株不裁剪、不 padding；实际原图到模型矩阵是：

```text
AΩ = [[384/2160, 0, (384/2160−1)/2],
      [0, 688/3840, (688/3840−1)/2],
      [0, 0, 1]]
```

`vggt_omega/utils/load_fn.py:209` 正确记录 PIL 像素中心变换，`visual_util.py:374` 用 A⁻¹K 导出。其轻微 sx/sy 差异来自实际 patch 尺寸对齐，并非将所有原图强行压成正方形。

标准流程先黑色 square padding（左/右各 840 原始像素），3840 square→1024 bicubic→518 bilinear，未裁剪原始视野。`vggt_foreground/preprocess.py:40` 的 build_affine 保留 padding 和比例，却遗漏 `(s−1)/2`；保存的 A 在二维方向各少了 −0.4325520833 模型像素的偏移。正确 K_original 相对现有 K 在 cx/cy 各增加 3.206563707 原始像素。

独立 float ramp 经过真实 CPU Pillow bicubic 与 torch bilinear：旧 A 误差中位数 3.206526 原始像素；修正 A 后 0.001906，P95 0.002818。两次 resize 的正确 A 复合误差为 0；修正 P 与 K 的投影误差 <1.3e−13 px。`controlled_fixes/pixel_transform_adapter.py` 是独立修复，原始流程未改；结果 `pixel_transform_control.json`。RGB 内容、模型权重、焦距和外参均未改变。

标准源码避免了官方 padded point2D 1024/518 混用问题：实际导出已使用 A⁻¹，而不是未经检查照搬官方 rescale 函数。剩余半像素问题应修，但不能用来解释几十个模型像素的独立跨视角偏差。

官方来源：[VGGT pose encoding](https://github.com/facebookresearch/vggt/blob/main/vggt/utils/pose_enc.py)、[VGGT loader](https://github.com/facebookresearch/vggt/blob/main/vggt/utils/load_fn.py)、[official COLMAP demo](https://github.com/facebookresearch/vggt/blob/main/demo_colmap.py)、[Ω pose encoding](https://github.com/facebookresearch/vggt-omega/blob/main/vggt_omega/utils/pose_enc.py)。两种发布相机 decoder 都将模型主点固定 W/2,H/2，并不是网络分别学习了非中心主点。

## 外参、场景对齐与导出

两种 pose decoder 的 E 都是 OpenCV camera-from-world：x 向右、y 向下、z 向前。COLMAP qvec/tvec 保留此约定。GGGS Reader 将 R 转置以适应内部存储，getWorld2View2 再转回 W2C。所有 COLMAP Camera Center 与对应 Checkpoint cameras.json 完全相同，focal 完全相同；旋转最大数值差仅浮点误差。

Ω 原始 E 与导出 E 满足 `E_export=E_raw @ inv(T_align)`，208 帧最大差 2.69e−7。旋转正交性最大差 Ω3.49e−7、标准5.98e−7，未见造成数十像素偏差的非正交四元数问题。未发现重复 Scene Alignment 或将 C2W 当 W2C 的证据。

社区原始 exporter 曾用 pycolmap 属性后置修改场景，实际 ForeSplat exporter 在构造相机、点时显式施加同一个 T；不能把社区旧源码缺陷当成本次仍然存在的错误。`source_snapshots/community_visual_util.py` 保存官方社区版本供复核。

## 独立 RGB 跨视角证据

不使用任何预测深度，使用同一份归档 RGB、同一 Mask 提取 SIFT，双向 Lowe 0.75 匹配，再用图像独立 Fundamental RANSAC（1.5 模型像素、至少12 inliers）固定两模型共享的匹配点。随后测量保存相机预测 F 的 Sampson 距离。共享 384×688 网格避免各模型输入分辨率混用。

| 视角间隔 | 尝试/接受 pair | Ω 逐 pair 误差中位数 | 标准 VGGT | 图像 RANSAC 自身中位数 |
|---:|---:|---:|---:|---:|
| 1 | 69/69 | 3.001 px | 23.531 px | 0.211 px |
| 5 | 68/57 | 4.568 px | 49.620 px | 0.236 px |
| 15 | 65/24 | 8.942 px | 81.711 px | 0.146 px |

相邻 pair 的 within1.5px 支持比例逐 pair 中位数 Ω18.43%、标准0。较远视角接受率降低且容易退化，完整分布保存，但主证据优先近邻。RANSAC 自身误差包含拟合选择偏差，不作为相机 GT。叶片重复纹理、薄结构与非刚体运动仍可能影响匹配；共用同一匹配集合和近邻多数视角可避免把不同匹配条件混为比较结论。

已直接查看 `epipolar_overlay_0_1.png`、`epipolar_overlay_75_76.png`：匹配点可辨认为同一叶片纹理；有些视角 Ω 更好，有些标准更好，不能声称每帧都 Ω 优。这里旧overlay文件名使用208可用序列的index，原source文件名保存在图标题和 `epipolar_match_coordinates.json`；例如index135→136对应source136→137。该pair上部植株控制标准69.45 px、Ω0.76，修正标准半像素仅降到69.10 px。不能把数十像素错误归因于 0.43 模型像素映射。

所有结果在 `epipolar_pairs.csv`、`epipolar_summary.json`、`epipolar_match_coordinates.json`、`epipolar_principal_point_control.json`。本轮直接使用当前 [OpenCV 官方匹配文档](https://github.com/opencv/opencv/blob/5.0.0/doc/py_tutorials/py_features/py_matcher/py_matcher.markdown)，本地 cv2=5.0.0。

## 原始点图跨投影与射线

`camera_projection_overlays.py` 从保存原始 depth 和 model K 生成射线，保持明确世界坐标系，将source前景点投影至另一真实训练 RGB。三列分别用：保存 K 缩放、实际 GGGS 中心 K、真实像素变换 K。点大小、颜色、网格固定。输出4组 `cross_view_overlay_*.png`。

直接查看 source26→41：标准 source point cloud 与目标植株存在明显跨视角错层/左侧散布；三种 K 小修正没有消除它。Ω 也有局部双层与噪声。该证据来自预测 depth 与 pose 同源，不能代替上述独立 RGB 证据；自投影误差为零只说明代数闭合。

## 当前归因边界

Confirmed：主点丢弃、标准 half-pixel transform 缺失、标准已运行前端相机不能支持大量真实近邻 RGB 对应。Rejected：相机主点小偏移足以解释标准总体几何失配；帧名字顺序错误；重复对齐是当前主要错误。Unresolved：标准相机错误有多少来自 square/黑前景/208输入条件、有多少来自模型能力；不能以本轮结果宣称标准 VGGT 在所有场景失败。后续相机因素与初始化、训练的因果验证见 ROOT_CAUSE_ANALYSIS.md。


## 独立匹配点与保存深度的真实跨投影

将上述 RGB-only 匹配的源像素从保存 depth 反投影，再投向真正匹配的目标 RGB 像素；不是向同源 depth 的最近邻点比较。共同14组间隔1视角：Ω源depth投影误差逐pair中位数8.013 shared-grid px（p20高置信度8.151），标准111.264（p20为104.209）。间隔5共同11组：Ω16.367，标准193.370。两者使用同一个目标匹配集合，且目标纹理确实可见。结果 `independent_match_depth_projection.csv` / `_summary.json`。此诊断同时涉及预测深度和相机，极线误差单独提供与深度无关的相机证据。

已有前端参照在同14相邻pair上：ΩSampson4.332、Fast3R16.193、AMB3R-SfM7.117、Fast3R+AMB3R-SfM16.616像素。该子集与全69pair统计不同，不能混读中位数。两套SfM208/208注册不等于像素精度已通过，也没有证据支持立即把另一个已完成模型接入GGGS。原始Bundle均只读。结果 `reference_epipolar_pairs.csv` / `_summary.json`。

本轮实际E1 Checkpoint runtime208Camera验证：1080×1920，focal与COLMAP按训练缩放最大差6.83e−13，W2C差5.93e−8，实际full_proj相机射线投影回原像素最大误差0.000151px。E5和E6分别208Camera均验证，最大射线投影误差0.000288px。E6没有GGGS的Fx/Fy/Cx/Cy属性，验证使用实际FoV和投影矩阵，未伪造相机属性。这里验证相机/像素转换的实现自洽，独立RGB匹配仍不自洽；两者不矛盾。逐帧运行时记录和矩阵在 render_error_analysis/checkpoint_rerender_v1/{E1,E5,E6}；实际验证在 [actual_runtime_camera_verification.json](camera_projection_comparison/actual_runtime_camera_verification.json)。

## 焦距和相对位姿的单因素拆分

[camera_factor_control.py](controlled_fixes/camera_factor_control.py) 已在固定共同RGB匹配上实际执行2×2控制，主点固定原图中心：分别只替换相对位姿或fx/fy。没有交换PLY或未经Sim(3)对齐的世界点；相对W2C消除世界坐标gauge。

| 固定主点的相机组合 | 相邻14对Sampson中位数 | 间隔5的11对 |
|---|---:|---:|
| Ω pose + Ω focal | 4.345 px | 6.634 px |
| Ω pose + Standard focal | 4.633 px | 10.836 px |
| Standard pose + Ω focal | 19.897 px | 21.440 px |
| Standard pose + Standard focal | 23.491 px | 20.696 px |

标准焦距替换为Ω焦距仍有19.897px；固定Ω焦距后仅改变相对位姿明显恶化近邻对应。这支持相对位姿是该已运行标准数据中的重要问题，不能用焦距大小差异单独解释。此为局部独立图像控制，不是全208相机已校准或Gaussian重训收益。原始结果 [camera_factor_results.json](controlled_fixes/camera_factor_results.json)。
