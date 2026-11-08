# ForeSplat Frontend Bundle v1

统一接口供后续 GGGS、2DGS、其他3DGS和Mesh消费者读取。本阶段不启动这些模型训练，也不宣称已通过GGGS专有COLMAP读取验证。

```python
from frontends.common import FrontendBundle

bundle = FrontendBundle('experiments/KongQueZhuYu/Fast3R')
print(bundle.metadata['scale'])
cameras = bundle.cameras
for frame in bundle.frames():
    xyz = frame['predictions']['world_points']  # H,W,3；Bundle共同世界坐标
    confidence = frame['predictions']['world_points_conf']  # H,W或H,W,1
    mask = frame['mask']                       # H,W；未经腐蚀
    rgb = frame['rgb']                         # H,W,3 uint8
    c2w = frame['c2w']                         # 4,4 Camera-to-World
    k_original = frame['intrinsics_original']  # 3,3 原始像素坐标
    state = frame['pose_status']              # 失败/未注册状态不能忽略
```

每个Bundle保存：

- `bundle.json`：schema、方法、帧名称、完整/子集范围、坐标与尺度语义、过滤方式和运行元数据。
- `raw_predictions/{frame}.npz`：原始几何字段。Fast3R同时保留官方字段名称及共同字段别名；AMB3R保留world_points/depth/depth_metric/pose/intrinsic等；SfM另存每次局部预测及官方memory状态。
- `pointmaps/`、`confidence/`：点图与置信度；不能把Local点图直接合并为世界点云。
- `cameras/cameras.npz`：`c2w`、`w2c`、`intrinsics_model`、`intrinsics_original`、`names`。以 `pose_status.json` 判断失败或未注册帧；身份矩阵并不自动代表成功。
- Fast3R 的 `cameras/estimated_focal_lengths.json` 显式记录官方 PnP 使用的模型像素焦距及映射回原图的 x/y 焦距。其他模型的对应焦距读取保存的内参矩阵。
- `rgb/`、`mask/`：实际几何预测栅格；`pixel_transforms.json` 含原图尺寸、实际输入尺寸、resize/padding/crop及变换A与A逆。
- `pointcloud/global_raw.ply`、`global_confidence.ply`、`global_mask.ply`、`global_mask_confidence.ply`：四种过滤，完整数据不被下采样覆盖；`global_mask_confidence_q85.ply` 另存分位数基线。
- `pointcloud/global_per_frame`：每帧世界系前景点云；Fast3R `local_per_frame` 保持每帧独立相机系。`visualization_only.ply` 仅供显示。
- `per_frame_metrics.csv`、`RUN_REPORT.md`：观测、限制和真实运行记录。

像素中心映射为 `p_model=A@p_original`，`K_original=A^-1@K_model`。A包含实际整数resize尺寸的独立x/y比例、half-pixel偏移和padding平移；Mask使用完全相同几何操作与nearest插值。世界点无需为了换图像坐标而重新缩放，只有像素/内参映射。

读取接口中的 RGB/Mask 使用模型输入栅格，直接消费时配对 `intrinsics_model`。若后续方法使用原始分辨率图像，应按 `name` 从本实验共享的 `manifest.csv` 查找原始 RGB/Mask 路径，并配对 `intrinsics_original`；清单同时保存原文件 SHA256 和不可读帧原因，不重新分割或替换源图。

Scale字段必须传给消费端：Fast3R和现有VGGT-Ω为不确定场景尺度；AMB3R基础模型另有预测公制深度。由 depth_metric 与对应 unprojection 派生的点图/相机属于模型预测公制，但无本植物的实际距离验证。将深度尺度直接乘到独立 world_points Head 的结果只是条件性派生，当前没有验证两个 Head 使用同一尺度，不能作为已验证公制点云。原生/混合SfM最终地图经过局部注册尺度对齐，不能因某次调用存在MetricHead就宣称全局公制。

AMB3R 的额外 global_depth_unprojection_mask_confidence_metric_predicted.ply 沿用原始主筛选 world_points_conf>1 和相同 Mask；OFF/ON 完整点数为2,010,474 /9,219,511。这是同筛选像素的深度几何派生，不是原生 depth_conf 筛选点云。depth_conf 分支的99.814%/100%覆盖和相邻一致性另记于 amb3r_depth_channel_comparison.csv；消费端选择深度分支时需明确其置信度字段，不得把两套门限混称。

SfM官方地图管理器不保存或优化内参，Bundle将保留最后实际参与更新的局部内参及来源记录。最终融合点图和相机并非严格由同一K生成；评价必须记录该限制。

未注册帧保留清单、局部原始预测（若产生）、原因及标记；消费者应只初始化成功注册的相机和有效前景几何，不应将失败零值/NaN/单位矩阵回退当成已注册。

Fast3R-SfM 若有未注册帧，`local_unmapped_predictions/` 保留最近一次实际返回的局部预测及来源。读取接口只补充独立相机系的 `local_points` / `local_confidence`，并返回 `local_prediction_provenance.global_registration=False`；其中的簇世界点图和簇相机不替代失败的全局点图或全局相机。

读取接口的 `source_index` 是原始数字帧号，`frame_index` 是 manifest 中可用帧编号，`bundle_index` 是本次完整序列或子集中的数组位置。编号的缺号来自清单中明确记录的不可读源帧；子集数组位置也不同于完整清单编号，不能把数组下标当作原视频帧号。

本接口不伪造COLMAP特征、匹配或Tracks。如果未来导出GGGS专有结构，需要单独执行其 `readColmapSceneInfo()` 读取验收；当前只有通用Bundle读取验证。

来源：本目录 `frontends/common.py` 与真实Bundle字段；[Fast3R](https://github.com/facebookresearch/fast3r)、[AMB3R](https://github.com/HengyiWang/amb3r) 官方源码。
