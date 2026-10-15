# Phase20E-3 预声明主提取协议

本文件及 `configs/*_MESH.json` 在首次 E3 GPU 渲染、TSDF 提取和质量观察之前写定。参考来自 Phase20E-2 的 `NEXT_PHASE_MESH_REFERENCE.json`，不是根据本次结果选模。三个 M0 分别为 WanNianQing1/B_R0/Seed0、CaoMei2/B_R0/Seed0 和 HongZhang/B-C 共享 C_R1/Seed0。M1/M2 使用同一协议。

## 本地实现与参数来源

执行本仓库 `2d-gaussian-splatting-main/utils/mesh_utils.py` 的 `GaussianExtractor.reconstruction`、`extract_mesh_bounded` 和 `post_process_mesh`。不假定上游 CLI 与本地一致。`render.py` 实际提供 bounded/unbounded、mesh_res、voxel_size、depth_trunc、sdf_trunc、num_cluster 等参数；本阶段包装直接调用其相同函数，并保存真实浮点渲染以复算。未使用额外边界收缩或其他 meshing_mode。

- `bounded=true`，`depth_ratio=0.0`（原冻结 pipeline，期望表面深度），融合时 `active_sh_degree=0`，背景 RGB 为黑色。
- 每场景相机尺度：在冻结训练相机中，用本地 `focus_point_fn` 求相机光轴焦点；`r = min_i ||camera_center_i-focus||`。
- `depth_trunc=2r`，`voxel_size=2r/1024`，`sdf_trunc=5*voxel_size`。这等于本地原生默认尺度规则，三场景按同一公式计算；所有数值见冻结 JSON。
- 融合视图为全部冻结 train cameras，排除 held-out。相机、顺序及 RGBA 来源沿用冻结数据；不新建 COLMAP。
- 背景深度过滤直接采用原生 `gt_alpha_mask < 0.5 → depth=0`；alpha 来自冻结 RGBA 经原 loader 的缩放。没有新增 rendered-alpha 阈值或反向调整 ROI。
- Open3D ScalableTSDFVolume / RGB8，`depth_scale=1`。RGB 融合色按原生裁剪到 [0,1] 后 uint8；深度使用真实 float，正式外观评价另用真实 float RGB。
- 同时保留 raw Mesh 和 post Mesh。`num_cluster=1`；原生后处理保留三角形数达到最大组件数且至少 50 面的组件，同规模并列可保留，随后移除未引用顶点与退化面。
- 该后处理会丢弃较小的分离组件，不能预设其都是漂浮噪声，可能包括植物叶片。原始及后处理结构与覆盖均单独报告。
- 不使用 unbounded contracted marching cubes 作为主结果，不通过提取失败后换参数将敏感性结果替换主结果。

## 固定补充敏感性分析

对三个 M0 的同一批缓存真实 depth/RGB 和相机，补充 voxel 因子 0.5 和 2.0（主值为 1），depth_trunc 保持不变，sdf_trunc 始终为 5 voxel。三场景都执行同一组值，单独标为 supplementary，不能选最优替换主实验。无需重训。

## 结构及重投影评价

所有 Mesh 要求非空顶点与面、坐标有限、法向可重算、PLY 重读后统计一致。报告原始/后处理顶点、面、组件、最大组件占比、至少50面组件数、边界边、非流形边、退化面比例、包围盒、面积及相邻面法向余弦。退化阈值为 `max(bbox_diagonal²*1e-12,1e-20)`（用于叉积范数），不是清理 Mesh 的新算法。

在原冻结 held-out cameras 上做 Open3D CPU raycasting。使用本地 `to_cam_open3d` 的像素中心约定 `(W-1)/2,(H-1)/2`、冻结 PINHOLE FOV 和 qvec/tvec；ray 的相机 z 分量为1，t_hit 对应相机轴深度。在共同可见且 Gaussian alpha>0.5 的固定 R0 ROI 内计算 Mesh/Gaussian depth MAE/中位数/P95/相对差；另报告 ROI 可见覆盖和 silhouette proxy IoU。

这些指标只描述提取与 Gaussian 的一致性及预测 ROI 覆盖，没有独立几何 GT，不代表真实几何精度。面积单位仅为继承 COLMAP 任意单位的平方，不能换算 cm²，不计算真值 Chamfer 或 F-score。

## 可视化规则

每场景采用冻结 held-out 名称排序后的首个、中间、末个相机，不按本次指标挑图。保存原始 RGB、Gaussian RGB/alpha/depth/normal、raw/post Mesh depth/normal，以及 Mesh 边界图；另生成 Gaussian 中心和 Mesh 的真实3D视图。图件的色标、视角与缩放对配对模型一致；图像缩放仅用于展示，正式数值来自浮点文件。

## 可复现命令与环境

入口：`scripts/extract_mesh.py --sample <scene> --method M0|M1|M2`；`scripts/evaluate_mesh.py` 复算重投影。解释器为主 2DGS venv，物理 GPU1 UUID `GPU-6836f890-98b2-5891-07ef-675c25199c65`；环境与扩展哈希见 `reports/ENVIRONMENT.json`。每次开始检查同一 GPU 空闲；不切换设备。所有大文件在 `/data/fj/F2DMAS_phase20e3_data/1771e53f_v1/`，不覆盖历史。

官方文档仅用于确认 Open3D 接口；算法参数以本地冻结源码为准。Context7 已按先 library 后 docs 的流程读取 Open3D：

- https://www.open3d.org/docs/latest/tutorial/pipelines/rgbd_integration.html
- https://www.open3d.org/docs/latest/tutorial/geometry/ray_casting.html
- https://www.open3d.org/docs/latest/python_api/open3d.t.geometry.RaycastingScene.html
