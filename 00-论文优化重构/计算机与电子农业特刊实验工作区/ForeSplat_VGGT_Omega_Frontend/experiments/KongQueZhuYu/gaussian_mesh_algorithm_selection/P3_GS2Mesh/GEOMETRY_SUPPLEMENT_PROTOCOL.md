# 固定尺度表面残差与真实边界拓扑补充

入口为 `scripts/mesh_surface_supplement.py`；只使用 CPU，不运行训练、不融合新深度、不更改原始 Mesh，也不发起新的 208 视角光栅化。可对 M1–M5、P1–P4、E12–E14 的每一个实际 raw/post/refined 文件分别调用。它补充已有相邻面 `1−cos(normal)` 指标，不替代轮廓、跨视角深度和完整性评价。

本次 Bocha 搜索返回 HTTP 403（配额不足）。局部平面点距定义参考实际抓取的 [CloudCompare CCCoreLib 官方源文件](https://github.com/CloudCompare/CCCoreLib/blob/master/src/Neighbourhood.cpp)，未把搜索摘要当作实验依据。源码快照保存在 `logs/official_neighbourhood_reference.cpp` 和 `logs/official_roughness_reference.cpp`。所有数值来自本地实际 Mesh 或明确标记的独立验证样例。

## 固定空间尺度和面积采样

标准 VGGT 使用历史冻结 `geometry_mesh_final_selection/configs/standard_spatial_scale.json` 的对角线 `D=0.8903337496969407`。Ω 单独使用 `omega_spatial_scale.json` 的 `D=0.6752206947257855`。不按每个输出 Mesh 的包围盒重新定尺度，不作 Sim(3) 或参考扫描拟合。180°刚体查看变换不影响距离和拓扑；如果要投影边界，必须使用该模型自身对应的实际相机。

默认目标采样间距是 `D/512`，支持样本数为 `ceil(三角形总面积/间距²)`，上限 **200,000**。支持点和 3,000 个独立查询点都以三角形面积为概率、用平方根重参数化的重心坐标均匀采样；Seed 0，两个独立随机流。这样不按网格顶点数量加权。支持样本达到上限时，记录 `effective_spacing=sqrt(area/count)`、`support_capped` 和各邻域有效比例，明确不同 Mesh 的样本间距可能增大。

默认半径固定为 `D/64` 和 `D/32`。标准 VGGT 分别是 `0.013911464839014698` 和 `0.027822929678029397`，Ω 分别是 `0.010550323355090399` 和 `0.021100646710180798`，单位为各自重建 gauge，不能解释为毫米。P1 原始约 15.4M 面，总面积约 4.213；200k cap 后平均间距约 0.00459，因此默认不采用更小的 `D/128`，避免仅少量邻居形成平面。CLI 允许事先约定其他半径，但跨模型比较必须使用相同半径和同一配置。

每个球形邻域用支持点质心和 3×3 协方差的最小特征向量拟合平面。至少需要 8 个邻居，并排除近似直线的退化邻域。独立查询点没有被加入拟合。

| 输出 | 定义 | 含义限制 |
|---|---|---|
| `query_plane_distance_*` | 查询点到邻域最小二乘平面的绝对距离 | 接近 CloudCompare 点云粗糙度定义；同时包含真实叶片曲率 |
| `local_plane_rms_*` | 邻域平面残差 RMS，即协方差最小特征值平方根 | 较少依赖单个查询点；仍受曲率、折叠和相邻薄表面影响 |
| `*_normalized_*` | 上述距离除以冻结 `D` | 标准 VGGT 内可直接对照；Ω 只是自身尺度归一化的历史代理 |
| `surface_variation_*` | 最小特征值 / 三个特征值之和 | 无单位的局部形状代理 |
| `valid_fraction` | 达到邻居和非退化平面条件的查询比例 | 必须一起报告，避免仅比较容易拟合的保留区域 |

每个半径分别报告 `all_spatial_neighbors` 和 `same_oriented_normal`。后者仅保留与查询三角面法线夹角≤60°的支持点，用于观察背面或邻近双层混入的影响。它依赖实际三角面绕序，并可能排除真实卷曲处；不把这一分支单独当作几何优劣结论。两个分支差异大时应结合 Depth/Normal/灰色 Mesh 检查相邻薄表面与错误粘连。

此指标**不是扫描 GT 距离误差**，也不能把较小残差直接解释为更准确：过度平滑、消失的叶尖和真实曲率减少均可能降低数值。各方法输出表面积和可见植物区域不同，也不是严格相同的解剖采样位置。

## Boundary Loops、开链与分叉网络

对实际三角形的无向边做 incidence 统计：incidence=1 为边界，incidence>2 为非流形边。以边界边构建顶点图，逐连通分量分类：

- 全部顶点 degree=2 且至少三个顶点：`closed_loop`，确定的简单闭环。
- degree≤2 且恰有两个 degree=1 端点：`open_chain`。此类可出现在非流形输入。
- 其他：`branched_boundary_network`，保留分叉顶点数量、端点数量和图 cycle rank `E−V+1`；不武断拆成多个孔洞。

保存每条闭环/链的有序真实顶点 ID、每个分量的真实边 ID、周长和最佳拟合平面上的投影多边形面积。**投影多边形面积不是孔洞面积**：完整叶片的外缘本身就围成闭环。不会将 boundary_edges、closed_boundary_loops 或 cycle_rank 标注为“真实叶片孔洞数”。面积采样没有改变拓扑判定，拓扑直接读取全部实际三角形。

15M 面的内存控制采用一个 int64 标量边码数组排序，避免显式构建 45M×2 的 int64 边矩阵。粗糙度计算以 100k 面分块求面积和法线，支持点 cap=200k，默认 4 个 KD-tree 工作线程。大型文件读取仍可能占用数 GiB，脚本记录实际峰值 RSS，不同时启动大量 Mesh 任务。

## 复用固定八视角的边界投影证据

可选传入 `--camera-index` 和 `--evaluation-dir`。前者接受实际 `camera_runtime.json` 的 `records` 列表，必须包含正确的 `Fx/Fy/Cx/Cy/world_view_transform` 和 source_id；后者已有 `views/<stage>_<source_id:04d>.npz`，至少包含 `mesh_depth`、`mesh_silhouette`、`foreground_mask`。使用固定八个 source_id：0、26、53、79、105、131、157、183。缺少缓存时记为 partial 或 caches_unavailable，不虚构评价。

每个边界分量按弧长采样，目标间距 `D/1024`、每分量上限 2048；只有相机 Z 与已有首交点 Depth 的 3×3 像素邻域相差≤`max(D/1024,0.002*Z)` 的点参与可见分类，遮挡点排除。三类证据分别保存：

- `foreground_mask_boundary_supported`：可见投影≥一半在 RAP 边缘 3px 内，属于前景外缘支持候选；RAP 包含花盆和标志物，不能声称全部是叶缘。
- `internal_missing_boundary_candidate`：可见投影≥一半临近 RAP 内部封闭缺失区域（距离≤3px），同时远离 RAP 外缘。内部缺失按与统一脚本相同的 3px 前景 rim 排除、连通域≥9px 定义。它仍是投影缺失代理，不能认证真实三维孔洞。
- `mixed_or_ambiguous` / `unobserved_or_occluded`：保留混合和不可见状态，不强行分配。

原始 RGB、模型 Depth 和空间位置可进一步验证候选，当前方法自身不推断物理叶片身份或多层真实缺口。

## 调用和输出

以下 `task_base` 只用于当前命令，不修改系统环境变量。Mesh 路径必须替换为实际存在的文件。

```bash
task_base='/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu'
task_p3="$task_base/gaussian_mesh_algorithm_selection/P3_GS2Mesh"
"$task_p3/.venv/bin/python" "$task_p3/scripts/mesh_surface_supplement.py" \
  --mesh "$task_p3/meshes_shared_depth_range/post.ply" \
  --out "$task_p3/geometry_supplement/post_shared" --frontend standard \
  --support-cap 200000 --queries 3000 --seed 0 --workers 4
```

已有八视角缓存后，仅补充边界投影，使用独立输出目录，不覆盖上面的整套记录：

```bash
"$task_p3/.venv/bin/python" "$task_p3/scripts/mesh_surface_supplement.py" \
  --mesh "$task_p3/meshes_shared_depth_range/post.ply" \
  --out "$task_p3/geometry_supplement/post_shared_projection" \
  --mode boundary --frontend standard \
  --camera-index "$task_base/geometry_mesh_final_selection/surface_inputs/e10_gggs_surface/camera_runtime.json" \
  --evaluation-dir '实际已有的统一评价目录' --stage post
```

M1 / VGGT-Ω 必须加 `--frontend omega`，投影时也使用 Ω 对应的相机缓存；其余固定标准前端使用 standard。`--mode roughness` 或 boundary 可以独立调用。不要给不同方法选用各自“更有利”的半径、采样上限或阈值。

主要输出为 `fixed_scale_roughness.csv`、`surface_samples_and_residuals.npz`、`boundary_components.csv`、`boundary_topology.npz`、可选 `boundary_projection_classes.csv` / `boundary_projection_views.csv`，以及 `roughness_complete.json`、`boundary_complete.json` 和总的 `supplement_complete.json`。总记录包含实际解析的绝对 Mesh 路径、大小、顶点数、面数、冻结 scale 文件、脚本 SHA256、依赖版本、耗时和峰值 RSS。

建议统一表另外保留固定半径/邻域命名的字段，例如 `plane_rms_r64_same_normal_normalized_mean`、`plane_rms_r32_same_normal_normalized_mean`、对应 valid_fraction、effective_spacing 和 support_capped；真实拓扑字段使用 `closed_boundary_loops`、`open_boundary_chains`、`branched_boundary_networks`、`boundary_graph_cycle_rank`。这些补充字段不覆盖旧 roughness 或 hole_area。

## 已执行验证

`scripts/validate_surface_supplement.py` 的独立小样例全部通过，结果在 `validation/surface_supplement/VALIDATION.json`：平面稀疏/密集重剖分仍为零、曲面非零、封闭四面体没有边界、环形面有两个简单闭环、共享单顶点的两个周期归为分叉网络、非流形额外三角面生成一条开链；前景内部缺失、RAP 自身空白和被前表面遮挡三种情形具有不同投影分类。验证只保存小型数值缓存，不导出或索引任何合成 Mesh 为真实实验模型。

实际 P3 shared-range Post（787,660 顶点 / 1,514,233 面）已执行默认 200k/3000 两半径补充。边界为 72,307 条，2,822 个简单闭环、84 个分叉网络、0 条开链，cycle rank=3,075、0 非流形边，数量与旧真实边统计一致。最终脚本的实际测量总耗时 5.54s，峰值 RSS 584 MiB；面积 1.49823，cap 后有效采样间距 0.002737。同向 r64 平面 RMS/D 的均值 0.00185817、有效比例 99.43%，r32 为 0.00495463、99.87%。只用于证明实际脚本可运行，尚不据此给全部方法排序。实际八视角投影尚待已有统一缓存，不在此制造新的全批评价。

P3 已有 `build_p3_views.py` 从首交点真实 primitive ID 的图像相邻转换形成线框，遮挡三角形不会透出；极小三角形仍受像素分辨率限制。最终统一实际边投影/纹理显示由共享脚本处理，本补充脚本不重复改变显示模块。
