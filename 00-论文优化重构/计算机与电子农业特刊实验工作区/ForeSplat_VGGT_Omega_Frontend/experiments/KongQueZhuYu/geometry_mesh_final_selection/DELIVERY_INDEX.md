# ForeSplat E11 / Mesh 最终交付

[最终流程决策](FINAL_PIPELINE_DECISION.md) · [后端分析](E11_BACKEND_ANALYSIS.md) · [Mesh分析](MESH_SELECTION_ANALYSIS.md) · [完整验收](configs/DELIVERY_VERIFICATION.json)

[渲染指标](comparisons/rendering_comparison.csv) · [结构指标](comparisons/mesh_structure_comparison.csv) · [208视角Mesh重投影](comparisons/mesh_reprojection_comparison.csv) · [扫描参照状态](comparisons/mesh_scan_comparison.csv) · [真实大文件索引](comparisons/LARGE_ASSET_INDEX.csv)

| 主实验 | 原始Mesh | Post Mesh |
|---|---|---|
| M1 Ω直接NKSR | [raw](meshes/M1_Omega_NKSR/raw.ply) | [post](meshes/M1_Omega_NKSR/post.ply) |
| M2 标准直接NKSR | [raw](meshes/M2_Standard_NKSR/raw.ply) | [post](meshes/M2_Standard_NKSR/post.ply) |
| M3 E10 GGGS→NKSR | [raw](meshes/M3_E10_GGGS_NKSR/raw.ply) | [post](meshes/M3_E10_GGGS_NKSR/post.ply) |
| M4 E10原生TSDF | [raw](meshes/M4_E10_GGGS_TSDF/raw.ply) | [post](meshes/M4_E10_GGGS_TSDF/post.ply) |
| M5 E11原生TSDF | [raw](meshes/M5_E11_2DGS_TSDF/raw.ply) | [post](meshes/M5_E11_2DGS_TSDF/post.ply) |

[固定八视角显示协议](comparisons/visualizations/protocol.json) · [有限真实视觉审阅](comparisons/visualizations/VISUAL_REVIEW.json) · [真实对照图](comparisons/visualizations/) · [E11正式模型/日志](E11_2DGS_Corrected/RUN_REPORT.md)

各Mesh目录同时保存全部组件/删除位置及208raw/post深度和轮廓；各surface_inputs目录保存208真实Depth/Normal/Alpha/Camera/RAP和点来源。大文件保留真实路径，不重复复制历史Checkpoint。E11→NKSR额外可选尝试两次OOM，没有成功Mesh，见[容量记录](meshes/supplementary/E11_2DGS_NKSR/RESOURCE_LIMIT.json)。未确认同株扫描，不提供虚构三维精度。
