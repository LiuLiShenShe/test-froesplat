# E14空TSDF恢复

同一缓存深度/RAP/K/W2C/voxel/truncation在Open3D0.20.0输出0体素0面，0.19.0首视角输出27332体素及7858三角面。该对照仅支持当前环境legacy TSDF实现故障，不宣称所有Open3D0.20安装都存在相同问题。官方融合参数与算法保持。

E14独立venv固定open3d==0.19.0，没有更改共享父环境。复用4×208已完成RGB/Alpha/Depth/Normal NPZ；通过实际runtime相机重建和逐项bitwise检查，避免加载无需参与融合的先验数组。原生SH0颜色重新渲染，深度/模型不变。三个阶段均输出非空raw/post，失败记录完整存于original_failed_records/。

|阶段|Raw顶点|Raw面|Post顶点|Post面|恢复提取秒|
|---|---:|---:|---:|---:|---:|
|cluster_before|1201727|2133288|1066227|1991108|30.558592796325684|
|cluster_after|735143|1327699|651950|1237026|27.310306549072266|
|final_30000|905989|1658690|836610|1588063|31.70599937438965|

原生Refinement及全历史评价继续运行，Mesh存在不等于几何质量认证。来源：实际缓存探测/提取日志；[Open3D官方RGBD融合文档](https://www.open3d.org/docs/latest/tutorial/pipelines/rgbd_integration.html)（Context7已读取）。博查搜索403不可用。
