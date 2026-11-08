# AMB3R_SfM RUN REPORT

状态：完成。实际输入 32 / 208 个可用视图；全量标志 False。物理 GPU1。

代码 AMB3R commit：`92c4081f910f98e683503092b85301861519175e`。
权重 SHA256：`2d9342282cce0a20e1074ad1513d9a18c3d100dc06c734a4a11f2d1baeea5239`。

长边 518，padding 对齐 14；完整视野，RGB/Mask 使用相同几何变换，Mask 最近邻，未腐蚀、未曝光处理。

执行官方聚类、初始化、coarse registration、keyframe BFS 与 non-keyframe refinement。仅增加观测与接口 shape bridge，没有修改网络或新建点云融合算法。

官方 allow_unmapped_frames 默认 false，本次 True。

成功注册 32；未映射 0。每帧状态、原始输入编号与原因见 registration_status.json 和 unmapped_frames.json。

pipeline 总时间 360.266 s；局部模型推理合计 65.736 s；峰值 allocated VRAM 9.766 GiB。

原始每次网络 geometry/confidence/K/metric depth 保存为 local_predictions/call_*.npz。memory confidence 为官方融合的 (conf-1)/conf；统一 confidence 仅代数恢复表示，不是原始网络输出。

最终地图为归一化/尺度不确定的 pointmap/pose；AMB3R metric depth 单独保留，没有把 SfM 地图标为真实公制。未使用扫描点云尺度矫正。

官方 memory 不优化内参。最终 K 来自最近实际参与该帧的局部推理，accepted call ids 可追溯；geometry_validation.json 的残差包含融合几何和局部 K 的差异。

未进行 GGGS/2DGS/3DGS/mesh/曝光训练，没有伪造 COLMAP 特征 tracks。

[源码审计](../../../docs/SFM_SOURCE_AUDIT.md)；[官方 SfM pipeline](https://github.com/HengyiWang/amb3r/blob/92c4081f910f98e683503092b85301861519175e/sfm/pipeline.py)；[AMB3R 论文](https://arxiv.org/abs/2511.20343)。
