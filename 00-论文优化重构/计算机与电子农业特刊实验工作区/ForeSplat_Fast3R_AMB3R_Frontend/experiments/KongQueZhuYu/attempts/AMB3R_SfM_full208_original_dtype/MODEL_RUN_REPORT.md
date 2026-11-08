# AMB3R_SfM RUN REPORT

状态：完成。实际输入 208 / 208 个可用视图；全量标志 True。物理 GPU1。

代码 AMB3R commit：`92c4081f910f98e683503092b85301861519175e`。
权重 SHA256：`2d9342282cce0a20e1074ad1513d9a18c3d100dc06c734a4a11f2d1baeea5239`。

长边 518，padding 对齐 14；完整视野，RGB/Mask 使用相同几何变换，Mask 最近邻，未腐蚀、未曝光处理。

执行官方聚类、初始化、coarse registration、keyframe BFS 与 non-keyframe refinement。仅增加观测与接口 shape bridge，没有修改网络或新建点云融合算法。

官方 allow_unmapped_frames 默认 false，本次 True。

成功注册 208；未映射 0。每帧状态、原始输入编号与原因见 registration_status.json 和 unmapped_frames.json。

pipeline 总时间 2644.641 s；局部模型推理合计 1164.768 s；峰值 allocated VRAM 11.617 GiB。

峰值 reserved VRAM 36.475 GiB；局部调用 1403 次。

官方 benchmark 提前返回 None：352 次；原生 Backend confidence 下降后退回 Frontend：32 次。这些是原始模型逻辑，None 调用没有伪造返回几何。

此旧 runner 未计时最终 CPU bundle 导出与完整端到端耗时；未用文件时间推算缺失值。

局部原始预测序列化采用 none；该阶段累计 1022.578 s。序列化策略只改变文件存储与运行开销，原始预测值不变。

地图参考帧：usable index 189，原始 source index 190，crop_0190.png。初始/最终 C2W 距 identity：0.000422903 / 0.341268。

原始每次网络 geometry/confidence/K/metric depth 保存为 local_predictions/call_*.npz。memory confidence 为官方融合的 (conf-1)/conf；统一 confidence 仅代数恢复表示，不是原始网络输出。

最终地图为归一化/尺度不确定的 pointmap/pose；AMB3R metric depth 单独保留，没有把 SfM 地图标为真实公制。未使用扫描点云尺度矫正。

官方 memory 不优化内参。最终 K 来自最近实际参与该帧的局部推理，accepted call ids 可追溯；geometry_validation.json 的残差包含融合几何和局部 K 的差异。

未进行 GGGS/2DGS/3DGS/mesh/曝光训练，没有伪造 COLMAP 特征 tracks。

[源码审计](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Fast3R_AMB3R_Frontend/docs/SFM_SOURCE_AUDIT.md)；[官方 SfM pipeline](https://github.com/HengyiWang/amb3r/blob/92c4081f910f98e683503092b85301861519175e/sfm/pipeline.py)；[AMB3R 论文](https://arxiv.org/abs/2511.20343)。

持久化：直接写入/data HDD，全部3795文件和208相机/RGB/Mask/预测记录已核验；persistence_verification.json记录核验。原始NPZ未更改。共享HDD I/O拥塞，模型/序列化/pipeline时间分别报告。
