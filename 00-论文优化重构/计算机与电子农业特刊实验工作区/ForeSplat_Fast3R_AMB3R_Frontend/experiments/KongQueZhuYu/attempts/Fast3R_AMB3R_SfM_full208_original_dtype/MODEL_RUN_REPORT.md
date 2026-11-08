# Fast3R_AMB3R_SfM RUN REPORT

状态：完成。实际输入 208 / 208 个可用视图；全量标志 True。物理 GPU1。

代码 AMB3R commit：`92c4081f910f98e683503092b85301861519175e`。
权重 SHA256：`1357d42e74539ba03510ea1986c7464a7facf17157779e2b7ebd6284e79e0427`。

长边 512，padding 对齐 16；完整视野，RGB/Mask 使用相同几何变换，Mask 最近邻，未腐蚀、未曝光处理。

执行官方聚类、初始化、coarse registration、keyframe BFS 与 non-keyframe refinement。仅增加观测与接口 shape bridge，没有修改网络或新建点云融合算法。

官方 allow_unmapped_frames 默认 false，本次 True。

成功注册 208；未映射 0。每帧状态、原始输入编号与原因见 registration_status.json 和 unmapped_frames.json。

pipeline 总时间 2073.706 s；局部模型推理合计 1260.933 s；峰值 allocated VRAM 4.606 GiB。

Fast3R commit：`33104d4b5b8df43795ecded236194958bbdac572`。

峰值 reserved VRAM 5.002 GiB；局部调用 1310 次。

官方 benchmark 提前返回 None：0 次；原生 Backend confidence 下降后退回 Frontend：0 次。这些是原始模型逻辑，None 调用没有伪造返回几何。

全部已返回局部调用的实际 PnP observer 计数：{'success': 9885}。not_called/started 为观测不完整；solver success 不代表几何精度。

最终 CPU bundle 导出 276.074 s。

局部原始预测序列化采用 none；该阶段累计 99.184 s。序列化策略只改变文件存储与运行开销，原始预测值不变。

地图参考帧：usable index 23，原始 source index 23，crop_0023.png。初始/最终 C2W 距 identity：0.00304128 / 0.234978。

原始每次网络 geometry/confidence/K/metric depth 保存为 local_predictions/call_*.npz。memory confidence 为官方融合的 (conf-1)/conf；统一 confidence 仅代数恢复表示，不是原始网络输出。

最终地图为归一化/尺度不确定的 pointmap/pose；AMB3R metric depth 单独保留，没有把 SfM 地图标为真实公制。未使用扫描点云尺度矫正。

官方 memory 不优化内参。最终 K 来自最近实际参与该帧的局部推理，accepted call ids 可追溯；geometry_validation.json 的残差包含融合几何和局部 K 的差异。

未进行 GGGS/2DGS/3DGS/mesh/曝光训练，没有伪造 COLMAP 特征 tracks。

[源码审计](../../../../docs/SFM_SOURCE_AUDIT.md)；[官方 SfM pipeline](https://github.com/HengyiWang/amb3r/blob/92c4081f910f98e683503092b85301861519175e/sfm/pipeline.py)；[AMB3R 论文](https://arxiv.org/abs/2511.20343)。

存储与时间：temporary SSD output; every artifact must be copied and verified before canonical completion。持久化复制 3250.702002219012 s；文件集合/尺寸已核验。全帧 camera、RGB、Mask、预测存档已经核验。共享 HDD 拥塞及 SSD/HDD 输出位置差异使 pipeline wall time 不能直接解释为模型速度差异。

[完整几何评价与统一图](GEOMETRY_RUN_REPORT.md)。派生数据16份已按独立清单逐SHA256同步；网络raw/位姿/RGB/Mask未替换。

从模型加载（输入已准备）到CPU bundle结束实测2465.544s；不含原始输入读取和SSD→HDD持久化。持久化复制及完整核验3250.702s。
