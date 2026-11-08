# KongQueZhuYu 几何前端运行总报告

更新：2026-11-08（Asia/Shanghai）。本阶段新运行的模型推理仅使用物理 GPU1，所有依赖通过 uv 管理。

## 实际执行范围

本阶段只处理 KongQueZhuYu。Fast3R、AMB3R Base OFF/ON、原生 AMB3R-SfM 和 Fast3R + AMB3R-SfM 均已真实完成208帧计算、完整持久化、统一评价与可视化。加上既有Ω参考和独立FP32对照，七份Bundle最终v2审计全部通过、issues=[]，每份208帧/208有限相机，累计6286个PLY检查。run_registry.json全部为complete，实际验收汇总见 shared_config/bundle_audit_summary.json。

输入直接来自 /data/fj/F2DMAS/03-SAM/KongQueZhuYu，用户确认 RAP-SAM3 与 RAP-FSAM3 指同一已有处理结果。210 个标准 RGB 名称均有对应 Mask；crop_0044.png、crop_0206.png 无法解码，其清单行与异常原因保留。208 个可读图像按原始帧号排序使用，没有主动抽帧。manifest.csv 保存路径、配对、原分辨率、状态及原文件 SHA256。

原图均为 W2160 × H3840。Fast3R 保持完整视野，缩放为 W288 × H512，无 padding/crop；AMB3R 缩放为 W291 × H518，再左右 padding 1/2 像素成为 W294 × H518。RGB bicubic、Mask nearest，二者共享相同几何操作。pixel_transforms.json 保存整数缩放参数、padding/crop、半像素中心变换 A 及其逆，K_original=A^-1 K_model。源图未修改，没有曝光补偿或 Mask 腐蚀。

## 环境、网络与权重

GPU1 为 NVIDIA RTX A6000 48GB，驱动 590.48.01；驱动显示的 CUDA 上限为 13.1，不等于各环境实际 Torch CUDA runtime。进程 CUDA_VISIBLE_DEVICES=1 后逻辑 cuda:0 指物理 GPU1。其他用户进程未被终止或更改。

| 隔离环境 | Python | PyTorch | Torch CUDA |
| --- | --- | --- | --- |
| .venv-fast3r | 3.11.9 | 2.10.0+cu128 | 12.8 |
| .venv-amb3r | 3.11.9 | 2.5.0+cu121 | 12.1 |

uv 版本为 0.10.12。两环境 uv pip check 均通过。Fast3R 未安装 DUSt3R cuROPE。AMB3R 的 FlashAttention 使用对应 Torch2.5/CUDA12/cp311/ABI-FALSE 的 2.7.3 wheel；其他 CUDA 扩展兼容细节、固定版本和安装记录见 ../../docs/AMB3R_SOURCE_AUDIT.md、../../frontends/amb3r/requirements.lock.txt。现有 VGGT-Ω 和 GGGS 环境未改动，也未升级全局 PyTorch。

| 模型 | 官方代码 Commit | 权重 SHA256 |
| --- | --- | --- |
| Fast3R | 33104d4b5b8df43795ecded236194958bbdac572 | 1357d42e74539ba03510ea1986c7464a7facf17157779e2b7ebd6284e79e0427 |
| AMB3R | 92c4081f910f98e683503092b85301861519175e | 2d9342282cce0a20e1074ad1513d9a18c3d100dc06c734a4a11f2d1baeea5239 |

Fast3R 官方仓库归档状态已直接核实，代码/权重采用 FAIR Noncommercial Research License。AMB3R 当前仓库根目录未发现 LICENSE，未自行指定许可证。官方网络和预训练权重未修改；独立 Adapter、输入/输出封装及观察记录器位于 frontends/。

## 流程与输出

Fast3R 先完成 3/16/64 视角检查，再一次输入全部 208 视角完成全局 BF16 前馈，完整保留 Global Fusion 交互。官方 PnP 使用 niter_PnP=100、first_view_from_global_head，实际 208/208 成功。Global Point Map 与 Local Point Map 分开保存。额外全208视角外层 FP32 对照与前景有效点 PnP 诊断独立保存，没有替换 BF16 主结果；官方内部 Attention BF16 保留，不能称为完全 FP32 网络。

AMB3R OFF/ON 均使用全部 208 帧，相同 RGB/Mask/顺序/像素变换/随机种子 42 和相同权重，分别调用官方 forward(iters=0/1)。原始 point head、depth/depth_metric、pose/intrinsic、pts3d_by_unprojection 保留。OFF/ON 配对核验保存 amb3r_off_on_geometry_controls.json 和 shared_config/amb3r_base_pair.json。

原生 SfM 使用官方聚类、粗注册、关键帧全局映射和非关键帧细化，208/208 注册、73 个关键帧、0 未映射。实际 1,403 次局部调用中，352 次是官方 benchmark early stop（没有返回几何），32 次返回结果采用官方 Backend-to-Frontend fallback。返回的 1,051 组局部原始几何另存，不把最终 memory 的内建融合结果称为原始网络预测。

混合 Adapter 同时实现 extract_amb3r_sfm_features 与 run_amb3r_sfm。特征来自 Fast3R 原有图像编码器的 patch pooling；每次局部推理完整保留簇内全局交互，输出在该簇第一相机参考系，再由官方 SfM 动态对齐/管理地图。公共接口 confidence 为 B,T,H,W,1，桥接器只移除尾部单例维以满足真实 pipeline，不改变置信度值。Fast3R 与 AMB3R 的 1+exp 表示可代数使用 (c-1)/c，但不代表统计校准。混合流程 208/208 注册、35 个关键帧、0 未映射，全部 9,885 个实际观察的局部视角 PnP 成功。

SfM 的默认失败处置通过官方 allow_unmapped_frames=True 明确保留未映射信息；本次两流程最终均为0。聚类/注册阈值未为本植物随意调整。关键帧、簇信息、局部调用来源、接受状态、原始 memory、相机及几何验证均归档。官方 memory 不优化内参，最终 K 来自最后实际参与该帧更新的局部调用；融合几何与 K 的自投影残差必须连同此限制解读。没有归档每次动态对齐的完整 Sim3 矩阵轨迹，不能把最终融合地图解释为每簇单一静态变换的直接拼接。

## 实测计算时间与显存

| 流程 | 已测范围 | 时间 | Torch allocated peak |
| --- | --- | ---: | ---: |
| Fast3R | 一次全208网络前向；不含PnP/序列化 | 20.09s | 13.04GiB |
| Fast3R | 网络+官方CPU PnP；不含读图/加载/导出 | 29.28s | 同上 |
| AMB3R OFF | 官方全208 forward(iters=0) | 43.65s | 23.67GiB |
| AMB3R ON | 官方全208 forward(iters=1) | 120.11s | 30.83GiB |
| AMB3R-SfM native | pipeline，含局部模型/官方注册与HDD局部导出 | 2644.64s | 11.62GiB |
| Fast3R + AMB3R-SfM | pipeline，含局部模型/PnP/官方注册与SSD局部导出 | 2073.71s | 4.61GiB |

原生/混合局部模型调用累计1164.77/1260.93s，局部原始结果导出累计1022.58/99.18s。混合SSD Bundle导出276.07s；从模型加载到Bundle结束实测2465.54s（此前已准备输入，不含原图读取及HDD复制）。这些范围不可互换；混合较短pipeline wall-time主要伴随更快原始输出存储，不能称其局部基础模型推理更快。原生总前端/Bundle导出没有独立计时，保留未知。持久化复制时间在 persistence_verification.json 中单独记录。以上显存为本任务Torch分配峰值，不是整卡其他进程占用之和。

## 评价与限制

comparison.csv / comparison.md 汇总主方法及独立 FP32 对照；final_analysis.md 给出七项技术结论。评价包含覆盖/叶缘/局部密度代理、相邻投影深度一致性、世界点距离、轨迹、置信度及预测公制深度。分位数 PLY 与 baseline conf>1 相交，OFF 的大量 confidence=1 不会被 q85 重新接纳。原始完整点云保留，可视化采样为单独副本。

现有 VGGT-Ω 参考由保存的深度、W2C/K、RGB/Mask 重新反投影，未使用旧融合或截断 PLY。其输入 W384×H688，且没有可用的原始 point-head 文件，因此这是不同分辨率、不同输出分支的现有工程基线比较。旧推理时间59.607s与峰值21.682GiB来自已有日志，不是本轮复测，旧物理GPU未核实；旧代码Commit无法从已有归档确认，保留unknown而不伪造。OFF/ON 才是严格同输入的 Backend 消融。自投影近零的深度分支是代数一致性，不是实测相机/三维精度。

五个实际人工2D GT 位于 /data/fj/F2DMAS/03-GT/KongQueZhuYu，对应原始0000/0025/0050/0075/0100。人工 Mask 含花盆与独立蓝色物体，不是叶片实例标注。统一五视图把真实照片/轮廓与各模型整体点云投影并排显示；原图坐标映射相同，没有伪造3D GT或执行扫描配准。ROI 为直接查看原RGB后确定的叶片/叶缘矩形窗口，未运行新分割。

Fast3R、VGGT-Ω 尺度不确定；AMB3R depth_metric 是未经本植物实测验证的学习型米制深度。将该因子传给独立 point head 的 _metric_derived 文件标记 gauge_validated=false。深度反投影的预测公制点图/相机和额外 PLY 单独标明，不能与已验证真实米制几何混淆。补充深度反投影预测公制PLY沿用主 world_points_conf>1，OFF/ON为2,010,474/9,219,511点；原生depth_conf分支的覆盖/一致性单独记录，不用于宣称这些PLY的覆盖率。SfM 全局尺度经跨簇注册，不因局部调用存在 Metric Head 就成为真实米制。

计时保留实际测量范围：Fast3R 单次网络/网络+PnP；Base 官方 forward；SfM pipeline/局部模型/局部原始输出/Bundle 导出。原生 SfM 在共享 HDD 上运行，混合流程在 SSD 暂存并另行持久化；共享 HDD 压力的实际快照保存 shared_config/host_io_pressure.json。不能直接把二者 wall-time 差异归因于算法加速。未记录的原生 Bundle/全前端时间保留空值，没有根据文件时间补造。

## 已完成的导出修复与验收

最终审计发现旧Ω导入缺少416份逐帧PLY及参数方法标签不一致。现已从本Bundle的完整按帧拼接PLY无损拆出两种逐帧输出，首/中/末三帧全部前景XYZ/RGB与原raw逐字节一致；原始预测、相机、指标与图未改。629项失败审计及旧参数留档，修复后v2通过。证明为 VGGT_Omega_Existing/lossless_split_audit.json，不是重跑或借用其他模型。

## 交付索引与验收

- manifest.csv、shared_config/：原始输入和运行依据。
- Fast3R/、AMB3R_Base_OFF/、AMB3R_Base_ON/、AMB3R_SfM/、Fast3R_AMB3R_SfM/：统一 Bundle、完整点图/置信度、RGB/Mask、相机、PLY、逐模型 RUN_REPORT。
- comparison.csv、comparison.md、amb3r_off_on_geometry_controls.csv、amb3r_depth_channel_comparison.csv：定量主表及分支/筛选对照。
- visualizations/：整株正侧顶、叶片/叶缘、统一灰色几何辅助图、Confidence、Trajectory、人工GT五视图、Global/Local诊断；PNG 与 PDF 均保存。
- ../../docs/FRONTEND_BUNDLE_INTERFACE.md：后续3DGS/Mesh的统一CPU读取协议。未宣称通过 GGGS readColmapSceneInfo 验收，未伪造特征/匹配/Tracks。
- 各 bundle_audit.json、shared_config/root_interface_verification.json：文件/数量/维度/来源检查与CPU读取、相机逆和像素内参回映验证，不能视为三维精度验收。

本阶段未启动 EO、GGGS、2DGS/3DGS、Mesh/NKSR、新分割、新的点云平均融合或扫描真实尺度配准，未处理另外18株。

来源：本地原始文件和真实运行记录；[Fast3R 官方源码/论文](https://github.com/facebookresearch/fast3r)、[Fast3R 论文](https://arxiv.org/abs/2501.13928)、[AMB3R 官方源码](https://github.com/HengyiWang/amb3r)、[AMB3R 论文](https://arxiv.org/abs/2511.20343)。论文/官方源码已直接阅读，审计见 docs/*SOURCE_AUDIT.md。博查检索实际返回403额度不足，发现服务不可用；本报告没有把该失败称为搜索成功。
