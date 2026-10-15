# Phase20E-3 Table 7 evidence boundary

本文件不修改历史 Table 7、四方法平均值或论文 `.tex`。Phase20B 的 `M2M3_IMPLEMENTATION_TRUTH.md`、`M2M3_FORMULA_CODE_MATRIX.csv`、`SCALE_RECOVERY_AUDIT.md` 和 `TABLE7_UNVERIFIED_CLAIMS.md` 继续作为历史审计依据。

## 可新增、不能冒充历史全量替换的证据

本阶段范围为 WanNianQing1、CaoMei2、HongZhang 三场景，每种机制仅一个固定 seed=0；M0 继承 Phase20E-1 模型，M1/M2 为本阶段正式训练。它是三场景机制 pilot，验证视角是共同相机的重复观测，不是独立训练重复。不能将三场景或全部视角拼成 20 序列结果。

- `M2M3_TRAINING_RUNS.csv`：可替换这三场景对应的模型数量、迭代数、Gaussian 数、训练参数与本次资源记录；M0 历史耗时与新增包装耗时的观察开销不同，不能用单次差值宣称稳定加速。
- `M2M3_HELDOUT_METRICS.csv`：可新增共同 canonical raw RGB 参考的真实浮点 PSNR/SSIM/LPIPS；不能把黑背景训练的全图 PSNR 直接解释为植物几何精度。
- `M2M3_FOREGROUND_PROXY_METRICS.csv`：可新增同一冻结 R0 ROI 上的前景 MSE/PSNR 和共同黑背景合成 SSIM/LPIPS。R0 是预测代理，合成卷积指标受边界窗口影响，不能改名为独立前景 GT。
- `M2M3_BACKGROUND_GEOMETRY_PROXY_METRICS.csv`：可新增背景 RGB 能量、alpha 支持与缺失、Gaussian 分布等代理；外部 alpha 连通区域不是经人工确认的漂浮高斯。
- `MESH_EXTRACTION_RUNS.csv`、`MESH_STRUCTURE_METRICS.csv`、`M2M3_MESH_COMPARISON.csv`：可新增本次真实 TSDF 成功率、顶点/面片/组件/边界/非流形边、任意尺度面积、深度重投影一致性与提取耗时；不能将这些结构统计写成真实几何 accuracy。
- `M2M3_PRUNING_EVENT_LOG.csv`：可新增加权额外剪枝、预算限制和 Floor40 阻止数量的直接证据。设置了 Floor40 或进入其时间窗口均不等于产生了容量保护收益。

具体数值、失败和方向必须读取本阶段最终 CSV 与 `PHASE20E3_FINAL_REPORT.md`，不能从旧 pilot 拼接。

## 本阶段已经真实更新的三场景 pilot 结果

- 六个新增 M1/M2 均完成 30,000 iterations、29,999 native optimizer updates，并通过 checkpoint/PLY 独立重载；`M2M3_TRAINING_RUNS.csv`、训练日志和外部模型索引可追溯命令、GPU UUID、环境哈希与模型 SHA256。
- 共同 GPU1 原生 RGB/alpha/depth/normal renderer 的描述性均值（不含加载、指标和序列化）已写入 `M2M3_RENDER_TIMING_SUMMARY.csv`：WanNianQing1 M0/M1/M2 为 5.931/6.242/6.180 ms，CaoMei2 为 7.026/5.660/6.083 ms，HongZhang 为 7.753/7.754/7.276 ms。固定顺序、单次会话和共享主机负载使其只能作为 pilot，不能写成稳健加速结论。
- 三个 M0 与六个 M1/M2 均有 raw/post Mesh 结构记录。主 post 顶点范围为 438,283–1,672,621，面片范围为 815,120–3,293,400；全部有限且可重载，但 raw 分离组件、开边和少量退化面仍保留。面积只使用 COLMAP 任意单位平方，不能转为 cm²。
- `M2M3_PRUNING_EVENT_LOG.csv` 证明 30/30 个额外剪枝事件真实执行；Floor40 的 15 个 M2 事件均被评估但 `floor_blocked_count=0`。因此本阶段可以更新“机制实际触发”和“Floor40 未介入”的证据，不能更新为容量保护正收益。
- 全图 raw RGB、固定 R0 ROI proxy、背景/深度/法向代理和 Mesh/Gaussian 深度一致性均已由真实浮点结果复算；它们只能作为共同参考下的三场景机制 pilot，不能冒充独立前景 GT 或真实几何精度。

## 仍不能保留为已验证结果的历史描述

原 Table 7 的四方法、20 序列平均值及未追溯到正式命令、模型和逐视角共同参考的数值，仍保持 `PAPER_REPORTED_UNVERIFIED`。本阶段没有运行全量 SfM/MVS/其他 Gaussian 方法的正式配对 baseline，因此不能更新为四方法公平排名。旧 pilot 可以证明历史代码与参数曾被使用，不能作为本次 M1/M2 的正式模型。

没有独立三维 GT，不能报告真值 Chamfer accuracy、F-score、Mesh 精确率/召回率。Mesh-to-Gaussian 深度差只反映二者的一致性；同一模型生成的 Mesh 不能成为另一模型的独立 GT。连通组件更少、面片更多或表面积更大也不能独立证明更准确。

## 四项表型仍缺少的数据

株高、冠幅、叶长、叶宽均缺少：每场景可追溯的物理尺度基准及其测量误差、基准与冻结 COLMAP 三维坐标的对应关系、明确的测量定义和器官身份、同一次采集对象的人工实测值与单位、测量人员/日期/重复测量记录。图像中出现标块不能替代其真实尺寸、身份和坐标记录。任意 COLMAP 长度和面积保持任意单位及其平方，不转换为 cm、mm 或 cm²。

Phase20F 的输入清单见 `PHASE20F_HANDOFF.md`。本阶段完成后停止，不自动开展尺度恢复、表型测量或论文改写。
