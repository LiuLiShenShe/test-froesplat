# 05 — FIGURE AND TABLE REDESIGN

**新旧图片、表格和证据映射**

- 核验日期：2026-10-20
- 原稿图表文件位于 `ForeSplat__.../Fig1.png … Fig9.png`（9 张）与 5 张表（`tab:dataset-protocol`、`tab:dataset-coverage`、`tab:vfm-comparison`、`tab:prompt-robustness`、`tab:rap-ablation`、`tab:injection-ablation`、`tab:reconstruction`、`tab:phenotyping`）。
- 本轮不生成图件，只给出证据映射与编辑动作。

---

## 1. New-to-Old Figure Mapping

### Fig. 1 — ForeSplat 总体架构

```text
New figure/table:   Fig. 1 — ForeSplat overall pipeline
Old figure/table:   Fig1.png（现 §3.1 总体框架图）
Scientific purpose:  一图说明真实默认流程：FFT 筛选 → SAM3 P6 逐实例候选 → CDR 重排序
                     → 前景约束 2DGS → (可选) 掩膜支持度加权剪枝 → TSDF 网格
Claim supported:    流程存在且可执行（不支撑任何精度主张）
Evidence source:    01_FINAL_METHOD_DEFINITION.md §2；冻结参数 FINAL_RAP_FSAM3_SPEC_DRAFT.md §2
Experiment version: P6 + per_instance + r1（refinement OFF）
Required action:    重绘。当前图把“多提示词 + 语义门控 + SB-FBPR + GCFC + 双向 M2M3”作为主链，
                     与默认运行不符。图中必须标出：CDR 位置、（可选）细化链灰显、
                     M2M3 灰显并标注“mask-supported weighted pruning (optional)”。
Expected section:   3.1
```

### Fig. 2 — RAP-FSAM3 实例选择与 CDR 修正案例

```text
New figure/table:   Fig. 2 — Candidate re-ranking and the two rescued frames
Old figure/table:   Fig2.png（现 RAP-FSAM3 方法图）+ Fig3.png 的一部分
Scientific purpose:  (a) 说明候选生成→评分解耦 contrast→选择；(b) 展示 2 帧切换案例
Claim supported:    R1/CDR 在 2/32 帧改变目标且 2 帧均改善；0 帧退化；1 帧仍完全失败
Evidence source:    阶段二十D_固定轮廓直接评价/R0_R1_TEST32_RESULTS.csv（含 switch 案例与
                     WanNianQing1_0223 失败帧）；figures/ 下 32 张 comparison figure + 2 张 switch PDF
Experiment version: Phase20D fixed-contour replay（ranking_policy base vs r1，其余冻结）
Required action:    新建。必须包含：候选掩膜叠图、score 分解（q_contrast 贡献）、切换前后对比、
                     以及 **显式标注 0223 失败帧**（不得只展示成功案例）。
                     图注必须写“fixed-contour rasterization evaluation, not dual-signature human GT”。
Expected section:   3.2.2 / 4.1
```

### Fig. 3 — 提示语义脆弱性

```text
New figure/table:   Fig. 3 — Prompt-level semantic bias (P1–P5)
Old figure/table:   Fig4.png（现提示词鲁棒性图）
Scientific purpose:  说明单提示词 SAM3 在盆栽场景的语义脆弱性
Claim supported:    P1–P5 的 F1 跨度极大（0.090–0.983），提示语义而非模型能力是主因
Evidence source:    阶段二十B.../tables/TABLE4_RECALCULATED.csv（potted_clean，Easy20 20 帧）
Required action:    改数不改图。必须把帧数/序列数标为“4 sequences × 5 frames”，
                     并把 P2 标签从“entire potted plant”改为“entire plant excluding pot”，
                     同时注明“该提示词实测盆区召回≈97.5%，命名与行为不一致”。
Expected section:   4.2
```

### Fig. 4 — 前景约束 2DGS 训练机制与 A5 对比

```text
New figure/table:   Fig. 4 — Foreground-constrained 2DGS: loss injection and its ablation
Old figure/table:   Fig5.png（现注入位置机制图）+ Fig7.png（现视觉证据）
Scientific purpose:  展示前景先验进入 2DGS 的四条路径，以及 A0→A5 的视觉/数量变化
Claim supported:    联合前景监督把外部非黑比例 0.991→0.029、泄漏 1.220→0.019、高斯 751k→593k
Evidence source:    kongquezhu_A0_A6_ablation_summary.csv；06-实验输出/KongQueZhuYu/{E2_2dgs_baseline,
                     A5_fg_rgb_alpha_bg_loss}；masks + renders
Experiment version: S10 A0 vs A5，KongQueZhuYu，30000 iters，27 views，单次运行
Required action:    重绘。图中必须标注“single scene, single run”；删除图中“m2m3 densify/keep”
                     类要素；补 A2/A3/A4 三行“未分离”的视觉对照（证明只有联合有效）。
Expected section:   3.3 / 4.4
```

### Fig. 5 — M2M3 容量—质量权衡

```text
New figure/table:   Fig. 5 — Capacity–quality trade-off (M0/M1/M2 × 3 scenes)
Old figure/table:   无直接对应（现 Fig6.png 画的是未实现的 M2M3 机制）
Scientific purpose:  展示跨场景方向不一致的容量—质量关系（含负向场景）
Claim supported:    加权剪枝真实触发（30/30 事件，额外删除 6.9k–12.3k），但效果跨场景 MIXED；
                     Floor40 从未拦截删除
Evidence source:    阶段二十E3.../M2M3_SCENE_SUMMARY.csv、M2M3_PRUNING_EVENT_LOG.csv、
                     M2M3_CAPACITY_EFFECT_REPORT.md、figures/*_M0_M1_M2_COMPARISON.png
Experiment version: Phase20E3 M0/M1/M2，seed 0，30000 iters，3 scenes
Required action:    新建（散点/配对图 + 3 场景并列）。必须显式包含 WanNianQing1 的**下降**。
                     不得画“容量下降→质量提升”的单向箭头叙事。
Expected section:   4.5
```

### Fig. 6 — 2D 与 3D 结果不一致及失败分析

```text
New figure/table:   Fig. 6 — Segmentation-to-reconstruction inconsistency
Old figure/table:   无对应（现稿完全缺失该发现）
Scientific purpose:  呈现“更好的 2D 掩膜/更准的 2D 选择未必带来更好的 3D”
Claim supported:    (a) 旧 RAP 掩膜在 3/3 样本降低 3D 前景 PSNR 并增加高斯数（S24）；
                     (b) R1 vs R0 的三维前景 PSNR 配对差异为 MIXED/INCONCLUSIVE（Phase20E2）
Evidence source:    S24.../experiment4b_per_sample.csv；阶段二十E2.../MULTISEED_SCENE_SUMMARY.csv、
                     PAIRED_SEED_EFFECTS.csv、figures/（按相机顺序冻结的 first/middle/last）
Experiment version: S24（旧 R0 权重掩膜）；Phase20E2（E1 B/C 配对，seed 0/1/2）
Required action:    新建。建议(a)为分组条形图，(b)为 3 种子配对差值的点图（含 SD 带与方向符号）。
                     图注必须写“all proxies are non-independent ground truth”。
Expected section:   4.6
```

### Fig. 7 — Gaussian、Mesh 与植物应用

```text
New figure/table:   Fig. 7 — Gaussian models, meshes, and the measurement chain
Old figure/table:   Fig8.png（现重建与网格）+ Fig9.png（现表型散点）
Scientific purpose:  展示从高斯→网格→测量的链路，并显式标出链路中**缺失的物理尺度与器官身份**
Claim supported:    网格可生成/可重载/结构可审计；当前协议下厘米级表型不可得
Evidence source:    阶段二十E3.../MESH_STRUCTURE_METRICS.csv、figures/*_GEOMETRY.png、
                     *_SURFACE.png、*_REPROJECTION.png；阶段二十F.../PHASE20F_FINAL_REPORT.md、
                     REMOVED_STRUCTURE_AUDIT.csv
Experiment version: Phase20E3 M0/M1/M2 + Phase20F（M0 raw/post）
Required action:    重绘 Fig.8 为 raw/post 网格对照 + 结构统计小表；**Fig.9（表型散点）必须删除**
                     （无来源链路）。可改为一张“测量链路与阻断点”示意图：
                     网格 ✓ → 器官身份 ✗ → 垂直基准 ✗ → 物理尺度 ✗ → 厘米级表型 ✗。
Expected section:   4.7 / 5.4
```

---

## 2. New-to-Old Table Mapping

### Table 1 — 最接近方法的技术比较（新增）

```text
New figure/table:   Table 1 — Comparison with the closest methods
Old figure/table:   无（新增）
Scientific purpose:  直接回应“与最近工作的差异”，防止 novelty 被直接判负
Claim supported:    ForeSplat 的增量在“目标实例选择”“注入 2DGS 优化目标”“负向结果披露”
Evidence source:    02_NOVELTY_POSITIONING.md §1（含 DOI）
Required action:    新建。列：方法 / 年份 / 场景 / 是否为盆栽 / 是否做多候选目标选择 /
                    掩膜是否进入优化目标 / 是否给出容量控制 / 是否有几何 GT。
Expected section:   2.4
```

### Table 2 — 数据集、算法版本与实验协议（重写）

```text
New figure/table:   Table 2 — Datasets, algorithm versions and evaluation protocols
Old figure/table:   tab:dataset-protocol + tab:dataset-coverage
Scientific purpose:  一次说清“哪张表来自哪个版本、哪些序列、哪些帧、哪种 GT”
Claim supported:    证据可追溯性（服务于审稿人对 4.1–4.7 的核查）
Evidence source:    04_CLAIM_TO_EVIDENCE_MATRIX.md 全文
Required action:    两表合一为一张三段表：
                    (i) 资源：20 序列 / 2 采集设置 / 21 植株 / 63 叶；
                    (ii) 分割评测：Easy20（4 序列×5 帧，potted_clean）、TEST32（8 序列×4 帧，固定轮廓）；
                    (iii) 三维评测：KongQueZhuYu 单场景 A0–A6 + 三场景 E1/E2/E3。
                    必须标出每一行的 **算法版本** 与 **GT 版本**。
Expected section:   3.6
```

### Table 3 — 历史 RAP-FSAM3 分割比较（换数）

```text
New figure/table:   Table 3 — VFM foreground-prior comparison (unified potted_clean GT)
Old figure/table:   tab:vfm-comparison
Scientific purpose:  在统一 GT 口径下比较基础 VFM 与本文前景先验
Claim supported:    RAP-FSAM3-v2 相对 SAM3-P2/Grounded-SAM/CLIPSeg 的分割表现
Evidence source:    阶段二十B.../tables/TABLE3_RECALCULATED.csv
Required action:    **换数 + 改口径**。现表 0.9706/0.9641/183.42/0.4643/0.001739 必须替换为
                    0.98373/0.97963/23.73/0.48521/0.001739（potted_clean，Easy20）。
                    标题与表注必须写“4 representative sequences, 20 frames”，
                    不得写“all 20 RGB sequences”。CLIPSeg 0.93029、Grounded-SAM 0.96220、
                    SAM3-P2 0.98283 同步替换。图注须声明该结果属 RAP-FSAM3-v2（score_select，无 R1）。
Expected section:   4.3
```

### Table 4 — 提示词鲁棒性（换标签 + 换数）

```text
New figure/table:   Table 4 — Prompt robustness on the same frames
Old figure/table:   tab:prompt-robustness
Scientific purpose:  说明直接提示的语义脆弱性
Claim supported:    P1–P5 的分割表现跨度
Evidence source:    阶段二十B.../tables/TABLE4_RECALCULATED.csv
Required action:    换数为 P1 0.76188 / P2 0.98283 / P3 0.47172 / P4 0.09024 / P5 0.23287
                    （potted_clean，Easy20）；**P2 文本标签** 改为 `entire plant excluding pot`；
                    补一句“该提示词实测盆区召回≈97.5%，命名与行为不一致”。
Expected section:   4.2
```

### Table 5 — CDR 候选排序消融（新增/替换）

```text
New figure/table:   Table 5 — Contrast-decoupled re-ranking on the TEST32 challenge set
Old figure/table:   tab:rap-ablation（当前为 RAP-FSAM3 模块消融，与 R1 无关）
Scientific purpose:  呈现 R1/CDR 的排序修正效果与代价
Claim supported:    F1 0.88887→0.95045；IoU 0.87222→0.93291；HD95 390.60→227.26；
                    Boundary-F1 0.24737→0.26349；Leakage 0.007471→0.006664；
                    improved 2 / regressed 0 / unchanged 30；严重失败 3→1
Evidence source:    阶段二十D_固定轮廓直接评价/R0_R1_TEST32_SUMMARY.csv、
                     R0_R1_TEST32_PER_SEQUENCE.csv、R0_R1_TEST32_RESULTS.csv、
                     R0_R1_PAIRED_DIFFERENCES.csv
Required action:    **新建**（现表 4 无对应内容）。建议同时给出 8 序列逐序列行与
                     GEOMETRY_AMBIGUITY13 / OTHER19 两个子集行（后者差异恒为 0，须显式展示）。
                     表注必须写“fixed-contour rasterization evaluation of manual contours;
                     not dual-signature human ground truth; DEV/TEST share no sequence identity
                     but neighbouring-frame correlation is not excluded”。
Expected section:   4.1
```

### Table 6 — 前景联合监督消融（删两行 + 加限定）

```text
New figure/table:   Table 6 — Foreground-constraint ablation (KongQueZhuYu)
Old figure/table:   tab:injection-ablation
Scientific purpose:  证明“只有把前景先验注入优化目标才能分离前景”
Claim supported:    A0→A5：PSNR_fg 24.209→25.106；outside 0.9908→0.0294；leak 1.2201→0.0190；
                    count 751,213→592,900；A2/A3/A4 均未分离
Evidence source:    kongquezhu_A0_A6_ablation_summary.csv（权威）；experiment4_b0_b5_summary.csv
                     （同一批运行的 B 系列重标）
Required action:    **删除两行**：
                      “Complete ForeSplat objective w/o M2M3-GO”（25.0478 / 596,840 —— 全仓无出处）
                      “Post hoc mask pruning” 的 Gaussian count（468,920 —— 全仓无出处；实际 E7 前景导出为 340,413）
                    **重排数值归属**（当前跨行混用）：
                      “Foreground-object loss injection” 行：PSNR_fg 应为 **25.1055**（A5），
                        count 592,900（A5），FG init = No；
                      “Complete ForeSplat objective” 行：应改用 A6 的 25.0072 / 591,623，
                        或直接改用 A6+M4(F1) 的 24.9723 / 585,594，勿与 A5 数值混用。
                    表注必须写“single scene, single run, no seed repetition; masks from the legacy
                    SAM mask set, not from the P6+R1 pipeline; PSNR_fg 是唯一可跨行比较的列”。
Expected section:   4.4
```

### Table 7 — M2M3 容量与质量对照（新增）

```text
New figure/table:   Table 7 — Mask-supported pruning and the global capacity floor
Old figure/table:   无（现稿无对应表）
Scientific purpose:  正负混合呈现容量控制的真实效果与失效边界
Claim supported:    3 场景 × M0/M1/M2 的 count 与 foreground PSNR；额外删除数；floor_blocked=0
Evidence source:    阶段二十E3.../M2M3_SCENE_SUMMARY.csv、M2M3_PRUNING_EVENT_LOG.csv、
                     M2M3_CAPACITY_EFFECT_REPORT.md
Required action:    **新建**。必须包含 WanNianQing1 的负向行；必须含
                    “extra_pruning_removed_total” 与 “floor_blocked_total = 0” 两列；
                    表注写“three scenes, seed 0; M0 is a frozen historical run, not a bitwise rerun;
                    pre-pruning trajectories already diverged at iteration 601, so count differences
                    are not a causal estimate of the extra pruning”。
Expected section:   4.5
```

### Table 8 — 二维—三维一致性与稳健性（新增，替换现 Table 7）

```text
New figure/table:   Table 8 — Segmentation-to-reconstruction consistency and seed robustness
Old figure/table:   tab:reconstruction（COLMAP/3DGS/2DGS/ForeSplat 四行“20 序列平均”）
Scientific purpose:  (a) 呈现 2D→3D 不一致；(b) 呈现 R0/R1 三种子稳健性
Claim supported:    S24 三种掩膜来源的 3D 结果；Phase20E2 的 3 种子配对差值与 SD
Evidence source:    S24.../experiment4b_summary.csv；阶段二十E2.../MULTISEED_SCENE_SUMMARY.csv、
                     R1_3D_ROBUSTNESS_DECISION.md
Required action:    **整表替换**。现表四行全部为 `PAPER_REPORTED_UNVERIFIED`
                    （tables/TABLE7_CLAIM_STATUS.csv：4/4 行 UNVERIFIED，理由
                    “No source set proves same 20 sequence/sample split, view split, metric ROI,
                    and runtime protocol”）。新表只放有来源的 3 场景 × 3 种子消融。
                    3DGS/2DGS/COLMAP 四方法比较若要保留，必须先补齐同源协议；
                    否则移入 Discussion 并标注 “application-level, protocol not closed”。
Expected section:   4.6
```

### Table 9 — 网格结构与农业应用（新增，替换现 Table 8）

```text
New figure/table:   Table 9 — Mesh structure records and the status of trait measurement
Old figure/table:   tab:phenotyping（MAE/RMSE/MAPE/Bias/R²）
Scientific purpose:  用可审计的结构记录替代不可追溯的 R² 表，并诚实交代测量阻断
Claim supported:    3 场景 raw/post 网格结构统计；代理深度一致性；表型测量的 BLOCKED 状态
Evidence source:    阶段二十E3.../MESH_STRUCTURE_METRICS.csv、metrics/*_MESH_REPROJECTION.csv；
                     阶段二十F.../PHASE20F_FINAL_REPORT.md、PHYSICAL_SCALE_PROVENANCE.csv
Required action:    **整表替换**。现表 8 的 R² 全部来自工作簿 `植株数据.xlsx` 的人工/虚拟记录，
                    无 mesh、无模型、无 mask 链路（`virtual_values_are_new_mesh_results=false`），
                    且 Phase20F 实测“0 个有效厘米值”。新表改为：
                    (i) 网格结构（顶点/面/连通分量/边界边/退化面比例，含 raw 与 post）；
                    (ii) 代理一致性（深度 MAE、ROI 覆盖、剪影 IoU，标注 non-GT）；
                    (iii) 表型状态行：株高 GEOMETRIC_ONLY、冠幅 GEOMETRIC_ONLY、
                         叶长 BLOCKED、叶宽 BLOCKED，缺失项写明原因。
                    如需保留旧 R²，必须改标为 “legacy recorded statistics (workbook)”, 移至补充材料。
Expected section:   4.7 / 5.4
```

### Appendix Table A1 — 后训练剪枝原型（新增）

```text
New figure/table:   Table A1 — Post-training pruning prototype
Old figure/table:   无
Scientific purpose:  记录已实现的压缩侧工作
Claim supported:    1 模型 × {random, opacity} × {0,0.2,0.4,0.6} 的 count/size/PSNR/FPS/VRAM
Evidence source:    阶段二十G0.../COMPRESSION_METRICS.csv、PHASE20G0_FINAL_REPORT.md
Required action:    新建于补充材料。必须写明“pruning only; no quantization, entropy coding or SH
                    compression; PUP and LightGaussian adaptations BLOCKED; single scene, single seed”。
Expected section:   Supplementary A
```

---

## 3. 旧图逐张处置

| 原图 | 原位置 | 处置 | 备注 |
|---|---|---|---|
| Fig1.png | §3.1 总体框架 | **重绘** | 流程与默认实现不符 |
| Fig2.png | §3.3 RAP-FSAM3 | **重绘** | 语义门控 softmax / SB-FBPR / GCFC 主链须改为可选 |
| Fig3.png | §5.1 VFM 比较 | **保留**（换数/换注） | 视觉对比可用，数值与图注须改 |
| Fig4.png | §5.2 提示词 | **保留**（改标签/帧数） | — |
| Fig5.png | §5.4 注入机制 | **重绘** | 补 A2/A3/A4 未分离对照；删 M2M3 densify 要素 |
| Fig6.png | §5.4 M2M3 机制 | **删除** | 画的是未实现的机制 |
| Fig7.png | §5.4 视觉证据 | **保留**（加“单场景”标注） | — |
| Fig8.png | §5.5 重建与网格 | **重绘** | raw/post 对照 + 结构统计 |
| Fig9.png | §5.5 表型散点 | **删除** | 无来源链路 |

---

## 4. 一致性检查清单（提交前逐项确认）

- [ ] 每张表的表注都写明 **算法版本 / 序列数 / 帧数 / GT 版本 / 聚合口径**。
- [ ] 正文出现的每个数字都能在 `MANUSCRIPT_EVIDENCE_REGISTRY.csv` 中找到行。
- [ ] 没有把不同版本（Easy20 分割 / TEST32 选择 / S10 二维 / E1–E3 三维）的数字拼进同一句端到端陈述。
- [ ] 所有 `*_fg` 指标与全图指标不混列（A0 全图 24.16 vs A5 全图 6.26 的陷阱）。
- [ ] 负向结果（WanNianQing1 FG PSNR 下降、0223 失败帧、R1 三维 MIXED、Floor40 NO_EFFECT、表型 BLOCKED）全部出现在正文而非仅在补充材料。
- [ ] 图注中不出现 “20 sequences”“20 RGB sequences” 等无法支撑的规模表述。
- [ ] 补充材料包含冻结配置与命令记录（`command.txt` 系列），便于复核。
