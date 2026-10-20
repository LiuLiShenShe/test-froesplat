# 02 — NOVELTY POSITIONING

**ForeSplat 三项贡献的重新定位、最近相关工作对比及英文 novelty statement**

- 核验日期：2026-10-20
- 文献检索：经 OpenAlex / arXiv API（`open-websearch fetch-web`）。搜索引擎（bing/duckduckgo/startpage/brave/exa）本次不可用。CompAg 官方 aims-and-scope 未能抓取，相关判断为“基于已发表语料的推断”。
- 代码/实验证据见 `01_FINAL_METHOD_DEFINITION.md` 与 `04_CLAIM_TO_EVIDENCE_MATRIX.md`。

---

## 1. 最接近已有工作（按重合度排序）

| # | 工作 | 出处/年 | 真实方法主张 | 与 ForeSplat 的重合 |
|---|---|---|---|---|
| 1 | **Object-Centric 2D Gaussian Splatting** | ICPRAM 2025 (doi:10.5220/0013305500003905) | 用物体掩膜做目标重建；**遮挡感知剪枝**在不损质量下压缩（可达 96% 更小、71% 更快）。 | **与贡献 2+3 机制重合最高**：掩膜约束 + 容量剪枝。非植物、无 alpha 损失、无掩膜支持度加权。 |
| 2 | **Object-centric 3DGS for Strawberry Plant Reconstruction & Phenotyping** | Smart Agric. Technol. 2026 (doi:10.1016/j.atech.2026.101810; arXiv 2511.02207) | SAM-2 + alpha 背景掩膜 → 干净草莓重建；DBSCAN+PCA 估株高/冠幅。 | **贡献 2 的植物域直接先例**：掩膜约束 GS + 背景去除 + 表型提取，动机表述几乎一致。 |
| 3 | **LCR-GS (contour-aware lightweight 2DGS)** | The Crop Journal 2026 (doi:10.1016/j.cj.2026.07.002)* | **2DGS** + **轮廓加权 Laplacian 正则**抑制叶片边界深度不连续 + 轻量化 + 器官级语义。 | **贡献 2 的 2DGS 核心最接近**（植物+2DGS+边界+效率）。（*缩写按标题匹配，未在该摘要中确认。） |
| 4 | **PlantGaussian** | The Crop Journal 2025 (doi:10.1016/j.cj.2025.01.011) | 号称“3DGS 在植物科学的首批应用”；集成 **SAM+跟踪**；网格分割 → 可测量植株网格。 | “SAM 引导前景分离”与“首次 3DGS 用于植物”**均已被占用**。 |
| 5 | **Plant3R** | Plant Phenomics 2026 (doi:10.1016/j.plaphe.2026.100200) | MASt3R 稀疏点云+位姿 + 3DGS，**盆栽小麦**、手持影像；PSNR>34，R²>0.94。 | 重合“盆栽 + 稀疏点云先验”，但无掩膜/候选选择。 |
| 6 | **AgriGaussian** | AI in Agriculture 2026 (doi:10.1016/j.aiia.2026.01.006) | 深度监督 + **作物外观提取模块（抑制光照/环境干扰、保留表型细节）** + 多尺度分块；PSNR 26.85。 | 重合“背景干扰抑制”动机，机制不同。 |
| 7 | **Cotton3DGaussians** | **CompAg 2025** (doi:10.1016/j.compag.2025.110293) | 标题：“Multiview 3D Gaussian Splatting for boll mapping and plant architecture analysis”。（方法细节未取到，仅标题/出处已核。） | **证明 CompAg 已发表作物 3DGS 流水线**。 |
| 8 | **IPENS** | Plant Phenomics 2025 (doi:10.1016/j.plaphe.2025.100106) | 把 SAM2 的 2D 掩膜经辐射场提升到 3D，提取目标点云；多目标协同。 | 掩膜引导 3D 提取的最近邻；但为 NeRF、无候选重排序、且是“提取全部目标”而非“选一个”。 |
| 9 | **Compact 3DGS** | CVPR 2024 (doi:10.1109/cvpr52733.2024.02052) | **可学习逐高斯掩膜** + 网格神经颜色 + VQ；>25× 压缩。 | “掩膜剪枝到紧凑模型”的最强通用先例。 |
| 10 | **LightGaussian** / **Mini-Splatting** | arXiv 2311.17245 / 2403.14166 | 全局显著性剪枝 + SH 蒸馏/VQ；约束高斯数的增密/简化。 | 重要性剪枝+容量预算已被充分占据；ForeSplat 仅在“**加权信号=掩膜支持度**”与“**显式全局容量下限**”上有差异。 |
| 11 | **Gaussian Grouping** / **SAGA** | ECCV 2024 / AAAI 2025 | SAM 掩膜监督的逐高斯 identity encoding / scale-gated affinity。 | 语义引导的“无剪枝”路线。 |
| 12 | **GFR-SAM；TF-SSD；Re-Prompting SAM3** | 均 2026（arXiv 2607.11732 / 2604.00549 / 2603.23788） | 训练无关的 **SAM3 候选生成 + 候选排序/过滤**：对比式（DINOv3 原型对齐）抑背景、质量掩膜过滤、跨帧目标检索重提示。 | **与 R1（候选重排序）最接近的已发表类比**，且全部 2026 年。 |
| 13 | 本课题组前作 **F2DMAS** | Front. Plant Sci. 2026 (doi:10.3389/fpls.2026.1885579) | FSAM3 分割 + 2DGS + TSDF，**盆栽**。 | 需在 Related Work 明确区分并引用，避免自我重叠。 |

---

## 2. 三项贡献逐项评估

### 贡献 1 —— RAP-FSAM3 目标实例可靠选择

- **问题新颖性（较强）**：农业领域**未检索到**任何专门解决“多实例 SAM/SAM3 候选中选出目标盆栽实例”的工作。这是一个真实、可表述的任务失配。
- **机制新颖性（弱）**：R1 的真实机制是“**把 `q_contrast` 从主评分中扣除**”——一个由 oracle 审计（Phase 17B/C）导出的启发式；无学习、无跨视角、无新特征。GFR-SAM 等 2026 工作已用“对比式候选排序”处理同类问题。
- **结论**：定位为 **“框架中的关键任务策略创新”**，**不是**独立算法创新。措辞应强调“**目标实例选择这一被忽视的问题** + 一个**最小、可解释、可复现的排序修正**”，而不是“新算法”。

### 贡献 2 —— Foreground-constrained 2DGS

- **机制新颖性（弱-中）**：掩膜约束 GS + 前景 RGB + alpha + 背景透明度 + 视图权重，是多个已知机制的**组合**；object-centric 2DGS (2025)、草莓 object-centric 3DGS (2026)、LCR-GS (2026) 已覆盖核心。
- **真实增量**：把 SAM3 前景先验作为 **2DGS 训练时目标函数约束**（而非仅预处理/后剪枝），并用受控消融证明“**后剪枝不足、仅透明度约束不足、必须约束 RGB 监督本身**”。这一 **实证结论** 比机制本身更有价值。
- **结论**：定位为 **“经过受控消融验证的方法组合 + 一个有意义的实证结论”**，**不是**新机制。避免宣称“首次”。

### 贡献 3 —— M2M3-GO 容量控制

- **机制新颖性（很弱）**：真实实现 = 掩膜支持度加权的额外剪枝 + 可选全局容量下限；重要性剪枝（LightGaussian/Compact3DGS/Mini-Splatting）已饱和。
- **实证结论（重要负向）**：Floor40 在 15/15 事件中 **从未拦截任何删除**（NO_EFFECT）；加权剪枝效果 **MIXED**（跨场景方向不一致）；无独立几何 GT。
- **结论**：**必须降级**为“可选紧凑化/前景支持剪枝模块”或实现细节，**不得**作为独立三维创新。

---

## 3. 逐维度打分

| 贡献 | 新颖性 | 技术实质 | 实验支持 | CompAg 相关性 | 建议权重 |
|---|---|---|---|---|---|
| 1 RAP-FSAM3 目标选择 | 中（问题新，机制弱） | 中（启发式，但最小可复现） | 中（TEST32 固定轮廓；无人类双重 GT） | 高 | **30%** |
| 2 前景约束 2DGS | 低-中（组合） | 中（真实损失链） | 中（A5 单场景 + R0/R1 三场景 pilot） | 高 | **30%** |
| 3 M2M3-GO | 低 | 低 | 弱（MIXED，无 GT） | 中 | **8%** |
| 4 网格+表型+负向结果 | 中（真实负向发现） | 中 | 弱-中（无尺度/无 GT） | **很高** | **22%** |
| 5 数据基础/方法学（20 序列、双采集设置） | — | — | 中 | 高 | **10%** |

> 与用户拟定的 40/35/15/10 相比：**RAP-FSAM3 与 2DGS 各降至约 30%**，**M2M3 降至约 8%**，把腾出的权重给 **“2D→3D 不一致 + 负向结果 + 应用验证”**（这是最真实、最 CompAg 的部分）。

---

## 4. 推荐的英文 Novelty Positioning（约 210 词）

> Object-centric Gaussian splatting has shown that segmentation masks can produce compact, object-level 3D representations, and recent plant-specific pipelines already combine SAM-style masking with 3D Gaussian Splatting for crop reconstruction and trait extraction. What these systems largely assume, however, is that a correct foreground mask for the *target* instance is already available. In cluttered potted-plant scenes this assumption fails: a promptable foundation model returns several candidate instances per frame, and the candidate that maximises generic geometric quality is frequently not the target plant, because a high foreground–background contrast can reward a salient flower or a neighbouring object. ForeSplat addresses this specific failure. First, we reframe foreground-prior generation as a *target-instance selection* problem and show that a single change—decoupling the contrast term from the candidate ranking score—recovers the target instance on the difficult frames without any retraining, weight search, or change to the candidate pool. Second, we show, through a controlled ablation, that this corrected prior only becomes useful for 3D reconstruction when it is injected into the 2DGS *optimisation objective*: post-hoc mask pruning and opacity-only regularisation cannot redirect the photometric gradients that allocate Gaussian capacity. Third, we report that improved two-dimensional selection does not automatically transfer to three-dimensional foreground quality, and we quantify the residual gap. The contribution is therefore a task-level reformulation with a minimal, reproducible mechanism, validated by controlled ablations and an explicit negative result, rather than a new Gaussian primitive or a new network.

---

## 5. 英文三项 Contributions（正式表述）

**Contribution 1.**
> We formulate potted-plant foreground-prior generation as a target-instance selection problem and propose RAP-FSAM3, a training-free pipeline that generates per-instance candidates with a single plant prompt and re-ranks them with a contrast-decoupled score (CDR). Removing the foreground–background contrast term from the ranking score—while leaving all weights, the denominator and the candidate pool unchanged—recovers the correct target on the difficult multi-instance frames and improves fixed-contour F1 from 0.889 to 0.950 on an eight-sequence challenge set, without retraining.

**Contribution 2.**
> We show that a VFM-derived foreground prior must be injected into the 2DGS *optimisation objective* rather than used as image preprocessing or post-processing. A controlled ablation on a representative potted plant demonstrates that restricting the photometric loss to foreground pixels, together with alpha and background-opacity supervision, reduces the outside-mask non-black ratio from 0.991 to 0.029, whereas alpha-only, opacity-only and post-hoc mask pruning all fail to separate the foreground; the joint foreground loss is necessary and acts as a coupled set of constraints.

**Contribution 3.**
> We provide an application-level evaluation with explicitly bounded claims: a mask-supported weighted Gaussian-pruning module with an optional global capacity floor, three-scene multi-seed noise-floor analysis of the selection mechanism at the 3D level, and a documented negative result showing that two-dimensional selection gains do not translate into three-dimensional foreground gains on the current pipeline. We also quantify why validated centimetre phenotyping is not yet attainable, because no physical scale reference exists in the current capture protocol.

---

## 6. 不可保留的 novelty 表述（黑名单）

- ❌ “首次将 3DGS/2DGS 用于盆栽/植物” —— PlantGaussian (2025)、Wheat3DGS (2025)、F2DMAS (2026) 已占位。
- ❌ “首次 SAM3 + 2DGS 组合” —— 组合不是贡献。
- ❌ “提出双向 mask-to-model / model-to-mask 优化” —— 代码未实现。
- ❌ “局部容量保护（Floor40）保证结构完整” —— 实为全局、且从未生效。
- ❌ “R1 带来稳定三维增益” —— Phase20E2 判定 MIXED / ROBUST_SELECTION_ONLY。
- ❌ “在 20 序列上达到 PSNR 31.09 / R² 0.988” —— 无来源（见 `04`、`06`）。
