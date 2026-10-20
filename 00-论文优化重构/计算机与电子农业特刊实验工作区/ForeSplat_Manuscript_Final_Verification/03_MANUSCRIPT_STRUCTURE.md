# 03 — MANUSCRIPT STRUCTURE

**最终论文目录及原稿章节迁移方案**

参考对象：*AgriGaussian*（AI in Agriculture 2026）的农业工程论文组织方式（问题组织 → 方法展开 → 实验呈现 → 农业应用论证）。**学习其组织逻辑，不复制其表述。**

---

## 1. 最终推荐目录

```latex
1  Introduction
   1.1  (不设小节，3 段式：需求 → 目标选择失败 + 前景冗余 → ForeSplat 与三项贡献)

2  Related Work
   2.1  Vision foundation models for plant segmentation
   2.2  Object-centric Gaussian reconstruction
   2.3  Gaussian pruning and compact representations
   2.4  Positioning of ForeSplat          % 新增：直接回答与最近工作的差异

3  Materials and Methods
   3.1  Problem formulation and ForeSplat overview
   3.2  RAP-FSAM3: target-aware foreground prior generation
        3.2.1  Frame-quality filtering and instance candidate generation
        3.2.2  Contrast-Decoupled Re-ranking (CDR) candidate selection
        3.2.3  Optional foreground refinement (disabled by default)
   3.3  Foreground-constrained 2D Gaussian Splatting
        3.3.1  Foreground-aware initialisation from COLMAP tracks
        3.3.2  Mask-conditioned joint supervision
        3.3.3  View-quality-aware soft loss weighting
   3.4  Mask-supported Gaussian pruning with optional capacity floor
   3.5  Mesh extraction, scale recovery and trait measurement
   3.6  Experimental setup: data, versions, baselines and metrics
        3.6.1  Datasets and acquisition settings
        3.6.2  Algorithm version and protocol definitions
        3.6.3  Baselines and evaluation metrics

4  Results
   4.1  Target-instance selection under geometric ambiguity
   4.2  Prompt-level semantic fragility of direct VFM prompting
   4.3  Historical RAP-FSAM3 segmentation baseline (unified GT)
   4.4  Foreground-conditioned 2DGS: the necessity of objective injection
   4.5  Gaussian capacity and quality trade-offs
   4.6  Segmentation-to-reconstruction consistency (negative result)
   4.7  Mesh reconstruction and agricultural applicability

5  Discussion
   5.1  Why target-instance selection, not prompt engineering, is the bottleneck
   5.2  Foreground priors must define the 3D optimisation target
   5.3  Model capacity versus reconstruction quality
   5.4  Limitations: scale, geometric ground truth and measurement validity
   5.5  Practical implications for low-cost phenotyping

6  Conclusions

Appendices (or supplementary)
   A  Post-training Gaussian pruning prototype
   B  Rasterisation sensitivity and secondary metrics
   C  Reproducibility: frozen configurations and command records
```

---

## 2. 与原稿的逐节对照

| 新节 | 原稿节 | 动作 | 理由 |
|---|---|---|---|
| **1 Intro** | `\section{Introduction}` | **重写** | 需把“目标选择失败”立为问题起点；删除未证实的端到端提升 |
| **2 Related Work** | 原稿无独立 Related Work（融在 Intro） | **新增** | CompAg 期望独立 RW；必须含 2.4 直接定位 |
| **3.1** | `Study Design and Overall Workflow` | **重写** | 按真实执行顺序（见 `01` §2） |
| **3.2.1** | `FFT-Based Frame-Quality Filtering` + `Multi-Prompt VFM Candidate Generation` | **合并重写** | 默认只有 P6 单提示词；P1–P5 的候选集仅历史使用；FFT 公式须改为实现口径 |
| **3.2.2** | `Semantic-Gated Multi-Prompt Selection` | **删除改写为 CDR** | softmax 未实现、门控默认关；改为对比解耦重排序 |
| **3.2.3** | `Structure-Balanced …` + `Geometry-Consistency …` | **降级为可选、默认关闭** | 与 R1 零联合证据 |
| **3.3.1** | `Camera Pose Estimation and Foreground Trajectory Initialisation` | **保留 + 小修** | 机制真实；需披露 mask 身份未闭合 |
| **3.3.2** | `Foreground RGB Supervision and Opacity Constraints` | **重写公式** | λ_mask 需改为 0.5·L1+0.5·Dice 的整体系数；补 λ_normal |
| **3.3.3** | `View-Quality-Aware Soft Loss Weighting` | **重写** | σ 公式未实现；改为外部预计算权重 clamp[0.6,1.0] |
| **3.4** | `Mask-Guided Multi-Cue Gaussian Pruning` + `M2M3-GO` 两节 | **合并为一节并大幅重写** | 公式几乎全部未实现 |
| **3.5** | `Mesh Extraction, Scale Recovery and Phenotypic Measurement` | **保留 + 删公式** | visibility/normal/mask 加权 TSDF 未在正式结果使用 |
| **3.6** | 原稿散落在 `Dataset…` + `Evaluation Metrics` + 段落内 | **集中为独立小节** | 原稿协议散落 4 处，易与结果混淆 |
| **4.1** | `RAP-FSAM3 Improves VFM-Based Plant Foreground Priors` | **重写** | 改 20 序列表述；用 potted_clean 重算值 |
| **4.2** | `Prompt Sensitivity Reveals Semantic Bias…` | **保留 + 修正标签** | P2 标签错；帧数改为 4 序列 20 帧 |
| **4.3** | `Ablation of RAP-FSAM3 Adaptation Modules` | **保留 + 替换数字** | 用 TABLE3/4/5 重算值；删无出处数字 |
| **4.4** | `VFM-Prior Injection into 2DGS Is Necessary…` | **保留 + 删两行 + 限定单场景** | 需删 25.0478 / 468920 两行（无来源） |
| **4.5** | （原稿无对应） | **新增** | 来自 Phase20E3，正负混合呈现 |
| **4.6** | （原稿无对应） | **新增** | S24 + Phase20E2；负向结果，最有价值的发现之一 |
| **4.7** | `Workflow-Level Reconstruction, Mesh Generation and Phenotypic Validation` | **重写、大幅删减** | Table 7 四行 UNVERIFIED；Table 8 legacy |
| **5.1** | `Vision Foundation Models Require Agricultural Prompt Adaptation` | **重写** | 改为“目标选择而非提示词工程是瓶颈” |
| **5.2** | `Foreground Priors Should Define the 3D Optimisation Target` | **保留** | 核心论点，证据最强 |
| **5.3** | `From Foreground-Object Supervision to Primitive-Level Capacity Allocation` | **重写** | 诚实写 MIXED + NO_EFFECT |
| **5.4** | `Phenotyping-Oriented Reconstruction…` + 段落内误差来源 | **合并 + 增强** | 加入 Phase20F 的尺度/身份阻断 |
| **5.5** | （原稿无） | **新增** | CompAg 关心的实践含义 |
| **6 Conclusions** | `\section{Conclusion}` | **重写** | 删除无证据的端到端性能声明 |
| **App. A** | （原稿无） | **新增** | Phase20G0 压缩原型 |
| **App. B/C** | （原稿无） | **新增** | 栅格化敏感性 + 冻结配置可复现性 |

---

## 3. 是否符合 CompAg 研究论文逻辑

| 判据 | 结论 |
|---|---|
| 问题导向 | ✅ 新结构以“目标实例选择失败”开篇，比原稿“VFM 提供新机会”更贴近 CompAg 的工程问题取向 |
| 农业应用闭环 | ✅ 4.7 + 5.4 明确“为什么当前协议下不能宣称厘米精度”，比原稿的 R² 表格更可信 |
| 方法层级统一 | ✅ 从“选对目标 → 定义 3D 目标 → 分配容量 → 应用”是一条递进链，且每级都有对应证据 |
| 负向结果处置 | ✅ 4.6 + 5.4 显式承载，符合农业工程期刊的实证偏好 |
| Related Work 位置 | ⚠️ 原稿无独立 RW；必须新增，否则“最近工作重合度高”会被审稿人直接判为 novelty 不足 |
| Experimental Setup 位置 | ⚠️ 建议放在 Methods 末（3.6）而非独立节，符合 CompAg 习惯 |

---

## 4. 反复出现的结构问题（原稿）

1. **协议散落 4 处**：数据集在 3.2、指标在 3.8、运行协议在 3.8 末段、20 序列声明在 4.5 段落内。审稿人无法核对“同一批样本”。→ 集中到 3.6。
2. **两张架构图/表格职责重叠**：Fig.1（总览）与 Fig.5（注入）都在讲同一件事；Table 5（injection ablation）与 §5.2 Discussion 重复叙述。→ Fig.5 改作 3.3 的机制图。
3. **论证断裂**：Intro 说“M2M3-GO 是 primitive-level refinement layer”，但 §3.7 又给出完整四公式与 sigmoid 效用，读者会期待一个对应的独立消融。→ 若不能给消融，就删公式。
4. **贡献列表与证据强度不匹配**：三项 contribution 中，第二项实际证据是单场景 A5，第三项无正向证据。→ 见 `02` §3 的权重重分配。

---

## 5. 必须重写/保留/删除的小节速查

**必须整段重写**：Abstract；Introduction；3.2.1；3.2.2；3.2.3（降级）；3.3.2；3.3.3；3.4（两节合一）；4.1；4.3；4.4；4.7；5.1；5.3；6 Conclusions。

**可保留并局部修改**：3.1；3.3.1；3.5；4.2；4.5（新增但主体数据来自 Phase20E3 表）；5.2；5.4。

**删除或移至补充材料**：3.2.2 的 softmax/语义门控公式；3.4 的四条 M2M3 公式；3.5 的 TSDF 加权公式；Table 5 中两行无来源数据；Table 7 的四行“20 序列平均”；Table 8 的 R² 结果表。
