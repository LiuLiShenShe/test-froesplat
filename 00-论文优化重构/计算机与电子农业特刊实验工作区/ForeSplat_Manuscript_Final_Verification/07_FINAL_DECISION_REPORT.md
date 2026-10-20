# 07 — FINAL DECISION REPORT

**最终结论、核心矛盾以及建议采用的论文主线**

- 核验日期：2026-10-20；仓库基线 `e31176ee`（R1 引入 commit `d704faa8`）
- 交付目录：`ForeSplat_Manuscript_Final_Verification/`（本目录）
- 本轮未修改 `ForeSplat.tex`、未修改生产代码、未重跑实验、未提交 Git。

---

## 1. 总体判断

论文的**故事线成立且比原稿更强**，但必须从“三模块算法创新”改为“**一个被忽视的任务失配 + 一个最小可复现的排序修正 + 一个受控的优化目标结论 + 一组诚实的负向结果**”。

原稿最大的风险不是 novelty 不够，而是 **Evidence 与 Claim 大面积错配**：8 类主张中有 2 类为 `CONTRADICTED`，5 类为 `PARTIALLY_VERIFIED`。若按原稿投稿，审稿人只需问三个问题即可致命：
1. “F1 0.9706 是在 20 个序列上算的吗？”——不是，是 4 个序列 20 帧，且用 legacy GT。
2. “R² 0.988 是从哪个模型量出来的？”——无 mesh/模型/mask 链路，Phase20F 实测 0 个有效厘米值。
3. “M2M3-GO 的双向公式在哪段代码？”——不存在。

---

## 2. 五个核心矛盾（按严重度）

### 矛盾 1（最严重）：论文规模表述与数据规模不符

| 原稿 | 实际 |
|---|---|
| "evaluation frames sampled from **all 20 RGB sequences**" | **4 个代表序列 × 5 帧 = 20 帧** |
| "Across **20 RGB multiview sequences** … PSNR = 31.09 dB" | 4 行结果全部 `UNVERIFIED`，无同源协议 |
| Table 8 的 R²（0.988/0.988/0.974/0.900） | 来自工作簿，**0 个有效厘米值** |

### 矛盾 2：论文算法与生产算法不一致

| 原稿默认 | 生产默认 |
|---|---|
| 多提示词 P1–P5 + 语义门控 softmax + SB-FBPR + GCFC | **P6 单提示词 + 逐实例 + R1，refinement 全关** |
| R1 是独立排序机制 | R1 只是 `ranking_policy.py` 里的一个分支，且只在单提示词路径生效 |
| M2M3-GO 四公式 + sigmoid 效用 + 局部 Floor40 | 只有中心投影支持度加权的额外剪枝 + 全局容量下限；其余全部未实现 |

### 矛盾 3：证据版本混用

- 分割主表 = 旧版 RAP-FSAM3-v2（`score_select`，无 R1）。
- 2DGS A5 消融 = 旧 `02-sam_masks`（SAM v1，无盆标注），**不是** P6+R1。
- M2M3/Mesh = E3 的 M0/M1/M2，前景损失**全部关闭**。
- 表型 = 2026-02 的工作簿，早于 R1 提交（2026-09-10）六个月以上。
- **没有任何一条链路是“R1 + A5 + M2M3”的端到端结果。**

### 矛盾 4：负向结果被隐藏，而它才是最有价值的部分

仓库自己已给出四条强负向结论，原稿一条都没写：
- S24：更好的旧 RAP 掩膜在 3/3 样本上**降低** 3D 前景 PSNR（24.66→23.22）并**增加**高斯数（402k→468k）。
- Phase20E2：R1 的三维前景 PSNR 配对差异 **MIXED/INCONCLUSIVE**，最终角色 `ROBUST_SELECTION_ONLY`。
- Phase20E3：Floor40 在 15/15 事件中 **从未拦截任何删除**（NO_EFFECT）。
- Phase20F：物理尺度与器官身份双缺失，**0 个**有效厘米值。

### 矛盾 5：贡献等级与证据强度倒挂

原稿把 M2M3-GO 作为第二项贡献的核心组成，但它既无正向证据也无几何 GT；而真正有说服力的“2D→3D 不一致”发现完全没有位置。

---

## 3. 建议采用的论文主线（最终推荐）

```text
植物三维重建需要目标可靠选择
    ↓  [RAP-FSAM3：把前景先验生成重新定义为目标实例选择问题]
RAP-FSAM3 修正错误实例选择（CDR：把对比项从排序主评分解耦）
    ↓  [Foreground-constrained 2DGS：把前景先验变成优化目标而非预处理]
前景先验约束 2DGS（联合前景监督：前景 RGB + alpha + 背景透明度）
    ↓  [可选：掩膜支持度加权剪枝 + 可选全局容量下限]
通过容量管理降低无关表示（MIXED，如实报告）
    ↓  [负向结果：2D 增益不自动转为 3D 增益]
形成适用于农业应用的紧凑植物对象模型（受尺度与身份缺失限制）
```

**这条主线可用一句话向 CompAg 审稿人解释清楚，且每一步都有可追溯证据。**

---

## 4. 关键决定

| 问题 | 决定 | 理由 |
|---|---|---|
| RAP-FSAM3 是否第一核心创新？ | **是，但降级为“任务级创新”** | 问题确实未被农业文献占用；机制只是最小启发式。应写“提出该问题 + 一个最小可复现的修正”，不写“新算法” |
| R1 如何命名与整合？ | **CDR — Contrast-Decoupled Re-ranking，作为 RAP-FSAM3 内部机制（3.2.2）** | TCR 暗示目标一致性学习，代码中不存在；CDR 精确描述“解耦对比项” |
| A5 是否足以作为第二技术贡献的主要证据？ | **不足，但可作为“实证结论”的主证据** | 单场景、单次运行、无种子；真正的单因子步是 A4→A5。写成“受控消融证明只有把先验注入优化目标才能分离前景”，而非“A5 模块” |
| M2M3 应保持什么地位？ | **可选紧凑化模块（3.4），不是独立贡献** | 加权剪枝真实触发但 MIXED；Floor40 NO_EFFECT；无几何 GT |
| 最有说服力的论文故事是什么？ | **“目标选择 → 优化目标 → 容量分配 → 诚实的能力边界”** | 四段递进，每段有证据；负向结果让论文更可信而非更弱 |

---

## 5. 三个必须现在接受的取舍

1. **放弃端到端性能叙事。** 31.09 dB / R² 0.988 全部不可追溯。取而代之的是可核验的分块证据。
2. **放弃 M2M3 的方法学贡献。** 删公式、改实现、并入 3.4。这不会削弱论文，反而消除最大的“公式-代码不一致”风险。
3. **接受“厘米级表型尚不可得”作为结论之一。** Phase20F 已经证明这是数据缺口而非方法缺陷；如实写出反而是 CompAg 欣赏的工程诚实。

---

## 6. 若后续可以补实验，优先补三件（按性价比）

1. **V1 正式运行**（P6 + r1 + refinement OFF）——只需 GPU 候选复用，可关闭 `P0-4`（R1 从未正式运行）这一最易被攻击的点。
2. **A5 的第二场景 + 第二种子**——把 4.4 从“单场景单次”升级为“两场景×两种子”。
3. **补盆径实测值**——关闭 `P0-1`，使尺度恢复与表型从 BLOCKED 变为可评估。

三者都不需要重写论文框架，只提升证据强度。

---

## 7. 交付物清单

| 文件 | 内容 |
|---|---|
| `01_FINAL_METHOD_DEFINITION.md` | 最终算法结构、模块命名、默认配置、真实公式 |
| `02_NOVELTY_POSITIONING.md` | 贡献重定位、最近工作对比（含 DOI）、英文 novelty statement 与三项贡献 |
| `03_MANUSCRIPT_STRUCTURE.md` | 最终目录与原稿章节迁移方案 |
| `04_CLAIM_TO_EVIDENCE_MATRIX.md` | C1–C8 论断与证据对应，含可写入/禁止写入边界 |
| `05_FIGURE_TABLE_REDESIGN.md` | 9 图 × 9 表的 Old-to-New 映射与编辑动作 |
| `06_LATEX_REVISION_PLAN.md` | 可直接执行的 `.tex` 编辑指令（按 P0-1…P0-7 优先级） |
| `07_FINAL_DECISION_REPORT.md` | 本文件 |
| `MANUSCRIPT_EVIDENCE_REGISTRY.csv` | 61 条结构化证据记录 |

---

## 8. 最终判断

```text
Method structure:      REVISE   → 按 01_FINAL_METHOD_DEFINITION.md 重写 3.2/3.3/3.4；
                                   R1 改称 CDR 并内嵌为 3.2.2；
                                   M2M3 改为 3.4 “mask-supported weighted pruning + optional global floor”
R1 integration:        CONFIRMED → R1 归属 RAP-FSAM3 内部，不作为平行算法；
                                   但必须披露“仅在 single+per_instance 路径生效”与“从未正式运行”
Novelty positioning:   REVISE   → 按 02_NOVELTY_POSITIONING.md §3 权重重分配（30/30/8/22/10）；
                                   Related Work 新增 2.4 + Table 1
Evidence mapping:      INCOMPLETE → 8 类主张中 2 类 CONTRADICTED、5 类 PARTIALLY_VERIFIED；
                                   Abstract/Table 7/Table 8 必须重写或删除
Paper structure:       REVISE   → 按 03_MANUSCRIPT_STRUCTURE.md；新增 Related Work、
                                   3.6 实验设置、4.5、4.6 两节

Ready for LaTeX revision: YES   → 可以进入正式重写，但必须遵守 06_LATEX_REVISION_PLAN.md
                                   的 P0-1…P0-7 编辑指令，不得跳过
```
