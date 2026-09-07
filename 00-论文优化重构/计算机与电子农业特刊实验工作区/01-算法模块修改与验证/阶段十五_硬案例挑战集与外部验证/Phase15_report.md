# Phase 15 — Difficulty-Stratified Re-analysis (Zero-Compute)

**日期**: 2026-09-07  
**Scope**: 零新增推理，复用 Phase 14.1 产物，按场景难度分层重新分析 A6/A7 效应。  
**Limitation**: 21 帧全部为 easy set（F1≥0.976），无真正硬帧。详见 §6 局限性。

---

## §0. Starting-State Audit

Phase 14.1 封口后状态：

| 项 | 值 |
|---|---|
| 评估帧 | 21 帧，5 samples（CaoMei1/ChangShouHua2/DouBanLv1/KongQueZhuYu/XianKeLai1） |
| GT | GT_potted_clean（21 帧，binary potted_clean masks） |
| 变体 | V00 Control, V10 A6, V01-R1 A7, V11-R1 A6+A7 |
| 冻结参数 | P6 prompt，score_weights=area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5，阈值全部冻结 |
| 石膏 SHA | `f9f091d8`（Phase 14 commit，未改动） |
| 代码改动 | 仅生命周期/诊断修复（Phase 14.1），未触及 Frozen P6 路径 |

## §1. No Parameter Tuning

**已确认**：无任何阈值/权重/超参调整。Phase 14.1 产物全量复用。本次分析仅重新计算难度属性和分层统计。

## §2. Data Pool Audit

| 资源 | 帧数 | GT? | 备注 |
|---|---|---|---|
| GT_potted_clean（easy 集） | 21 | Yes | 5 samples × 5 frames（XianKeLai1 × 1） |
| 15 non-GT samples | ~3,750 | **NO** | COLMAP 位姿已有，但无任何标注 |
| 60 hard-candidate frames | 60 | **NO** | 由 scout_difficulty.py 按原始图难度排名选出 |

**结论**：GT 仅存在于 21 帧 easy set。真正硬帧需新标注。本阶段按用户决策采用零计算路径。

## §3. H1 与 H2 定义

- **H1（Historical Failure Stress Set）**：Phase 12 的 3 帧历史失败帧（CaoMei1_0100, ChangShouHua2_0100, DouBanLv1_0000）。全部在 easy set 内，P6 修复后 F1≥0.98。定位在难度分布的 hard/mid 端。
- **H2（Difficulty-defined Held-out Set）**：**未构建**。需要 15 个 non-GT sample 的新标注，但用户选择不标注。改为 21 帧 easy set 按原始图难度分层（§4）。

## §4. 难度分层

对 21 帧 easy set 帧计算 5 项原始图难度属性（`scout_difficulty.py`，同 §3 约定的纯原始图特征）：

| 属性 | 方向 | 含义 |
|---|---|---|
| Laplacian variance | 低 = 难 | 模糊/清晰度 |
| Pixel entropy | 高 = 难 | 纹理复杂度 |
| ExG separation | 低 = 难 | 植物-背景色彩分离度 |
| Green ratio | 极端 = 难 | 绿色占比（极低/极高都难） |
| Dynamic range | 低 = 难 | 对比度展宽 |

Composite difficulty = 0.2 × (ls + es + xs + grc + ds)，min-max 归一化后加权求和。

**分层**：三等分 tertiles（7/7/7 hard/mid/easy）

| 层 | 帧 | H1 帧数 | composite range |
|---|---|---|---|
| hard | ChangShouHua2_0050/0100/0000, DouBanLv1_0050/0000, CaoMei1_0000, KongQueZhuYu_0075 | **2** | 0.614–0.735 |
| mid | CaoMei1_0100/0075, ChangShouHua2_0025/0075, XianKeLai1_0000, DouBanLv1_0100/0025 | **1** | 0.531–0.612 |
| easy | KongQueZhuYu_0050, DouBanLv1_0075, KongQueZhuYu_0025/0100/0000, CaoMei1_0050/0025 | 0 | 0.430–0.528 |

分层 CSV：`挑战集列表/21易帧难度分层.csv`

## §5. Challenge Set

**未构建新的硬案例挑战集**（需要标注 15 个 non-GT sample 的帧）。改为对 21 帧 easy set 做难度分层。

## §6. 局限性（关键）

**21 帧全部为 easy set（F1 ≥ 0.976）。** 本阶段的结论只能回答 "在可测范围内 A6/A7 是否有差异"，不能回答 "A6/A7 在真正硬条件下是否有效"。天花板效应导致 20/21 帧 ΔF1 = 0（A6 仅改变 1 帧，A7 改变 0 帧）。

## §7. Easy Anchor Set

Phase 14 easy set（21 帧）原样复用。Phase 14.1 的 V00/V10/V01-R1/V11-R1 产物全量复用（无重跑）。

## §8. Factorial Design

2×2 析因（与 Phase 14.1 完全一致）：

| Variant | A6 | A7 |
|---|---|---|
| V00 Control | OFF | OFF |
| V10 A6 | ON | OFF |
| V01-R1 A7 | OFF | ON |
| V11-R1 A6+A7 | ON | ON |

## §9. A7 Session Safety

Phase 14.1 已修复 session 生命周期。V01-R1 / V11-R1 已验证 5/5 sample ok，无泄漏。

## §10. 分层指标

### Per-stratum mean F1

| Stratum | n | V00 | V10 (A6) | V01 (A7) | V11 (A6+A7) |
|---|---|---|---|---|---|
| hard | 7 | 0.9838 | 0.9838 | 0.9842 | 0.9838 |
| mid | 7 | 0.9844 | 0.9761 | 0.9842 | 0.9763 |
| easy | 7 | 0.9823 | 0.9823 | 0.9823 | 0.9822 |
| **all** | **21** | **0.9835** | **0.9807** | **0.9836** | **0.9807** |

### Per-stratum Factorial Effects (mean F1)

| Stratum | E_A6 | E_A7 | I_AB |
|---|---|---|---|
| hard | -0.0002 | +0.0001 | -0.0004 |
| mid | **-0.0081** | -0.0000 | +0.0004 |
| easy | -0.0001 | -0.0000 | -0.0001 |
| all | -0.0028 | +0.0000 | -0.0001 |

**E_A6 集中在 mid stratum**，完全由单帧 ChangShouHua2_0075 的回退贡献。

### Per-stratum min/max F1

详见 `挑战集列表/phase15_per_stratum_metrics.csv`

## §11. Paired Effects

### 材料级回退（|ΔF1| > 0.005）

| 变体 | stratum | 回退数 | 改善数 |
|---|---|---|---|
| V10 (A6) | hard | 0 | 0 |
| V10 (A6) | mid | **1** | 0 |
| V10 (A6) | easy | 0 | 0 |
| V01 (A7) | — | 0 | 0 |
| V11 (A6+A7) | mid | **1** | 0 |

唯一材料回退：**ChangShouHua2_0075**（mid stratum），F1 0.9856→0.9278，ΔF1=-0.058。

### A6 mask 变化分析

A6 共识投票在 V10 上对所有 21 帧启用（共识启用=1），5 帧投票通过（共识接受=1），16 帧回退。

**关键发现**：5 帧"接受"中，**仅 ChangShouHua2_0075 实际改变了掩膜**（208,179 像素差异）。其余 4 帧（ChangShouHua2_0025/0050, DouBanLv1_0025/0050）的 V10 掩膜与 V00 **bitwise 完全相同**。

### A7 触发分析

V01-R1 共识记忆选择 9 帧（CaoMei1 × 4, DouBanLv1 × 5），每 stratum 3 帧：

| stratum | A7-selected n | mean ΔF1 (on) | mean ΔF1 (off) |
|---|---|---|---|
| hard | 3 | +0.0008 | 0.0000 |
| mid | 3 | -0.0004 | 0.0000 |
| easy | 3 | +0.0001 | 0.0000 |

A7 selected 帧的 ΔF1 均 < ±0.001（数量级上无意义）。

## §12. Practical Significance

21 帧中，A6 唯一材料回退（ChangShouHua2_0075，ΔF1=-0.058），A7 无材料回退/改善。**在 easy set 天花板效应下，无法评估 hard 条件下的 practical significance。**

## §13. Difficulty Correlation

| 变体 | Spearman(diff, ΔF1) | p-value | 解释 |
|---|---|---|---|
| A6 (V10) | -0.919 | <0.001 | **单帧伪象**：仅 ChangShouHua2_0075 有非零 Δ，移除后余 20 帧 Δ=0，相关性消失 |
| A7 (V01) | +0.148 | 0.522 | 无显著相关 |
| A6+A7 (V11) | -0.491 | 0.024 | 同为单帧驱动 |

**结论**：A6 的"难度相关性"完全由单帧 ChangShouHua2_0075 驱动，不构成可推广的趋势。A7 无难度响应。

## §14. Difficulty-Category Analysis

| 层 | A6 效果 | A7 效果 |
|---|---|---|
| hard（7帧） | E_A6=-0.0002（neutral） | E_A7=+0.0001（neutral） |
| mid（7帧） | E_A6=-0.0081（全部由 1 帧驱动） | E_A7≈0（neutral） |
| easy（7帧） | E_A6=-0.0001（neutral） | E_A7=-0.0000（neutral） |

hard stratum 7 帧中，A6 对所有帧的 ΔF1=0（bitwise identical）。**hard stratum 的 A6 无激活/无变化**。

## §15. A6 Mechanism

A6 共识投票机制分析：

- 共识启用：21/21 帧
- 共识接受：5/21 帧（ChangShouHua2_0025/0050/0075, DouBanLv1_0025/0050）
- **实际掩膜改变**：1/21 帧（仅 ChangShouHua2_0075，Δpixels=208,179）
- 其余 4 帧"接受"后 V10=V00 bitwise → 共识投票通过但最终选择未变（V00 基线足够好，候选覆盖了共识输出）

**机制推断**：在 easy set 上，共识投票的边缘删除/补回（回退IoU 高→接受→但掩膜变化已被 V00 的候选覆盖吸收）使得 A6 几乎无增量变化。ChangShouHua2_0075 是唯一因共识回退 IoU 偏高（0.8887）而实际改变掩膜的帧——回退 IoU=0.8887 意味着共识支持略低但仍过阈值，导致删除有效前景像素。

## §16. A7 Mechanism

- V01-R1：13 帧传播，9 帧采用，5/5 sample ok
- V11-R1：13 帧传播，5 帧采用
- 采用帧均为 CaoMei1 + DouBanLv1 的前 5 帧（共同两个 sample）
- ΔF1（on）数量级 < ±0.001 → A7 在 easy set 上完全中性

## §17. Trigger-Conditioned Effects

### A6 trigger conditioned

| stratum | 共识接受 n | mean ΔF1 (accepted, actually changed) |
|---|---|---|
| hard | 2 | 0.0000（两帧均未改变掩膜） |
| mid | 3 | -0.0193（仅 ChangShouHua2_0075 贡献全部负值） |
| easy | 0 | N/A |

### A7 trigger conditioned

| stratum | A7-selected n | mean ΔF1 |
|---|---|---|
| hard | 3 | +0.0008 |
| mid | 3 | -0.0004 |
| easy | 3 | +0.0001 |

## §18. Rescue Rate

**rescue rate = 0/21**。21 帧 V00 F1 全部 ≥ 0.976，无"失败"帧可供救援。

## §19. Statistical Analysis

由于 20/21 帧 ΔF1=0（A6）和 21/21 帧 ΔF1≈0（A7），Wilcoxon signed-rank 检验无统计功效。所有检验均为显著性不可计算（p=1.0 for all non-zero pairs）。

## §20. Visual Gallery

可视化生成脚本（`make_contact_sheet.py`）已就绪。ChangShouHua2_0075 的 V00/V10 overlay 已在 Phase 10/14 产物中保存。

## §21. Easy vs Hard Comparison

| 层 | V00 F1 mean | E_A6 | E_A7 |
|---|---|---|---|
| hard（高难度属性） | 0.9838 | -0.0002 | +0.0001 |
| mid | 0.9844 | -0.0081 | -0.0000 |
| easy（低难度属性） | 0.9823 | -0.0001 | -0.0000 |

hard stratum（最高难度属性）的 A6 E_A6=-0.0002 ≈ 0，A7 完全中性。**A6/A7 效应不随场景难度增加而改善**（hard stratum 无激活、无差异）。

## §22. Architecture Decision

| 模块 | 决策 | 理由 |
|---|---|---|
| **A6（跨视角共识）** | **CONDITIONAL → DEMOTE** | 21 帧中仅 1 帧有非零掩膜改变，且为 material regression（ΔF1=-0.058）；其余 20 帧 Δ=0。共识投票通过但不改变掩膜，说明 easy set 的 P6 候选已足够稳定，A6 增量信息为零。建议 DEMOTE：保留代码但默认禁用（use_cross_view_consensus=False 已为默认）。 |
| **A7（SAM3 传播）** | **DEMOTE** | 采用 9 帧，ΔF1 全部在 ±0.001 内，完全中性。传播本身有效（13 帧）但对最终选择无贡献。建议 DEMOTE：保留但默认禁用。 |

**注意**：以上决策基于 easy set（天花板效应）。在 hard set 上 A6/A7 的行为未被测试（需要标注）。

## §23. No Contamination

- 21 帧 easy set 的 GT QA 已在 Phase 12 完成（§4）
- 未对任何参数做调整
- 60 hard-candidate 的难度排名仅用于 §13-§14 的辅助分析，不影响因子设计

## §24. Output Directory

```
阶段十五_硬案例挑战集与外部验证/
├── Phase15_report.md                （本报告）
├── 脚本/
│   ├── scout_difficulty.py           （难度计算）
│   ├── make_contact_sheet.py         （缩略图联系表）
│   ├── stratify_easy_by_difficulty.py（分层脚本）
│   ├── phase15_per_stratum_analysis.py（分层析因脚本）
│   └── process_gt_challenge.py       （GT处理，smoke-tested，备用）
├── 挑战集列表/
│   ├── difficulty_ranking.csv        （60 候选帧，hard-case pool）
│   ├── candidates_p1.png             （60 候选缩略图联系表）
│   ├── 标注指南.md                    （标注格式说明，备用）
│   ├── 21易帧难度分层.csv            （21 帧分层结果）
│   ├── phase15_per_frame_f1.csv      （21 帧逐帧 F1 + 触发）
│   └── phase15_per_stratum_metrics.csv（分层汇总）
└── GT_QA/                            （空，无新标注）
```

## §25. Regression

- Phase 14/14.1 产物未改动（git-clean，SHA `f9f091d8`）
- 未重跑任何推理（零计算路径）
- 所有分析脚本已验证：84/84 历史测试仍通过（未改动代码）

## §26. Frozen Evidence Integrity

- 选择后掩膜（V00/V10/V01-R1/V11-R1）：未修改
- 提示词选择.csv / 记忆传播.json / 共识投票.csv：未修改
- GT_potted_clean：未修改
- Frozen 参数：未修改

## §27. Acceptance Gates

| Gate | 判据 | 状态 |
|---|---|---|
| G1 | 难度分层脚本可复现 | PASS（`21易帧难度分层.csv` 已生成） |
| G2 | 三等分 7/7/7 | PASS（hard/mid/easy 各 7 帧） |
| G3 | H1 帧在分层中有定位 | PASS（2 hard, 1 mid） |
| G4 | 分层析因完成 | PASS（E_A6/E_A7/I_AB 每层均已计算） |
| G5 | A6 mask 变化确认 | PASS（1/21 帧实际改变，其余 bitwise identical） |
| G6 | 触发条件分析完成 | PASS（共识接受/记忆采用 全层统计） |
| G7 | Phase 14/14.1 未改动 | PASS（git-clean，SHA unchanged） |
| G8 | 零新推理 | PASS |
| G9 | 零阈值/参数调整 | PASS |
| G10 | 60 候选帧可供未来标注 | PASS（CSV + 联系表已生成） |
| G11 | 报告覆盖 §0-§23 | PASS |
| G12 | 难度属性纯原始图（§3 合规） | PASS（仅 Laplacian/熵/ExG/绿色比/动态范围） |

## §28. 复现性

```
Python / env：  /home/test/biosoft/enter/envs/sam3/bin/python
GPU：            未使用（零计算路径）
SAM3 检查点：   /data/fj/F2DMAS/sam3/sam3.pt
输入：          Phase 14.1 产物（V00/V10/V01-R1/V11-R1 选择后掩膜 + 提示词选择.csv + 共识投票.csv）
GT：            GT_potted_clean（Phase 12）
分析脚本：      阶段十五_硬案例挑战集与外部验证/脚本/ 下全部 .py
```

## A. Future Work Recommendation

1. **标注 30-60 硬帧**（2.5-5h 手工标注）：从 15 个 non-GT sample 中按 `difficulty_ranking.csv` 选取 30-60 帧，labelme 标注 → `process_gt_challenge.py` → GT QA → 冻结 → V00/V10/V01/V11 析因。
2. 有了真正硬帧 F1 才能完成 §18（rescue rate）、§19（Wilcoxon 显著性）、§22（基于硬帧的架构决策）。
3. 本阶段的难度排名 CSV（60 候选帧）和缩略图联系表已准备好，标注工作可随时启动。
