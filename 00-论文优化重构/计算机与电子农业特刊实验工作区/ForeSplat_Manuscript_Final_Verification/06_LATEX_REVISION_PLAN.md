# 06 — LATEX REVISION PLAN

**可指导下一轮实际修改 `ForeSplat.tex` 的详细编辑方案**

- 目标文件：`ForeSplat__Foreground_Object_2D_Gaussian_Splatting_via_Semantic_Mask_Guided_Task_Reformulation_for_Plant/ForeSplat.tex`（850 行，8 节）
- 本轮 **不修改** 正文。本文件是给下一轮的编辑指令。
- 原稿行号以当前 `ForeSplat.tex` 为准。

---

## A. 按优先级排序的修改清单

### P0-1　Abstract：删除无法追溯的端到端性能

| 原稿位置 | 原始主张 | 实际问题 | 修改方向 | 证据 |
|---|---|---|---|---|
| L88–95 `\begin{abstract}` | "RAP-FSAM3 achieved an F1-score of 0.9706, an mIoU of 0.9641, an HD95 of 183.42 px, a Boundary-F1 of 0.4643 and a leakage energy of 0.001739 **on manually annotated evaluation frames sampled from all 20 RGB sequences**" | ① 实为 **4 个代表序列 × 5 帧 = 20 帧**，非 20 序列；② 用 legacy union GT；③ 5 种 GT 变体重算无一能复现；④ 属旧版 RAP-FSAM3-v2（`score_select`，无 R1） | 换成 potted_clean 下 Easy20 的 0.98373 / 0.97963 / 23.73 / 0.48521 / 0.001739，并写 "on 20 manually annotated frames from four representative sequences" | `P0_BLOCKERS` P0-3；`E1_.../方法覆盖与运行状态.md`；`TABLE3_RECALCULATED.csv` |
| 同上 | "Across 20 RGB multiview sequences … ForeSplat achieved PSNR = 31.09 dB, SSIM = 0.9711 and LPIPS = 0.0365" | `TABLE7_CLAIM_STATUS.csv`：4/4 行 **UNVERIFIED**，无同源 20 序列清单/划分/ROI/计时边界 | **整句删除**。改为“在 3 个场景的配对评测中报告前景指标”，或全部移入 Discussion 并标为未闭合协议 | `TABLE7_CLAIM_STATUS.csv`；`阶段二十B.../reports/TABLE7_PROVENANCE_AUDIT.md` |
| 同上 | "\(R^2\) values of 0.988, 0.988, 0.974 and 0.900" | 无 mesh/model/mask 链路；Phase20F 实测 0 个有效厘米值；`virtual_values_are_new_mesh_results=false` | **整句删除**。改为“说明当前协议下厘米级表型测量的阻断”，或移入补充材料作 legacy 记录 | `阶段二十F.../PHASE20F_FINAL_REPORT.md`；`HISTORICAL_PHENOTYPE_MATCHING.csv` |
| 同上 | "Objective ablation showed that foreground RGB supervision reduced the outside-mask non-black ratio from 0.9908 to 0.0294 … whereas post hoc mask pruning after full-scene training remained insufficient" | 数值真实（A0→A5），但为 **单场景单次运行**，且掩膜来自旧 SAM 集而非 P6+R1 | 保留数值，补 "on a single representative potted-plant sequence" 与 "using the legacy SAM mask set" | `kongquezhu_A0_A6_ablation_summary.csv` |
| 同上 | "a lightweight M2M3-GO primitive-level refinement layer" | "primitive-level refinement"暗示 keep/densify/prune 联合决策；代码只做剪枝 | 改为 "an optional mask-supported weighted Gaussian-pruning module" | `M2M3_MANUSCRIPT_CORRECTION_SPEC.md` |

**P0-1 编辑指令：** 重写整个 abstract，只用 C1/C3/C5 三类有来源的结论；M2M3 不再进摘要。

---

### P0-2　Introduction：novelty 重定位

| 原稿位置 | 原始主张 | 实际问题 | 修改方向 | 证据 |
|---|---|---|---|---|
| L100–101 "These advances also leave a practical agricultural question unresolved: **is the reconstructed 3D object the plant object that needs to be measured?**" | 方向正确，但没有落到“目标实例选择”这一具体失败 | 这是全篇最好的问题意识，但下文没有承接 | 保留并强化：加一段“同一帧里 SAM3 返回多个实例，几何质量最好的候选常不是目标植株，因为高前景-背景对比度会奖励花朵/邻株” | `阶段十七B_Phase17B_protocol.md`；`R0_R1_TEST32_RESULTS.csv` |
| L104 "Vision foundation models provide a promising route for this coupling, and recent work has explored their adaptation… However, they are not plug-and-play tools" | 把 SAM3+2DGS 的组合当作主要 novelty | 组合不是贡献 | 改为：问题 = **目标实例选择** + **前景先验必须进入优化目标** | `02_NOVELTY_POSITIONING.md` |
| L106–108 "Contributions of this study are as follows" 三条 | ②③ 与证据强度不匹配（② 单场景；③ 无正向证据） | — | 按 `02` §5 的三条英文贡献整体替换；M2M3 从独立贡献降为第三条贡献内的一个组件 | `02_NOVELTY_POSITIONING.md` §3、§5 |

---

### P0-3　RAP-FSAM3 方法节：默认路径与公式重写

| 原稿位置 | 原始主张 | 实际问题 | 修改方向 | 证据 |
|---|---|---|---|---|
| L189 `\subsubsection{Multi-Prompt VFM Candidate Generation}` + 公式 `K=5` | 默认生产只有 **P6 单提示词** | `prompt_selection_mode=single`, `use_prompt_ensemble=false` | 改写为 “single plant-instance prompt (P6) + per-instance candidates”；把 P1–P5 的候选集作为历史变体在 3.6.2 的版本定义中说明 | `FINAL_RAP_FSAM3_SPEC_DRAFT.md` §2 |
| L199–241 `\subsubsection{Semantic-Gated Multi-Prompt Selection}`（`φ`/`ψ`/`σ`/softmax/`τ_s`） | softmax 与温度 **未实现**；生产默认门控关闭 | `grep softmax|temperature|tau_s` 仅命中 `q_sam` 的 sigmoid | **整节替换为 3.2.2 CDR**：给出 `S_R0`、`S_R1` 与 `denom=5.5`，并说明 contrast 项被降级为诊断信号而非删除 | `P1-2`；`ranking_policy.py`；`R1_FORMULA_VERIFICATION.md` |
| L242–274 `SB-FBPR` | 与 R1 **零联合证据**（78 份运行中 `single+SPNP` 0 例） | `P0-5` | 降级为 3.2.3 “Optional foreground refinement (disabled by default)”，一句话带过 + 说明未在 R1 配置下单独验证 | `R1_CALL_PATH_AUDIT.md` |
| L275–306 `GCFC` | 同上 | 同上 | 合并进 3.2.3 | 同上 |
| L189–192 FFT 公式 `Q_FFT(I)=‖W_H⊙log(1+\|F(I)\|)\|_1/(…)` | 与实现不符：实现是 `1 − low_energy/total_energy`、平方能量、中心 10% 圆盘、**无 log 压缩**，且叠加 50% Laplacian 与 `max_consecutive_drop=2` | `P1-3` | 按实现改写，或删除公式改为文字描述“normalised high-frequency saliency + per-sequence first-quartile threshold + continuity guard” | `P1-3`；`生成RAP-FSAM3掩膜.py` |

---

### P0-4　2DGS 损失公式：与代码对齐

| 原稿位置 | 原始主张 | 实际问题 | 修改方向 | 证据 |
|---|---|---|---|---|
| L376–391 `L_core(t)=L_rgb-fg + χ(t)(λ_mask L_mask + λ_bg L_bg) + L_reg`；"We used λ_mask=0.08 and λ_bg=0.02" | ① `0.08` 实际是 `(0.5·L1 + 0.5·Dice)` 的整体系数；② `L_reg` 未给权重（实为 `λ_normal=0.05`，iter>7000；`λ_dist=0.0`）；③ `χ(t)` 语义正确 | 代码 L185–206、L555–581 | 改写为显式三项拆分，并补 `λ_normal`。可写作：`L = λ_fg_rgb[(1−λ_dssim)L1_fg + λ_dssim(1−SSIM_fg)] + χ·0.08(0.5 L1^α + 0.5 Dice^α) + χ·0.02 L_bg + λ_normal L_normal` | `train.py` L136–206, 555–581；`command.txt` |
| L393–411 `View-Quality-Aware Soft Loss Weighting` 的 `q_i=q_min+(q_max−q_min)σ((s_i−μ_s)/σ_s)` | 该公式 **未实现**；`view_quality_mode` 是死标志 | 权重来自外部预计算 CSV，clamp[0.6,1.0]，仅缩放 RGB 损失 | 改写为：“view weights are precomputed per sequence from mask coverage, boundary sharpness and foreground contrast, clamped to [0.6, 1.0], and applied only to the RGB term (`view_weight_mode=rgb_only`)” | `train.py` L74–114；`arguments/__init__.py:58` |
| L313–332 前景轨迹初始化 | 机制真实 | 需披露 mask 身份未闭合 | 保留公式，表注加 “the mask set used by the 2DGS stage is the legacy SAM mask set; pixel-level identity with the P6+R1 masks has not been verified” | `P0-2` |

---

### P0-5　M2M3-GO 两节：删公式、改定位

| 原稿位置 | 原始主张 | 实际问题 | 修改方向 | 证据 |
|---|---|---|---|---|
| L412–438 `Mask-Guided Multi-Cue Gaussian Pruning`（`z_j`、`P_keep=σ(w_g^T z − b_g)`） | sigmoid 效用 **未实现** | 实际是五项归一化正惩罚加权和 | 整节与 3.4 合并，用 `M2M3_MANUSCRIPT_CORRECTION_SPEC.md` 的真实 `q_i` 公式替换 | `M2M3_IMPLEMENTATION_TRUTH.md` |
| L439–549 `\subsection{M2M3-GO…}` 的 `S^{m2g}`、`H_i`、`S^{g2m}`、`B^∂`、`U_j`、`Prune/Densify/Keep` | **全部未实现** | 仅 `mask_consistency_scores` 一个方向（中心投影代理）；无 deficit、无边界项、无 sigmoid、无增密 | **删除这些公式**；改为：中心投影支持度 `s_i`、剪枝分数 `q_i`、预算 `B=⌊0.03N⌋`、全局下限 `F_t=min(N, max(0, round(0.4·N_max_seen)))` | `train.py` L240–465 |
| L530–547 Floor40 局部块 `N_k^min=ρ_floor N_k^ref` | 实为 **全局单一标量** | `resolve_capacity_floor_count` 返回全局值 | 改为全局表述，并**必须写** “in the frozen runs the floor never blocked a deletion (`floor_blocked_count = 0` in all 15 M2 events)” | `M2M3_CAPACITY_EFFECT_REPORT.md` |
| L441–448 "M2M3-GO does not serve as a standalone compression or reconstruction module" | 定位表述本身是对的 | 但 §3.7 又给出完整算法，读者会期待独立消融 | 保留该定位句，并把公式全部删到只剩实现级描述 | `M2M3_METHOD_DECISION.md` |

---

### P0-6　Results：删无来源行、限定范围、补新表

| 原稿位置 | 原始主张 | 实际问题 | 修改方向 | 证据 |
|---|---|---|---|---|
| L578–584 `RAP-FSAM3 achieved F1=0.9706 … sampled from all 20 RGB sequences` | 同 P0-1 | 同 P0-1 | 换数 + 改口径 + 声明版本 | `TABLE3_RECALCULATED.csv` |
| L595–604 Table `tab:vfm-comparison` | 同上 | 同上 | 换 4 行数值；表注改 "four representative sequences, 20 frames" | 同上 |
| L617–629 Table `tab:prompt-robustness`，P2 标签 "entire potted plant" | 实为 "entire plant excluding pot"；且 P2 盆召回≈97.5%，命名与行为不符 | `P1-1` | 改正标签；补披露；数值换为 TABLE4 重算值 | `TABLE4_RECALCULATED.csv` |
| L637–665 Table `tab:rap-ablation`，`w/o GCFC` 行 Boundary-F1=0.4543 / Leakage=0.001935 | **全仓无出处** | 且该行 F1/mIoU 取自 A5c、HD95 取自 A2，跨聚合混用 | **删除该行**；其余行换 TABLE5 重算值；补“Geometry-only selector 为 OFFLINE_REPLAY；Without_GCFC 为中间产物，非独立消融” | `HISTORICAL_EVIDENCE_ERRATA.md`；`TABLE5_RECALCULATED.csv` |
| L686–704 Table `tab:injection-ablation` | "Complete ForeSplat objective w/o M2M3-GO" 行 25.0478 / 596,840 与 "Post hoc mask pruning" 行 468,920 **全仓无出处** | 当前数值还跨 A5/A6/F1 混用 | 删 25.0478 / 596,840 / 468,920；重排归属：注入行用 A5（25.1055 / 592,900），完整行用 A6（25.0072 / 591,623）或 A6+M4（24.9723 / 585,594）；后剪枝行的真实计数改为 E7 前景导出 **340,413** | `kongquezhu_A0_A6_ablation_summary.csv`；PLY 头 |
| L731–793 Table `tab:reconstruction` 四行 | 全部 UNVERIFIED | 无同源协议 | **整表删除或整体降级**；用 Phase20E1/E2 的三场景配对表替换 | `TABLE7_CLAIM_STATUS.csv` |
| L770–792 表型 `tab:phenotyping` 与 R² 叙述 | 无链路 | Phase20F 0 个有效厘米值 | **整表删除**；改为表型状态表（GEOMETRIC_ONLY / BLOCKED） | `PHASE20F_FINAL_REPORT.md` |
| — | 无 M2M3 容量节、无 2D→3D 节 | — | **新增** 4.5 与 4.6 两节 | `M2M3_SCENE_SUMMARY.csv`；`S24.../experiment4b_summary.csv` |

---

### P0-7　Discussion 与 Conclusion

| 原稿位置 | 原始主张 | 实际问题 | 修改方向 | 证据 |
|---|---|---|---|---|
| L794–797 "Vision Foundation Models Require Agricultural Prompt Adaptation" | 把问题归为“需要提示词适配” | 真实瓶颈是 **目标实例选择**（Phase17B：正确候选已生成，未选中） | 改为 “Why target-instance selection, not prompt engineering, is the bottleneck” | `Phase17B_protocol.md` |
| L805–810 "From Foreground-Object Supervision to Primitive-Level Capacity Allocation" | 暗示 M2M3 有稳定收益 | 实测 MIXED + Floor40 NO_EFFECT | 改为如实讨论：真实触发、方向不一致、floor 未干预、容量差不可因果归因（iteration 601 已分叉） | `M2M3_METHOD_DECISION.md`；`M2M3_CAPACITY_EFFECT_REPORT.md` |
| L818–820 "Relationship to Reference Studies" | 只与 IPENS/LCR-GS/Gaussian Grouping/SAGA 比较 | 未覆盖 object-centric 2DGS (2025)、草莓 object-centric 3DGS (2026)、LCR-GS (2026)、PlantGaussian (2025)、F2DMAS (2026) | 扩写为独立 Related Work 2.4，并加 Table 1 | `02_NOVELTY_POSITIONING.md` §1 |
| L822–832 Conclusion | "Validation across 20 sequences and 21 plants indicates that the workflow can support low-cost measurement of plant height, canopy width, leaf length and leaf width" | 无证据 | 改为：“我们给出可审计的网格结构记录，并说明在当前协议下厘米级表型测量尚不可得及其原因” | `PHASE20F_FINAL_REPORT.md` |

---

## B. 公式清单（必须调整）

| 公式 | 原稿位置 | 动作 |
|---|---|---|
| FFT `Q_FFT` | L183–187 | 改为实现口径或删公式 |
| `K=5` 候选集 | L195–198 | 改为单提示词；K=5 移入版本定义 |
| `φ_i, ψ_i, c_i, a_i, τ_s`（语义门控） | L201–239 | **删除** |
| `P_i^+, R_i^-, P_i^-`（SB-FBPR） | L244–272 | 降级为文字描述 |
| `H_i(p), G_i, M_i^geo`（GCFC） | L277–305 | 降级为文字描述 |
| `Γ(X_j), Keep(X_j)`（前景轨迹） | L317–331 | **保留** |
| `L_rgb-full, w_M(p), L_rgb-fg, L_mask, L_bg, L_core, χ(t)` | L339–391 | **重写**（λ 拆分 + λ_normal） |
| `q_i, s_i, μ_s, σ_s`（视图质量） | L397–409 | **重写**（删 σ 公式） |
| `z_j, P_keep(g_j), Prune(g_j)`（多线索剪枝） | L418–436 | **删除**，换 `q_i` |
| `S^{m2g}, H_i, S^{g2m}, B^∂, U_j, Prune/Densify/Keep` | L445–528 | **删除** |
| `B={B_k}, N_k^min, Prune_floor`（局部 Floor40） | L530–547 | **改写**为全局 `F_t` |
| `D(x), w_c(x)`（mask-aware TSDF） | L556–573 | **删除**（正式结果未使用该加权） |
| outside/leakage 阈值 | L566–568 | **保留**（口径清楚） |

---

## C. 表格清单（必须调整）

| 表 | 原稿 label | 动作 |
|---|---|---|
| Dataset use and evaluation protocol | `tab:dataset-protocol` | 与 `tab:dataset-coverage` 合并为新 Table 2 |
| Dataset coverage | `tab:dataset-coverage` | 同上 |
| VFM comparison | `tab:vfm-comparison` | 换 4 行数值；改口径注；声明版本 |
| Prompt robustness | `tab:prompt-robustness` | 换数值 + 改正 P2 标签 |
| RAP-FSAM3 module ablation | `tab:rap-ablation` | 删 `w/o GCFC` 行；换其余数值；补证据类型 |
| Prior injection ablation | `tab:injection-ablation` | 删 2 个无来源数值；重排 A5/A6/F1 归属；后剪枝计数改 340,413 |
| Workflow reconstruction | `tab:reconstruction` | **整表删除** 或整体降级为未闭合协议 |
| Phenotyping | `tab:phenotyping` | **整表删除**；换为表型状态表 |

---

## D. 图件清单（必须调整）

见 `05_FIGURE_TABLE_REDESIGN.md` §3。要点：Fig1/Fig2/Fig5/Fig6/Fig8 重绘；Fig6（M2M3 机制）与 Fig9（表型散点）**删除**；Fig3/Fig4/Fig7 保留但改标注。

---

## E. 建议的 `\label` 与新增结构

```latex
% Related Work 独立成节（原稿没有）
\section{Related Work}\label{sec:related-work}
  \subsection{Vision foundation models for plant segmentation}
  \subsection{Object-centric Gaussian reconstruction}
  \subsection{Gaussian pruning and compact representations}
  \subsection{Positioning of ForeSplat}\label{sec:positioning}   % 内含新 Table 1

% 3.2 结构调整
\subsubsection{Frame-quality filtering and instance candidate generation}\label{...}
\subsubsection{Contrast-Decoupled Re-ranking (CDR)}\label{...}          % 原 Semantic-Gated
\subsubsection{Optional foreground refinement}\label{...}               % 原 SB-FBPR + GCFC 合并

% 3.4 两节合一
\subsection{Mask-supported Gaussian pruning with optional capacity floor}\label{...}

% 3.6 新增
\subsection{Experimental setup}\label{...}

% Results 新增两节
\subsection{Gaussian capacity and quality trade-offs}\label{...}
\subsection{Segmentation-to-reconstruction consistency}\label{...}
```

---

## F. 执行顺序建议（下一轮）

1. **先改 3.2 / 3.3 / 3.4 的方法节**（公式与默认路径），因为它们决定图与表的标注口径。
2. **再改 Abstract / Introduction / Contributions**（此时已有准确的方法表述可引用）。
3. **再换 Results 的表与数字**（按 `04` 的状态逐行替换）。
4. **最后重绘 Fig1 / Fig2 / Fig5，删除 Fig6 / Fig9**，并按 `05` 的表注规范补全算法版本/G T/聚合口径。
5. **全篇一致性扫描**：grep `20 sequences`、`31.09`、`0.9706`、`R^2`、`M2M3-GO`、`softmax`、`τ_s`，确认全部处理完毕。
