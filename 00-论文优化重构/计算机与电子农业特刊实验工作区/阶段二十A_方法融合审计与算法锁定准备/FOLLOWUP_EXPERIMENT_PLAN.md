# 后续最小必要验证实验设计 / FOLLOWUP_EXPERIMENT_PLAN

**阶段：** Phase 20A（**只输出计划，不执行推理、不启动标注**）

---

## 1. V0–V3 完整链路对照（方案 A：P6 + per-instance）

### 1.1 四组定义与固定变量

| 组别 | Prompt | Ranking | Refinement | 其他固定量 |
|---|---|---|---|---|
| **V0** | P6 固定 | R0 (`base`) | OFF | 见 §1.2 冻结表 |
| **V1** | P6 固定 | R1 | OFF | 同上 |
| **V2** | P6 固定 | R0 (`base`) | ON | 同上 |
| **V3** | P6 固定 | R1 | ON | 同上 |

### 1.2 冻结变量表（每组 `参数.json` 必须显式写出，禁止依赖 CLI 默认）

```
--prompt_list P6
--default_prompt_id P6
--use_prompt_ensemble false          # ← 关键！为 true 则落入 L1688 score_select，R1 被绕过
--prompt_selection_mode single
--candidate_mode per_instance
--sam3_mask_threshold 0.5
--score_weights area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5
--base_cleanup_mode fsam3_basic
--confidence_threshold 0.3
--area_min_ratio 0.01 --area_max_ratio 0.80
--vertical_coverage_min_ratio 0.30
--use_temporal_alignment false
--use_semantic_gate false
--use_reprompt_detection false
--use_cross_view_consensus false     # A6 关
--use_memory_propagation false       # A7 关
--save_candidate_masks --save_intermediate_masks --save_raw_instance_masks --save_foreground_rgb --save_rgba --save_overlay
--ranking_policy {base|r1}           # ← 唯一变量
# V2/V3 额外：
--use_spnp_refinement true
--spnp_backend sam3_if_supported
--spnp_positive_points 5 --spnp_negative_points 8 --spnp_ring_radius 15
--spnp_min_refined_iou 0.65
--use_residual_repair true
--thin_repair_mode adjacent
--closing_kernel 15 --opening_kernel 3
--use_corrective_geometry true
--corrective_geometry_backend colmap_tracks
--colmap_dir <absolute 03-final_locked>
```

> **唯一允许变化的是 `ranking_policy` 与 refinement 三开关。**
> 任何其它差异都会破坏公平性。

### 1.3 现有产物盘存与缺口

| 组别 | 现有产物 | 缺口 | 可否离线重放 | 需 GPU |
|---|---|---|---|---|
| **V0** | ✅ Phase 12 `P6_raw_baseline`（21 帧）+ Phase 17A `P00_Control`（10 帧）+ `U00_Control`（32 帧）均等价于 V0（等价性已由 Phase 17C T1 在 10/10 DEV10 帧上证明） | 无正式 V0 需求 | **是**（候选已冻结） | ❌（若只需重排序） |
| **V1** | ❌ 仅 CSV 重排序（`Phase17C_ranking_benchmark.csv`、`Phase17C_test32_safety.csv`） | **缺正式运行产物（最终掩膜）** | 候选可复用；**最终掩膜需重算**（离线，从 `候选掩膜/raw_instance_P6/` 选实例再走 refine 链，零细化） | **建议至少一次正式 GPU 运行** |
| **V2** | ❌ 现有 refinement 运行全是 `score_select`（R0） | **缺 single+R0+refinement** | 否（候选可复用，但需重跑 refine 阶段） | **必需** |
| **V3** | ❌ **零例** | 全部缺 | 否 | **必需** |

**结论：GPU 必需（V2、V3；V1 建议正式运行以产出可引用的掩膜）。**

### 1.4 候选缓存复用策略（降低 GPU 成本）

1. **V1 可用 `--reuse_existing_candidates --candidate_source_dir <Phase17A P00_Control 候选掩膜>`**
   → 零 SAM3 推理，仅做评分 + R1 选择。
2. **V2/V3 同样可复用候选**，但 SB-FBPR 后端为 `sam3_if_supported`，需要 torch/model 在内存中
   （`main()` L2771-2775 会因 `use_spnp_refinement and spnp_backend==sam3_if_supported` 而加载 SAM3）。
   因此 **V2/V3 仍需 GPU**，但可**不重新做候选生成**（省一次全量推理）。
3. **候选复用会改变候选掩膜的来源目录** → 必须在 `参数.json` 与报告注明。

### 1.5 评价指标

| 层级 | 指标 | 备注 |
|---|---|---|
| **候选选择前** | oracle F1（最佳原始实例 vs GT）、`#inst` | 复用 Phase 17B 的 `Phase17B_candidate_oracle.csv` 口径 |
| **选择后（R0 vs R1）** | selected F1、**oracle regret = oracle_F1 − selected_F1** | 这是 R1 的核心指标 |
| **refinement 后（V0→V2、V1→V3）** | final F1、**refinement ΔF1 = final_F1 − selected_F1** | 隔离细化模块的净贡献 |
| **交互项** | `interaction = (V3 − V2) − (V1 − V0)` | R1 × 细化的交互效应 |
| **安全性** | 空掩膜数、每帧连通域数、面积比例分布 | 防止细化引入结构性破坏 |
| **泄漏** | 外部非黑比例、泄漏能量 | 与论文 Table 3/5 同口径 |

**评价数据集：**
- 有 GT：Phase 16 GT10（10 帧，BaiZhang 0033–0037 + DouBanLv2 0037–0041），角色 = **DEV**；
  Phase 12 GT_potted_clean（21 帧，Easy21），角色 = **DEV/安全集**。
- ⚠️ **两者均为开发集**。V0–V3 全部属于开发集消融，不得作为独立验证。

---

## 2. 独立验证计划（增加有 GT 的独立验证）

### 2.1 目标与约束

- **目标**：为 R1 提供一次**不参与任何调参**的独立验证。
- **约束**：
  - R1 **冻结**（公式、权重、提示词、细化开关全部锁死）；
  - 按**序列**切分，非按帧（避免同序列帧泄漏）；
  - 每帧明确标注 plant-only 或 plant-plus-pot；
  - **不得针对测试集再次调参**；
  - 不把 GT10 或 TEST32 当独立测试集。

### 2.2 建议设计

| 项 | 建议 | 理由 |
|---|---|---|
| **冻结集** | Phase 16 TEST32 的 **8 个序列**（CaoMei2, ChangShouHua3, DouBanLv3, HongZhang, WanNianQing1, WangWenCao2, XianKeLai3, XiangPiShu2） | 已按序列级 DEV/TEST 隔离；当前无 GT |
| **标注量** | 每序列 4 帧（与 TEST32 对齐）= **32 帧**；或每序列 5 帧 = 40 帧 | 与 Phase 12 的 21 帧同量级，可比 |
| **标注口径** | `potted_clean`（含盆、去蓝块），与 Phase 12/16 一致 | 见 `PLANT_POT_SCALE_CONSISTENCY.md` §6 |
| **必须补标 pot 多边形** | 建议：`potted_plant` + `pot` + `plant` + `blue_cube` 四类全标 | 否则无法评估 plant-only 变体；现有挑战 GT 缺 pot（`formal_p2_valid=False`） |
| **D 类标签** | 若用于困难场景图，需**人工视觉核查**并记录 `label_origin=human_verified` | 现有 D1/D11 为 `green_pct` 阈值启发式，53/58 帧同标签 |
| **执行** | R0 与 R1 各跑一次（同一候选，两次选择），或一次运行 + 一次离线重排序 | R1 冻结，无需为 R1 单独重跑推理 |
| **报告口径** | 明确写"独立验证集，开发过程中未接触" | 不得称为 "test set" 于开发阶段 |

### 2.3 样本量与检出力提示

- 32 帧、单一方案对比时，F1 均值差 0.02 量级需要逐帧配对检验（paired），非独立样本 t 检验。
- 建议报告：均值/中位数、逐帧配对差、≥0.005 的实质回归帧数、**空掩膜数**。

---

## 3. 建议的多提示词公平对照（若未来需要评估方案 B）

> 本审计不推荐方案 B；若仍要评估，必须**新建**对照，不得混用旧 P1–P5 结果。

| 组别 | Prompt | Ranking | Refinement | 说明 |
|---|---|---|---|---|
| M0 | P1–P5 score_select | R0 | OFF | 复现旧口径基线 |
| M1 | P1–P5 score_select | R1 | OFF | **需要改代码**（L1688 → `select_best_policy`） |
| M2 | P6 single per_instance | R0 | OFF | = V0 |
| M3 | P6 single per_instance | R1 | OFF | = V1 |

**公平性要求：**
- M0/M1 与 M2/M3 必须在**同一帧集、同一 GT、同一指标实现**下评估；
- 不得把 M0（旧 P1–P5，legacy-union GT）的数字与 M2/V0（`potted_clean` GT）的数字并列比较；
- 需记录"提示词集不同"这一混淆变量，不得把 M1−M0 的差全部归因于 R1（提示词集与细化开关同时变化会混淆归因）。

---

## 4. 本阶段明确不执行

- ❌ 不运行任何 SAM3 推理、2DGS 训练、TSDF 网格化；
- ❌ 不启动新的人工标注；
- ❌ 不修改生产代码以"让实验跑通"；
- ❌ 不对 TEST32 计算任何准确率；
- ❌ 不把 GT10 当独立测试集。