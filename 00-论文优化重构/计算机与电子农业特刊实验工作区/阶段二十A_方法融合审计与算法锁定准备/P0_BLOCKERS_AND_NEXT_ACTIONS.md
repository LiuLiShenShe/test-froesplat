# P0 阻断项与下一步行动 / P0_BLOCKERS_AND_NEXT_ACTIONS

**阶段：** Phase 20A
**规则：** 本阶段只记录问题与建议，**不修复任何生产代码或历史数据**。

---

## 0. 阻断项分级

| 级别 | 含义 |
|---|---|
| **P0** | 阻断最终方法锁定或论文可复现性，必须在 Phase 20B 解决 |
| **P1** | 影响论文表述准确性，不阻断锁定，但投稿前必须处理 |
| **P2** | 影响叙事完整性，可作为 limitations 或 future work |

---

## 1. P0 阻断项

### P0-1　尺度参照（花盆直径）在仓库中完全不存在

- **证据**：
  - `grep pot_diameter | pot diameter | 花盆直径 | 盆径` 全仓（md/py/json/csv/xlsx）**0 数值命中**；
  - `植株数据.xlsx`（3 份副本，22 行 × 17 列）**无盆径列**；
  - `ForeSplat_real_release/scripts/measure_traits_entry.py:14` 仅转发 `--scale-marker`，无实现；
  - `configs/default_foresplat.yaml:12` 指向 `examples/demo_sequence/scale_marker.json` —— **该路径不存在**；
  - `06-MESH-new/*_mesh.log`（13 样本）不含 pot / diameter / scale 字段。
- **冲突**：论文 §4.7 称"Scale was recovered using the known physical diameter of the pot"，
  但 2DGS 的 `mask_dir`（`03-分割Mask/02-sam_masks/`）无 pot 标注，且目标声明是剔除盆体
  （`ForeSplat_zh.md:89, :194`）。若网格 plant-only，则无法从网格量取盆径。
- **影响**：Table 8 全部表型数字的上游不可复现。
- **下一步**：
  1. 从采集记录中找回每品种/每样本盆径实测值（含测量工具与不确定度）；
  2. 明确尺度参照域：**原始图像中的花盆** vs **重建网格中的花盆**；
  3. 若为前者，说明传递路径（图像域盆径 → 相机尺度 → 网格尺度）；
  4. 若为后者，需确认重建掩膜是否含盆（与 P0-2 联动）。
- **需 GPU：** 否。

### P0-2　2DGS / M2M3-GO 实际使用的掩膜与 P6 掩膜的同一性未核对

- **证据**：
  - 2DGS 各损失项（`train.py:134,161-189`）、COLMAP 前景初始化（`scene/__init__.py:119-173`）、
    掩膜剪枝（`train.py:241-297`；`m2m3-floor40/train.py:362-465`）使用**同一** `mask_dir`：
    `00-论文优化重构/数据管理/03-分割Mask/02-sam_masks/<Sample>/`；
  - 该目录 19 个样本、每样本 206–215 张掩膜，**无任何 pot 标注**；
  - P6 带盆输出在另一个目录（`03-分割Mask/05-RAP-FSAM3掩膜/E1_*_P6带盆整株_101帧/`）。
- **缺失**：逐像素/逐 IoU 比较 `02-sam_masks/` 与任一 P6 产物的结果。
- **影响**：无法判定"若换用 P6+R1 掩膜，2DGS 是否需要重训"。
- **下一步**：对 19 样本执行离线比较（IoU、changed-pixel ratio、前景面积差、盆区覆盖差），
  写入 `RECONSTRUCTION_EVIDENCE_REUSE_MATRIX.csv`。
- **需 GPU：** 否。

### P0-3　论文 Table 3 / Table 5 的 F1、mIoU 无法复现，且混用聚合口径

- **证据**：
  - 论文数字的权威来源已定位为
    `06-实验一四样本代表集/E1_Representative4_PottedPlant_VFM/metrics/summary_metrics.csv`：
    `SAM3_P2 F1=0.969238523974442`、`RAP-FSAM3-v2 F1=0.9706185273336799`，
    **两行 GT 均为** `project LabelMe GT foreground: union of closed target-object linestrips`（旧并集，含蓝块）。
  - Phase 20A 用 5 种 GT 变体重算，**无一种能复现这两个值**：
    | GT 变体 | SAM3_P2 F1 | RAP-FSAM3 F1 |
    |---|---|---|
    | potted_plant | 0.982820 | 0.983717 |
    | potted ∪ cube | 0.967460 | 0.968410 |
    | plant_only | 0.767258 | 0.766073 |
    | E1 `gt_masks/` | 0.967696 | 0.968656 |
    | potted ∧ ¬cube | 0.982834 | 0.983731 |
    | **论文/CSV 报告值** | **0.969239** | **0.970619** |
  - Phase 12/16/17/18 全部使用 `potted_clean`，与 E1 的旧并集**不是同一口径**。
  - Table 5 另有更严重问题：`w/o GCFC` 行的 `Boundary-F1 = 0.4543`、`leakage = 0.001935`
    **全仓无出处**（仅存在于论文）；且该行 F1/mIoU 取自 A5c、HD95 取自 A2 —— 跨聚合混用。
- **下一步**：
  1. 决定最终论文统一口径（建议 `potted_clean`）；
  2. 用 `gt_masks_split_v2` 的 `mask_potted_plant ∧ ¬mask_blue_cube` **离线重算**
     Table 3 / Table 4 / Table 5 的全部指标；
  3. 删除或重做 Table 5 的无出处数字；
  4. 停止使用"all 20 RGB sequences"表述，改为"4 个代表序列共 20 帧"
     （依据：`实验一_视觉基础模型横向对比分割表.csv` 的"样本范围"列）。
- **需 GPU：** 否（候选掩膜已冻结）。

### P0-4　R1 从未作为正式运行执行过

- **证据**：78 份 `参数.json` **无一**含 `ranking_policy` 键（全部早于 R1 提交 `d704faa8`）；
  Phase 17C 是 **zero-inference** 重排序，其 T1 测试以 `policy="base"` 显式重放并复现历史选择（10/10）。
- **影响**：R1 的证据全部是"离线重排序"，不是"实际运行产出"。
- **下一步**：执行 V1（`--ranking_policy r1`，refinement OFF）正式运行，产出最终掩膜。
- **需 GPU：** 是（可 `--reuse_existing_candidates` 省去候选生成）。

### P0-5　R1 × 细化链（SBNP / 残差修复 / GCFC）联合零证据

- **证据**：
  - 78 份运行中，`prompt_selection_mode="single"` 且 `use_spnp_refinement=true` 的组合 **0 例**；
  - `single` 且 `use_corrective_geometry=true` 的组合 **1 例**
    （`阶段十一修复验证/legacy_对照_CaoMei1_0100`，仅 GCFC，且早于 R1 提交）；
  - `ranking_policy` 显式设置且 refinement 开启的组合 **0 例**。
- **静默风险**：历史开启 SPNP 的运行全部 `use_prompt_ensemble=true` + `score_select`；
  若照抄历史命令并追加 `--use_spnp_refinement`，R1 会被 L1682 早退完全绕过，
  而 CSV 的"主评分"列看起来正常，极易误判。
- **下一步**：执行 V2（base+refine）与 V3（r1+refine），计算
  `interaction = (V3−V2) − (V1−V0)`。
- **需 GPU：** 是。

---

## 2. P1 项（投稿前必须处理）

### P1-1　论文 Table 4 把 P2 标为 "entire potted plant"（语义错标）

- 归档 `实验二_提示词鲁棒性四样本汇总表.csv`：P2 = **`entire plant excluding pot`**，F1 = 0.969239。
- 论文数字一致，**标签被改写**。
- 附加事实（Phase 20A 实测）：P2 的盆区召回 ≈97.5%，**与 P6 差 < 0.1 个百分点**——
  "excluding pot" 提示词在本数据集上并未排除花盆。
- **处理**：改正标签；在方法节披露 P2 行为与命名不符；避免在论文中建立"带盆 vs 去盆"的强区分叙事。

### P1-2　论文 SG-MPS 的"温度 softmax"无实现

- `grep softmax|temperature|tau_s` 于生产脚本 → 仅 L1591 的 `q_sam` sigmoid。
- **处理**：改写为加性启发式语义门控（`semantic_total` 加法项），
  或补实现 + 补实验（后者需 GPU）。

### P1-3　论文 FFT 式(1) 与实现不一致

- 实现：`1 − low_energy/total_energy`，平方能量，中心 10% 圆盘作低频，**无 log 压缩**；
  且叠加 **50% Laplacian** 与 **max_consecutive_drop=2** 连续性约束（论文均未提）。
- **处理**：按实现改写方法节；保留"第一个四分位阈值"表述（一致）。

### P1-4　M2M3-GO 的四条公式无实现代码

- `mask_to_model_support` / `model_to_mask_deficit` / `boundary_support` / `utility_score`
  仅存在于 `ForeSplat_real_release/configs/m2m3_go.yaml`；全仓 `.py` **0 命中**。
- 实际实现是 `m2m3_prune_scores` + `capacity_floor_ratio` 的**全局计数下限**机制。
- **处理**：方法节改写为实现机制；或补实现。若保留论文公式，需在 limitations 声明"公式为设计表述"。

### P1-5　Floor40 机制描述不符

- 值 `ρ=0.40` 存在（`arguments/__init__.py:143`），但为**全局**计数下限，
  非论文所述的**局部块**下限；`min_local_retention` 在 `.py` 中 0 命中。
- **处理**：改写描述为全局容量下限（`capacity_floor_ratio`, reference=`max_seen`）。

### P1-6　Table 7 的 wall-clock 时间与"20 序列平均"无出处

- `5044.5 / 12913.7 / 5413.5 / 599.5`（训练）与 `55.0 / 157.9 / 642 / 78`（网格）
  仅出现在论文散文（`ForeSplat_zh.md:208-213`）；全仓 csv/json **无计时记录**。
- "averaged over all 20 RGB sequences"：数据管理下 ForeSplat/A6 输出仅 **4 样本**，
  M2M3 pilot 仅 **3 场景**，网格 `06-MESH-new` 仅 **13 样本**。
- **处理**：重测时间列，或删除该列；把"20 序列平均"改为实际样本数。

### P1-7　Table 6 的网格→性状几何测量无实现

- `生成实验一五当前结果.py` 只从 xlsx 读取**已记录**的虚拟值并算误差统计，
  不执行几何测量；`measure_traits_entry.py:30-33` 明示完整工具未随发布提供。
- **处理**：提供测量脚本或明确标注外部工具 + 人工步骤。

### P1-8　Table 5 "Geometry-only selector" 行是离线重放而非真实运行

- `实验三_RAP-FSAM3模块消融多样本主表.csv` A1 行 status = `已评估-离线重放`，
  说明为"由 RAP-FSAM3-v2 提示词评分离线重放旧 A1：old_score = 总分 − 语义门控总修正"。
- **处理**：论文中标注该行为"离线重放诊断"，不得表述为独立运行实验。

### P1-9　Phase 19A 的两处结论被本审计推翻

| Phase 19A 结论 | 本审计判定 | 证据 |
|---|---|---|
| `P6_raw_baseline` 在 `prompt_list=P2` 下产出（5 处） | **REFUTED** | `P6_raw_baseline/参数.json` 与 `参数/phase12_p6_raw_baseline.json` **双记录**均为 `prompt_list=P6`, `default_prompt_id=P6`, `prompt_ids=["P6"]`；全仓 0 例 `prompt_texts_json` 覆盖 |
| Phase 12 P6 基线"应用了 GCFC"（`Phase19A_algorithm_delta.md` F7） | **REFUTED** | 两份参数记录 `use_corrective_geometry=false`；21/21 帧 `几何修正掩膜` 与 `最终掩膜` **逐字节相同** |
- **处理**：Phase 20B 必须修正 Phase 19A 相关条款，否则新论文会继承错误。

---

## 3. P2 项

### P2-1　D3 / D6 / D12 零帧

- 58 帧冻结挑战集中：53 帧 `D1+D11`，5 帧 `D11+D14`，**D3/D6/D12 各 0 帧**。
- 标签来自 `phase16_split.py` 的 `green_pct >= 20` 单一阈值二分支，**非人工视觉核查**，
  且 D1/D11/D14 **不互斥**。
- `Phase16_report.md` §6 的分布表与冻结清单不可调和（表 TEST 列 19+8+1=28 ≠ 32；D1 合计 16 ≠ 53）。
- **处理**：论文困难场景叙事只围绕 D1 + R1 排序修复；
  补齐 D3/D6/D12 需新的人工标注批次（建议每序列各 1–2 帧）。

### P2-2　挑战集 GT 无 plant/pot 拆分

- 10 帧全为单一 `"potted plant"` 多边形；`pot_status=missing`、`formal_p2_valid=False`。
- **处理**：补标 pot 多边形后方可评估 plant-only 变体。

### P2-3　`gt_masks_split/`（v1 启发式）对蓝块失效

- cube IoU 均值 **0.048**（20/21 帧失败）；`gt_audit_summary.csv` 报 `has_cube` 1/21（真实 19/21）；
  `gt_pending_review.csv` 为空（阈值 0.5，12 帧 `min_confidence=0.6` 未被标记）。
- **处理**：停用 v1，统一使用 `gt_masks_split_v2/`（标签驱动）。

### P2-4　Phase 16 规模未达标

- 协议目标 54 帧 / 15 样本；实际 10 帧 / 2 样本（`gt/DEV`、`gt/TEST` 为空目录）。
- **处理**：论文不得暗示"58 帧困难集已全面标注"。

### P2-5　死代码与命名误导

- `compose_plant_only()` / `compose_potted()` / `detect_semantic_collapse()` / `select_single_object_id()` 定义但从未调用；
- `Candidate` 无 `track_mask` 字段 → `risk_penalty += 0.15` 为死分支。
- **处理**：论文只声明实际生效的机制；或补实现（需 GPU + 实验）。

---

## 4. 下一步行动清单（按依赖排序）

| 顺序 | 行动 | 需 GPU | 阻断的规格升级 |
|---|---|---|---|
| 1 | P0-3 离线重算 Table 3/4/5 到 `potted_clean` | ❌ | 无（论文表可用性） |
| 2 | P0-2 掩膜同一性比较（19 样本） | ❌ | P0-2 |
| 3 | P0-1 找回盆径数值 + 确定尺度传递路径 | ❌ | P0-1 |
| 4 | P1-9 修正 Phase 19A 两处错误结论 | ❌ | 无（文档一致性） |
| 5 | P0-4 V1 正式运行（P6+R1, refine OFF） | ✅ | P0-4 |
| 6 | P0-5 V2 / V3 正式运行 + interaction 计算 | ✅ | P0-5 |
| 7 | 独立验证集标注（TEST32 32 帧，含 pot 多边形） | ❌（人工） | 独立验证 |
| 8 | P1-1…P1-8 论文表述修正 | ❌ | 投稿就绪 |
| 9 | 依 V0–V3 与独立验证结果，将 `FINAL_RAP_FSAM3_SPEC_DRAFT.md` 升级为 `LOCKED_CANDIDATE` | — | 规格锁定 |

**本阶段到此为止，不进入 Phase 20B。**