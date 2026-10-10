# R1 公式精确复核 / R1 Formula Verification

**阶段：** Phase 20A
**复核对象：** 生产代码最终规则（非论文草稿推导）
**生产代码：** `00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本/ranking_policy.py`
**集成代码：** 同目录 `生成RAP-FSAM3掩膜.py`
**审计基线 commit：** `6df0c3df`（HEAD，Phase 19A）；R1 引入 commit：`d704faa8`（Phase 17C）

---

## 0. 结论摘要

| 复核项 | 结论 | 状态 |
|---|---|---|
| 分母是否仍为 5.5 | 是。冻结全权重和 = 1+1+1+1+1+0.5 = 5.5 | VERIFIED |
| `q_contrast` 是否只被移除一次 | 是。仅在 `primary_and_tiebreak()` 中移除一次 | VERIFIED |
| 风险惩罚是否正确相减 | 是。`− risk_penalty`，正值相减 | VERIFIED |
| `q_temp` 无时序对齐时是否中性 | 是。固定 0.5 | VERIFIED |
| `semantic_total` 门控关闭时是否为零 | 是。恒为 0.0 | VERIFIED |
| R1 与 R0 在相同候选集上的唯一差异 | 主评分中扣除 `w_contrast·q_contrast/denom` | VERIFIED |
| 是否需要重新选权重 | **不需要。R1 不改任何权重，只改项的取舍** | VERIFIED |

---

## 1. 生产代码原文（唯一权威来源）

### 1.1 `ranking_policy.py` 全文（68 行，Phase 20A 逐行核对）

```python
POLICY_NAMES = ("base", "r1", "r2", "r3")          # L23

def primary_and_tiebreak(base_total, q_contrast, q_area,
                         w_contrast, denom,
                         semantic_total, penalty, policy):
    if policy == "base":                            # L42-43
        return base_total + semantic_total - penalty, 0.0
    if policy in ("r1", "r2"):                      # L44-48
        primary = (base_total * denom - w_contrast * q_contrast) / denom \
            + semantic_total - penalty
        tie = q_contrast if policy == "r2" else 0.0
        return primary, tie
    if policy == "r3":                              # L49-53
        primary = (base_total * denom - w_contrast * q_contrast
                   + w_contrast * q_contrast * q_area) / denom \
            + semantic_total - penalty
        return primary, 0.0
    raise ValueError(...)                           # L54

def select_best_policy(score_records, policy):      # L57-68
    return max(score_records,
               key=lambda r: (getattr(r, "primary_score", 0.0),
                              getattr(r, "tie_break_score", 0.0)))
```

### 1.2 R1 的最终形式化简

设 `w_contrast = weights["contrast"] = 1.0`，`denom = 5.5`：

```
primary_score(r1)
  = (base_total·denom − 1.0·q_contrast)/denom + semantic_total − penalty
  = base_total − q_contrast/5.5 + semantic_total − penalty
```

在生产默认配置下 `semantic_total = 0.0`，因此：

```
primary_score(r1) = base_total − q_contrast/5.5 − penalty
```

**与历史审计给出的形式完全一致**（`primary_score = base_total − q_contrast/5.5 − risk_penalty`）。

对照 `base`（= R0）：`primary_score(base) = base_total + 0 − penalty = total_score`。

---

## 2. 分子 / 分母逐项复核（`score_candidate()`，`生成RAP-FSAM3掩膜.py`）

### 2.1 分母

```python
denom = max(sum(weights.values()), 1e-6)            # L1592
```

生产默认 `--score_weights "area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5"`（L655），
未提供 `leak` / `side` 键 → `parse_key_values()`（L2777-2780）以默认值初始化后仅覆盖给定键，
`leak`/`side` 保持 **缺失** → `weights.get("leak", 0.0)` = 0.0（L1599-1600）。

```
denom = 1(area) + 1(comp) + 1(edge) + 1(temp) + 1(contrast) + 0(leak) + 0(side) + 0.5(sam) = 5.5
```

**复核结论：分母 = 5.5，且在所有策略下不变**（`ranking_policy.py` L18 文档字符串亦如此声明）。
`leak` / `side` 权重为 0 只影响分子项，不影响分母——分母是全权重和，不是实际激活权重和。**这是设计上的冻结口径，Phase 20A 不做修改。**

### 2.2 分子

```python
base_total = (w_area·q_area + w_comp·q_comp + w_edge·q_edge
              + w_temp·q_temp + w_contrast·q_contrast
              + w_leak·q_leak + w_side·q_side
              + w_sam·q_sam) / denom                  # L1593-1602
```

| 分量 | 计算式 | 代码行 | 生产默认权重 |
|---|---|---|---|
| `q_area` | `area_quality(area_ratio, args)` 三角合理性 | L1581, 1211 | 1.0 |
| `q_comp` | `1/(1+max(comp_count-1,0))` | L1582 | 1.0 |
| `q_edge` | `edge_quality(mask, args)` 边界贴合度 | L1584, 1227 | 1.0 |
| `q_temp` | 无时序对齐 → **固定 0.5** | **L1586** | 1.0 |
| `q_contrast` | `min(1.0, contrast/0.25)` | L1587 | 1.0 |
| `q_leak` | `leakage_quality(bottom_fraction)` | L1588 | **0.0** |
| `q_side` | `side_leakage_quality(side_fraction)` | L1589 | **0.0** |
| `q_sam` | `1/(1+exp(-6(sam_score-0.5)))` | L1591 | 0.5 |

### 2.3 `q_temp` 中性值复核 —— **CONFIRMED**

```python
temporal_iou = mask_iou(mask, prev) if (prev is not None
                and getattr(args, "use_temporal_alignment", False)) else 0.5   # L1579
q_temp = temporal_iou if getattr(args, "use_temporal_alignment", False) else 0.5  # L1586
```

`--use_temporal_alignment` 为 `store_true`（L664），生产默认关闭。
所有生产运行 `参数.json` 记录 `use_temporal_alignment = False`。

**结论：`q_temp` 在无时序对齐时恒为 0.5（中性），不是满分。此项 Phase 11 §4.5 的修正已在代码中生效。**

### 2.4 `semantic_total` 零值复核 —— **CONFIRMED**

```python
semantic = {"semantic_total": 0.0, ...}             # L1615-1622
if args.use_semantic_gate and semantic_context is not None:   # L1623
    semantic = semantic_gate_scores(mask, semantic_context, args)
```

`--use_semantic_gate` 为 `store_true`（L683），生产默认关闭；
Phase 12 `P6_raw_baseline` 与 Phase 17A 全部记录 `use_semantic_gate = False`。

**结论：门控关闭时 `semantic_total ≡ 0.0`，对主评分无贡献（不是隐式贡献）。**

### 2.5 风险惩罚复核 —— **CONFIRMED**

```python
risk_penalty = 0.0
upper_coverage = float(ys.min() / max(h,1)) if len(ys) else 1.0   # L1607
vertical_coverage = 1.0 - upper_coverage                          # L1608
if vertical_coverage < 0.30:  risk_penalty += 0.25                # L1609-1610
if track_mask is not None and track_mask.any():
    if mask_iou(mask, track_mask) < 0.20: risk_penalty += 0.15    # L1611-1614
total = base_total + semantic["semantic_total"] - risk_penalty     # L1625
```

- 惩罚项为 **正值**，在 `total` 与 `primary_and_tiebreak()` 中均以 **减法** 施加 → 符号正确。
- `0.25` 分支：`vertical_coverage_min_ratio` 默认 0.30（L667）。
- `0.15` 分支：`box_track_overlap_min` 默认 0.20（L669）。
  **注意：** 实践证追踪重叠惩罚需要 `candidate.track_mask` 非空。
  在 `per_instance` 生产路径中 `Candidate.track_mask` 未被赋值（字段存在于 L61 的注释区，
  但 `infer_candidates()` 未设置），故 **该 0.15 分支在生产默认下实际不触发**。
  这是**已识别的死分支**，记录于 `P0_BLOCKERS_AND_NEXT_ACTIONS.md`，本阶段不修改。

---

## 3. 数值复核（在真实归档产物上重算）

### 3.1 数据源

`阶段十七C_候选排序修复开发/Phase17C_ranking_benchmark.csv`（182 条候选记录，
来自 Phase 17A `P00_Control` 与 Phase 14 `V00_Control` 的 `提示词评分.csv`，commit `d704faa8`）。

### 3.2 重算结果

```
denom = 5.5
rows  = 182
以 base_total 公式重算并与记录 total_score 比较：
    |diff| > 1e-6 的记录数 = 1
    最大 |diff| = 0.24999998
```

**该唯一 1 条偏差 = 恰好 0.25**，即触发了 `vertical_coverage < 0.30` 的纵向覆盖硬门控惩罚。
这独立验证了两件事：
1. `base_total` 分子/分母口径与生产代码逐项一致；
2. `penalty = 0.25` 分支在真实数据上确实被触发（而非仅存在于代码中）。

### 3.3 R0 vs R1 逐帧重算（按 `(source, sample, frame)` 正确分组，51 组）

| 帧组 | R0 选中 | R0 F1 | R1 选中 | R1 F1 | 变化 |
|---|---|---|---|---|---|
| DouBanLv2 0039 | `DouBanLv2_0039_00` | **0.0000** | `DouBanLv2_0039_01` | **0.9867** | **RESCUED** |
| DouBanLv2 0040 | `DouBanLv2_0040_01` | **0.0000** | `DouBanLv2_0040_00` | **0.9882** | **RESCUED** |
| DouBanLv2 0041 | `DouBanLv2_0041_00` | **0.0000** | `DouBanLv2_0041_03` | **0.9865** | **RESCUED** |
| BaiZhang 0033–0037 (5 帧) | — | 0.9536–0.9623 | 同 R0 | 同 R0 | 无变化 |
| DouBanLv2 0037–0038 (2 帧) | — | 0.9855 / 0.9898 | 同 R0 | 同 R0 | 无变化 |
| Easy21 全 21 帧 | — | 0.9761–0.9891 | 同 R0 | 同 R0 | **0 变化** |
| DouBanLv2 0042–0046（上下文帧，无 GT） | 5 帧选择改变 | — | — | 无 GT，不可评价 | — |

总 churn：8/51 组（其中 5 组为无 GT 的 DouBanLv2 上下文帧）。
**Phase 20A 独立重算与 Phase 17C 报告（3/3 rescue，Easy21 0 churn）在数值上一致。**

---

## 4. `q_contrast` 的最终地位

| 用途 | 代码位置 | 是否仍在链路中 |
|---|---|---|
| 仍计入 `base_total` / `total_score` | L1598, L1625 | **是**（诊断口径） |
| 从 R1 主评分中扣除 | `ranking_policy.py` L45-46 | **是（唯一一次扣除）** |
| 保留为 `ScoreRecord.q_contrast` 字段 | L1650 | **是** |
| 保留为 `contrast_effective = q_contrast × q_area` | L1638, L1671 | **是** |
| 保留在 `提示词评分.csv` 的 `总分`/`主评分` 双列 | L2611, L2635 | **是** |

**结论：`q_contrast` 未被删除，只从"主排序信号"降级为"诊断信号"。
删除一次、且仅在 `primary_and_tiebreak()` 内部删除一次 —— 无重复扣除、无遗漏。**

---

## 5. 与 Phase 18《最终算法定义》的逐条比对

| Phase 18 表述 | 生产代码核对 | 结论 |
|---|---|---|
| `primary_score = base_total − q_contrast/5.5 − penalty` | L45-46 + L1592 | **一致** |
| `denom = 1+1+1+1+1+0.5 = 5.5` | L1592 + 默认 `score_weights` | **一致** |
| `penalty = 0.25 if vertical_coverage < 0.30` | L1609-1610 | **一致**（另有 0.15 分支，Phase 18 未提但存在） |
| `semantic_total = 0.0`（门控默认关） | L1616, L1623 | **一致** |
| `q_temp` 默认恒为 0.5 | L1586 | **一致** |
| `q_leak` / `q_side` 权重 0.0，仅诊断 | L1599-1600 | **一致** |
| R1 默认 `--ranking_policy r1` | L657-662 | **一致** |
| 选择为 `(primary_score, tie_break_score)` 字典序 | L65-68 | **一致** |

---

## 6. 本阶段未做的事（边界声明）

- 未修改 `ranking_policy.py` 或 `生成RAP-FSAM3掩膜.py`。
- 未重新选择或调整任何 R1 权重。
- 未推导任何新公式 —— 本文件所有公式均直接摘录自生产代码行号。
- 未把 GT10 当作独立测试集；本节所有 GT10 数字均标注为 DEV/诊断用途。
- TEST32 无 GT 帧未计算任何准确率。

---

## 7. 复核方法与可复现命令

```bash
# 逐行读生产规则
sed -n '1,68p' \
  00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本/ranking_policy.py

# 分子/分母/惩罚/中性值
sed -n '1540,1675p' \
  00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py

# 非 GPU 测试复跑（Phase 20A 实测 150/150 PASS）
cd 00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证
python3 -I -m pytest $(find . -name "test_*.py" -path "*/tests/*") -q
```

**Phase 20A 实测：`150 passed in 74.57s`**（Phase 17C 当时报告 142/142；
差额来自 Phase 17C 之后新增的 Phase 18 与 Phase 12 hotfix 测试）。