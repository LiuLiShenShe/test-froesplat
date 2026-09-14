# Phase 19A — WP5 排名公式审计（R1 Ranking-Formula Audit）

> 来源代码（本次审计逐行核对，全部为当前工作区实际代码）：
> - `生成RAP-FSAM3掩膜.py`（生产脚本，3381 行）
> - `ranking_policy.py`（Phase 17C 排序策略单一事实源）
> - `Phase18_final_algorithm_spec.md`（阶段十八最终算法规格，基线 commit `d704faa8`）
>
> 审计基准 commit：`d704faa8`（Phase 17C 最终提交，即 S20 生产默认）。

---

## 1. Purpose / 目的

本审计为论文重构（Phase 19A）消除三类歧义，给出可直接写入手稿的精确符号表：

1. **P2 vs P6 歧义** — 论文中 "P6" 一词同时指代 (a) 生产 prompt 集合 `P6`（带盆整株提示词）与 (b) 阶段十二冻结基线运行名 `P6_raw_baseline`。二者是不同的东西，必须区分。
2. **惩罚符号歧义** — 规格文中 `penalty = 0.25` 的记法易被读作"加惩罚值"；实际实现是**从主评分中减去一个正风险惩罚**（`risk_penalty`，正值相减）。
3. **分母歧义** — `base_total` 的分母 `denom` 是权重和（默认 5.5），R1 主评分在与 `base_total` 相同的分母上相除。论文公式必须保留 `denom` 显式项，否则 `primary = base_total − q_contrast/5.5 − penalty` 的推导无法成立。

所有结论均标注精确 `文件:行号`，可直接核对。

---

## 2. Symbol Table / 符号表

### 2.1 特征分 q_*（每个候选掩膜计算一次）

| 符号 | 代码变量 | 精确实现 | 论文记号（建议） | 来源 |
|---|---|---|---|---|
| `q_area` | `q_area` | `area_quality(area_ratio, args)` — 面积比相对 `[area_min_ratio, area_max_ratio]` 的三角形合理性分 | `q_area` | `生成RAP-FSAM3掩膜.py:1581`（函数 `area_quality` 调用；三角映射定义见 `:1212-1215`） |
| `q_comp` | `q_comp` | `1.0 / (1.0 + max(comp_count - 1, 0))` — 单连通域得满分，多连通域递减 | `q_comp` | `生成RAP-FSAM3掩膜.py:1582` |
| `q_edge` | `q_edge` | `edge_quality(mask, args)` — 边界贴合度（惩罚贴图框边缘），`border_band=2px` | `q_edge` | `生成RAP-FSAM3掩膜.py:1584` |
| `q_temp` | `q_temp` | `temporal_iou`（当 `use_temporal_alignment=True`），否则**固定 0.5**（中性值，时序门控未配准时不禁用） | `q_temp` | `生成RAP-FSAM3掩膜.py:1586`；门控默认关 `:664-665` |
| `q_contrast` | `q_contrast` | `min(1.0, contrast / 0.25)` — 前景-背景 L2 对比度归一化（0.25 为饱和点） | `q_contrast` | `生成RAP-FSAM3掩膜.py:1587` |
| `q_leak` | `q_leak` | `leakage_quality(bottom_leak_fraction, args)` — 底部纵向溢出质量分 | `q_leak` | `生成RAP-FSAM3掩膜.py:1588` |
| `q_side` | `q_side` | `side_leakage_quality(side_leak_fraction, args)` — 侧边盆沿侵入质量分 | `q_side` | `生成RAP-FSAM3掩膜.py:1589` |
| `q_sam` | `q_sam` | `1.0 / (1.0 + exp(-6.0 * (sam_score - 0.5)))` — SAM3 实例置信度 logit 的 sigmoid 归一化 | `q_sam` | `生成RAP-FSAM3掩膜.py:1591` |

### 2.2 权重 w_* 与分母

| 符号 | 代码变量 | 精确实现 | 论文记号（建议） | 来源 |
|---|---|---|---|---|
| `w_area` | `weights["area"]` | 默认 `1.0`（来自 `--score_weights "area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5"`） | `w_area` | 默认串 `:655-656`；解析 `parse_key_values` `:806-817`；调用点 `:2777-2780` |
| `w_comp` | `weights["comp"]` | 默认 `1.0` | `w_comp` | 同上 |
| `w_edge` | `weights["edge"]` | 默认 `1.0` | `w_edge` | 同上 |
| `w_temp` | `weights["temp"]` | 默认 `1.0` | `w_temp` | 同上 |
| `w_contrast` | `weights["contrast"]` | 默认 `1.0` | `w_contrast` | 同上 |
| `w_leak` | `weights["leak"]` | **默认 0.0**（不在默认串中，解析仅当显式传入时出现） | `w_leak` | 解析 `:806-817`；调用点 `:2777-2780` |
| `w_side` | `weights["side"]` | **默认 0.0**（同上） | `w_side` | 同上 |
| `w_sam` | `weights["sam"]` | 默认 `0.5` | `w_sam` | 默认串 `:655`；解析 `:806-817` |
| `denom` | `denom` | `max(sum(weights.values()), 1e-6)`；默认 `1+1+1+1+1+0.5 = 5.5`（冻结） | `D` | `生成RAP-FSAM3掩膜.py:1592`；规格 `Phase18_final_algorithm_spec.md:45` |
| `base_total` | `base_total` | `(Σ w_k·q_k + w_sam·q_sam) / denom`，k∈{area,comp,edge,temp,contrast,leak,side}；默认下 leak/side 权重 0 | `B` | `生成RAP-FSAM3掩膜.py:1593-1602` |

### 2.3 修正项（semantic / penalty）

| 符号 | 代码变量 | 精确实现 | 论文记号（建议） | 来源 |
|---|---|---|---|---|
| `semantic_total` | `semantic["semantic_total"]` | 默认 **0.0**（`--use_semantic_gate` 默认 False）；仅当门控开启且 `semantic_context` 非空时由 `semantic_gate_scores()` 计算（target_box/垂直覆盖/中心先验 − pot_overlap − side_distractor 加权和） | `S` | 默认 `:683`；占位 0.0 `:1615-1622`；计算 `:1623-1624`；`semantic_gate_scores` 计算式 `:1495-1502` |
| `penalty`（垂直覆盖） | `risk_penalty` | `if vertical_coverage < vertical_coverage_min_ratio(0.30): risk_penalty += 0.25` — **+0.25 是正量，随后从主评分中减去** | `P_v = 0.25` | `生成RAP-FSAM3掩膜.py:1604-1610`；阈值 `:667-668` |
| `penalty`（track 重叠） | `risk_penalty` | `if track_mask available and mask_iou(mask, track_mask) < box_track_overlap_min(0.20): risk_penalty += 0.15` | `P_t = 0.15` | `生成RAP-FSAM3掩膜.py:1611-1614`；阈值 `:669-670` |
| `risk_penalty`（汇总） | `risk_penalty` | `P = P_v + P_t`，取值集合 {0.0, 0.15, 0.25, 0.40}；存入 `ScoreRecord.leak_penalty` | `P` | `:1604-1614`、`leak_penalty=float(risk_penalty)` `:1667` |

> **命名提醒**：`ScoreRecord.leak_penalty` 字段实际存的是**风险惩罚汇总** `risk_penalty`（垂直覆盖 + track 重叠），不是底部泄漏分；底部泄漏分是 `q_leak` 特征。手稿不要用字段名直译。

### 2.4 主评分与平局

| 符号 | 代码变量 | 精确实现 | 论文记号（建议） | 来源 |
|---|---|---|---|---|
| `primary_score` | `primary_score` | R1 下 `(base_total·denom − w_contrast·q_contrast)/denom + semantic_total − penalty` | `F` | `ranking_policy.py:44-48`；调用点 `生成RAP-FSAM3掩膜.py:1628-1637` |
| `tie_break` | `tie_break_score` | R1 下固定 `0.0`（无字典序平局决胜；R2 才用 `q_contrast`） | `T` | `ranking_policy.py:47` |
| 选择规则 | `select_best_policy` | `(primary_score, tie_break_score)` 字典序取最大，`max` 保持首见平局 | — | `ranking_policy.py:57-68`；调用点 `:1725` |
| 旧 total（仅诊断） | `total_score` | `base_total + semantic_total − risk_penalty`（R1 下**不再用于选择**） | `T_old` | `生成RAP-FSAM3掩膜.py:1625`；`:1643` |
| 空掩膜硬门 | `empty_flag` | `not mask.any()` → `total=0.0, primary=0.0, tie=0.0, q_*=0.0` | — | `:1550-1569` |

### 2.5 超参与流程参数

| 符号 | 代码变量 | 默认值 | 论文记号（建议） | 来源 |
|---|---|---|---|---|
| 提示词集合 | `--prompt_list` | `"P2"` | `P2`（生产默认 prompt 集合） | `生成RAP-FSAM3掩膜.py:645` |
| 默认提示词 | `--default_prompt_id` | `"P2"` | `P2` | `:647` |
| 候选模式 | `--candidate_mode` | `"per_instance"`（choices: legacy_union / per_instance） | per-instance | `:628-635` |
| SAM3 掩膜阈值 | `--sam3_mask_threshold` | `0.5` | `τ_mask = 0.5` | `:636-639` |
| SAM3 置信度阈值 | `--confidence_threshold` | `0.3` | `τ_conf = 0.3` | `:622` |
| 垂直覆盖门限 | `--vertical_coverage_min_ratio` | `0.30` | `V_min = 0.30` | `:667-668` |
| track 重叠门限 | `--box_track_overlap_min` | `0.20` | `O_min = 0.20` | `:669-670` |
| 面积下限 | `--area_min_ratio` | `0.01` | `A_min = 0.01` | `:671` |
| 面积上限 | `--area_max_ratio` | `0.80` | `A_max = 0.80` | `:672` |
| 排名策略 | `--ranking_policy` | `"r1"`（choices: base/r1/r2/r3） | R1 | `:657-662` |
| 语义门控 | `--use_semantic_gate` | `False`（store_true） | 门控默认关 | `:683` |
| A6 跨视角共识 | `--use_cross_view_consensus` | `False`（store_true） | A6 默认关 | `:775` |
| A7 记忆传播 | `--use_memory_propagation` | `False`（store_true） | A7 默认关 | `:792` |

### 2.6 P2/P6 生产 prompt 文本（BUILTIN_PROMPTS）

| ID | 英文文本 | 中文目录名 | 来源 |
|---|---|---|---|
| `P2` | `"entire plant excluding pot"` | `P2_整株去花盆` | `:39`, `:51` |
| `P6` | `"whole potted plant including the pot"` | `P6_带盆整株` | `:43`, `:55` |

---

## 3. R1 主评分推导（R1 Primary Score Derivation）

### 3.1 默认配置下的精确代数

**第 1 步 — base_total**（`生成RAP-FSAM3掩膜.py:1593-1602`）：

```
B = (w_area·q_area + w_comp·q_comp + w_edge·q_edge + w_temp·q_temp
     + w_contrast·q_contrast + w_leak·q_leak + w_side·q_side + w_sam·q_sam) / denom
```

默认 `w_leak = w_side = 0`，故默认下：

```
B = (q_area + q_comp + q_edge + q_temp + q_contrast + 0.5·q_sam) / 5.5
```

**第 2 步 — R1 主评分**（`ranking_policy.py:44-48`，r1 分支）：

```
F = (B·denom − w_contrast·q_contrast) / denom + S − P
```

代入默认 `w_contrast = 1.0`、`denom = 5.5`、`S = 0`：

```
F = (5.5·B − q_contrast) / 5.5 + 0 − P
  = B − q_contrast/5.5 − P
```

**这就是论文应使用的公式**（`Phase18_final_algorithm_spec.md:113` 一致）：

```
primary_score = base_total − q_contrast / 5.5 − penalty
```

### 3.2 该公式如何"剔除"对比项

`q_contrast` 在 `B` 中占 `1.0·q_contrast`（即 `q_contrast/5.5`）。R1 从分子中减去 `w_contrast·q_contrast` 后再除以 `denom`，等价于从归一化总和中**精确移除对比项的贡献 `q_contrast/5.5`**，其余五项与 SAM 项权重不变、`denom` 保持冻结 5.5（不重归一化到新权重和 4.5）。因此：

- 对比项完全不参与 R1 排序；
- 其余特征的相对权重不变（仍是各自 /5.5）；
- `q_contrast` 仍保留在 `ScoreRecord` 中（字段 `q_contrast`、`contrast`、`contrast_effective = q_contrast·q_area`、`sam_scores`）仅作诊断，不参与选择（`生成RAP-FSAM3掩膜.py:1638, 1648, 1655, 1671`；规格 `:133`）。

### 3.3 必须显式保留 `denom` 的原因

若论文把公式写成 `F = B − q_contrast − P`（去掉 `/5.5`），会错误地把对比项当作 1.0 整值扣减。正确写法依赖 `denom = 5.5` 的显式分母：

```
F = (Σ_weights w_i·q_i + w_sam·q_sam − w_contrast·q_contrast)/5.5 + S − P
  = B − q_contrast/5.5 − P          （默认 w_contrast = 1.0）
```

权重和（denom）在**所有**策略（base/r1/r2/r3）下保持冻结 5.5（`ranking_policy.py:18`），不随策略改变。

> 保守标注（供手稿校对）：`denom = 5.5` 成立的前提是 `--score_weights` 取默认串；若实验显式传入非零 `leak`/`side` 权重，denom 将不再是 5.5（`parse_key_values` 会把这些键加入权重字典，`:806-817`）。生产默认与 Phase 12 冻结基线均为 5.5。

### 3.4 策略间关系（r2/r3 仅作对照，非生产默认）

| 策略 | primary_score | tie_break | 代码 |
|---|---|---|---|
| base | `B + S − P` | `0.0` | `ranking_policy.py:42-43` |
| **r1（默认）** | `(B·denom − w_c·q_c)/denom + S − P` | `0.0` | `:44-47` |
| r2 | 同 r1 | `q_contrast` | `:47` |
| r3 | `(B·denom − w_c·q_c + w_c·q_c·q_area)/denom + S − P` | `0.0` | `:49-53` |

---

## 4. P2 vs P6 解析（P2/P6 Resolution）

论文手稿中 "P6" 一词存在**两义**，本审计给出区分规则：

| 指代 | 是什么 | 生产状态 | 证据 |
|---|---|---|---|
| **P2（生产默认 prompt 集合）** | `BUILTIN_PROMPTS["P2"] = "entire plant excluding pot"`（整株去花盆） | 生产默认：`--prompt_list "P2"`、`--default_prompt_id "P2"`、`--prompt_selection_mode "single"`（单 prompt 分支）；候选生成在该 prompt 下逐实例评分选择 | 文本 `:39`、目录名 `:51`、默认 `:645, :647, :649-653` |
| **P6（生产 prompt 集合）** | `BUILTIN_PROMPTS["P6"] = "whole potted plant including the pot"`（带盆整株） | 生产 prompt 字典成员；阶段十一 §5 起作为独立语义候选（与 P2 不再共用终选掩膜）；`P2/P6 语义坍缩检测` 用于当两者面积/盆区覆盖几乎相同时受控组合 | 文本 `:43`、独立候选 §5 `:1743-1750`、坍缩判定 `:675`、受控组合 `:1797` |
| **P6_raw_baseline（阶段十二冻结基线运行名）** | **不是生产 prompt 集合**，而是阶段十二历史冻结的运行/基线标识符（其指标 `f1_mean=0.983498` 在**生产有效配置**下产出，即默认 `prompt_list=P2` + `ranking_policy` 下当时的排序行为） | 历史基准，仅作对照数字；当前生产默认排序为 R1 | 阶段十八文档与 Phase 12 存档；`f1_mean=0.983498` 为冻结基线指标 |

**论文必须遵守的写法**：

1. 论文正文每次使用 "P6" 时，明确限定为**生产 prompt 集合 P6**（带盆整株提示词，`P6 = "whole potted plant including the pot"`），并注明它只在 `--prompt_list` 显式包含 P6 时参与生成（默认只含 P2）。
2. "P6 基线指标 f1_mean=0.983498" 的出处应改写为 "阶段十二冻结基线（代号 `P6_raw_baseline`，在该配置下以默认 P2 prompt 集合运行产出）"——**避免让读者误以为该指标来自 P6 prompt 集合**。
3. 若论文报告 P6 prompt 集合的消融，应使用 `--prompt_list P6`（或 `P2,P6`）的实际运行结果，而非阶段十二基线数字。
4. 该区分的代码级事实：`parse_prompt_ids()`（`生成RAP-FSAM3掩膜.py:842-845`）按逗号切分 `--prompt_list`；默认值 `"P2"` 意味着生产运行**只有一个 prompt 分支（P2）**，`default_prompt_id` 保证 P2 一定在列表中（`:2745-2747`）。

---

## 5. 惩罚符号解析（Penalty Sign Resolution）

**结论：惩罚是"正量、随后从主评分中减去"（风险惩罚），不是负值相加。**

代码事实链（`生成RAP-FSAM3掩膜.py`）：

```
风险惩罚是正值累加（+=）：
  risk_penalty = 0.0                                    :1604
  if vertical_coverage < 0.30: risk_penalty += 0.25     :1609-1610
  if track_mask ... IoU < 0.20: risk_penalty += 0.15    :1612-1614

修正发生在主评分计算时（减法，负号在公式中）：
  total = base_total + semantic_total - risk_penalty    :1625
  primary_and_tiebreak(..., penalty=risk_penalty, ...)  :1628-1637

ranking_policy.py 中 penalty 被相减：
  r1: primary = (base_total·denom − w_c·q_c)/denom + semantic_total − penalty   :44-48
```

因此：

- **规格文记法** `penalty = 0.25 if vertical_coverage < 0.30` 的含义是：当垂直覆盖低于门限时，**产生 +0.25 的正风险量，该量在 `primary_score` 中作为减项**。
- 与直观相反，"penalty 为 0.25" 不是给评分加上 0.25，而是从评分中**扣去 0.25**。
- `risk_penalty` 的取值集合：`{0.0, 0.15, 0.25, 0.40}`（两项独立触发，track 重叠惩罚仅在候选携带有效 `track_mask` 且非空时评估，`candidate.track_mask`，`:1611-1612`）。
- 空掩膜候选（`empty_flag=True`）直接返回 `primary_score=0.0`，不进入惩罚评估（`:1551-1569`）。
- 注意不要把 `risk_penalty` 与特征 `q_leak`/`q_side`（权重 0，不进入 `base_total` 的贡献）混为一谈：`q_leak`/`q_side` 是结构质量分，`risk_penalty` 是硬门控风险惩罚，二者是不同的机制。`ScoreRecord.leak_penalty` 字段名（`:1667`）存的是后者，命名易混淆，手稿应按语义表述为"风险惩罚（垂直覆盖/track 重叠）"。

**论文公式书写建议**（与实现符号一致）：

```
F = B − q_contrast/5.5 − P,    P = 0.25·1[V < 0.30] + 0.15·1[track_mask 有效 且 IoU < 0.20]
```

其中 `1[·]` 为指示函数；track 项仅当 track_mask 存在且非空时评估。

---

## 6. A6/A7 公式地位（A6/A7 Formula Status）

**结论：A6（跨视角共识）与 A7（记忆传播）在评分公式中无任何项；二者是"选择后"的实验分支，默认关闭。**

| 模块 | 标志 | 默认 | 作用位置 | 是否进入评分公式 |
|---|---|---|---|---|
| A6 跨视角共识 | `--use_cross_view_consensus`（store_true） | `False` | 在每样本**选出的掩膜集合**上执行共识细化（`apply_cross_view_consensus`，`生成RAP-FSAM3掩膜.py:234-407`），由主循环 `:2956` 调用 | **否**。不修改 `score_candidate` 的任何项；无 `q`/`w`/`penalty` 贡献 |
| A7 记忆传播 | `--use_memory_propagation`（store_true） | `False` | 在**已选定种子帧掩膜**基础上调用 SAM3 记忆引擎做双向传播，产出替代掩膜集（`propagate_memory_masks`，`:419-566`），由 `:3011-3062` 调用 | **否**。作用于掩膜输出，不进入候选评分 |

代码证据：

- 两个标志都是 `action="store_true"`，默认 `False`：`:775`（A6）、`:792`（A7）。
- `score_candidate()`（`:1540-1672`）内部**不读取** `use_cross_view_consensus` / `use_memory_propagation`；评分公式只由 `--score_weights`、`--ranking_policy`、`--vertical_coverage_min_ratio`、`--box_track_overlap_min`、`--use_semantic_gate` 决定。
- A6 在每样本选择循环之后、以**已选掩膜**为输入（`:2956` 附近），A7 以 `base_for_memory`（已选掩膜）为种子（`:3011-3015`）——均为后处理分支。
- Phase18 规格一致：`Component 5`（A6 DEMOTED / DEFAULT OFF，`Phase18_final_algorithm_spec.md:139-153`）、`Component 6`（A7 DEMOTED / DEFAULT OFF，`:156-169`）、`Component 7`（A6+A7 组合 DEMOTED，`:173-184`）；方法流程图将 A6/A7 标为 "Optional — not in production default"（`:215-218`）。

**论文应表述**："A6（跨视角共识）与 A7（记忆传播）作为独立的后处理/替代掩膜分支存在，默认关闭；它们不参与候选评分公式，评分公式（R1）在是否启用 A6/A7 时保持逐位一致。"

---

## 7. 附录：核对清单（Audit Checklist）

本次审计对前置审计给出的"已验证事实"逐条重核，全部与当前代码一致：

- `parse_args()` 位于 `生成RAP-FSAM3掩膜.py:613`；`--ranking_policy` 默认 `"r1"` `:657-659`；`--score_weights` 默认串 `:655`；`--prompt_list "P2"` `:645`；`--candidate_mode "per_instance"` `:628-631`；`--sam3_mask_threshold 0.5` `:637`；`--vertical_coverage_min_ratio 0.30` `:667`；`--box_track_overlap_min 0.20` `:669`。
- `base_total` 在 `score_candidate()` 内，`:1593-1602`，`denom = max(sum(weights.values()), 1e-6)`（`:1592`），默认 5.5；leak/side 默认权重 0.0。
- 风险惩罚 `:1604-1614`：+0.25（垂直覆盖 < 0.30），+0.15（track IoU < 0.20，仅当 track_mask 有效）。
- `ranking_policy.py` `primary_and_tiebreak` `:26-54`：base `:42-43`；r1 `:44-47`（tie=0.0）；r2 `:47`（tie=q_contrast）；r3 `:49-53`。
- `semantic_total` 默认 0.0（`--use_semantic_gate` store_true 默认 False，`:683`）。
- P2 = 生产默认 prompt 集合；P6 = 生产 prompt 字典成员（带盆整株）；P6_raw_baseline = 阶段十二冻结基线运行名（指标 f1_mean=0.983498 在**生产有效配置**下产出）。

**审计边界**：本审计只核对 `生成RAP-FSAM3掩膜.py`（生产脚本）、`ranking_policy.py`、`Phase18_final_algorithm_spec.md` 三个文件；未修改任何现有文件。
