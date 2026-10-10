# R1 调用路径审计 / R1 Call-Path Audit

**阶段：** Phase 20A
**审计问题：** R1 是否已经正确融入原 RAP-FSAM3？在**全部真实使用过的**选择分支里是否生效？旧多提示词模式是否绕过 R1？
**权威代码：** `00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py`（HEAD `6df0c3df`）

---

## 0. 结论（先看这里）

| 问题 | 结论 | 严重度 |
|---|---|---|
| R1 是否融入 RAP-FSAM3 生产脚本？ | **是**。已作为 `--ranking_policy` 默认值接入 `score_candidate()` 与 `select_mask()`。 | — |
| R1 在 `single` + `per_instance` 多实例分支是否生效？ | **是**。`select_mask()` L1725 调用 `select_best_policy()`，按 `(primary_score, tie_break_score)` 字典序选。 | — |
| R1 在 `score_select`（多提示词）是否生效？ | **否 —— 绕过 R1**。L1688 直接用 `row.total_score`。 | **HIGH（论文方案 B 的核心风险）** |
| R1 在 `weighted_fusion` 是否生效？ | **否 —— 绕过 R1**。L1699-1704 用 `row.total_score` 计算融合权重。 | **HIGH** |
| R1 在 A6/A7 变体选择（Pass 2）是否生效？ | **否 —— 绕过 R1**。L3162 用 `rec.total_score`。 | **HIGH**（但 A6/A7 生产默认关闭，故默认路径下不触发） |
| 是否存在 R1 与 SB-FBPR/GCFC 同时启用的真实运行？ | **不存在**。全仓 78 份 `参数.json` 中，所有开启 SPNP/Residual/GCFC 的运行一律为 `score_select` 模式。 | **HIGH（Q5 的直接答案）** |

> **一句话结论：R1 只在"单提示词 + 每实例"这一条路径上生效。凡是真正跑过 SB-FBPR / GCFC / A6 / A7 的实验，走的都是不经过 R1 的 `score_select` 路径。**

---

## 1. 全仓 `total_score` vs `primary_score` 使用点穷举

以下为对生产脚本的完整 grep 结果（无遗漏）：

| 行号 | 表达式 | 用途 | 是否最终掩膜选择？ |
|---|---|---|---|
| 81 | `total_score: float` | 字段声明 | — |
| 1557 | `total_score=0.0` | 空掩膜硬门返回值 | — |
| 1643 | `total_score=float(total)` | 计算并写入记录 | — |
| **1688** | `best = max(selectable, key=lambda row: row.total_score)` | **`score_select` 选择** | **是 → 绕过 R1** |
| 1697 | `return ..., best.total_score, False` | `score_select` 返回值 | — |
| **1699** | `score_by_prompt = {…max(row.total_score,0.0)…}` | **`weighted_fusion` 权重来源** | **是 → 绕过 R1** |
| 1732 | `sorted(score_records, key=lambda r: r.total_score, reverse=True)` | 重提示触发排序 | 否（辅助报警） |
| 1733 | `sorted_recs[0].total_score − sorted_recs[1].total_score < gap` | 重提示触发判据 | 否（辅助报警） |
| **1734** | `best_score = float(getattr(best_rec,"primary_score",…))` | **返回给调用者的选择分数** | 否（记录） |
| 1738 | `row.total_score < reprompt_min_score` | 单候选分支重提示判据 | 否（辅助报警） |
| 1739 | `float(row.total_score if row else 0.0)` | 单候选分支返回分数 | 否（记录） |
| 2611 | `"总分": row.total_score` | CSV 列 | 否（记录） |
| 2635 | `"主评分": getattr(row,"primary_score",0.0)` | CSV 列（**R1 顺序**） | 否（记录） |
| 2866-2867 | `(primary_score, total_score)` | **CSV `排名` 列排序** | 否（记录，R1 顺序） |
| 3160 | `rec.total_score = rec.total_score − penalty` | 变体重叠泄漏惩罚 | — |
| **3162** | `best_record = max(variant_records, key=lambda r: r.total_score)` | **A1s/A6/A7 变体选择** | **是 → 绕过 R1** |
| 3166 | `selected_score = best_record.total_score` | 返回分数 | 否（记录） |

**唯一真正用 R1 主评分做掩膜选择的调用点：L1725 `select_best_policy(default_recs, …)`。**

> 注意 L2635 与 L2866：CSV 的 `排名` 列**已经**按 R1 的 `primary_score` 排序。
> 因此论文里"候选排名"表若取自该 CSV，是 R1 口径；但最终掩膜未必是 R1 所选。这一区分是本次审计的关键。

---

## 2. 四个候选模式分支逐一核对

### 2.1 单提示词 + 每实例（`prompt_selection_mode=single`, `candidate_mode=per_instance`）

```
select_mask()
  L1682  if use_prompt_ensemble and prompt_selection_mode=="score_select":  → 不进入
  L1698  if use_prompt_ensemble and prompt_selection_mode=="weighted_fusion": → 不进入
  L1710  prompt_id = args.default_prompt_id                       # 例如 P6
  L1713  default_cands = [c for c in candidates if c.prompt_id==P6]
  L1717  if len(default_cands) > 1 and args.candidate_mode == "per_instance":
  L1718      rec_by_id = {(r.prompt_id, r.instance_id): r for r in score_records}
  L1719-1723 default_recs = [...同一 (prompt_id, instance_id) 的记录...]
  L1725      best_rec = select_best_policy(default_recs, args.ranking_policy)   ← R1 生效
  L1726-1730 best_item = 对应候选
  L1732-1733 重提示触发判据（用 total_score，仅报警）
  L1734-1735 return best_item.mask, "P6#{instance}", best_rec.primary_score, needs_reprompt
```

**判定：R1 生效。**
**前提：`default_prompt_id` 对应 prompt 必须返回 > 1 个实例。** 若只返回 1 个实例，则落入 L1736-1739 分支，
直接取 `default_cands[0].mask`，无任何排序 —— 此时 R1 无关紧要（无选择可做）。

**历史实验实际状态：** Phase 17A/17B/17C 的 P6 运行均为此模式
（`prompt_selection_mode=single`, `candidate_mode=per_instance`, `default_prompt_id=P6`）。
`Phase17C_ranking_benchmark.csv` 中每帧存在 1–9 个实例 → 该分支确实被触发。

### 2.2 多提示词 `score_select`

```
L1682  if use_prompt_ensemble and prompt_selection_mode=="score_select":
L1684      non_empty = {item.prompt_id for item in candidates if item.mask.any()}
L1685      selectable = [row for row in score_records if row.prompt_id in non_empty]
L1686      if not selectable: selectable = score_records
L1688      best = max(selectable, key=lambda row: row.total_score)      ← 用 R0 的 total_score
L1690-1696 best_cand = 匹配 (prompt_id, instance_id) 的候选
L1697      return best_cand.mask.copy(), best.prompt_id, best.total_score, False
```

**判定：R1 不生效。最终候选选择完全由 R0 的 `total_score` 决定。**
语义门控若开启，其影响仅通过 `semantic_total` 进入 `total_score`，不进入 R1 主评分路径。

**历史实验实际状态：** 仓库中所有 32 份 `参数.json` 记录 `use_prompt_ensemble=True` 的运行，
`prompt_selection_mode` **全部为 `score_select`**，无一例外。这些运行同时是唯一一批开启 SB-FBPR / Residual / GCFC / A6 / A7 的运行。

### 2.3 多提示词 `weighted_fusion`

```
L1698  if use_prompt_ensemble and prompt_selection_mode=="weighted_fusion":
L1699      score_by_prompt = {row.prompt_id: max(row.total_score,0.0) for row in score_records}
L1700-1704 weights = {pid: score/score_sum}                         ← 融合权重来自 R0 total_score
L1705-1708 fused = Σ weights[pid]·mask(pid)；return fused >= fusion_threshold, "weighted_fusion", …
```

**判定：R1 不生效。**
融合权重直接取各 prompt 的最高 `total_score`，未使用 `primary_score`。
`score_candidate()` 已算出的 `primary_score` 在此路径中完全未被读取。

**历史实验实际状态：** 仓库中 **无任何** `prompt_selection_mode=weighted_fusion` 的运行。
→ 该路径为**未经实验验证的代码路径**。

### 2.4 单提示词 + 单实例兜底（L1736-1739）

```
L1736  row = next((r for r in score_records if r.prompt_id==prompt_id), None)
L1738  needs_reprompt = (row is None) or (row.total_score < reprompt_min_score) or row.empty_flag
L1739  return default_cands[0].mask.copy(), prompt_id, float(row.total_score if row else 0.0), …
```

**判定：无选择发生，R1 不参与（也不需要）。**
返回分数仍记录 `total_score`（R0 口径）而非 `primary_score`。
这是 `Easy21` 里那些只有单实例的帧（多数 Easy21 帧 `_00` 单实例）实际走的路径。

---

## 3. 其他候选相关决策点核对

| 决策点 | 位置 | 依据的分数 | 是否 R1 口径 |
|---|---|---|---|
| 空掩膜硬门 | L1550-1569 | 恒 `total=0, primary=0, empty_flag=True` | 中性（两者一致） |
| 重提示触发（多候选分支） | L1732-1733 | `total_score` 排序后取 top1−top2 差 | **否（R0）** |
| 重提示触发（单候选分支） | L1738 | `total_score < reprompt_min_score` | **否（R0）** |
| A1s/A6/A7 变体选择（Pass 2） | L3162 | `rec.total_score` | **否（R0）** |
| 变体泄漏惩罚 | L3146-3161 | 在 `total_score` 上再减 `leak_penalty` | **否（R0 + 惩罚）** |
| A7 种子选择 `memory_seed_mode=consensus_best` | L3044 | `per_frame_info["回退IoU"]` | 与 R1 无关（IoU 回退） |
| A7 种子选择 `memory_seed_mode=best_score`（默认） | L3046, L3048-3049 | `scores_by_stem[stem]` = Pass1 返回的 `selected_score` | Pass1 单提示词分支下 **是 R1 口径** |
| `提示词评分.csv` 的 `排名` 列 | L2862-2870 | `(primary_score, total_score)` | **是 R1 口径** |
| 语义门控评分写入 | L1624 | `semantic_total` → 进入 `total`，不进入 R1 路径 | **否** |
| `compute_geometry_feedback` 低分报警 | L3318-3319 | `geometry_score < 低` | 与 R1 无关（独立报警） |

### 3.1 关于"使用旧 `total_score` 不一定是 bug"的判定

按任务要求，把 `total_score` 的保留使用分为五类：

| 类别 | 本审计中的实例 | 是否构成兼容性问题 |
|---|---|---|
| ① 最终候选选择 | L1688（score_select）、L1699（weighted_fusion）、L3162（变体） | **是 —— 若这些路径用于正式配置** |
| ② 低置信度报警 | L1732-1733、L1738 | **否** —— 属辅助报警，不改掩膜 |
| ③ 历史记录 | L1739、L2611、L3166 | **否** |
| ④ 辅助排序显示 | L2866（与 R1 主评分并列，R1 优先） | **否** |
| ⑤ 实验性分支 | L3046/3049（A7 种子，A7 已默认关闭） | **条件性** —— A6/A7 启用时需注意 |

---

## 4. 决定性证据：全仓 78 份 `参数.json` 的交叉扫描

扫描脚本：遍历仓库所有 `参数.json`（排除 `.git`），共 **78** 份。

### 4.1 开启 refinement 的运行（27 份）

**全部 27 份的 `prompt_selection_mode` 均为 `score_select`，`ranking_policy` 均为 `None`（早于 R1 提交）。**

典型样本：

| 运行 | prompt_list | selection_mode | SPNP | Residual | GCFC | rank |
|---|---|---|---|---|---|---|
| `XianKeLai1_A5c_语义门控几何修正_GT1冒烟` | P1..P5 | score_select | **ON** | **ON** | **ON** | None |
| `E3v2_KongQueZhuYu_GT5_A5c_完整RAPFSAM3v2` | P1..P5 | score_select | **ON** | **ON** | **ON** | None |
| `E3v2_KongQueZhuYu_GT5_A1s结构化正负提示` | P1..P5 | score_select | **ON** | OFF | OFF | None |
| `E4b_downstream/CaoMei2` | P1..P5 | score_select | **ON** | **ON** | **ON** | None |
| `E1_CaoMei1_P6带盆整株_101帧` | **P6** | score_select | **ON** | OFF | OFF | None |
| `E1_XianKeLai1_P6带盆整株_101帧` | **P6** | score_select | **ON** | OFF | OFF | None |
| `E1_DouBanLv1_P6带盆整株_101帧` | **P6** | score_select | **ON** | OFF | OFF | None |
| `E1_ChangShouHua2_P6带盆整株_101帧` | **P6** | score_select | **ON** | OFF | OFF | None |

**关键发现（Phase 20A 新发现，Phase 19A 未记录）：**

1. **SB-FBPR / Residual / GCFC 从未在 `single` + `per_instance` 模式下运行过。**
2. **R1 从未与 SB-FBPR / Residual / GCFC 同时运行过**（因为 R1 在 `score_select` 下被绕过）。
3. 存在 **4 份 P6 + score_select + SPNP** 的 101 帧运行（CaoMei1 / XianKeLai1 / DouBanLv1 / ChangShouHua2），
   这些是"带盆整株 + SPNP 正负提示细化"的真实产物，但**它们的候选选择仍是 R0 口径**。
4. 唯一的 P2 + single + 历史运行是 `阶段十一修复验证/legacy_对照_CaoMei1_0100`，且它**只开了 GCFC**（未开 SPNP/Residual），早于 R1 提交。

### 4.2 开启语义门控的运行（25 份）

全部 `score_select` + `gate_backend=heuristic`。**语义门控从未在 `single` 模式下运行。**
→ 论文中"Semantic-gated multi-prompt selection（语义门控多提示词选择）"这一核心模块，
**从未以论文描述的形式（对多提示词做 softmax 选择）被验证过**。见 `METHOD_CONFIGURATION_AUDIT.md` 第 2 节。

### 4.3 开启 A6/A7 的运行（25 份）

全部 `score_select`。含 Phase 17A 的 P10/P01/P11、U10/U01/U11。
→ **A6/A7 也从未在 `single` 模式下运行过**，因此 A1s/A6/A7 变体选择路径（L3162，R0 口径）是唯一验证过 A6/A7 的路径。

### 4.4 `ranking_policy` 字段

78 份 `参数.json` 中 **无一份**含 `ranking_policy` 键 —— 因为该键在 Phase 17C（`d704faa8`）才引入，
而所有 78 份是更早的冻结产物。**因此历史上没有任何一次运行显式请求过 R1。**
Phase 17A/17B/17C 的"实际配置"等价于 R0（`base`）：`select_mask()` 的 L1725 在 `ranking_policy` 缺省
时通过 `getattr(args, "ranking_policy", "r1")` 回退 —— 但那些运行的候选文件早于该字段存在，
Phase 17C 以 `policy="base"` 显式重放（其 T1 测试：10/10 DEV10 帧复现历史 `提示词选择.csv`）。

---

## 5. 语义门控（SG-MPS）实现与论文描述的结构性差异

论文 §2.3.3 描述：
```
c_i^(k) = σ(w_φ^T φ_i^(k) − w_ψ^T ψ_i^(k) − b)
a_i^(k) = exp(c_i^(k)/τ_s) / Σ_l exp(c_i^(l)/τ_s)      ← 温度控制 softmax
k* = argmax_k a_i^(k)
```

生产代码（`build_semantic_gate_context()` L1384 + `semantic_gate_scores()` L1461）：
- `semantic_gate_scores()` 返回的是一个**加性启发式得分字典**：
  `semantic_total`、`target_box_score`、`vertical_coverage_score`、`pot_overlap_penalty`、
  `side_distractor_penalty`、`center_prior_score`（L1615-1622）。
- **仓库中不存在任何 `softmax`、`temperature`、`τ_s` 实现**：
  ```
  $ grep -n "softmax\|temperature\|tau_s" 生成RAP-FSAM3掩膜.py
  1591:    q_sam = 1.0 / (1.0 + math.exp(-6.0 * (float(candidate.sam_score) - 0.5)))
  ```
  唯一的 `exp()` 是 SAM3 分数的 sigmoid 归一化，与 softmax 无关。

**结论：论文的"温度控制 softmax 多提示词选择"在生产代码中没有对应实现；生产代码是逐候选加性启发式语义门控。**
这不影响 R1 的数学正确性，但直接影响"论文方法与真实方法是否一致"——
即**原论文 SG-MPS 描述与真实实现存在结构性差异**，属于 `METHOD_CONFIGURATION_AUDIT.md` 的 GAP 项。

---

## 6. 与论文方法描述的关系（是否构成"绕过"）

| 论文 RAP-FSAM3 步骤 | 生产代码入口 | 该路径是否使用 R1 |
|---|---|---|
| FFT 帧质量筛选 | `02-FFT` 独立脚本（不在本脚本内） | 与 R1 无关 |
| 多提示词候选生成（K=5） | `infer_candidates()` L1045 | 是 R1 的**输入** |
| 语义门控多提示词选择 | `semantic_gate_scores()` L1461 | **否**（加性，进入 `total_score`） |
| 最终候选选择 | `select_mask()` L1675 | **仅 `single`+`per_instance` 分支是** |
| 结构平衡正负提示细化 | `apply_spnp_refinement()` L2061 | **否**（其输入掩膜由上游决定） |
| 几何一致性修正 | `apply_corrective_geometry()` L2374 | **否** |

**直接回答任务问题 2 与 4：**
- **R1 已正确接入生产脚本，但只在一条分支生效。**
- **旧多提示词 `score_select` 模式确实绕过 R1**（L1688），且这是**唯一**真正运行过
  SB-FBPR/GCFC/A6/A7 的模式。**这是本阶段最重要的审计发现。**

---

## 7. 影响面评估

若论文最终方法采用：

- **方案 A（P6 + per-instance + R1）**：R1 生效。SB-FBPR/GCFC 需显式开启且**必须走 R1 生效的路径**
  → 当前代码下开启 SPNP 会强制走 `score_select`（L1682 在 L1710 之前返回），从而**静默绕过 R1**。
  这是一个真实的、非显然的交互风险，已记录于 `P0_BLOCKERS_AND_NEXT_ACTIONS.md`。
- **方案 B（原多提示词 + R1）**：**当前代码不产生 R1 效果**。多提示词选择走 L1688 的 `total_score`。
  要使 R1 生效需修改 `select_mask()`（L1688 改用 `select_best_policy(selectable, policy)`），
  但这是**生产代码修改**，Phase 20A 不得实施，只能给出建议。

两方案的详细对比见 `METHOD_SELECTION_DECISION.md`。

---

## 8. 审计边界

- 未修改 `生成RAP-FSAM3掩膜.py` 或 `ranking_policy.py`。
- 未把 GT10 当独立测试集；本文件所有 GT10 数字仅用于 DEV 诊断。
- 未对 TEST32 无 GT 帧计算准确率。
- 本文件所有行号对应 commit `6df0c3df`。