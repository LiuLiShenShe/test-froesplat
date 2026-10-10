# 单提示词 vs 多提示词：最终方案选择建议 / METHOD_SELECTION_DECISION

**阶段：** Phase 20A（只给推荐，不实施代码修改）

---

## 0. 两方案定义

### 方案 A：P6 + per-instance + R1

```
SAM3 P6（"whole potted plant including the pot"）
  ↓  per_instance：每个 SAM3 实例独立为 Candidate
R1（主评分 = base_total − q_contrast/5.5 − penalty，L1725 生效）
  ↓
Configured refinement（按需开启，SPNP → 残差修复 → GCFC）
  ↓
Final foreground mask
```

### 方案 B：原多提示词 RAP-FSAM3 + R1

```
SAM3 P1–P5（多提示词）
  ↓  per_instance：每 prompt 每实例为 Candidate
R1 candidate selection（预期）
  ↓  Configured refinement
Final foreground mask
```

---

## 1. 证据核对表（逐问回答任务书 §10 要求）

| 要求 | 证据 | 结论 |
|---|---|---|
| A 的困难集证据是否一致 | Phase 17B：0039/0040/0041 的正确 P6 候选 F1=0.9867/0.9882/0.9865（PROVEN）；Phase 17C：R1 3/3 rescued，Easy21 0 回归 | **一致** |
| A 是否有实际运行支持 | Phase 12 `P6_raw_baseline`（21 帧，prompt=P6，per_instance，single）✓；Phase 17A P00/P10/P01/P11（P6，per_instance，single）✓；81 份已冻结配置交叉证实 | **有** |
| A 的带盆前景是否符合原稿任务 | 原稿研究对象为"目标盆栽植株实例"（§2.2）；GT `potted_clean` 含盆（Phase 20A 逐像素验证一致率 1.000000）；P6 实测盆召回 ≈97.5% | **符合** |
| A 是否与原 2DGS 输入一致 | **不一致待查**：2DGS 用 `03-分割Mask/02-sam_masks/`（无 pot 标注）而 P6 输出在 RAP-FSAM3 掩膜目录；未定位 2DGS 掩膜与 P6 掩膜的同一性比较 | **UNVERIFIED**（P0 清单） |
| A 是否保留花盆 | **是**（实测 88.1% 全均值；剔除 2 退化帧后 ≈97.5%） | **是** |
| A 是否兼容原表型尺度恢复 | **形式一致**（盆在场景中）；但尺度恢复**无实现代码**，且若网格 plant-only 则冲突 | **条件一致** |
| B 的多提示词选择是否绕过 R1 | `select_mask()` L1688 用 `row.total_score` → **是，绕过** | **绕过（代码事实）** |
| B 的原 Table 3/5 实验是否真正使用多提示词 | 是（`prompt_ids=['P1'..'P5']`，35 份运行） | **是** |
| B 的多提示词与 R1 是否有真实联合验证 | **否，零例** | **否** |
| B 是否需要修改代码才能严格实现 | **是**：L1688 改用 `select_best_policy(selectable, policy)`（最小修改，建议见 §4） | **是** |
| 是否可直接把旧多提示词结果与新单提示词结果混合比较 | **否**（禁止） | — |
| 是否可把所有收益归于 R1 | **否**（禁止；同时改 prompt+ranking+refinement 即混淆归因） | — |

---

## 2. 额外影响方案选择的事实（Phase 20A 新发现）

| # | 发现 | 对方案选择的影响 |
|---|---|---|
| **F1** | **P2 实测与 P6 行为不可区分**：P2（"excluding pot"）的盆召回 ≈97.5%，与 P6 差 < 0.1 个百分点。P2 的命名是误导。 | 削弱"必须在 P2/P6 间二选一"的语义负担 —— 两者实际产物近似，"带盆 vs 去盆"不是有效区分轴。真正重要的是**提示词稳定性**（P6 在 17A/17B 困难集上生成了正确候选，P2 未被同等验证）。 |
| **F2** | **E1 的全部 10 个方法行（含 SAM3 与 RAP）用同一个 legacy-union GT（potted ∪ cube）评分。** 论文 Table 3 的四个数字全部来自 `summary_metrics.csv`（与该 GT 对应），而 Phase 12/16/17/18 全部使用 `potted_clean`。 | 若新论文采用 `potted_clean`（建议，见尺度一致性文件），**Table 3 的全部四个数字都要重算**（离线重算即可，不需 GPU）。`summary_metrics.csv` 的口径**不得直接继承**。 |
| **F3** | **M2M3-GO 的 S_m2g/S_g2m/B^∂/U 无实现代码**；Floor40 为全局计数下限而非论文的局部块下限；2DGS 四个 wall-clock 时间无出处；尺度恢复无实现。 | 方案 A/B 的**上游（分割）选择**不受影响，但下游（重建）部分现状是：W13/W15/W18/W14/W24/W22/W23 为部分或全部 `UNVERIFIED`。最终方法在重建侧**不能**宣布 LOCKED。 |
| **F4** | **D3/D6/D12 无帧**；53/58 帧同标签。 | 困难场景叙事只能围绕 D1（邻株交叠）+ R1 排序修复展开，不能声称五类覆盖。 |
| **F5** | **开启 SPNP 的运行 100% 为 score_select**；若照抄历史命令并在默认 single 上开 SPNP，`use_prompt_ensemble` 仍为 false → 走 R1 分支（L1710）——这与"历史 SPNP 产物全是 R0"形成**静默口径切换风险**。 | 任何 V0–V3 实验必须在 `参数.json` 中同时固定 `use_prompt_ensemble=false` 与 `prompt_selection_mode=single`，否则 R1/R0 混用。 |

---

## 3. 推荐方案

> **推荐：方案 A（P6 + per-instance + R1，refinement 显式配置）。**

**证据（按强度排序）：**

1. **正确候选存在性（PROVEN）**：Phase 17B 证明 P6 已生成 F1≈0.987 的正确候选，瓶颈在排序而非生成。更换提示词集不能解决排序问题。
2. **最小充分干预（SUPPORTED）**：Phase 17C 证明仅移除 `q_contrast` 一项即修复 3/3 失败，Easy21 零回归。Phase 20A 独立重算确认。
3. **运行一致性**：Phase 12（21 帧）与 Phase 17A（10+32 帧）均为 P6 + per-instance + single。
   R1 是与该链路**唯一兼容**的排序策略（R1 在该分支生效，在 score_select 分支绕过）。
4. **口径一致性**：P6（含盆）与 `potted_clean` GT、与花盆尺度参照自洽。
5. **简单性**：方案 A 不需要任何生产代码改动；方案 B 需要改 `select_mask()`（L1688）。

**已有结果的继承程度：**

| 结果 | 继承程度 | 条件 |
|---|---|---|
| Phase 12 P6 基线（F1=0.983498，21 帧） | **可继承为历史基线**（R0 口径） | 标注为 `base` 策略下的冻结基线，而非 R1 口径 |
| Phase 17B oracle 候选（F1≈0.987） | **可继承为诊断证据** | 角色 = DIAGNOSTIC_DEV，不进入验证结论 |
| Phase 17C 重排序（R1，3/3 rescue） | **可继承为 R1 的 DEV 证据** | 口径 = 候选重排序，非正式运行；TEST32 仅行为审计 |
| RAP-FSAM3-v2（E1，P1–P5 全链路） | **降级为历史对照**（方案 B 的产物） | 不得作为方案 A 的结果引用 |
| Table 3/5 数字 | **见 `RECONSTRUCTION_EVIDENCE_REUSE_MATRIX.csv`** | F1/mIoU 需离线重算到 `potted_clean` |

**风险：**

| # | 风险 | 缓解 |
|---|---|---|
| R-A1 | R1 从未作为正式运行执行过（零正式运行证据） | V1（P6+R1+零细化）为必做最小实验（GPU 必需见 V 计划） |
| R-A2 | R1 + SPNP/Residual/GCFC 联合从未运行 | V3（P6+R1+全细化）为必做联合实验；若 V3 反常，细化模块需单独审查 |
| R-A3 | 2DGS 输入掩膜与 P6 掩膜的同一性未核对 | P0-2：逐像素比较（离线） |
| R-A4 | 带盆前景与 plant-only 需求的张力（P2 命名误导） | 若需 plant-only 变体，用 `compose_plant_only()` 受控减除（需先实现验证） |

---

## 4. 若实施方案 B 所需的最小代码修改建议（精确到函数）

> **本阶段只给建议，不实施。**

```python
# 文件：生成RAP-FSAM3掩膜.py，函数 select_mask()，行 1682-1697
# 现状：
best = max(selectable, key=lambda row: row.total_score)
# 最小修改：
best = select_best_policy(selectable, getattr(args, "ranking_policy", "r1"))
# 同时：
return best_cand.mask.copy(), best.prompt_id, best.primary_score, False
#   （把返回分数从 best.total_score 改为 best.primary_score，保证记录口径一致）
```

```python
# 文件：生成RAP-FSAM3掩膜.py，函数 select_mask()，行 1698-1708（weighted_fusion）
# 现状：
score_by_prompt = {row.prompt_id: max(row.total_score, 0.0) for row in score_records}
# 最小修改（二选一）：
#  (a) 废弃 weighted_fusion（推荐：无任何运行记录）；
#  (b) 改为 score_by_prompt = {row.prompt_id: max(row.primary_score,0.0) …}
```

**回归约束：** 修改后必须复跑 `tests/test_phase17c.py` 的 T1（R0 复现历史 `提示词选择.csv` 10/10）
与 `tests/test_phase12.py`，确认 `policy="base"` 显式路径不受影响。

---

## 5. 是否需要补充联合实验

**是。** 详见 `FOLLOWUP_EXPERIMENT_PLAN.md`。最小集合：

| 组别 | Prompt | Ranking | Refinement | 是否已有产物 | 需 GPU |
|---|---|---|---|---|---|
| V0 | P6 | R0（base） | OFF | ✅（Phase 17A P00，如有候选缓存可复用） | ❌（仅重排序则离线；正式运行需 GPU） |
| V1 | P6 | R1 | OFF | ❌（仅 CSV 重排序） | **必需（正式运行）** |
| V2 | P6 | R0 | ON | ❌（只有 score_select 版，无 single 版） | **必需** |
| V3 | P6 | R1 | ON | ❌ | **必需** |

若推荐多提示词方案（本审计不推荐），应另行设计公平的多提示词对照。

---

## 6. 尚未解决的风险（与最终规格草稿的对应）

| # | 风险 | DRAFT 标注 |
|---|---|---|
| U-A | R1 正式运行缺失（仅重排序证据） | `FINAL_RAP_FSAM3_SPEC_DRAFT.md` → `DRAFT` |
| U-B | R1 × 细化联合缺失 | → `DRAFT` |
| U-C | 2DGS 输入掩膜同一性未核对 | → `DRAFT` |
| U-D | M2M3-GO 公式无实现（M16–M20） | → 方法节必须降级为"实现机制描述" |
| U-E | 盆径数值与尺度恢复实现缺失 | → `DRAFT` |
| U-F | Table 7 时间与"20 序列平均"无出处 | → `DRAFT` |

**因此最终方法状态 = `DRAFT`，不是 `LOCKED_CANDIDATE`。**