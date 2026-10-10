# Phase 20A 最终报告 / PHASE20A_FINAL_REPORT

**阶段：** Phase 20A — ForeSplat 方法融合审计与配置锁定准备
**日期：** 2026-10-10
**基线 commit：** `6df0c3df`（Phase 19A）
**分支：** `main`
**远程：** `https://github.com/LiuLiShenShe/test-froesplat.git`

---

## 0. 一句话结论

> **原 RAP-FSAM3 框架可以保留，R1 应当**替换**原候选评分（而非作为后处理）；
> 但 R1 目前只在"单提示词 + 每实例"一条路径上生效，而历史上**所有**跑过 SB-FBPR / GCFC / A6 / A7
> 的实验走的是**不经过 R1** 的 `score_select` 路径。
> 因此推荐方案 A（P6 + per-instance + R1），但最终算法**尚不能锁定**（DRAFT）。**

---

## 1. 审计执行摘要

| 项目 | 结果 |
|---|---|
| Git 状态 | 起始工作区干净（无未提交修改），未使用 `reset --hard` / `clean -fd` |
| 生产脚本定位 | `00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py`（3381 行） |
| 退稿论文定位 | `…/ForeSplat__Foreground_Object_…_for_Plant/ForeSplat.tex`（850 行） |
| 代码阅读 | `生成RAP-FSAM3掩膜.py` 关键路径 613–804 / 1540–1739 / 1771–1830 / 2061–2481 / 2742–3381；`ranking_policy.py` 全文 |
| 全仓配置扫描 | **78 份 `参数.json`** 逐份解析 |
| 像素级重算 | `potted_clean` 定义（21 帧）、P6 花盆覆盖（21 帧）、Table 3 口径判定（5 种 GT × 2 方法 × 20 帧）、Phase 12 细化链逐字节同一性（21 帧 × 6 目录） |
| R1 数值重算 | 182 条归档候选独立重算；51 个 (source, sample, frame) 组重排 |
| 非 GPU 测试 | **150 passed**（Phase 17C 当时报告 142，差额为后续新增测试） |
| 后台并行调查 | 4 项：论文表格溯源 / 2DGS-M2M3-GO 链路 / GT 类别定义 / Phase 16 困难集 |

---

## 2. 对任务书 12 个问题的直接回答

### Q1　原 RAP-FSAM3 是否能够完整保留？
**部分保留。**
- FFT 筛选、SAM3 候选生成、语义门控、SB-FBPR、残差修复、GCFC 的**代码入口全部存在**。
- 但**论文描述与实现存在 5 处结构性差异**：FFT 公式（P1-3）、SG-MPS 的 softmax 温度选择（P1-2）、
  M2M3-GO 四条公式无实现（P1-4）、Floor40 机制（P1-5）、Table 4 的 P2 语义错标（P1-1）。
- 另有 4 个纯死代码函数（`compose_plant_only` / `compose_potted` / `detect_semantic_collapse` /
  `select_single_object_id`）与 1 个死分支（`track_mask` → +0.15 罚项）。

### Q2　R1 是否应该替换原候选评分，而不是作为额外后处理？
**应该替换。**
理由：(a) Phase 17B 已 PROVEN 失败发生在**选择**环节（正确候选 F1≈0.987 存在但未被选中，
regret ≈ 0.987）；(b) 若作为后处理则无法改变被选中的候选，只能在其上修补，语义不成立；
(c) Phase 17C 已证明该替换是最小充分干预（3/3 修复，0 回归）。

### Q3　现在的 R1 在全部真实使用的选择分支里是否生效？
**否，只在一条分支生效。**

| 分支 | R1 是否生效 | 证据 |
|---|---|---|
| `single` + `per_instance`（多实例） | **是** | `select_mask()` L1725 → `select_best_policy()` |
| `single`（单实例） | 不适用（无选择） | L1736-1739 直接取唯一候选 |
| `score_select`（多提示词） | **否** | L1688 用 `row.total_score` |
| `weighted_fusion` | **否** | L1699 用 `row.total_score` |
| A1s/A6/A7 变体选择 | **否** | L3162 用 `rec.total_score` |

### Q4　旧多提示词模式是否绕过 R1？
**是，明确绕过。**
`select_mask()` L1682-1697 在 `use_prompt_ensemble=true` + `score_select` 时早退，
`best = max(selectable, key=lambda row: row.total_score)`（L1688）——纯 R0 口径。
**且这正是历史上唯一真正执行过 SB-FBPR / GCFC / A6 / A7 的模式。**

### Q5　SPNP、Residual repair、GCFC 是否确实在历史正式实验中开启？与 R1 同时开启是否有真实证据？
- **是否开启过：是，但不在最终正式实验中。**
  27 份运行开启 refinement（12 份同时开启 GCFC），**100% 为 `score_select`**；
  Phase 12 正式 P6 基线与 Phase 17A 全部 4 个变体**全部关闭**。
- **与 R1 同时开启：零证据。**
  `prompt_selection_mode="single"` + `use_spnp_refinement=true` = **0/78 例**；
  显式设 `ranking_policy` 且开 refinement = **0/78 例**。
- **状态：`INTEGRATION_NOT_YET_VERIFIED`。**

### Q6　P2 与 P6 哪个更适合论文最终研究对象？推荐依据是什么？
**推荐 P6（带盆整株）。**
- GT `potted_clean` 含盆（逐像素验证一致率 1.000000），P6 实测盆召回 ≈97.5%，蓝块 0 px → 口径自洽。
- 花盆是论文尺度参照的物理载体，含盆才可能从场景恢复尺度。
- Phase 16/17 全部困难集证据建立在 P6 之上。
- **补充事实**：P2 实测与 P6 在盆覆盖上**不可区分**（差 < 0.1 个百分点），
  故"去盆 vs 带盆"不是有效区分轴；真正的区分是提示词稳定性（P6 在困难集上生成了正确候选）。

### Q7　Phase 17B/17C 的实验是否支持拟定的最终方案？
**支持，且 Phase 20A 已独立重算确认。**
- 17B：正确 P6 候选存在（0.9867/0.9882/0.9865），失败在选择 → 直接支持"P6 + 修排序"。
- 17C：R1 3/3 rescue、DEV10 其余 7 帧 0 回归、Easy21 21 帧 0 变化、TEST32 churn 2/32。
- **Phase 20A 独立重算**：0039/0040/0041 由 0.0000 → 0.9867/0.9882/0.9865；
  Easy21 21 帧全无变化；总 churn 8/51 组（其中 5 组为无 GT 上下文帧）。
- **限制**：均为开发集/诊断角色，不是独立验证。

### Q8　原来 ForeSplat 的 Table 3、5、6、7、8 哪些可以继承？
| 表 | 判定 |
|---|---|
| **Table 3** | 数字有出处（`E1/metrics/summary_metrics.csv`），但 GT 为**旧并集**（含蓝块），与 Phase 12/16/17/18 的 `potted_clean` 不同口径，且 Phase 20A 用 5 种 GT 变体**均无法复现**该 F1。→ `TARGETED_RERUN_REQUIRED`（离线重算） |
| **Table 4** | 数字可继承；**P2 标签必须改正**。→ `REUSE_AS_LEGACY_BASELINE` |
| **Table 5** | 混合聚合口径 + `0.4543` / `0.001935` 两个数字全仓无出处；`w/o GCFC` 行的 F1/mIoU 取自 A5c、HD95 取自 A2；`Geometry-only` 行是离线重放。→ `TARGETED_RERUN_REQUIRED` |
| **Table 6** | 单样本（KongQueZhuYu）形态消融，论文描述与实现相符。→ `REUSE_AS_LEGACY_BASELINE` |
| **Table 7** | 四个 wall-clock 时间无出处；"20 序列平均"无产物支撑（ForeSplat 输出仅 4 样本、M2M3 pilot 仅 3 场景、网格 13 样本）。→ `UNVERIFIED` |
| **Table 8** | 统计量 n=21 / n=63 与归档一致；但上游几何测量无实现。→ `PARTIAL` |

### Q9　哪些 2DGS 样本可能需要重新训练？
**目前判定：`NOT VERIFIED` → 倾向于"先离线比对，暂不重训"。**
- 2DGS 全部损失项与 COLMAP 前景初始化使用**同一** `mask_dir`
  （`03-分割Mask/02-sam_masks/<Sample>/`，19 样本）。
- 该目录与 P6 输出的同一性**尚未逐像素比较**（`P0-2`）。
- 若比较结果显示差异极小（例如 mean IoU > 0.99），**无需重训**；
  若差异显著，则需按 `RECONSTRUCTION_EVIDENCE_REUSE_MATRIX.csv` 的分类逐样本判定
  `REUSE_DIRECTLY` / `TARGETED_RERUN_REQUIRED`。
- **本阶段无依据要求全量重跑 20 个序列。**

### Q10　D1/D3/D6/D11/D12 中有哪些真实可用的代表图？
| 类别 | 可用帧 | GT | 用途 |
|---|---|---|---|
| **D1** | DouBanLv2 0039/0040/0041（Panel A）、BaiZhang 0033–0037 与 DouBanLv2 0037/0038（Panel B） | ✅ 10 帧有 GT | 定性 + 可定量（DEV） |
| **D11** | XiangPiShu2 0133/0134/0136、XianKeLai3 0117（Panel C） | ❌ 无 GT | 仅定性，必须标注 |
| **D3** | **0 帧 — BLOCKED** | — | 不得声称覆盖 |
| **D6** | **0 帧 — BLOCKED** | — | 不得声称覆盖 |
| **D12** | **0 帧 — BLOCKED** | — | 不得声称覆盖 |

53/58 帧携带同一标签对 `D1+D11`；标签来自 `green_pct` 单一阈值二分支，**非人工视觉核查**、**不互斥**。
已生成 58 行资产清单（`DIFFICULT_CASE_ASSET_MANIFEST.csv`），所有源图经核实在盘。

### Q11　最终算法目前是否可以锁定？如果不能，还差哪几项验证？
**不能锁定，当前 `DRAFT`。** 缺：
1. **V1 正式运行**（P6 + R1 + refinement OFF）—— R1 至今只有离线重排序证据；
2. **V2/V3 正式运行** + `interaction = (V3−V2) − (V1−V0)` —— R1 × 细化零联合证据；
3. **2DGS 输入掩膜同一性比对**（`P0-2`）—— 决定是否需重训；
4. **尺度参照（盆径）数值与传递路径落实**（`P0-1`）—— Table 8 上游不可复现；
5. **Table 3/4/5 离线重算到 `potted_clean`**（`P0-3`）。

### Q12　接下来 Phase 20B 最小需要做什么？
见 `P0_BLOCKERS_AND_NEXT_ACTIONS.md` §4 的 9 步清单。最小子集：
- **不需 GPU**：P0-3 重算、P0-2 比对、P0-1 找回盆径、P1-9 修正 Phase 19A；
- **需 GPU**：V1、V2、V3；
- **需人工**：TEST32 的 32 帧独立验证标注（含 pot 多边形）。

---

## 3. 本阶段最重要的发现（按严重度）

| # | 发现 | 严重度 |
|---|---|---|
| **1** | **R1 与所有细化模块零联合运行证据**；历史上开启 refinement 的 27 份运行 100% 走不经过 R1 的 `score_select` | 高 |
| **2** | **论文 Table 5 中 `Boundary-F1=0.4543` 与 `leakage=0.001935` 全仓无出处**；各行混用逐帧均值与样本等权均值 | 高 |
| **3** | **论文 Table 3/4/5 的 GT 口径是旧并集（含蓝块）**，与 Phase 12/16/17/18 的 `potted_clean` 不同；且该 F1 用 5 种 GT 变体均无法复现 | 高 |
| **4** | **论文称分割评测覆盖"全部 20 个序列"，实为 4 个代表序列共 20 帧** | 高 |
| **5** | **花盆直径在全仓不存在**，尺度恢复无实现代码；且重建掩膜若 plant-only 则方法不闭合 | 高 |
| **6** | **M2M3-GO 的 S_m2g / S_g2m / B^∂ / U 与 keep-densify-prune 阈值仅存在于 YAML，无任何 Python 消费** | 高 |
| **7** | **论文把 P2（"entire plant excluding pot"）标为 "entire potted plant"**；且 P2 实测并不去盆（盆召回 ≈97.5%） | 中高 |
| **8** | **Table 7 的四个 wall-clock 时间无出处**；"20 序列平均"无产物支撑（实际 4/3/13 样本） | 中高 |
| **9** | **D3/D6/D12 各 0 帧**；53/58 帧同标签；标签为单阈值启发式、非人工、且 D1/D11/D14 不互斥 | 中 |
| **10** | **Phase 19A 两处结论被推翻**：`P6_raw_baseline` 实为 P6（非 P2）；GCFC 在该基线上未生效（21/21 逐字节恒等） | 中 |

---

## 4. 验收门自检

| 门 | 判定 | 说明 |
|---|---|---|
| **G1** 论文与生产代码已完整定位 | **PASS** | `.tex` 850 行 + 主脚本 3381 行 + `ranking_policy.py` 68 行 |
| **G2** 各模块有明确代码入口与历史实验配置 | **PASS** | `METHOD_VERSION_MATRIX.csv`（27 行 / 12 列，无畸形行）+ `PROMPT_AND_RUN_PROVENANCE_MATRIX.csv`（78 行 / 21 列） |
| **G3** R1 在 single/per_instance/score_select/weighted_fusion 的实际作用已查清 | **PASS** | `R1_CALL_PATH_AUDIT.md` 含全部 18 处 `total_score` 与 12 处 `primary_score` 使用点 |
| **G4** R1 公式与生产代码逐项一致 | **PASS** | `R1_FORMULA_VERIFICATION.md`；182 条候选重算，181/182 精确一致（1 条差 = 0.25 罚项） |
| **G5** P2/P6/`P6_raw_baseline` 实际运行身份已核实 | **PASS** | 两份独立参数快照 + 产物命名 + 逐字节链路验证 |
| **G6** SPNP / Residual / GCFC 执行状态与顺序已核实 | **PASS** | `REFINEMENT_CHAIN_AUDIT.md`；顺序确认、开关确认、21/21 逐字节证伪 |
| **G7** 原论文对应真实完整配置已确定或指出缺失 | **GAP** | 论文 Table 5 的无出处数字、Table 7 的时间来源无法确定（`P0-3`） |
| **G8** Plant/pot GT 与尺度恢复一致性已核实 | **PARTIAL** | GT 定义与 P6 覆盖 **CONSISTENT**；尺度恢复 **NOT VERIFIED**（`P0-1`） |
| **G9** Table 6/7/8 及主要图片有版本归属或未核验记录 | **PARTIAL** | Table 6 `REUSE_AS_LEGACY_BASELINE`；Table 7/8 `UNVERIFIED`/`PARTIAL` |
| **G10** D1/D3/D6/D11/D12 有真实资产清单与人工核验状态 | **PASS（含 GAP）** | 58 行清单；D3/D6/D12 = 0 帧（BLOCKED）；全部 `manually_verified=false` |
| **G11** 后续是否需改代码有明确结论 | **PASS** | 方案 A 不需改；方案 B 需改 `select_mask()` L1688（已给函数级建议） |
| **G12** 后续补实验有最小工作量计划 | **PASS** | `FOLLOWUP_EXPERIMENT_PLAN.md` V0–V3 + 独立验证计划 |
| **G13** 未修改生产算法、历史数据、原论文 | **PASS** | `git status` 仅新增本阶段目录 |
| **G14** 所有结论有文件路径/函数/参数/真实产物支持 | **PASS** | 全部结论标注路径+行号或产物路径 |

---

## 5. 最终判定

```text
PHASE20A VERDICT:
PARTIAL

FINAL METHOD STATUS:
DRAFT

CODE CHANGE REQUIRED:
NO            # 推荐方案 A 不需要改生产代码
                       # 仅当采纳方案 B 时需改 select_mask() L1688（见 METHOD_SELECTION_DECISION.md §4）

GPU RERUN REQUIRED:
CONDITIONAL   # 分割侧：V1/V2/V3 必需（其中 V2/V3 不可离线）
              # 2DGS 侧：待 P0-2 掩膜同一性比对结果，当前不建议重训
              # 已完成且无需重跑：Phase 12 P6 基线、Phase 17A 四变体

OLD 3D RESULTS REUSABLE:
NOT VERIFIED  # 待 P0-2 逐样本比对后才能给出 REUSE_DIRECTLY / TARGETED_RERUN_REQUIRED 分类
              # Table 6 可作历史形态基线继承；Table 7/8 待 P0-1 与 P1-6 解决

RECOMMENDED CONFIG:
方案 A —— P6("whole potted plant including the pot") + candidate_mode=per_instance
         + prompt_selection_mode=single + use_prompt_ensemble=false
         + ranking_policy=r1 + A6/A7/semantic_gate/temporal/reprompt 全关
         + refinement(SPNP→残差修复→GCFC) 显式开启（待 V3 验证）
理由：①R1 仅在该分支生效，②Phase 12/16/17 全部证据均在此配置下取得，
      ③P6 与 potted_clean GT 及花盆尺度参照自洽，④无需修改生产代码。

NEXT PHASE:
Phase 20B —— 先完成 4 项免 GPU 工作（P0-3 表格重算 / P0-2 掩膜比对 /
P0-1 找回盆径与尺度路径 / P1-9 修正 Phase 19A），再执行 V1/V2/V3 三组 GPU 运行
并计算 interaction，最后据结果将 FINAL_RAP_FSAM3_SPEC_DRAFT.md 升级为 LOCKED_CANDIDATE。
```

---

## 6. 本阶段边界声明

- 未修改 `生成RAP-FSAM3掩膜.py`、`ranking_policy.py`、2DGS / M2M3-GO / COLMAP / Mesh 实现；
- 未改变任何生产算法默认参数；
- 未运行 SAM3、2DGS 或任何 GPU 推理/训练；
- 未修改历史实验结果、GT、掩膜、CSV 或 manifest；
- 未修改论文 `.tex` 正文；
- 未新增算法模块，未重新选择 R1 权重；
- 未把 TEST32 无 GT 帧计算为分割准确率；
- 未把 GT10 开发集视为独立测试集；
- 未把 CLI 默认值直接当作所有历史实验的真实配置（全部以 `参数.json` 为准）；
- 未编造缺失的运行参数或实验来源（缺失一律标 `UNKNOWN` / `NOT VERIFIED`）。

## 7. 非 GPU 测试复现

```bash
cd 00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证
python3 -I -m pytest $(find . -name "test_*.py" -path "*/tests/*") -q
# Phase 20A 实测：150 passed in 74.57s
```