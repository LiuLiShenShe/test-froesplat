# 方法配置逐模块对照审计 / METHOD_CONFIGURATION_AUDIT

**阶段：** Phase 20A
**原退稿论文：** `00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat__Foreground_Object_2D_Gaussian_Splatting_via_Semantic_Mask_Guided_Task_Reformulation_for_Plant/ForeSplat.tex`（850 行）
**生产脚本：** `00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py`（3381 行，HEAD `6df0c3df`）
**审计原则：** `argparse` 默认关闭 **不等于** 实验未启用。必须以 `参数.json` / 命令 / 掩膜产物为准。

---

## 0. 状态词定义

| 状态 | 含义 |
|---|---|
| `VERIFIED_ACTIVE` | 有代码入口 + 有历史运行以启用状态执行 + 有产物佐证 |
| `VERIFIED_INACTIVE` | 有代码入口 + 全部历史运行均为关闭状态 + 产物证实未生效 |
| `OPTIONAL` | 有代码入口 + 至少一次运行启用 + 明确非默认 |
| `LEGACY_ONLY` | 仅存在于历史更早的阶段，当前主链路不再使用 |
| `UNVERIFIED` | 代码存在但找不到运行配置或产物佐证 |

---

## 1. 逐模块对照表（论文 Methods ↔ 真实代码 ↔ 历史实验）

| 模块 | 原论文描述（ForeSplat.tex 行号） | 真实代码入口 | 开关（CLI） | 历史实验实际状态 | R1 后状态 | 结论 |
|---|---|---|---|---|---|---|
| FFT 帧质量筛选 | §2.3.1 L177–187，式(1) | `02-FFT/blur_filter.py::fft_blur_score()` L30-50；`smart_filter()` L100-141；`process_folder()` L192 | 独立脚本，非本入口 | **已执行**：20 样本各一份 `filter_log.json`（如 `02-FFT/BaiZhang/filter_log.json`：total 250 / removed 36 / kept 214 / `blur_percentile=25.0`） | 与 R1 无关 | `VERIFIED_ACTIVE`（但**公式与论文不一致**，见 §2.0） |
| SAM3 多提示词候选生成（K=5） | §2.3.2 L189–197 | `infer_candidates()` L1045；`candidate_mode` 默认 `per_instance` L630 | `--candidate_mode`, `--sam3_mask_threshold` | **已执行**（K=5 与 K=1 两种都真实跑过） | 不变 | `VERIFIED_ACTIVE` |
| 语义门控多提示词选择（SG-MPS） | §2.3.3 L199–240，式(2)–(4)，**含 softmax + 温度 τ_s** | `build_semantic_gate_context()` L1384；`semantic_gate_scores()` L1461；被 `score_candidate()` L1623-1624 调用 | `--use_semantic_gate`（`store_true`，默认关） | **25 份运行开启，全部为 `score_select` + `gate_backend=heuristic`** | 开启时其得分进入 `semantic_total` → 仅进入 `total_score`，**不进入 R1 主评分** | `OPTIONAL` + **公式 GAP**（见 §2.1） |
| 最终候选选择 | §2.3.3 末 M*_i = M_i^(k*) | `select_mask()` L1675-1739 | `--prompt_selection_mode`, `--use_prompt_ensemble`, `--ranking_policy` | `single`+`per_instance`（P6）与 `score_select`（P1–P5）**两类都真实跑过** | **`single` 分支用 R1；`score_select` 分支绕过 R1** | `VERIFIED_ACTIVE` / **分支不一致（见 `R1_CALL_PATH_AUDIT.md`）** |
| 结构平衡正负提示细化（SB-FBPR） | §2.3.4 L242–273，式(5)–(8) | `apply_spnp_refinement()` L2061-2139；`sam3_box_refinement()` L1976；`sample_points()` L1827 | `--use_spnp_refinement`（默认关）+ `--spnp_backend` | **27 份运行开启，100% 为 `score_select`**；Phase 12 P6 与 Phase 17A 全部关闭 | **从未在 R1 生效路径上运行过** | `OPTIONAL` + **R1 联合未验证** |
| 残差/细结构修复 | 论文未单列（并入 refinement） | `repair_residual_mask()` L2142-2175 | `--use_residual_repair` + `--thin_repair_mode` | 开启的运行 100% 为 `score_select` | 同上 | `OPTIONAL` + **R1 联合未验证** |
| 几何一致性前景校正（GCFC） | §2.3.5 L275–298，式(9)–(11) | `apply_corrective_geometry()` L2374-2481 | `--use_corrective_geometry` + `--corrective_geometry_backend` | 开启的运行 100% 为 `score_select`；Phase 12 P6 与 Phase 17A 关闭 | 同上 | `OPTIONAL` + **R1 联合未验证** |
| A6 跨视角共识 | 论文未描述（Phase 14 引入） | `apply_cross_view_consensus()` L234；Pass2 变体选择 L3109-3166 | `--use_cross_view_consensus` | 25 份运行开启，100% `score_select` | 变体选择用 `total_score` → 绕过 R1 | `OPTIONAL`，Phase 17A 判定为**降级/默认关** |
| A7 SAM3 记忆传播 | 论文未描述 | `propagate_memory_masks()` L419 | `--use_memory_propagation` | 同上 | 同上 | `OPTIONAL`，Phase 17A 判定为**降级/默认关** |
| 重提示检测 | 论文未描述 | `select_mask()` L1732-1733 / L1738；`reprompt_score()` L2178 | `--use_reprompt_detection` | **全部历史运行关闭**（78 份 `参数.json`） | 仅报警，不影响选择 | `LEGACY_ONLY`（已实现但从未启用） |
| 语义坍缩检测 | 论文未描述 | `detect_semantic_collapse()` L1745 | 无（纯函数） | **从未被调用** | — | `LEGACY_ONLY`（死代码） |
| 受控组合 plant_only / potted | 论文未描述 | `compose_plant_only()` L1771；`compose_potted()` L1793 | 无（纯函数） | **从未被调用** | — | `LEGACY_ONLY`（死代码） |
| A7 单 object ID 选择 | 论文未描述 | `select_single_object_id()` L1807 | 无（纯函数） | **从未被调用** | — | `LEGACY_ONLY`（死代码） |
| R1 候选排序 | **论文完全未描述** | `ranking_policy.py`（68 行，全文）；`score_candidate()` L1626-1637 | `--ranking_policy`（默认 `r1`，L657-662） | **无任何历史运行显式请求过**（78 份 `参数.json` 均无该键）；Phase 17C 以 `policy="base"` 重放历史并证明逐帧一致 | 当前默认 | `VERIFIED_ACTIVE`（作为代码）/ `UNVERIFIED`（作为已跑实验） |

---

## 2. 论文方法描述 ↔ 生产代码的结构性差异（GAP 项）

### 2.0 FFT 质量指标公式不一致 —— `INCONSISTENT`

**论文式(1)：**
```
Q_FFT(I) = ‖W_H ⊙ log(1+|F(I)|)‖₁ / (‖log(1+|F(I)|)‖₁ + ε)
```

**实现（`02-FFT/blur_filter.py` L36-50）：**
```python
f = np.fft.fft2(gray); fshift = np.fft.fftshift(f)
magnitude = np.abs(fshift)
radius = int(min(h, w) * 0.10)
low_freq_mask = ((X-cx)**2 + (Y-cy)**2) <= radius**2
total_energy = np.sum(magnitude**2) + 1e-8
low_energy   = np.sum(magnitude[low_freq_mask]**2)
high_energy_ratio = 1.0 - low_energy / total_energy
```

| 要素 | 论文 | 实现 | 判定 |
|---|---|---|---|
| 对数压缩 `log(1+|·|)` | 有 | **无** | 不一致 |
| 高通掩码 | `W_H ∈[0,1]^{H×W}` 径向 | 中心 **低频**圆盘（半径 = 10%·min(h,w)），取补 | 语义等价但形式不同 |
| 范数 | L1 | **平方能量 L2** | 不一致 |
| 归一化 | 除以总范数 | `1 − low/total` | 等价形式 |
| Laplacian 项 | **未提及** | **50% FFT + 50% Laplacian**（L141） | **论文遗漏** |
| COLMAP 连续性约束 | 未提及 | `max_consecutive_drop = 2`（强制保留） | **论文遗漏** |
| 分位数阈值 | 第一个四分位 | `blur_percentile = 25.0` | **一致 ✓** |

**影响：** 该模块位于候选生成之前，不影响 R1 判定；但论文方法节必须改写为实现口径，
否则属于公式–代码不一致（审稿人易查）。

### 2.1 SG-MPS 的 softmax 温度选择不存在 —— `INCONSISTENT`（高优先级）

**论文 §2.3.3 描述：** 门控置信度 `c_i^(k)=σ(...)` → 温度 softmax `a_i^(k)=exp(c/τ_s)/Σexp(c/τ_s)` → `k*=argmax a`。

**实现：** `semantic_gate_scores()` L1461 返回**加性启发式字典**
`{semantic_total, target_box_score, vertical_coverage_score, pot_overlap_penalty, side_distractor_penalty, center_prior_score}`，
`semantic_total` 直接以加法进入 `total = base_total + semantic_total − penalty`（L1625）。

**全仓检索证据：**
```
$ grep -n "softmax\|temperature\|tau_s\|exp(" 生成RAP-FSAM3掩膜.py
1591:    q_sam = 1.0 / (1.0 + math.exp(-6.0 * (float(candidate.sam_score) - 0.5)))
```
→ **除 SAM3 分数的 sigmoid 外，代码中不存在任何 softmax / 温度参数。**

**判定：论文的"温度控制 softmax 多提示词选择"在生产代码中无对应实现。
该差异必须在重写稿中处理（改写方法节 或 补实现+补实验）。**

### 2.2 论文的 P2 与生产 P2 文本不一致 —— `INCONSISTENT`

| 来源 | P2 文本 |
|---|---|
| 论文 Table 4 / §2.3.2 | `entire potted plant`（"带盆整株"） |
| 生产 `BUILTIN_PROMPTS` L39 | `P2 = "entire plant excluding pot"` |
| 生产 `BUILTIN_PROMPTS` L42 | `P6 = "whole potted plant including the pot"` |

**全仓 78 份 `参数.json` 中，`prompt_texts_json` 一项无一被设置（0 例）。**
即：**所有历史运行都使用内置 P2 = "entire plant excluding pot"。**
论文中标注为 `entire potted plant` 且 F1=0.9692 的那条提示词，其文本对应的是 **P6**，不是 P2。

**判定：论文的提示词编号体系与生产提示词典不一致，属必须澄清项。**

### 2.3 论文"5 个提示词"与生产的 P6 未纳入 K=5

论文 K=5 = {P1, P2, P3, P4, P5}（绿区/整株/器官/幼苗/去背景）。
所有 `score_select` 运行的 `prompt_ids` 确为 `['P1','P2','P3','P4','P5']` ✓。
但 **P6（带盆整株）在论文的 K=5 中不存在**，而：
- Phase 12 正式验收使用 P6；
- Phase 16/17A/17B/17C 全部困难集证据使用 P6；
- `E1_*_P6带盆整株_101帧` 系列使用 P6。

**这是论文与当前最佳证据之间的核心口径分歧，见 `METHOD_SELECTION_DECISION.md`。**

---

## 3. 关键实验实际配置（逐份 `参数.json` 核对，非 CLI 默认值推断）

### 3.1 Phase 12 `P6_raw_baseline`（21 帧，正式冻结基线）

来源：`阶段十二_GT_v2_QA与P6正式验收/P6_raw_baseline/参数.json`
交叉印证：`阶段十二_GT_v2_QA与P6正式验收/参数/phase12_p6_raw_baseline.json`

| 字段 | 值 | 证据 |
|---|---|---|
| `prompt_list` | **P6** | 两份 JSON 一致 |
| `default_prompt_id` | **P6** | 两份 JSON 一致 |
| `prompt_ids` | `["P6"]` | `参数.json` |
| `candidate_mode` | `per_instance` | 两份 |
| `prompt_selection_mode` | `single` | `参数.json` |
| `use_prompt_ensemble` | `false` | 两份 |
| `ranking_policy` | **键不存在**（早于 R1 提交） | `参数.json` 键表中无此项 |
| `use_semantic_gate` | `false` | 两份 |
| `use_spnp_refinement` | `false` | 两份 |
| `use_residual_repair` | `false` | `参数.json` |
| `use_corrective_geometry` | `false` | 两份 |
| `use_cross_view_consensus` | `false` | 两份 |
| `use_memory_propagation` | `false` | 两份 |
| `use_temporal_alignment` | `false` | 两份 |
| `use_reprompt_detection` | `false` | 两份 |
| `score_weights` | `area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5` | 两份 |
| `base_cleanup_mode` | `fsam3_basic` | `参数.json` |

**产物侧独立佐证（Phase 20A 实测）：**
1. 6 个掩膜目录 `选择后掩膜 / 正负提示细化掩膜 / 残差修复掩膜 / 几何修正掩膜 / 最终掩膜 / A5c_final_mask`
   的 **21/21 帧全部逐字节相同**（MD5 逐帧比对）→ refinement 链路在本次运行中**完全未产生任何像素变化**。
2. `提示词选择.csv` 中 `SPNP后端 = disabled`、`SPNP原因 = module_off`（21/21 帧）。

**结论：Phase 12 P6 基线是"单提示词 P6 + 每实例 + 零细化"的原始基线。其命名 `P6_raw_baseline` 与实际配置一致。**

### 3.2 Phase 19A 关于 `P6_raw_baseline` 的解释 —— **REFUTED**

Phase 19A 在 5 处声称该运行"在默认 `prompt_list = P2` 配置下产出"：

| 文件 | 原文要点 |
|---|---|
| `Phase19A_formula_audit.md` L178, L183, L256 | "其指标 f1_mean=0.983498 在**生产有效配置**下产出，即默认 `prompt_list=P2`" |
| `Phase19A_exclusion_list.md` L83-84 | "produced under production-valid config with the **default `prompt_list = "P2"`**" |
| `Phase19A_method_blueprint.md` L52 | "produced under default `prompt_list=P2` config" |
| `Phase19A_experiment_asset_registry.csv` P12-A01 | "the P6_raw_baseline RUN was produced under production-valid config with default prompt_list=P2" |
| `Phase19A_report.md` L106 | "P6_raw_baseline 不得误标为 P6 提示集输出（实为默认 P2 提示集冻结运行）" |

**反驳证据（两份独立归档 + 运行时产物）：**

| 证据 | 内容 |
|---|---|
| `P6_raw_baseline/参数.json` | `prompt_list = "P6"`, `default_prompt_id = "P6"`, `prompt_ids = ["P6"]` |
| `参数/phase12_p6_raw_baseline.json`（独立的命令行参数快照） | `"prompt_list": "P6"`, `"default_prompt_id": "P6"` |
| `P6_汇总.json` | 产物命名为 `P6_*`（`P6_全部21帧.csv` / `P6_汇总.json` / `P6_回归帧.csv` / `P6_三失败帧.csv`） |
| `P6_全部21帧.csv` 的 `p6_path` 列 | 全部指向 `P6_raw_baseline/最终掩膜/mask_*.png` |
| `指标/计算P6指标.py` | 评估脚本自身以 P6 命名 |
| 全仓 0 例 `prompt_texts_json` 覆盖 | P2 的文本恒为 "entire plant excluding pot"，与带盆整株语义不符 |

**判定：Phase 19A 的 `prompt_list=P2` 解释被运行快照直接推翻。任务书提出的怀疑成立。
`P6_raw_baseline` 确系 P6 提示集运行。Phase 19A 相关条款应视为错误，不得沿用。**

### 3.3 Phase 19A 关于 GCFC 的解释 —— **REFUTED**

`Phase19A_algorithm_delta.md` 第 7 行称：
> "The RAP-FSAM3 *evaluated pipeline* (Phase 12 P6_raw_baseline, which produced the reported F1=0.9706 masks) applied the geometry-consistency correction as part of the pipeline."

**反驳证据：**
1. `P6_raw_baseline/参数.json`：`use_corrective_geometry = false`；
2. `P6_raw_baseline/参数/phase12_p6_raw_baseline.json`：`"use_corrective_geometry": false`；
3. **逐字节比对：`几何修正掩膜/` 21/21 帧 == `最终掩膜/` 21/21 帧** → GCFC 未产生任何像素变化；
4. `几何修正掩膜/` 目录虽然存在，但按设计总会被写出（`main()` L3212 在 `save_intermediate_masks` 下无条件保存），
   **目录存在 ≠ 模块生效**。

**判定：Phase 19A §F7 "描述被评估配置（GCFC applied）" 的建议会造成论文失实，必须否决。
在 Phase 12 P6 基线上 GCFC 未生效。**

> ⚠️ 但需注意：F1=0.9706 是**论文 Table 3 的 RAP-FSAM3 数字**，
> 而 Phase 12 P6 基线的 F1 是 **0.983498**（21 帧）。二者不同，
> 说明 0.9706 另有来源 —— 该来源的溯源见 `RECONSTRUCTION_EVIDENCE_REUSE_MATRIX.csv`
> 与 `P0_BLOCKERS_AND_NEXT_ACTIONS.md` 的 P0-3（本次 Phase 20A 尚未定位到该数字的生成运行）。

---

## 4. 生产默认值 vs 推荐论文配置（差异已明确）

| 参数 | 生产默认（HEAD） | Phase 12/17 实际实验 | 论文若采方案 A 的建议 |
|---|---|---|---|
| `default_prompt_id` / `prompt_list` | **P2** | **P6** | **P6**（需显式 `--prompt_list P6 --default_prompt_id P6`） |
| `candidate_mode` | `per_instance` | `per_instance` | 同（无需改动） |
| `prompt_selection_mode` | `single` | `single` | 同 |
| `use_prompt_ensemble` | `false` | `false` | 同（**必须 false，否则 R1 被绕过**） |
| `ranking_policy` | `r1` | 键不存在（≡ `base`） | `r1` |
| `use_semantic_gate` | `false` | `false` | 待定，见 §5 |
| `use_spnp_refinement` | `false` | `false` | 若保留 SB-FBPR 则须开启 → **触发 `score_select`，R1 被绕过** |
| `use_residual_repair` | `false` | `false` | 同上 |
| `use_corrective_geometry` | `false` | `false` | 同上 |
| `use_cross_view_consensus` / `use_memory_propagation` | `false` | `false` | 保持 false（Phase 17A 证据支持） |

---

## 5. 未决问题（不在本阶段解决）

| 编号 | 问题 | 状态 |
|---|---|---|
| U1 | 论文 Table 3/5 中 RAP-FSAM3 的 F1=0.9706 / mIoU=0.9641 / HD95=183.42 / B-F1=0.4643 / leakage=0.001739 由**哪一次运行**产生？ | `UNVERIFIED` |
| U2 | 论文 Table 4 中 F1=0.9692 的"plant-instance prompt"是 P2 还是 P6？ | 倾向 P6，但未定位生成运行 |
| U3 | SG-MPS 论文描述的 softmax/τ_s 是历史真实实现（代码已被覆盖）还是论文撰写时的美化？ | `UNVERIFIED`，无 git 历史可查（`ranking_policy.py` 为 Phase 17C 新建；主脚本更早版本需 git log 追溯） |
| U4 | `candidate.track_mask` 从未被赋值 → `risk_penalty += 0.15` 为死分支。论文是否声明了该项？ | 代码事实已确认；论文未提及 → 一致 |

---

## 6. 复现命令

```bash
# 生产默认值（不依赖 GPU）
sed -n '613,804p' \
  00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py

# 全仓参数.json 交叉扫描（78 份）
python3 -I - <<'EOF'
import json, pathlib
for p in pathlib.Path('.').rglob('参数.json'):
    if '.git' in p.parts: continue
    d = json.loads(p.read_text(encoding='utf-8'))
    on = {k: d.get(k) for k in ('use_spnp_refinement','use_residual_repair','use_corrective_geometry') if d.get(k) is True}
    if on: print(p, on, d.get('prompt_selection_mode'), d.get('prompt_ids'), d.get('ranking_policy'))
EOF

# Phase 12 refinement 链逐字节同一性（Phase 20A 实测 21/21 相同）
python3 -I -c "
import hashlib, pathlib
d=['选择后掩膜','正负提示细化掩膜','残差修复掩膜','几何修正掩膜','最终掩膜','A5c_final_mask']
f=sorted(pathlib.Path('.../P6_raw_baseline/选择后掩膜').glob('*.png'))
print(sum(all(hashlib.md5((pathlib.Path(x)/f0.name).read_bytes()).hexdigest()
      == hashlib.md5(f0.read_bytes()).hexdigest() for x in d[1:]) for f0 in f), '/', len(f))
"
```