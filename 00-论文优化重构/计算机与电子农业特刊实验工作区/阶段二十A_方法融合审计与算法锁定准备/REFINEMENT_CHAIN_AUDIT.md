# 前景细化与几何修正链路审计 / REFINEMENT_CHAIN_AUDIT

**阶段：** Phase 20A
**审计问题：** SPNP → 残差修复 → 几何修正 是否为真实代码顺序？各模块开关？未开启时的行为？Phase 12/17 是否开启？论文 Table 5 消融是否对应？与 R1 同时启用是否有真实证据？

---

## 0. 结论

| 问题 | 结论 |
|---|---|
| 链路顺序是否为 SPNP → 残差修复 → 几何修正？ | **是**，真实代码顺序确认（`main()` Pass 2，L3184 → L3198 → L3203） |
| 三个模块默认是否开启？ | **全部默认关闭**（均为 `store_true`） |
| Phase 12 `P6_raw_baseline` 是否开启？ | **全部关闭**，且 21/21 帧逐字节证明未产生任何像素变化 |
| Phase 17A P00/P10/P01/P11 是否开启？ | **全部关闭** |
| **R1 与三个细化模块同时启用的真实运行是否存在？** | **不存在。零例。** |
| 最终状态 | **`INTEGRATION_NOT_YET_VERIFIED`** |

---

## 1. 真实执行顺序（`main()` Pass 2，代码行号）

```
Pass 1（逐帧，无全局变体）
  L2819-2920  候选生成 → 评分 → select_mask() → 存入 selected_by_stem

[若开启 A6] L2927-3006  跨视角共识 → consensus_result
[若开启 A7] L3011-3100  记忆传播     → memory_masks

Pass 2（逐帧，最终链路）
  L3109-3140  构造变体 {A1s, A6共识, A7记忆}，逐个 score_candidate()
  L3162       best_record = max(variant_records, key=lambda r: r.total_score)   ← R0 口径
  L3164       selected_mask = best_variant.mask
        ↓
  L3184-3193  apply_spnp_refinement(image, selected_mask, …)     ← SB-FBPR
        ↓
  L3198       repaired_mask, restored = repair_residual_mask(spnp_mask, args)   ← 残差/细结构修复
        ↓
  L3203-3208  final_mask, corrective_info = apply_corrective_geometry(repaired_mask, …)  ← GCFC
        ↓
  L3222       save_mask(final_mask, dirs["final"]/…)
```

**确认：顺序为 SPNP → 残差修复 → GCFC，与任务书给出的链路图一致。**

---

## 2. 每个模块的开关、输入输出、接受条件、未开启行为

### 2.1 SB-FBPR — `apply_spnp_refinement()`（L2061-2139）

| 项 | 内容 | 代码行 |
|---|---|---|
| 开关 | `--use_spnp_refinement`（`store_true`，默认关） | L709 |
| 后端开关 | `--spnp_backend`，默认 `"none"`；可选 `sam3_if_supported` / `postprocess_only`（`sam_family` 抛 `NotImplementedError`） | L710-714, L2103-2107 |
| 输入 | `selected_mask`（Pass2 变体选择后的掩膜） | L3186 |
| 正提示点 | `sample_points(mask, spnp_positive_points=5, True, spnp_ring_radius=15)`（距离变换 TopK） | L2082, L1827 |
| 负提示点 | `sample_points(mask, spnp_negative_points=8, False, 15)`（外环） | L2083 |
| 证据引导 | 若 `use_spnp_evidence_guidance` 且有 A6 geo_support → 改用 `build_spnp_evidence()`（默认关，L786） | L2072-2080, L3174-3183 |
| 重查询 | `sam3_box_refinement()`（`sam3_if_supported` 后端）：正框 + 负框 | L2091-2102, L1976 |
| 接受条件 | `refined_iou ≥ --spnp_min_refined_iou`（默认 **0.65**） | L724-726 |
| `postprocess_only` 接受条件 | `0 < remove.sum() ≤ max_remove`（`spnp_max_remove_ratio=0.15`）且需 `spnp_use_lower_negative` | L2114-2134 |
| **未开启行为** | `return mask, points, {"backend":"disabled","accepted":0,"reason":"module_off"}` —— **掩膜原样返回，逐像素不变** | **L2089-2090** |
| 后端为 `none` 行为 | `return mask, points, {"backend":..., "accepted":0, "reason":"backend_off"}` —— 同样原样返回 | L2108-2109 |

### 2.2 残差/细结构修复 — `repair_residual_mask()`（L2142-2175）

| 项 | 内容 | 代码行 |
|---|---|---|
| 开关 | `--use_residual_repair`（`store_true`）**且** `--thin_repair_mode != "none"`（默认 `"none"`） | L728, L730-734 |
| 输入 | `spnp_mask` | L3198 |
| 流程 | `binary_closing(closing_kernel=15)` → `binary_opening(opening_kernel=3)` → `residual = closed & ~opened` | L2145-2151 |
| 恢复判据（全部须满足） | ① `0 < area ≤ thin_repair_max_area_ratio×size`（默认 0.003）② 与 `binary_dilation(opened, kernel(5))` 相邻 ③ 若 mode ∈ {skeleton, shape_filter} 则 `elongation ≥ thin_repair_min_elongation`（默认 2.5） | L2154-2168 |
| 后处理 | `fill_holes`（默认 true）→ `keep_largest_component`（默认 true） | L2171-2174 |
| 返回 | `(repaired, restored)`；`restored` 用于写入 `恢复细结构像素` 列 | L3198, L3265 |
| **未开启行为** | `return mask, np.zeros_like(mask)` —— **掩膜原样返回** | **L2143-2144** |

> ⚠️ 副作用提示：`keep_largest_component`（默认开）在本函数**内部**执行。
> 若 `use_residual_repair=true` 但 `thin_repair_mode="none"`，函数直接早退，**最大连通域筛选也不会发生**。
> 也就是说 `残差修复掩膜/` 目录存在且文件齐全，**不代表任何修复发生**。

### 2.3 GCFC — `apply_corrective_geometry()`（L2374-2481）

| 项 | 内容 | 代码行 |
|---|---|---|
| 开关 | `--use_corrective_geometry`（`store_true`） | L760 |
| 后端 | `--corrective_geometry_backend`，默认 `colmap_tracks`（可选 `mask_shape`） | L761-765 |
| 依赖数据 | `colmap_observations` 仅在 `use_corrective_geometry AND backend=="colmap_tracks"` 时加载（`main()` L2786-2790） | L2786-2790 |
| 正修正 | `positive = points_to_support_mask(obs.points, shape, geometry_track_dilation=19) & ~base`；点数 < `geometry_positive_min_points`（12）则清零 | L2419, L2433-2435 |
| 负修正 | `unsupported = base & ~broad_support(dilation=29) & suspicious_region`；**默认关闭**（`geometry_enable_negative_correction` 默认 `False`）→ 恒为空 | L2436-2439, L772 |
| 合并 | `corrected = (base | positive) & ~unsupported` → `fill_holes` → `keep_largest_component` | L2442-2446 |
| **接受条件（三重守卫）** | ① `delta.any()` 否则 `no_corrective_delta` ② `delta_ratio ≤ geometry_correct_max_delta_ratio`（0.08）否则 `delta_guard` ③ `corrected_iou ≥ geometry_correct_min_iou`（0.75）否则 `iou_guard` | **L2451-2461** |
| 最终 | `final = corrected if accepted else base`（守卫不通过则回退原掩膜） | L2463 |
| 无 COLMAP 观测时 | `return base, {..., "原因":"missing_colmap_observation", "接受":0}` —— **原样返回** | L2410-2418 |
| **未开启行为** | `return base, {"后端":"disabled","接受":0,"原因":"module_off","几何修正像素比例":0.0}` —— **掩膜原样返回** | **L2389-2397** |

---

## 3. Fallback 汇总

| 模块 | 触发条件 | Fallback 行为 |
|---|---|---|
| SB-FBPR | 开关关 / 后端 `none` | 原掩膜透传 |
| SB-FBPR | 重查询 IoU < 0.65 | 拒绝细化，用原掩膜 |
| SB-FBPR | `postprocess_only` 删除像素 > 15% | `reason="remove_guard"`，用原掩膜 |
| 残差修复 | 开关关或 mode=`none` | 原掩膜透传 |
| GCFC | 开关关 | 原掩膜透传 |
| GCFC | `observation is None` | 原掩膜透传，`missing_colmap_observation` |
| GCFC | 守卫不通过 | **回退到 `base`**（不是回退到上一级，是回退到 GCFC 输入） |

**重要：** GCFC 的 `base` 就是 `repaired_mask`。因此若 SB-FBPR/残差修复已改变掩膜，GCFC 守卫失败时
**保留的是 `repaired_mask`，不是 `selected_mask`**。链路不会因 GCFC 失败而丢失前级改进。

---

## 4. 影响最终掩膜的参数清单

| 参数 | 默认 | 影响阶段 | 影响性质 |
|---|---|---|---|
| `use_spnp_refinement` / `spnp_backend` | false / none | SB-FBPR | 开关 |
| `spnp_positive_points` / `spnp_negative_points` / `spnp_ring_radius` | 5 / 8 / 15 | SB-FBPR | 提示点采样 |
| `spnp_min_refined_iou` | 0.65 | SB-FBPR | **接受阈值** |
| `spnp_max_remove_ratio` / `spnp_use_lower_negative` / `spnp_lower_band_ratio` / `spnp_lower_exg_threshold` | 0.15 / false / 0.70 / 0.0 | SB-FBPR(postprocess_only) | 删除守卫 |
| `use_residual_repair` / `thin_repair_mode` | false / none | 残差修复 | 开关 |
| `closing_kernel` / `opening_kernel` | 15 / 3 | 残差修复 | 形态学核 |
| `thin_repair_max_area_ratio` / `thin_repair_min_elongation` | 0.003 / 2.5 | 残差修复 | 恢复判据 |
| `fill_holes` / `keep_largest_component` | true / true | 全部三级 | 公共后处理 |
| `use_corrective_geometry` / `corrective_geometry_backend` | false / colmap_tracks | GCFC | 开关 |
| `geometry_track_dilation` / `geometry_unsupported_dilation` | 19 / 29 | GCFC | 支撑场尺度 |
| `geometry_positive_min_points` | 12 | GCFC | 正修正门槛 |
| `geometry_correct_max_delta_ratio` / `geometry_correct_min_iou` | 0.08 / 0.75 | GCFC | **接受守卫** |
| `geometry_enable_negative_correction` | **false** | GCFC | 负修正总开关（论文称"负删除默认禁用"，与代码一致 ✓） |
| `use_spnp_evidence_guidance` | false | SB-FBPR | 证据引导 |
| `semantic_pot_lower_start_ratio` / `semantic_side_band_ratio` | 0.58 / 0.18 | GCFC | 可疑区带定义 |

---

## 5. Phase 12 / Phase 17 的实际开启状态

### 5.1 Phase 12 `P6_raw_baseline` — 全关，且产物逐字节证实未生效

`参数.json` 与 `参数/phase12_p6_raw_baseline.json` **两份独立记录**均为：
```json
"use_spnp_refinement": false,  "use_residual_repair": false,
"use_corrective_geometry": false
```

**Phase 20A 实测逐字节核对（21 帧）：**
```
选择后掩膜 / 正负提示细化掩膜 / 残差修复掩膜 / 几何修正掩膜 / 最终掩膜 / A5c_final_mask
→ 21/21 帧六目录 MD5 完全一致
```
`提示词选择.csv` 21/21 帧记录 `SPNP后端=disabled, SPNP原因=module_off`。

**判定：`VERIFIED_INACTIVE`。目录齐全是保存逻辑（L3212 无条件保存）造成的假象。**

### 5.2 Phase 17A（GT10 pilot，10 帧 × 4 变体；TEST32 × 4 变体）

`P00_Control/参数.json` 与 Phase 17A 报告 §4 一致：
`use_spnp_refinement=false, spnp_backend="none", use_residual_repair=false, thin_repair_mode="none",
use_corrective_geometry=false, use_geometry_feedback=false, geometry_feedback_mode="none"`

**判定：`VERIFIED_INACTIVE`。**

### 5.3 真正开启过细化的运行（27 份）—— 全部在 `score_select` 分支

详见 `R1_CALL_PATH_AUDIT.md` §4.1 表格。要点：

- 27/27 份的 `prompt_selection_mode = "score_select"`；
- 27/27 份的 `ranking_policy` 键不存在（早于 R1 提交）；
- 其中 4 份为 P6 + score_select + SPNP（`E1_{CaoMei1,XianKeLai1,DouBanLv1,ChangShouHua2}_P6带盆整株_101帧`）。

**这些运行是"R1 路径之外"的产物，不能用于证明 R1 与细化模块兼容。**

---

## 6. R1 × 细化模块的兼容性证据矩阵

| 组合 | 是否有真实运行 | 证据 |
|---|---|---|
| R1 + SB-FBPR | **否** | 0 份 `参数.json` 同时含 `ranking_policy ∈ {r1,r2,r3}` 与 `use_spnp_refinement=true` |
| R1 + 残差修复 | **否** | 同上 |
| R1 + GCFC | **否** | 同上 |
| R1 + 三者全开 | **否** | 同上 |
| R0 + 三者全开 | **是**（`score_select` 分支） | `XianKeLai1_A5c_语义门控几何修正_GT1冒烟`、`E3v2_KongQueZhuYu_GT5_A5c_完整RAPFSAM3v2`、`E4b_downstream/{CaoMei2,KongQueZhuYu,XianKeLai1}`、`E1_Representative4_PottedPlant_VFM/rap_runs/RAP-FSAM3-v2/{4 样本}` |

**判定：`INTEGRATION_NOT_YET_VERIFIED`。**

---

## 7. 一个此前未被记录的关键代码交互风险

```python
def select_mask(...):
    if args.use_prompt_ensemble and args.prompt_selection_mode == "score_select":
        ...                       # ← L1682，早退
        best = max(selectable, key=lambda row: row.total_score)   # ← R0
        return ...
    ...
    if len(default_cands) > 1 and args.candidate_mode == "per_instance":
        best_rec = select_best_policy(default_recs, ...)          # ← L1725，R1
```

- 开启 `--use_spnp_refinement` **本身不改变** `use_prompt_ensemble` / `prompt_selection_mode`；
- 但历史上**所有**开启 SPNP 的运行都同时设置了 `use_prompt_ensemble=true` + `prompt_selection_mode=score_select`；
- 若有人"照抄历史命令"并只追加 `--use_spnp_refinement`，会**静默落入 R0 选择路径**，
  得到"R1 + 细化"的错误组合，且日志中 `主评分` 列看起来正常（CSV 有 R1 列，但最终掩膜由 R0 选出）。

**这是一条非显然的陷阱，必须写入 P0 清单与最终算法规格的显式约束。**

---

## 8. 论文 Table 5 消融与本链路的对应关系

论文 Table 5（`tab:rap-ablation`，L641-662）五行：

| 论文行 | SG-MPS | SB-FBPR | GCFC | 本审计可对应的真实运行 |
|---|---|---|---|---|
| Fixed plant-instance prompt | – | – | – | Phase 12 `P6_raw_baseline`（P6 单提示词、零细化）**最接近** |
| Geometry-only multi-prompt selector w/o SG-MPS | No | – | – | **未定位到运行** |
| Semantic-gated selected prior | Yes | – | – | `E3v2_KongQueZhuYu_GT5_A1s_语义门控选择`（但 GT5、score_select、含 SPNP） |
| RAP-FSAM3 w/o GCFC | Yes | Yes | No | `E3v2_KongQueZhuYu_GT5_A5c_完整RAPFSAM3v2`（全开）**无法直接给出 w/o GCFC** |
| RAP-FSAM3 (ours, full) | Yes | Yes | Yes | 同上（但含 A6/A7/证据引导等额外变量） |

**关键判定：论文 Table 5 的五行无法在冻结实验产物中一一对应。**
- "Fixed plant-instance prompt" 行与 Phase 12 P6 基线**指标不匹配**（论文 0.9692 vs Phase 12 的 0.983498）；
- 语义门控开启的运行全部 `score_select`，其"选择"用的是 R0，与论文"语义门控选择到 P6 后再细化"的叙事不一致；
- 无任何一次运行是"仅 SB-FBPR 开启、GCFC 关闭"的干净配置。

**故 Table 5 的可继承性判定为 `REUSE_AS_LEGACY_BASELINE`，需重做，见 `RECONSTRUCTION_EVIDENCE_REUSE_MATRIX.csv`。**

---

## 9. 本阶段边界

- 未修改任何生产代码或历史产物。
- 未开启任何模块（未运行 GPU 推理）。
- 未把"目录存在"当作"模块生效"—— 全部结论以 `参数.json` + 逐字节比对为准。
- 未对 TEST32 无 GT 帧计算准确率。