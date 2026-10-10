# 最终 RAP-FSAM3 算法规格草稿 / FINAL_RAP_FSAM3_SPEC_DRAFT

> ## 状态：`DRAFT`
>
> **本规格不是 `LOCKED_CANDIDATE`。**
>
> 允许升级为 `LOCKED_CANDIDATE` 的前置条件（全部满足才可）：
> 1. V1（P6 + R1 + refinement OFF）正式运行完成并产出掩膜；
> 2. V3（P6 + R1 + refinement ON）正式运行完成并产出掩膜；
> 3. `interaction = (V3−V2) − (V1−V0)` 无实质负值（即细化与 R1 不冲突）；
> 4. 2DGS 输入掩膜与本规格 P6 掩膜的逐像素同一性已核对（`P0-2`）；
> 5. 尺度参照（盆径）数值与传递路径已落实（`P0-1`）。
>
> 状态若为 `BLOCKED`，则表示前置条件在**理论上不可满足**（例如尺度参照在数据中根本不存在）。
> 当前为 `DRAFT`：证据充分但联合运行缺失，且尺度项存在实证冲突。

---

## 0. 适用范围

本规格仅覆盖 **RAP-FSAM3 前景先验生成** 部分（Phase 20A 审计范围）。
2DGS / M2M3-GO / 网格 / 表型部分的状态见 `METHOD_VERSION_MATRIX.csv`，其中多项为 `UNVERIFIED`。

---

## 1. 算法总流程（推荐方案 A）

```
① FFT 帧质量筛选                [上游独立脚本 02-FFT/blur_filter.py]
       ↓  每序列按 25 分位阈值剔除模糊帧（附 max_consecutive_drop=2 连续性保护）
② SAM3 P6 候选生成              [生成RAP-FSAM3掩膜.py::infer_candidates()]
   prompt = "whole potted plant including the pot"
   candidate_mode = per_instance，sam3_mask_threshold = 0.5
   每个 SAM3 实例独立为 Candidate，不 OR 合并、不取最大连通域
       ↓
③ 逐候选特征与评分              [score_candidate(), L1540-1672]
   q_area / q_comp / q_edge / q_temp(=0.5) / q_contrast / q_leak(w=0) / q_side(w=0) / q_sam
   base_total = (1·q_area + 1·q_comp + 1·q_edge + 1·q_temp + 1·q_contrast
                 + 0·q_leak + 0·q_side + 0.5·q_sam) / 5.5
   risk_penalty = 0.25 if vertical_coverage < 0.30 else 0
                 (+0.15 track-overlap branch: 生产默认下为死分支，见 §5)
   semantic_total = 0（语义门控关闭）
       ↓
④ R1 候选选择                   [select_mask() L1717-1735 → select_best_policy() L1725]
   primary_score = base_total − q_contrast/5.5 − risk_penalty
   tie_break_score = 0
   argmax(primary_score, tie_break_score)，同分取先出现者
       ↓
⑤ 细化链（显式配置，当前建议 ON，见 §3）
   ⑤a SB-FBPR  [apply_spnp_refinement()]      后端 sam3_if_supported
   ⑤b 残差/细结构修复 [repair_residual_mask()]  thin_repair_mode = adjacent
   ⑤c GCFC    [apply_corrective_geometry()]   backend = colmap_tracks
       ↓
⑥ 最终前景掩膜 → foreground RGB / rgba / overlay / 最终掩膜/
```

---

## 2. 冻结参数

```json
{
  "prompt_list": "P6",
  "default_prompt_id": "P6",
  "use_prompt_ensemble": false,
  "prompt_selection_mode": "single",
  "candidate_mode": "per_instance",
  "sam3_mask_threshold": 0.5,
  "confidence_threshold": 0.3,
  "base_cleanup_mode": "fsam3_basic",
  "closing_kernel": 15,
  "fill_holes": true,
  "keep_largest_component": true,

  "score_weights": "area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5",
  "ranking_policy": "r1",
  "area_min_ratio": 0.01,
  "area_max_ratio": 0.80,
  "component_min_area_ratio": 0.0005,
  "vertical_coverage_min_ratio": 0.30,
  "leakage_max_bottom_fraction": 0.02,
  "leakage_max_side_fraction": 0.004,

  "use_temporal_alignment": false,
  "use_semantic_gate": false,
  "use_reprompt_detection": false,
  "use_cross_view_consensus": false,
  "use_memory_propagation": false,

  "use_spnp_refinement": true,
  "spnp_backend": "sam3_if_supported",
  "spnp_positive_points": 5,
  "spnp_negative_points": 8,
  "spnp_ring_radius": 15,
  "spnp_min_refined_iou": 0.65,
  "spnp_max_remove_ratio": 0.15,

  "use_residual_repair": true,
  "thin_repair_mode": "adjacent",
  "closing_kernel": 15,
  "opening_kernel": 3,
  "thin_repair_max_area_ratio": 0.003,
  "thin_repair_min_elongation": 2.5,

  "use_corrective_geometry": true,
  "corrective_geometry_backend": "colmap_tracks",
  "geometry_positive_min_points": 12,
  "geometry_track_dilation": 19,
  "geometry_unsupported_dilation": 29,
  "geometry_correct_max_delta_ratio": 0.08,
  "geometry_correct_min_iou": 0.75,
  "geometry_enable_negative_correction": false
}
```

**⚠️ 不变式（违反即视为不同配置）：**
- `use_prompt_ensemble` **必须为 `false`**。若为 `true` 且 `prompt_selection_mode="score_select"`，
  `select_mask()` 在 L1682 早退，**R1 被完全绕过**。
- `ranking_policy` **必须显式写出**。78 份历史 `参数.json` 均无此键（早于 R1 提交），
  省略会导致"历史产物被误当作 R1 产物"。

---

## 3. 为什么细化链标记为"建议 ON"而非"确认 ON"

| 证据 | 结论 |
|---|---|
| 27 份开启 refinement 的运行 100% 为 `score_select` | refinement 产物 **全部是 R0 口径** |
| 0 份运行同时含 `ranking_policy ∈ {r1,r2,r3}` 与 refinement 开关 | R1 × refinement **零联合证据** |
| Phase 12 P6 基线 21/21 帧 refinement 链逐字节恒等 | 该基线上 refinement **完全未生效** |
| V2/V3 计划（`FOLLOWUP_EXPERIMENT_PLAN.md`） | **未执行** |

**因此细化链的 ON/OFF 是待验证项，不是已确认项。**
在 V3 完成前，论文若保留 SB-FBPR / GCFC 叙述，必须标注"该模块未在 R1 配置下单独验证"。

---

## 4. 预期可复现的产物结构

```
<output_dir>/
  参数.json                  ← 权威配置记录（含 ranking_policy）
  候选掩膜/raw_instance_P6/mask_<stem>_<NN>.png
  候选掩膜/候选评分明细_<stem>.csv     ← 含 总分 / 主评分 / 平局决胜分
  选择后掩膜/mask_<stem>.png           ← R1 选择结果
  正负提示细化掩膜/mask_<stem>.png
  残差修复掩膜/mask_<stem>.png + thin_<stem>.png
  几何修正掩膜/mask_<stem>.png
  最终掩膜/mask_<stem>.png              ← 交付物
  前景图/crop_<stem>.png
  透明图/rgba_<stem>.png
  叠加图/overlay_<stem>.png
  提示词评分.csv / 提示词选择.csv / 重提示帧标记.csv / 几何修正提示.csv
  运行日志.json / 失败汇总.json
```

---

## 5. 已识别的规格偏差（需在论文中披露或修复）

| # | 偏差 | 影响 | 建议 |
|---|---|---|---|
| S1 | `Candidate` 无 `track_mask` 字段 → `risk_penalty += 0.15` 为**死分支**（L1611-1614） | 论文若声明该罚项，则与实现不符 | 论文只声明 0.25 分支；或补实现并验证 |
| S2 | `compose_plant_only()` / `compose_potted()` / `detect_semantic_collapse()` / `select_single_object_id()` **定义但从未调用** | 无法由 P6 得到 plant-only 变体 | 若论文需要 plant-only，需先实现并验证 |
| S3 | 论文 SG-MPS 的"温度 softmax"无实现（仓库无 softmax/temperature） | 方法–代码不一致 | 改写为加性启发式语义门控，或补实现+实验 |
| S4 | 论文 FFT 式(1) 与 `blur_filter.py` 实现不一致（无 log 压缩、用 L2 能量、多 50% Laplacian） | 公式–代码不一致 | 改写为实现口径 |
| S5 | 论文 Table 4 把 P2 标为 "entire potted plant"，归档为 "entire plant excluding pot" | 语义错标 | 改正；并披露 P2 实测不去盆 |
| S6 | M2M3-GO 的 S_m2g / S_g2m / B^∂ / U / keep-densify-prune **无实现**（仅 YAML） | 方法节公式无对应代码 | 方法节改写为实现的容量下限机制，或补实现 |
| S7 | Floor40 值 0.40 存在，但机制是**全局计数下限**而非论文的**局部块下限** | 公式–代码不一致 | 改写描述 |
| S8 | 尺度恢复（盆径）无实现代码、无数值 | 论文结论不可复现 | 补录数值 + 说明传递路径 |
| S9 | Table 7 的四个 wall-clock 时间无出处；"20 序列平均"无产物支撑 | 结果不可复现 | 重测或删除时间列 |

---

## 6. 证据强度声明

| 论断 | 强度 | 来源 |
|---|---|---|
| `denom = 5.5` | **PROVEN** | Phase 20A 对 182 条归档候选重算，181/182 精确一致（唯一 1 条差 = 恰好的 0.25 罚项） |
| R1 = `base_total − q_contrast/5.5 − penalty` | **PROVEN** | `ranking_policy.py` 全文 + 数值复核 |
| R1 修复 0039/0040/0041 | **SUPPORTED（开发集）** | Phase 17C + Phase 20A 独立重算一致；角色 DEV，非独立验证 |
| R1 在 Easy21 零回归 | **SUPPORTED（开发集）** | 同上 |
| R1 不回归到正确帧（DEV10 其余 7 帧） | **SUPPORTED（开发集）** | 同上 |
| R1 × SB-FBPR 兼容 | **NOT VERIFIED** | 零联合运行 |
| R1 × GCFC 兼容 | **NOT VERIFIED** | 零联合运行 |
| P6 含盆且与 `potted_clean` 一致 | **PROVEN** | Phase 20A 逐像素重算（盆召回 ≈97.5%，蓝块 0 px） |
| R1 在 `score_select` 路径不生效 | **PROVEN** | 代码 L1688 |
| TEST32 上 R1 行为安全 | **SUPPORTED（仅行为，无准确率）** | Phase 17C §9，churn 2/32 |