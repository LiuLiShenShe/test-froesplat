# 03 — RAP-FSAM3 修复日志

生成时间：2026-10-25
修复对象：`数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py`
新增文件：`…/脚本/sam3_prompt_adapter.py`

原则：**只修真实存在的缺陷，不改论文去迁就代码；每一处改动都留下可复现的证据。**

---

## 修复 1（P0-A）GCFC 后端 `colmap_tracks_gaussian` 从未加载 COLMAP 观测

### 症状

配置 `--corrective_geometry_backend colmap_tracks_gaussian` 时，`main()` 的观测加载条件只匹配 `colmap_tracks`：

```python
# 修复前
if args.use_corrective_geometry and args.corrective_geometry_backend == "colmap_tracks":
    colmap_observations = load_colmap_observations(...)
```

于是 `colmap_observations` 恒为空字典，`apply_corrective_geometry` 每帧走 `missing_colmap_observation` 分支并**静默接受原掩膜**。GCFC 在报告里"执行了"，但实际从未使用任何几何证据——这是最危险的一类缺陷：产物看起来正常，机制完全空转。

### 修复

```python
colmap_observations: dict[str, ColmapObservation] = {}
if args.use_corrective_geometry and args.corrective_geometry_backend in (
    "colmap_tracks", "colmap_tracks_gaussian",
):
    if args.colmap_dir is None:
        raise ValueError("--use_corrective_geometry requires --colmap_dir")
    colmap_observations = load_colmap_observations({p.stem for p in images}, args)
    expected = {p.stem for p in images}
    missing = sorted(expected - set(colmap_observations.keys()))
    if missing and args.strict_colmap_observations:
        raise RuntimeError(
            "GCFC strict mode: no COLMAP observation for frames "
            f"{missing}. Provide a COLMAP sparse model that registers these images."
        )
    if missing:
        print(f"[GCFC] WARNING: no COLMAP observation for {missing}")
```

配套：
- 共识路径改为 `colmap_observations if colmap_observations else load_colmap_observations(...)`，避免重复解析。
- 新增 `--strict_colmap_observations`：缺观测时 **raise**，不回落。用户要求"缺失时应报 BLOCKED 而非静默 fallback"，此开关即该语义。
- `colmap_mask_stem_candidates()`（`:2815`）逐 stem 生成候选名，兼容单场景与多场景 sparse 目录。

### 验证

KongQueZhuYu 5 帧实测注册点：1542 / 1891 / 631 / 98 / 5064，投影点全部落在图像范围内；`gcfc_has_real_observation=1` 5/5；$G_i$ = 0.9677–0.9918。

---

## 修复 2（P0-A）GCFC 拒绝原因被 `no_corrective_delta` 掩盖

### 症状

原因判定顺序中 `not delta.any()` 在 `support_gate` 之前，导致"支撑场根本没达到阈值"被报告成"没有修正增量"，掩盖了几何证据不足这一真实原因。

### 修复

```python
if args.corrective_geometry_backend == "colmap_tracks_gaussian" and support_gate_failed:
    reason, accepted = "support_gate", False
elif not delta.any():
    reason, accepted = "no_corrective_delta", False
elif delta_ratio > args.geometry_correct_max_delta_ratio:
    reason, accepted = "delta_guard", False
elif corrected_iou < args.geometry_correct_min_iou:
    reason, accepted = "iou_guard", False
```

### 验证

frame 0075（仅 98 个注册点）报 `no_corrective_delta`（真实：支撑场存在但未落在边界带内），其余 4 帧 `ok`。语义正确。

---

## 修复 3（P0-A）GCFC 缺少可视调试图

### 修复

新增 `save_gcfc_debug_artifacts(debug_dir, stem, before, after, support_field, boundary, positive)`（`:2960`），在 `apply_corrective_geometry(..., debug_dir=...)` 内调用，逐帧写出：

| 文件 | 内容 |
|---|---|
| `GCFC_before_<stem>.png` | 修正前掩膜 |
| `GCFC_support_field_<stem>.png` | $H_i$ 归一化灰度图 |
| `GCFC_boundary_band_<stem>.png` | $\mathcal B_i$ 边界带 |
| `GCFC_added_pixels_<stem>.png` | $M^{geo}\setminus M$ 新增像素 |
| `GCFC_after_<stem>.png` | 修正后掩膜 |

同时把 `boundary` 用 `boundary_band(base, args.geometry_boundary_band_px)` 真实物化（此前只在部分分支构造），确保调试图与实算区域一致。

---

## 修复 4（P0-B）SB-FBPR 发送的是 BOX 而不是论文要求的点提示

### 症状

论文 §2.3.4 定义 $\mathcal P_i^+=\operatorname{TopK}(d_{M^*},K_+)$、$\mathcal P_i^-=\operatorname{NMS}(\zeta,K_-,r_{nms})$——**点提示**。代码通过 `processor.add_geometric_prompt(box=...)` 发送**正负框**。距离变换 TopK 与 NMS 负采样算出来了，却从未送进模型。

### 第 0 步：先查 SAM3 到底支持什么

读 `第三方源码/SAM3-latest/sam3/model/geometry_encoders.py`：

```python
def append_points(self, points, labels, mask=None):   # :343
    # points: (N_points, bs, 2) 归一化 cx,cy
    # labels: (N_points, bs)    1=正, 0=负
def append_boxes(self, boxes, labels, mask=None):     # 类似
```

`SequenceGeometryEncoder.forward()` 调 `_encode_points()`（`:589`）与 `_encode_boxes()`，两者输出拼进同一 prompt token 序列。

**结论：SAM3 原生支持点提示，且可与 text prompt 共存。**

### 第 1 步：实测验证（不是读文档就下结论）

把点 append 到 `state["geometric_prompt"]` 后调用 `processor._forward_grounding(state)`，观测到**实例数与掩膜面积都发生变化**，确认点提示真的生效。

过程中踩到并解决的两个坑（保留作为接口证据）：

| 错误 | 原因 | 解法 |
|---|---|---|
| `AssertionError` in `Prompt.append_points` | 张量形状写成 `(1,2,2)` | 改为 `(N, 1, 2)`（`.view(-1,1,2)`），labels `(N,1)` |
| `RuntimeError: Inference tensors cannot be saved for backward` | 直接调 `model.forward_grounding` | 改用 `processor._forward_grounding`（带 `@torch.inference_mode()`） |
| `KeyError: 'scores'` | 直接调 model 不产出后处理字段 | 同上，processor 版会写 `state["scores"]` |

### 第 2 步：适配层

新增 `sam3_prompt_adapter.py`：

```python
def append_point_prompts(state, positive_points, negative_points, device):
    if "geometric_prompt" not in state:
        raise RuntimeError("geometric_prompt missing: call set_text_prompt() first")
    prompt = state["geometric_prompt"]
    if positive_points:
        prompt.append_points(normalized_points_tensor(positive_points, device),
                             labels_tensor([1]*len(positive_points), device))
    if negative_points:
        prompt.append_points(normalized_points_tensor(negative_points, device),
                             labels_tensor([0]*len(negative_points), device))
```

坐标归一化为 `(x/W, y/H)`，与 SAM3 内部约定一致。

### 第 3 步：论文口径的采样器

| 函数 | 实现 | 行 |
|---|---|---|
| `sample_points_from_region` | $\zeta$ = 区域距离变换；前 K 深点，贪心 NMS 抑制半径 $r_{nms}$ | `:2182` |
| `manuscript_negative_regions` | 返回 `ring` / `table` / `bg` / `neighbour` 四个子区，各自 `& ~mask`（保证花盆不被当负例） | `:2212` |
| `manuscript_negative_region` | 四区并集 | `:2257` |
| `sample_negative_prompts` | 每个非空子区保底 1 点，其余按深度填充；NMS 去聚簇；离图像边界 ≥1% 边长 | `:2270` |
| `build_sb_fbpr_prompts` | 组装正/负点集与调试用负区并集 | `:2344` |

### 第 4 步：接线

`sam3_box_refinement` 新签名：

```python
def sam3_box_refinement(image, mask, prompt_text, positive_points, negative_points,
                        processor, torch_module, args, evidence=None):
    prompt_form = str(getattr(args, "spnp_prompt_form", "boxes"))
    send_boxes  = prompt_form in ("boxes", "boxes+points")
    send_points = prompt_form in ("points", "boxes+points")
    ...
    output = processor.set_text_prompt(state=state, prompt=prompt_text)
    if send_boxes:  ... add_geometric_prompt(...)
    if send_points:
        appended = append_point_prompts(state, pos_norm, neg_norm, args.device)
        if appended["positive"] or appended["negative"]:
            output = processor._forward_grounding(state)
```

新 CLI：
- `--spnp_prompt_form {boxes,points,boxes+points}`，默认 `boxes`（保持历史 A6/A7 回放**字节级一致**）
- `--spnp_negative_region_mode {ring,manuscript}`，默认 `ring`
- `--spnp_negative_nms_radius`（默认 0 → 4% 图像边长）
- `--spnp_table_band_start_ratio`（0.58）/ `--spnp_side_band_ratio`（0.18）

### 验证

| 证据 | 结果 |
|---|---|
| `SB-FBPR提示证据.csv` | 5 帧全部 `points_sent_positive=5`、`points_sent_negative=8`、`boxes_positive=0` |
| 归一化坐标 | 全部落在 [0,1]，逐点落盘 |
| 历史回放 | `ranking_policy=base` + `prompt_form=boxes` 时，与阶段二十C `V0/Easy21` 掩膜 **0/21 不同**（字节级一致） |

---

## 修复 5（P0-C）SG-MPS 的 softmax 归一化维度错误

### 症状

论文 $a_i^{(k)}$ 的 $k$ 是**提示词索引**（$\sum_{l=1}^{K}$）。原实现按**候选实例**归一化：同一提示词返回 23 个实例时，每个实例各拿到一个概率，softmax 分母被实例数污染；提示词级概率失去意义。

### 修复

```python
best_by_prompt = {}
for rec in usable:
    v = float(getattr(rec, "gate_confidence", 0.0))
    best_by_prompt[rec.prompt_id] = max(best_by_prompt.get(rec.prompt_id, -1.0), v)
prompt_ids = sorted(best_by_prompt)
peak = max(best_by_prompt.values())
exps = {pid: math.exp((best_by_prompt[pid] - peak) / tau) for pid in prompt_ids}
prompt_prob = {pid: exps[pid] / sum(exps.values()) for pid in prompt_ids}
for rec in usable:
    rec.gate_softmax = float(prompt_prob.get(rec.prompt_id, 0.0))
```

- 分母 = 有可用候选的提示词数（P4 无检测时自动剔除）。
- 同一提示词的多个实例共享该提示词概率（论文的 $a_i^{(k)}$ 本来就只定义到提示词）。
- 无可用候选的提示词写 `-1.0` 哨兵，不参与 softmax。

### 验证

`逐帧报告.csv` 的 `sg_mps_softmax` 列 5 帧均满足 $\sum_k a_i^{(k)}=1$ 且只在有候选的提示词上非零；frame 0075 门控真实改选 P1（0.5396 vs P2 0.4604）。

---

## 修复 6（P0-C）SG-MPS 与 CDR 双重计权 `q_contrast` 的风险

### 分析

$q_{contrast}$ 同时出现在 $\phi_i^{(k)}$（论文 Eq. c）与 $S_{CDR}=S_{base}-q_{contrast}/5.5$（R1 排名）中。若两者都启用，同一物理量被计权两次，机制不再正交。

### 处置

**不改公式，改默认开关**：`--semantic_gate_use_contrast` 默认 `False`，代码中 `w_phi[4]=0.0`。因此：

- 门控 $c_i^{(k)}$ **不含**对比度，只由 $q_{plant},q_{area},q_{comp},q_{edge},q_{temp}$ 与 $\psi$ 决定；
- 对比度只经 CDR/R1 进入实例级排序。

两者分层不冲突：**SG-MPS 选提示词 → CDR 在该提示词内选实例**。

---

## 修复 7 SG-MPS 分层元数据缺失（可追溯性）

### 修复

`select_mask(..., selection_meta: dict | None = None)` 记录 `sg_mps_prompt`、`sg_mps_softmax`、`sg_mps_candidates`、`ranking_shortlist_prompt`、`selected_by_ranking_prompt`、`ranking_policy`、`selected_prompt`、`selected_instance`、`selection_path`。

`selection_path` 三值语义：

| 值 | 含义 |
|---|---|
| `sg_mps_then_ranking` | 门控先选提示词，排序再选实例 |
| `ranking_only` | 未开门控，排序直接选 |
| `default_prompt` | 无可用候选，回落默认提示词 |

---

## 修复 8 逐帧报告从 23 列扩到 39 列

按用户 §6 清单补齐 `sam3_inference_ran`、`selection_path`、`sg_mps_softmax`、`cdr_policy`、`sb_fbpr_pos/neg_points_sent`、`gcfc_registered_points/support_G/added_px`、`iou/hd95_if_gt_available` 等，使每一帧的"哪个机制真的跑了、选了什么、为什么被拒"都能从单个 CSV 回答。

---

## 修复 9 提示证据列的可读性缺陷

`point_coords_normalized` 用 `+`/`-` 前缀表示正负标签，坐标本身带负号时（不会发生，但读者会误判）极易读错。新增：

- `point_coords_unsigned`：真实 [0,1] 坐标；
- `point_labels`：`1`/`0`。

---

## 修复 10 死代码清理

`sample_negative_prompts` 中 `non_empty` 被计算并判空两次（第一次结果被立刻覆盖），删除冗余分支。**行为不变**，仅去掉误导性的死代码。改后 54/54 测试通过。

---

## 回归验证

```
阶段二十一_原稿方法全量实现与端到端验证/tests  54 passed in 2.88s
```

历史回放（`--ranking_policy base --spnp_prompt_form boxes --spnp_negative_region_mode ring`）与阶段二十C `V0/Easy21` 最终掩膜 **0/21 字节不同**，证明新增分支没有污染历史路径。

---

## 未修复 / 明确保留的问题

| 编号 | 问题 | 为什么不修 |
|---|---|---|
| N1 | `sam3_box_refinement` 名字仍含 `box` | 改名会破坏历史调用与测试；文档已说明其真实语义由 `--spnp_prompt_form` 决定 |
| N2 | $H_i$ 的 $\omega_{ij}=1$ | 论文未定义 $\omega$ 的计算方式；实现常数是最小假设，需论文先定 |
| N3 | $q_{plant}/q_{table}/q_{neighbour}$ 是启发式代理 | 换成学习打分器是**方法变更**，超出"恢复原稿"范围；已如实记录待论文限定措辞 |
| N4 | 正点采样附带 4% 图像边长的 NMS 间距 | 论文 $\mathcal P_i^+$ 未写间距；保留是为了点击多样性，属实现细节，需在论文中说明 |
| N5 | `METADATA_UNVERIFIED: P4/0` 警告 | P4（`crop seedling`）在多数帧 0 检测，缓存里存的是全零占位掩膜；评分阶段空掩膜硬门已拦截（`primary_score=0`），警告本身是**准确**的，不应消除 |
