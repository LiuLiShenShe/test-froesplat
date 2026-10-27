# 01 — 论文↔代码映射（ForeSplat 原稿方法恢复）

生成时间：2026-10-25
权威论文版本：`ForeSplat__Foreground_Object_2D_Gaussian_Splatting_via_Semantic_Mask_Guided_Task_Reformulation_for_Plant/ForeSplat.md`（§2.3 原稿英文版，含 SG-MPS / SB-FBPR / GCFC）
生产入口：`数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py`
适配层：`…/脚本/sam3_prompt_adapter.py`（本轮新增）
FFT 模块：`/data/fj/F2DMAS/02-FFT/blur_filter.py`

本表只回答一个问题：**论文里的每一个公式/机制，在代码里的哪一行真的被执行**。状态口径见 §0。

---

## 0. 验收等级与状态口径

| 等级 | 含义 |
|---|---|
| L1 | 数学定义与论文公式一致 |
| L2 | 代码中存在真实实现（非注释、非空壳） |
| L3 | 该实现处于真实调用链上（有输入、有输出、被主流程调用） |
| L4 | 在真实数据上真实运行过（有落盘产物） |
| L5 | 有功能行为证明（开关前后产物不同，且差异可归因于该机制） |
| L6 | 有独立实验（独立数据集/独立运行，能给出该机制的量化贡献） |

状态取值：`VERIFIED`(L1–L6) / `IMPLEMENTED`(L1–L3) / `RUNNABLE`(L1–L4) / `PARTIALLY_VERIFIED`(部分等级) / `BLOCKED`。

⚠️ 本文件中的状态是**本轮工作的记录**，最终判定以 `10_FINAL_ACCEPTANCE.md` 为准。

---

## 1. §2.3.1 FFT 帧质量过滤（Eq. 1）

| 论文 | 代码 | 位置 | 状态 |
|---|---|---|---|
| $Q_{FFT}(I)=\dfrac{\lVert W_H\odot\log(1+\lvert\mathcal F(I)\rvert)\rVert_1}{\lVert\log(1+\lvert\mathcal F(I)\rvert)\rVert_1+\varepsilon}$ | `fft_blur_score_paper()` | `02-FFT/blur_filter.py:74` | VERIFIED (L1–L6) |
| $W_H$ 径向高通权重 | `radial_high_pass_weight()` | `02-FFT/blur_filter.py`（`cutoff_ratio=0.10, transition_ratio=0.10`） | VERIFIED |
| 逐序列第一四分位阈值 | `--blur_percentile 25` | 同上 CLI | VERIFIED |

- 两条公式并存：`fft_blur_score`（legacy，绝对高频和）与 `fft_blur_score_paper`（论文 Eq.1），由 `--fft_formula {legacy,paper}` 选择，默认 `paper`。
- 实测差异：KongQueZhuYu 210 帧中 33 帧判定不同（`阶段二十一_…/metrics/fft_formula_kqz.csv`）。
- ⚠️ 已知口径差异：生产入口 `生成RAP-FSAM3掩膜.py` **不调用** FFT，它假定输入目录已是过滤后的帧。FFT 是独立的上游阶段（`02-FFT/blur_filter.py`），由 `run_pipeline.py` 的 `fft` 阶段串联。

---

## 2. §2.3.2 多提示词候选生成

| 论文 | 代码 | 位置 |
|---|---|---|
| $K=5$ 提示词，$\mathcal M_i=\{M_i^{(1)},\dots,M_i^{(K)}\}$ | `--prompt_list P1,P2,P3,P4,P5` + `infer_candidates()` | `:1122` |
| 五个提示词文本 | `DEFAULT_PROMPT_TEXTS` / `--prompt_texts_json` | `load_prompt_texts():907` |
| 逐实例候选（非并集） | `--candidate_mode per_instance`，`_masks_scores_boxes():1064` | `:1064` |

| 提示词 | 文本 | 语义角色 |
|---|---|---|
| P1 | `green plant` | 绿色区域 |
| P2 | `entire potted plant` | 带盆整株（默认主提示词；§2.3.2 锁定的目标） |
| P3 | `leaves and stems` | 器官级（去盆，仅叶片/茎） |
| P4 | `crop seedling` | 幼苗取向 |
| P5 | `plant body without background` | 排除背景的植株 |

状态：**VERIFIED (L1–L6)**。R6 实跑 5 帧分别产出 P1..P5 候选 23/7/5/5/13 个（`runs/R6_RAP_FULL_real/候选掩膜/raw_instance_*`）。
注意：P4（`crop seedling`）在本场景多数帧返回 0 个可用实例，SG-MPS 中被标记为 `-1.0` 并自动从 softmax 分母中排除——这是真实行为，不是代码缺陷。

> **P2 目标口径（本轮修正）**：论文 §2.3.2 / §3.2 / 表 4 明确把植株实例定义为 `entire potted plant`（含可见花盆，F1=0.9692），而修正前代码是 `entire plant excluding pot`。这等于在没有告知的情况下把"带盆整株"改成了"纯植株"，并让 SG-MPS 的目标函数与论文相反。现已统一为 `entire potted plant`；`validate_prompt_contract()` 会在 P2 文本含 `excluding pot` / `without pot` 时直接抛错。P6（`whole potted plant including the pot`）、`POT`、`CUBE` 只保留为诊断提示词，不属于论文的 K=5 集合（`PAPER_PROMPT_IDS=("P1","P2","P3","P4","P5")`）。

---

## 3. §2.3.3 SG-MPS 语义门控多提示词选择

| 论文 | 代码 | 位置 | 状态 |
|---|---|---|---|
| $\phi_i^{(k)}=[q_{plant},q_{area},q_{comp},q_{edge},q_{contrast},q_{temp}]^\top$ | `phi = [...]`（`semantic_gate_scores`） | `:1658` | VERIFIED (L1–L4, L5 部分) |
| $\psi_i^{(k)}=[q_{table},q_{bg},q_{neighbour},q_{outlier}]^\top$ | `psi = [...]` | `:1659` | VERIFIED |
| $c_i^{(k)}=\sigma(w_\phi^\top\phi-w_\psi^\top\psi-b)$ | `logit = Σwφ·φ − Σwψ·ψ − bias; gate_confidence = sigmoid(logit)` | `:1665-1666` | VERIFIED |
| $w_\phi$ 默认 `0.5,0.3,0.15,0.15,0.0,0.1`，$w_\psi$ 默认 `0.3,0.3,0.3,0.2`，$b=0.5$ | `--semantic_phi_weights` / `--semantic_psi_weights` / `--semantic_gate_bias` | `:1660-1664` | VERIFIED |
| §2.3.3「可见花盆不是负类」⇒ $q_{bg}$ 必须排除花盆、且 $\phi$ 需要正向的包含花盆证据 $q_{pot}$ | `background_region = ~(target_mask | pot_mask)`；`phi` 追加 `q_pot`，权重 `--semantic_pot_weight`（默认 0.2） | `semantic_gate_scores` | VERIFIED (本轮修正) |
| $\phi$ 的 $q_{plant}$ 在带盆口径下是**植株区域的召回**，不是"候选里绿色像素的占比" | `green_fraction = |M∩green_ref|/|green_ref|`（`green_ref` 由候选并集支持的绿色分量给出） | `semantic_gate_scores` | VERIFIED (本轮修正) |
| 语义参考必须是**植株**，不是场景里最大的绿色块 | `largest_green_component_mask(image, args, support=union)`：取与候选并集重叠最大的绿色连通域 | `:1564` | VERIFIED (本轮修正) |
| 花盆带相对植株底部锚定 | `pot_top = plant_box.y0 + plant_height·semantic_pot_lower_start_ratio` | `build_semantic_gate_context` | VERIFIED (本轮修正) |
| $a_i^{(k)}=\operatorname{softmax}_{k}(c_i^{(k)}/\tau_s)$，**按提示词**归一化 | `apply_semantic_gate_softmax()`：先按 prompt 取 `max` 代表值，再跨 prompt 做 softmax | `:1695` | VERIFIED (L1–L5) |
| $k^*=\arg\max_k a_i^{(k)}$ | `semantic_gate_prompt()` | `:1733` | VERIFIED |
| 面积合理区间是选择约束而非仅诊断 | `--candidate_area_admissibility {off,gate,all}`；`gate_usable_records()` 跳过不可采纳候选 | `:1832` | VERIFIED (本轮修正) |
| 门控只选提示词，实例级排序交给 ranking | `select_mask(selection_meta=...)` 分层：`sg_mps_then_ranking` | `:1940` | VERIFIED |

> 受控开关：`--semantic_gate_target_mode {potted_plant,plant_only}`。默认 `potted_plant` 是论文口径；`plant_only` 精确复现修正前的"绿色参考 + 花盆算背景 + 无 $q_{pot}$"行为，**只用于消融**，把目标口径这一因素单独隔离出来。同一候选池下 `potted_plant` 与 `plant_only` 的差异就是"目标定义"的贡献（见 `05_RAP_FSAM3_ABLATION.md` §5）。

### 3.1 逐提示词归一化（用户 P0-C 项）——已修正并验证

本轮修正前，`apply_semantic_gate_softmax` 在**候选实例级**归一化；论文要求按**提示词**归一化。现实现：

```
best_by_prompt[prompt_id] = max over instances of that prompt of gate_confidence
prompt_prob[pid] = exp((best_by_prompt[pid] - peak)/τ_s) / Σ_l exp((best_by_prompt[l] - peak)/τ_s)
每个实例写回：rec.gate_softmax = prompt_prob[rec.prompt_id]
```

`逐帧报告.csv` 的 `sg_mps_softmax` 列直接落盘该分布，R6 实测：

| frame | SG-MPS softmax（按提示词） | 门控选中 | 最终选中 |
|---|---|---|---|
| 0000 | P1 0.2556 / P2 0.2600 / P3 0.2543 / P5 0.2301（P4 无候选） | P2 | P2#1 |
| 0025 | P1 0.2484 / P2 0.2549 / P3 0.2484 / P5 0.2483 | P2 | P2#1 |
| 0050 | P1 0.3305 / P2 0.3390 / P3 0.3305 | P2 | P2#0 |
| 0075 | P1 0.5396 / P2 0.4604 | **P1** | P1#0 |
| 0100 | P1 0.2499 / P2 0.2501 / P3 0.2500 / P5 0.2500 | P2 | P2#1 |

frame 0075 是真实提示词切换案例（P2 在该帧退化，门控改选 P1），证明 SG-MPS **确实影响提示词选择**（L5）。

**但"有影响"不等于"有益"**：同一帧按带盆 GT 的 F1 只有 0.9227（相邻帧 0.981–0.987），HD95 282.8 px；按纯植株 GT 反而 0.9800。跨场景 16 帧消融进一步显示门控只选对 2/16（oracle 全部为 P2）。详见 `05_RAP_FSAM3_ABLATION.md` §3–§4。

### 3.2 SG-MPS 与 CDR 不冲突（用户 P0-C 项）

分层关系：

```
第 1 层  SG-MPS  → 选提示词 k*
第 2 层  CDR/R1  → 在 k* 的实例集合内重排
```

证据：`--semantic_gate_use_contrast` 默认 **False**，因此 $w_\phi[4]=0$，$c_i^{(k)}$ 中**不含** $q_{contrast}$；对比度只通过 CDR 的 $S_{CDR}=S_{base}-q_{contrast}/5.5$ 进入决策。同一物理量不会在两个机制里被重复计权。

`selection_path` 列三种取值：`sg_mps_then_ranking`（门控+排序） / `ranking_only`（仅排序） / `default_prompt`（无可选候选，回落默认）。

### 3.3 诚实说明：$\phi,\psi$ 中三项是启发式代理

论文把 $q_{plant},q_{table},q_{neighbour}$ 描述为"语义证据"，代码中它们的**实际计算**是：

| 符号 | 实际实现 | 代理性质 |
|---|---|---|
| $q_{plant}$ | `0.5·target_box_score + 0.5·green_fraction`，`target_box_score` 为候选与"目标掩膜"（绿分量/语义框）的召回/精度均值，`green_fraction` 为 ExG>12 的像素占比 | 用颜色+框重叠代理植物语义 |
| $q_{table}$ | 候选在 `lower_band`（y>0.58H）内、且不在 pot/target 区域的像素占比 | 用图像下半区代理桌面区域 |
| $q_{neighbour}$ | `side_distractor_penalty`：候选落在左右侧带（各 0.18W）的像素占比 | 用图像侧带代理邻株 |
| $q_{bg}$ | 候选落在 `target_mask` 之外的像素占比 | 依赖 target_mask 质量 |
| $q_{outlier}$ | `1 − q_comp`（连通域惩罚的补） | 组件数代理离群碎块 |

这些**不是**学习到的语义打分器，是手工几何/颜色先验。论文 §2.3.3 的措辞（"measures plant-body semantic evidence"）容易被读成有监督语义模型，**必须在下轮论文同步时限定为 hand-crafted proxy**，见 `09_MANUSCRIPT_SYNC_REPORT.md` 的 M-SG-1 项。

---

## 4. §2.3.4 SB-FBPR 结构平衡前后景提示细化

| 论文 | 代码 | 位置 | 状态 |
|---|---|---|---|
| $\mathcal P_i^+=\operatorname{TopK}_{p\in M_i^*}(d_{M_i^*}(p),K_+)$ | `sample_points_from_region(mask, K_+, nms_radius=0)`：`distance_transform_edt(mask)`，取前 K 大 | `:2182` | VERIFIED (L1–L5) |
| $\mathcal R_i^-=(\operatorname{Dilate}(M_i^*,r)\setminus M_i^*)\cup\mathcal I_{table}\cup\mathcal I_{bg}\cup\mathcal I_{neighbour}$ | `manuscript_negative_regions()` 返回 `{"ring","table","bg","neighbour"}`，每个子区都 `& ~mask` | `:2212` | VERIFIED |
| 并集 | `manuscript_negative_region()` | `:2257` | VERIFIED |
| $\mathcal P_i^-=\operatorname{NMS}_{p\in\mathcal R_i^-}(\zeta_i(p),K_-,r_{nms})$ | `sample_negative_prompts(sub_regions, K_-, r_nms)`：每个非空子区保底 1 点，其余按面积加权填充，NMS 最小间距，边界安全 | `:2270` | VERIFIED |
| $\mathcal P_i=(\mathcal P_i^+,\mathcal P_i^-)$ → 重查询 VFM | `sam3_box_refinement()` → `append_point_prompts()` + `processor._forward_grounding()` | `:2404` / `sam3_prompt_adapter.py:71` | VERIFIED (L1–L5) |

### 4.1 用户 P0-B 项：SAM3 到底支不支持点提示

**结论：支持。** 实测接口链路（`第三方源码/SAM3-latest/sam3/model/geometry_encoders.py`）：

```
Prompt.append_points(points (N,bs,2) 归一化, labels (N,bs) 1/0)   # :343
        ↓
SequenceGeometryEncoder.forward → _encode_points()               # :589
        ↓
与 text prompt 的 token 序列拼接，一次 forward 同时使用
```

因此本轮**没有**退化成"用等价 box 代替点"，而是真的把正/负**点**送进模型：

| 证据 | 文件 |
|---|---|
| 每帧实际发送 5 正点 + 8 负点、0 个 box | `runs/R6_RAP_FULL_real/SB-FBPR提示证据.csv` |
| 归一化坐标 + 标签逐点落盘 | 5 帧 run：`point_coords_normalized`（前缀 ± 为标签）；210 帧 run 起：新增 `point_coords_unsigned` / `point_labels` 两列 |
| 提示形式开关 | `--spnp_prompt_form {boxes,points,boxes+points}`（默认 `boxes` 以保持历史字节级可复现） |
| 负区模式开关 | `--spnp_negative_region_mode {ring,manuscript}`（默认 `ring`） |

> 命名说明：函数名仍是 `sam3_box_refinement`（历史遗留）。它现在同时支持 box / point / box+point 三种提示形式，实际行为由 `--spnp_prompt_form` 决定。**未改名为** `sam3_prompt_refinement`，以免破坏历史调用与测试；语义以本表为准。

### 4.2 提示证据列的口径澄清

`SB-FBPR提示证据.csv` 的 `point_coords_normalized` 用前缀 `+`/`-` 表示**正/负标签，不是坐标符号**（例：`-0.860584,0.217244` 表示负点位于 x=0.8606,y=0.2172）。为避免误读，本轮新增两列（自 210 帧 run 起写入；5 帧 run 早于该改动，未重跑以保留历史产物）：
- `point_coords_unsigned`：全部点的真实 [0,1] 坐标；
- `point_labels`：`1`=正 / `0`=负。

---

## 5. §2.3.5 GCFC 几何一致性前景修正

| 论文 | 代码 | 位置 | 状态 |
|---|---|---|---|
| $H_i(p)=\sum_{X_j\in X_i}\omega_{ij}\kappa_\sigma(p-\pi_i(X_j))$ | `points_to_support_field()`：COLMAP 3D 点投影 + 高斯核，$\sigma$=`--geometry_gaussian_sigma 3.0` | `:2912` | VERIFIED (L1–L5) |
| $G_i=\dfrac{\langle H_i,M_i\rangle}{\lVert H_i\rVert_1+\varepsilon}$ | `apply_corrective_geometry()` 内 `support_G` | `:2987` | VERIFIED |
| $\mathcal B_i=\operatorname{Dilate}(M,r_g)\setminus\operatorname{Erode}(M,r_g)$ | `boundary_band(mask, r_g)`，`r_g=` `--geometry_boundary_band_px 12` | `:2947` | VERIFIED |
| $M^{geo}=M\vee[\mathbf 1(H>\tau_g)\odot\mathbf 1(p\in\mathcal B_i)]$ | 同上，$\tau_g$=`--geometry_support_threshold 0.15` | `:2987` | VERIFIED |
| 默认关闭负向删除 | `--geometry_enable_negative_correction` 默认 `False` | `:217` | VERIFIED |
| $\pi_i$、$X_i$ 来自 COLMAP | `load_colmap_observations()` → `load_colmap_loader()` | `:2869` / `:2759` | VERIFIED |
| $\omega_{ij}$ 单调假设 | 代码取 $\omega_{ij}=1$（所有注册点等权） | 同上 | **简化项，见 §8-S4** |

### 5.1 用户 P0-A 项：观测加载缺陷（已修复）

**缺陷（修复前）**：`main()` 只在 `--corrective_geometry_backend colmap_tracks` 时加载 COLMAP 观测；新后端 `colmap_tracks_gaussian` 拿到空字典 → 每帧静默回落 `missing_colmap_observation`，GCFC 等于从未执行。

**修复**：

```python
if args.use_corrective_geometry and args.corrective_geometry_backend in (
    "colmap_tracks", "colmap_tracks_gaussian",
):
    colmap_observations = load_colmap_observations(stems, args)
    missing = sorted(expected - colmap_observations.keys())
    if missing and args.strict_colmap_observations:
        raise RuntimeError("GCFC strict mode: ... Provide a COLMAP sparse model ...")
    if missing:
        print("[GCFC] WARNING: no COLMAP observation for ...")
```

- 多场景目录：`load_colmap_observations` 内部按 `colmap_mask_stem_candidates()`（`:2815`）逐个 stem 解析，支持单场景/多场景 sparse 目录。
- `--strict_colmap_observations`：有帧缺观测就 **raise**，而不是回落。
- 原因字段顺序修正：`support_gate` 先于 `no_corrective_delta` 判定，避免支撑不足被掩盖成"无修正增量"。

### 5.2 真实运行证据（R6，5 帧）

| frame | 注册点数 | $G_i$ | 新增像素 | 接受 | 原因 |
|---|---|---|---|---|---|
| 0000 | 1542 | 0.9752 | 151 | 1 | ok |
| 0025 | 1891 | 0.9892 | 227 | 1 | ok |
| 0050 | 631 | 0.9677 | 456 | 1 | ok |
| 0075 | 98 | 0.9897 | 0 | 0 | `no_corrective_delta` |
| 0100 | 5064 | 0.9918 | 223 | 1 | ok |

frame 0075 只有 98 个注册点，支撑场稀疏到不产生任何 $H>\tau_g$ 且落在边界带内的像素——这是**真实几何证据不足**，不是代码 bug。

### 5.3 调试产物（用户 P0-A 要求）

`apply_corrective_geometry(..., debug_dir=...)` + `save_gcfc_debug_artifacts()`（`:2960`）逐帧写出到 `几何修正提示图/`：

```
GCFC_before_<stem>.png
GCFC_support_field_<stem>.png
GCFC_boundary_band_<stem>.png
GCFC_added_pixels_<stem>.png
GCFC_after_<stem>.png
```

外加 `几何修正提示.csv` / `corrective_geometry_delta.csv`（逐帧 $G$、新增像素、接受/拒绝原因）。

---

## 6. 残余/细结构修复（论文未单列公式，属 §2.3.4 的工程实现）

| 机制 | 代码 | 位置 | 说明 |
|---|---|---|---|
| 开运算移除细结构 + 邻接恢复 | `repair_residual_mask()` | `:2684` | `--use_residual_repair`，`--thin_repair_mode {adjacent,elongation}` |

状态：**RUNNABLE (L1–L4)**；论文没有对应公式，因此没有 L1 可对照。R5 vs R4p 的消融给出量化贡献。

---

## 7. 实例级排序与 CDR/R1（论文 §2.3.3 末段引用，实际定义在 ranking policy）

| 机制 | 代码 | 位置 |
|---|---|---|
| $S_{base}$ | `score_candidate()` | `:1786` |
| $S_{CDR}=S_{base}-q_{contrast}/5.5$ | `ranking_policy.py`，`--ranking_policy r1` | 同目录 |
| 清洗与几何守卫 | `basic_cleanup()`, `collapse_area_threshold`, `leakage_*` | `:1011` |

状态：**VERIFIED**（阶段 17C，commit `d704faa8`；R1 救援 0039/0040/0041 至 F1≥0.95）。

> ⚠️ 论文 §2.3.3 末句写"ranking policy of Section 2.3.2"，但 §2.3.2 是多提示词生成，不含 ranking policy 定义。这是**论文内部交叉引用错误**，见 M-XREF-1。

---

## 8. 已记录的偏差清单（与阶段二十A `FINAL_RAP_FSAM3_SPEC_DRAFT.md` S1–S9 对齐）

| 编号 | 偏差 | 影响 | 处置 |
|---|---|---|---|
| S1 | FFT 不在生产入口内，是上游独立阶段 | 生产入口假定输入已过滤 | 由 `run_pipeline.py` 串联，不隐瞒 |
| S4 | $H_i$ 的 $\omega_{ij}$ 取常数 1 | 未对不确定点降权 | 论文需写明或实现加权 |
| S5 | 负向删除默认关闭（与论文一致） | 只做正向补洞 | 已由 `--geometry_enable_negative_correction` 显式控制 |
| S6 | $q_{plant}/q_{table}/q_{neighbour}$ 为启发式代理 | 见 §3.3 | 论文措辞需限定 |
| S7 | `sam3_box_refinement` 函数名与"point prompt"不符 | 仅命名问题 | 保留历史名，本表说明 |
| S8 | SB-FBPR 负区子区（ring/table/bg/neighbour）阈值是图像比例启发式 | 场景相关 | 暴露为 CLI |
| S9 | $\tau_s$ 默认 0.25 使 $a_i^{(k)}$ 接近均匀 | 门控主要在近 ties 时起作用 | 消融给出 $\tau$ 敏感性 |

---

## 9. 本轮实际修改的代码（对应 `03_RAP_FSAM3_FIX_LOG.md`）

| 改动 | 文件 | 目的 |
|---|---|---|
| 观测加载覆盖 `colmap_tracks_gaussian`；`--strict_colmap_observations` | 生产入口 `main()` | 修 P0-A |
| GCFC 拒绝原因顺序 + 5 张调试图 | `apply_corrective_geometry` / `save_gcfc_debug_artifacts` | 修 P0-A |
| SB-FBPR 改为真实 Distance-Transform TopK 正点 + NMS 负点 | `:2182-2404` | 修 P0-B |
| 点提示适配层 | `sam3_prompt_adapter.py`（新增） | 修 P0-B |
| SG-MPS 改为按提示词 softmax | `apply_semantic_gate_softmax` | 修 P0-C |
| 分层选择元数据 `selection_path` | `select_mask` | 修 P0-C |
| 提示证据新增无符号坐标 + 标签列 | `sam3_box_refinement` / CSV 写出 | 证据可读性 |
| 逐帧报告扩到 39 列 | `main()` | 用户要求的报告字段 |

---

## 10. §2.8 TSDF 网格融合（本轮新增实现）

论文 §2.8 写的是 weighted fusion：

$$D(x)=\frac{\sum_c w_c(x)d_c(x)}{\sum_c w_c(x)+\varepsilon},\qquad w_c=w_c^{\text{vis}}w_c^{\text{normal}}w_c^{\text{mask}}$$

修正前生产路径 `render.py` 只调用 Open3D `ScalableTSDFVolume`（每视角**均匀权重** + 掩膜背景清零），**没有**实现 §2.8 的三项权重，论文该公式在代码里不存在。本轮新增：

| 论文 | 代码 | 位置 | 状态 |
|---|---|---|---|
| $D(x)=\sum w_c d_c/(\sum w_c+\varepsilon)$ | `GaussianExtractor.extract_mesh_weighted()`（逐体素、逐相机加权平均，`tsdf[idx]=(tsdf·prev+sdf·w)/new_w`） | `utils/mesh_utils.py` | VERIFIED |
| $w^{vis}=\mathrm{clamp}(1-\|d_c\|/\delta,\ w_{vis}^{min},1)$ | `w_vis = (1 - |sdf|/sdf_trunc).clamp(vis_weight_min, 1)` | 同上 | VERIFIED |
| $w^{normal}=w_n^{min}+(1-w_n^{min})\lvert n\cdot v\rvert^{p}$ | 渲染法向 `surf_normal` 旋到世界系后与视线夹角；`--mesh_normal_sharpness` | 同上 | VERIFIED |
| $w^{mask}$ | 双线性采样 RAP-FSAM3 前景掩膜；`--mesh_mask_weight_min` | 同上 | VERIFIED |
| 三项相乘 | `w_total = w_vis*w_normal*w_mask` | 同上 | VERIFIED |
| 生产入口 | `render.py --mesh_weight_mode visibility_normal_mask`（输出 `fuse_weighted.ply`），诊断 JSON `--mesh_weight_diagnostics` | `render.py` | VERIFIED |
| 独立数值验证 | `scripts/check_weighted_tsdf.py`：已知平面 z=1.0 上验证几何（内点 ≤0.5 体素）、$w_{mask}$ 开洞（掩膜外不重建）、$w_{normal}$（掠射视角不污染） | 新增脚本，GS venv 运行 | VERIFIED |

> 注：三项权重的**具体估计器**论文未给出闭式定义，本实现按"可见性/法向/掩膜支撑"的物理含义选取并全部暴露为 CLI；`09_MANUSCRIPT_SYNC_REPORT.md` 已记录应在论文里补写这三项的定义。
