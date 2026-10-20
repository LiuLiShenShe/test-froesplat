# 01 — FINAL METHOD DEFINITION

**ForeSplat — 最终算法结构、模块命名、默认配置与真实公式**

- 核验日期：2026-10-20
- 仓库基线 commit：`e31176ee`（HEAD）；R1 引入 commit：`d704faa8`（Phase 17C）
- 权威生产代码：
  - RAP-FSAM3 掩膜生成：`00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py`、`.../脚本/ranking_policy.py`
  - 前景约束 2DGS + M2M3：`2d-gaussian-splatting-m2m3-floor40/{train.py, arguments/__init__.py, scene/__init__.py, utils/*}`（M0 基线用 `2d-gaussian-splatting-main/`）
- 本文件所有结论均来自上述源码行号，而非以前总结报告或 README。

---

## 0. 一句话结论

真实生产体系是 **四段式**：`(FFT 帧筛选) → (SAM3 P6 单提示词逐实例候选) → (R1=对比解耦重排序选择) → (前景约束 2DGS 训练) → (可选 M2M3 掩膜支持度加权剪枝 + 可选全局容量下限) → (标准 TSDF 网格 + 表型测量)`。

原稿描述的 **多提示词 P1–P5 + 语义门控 softmax + SB-FBPR + GCFC 联合链**与实际默认运行 **不一致**：
- 语义门控（semantic gate）生产默认 **关闭**；
- 论文中的温度 softmax（`τ_s`）**未实现**（实际是加性启发式 `semantic_total`）；
- 多提示词路径（`score_select` / `weighted_fusion`）**绕过 R1**；
- 全仓 **不存在** “R1 + refinement 同时开启”的一次真实运行。
- M2M3-GO 论文公式（双向 deficit / 边界效用 / sigmoid 效用 / 局部 Floor40 / deficit 增密）**在代码中不存在**。

因此最终方法定义必须按真实默认路径重写，而不是按原稿公式反推。

---

## 1. 模块总表（保留 / 修正 / 降级）

| 拟用名称 | 真实身份 | 状态 | 说明 |
|---|---|---|---|
| **ForeSplat** | 整体框架（VFM 前景先验 → 前景对象 2DGS → 网格/表型） | **保留** | 名字可用，但“VFM-guided”必须落到真实机制 |
| **RAP-FSAM3** | 前景先验生成模块（帧筛选 + SAM3 候选 + 重排序 + 可选细化） | **保留，但需重新定义边界** | 默认 = 单提示词 P6 + 逐实例 + R1；不是多提示词语义门控链 |
| **R1** | RAP-FSAM3 内部的候选排序修正机制 | **保留为内部机制，改名** | 建议 **CDR (Contrast-Decoupled Re-ranking)**；见 §3 |
| **Foreground-constrained 2DGS** | 前景约束 2DGS 训练（前景初始化 + 前景 RGB + alpha + 背景透明度 + 视图权重） | **保留（机制真实）** | 全部为 CLI 开关，代码默认关闭；实际运行按命令开启 |
| **M2M3-GO** | 掩膜支持度加权的额外高斯剪枝 + 可选全局容量下限 | **降级为可选紧凑化模块** | 名称易暗示双向机制；建议改称 “mask-supported weighted pruning with optional global capacity floor” |
| **Mesh + Phenotyping** | 标准 2DGS TSDF 网格 + 几何代理测量 | **保留为应用验证（受限）** | 无几何 GT、无物理尺度；表型数字为 legacy |

---

## 2. 最终默认流程（真实执行顺序）

```text
Input multiview RGB images
    ↓
[Step 0] FFT 帧质量筛选（每序列第一四分位阈值；max_consecutive_drop=2）
    ↓
[Step 1] SAM3 候选生成（默认 default_prompt_id = P6 "whole potted plant including the pot"；
         candidate_mode = per_instance；sam3_mask_threshold = 0.5）
    ↓
[Step 2] 逐候选评分 score_candidate()（base_total + semantic_total − risk_penalty）
    ↓
[Step 3] R1 候选选择 select_best_policy(policy="r1")
         primary = base_total − q_contrast/5.5 + semantic_total − penalty
    ↓
[Step 4] (可选) 细化链：SB-FBPR(SPNP) → 残差修复 → GCFC —— 默认建议 OFF
    ↓
[Step 5] 前景约束 2DGS 训练（前景轨迹初始化 + 前景 RGB 损失 + alpha 掩膜损失
         + 背景透明度损失 + 视图权重）
    ↓
[Step 6] (可选) M2M3 掩膜支持度加权额外剪枝（+ 可选全局 Floor40）
    ↓
[Step 7] 标准 TSDF 融合 + Marching Cubes → 网格 → 尺度恢复 → 表型测量
```

**层级划分：**

| 类别 | 内容 |
|---|---|
| **Default main pipeline** | Step 0–3（RAP-FSAM3：FFT+P6+R1），Step 5（前景约束 2DGS），Step 7（TSDF 网格） |
| **Optional methods** | Step 4（细化链，R1 下未联合验证），Step 6（M2M3 剪枝 / Floor40） |
| **Historical variants** | A0–A6 消融（2DGS 先验注入），B0–B5（先验注入位置），E2/E3/E6/E7/E8 |
| **Experimental-only** | 视图权重三信号公式、局部 Floor40、双向 deficit、边界效用、deficit 增密 |

> 注意：Step 5 存在“掩膜身份未闭合”问题——历史 2DGS 使用的是 `03-分割Mask/02-sam_masks/`（无盆标注、旧 SAM/RAP 掩膜），**不是** P6+R1 掩膜（见 `P0-2`）。所以真实“默认流程”中，RAP-FSAM3 的 **输出** 与 2DGS 的 **输入** 之间尚无逐像素同一性证据。

---

## 3. RAP-FSAM3 与 R1 的真实关系

### 3.1 R0 / R1 评分（`ranking_policy.py` 全文 68 行）

```python
POLICY_NAMES = ("base", "r1", "r2", "r3")

def primary_and_tiebreak(base_total, q_contrast, q_area,
                         w_contrast, denom, semantic_total, penalty, policy):
    if policy == "base":
        return base_total + semantic_total - penalty, 0.0
    if policy in ("r1", "r2"):
        primary = (base_total * denom - w_contrast * q_contrast) / denom \
            + semantic_total - penalty
        tie = q_contrast if policy == "r2" else 0.0
        return primary, tie
    if policy == "r3":
        primary = (base_total * denom - w_contrast * q_contrast
                   + w_contrast * q_contrast * q_area) / denom \
            + semantic_total - penalty
        return primary, 0.0
    raise ValueError(...)

def select_best_policy(score_records, policy):
    return max(score_records,
               key=lambda r: (r.primary_score, r.tie_break_score))
```

### 3.2 完整生产公式

`base_total`（`生成RAP-FSAM3掩膜.py` L1592–1602）：

```
denom = max(sum(weights.values()), 1e-6)                                   # = 5.5
base_total = (w_area·q_area + w_comp·q_comp + w_edge·q_edge
              + w_temp·q_temp + w_contrast·q_contrast
              + w_leak·q_leak + w_side·q_side
              + w_sam·q_sam) / denom
```

生产默认权重：`area=1, comp=1, edge=1, temp=1, contrast=1, sam=0.5`（`leak`/`side` 未给出 → 0）→ **denom = 5.5**。

各分量（L1581–1591）：

| 分量 | 定义 | 生产默认 |
|---|---|---|
| `q_area` | `area_quality(area_ratio)` 三角合理性 | 权重 1.0 |
| `q_comp` | `1/(1+max(comp_count-1,0))` | 1.0 |
| `q_edge` | `edge_quality(mask)` 边界贴合度 | 1.0 |
| `q_temp` | 无时序对齐 → **恒 0.5**（L1586，`use_temporal_alignment` 默认关） | 1.0 |
| `q_contrast` | `min(1.0, contrast/0.25)` | 1.0 |
| `q_leak` | `leakage_quality` | **0.0（仅诊断）** |
| `q_side` | `side_leakage_quality` | **0.0（仅诊断）** |
| `q_sam` | `1/(1+exp(-6(sam_score-0.5)))` | 0.5 |

风险惩罚（L1607–1614）：`risk_penalty = 0.25 if vertical_coverage<0.30`（`+0.15` 的 track-overlap 分支在生产 `per_instance` 路径为 **死分支**，因 `Candidate.track_mask` 未被赋值）。

语义项：`semantic_total = 0.0`（`use_semantic_gate` 默认关，L1623）。

因此：

```
S_R0(c) = base_total(c) + 0 − risk_penalty(c)                          （= total_score）
S_R1(c) = base_total(c) − w_contrast·q_contrast(c)/denom + 0 − risk_penalty(c)
        = S_R0(c) − 1.0 · q_contrast(c) / 5.5
```

**用户提出的公式 `S_R1(c) = S_R0(c) − w_contrast × q_contrast(c) / 5.5` 在生产默认下是精确的**（`w_contrast=1.0`, denom=5.5, semantic_total=0）。需要补充的 omitted terms：`semantic_total`（门控开启时非零）与 `risk_penalty`（两者共有，相减抵消，故上述化简成立）。

### 3.3 R1 到底改了什么（逐项回答任务 A1）

| 检查项 | 结论 | 证据 |
|---|---|---|
| R0 的候选生成与评分 | 见 §3.2 | L1540–1672 |
| R1 具体修改 | **仅**从主评分中扣除 `q_contrast` 一项 | `ranking_policy.py` L44–48 |
| contrast 如何进入原始评分 | 作为 `base_total` 中 `w_contrast·q_contrast/denom` 一项 | L1598 |
| R1 是否移除 contrast 主评分贡献 | **是**（仅一次，在 `primary_and_tiebreak` 内） | L45–46 |
| 其余权重/分母/归一化/tie-break | **全部不变**（分母仍 5.5；tie-break 仍 0） | `R1_FORMULA_VERIFICATION.md` |
| R1 是否新增目标一致性特征 | **否**（无任何新特征） | 源码逐行 |
| R1 是否改变候选池 | **否**（同一候选集） | `rank_candidates.py` 断言 R0/R1 仅 `ranking_policy` 不同 |
| R1 是否具有真正跨视角一致性约束 | **否**（`use_cross_view_consensus=false`，无跨视角项） | 冻结参数 |
| RAP-FSAM3 最终默认是否 R1 | **是**（`--ranking_policy r1`） | `ranking_policy.py`、`生成RAP-FSAM3掩膜.py` L657–662 |
| 其他默认开关 | `use_temporal_alignment=false`；`use_semantic_gate=false`；`use_reprompt_detection=false`；`use_cross_view_consensus=false`；`use_memory_propagation=false`；`use_prompt_ensemble=false`；`prompt_selection_mode=single`；`candidate_mode=per_instance`；`init_pcd=...` 仅 2DGS 侧 | 冻结参数 JSON |

### 3.4 命名建议：**CDR — Contrast-Decoupled Re-ranking（对比解耦重排序）**

- **TCR（Target-Consistency Re-ranking）不准确**：代码中 **没有** 任何“目标一致性”特征或学习；R1 只是把 `q_contrast` 从主排序信号降为诊断信号。使用 TCR 会被审稿人视为过度包装。
- **CDR 准确**：直接描述“将前景-背景对比度项从候选主评分中解耦”。
- 备选：**CER（Contrast-Excluded Re-ranking）**。仍推荐 **CDR**，因为“decoupled”同时表达了“该项仍被保留为诊断值”这一事实（`q_contrast` 未删除，仅移出主排序，见 `R1_FORMULA_VERIFICATION.md` §4）。

**推荐命名机制**：`CDR — Contrast-Decoupled Re-ranking`，作为 **RAP-FSAM3 内部的候选排序修正机制**（不独立成算法）。

### 3.5 关键限制（必须在论文中披露）

`R1_CALL_PATH_AUDIT.md`（Phase 20A）以全仓 grep 证明：

| 选择分支 | R1 是否生效 | 行号 |
|---|---|---|
| `single` + `per_instance`（**生产默认**） | **生效** | L1725 `select_best_policy()` |
| `score_select`（多提示词） | **绕过** | L1688 用 `row.total_score` |
| `weighted_fusion`（多提示词） | **绕过** | L1699 |
| A6/A7 变体选择 | **绕过** | L3162 |

> **R1 只在“单提示词 + 逐实例”路径生效。全仓 78 份 `参数.json` 中，所有开启 SPNP/Residual/GCFC 的运行一律走 `score_select`，即 `R1 × refinement` 联合 **零证据**。**
> 并且：全仓 78 份历史 `参数.json` **无一**含 `ranking_policy` 键（全部早于 R1 提交 `d704faa8`）。即 R1 **从未作为正式运行产出过掩膜**；其证据全部是离线重排序与 Phase 20D 的固定轮廓重放。

---

## 4. Foreground-constrained 2DGS — 真实损失与机制

### 4.1 总损失（`train.py` L555–581）

```
total_loss = loss + dist_loss + normal_loss + mask_loss + bg_loss     # L581
loss       = rgb_reconstruction_loss(...) · view_weight               # L555–557
normal_loss= λ_normal · normal_error,  λ_normal = 0.05 if iter>7000 else 0   # L561,568
dist_loss  = λ_dist  · rend_dist,      λ_dist   = 0.0  if iter>3000 else 0   # L562,569
```

### 4.2 各分项

**（a）前景 RGB 损失**（L136–162）：掩膜忽略 2px 边界带；背景像素默认权重 0：
```
stable_region = stable_mask_region(target_mask, mask_ignore_boundary_px)   # 去 dilate−erode 带
fg_weight     = target_mask · stable_region
bg_weight     = (1 − target_mask) · stable_region · max(0, foreground_bg_rgb_weight)   # 默认 0
ll1 = weighted_rgb_mean(|image − gt_image|, rgb_weight)
loss = ((1 − λ_dssim)·ll1 + λ_dssim·ssim_loss) · λ_fg_rgb      # λ_dssim=0.2, λ_fg_rgb=1.0
```

**（b）alpha 掩膜损失**（L185–199，`l1_dice` 分支）：
```
mask_loss = (0.5·l1 + 0.5·dice) · λ_mask · χ(t)               # λ_mask = 0.08
```

**（c）背景透明度损失**（L203–206）：
```
bg_loss = masked_mean(render_alpha, (1−target)·stable) · λ_bg · χ(t)   # λ_bg = 0.02
```

**（d）warm-up**：`χ(t)=ramp_weight(t; start=500, warmup=1500)`（`plant_aware_utils.py` L137–142）。

### 4.3 与用户拟用公式的差异

用户/原稿：`L_fg = L_RGB + 0.08 L_alpha + 0.02 L_bg`。

真实实现（实际运行命令，见 `m2m3实验/.../command.txt`）：
```
L = 1.0·L_rgb-fg(+0.2·dssim 内含) + χ·0.08·(0.5·L1 + 0.5·Dice) + χ·0.02·L_bg
    + λ_dist·L_dist(=0) + λ_normal·L_normal(=0.05, iter>7000)
```

需要修正的细节：
1. `0.08` 在代码中 **不是** alpha 损失整体系数，而是 `(0.5·L1 + 0.5·Dice)` 的整体系数——即 **0.04 L1 + 0.04 Dice**。
2. 存在额外正则项 `λ_normal=0.05`（iter>7000），原稿 `L_reg` 未给权重。
3. 所有 `use_*` 开关与 `λ` **代码默认全为关/0**；0.08/0.02 仅存在于实际运行命令与一个 **未被代码读取** 的 YAML（`ForeSplat_real_release/configs/default_foresplat.yaml`）。

因此 **A5 不是“三个独立验证的创新模块”，而是一组联合监督设置**（同一场景同一次训练同时开启 fg RGB + alpha + bg）。A2/A3/A4 是单独/部分开启的对照——它们 **并未** 产生前景分离（见表 §C3），只有联合开启才生效。这一点对论文叙事至关重要。

### 4.4 视图质量加权 —— 原稿公式未实现

- 真实实现：`read_view_weights()` 读取 **外部预计算** 的逐视图权重 CSV，clamp 到 `[0.6, 1.0]`，仅缩放 RGB 损失（`view_weight_mode=rgb_only`）。
- 原稿三信号 `q_i = q_min+(q_max−q_min)σ((s_i−μ)/σ)`（掩膜覆盖/边界锐度/前景对比）在 **训练代码中不存在**；权重由外部脚本 `01_prepare_three_scene_assets.py` 生成。
- `arguments/__init__.py:58` 的 `view_quality_mode` 声明后 **从未被读取**（死标志）。

---

## 5. M2M3-GO — 真实实现（对齐 Phase20E3）

**权威描述**：`M2M3_MANUSCRIPT_CORRECTION_SPEC.md` + `train.py` L240–465。

### 5.1 mask-to-model 支持度（唯一实现方向）

`mask_consistency_scores()`（`train.py` L327–359）：
```
对最多 12 个 stride 抽样训练相机：
  v_ic = 高斯中心在相机前方且投影落入 NDC (−1,1)²   （无遮挡检测）
  s_i  = Σ_c v_ic·1[a_c(π_c(x_i))>0.5] / Σ_c v_ic     若 Σv>0
  s_i  = 1.0                                             若 Σv=0（无有效投影默认支持度 1，非 0）
```
`a_c` 为 `grid_sample(mode="nearest", align_corners=True)` 读取的冻结 alpha。这是 **中心投影代理**，不是原稿的 footprint 平均 `S^{m2g}`。

### 5.2 剪枝评分（冻结实验）

全局最大归一化：`N(z)_i = [z_i]_+ / max(max_j [z_j]_+, 1e-6)`。

```
q_i = N([0.005−α_i]_+) + N([0.01−b_i]_+) + 3·N([0.45−s_i]_+)
      + 0.5·N([ℓ_i−median_torch(ℓ)]_+) + 0.5·N([r_i]_+)
```
其中 `α_i=sigmoid(_opacity)`，`b_i`=SH DC 亮度，`ℓ_i=max_k exp(_scaling_ik)`，`r_i=max_radii2D_i`。
权重 `m2m3_mask=3.0, opacity=1.0, brightness=1.0, scale=0.5, view=0.5`。

### 5.3 预算、时机、floor

- 时机：`pruning_start=18000, interval=3000` → 18000/21000/24000/27000/30000。
- 候选集：`pruning_mode=mask` → `C={i: s_i<0.45}`（**仅 mask 支持不足者**；opacity/brightness 在 mask 模式下不单独生成候选）。
- 预算：`B=floor(0.03·N)`，按分数取 top-B。
- 调用顺序：训练损失/反向 → 原生 `densify_and_prune()`（仅 iter<15000）→ `update_capacity_state()` → optimizer step → **额外 M2M3 剪枝**。
- **不与原生 densification 联动**；M2M3 不做增密。
- Floor40（仅 `capacity_control_mode=m2m3_floor`）：`F_t = min(N, max(0, round(0.4·N_max_seen)))`，为 **全局单一标量**，**非** 原稿的按前景块 `N_k^min=0.4·N_k^ref`。

### 5.4 未实现（原稿公式 → 必须删除/降级）

| 原稿机制 | 代码状态 | 原稿位置 |
|---|---|---|
| Model-to-mask deficit `H=M(1−A)`, `S^{g2m}` | **NOT FOUND** | L464–478 |
| Boundary-aware utility `B^∂`, `σ_b` | **NOT FOUND** | L480–490 |
| sigmoid `P_keep`/`U_j` 效用 | **NOT FOUND** | L492–507 |
| 双向 mask↔model 反馈 | **NOT FOUND** | L441–547 |
| 局部块 Floor40 | **NOT FOUND**（仅全局） | L530–547 |
| mask-deficit 增密 / keep-prune-densify 联合决策 | **NOT FOUND** | L509–528 |

### 5.5 Floor40 实测结论

`M2M3_CAPACITY_EFFECT_REPORT.md`：15/15 M2 事件均计算并激活，但 **`floor_blocked_count = 0` 全部**，累计阻止删除 **0**。原因：五轮每轮 ≤3% 的预算下界恒高于 `0.4·peak`，**预算本身保证不可能触及 floor**。判定 `FLOOR40_BENEFIT = NO_EFFECT`。

---

## 6. 网格与表型（真实实现）

- **网格**：标准 2DGS TSDF。`extract_mesh.py → GaussianExtractor.extract_mesh_bounded() → Open3D ScalableTSDFVolume`，参数 `voxel=2r/1024, sdf_trunc=5·voxel, depth_trunc=2r, mesh_res=1024`；后处理 `num_cluster=1`（只保留最大连通分量）。
- **原稿的 visibility/normal/mask 加权 TSDF 权重公式 `w_c=w_c^vis·w_c^normal·w_c^mask` 未在正式结果中使用**；一个自定义 `edge_aware_postprocess_mesh`（`mesh_utils.py` L47–125）存在但被明确排除（`MESH_EXTRACTION_PROTOCOL.md`：不使用额外边界收缩）。
- **尺度恢复**：**无实现**。Phase20F：`PHYSICAL_SCALE = NOT_AVAILABLE`，`VERTICAL_REFERENCE = NOT_AVAILABLE`，`0` 个有效厘米值。

---

## 7. 最终默认配置（冻结）

```json
{
  "prompt_list": "P6",
  "default_prompt_id": "P6",
  "use_prompt_ensemble": false,
  "prompt_selection_mode": "single",
  "candidate_mode": "per_instance",
  "sam3_mask_threshold": 0.5,
  "score_weights": "area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5",
  "ranking_policy": "r1",
  "use_temporal_alignment": false,
  "use_semantic_gate": false,
  "use_reprompt_detection": false,
  "use_cross_view_consensus": false,

  "use_foreground_rgb_loss": true, "lambda_fg_rgb": 1.0,
  "use_mask_loss": true, "lambda_mask": 0.08, "mask_loss_type": "l1_dice",
  "use_bg_opacity_loss": true, "lambda_bg": 0.02,
  "mask_ignore_boundary_px": 2, "mask_loss_start_iter": 500, "mask_loss_warmup_iters": 1500,
  "view_weight_mode": "rgb_only", "view_weight_min": 0.6, "view_weight_max": 1.0,

  "pruning_mode": "mask", "pruning_start_iter": 18000, "pruning_interval": 3000,
  "pruning_opacity_threshold": 0.005, "pruning_brightness_threshold": 0.01,
  "pruning_mask_threshold": 0.45, "pruning_max_remove_ratio": 0.03,
  "m2m3_max_remove_ratio": 0.03,
  "capacity_control_mode": "m2m3_floor",
  "capacity_floor_ratio": 0.4, "capacity_floor_reference": "max_seen",
  "capacity_floor_start_iter": 18000,

  "iterations": 30000
}
```

来源：`stage20A FINAL_RAP_FSAM3_SPEC_DRAFT.md` §2 冻结参数 + `m2m3实验/实验输出/runs/20260610_151144_full/*/command.txt`。

---

## 8. 必须在论文中如实披露的 5 个结构性问题

1. **掩膜身份未闭合**：2DGS 的 `mask_dir` 是 `03-分割Mask/02-sam_masks/`（旧、无盆），不是 P6+R1 输出（`P0-2`）。
2. **R1 从未正式运行产出掩膜**（`P0-4`），证据为离线重排序 + Phase20D 固定轮廓重放。
3. **R1 × refinement 零联合证据**（`P0-5`）；78 份运行中 `single+SPNP` 组合 0 例。
4. **语义门控 softmax 未实现**（`P1-2`）；生产默认门控关闭。
5. **M2M3 四条公式无实现**（`P1-4`），Floor40 为全局而非局部（`P1-5`），且实测从未生效。

---

## 9. 证据来源清单

| 结论 | 文件 |
|---|---|
| R1 公式 | `阶段二十A.../R1_FORMULA_VERIFICATION.md`；`S20.../脚本/ranking_policy.py`；`S20.../脚本/生成RAP-FSAM3掩膜.py` L1540–1672 |
| R1 调用路径 | `阶段二十A.../R1_CALL_PATH_AUDIT.md` |
| 2DGS 损失 | `2d-gaussian-splatting-m2m3-floor40/train.py` L136–206,555–581 |
| 视图权重 | `train.py` L74–114；`arguments/__init__.py:58`（死标志） |
| M2M3 实现 | `train.py` L240–465；`M2M3_IMPLEMENTATION_TRUTH.md`；`M2M3_MANUSCRIPT_CORRECTION_SPEC.md` |
| Floor40 实测 | `M2M3_CAPACITY_EFFECT_REPORT.md`；`M2M3_METHOD_DECISION.md` |
| 默认命令 | `m2m3实验/实验输出/runs/20260610_151144_full/*/command.txt` |
| 冻结参数 | `阶段二十A.../FINAL_RAP_FSAM3_SPEC_DRAFT.md` |
| 网格 | `阶段二十E3.../MESH_EXTRACTION_PROTOCOL.md`；`MESH_REFERENCE_RESULTS.md` |
| 尺度 | `阶段二十F.../PHASE20F_FINAL_REPORT.md` |
