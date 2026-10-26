# 02 — RAP-FSAM3 全链路审计

生成时间：2026-10-25
审计对象：`数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py`（3946 行级主流程）
判定口径：见 `01_PAPER_CODE_MAPPING.md` §0

---

## 0. 一句话结论

RAP-FSAM3 的**六个原稿机制全部在代码中真实存在**，且**全部在真实调用链上**；本轮修复的三个 P0 缺陷（GCFC 观测加载、SB-FBPR 提示类型、SG-MPS 归一化维度）已全部修正，并在 5 帧真实 COLMAP 场景上完成了一次端到端实跑（R6）。
**尚未完成**：更大规模帧集、第二个场景、下游 2DGS/M2M3-GO 接入与论文同步。

---

## 1. 全链路拓扑（真实执行顺序）

```
输入 RGB 帧
  │
  ├─[1] FFT 帧质量过滤        ← 上游独立阶段，blur_filter.py --fft_formula paper
  │
  └─[2] SAM3 多提示词逐实例候选生成   infer_candidates()                    L2/L3 ✔
        └─ 候选掩膜/raw_instance_P{1..5}/mask_<stem>_<inst>.png
  │
  ├─[3] SG-MPS 语义门控选提示词   semantic_gate_scores() + apply_semantic_gate_softmax()
  │      └─ c_i^(k)=σ(wφ·φ − wψ·ψ − b)；a_i^(k)=softmax over PROMPTS (τ_s)
  │
  ├─[4] CDR/R1 实例级重排        select_mask() + ranking_policy r1
  │      └─ S_CDR = S_base − q_contrast/5.5
  │
  ├─[5] SB-FBPR 提示细化         sample_points_from_region / manuscript_negative_regions /
  │      sample_negative_prompts / build_sb_fbpr_prompts → sam3_box_refinement
  │      └─ 正点 = Distance-Transform TopK(K_+)
  │      └─ 负点 = NMS(ζ, K_-, r_nms) 覆盖 ring ∪ table ∪ bg ∪ neighbour
  │      └─ append_point_prompts() → processor._forward_grounding()
  │
  ├─[6] 残余/细结构修复          repair_residual_mask()
  │      └─ 开运算移除 + 邻接/长宽比恢复
  │
  └─[7] GCFC 几何一致性修正      apply_corrective_geometry()
         └─ H_i(p)=Σ κ_σ(p−π_i(X_j))，G_i=⟨H_i,M_i⟩/(‖H_i‖₁+ε)
         └─ 只在 B_i=Dilate(M,r_g)\Erode(M,r_g) 内正向补洞
  │
  ↓
最终掩膜/mask_<stem>.png + 前景图 + 透明图 + 逐帧报告.csv
```

每个方框的 `L2/L3` 判定依据：**该函数被 `main()` 的真实分支调用，且产物落盘**（不是只被单测引用）。

---

## 2. 六个机制的调用链证据

| # | 机制 | 调用点（生产入口内） | 落盘产物 | 调用链判定 |
|---|---|---|---|---|
| 1 | FFT 过滤 | 上游阶段，`run_pipeline.py: fft` | `filter_log.json` | ✔ 独立阶段串联 |
| 2 | 多提示词候选 | `infer_candidates()` ← `:3562` 分支 | `候选掩膜/raw_instance_P*/`、`候选元数据_*.json`、`候选评分明细_*.csv` | ✔ 真调用 |
| 3 | SG-MPS | `apply_semantic_gate_softmax()` ← `:3600` 附近；`semantic_gate_prompt()` | `语义门控评分.csv`、`语义门控框/`、`逐帧报告.csv` 的 `sg_mps_*` 列 | ✔ 真调用 |
| 4 | CDR/R1 | `ranking_policy` 注入 `select_mask()` | `提示词评分.csv`、`选择后掩膜/`、`逐帧报告.csv` 的 `cdr_policy` | ✔ 真调用 |
| 5 | SB-FBPR | `sam3_box_refinement()` ← `apply_spnp_refinement()` ← `:3990` 附近 | `正负提示细化掩膜/`、`正负提示点.csv`、`SB-FBPR提示证据.csv` | ✔ 真调用 |
| 6 | 残余修复 | `repair_residual_mask()` ← `:3992` 附近 | `残差修复掩膜/`、`逐帧报告.csv` 的 `residual_restored_px` | ✔ 真调用 |
| 7 | GCFC | `apply_corrective_geometry()` ← `:3997` 附近 | `几何修正掩膜/`、`几何修正提示图/GCFC_*.png`、`几何修正提示.csv` | ✔ 真调用 |

**辅助链路**（原稿有、代码有，但非 RAP 必需）：`记忆传播掩膜/`（`propagate_memory_masks`）、`共识投票掩膜/`（`apply_cross_view_consensus`）、`compute_geometry_feedback()`。这些在本轮主链路中**未开启**，状态记为 RUNNABLE，不计入 R6 结果。

---

## 3. P0 缺陷修复记录（对应 `03_RAP_FSAM3_FIX_LOG.md`）

### 3.1 缺陷 A — GCFC 观测加载（用户 P0-A）

| 项 | 修复前 | 修复后 |
|---|---|---|
| `colmap_tracks_gaussian` 后端加载观测 | ❌ 不加载 → `missing_colmap_observation` 静默回落 | ✅ 与 `colmap_tracks` 同等加载 |
| 单/多场景 sparse 目录 | 仅单场景 | ✅ `colmap_mask_stem_candidates()` 逐个 stem 解析 |
| 缺观测时的行为 | 静默 fallback | ✅ `--strict_colmap_observations` 时 raise；否则显式 WARNING |
| 拒绝原因排序 | `no_corrective_delta` 可能掩盖 `support_gate` | ✅ `support_gate` 先判定 |
| 调试图 | 无 | ✅ 5 张/帧（before / support_field / boundary_band / added_pixels / after） |

### 3.2 缺陷 B — SB-FBPR 提示类型（用户 P0-B）

| 项 | 修复前 | 修复后 |
|---|---|---|
| 正提示 | BOX（`add_geometric_prompt`） | ✅ 点：`TopK(distance_transform, K_+)` |
| 负提示 | BOX | ✅ 点：`NMS(ζ, K_-, r_nms)`，覆盖 ring∪table∪bg∪neighbour |
| 模型是否支持点 | 假设不支持 | ✅ 实测支持（`Prompt.append_points` → `_encode_points`） |
| 证据落盘 | 只有 box 坐标 | ✅ 逐点归一化坐标 + 标签 + 发送计数 |

### 3.3 缺陷 C — SG-MPS 归一化维度（用户 P0-C）

| 项 | 修复前 | 修复后 |
|---|---|---|
| softmax 维度 | 候选实例级 | ✅ 提示词级（论文 $a_i^{(k)}$ 的 $k$ 是提示词索引） |
| 一提示词多实例 | 逐实例独立归一化 | ✅ 取该提示词的 max 作为代表值，概率写回其所有实例 |
| 与 CDR 关系 | 可能双重计权 $q_{contrast}$ | ✅ $w_\phi[4]=0$ 默认关闭，对比度只经 CDR 进入 |

---

## 4. R6 实跑记录（`runs/R6_RAP_FULL_real/`）

**输入**：`S21_KongQueZhuYu_E2_E3/selected_frames` 5 帧（0000/0025/0050/0075/0100，2160×3840）
**COLMAP**：`04-COLMAP/KongQueZhuYu/sparse/0`（实测注册点 1542/1891/631/98/5064）
**开关**：`--use_semantic_gate --semantic_gate_mode gated_softmax --semantic_gate_temperature 0.25 --ranking_policy r1 --spnp_prompt_form points --spnp_negative_region_mode manuscript --use_residual_repair --thin_repair_mode adjacent --corrective_geometry_backend colmap_tracks_gaussian --strict_colmap_observations`
**设备**：CUDA（CPU 上 SAM3/ViTDet 有 BFloat16/Float dtype 不匹配，属上游已知限制）

| frame | SAM3 候选 | SG-MPS 选 | CDR 政策 | SB-FBPR 实发 | 修复 px | GCFC $G$ | GCFC 新增 | 接受 |
|---|---|---|---|---|---|---|---|---|
| 0000 | 23 | P2 | r1 | 5 正 + 8 负点，0 box | 26 | 0.9752 | 151 | ✔ |
| 0025 | 7 | P2 | r1 | 5 + 8 点 | 21 | 0.9892 | 227 | ✔ |
| 0050 | 5 | P2 | r1 | 5 + 8 点 | 24 | 0.9677 | 456 | ✔ |
| 0075 | 5 | **P1** | r1 | 5 + 8 点 | 12 | 0.9897 | 0 | ✗ `no_corrective_delta` |
| 0100 | 13 | P2 | r1 | 5 + 8 点 | 10 | 0.9918 | 223 | ✔ |

**帧 0075 的价值**：唯一一次提示词从 P2 切到 P1（SG-MPS softmax P1 0.5396 / P2 0.4604），同时 GCFC 因该帧仅 98 个注册点而正确地报 `no_corrective_delta`——两个机制的接受/拒绝逻辑都给出了可解释的真实行为。

**但这一帧也是负增益的证据**：带盆 GT 的 F1 = 0.9227（相邻帧 0.981–0.987），HD95 从 13–25 px 涨到 282.8 px；同一帧按纯植株 GT 则为 0.9800。即门控确实在起作用，只是在优化与论文 §2.3.2 声明不同的目标。详见 `05` §3 与 `metrics/kqz5_R6_vs_splitGT.json`。

---

## 5. 逐帧报告字段对齐核查（用户 §6 要求）

用户要求 23 个字段，`逐帧报告.csv` 实际 39 列，全部要求字段命中：

| 要求字段 | 实际列名 | 命中 |
|---|---|---|
| frame_id | `frame_id` | ✔ |
| input_image_sha256 | `input_image_sha256` | ✔ |
| sam3_candidates | `sam3_candidates` | ✔ |
| selected_prompt / selected_instance | 同名 | ✔ |
| sg_mps_executed / cdr_executed | 同名 | ✔ |
| sb_fbpr_executed / sb_fbpr_accepted | 同名 | ✔ |
| residual_repair_executed | 同名 | ✔ |
| gcfc_executed / gcfc_has_real_observation / gcfc_accepted | 同名 | ✔ |
| final_mask_sha256 | 同名 | ✔ |
| f1 / boundary_f1_if_gt_available | `f1_if_gt_available` / `boundary_f1_if_gt_available` | ✔ |
| failure_reason | 同名 | ✔ |

额外 16 列为支撑字段：`sam3_inference_ran`、`sam3_prompts`、`selection_path`、`variant_selected`、`sg_mps_mode/prompt/softmax/gate_confidence`、`cdr_policy`、`sb_fbpr_prompt_form/reason/pos_points_sent/neg_points_sent`、`residual_repair_mode`、`gcfc_backend/registered_points/support_G/added_px/reason`、`iou_if_gt_available`、`hd95_if_gt_available`。

---

## 6. GT 口径（必须显式声明）

`S21_KongQueZhuYu_E2_E3/gt_masks/` 是 **并集 GT**（植株 + 花盆 + 蓝立方块）。
拆分 GT 在 `GT口径拆分审计/gt_masks_split_v2/KongQueZhuYu/`，含 `mask_plant_*` / `mask_pot_*` / `mask_potted_plant_*` / `mask_blue_cube_*`。

**RAP-FSAM3 的论文目标是 potted-plant 前景**，因此消融与最终指标以 **`mask_potted_plant_*` 为主口径**，同时并列报告 merged 与 plant_only 以便对照。混用两种 GT 而不声明是不可接受的——所有指标表都会带 `gt_convention` 列。

---

## 7. 阻塞项与未验证项（诚实清单）

| 编号 | 项 | 状态 | 说明 |
|---|---|---|---|
| B1 | 仅 5 帧 / 1 个场景 | ✅ 已解除 | KQZ 210 帧全场景跑完（`04` §4）；跨场景 dev 集 4 场景 16 帧（`05`） |
| B2 | 下游 2DGS 未接入全 RAP 掩膜 | ✅ 已解除 | 2×2 四档全部 COMPLETE；⚠ 完整 RAP 掩膜的 3D 指标更差（`06` §6.1） |
| B3 | M2M3-GO 未用全 RAP 掩膜回归 | ✅ 已解除 | 两档真实执行（19 次更新/档）；ΔPSNR_fg +0.011 dB（C_R1）/ −0.125 dB（RAP） |
| B4 | $\omega_{ij}$ 未实现加权 | PARTIALLY_VERIFIED | 取常数 1 |
| B5 | $q_{plant}/q_{table}/q_{neighbour}$ 是启发式代理 | PARTIALLY_VERIFIED | 论文措辞需限定 |
| B6 | 论文 §2.3.3 交叉引用指向 §2.3.2（错误） | 论文侧缺陷 | 待同步 |
| B7 | $\tau_s$ 敏感性未在本轮实跑中扫描 | PARTIALLY_VERIFIED | 沿用阶段 17 冻结消融 |
| B8 | 时间项 $q_{temp}$ 在本轮取 `temporal_iou`，5 帧序列下区分度有限 | PARTIALLY_VERIFIED | 需更长序列验证 |

---

## 8. 结论

| 验收层 | 判定 |
|---|---|
| 全部机制有真实代码（L2） | **VERIFIED** |
| 全部机制在真实调用链上（L3） | **VERIFIED** |
| 全部机制在真实数据上运行过（L4） | **VERIFIED**（210 帧 KQZ 全场景 + 5 帧 KQZ + 16 帧跨 4 场景 dev + 四档下游训练 + 四档 TSDF 网格） |
| 有功能行为证明（L5） | **VERIFIED（但结论不利）**：SG-MPS 提示词切换、SB-FBPR 点提示生效、GCFC 真实修正均有逐帧证据；同时 R0–R6 消融证明 SG-MPS 在 dev 集上**负增益**（见 `05`） |
| 独立实验（L6） | **PARTIALLY_VERIFIED**：4 场景 16 帧 × 10 档 × 3 GT 口径已完成（`metrics/dev_ablation_*.csv`），但样本量小、结论为负 |
| 下游全 RAP 掩膜进入 3D 训练 | **VERIFIED（结论不利）**：2×2 四档全部 COMPLETE（210 帧 RAP 掩膜 + 冻结 C_R1 × M2M3-GO on/off），PSNR_fg −0.285 dB、leakage 10×（`06` §6.1） |
| TSDF 网格提取（论文 §2.8） | **VERIFIED**：四档 `render.py` 标准 TSDF，`runs/mesh/<tag>/` + `metrics/mesh_phenotype.csv` |
| 物理表型（cm） | **BLOCKED**：无经过核验的真实尺度 → `PHYSICAL_SCALE_NOT_AVAILABLE` |
