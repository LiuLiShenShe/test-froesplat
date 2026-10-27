# 06 — Foreground-Object 2DGS 接入（完整 RAP 掩膜）

生成时间：2026-10-25（2026-10-26 增补 v3 重跑状态）
状态：**2×2 四档已完成（修正前 210 掩膜），结论为负；Round-3 修正后的 210 掩膜重跑与下游重训进行中（见 §9）。**

> **§9 重要先行说明（2026-10-26）**：本文 §2/§5/§6 的 2×2 结果来自**修正前**的全场景 RAP 掩膜（`runs/KongQueZhuYu_RAP_FULL_210`，P2 文本为 `entire plant excluding pot`、门控参考为温室背景绿、花盆带切不到花盆、无 $q_{pot}$）。该掩膜在 `05` §10 已被证明与论文目标口径不一致（R6 F1 仅 0.7517）。**Round-3 修正后的 210 掩膜正在后台重跑**（`runs/KongQueZhuYu_RAP_FULL_210_v3`，15:47 启动，~5.8 h），完成后再用其 `最终掩膜/` 重跑下游 2×2 与网格。**在 v3 重跑完成前，本文 §6.1 的指标只能算"修正前掩膜的下游表现"，不得当作方法的最终结论引用。**

---

## 1. 要替换什么

用户要求：**训练用掩膜必须是完整 RAP-FSAM3 输出，不是旧的冻结 `C_R1` 掩膜。**

事实核查（`03-分割Mask/04-fsam3_masks/KongQueZhuYu/参数.json`）：

| 参数 | 冻结 `C_R1` 掩膜 | 完整 RAP（R6） |
|---|---|---|
| `prompt_list` | `P2`（单提示词） | `P1,P2,P3,P4,P5` |
| `candidate_mode` | 旧并集模式 | `per_instance` |
| `ranking_policy` | 未设置 | `r1`（CDR） |
| `use_semantic_gate` | `False` | `True`（gated_softmax, τ=0.25） |
| `use_spnp_refinement` | `False` | `True`（点提示 + manuscript 负区） |
| `use_residual_repair` | `False` | `True`（adjacent） |
| `use_corrective_geometry` | `False` | `True`（colmap_tracks_gaussian） |

结论：`C_R1` 是**纯单提示词基线**，不含任何 RAP 机制。用它做下游训练等同于"没有 RAP-FSAM3"。这正是必须替换的理由。

---

## 2. 掩膜来源与路径

| 项 | 值 |
|---|---|
| 全场景完整 RAP 输出 | `runs/KongQueZhuYu_RAP_FULL_210/最终掩膜/mask_<stem>.png`（**210/210 完成**） |
| 输入帧 | `02-位姿COLMAP/03-final_locked/KongQueZhuYu/images`（210 帧，全部有效 JPEG） |
| COLMAP | `02-位姿COLMAP/03-final_locked/KongQueZhuYu/sparse/0` |
| 下游准备目录 | `runs/downstream/KongQueZhuYu_RAP_FULL/prepared_masks/`（符号链接，不复制） |
| 基线准备目录 | `runs/downstream/KongQueZhuYu_C_R1_baseline/prepared_masks/`（已就绪，210/210） |

---

## 3. 掩膜来源哈希（防"偷偷换掩膜"）

训练脚本（`2d-gaussian-splatting-m2m3-floor40/train.py`）支持：

```
--mask_provenance_path     写入 camera→mask path→sha256→前景占比 清单
--mask_provenance_expected 与预期清单逐帧比对
--mask_provenance_required 有缺失直接报错
```

`scripts/run_downstream_2dgs.py` 在训练**之前**生成 `mask_provenance_expected.json`（含每张掩膜 SHA256），并在训练命令中带上 `--mask_provenance_required`。训练启动后 `train.py` 会写 `mask_provenance.json`；若两者不一致（有人换了掩膜文件），训练直接 `ValueError` 中止。

基线已生成：`runs/downstream/KongQueZhuYu_C_R1_baseline/mask_provenance_expected.json`（210 cameras, 0 missing）。

---

## 4. 训练配方（与 E4b 掩膜来源实验完全一致，保证可比）

来自 `06-实验输出/KongQueZhuYu/E4b_RAP_FSAM3_v2/command.txt`：

```
--eval --resolution 2
--mask_mode alpha --mask_dir <prepared_masks> --mask_pattern mask_{stem}.png --mask_threshold 0.5
--init_pcd_mode foreground_track --init_pcd_min_observations 3
--init_pcd_foreground_threshold 0.9 --init_pcd_dilate_mask_px 0
--use_foreground_rgb_loss --lambda_fg_rgb 1.0 --foreground_bg_rgb_weight 0.0
--foreground_rgb_crop_padding 12
--use_mask_loss --use_bg_opacity_loss --lambda_mask 0.08 --lambda_bg 0.02
--mask_loss_type l1_dice --mask_ignore_boundary_px 2
--mask_loss_start_iter 500 --mask_loss_warmup_iters 1500
--pruning_mode mask --pruning_start_iter 18000 --pruning_interval 3000
--pruning_opacity_threshold 0.005 --pruning_brightness_threshold 0.01
--pruning_mask_threshold 0.45 --pruning_mask_max_views 12
--pruning_max_remove_ratio 0.03 --pruning_mask_score_weight 3.0 --save_pruning_report
--iterations 30000 --save_iterations 30000
```

注：E4b 原命令用 `--resolution 4`；本恢复实验用 **`--resolution 2`** 以便与阶段二十一的 M2M3 档位对齐（分辨率差异已在此声明，两档内部仍然同分辨率对比）。

---

## 5. 待执行的四次训练（串行，避免 OOM）

| # | tag | 掩膜来源 | M2M3-GO | 状态 |
|---|---|---|---|---|
| 1 | `C_R1_baseline` | 冻结 C_R1 | off | ✅ COMPLETE（2201.3 s） |
| 2 | `RAP_FULL` | 完整 RAP 210 帧 | off | ✅ COMPLETE（2152.9 s） |
| 3 | `RAP_FULL_M2M3GO` | 完整 RAP 210 帧 | `full`（τ_p quantile） | ✅ COMPLETE（2131.8 s） |
| 4 | `C_R1_M2M3GO` | 冻结 C_R1 | `full` | ✅ COMPLETE（2133.5 s） |

命令：

```bash
PY=/home/test/.venvs/sam3/bin/python
S=00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Original_Method_Recovery

# 1 基线
$PY $S/scripts/run_downstream_2dgs.py --scene KongQueZhuYu \
    --mask-run-dir 00-论文优化重构/数据管理/03-分割Mask/04-fsam3_masks/KongQueZhuYu \
    --tag C_R1_baseline --iterations 30000 --resolution 2

# 2 完整 RAP
$PY $S/scripts/run_downstream_2dgs.py --scene KongQueZhuYu \
    --mask-run-dir $S/runs/KongQueZhuYu_RAP_FULL_210 \
    --tag RAP_FULL --iterations 30000 --resolution 2

# 3 完整 RAP + M2M3-GO
$PY $S/scripts/run_downstream_2dgs.py --scene KongQueZhuYu \
    --mask-run-dir $S/runs/KongQueZhuYu_RAP_FULL_210 \
    --tag RAP_FULL_M2M3GO --iterations 30000 --resolution 2 --m2m3-go --port 6050

# 4 冻结 C_R1 + M2M3-GO（补齐 2×2）
$PY $S/scripts/run_downstream_2dgs.py --scene KongQueZhuYu \
    --mask-run-dir 00-论文优化重构/数据管理/03-分割Mask/04-fsam3_masks/KongQueZhuYu \
    --tag C_R1_M2M3GO --iterations 30000 --resolution 2 --m2m3-go --port 6050
```

---

## 6. 评价口径

`scripts/evaluate_foreground.py`（带 `--with_lpips`、`--boundary_ignore 2`）输出：

| 指标 | 含义 |
|---|---|
| PSNR_fg / SSIM_fg / LPIPS_fg | 前景区域重建质量 |
| outside ratio | 渲染 alpha 落在掩膜外的比例（前景泄漏） |
| leakage energy | 泄漏能量 |
| Gaussian count | 高斯数量 |
| mask provenance | 逐帧掩膜 SHA256 与前景占比 |

### 6.1 已有结果（`metrics/downstream_2x2.csv`，由 `scripts/build_downstream_2x2.py` 生成）

| tag | 掩膜 | M2M3-GO | PSNR_fg | SSIM_fg | LPIPS_fg | outside ratio | leakage energy | Gaussians | wall |
|---|---|---|---|---|---|---|---|---|---|
| `C_R1_baseline` | 冻结 C_R1 | off | 25.019 | 0.8384 | 0.1183 | **0.0292** | **0.0128** | 699 328 | 2201 s |
| `C_R1_M2M3GO` | 冻结 C_R1 | full | **25.029** | **0.8387** | **0.1182** | 0.0292 | 0.0128 | 634 137 | 2134 s |
| `RAP_FULL` | 完整 RAP 210 | off | 24.734 | 0.8244 | 0.1280 | 0.0657 | 0.1316 | 715 587 | 2153 s |
| `RAP_FULL_M2M3GO` | 完整 RAP 210 | full | 24.610 | 0.8135 | 0.1307 | 0.0645 | 0.1319 | 657 711 | 2132 s |

读法（2×2 的两个主效应）：

| 对比 | ΔPSNR_fg | ΔSSIM | ΔLPIPS | 说明 |
|---|---|---|---|---|
| **掩膜来源**（C_R1 → RAP，M2M3 关） | **−0.285 dB** | −0.0140 | +0.0097 | RAP 掩膜让重建**变差**，泄漏 5×/10× |
| **掩膜来源**（C_R1 → RAP，M2M3 开） | −0.420 dB | −0.0252 | +0.0125 | 同上，幅度更大 |
| **M2M3-GO**（关 → 开，C_R1 掩膜） | **+0.011 dB** | +0.0003 | −0.0001 | 可忽略（正向但量级接近噪声） |
| **M2M3-GO**（关 → 开，RAP 掩膜） | **−0.125 dB** | −0.0109 | +0.0027 | 负向 |

**结论**：在本场景，**掩膜来源的影响（−0.29 dB）远大于 M2M3-GO 的影响（±0.12 dB）**，而且方向都是"完整 RAP 更差"。M2M3-GO 既救不回坏掩膜，在好掩膜上的正增益也只有 0.011 dB。

⚠️ **这是不利结果，必须如实报告**：换成完整 RAP 掩膜后，前景重建质量与泄漏都**变差**（PSNR_fg −0.28 dB，outside ratio +0.0365，leakage energy **10× 恶化**）。

**归因（有逐帧证据）**：`metrics/mask_geometry_rap_vs_cr1.json` 逐帧比对了 210 张 RAP 掩膜与冻结 C_R1 掩膜：

- 整体 IoU 0.9345（中位 0.9768），**22 帧 IoU < 0.75**；
- 按门控选中的提示词分组：P2 156 帧 IoU 0.9662（与 C_R1 几乎一致），但 **P1 29 帧 0.8506、P3 15 帧 0.8007、P5 5 帧 0.7774**；
- 0008 帧 IoU = **0.000**：SG-MPS 选 P1，该帧前景占比从 C_R1 的 0.2266 塌到 **0.0086**（掩膜几乎为空），同时该帧 GCFC 被 `support_gate` 拒绝、SB-FBPR 被 `low_iou_guard` 拒绝 —— 三个保护机制都没能拦住一个近乎空的最终掩膜进入下游训练。

也就是说：**下游指标变差的直接原因是完整 RAP 在这 54 帧（26%）上换用了非 P2 提示词，其中若干帧的掩膜与冻结基线显著不同甚至退化**。这与 `05` 消融在 dev 集上得到的结论一致，且现在是同一套代码在全场景上的复现。论文不能只报"完整 RAP 掩膜已进入 3D 训练"，必须同时报这个对比。

⚠️ **自指限制**：`evaluate_foreground.py` 用的 `--mask_dir` 与训练相同，因此"RAP_FULL 的 outside ratio 0.0657"是在 RAP 自己的掩膜上算的泄漏，**不能**单独用来证明 RAP 掩膜更好。掩膜本身的质量由 `05_RAP_FSAM3_ABLATION.md` 的 GT 指标回答。两者的差别正好暴露问题：RAP 训练**忠实于**自己给的掩膜（甚至把掩膜的更大前景也学进去了），而那个掩膜相对 C_R1 本身就是错的。

---

## 7. 已知风险

| 编号 | 风险 | 说明 |
|---|---|---|
| D1 | RAP_FULL 可能显著差于 C_R1 | `05` 报告显示 SG-MPS 在开发集上选错提示词；KongQueZhuYu 上 5 帧样本里门控选了 P2（4/5），但全场景 210 帧未验证。若 RAP_FULL 明显更差，则"完整 RAP 优于基线"的论文主张不成立，必须如实报告 |
| D2 | 磁盘为机械盘且曾被其它租户限速 | 210 段加载可能耗时 8–70 min，需预留 |
| D3 | 内存 | 两台 GPU 并行训练会触发 OOM killer，**必须串行** |
| D4 | 掩膜自指评价 | 见 §6 |

---

## 8. 完成判据

下游接入算"完成"需要同时满足（逐条核对结果）：

1. `runs/downstream/KongQueZhuYu_*/run_record.json` 中 `status == "COMPLETE"` —— ✅ 四档均 `train_returncode=0`；
2. `mask_provenance.json` 的 `camera_count == 183`、`missing_count == 0`，且 183 条 SHA256 与 `mask_provenance_expected.json` 逐条一致 —— ✅ 四档一致（⚠️ **183 不是 210**：`--eval` 把 210 个相机切成 183 train / 27 test，`mask_provenance` 只对**训练相机**做哈希校验；27 个 test 相机另由 `eval/foreground_test.json` 的 `missing_masks` 覆盖）；
3. `eval/foreground_test.json` 存在且含逐视角前景指标 —— ✅ 四档 27 个 test 视角；
4. 四档训练使用同一 `source_path`、`resolution=2`、`iterations=30000`、损失与剪枝配置 —— ✅（`run_record.json` 的 `train_cmd` 逐字可查）；
5. 结果（含失败）写入 `08_END_TO_END_VALIDATION.md` —— ✅ `08` §2.3 / §4。

**结论：下游接入已完成；结论为负（`06` §6.1），必须如实写进论文。**

---

## 9. Round-3（v3）掩膜重跑与下游重训（2026-10-26，进行中）

Round-3 把目标口径统一到论文 §2.3.2 的 potted-plant 后，`05` §10 已证明 16 帧 dev 集上 R6 F1 0.7517→0.9774。全场景 210 帧必须用同一份修正代码重跑，旧掩膜不得复用（其 P2 候选来自错误的提示词文本，无法"事后修正"）。

**已启动（后台，~5.8 h）：**

```bash
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_rap_full_210_v3.sh \
  >/dev/null 2>&1 &
# 脚本内部等价于：
python 生成RAP-FSAM3掩膜.py \
  --input_dir .../03-final_locked/KongQueZhuYu/images \
  --output_dir runs/KongQueZhuYu_RAP_FULL_210_v3 \
  --candidate_mode per_instance --prompt_list P1,P2,P3,P4,P5 \
  --default_prompt_id P2 --use_prompt_ensemble --prompt_selection_mode score_select \
  --ranking_policy r1 --use_semantic_gate --semantic_gate_mode gated_softmax \
  --semantic_gate_temperature 0.25 \
  --semantic_gate_target_mode potted_plant \   # ← Round-3 修正
  --candidate_area_admissibility gate \        # ← Round-3 修正
  --use_spnp_refinement --spnp_prompt_form points --spnp_negative_region_mode manuscript \
  --use_residual_repair --use_corrective_geometry \
  --corrective_geometry_backend colmap_tracks_gaussian \
  --colmap_dir .../03-final_locked/KongQueZhuYu --strict_colmap_observations --force
```

| 项 | 值 |
|---|---|
| 输出目录 | `runs/KongQueZhuYu_RAP_FULL_210_v3/`（**不覆盖**旧 `runs/KongQueZhuYu_RAP_FULL_210/`） |
| 日志 | `logs/KongQueZhuYu_RAP_FULL_210_v3.log` |
| 启动脚本 | `scripts/run_rap_full_210_v3.sh` |
| 启动时间 | 2026-10-26 15:47 |
| 运行中证据 | `runs/KongQueZhuYu_RAP_FULL_210_v3/参数.json` 已含 `semantic_gate_target_mode=potted_plant`、`candidate_area_admissibility=gate`；`候选掩膜/raw_instance_P{1..5}/` 持续增长 |
| 预计完成 | ~21:30（受 GPU 争用影响） |

**完成后必须执行的下一步（当前尚未执行，不得标记为通过）：**

```bash
# 1. 校验 210/210 掩膜齐全且提示词分布合理
PY=/home/test/.venvs/sam3/bin/python; S=<recovery>
python - <<'EOF'
import csv, pathlib
S=pathlib.Path("runs/KongQueZhuYu_RAP_FULL_210_v3")
masks=list((S/"最终掩膜").glob("mask_*.png")); print("final masks", len(masks))
rows=list(csv.DictReader(open(S/"逐帧报告.csv")))
from collections import Counter; print("selected", Counter(r["selected"] for r in rows))
print("sg_mps", Counter(r.get("sg_mps_prompt","") for r in rows))
EOF

# 2. 用 v3 掩膜重跑四档下游（串行，避免 OOM；§2.5 视图权重需 on/off 对照）
$PY $S/scripts/run_downstream_2dgs.py --scene KongQueZhuYu \
    --mask-run-dir $S/runs/KongQueZhuYu_RAP_FULL_210_v3 \
    --tag RAP_FULL_v3 --iterations 30000 --resolution 2 --view-weight-mode rgb_only
$PY $S/scripts/run_downstream_2dgs.py --scene KongQueZhuYu \
    --mask-run-dir $S/runs/KongQueZhuYu_RAP_FULL_210_v3 \
    --tag RAP_FULL_v3_M2M3GO --iterations 30000 --resolution 2 --m2m3-go --view-weight-mode rgb_only --port 6050
# §2.5 消融对照（同一 v3 掩膜，仅视图权重关）
$PY $S/scripts/run_downstream_2dgs.py --scene KongQueZhuYu \
    --mask-run-dir $S/runs/KongQueZhuYu_RAP_FULL_210_v3 \
    --tag RAP_FULL_v3_novw --iterations 30000 --resolution 2 --view-weight-mode none --port 6060

# 3. 网格（标准 + §2.8 加权）
GS=/data/fj/F2DMAS/2d-gaussian-splatting-m2m3-floor40/venv/bin/python
$GS scripts/run_mesh_phenotype.py --tags RAP_FULL_v3,RAP_FULL_v3_M2M3GO
# §2.8 加权网格对照
cd /data/fj/F2DMAS/2d-gaussian-splatting-m2m3-floor40 && $GS render.py \
    -m <model_dir> --skip_train --skip_test --mesh_weight_mode visibility_normal_mask \
    --mesh_weight_diagnostics <out>/weighted_diag.json
```

**状态：⚠️ 210 帧 v3 重跑进行中，下游 2×2 / 网格 / §2.5 消融均未执行。在它们完成之前，本文不声称"v3 掩膜已进入下游 2DGS 训练"。**

### 9.1 首轮 v3 启动的配置缺陷（已发现并修正，2026-10-26）

首轮 `run_rap_full_210_v3.sh` **漏传了两个 flag**，其默认值使两个细化阶段被静默跳过：

| 漏传 flag | 默认值 | 后果 |
|---|---|---|
| `--spnp_backend` | `none` | SB-FBPR 全部 `reason=backend_off`，210/210 帧 0 点发送 |
| `--thin_repair_mode` | `none` | 残差修复 210/210 帧 `executed=0` |

因此 `runs/KongQueZhuYu_RAP_FULL_210_v3/最终掩膜/` 的 210 张掩膜**只包含 SG-MPS + CDR + GCFC**，不是论文的完整 R6 链（缺 SB-FBPR 与残差修复）。它们**不得**作为"完整 RAP"掩膜进入下游。

修正：新增 `scripts/run_rap_full_210_v3full.sh`，补 `--spnp_backend sam3_if_supported` + `--thin_repair_mode adjacent`，并用 `--reuse_existing_candidates --candidate_source_dir runs/KongQueZhuYu_RAP_FULL_210_v3` 复用已生成的候选池（避免再跑 5.8 h 的 SAM3 多提示词推理）。已核验：**复用产出的选择掩膜与首轮 v3 逐字节一致**（7/7 SHA256 相同），即选择阶段忠实复现，v3full 只在其上补 SB-FBPR→残差修复→GCFC。

| 项 | 值 |
|---|---|
| 输出目录 | `runs/KongQueZhuYu_RAP_FULL_210_v3full/`（不覆盖 v3） |
| 日志 | `logs/KongQueZhuYu_RAP_FULL_210_v3full.log` |
| 启动 | 2026-10-26 22:37，两遍循环（先全帧选择，再全帧细化） |
| 预计 | 因复用需逐帧从机械盘重读候选掩膜，loop1 ~1.5 min/帧 → 全链 ~6–8 h |

