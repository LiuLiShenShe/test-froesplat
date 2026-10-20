# 04 — CLAIM TO EVIDENCE MATRIX

**C1–C8 论断与证据对应关系**

- 核验日期：2026-10-20；基线 commit `e31176ee`
- 状态定义：**VERIFIED** / **PARTIALLY_VERIFIED** / **UNSUPPORTED** / **CONTRADICTED**
- 全部数值均追溯到原始 CSV/JSON/PLY/源码行，未使用总结性 markdown 报告作为唯一依据。

---

## 0. 总览

| Claim | 主题 | 状态 | 核心证据 | 可进入的位置 |
|---|---|---|---|---|
| **C1** | R1 目标实例选择（TEST32） | **PARTIALLY_VERIFIED** | `R0_R1_TEST32_SUMMARY.csv` / `RESULTS.csv` / `PHASE20D_DIRECT_EVALUATION_REPORT.md` | Results（附口径说明）+ Discussion |
| **C2** | 历史 RAP-FSAM3 分割改进 | **PARTIALLY_VERIFIED** | `阶段二十B.../tables/TABLE3/4/5_RECALCULATED.csv` | Results（作为 legacy baseline） |
| **C3** | A5 联合前景监督 | **VERIFIED（数值）/ 范围受限** | `kongquezhu_A0_A6_ablation_summary.csv` + `foreground_object_results.json` + PLY 头 | Results + Abstract（须限定单场景） |
| **C4** | M2M3-GO 容量控制 | **VERIFIED（数值）/ MIXED（结论）** | `M2M3_SCENE_SUMMARY.csv` / `M2M3_PRUNING_EVENT_LOG.csv` / `M2M3_CAPACITY_EFFECT_REPORT.md` | Results（正负混合）+ Discussion |
| **C5** | 2D→3D 不一致 | **VERIFIED** | `S24.../experiment4b_summary.csv` + Phase20E2 `PHASE20E2_FINAL_REPORT.md` / `R1_3D_ROBUSTNESS_DECISION.md` | Results（专节）+ Discussion |
| **C6** | 后训练压缩 | **PARTIALLY_VERIFIED** | `COMPRESSION_METRICS.csv` / `PHASE20G0_FINAL_REPORT.md` | **补充材料** |
| **C7** | Mesh 重建 | **PARTIALLY_VERIFIED** | `MESH_STRUCTURE_METRICS.csv` / `MESH_REFERENCE_RESULTS.md` | Results（定性/结构）+ 补充材料 |
| **C8** | 表型测量 | **CONTRADICTED（作为 ForeSplat 结果）** | `PHASE20F_FINAL_REPORT.md` / `PHYSICAL_SCALE_PROVENANCE.csv` / `HISTORICAL_PHENOTYPE_MATCHING.csv` | **仅作 legacy 记录 + Limitations** |

---

## C1 — RAP-FSAM3 / R1 目标实例选择

### 核验到的原始事实

| 检查项 | 结论 | 证据 |
|---|---|---|
| 8 个序列、32 帧 | ✅ | `R0_R1_TEST32_RESULTS.csv`：CaoMei2 / ChangShouHua3 / DouBanLv3 / HongZhang / WanNianQing1 / WangWenCao2 / XianKeLai3 / XiangPiShu2，各 4 帧 |
| 序列等权统计 | ✅（`sample_equal_mean`） | `R0_R1_TEST32_SUMMARY.csv`；8×4 使 `frame_equal_mean` 与之数值相同 |
| R0/R1 相同候选池 | ✅ | `阶段二十E0.../scripts/rank_candidates.py`：同 `verify_cache` 候选，仅 `ranking_policy` 不同；代码内显式断言配置仅此一项不同 |
| 只有 2 帧有益切换 | ✅ | `R0_R1_TEST32_RESULTS.csv`：`f1_effect` 计数 = IMPROVED 2 / UNCHANGED 30 / REGRESSED 0 |
| 其他 30 帧不变 | ✅ | 同上；`selection_changed=0` 30 行 |
| 具体 2 帧 | ✅ | `CaoMei2_0146`（F1 0.0001→0.9855）、`WanNianQing1_0227`（F1 0.0→0.985） |
| 改善帧存在 GT 轮廓歧义 | ✅ **是** | 2 帧均属 `GEOMETRY_AMBIGUITY` 子集（13 帧）；`CONTOUR_GEOMETRY_AUDIT.csv`：13 帧/17 形状有自交/回退数值警告，10 帧影响主 GT |
| 仍存在目标选择失败 | ✅ **是** | `WanNianQing1_0223` 仍 F1=0，oracle regret≈0.987，未被删除 |
| DEV10 与 TEST32 相互独立 | ⚠️ **部分** | `TEST32_GT_AUDIT.md`：DEV26(BaiZhang/DouBanLv2) 与 TEST 无 sample/sequence ID 交集；Easy21/GT10 的 31 个 DEV RGB 亦无序列交集；TEST 内与 TEST/DEV 的完全重复哈希 = 0。**但**“邻近帧存在相关性；未人工验证近重复/跨序列物理身份，不能声称完全排除所有泄漏” |
| 指标数值 | ✅ | R0 F1 0.88887→R1 0.95045；IoU 0.87222→0.93291；HD95 390.60→227.26；Boundary-F1 0.24737→0.26349；Leakage 0.007471→0.006664（opencv rasterizer；pillow 复算一致至 1e-5） |

### 决定性限制

1. **官方人类 GT 门禁仍为 BLOCKED**：`阶段二十D_独立GT与困难场景验证/PHASE20D_STATUS.json` = `human_gt: USER_ANNOTATION_REQUIRED`, `test32_evaluation: BLOCKED`, `r1_independent_validation: NOT VERIFIED`, `human_verified_difficulty_frames: 0`。`TEST32_GT_AUDIT.md`：32/32 帧的人工 JSON = **0**，合格双人复核 = **0**。
2. 本次指标使用的是 **确定性栅格化的既有手工轮廓**（`阶段二十D_固定轮廓直接评价/PHASE20D_DIRECT_EVALUATION_REPORT.md`："Fixed-contour rasterization evaluation … **not** dual-signature standard human GT"）。
3. `FINAL_FOREGROUND_METHOD_LOCK.md` 七项独立结论中，“困难多实例目标的正确抢救”“TEST32 退化数”“错误物理目标选择”“收益是否集中于少数严重失败帧”四项均为 **NOT VERIFIED**。
4. R1 **从未作为正式运行产出过掩膜**（78 份历史 `参数.json` 无 `ranking_policy` 键）；证据为离线重排序 + 本次固定轮廓重放。

### 结论与可用强度

- **状态：PARTIALLY_VERIFIED。**
- **可写入 Results**：“在 8 序列 32 帧的固定轮廓挑战集上，CDR/R1 将 F1 由 0.889 提升至 0.950（+0.062），IoU 0.872→0.933，HD95 390.6→227.3 px；32 帧中 2 帧目标切换且两帧均改善，0 帧退化。”
- **必须同时写入的限定**：“评估基于既有手工轮廓的确定性栅格化，尚未完成双人独立标注的 GT 门禁；收益集中于 2 帧，其中 1 帧仍为完全失败。”
- **不可写入**：Abstract 中的百分比提升、任何“可靠/鲁棒地解决了目标误选”的强断言、任何 3D 收益的暗示。

---

## C2 — 历史 RAP-FSAM3 分割改进（Phase20B）

### 统一口径重算（potted_clean = potted_plant ∧ ¬blue_cube；Easy20 = 4 序列 × 5 帧）

| 方法 | F1 | IoU_fg | mIoU | HD95(px) | Boundary-F1 | Leakage |
|---|---:|---:|---:|---:|---:|---:|
| CLIPSeg_potted | 0.93029 | 0.87018 | 0.91732 | 159.62 | 0.09126 | 0.018291 |
| GroundedSAM1_potted | 0.96220 | 0.93161 | 0.95473 | 87.89 | 0.42074 | 0.004373 |
| **RAP-FSAM3-v2** | **0.98373** | 0.96824 | 0.97963 | 23.73 | 0.48521 | 0.001739 |
| SAM3_P2 | 0.98283 | 0.96634 | 0.97822 | 26.82 | 0.42235 | 0.002272 |

来源：`阶段二十B_历史实验与证据修复/tables/TABLE3_RECALCULATED.csv`（`source_commit=8ffa174f`）。

### 关键勘误（不得沿用的旧数字）

| 旧数字 | 问题 | 依据 |
|---|---|---|
| Table 3 `F1=0.9706, mIoU=0.9641` | 混合 pooled confusion 与逐帧统计；Phase20A 用 5 种 GT 变体重算 **无一种**能复现 | `P0_BLOCKERS` P0-3 |
| Table 3 `HD95=183.42` | 同上 | 同上 |
| Table 5 `Boundary-F1=0.4543`, `Leakage=0.001935` | **全仓无出处**，无预测/GT/帧列表绑定 | `HISTORICAL_EVIDENCE_ERRATA.md` |
| “sampled from all 20 RGB sequences” | 实为 **4 个代表序列共 20 帧** | `E1_.../方法覆盖与运行状态.md` |
| Table 4 P2 标签 “entire potted plant” | 归档实为 **`entire plant excluding pot`**；且 P2 实测盆召回≈97.5%，与 P6 差 <0.1pp，“含盆/去盆”不是有效区分轴 | `P1-1`、`FINAL_RAP_FSAM3_SPEC_DRAFT.md` F1 |
| Table 3 的 GT | 原值用 **legacy union GT**（potted ∪ cube）；重算用 `potted_clean`，两者不可混列 | `METHOD_SELECTION_DECISION.md` F2 |

### 版本归属

重算值属于 **旧 RAP-FSAM3-v2**（`prompt_selection_mode=score_select`，无 R1、无几何反馈）。**R1 从未以生产运行形式产出过分割结果**（182/182 只是离线公式复核）。

### 结论

- **状态：PARTIALLY_VERIFIED**（作为“统一 GT 下的历史基线”完全可用；作为“本文最终方法”不可用）。
- **可写入**：Results 中作为“历史多提示词 RAP-FSAM3 在统一 GT 下的基线”，并明确标注版本。
- **不可写入**：Abstract 的 0.9706/0.9641/183.42/0.4643 组合；“在 20 个序列上评估”。

---

## C3 — A5 联合前景监督（S10）

### 核验到的原始事实

| 项 | A0（full-scene 2DGS, E2） | A5（fg RGB + alpha + bg） |
|---|---:|---:|
| PSNR_fg | 24.208995 | **25.105544** |
| SSIM_fg | 0.851372 | 0.856108 |
| LPIPS_fg(black bg) | 0.048028 | 0.043739 |
| outside non-black ratio | 0.990785 | **0.029387** |
| leakage energy ratio | 1.220132 | **0.019000** |
| Gaussian count @30000 | 751,213 | **592,900** |
| 评价视角 | 27 | 27 |
| 迭代 | 30000 | 30000 |

来源：`数据管理/05-评测结果/KongQueZhuYu/S10_small_matrix/kongquezhu_A0_A6_ablation_summary.csv`（A0/A5 行）+ `06-实验输出/KongQueZhuYu/{E2_2dgs_baseline,A5_fg_rgb_alpha_bg_loss}/foreground_object_results.json` + PLY 头 `element vertex`。

A5 配置（`S10-KongQueZhuYu小矩阵/configs/kongquezhu_A5_fg_rgb_alpha_bg_loss.json`）：
`use_foreground_rgb_loss=true, lambda_fg_rgb=1.0, foreground_bg_rgb_weight=0.0, foreground_rgb_crop_padding=12, use_mask_loss=true, use_bg_opacity_loss=true, lambda_mask=0.08, lambda_bg=0.02, mask_loss_type=l1_dice, mask_ignore_boundary_px=2, mask_loss_start_iter=500, mask_loss_warmup_iters=1500, pruning_mode=none, mask_mode=alpha, mask_threshold=0.5`。

### 重要限定

1. **掩膜来源是旧 `02-sam_masks`（SAM v1），不是 RAP-FSAM3 / P6 / R1。** 与论文的分割贡献不同族。
2. **单场景、单次运行、无随机种子重复、无多场景复现。**
3. **A0→A5 同时改变三个损失项**，不是单因子消融。真正的单因子步是 **A4→A5（仅前景 RGB 损失）：24.8126→25.1055**。
4. A0 的前景评测命令未在 `run_status.json` 中留日志（仅全图指标步骤），同一性由两者 `mask_ratio_mean` 字节一致推断。
5. **全图指标在 A0/A5 之间不可比**（A5 全图 PSNR 6.256 vs A0 24.159，因 A5 有意不重建背景）——只有 `*_fg` 列可跨行比较。
6. 论文中若同时引用 A 系列与 B 系列编号，需统一（B0≡A0、B3≡A5、B5≡A6+M4）。

### 结论

- **状态：VERIFIED（数值）/ 范围受限（外推）。**
- **可写入**：“在代表性宽叶盆栽单场景上，联合前景监督把前景 PSNR 由 24.21 dB 提升至 25.11 dB，外部非黑比例由 0.991 降至 0.029，泄漏能量比由 1.220 降至 0.019，高斯数量由 751k 降至 593k。”
- **不可写入**：多场景普遍性；三个损失项各自独立的贡献；端到端 ForeSplat 性能。
- **必须回答**：“是否支持联合前景约束还是某个单独损失项？”→ **支持联合约束**。单独开启 alpha（A2：24.3422）、单独背景透明度（A3：24.7508）、两者联合（A4：24.8126）**均未产生前景分离**（outside≈0.99）；只有加入前景 RGB 约束后才分离。**结论是“RGB 监督范围的约束是必要条件，alpha/bg 项起辅助稳定作用”。**

---

## C4 — M2M3-GO 容量控制（Phase20E3）

### 核验到的原始事实

| Scene | M0 count | M1 count | M2 count | M0 FG PSNR | M1 FG PSNR | M2 FG PSNR | held-out views |
|---|---:|---:|---:|---:|---:|---:|---:|
| WanNianQing1 | 128,447 | 132,947 | **128,179** | 23.083 | 22.057 | **22.306** | 27 |
| CaoMei2 | 130,164 | 119,229 | **116,663** | 20.000 | 21.815 | **21.066** | 14 |
| HongZhang | 162,458 | 147,594 | **147,876** | 20.322 | 20.463 | **21.324** | 27 |

来源：`阶段二十E3.../M2M3_SCENE_SUMMARY.csv`（9 行，3 场景 × M0/M1/M2，**无 M3**）。

- **协议一致性**：三场景同迭代（30000）、同 seed（0）、同 GPU UUID、同一 M2M3 配置族；M1/M2 配置唯一差异是 `capacity_control_mode`（`m2m3` vs `m2m3_floor`）。
- **额外剪枝确实触发**：30/30 事件 EVALUATED 且删除数>0；额外删除 WanNianQing1 M1/M2 = 9,017/9,256；CaoMei2 = 12,312/6,882；HongZhang = 8,680/8,702。
- **Floor40 从未生效**：15/15 M2 事件 `floor_blocked_count=0` → `FLOOR40_BENEFIT = NO_EFFECT`。
- **容量差不可作因果估计**：M0/M1/M2 首次容量差异在 **iteration 601**（`PRE_PRUNING_TRAJECTORY_COMPARISON.csv`），即额外剪枝（18000 起）之前训练轨迹已分叉。
- **无独立三维 GT**：`independent_3D_GT=NONE`，`physical_units=arbitrary inherited COLMAP units`。
- **M0 是历史冻结模型**，非逐位相同重跑。

### 结论

- **状态：VERIFIED（数值）/ MIXED（效果方向）。**
- **展示方式**：**必须正负混合**。建议正文表格给出 M0/M1/M2 三行 × 三场景，明确标注 WanNianQing1 前景 PSNR **下降**（23.08→22.31）而 CaoMei2/HongZhang **上升**。
- **不可写入**：M2M3 提升质量；M2M3 降低容量（仅 CaoMei2/HongZhang 成立，WanNianQing1 反而略增）；Floor40 有保护作用；任何跨种子/跨序列结论。

---

## C5 — 2D 分割与 3D 重建收益不一致

### C5a：S24 / E4b 掩膜来源 → 2DGS 闭环（3 样本，同协议）

| 掩膜来源 | PSNR_fg | SSIM_fg | LPIPS_fg | outside ratio | leakage | Gaussian count | mesh 连通分量 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAM3 single prompt | 24.6598 | 0.8454 | 0.0358 | 0.023111 | 0.013969 | 402,497 | 5.67 |
| FSAM3-base | 24.6602 | 0.8458 | 0.0357 | 0.023313 | 0.014109 | 405,016 | 3.33 |
| **RAP-FSAM3-v2** | **23.2179** | 0.8303 | 0.0397 | 0.043381 | 0.032505 | **468,252** | 8.00 |

来源：`数据管理/05-评测结果/S24_E4b_mask_source_2DGS_closed_loop/experiment4b_summary.csv` + `experiment4b_per_sample.csv`（逐样本：KongQueZhuYu / XianKeLai1 / CaoMei2，3/3 样本中 RAP 掩膜的前景 PSNR 均最低且高斯数最多）。
**版本证据**：S24 的 RAP 掩膜 `参数.json` 中 `score_weights="area=1,comp=1,edge=1,temp=1,contrast=1"`（**无 `sam=0.5`**）→ R0 权重集，早于 R1 定稿。

✅ **“旧 RAP 掩膜使 3D 前景 PSNR 下降”成立**；✅ “高斯数量增加”成立；✅ 受控对照（同脚本、同场景、同评测口径）。

### C5b：Phase20E2 R1 vs R0 的 3D 稳健性（3 种子 × 2 场景）

| Scene | Endpoint | ΔS0 | ΔS1 | ΔS2 | 配对均值 | 配对 SD | B SD | C SD | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| WanNianQing1 | PSNR | +0.0017 | −0.0887 | +0.0007 | −0.0288 | 0.0519 | 0.0006 | 0.0513 | **INCONCLUSIVE** |
| WanNianQing1 | foreground_PSNR | +0.3281 | −1.0852 | +0.0214 | −0.2452 | 0.7435 | 0.1033 | 0.6411 | INCONCLUSIVE |
| CaoMei2 | PSNR | −0.0253 | −0.0251 | −0.0137 | −0.0214 | 0.0066 | 0.0093 | 0.0150 | **MIXED** |
| CaoMei2 | foreground_PSNR | −1.1009 | +0.6891 | −0.1144 | −0.1754 | 0.8965 | 0.6031 | 0.5601 | MIXED |

HongZhang B/C **共用同一 checkpoint**，无法建立 R1-vs-R0 差异。
最终角色判定（`PHASE20E2_FINAL_REPORT.md`）：`R1_3D_BENEFIT: MIXED`，`R1_FINAL_ROLE: ROBUST_SELECTION_ONLY`。

### 结论

- **状态：VERIFIED。**
- **可写入**：“在本管线上，二维分割改进不自动转化为三维前景质量改进：旧 RAP 掩膜在 3/3 样本上降低了 3D 前景 PSNR 并增加了高斯数量（S24）；在配对多种子 2DGS 训练中，R1 相对 R0 的三维前景 PSNR 差异为 MIXED（WanNianQing1：INCONCLUSIVE；CaoMei2：MIXED），HongZhang 因 B/C 共用 checkpoint 无法比较。”
- **必须禁止**：把该发现表述成“已解决问题”；或暗示 R1 已证明带来三维收益；或把 R1 的作用扩到 R1×refinement 联合（零证据）。

---

## C6 — 后训练压缩（Phase20G0）

| 项 | 内容 |
|---|---|
| 插件 | `plugins/gaussian_compression/`（commit `e31176ee`） |
| 方法 | **仅逐行剪枝**；`io.py::write_subset` 重写 PLY 保留行。**无量化、无熵编码、无 SH 压缩、无恢复微调**（`2DGS_COMPRESSION_COMPATIBILITY.md`：SH distillation/quantization = OUT OF SCOPE） |
| 模型 | **1 个**：WanNianQing1 / B_R0 / seed0 / M0，128,447 高斯，31,342,576 bytes |
| 运行 | 8 组 = {random, opacity} × {0, 0.2, 0.4, 0.6}，同 27 个 held-out 相机，1060×1900，SH degree 3，黑背景 |
| 指标 | 全图 PSNR 5.21→5.18/5.21→5.18；前景代理 PSNR 23.08→19.81(random,60%) / 19.16(opacity,60%)；PLY bytes 31.3MB→12.5MB；FPS 167.9→221.9；峰值 reserved VRAM 989.9MB→557.8MB |
| 阻塞项 | **PUP_2DGS = BLOCKED**（原生 ABI 无 Fisher buffer；2DGS 为 5 空间参数 + surfel 协方差）；**LightGaussian_2DGS = BLOCKED**（原生 ABI 无 count-render；官方 kernel 假设三激活尺度） |
| 状态 | `PHASE20G0 STATUS: PARTIAL`，`NEXT_PHASE: REMEDIATION_REQUIRED` |

### 结论

- **状态：PARTIALLY_VERIFIED。**
- **定位：工程级受控原型 / 基线基准**，不是压缩方法学贡献。
- **建议：正文一句话 + 补充材料表格。** 不可作为主结果或独立贡献；不可称“压缩方法”。

---

## C7 — Mesh 重建

- **产出**：3 场景 × M0/M1/M2 × raw/post = 18 个网格，全部经 TSDF 融合 → Marching Cubes → 原生连通分量后处理（`num_cluster=1`）→ 文件重读 → 结构复算，全部 `reload_verified=True`。
- **示例（`MESH_STRUCTURE_METRICS.csv`）**：WanNianQing1 M0 raw 1,271,632 顶点 / 2,388,092 面 / 20,147 连通分量 → post 1,135,076 / 2,241,526 / 1；CaoMei2 raw 792,827/1,226,040/41,555 → post 438,283/815,120/1；HongZhang raw 2,575,549/4,296,140/126,275 → post 1,672,621/3,293,400/1。
- **无几何 GT**：`independent_3D_GT=NONE`、`physical_units=arbitrary inherited COLMAP units`；协议明确“不计算真值 Chamfer 或 F-score”。
- **有代理指标**：网格-高斯深度 MAE/median/P95、ROI 覆盖代理、剪影 IoU —— **均为自洽性代理，不是精度**。
- **后处理保留最大连通分量是规则结果**，非“更准确”的证据；CaoMei2/HongZhang 原始最大组件仅占 66.5%/76.7% 面片，后处理可能删除真实孤立叶片。
- **无自定义 TSDF 机制**：正式路径为标准 2DGS TSDF；自定义 `edge_aware_postprocess_mesh` 存在但被明确排除。

### 结论

- **状态：PARTIALLY_VERIFIED（结构/定性）；几何精度 UNSUPPORTED。**
- **可写入**：网格可稳定生成、可重载、保留可审计的 raw/post 结构记录与代理一致性指标。
- **不可写入**：任何“结构精度/几何准确度提升”；任何 Chamfer/F-score；任何厘米级尺寸。

---

## C8 — 表型测量

### 核验到的原始事实

- 论文 Table 8 的四个 R²（0.9878 / 0.9879 / 0.9738 / 0.8999）与 MAE/MAPE 全部来自 **工作簿 `植株数据.xlsx`**（1 sheet、21 植株行、17 列、人工“真值”vs“虚拟”值），由 `03-扩展实验后续执行/生成实验一五当前结果.py` 汇总。该脚本 **不读取任何 mesh / 2DGS 模型 / mask**。
- 工作簿 **无盆径、无 scale marker、无 scale factor、无 specimen ID、无采集时间**。
- 配对状态（`阶段二十F.../HISTORICAL_PHENOTYPE_MATCHING.csv`）：多数 `NO_MATCH`；WanNianQing/CaoMei `AMBIGUOUS`（各两行）；HongZhang `SAMPLE_MATCH_ONLY`；叶位 `LEAF_ID_NOT_AVAILABLE`；`virtual_values_are_new_mesh_results=false` 处处成立。
- **Phase20F 实测**（`PHASE20F_FINAL_REPORT.md`）：`PHYSICAL_SCALE: NOT_AVAILABLE`、`VERTICAL_REFERENCE: NOT_AVAILABLE`、`PLANT_HEIGHT: GEOMETRIC_ONLY`、`CANOPY_WIDTH: GEOMETRIC_ONLY`、`LEAF_LENGTH: BLOCKED`、`LEAF_WIDTH: BLOCKED`；“**How many valid centimetre values were computed? Zero.**”；`PHENOTYPE_ERROR_METRICS.csv` 为**仅表头**；`TABLE8_STATUS: REPLACEMENT_REQUIRED`。
- Phase20F 分析的 3 个网格是 **M0 raw/post**（源自 E1 的 B_R0/C_R1 模型），而这些运行 `use_foreground_rgb_loss=false, use_mask_loss=false, use_bg_opacity_loss=false, mask_mode=none` —— **既不是 A5 联合损失配置，也不是 M2M3**。表型统计也 **不** 来自这些网格。
- `ForeSplat_real_release/examples/paper_tables/strict_multimask_status.csv` 自身声明：`not_supported_claim = "specific segmentation models causally produce specific phenotype errors"`。

### 结论

- **状态：CONTRADICTED（作为 ForeSplat 方法结果）。**
- **可写入**：作为 **legacy 记录统计**（明确标注来源为工作簿、无可复现 3D 链路），或移入 Limitations。
- **不可写入**：Abstract / Contributions / Conclusion 中 “ForeSplat achieved R²=0.988 …” 一类表述；任何“通过 ForeSplat 验证了虚拟表型精度”的因果陈述。
- **替代路径**：把 Phase20F 的真实结论写进 Discussion —— “在当前采集与建模协议下，由于缺少物理尺度参照与器官身份标识，可验证的厘米级表型测量尚不可得”，这是一条**有价值且诚实**的研究结论。

---

## 附：证据分类速查

| 类别 | 内容 |
|---|---|
| **新版正式实验** | Phase20D 固定轮廓 32 帧；Phase20E1/E2 配对多种子 2DGS；Phase20E3 M0/M1/M2 + Mesh；Phase20G0 压缩 |
| **历史算法验证** | S10 A0–A6（KongQueZhuYu）；S24/E4b 掩膜来源闭环；Phase20B 重算的 Easy20 分割基线 |
| **单序列实验** | 全部 A0–A6（A5 证据仅 1 场景 1 次运行） |
| **工程 smoke test** | Phase20G0 压缩（1 模型 2 基线）；`A6_*_smoke*` 目录 |
| **未实现方法设计** | 语义门控 softmax；视图质量三信号公式；M2M3 双向 deficit/边界效用/局部 Floor40；mask-aware TSDF 权重公式 |
| **负向实验结果** | S24（更好的 2D 掩膜→更差的 3D）；Phase20E2（R1 三维 MIXED）；Phase20E3（Floor40 NO_EFFECT、M2M3 MIXED）；Phase20F（表型 BLOCKED） |
