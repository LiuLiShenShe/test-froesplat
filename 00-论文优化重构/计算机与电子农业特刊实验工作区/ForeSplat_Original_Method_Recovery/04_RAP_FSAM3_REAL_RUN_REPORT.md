# 04 — RAP-FSAM3 真实运行报告

生成时间：2026-10-25（持续更新）
判定口径：见 `01_PAPER_CODE_MAPPING.md` §0

本报告只记录**真实运行**的产物，不包含单测结果、配置存在性声明或历史结果复用。
每一节都给出可复核的命令、落盘路径与哈希来源。

---

## 1. 运行环境

| 项 | 值 |
|---|---|
| 解释器 | `/home/test/.venvs/sam3/bin/python`（system-site-packages + iopath/timm/ftfy/pycocotools） |
| GPU | NVIDIA 2×48 GB（SAM3 需 `--device cuda`；CPU 上 ViTDet 有 BFloat16/Float dtype 不匹配，属上游已知限制） |
| SAM3 | `第三方源码/SAM3-latest` + `sam3/sam3.pt` |
| COLMAP loader | `2d-gaussian-splatting-main/scene/colmap_loader.py` |
| 生产入口 | `数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py` |

---

## 2. 运行 R6-1：KongQueZhuYu 5 帧（首次全链路实跑）

| 项 | 值 |
|---|---|
| 目录 | `runs/R6_RAP_FULL_real/` |
| 输入 | `S21_KongQueZhuYu_E2_E3/selected_frames/{0000,0025,0050,0075,0100}.jpg`（2160×3840） |
| COLMAP | `04-COLMAP/KongQueZhuYu/sparse/0`（注册点 1542/1891/631/98/5064） |
| 开关 | `--use_semantic_gate --semantic_gate_mode gated_softmax --semantic_gate_temperature 0.25 --ranking_policy r1 --spnp_prompt_form points --spnp_negative_region_mode manuscript --use_residual_repair --thin_repair_mode adjacent --corrective_geometry_backend colmap_tracks_gaussian --strict_colmap_observations` |
| 耗时 | 484.4 s（5 帧，含 SAM3 加载） |
| 状态 | ✅ 完整 RAP-FSAM3，6/6 机制执行 |

### 2.1 逐帧执行记录

| frame | SAM3 候选 | SG-MPS 选中 | CDR | SB-FBPR 实发 | 修复 px | GCFC 注册点 | $G_i$ | GCFC 新增 | GCFC 接受 | F1 并集GT | F1 带盆GT | F1 纯植株 | HD95(带盆) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 23 | P2 | r1 | 5 正 + 8 负点，0 框 | 26 | 1542 | 0.9752 | 151 | ✔ ok | 0.9740 | **0.9832** | 0.8215 | 12.7 |
| 0025 | 7 | P2 | r1 | 5 + 8 点 | 21 | 1891 | 0.9892 | 227 | ✔ ok | 0.9872 | **0.9872** | 0.8572 | 19.0 |
| 0050 | 5 | P2 | r1 | 5 + 8 点 | 24 | 631 | 0.9677 | 456 | ✔ ok | 0.9747 | **0.9810** | 0.8067 | 25.1 |
| 0075 | 5 | **P1** | r1 | 5 + 8 点 | 12 | 98 | 0.9897 | 0 | ✗ `no_corrective_delta` | 0.9144 | **0.9227** | **0.9800** | 282.8 |
| 0100 | 13 | P2 | r1 | 5 + 8 点 | 10 | 5064 | 0.9918 | 223 | ✔ ok | 0.9838 | **0.9874** | 0.9911 | 13.7 |

**指标来源**：并集 GT 的 F1 来自生产脚本自身的 `f1_if_gt_available` 列（`--gt_mask_dir` 指向 `S21_KongQueZhuYu_E2_E3/gt_masks`）；带盆 / 纯植株 / 花盆 / 蓝立方块四口径由 `metrics/kqz5_R6_vs_splitGT.json` 给出（含逐帧掩膜与 GT 的 SHA256、面积、Boundary-F1、HD95、15 px 带外泄漏率）。**上一版表中 0075 的"带盆GT=0.9833"是错的**——重新按拆分 GT 计分后为 0.9227，已更正。

### 2.2 关键观察

1. **SG-MPS 真实改变了提示词选择**：frame 0075 的 softmax 为 P1 0.5396 / P2 0.4604，门控把选择从默认 P2 切到 P1。这正是论文所述"语义不一致提示词的纠偏"行为，不是空转。
2. **SB-FBPR 真的发送了点提示**：`SB-FBPR提示证据.csv` 5 帧全部 `points_sent_positive=5`、`points_sent_negative=8`、`boxes_positive=0`。**注意**：本 run 的 CSV 只有 12 列，逐点坐标保存在 `point_coords_normalized` 里，用**前导 `+`/`-` 标记正负**（这是标签标记，不是坐标符号）。为消除歧义，生产脚本随后新增了 `point_coords_unsigned`（纯 [0,1] 坐标）与 `point_labels`（`1`=正 / `0`=负）两列；这两列**只在本次 210 帧 run 及之后生效**，本 5 帧 run 未重跑（不覆盖历史产物）。
3. **GCFC 用的是真实 COLMAP 观测**：`gcfc_has_real_observation=1` 5/5，$G_i$ ∈ [0.9677, 0.9918]，新增像素 0–456。
4. **frame 0075 的拒绝是真实的**：该帧仅 98 个注册点，支撑场稀疏到不产生任何 $H>\tau_g$ 且落在边界带内的像素，因此 GCFC 正确报告 `no_corrective_delta` 并拒绝修正。这不是失败案例，是机制按设计工作。
5. **frame 0075 是本次 5 帧样本中唯一一次 SG-MPS 改变选择，也是唯一一次掉分**：门控从 P2 切到 P1，带盆 GT 的 F1 从相邻帧的 0.981–0.987 掉到 **0.9227**，HD95 从 13–25 px 涨到 **282.8 px**；同一帧按纯植株 GT 反而 0.9800。**这一帧同时证明了两件事**：(a) SG-MPS 确实在起作用，不是空转；(b) 它的作用是在优化"纯植株"口径，与论文 §2.3.2 声明的 potted-plant 目标不一致。该结论与 `05` 的 16 帧消融完全一致。

   附注：0075 帧的 98 个注册点也解释了为什么 GCFC 在该帧无法补洞（`no_corrective_delta`）——支撑场太稀疏，边界带内不存在 $H>\tau_g$ 的像素。

---

## 3. GT 口径声明（重要）

仓库里存在两套 GT：

| GT | 路径 | 内容 | 是否本次主口径 |
|---|---|---|---|
| 并集 GT | `S21_KongQueZhuYu_E2_E3/gt_masks/` | 植物 + 花盆 + 蓝立方块 | 仅作对照 |
| 拆分 GT | `GT口径拆分审计/gt_masks_split_v2/<scene>/` | `mask_plant_*` / `mask_pot_*` / `mask_potted_plant_*` / `mask_blue_cube_*` | ✅ 主口径：`mask_potted_plant_*` |

理由：论文 §2.3.2 明确把 RAP-FSAM3 的目标定义为 **potted-plant 前景**（"a potted-plant foreground prompt rather than a strict plant-body-only prompt"）。用并集 GT 评价一个以"带盆植株"为目标的机制，等于用错误的目标函数打分。

**所有指标表均带 `gt_convention` 列**，绝不混用两种 GT 而不声明。

---

## 4. 运行 R6-2：KongQueZhuYu 全场景 210 帧（规模验证）

| 项 | 值 |
|---|---|
| 目录 | `runs/KongQueZhuYu_RAP_FULL_210/` |
| 输入 | `02-位姿COLMAP/03-final_locked/KongQueZhuYu/images`（210 帧，全部有效 JPEG） |
| COLMAP | 同目录（`--colmap_dir` 指向 dataset 根，`find_sparse_dir` 自动定位 `sparse/0`） |
| 状态 | ✅ **已完成**（210/210 最终掩膜，5 h 47 min） |

用途：证明 GCFC 在全场景、稀疏点分布高度不均（部分帧极少观测）时仍不会静默失败——`--strict_colmap_observations` 会因缺观测直接 raise。

> ⚠️ CaoMei1 的 `04-COLMAP/CaoMei1/images` 中有 4 个损坏文件（`0030.jpg`/`0032.jpg` 是许可证文本、`0033.jpg` 是 HTML、`0140.jpg` 是二进制垃圾），PIL 抛 `UnidentifiedImageError`。这是**原始数据损坏**，不是代码缺陷。`02-位姿COLMAP/03-final_locked/CaoMei1/images` 同样含这 4 个坏文件，因此 CaoMei1 不能直接跑 210 帧。已改用 `01-输入图像/02-fft_frames/CaoMei1`（210 帧全部有效）做下游全场景。

---

## 5. 运行 R0–R6：跨场景消融（开发集，与冻结评测集不同）

| 项 | 值 |
|---|---|
| 目录 | `runs/dev_ablation_v2/` |
| 脚本 | `scripts/run_dev_ablation.py` |
| 开发集 | CaoMei1 / ChangShouHua2 / DouBanLv1 各 5 帧 + XianKeLai1 1 帧 = 16 帧 / 4 场景 |
| 排除 | KongQueZhuYu（S21 冻结评测场景） |
| 输入源 | `01-输入图像/02-fft_frames/<scene>/`（已过 FFT 过滤的有效帧） |
| GT | `gt_masks_split_v2/<scene>/mask_potted_plant_*`（主）+ `mask_plant_*` + `mask_pot_*` |
| 配置 | R0 base / R1 r1 / R2 SG-MPS / R3 SG-MPS+CDR / R4b 框提示 / R4p 点提示 / R5 +残差修复 / R6 +GCFC |

**候选池隔离**：每个场景先跑 R6（真实 SAM3 推理，P1–P5 逐实例候选落盘），其余档位用 `--reuse_existing_candidates --candidate_source_dir` 复用**同一个候选池**。因此 R0–R6 的差异**只来自选择/细化规则**，不来自候选集不同——这是消融有效性的前提。

**COLMAP 严格模式**：R6 使用 `--strict_colmap_observations`，任一帧缺观测即 raise，不会静默降级。

### 5.1 运行状态（全部 rc=0）

| 场景 | 档位数 | 状态 |
|---|---|---|
| CaoMei1 | 10（R6 + R0/R1/R2/R3/R3n/R3g/R4b/R4p/R5） | ✅ 全部 rc=0 |
| ChangShouHua2 | 10 | ✅ 全部 rc=0 |
| DouBanLv1 | 10 | ✅ 全部 rc=0 |
| XianKeLai1 | 10（1 帧） | ✅ 全部 rc=0 |

逐档实测 wall time 见 `05_RAP_FSAM3_ABLATION.md` §8（受同期 GPU 任务影响，只作量级参考）；逐档完整命令行见 `logs/dev_<scene>_<config>.log` 首行。评分由 `scripts/run_dev_ablation.py --score-only --workers 12` 完成（480 个 (场景×档位×帧×GT口径) 组合），结果在 `metrics/dev_ablation_*.csv`。

---

## 6. 运行 R6-3：KongQueZhuYu 全场景完整 RAP（下游用）

| 项 | 值 |
|---|---|
| 目录 | `runs/KongQueZhuYu_RAP_FULL_210/` |
| 用途 | 产出 210 帧完整 RAP-FSAM3 掩膜，喂给 Foreground-Object 2DGS，替代冻结的 `C_R1` 掩膜 |
| 输入 | `02-位姿COLMAP/03-final_locked/KongQueZhuYu/images` |
| 掩膜命名 | `最终掩膜/mask_<stem>.png`，与 2DGS `--mask_pattern 'mask_{stem}.png'` 一致 |
| 状态 | ✅ **完成，210/210**（2026-10-25 13:03 → 18:50，约 5 h 47 min） |

### 6.1 全场景逐阶段统计（`runs/KongQueZhuYu_RAP_FULL_210/逐帧报告.csv`，40 列 × 210 帧）

| 阶段 | 执行帧数 | 结果 |
|---|---|---|
| SAM3 多提示词逐实例候选 | 210 | 每帧 5 个提示词全部推理 |
| SG-MPS 门控 | 210 | 选中分布：**P2 156（74.3%）**、P1 29（13.8%）、P3 15（7.1%）、P5 5、P4 5 |
| CDR/R1 | **210**（`cdr_scope=multi_prompt`） | 全部帧都在多提示词候选池上执行 |
| SB-FBPR 点提示 | 210 | 接受 **209**，1 帧 `low_iou_guard`（0008） |
| 残差/细结构修复 | 210 | 恢复 4–101 px/帧，均值 17.0 px |
| GCFC | 210（`gcfc_has_real_observation=1` **210/210**） | 接受 **205**，5 帧 `support_gate` 拒绝（0008/0070/0071/0072/0073） |

GCFC 支撑场 $G_i$：均值 **0.8209**，范围 0.0024–0.9854；新增像素 0–12252 px/帧，均值 3049 px。
被拒的 5 帧全部是 `support_gate`（$G_i$ 低于 $\tau_g=0.15$，其中 0008 帧只有 0.0024），**不是静默失败**——拒绝原因逐帧落盘在 `gcfc_reason` 与 `failure_reason`。

### 6.2 与 5 帧样本的对照

| 指标 | 5 帧（0000/0025/0050/0075/0100） | 210 帧全场景 |
|---|---|---|
| SG-MPS 选 P2 的比例 | 4/5 | 156/210（74.3%） |
| GCFC 接受率 | 4/5 | 205/210（97.6%） |
| GCFC 平均新增 px | 211 | 3049 |
| 逐帧报告列数 | 39 | **40**（多 `cdr_scope`） |

⚠️ 5 帧样本的 GCFC 接受率（4/5）**明显低估**了全场景表现（205/210），因为样本里包含了注册点最少的 0075 帧。这说明"用 5 帧抽样判断机制行为"是不够的——这也是本轮必须跑完 210 帧的理由。

**掩膜来源哈希（已验证）**：`runs/downstream/KongQueZhuYu_RAP_FULL/mask_provenance_expected.json` 在训练前记录了
210 张掩膜的 SHA256；训练启动后 `train.py` 写出 `mask_provenance.json`，其中 **183 个有相机张量的训练视角**
的掩膜 SHA256 与预期清单**逐条一致（0 处不符）**，`missing_count = 0`。因此 RAP_FULL 训练消费的确实是
`runs/KongQueZhuYu_RAP_FULL_210/最终掩膜` 的产物，不是任何历史掩膜。

---

## 7. 历史回放一致性（证明改动没有污染历史路径）

```bash
# 与阶段二十C 的 V0/Easy21 比较
A=../阶段二十C_FIX_候选元数据与精简细化验证/runs/V0/Easy21/最终掩膜
B=runs/replay_check/Easy21_V0/最终掩膜
```

| 配置 | 结果 |
|---|---|
| `--ranking_policy base --spnp_prompt_form boxes --spnp_negative_region_mode ring` | **0/21 掩膜字节不同** |

新增的 `points` / `manuscript` 分支与 GCFC 观测加载改动，在默认参数下不影响历史字节级输出。

---

## 8. 失败案例与异常记录（不隐藏）

| 编号 | 现象 | 归因 | 处置 |
|---|---|---|---|
| F1 | `METADATA_UNVERIFIED: P4/0` 警告 | P4（`crop seedling`）在多帧 0 检测，缓存中存全零占位掩膜 | 评分阶段空掩膜硬门已拦截（`primary_score=0`），警告**准确**，不应消除 |
| F2 | CaoMei1 4 个损坏 JPEG | 原始数据损坏（文本/HTML/垃圾二进制） | 换用 `02-fft_frames/CaoMei1`；记录在案 |
| F3 | GCFC frame 0075 `no_corrective_delta` | 该帧仅 98 个注册点，支撑场太稀疏 | 机制按设计拒绝修正，非缺陷 |
| F4 | SAM3 在 CPU 上 dtype 报错 | SAM3/ViTDet 上游限制 | 全部运行用 `--device cuda` |
| F5 | 04 早期表格中 0075 帧"带盆GT = 0.9833" | 早期未经拆分 GT 计分、按相邻帧推断 | 已用 `metrics/kqz5_R6_vs_splitGT.json` 更正为 **0.9227**，并在表下标注 |

---

## 9. 尚未运行 / 阻塞项

| 编号 | 项 | 状态 |
|---|---|---|
| P-1 | ChangShouHua2 / DouBanLv1 / XianKeLai1 消融档位 | ✅ 已完成（4 场景 × 10 档全部 rc=0，指标见 `05`） |
| P-2 | KongQueZhuYu 210 帧完整 RAP | ✅ **已完成**（210/210，逐阶段统计见 §6.1） |
| P-3 | 完整 RAP 掩膜 → Foreground-Object 2DGS 训练 | ✅ **已完成**：2×2 四档全部 COMPLETE（2132–2201 s，`metrics/downstream_2x2.csv`） |
| P-4 | M2M3-GO 回归 + mesh | ✅ M2M3-GO 两档真实执行（19 次更新/档）+ 四档 TSDF 网格已提取（`runs/mesh/`）；⚠ 结论为负，见 `07` §3.1/§6、`08` §4 |
| P-5 | 论文公式/表格同步 | 未开始（差异清单见 `09`） |
| P-6 | SG-MPS 定位的论文级决策（安全机制 vs 换 $\phi$ 定义） | 需作者拍板，见 `05` §3 与 §9 |
