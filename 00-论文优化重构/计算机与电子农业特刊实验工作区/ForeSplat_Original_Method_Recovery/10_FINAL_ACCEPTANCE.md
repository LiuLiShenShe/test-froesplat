# 10 — 最终验收

生成时间：2026-10-25
判定口径：`01_PAPER_CODE_MAPPING.md` §0（L1–L6）
**本文件是唯一权威判定。其余文件中的 PASS/VERIFIED 若与此冲突，以本文件为准。**

---

## 0. 一句话结论

**RAP-FSAM3 的六个原稿机制已全部在真实代码中实现、在真实调用链上、并在真实数据上端到端运行（L1–L4 达成，L5 部分达成）。下游闭环也已贯通到 mesh（完整 RAP 掩膜 → 2DGS → M2M3-GO → TSDF 网格，四档全部完成）。但物理表型因缺少经过核验的真实尺度而 BLOCKED，论文同步未做，且消融与 2×2 都给出了对 SG-MPS / 完整 RAP 掩膜不利的明确证据。因此不能声明"方法有效"，更不能声明"全部完成"。**

---

## 1. 逐模块验收

| 模块 | L1 数学 | L2 代码 | L3 调用链 | L4 真实运行 | L5 行为证明 | L6 独立实验 | 状态 |
|---|---|---|---|---|---|---|---|
| FFT 帧质量过滤（Eq.1） | ✔ | ✔ | ✔ | ✔ | ✔ 33/210 帧判定不同 | ✔ | **VERIFIED** |
| 多提示词逐实例候选（K=5） | ✔ | ✔ | ✔ | ✔ | ✔ 23/7/5/5/13 候选 | ✔ 4 场景 16 帧 | **VERIFIED** |
| SG-MPS：φ/ψ/c | ✔ | ✔ | ✔ | ✔ | ✔ 落盘逐帧 | ✔ | **IMPLEMENTED** |
| SG-MPS：提示词级 softmax a | ✔ | ✔ | ✔ | ✔ | ✔ Σ=1，同提示词共享概率 | ✔ | **VERIFIED** |
| SG-MPS：k* 选择 | ✔ | ✔ | ✔ | ✔ | ✔ 真实切换（KQZ 0075） | ✔ 4 场景 16 帧 | **PARTIALLY_VERIFIED**（dev 集上选错） |
| SG-MPS：承诺裕度 | 新增参数，非原稿 | ✔ | ✔ | ✔ | ✔ 近 ties 让位，但宽裕度无效 | ✔ | **IMPLEMENTED** |
| CDR/R1 实例重排 | ✔ | ✔ | ✔ | ✔ | ✘ 逐帧选择与 base 完全一致 | ✘ | **PARTIALLY_VERIFIED** |
| SB-FBPR 正点 TopK | ✔ | ✔ | ✔ | ✔ | ✔ 单测 + 真实 5 点 | ✔ | **VERIFIED** |
| SB-FBPR 负区四分区 | ✔ | ✔ | ✔ | ✔ | ✔ 单测 + 真实 8 点跨区 | ✔ | **VERIFIED** |
| SB-FBPR NMS 负点 | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | **VERIFIED** |
| SB-FBPR 点提示注入 SAM3 | ✔ | ✔ | ✔ | ✔ | ✔ 5 正 + 8 负、0 框 | ✔ | **VERIFIED** |
| 残余/细结构修复 | 无原稿公式 | ✔ | ✔ | ✔ | ✔ 10–26 px | ✘ 零增量 | **RUNNABLE** |
| GCFC 支撑场 H | ✔ | ✔ | ✔ | ✔ | ✔ 调试图 | ✔ | **VERIFIED** |
| GCFC 一致性 G | ✔ | ✔ | ✔ | ✔ | ✔ 0.9677–0.9918 | ✔ | **VERIFIED** |
| GCFC 边界带正向补洞 | ✔ | ✔ | ✔ | ✔ | ✔ 151–456 px | ✔ | **VERIFIED** |
| GCFC 真实观测加载 | ✔ | ✔ | ✔ | ✔ | ✔ 5/5 帧 | ✔ | **VERIFIED** |
| Foreground-Object 2DGS | ✔ | ✔ | ✔ | ✔ 4 档中 3 档完成 | ⚠ 行为忠实但指标下降 | ✔ | **VERIFIED（实现）/ 结果不利** |
| M2M3-GO | ✔ | ✔ | ✔ | ✔ RAP 掩膜上已跑 | ⚠ 执行真实、无增益 | 部分（2×2 收尾中） | **RUNNABLE** |
| TSDF mesh 提取（§2.8） | ✔ | ✔ | ✔ | ✔ 四档 | ✔ 连通域/碎块/几何量 | ✔ 四档 | **VERIFIED** |
| 表型几何（探索性，COLMAP 单位） | ✔ | ✔ | ✔ | ✔ | ✔ 高/冠幅可测 | ✔ | **PARTIALLY_VERIFIED** |
| 物理表型（cm） | — | ✔ 接口齐备 | — | ✘ | ✘ | ✘ | **BLOCKED**（无真实尺度） |

---

## 2. 用户明确要求的四件事

### 2.1 P0-A：GCFC 观测加载 ✅ 已修复并验证

- 修复前：`colmap_tracks_gaussian` 后端拿到空观测 → 每帧静默 `missing_colmap_observation`。
- 修复后：两后端同等加载；`--strict_colmap_observations` 缺观测直接 raise。
- 真实证据：KQZ 5 帧注册点 1542/1891/631/98/5064，`gcfc_has_real_observation=1` 5/5，G=0.9677–0.9918，新增 151/227/456/0/223 px。
- 调试图：`GCFC_before/support_field/boundary_band/added_pixels/after` 逐帧落盘。

### 2.2 P0-B：SB-FBPR 点提示 ✅ 已修复并验证

- 先查后改：确认 SAM3 原生支持 `Prompt.append_points`（geometry_encoders.py:343 → _encode_points:589）。
- 真实证据：`SB-FBPR提示证据.csv` 5 帧全部 `points_sent_positive=5`、`points_sent_negative=8`、`boxes_positive=0`，逐点归一化坐标 + 标签落盘。
- 论文口径：正点 = 距离变换 TopK；负点 = NMS，覆盖 ring∪table∪bg∪neighbour。
- 兼容性：默认 `--spnp_prompt_form boxes` + `--spnp_negative_region_mode ring` 时与历史 `V0/Easy21` **0/21 字节差异**。

### 2.3 P0-C：SG-MPS 归一化与分层 ✅ 已修复，⚠ 结论不利

- 归一化：已改为按**提示词**做 softmax，同提示词多实例共享概率。
- 分层：SG-MPS 选提示词 → CDR 选实例；`--semantic_gate_use_contrast` 默认 False，避免与 CDR 重复计权。
- **但**：dev 集上 SG-MPS 选错提示词，带盆 GT 的 16 帧 F1 从 0.9763（无门控）掉到 0.7504。见 §3。

### 2.4 全链路真实运行 ✅ 5 帧 + 210 帧 + 下游 + 网格

- 5 帧：`runs/R6_RAP_FULL_real`（484.4 s，39 列逐帧报告；四口径 GT 计分见 `metrics/kqz5_R6_vs_splitGT.json`）。
- 210 帧：`runs/KongQueZhuYu_RAP_FULL_210`（最终掩膜 **210/210**，5 h 47 min）。
- 下游 2×2：`runs/downstream/` 四档全部 COMPLETE（`metrics/downstream_2x2.csv`）。
- TSDF 网格：`runs/mesh/<tag>/` ×4 + `metrics/mesh_phenotype.csv`、`metrics/mesh_geometry.json`。

### 2.5 KQZ 5 帧四口径计分（`metrics/kqz5_R6_vs_splitGT.json`）

| 帧 | 带盆 GT F1 | 纯植株 F1 | 花盆 F1 | 泄漏率 | 备注 |
|---|---|---|---|---|---|
| 0000 | 0.9832 | 0.8215 | 0.4382 | 0.00013 | 门控选 P2 |
| 0025 | 0.9872 | 0.8572 | 0.3795 | 0.00013 | 门控选 P2 |
| 0050 | 0.9810 | 0.8067 | 0.4632 | 0.00003 | 门控选 P2，GCFC 新增 456 px |
| 0075 | **0.9227** | **0.9800** | 0.0043 | 0.00018 | **门控切到 P1**：带盆口径掉分、纯植株口径反超 |
| 0100 | 0.9874 | 0.9911 | 0.0005 | 0.0 | 门控选 P2 |

---

## 3. 消融的核心发现（负面，必须报告）

完整数据见 `05_RAP_FSAM3_ABLATION.md` §1–§5（`metrics/dev_ablation_*.csv`，16 帧 / 4 场景 / 10 档 / 480 次计分）。

带盆 GT 的 16 帧均值：

| 档位 | 全 16 帧 F1 | oracle 选择率 | HD95 mean |
|---|---|---|---|
| R0 单提示 P2 + base | **0.9828** | 1.000 | 41.3 |
| R1 单提示 P2 + CDR | 0.9828 | 1.000 | 41.3 |
| R3n 多提示无门控 + CDR | 0.9763 | 0.938 | 69.1 |
| R3g 门控 + 裕度 0.05 | 0.9435 | 0.812 | 145.5 |
| R2 / R3 门控 | 0.7504 | 0.125 | 547.4 |
| R4b 框提示 | 0.7559 | 0.125 | 545.9 |
| R4p 点提示 | 0.7516 | 0.125 | 548.8 |
| R5 +残差修复 | 0.7516 | 0.125 | 548.8 |
| R6 完整 RAP | 0.7517 | 0.125 | 548.1 |

**归因**：回退完全由 SG-MPS 的提示词选择造成。逐帧检查：**oracle 在 16/16 帧都选 P2，门控只选对 2/16**（分布 P1×8、P3×3、P5×2、P2×2、P4×1）。裕度 0.05 只能消掉近 ties 的错选，DouBanLv1 0025/0050 两帧差值 0.0567/0.0540 越过阈值仍被错选。

**这不是实现 bug**：$c$ 公式、提示词级 softmax、argmax 全部按论文实现并单测通过。

**关键的互补证据**：把同一批掩膜换成纯植株 GT，R6 = 0.9575、R0 = 0.7392（完全颠倒）。所以 SG-MPS 并不是"随机出错"，而是在**正确地优化一个与论文声明不同的目标**——它偏好 `green plant` 类提示词。这是 $\phi$ 缺少"花盆应被包含"证据项导致的**定义冲突**，需论文层面二选一（见 §6.5）。

---

## 4. 下游闭环

| 项 | 状态 | 证据 |
|---|---|---|
| C_R1 基线 2DGS（30k, res 2） | ✅ COMPLETE | PSNR_fg 25.019 / SSIM_fg 0.8384 / LPIPS_fg 0.1183 / outside 0.0292 / leakage 0.0128，27 test views，`mask_provenance` 183 cameras 0 missing |
| C_R1 + M2M3-GO | ✅ COMPLETE | PSNR_fg 25.029 / SSIM 0.8387 / LPIPS 0.1182 / outside 0.0292 / leakage 0.0128；19 次更新，剪枝 115 825 |
| RAP_FULL 2DGS | ✅ COMPLETE | PSNR_fg 24.734 / SSIM 0.8244 / LPIPS 0.1280 / outside 0.0657 / leakage 0.1316；provenance 183/183 SHA256 与预期一致 |
| RAP_FULL + M2M3-GO | ✅ COMPLETE | PSNR_fg 24.610 / SSIM 0.8135 / LPIPS 0.1307 / outside 0.0645 / leakage 0.1319；19 次更新，剪枝 107 955 |
| TSDF 网格（四档） | ✅ 已提取 | 标准 TSDF，74–106 s/档；`runs/mesh/<tag>/mesh_post.ply` + `mesh_metrics.json` |
| 表型几何（探索性） | ⚠ 已测，无尺度 | 高 3.541–3.561 u、冠幅 3.122–3.142 u；单位是 COLMAP 单位，`physical_status=PHYSICAL_SCALE_NOT_AVAILABLE` |
| 物理表型（cm） | ⛔ BLOCKED | 无经过核验的真实世界距离（与阶段二十F 同一证据门） |

**下游 2×2 结论（全部完成，结论为负）**：

| 主效应 | ΔPSNR_fg | ΔSSIM |
|---|---|---|
| 掩膜来源 C_R1→RAP（M2M3 关） | **−0.285 dB** | −0.0140 |
| M2M3-GO（C_R1 掩膜） | +0.011 dB | +0.0003 |
| M2M3-GO（RAP 掩膜） | −0.125 dB | −0.0109 |

即：**换用完整 RAP 掩膜让 3D 前景重建变差（−0.29 dB、leakage 10×），M2M3-GO 也救不回来，在好掩膜上仅 +0.011 dB（可忽略）。** 两项原稿机制在本场景的 3D 端都**没有表现出增益**。

---

## 5. 交付物清单

| 文件 | 状态 |
|---|---|
| `01_PAPER_CODE_MAPPING.md` | ✅ |
| `02_RAP_FSAM3_FULL_CHAIN_AUDIT.md` | ✅ |
| `03_RAP_FSAM3_FIX_LOG.md` | ✅（10 项修复 + 5 项明确保留） |
| `04_RAP_FSAM3_REAL_RUN_REPORT.md` | ✅（210 帧全量） |
| `05_RAP_FSAM3_ABLATION.md` | ✅（4 场景 16 帧 × 10 档，完整指标表 + 负面结果） |
| `06_FOREGROUND_2DGS_INTEGRATION.md` | ✅（2×2 四档全部 COMPLETE） |
| `07_M2M3_GO_INTEGRATION.md` | ✅（2×2 回归 + TSDF 网格 + 表型几何） |
| `08_END_TO_END_VALIDATION.md` | ✅ |
| `09_MANUSCRIPT_SYNC_REPORT.md` | ✅（**未改论文**，只列差异） |
| `10_FINAL_ACCEPTANCE.md` | ✅（本文件） |
| `feature_matrix.csv` | ✅ 24 行 |
| `run_manifest.json` | ✅ 55 个 run（含 4 个 mesh run；`scripts/build_deliverables.py` 生成） |
| `configs/` | ✅ 5 个快照 + README（`scripts/build_configs.py` 从各 run 的 `参数.json` 生成，不手写） |
| `scripts/` | ✅ 9 个：`run_rap_ablation.py`、`run_dev_ablation.py`、`run_downstream_2dgs.py`、`run_downstream_2x2.sh`、`run_mesh_phenotype.py`、`build_deliverables.py`、`build_configs.py`、`build_figures.py`、`build_downstream_2x2.py` |
| `tests/` | ✅ 9 项契约测试（含网格/表型 3 项），与阶段二十一 54 项合计 **63 passed** |
| `metrics/` | ✅ dev 消融全套 + KQZ 5 帧四口径计分 + rap 消融 + 2×2 + mesh 几何 |
| `figures/` | ✅ 6 张（`scripts/build_figures.py` 从 run 产物直接生成）+ README |

---

## 6. 不能声明完成的原因（逐条）

1. **L6 部分达成**：dev 集有 GT 的帧只有 16 个，且 SG-MPS 在其上表现极差；无法据此声称 RAP-FSAM3 方法有效。
2. **完整 RAP 掩膜已齐**（210/210），2×2 下游四档已跑完，但**结论为负**（见 §4）。
3. **物理表型未交付**：网格与探索性几何已出，但无真实尺度 → `PHYSICAL_SCALE_NOT_AVAILABLE`（证据门，不是遗漏）。
4. **论文未修改**（用户要求"论文方法定义 = 代码"，当前中文草稿仍是简化版 FSAM3+PCA，与原稿和代码都不一致）。
5. **SG-MPS 的方法学问题未解决**：这是**论文级决策**——是把门控降级为"安全机制"并如实报告，还是给 $\phi$ 补上花盆证据项，**需要用户拍板**，不属于可自动决定的代码修改。
6. `configs/` 与 `figures/` 未按交付物清单建立。→ **已补上**（`scripts/build_configs.py`、`scripts/build_figures.py`）。

---

## 7. 建议的下一步（按优先级）

| 顺序 | 动作 | 理由 |
|---|---|---|
| 1 | **论文侧决定 SG-MPS 定位**（安全机制 or 给 $\phi$ 补 $q_{pot}$） | 方法学决策，需人工拍板；`05` §3 给出两种解释的实测依据。这是当前唯一挡住"论文=代码"的前置决策 |
| 2 | 论文同步（补 M-1~M-5、修 M-XREF-1/M-SG-1 等） | 依赖 1 的结论；差异清单见 `09` |
| 3 | 若要做物理表型：补一段经过核验的真实世界距离 | 唯一能让 `PHYSICAL_SCALE_NOT_AVAILABLE` 变成可报 cm 的输入 |
| 4 | 补终态面积门（0008 类失效） | 需要先决定是否改动方法定义；当前只记录为已知失效路径 |

> 实验侧已无待跑任务：RAP 210 帧、2×2 四档训练、四档 TSDF 网格全部完成。

> 注：`--semantic_gate_margin 0.05` **不建议**设为默认。实测它把 0.7504 提到 0.9435，但仍低于无门控的 0.9763，且它的作用只是"少说话"，并未修正 $\phi$ 的目标定义错误。设为默认会让默认配置**看起来**更好而问题依旧存在，属于用户明令禁止的"调参掩盖问题"。
