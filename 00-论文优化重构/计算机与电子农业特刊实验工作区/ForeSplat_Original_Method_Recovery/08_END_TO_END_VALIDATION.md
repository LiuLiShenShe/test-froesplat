# 08 — 端到端验证

生成时间：2026-10-25
状态：**端到端链路已贯通。** RAP 掩膜全链路、2×2 下游训练、TSDF 网格提取均已完成；
物理表型因缺少真实尺度而按证据门留空（`PHYSICAL_SCALE_NOT_AVAILABLE`），论文同步未做。

---

## 1. 端到端链路当前真实状态

```
RGB 帧
  │
  ├─ [1] FFT 帧质量过滤                      ✔ 独立阶段，blur_filter.py --fft_formula paper
  │
  ├─ [2] SAM3 多提示词逐实例候选              ✔ 真实推理，P1–P5，KQZ 210 帧全跑
  │
  ├─ [3] SG-MPS 语义门控选提示词              ✔ 执行（5 帧 dev + 210 帧 KQZ 均落盘 a_max/argmax_prompt）
  │
  ├─ [4] CDR/R1 实例级重排                    ✔ 执行（选择与 base 在该子集一致，见 05 §2）
  │
  ├─ [5] SB-FBPR 点提示细化                   ✔ 执行，5 正 + 8 负点真实送入 SAM3，0 框
  │
  ├─ [6] 残余/细结构修复                      ✔ 执行（210 帧全部执行，4–101 px/帧，均值 17.0）
  │
  ├─ [7] GCFC 几何一致性修正                  ✔ 执行，真实 COLMAP 观测 210/210，接受 205/210
  │
  └─ [8] 最终掩膜 210 帧                      ✅ 210/210 完成
       │
       ├─ [9] Foreground-Object 2DGS           ✅ 四档 COMPLETE（2132–2201 s）
       │    └─ [10] M2M3-GO                    ✅ 两档真实执行（19 次更新/档）
       │         └─ [11] TSDF mesh             ✅ 四档 mesh_post.ply（74–106 s/档）
       │              └─ [12] 表型几何          ⚠ 探索性（COLMAP 单位）；物理表型 BLOCKED（无尺度）
       └─ 论文同步                             ⏸ 未做（见 09）
```

---

## 2. 已完成的端到端证据

### 2.1 RAP-FSAM3 全链路（5 帧，KQZ，`runs/R6_RAP_FULL_real`）

| 验收项 | 证据 | 判定 |
|---|---|---|
| 真实 RGB 输入 | `input_image_sha256` 逐帧落盘 | ✔ |
| 真实 SAM3 推理 | 23/7/5/5/13 个候选，`raw_instance_P{1..5}/` | ✔ |
| 多提示词逐实例候选 | `候选元数据_*.json`（prompt/instance/box/sha256） | ✔ |
| SG-MPS 真实影响选择 | 0075 帧 P2→P1 切换，softmax 0.5396/0.4604；该帧带盆 GT F1 0.9227（相邻帧 0.981–0.987），纯植株 GT F1 0.9800 | ✔ |
| CDR 真实生效 | `cdr_policy=r1`，`cdr_scope` 记录候选集范围 | ✔ |
| SB-FBPR 真实点提示 | `SB-FBPR提示证据.csv`：5 正 + 8 负点、0 框、逐点坐标（本 5 帧 run 为 12 列，见 `04` §2.2） | ✔ |
| 几何修正 | `几何修正提示图/GCFC_*.png` ×5 + `几何修正提示.csv` | ✔ |
| 最终掩膜 | `最终掩膜/mask_<stem>.png`，`final_mask_sha256` 逐帧 | ✔ |
| 逐帧报告 39 列 | `逐帧报告.csv`，含用户要求的全部 23 个字段 | ✔ |
| 四口径 GT 计分 | `metrics/kqz5_R6_vs_splitGT.json`：带盆 0.9832/0.9872/0.9810/0.9227/0.9874，纯植株 0.8215–0.9911，泄漏率 ≤0.0002 | ✔ |

### 2.2 全场景规模验证（KQZ 210 帧，`runs/KongQueZhuYu_RAP_FULL_210`）

| 项 | 状态 |
|---|---|
| 候选生成 | ✔ 210/210（`选择后掩膜` 210 个） |
| SG-MPS 落盘 | ✔ 每帧 `argmax_prompt` 与 `a_max` 写入运行日志与 `逐帧报告.csv` |
| 门控选中分布 | P2 **156**（74.3%）、P1 29、P3 15、P5 5、P4 5 |
| CDR | ✔ 210/210，`cdr_scope=multi_prompt` |
| SB-FBPR | ✔ 210/210 执行，209 接受（0008 `low_iou_guard`） |
| 残差修复 | ✔ 210/210，4–101 px/帧，均值 17.0 |
| GCFC | ✔ 210/210 有真实观测；接受 205，5 帧 `support_gate`（0008/0070/0071/0072/0073）；$G_i$ 均值 0.8209 |
| 最终掩膜 | ✅ **210/210** |
| 逐帧报告 | 40 列 × 210 行 |

### 2.3 下游 2DGS 2×2（`runs/downstream/`，`metrics/downstream_2x2.csv`）

| tag | 掩膜来源 | M2M3-GO | 状态 | PSNR_fg | SSIM_fg | LPIPS_fg | outside | leakage |
|---|---|---|---|---|---|---|---|---|
| `C_R1_baseline` | 冻结 C_R1 | off | ✅ COMPLETE | **25.019** | **0.8384** | **0.1183** | **0.0292** | **0.0128** |
| `RAP_FULL` | 完整 RAP 210 | off | ✅ COMPLETE | 24.734 | 0.8244 | 0.1280 | 0.0657 | 0.1316 |
| `RAP_FULL_M2M3GO` | 完整 RAP 210 | full | ✅ COMPLETE | 24.610 | 0.8135 | 0.1307 | 0.0645 | 0.1319 |
| `C_R1_M2M3GO` | 冻结 C_R1 | full | ✅ COMPLETE | **25.029** | **0.8387** | **0.1182** | 0.0292 | 0.0128 |

- 四档均为 30k 迭代、`--resolution 2`、同一损失/剪枝配方，27 个 test 视角；
- 四档 `mask_provenance.json` 均为 **183 cameras / 0 missing**，183 条 SHA256 与 `mask_provenance_expected.json` **逐条一致**；
- 训练耗时 2132–2201 s，四档一致。

| 主效应 | ΔPSNR_fg | ΔSSIM | 结论 |
|---|---|---|---|
| 掩膜来源 C_R1→RAP（M2M3 关） | **−0.285** | −0.0140 | RAP 掩膜更差，outside 2.25×、leakage 10× |
| 掩膜来源 C_R1→RAP（M2M3 开） | −0.420 | −0.0252 | 同上，幅度更大 |
| M2M3-GO（C_R1 掩膜） | +0.011 | +0.0003 | 可忽略 |
| M2M3-GO（RAP 掩膜） | −0.125 | −0.0109 | 负向 |

⚠️ **这是本轮最重要的负面结果**：完整 RAP 掩膜在 3D 端比冻结的单提示词基线更差，M2M3-GO 既救不回来、在好掩膜上也只 +0.011 dB。归因见 `06` §6.1：完整 RAP 在 54/210 帧（26%）选用了非 P2 提示词，其中 22 帧与冻结基线的掩膜 IoU < 0.75，0008 帧前景占比塌到 0.86%（近乎空掩膜）。与 `05` 的 dev 集消融结论一致。

### 2.4 TSDF 网格 + 探索性表型几何（`metrics/mesh_phenotype.csv`）

四档模型逐一执行 `render.py --skip_train --skip_test`（standard TSDF，`voxel_size 0.02 / depth_trunc 6.0 / sdf_trunc 0.08 / mesh_res 256`），
产物归档在 `runs/mesh/<tag>/`，测量最大连通域：

| tag | post 顶点 | 连通域 | 最大域占比 | 高 (u) | 最大冠幅 (u) | 网格耗时 |
|---|---|---|---|---|---|---|
| `C_R1_baseline` | 170 040 | 14 | 0.9443 | 3.5420 | 3.1256 | 106.1 s |
| `C_R1_M2M3GO` | 168 920 | **5** | **0.9931** | 3.5411 | 3.1218 | 73.9 s |
| `RAP_FULL` | 175 350 | 12 | 0.9171 | 3.5598 | 3.1418 | 81.5 s |
| `RAP_FULL_M2M3GO` | 181 622 | 8 | 0.9284 | 3.5611 | 3.1393 | 78.4 s |

- **单位是 COLMAP 场景单位，不是 cm**；`physical_status = PHYSICAL_SCALE_NOT_AVAILABLE`（本场景没有经过核验的真实距离）；
- 网格是**含花盆的整体对象**，因此"高"= 盆栽整体高度，不等于论文的 `plant_height`；
- 叶片级性状 `LEAF_ID_NOT_AVAILABLE`（无器官身份与地标）；
- 图：`figures/fig6_mesh_phenotype_geometry.png`。

两条可报告的事实：

1. **粗几何对掩膜来源不敏感**（四档高度差异 ≤0.54%），说明下游 −0.285 dB 的下降来自局部泄漏/纹理，不是整体尺度漂移；
2. **M2M3-GO 在 C_R1 掩膜上把连通域 14→5、最大域占比 0.9443→0.9931、碎块 12→4，顶点仅 −0.7%** ——
   这是本次对 M2M3-GO 唯一的**正面**证据，但它是**几何结构**指标，不是重建精度指标（PSNR_fg 只 +0.011 dB）。
   在 RAP 掩膜上同一机制则使顶点 +3.6% 且 PSNR 下降。

---

## 3. 尚未完成（不假装完成）

| 编号 | 项 | 阻塞原因 |
|---|---|---|
| E1 | 完整 RAP 210 帧最终掩膜 | ✅ 210/210 |
| E2 | RAP_FULL 2DGS 训练 | ✅ COMPLETE（2152.9 s，provenance 183/183） |
| E3 | RAP_FULL_M2M3GO 2DGS 训练 | ✅ COMPLETE（2131.8 s，19 次 M2M3-GO 更新） |
| E4 | 2×2 回归（两套掩膜 × M2M3-GO on/off） | ✅ 四档全部 COMPLETE，见 §2.3 |
| E5a | TSDF 网格提取 | ✅ 四档 `runs/mesh/<tag>/mesh_post.ply`，见 §2.4 |
| E5b | 物理表型（cm） | ⛔ 阻塞：无经过核验的真实世界距离 → `PHYSICAL_SCALE_NOT_AVAILABLE`。这是证据门，不是遗漏 |
| E6 | 论文公式/表格/图同步 | ⏸ 独立，未做（差异清单见 `09`；SG-MPS 定位待用户拍板） |
| E7 | 完整消融指标表（Boundary-F1/HD95/泄漏/oracle 选择率） | ✅ 已完成，见 `05` §2 与 `metrics/dev_ablation_summary.csv` |

---

## 4. 已知的、必须写进论文的负面结果

以下全部来自真实运行数据（`metrics/dev_ablation_*.csv`、`metrics/kqz5_R6_vs_splitGT.json`），不是推测：

1. **SG-MPS 在开发集上显著有害**：带盆 GT 的 16 帧均值 0.9763（无门控 R3n）→ 0.7504（R3 门控），CaoMei1 单场景 0.9833→0.5943。oracle 在 16/16 帧都选 P2，门控只选对 2/16。
2. **加承诺裕度只能缓解不能消除**：R3g 0.9435，仍低于无门控 0.9763；DouBanLv1 0025/0050 两帧门控差值 0.0567/0.0540 刚好越过 0.05 阈值，"自信地"选错。
3. **换 GT 口径结论翻转**：同一批掩膜在纯植株口径下 R6 = 0.9575 远高于 R0 = 0.7392 —— SG-MPS 实际在优化"纯植株"，与论文声明的 potted-plant 目标冲突。这是方法定义问题，必须由论文二选一。
4. **CDR/R1 在 dev 子集上零增量**（R1 与 R0 逐帧完全一致；R0 已 100% 选中 oracle 实例）。
5. **残差修复在 dev 子集上零增量**（R5 与 R4p 的最终掩膜逐字节一致）。
6. **GCFC 触发率高但增益极小**：dev 集 R6 中 12/16 帧被接受，F1 增量 ≤0.0002；KQZ 5 帧 4/5 被接受，新增 151/227/456/0/223 px。
7. **SB-FBPR 点提示在 XianKeLai1 上负向**（0.9827 框 → 0.9641 点），且点提示整体只比框提示低 0.0043。
8. **仅扩展候选集到 5 提示词就会掉分**（ChangShouHua2 0.9801→0.9594，16 帧均值 −0.0065）。
9. **完整 RAP 掩膜进 3D 训练后下游指标变差**：PSNR_fg 25.019→24.734，outside ratio 0.0292→0.0657，leakage energy 0.0128→0.1316（10×）。逐帧归因见 `06` §6.1 与 `metrics/mask_geometry_rap_vs_cr1.json`。
10. **0008 帧的近乎空掩膜进入了训练**：前景占比 0.0086（基线 0.2266），而 `--area_min_ratio 0.01` 的面积门、SB-FBPR 的 `low_iou_guard`、GCFC 的 `support_gate` **都没能拦住它**。这是一个真实的失效路径，应作为论文的局限性声明，或在代码里补终态面积门。
11. **M2M3-GO 在 3D 端无精度增益**：C_R1 掩膜上 ΔPSNR_fg = +0.011 dB（噪声量级），RAP 掩膜上 −0.125 dB。它唯一的正面证据是几何结构清理：连通域 14→5、最大域占比 0.9443→0.9931、碎块 12→4（`metrics/mesh_phenotype.csv`）。
12. **物理表型无法交付**：KongQueZhuYu 没有经过核验的真实世界距离，四档全部 `PHYSICAL_SCALE_NOT_AVAILABLE`；论文 §2.8 若声称表型精度，必须补尺度来源。
13. **粗几何对掩膜来源不敏感**：四档高度 3.541–3.561 u、最大冠幅 3.122–3.142 u（差异 ≤0.54%），说明下游 PSNR 下降来自局部泄漏而非尺度漂移。

---

## 5. 复现命令

```bash
# R6 全链路（5 帧）
python 生成RAP-FSAM3掩膜.py --input_dir .../selected_frames \
  --output_dir runs/R6_RAP_FULL_real \
  --use_semantic_gate --semantic_gate_mode gated_softmax --semantic_gate_temperature 0.25 \
  --ranking_policy r1 --spnp_prompt_form points --spnp_negative_region_mode manuscript \
  --use_residual_repair --thin_repair_mode adjacent \
  --corrective_geometry_backend colmap_tracks_gaussian --colmap_dir .../KongQueZhuYu \
  --strict_colmap_observations --device cuda

# 全场景完整 RAP（210 帧）
python 生成RAP-FSAM3掩膜.py --input_dir .../final_locked/KongQueZhuYu/images \
  --output_dir runs/KongQueZhuYu_RAP_FULL_210 ... （同上开关）

# 跨场景消融（16 帧 / 4 场景，与冻结评测集不同）
python scripts/run_dev_ablation.py

# 下游 2DGS
python scripts/run_downstream_2dgs.py --scene KongQueZhuYu \
  --mask-run-dir <RAP 输出> --tag RAP_FULL --iterations 30000 --resolution 2

# 2×2 汇总表（高斯数直接从 PLY 头部读出）
python scripts/build_downstream_2x2.py

# TSDF 网格 + 探索性表型几何（标准 TSDF 协议）
/data/fj/F2DMAS/2d-gaussian-splatting-m2m3-floor40/venv/bin/python \
  scripts/run_mesh_phenotype.py          # 跑 render.py 并归档；--measure-only 复用；--tags 选档

# 交付物清单
python scripts/build_deliverables.py
python scripts/build_figures.py
```

---

## 6. 完成判据（全部满足才可声明"完成"）

- [x] E7 完整消融指标表生成
- [x] E1–E4 全部 `COMPLETE`
- [x] E5a 四档 `mesh_post.ply` + `mesh_metrics.json` 归档于 `runs/mesh/<tag>/`
- [x] 每档 `mask_provenance.json` 与对应掩膜来源哈希一致（183/183 SHA256）
- [ ] E5b 物理表型（cm）—— **阻塞，非遗漏**：无经过核验的真实尺度
- [ ] E6 论文同步完成且改动可追溯（差异清单见 `09`；SG-MPS 定位需用户拍板）
- [x] `10_FINAL_ACCEPTANCE.md` 逐条给出 L1–L6 判定

当前判定：**RAP-FSAM3 核心方法已在真实数据上端到端运行，下游闭环已贯通到 mesh（L1–L5 达成）；物理表型与论文同步未完成（L6 部分 BLOCKED）。**
