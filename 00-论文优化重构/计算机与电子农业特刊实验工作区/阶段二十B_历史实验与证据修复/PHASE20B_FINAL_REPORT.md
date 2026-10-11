# ForeSplat Phase 20B 最终报告

## 1. 审计范围与边界

- 审计基线：`c119498c`；当前工作仅新增本阶段目录，不修改 RAP-FSAM3、`ranking_policy.py`、2DGS、M2M3-GO、GT、论文 `.tex` 或历史实验文件。
- 已读取 Phase 20A 要求的八份材料：`PHASE20A_FINAL_REPORT.md`、`METHOD_SELECTION_DECISION.md`、`P0_BLOCKERS_AND_NEXT_ACTIONS.md`、`R1_FORMULA_VERIFICATION.md`、`RECONSTRUCTION_EVIDENCE_REUSE_MATRIX.csv`、`PROMPT_AND_RUN_PROVENANCE_MATRIX.csv`、`FINAL_RAP_FSAM3_SPEC_DRAFT.md`、`FOLLOWUP_EXPERIMENT_PLAN.md`。
- 仅使用仓库真实 GT、预测、掩膜、模型目录、日志、CSV、JSON、XLSX 和源码；没有 synthetic、mock、placeholder 或虚构实验数字。
- 未运行 SAM3 GPU 推理、2DGS GPU 训练、COLMAP、Mesh 重建或 Phase 20C；V0–V3 只生成配置、输入清单和预检结果。
- 外部 Bocha 搜索按要求尝试但返回 HTTP 403（余额/套餐配额不足），因此本报告不把网络搜索当作证据；结论只来自本地仓库和本阶段脚本。

## 2. 最终判定

```text
PHASE20B STATUS:
PARTIAL

SEGMENTATION TABLES:
PARTIAL

OLD 3D EVIDENCE:
PARTIAL

SCALE RECOVERY:
USER_INPUT_REQUIRED

M2M3 CODE ALIGNMENT:
MISMATCH

R1 FORMULA:
PASS

V0-V3 READY:
YES

NEXT PHASE:
PHASE20C
```

`PARTIAL` 是有意保守的结论：本阶段已经恢复了可重算的分割和 R1 证据，但原 Table 7 的 20 序列共同来源、完整物理尺度链路以及旧模型到 P6+R1 的整套输入等价性仍未闭合。

## 3. 直接回答核心问题

### 3.1 原有多少结果可以直接继承

可以直接继承为“证据”的内容：

1. 统一 `potted_clean` 口径后的 Easy20 分割重算结果；它们是历史基线，不是新 P6+R1 结果。
2. `R1_FORMULA_RECHECK.csv` 中 182 条归档候选的公式复核结果，全部 `PASS`；这是离线公式证据，不是生产推理结果。
3. 旧 2DGS 模型、Mesh、M2M3 pilot 的原始文件，可作为带原输入版本的 legacy/pilot 证据。
4. 表型工作簿中已经记录的人工/虚拟统计值，可作为 `LEGACY_RECORDED_STATISTICS`。

不能直接继承为新方法结果的内容：

1. 旧 2DGS 不能直接改名为 P6+R1 训练结果；19 个旧掩膜样本中没有一条有效版本-帧比较为 `EXACT_MATCH`。
2. 原论文 Table 7 的四行“20 序列平均”不能标记为 `VERIFIED`。
3. 原 Table 5 的无法绑定来源的 `Boundary-F1=0.4543`、`Leakage=0.001935` 不能沿用。
4. 盆径、尺度因子和虚拟测量过程不能由现有工作簿或 Mesh 日志补造。

因此，**目前没有一套旧三维结果可以在证据意义上直接继承为 P6+R1 联合结果**；最小路径是先执行 Phase 20C 的 V0–V3，再按实际掩膜差异做定向 2DGS/表面重建实验，而不是无依据地全量重跑 19 或 20 个序列。

## 4. Table 3/4/5：GT、指标和重算结果

### 4.1 GT 口径

本阶段只使用：

```text
potted_clean = potted_plant AND NOT blue_cube
```

Easy20 为 4 个序列、每序列 5 帧、共 20 帧。`GT_DEFINITION_AUDIT.csv` 共 31 条可用 GT 记录，31/31 `VERIFIED`；Phase 12 clean、`gt_masks_split_v2` 的内存逐像素逻辑与 `potted_plant ∧ ¬blue_cube` 一致。没有写回或修改任何 GT 文件。Challenge10 没有可用于本阶段盆区拆分的独立 pot 标注，因此不混入 Table 3–5。

### 4.2 指标定义和主聚合

- `F1 = 2TP / (2TP + FP + FN)`。
- `IoU_fg = TP / (TP + FP + FN)`。
- `mIoU = (IoU_fg + IoU_bg) / 2`。
- `HD95` 使用旧实现的双向边界距离、OpenCV `DIST_L2`、3×3 近似距离变换，单位为原图像素。
- `Boundary-F1` 沿用旧实现：3×3 方形腐蚀取边界，7×7 椭圆核膨胀，容差半径 3 像素；空边界约定也保持旧实现。
- `Leakage energy = FP / (H×W)`，是旧分割脚本的像素泄漏比例，不是三维 RGB 辐射能量。
- 主列使用 `sample_equal_mean`：先对每个序列有效帧取算术平均，再对序列均值等权平均。per-frame mean、per-sample mean、median、population SD 分列保存，不能混用。

### 4.3 Table 3 重算

| 方法 | F1 | IoU_fg | mIoU | HD95(px) | Boundary-F1 | Leakage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CLIPSeg_potted | 0.93029246 | 0.87018212 | 0.91732377 | 159.61876545 | 0.09126244 | 0.01829091 |
| GroundedSAM1_potted | 0.96219722 | 0.93161435 | 0.95472870 | 87.89367442 | 0.42074027 | 0.00437349 |
| RAP-FSAM3-v2 | 0.98373096 | 0.96823591 | 0.97963466 | 23.72945986 | 0.48520898 | 0.00173863 |
| SAM3_P2 | 0.98283445 | 0.96634135 | 0.97822086 | 26.82004576 | 0.42235026 | 0.00227178 |

### 4.4 Table 4 重算

| 提示词 | F1 | IoU_fg | mIoU | HD95(px) | Boundary-F1 | Leakage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 | 0.76188164 | 0.62969101 | 0.77696109 | 524.68527765 | 0.34601976 | 0.00095756 |
| P2 | 0.98283445 | 0.96634135 | 0.97822086 | 26.82004576 | 0.42235026 | 0.00227178 |
| P3 | 0.47172450 | 0.39749664 | 0.64280275 | 2065.68354380 | 0.21124224 | 0.00144877 |
| P4 | 0.09024434 | 0.06455275 | 0.43526827 | 3848.14087383 | 0.04706555 | 0.00005827 |
| P5 | 0.23287355 | 0.19651367 | 0.52133300 | 3240.94165047 | 0.10612231 | 0.00022008 |

### 4.5 Table 5 重算

| 配置 | F1 | IoU_fg | mIoU | HD95(px) | Boundary-F1 | Leakage | 证据类型 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Fixed_prompt_P2 | 0.98283445 | 0.96634135 | 0.97822086 | 26.82004576 | 0.42235026 | 0.00227178 | PIPELINE_OUTPUT |
| Geometry_only_selector | 0.78581163 | 0.77220740 | 0.86814851 | 905.17303878 | 0.33307109 | 0.00203470 | OFFLINE_REPLAY |
| RAP_FSAM3_full | 0.98373096 | 0.96823591 | 0.97963466 | 23.72945986 | 0.48520898 | 0.00173863 | PIPELINE_OUTPUT |
| Semantic_gated_selected | 0.98283445 | 0.96634135 | 0.97822086 | 26.82004576 | 0.42235026 | 0.00227178 | PIPELINE_OUTPUT |
| Without_GCFC | 0.98419630 | 0.96913713 | 0.98015396 | 23.14922662 | 0.48918913 | 0.00155845 | INTERMEDIATE_OUTPUT_NOT_INDEPENDENT_RUN |

`Geometry_only_selector` 是归档候选的离线重放，不能写成生产运行；`Without_GCFC` 是同一流水线的残差修复中间掩膜，不是独立消融。表格数字和所有帧级结果分别见 `tables/TABLE3_RECALCULATED.csv`、`tables/TABLE4_RECALCULATED.csv`、`tables/TABLE5_RECALCULATED.csv` 与 `tables/SEGMENTATION_PER_FRAME_METRICS.csv`。

### 4.6 哪些旧数字不能沿用

- 早期 union GT 下的 Table 3/4/5 数字不能与本阶段 `potted_clean` 结果混列。
- 原 Table 5 的 `Boundary-F1=0.4543` 和 `Leakage=0.001935` 没有可追溯预测绑定。
- 原 Table 3 的 `F1=0.9706`、`mIoU=0.9641` 混合了 pooled confusion 和逐帧统计，不能作为本阶段统一统计列。
- 本阶段重算数字只能作为“统一 GT 的历史基线”；它们不是 P6+R1 新算法实验。

## 5. 旧 2DGS 前景掩膜身份

### 5.1 覆盖和像素比较

- 旧 SAM 掩膜目录：19 个样本。
- 有身份验证的 P6 交叉样本：6/19（`CaoMei1`、`CaoMei2`、`ChangShouHua2`、`DouBanLv1`、`KongQueZhuYu`、`XianKeLai1`）。
- 有效版本-帧比较：43 个；唯一旧帧：23 个；`EXACT_MATCH`：0 个。
- 有效比较分类：`SIMILAR_BUT_DIFFERENT=30`、`DIFFERENT=13`；全部比较记录还包含 `NOT_VERIFIED=37` 和 `UNMATCHED=24`。
- 旧 2DGS 模型继承决策：`REUSE_AS_LEGACY_BASELINE=6`、`TARGETED_RERUN_REQUIRED=4`、`NOT_VERIFIED=9`、`REUSE_DIRECTLY=0`。

比较依据是真实 RGB 文件名、图像身份、尺寸和坐标系，不按目录排序猜帧。原始掩膜对比不做 resize；COLMAP undistorted 尺寸与 raw RGB 不同（例如 `CaoMei1` 为 2160×3840 对 2120×3800），因此局部 raw-mask 比较不能证明旧训练输入与新 COLMAP 训练域一致。IoU 接近 1 也不等于输入完全相同。

### 5.2 继承判断

没有证据支持任何旧三维模型 `REUSE_DIRECTLY` 作为 P6+R1 结果。旧模型可作为原输入版本下的 legacy baseline；只有在 Phase 20C 生成最终 P6+R1 掩膜并完成模型训练输入哈希闭合后，才可决定 4 个 `TARGETED_RERUN_REQUIRED` 样本是否重跑。不得据此要求全量 19/20 序列重跑。

## 6. 物理尺度和表型来源

- `植株数据.xlsx`：1 个 sheet、21 行植株、17 列，包含人工/虚拟株高、冠幅和三片叶的记录。
- 工作簿没有逐样本盆径、半径、scale marker 或 scale factor 字段。
- `measure_traits_entry.py` 只要求外部 `--scale-marker`，生成外部测量 manifest；完整交互/半自动工具没有随轻量发布包提供。
- 12 个 `06-MESH-new/*_mesh*.log` 没有发现 pot/scale/trait 的传递字段。
- `生成实验一五当前结果.py` 读取工作簿并计算统计，没有 Mesh 读取、坐标尺度变换、landmark 或几何提取流程。
- 因此 Table 8 的人工/虚拟数字可以作为 `LEGACY_RECORDED_STATISTICS`，不能标为 `REPRODUCIBLE_3D_EXTRACTION`。

真正需要用户补充的资料已集中在 `reports/USER_MEASUREMENT_INPUT_REQUIRED.md`：21 个样本的真实盆径及单位/时间、每个虚拟值对应的 Mesh 版本和坐标系、尺度恢复系数、测量端点/landmark 记录，以及外部测量工具的版本和导出文件。

## 7. Table 7 三维证据

### 7.1 论文声称与本地证据

| 方法 | 论文 PSNR | 论文 SSIM | 论文 LPIPS | 论文训练时间(s) | 论文 Mesh 时间(s) | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| COLMAP | 13.63 | 0.8745 | 0.1072 | 599.5 | 78.0 | UNVERIFIED |
| 3DGS + VFM | 30.17 | 0.9587 | 0.0386 | 5413.5 | 642.0 | UNVERIFIED |
| Standard 2DGS | 29.58 | 0.9574 | 0.0487 | 12913.7 | 157.9 | UNVERIFIED |
| ForeSplat | 31.09 | 0.9711 | 0.0365 | 5044.5 | 55.0 | UNVERIFIED |

本地盘点找到 11 个历史 2DGS `cfg_args` 模型目录、12 个 Mesh 日志和 2 个 M2M3 full-summary CSV。M2M3 两个 CSV 各覆盖 3 个场景、3 个方法，共 18 行 pilot 记录；这些行有真实 PSNR/SSIM/LPIPS 和 train/render/metrics elapsed，可保留为 M2M3 pilot，但不支持原 Table 7 的 20 序列平均。

### 7.2 不能核验的原因

没有同时找到四种方法共享的 20 序列清单、训练/验证/测试视图划分、评价 ROI、失败样本处理、计时起止边界和 Mesh 后处理边界。因此原 Table 7 的样本数、PSNR/SSIM/LPIPS 以及训练/Mesh 时间都不能标记为 `VERIFIED`。需要补充完整渲染、GT、split 和日志后，才能进行 CPU 复评或统一计时；本阶段不修改原 Table 7。

## 8. M2M3-GO 公式与代码

### 8.1 实际执行

- `Mask-to-model support` 只执行多视图投影后对 `gt_alpha_mask` 最近采样的 `hits/visible` 一致性分数；pilot 使用最多 12 个视图、阈值 0.45。
- 实际 prune score 是 opacity、brightness、mask、scale、screen-radius 的归一化正 penalty 加权求和；pilot 权重为 1、1、3、0.5、0.5。
- `foreground` 区域门控和 `maybe_prune_gaussians()` 调用链真实执行。
- 代码仍使用标准 `gaussian_model.densify_and_prune()`；没有 M2M3 独立 densify 控制器。
- M2M3 仅进入 prune candidate path，并按 `m2m3_max_remove_ratio=0.03` 和 pruning interval/start 做 top-k 删除。

### 8.2 未执行或与论文不一致

- 没有逐 Gaussian 的 model-to-mask deficit `H_i=M_i(1-A_i)`。
- 没有 boundary-distance support 或 `sigma_b`。
- 没有论文草稿中的 sigmoid utility、残差项和显式 view-count utility。
- 没有显式 keep threshold 或 M2M3 densify decision。
- Floor40 不是 local block 保护，而是全局 `floor_count = round(reference_count × 0.4)`；pilot 的 reference 是 `max_seen`，起始迭代为 18000。

因此代码对齐结论为 `MISMATCH`。后续只能在两个选项中选择：方案 A 按真实实现重写论文；方案 B 补齐公式、局部 Floor40 和单元测试后重新实验。本阶段不替用户选择，也没有修改 M2M3 代码。

## 9. R1 最终离线核查

`tables/R1_FORMULA_RECHECK.csv` 共 182 条，逐条 `PASS`。重算使用原始实例掩膜重新得到 `risk_penalty`，没有用归档 `total_score` 反推惩罚；冻结分母 5.5，权重为 `1,1,1,1,1,0.5`，semantic gate 关闭，`tie_break=0`。

最大 `total_score` 误差为 `6.81443810269e-08`，在归档 SAM 分数六位小数容差内；唯一明显的 0.25 差异是 `BaiZhang_0042_09` 的 `base_total=0.609126836742` 与扣除 `risk_penalty=0.25` 后的 `total_score=0.359126836742` 被错误比较。没有修改参数、归档 CSV 或 `ranking_policy.py`。

## 10. Phase 20C V0–V3 准备状态

四组固定为：

| 组别 | Ranking | Refinement |
| --- | --- | --- |
| V0 | base | OFF |
| V1 | r1 | OFF |
| V2 | base | ON |
| V3 | r1 | ON |

其他参数统一为 `P6`、`per_instance`、`single`、无 prompt ensemble、相同 score weights、同一候选缓存。Refinement ON 显式使用源码支持的 `SPNP`、`residual repair`、`corrective geometry` 以及 `sam3_if_supported`、`adjacent`、`colmap_tracks` 后端；没有凭空增加 CLI 参数。

预检结果：20 个配置行（4 组×5 个样本）全部 `READY_FOR_PHASE20C`；可用输入帧为 `CaoMei1=5`、`ChangShouHua2=5`、`DouBanLv1=5`、`KongQueZhuYu=5`、`XianKeLai1=1`，合计 21 帧。候选缓存逐帧存在，COLMAP sparse、SAM3 repo 和 checkpoint 均通过逐样本存在性检查。`V2/V3` 一旦真正执行会加载 SAM3 并需要 GPU；本阶段没有启动任何命令。

配置、命令、输入清单和预检分别见：

- `reports/configs/V0_*.json` 至 `V3_*.json`
- `reports/V0_V3_COMMANDS.md`
- `tables/V0_V3_CONFIG_MATRIX.csv`
- `tables/V0_V3_INPUT_MANIFEST.csv`
- `reports/V0_V3_PREFLIGHT_REPORT.md`
- `reports/PHASE20C_EXECUTION_PLAN.md`

## 11. 旧实验分类

### 可保留

- 原始 GT、原始预测、掩膜、日志和模型文件本身。
- 本阶段统一 `potted_clean` 后的 Table 3/4/5 重算值，作为历史基线。
- R1 182 条离线公式复核。
- 旧 2DGS/Mesh 作为带原输入版本的 legacy baseline。
- M2M3 3 场景 pilot 作为实现证据。
- `植株数据.xlsx` 的人工/虚拟统计作为 legacy recorded statistics。

### 需要重算

- 所有混用 union GT、pooled confusion 和逐帧均值的 Table 3/4/5 汇总。
- 若要与 Table 7 做公平比较，需在统一 ROI、split 和失败处理下 CPU 重算已有渲染指标。
- 若要报告表型复现误差，需在补齐尺度和 Mesh provenance 后重新生成可追溯统计。

### 需要重跑

- Phase 20C 的 V0–V3 掩膜/细化实验；本阶段只准备，没有执行。
- 只有在最终 P6+R1 掩膜与旧训练输入差异被确认后，才对受影响样本做定向 2DGS/Mesh 重跑。
- 若坚持原 Table 7 数字，需在统一机器和计时边界下重测训练、渲染和 Mesh 时间。
- 若选择 M2M3 方案 B，需先实现、单测并重新跑 smoke/pilot；不能沿用现有 pilot 冒充新公式结果。

### 无法核验

- 原 Table 7 的 20 序列平均和四方法 wall-clock。
- 原 Table 5 无来源绑定的 BF1/Leakage 数字。
- 旧 2DGS 是否全部以同一 raw/undistorted mask 域训练。
- 每个虚拟表型值对应的 Mesh、尺度系数和交互测量记录。
- D3/D6/D12 的可靠确认帧。

## 12. 14 个验收问题逐项回答

1. **Table 3/4/5 具体是多少？** 已在第 4 节给出；完整数字和逐帧明细在三张重算表及 `SEGMENTATION_PER_FRAME_METRICS.csv`。
2. **哪些旧论文数字不能沿用？** 早期 union-GT 汇总、Table 3 pooled/逐帧混合值、Table 5 的 BF1 `0.4543`、Leakage `0.001935`、以及 Table 7 四行 20 序列平均和时间。
3. **19 个旧 2DGS 样本中多少有可匹配 P6？** 6/19 有身份验证的交叉样本；对应 23 个唯一帧和 43 个有效版本-帧比较。
4. **新旧掩膜多少完全一致？** 0/43 个有效比较为 `EXACT_MATCH`。
5. **是否有旧 3D 结果可直接继承？** 没有 P6+R1 直接继承证据；旧模型只能作为 legacy baseline。
6. **Table 7 的 20 序列平均是否有依据？** 没有闭合依据；本地仅有 11 个历史模型目录、12 个 Mesh 日志和 3 场景 M2M3 pilot。
7. **原训练和 Mesh 时间能否核验？** M2M3 pilot 的局部日志时间可核验；原 Table 7 四方法时间不能核验，必须统一重测。
8. **盆径和尺度恢复缺什么？** 21 个样本的真实盆径/单位、scale marker 或系数、Mesh 版本、坐标系、端点/landmark 和外部测量工具 provenance。
9. **M2M3 哪些公式真实执行？** 多视图 mask consistency、加权 penalty pruning、foreground gate 和调用链执行；model-to-mask、boundary support、sigmoid utility、独立 densify 未执行。
10. **Floor40 的真实机制？** 全局 `round(max_seen × 0.4)` 容量下限，不是 local block 保护。
11. **R1 182 条是否通过？** 是，`182/182 PASS`，无参数和历史数据修改。
12. **V0–V3 是否准备好？** 是，20/20 配置行资产预检 `READY_FOR_PHASE20C`，但没有运行结果。
13. **哪些任务被用户数据或实验产物阻断？** Table 8 尺度/虚拟测量 provenance、Table 7 20 序列共同产物、完整 P6+R1 最终掩膜和 D3/D6/D12 确认帧。
14. **下一阶段最小 GPU 实验是什么？** 先按同一候选缓存执行 V0/V1，再执行需要 SAM3 的 V2/V3；比较 selected-mask、R1 score、refinement 接受率和哈希，随后只对差异明确且有旧模型的样本做定向 2DGS/Mesh 重跑，不默认全量。

## 13. 复现和验收

CPU 审计脚本位于 `scripts/`，输出表和报告均在本阶段目录。测试覆盖 GT 数量、Table 3/4/5 统一口径、R1 182 条通过、掩膜身份保守分类、Table 7 未验证状态、M2M3 pilot 行数、V0–V3 配置一致性和生产文件未修改。执行：

```bash
STAGE='00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十B_历史实验与证据修复'
PYTHONDONTWRITEBYTECODE=1 python -m pytest "$STAGE/tests" -q
git diff --check
git diff --stat
git status --short
```

本报告结束于 Phase 20B；不会自动进入 Phase 20C。
