# 重建证据复用矩阵 / RECONSTRUCTION_EVIDENCE_REUSE_MATRIX

**阶段：** Phase 20A
**分类规则：** `REUSE_DIRECTLY` / `REUSE_AS_LEGACY_BASELINE` / `TARGETED_RERUN_REQUIRED` / `UNVERIFIED`
**原则：** 不得无依据地要求全量重跑 20 序列。

---

## 0. 结论摘要

| 论文要素 | 实际数据源（已定位） | 与最终方法输入一致性 | 分类 |
|---|---|---|---|
| Table 3（VFM 分割对比） | `06-实验一四样本代表集/…/metrics/`（4 样本 × 5 帧 = **20 帧**，非"全部 20 序列"） | 基于 P2/P1-P5 候选掩膜；RAP-FSAM3 行基于 **score_select 多提示词 + SPNP/GCFC** 运行 | `REUSE_AS_LEGACY_BASELINE`（**口径需重跑或降级**） |
| Table 4（提示词鲁棒性） | `04-结果表格模板/实验二_提示词鲁棒性四样本汇总表.csv`（20 帧） | 数字与归档**逐位一致**；但 **P2 文本被论文改写**（真实 = "entire plant excluding pot"，论文写 "entire potted plant"） | `REUSE_AS_LEGACY_BASELINE`（数字可继承，**标签必须改正**） |
| Table 5（RAP 模块消融） | `实验三_RAP-FSAM3模块消融多样本主表.csv`（20 帧） | **混合聚合口径 + 至少一个数字无出处**（见 §3） | `TARGETED_RERUN_REQUIRED` |
| Table 6（先验注入消融） | `实验四_先验注入位置消融结果表.csv` | 单样本（KongQueZhuYu）代表形态 | 见 §4 |
| Table 7（重建对比） | `数据管理/06-实验输出/KongQueZhuYu/*/foreground_object_results.json` | 仅 KongQueZhuYu 单样本 | 见 §5 |
| Table 8（表型测量） | `实验五_人工虚拟表型汇总.csv` | 待 2DGS 代理返回后更新 | 见 §6 |

---

## 1. 关键背景：论文声称的"评测口径" vs 归档事实

**论文多处声称：**
> "manually annotated evaluation frames sampled from **all 20 RGB sequences**"
> （Table 1 协议表、Table 3、Table 4、Table 5 图注、正文多次）

**归档事实：**
- 分割评测实际范围 = **`FourSample_GT20`**：KongQueZhuYu、DouBanLv1、ChangShouHua2、CaoMei1，
  每样本 5 帧，共 **20 帧**（4 个序列）。
- 来源：`实验一_视觉基础模型横向对比分割表.csv` 的"样本范围"列逐行标注
  "四样本代表集：KongQueZhuYu、DouBanLv1、ChangShouHua2、CaoMei1；每样本5帧，共20帧GT"。

**判定：论文"all 20 RGB sequences"的表述与归档评测集不一致。
该 20 帧 = Phase 12 Easy21 的 21 帧的一部分（CaoMei1 0000/0025/0050/0075/0100 等），
但**不是** Phase 16 的 GT10 困难集，也**不是**覆盖全部 20 序列的抽样。**

---

## 2. 逐表溯源

### 2.1 Table 3（VFM 对比）— 数字与归档一致，但范围与叙事不符

| 论文行 | 归档来源 | F1（论文/归档） | 一致性 |
|---|---|---|---|
| CLIPSeg 0.9221 | 实验一表 CLIPSeg 行 | 0.9221 / 0.9221 | 一致 |
| Grounded-SAM 0.9443 | 实验一表 Grounded-SAM 行 | 0.9443 / 0.9443 | 一致 |
| Single-prompt SAM3 prior 0.9692 | 实验一表 "SAM3单提示词" 行（**P2 = "entire plant excluding pot"**） | 0.9692 / 0.9692 | 一致 |
| RAP-FSAM3 0.9706 | 实验一表 "RAP-FSAM3-v2" 行 | 0.9706 / 0.9706 | 一致 |

**⚠️ 关键陷阱：** 论文把基线行命名为 "Single-prompt SAM3 prior" 并在 Table 4 把同一数字
（0.9692）标注为 "entire potted plant"。归档证据表明该数字的真实提示词是 **P2 = "entire plant excluding pot"**
（`实验一` 表 "文本提示" 列：`P2: entire plant excluding pot`；`实验二` 表：`P2,entire plant excluding pot`）。
**论文把 P2（去盆）标注为"带盆整株"——这是同一数字、两种冲突的提示词语义。**

### 2.2 Table 4（提示词鲁棒性）— 数字一致，P2 标签错误

| 提示词 | 论文 F1 | 归档 F1（实验二汇总表） | 论文文本 | 归档文本 | 判定 |
|---|---|---|---|---|---|
| P1 | 0.7972 | 0.797210 | green plant | green plant | 一致 |
| P2 | 0.9692 | 0.969239 | **entire potted plant** | **entire plant excluding pot** | **标签不符** |
| P3 | 0.6471 | 0.647129 | leaves and stems | leaves and stems | 一致 |
| P4 | 0.0768 | 0.076755 | crop seedling | crop seedling | 一致 |
| P5 | 0.4051 | 0.405097 | plant body without background | plant body without background | 一致 |

### 2.3 Table 5（RAP 模块消融）— 混合口径 + 无出处数字

| 论文行 | 论文值 | 可对应的归档 | 判定 |
|---|---|---|---|
| Fixed plant-instance prompt | 0.9692/0.9624/193.71/0.4051/0.002272 | 实验二 P2 行（F1 0.969239, HD95 193.711, BF1 0.405071, Leak 0.002272）；主表 A0 为 sample_equal_mean（0.966426） | **F1/mIoU 取帧均值口径，HD95/BF1/Leak 同**；两个聚合不同 |
| Geometry-only w/o SG-MPS | 0.8429/0.8533/1008.90/0.3214/0.002035 | 主表 A1（0.842924/0.853311/1008.904/0.321366/0.002035，**sample_equal_mean**） | F1/mIoU 一致（sample 均值）；**但逐帧均值实为 0.7755/0.8553** |
| Semantic-gated selected prior | 0.9692/…（同 row1） | 主表 A1s == A0（0.966426/…，sample 均值） | 与 row1 同问题 |
| RAP-FSAM3 w/o GCFC | 0.9677/0.9613/190.70/0.4543/0.001935 | F1/mIoU == 主表 A5c sample 均值（0.967662/0.961313）；HD95 190.70 == A2 逐帧均值；**BF1 0.4543 与 Leak 0.001935 全仓无出处** | **混合 + 无出处数字** |
| RAP-FSAM3 (full) | 0.9706/0.9641/183.42/0.4643/0.001739 | HD95/BF1/Leak == A5c（183.416/0.464256/0.001739）；**F1/mIoU 无任何聚合能复现（A5c 逐帧 0.968656/0.962239；sample 均值 0.967662/0.961313）** | **F1/mIoU 来源未明** |

**Phase 20A 重算（逐帧均值，见上方表格）证明：论文 Table 5 各行的 F1/mIoU 在
"逐帧均值 / 样本等权均值"之间来回混用，且 "w/o GCFC" 与 "full" 两行的某些数字在归档中不存在。**

### 2.4 Table 6（先验注入消融）

来源：`04-结果表格模板/实验四_先验注入位置消融结果表.csv`
样本：KongQueZhuYu（单样本代表形态，论文亦自称"a representative broad-leaf foreground morphology" ✓）。
需 2DGS 代理确认 `foreground_object_results.json` 中各运行（A3/A4/A6_M1_soft_M4 等）的配置键。

### 2.5 Table 7（重建对比）

- COLMAP / 3DGS+VFM / 2DGS / ForeSplat 四行的时间与 PSNR 数字，归档位置 `数据管理/06-实验输出/KongQueZhuYu/*/foreground_object_results.json`
  与 `00-论文优化重构/m2m3实验/实验输出/runs/*/`。
- 论文声称 "averaged over all 20 RGB sequences"，但已定位的产物绝大多数为 **KongQueZhuYu 单样本** 或
  m2m3 小闭环（CaoMei2 / ChangShouHua2 / KongQueZhuYu 等子集）。**"20 序列平均"的证据待 2DGS 代理确认。**

### 2.6 Table 8（表型测量）

来源：`04-结果表格模板/实验五_人工虚拟表型汇总.csv`（n=21 植株 / n=63 叶）。
待 2DGS 代理返回后写入精确路径。

---

## 3. 结论分类（完整矩阵见 CSV）

| 论文要素 | 分类 | 理由 |
|---|---|---|
| Table 3 全部基线行（CLIPSeg/Grounded-SAM/Single-prompt SAM3） | `REUSE_AS_LEGACY_BASELINE` | 数字有出处，但评测集 = 20 帧非 20 序列；提示词口径与论文标签冲突 |
| Table 3 RAP-FSAM3 行 | `TARGETED_RERUN_REQUIRED` | F1/mIoU 0.9706/0.9641 无法从任何归档聚合复现；且该数字的生成运行未定位（0.9706 只出现在实验一表与论文） |
| Table 4 | `REUSE_AS_LEGACY_BASELINE`（数字）/ `TARGETED_RERUN_REQUIRED`（若论文坚持"带盆整株"口径） | 数字可继承，标签必改；若新论文需要真正的"带盆整株"单提示词对比，必须补跑 P6 |
| Table 5 | `TARGETED_RERUN_REQUIRED` | 混合聚合 + 无出处数字 |
| Table 6 | `REUSE_AS_LEGACY_BASELINE` | 单样本消融本身成立，但仅代表一种形态，论文描述基本准确 |
| Table 7 | `UNVERIFIED`（待 2DGS 代理返回） | "20 序列平均"证据未定位 |
| Table 8 | `UNVERIFIED`（待 2DGS 代理返回） | 待精确路径 |

---

## 4. 不允许做的事情（复核清单）

- ❌ 不得把 FourSample_GT20（4 序列 20 帧）说成"从全部 20 序列抽样"。
- ❌ 不得把 P2（entire plant excluding pot）的数字标注为带盆整株。
- ❌ 不得沿用 "RAP-FSAM3 w/o GCFC" 行中 BF1=0.4543 / Leak=0.001935 两个无出处数字。
- ❌ 不得在未定位 0.9706 生成运行前，直接引用该数字。
- ❌ 不得用 GT10（DEV）作为独立测试集。
- ❌ 不得用 TEST32（无 GT）计算准确率。
