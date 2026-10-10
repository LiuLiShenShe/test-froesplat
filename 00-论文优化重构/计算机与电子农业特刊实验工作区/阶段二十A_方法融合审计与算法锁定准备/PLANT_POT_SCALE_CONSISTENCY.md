# 前景 / 花盆 / 尺度恢复一致性核查 / PLANT_POT_SCALE_CONSISTENCY

**阶段：** Phase 20A
**判定词：** `CONSISTENT` / `INCONSISTENT` / `NOT VERIFIED`

---

## 0. 结论摘要

| 核查项 | 判定 | 依据 |
|---|---|---|
| 1. `plant` / `pot` / `potted_plant` / `blue_cube` 标注定义 | **VERIFIED** | LabelMe 标签解析 + Phase16 协议 |
| 2. GT 是否实际包含整个花盆 | **是** | 21/21 帧 pot∩potted ∈ [98.2%, 99.8%] |
| 3. `potted_clean = potted_plant AND NOT blue_cube` | **CONFIRMED（逐像素）** | Phase 20A 重算 21/21 帧一致率 = **1.000000** |
| 4. P6 掩膜的花盆覆盖程度 | **88.1% 全样本均值 / ≈97.5%（剔除 2 个退化帧）** | Phase 20A 像素级重算 |
| 5. 原 ForeSplat 2DGS 输入掩膜是否含盆 | **NOT VERIFIED（倾向"不含"）** | 掩膜目录无标注；目标声明为剔除盆体 |
| 6. COLMAP 前景初始化用哪种目标定义 | **`plant_only` 语义**（掩膜同 2DGS） | 同一 `mask_dir` |
| 7. 2DGS 最终重建是否保留花盆 | **NOT VERIFIED，证据倾向 NO** | 同上 |
| 8. 物理尺度恢复是否依赖花盆三维几何 | **NOT VERIFIED —— 且与 5/7 冲突** | 无实现代码 |
| 9. 只生成 plant-only mesh 时尺度参考如何传入 | **NOT VERIFIED —— 仓内无答案** | 无 `scale_marker` 实现 |
| 10. 新论文应统一的评价协议 | 见 §6 | — |

---

## 1. GT 类别定义（VERIFIED）

权威源：`/data/fj/F2DMAS/03-GT-区分/`（21 帧 LabelMe JSON，5 样本）。

| 标签 | 出现数 | 含义 |
|---|---|---|
| `potted_plant` | 21（每帧 1 个） | **目标植物，已包含花盆** |
| `pot` | 22 | 花盆 / 托盘（**单独标注，嵌套在 potted_plant 内**） |
| `plant` | 7 | 仅叶茎（**仅 CaoMei1×5 + ChangShouHua2×2 人工标注**；其余 14 帧由 `potted_plant − pot` 自动推算） |
| `blue_cube` | 19 | 蓝色标定块 |

协议依据：`阶段十六_HardCase_GT构建与锁定/Phase16_GT_annotation_protocol.md` §2、§4
> "If labeled as `pot`, included in `potted_plant` (whole pot = potted_plant)"

**Phase 20A 实测嵌套度：** pot∩potted ∈ [98.2%, 99.8%] → `potted_plant` 是一个**已包含盆的闭合区域**。
`potted_plant XOR (plant ∪ pot)` 在 20/21 帧非零（358–42,699 px，0.02%–1.5%），属标注 slop。

**无颜色编码掩膜**：所有 `gt_*.png` 为单通道 0/255 二值；颜色映射仅存在于
`阶段十二…/GT_QA/处理GT_v2.py` L124-150 的可视化分支，不落盘为掩膜。

---

## 2. `potted_clean` 定义 —— **CONFIRMED**

三处独立代码 + 一处协议一致：

| 来源 | 原文 |
|---|---|
| `阶段十二…/GT_QA/处理GT_v2.py` L8 | `GT_potted_clean/<sample>/mask_potted_clean_<frame>.png  (potted & ~cube)` |
| 同上 L67, L69 | `# potted_clean = potted & ~cube` / `potted_clean = potted & ~cube` |
| `阶段十六…/脚本/process_gt_challenge.py` L12, L103 | `Same target definition as the Phase 12 easy set: potted_clean = potted & ~cube.` |
| `Phase16_GT_annotation_protocol.md` §1 | `potted_clean = potted_plant AND NOT blue_cube` |

**Phase 20A 逐像素复核（21 帧）：**

```
mean agreement  potted_clean == (potted_plant AND NOT blue_cube)  =  1.000000
```
每帧单独一致率均为 1.000000 → **定义完全成立，且不含任何 pot 减除。**

**实际影响极小：** 21 帧中仅 1 帧被该减法改变（`CaoMei1_0050`，cube 重叠 393 px，
683,626 → 683,233）。其余 20 帧 `clean_area == potted_area`。

---

## 3. P6 掩膜的花盆覆盖（Phase 20A 像素级重算）

数据：`阶段十二…/P6_raw_baseline/最终掩膜/mask_<sample>_<frame>.png`
vs `GT口径拆分审计/gt_masks_split_v2/{mask_pot, mask_plant, mask_potted_plant, mask_blue_cube}`

| 指标 | 值 |
|---|---|
| **盆区召回率（P6 ∩ pot）/ pot** | **88.10%**（21 帧均值） |
| 盆区召回率（剔除 2 个退化帧） | **≈97.5%** |
| 植株区召回率 | 97.13% |
| 带盆区召回率 | 97.27% |
| 蓝色标定块重叠 | **0 px，全部 21 帧** |

逐帧（部分）：

| 帧 | P6 面积 | GT pot | 盆召回 | GT plant | 植株召回 |
|---|---|---|---|---|---|
| CaoMei1_0000 | 1,056,759 | 601,751 | 97.5% | 479,387 | 97.6% |
| DouBanLv1_0000 | 1,892,117 | 653,259 | 97.2% | 1,280,072 | 98.0% |
| KongQueZhuYu_0075 | 2,129,097 | 249,803 | 91.3% | 1,933,933 | 97.6% |
| KongQueZhuYu_0100 | 2,390,131 | 19,380 | 6.3% ⚠ | 2,427,652 | 98.2% |
| ChangShouHua2_0100 | 2,602,120 | 0 | n/a（GT 缺盆多边形） | 2,673,181 | 96.8% |

> ⚠ `KongQueZhuYu_0100` 的 GT 盆多边形仅 19,380 px（占带盆区 0.8%），
> `ChangShouHua2_0100` 的 `has_pot=0` —— 这是 **GT 侧的退化帧**，非 P6 的失败。

**判定：P6 掩膜确实包含花盆，与 `potted_clean` GT 口径一致 → `CONSISTENT`。**

---

## 4. P2 提示词实际上**并未**排除花盆 —— 命名误导

生产提示词典（`生成RAP-FSAM3掩膜.py` L37-47）：
```python
"P2": "entire plant excluding pot",
"P6": "whole potted plant including the pot",
"POT": "flowerpot and tray",
"CUBE": "blue calibration cube",
```

**Phase 20A 对 E1 的 `method_masks/SAM3_P2/`（20 帧）实测：**

| 提示词 | 盆区召回率（均值） | 植株区召回率 |
|---|---|---|
| **P2**（"excluding pot"） | **≈97.5%** | 97.7% |
| **P6**（"including the pot"） | ≈97.5% | 97.1% |

**结论：P2 与 P6 在花盆覆盖上统计不可区分（差 < 0.1 个百分点）。**

仓内旁证：`04-结果表格模板/实验三_P2去盆vsP6带盆_四样本16GT对比.csv` 第 6 行
> `CaoMei1,0100,去盆,0.229,0.235,0.178,SAM3候选失败(SAM3只割到盆)两口径同败`

**影响：**
1. P2 的命名（"去盆" / "excluding pot"）与实际行为**不符**；
2. 论文 Table 4 把 P2 的数字标为 "entire potted plant"（错标方向相反，但仍不准确）；
3. **"P2 与 P6 谁更适合论文"这一选择，在花盆覆盖维度上不是有效区分轴**（见 `METHOD_SELECTION_DECISION.md`）。

---

## 5. 重建侧：尺度恢复 —— **NOT VERIFIED（P0 阻断项）**

### 5.1 无实现代码

| 检索目标 | 结果 |
|---|---|
| `pot_diameter` / `pot diameter` / `花盆直径` / `盆径`（全仓 md/py/json/csv/xlsx） | **0 命中**（仅散文提及） |
| `restore_scale` / `scale_marker` 实现 | `ForeSplat_real_release/scripts/measure_traits_entry.py:14` 只有 `--scale-marker` 参数转发 |
| `configs/default_foresplat.yaml:12` | 声明 `scale_marker: examples/demo_sequence/scale_marker.json` —— **该路径在发布树中不存在** |
| `植株数据.xlsx`（3 份副本） | 22 行 × 17 列，**无任何盆径列**（仅株高/冠幅/叶长/叶宽的真值与虚拟值） |
| `06-MESH-new/*_mesh.log`（13 样本） | 不含 pot / 直径 / diameter / scale 任何字段；样本目录仅有 `fuse.ply`、`fuse_post.ply` |

散文级唯一陈述（`论文cc/02_manuscript_chinese_draft.md:439`）：
> "尺度恢复：绝对尺度使用单个已知物理尺寸（花盆直径，使用数字卡尺测量，精度 ±0.5 mm）恢复。"

**判定：花盆直径这一尺度参照量在全仓中不存在具体数值，也无实现代码 → `NOT VERIFIED`。**

### 5.2 冲突：重建掩膜若为 plant-only，则无法从网格量取盆径

- 2DGS / M2M3-GO / 掩膜剪枝使用的统一 `mask_dir`：
  `/data/fj/F2DMAS/00-论文优化重构/数据管理/03-分割Mask/02-sam_masks/<Sample>/`（19 样本）
- 该目录**无任何 pot 标注**；`02-sam_masks` 与 RAP-FSAM3 的 P6 带盆目录是**两个不同目录**。
- 目标声明（`ForeSplat_zh.md:89, :194`）是把盆/基质作为待剔除对象。
- 发布版网格入口 `ForeSplat_real_release/scripts/extract_mesh_entry.py:35` 只写 manifest，
  明示完整实现未随发布提供。

**判定：若重建掩膜为 plant-only，则论文所述"用花盆直径从重建场景恢复尺度"在方法上不闭合。
仓内无任何代码或记录解决这一点。→ `INCONSISTENT`（方法叙述 vs 实现状态）。**

### 5.3 表型测量本身也是"只算统计不算几何"

`03-扩展实验后续执行/生成实验一五当前结果.py`：
- L804-822 `load_phenotype_pairs()` 读取 **已记录**的 `株高虚拟植` 等列；
- L825-846 `trait_metrics()` 算 MAE/RMSE/MAPE/bias/R²；
- **不执行**从网格到性状的任何几何测量。

唯一的"测量工具"是桩件：`ForeSplat_real_release/scripts/measure_traits_entry.py:30-33`
> "The full measurement tool is interactive/semi-automatic and is not bundled with this lightweight subset."

n=21 / n=63 与论文 Table 8 一致（`04-结果表格模板/实验五_人工虚拟表型汇总.csv`）：
```
株高,21,...,0.9848,1.2073,6.9064,0.5771,0.9878
冠幅,21,...,0.8648,0.9880,4.5000,0.6410,0.9879
叶长,63,...,0.5063,0.6370,7.4491,0.3133,0.9738
叶宽,63,...,0.4502,0.6378,9.7256,0.3825,0.8999
```
**判定：Table 8 的统计量 `VERIFIED`，但其上游几何测量链 `NOT VERIFIED`。**

---

## 6. 新论文应统一的评价协议（建议）

**必须统一的三条线：**

1. **分割评价目标 = `potted_clean`（含盆、去蓝色标定块）**
   - 与 Phase 12 / 16 / 17 / 18 全部已报数字一致；
   - 与 P6 提示词语义一致；
   - 与花盆尺度参照自洽。
   - ❌ 停止使用 `03-GT/` 的旧并集（含 cube），停止使用 `gt_masks_split/` v1 启发式版本
     （其 cube IoU 均值仅 0.048，20/21 帧失败）。

2. **提示词命名必须与实际行为一致**
   - 既然 P2 实测并不去盆，建议**废弃 P2 这一命名**，或改用行为中性的名称；
   - 论文若要比较"去盆 vs 带盆"，必须用 `compose_plant_only()`（`生成RAP-FSAM3掩膜.py` L1771）
     做受控减除 —— **但该函数当前是死代码，从未被调用**，需先实现并验证。

3. **尺度恢复必须补齐可复现证据**
   - 提供每样本/每品种的盆径实测值（含测量工具与不确定度）；
   - 明确尺度参照来自**原始图像中的花盆**还是**重建网格中的花盆**；
   - 若网格为 plant-only，必须说明参照如何传入（例如以图像域盆径 → 相机尺度 → 网格尺度）。

---

## 7. 不一致项的最低成本修正方案

| 编号 | 不一致 | 最低成本修正 | 需 GPU |
|---|---|---|---|
| P1 | `potted_clean` 定义正确，但 `03-GT/` 旧并集仍在流通 | 论文所有分割数字统一重算到 `potted_clean`；旧并集仅作历史 | ❌ 离线 |
| P2 | P2 命名与行为不符（不去盆） | 改名或废弃 P2；`plant_only` 评测改用 `compose_plant_only()` 受控减除 | ❌（若复用候选掩膜） |
| P2-2 | 挑战集 GT 无 plant/pot 拆分（10 帧全 `pot_status=missing`，`formal_p2_valid=False`） | 补标 10 帧的 pot 多边形 | ❌（人工标注） |
| P3 | 花盆直径无记录、无代码 | 补录实测值 + 实现或说明尺度传递路径 | ❌ |
| P4 | 尺度参照与 plant-only 网格冲突 | 明确参照域并写入方法节 | ❌ |
| P5 | 网格→性状几何测量无实现 | 至少提供测量脚本或说明外部工具 | ❌ |

**上述 P1–P5 均不需要重新训练 2DGS。**

---

## 8. 审计边界

- 未修改任何 GT、掩膜或历史指标。
- 本文件所有像素级统计均由 Phase 20A 现场只读重算得出，未写入任何文件。
- 未把"缺少 pot GT 多边形"误判为"P6 漏检盆"——两者已在 §3 分别量化。
- 未把 TEST32 无 GT 帧纳入任何一致率统计。