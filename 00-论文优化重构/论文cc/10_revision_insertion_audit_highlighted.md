# ForeSplat 修稿插入审计与高亮标注

审计对象：`02_manuscript_chinese_draft copy.md`

审计结论：当前稿件的主要问题已经不是“没有数据”，而是正文叙事仍按“3 个代表样本 + 内部消融”来防守。既然已经有 20 个序列的完整重建和表型结果、人工真值分割、SEEM 对比图和数据，以及 2DGS-full-scene / 3DGS-FSAM3 / SuGaR 数据，主文应改成“全数据集验证 + 代表样本机制消融 + 外部 Gaussian baseline 对比”的结构。

标记规则：

- `<mark>必须加入</mark>`：不加会继续被审稿人认为证据链不闭合。
- `<mark>建议加入</mark>`：能明显增强说服力，但可放主文或补充材料。
- `<mark>必须替换</mark>`：当前文字已经与现有数据事实不一致。
- `<mark>降调/删除</mark>`：保留会造成过强 claim 或自我削弱。

---

## 1. 总体结构调整

### 当前核心风险

1. 文章仍多次写成“三个代表样本验证重建”，但你已经有 20 个序列完整重建和表型数据。
2. 4.1 的 FSAM3 vs SEEM 仍是占位语，手工分割 GT 没有进入证据链。
3. 外部 baseline 在局限性中被写成“未包括”，但你已有 2DGS-full-scene、3DGS-FSAM3、SuGaR 数据。
4. 表型验证只报告 ForeSplat 人工-虚拟一致性，还没有充分显示 ForeSplat 相对 baseline 是否降低表型误差。
5. 图表编号已混乱，新增表图后必须整体重排。

### 建议主文结果顺序

<mark>必须替换 4 章结构为以下逻辑：</mark>

1. **4.1 FSAM3 segmentation benchmark and mask prior analysis**  
   人工真值分割 + SEEM 等方法对比，证明 mask 先验可靠。

2. **4.2 Objective ablation on foreground-object 2DGS**  
   保留 KongQueZhuYu 单样本机制消融，证明 foreground RGB supervision 是关键。

3. **4.3 Full-dataset ForeSplat reconstruction across 20 sequences**  
   新增 20 个序列完整重建结果，解决跨物种/全数据集验证不足。

4. **4.4 Comparison with Gaussian-based reconstruction baselines**  
   新增 2DGS-full-scene、3DGS-FSAM3、SuGaR 对比，解决外部 baseline 缺失。

5. **4.5 View-quality strategy ablation**  
   保留单样本或代表样本视角质量消融，定位为机制实验。

6. **4.6 Compactness and mesh export analysis**  
   如果紧凑化只跑 3 个样本，明确是 compactness ablation；如果 20 个都有，则升级为 full-dataset compactness table。

7. **4.7 Phenotypic measurement validation and baseline-level phenotype errors**  
   ForeSplat 全数据集表型结果 + baseline 表型误差对比。

---

## 2. 摘要与 Highlights

### 位置

原稿第 38-44 行 Highlights；第 48-50 行摘要。

### 必须替换的逻辑

<mark>必须替换</mark> “完整 ForeSplat 流程在三个代表样本上...” 这一摘要核心证据。现在应写成：

> <mark>实验覆盖 10 个物种的 20 个多视角序列；ForeSplat 在全部序列上完成 plant-only 重建和表型测量，机制消融在代表性复杂背景样本上进行。</mark>

摘要里建议加入四个数字组：

1. <mark>FSAM3 vs SEEM 分割 benchmark：</mark> mIoU / F1 / HD95 / runtime，写平均值或范围。
2. <mark>20 序列 ForeSplat 重建：</mark> PSNR_fg、outside、leakage、mesh success rate、Gaussian count 的 mean ± SD。
3. <mark>外部 baseline 对比：</mark> 2DGS-full-scene、3DGS-FSAM3、SuGaR 与 ForeSplat 的关键差异，优先写 outside/leakage 和表型误差。
4. <mark>20/21 株表型：</mark> 株高、冠幅、叶长、叶宽 R² / MAPE。

### 建议摘要替换骨架

<mark>必须加入</mark> 以下信息骨架，数字用你的数据填：

```text
Experiments were conducted on 20 multi-view sequences covering 10 potted plant species. A manually annotated mask subset was used to compare FSAM3 with SEEM and other segmentation baselines, where FSAM3 achieved [mIoU], [F1], [HD95] and [time]. Across all 20 sequences, ForeSplat completed foreground-only reconstruction with [PSNR_fg mean ± SD], [outside mean ± SD] and [leakage mean ± SD]. Compared with 2DGS-full-scene, 3DGS-FSAM3 and SuGaR under matched input views and masks, ForeSplat reduced background leakage and produced cleaner phenotype-ready meshes. Manual-virtual measurements over [20 sequences / 21 plants] achieved R² of 0.991, 0.993, 0.980 and 0.956 for plant height, canopy width, leaf length and leaf width, respectively.
```

### Highlights 建议改为 5 条

<mark>必须加入</mark>

- FSAM3 was validated against manually annotated plant masks and SEEM on a representative mask benchmark.
- ForeSplat completed plant-only reconstruction for all 20 multi-species sequences.
- Foreground RGB supervision, rather than alpha/background regularization alone, is the decisive mechanism.
- ForeSplat was compared with 2DGS-full-scene, 3DGS-FSAM3 and SuGaR under matched input conditions.
- Full-dataset phenotype validation showed high agreement for global traits, while leaf width remained boundary-sensitive.

---

## 3. 引言贡献点

### 位置

原稿第 67-72 行贡献列表。

### 当前问题

贡献 1 把 FSAM3 写得像独立分割创新，但结果部分没有完整分割 benchmark。现在你有手工真值和 SEEM 数据，可以支撑它，但仍建议定位为“重建导向 mask 先验”，避免被要求做通用分割 SOTA。

### 修改建议

<mark>必须替换贡献 1：</mark>

```text
1. We introduce FSAM3 as a reconstruction-oriented plant foreground prior. It combines FFT-based frame screening, promptable plant segmentation and PCA-guided component refinement, and is evaluated on manually annotated plant masks against SEEM and other segmentation baselines.
```

<mark>必须替换贡献 3：</mark>

```text
3. We evaluate ForeSplat on all 20 multi-view sequences for reconstruction and phenotypic measurement, while using representative controlled ablations to isolate the effects of foreground RGB supervision, soft view weighting and mask-guided Gaussian cleanup.
```

<mark>必须加入贡献 4 或合并进贡献 3：</mark>

```text
4. We compare ForeSplat with Gaussian-based external baselines, including 2DGS-full-scene, 3DGS-FSAM3 and SuGaR, under matched input views and mask conditions, and report both foreground reconstruction metrics and downstream phenotypic errors.
```

---

## 4. 方法部分新增内容

## 4.1 数据集与实验用途

### 位置

原稿第 140-148 行。

### 必须替换

<mark>必须替换</mark> 第 146 行“完整配置跨样本验证使用 KongQueZhuYu、XianKeLai1 和 CaoMei2”的口径。应改为：

```text
ForeSplat full-pipeline reconstruction and phenotype extraction were performed on all 20 sequences. KongQueZhuYu was used for objective-function ablation because it contains strong background leakage cues. KongQueZhuYu, XianKeLai1 and CaoMei2 were used as representative visualization and mechanism-analysis cases, corresponding to complex background, thin-leaf geometry and dense occlusion. External Gaussian baselines were evaluated on [all 20 sequences / the same 20 sequences / specify subset], using matched retained frames and FSAM3 masks where applicable. The segmentation benchmark used [N] manually annotated frames from [M] sequences.
```

### Table 1 必须扩展

<mark>必须加入 Table 1 字段：</mark>

| 字段 | 目的 |
|---|---|
| Sample ID | 20 个序列逐一列出 |
| Species CN / Latin candidate | 解决物种命名 |
| Growth type | 便于解释泛化，如 rosette / thin-leaf / dense canopy |
| Raw frames | 原始采集规模 |
| FFT-retained frames | 质量筛选后帧数 |
| COLMAP registered frames | SfM 成功率 |
| FSAM3 masks generated | mask 是否完整 |
| Manual mask GT frames | 哪些序列参与分割 benchmark |
| ForeSplat reconstruction | all yes |
| External baselines | 2DGS / 3DGS / SuGaR 是否完成 |
| Phenotype GT | 株高、冠幅、叶长、叶宽 |

---

## 4.2 FSAM3 分割 benchmark 协议

### 位置

原稿第 150-186 行后，新增 `3.3.4 Segmentation benchmark protocol`。

### 必须加入

<mark>必须加入</mark> 手工真值标注协议：

```text
For segmentation evaluation, [N] frames were manually annotated from [M] sequences covering broad leaves, thin leaves, dense canopy and complex backgrounds. The plant foreground excluded pot, soil, table and background objects. All methods were evaluated against the same binary masks after resizing to the reconstruction resolution. We report mIoU, F1-score/Dice, HD95 and average processing time.
```

### 必须加入 baseline 设置

<mark>必须加入</mark>

| 方法 | 设置必须写清楚 |
|---|---|
| FSAM3 | prompt P2, FFT screening, PCA refinement, threshold 0.5 |
| SEEM | prompt / text query / region setting，是否后处理 |
| SAM/SAM2/SAM3 raw if available | 同 prompt，不做 PCA 或只做最小后处理 |
| ExG/Otsu/HSV if available | 传统颜色 baseline，可放补充材料 |

### 关键措辞

<mark>降调/删除</mark> 不要写 “FSAM3 achieves SOTA segmentation”。建议写：

```text
The purpose of this benchmark is not to claim general-purpose segmentation superiority, but to test whether FSAM3 provides more reconstruction-suitable plant foreground masks under our acquisition setting.
```

---

## 4.3 外部 baseline 公平性协议

### 位置

原稿第 297-309 行 `3.8 实验矩阵与验证设计` 后扩展。

### 必须加入

<mark>必须加入</mark> baseline 公平性说明：

```text
For external Gaussian-based baselines, all methods used the same FFT-retained input views and the same train/evaluation split. Methods requiring foreground masks used the same FSAM3 masks. 2DGS-full-scene was trained with the original full-image RGB supervision and no foreground restriction. 3DGS-FSAM3 used the same foreground masks for input masking or alpha handling [specify exact mode]. SuGaR was initialized from [3DGS / its native pipeline] and converted to mesh using its native surface-aligned extraction. All methods were evaluated using the same foreground masks and phenotype measurement protocol.
```

### 必须避免

<mark>降调/删除</mark> 不要写“same pipeline”笼统比较。要拆成：

- same retained frames
- same camera poses or method-native pose handling
- same FSAM3 masks where applicable
- same foreground evaluation masks
- same phenotype measurement definitions
- method-native mesh extraction allowed

---

## 4.4 评价指标

### 位置

原稿第 311-324 行。

### 必须补全

<mark>必须加入 outside_nonblack 阈值</mark>：

```text
outside_nonblack_ratio_mean was computed as the proportion of background pixels whose rendered RGB intensity exceeded [threshold, e.g., 10/255 or 0.04 after normalization].
```

<mark>必须加入公式</mark>：

```text
PSNR_fg = 10 log10(MAX_I^2 / MSE_fg)
MSE_fg = sum_p M(p)||R(p)-I(p)||_2^2 / sum_p M(p)

outside = sum_p (1-M(p)) 1[||R(p)||_1/3 > tau] / sum_p (1-M(p))

leakage = sum_p (1-M(p))||R(p)||_2^2 / (sum_p M(p)||R(p)||_2^2 + epsilon)

MAE = mean_i |y_i - x_i|
RMSE = sqrt(mean_i (y_i - x_i)^2)
MAPE = mean_i |(y_i - x_i)/x_i| * 100%
Bias = mean_i (y_i - x_i), where y_i is virtual and x_i is manual.
```

---

## 5. 结果部分需要加入的表和图

## 5.1 4.1 FSAM3 mask benchmark

### 原稿位置

第 338-344 行；第 344 行目前是占位句。

### 必须替换占位句

<mark>必须替换</mark> 第 344 行：

```text
To quantify mask quality, we evaluated FSAM3 against manually annotated plant masks and compared it with SEEM and [other baselines]. Table 2 reports overlap, boundary and runtime metrics, while Fig. 3 shows representative success and failure cases.
```

### Table 2：Segmentation benchmark

<mark>必须加入主文表</mark>

| Method | Prompt / setting | Postprocess | n frames | mIoU ↑ | F1/Dice ↑ | HD95 ↓ | Time/image ↓ | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| FSAM3 | P2 | FFT + PCA | [N] | [ ] | [ ] | [ ] | [ ] | reconstruction prior |
| SEEM | [prompt] | [yes/no] | [N] | [ ] | [ ] | [ ] | [ ] | baseline |
| SAM3 raw | P2 | none/minimal | [N] | [ ] | [ ] | [ ] | [ ] | optional |
| ExG/Otsu | automatic | morphology | [N] | [ ] | [ ] | [ ] | [ ] | optional/supplement |

### Figure 2 or Figure 3：SEEM 对比图

<mark>必须加入图</mark>

建议布局：

- 列：RGB / manual GT / FSAM3 / SEEM / error map
- 行：至少 4 类样本：宽叶、薄叶、密集遮挡、复杂背景
- error map：false positive 用红色，false negative 用蓝色
- caption 要写：FSAM3 不是通用分割 SOTA，而是更适合本文重建先验。

---

## 5.2 4.2 Objective ablation

### 原稿位置

第 348-373 行。

### 保留但补齐

<mark>必须补齐</mark> 表中 E7 的 LPIPS 和 Gaussian 数量，或删除该列在 E7 的占位并在表注说明 “not available because ...”。主文正式投稿不能保留 `[占位：待统一统计]`。

### Figure 4

<mark>必须加入</mark> 当前 Fig. 4 仍然需要，但建议增加 E7 post-hoc pruning 一列：

列顺序：

1. 2DGS-full-scene
2. input foreground masking
3. fg RGB + alpha/bg
4. ForeSplat core
5. full-scene + post-hoc mask pruning

行：

1. RGB render
2. alpha / opacity map
3. leakage heatmap outside mask
4. leaf-edge zoom

---

## 5.3 4.3 Full-dataset ForeSplat reconstruction

### 原稿位置

第 375-387 行。

### 必须替换

<mark>必须替换</mark> 当前“前景对象重建目标完整配置在三个代表性结构样本上...”为：

```text
ForeSplat was evaluated on all 20 multi-view sequences to test whether the foreground-object objective remains stable beyond representative ablation cases. Table 4 summarizes per-sequence reconstruction metrics, and Fig. 5 shows the distribution of foreground quality and leakage across species and growth types.
```

### Table 4：20 序列完整重建结果

<mark>必须加入主文表或主文精简表 + Supplement 全表</mark>

建议主文表：

| Sample | Species | Growth type | Registered views | PSNR_fg ↑ | SSIM_fg ↑ | LPIPS_fg ↓ | outside ↓ | leakage ↓ | Gaussians ↓ | Mesh success | Phenotype measured |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| S01 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | yes/no | yes/no |

如果表太长，主文可放 aggregation：

| Group | n seq | PSNR_fg mean ± SD | outside mean ± SD | leakage mean ± SD | Gaussians mean ± SD | Mesh success |
|---|---:|---:|---:|---:|---:|---:|
| Broad-leaf / rosette | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Thin-leaf | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Dense canopy | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| All | 20 | [ ] | [ ] | [ ] | [ ] | [ ] |

### Figure 5：20 序列分布图

<mark>必须加入</mark>

建议布局：

- a: PSNR_fg by sample/growth type
- b: outside and leakage scatter, foreground-only 阈值线画出来
- c: Gaussian count distribution
- d: representative render thumbnails from 6-8 species

---

## 5.4 4.4 External Gaussian baseline comparison

### 插入位置

建议放在 full-dataset ForeSplat 结果之后，视角质量消融之前。即新增为新的 `4.4`，原 4.4 后移。

### 必须加入

<mark>必须加入</mark> 外部 baseline 结果，因为这是评审最致命问题之一。

### Table 5：外部 Gaussian baseline 定量对比

<mark>必须加入主文表</mark>

| Method | Input / mask setting | n seq | PSNR_fg ↑ | SSIM_fg ↑ | LPIPS_fg ↓ | outside ↓ | leakage ↓ | Representation size ↓ | Mesh success ↑ | Runtime ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2DGS-full-scene | full RGB, no mask objective | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| 3DGS-FSAM3 | same masks, method-native 3DGS | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| SuGaR | 3DGS-to-mesh baseline | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| ForeSplat | foreground-object 2DGS | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

建议报告 mean ± SD，而不是只写百分比提升。

### Table 6：表型误差 baseline 对比

<mark>强烈建议加入</mark> 如果你有 baseline 表型数据，这张表是最能打 CompAg 的表。

| Method | Height MAE / MAPE | Canopy MAE / MAPE | Leaf length MAE / MAPE | Leaf width MAE / MAPE | Mean trait MAPE |
|---|---:|---:|---:|---:|---:|
| 2DGS-full-scene | [ ] | [ ] | [ ] | [ ] | [ ] |
| 3DGS-FSAM3 | [ ] | [ ] | [ ] | [ ] | [ ] |
| SuGaR | [ ] | [ ] | [ ] | [ ] | [ ] |
| ForeSplat | [ ] | [ ] | [ ] | [ ] | [ ] |

如果叶长/叶宽 baseline 测量成本高，最低限度主文保留株高和冠幅，叶片性状放补充材料。

### Figure 6：外部 baseline 可视化

<mark>必须加入图</mark>

建议布局：

- 行：3 个代表样本（复杂背景 / 薄叶 / 密集遮挡）
- 列：RGB input / 2DGS-full-scene / 3DGS-FSAM3 / SuGaR / ForeSplat
- 每格包含 render 或 mesh；另加 leakage heatmap 或 mesh zoom

图注核心句：

```text
The comparison uses matched retained views and evaluation masks. ForeSplat differs from the baselines by optimizing the RGB supervision domain itself rather than only masking inputs or extracting surfaces after full-scene training.
```

---

## 5.5 视角质量与紧凑性消融

### 原稿位置

第 389-434 行。

### 建议保留但重命名

<mark>必须降调</mark> 这两节不要再承担“跨物种主证据”，而是写成“mechanism ablation on representative sequences”。

建议标题：

- `4.5 Mechanism ablation: hard filtering versus soft view weighting`
- `4.6 Mechanism ablation: compact foreground cleanup`

### Table 7：视角质量策略

保留当前表，但表编号重排。

<mark>建议加入</mark> 如果 20 序列也有 soft weighting final 数据，可加一列或补充表：

| Setting | n seq | PSNR_fg mean ± SD | outside mean ± SD | leakage mean ± SD | Gaussians mean ± SD |

### Figure 7：视角质量图

保留当前 Fig. 5 方案：

- hard filtering angle gaps
- soft weights around camera trajectory
- PSNR / leakage / Gaussian count bars

### Table 8：紧凑性

如果只有 3 个样本，保留当前表并写明 representative compactness ablation。  
如果 20 个序列都有 final compact ForeSplat，升级为：

| Group / all | n seq | Core Gaussians | Compact Gaussians | Reduction % | PSNR change | outside change | leakage change |

### Figure 8：紧凑性 trade-off

建议从 3 样本柱状图升级成 scatter：

- x: Gaussian reduction %
- y: PSNR_fg change
- point color: growth type
- point size: initial Gaussian count

这比 3 个柱状图更能体现 20 序列全数据。

---

## 5.6 Mesh 和表型验证

### 原稿位置

第 436-476 行。

### Mesh section

如果只有两个样本的 TSDF 变体实验，当前写法可保留。  
如果 20 序列都有 mesh success 和顶点数，必须加 full-dataset mesh summary：

<mark>建议加入 Table 9A：</mark>

| Method / setting | n seq | Mesh success | Vertices mean ± SD | Components mean ± SD | Largest component ratio | Mesh time |
|---|---:|---:|---:|---:|---:|---:|

### Phenotype section

<mark>必须替换</mark> “跨 10 个物种、20 株植物”与摘要里的“21 株植物”口径不一致。统一成：

- 如果确实是 20 序列、21 株：写 “20 sequences containing 21 plants”
- 如果是 20 序列、20 株：把所有 n=21 改成 n=20，并重新核对表型统计

### Table 10：ForeSplat 人工-虚拟一致性

保留当前表，但加入 95% LoA 更好：

| Trait | n | MAE | RMSE | MAPE | Bias | 95% LoA | R² |

### Table 11：baseline-level phenotype error

<mark>强烈建议加入主文</mark>，见第 5.4 的 Table 6。  
如果篇幅有限，可把 baseline-level phenotype 放在 4.7，而外部重建指标放在 4.4。

### Figure 9 or 10：表型验证

当前 Fig. 8 应保留，但建议升级：

- 2 x 2 manual-vs-virtual scatter
- 每个 panel 标注 R²、MAE、MAPE、Bias
- 每个 panel 增加 Bland-Altman inset，至少叶宽必须有
- 点颜色按 species 或 growth type

---

## 6. 讨论与局限性必须同步更新

## 6.1 FSAM3 讨论

### 原稿位置

第 492-496 行。

### 必须替换

<mark>必须替换</mark> “缺少密集像素级 ground truth”相关句子，因为你已经有手工 GT。

建议改为：

```text
The manually annotated mask benchmark supports FSAM3 as a reconstruction-oriented foreground prior under our acquisition setting. However, the benchmark is still limited to [N] representative frames and is not intended to establish general segmentation state-of-the-art performance across arbitrary agricultural scenes.
```

## 6.2 与三维重建方法关系

### 原稿位置

第 516-520 行。

### 必须加入

<mark>必须加入</mark> 外部 baseline 讨论：

```text
The comparison with 2DGS-full-scene, 3DGS-FSAM3 and SuGaR shows that the main advantage of ForeSplat is not merely the use of Gaussian primitives or post-hoc surface extraction, but the alignment between the RGB supervision domain and the plant-only measurement target.
```

## 6.3 边界条件与未来方向

### 原稿位置

第 522-526 行。

### 必须替换

<mark>必须删除/替换</mark>：

- “三维重建层面的定量验证则基于代表性结构样本”
- “未来工作应...扩展前景对象重建...并纳入逐物种定量重建比较”
- “建立代表性帧的人工 mask ground truth”
- “加入 ... 3DGS ... baseline”

这些都已经被你的现有数据覆盖，继续写会自我削弱。

### 新局限性建议

<mark>必须加入</mark> 更准确的新局限：

1. **消融验证范围**：完整重建覆盖 20 序列，但目标函数和视角质量消融主要在代表样本上完成。
2. **baseline 范围**：已比较 Gaussian-based baselines，但尚未系统比较 COLMAP+MVS、NeRF/NeuS 等非 Gaussian 流程。
3. **分割 benchmark 范围**：有人工真值，但标注帧数仍是代表性子集，不是大规模分割数据集。
4. **环境范围**：室内或半受控环境，田间强光、风、自然复杂背景未验证。
5. **测量协议**：虚拟 landmark 单操作者，缺少操作者间/操作者内重复性。
6. **尺度恢复**：单一花盆直径尺度参照，未做多点标定。

---

## 7. 需要的最终图表清单

## 主文推荐图

<mark>必须加入/保留 10 幅左右主图；篇幅紧可把部分放 Supplement。</mark>

| 图号建议 | 图名 | 状态 | 目的 |
|---|---|---|---|
| Fig. 1 | ForeSplat task and pipeline overview | 保留/合并原 Fig.1+Fig.3 | 说明 full-scene 到 plant-only 的任务重定义 |
| Fig. 2 | FSAM3 pipeline and segmentation examples | 必须加入 | 展示 FFT/SAM3/PCA 与 mask 输出 |
| Fig. 3 | FSAM3 vs SEEM on manual GT | 必须加入 | 关闭 FSAM3 证据链问题 |
| Fig. 4 | Objective ablation and leakage maps | 必须加入 | 支撑 foreground RGB 是关键 |
| Fig. 5 | Full-dataset ForeSplat reconstruction distribution | 必须加入 | 支撑 20 序列泛化 |
| Fig. 6 | External Gaussian baseline visual comparison | 必须加入 | 支撑 2DGS/3DGS/SuGaR 对比 |
| Fig. 7 | Hard filtering vs soft weighting | 建议保留 | 支撑视角质量机制 |
| Fig. 8 | Compactness trade-off across samples | 建议加入 | 支撑紧凑化收益 |
| Fig. 9 | Mesh structure and boundary examples | 建议保留 | 支撑 phenotype-ready mesh |
| Fig. 10 | Manual vs virtual phenotype + Bland-Altman | 必须加入 | 支撑表型测量有效性 |

## 主文推荐表

<mark>必须重排表编号，不要继续出现多个 Table 3 / Table 4。</mark>

| 表号建议 | 表名 | 主文/补充 | 必要性 |
|---|---|---|---|
| Table 1 | Dataset summary and experiment coverage for 20 sequences | 主文 | 必须 |
| Table 2 | Segmentation benchmark against manual masks | 主文 | 必须 |
| Table 3 | Objective ablation on KongQueZhuYu | 主文 | 必须 |
| Table 4 | Full-dataset ForeSplat reconstruction across 20 sequences | 主文或主文汇总 + S2 全表 | 必须 |
| Table 5 | External Gaussian baseline comparison | 主文 | 必须 |
| Table 6 | Baseline-level phenotype error comparison | 主文，若有数据 | 强烈建议 |
| Table 7 | View-quality strategy ablation | 主文或补充 | 建议 |
| Table 8 | Compactness ablation / full-dataset compactness | 主文或补充 | 建议 |
| Table 9 | Mesh structural and efficiency metrics | 主文或补充 | 建议 |
| Table 10 | Manual-vs-virtual phenotype validation | 主文 | 必须 |

## 补充材料建议

<mark>建议加入 Supplementary Tables/Figures：</mark>

- Table S1：中文名、英文名、拉丁名候选、growth type。
- Table S2：20 序列逐样本 ForeSplat 完整指标。
- Table S3：20 序列逐样本外部 baseline 指标。
- Table S4：逐样本表型误差。
- Table S5：分割 benchmark 每帧结果。
- Fig. S1：全部 20 序列 render thumbnails。
- Fig. S2：全部 20 序列 mesh thumbnails。
- Fig. S3：FSAM3 failure cases。
- Fig. S4：external baseline failure cases。

---

## 8. 逐位置修改清单

| 原稿位置 | 当前问题 | 动作 |
|---|---|---|
| 第 9-10 行 | 图表数仍是 8 图 7 表 | <mark>必须更新</mark> 为新版图表数量 |
| 第 38-44 行 | Highlights 缺少 GT 分割、外部 baseline、20 序列重建 | <mark>必须替换</mark> |
| 第 50 行 | 摘要仍以 3 样本 compactness 为主证据，结尾宣传化 | <mark>必须替换</mark> |
| 第 67-72 行 | 贡献点未纳入 20 序列与外部 baseline | <mark>必须替换</mark> |
| 第 140-148 行 | 实验用途写成 3 样本重建验证 | <mark>必须替换</mark> |
| 第 150-186 行 | 缺少人工真值分割 benchmark 协议 | <mark>必须加入</mark> |
| 第 297-309 行 | 缺少 20 序列、外部 baseline、公平性协议 | <mark>必须扩展</mark> |
| 第 311-324 行 | outside 阈值与公式缺失 | <mark>必须补齐</mark> |
| 第 338-344 行 | SEEM 对比占位 | <mark>必须替换为 Table 2 + Fig. 3</mark> |
| 第 352-363 行 | E7 有占位数据 | <mark>必须补齐或改表注</mark> |
| 第 375-387 行 | 只报告 3 样本跨样本验证 | <mark>必须替换为 20 序列 full-dataset 结果</mark> |
| 第 389 行后 | 缺少外部 baseline 结果节 | <mark>必须插入新 4.4</mark> |
| 第 459-476 行 | 表型缺 baseline-level error；20/21 株口径不一 | <mark>必须统一并建议加入 baseline 表型表</mark> |
| 第 492-496 行 | 仍说缺 pixel-level GT | <mark>必须替换</mark> |
| 第 522-526 行 | 未来工作写了已完成事项 | <mark>必须替换</mark> |
| 第 538-546 行 | 局限性仍写“重建样本少”“无外部 baseline” | <mark>必须替换</mark> |
| 第 558-572 行 | 结论未体现 20 序列、SEEM、外部 baseline | <mark>必须替换</mark> |
| 第 578、586、594 行 | URL/作者贡献/资助占位 | <mark>投稿前必须补齐</mark> |

---

## 9. 最稳的论文主张边界

<mark>建议最终主张写成：</mark>

```text
ForeSplat is a foreground-object reformulation of 2DGS for plant-only reconstruction and structural phenotyping. Its main contribution is aligning the RGB supervision domain, Gaussian representation and phenotype measurement target. Full-dataset experiments on 20 multi-view sequences support its robustness under indoor/semi-controlled multi-species potted-plant acquisition, while controlled ablations identify foreground RGB supervision as the decisive mechanism. Segmentation and Gaussian-baseline comparisons further show that the gains are not explained by mask preprocessing or mesh extraction alone.
```

<mark>不要写成：</mark>

```text
ForeSplat is a universal multi-species segmentation and reconstruction solution for all plant phenotyping scenarios.
```

---

## 10. 数据填表前检查清单

在正式改主稿前，先统一以下口径：

- [ ] 20 个序列与 21 株植物的关系：是否有一个序列包含多株，还是表型 n 应改为 20？
- [ ] FSAM3 segmentation benchmark 的人工标注帧数 N、序列数 M、抽样规则。
- [ ] SEEM 的 prompt、输入分辨率、是否后处理。
- [ ] 2DGS-full-scene、3DGS-FSAM3、SuGaR 是否全部覆盖 20 序列。
- [ ] 外部 baseline 是否都有 mesh 和表型测量；如果没有，哪些性状可比。
- [ ] 所有方法是否使用同一 retained frame list、同一 COLMAP poses、同一 evaluation masks。
- [ ] outside_nonblack 阈值的实际数值。
- [ ] Runtime 是训练时间、mesh 时间还是端到端时间，表头必须区分。
- [ ] 统计报告采用 mean ± SD，还是 median [IQR]；全文统一。
- [ ] 表图编号重新生成，交叉引用统一。
