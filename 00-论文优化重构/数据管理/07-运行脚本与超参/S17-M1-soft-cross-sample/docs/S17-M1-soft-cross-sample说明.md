# S17 M1-soft Cross-sample 说明

## 目的

S16 已在 `KongQueZhuYu` 上证明：M1 的合理形式不是 retained-list hard filtering，而是在保留多视角覆盖的前提下进行 soft view weighting。S17 按 P0 方案扩展到代表样本：

- `XianKeLai1`：薄叶/细结构；
- `CaoMei2`：密集叶/遮挡。

目标是验证 M1-soft 不是 `KongQueZhuYu` 单样本特例。

## 当前策略

```text
A6 + M1-soft = A6 foreground-object objective
             + full view coverage
             + H-VQG soft RGB loss weight
```

与 S14/S15 不同，S17 不使用 raw/mask/geo retained-list 删除视角：

- `raw_gate_mode=none`
- `mask_gate_mode=none`
- `geo_gate_mode=none`
- `view_weight_mode=rgb_only`

## Soft Weight 文件

| Sample | Views | Min | Max | Mean | File |
|---|---:|---:|---:|---:|---|
| XianKeLai1 | 203 | 0.793431 | 0.922266 | 0.880234 | `数据管理/05-评测结果/S17_M1_soft_cross_sample/xiankelai1_hvqg_soft_view_weights.csv` |
| CaoMei2 | 203 | 0.797829 | 0.909479 | 0.866690 | `数据管理/05-评测结果/S17_M1_soft_cross_sample/caomei2_hvqg_soft_view_weights.csv` |

注意：`XianKeLai1` 的 sanitized A6 场景使用 `crop_XXXX` 图像名，权重文件已按训练实际图像名输出，避免与 camera image_name 不匹配。

## 执行状态

P0 已完成：

1. `XianKeLai1 A6+M1-soft` 30k 训练、渲染、full metrics 和 foreground-object eval 已完成；
2. `CaoMei2 A6+M1-soft` 30k 训练、渲染、full metrics 和 foreground-object eval 已完成；
3. 三代表样本 `A6` vs `A6+M1-soft` 汇总已落盘。

汇总文件：

```text
数据管理/05-评测结果/S17_M1_soft_cross_sample/m1_soft_cross_sample_summary.md
数据管理/05-评测结果/S17_M1_soft_cross_sample/m1_soft_cross_sample_summary.csv
数据管理/05-评测结果/S17_M1_soft_cross_sample/m1_soft_cross_sample_summary.json
```

## 结果

| Sample | Variant | Eval images | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| KongQueZhuYu | A6 | 27 | 25.0072 | 0.8548 | 0.0438 | 0.0294 | 0.0189 | 591623 |
| KongQueZhuYu | A6+M1-soft | 27 | 24.9566 | 0.8543 | 0.0440 | 0.0284 | 0.0184 | 532264 |
| XianKeLai1 | A6 | 26 | 23.7276 | 0.8278 | 0.0309 | 0.0484 | 0.0379 | 253827 |
| XianKeLai1 | A6+M1-soft | 26 | 23.6632 | 0.8274 | 0.0312 | 0.0478 | 0.0374 | 220947 |
| CaoMei2 | A6 | 26 | 25.0833 | 0.8121 | 0.0250 | 0.0147 | 0.0081 | 370844 |
| CaoMei2 | A6+M1-soft | 26 | 25.0046 | 0.8107 | 0.0253 | 0.0140 | 0.0077 | 249944 |

跨样本平均变化：

| Metric | A6+M1-soft - A6 |
|---|---:|
| PSNR_fg | -0.0646 dB |
| SSIM_fg | -0.0008 |
| LPIPS_fg | +0.0002 |
| Outside | -0.0007 |
| Leakage | -0.0005 |
| total Gaussians | -213139 (-17.52%) |

结论：S17 证明 M1-soft 不是 `KongQueZhuYu` 单样本特例。它在三代表样本上均基本保持 A6 foreground-object quality，同时略降 outside/leakage 并减少 Gaussian 数量，可正式记为 M1 方向修正后的跨样本正证据。

## 后续执行顺序

P1：

- M4 扩样本，优先补 `KongQueZhuYu` 或 `CaoMei2`。

P2：

| Variant | 目的 |
|---|---|
| A6 | core |
| A6+M1-soft | view weighting |
| A6+M4 | cleanup/export |
| A6+M1-soft+M4 | full practical version |

## 判定标准

M1-soft 的论文定位不是“显著提升 PSNR”，而是：

```text
M1-soft improves model compactness and leakage control while preserving foreground reconstruction quality.
```

中文：

```text
M1-soft 在基本保持前景重建质量的同时，提高模型紧凑性，并略微降低背景泄漏。
```

S17 已满足三代表样本上的判定标准：前景质量接近 A6、leakage 不升、Gaussian 数下降。因此 M1-soft 可进入 `Ours-full` 候选；是否成为最终 full method 仍需与 M4 的组合实验共同确认。
