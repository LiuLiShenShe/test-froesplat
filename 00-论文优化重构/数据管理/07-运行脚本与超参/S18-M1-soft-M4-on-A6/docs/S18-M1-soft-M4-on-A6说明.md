# S18 M1-soft + M4 on A6 说明

## 目的

S18 用于验证两个已单独成立的增强模块是否可以叠加：

```text
A6 foreground-object objective
  + M1-soft view weighting
  + M4 mask pruning
```

S17 已证明 M1-soft 在三代表样本上基本保持 A6 前景质量并减少 Gaussian；S13 已证明 M4 在 XianKeLai1 和 CaoMei2 上可作为轻量化/export 增强。S18 首先选择 `CaoMei2`，因为该样本在 S17 中 M1-soft Gaussian 降幅最大，在 S13 中 M4 降幅也最大，适合检验组合是否继续稳定；随后选择 `XianKeLai1`，因为它是薄叶/细结构样本，outside 接近 foreground-only 阈值，更能检验组合版对边界和 mask 稳定性的影响；最后补充 `KongQueZhuYu`，用于形成主样本 full variant 故事闭环。

## 配置

| Sample | Config | Output | M1 weight file | M4 pruning |
|---|---|---|---|---|
| `CaoMei2` | `数据管理/07-运行脚本与超参/S18-M1-soft-M4-on-A6/configs/caomei2_A6_M1_soft_M4.json` | `数据管理/06-实验输出/CaoMei2/A6_M1_soft_M4` | `数据管理/05-评测结果/S17_M1_soft_cross_sample/caomei2_hvqg_soft_view_weights.csv` | mask pruning, start 18000, interval 3000 |
| `XianKeLai1` | `数据管理/07-运行脚本与超参/S18-M1-soft-M4-on-A6/configs/xiankelai1_A6_M1_soft_M4.json` | `数据管理/06-实验输出/XianKeLai1/A6_M1_soft_M4` | `数据管理/05-评测结果/S17_M1_soft_cross_sample/xiankelai1_hvqg_soft_view_weights.csv` | mask pruning, start 18000, interval 3000 |
| `KongQueZhuYu` | `数据管理/07-运行脚本与超参/S18-M1-soft-M4-on-A6/configs/kongquezhu_A6_M1_soft_M4.json` | `数据管理/06-实验输出/KongQueZhuYu/A6_M1_soft_M4` | `数据管理/05-评测结果/S16_M1_soft_weighting_on_A6/kongquezhu_hvqg_soft_view_weights.csv` | mask pruning, start 18000, interval 3000 |

## 执行命令

```bash
/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python \
  00-论文优化重构/数据管理/07-运行脚本与超参/S4-2DGS-baseline回归/scripts/run_2dgs_baseline.py \
  --config 00-论文优化重构/数据管理/07-运行脚本与超参/S18-M1-soft-M4-on-A6/configs/caomei2_A6_M1_soft_M4.json
```

训练完成后运行 foreground-object eval：

```bash
/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python \
  00-论文优化重构/数据管理/07-运行脚本与超参/S12-代表样本A6扩展/scripts/run_foreground_eval.py \
  00-论文优化重构/数据管理/07-运行脚本与超参/S18-M1-soft-M4-on-A6/configs/caomei2_A6_M1_soft_M4.json
```

## 判定标准

`A6+M1-soft+M4` 不要求显著提升 `PSNR_fg`，而是判断是否能同时保持质量和降低模型规模：

| Metric | Expected |
|---|---|
| PSNR_fg / SSIM_fg / LPIPS_fg | 接近 A6、A6+M1-soft、A6+M4 |
| Outside | 不高于 foreground-only 阈值 0.05 |
| Leakage | 不高于 foreground-only 阈值 0.10 |
| Gaussians | 接近或少于 A6+M1-soft / A6+M4 |
| Train/render/eval | 稳定完成 |

若组合结果保持前景质量并继续减少 Gaussian，可将其记为 `Ours-full` 的首个 practical-version 证据。若组合结果不优于单独 M1-soft 或 M4，则说明两个模块存在收益重叠，最终论文可保留 `Ours-core=A6` 与单模块增强表，而不强行合并为唯一主方法。

## 执行结果

`CaoMei2 A6+M1-soft+M4` 已完成 30k 训练、渲染、full-frame metrics 和 foreground-object eval。

| Variant | Eval images | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians |
|---|---:|---:|---:|---:|---:|---:|---:|
| A6 | 26 | 25.0833 | 0.8121 | 0.0250 | 0.0147 | 0.0081 | 370844 |
| A6+M1-soft | 26 | 25.0046 | 0.8107 | 0.0253 | 0.0140 | 0.0077 | 249944 |
| A6+M4 | 26 | 25.0303 | 0.8108 | 0.0251 | 0.0144 | 0.0080 | 284757 |
| A6+M1-soft+M4 | 26 | 24.9718 | 0.8101 | 0.0252 | 0.0136 | 0.0076 | 246452 |

相对 A6，组合版 `PSNR_fg` 下降 `0.1115 dB`、`SSIM_fg` 下降 `0.0020`、`LPIPS_fg` 增加 `0.0002`，仍属于接近 A6 的前景质量范围；同时 `outside` 从 `0.0147` 降到 `0.0136`，`leakage` 从 `0.0081` 降到 `0.0076`，Gaussian 数从 `370844` 降到 `246452`，减少 `124392`，约 `33.54%`。

结论：CaoMei2 是 `Ours-full` practical-version 的首个正证据。M1-soft 与 M4 可以叠加，但 compactness 收益有重叠：M1-soft 已经贡献主要 Gaussian 降幅，M4 在其上继续带来较小的额外减少和更低 leakage/outside。

汇总文件：

```text
数据管理/05-评测结果/S18_M1_soft_M4_on_A6/caomei2_A6_M1_soft_M4_summary.md
数据管理/05-评测结果/S18_M1_soft_M4_on_A6/caomei2_A6_M1_soft_M4_summary.csv
数据管理/05-评测结果/S18_M1_soft_M4_on_A6/caomei2_A6_M1_soft_M4_summary.json
```

## XianKeLai1 跨样本复核结果

`XianKeLai1 A6+M1-soft+M4` 已完成 30k 训练、渲染、full-frame metrics 和 foreground-object eval。

| Variant | Eval images | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians |
|---|---:|---:|---:|---:|---:|---:|---:|
| A6 | 26 | 23.7276 | 0.8278 | 0.0309 | 0.0484 | 0.0379 | 253827 |
| A6+M1-soft | 26 | 23.6632 | 0.8274 | 0.0312 | 0.0478 | 0.0374 | 220947 |
| A6+M4 | 26 | 23.7256 | 0.8279 | 0.0310 | 0.0486 | 0.0376 | 251047 |
| A6+M1-soft+M4 | 26 | 23.7070 | 0.8273 | 0.0312 | 0.0479 | 0.0373 | 219661 |

相对 A6，组合版 `PSNR_fg` 仅下降 `0.0206 dB`，`SSIM_fg` 下降 `0.0005`，`LPIPS_fg` 增加 `0.0003`；同时 `outside` 从 `0.0484` 降至 `0.0479`，仍低于 `0.05` 阈值，`leakage` 从 `0.0379` 降至 `0.0373`，Gaussian 数减少 `34166`，约 `13.46%`。

## KongQueZhuYu 主样本闭环结果

`KongQueZhuYu A6+M1-soft+M4` 已完成 30k 训练、渲染、full-frame metrics 和 foreground-object eval。

| Variant | Eval images | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians |
|---|---:|---:|---:|---:|---:|---:|---:|
| A6 | 27 | 25.0072 | 0.8548 | 0.0438 | 0.0294 | 0.0189 | 591623 |
| A6+M1-soft | 27 | 24.9566 | 0.8543 | 0.0440 | 0.0284 | 0.0184 | 532264 |
| A6+M1-soft+M4 | 27 | 24.9423 | 0.8540 | 0.0441 | 0.0284 | 0.0182 | 530936 |

相对 A6，组合版 `PSNR_fg` 下降 `0.0649 dB`，`SSIM_fg` 下降 `0.0008`，`LPIPS_fg` 增加 `0.0003`；同时 `outside` 从 `0.0294` 降至 `0.0284`，`leakage` 从 `0.0189` 降至 `0.0182`，Gaussian 数减少 `60687`，约 `10.26%`。该结果完成了主样本上的 hard filtering 负证据、soft weighting 正证据和 compact full variant 的闭环。

## 跨样本结论

`CaoMei2`、`XianKeLai1` 与 `KongQueZhuYu` 三个组合实验合计，`A6+M1-soft+M4` 相对 A6 的平均变化为：`PSNR_fg -0.0657 dB`、`SSIM_fg -0.0011`、`LPIPS_fg +0.0003`、`outside -0.0009`、`leakage -0.0006`；Gaussian 总数从 `1216294` 降至 `997049`，减少 `219245`，约 `18.03%`。

当前状态：S18 已从单样本首条正证据升级为三样本正证据，并完成主样本闭环。论文中可正式采用：

```text
Ours-core = A6
Ours-full / Ours-compact = A6 + M1-soft + M4
```

需要注意，`Ours-full` 的准确主张不是显著提升 PSNR/SSIM/LPIPS，而是在小幅前景质量变化内进一步降低背景泄漏和 Gaussian 数量，得到更紧凑的 foreground-object representation。
