# S13 M4 on A6 运行说明

## 目标

在已经完成的 A6 foreground-object reconstruction 上补 M4 pruning/export 证据。M4 不替代 A6 的 foreground objective，而是作为系统增强：减少 Gaussian 数、降低背景泄漏，并服务后续 plant-only export 和 mesh。

2026-05-21 更新：`XianKeLai1 A6+M4` 与 `CaoMei2 A6+M4` 均已完成并形成稳定证据。S13 现在可作为 P1 M4 扩样本结果，并为 S18 `CaoMei2 A6+M1-soft+M4` 组合矩阵提供参照。

## 当前证据

`KongQueZhuYu/F1_high_precision_foreground` 已经是 A6 + lightweight mask pruning 的成功结果，可作为第一条 A6+M4 证据：

```text
数据管理/06-实验输出/KongQueZhuYu/F1_high_precision_foreground/
```

对照纯 A6：

```text
数据管理/06-实验输出/KongQueZhuYu/A6_foreground_track_init_fg_rgb_alpha_bg/
```

## 本轮新增

先跑 `XianKeLai1/A6_M4_mask_pruning`，保持 A6 训练目标不变，只打开 M4 mask pruning：

```bash
/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python \
  00-论文优化重构/数据管理/07-运行脚本与超参/S4-2DGS-baseline回归/scripts/run_2dgs_baseline.py \
  --config 00-论文优化重构/数据管理/07-运行脚本与超参/S13-M4-on-A6/configs/xiankelai1_A6_M4_mask_pruning.json
```

P1 扩样本新增 `CaoMei2/A6_M4_mask_pruning`：

```bash
/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python \
  00-论文优化重构/数据管理/07-运行脚本与超参/S4-2DGS-baseline回归/scripts/run_2dgs_baseline.py \
  --config 00-论文优化重构/数据管理/07-运行脚本与超参/S13-M4-on-A6/configs/caomei2_A6_M4_mask_pruning.json
```

训练完成后运行 foreground-object eval：

```bash
/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python \
  00-论文优化重构/数据管理/07-运行脚本与超参/S12-代表样本A6扩展/scripts/run_foreground_eval.py \
  00-论文优化重构/数据管理/07-运行脚本与超参/S13-M4-on-A6/configs/xiankelai1_A6_M4_mask_pruning.json
```

`CaoMei2` 对应命令：

```bash
/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python \
  00-论文优化重构/数据管理/07-运行脚本与超参/S12-代表样本A6扩展/scripts/run_foreground_eval.py \
  00-论文优化重构/数据管理/07-运行脚本与超参/S13-M4-on-A6/configs/caomei2_A6_M4_mask_pruning.json
```

## 判断标准

- `PSNR_fg` / `SSIM_fg` / `LPIPS_fg_black_bg` 不明显恶化。
- `gaussians_30000` 下降，或 `outside_nonblack_ratio_mean` / `leakage_energy_ratio_mean` 下降。
- pruning report 存在：`pruning/pruning_iter_*.json`。

## 汇总结果

| Sample | Variant | Eval images | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| XianKeLai1 | A6 | 26 | 23.7276 | 0.8278 | 0.0309 | 0.0484 | 0.0379 | 253827 |
| XianKeLai1 | A6+M4 | 26 | 23.7256 | 0.8279 | 0.0310 | 0.0486 | 0.0376 | 251047 |
| CaoMei2 | A6 | 26 | 25.0833 | 0.8121 | 0.0250 | 0.0147 | 0.0081 | 370844 |
| CaoMei2 | A6+M4 | 26 | 25.0303 | 0.8108 | 0.0251 | 0.0144 | 0.0080 | 284757 |

跨样本总 Gaussian 数从 `624671` 降至 `535804`，减少 `88867`，约 `14.23%`。平均 foreground-object 变化为：`PSNR_fg=-0.0275 dB`，`SSIM_fg=-0.0006`，`LPIPS_fg=+0.0001`，`leakage=-0.0002`。因此 M4 当前定位应写成：在基本保持前景质量的同时减少模型规模，并略微降低 leakage。

汇总文件：

```text
数据管理/05-评测结果/S13_M4_on_A6/m4_cross_sample_summary.md
数据管理/05-评测结果/S13_M4_on_A6/m4_cross_sample_summary.csv
数据管理/05-评测结果/S13_M4_on_A6/m4_cross_sample_summary.json
```

## 后续组合

- `CaoMei2 A6+M1-soft+M4`，用于验证 view weighting 与 pruning/export 是否能叠加。
