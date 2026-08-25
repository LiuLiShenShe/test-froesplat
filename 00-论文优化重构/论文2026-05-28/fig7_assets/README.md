# Fig. 7 Assets

这个目录收集图 7（视角质量策略与表示紧凑性）所需的作图数据。

## Panel a: View-coverage strategies

使用：

`fig7_panel_a_view_coverage.csv`

字段含义：

| 字段 | 含义 |
|---|---|
| `view_order` | KongQueZhuYu 测试视角顺序 |
| `view_id` | 原始视角编号 |
| `angle_deg` | 用于画环形示意的等间隔角度 |
| `hard_drop_image_quality_retained` | hard IQ 策略是否保留该测试视角 |
| `hard_drop_mask_quality_retained` | hard mask 策略是否保留该测试视角 |
| `soft_weighting_retained` | soft weighting 是否保留该测试视角 |
| `soft_weight` | soft RGB loss 权重 |
| `soft_weight_band` | 便于上色的 high/mid/low 分组 |

对应数量：

| 策略 | 保留视角 |
|---|---:|
| Hard drop (image/IQ quality) | 17 / 27 |
| Hard drop (mask quality) | 24 / 27 |
| Soft weighting | 27 / 27 |

## Panel b: Comparison of view-quality strategies

使用：

`fig7_panel_b_view_quality_strategies.csv`

包含 `Full config`、`Hard drop (IQ)`、`Hard drop (Mask)`、`Soft weight` 四个策略的：

- `PSNR_fg`
- `SSIM_fg`
- `LPIPS_fg_black_bg`
- `outside_nonblack_ratio_mean`
- `leakage_energy_ratio_mean`
- `gaussians_30000`
- `gaussians_1e5`

## Panel c: Quality-compactness trade-off

使用：

`fig7_panel_c_quality_compactness_tradeoff.csv`

包含 `CaoMei2`、`XianKeLai1`、`KongQueZhuYu` 的 A6 / A6+M1-soft / A6+M4 / A6+M1-soft+M4 指标。

注意：当前 S18 汇总里 `KongQueZhuYu` 没有单独的 `A6+M4` 行，所以 panel c 若要四个点完全对称，需要补跑或改为三点轨迹。

## Panel d: Effect of compact configuration

使用：

- `fig7_panel_d_compact_before_after.csv`
- `fig7_panel_d_compact_per_sample_delta.csv`

汇总结论：

| 指标 | Before | After |
|---|---:|---:|
| Total Gaussians | 1,216,294 | 997,049 |
| Total Gaussians (10^5) | 12.16294 | 9.97049 |
| Change | 0 | -18.03% |

平均变化：

| 指标 | After - Before |
|---|---:|
| Avg. PSNR_fg | -0.0657 dB |
| Avg. SSIM_fg | -0.0011 |
| Avg. LPIPS_fg | +0.0003 |

## 可用于 panel d 右侧渲染对比的图片

Before:

`/data/fj/F2DMAS/00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A6_foreground_track_init_fg_rgb_alpha_bg/test/ours_30000/renders/00000.png`

After:

`/data/fj/F2DMAS/00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A6_M1_soft_M4/test/ours_30000/renders/00000.png`

## 原始数据源

| 用途 | 源文件 |
|---|---|
| A6 跨样本基线 | `00-论文优化重构/数据管理/05-评测结果/S12_representative_A6_extension/representative_A6_summary.csv` |
| hard IQ / hard mask | `00-论文优化重构/数据管理/05-评测结果/S15_M1_coverage_aware_on_A6/kongquezhu_A6_M1_reject_only_summary.csv` |
| soft weighting | `00-论文优化重构/数据管理/05-评测结果/S16_M1_soft_weighting_on_A6/kongquezhu_A6_M1_soft_weighting_summary.md` |
| cross-sample soft weighting | `00-论文优化重构/数据管理/05-评测结果/S17_M1_soft_cross_sample/m1_soft_cross_sample_summary.csv` |
| compact combination | `00-论文优化重构/数据管理/05-评测结果/S18_M1_soft_M4_on_A6/s18_combo_cross_sample_summary.csv` |

