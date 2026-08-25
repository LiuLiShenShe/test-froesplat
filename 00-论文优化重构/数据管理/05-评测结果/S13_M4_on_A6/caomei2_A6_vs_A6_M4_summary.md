# CaoMei2 A6 vs A6+M4 summary

## Runs

| variant | output_dir |
|---|---|
| A6 | `00-论文优化重构/数据管理/06-实验输出/CaoMei2/A6_foreground_track_init_fg_rgb_alpha_bg` |
| A6+M4 | `00-论文优化重构/数据管理/06-实验输出/CaoMei2/A6_M4_mask_pruning` |

## Metrics

| variant | gaussians_30000 | PSNR | SSIM | LPIPS | PSNR_fg | SSIM_fg | LPIPS_fg_black_bg | leakage_energy_ratio_mean | outside_nonblack_ratio_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A6 | 370844 | 5.1344 | 0.1076 | 0.7066 | 25.0833 | 0.8121 | 0.0250 | 0.0081 | 0.0147 |
| A6+M4 | 284757 | 5.1341 | 0.1074 | 0.7070 | 25.0303 | 0.8108 | 0.0251 | 0.0080 | 0.0144 |
| Delta A6+M4 - A6 | -86087 | -0.0003 | -0.0002 | +0.0004 | -0.0529 | -0.0013 | +0.0002 | -0.0001 | -0.0003 |

Gaussian reduction: `86087 / 370844 = 23.21%`.

## Pruning Reports

| iteration | gaussians_before | gaussians_after | removed | pruning_ratio |
|---:|---:|---:|---:|---:|
| 18000 | 286806 | 285504 | 1302 | 0.0045 |
| 21000 | 285504 | 285165 | 339 | 0.0012 |
| 24000 | 285165 | 284963 | 202 | 0.0007 |
| 27000 | 284963 | 284834 | 129 | 0.0005 |
| 30000 | 284834 | 284757 | 77 | 0.0003 |

## Takeaway

On CaoMei2, adding M4 mask pruning to A6 removes 86,087 Gaussians at 30k, about 23.21% of the A6 model size. Foreground quality remains close to A6: `PSNR_fg` changes by `-0.0529 dB`, `SSIM_fg` by `-0.0013`, and `LPIPS_fg_black_bg` by `+0.0002`. Leakage and outside nonblack are slightly lower, so this result supports M4 as a practical compactness/export enhancement rather than a reconstruction-quality booster.
