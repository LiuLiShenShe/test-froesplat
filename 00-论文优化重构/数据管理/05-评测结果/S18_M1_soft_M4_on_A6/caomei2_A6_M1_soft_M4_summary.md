# S18 CaoMei2 A6+M1-soft+M4 summary

## Runs

| variant | output_dir |
|---|---|
| A6 | `00-论文优化重构/数据管理/06-实验输出/CaoMei2/A6_foreground_track_init_fg_rgb_alpha_bg` |
| A6+M1-soft | `00-论文优化重构/数据管理/06-实验输出/CaoMei2/A6_M1_soft_weighting` |
| A6+M4 | `00-论文优化重构/数据管理/06-实验输出/CaoMei2/A6_M4_mask_pruning` |
| A6+M1-soft+M4 | `00-论文优化重构/数据管理/06-实验输出/CaoMei2/A6_M1_soft_M4` |

## Metrics

| Variant | Eval images | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians |
|---|---:|---:|---:|---:|---:|---:|---:|
| A6 | 26 | 25.0833 | 0.8121 | 0.0250 | 0.0147 | 0.0081 | 370844 |
| A6+M1-soft | 26 | 25.0046 | 0.8107 | 0.0253 | 0.0140 | 0.0077 | 249944 |
| A6+M4 | 26 | 25.0303 | 0.8108 | 0.0251 | 0.0144 | 0.0080 | 284757 |
| A6+M1-soft+M4 | 26 | 24.9718 | 0.8101 | 0.0252 | 0.0136 | 0.0076 | 246452 |

Full-frame metrics for `A6+M1-soft+M4` are `PSNR=5.1327`, `SSIM=0.1073`, `LPIPS=0.7070`. They are recorded only as pipeline completeness checks; foreground-object metrics remain the formal criterion.

## Deltas

| Comparison | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians |
|---|---:|---:|---:|---:|---:|---:|
| Combo - A6 | -0.1115 | -0.0020 | +0.0002 | -0.0010 | -0.0005 | -124392 (-33.54%) |
| Combo - A6+M1-soft | -0.0328 | -0.0006 | -0.0001 | -0.0004 | -0.0001 | -3492 (-1.40%) |
| Combo - A6+M4 | -0.0585 | -0.0007 | +0.0001 | -0.0007 | -0.0004 | -38305 (-13.45%) |

## Pruning Reports

| iteration | gaussians_before | gaussians_after | removed | pruning_ratio |
|---:|---:|---:|---:|---:|
| 18000 | 248026 | 247043 | 983 | 0.0040 |
| 21000 | 247043 | 246757 | 286 | 0.0012 |
| 24000 | 246757 | 246614 | 143 | 0.0006 |
| 27000 | 246614 | 246517 | 97 | 0.0004 |
| 30000 | 246517 | 246452 | 65 | 0.0003 |

## Takeaway

On CaoMei2, `A6+M1-soft+M4` is a valid practical full-version candidate. It gives the smallest model and the lowest outside/leakage among the tested A6 variants, while keeping foreground-object quality close to A6. The trade-off is small: compared with A6, `PSNR_fg` decreases by `0.1115 dB`, `SSIM_fg` by `0.0020`, and `LPIPS_fg` increases by `0.0002`, while Gaussian count drops by `124392` (`33.54%`).

This suggests that M1-soft and M4 are compatible, but their compactness gains overlap: most of the model-size reduction already comes from M1-soft, and M4 adds a smaller extra reduction on top.
