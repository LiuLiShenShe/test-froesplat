# XianKeLai1 A6 vs A6+M4 summary

## Runs

| variant | output_dir |
|---|---|
| A6 | `00-论文优化重构/数据管理/06-实验输出/XianKeLai1/A6_foreground_track_init_fg_rgb_alpha_bg` |
| A6+M4 | `00-论文优化重构/数据管理/06-实验输出/XianKeLai1/A6_M4_mask_pruning_20260519_151007` |

## Metrics

| variant | gaussians_30000 | PSNR | SSIM | LPIPS | PSNR_fg | SSIM_fg | LPIPS_fg_black_bg | leakage_energy_ratio_mean | outside_nonblack_ratio_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A6 | 253827 | 22.6258 | 0.9287 | 0.0756 | 23.7276 | 0.8278 | 0.0309 | 0.0379 | 0.0484 |
| A6+M4 | 251047 | 22.6946 | 0.9288 | 0.0761 | 23.7256 | 0.8279 | 0.0310 | 0.0376 | 0.0486 |
| Delta A6+M4 - A6 | -2780 | +0.0687 | +0.0001 | +0.0005 | -0.0020 | +0.0001 | +0.0001 | -0.0003 | +0.0002 |

## Pruning Reports

| iteration | gaussians_before | gaussians_after | removed | pruning_ratio |
|---:|---:|---:|---:|---:|
| 18000 | 253581 | 252197 | 1384 | 0.0055 |
| 21000 | 252197 | 251664 | 533 | 0.0021 |
| 24000 | 251664 | 251352 | 312 | 0.0012 |
| 27000 | 251352 | 251169 | 183 | 0.0007 |
| 30000 | 251169 | 251047 | 122 | 0.0005 |

## Takeaway

On XianKeLai1, adding M4 mask pruning to A6 removes 2,780 Gaussians at 30k while keeping foreground-object reconstruction essentially unchanged. Leakage energy is slightly lower, while foreground PSNR/SSIM/LPIPS remain stable within noise-level differences.
