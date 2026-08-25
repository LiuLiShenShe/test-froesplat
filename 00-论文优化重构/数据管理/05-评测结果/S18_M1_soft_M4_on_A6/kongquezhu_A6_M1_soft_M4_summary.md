# S18 KongQueZhuYu A6+M1-soft+M4 summary

## Runs

| variant | output_dir |
|---|---|
| A6 | `00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A6_foreground_track_init_fg_rgb_alpha_bg` |
| A6+M1-soft | `00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A6_M1_soft_weighting_20260521_110854` |
| A6+M1-soft+M4 | `00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A6_M1_soft_M4` |

## Metrics

| Variant | Eval images | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians |
|---|---:|---:|---:|---:|---:|---:|---:|
| A6 | 27 | 25.0072 | 0.8548 | 0.0438 | 0.0294 | 0.0189 | 591623 |
| A6+M1-soft | 27 | 24.9566 | 0.8543 | 0.0440 | 0.0284 | 0.0184 | 532264 |
| A6+M1-soft+M4 | 27 | 24.9423 | 0.8540 | 0.0441 | 0.0284 | 0.0182 | 530936 |

Full-frame metrics for `A6+M1-soft+M4` are `PSNR=6.2533`, `SSIM=0.2679`, `LPIPS=0.5564`. They are recorded only as pipeline completeness checks; foreground-object metrics remain the formal criterion.

## Deltas

| Comparison | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians |
|---|---:|---:|---:|---:|---:|---:|
| Combo - A6 | -0.0649 | -0.0008 | +0.0003 | -0.0010 | -0.0007 | -60687 (-10.26%) |
| Combo - A6+M1-soft | -0.0143 | -0.0003 | +0.0001 | -0.0000 | -0.0002 | -1328 (-0.25%) |

## Pruning Reports

| iteration | gaussians_before | gaussians_after | removed | pruning_ratio |
|---:|---:|---:|---:|---:|
| 18000 | 533247 | 531691 | 1556 | 0.0029 |
| 21000 | 531691 | 531311 | 380 | 0.0007 |
| 24000 | 531311 | 531120 | 191 | 0.0004 |
| 27000 | 531120 | 531010 | 110 | 0.0002 |
| 30000 | 531010 | 530936 | 74 | 0.0001 |

## Takeaway

On `KongQueZhuYu`, `A6+M1-soft+M4` completes the main-sample closure for the practical full variant. Compared with A6, foreground quality remains very close: `PSNR_fg` decreases by `0.0649 dB`, `SSIM_fg` by `0.0008`, and `LPIPS_fg` increases by `0.0003`. At the same time, `outside` decreases from `0.0294` to `0.0284`, `leakage` decreases from `0.0189` to `0.0182`, and Gaussian count drops by `60687` (`10.26%`).

Compared with `A6+M1-soft`, M4 adds only a small extra cleanup: Gaussian count drops by another `1328` (`0.25%`) and leakage decreases slightly. This confirms the same pattern seen in CaoMei2 and XianKeLai1: M1-soft provides most of the compactness gain, while M4 acts as a safe export/cleanup step.
