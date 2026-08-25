# S18 XianKeLai1 A6+M1-soft+M4 summary

## Runs

| variant | output_dir |
|---|---|
| A6 | `00-论文优化重构/数据管理/06-实验输出/XianKeLai1/A6_foreground_track_init_fg_rgb_alpha_bg` |
| A6+M1-soft | `00-论文优化重构/数据管理/06-实验输出/XianKeLai1/A6_M1_soft_weighting` |
| A6+M4 | `00-论文优化重构/数据管理/06-实验输出/XianKeLai1/A6_M4_mask_pruning_20260519_151007` |
| A6+M1-soft+M4 | `00-论文优化重构/数据管理/06-实验输出/XianKeLai1/A6_M1_soft_M4` |

## Metrics

| Variant | Eval images | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians |
|---|---:|---:|---:|---:|---:|---:|---:|
| A6 | 26 | 23.7276 | 0.8278 | 0.0309 | 0.0484 | 0.0379 | 253827 |
| A6+M1-soft | 26 | 23.6632 | 0.8274 | 0.0312 | 0.0478 | 0.0374 | 220947 |
| A6+M4 | 26 | 23.7256 | 0.8279 | 0.0310 | 0.0486 | 0.0376 | 251047 |
| A6+M1-soft+M4 | 26 | 23.7070 | 0.8273 | 0.0312 | 0.0479 | 0.0373 | 219661 |

Full-frame metrics for `A6+M1-soft+M4` are `PSNR=22.7524`, `SSIM=0.9291`, `LPIPS=0.0757`. They are recorded only as pipeline completeness checks; foreground-object metrics remain the formal criterion.

## Deltas

| Comparison | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians |
|---|---:|---:|---:|---:|---:|---:|
| Combo - A6 | -0.0206 | -0.0005 | +0.0003 | -0.0004 | -0.0006 | -34166 (-13.46%) |
| Combo - A6+M1-soft | +0.0438 | -0.0001 | +0.0001 | +0.0001 | -0.0001 | -1286 (-0.58%) |
| Combo - A6+M4 | -0.0186 | -0.0006 | +0.0002 | -0.0007 | -0.0003 | -31386 (-12.50%) |

## Pruning Reports

| iteration | gaussians_before | gaussians_after | removed | pruning_ratio |
|---:|---:|---:|---:|---:|
| 18000 | 221842 | 220657 | 1185 | 0.0053 |
| 21000 | 220657 | 220198 | 459 | 0.0021 |
| 24000 | 220198 | 219953 | 245 | 0.0011 |
| 27000 | 219953 | 219774 | 179 | 0.0008 |
| 30000 | 219774 | 219661 | 113 | 0.0005 |

## Takeaway

On XianKeLai1, `A6+M1-soft+M4` passes the foreground-only thresholds and remains very close to A6. Compared with A6, `PSNR_fg` decreases by only `0.0206 dB`, `SSIM_fg` decreases by `0.0005`, and `LPIPS_fg` increases by `0.0003`; meanwhile `outside` decreases from `0.0484` to `0.0479`, `leakage` decreases from `0.0379` to `0.0373`, and Gaussian count drops by `34166` (`13.46%`).

This is an important thin-leaf / fine-structure validation: the combined practical variant does not push the sample over the `outside < 0.05` boundary and still provides compactness.
