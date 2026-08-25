# KongQueZhuYu A6 vs A6+M1/H-VQG summary

## Runs

| variant | output_dir |
|---|---|
| A6 | `00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A6_foreground_track_init_fg_rgb_alpha_bg` |
| A6+M1/H-VQG | `00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A6_hvqg_smoke_v2` |

## M1 Retained Views

Existing retained lists were reused from:

```text
00-论文优化重构/数据管理/05-评测结果/KongQueZhuYu/M1_hvqg_smoke_v2/
```

During training/render loading:

| gate | train retained | test retained |
|---|---:|---:|
| raw gate | 163/183 | 24/27 |
| mask gate | 163/163 | 24/24 |
| geometry gate | 123/163 | 17/24 |

## Metrics

| variant | num_eval_images | gaussians_30000 | PSNR | SSIM | LPIPS | PSNR_fg | SSIM_fg | LPIPS_fg_black_bg | leakage_energy_ratio_mean | outside_nonblack_ratio_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A6 | 27 | 591623 | 6.2555 | 0.2684 | 0.5567 | 25.0072 | 0.8548 | 0.0438 | 0.0189 | 0.0294 |
| A6+M1/H-VQG | 17 | 597116 | 6.3174 | 0.2757 | 0.5578 | 12.5478 | 0.6018 | 0.1179 | 0.3020 | 0.1743 |
| Delta A6+M1 - A6 | -10 | +5493 | +0.0619 | +0.0074 | +0.0011 | -12.4594 | -0.2530 | +0.0740 | +0.2831 | +0.1448 |

## Takeaway

This reused M1/H-VQG retained list does **not** improve A6 foreground-object reconstruction. It removes too many geometry/test views and breaks the foreground-object objective on the remaining evaluation set. This is useful negative evidence: M1 cannot be framed as hard top-score filtering only; it must be redesigned as coverage-balanced reconstruction-effective view selection.

Recommended next M1 revision:

- keep raw/mask gates conservative;
- weaken geometry hard filtering;
- add coverage-balanced selection by trajectory/image-order buckets;
- evaluate on the full comparable test set or explicitly report retained-test scope;
- compare filtering against view-weighted training, not only hard deletion.
