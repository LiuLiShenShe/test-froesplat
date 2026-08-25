# KongQueZhuYu A6 vs M1 Hard/Reject-only Summary

## Runs

| Variant | Output dir |
|---|---|
| A6 | `00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A6_foreground_track_init_fg_rgb_alpha_bg` |
| A6+M1-hard | `00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A6_hvqg_smoke_v2` |
| A6+M1-reject-only/raw-mask | `00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A6_M1_reject_only_raw_mask` |

## Retained Views

| Variant | Raw gate | Mask gate | Geometry gate | Eval images |
|---|---:|---:|---:|---:|
| A6 | none | none | none | 27 |
| A6+M1-hard | 163/183 train, 24/27 test | 163/163 train, 24/24 test | 123/163 train, 17/24 test | 17 |
| A6+M1-reject-only/raw-mask | 163/183 train, 24/27 test | 163/163 train, 24/24 test | none | 24 |

## Metrics

| Variant | PSNR_fg ↑ | SSIM_fg ↑ | LPIPS_fg ↓ | Outside ↓ | Leakage ↓ | Gaussians | Full PSNR | Full SSIM | Full LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A6 | 25.0072 | 0.8548 | 0.0438 | 0.0294 | 0.0189 | 591623 | 6.2555 | 0.2684 | 0.5567 |
| A6+M1-hard | 12.5478 | 0.6018 | 0.1179 | 0.1743 | 0.3020 | 597116 | 6.3174 | 0.2757 | 0.5578 |
| A6+M1-reject-only/raw-mask | 13.4557 | 0.6244 | 0.1115 | 0.1450 | 0.2848 | 579612 | 6.2292 | 0.2668 | 0.5594 |

## Delta vs A6

| Variant | Delta PSNR_fg | Delta SSIM_fg | Delta LPIPS_fg | Delta Outside | Delta Leakage | Delta Gaussians |
|---|---:|---:|---:|---:|---:|---:|
| A6+M1-hard - A6 | -12.4594 | -0.2530 | +0.0740 | +0.1448 | +0.2831 | +5493 |
| A6+M1-reject-only/raw-mask - A6 | -11.5516 | -0.2304 | +0.0677 | +0.1156 | +0.2659 | -12011 |

## Interpretation

`A6+M1-reject-only/raw-mask` improves slightly over `A6+M1-hard`, especially by keeping 24 test views instead of 17 and reducing leakage from 0.3020 to 0.2848. However, it remains far below A6 and still exceeds the foreground-only leakage threshold.

This means the S14 failure is not caused only by the geometry hard delete. The existing raw/mask retained list still behaves as hard filtering and likely removes views that are important for reconstruction coverage or disrupts the training/evaluation view distribution.

## Formal Conclusion

| Variant | Role | Conclusion |
|---|---|---|
| A6 | main baseline | foreground-object reconstruction is valid |
| A6+M1-hard | strong negative control | hard filtering severely breaks reconstruction |
| A6+M1-reject-only/raw-mask | weak negative control | keeping more views is slightly better than M1-hard, but still fails clearly |

Foreground-only thresholds:

```text
outside < 0.05
leakage < 0.10
```

A6 passes these thresholds (`outside=0.0294`, `leakage=0.0189`). Both retained-list variants fail them (`M1-hard leakage=0.3020`, `M1-reject-only leakage=0.2848`). Therefore the result should be recorded as negative evidence against the current M1 retained-list mechanism, not as a successful M1 module.

Current conclusion:

- `M1-hard` remains a strong negative control.
- `M1-reject-only/raw-mask` is also negative or at most weakly informative; it is not an effective M1.
- The next M1 should not continue with retained-list hard filtering.
- The next useful direction is `soft view weighting` or a true coverage-balanced selection that preserves anchor views in every trajectory/angle bin.

Paper wording:

> These negative results indicate that view quality control for foreground-object 2DGS should not be implemented as hard retained-list filtering. Instead, it should preserve multi-view coverage and regulate view contribution through coverage-balanced selection or soft weighting.
