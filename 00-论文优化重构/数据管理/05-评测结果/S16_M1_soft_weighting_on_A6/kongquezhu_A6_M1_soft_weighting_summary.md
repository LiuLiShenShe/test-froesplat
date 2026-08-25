# KongQueZhuYu A6+M1-soft Weighting Summary

## Run

| Variant | Output dir |
|---|---|
| A6+M1-soft weighting | `00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A6_M1_soft_weighting_20260521_110854` |

## Method

```text
A6 + M1-soft = A6 foreground-object objective
             + full view coverage
             + per-view RGB loss weight
```

This run does not delete train or eval views. It uses the H-VQG score only as a soft RGB reconstruction loss weight.

## View Weights

| Item | Value |
|---|---:|
| weighted views | 210 |
| min weight | 0.846018 |
| max weight | 0.941009 |
| mean weight | 0.892151 |
| weight mode | `rgb_only` |

Weight file:

```text
00-论文优化重构/数据管理/05-评测结果/S16_M1_soft_weighting_on_A6/kongquezhu_hvqg_soft_view_weights.csv
```

## Metrics

| Variant | Eval images | PSNR_fg up | SSIM_fg up | LPIPS_fg down | Outside down | Leakage down | Gaussians | Full PSNR | Full SSIM | Full LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A6 | 27 | 25.0072 | 0.8548 | 0.0438 | 0.0294 | 0.0189 | 591623 | 6.2555 | 0.2684 | 0.5567 |
| A6+M1-soft weighting | 27 | 24.9566 | 0.8543 | 0.0440 | 0.0284 | 0.0184 | 532264 | 6.2539 | 0.2681 | 0.5567 |

## Delta vs A6

| Variant | Delta PSNR_fg | Delta SSIM_fg | Delta LPIPS_fg | Delta Outside | Delta Leakage | Delta Gaussians |
|---|---:|---:|---:|---:|---:|---:|
| A6+M1-soft weighting - A6 | -0.0506 | -0.0005 | +0.0002 | -0.0010 | -0.0005 | -59359 |

## Foreground-only Thresholds

```text
outside < 0.05
leakage < 0.10
```

`A6+M1-soft weighting` passes the foreground-only thresholds:

- `outside_nonblack_ratio_mean=0.0284`
- `leakage_energy_ratio_mean=0.0184`

## Interpretation

S16 shows that soft view weighting preserves multi-view coverage and avoids the retained-list hard filtering failure observed in S14/S15. Foreground quality is effectively tied with A6, leakage is slightly lower, and the Gaussian count is reduced by 59,359, about 10.03%.

Paper wording:

```text
M1-soft improves model compactness and leakage control while preserving foreground reconstruction quality.
```

This supports the updated M1 conclusion:

- retained-list hard filtering is negative evidence;
- view quality should not directly delete views;
- M1 is more suitable as soft gradient contribution control, or as a future coverage-balanced anchor-view strategy.

Full-frame metrics are recorded only as a run-completeness check. They are not used to judge foreground-object reconstruction quality.
