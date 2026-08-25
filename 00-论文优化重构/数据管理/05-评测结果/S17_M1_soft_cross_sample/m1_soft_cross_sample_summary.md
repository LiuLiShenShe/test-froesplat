# S17 M1-soft Cross-sample Summary

## Purpose

S17 extends the S16 `KongQueZhuYu` result to two additional representative samples:

- `XianKeLai1`: thin leaves / fine structures;
- `CaoMei2`: dense leaves / occlusion.

The goal is to test whether M1-soft is a cross-sample positive candidate rather than a single-sample accident.

## Result Table

| Sample | Variant | Eval images | PSNR_fg up | SSIM_fg up | LPIPS_fg down | Outside down | Leakage down | Gaussians down | Full PSNR | Full SSIM | Full LPIPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KongQueZhuYu | A6 | 27 | 25.0072 | 0.8548 | 0.0438 | 0.0294 | 0.0189 | 591623 | 6.2555 | 0.2684 | 0.5567 |
| KongQueZhuYu | A6+M1-soft | 27 | 24.9566 | 0.8543 | 0.0440 | 0.0284 | 0.0184 | 532264 | 6.2539 | 0.2681 | 0.5567 |
| XianKeLai1 | A6 | 26 | 23.7276 | 0.8278 | 0.0309 | 0.0484 | 0.0379 | 253827 | 22.6258 | 0.9287 | 0.0756 |
| XianKeLai1 | A6+M1-soft | 26 | 23.6632 | 0.8274 | 0.0312 | 0.0478 | 0.0374 | 220947 | 22.7146 | 0.9290 | 0.0755 |
| CaoMei2 | A6 | 26 | 25.0833 | 0.8121 | 0.0250 | 0.0147 | 0.0081 | 370844 | 5.1344 | 0.1076 | 0.7066 |
| CaoMei2 | A6+M1-soft | 26 | 25.0046 | 0.8107 | 0.0253 | 0.0140 | 0.0077 | 249944 | 5.1332 | 0.1074 | 0.7067 |

Full-frame metrics are recorded only as run-completeness checks. They are not the performance criterion for foreground-object reconstruction.

## Delta Summary

Across three samples, `A6+M1-soft` compared with A6:

| Metric | Mean / Total delta |
|---|---:|
| mean delta PSNR_fg | -0.0646 dB |
| mean delta SSIM_fg | -0.0008 |
| mean delta LPIPS_fg | +0.0002 |
| mean delta Outside | -0.0007 |
| mean delta Leakage | -0.0005 |
| total Gaussian reduction | -213139 |
| total Gaussian reduction ratio | -17.52% |

Per-sample Gaussian reductions:

| Sample | A6 Gaussians | A6+M1-soft Gaussians | Reduction | Reduction ratio |
|---|---:|---:|---:|---:|
| KongQueZhuYu | 591623 | 532264 | 59359 | 10.03% |
| XianKeLai1 | 253827 | 220947 | 32880 | 12.95% |
| CaoMei2 | 370844 | 249944 | 120900 | 32.60% |

## Interpretation

S17 confirms the S16 mechanism on three representative samples. M1-soft does not reproduce the coverage failure of M1-hard or M1-reject-only retained-list filtering. Instead, it keeps all train/eval views, preserves A6 foreground-object quality within noise-level differences, slightly lowers outside/leakage metrics, and reduces Gaussian count.

This supports the method conclusion:

```text
View quality should regulate gradient contribution rather than remove views from foreground-object 2DGS training.
```

Recommended paper wording:

```text
M1-soft improves model compactness and leakage control while preserving foreground reconstruction quality.
```

Chinese wording:

```text
M1-soft 在基本保持前景重建质量的同时，提高模型紧凑性，并略微降低背景泄漏。
```

Current status:

- `M1-hard` and `M1-reject-only/raw-mask` remain negative evidence for hard view deletion.
- `M1-soft` is now positive cross-sample evidence for the corrected M1 design.
- The next step is P1/P2: expand M4 and test the practical combination `A6+M1-soft+M4`.
