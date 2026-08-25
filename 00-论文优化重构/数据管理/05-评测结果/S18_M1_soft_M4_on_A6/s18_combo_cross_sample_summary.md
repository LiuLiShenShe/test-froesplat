# S18 A6+M1-soft+M4 cross-sample summary

## Purpose

S18 tests whether the current practical full variant can combine:

```text
A6 foreground-object reconstruction
  + M1-soft view weighting
  + M4 mask pruning
```

After CaoMei2 provided the first positive combination evidence and XianKeLai1 validated the thin-leaf / fine-structure case, KongQueZhuYu was added for main-sample closure.

## Metrics

| Sample | Variant | Eval images | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CaoMei2 | A6 | 26 | 25.0833 | 0.8121 | 0.0250 | 0.0147 | 0.0081 | 370844 |
| CaoMei2 | A6+M1-soft | 26 | 25.0046 | 0.8107 | 0.0253 | 0.0140 | 0.0077 | 249944 |
| CaoMei2 | A6+M4 | 26 | 25.0303 | 0.8108 | 0.0251 | 0.0144 | 0.0080 | 284757 |
| CaoMei2 | A6+M1-soft+M4 | 26 | 24.9718 | 0.8101 | 0.0252 | 0.0136 | 0.0076 | 246452 |
| XianKeLai1 | A6 | 26 | 23.7276 | 0.8278 | 0.0309 | 0.0484 | 0.0379 | 253827 |
| XianKeLai1 | A6+M1-soft | 26 | 23.6632 | 0.8274 | 0.0312 | 0.0478 | 0.0374 | 220947 |
| XianKeLai1 | A6+M4 | 26 | 23.7256 | 0.8279 | 0.0310 | 0.0486 | 0.0376 | 251047 |
| XianKeLai1 | A6+M1-soft+M4 | 26 | 23.7070 | 0.8273 | 0.0312 | 0.0479 | 0.0373 | 219661 |
| KongQueZhuYu | A6 | 27 | 25.0072 | 0.8548 | 0.0438 | 0.0294 | 0.0189 | 591623 |
| KongQueZhuYu | A6+M1-soft | 27 | 24.9566 | 0.8543 | 0.0440 | 0.0284 | 0.0184 | 532264 |
| KongQueZhuYu | A6+M1-soft+M4 | 27 | 24.9423 | 0.8540 | 0.0441 | 0.0284 | 0.0182 | 530936 |

## Combo vs A6

| Sample | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians |
|---|---:|---:|---:|---:|---:|---:|
| CaoMei2 | -0.1115 | -0.0020 | +0.0002 | -0.0011 | -0.0005 | -124392 (-33.54%) |
| XianKeLai1 | -0.0206 | -0.0005 | +0.0003 | -0.0005 | -0.0006 | -34166 (-13.46%) |
| KongQueZhuYu | -0.0649 | -0.0008 | +0.0003 | -0.0010 | -0.0007 | -60687 (-10.26%) |

Across these three samples, the combination has mean deltas of `-0.0657 dB` PSNR_fg, `-0.0011` SSIM_fg, and `+0.0003` LPIPS_fg relative to A6. Mean outside decreases from `0.0308` to `0.0300`, and mean leakage decreases from `0.0216` to `0.0210`.

Total Gaussian count decreases from `1216294` to `997049`, a reduction of `219245` (`18.03%`).

## Takeaway

S18 now provides three-sample positive evidence and main-sample closure. `A6+M1-soft+M4` remains close to A6 on dense/occluded, thin-leaf, and main complex-background samples, stays within the foreground-only thresholds, and consistently reduces Gaussian count.

The accurate paper positioning is:

```text
Ours-core = A6
Ours-full / Ours-compact = A6 + M1-soft + M4
```

This is not a claim that the full variant significantly improves PSNR/SSIM/LPIPS. The correct claim is that `Ours-full` improves compactness and leakage control while preserving foreground-object reconstruction quality within a small margin.
