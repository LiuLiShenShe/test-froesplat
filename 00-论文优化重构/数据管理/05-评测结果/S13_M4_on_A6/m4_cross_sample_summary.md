# S13 M4 on A6 cross-sample summary

## Purpose

S13 tests whether M4 mask pruning can act as a practical enhancement after A6 foreground-object reconstruction. The expected benefit is not higher foreground PSNR, but a smaller Gaussian model and cleaner export while preserving foreground quality.

## Results

| Sample | Variant | Eval images | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| XianKeLai1 | A6 | 26 | 23.7276 | 0.8278 | 0.0309 | 0.0484 | 0.0379 | 253827 |
| XianKeLai1 | A6+M4 | 26 | 23.7256 | 0.8279 | 0.0310 | 0.0486 | 0.0376 | 251047 |
| CaoMei2 | A6 | 26 | 25.0833 | 0.8121 | 0.0250 | 0.0147 | 0.0081 | 370844 |
| CaoMei2 | A6+M4 | 26 | 25.0303 | 0.8108 | 0.0251 | 0.0144 | 0.0080 | 284757 |

## Cross-sample Delta

| Metric | A6+M4 - A6 |
|---|---:|
| PSNR_fg | -0.0275 dB |
| SSIM_fg | -0.0006 |
| LPIPS_fg | +0.0001 |
| Outside | -0.0000 |
| Leakage | -0.0002 |
| total Gaussians | -88867 (-14.23%) |

## Takeaway

Across XianKeLai1 and CaoMei2, M4 preserves foreground-object reconstruction quality within noise-level differences and reduces total Gaussian count by 88,867, about 14.23%. This supports M4 as a compactness/export module after A6, not as a replacement for the A6 foreground-object objective.
