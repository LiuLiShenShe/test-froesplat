# Four-experiment comparison (KongQueZhuYu, all-view training)

Metrics are computed on the **training views** (all 208 views participate in
training; ``eval=False``). They are Training-view Metrics, not novel-view
generalisation numbers.

| Method | EO | Foreground | PSNR | SSIM | LPIPS | FG-PSNR | FG-SSIM | Gaussian Count | Training Time (min) |
|---|---|---|---|---|---|---|---|---|---|
| GGGS | No | No | 17.8947 | 0.7858 | 0.2406 | 15.2787 | 0.6117 | 600458 | 40.3 |
| GGGS + EO | Yes | No | 17.3925 | 0.7844 | 0.2492 | 14.7497 | 0.6093 | 607422 | 41.3 |
| GGGS + FG | No | Yes | 15.1640 | 0.6509 | 0.3795 | 13.1239 | 0.4609 | 1037020 | 46.4 |
| ForeSplat-GGGS | Yes | Yes | 14.2161 | 0.6488 | 0.3547 | 11.5697 | 0.4041 | 814255 | 37.5 |
