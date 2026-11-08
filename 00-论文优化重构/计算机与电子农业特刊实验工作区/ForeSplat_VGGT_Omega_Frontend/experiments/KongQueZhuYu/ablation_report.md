# ForeSplat-GGGS ablation report - KongQueZhuYu

## A. Experimental setup

- Plant: KongQueZhuYu, full sequence from the ForeSplat VGGT-Omega frontend.
- Views: 208 training views (all valid views; eval split disabled).
- Initialisation: 1252383 fused foreground points (VGGT-Omega multi-view fusion);
  identical for all four experiments, re-created into a fresh GaussianModel each run.
- Resolution divisor 2, SH degree 0, SG degree 0, seed 0,
  30000 iterations, identical optimiser / densification / pruning schedule
  (pruning only in the Foreground experiments, as defined by the factor).
- Coordinate frame: VGGT-Omega native. No laser-scan Sim(3) registration, no scale change.
- GPU: single RTX A6000 (48 GB, GPU 1); experiments run sequentially, never concurrently.

## B. Quantitative comparison

| Method | EO | Foreground | PSNR | SSIM | LPIPS | FG-PSNR | FG-SSIM | Gaussian Count | Training Time (min) |
|---|---|---|---|---|---|---|---|---|---|
| GGGS | No | No | 17.8947 | 0.7858 | 0.2406 | 15.2787 | 0.6117 | 600458 | 40.3 |
| GGGS + EO | Yes | No | 17.3925 | 0.7844 | 0.2492 | 14.7497 | 0.6093 | 607422 | 41.3 |
| GGGS + FG | No | Yes | 15.1640 | 0.6509 | 0.3795 | 13.1239 | 0.4609 | 1037020 | 46.4 |
| ForeSplat-GGGS | Yes | Yes | 14.2161 | 0.6488 | 0.3547 | 11.5697 | 0.4041 | 814255 | 37.5 |

All metrics are Training-view Metrics over the 208 training views.
FG-PSNR / FG-SSIM are computed inside each view's GT-mask bounding box.

## C. Ablation analysis (data-driven)

1. **Exposure compensation (E2 - E1)**: full-image PSNR -0.5022, SSIM -0.0014. EO did not improve the rendered metrics on this data.
2. **Foreground constraints (E3 - E1)**: full-image PSNR -2.7307, SSIM -0.1349, foreground-region PSNR -2.1548. The foreground loss did not improve the foreground-region metric here.
3. **Background suppression**: E3/E4 apply the mask + background-opacity losses and mask-guided pruning; Gaussian counts are 600458 (E1), 607422 (E2), 1037020 (E3), 814255 (E4). Foreground runs did not reduce the primitive budget on this data.
4. **Joint effect (E4 - E1)**: full-image PSNR -3.6787; best PSNR among the four: E1_GGGS_Baseline. The combined method is not the best on PSNR - report this honestly rather than assuming E4 wins.
5. **Caveats**: metrics are on training views (all views trained on); per-view LPIPS/SSIM
   variation is in `metrics/per_view_metrics.csv`; no novel-view split was held out.

## D. Preparation for scan-based geometric evaluation (not performed here)

- Every experiment keeps the Gaussian cloud (means / RGB / opacity / scale / rotation) in
  `gaussian/gaussians.npz` and `gaussian/point_cloud.ply`, in the VGGT-Omega native frame.
- Native meshes (GGGS TSDF fusion) are in `mesh/` when extraction succeeded.
- The camera metadata and the scene alignment matrix are in the dataset metadata;
  a later stage can register these outputs to the laser scan once, then evaluate geometry.
