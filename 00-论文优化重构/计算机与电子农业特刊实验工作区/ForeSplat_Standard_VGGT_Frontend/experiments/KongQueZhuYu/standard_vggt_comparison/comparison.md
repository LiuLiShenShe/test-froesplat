# KongQueZhuYu — Standard VGGT frontend vs VGGT-Omega frontend (GGGS backend)

**Metric scope: Training-view Metrics.** All 208 views participate in training
(`eval=False`, same as the E1..E4 four-experiment comparison). Numbers are
*fit-to-training-images* metrics, not novel-view generalisation.

E1 and E5 share the identical GGGS recipe: original `train_foresplat.py`
E1 config — iterations 30000, resolution divisor 2, seed 0, EO off, FG off
(pruning none), black background, same initial count 1,252,383 points (one per
frontend's frontend-exported initialization cloud). E6 keeps the same
standard-VGGT dataset/cameras/init cloud but swaps the 3D layer to **official
2DGS (hbb1) with its own regularization** (lambda_dist 0.2, lambda_normal 0.05),
not GGGS hyperparameters. The E1↔E5 axis varies only the frontend; the E5↔E6
axis varies only the 3D layer on a fixed frontend — together they separate the
two hypotheses.

| Frontend | PSNR | SSIM | LPIPS | FG-PSNR | FG-SSIM | Init Gaussians | Final Gaussians | Training Time (min) | Peak VRAM (MiB) |
|---|---|---|---|---|---|---|---|---|---|
| VGGT-Omega balanced@512 (E1) | 17.8947 | 0.7858 | 0.2406 | 15.2787 | 0.6117 | 1,252,383 | 600,458 | 40.3 | 8,950 |
| Standard VGGT square@518 + GGGS (E5) | **13.1057** | 0.7193 | 0.3546 | 10.4520 | 0.5006 | 1,252,383 | **5,153,165** | 49.6 | 18,331 |
| Standard VGGT square@518 + **official 2DGS** (E6) | **13.2018** | 0.7134 | 0.3766 | 10.5194 | 0.4772 | 1,252,383 | 119,836 | 45.6 | — |
| Δ (Standard GGGS − Omega) | **−4.7890** | −0.0665 | +0.1140 | −4.8267 | −0.1111 | 0 | +4,552,707 | +9.3 | +9,381 |
| Δ (2DGS − GGGS on std VGGT) | +0.0961 | −0.0059 | +0.0220 | +0.0674 | −0.0234 | 0 | −5,013,329 | −4.0 | — |

Per-view metrics: `E1_GGGS_Baseline/metrics/per_view_metrics.csv`,
`E5_Standard_VGGT_GGGS_Baseline/metrics/per_view_metrics.csv`, and
`E6_2DGS_Standard_VGGT/metrics/per_view_metrics.csv`.

## Key observations (data-driven)

1. **Standard VGGT frontend is strictly worse on this plant.** Every rendered
   metric regresses: PSNR −4.79 dB, SSIM −0.067, LPIPS +0.114 (worse — lower is
   better), FG-PSNR −4.83 dB, FG-SSIM −0.111. This is not a marginal change; it
   is a categorical regression, not the marginal PSNR change the spec warned
   against over-interpreting.

2. **The densification trajectory diverges.** E1 (fused init) peaks at 1.25M
   gaussians and prunes down to 600K; E5 (raw VGGT back-projection init)
   densifies up to 5.15M gaussians. Same `densify_grad_threshold`, same
   schedule; the difference is that the raw standard-VGGT cloud has a
   distribution that triggers continued GGGS densification (raw depth
   back-projections have more noisy / sparsely-observed voxels to densify
   into), while the VGGT-Omega fused cloud is more uniformly covered from the
   start.

3. **Standard VGGT 518-square preprocessing wastes 44% of the model grid on
   black bars** for a 2160×3840 portrait image (content strip = 291×518 =
   150,738 px vs VGGT-Omega's 688×384 = 264,192 px). Native-grid per-pixel
   depth gradient is the same (0.0118 vs 0.0118), so the per-pixel depth is not
   smoother, but the content pixel budget is 43% lower.

4. **The blur is not resolved by standard VGGT.** The question was whether
   standard VGGT's sharper depth would help GGGS; the answer is no — GGGS's
   blur persists and is worse here.

5. **E6 (official 2DGS) ties E5 (GGGS) on the same standard-VGGT inputs.** An
   entirely different 3DGS-family algorithm — surfel-based 2DGS (hbb1) with its
   own official regularization (lambda_dist 0.2, lambda_normal 0.05) — trained
   from the same COLMAP dataset/cameras/init cloud reproduces E5 within noise
   (ΔPSNR +0.10 dB, ΔSSIM −0.006, ΔLPIPS +0.022, ΔFG-PSNR +0.07 dB). 2DGS even
   pruned to 119,836 gaussians (vs GGGS's 5.15M) yet scores the same. This
   isolates the failure: it is **not** the GGGS reconstruction layer — any
   3DGS-family backend fed standard-VGGT frontend outputs lands at ~13 dB while
   the VGGT-Omega frontend reaches 17.9 dB. The bottleneck is upstream, in the
   standard-VGGT frontend (square@518 preprocessing content budget, and/or the
   per-view geometry it exports).

6. **2DGS is not even sharper — it is smoother.** Foreground gradient magnitude
   over the 8 shared views (`render_comparison/sharpness.csv`): GT 0.0360 /
   E1 0.0143 / E5 0.0111 / **E6 0.0028**. The surfel representation
   over-smooths, so the 2DGS render loses more high-frequency detail than GGGS
   while scoring the same PSNR. Sharpness-wise the ordering is GT ≫ E1 > E5 > E6.

## Depth / geometry quality (pre-training, 12-view sample)

| metric | VGGT-Omega balanced@512 | Standard VGGT square@518 | note |
|---|---|---|---|
| model grid (content pixels) | 688×384 = 264,192 px | 291×518 = 150,738 px | square padding wastes 44% on black bars |
| fg depth-gradient, upsampled to orig | 0.0022 | 0.0017 | original-res normalized depth |
| fg depth-gradient, native grid | 0.0118 | 0.0118 | per-pixel local gradient identical |
| reprojection error (208 views) | — (fused cloud, fused voxel cores) | mean 0.0004 px, p95 0.0011 px | standard VGGT W2C + exact affine |