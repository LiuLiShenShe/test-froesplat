# E6_2DGS_Standard_VGGT — RUN REPORT

- method: **Official 2DGS (hbb1/2d-gaussian-splatting)** on the standard-VGGT COLMAP dataset
- plant: KongQueZhuYu | views: 208 (all training, eval=False, same-viewpoint vs E1/E5)
- source_path: `experiments/KongQueZhuYu/standard_vggt_comparison/Standard_VGGT_Dataset`
- resolution: 2 (1080×1920, same as E1/E5)
- iterations: 30000
- regularization: lambda_dist 0.2 (depth-distortion, official 2DGS default),
  lambda_normal 0.05 (official default); **2DGS's own recipe, not GGGS hyperparameters**
- initialization: the same 1,252,383-point standard-VGGT cloud (points3D.ply)
- GPU: 1 (A6000 48 GB)
- repo: /data/fj/F2DMAS/2d-gaussian-splatting-main (venv torch 2.5.1+cu121, py3.12)
  - extensions reinstalled from current path (editable finder pointed at a stale /data/fj/... path)
  - matplotlib compat: `colormap()` now uses `fig.canvas.buffer_rgba()`
    (`tostring_rgb()` removed in matplotlib ≥3.9; crash at the first test-iteration save otherwise)

## Notes

- Official hbb1 recipe: `python train.py -s <path> -m <path> -r 2 --iterations 30000
  --lambda_dist 0.2 --lambda_normal 0.05` (defaults for dist/normal in the official repo;
  this local tree's .py default is 0.0 so both were passed explicitly).
- `--eval` was deliberately NOT used so all 208 views train — matching E1/E5 which train on
  all views (Training-view Metrics).
- Evaluation via `evaluate_e6.py` — same metric semantics as E1/E5 (image_metrics +
  foreground bbox logic inlined; imports no GGGS modules so the 2DGS venv resolves its own
  gaussian_renderer/Scene).

## Results (208 training views, Training-view Metrics)

- PSNR: 13.2018 (E1 reference: 17.8947, E5: 13.1057)
- SSIM: 0.7134 (E1: 0.7858, E5: 0.7193)
- LPIPS: 0.3766 (E1: 0.2406, E5: 0.3546)
- FG-PSNR: 10.5194 (E1: 15.2787, E5: 10.4520)
- FG-SSIM: 0.4772 (E1: 0.6117, E5: 0.5006)
- final Gaussian count: 119,836 (2DGS prunes aggressively vs GGGS 5,153,165)

Conclusion: E6 ≈ E5 within noise on every metric. An entirely different 3DGS-family
algorithm (surfel-based 2DGS with official regularization) trained on the SAME
standard-VGGT COLMAP dataset reproduces the same output quality as GGGS → the
reconstruction layer is not the bottleneck; the standard-VGGT frontend inputs are.
Attribution stays case C; 2DGS does NOT resolve the blur.

## Outputs

- checkpoints: `checkpoints/` (iter 30000 + chkpnt30000.pth)
- training wall time: 45.6 min (17:35→18:20), in-training snapshot PSNR 15.27 (iter 30000)
- metrics: `metrics/per_view_metrics.csv`, `metrics/render_metrics.json`
  (via `evaluate_e6.py`, same metric semantics as E1/E5)
- full-render dump (208 views) + bounded TSDF mesh (`fuse.ply`, `post_fuse.ply`):
  `checkpoints/train/ours_30000/` (official `render.py --skip_test -r 2`)
- comparison-view PNGs: `../render_comparison/e6_renders/`
  (composed into GT|E1|E5|E6 quad panels by `render_comparison.py`)