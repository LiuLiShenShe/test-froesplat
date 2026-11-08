# E5_Standard_VGGT_GGGS_Baseline - RUN REPORT

- plant: KongQueZhuYu
- frontend: **standard VGGT-1B** (square 518 preprocessing, black background)
  vs E1 reference frontend: VGGT-Omega balanced@512
- exposure compensation (EO): off
- ForeSplat foreground constraints (FG): off
- iterations: 30000 | resolution divisor: 2 | seed: 0
- initial points: 1252383 (deterministic seed-0 per-view stratified sampling of
  standard-VGGT raw depth back-projections after fg & depth>0 & conf>=p20; no fusion)
- training wall time: 49.6 min
- final Gaussian count: 5,153,165
- peak VRAM allocated: 18,331 MiB
- coordinate frame: standard VGGT native world frame

## Rendered metrics (208 training views, Training-view Metrics)

- PSNR: 13.1057 (E1 reference: 17.8947)
- SSIM: 0.7193 (E1: 0.7858)
- LPIPS: 0.3546 (E1: 0.2406)
- FG-PSNR: 10.4520 (E1: 15.2787)
- FG-SSIM: 0.5006 (E1: 0.6117)
- per-view PSNR range: 8.25 / 12.69 (median) / 32.66 dB — much higher variance than E1

Conclusion: strict regression vs E1 on every metric. Attribution: case C
(no improvement from standard VGGT; blur persists and worsens).

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=1 python gggs_community/train_foresplat.py \
  --source_path experiments/KongQueZhuYu/standard_vggt_comparison/Standard_VGGT_Dataset \
  --model_path experiments/KongQueZhuYu/standard_vggt_comparison/E5_Standard_VGGT_GGGS_Baseline/checkpoints \
  --resolution 2 --iterations 30000 \
  --mask_dir experiments/KongQueZhuYu/standard_vggt_comparison/Standard_VGGT_Dataset/masks
```
(identical flags to E1, minus nothing; E1 used the same `train_foresplat.py` recipe)

## Outputs

- checkpoints + cfg: `checkpoints/`
- final Gaussian ply: `checkpoints/final_gaussians/point_cloud.ply`
- training metrics csv: `metrics/train_metrics.csv`
- training log: `train_30000.log` (200-iter smoke in `smoke/`)

## Frontend source data

See `../source_data_audit.md` and `../model_metadata.json`.