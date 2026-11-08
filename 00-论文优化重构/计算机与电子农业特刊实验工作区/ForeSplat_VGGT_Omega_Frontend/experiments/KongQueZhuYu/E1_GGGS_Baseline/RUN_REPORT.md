# E1_GGGS_Baseline - RUN REPORT

- plant: KongQueZhuYu
- exposure compensation (EO): off
- ForeSplat foreground constraints (FG): off
- iterations: 30000 | resolution divisor: 2 | seed: 0
- training wall time: 41.3 min
- final Gaussian count: 600458
- peak VRAM allocated: 8950 MiB
- coordinate frame: VGGT-Omega native (no scan registration)

## Outputs

- checkpoints + cfg: `checkpoints/`
- final Gaussian ply: `checkpoints/final_gaussians/point_cloud.ply`
- training metrics csv: `metrics/train_metrics.csv`

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=1 python gggs_community/train_foresplat.py --source_path runs/full_sequence_batch/<plant>/GGGS_Dataset --resolution 2 --iterations 30000 --mask_dir runs/full_sequence_batch/<plant>/GGGS_Dataset/masks 
```
