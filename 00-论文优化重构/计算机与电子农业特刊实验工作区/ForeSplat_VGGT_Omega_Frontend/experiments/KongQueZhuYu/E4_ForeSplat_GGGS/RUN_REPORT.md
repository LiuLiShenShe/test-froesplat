# E4_ForeSplat_GGGS - RUN REPORT

- plant: KongQueZhuYu
- exposure compensation (EO): ON
- ForeSplat foreground constraints (FG): ON
- iterations: 30000 | resolution divisor: 2 | seed: 0
- training wall time: 38.9 min
- final Gaussian count: 814161
- peak VRAM allocated: 10648 MiB
- coordinate frame: VGGT-Omega native (no scan registration)

## Outputs

- checkpoints + cfg: `checkpoints/`
- final Gaussian ply: `checkpoints/final_gaussians/point_cloud.ply`
- training metrics csv: `metrics/train_metrics.csv`
- per-view exposure params: `exposure_params/exposure_matrices.npz`

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=1 python gggs_community/train_foresplat.py --source_path runs/full_sequence_batch/<plant>/GGGS_Dataset --resolution 2 --iterations 30000 --mask_dir runs/full_sequence_batch/<plant>/GGGS_Dataset/masks --enable_eo --use_foreground_rgb_loss --foreground_rgb_crop_padding 12 --use_mask_loss --use_bg_opacity_loss --lambda_mask 0.08 --lambda_bg 0.02 --mask_loss_type l1_dice --mask_ignore_boundary_px 2 --mask_loss_start_iter 500 --mask_loss_warmup_iters 1500 --pruning_mode mask --pruning_start_iter 18000 --pruning_interval 3000 --pruning_opacity_threshold 0.005 --pruning_brightness_threshold 0.01 --pruning_mask_threshold 0.45 --pruning_mask_max_views 12 --pruning_max_remove_ratio 0.03 --pruning_mask_score_weight 3.0 --save_pruning_report
```
