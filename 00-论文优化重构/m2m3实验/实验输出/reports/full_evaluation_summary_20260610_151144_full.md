# M2M3 full evaluation summary: 20260610_151144_full

PSNR/SSIM higher is better; LPIPS lower is better. Reduction columns are relative to the scene baseline.

## Output files

- Full run directory: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/runs/20260610_151144_full`
- Training/evaluation logs: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/logs`
- Summary CSV: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/reports/full_evaluation_summary_20260610_151144_full.csv`
- Summary Markdown: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/reports/full_evaluation_summary_20260610_151144_full.md`
- Evaluation runtime JSON: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/reports/evaluation_runtime_20260610_151144_full.json`
- Capacity and point-cloud summary: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/reports/capacity_and_pointcloud_summary.csv`
- Per-model final PLY: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/runs/20260610_151144_full/<SCENE>/<METHOD>/point_cloud/iteration_30000/point_cloud.ply`
- Per-model rendered test views: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/runs/20260610_151144_full/<SCENE>/<METHOD>/test/ours_30000/{renders,gt,vis}`
- Per-model average metrics: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/runs/20260610_151144_full/<SCENE>/<METHOD>/results.json`
- Per-model per-view metrics: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/runs/20260610_151144_full/<SCENE>/<METHOD>/per_view.json`
- Per-model capacity reports: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/runs/20260610_151144_full/<SCENE>/<METHOD>/capacity_control/`
- Per-model pruning reports: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/runs/20260610_151144_full/<SCENE>/<METHOD>/pruning/`

## CaoMei2

| method | PSNR | SSIM | LPIPS | points | PLY MB | size red. | point red. | train | PSNR d | SSIM d | LPIPS d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fg2dgs_baseline | 5.323 | 0.1196 | 0.7070 | 593310 | 144.8 | -0.0% | -0.0% | 38:15 | 0.000 | 0.0000 | 0.0000 |
| fg2dgs_m2m3 | 5.326 | 0.1202 | 0.7072 | 404330 | 98.7 | 31.9% | 31.9% | 15:48 | 0.003 | 0.0007 | 0.0001 |
| fg2dgs_m2m3_floor40 | 5.328 | 0.1193 | 0.7073 | 448641 | 109.5 | 24.4% | 24.4% | 21:58 | 0.005 | -0.0002 | 0.0003 |

## ChangShouHua2

| method | PSNR | SSIM | LPIPS | points | PLY MB | size red. | point red. | train | PSNR d | SSIM d | LPIPS d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fg2dgs_baseline | 15.423 | 0.4387 | 0.4714 | 57408 | 14.0 | -0.0% | -0.0% | 11:51 | 0.000 | 0.0000 | 0.0000 |
| fg2dgs_m2m3 | 15.479 | 0.5163 | 0.4326 | 179105 | 43.7 | -212.0% | -212.0% | 12:50 | 0.056 | 0.0776 | -0.0389 |
| fg2dgs_m2m3_floor40 | 15.248 | 0.4165 | 0.4760 | 131099 | 32.0 | -128.3% | -128.4% | 12:13 | -0.175 | -0.0222 | 0.0045 |

## KongQueZhuYu

| method | PSNR | SSIM | LPIPS | points | PLY MB | size red. | point red. | train | PSNR d | SSIM d | LPIPS d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fg2dgs_baseline | 6.253 | 0.2724 | 0.5571 | 538114 | 131.3 | -0.0% | -0.0% | 17:33 | 0.000 | 0.0000 | 0.0000 |
| fg2dgs_m2m3 | 6.252 | 0.2717 | 0.5572 | 536464 | 130.9 | 0.3% | 0.3% | 17:01 | -0.001 | -0.0007 | 0.0000 |
| fg2dgs_m2m3_floor40 | 6.253 | 0.2722 | 0.5573 | 536761 | 131.0 | 0.3% | 0.3% | 17:10 | -0.000 | -0.0002 | 0.0001 |

