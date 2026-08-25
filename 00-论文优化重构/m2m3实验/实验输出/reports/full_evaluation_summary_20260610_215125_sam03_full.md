# M2M3 full evaluation summary: 20260610_215125_sam03_full

PSNR/SSIM higher is better; LPIPS lower is better. Reduction columns are relative to the scene baseline.

## Output files

- Full run directory: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/runs/20260610_215125_sam03_full`
- Training/evaluation logs: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/logs`
- Summary CSV: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/reports/full_evaluation_summary_20260610_215125_sam03_full.csv`
- Summary Markdown: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/reports/full_evaluation_summary_20260610_215125_sam03_full.md`
- Evaluation runtime JSON: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/reports/evaluation_runtime_20260610_215125_sam03_full.json`
- Capacity and point-cloud summary: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/reports/capacity_and_pointcloud_summary.csv`
- Per-model final PLY: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/runs/20260610_215125_sam03_full/<SCENE>/<METHOD>/point_cloud/iteration_30000/point_cloud.ply`
- Per-model rendered test views: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/runs/20260610_215125_sam03_full/<SCENE>/<METHOD>/test/ours_30000/{renders,gt,vis}`
- Per-model average metrics: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/runs/20260610_215125_sam03_full/<SCENE>/<METHOD>/results.json`
- Per-model per-view metrics: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/runs/20260610_215125_sam03_full/<SCENE>/<METHOD>/per_view.json`
- Per-model capacity reports: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/runs/20260610_215125_sam03_full/<SCENE>/<METHOD>/capacity_control/`
- Per-model pruning reports: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/runs/20260610_215125_sam03_full/<SCENE>/<METHOD>/pruning/`

## CaoMei2

| method | PSNR | SSIM | LPIPS | points | PLY MB | size red. | point red. | train | PSNR d | SSIM d | LPIPS d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fg2dgs_baseline | 5.199 | 0.1101 | 0.7042 | 245315 | 59.9 | -0.0% | -0.0% | 13:55 | 0.000 | 0.0000 | 0.0000 |
| fg2dgs_m2m3 | 5.198 | 0.1101 | 0.7041 | 248852 | 60.7 | -1.4% | -1.4% | 13:53 | -0.000 | -0.0000 | -0.0001 |
| fg2dgs_m2m3_floor40 | 5.198 | 0.1101 | 0.7044 | 250033 | 61.0 | -1.9% | -1.9% | 13:45 | -0.000 | 0.0000 | 0.0002 |

## ChangShouHua2

| method | PSNR | SSIM | LPIPS | points | PLY MB | size red. | point red. | train | PSNR d | SSIM d | LPIPS d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fg2dgs_baseline | 15.386 | 0.4941 | 0.4439 | 131778 | 32.2 | -0.0% | -0.0% | 12:01 | 0.000 | 0.0000 | 0.0000 |
| fg2dgs_m2m3 | 15.625 | 0.5316 | 0.4272 | 147534 | 36.0 | -12.0% | -12.0% | 12:14 | 0.239 | 0.0375 | -0.0167 |
| fg2dgs_m2m3_floor40 | 15.510 | 0.5023 | 0.4349 | 126941 | 31.0 | 3.7% | 3.7% | 11:57 | 0.124 | 0.0081 | -0.0089 |

## KongQueZhuYu

| method | PSNR | SSIM | LPIPS | points | PLY MB | size red. | point red. | train | PSNR d | SSIM d | LPIPS d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fg2dgs_baseline | 6.251 | 0.2678 | 0.5566 | 532641 | 130.0 | -0.0% | -0.0% | 17:00 | 0.000 | 0.0000 | 0.0000 |
| fg2dgs_m2m3 | 6.251 | 0.2679 | 0.5570 | 534751 | 130.5 | -0.4% | -0.4% | 17:12 | 0.000 | 0.0001 | 0.0003 |
| fg2dgs_m2m3_floor40 | 6.251 | 0.2678 | 0.5569 | 535116 | 130.6 | -0.5% | -0.5% | 16:54 | 0.000 | 0.0000 | 0.0003 |

