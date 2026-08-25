# KongQueZhuYu 小矩阵结果汇总

| method_tag | status | psnr | ssim | lpips | gaussians_30000 | pruning_removed_total | mesh_mode | post_mesh_vertices |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E2_2dgs_baseline | success | 24.159228 | 0.887980 | 0.276462 | 751213 |  |  |  |
| E3_fsam3_preprocess | success | 6.134541 | 0.237451 | 0.575957 | 263108 |  |  |  |
| E6_mask_constrained | success | 23.913802 | 0.886522 | 0.284373 | 756977 |  |  |  |
| E7_mask_pruning | success | 24.710588 | 0.897897 | 0.265125 | 694992 | 50652 |  |  |
| E8_full_plant_aware | success | 23.251584 | 0.872133 | 0.288421 | 737919 | 53780 | post_boundary | 169986 |

说明：E3 训练阶段使用 `mask_mode=preprocess`，渲染/评测阶段覆盖为 `mask_mode=alpha`，以保留全图 GT 指标口径。
E8 训练阶段使用 H-VQG retained list，渲染阶段覆盖为全测试集，以和 E2/E3/E6/E7 保持可比。
