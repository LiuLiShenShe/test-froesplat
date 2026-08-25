# KongQueZhuYu foreground-object 指标汇总

| method_tag | is_separated_foreground | fg_psnr | fg_ssim | fg_lpips_black_bg | outside_nonblack_ratio_mean | leakage_energy_ratio_mean |
| --- | --- | --- | --- | --- | --- | --- |
| E2_2dgs_baseline | False | 24.208995 | 0.851372 | 0.048028 | 0.990785 | 1.220132 |
| E3_fsam3_preprocess | True | 20.729053 | 0.750485 | 0.069642 | 0.007278 | 0.004234 |
| E6_mask_constrained | False | 23.959506 | 0.846319 | 0.048354 | 0.990034 | 1.212727 |
| E7_mask_pruning | False | 24.752678 | 0.867630 | 0.044805 | 0.990334 | 1.226870 |
| E8_full_plant_aware | False | 24.037721 | 0.849281 | 0.047691 | 0.983076 | 1.196330 |
| E7_mask_pruning_foreground_object | False | 24.691759 | 0.865808 | 0.044920 | 0.750921 | 0.789977 |
| F1_high_precision_foreground | True | 24.972310 | 0.854043 | 0.043838 | 0.029277 | 0.018613 |

说明：`fg_*` 只表示 mask 内重建质量，不能单独证明已经分离 foreground object。
`outside_nonblack_ratio_mean` 和 `leakage_energy_ratio_mean` 是 mask 外泄漏指标；当前判定阈值暂定为 outside_nonblack < 0.05 且 leakage < 0.10。
因此 E7 虽然 mask 内质量高，但属于 full-scene 方法，背景泄漏极高，不能算 foreground-object 分离成功。F1_high_precision_foreground 满足分离阈值，同时取得当前最高 PSNR_fg 和最低 LPIPS_fg_black_bg；SSIM_fg 略低于 E7，但 E7/E7_export 均未通过背景泄漏约束。

E3 评测口径复核：`foreground_object_results.json` 中的 `PSNR_fg/SSIM_fg/LPIPS_fg_black_bg` 已对 render 和 GT 同时应用 foreground mask，不是用完整背景直接对比分数。为排除黑背景画布影响，已补充 mask bbox crop 复核指标：`SSIM_fg_crop=0.750485`，与 `SSIM_fg=0.750485` 基本一致；`LPIPS_fg_crop=0.122604`。因此 E3 低分主要来自前景自身重建质量和边界/纹理误差，而不是评测拿错了未分割背景。

F1 30k 正式结果：初始点云用 COLMAP track + mask 从 177918 点过滤到 118119 点，保留 66.39%；foreground-object eval 得到 PSNR_fg=24.972310、SSIM_fg=0.854043、LPIPS_fg_black_bg=0.043838、outside_nonblack_ratio_mean=0.029277、leakage_energy_ratio_mean=0.018613。full-frame PSNR=6.254815 是预期现象，表示该分支不再重建背景。
