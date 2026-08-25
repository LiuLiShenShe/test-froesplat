# KongQueZhuYu A0-A6 foreground-object objective 消融

| ablation_id | method_tag | fg_init | fg_rgb_loss | alpha_mask_loss | bg_opacity_loss | fg_psnr | fg_ssim | fg_lpips_black_bg | outside_nonblack_ratio_mean | leakage_energy_ratio_mean | gaussians_30000 | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | E2_2dgs_baseline | False | False | False | False | 24.208995 | 0.851372 | 0.048028 | 0.990785 | 1.220132 | 751213 | success |
| A1 | E3_fsam3_preprocess | False | implicit | False | False | 20.729053 | 0.750485 | 0.069642 | 0.007278 | 0.004234 | 263108 | success |
| A2 | A2_alpha_mask_loss_only | False | False | True | False | 24.342241 | 0.847842 | 0.049075 | 0.989829 | 1.226045 | 768067 | success |
| A3 | A3_bg_opacity_only | False | False | False | True | 24.750785 | 0.867198 | 0.045077 | 0.989995 | 1.225467 | 742931 | success |
| A4 | A4_alpha_mask_bg_opacity | False | False | True | True | 24.812574 | 0.868724 | 0.044475 | 0.989614 | 1.226591 | 763266 | success |
| A5 | A5_fg_rgb_alpha_bg_loss | False | True | True | True | 25.105544 | 0.856108 | 0.043739 | 0.029387 | 0.019000 | 592900 | success |
| A6 | A6_foreground_track_init_fg_rgb_alpha_bg | True | True | True | True | 25.007229 | 0.854780 | 0.043844 | 0.029438 | 0.018884 | 591623 | success |
| A6+M4 | F1_high_precision_foreground | True | True | True | True | 24.972310 | 0.854043 | 0.043838 | 0.029277 | 0.018613 | 585594 | success |

说明：A0/A1 复用既有 E2/E3 正式输出；A6+M4 复用既有 `F1_high_precision_foreground`，用于说明当前已验证完整结果。
核心论文消融应优先比较 A0-A6；M4 pruning、M1 foreground gate 和 M5 mesh 作为后续模块单独汇报。
foreground-only 分离阈值暂定为 outside_nonblack < 0.05 且 leakage < 0.10。
