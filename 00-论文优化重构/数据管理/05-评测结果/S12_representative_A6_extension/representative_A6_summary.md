# S12 representative A6 extension summary

| sample | role | status | fg_psnr | fg_ssim | fg_lpips_black_bg | outside_nonblack_ratio_mean | leakage_energy_ratio_mean | gaussians_30000 | sanitized_images_after | sanitized_dropped_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KongQueZhuYu | 复杂背景/主样本 | success | 25.007229 | 0.854780 | 0.043844 | 0.029438 | 0.018884 | 591623 |  |  |
| XianKeLai1 | 薄叶/细结构 | success | 23.727611 | 0.827828 | 0.030883 | 0.048356 | 0.037885 | 253827 | 203 | 5 |
| CaoMei2 | 密集叶/遮挡 | success | 25.083288 | 0.812117 | 0.024955 | 0.014656 | 0.008135 | 370844 | 203 | 7 |

Interpretation focus: A6 should preserve foreground-object quality across the thin-structure and dense-occlusion samples, while keeping leakage metrics low enough for plant-only export / mesh follow-up.
KongQueZhuYu uses the original final_locked scene; XianKeLai1 and CaoMei2 use sanitized COLMAP views only because a few locked image/mask files are unreadable.
