# KongQueZhuYu 小矩阵运行说明

更新日期：2026-06-06

## 目标

在 `KongQueZhuYu` 已有 `E2_2dgs_baseline` 正式 30k baseline 的基础上，补齐同一样本可比较小矩阵：

```text
E2_2dgs_baseline
E3_fsam3_preprocess
E6_mask_constrained
E7_mask_pruning
E8_full_plant_aware
```

2026-06-05/06 追加特刊实验四 B0-B5 位置消融所需的严格缺项：

```text
B2_foreground_track_init_only
B4_mask_pruning_only
```

## 统一口径

- 输入场景固定为 `数据管理/02-位姿COLMAP/03-final_locked/KongQueZhuYu`。
- mask 入口固定为 `数据管理/03-分割Mask/02-sam_masks/KongQueZhuYu`。
- 训练均为 30000 iter，`resolution=4`，`eval=true`。
- RGB 指标均使用 `evaluate_rendered_metrics.py` 计算 PSNR/SSIM/LPIPS。
- E3 训练阶段使用 `mask_mode=preprocess`，渲染评测阶段覆盖为 `mask_mode=alpha`，避免 GT 被 mask 后和 E2 指标口径不一致。
- E8 训练阶段使用 H-VQG retained list，渲染阶段覆盖回全测试集，保持与 E2/E3/E6/E7 同一测试集。
- B2 只启用 `init_pcd_mode=foreground_track`，mask 只用于筛选初始 COLMAP 点，不启用 RGB/mask/opacity loss 和 pruning。
- B4 只启用 `pruning_mode=mask`，mask 只用于 15000-30000 iter 的 pruning score，不启用前景初始化和任何训练损失。

## 配置位置

```text
数据管理/07-运行脚本与超参/S10-KongQueZhuYu小矩阵/configs/
```

新增配置：

```text
kongquezhu_B2_foreground_track_init_only.json
kongquezhu_B4_mask_pruning_only.json
```

## 汇总输出

```text
数据管理/05-评测结果/KongQueZhuYu/S10_small_matrix/
├── kongquezhu_small_matrix_summary.csv
├── kongquezhu_small_matrix_summary.json
└── kongquezhu_small_matrix_summary.md
```

特刊实验四 B0-B5 位置消融汇总脚本：

```text
数据管理/07-运行脚本与超参/S10-KongQueZhuYu小矩阵/scripts/summarize_experiment4_b0_b5.py
```

输出到：

```text
计算机与电子农业特刊实验工作区/04-结果表格模板/实验四_先验注入位置消融结果表.csv
计算机与电子农业特刊实验工作区/05-图件与论文映射/实验四_2DGS先验注入位置消融/
```

## 当前结果

已于 2026-05-18 完成 E2/E3/E6/E7/E8 小矩阵。

```text
E2_2dgs_baseline        PSNR=24.1592  SSIM=0.8880  LPIPS=0.2765  Gaussians=751213
E3_fsam3_preprocess    PSNR=6.1345   SSIM=0.2375  LPIPS=0.5760  Gaussians=263108
E6_mask_constrained    PSNR=23.9138  SSIM=0.8865  LPIPS=0.2844  Gaussians=756977
E7_mask_pruning        PSNR=24.7106  SSIM=0.8979  LPIPS=0.2651  Gaussians=694992
E8_full_plant_aware    PSNR=23.2516  SSIM=0.8721  LPIPS=0.2884  Gaussians=737919
```

补充记录：

- E7 pruning report 共 7 份，总移除 50652 个 Gaussian。
- E8 pruning report 共 7 份，总移除 53780 个 Gaussian。
- E8 post-boundary mesh：raw vertices=185316，post vertices=169986，使用 183 张 mask，mean shrink scale=0.9956。
- E3 的全图 RGB 指标很低，主要用于证明 preprocess-only 在全图 GT 口径下不可作为最终方法。
- 当前小矩阵中 E7 是最佳 RGB 重建组合；E8 完成 full 链路和 mesh 输出，但需继续分析 H-VQG retained list 对全测试集指标的影响。

## 2026-06-05/06 B0-B5 补跑记录

B2 `B2_foreground_track_init_only`：

- 运行时间：2026-06-05T23:40:52 到 2026-06-06T00:02:56。
- runner status：`success`。
- 前景初始化报告：177918 个 COLMAP 点筛到 118119 个前景初始点。
- full-frame：PSNR=21.6245，SSIM=0.8329，LPIPS=0.3499。
- foreground-object：PSNR_fg=22.5636，SSIM_fg=0.7966，LPIPS_fg=0.0604，外部非黑比例=0.9919，泄漏能量=1.1849，Gaussians=683429。

B4 `B4_mask_pruning_only`：

- 运行时间：2026-06-06T00:06:19 到 2026-06-06T00:26:28。
- runner status：`success`。
- pruning report：15000、17500、20000、22500、25000、27500、30000 iter 共 7 份，总移除 50274 个 Gaussian，全部由 mask score 触发。
- full-frame：PSNR=23.4466，SSIM=0.8744，LPIPS=0.3007。
- foreground-object：PSNR_fg=23.5861，SSIM_fg=0.8287，LPIPS_fg=0.0515，外部非黑比例=0.9908，泄漏能量=1.2018，Gaussians=689821。

结论：B2/B4 能提升或保持部分重建指标，但单独使用时仍没有形成前景对象；B3/B5 才把外部非黑比例压到约 3%、泄漏能量压到约 0.019。
