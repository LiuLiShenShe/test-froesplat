# Foreground-object 导出与评测记录

## 目标口径

本分支的目标是导出 **mask foreground object**，不是 leaf-only，也不是去花盆后的纯叶片植株。

因此，只要某部分被 SAM/FSAM mask 标为前景，就属于当前分支保留对象。对 `KongQueZhuYu` 来说，前景包含叶片、花盆、盆贴标以及 mask 标出的其他前景小物体。

## 已确认事实

- mask 可以被读取，命名 `mask_{stem}.png` 与 `cameras.json` 的 `img_name` 对齐。
- 训练图像尺寸为 `2120x3791`，mask 原始尺寸为 `2160x3840`，代码会用 nearest resize 到相机尺寸。
- overlay 诊断图显示 mask 与 RGB 基本对齐，诊断图位于：

```text
数据管理/06-实验输出/KongQueZhuYu/E7_mask_pruning/plant_only/diagnostics/
```

## E7 foreground-object 导出

输入：

```text
数据管理/06-实验输出/KongQueZhuYu/E7_mask_pruning/point_cloud/iteration_30000/point_cloud.ply
```

导出产物：

```text
数据管理/06-实验输出/KongQueZhuYu/E7_mask_pruning/plant_only/
├── plant_only_gaussians.ply
├── plant_only_report.json
└── plant_only_scores.npz
```

当前参数：

- `min_observations=3`
- `foreground_ratio_threshold=0.35`
- `dilate_mask_px=5`
- 210 张 mask 全部参与投票

当前导出结果：

- 输入 Gaussian：694992
- 保留 Gaussian：340413
- 移除 Gaussian：354579
- 保留比例：48.98%

## 可渲染评测目录

为了对 foreground-object PLY 做渲染和 eval，已将导出的 PLY 包装为独立 model 目录：

```text
数据管理/06-实验输出/KongQueZhuYu/E7_mask_pruning_foreground_object/
├── point_cloud/iteration_30000/point_cloud.ply
├── test/ours_30000/renders/
├── test/ours_30000/gt/
├── results.json
├── foreground_object_results.json
└── foreground_object_eval_summary.json
```

## Eval 结果

full-frame 口径：

```text
PSNR=11.0135
SSIM=0.6393
LPIPS=0.4298
```

说明：full-frame 会惩罚被移除的背景，因此不适合作为 foreground-object 质量主指标。

mask foreground-object 口径：

```text
PSNR_fg=24.6918
SSIM_fg=0.8658
LPIPS_fg_black_bg=0.0449
mask_ratio_mean=0.3018
num_images=27
```

说明：`LPIPS_fg_black_bg` 是将 render/GT 都按 mask 置黑背景后的 LPIPS，不与原始全图 LPIPS 直接横向比较。

## 后续注意

- 若论文目标是 “mask foreground object reconstruction”，当前分支口径成立。
- 若目标改为 “leaf-only / target-plant-only”，则需要额外做 target-mask refinement。
- 当前导出仍是 Gaussian 中心点多视角投票，可能保留少量被遮挡背景；如 foreground-object 可视化仍有明显背景，可继续增加可见性约束或调高 `foreground_ratio_threshold`。
