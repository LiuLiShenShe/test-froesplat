# M3：Mask-constrained Gaussian Optimization

## 1. 模块定位

M3 是新论文最重要的方法模块之一。它把 FSAM3 mask 从“图像预处理结果”升级为“2DGS 优化约束”。

原始路径：

```text
image -> mask preprocessing -> 2DGS
```

新路径：

```text
image + mask -> 2DGS RGB loss + mask consistency loss + background opacity loss
```

## 2. 核心目标

解决：

- background Gaussian competition
- background adhesion
- floating artifacts from background regions
- foreground outline mismatch

不解决：

- M3 不直接负责 mesh 边界收缩。
- M3 不直接负责 pruning，但会为 M4 提供更干净的 Gaussian field。

## 3. Loss 设计

总损失：

```text
L = L_rgb + lambda_mask * L_mask + lambda_bg * L_bg-opacity + lambda_normal * L_normal
```

其中：

```text
L_mask = || A_render - M_gt ||
L_bg-opacity = sum_{p in background} A_render(p)
```

符号说明：

- `A_render`：2DGS 渲染得到的 alpha/opacity map。
- `M_gt`：FSAM3 foreground mask。
- `L_mask`：约束渲染轮廓与植物 mask 一致。
- `L_bg-opacity`：惩罚背景区域不透明度，减少背景 Gaussian。

## 4. 实现接口

推荐参数：

```bash
--use_mask_loss
--use_bg_opacity_loss
--mask_dir <path>
--lambda_mask <float>
--lambda_bg <float>
--mask_loss_type {l1,bce,dice,l1_dice}
--mask_erode_px <int>
--mask_ignore_boundary_px <int>
```

默认：

```bash
--lambda_mask 0.0
--lambda_bg 0.0
```

默认不启用任何 mask loss。

## 5. 关键实现细节

### 5.1 mask 加载

要求：

- mask 文件名必须和训练图像可映射。
- mask 分辨率必须和渲染输出一致，或在加载时明确 resize。
- resize mask 时使用 nearest neighbor。
- mask 值统一为 `{0,1}` 或 `[0,1]` float。

### 5.2 alpha render

需要确认 2DGS renderer 能返回：

- RGB render
- alpha/opacity accumulation map
- depth
- normal

如果 renderer 当前不返回 alpha，需要最小侵入式增加返回项，并保证默认训练不受影响。

### 5.3 boundary ignore

mask 边界通常存在标注/分割误差。建议支持：

```text
--mask_ignore_boundary_px 0/2/4/8
```

做法：

- 对 mask 做 erode/dilate 得到 uncertain band。
- 在 uncertain band 内不计算 `L_mask` 或降低权重。

### 5.4 warm-up schedule

建议支持：

```bash
--mask_loss_start_iter <int>
--mask_loss_warmup_iters <int>
```

原因：

- 训练早期 Gaussian 尚未稳定，过强 mask 约束可能导致错误收缩。

## 6. 消融实验

最小必做：

| ID | Method | Mask preprocessing | Mask loss | BG opacity |
|---|---|---|---|---|
| M3-A0 | raw 2DGS | no | no | no |
| M3-A1 | FSAM3-preprocessed 2DGS | yes | no | no |
| M3-A2 | mask consistency only | yes | yes | no |
| M3-A3 | bg opacity only | yes | no | yes |
| M3-A4 | mask-constrained 2DGS | yes | yes | yes |

参数敏感性：

| lambda_mask | lambda_bg | PSNR | bg Gaussian ratio | leaf width MAPE |
|---:|---:|---:|---:|---:|
| 0.01 | 0.001 |  |  |  |
| 0.05 | 0.005 |  |  |  |
| 0.10 | 0.010 |  |  |  |

## 7. 指标

渲染：

- PSNR
- SSIM
- LPIPS

Gaussian 结构：

- Gaussian number
- background Gaussian ratio
- foreground opacity concentration
- floating artifact ratio

效率：

- train time
- mesh time
- GPU memory

表型：

- plant height MAE
- canopy width MAE
- leaf length MAE
- leaf width MAE/MAPE/Bias

## 8. 验收标准

M3 相比 preprocessing-only 至少需要在 2-3 项指标上提升：

- background Gaussian ratio 降低。
- floating artifacts 减少。
- mesh time 降低。
- trait MAE 降低。
- PSNR/SSIM/LPIPS 不明显下降。

硬性工程验收：

- 不传 `--use_mask_loss` 时 baseline 完全不需要 mask。
- 缺少 mask 时错误信息必须明确，不能静默跳过。
- `baseline+M3` 如果无 mask，必须提示需要 `--mask_dir` 或先运行 M2。

## 9. 风险与备选

风险：

- mask 错误会把真实叶片边缘压掉。
- lambda 过大会导致渲染质量下降。
- background opacity loss 可能误惩罚透明/细柄区域。

备选：

- 使用 eroded foreground 作为强监督区域，boundary band 降权。
- 使用 Dice + L1 混合 loss，避免前景/背景像素不均衡。
- 对 M3 只在训练中后期启用。

