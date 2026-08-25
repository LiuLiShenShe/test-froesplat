# S16 M1 Soft Weighting on A6 说明

## 目的

S14/S15 已证明 retained-list hard filtering 会破坏 `KongQueZhuYu` 的 foreground-object reconstruction。S16 改为 soft view weighting：不删除训练/测试视角，只在训练时降低低质量视角的 RGB loss 权重。

## 策略

```text
A6 + M1-soft = A6 foreground-object objective
             + full view coverage
             + per-view RGB loss weight
```

当前实现：

- `view_weight_mode=rgb_only`：只加权 foreground RGB reconstruction loss；
- mask loss / bg opacity loss 默认不加权，避免削弱 foreground-only 分离约束；
- `view_weight_list` 使用 `image_name,weight` CSV；
- 默认权重范围为 `[0.6, 1.0]`，避免低分视角被等价删除。

## 相关文件

```text
scripts/build_hvqg_view_weights.py
configs/kongquezhu_A6_M1_soft_weighting_smoke.json
configs/kongquezhu_A6_M1_soft_weighting.json
```

权重输出：

```text
数据管理/05-评测结果/S16_M1_soft_weighting_on_A6/kongquezhu_hvqg_soft_view_weights.csv
```

## 验收

第一步只做 120 iter smoke：

- 训练入口可读取 `view_weight_list`；
- 不触发 raw/mask/geo retained-list filtering；
- `baseline_guard.json` 标记 `uses_view_weighting=true`；
- 输出目录和日志可追溯。

## 当前 smoke 结果

2026-05-20 已完成 `A6_M1_soft_weighting_smoke`：

- 输入相机：读取 210/210，没有启用 raw/mask/geo retained-list filtering；
- foreground init：保留 `118119/177918` 点；
- soft view weights：成功读取 210 个视角权重；
- 权重范围：`0.846018` 到 `0.941009`，均值 `0.892151`；
- 120 iter 训练完成并保存 checkpoint；
- `baseline_guard.json` 已标记 `uses_view_weighting=true`。

输出目录：

```text
数据管理/06-实验输出/KongQueZhuYu/A6_M1_soft_weighting_smoke/
```

## 正式 30k 结果

2026-05-21 已完成 `A6_M1_soft_weighting_20260521_110854`：

- 30k 训练完成；
- render 完成；
- full-frame metrics 完成；
- foreground-object eval 完成；
- eval images：`27/27`，未删除测试视角；
- final Gaussians：`532264`。

输出目录：

```text
数据管理/06-实验输出/KongQueZhuYu/A6_M1_soft_weighting_20260521_110854/
```

### Full-frame metrics

| Variant | Full PSNR | Full SSIM | Full LPIPS |
|---|---:|---:|---:|
| A6+M1-soft weighting | 6.2539 | 0.2681 | 0.5567 |

Full-frame metrics 只作为流程完整性检查，不作为 foreground-object reconstruction 的核心评价依据。

### Foreground-object metrics

| Variant | Eval images | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians |
|---|---:|---:|---:|---:|---:|---:|---:|
| A6+M1-soft weighting | 27 | 24.9566 | 0.8543 | 0.0440 | 0.0284 | 0.0184 | 532264 |

结果文件：

```text
数据管理/06-实验输出/KongQueZhuYu/A6_M1_soft_weighting_20260521_110854/foreground_object_results.json
数据管理/05-评测结果/S16_M1_soft_weighting_on_A6/kongquezhu_A6_M1_soft_weighting_summary.md
```

## 阶段结论

S16 与 A6 基本持平，并通过 foreground-only 阈值：

```text
outside < 0.05
leakage < 0.10
```

相对 A6，S16 的 `PSNR_fg` 仅下降 `0.0506 dB`，`SSIM_fg` 仅下降 `0.0005`，`LPIPS_fg` 基本不变；同时 `outside` 和 `leakage` 略低，Gaussian 数从 `591623` 降到 `532264`。

因此，S16 证明 soft view weighting 没有重现 hard/reject-only retained-list 的覆盖破坏问题。当前 M1 结论应更新为：retained-list hard filtering 是负证据；soft weighting 是更合理的 M1 方向，可作为保持多视角覆盖前提下的视角质量调节策略。
