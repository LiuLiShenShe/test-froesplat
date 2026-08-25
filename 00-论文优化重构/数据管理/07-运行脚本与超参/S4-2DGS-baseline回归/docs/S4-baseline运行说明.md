# S4 纯 2DGS baseline 运行说明

更新日期：2026-05-18

## 目标

S4 的目标是在不修改 M1-H-VQG、M2、M3、M4、M5 算法模块的前提下，先锁定纯 2DGS baseline。

当前策略：

1. 先跑 `KongQueZhuYu` 单场景 smoke，验证路径、环境、输出规范和指标链路。
2. smoke 成功后，在 GPU 空闲时用同一 runner 切换到 full baseline 配置。
3. 单场景 full baseline 成功后，再扩展到 2-3 个代表样本，最后批量全场景。

## baseline 约束

本阶段必须保持纯 baseline：

- 不使用 FSAM3/SAM mask。
- 不使用 M1-H-VQG 筛选。
- 不使用 M3 mask loss。
- 不使用 M4 pruning。
- 不使用 M5 edge-aware meshing。
- 输入使用 `数据管理/02-位姿COLMAP/03-final_locked/<sample>`。

## 当前配置

smoke 配置：

```text
数据管理/07-运行脚本与超参/S4-2DGS-baseline回归/configs/kongquezhu_smoke_baseline.json
```

full 配置：

```text
数据管理/07-运行脚本与超参/S4-2DGS-baseline回归/configs/kongquezhu_full_baseline.json
```

执行脚本：

```text
数据管理/07-运行脚本与超参/S4-2DGS-baseline回归/scripts/run_2dgs_baseline.py
```

说明：

- `test_iterations` 当前设置为 `[-1]`，用于跳过 `train.py` 内置 TensorBoard validation report。
- 原因是当前环境的 matplotlib `FigureCanvasAgg` 不再提供 `tostring_rgb`，会在训练末尾的深度图可视化处报错。
- 该设置不改变 2DGS 优化过程；指标统一在训练结束后通过 `render.py` 和 `metrics.py` 计算。
- 原仓库 `lpipsPyTorch/modules/lpips.py` 当前被日志文本污染，无法 import。因此 S4 使用独立脚本 `evaluate_rendered_metrics.py` 读取 `render.py` 已导出的 `renders/gt`，计算 PSNR、SSIM 和 pip 包 `lpips` 的 LPIPS。

## 输出位置

```text
数据管理/06-实验输出/KongQueZhuYu/<method_tag>/
├── baseline_guard.json
├── command.txt
├── config.json
├── config.yaml
├── cfg_args
├── logs/
├── run_status.json
├── point_cloud/
├── test/
├── results.json
└── per_view.json
```

## 已完成 smoke 记录

运行目录：

```text
数据管理/06-实验输出/KongQueZhuYu/E2_2dgs_baseline_smoke_20260517_194908/
```

运行结果：

| step | status | 说明 |
|---|---|---|
| train | success | 300 iter 完成，保存 `point_cloud/iteration_300/point_cloud.ply` |
| render | success | 导出 27 张 test renders 和对应 gt |
| metrics | recovered | 原 `metrics.py` 因 `lpipsPyTorch` 文件损坏失败，已用 `evaluate_rendered_metrics.py` 补齐 |

smoke 指标：

| iteration | PSNR | SSIM | LPIPS |
|---:|---:|---:|---:|
| 300 | 18.7228 | 0.7459 | 0.4008 |

注意：

- 该结果只用于验证流程，不作为论文正式 baseline 数值。
- smoke 已证明 `final_locked/KongQueZhuYu` 可被 2DGS 正常读取、训练、渲染和评测。

## 已完成 full baseline 记录

运行目录：

```text
数据管理/06-实验输出/KongQueZhuYu/E2_2dgs_baseline/
```

运行配置：

```text
数据管理/07-运行脚本与超参/S4-2DGS-baseline回归/configs/kongquezhu_full_baseline.json
```

运行结果：

| step | status | elapsed | 说明 |
|---|---|---:|---|
| train | success | 1168.287 s | 30000 iter 完成，保存 `iteration_7000` 和 `iteration_30000` |
| render | success | 49.465 s | 导出 27 张 test renders、gt 和 depth vis |
| metrics | success | 6.922 s | 使用 `evaluate_rendered_metrics.py` 计算 PSNR/SSIM/LPIPS |

full baseline 指标：

| iteration | PSNR | SSIM | LPIPS |
|---:|---:|---:|---:|
| 30000 | 24.1592 | 0.8880 | 0.2765 |

运行状态文件：

```text
数据管理/06-实验输出/KongQueZhuYu/E2_2dgs_baseline/run_status.json
```

结果文件：

```text
数据管理/06-实验输出/KongQueZhuYu/E2_2dgs_baseline/results.json
数据管理/06-实验输出/KongQueZhuYu/E2_2dgs_baseline/per_view.json
```

说明：

- 本次运行使用物理 GPU1，通过配置中的 `cuda_visible_devices=1` 指定。
- 该结果是 `KongQueZhuYu` 单场景正式 30k 纯 2DGS baseline，可作为后续同场景 M1-M5 消融参照。
- 这不代表全场景 baseline 已完成；下一步需要扩展到 2-3 个代表样本，然后再批量全场景。
