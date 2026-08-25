# Fig. 4 Data And Layout Plan

图 4 的主问题：

训练期定义“仅植物重建目标”是否必要；或者说，整场景 2DGS 训练后再做 mask/post-hoc pruning 能否等价替代 ForeSplat 的前景对象优化。

## 数据范围

当前图 4 的可靠数据主要来自 `KongQueZhuYu` 一个主样本，不是跨样本统计图。

主数据源：

`00-论文优化重构/数据管理/05-评测结果/KongQueZhuYu/S10_small_matrix/kongquezhu_A0_A6_ablation_summary.csv`

补充后验剪枝数据源：

`00-论文优化重构/数据管理/05-评测结果/KongQueZhuYu/S10_small_matrix/kongquezhu_foreground_object_summary.csv`

我整理出的当前表：

`00-论文优化重构/论文2026-05-28/figures/fig4_data_table_current.csv`

渲染候选总览：

`00-论文优化重构/论文2026-05-28/figures/fig4_candidate_renders_contact_sheet.png`

## 可直接使用的实验行

| 图中简称 | 实验目录 | 含义 | 可用于 |
|---|---|---|---|
| A0 full-scene | `E2_2dgs_baseline` | 标准整场景 2DGS | 图 4a, b, c |
| A1 input mask | `E3_fsam3_preprocess` | 输入域前景掩膜训练，质量下降但外部低 | 图 4b 或补充图 |
| A2 alpha | `A2_alpha_mask_loss_only` | 仅 alpha mask loss | 图 4a, b |
| A3 bg opacity | `A3_bg_opacity_only` | 仅背景不透明度抑制 | 图 4a, b |
| A4 alpha+bg | `A4_alpha_mask_bg_opacity` | alpha + bg opacity | 图 4a, b |
| A5 fg RGB | `A5_fg_rgb_alpha_bg_loss` | 前景 RGB + alpha/bg，关键成功项 | 图 4a, b, c |
| A6 fg init+fg RGB | `A6_foreground_track_init_fg_rgb_alpha_bg` | 前景初始化 + A5 | 图 4a, b |
| A6+M4/F1 | `F1_high_precision_foreground` | A6 + lightweight mask pruning | 可作完整配置参考 |
| E7 post-hoc current | `E7_mask_pruning_foreground_object` | 当前可复现的后验 foreground object export | 图 4d |

## 关键指标

来自 `fig4_data_table_current.csv`。

| 方法 | PSNR_fg | SSIM_fg | LPIPS_fg | outside_nonblack | leakage | Gaussians |
|---|---:|---:|---:|---:|---:|---:|
| A0 full-scene | 24.2090 | 0.8514 | 0.0480 | 0.9908 | 1.2201 | 751,213 |
| A1 input mask | 20.7291 | 0.7505 | 0.0696 | 0.0073 | 0.0042 | 263,108 |
| A2 alpha | 24.3422 | 0.8478 | 0.0491 | 0.9898 | 1.2260 | 768,067 |
| A3 bg opacity | 24.7508 | 0.8672 | 0.0451 | 0.9900 | 1.2255 | 742,931 |
| A4 alpha+bg | 24.8126 | 0.8687 | 0.0445 | 0.9896 | 1.2266 | 763,266 |
| A5 fg RGB | 25.1055 | 0.8561 | 0.0437 | 0.0294 | 0.0190 | 592,900 |
| A6 fg init+fg RGB | 25.0072 | 0.8548 | 0.0438 | 0.0294 | 0.0189 | 591,623 |
| A6+M4/F1 | 24.9723 | 0.8540 | 0.0438 | 0.0293 | 0.0186 | 585,594 |
| E7 post-hoc current | 24.6918 | 0.8658 | 0.0449 | 0.7509 | 0.7900 | NA |

## Important consistency note

当前 `ForeSplat_zh.md` 表 3 中写的“整场景训练后验掩膜剪枝：PSNR_fg=21.34, outside=0.31, leakage=0.28”没有在当前 `S10_small_matrix` 汇总表里找到对应数据。

当前可复现的后验导出数据是：

- `E7_mask_pruning_foreground_object`
- PSNR_fg = 24.6918
- SSIM_fg = 0.8658
- LPIPS_fg = 0.0449
- outside_nonblack = 0.7509
- leakage = 0.7900

如果继续使用 0.31 / 0.28，需要找到对应原始评测文件或脚本输出；否则建议稿件中改成当前可追溯数值，避免图 4、表 3 和数据源不一致。

## 建议图 4 panel

### Fig. 4a: Visual ablation grid

建议放 6 列：

1. A0 full-scene
2. A2 alpha
3. A3 bg opacity
4. A4 alpha+bg
5. A5 fg RGB
6. A6 fg init+fg RGB

每列放同一个 00000 正面渲染：

`test/ours_30000/renders/00000.png`

图像信息清楚：

- A0/A2/A3/A4 都保留背景、花盆和桌面
- A5/A6 为黑底前景，视觉上支持“前景 RGB 监督是转折点”

### Fig. 4b: Leakage bar chart

用 `fig4_data_table_current.csv` 画两个指标：

- outside_nonblack_ratio_mean
- leakage_energy_ratio_mean

建议只放 A0-A6，另用 dashed line 标：

- outside threshold = 0.05
- leakage threshold = 0.10

A1 input mask 可以保留，但要注明 “foreground clean but quality degraded”。

### Fig. 4c: Local crop before/after foreground RGB

最干净的对比：

- A0 full-scene vs A5 fg RGB
- 或 A4 alpha+bg vs A5 fg RGB

建议裁切叶缘/花盆边界/桌面区域，展示同一视角 00000 中背景泄漏从整场景转为黑底前景。

### Fig. 4d: Post-hoc pruning vs training-time foreground objective

建议放：

1. A0 full-scene
2. E7 post-hoc current
3. A6 or A6+M4

E7 的 00000 渲染有明显背景残留与伪影，很适合作为“后验导出不等价”的视觉证据。

对应数值应使用当前可追溯的 E7：

- outside = 0.7509
- leakage = 0.7900

## 可用视觉文件路径

所有方法的 00000 渲染图都在：

`00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/<method>/test/ours_30000/renders/00000.png`

推荐方法目录：

- `E2_2dgs_baseline`
- `A2_alpha_mask_loss_only`
- `A3_bg_opacity_only`
- `A4_alpha_mask_bg_opacity`
- `A5_fg_rgb_alpha_bg_loss`
- `A6_foreground_track_init_fg_rgb_alpha_bg`
- `E7_mask_pruning_foreground_object`
- `F1_high_precision_foreground`

## 当前缺口

1. A0-A6 多数只有 Gaussian `point_cloud.ply` 和渲染图，没有 TSDF mesh。
2. 真正现成的 TSDF mesh 主要在 `A6_M1_soft_M4/train/ours_30000/fuse_post.ply`，不适合直接作为 A0-A6 每列 mesh 消融。
3. 因此图 4a 建议写“rendering and foreground-object evidence”，不要承诺每个消融都有 mesh。
4. 如果图注继续写“渲染和网格可视化”，需要额外为 A0-A6 批量导出 mesh 或改图注。

## Mockup 逐格替换清单

你发来的 mockup 可以保留版式，但植物图必须换成 `KongQueZhuYu` 的真实输出。建议不要再使用 mockup 里的玉米/盆栽示意。

### Panel a: Qualitative comparison of training objectives

mockup 现在有 6 列：

1. Baseline
2. Input-mask
3. Alpha reg.
4. FG RGB sup.
5. + FG-track init
6. Post-hoc prune

建议替换为：

| mockup 列 | 我们的实验 | 放置文件 |
|---|---|---|
| Baseline | A0 full-scene | `00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/E2_2dgs_baseline/test/ours_30000/renders/00000.png` |
| Input-mask | A1 input mask | `00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/E3_fsam3_preprocess/test/ours_30000/renders/00000.png` |
| Alpha reg. | A2 alpha mask loss | `00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A2_alpha_mask_loss_only/test/ours_30000/renders/00000.png` |
| FG RGB sup. | A5 foreground RGB | `00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A5_fg_rgb_alpha_bg_loss/test/ours_30000/renders/00000.png` |
| + FG-track init | A6 foreground init + fg RGB | `00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A6_foreground_track_init_fg_rgb_alpha_bg/test/ours_30000/renders/00000.png` |
| Post-hoc prune | E7 post-hoc export | `00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/E7_mask_pruning_foreground_object/test/ours_30000/renders/00000.png` |

如果需要把 A3/A4 也画进去，就把 `Input-mask` 移到补充图或图 4b，只在 Panel a 里保留：

`Baseline / Alpha reg. / Bg opacity / Alpha+Bg / FG RGB sup. / +FG-track init / Post-hoc prune`

对应 A3/A4 的文件：

- A3 bg opacity: `00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A3_bg_opacity_only/test/ours_30000/renders/00000.png`
- A4 alpha+bg: `00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A4_alpha_mask_bg_opacity/test/ours_30000/renders/00000.png`

### Panel a 的 TSDF mesh 行怎么处理

mockup 里第二行写的是 `TSDF mesh`，但我们现在 A0-A6 每列没有现成 TSDF mesh，只有 Gaussian `point_cloud.ply`。所以有两个选择：

**推荐选择 A：把这一行标题改成 `Gaussian representation` 或 `Gaussian point cloud`。**

这样可以用每个方法的：

`point_cloud/iteration_30000/point_cloud.ply`

后续再统一渲成灰色或彩色 Gaussian 点云。这个和现有数据最一致。

**选择 B：保留 `TSDF mesh`，但只在 ForeSplat 完整配置列放 mesh。**

可用 mesh：

`00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/A6_M1_soft_M4/train/ours_30000/fuse_post.ply`

这种做法不推荐用于 Panel a，因为会让读者以为每个消融都做了 TSDF mesh。

### Panel b: Background leakage metrics

mockup 中的柱状图应换成当前数据：

优先画三组：

| label | outside_nonblack | leakage |
|---|---:|---:|
| Baseline | 0.9908 | 1.2201 |
| +FG-track init / A6 | 0.0294 | 0.0189 |
| Post-hoc prune / E7 current | 0.7509 | 0.7900 |

如果要强调“前景 RGB 是转折点”，可以画 A0-A6 全部：

- A0: 0.9908 / 1.2201
- A1: 0.0073 / 0.0042, 但 PSNR_fg 降到 20.7291
- A2: 0.9898 / 1.2260
- A3: 0.9900 / 1.2255
- A4: 0.9896 / 1.2266
- A5: 0.0294 / 0.0190
- A6: 0.0294 / 0.0189

注意：mockup 里的 post-hoc prune `0.31 / 0.28` 不建议直接沿用，除非找到对应原始评测文件。当前可追溯 E7 是 `0.7509 / 0.7900`。

### Panel c: Local zoom

mockup 的两列 `Without FG RGB sup.` / `With FG RGB sup.` 可以换成：

| mockup 位置 | 我们的图 |
|---|---|
| Without FG RGB sup. | A4 alpha+bg: `A4_alpha_mask_bg_opacity/test/ours_30000/renders/00000.png` |
| With FG RGB sup. | A5 fg RGB: `A5_fg_rgb_alpha_bg_loss/test/ours_30000/renders/00000.png` |

裁切建议：

1. 上排裁盆和桌面边界区域，展示背景/花盆残留。
2. 下排裁叶缘区域，展示叶缘保留而非背景泄漏。

如果 A4 与 A5 的局部差异不够直观，可以用 A0 vs A5。

### Panel d: Post-hoc pruning vs training-time foreground-object optimization

mockup 的 `Post-hoc prune` 应换成：

`00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/E7_mask_pruning_foreground_object/test/ours_30000/renders/00000.png`

mockup 的 `Foreground-object optimization` 应换成二选一：

1. 训练期前景目标核心配置：`A6_foreground_track_init_fg_rgb_alpha_bg/test/ours_30000/renders/00000.png`
2. 完整 ForeSplat 配置参考：`F1_high_precision_foreground/test/ours_30000/renders/00000.png`

我建议用 A6，因为它和表 3 的主消融完全一致；F1/A6+M4 可以留给后续紧凑化或完整配置图。

### Caption 也要同步改

如果 Panel a 第二行不再是 mesh，而是 Gaussian 点云，图注应改为：

`a，不同训练目标的正面渲染和 Gaussian 表示可视化，显示背景、花盆和桌面是否进入最终表示。`

不要继续写“TSDF mesh”，除非已经为每个消融导出了对应 mesh。
