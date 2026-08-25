# Fig. 4 Assets

这个目录是为了画图 4 专门整理出来的小目录。Draw.io 里优先从这里插图，不用再去 `06-实验输出` 里找。

目录：

`00-论文优化重构/论文2026-05-28/figures/fig4_assets/`

## Panel a 替换清单

mockup 的第一行 `Render` 用这些文件：

| mockup 列 | 文件名 | 含义 |
|---|---|---|
| Baseline | `a01_baseline_A0_full_scene.png` | A0 标准整场景 2DGS |
| Input-mask | `a02_input_mask_A1.png` | A1 输入域 mask preprocess |
| Alpha reg. | `a03_alpha_reg_A2.png` | A2 仅 alpha mask loss |
| FG RGB sup. | `a04_fg_rgb_sup_A5.png` | A5 前景 RGB + alpha/bg |
| + FG-track init | `a05_fg_track_init_A6.png` | A6 前景轨迹初始化 + A5 |
| Post-hoc prune | `a06_post_hoc_prune_E7.png` | E7 后验 mask pruning/export |

mockup 的第二行不要写 `TSDF mesh`，改成：

`Gaussian representation`

或：

`Gaussian point cloud`

因为 A0-A6 现在没有每列对应的 TSDF mesh，只有 Gaussian point cloud。我已经把第二行可直接插入的 PNG 渲出来了：

| mockup 列 | 第二行文件名 | 含义 |
|---|---|---|
| Baseline | `g01_baseline_A0_gaussian.png` | A0 整场景 Gaussian，背景点云明显 |
| Input-mask | `g02_input_mask_A1_gaussian.png` | A1 输入掩膜后的 Gaussian |
| Alpha reg. | `g03_alpha_reg_A2_gaussian.png` | A2 alpha 正则，仍有大量背景点 |
| FG RGB sup. | `g04_fg_rgb_sup_A5_gaussian.png` | A5 前景 RGB 后的 Gaussian |
| + FG-track init | `g05_fg_track_init_A6_gaussian.png` | A6 前景初始化后的 Gaussian |
| Post-hoc prune | `g06_post_hoc_prune_E7_gaussian.png` | E7 后验剪枝导出，仍有背景残留 |

快速预览：

`panel_a_second_row_gaussian_contact_sheet.png`

## Panel c 局部放大

推荐两列：

| mockup 位置 | 文件 |
|---|---|
| Without FG RGB sup. | `optional_alpha_bg_A4.png` |
| With FG RGB sup. | `a04_fg_rgb_sup_A5.png` |

如果 A4/A5 局部差异不够明显，可以改成：

| mockup 位置 | 文件 |
|---|---|
| Without FG RGB sup. | `a01_baseline_A0_full_scene.png` |
| With FG RGB sup. | `a04_fg_rgb_sup_A5.png` |

## Panel d 替换清单

| mockup 位置 | 文件 | 含义 |
|---|---|---|
| Post-hoc prune | `a06_post_hoc_prune_E7.png` | 后验剪枝仍有明显残留 |
| Foreground-object optimization | `a05_fg_track_init_A6.png` | 训练期前景目标，干净黑底 |

如果你想放完整配置参考，可以把右侧换成：

`optional_A6_plus_M4_F1.png`

但主文更推荐用 `a05_fg_track_init_A6.png`，因为它和表 3 的主消融行完全一致。

## Panel b 数据

柱状图数据在：

`fig4_data_table_current.csv`

优先画三组：

| label | outside_nonblack | leakage |
|---|---:|---:|
| Baseline / A0 | 0.9908 | 1.2201 |
| +FG-track init / A6 | 0.0294 | 0.0189 |
| Post-hoc prune / E7 | 0.7509 | 0.7900 |

阈值线：

- outside threshold = 0.05
- leakage threshold = 0.10

## 原始来源路径

这些图片来自：

`00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/<method>/test/ours_30000/renders/00000.png`

对应 method：

- `E2_2dgs_baseline`
- `E3_fsam3_preprocess`
- `A2_alpha_mask_loss_only`
- `A5_fg_rgb_alpha_bg_loss`
- `A6_foreground_track_init_fg_rgb_alpha_bg`
- `E7_mask_pruning_foreground_object`
- `A3_bg_opacity_only`
- `A4_alpha_mask_bg_opacity`
- `F1_high_precision_foreground`
