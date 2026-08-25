# Fig. 6 Assets

这个目录收集图 6 需要的原始素材。

## 面板对应

| 样本 | 输入图 | 前景掩膜 | 仅前景渲染 | TSDF 网格 |
|---|---|---|---|---|
| KongQueZhuYu | `01_kongquezhuyu_input_crop_0000.png` | `01_kongquezhuyu_mask_0000.png` | `01_kongquezhuyu_render_00000.png` | `01_kongquezhuyu_mesh_fuse_post.ply` |
| XianKeLai1 | `02_xiankelai1_input_crop_0000.png` | `02_xiankelai1_mask_0000.png` | `02_xiankelai1_render_00000.png` | `02_xiankelai1_mesh_fuse_post.ply` |
| CaoMei2 | `03_caomei2_input_crop_0000.png` | `03_caomei2_mask_0000.png` | `03_caomei2_render_00000.png` | `03_caomei2_mesh_fuse_post.ply` |

## 面板 b 数据

`fig6_metrics_current.csv`

如果只画 `PSNR_fg (dB)` 和 `Gaussians (10^5)`，用：

`fig6_panel_b_psnr_gaussians.csv`

## 说明

- `KongQueZhuYu` 使用原始 `03-final_locked` 场景。
- `XianKeLai1` 和 `CaoMei2` 使用 `04-sanitized_for_A6` 场景。
- 目前仓库里没有现成的局部误差放大图；panel c 需要从上面的 `render` 和 `fuse_post.ply` 再裁切/渲染生成。
