# Fig. 3a Qualitative Comparison Placement Guide

资产目录：

`00-论文优化重构/论文2026-05-28/figures/fig3a_rendered_assets/`

总览检查图：

`fig3a_front_view_contact_sheet.png`

## 放置规则

这张图是 3 行 × 4 列。

列从左到右：

1. COLMAP
2. 3DGS-FSAM3
3. Standard 2DGS
4. ForeSplat

行从上到下：

1. Novel view：同一个正面测试视角的完整新视角渲染或对应真实图
2. Fore-grnd：前景约束或前景化结果
3. TSDF mesh：重建几何结果，统一用正面相机方向

## 每个格子放哪张图

| 行 | COLMAP | 3DGS-FSAM3 | Standard 2DGS | ForeSplat |
|---|---|---|---|---|
| Novel view | `colmap_row1_gt_front_view.png` | `3dgs_fsam3_row1_e3_scene_render.png` | `standard_2dgs_row1_scene_render.png` | `foresplat_row1_scene_render.png` |
| Fore-grnd | `colmap_row2_gt_front_view.png` | `3dgs_fsam3_row2_e3_foreground_render.png` | `standard_2dgs_row2_foreground_masked_render.png` | `foresplat_row2_foreground_render.png` |
| TSDF mesh | `colmap_row3_fuse_pointcloud.png` | `3dgs_fsam3_row3_sugar_textured_mesh.png` | `standard_2dgs_row3_tsdf_mesh.png` | `foresplat_row3_tsdf_mesh.png` |

## 逐格说明

### Novel view

COLMAP：放 `colmap_row1_gt_front_view.png`。COLMAP 本身没有可比的 neural novel-view render，所以这里放同视角真实图，作为传统 SfM/MVS 的输入参考。

3DGS-FSAM3：放 `3dgs_fsam3_row1_e3_scene_render.png`。这是 FSAM3 preprocess 分支在 00000 正面视角下的渲染结果。

Standard 2DGS：放 `standard_2dgs_row1_scene_render.png`。这是标准 2DGS 在同一个 00000 正面视角下的完整渲染。

ForeSplat：放 `foresplat_row1_scene_render.png`。这是 ForeSplat A6_M1_soft_M4 在同一个 00000 正面视角下的完整渲染。

### Fore-grnd

COLMAP：放 `colmap_row2_gt_front_view.png`。COLMAP 没有 foreground-only 输出，所以这里重复同视角真实图，表示传统方法没有显式前景建模结果。图注里可以注明 “N/A for explicit foreground prior” 或中文 “无显式前景先验”。

3DGS-FSAM3：放 `3dgs_fsam3_row2_e3_foreground_render.png`。它和 row1 同源，但黑背景体现了 FSAM3 preprocess 后的前景化输入/结果。

Standard 2DGS：放 `standard_2dgs_row2_foreground_masked_render.png`。这是用 FSAM3 mask 对标准 2DGS 渲染做的前景裁剪，用来对比没有前景建模时的残留与不完整。

ForeSplat：放 `foresplat_row2_foreground_render.png`。这是 ForeSplat foreground-only 分支的正面视角结果，是这一行的重点对比对象。

### TSDF mesh

COLMAP：放 `colmap_row3_fuse_pointcloud.png`。这是 COLMAP stereo fusion 的点云结果，不是 TSDF mesh，但放在这一行作为传统几何重建基线。

3DGS-FSAM3：放 `3dgs_fsam3_row3_sugar_textured_mesh.png`。这是 3DGS/FSAM3 对应的 SuGaR textured mesh 正面渲染。当前 native 3DGS renderer 因 CUDA extension 缺失未直接导出，所以这一格用 SuGaR mesh 作为几何结果。

Standard 2DGS：放 `standard_2dgs_row3_tsdf_mesh.png`。这是标准 2DGS 的 `fuse_post.ply` 正面渲染。

ForeSplat：放 `foresplat_row3_tsdf_mesh.png`。这是 ForeSplat A6_M1_soft_M4 的 `fuse_post.ply` 正面渲染，是这一行的重点结果。

## Draw.io 里怎么放

1. 先打开 `fig3a_front_view_contact_sheet.png` 对照整体效果。
2. 对每个虚线框执行 Insert Image，选择上表对应的 PNG。
3. 勾选或保持原图比例，不要拉伸变形。
4. 图片放进虚线框后，使用 center crop 或等比例缩放到填满框。
5. Novel view 和 Fore-grnd 两行是竖图，建议让图像高度贴近虚线框高度，左右留少量白边。
6. TSDF mesh 行是白底几何渲染，建议让主体居中，底部蓝色标记保留可见。
7. 不要使用 `_deprecated/` 里的旧图。里面是上一轮俯视 overlay 和空白 sparse 替代图。

## 推荐图注口径

“All method outputs are shown from the same frontal test view. COLMAP is shown with the aligned input/reference view and stereo-fusion point cloud because it does not produce neural foreground-only rendering or TSDF mesh in the same pipeline.”

