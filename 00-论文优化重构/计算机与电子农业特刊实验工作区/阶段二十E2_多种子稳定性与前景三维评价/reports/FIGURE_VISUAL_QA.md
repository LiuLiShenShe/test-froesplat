# Figure Visual QA

复核日期：2026-10-15（Asia/Shanghai）。本记录针对最终图像刷新，属于显示层检查，不替代数值验收。

通过当前模型的 `functions.view_image` 能力 directly inspected 以下实际生成的本地图片：

- `figures/WanNianQing1_0000_RGB.png`
- `figures/WanNianQing1_0000_BC_ABSOLUTE_DIFFERENCE.png`
- `figures/WanNianQing1_0000_DEPTH.png`
- `figures/CaoMei2_crop_0095_NORMAL.png`

RGB 图中的 canonical raw RGB、固定 R0 proxy ROI（明确标注 non-GT）及 B/C 三种子面板均完整显示。黑背景来自冻结模型的真实渲染。差值图三种子的共同色阶仍为 `[0, 0.1]`；缩短后的 `Mean absolute RGB error` 色条标签完整可见，未观察到标签裁切。

Depth 与 native normal 图的标题、种子标签和六模型面板完整显示。Depth 色条继续标注 inherited arbitrary COLMAP units 与共享 1%–99% 显示范围。外部深度及法线结构保留在图中，仅为描述性结果；不能据此将全部外部连通区域判定为真实 floating Gaussian，也不能推导独立几何 GT 下的质量结论。

刷新仅缩短差值色条的文字标签；真实 float 张量、色阶范围和评价 CSV 不变。57 个索引图的存在性与 SHA256 已由 `scripts/refresh_figures.py` 检查，记录见 `reports/FIGURE_REFRESH_PROGRESS.json`。显示 PNG 不参与正式指标计算；正式指标仍以真实 float 渲染的重算结果为准。

本次按展示完整性抽查四张图，不声称逐张人工审核全部图片。首次尝试打开不存在的 `CaoMei2_crop_0124_NORMAL.png` 未得到图像，随后依据实际文件索引查看了上述 `crop_0095` 文件；未对不存在的图片作任何内容判断。
