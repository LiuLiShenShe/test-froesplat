# 图表需求规格文档 — Plant-aware 2DGS 论文

## Figure Specifications (8 figures)

### Fig. 1: Task Reformulation Concept
- **类型:** 概念对比图
- **内容:** 左侧：Full-scene 2DGS 输出（植物+花盆+背景+桌面全可见）；右侧：Plant-aware 2DGS 输出（仅植物前景，背景压黑）
- **视觉元素:** 渲染图对比 + 背景泄漏热力图叠加
- **Caption:** "From full-scene to foreground-object 2D Gaussian Splatting for plant phenotyping. (a) Standard 2DGS reconstructs all visible content including pots, background, and supports. (b) Plant-aware 2DGS (Ours-core) restricts reconstruction to the mask-defined plant foreground object."

### Fig. 2: FSAM3 Pipeline Architecture
- **类型:** 流程图
- **内容:** Raw frames → FFT quality screening → SAM3 promptable segmentation (展示5个prompt) → PCA main-component refinement → Output masks (binary, alpha, foreground RGB)
- **视觉元素:** 每个阶段的示例输入/输出缩略图
- **Caption:** "FSAM3: Frequency-Spatial plant mask prior pipeline. Stage 1: FFT-based frequency-domain frame quality screening removes blurry frames. Stage 2: SAM3 text-prompted segmentation extracts plant foreground (five prompts evaluated). Stage 3: PCA-guided main-component refinement suppresses false-positive fragments."

### Fig. 3: Plant-aware 2DGS Method Overview
- **类型:** 系统架构图
- **内容:** 完整pipeline: Multi-view RGB + FSAM3 masks → COLMAP SfM → Foreground track init → Foreground-object 2DGS optimization (L_rgb_fg + L_mask + L_bg) → M1-soft view weighting → M4 compact cleanup → TSDF mesh extraction → Phenotype measurement
- **视觉元素:** 每个模块标注其核心贡献
- **Caption:** "Overview of Plant-aware 2DGS framework. Orange modules are our contributions; gray modules are standard components."

### Fig. 4: A0-A6 Ablation Visual Comparison
- **类型:** 渲染对比图
- **内容:** 4个子图 (A0, A1, A5, A6)，每个包含：(上排)测试视角RGB渲染，(下排)背景泄漏热力图
- **数据来源:** KongQueZhuYu 测试集
- **Caption:** "Ablation study: foreground-object reconstruction quality and background leakage. A0: full-scene 2DGS (high leakage). A1: mask preprocessing (clean but poor quality). A5: foreground RGB supervision (clean with good quality). A6: Ours-core (complete method with foreground track initialization)."

### Fig. 5: M1 View Quality Strategy — Hard vs Soft
- **类型:** 对比图
- **内容:** 左侧：M1-hard 视角覆盖图（红色标记缺失视角）；右侧：M1-soft 视角权重分布热力图；底部：PSNR_fg + Gaussian count 柱状对比 (A6, M1-hard, M1-reject-only, M1-soft)
- **数据来源:** KongQueZhuYu
- **Caption:** "Hard view filtering damages multi-view coverage; soft weighting preserves reconstruction quality. (a) M1-hard removes 10/27 views, creating angular coverage gaps. (b) M1-soft preserves all views while modulating their loss contribution. (c) Quantitative comparison."

### Fig. 6: Ours-core vs Ours-full Cross-sample Comparison
- **类型:** 分组柱状图
- **内容:** 3个样本 × 4个指标 (Gaussian count, PSNR_fg, outside_nonblack, leakage_energy)，每组2个柱子 (Ours-core 蓝色, Ours-full 橙色)
- **数据来源:** KongQueZhuYu, XianKeLai1, CaoMei2
- **Caption:** "Ours-full reduces Gaussian count while maintaining foreground quality across three representative plant architectures. Bars show Ours-core (A6, blue) and Ours-full (A6+M1-soft+M4, orange)."

### Fig. 7: Mesh Structural Evaluation
- **类型:** 网格可视化对比图
- **内容:** 2列 (KongQueZhuYu, XianKeLai1) × 3行 (Standard TSDF, Smaller truncation, Post-boundary)，边界区域放大插图
- **数据来源:** S19 mesh 输出
- **Caption:** "Mesh structural comparison across TSDF variants. Smaller truncation produces more compact but more fragmented meshes. Post-boundary cleanup preserves connected components while adjusting boundary edges. Zoom-in insets highlight boundary quality differences."

### Fig. 8: Manual-vs-Virtual Phenotype Validation
- **类型:** 2×2 散点图
- **内容:** (a) Plant height, (b) Canopy width, (c) Leaf length, (d) Leaf width。y=x参考线(灰色)，标注R²和n值。可选：叶宽Bland-Altman插图
- **数据来源:** 植株数据.xlsx (21 plants, 10 species)
- **Caption:** "Manual vs. virtual phenotype measurement across 21 plants from 10 species. Each point represents one measurement. Gray line: y=x perfect agreement. Leaf width (d) shows the largest deviation (MAPE = 9.73%)."

---

## Table Specifications (7 tables)

### Table 1: Dataset summary
- **列:** Sample ID | Species (CN) | Raw frames | FFT-retained | Retention% | Scene type | COLMAP registered | FSAM3 masks | Manual GT | Usage
- **行:** 20 (S01-S20)
- **数据来源:** dataset_summary.md + FFT screening logs

### Table 2: A0-A6 foreground-object objective ablation (KongQueZhuYu)
- **列:** ID | Description | fg_init | fg_rgb_loss | alpha_mask_loss | bg_opacity_loss | PSNR_fg↑ | SSIM_fg↑ | LPIPS_fg↓ | outside↓ | leakage↓ | Gaussians↓ | FG-only?
- **行:** A0, A1, A2, A3, A4, A5, A6

### Table 3: Ours-core (A6) cross-sample validation
- **列:** Sample | Architecture | PSNR_fg↑ | SSIM_fg↑ | LPIPS_fg↓ | outside↓ | leakage↓ | Gaussians↓
- **行:** KongQueZhuYu, XianKeLai1, CaoMei2

### Table 4: M1 view quality strategy comparison (KongQueZhuYu)
- **列:** Variant | Eval views | PSNR_fg↑ | SSIM_fg↑ | LPIPS_fg↓ | outside↓ | leakage↓ | Gaussians↓
- **行:** A6, A6+M1-hard, A6+M1-reject-only, A6+M1-soft

### Table 5: Ours-full cross-sample compactness
- **列:** Sample | Variant | PSNR_fg↑ | SSIM_fg↑ | LPIPS_fg↓ | outside↓ | leakage↓ | Gaussians↓
- **行:** KongQueZhuYu (A6, A6+M1-soft, A6+M4, A6+M1-soft+M4), XianKeLai1 (同), CaoMei2 (同)

### Table 6: Mesh structural and efficiency metrics
- **列:** Sample | Mesh variant | Vertices | Components | Largest comp. ratio | Small comps | Boundary edges | Boundary consistency | Mean disp. | P95 disp. | Mesh time(s)
- **行:** KongQueZhuYu (Standard, Smaller truncation, Post-boundary), XianKeLai1 (同)

### Table 7: Manual-vs-virtual phenotype validation
- **列:** Trait | n | MAE | RMSE | MAPE(%) | Bias | R²
- **行:** Plant height, Canopy width, Leaf length, Leaf width

---

## 数据来源汇总

| 数据 | 源文件 |
|------|--------|
| A0-A6消融数据 | 项目总结v1.0 §6.5 |
| 三样本A6数据 | 项目总结v1.0 §8.2 |
| M1-hard/soft数据 | 项目总结v1.0 §9 |
| M4跨样本数据 | 项目总结v1.0 §10 |
| S18 Ours-full数据 | 项目总结v1.0 §11 |
| S19 Mesh数据 | 项目总结v1.0 §12.3 |
| 表型对比数据 | 植株数据.xlsx |
| 数据集概览 | dataset_summary.md |
