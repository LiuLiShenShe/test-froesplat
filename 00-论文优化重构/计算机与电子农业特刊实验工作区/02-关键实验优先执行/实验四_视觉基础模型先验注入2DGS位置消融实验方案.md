# 实验四：视觉基础模型先验注入 2DGS 位置消融实验方案

## 优先级

最高。该实验直接支撑论文主线：视觉基础模型掩膜的价值不只是生成二维掩膜，而是把植物前景先验转化为三维重建优化约束。

## 当前状态

截至 2026-06-06，实验四 `KongQueZhuYu` 主样本 B0-B5 位置消融已收口。项目中可复用和本轮新增的 2DGS/ForeSplat 证据包括：

- `KongQueZhuYu` A0-A6 foreground-object objective 消融已完成，能说明输入遮罩、mask loss、背景透明度、前景 RGB loss 和前景 track 初始化的作用。
- `S12` 已把 A6 扩展到 `KongQueZhuYu`、`XianKeLai1`、`CaoMei2` 三个代表样本。
- `S18` 已完成 A6、A6+M1-soft、A6+M4、A6+M1-soft+M4 的三样本组合验证，证明 compact/full 变体在小幅质量损失内降低泄漏和高斯数量。
- 本轮补跑了严格 B2 和 B4 缺项：B2 只用 `foreground_track` 筛选 COLMAP 初始点，不启用 RGB/mask/opacity loss 和 pruning；B4 只启用 `pruning_mode=mask`，不启用前景初始化和任何训练损失。

正式表格、源数据和图件均已生成；`XianKeLai1` 与 `CaoMei2` 的 A6/full 结果保留为跨样本支持，不纳入 B0-B5 单因素主表。

## 核心问题

植物前景先验应注入 2DGS 的哪个阶段，才能真正改善前景重建并抑制背景泄漏？

## 假设

只做输入遮罩或事后剪枝是不够的。把掩膜先验注入初始化、RGB 损失、透明度约束和剪枝流程，才能稳定形成前景对象重建。

## 消融设置

| 编号 | 设置 | 输入遮罩 | 稀疏点初始化 | RGB 损失 | 透明度损失 | 剪枝 |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | 标准 2DGS | 否 | 否 | 否 | 否 | 否 |
| B1 | 仅输入遮罩 | 是 | 否 | 否 | 否 | 否 |
| B2 | 仅稀疏点初始化 | 否 | 是 | 否 | 否 | 否 |
| B3 | 仅损失约束 | 否 | 否 | 是 | 是 | 否 |
| B4 | 仅事后剪枝 | 否 | 否 | 否 | 否 | 是 |
| B5 | 完整 ForeSplat | 可选 | 是 | 是 | 是 | 是 |

## B0-B5 映射口径

| 编号 | 结果来源 | 口径说明 |
| --- | --- | --- |
| B0 | `E2_2dgs_baseline` | 标准全场景 2DGS 基线 |
| B1 | `E3_fsam3_preprocess` | 仅在训练输入阶段做 mask preprocess |
| B2 | `B2_foreground_track_init_only` | 2026-06-05/06 补跑；mask 只用于前景 COLMAP track 初始化 |
| B3 | `A5_fg_rgb_alpha_bg_loss` | foreground RGB、alpha mask 与 background opacity loss 组合，代表损失注入位置 |
| B4 | `B4_mask_pruning_only` | 2026-06-06 补跑；mask 只用于 15000-30000 iter 的 pruning score |
| B5 | `F1_high_precision_foreground` | 完整 ForeSplat，高精度前景对象设置 |

## 数据范围

- 主表使用 `KongQueZhuYu` 一个代表性样本，保证 B0-B5 单因素口径严格一致。
- `XianKeLai1`、`CaoMei2` 的 A6/full 结果作为跨样本支持证据。
- 后续若扩展到 6 个代表性样本，应覆盖复杂背景、细叶、密集遮挡、花叶混合、花盆泄漏风险和成熟植株。

## 所需输入

| 输入 | 说明 | 状态 |
| --- | --- | --- |
| 最终掩膜 | 优先使用验证后的 RAP-FSAM3 掩膜 | B0-B5 主表统一使用 `KongQueZhuYu` 现有 SAM/RAP-FSAM mask 评估口径 |
| COLMAP 稀疏轨迹 | 用于前景初始化 | B2/B5 均已使用 foreground track initialization |
| 2DGS 训练代码与配置 | 用于 B0-B5 设置 | B2/B4 配置已补充到 `S10-KongQueZhuYu小矩阵/configs/` |
| 渲染结果 | 用于计算重建指标 | B0-B5 `ours_30000` render 与 full-frame metrics 均已生成 |
| 泄漏评估掩膜 | 用于外部非黑比例和泄漏能量 | B0-B5 foreground-object metrics 均已生成 |

## 指标

| 指标 | 作用 |
| --- | --- |
| PSNR_fg | 前景重建质量 |
| SSIM_fg | 结构相似性 |
| LPIPS_fg | 感知误差 |
| 外部非黑比例 | 背景残留比例 |
| 泄漏能量 | 背景泄漏强度 |
| 高斯数量 | 表示紧凑性 |
| 训练时间 | 计算效率 |
| 网格化时间 | 导出效率 |
| 前景对象成功 | 是否可用于表型测量 |

## 结果表

已正式填写：

- `../04-结果表格模板/实验四_先验注入位置消融结果表.csv`

图件与源数据：

- `../05-图件与论文映射/实验四_2DGS先验注入位置消融/figures/Fig_E4_prior_injection_B0_B5_metrics.*`
- `../05-图件与论文映射/实验四_2DGS先验注入位置消融/source_data/experiment4_b0_b5_summary.csv`
- `../05-图件与论文映射/实验四_2DGS先验注入位置消融/source_data/experiment4_b0_b5_paper_table.csv`

## 最新结果摘要

| 设置 | PSNR_fg | SSIM_fg | LPIPS_fg | 外部非黑比例 | 泄漏能量 | 高斯数量 | 前景对象成功 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0 标准2DGS | 24.2090 | 0.8514 | 0.0480 | 0.9908 | 1.2201 | 751213 | 全图可用但前景未分离 |
| B1 仅输入遮罩 | 20.7291 | 0.7505 | 0.0696 | 0.0073 | 0.0042 | 263108 | 部分 |
| B2 仅稀疏点初始化 | 22.5636 | 0.7966 | 0.0604 | 0.9919 | 1.1849 | 683429 | 否 |
| B3 仅损失约束 | 25.1055 | 0.8561 | 0.0437 | 0.0294 | 0.0190 | 592900 | 是 |
| B4 仅事后剪枝 | 23.5861 | 0.8287 | 0.0515 | 0.9908 | 1.2018 | 689821 | 全图可用但前景未分离 |
| B5 完整ForeSplat | 24.9723 | 0.8540 | 0.0438 | 0.0293 | 0.0186 | 585594 | 是 |

关键读法：B1 能快速压低背景泄漏但牺牲全图/前景质量；B2 和 B4 单独使用时仍保留明显背景；B3 与 B5 才同时保持较高前景质量并把外部非黑比例压到约 3%，说明先验需要进入训练损失，完整 ForeSplat 进一步结合初始化和剪枝得到更紧凑的 Gaussian 表示。

## 完成标准

- [x] 将已有 `KongQueZhuYu` A0-A6/S18 结果映射到 B0-B5，并补跑严格 B2/B4 缺项。
- [x] B0-B5 至少在一个代表性样本上全部跑通并填入正式表。
- [x] `XianKeLai1` 和 `CaoMei2` 的 A6/full 结果作为跨样本支持保留。
- [x] 表格同时包含前景质量、泄漏和效率指标。
- [x] 可视化结果能解释指标差异。
