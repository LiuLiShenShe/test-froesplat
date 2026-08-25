# Task Plan: 实验三与实验四收口

## Goal

完成实验三 RAP-FSAM3-v2 图板收口，并补齐实验四 B0-B5 先验注入位置消融表、图件和文档状态。

## Phases

- [x] Phase 1: 生成实验三 v2 正式机制图板。
- [x] Phase 2: 补跑实验四严格 B2/B4 缺项。
- [x] Phase 3: 运行 B2/B4 foreground-object 指标并汇总 B0-B5 正式表。
- [x] Phase 4: 生成实验四图件和源数据。
- [x] Phase 5: 更新实验三/四方案、状态板、图件规划和论文映射。
- [x] Phase 6: 校验产物并交付结论。

## Key Questions

1. B2/B4 是否能用现有 2DGS 统一入口构成严格单因素对照？
2. B0-B5 的正式表是否能同时呈现前景质量、泄漏、高斯数量和效率？
3. 实验三图板是否替代旧 A0 vs A2/A5 诊断图作为正式用途？

## Decisions Made

- B2 使用 `foreground_track` 初始化，加载 mask 只用于筛初始 COLMAP 点，不开启 RGB/mask/opacity loss，不开启 pruning。
- B4 使用 `pruning_mode=mask`，加载 mask 只用于 pruning score，不开启初始化和任何训练损失。
- 实验三正式图板使用 `XianKeLai1` A1 old vs A1s 纠错和 `KongQueZhuYu` A2 vs A5c 几何 delta。
- B0-B5 正式主表使用 `KongQueZhuYu` 单样本严格位置消融；`XianKeLai1` 和 `CaoMei2` A6/full 结果作为跨样本支持，不混入主表。

## Errors Encountered

- `git status` 在 `/data/fj/F2DMAS` 返回非 git 仓库；本轮改用文件级校验，不依赖 git diff。

## Status

**Complete as of 2026-06-06 00:29 CST.**

已完成产物：

- 实验三新版机制图板：`../05-图件与论文映射/实验三_RAP-FSAM3v2新版图板/figures/Fig_E3v2_A1s_A5c_mechanisms.*`
- 实验四正式表：`../04-结果表格模板/实验四_先验注入位置消融结果表.csv`
- 实验四源数据：`../05-图件与论文映射/实验四_2DGS先验注入位置消融/source_data/experiment4_b0_b5_summary.csv`
- 实验四图件：`../05-图件与论文映射/实验四_2DGS先验注入位置消融/figures/Fig_E4_prior_injection_B0_B5_metrics.*`

补跑记录：

- B2 `B2_foreground_track_init_only`：2026-06-05T23:40:52 到 2026-06-06T00:02:56，runner status 为 `success`。
- B4 `B4_mask_pruning_only`：2026-06-06T00:06:19 到 2026-06-06T00:26:28，runner status 为 `success`；15000 到 30000 iter 共 7 份 pruning report，总移除 50274 个 Gaussian，全部由 mask score 触发。

核心结果：

- B2 仅初始化：PSNR_fg=22.5636，外部非黑比例=0.9919，泄漏能量=1.1849，说明只筛初始点不能形成前景对象。
- B4 仅事后剪枝：PSNR_fg=23.5861，外部非黑比例=0.9908，泄漏能量=1.2018，说明只剪枝仍不能替代训练期前景约束。
- B3/B5 的外部非黑比例约 0.03、泄漏能量约 0.019，是当前 B0-B5 中真正前景分离成功的设置。
