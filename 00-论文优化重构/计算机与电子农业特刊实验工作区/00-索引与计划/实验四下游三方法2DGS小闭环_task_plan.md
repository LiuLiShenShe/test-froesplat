# Task Plan: 实验四下游三方法 2DGS 小闭环

## Goal
补齐 SAM3 single prompt、FSAM3-base、RAP-FSAM3-v2 三类 mask source 在相同样本、COLMAP、2DGS 配置和训练轮数下的 2DGS downstream 对比脚本与执行说明。

## Phases
- [x] Phase 1: Inspect current 2DGS, mask, COLMAP and experiment-four artifacts
- [x] Phase 2: Add reusable execution and comparison scripts
- [x] Phase 3: Validate scripts with dry-run/syntax checks
- [x] Phase 4: Update `规划补充实验.md` with concrete run instructions
- [x] Phase 5: Generate RAP-FSAM3-v2 full-sequence masks for the three E4b samples
- [x] Phase 6: Complete RAP-FSAM3-v2 train/render/metrics/foreground-metrics/mesh closed loop
- [x] Phase 7: Rebuild full manifest and final summary tables

## Key Questions
1. Which paths should represent the three mask sources without changing the locked COLMAP input?
2. Which existing 2DGS flags map to the fixed ForeSplat/downstream configuration?
3. How should the comparison script read PSNR_fg, SSIM_fg, LPIPS_fg, leakage, Gaussian count and mesh topology robustly?

## Decisions Made
- Treat this as an E4b downstream mask-source closed loop, separate from the existing E4 B0-B5 prior-injection-position ablation.
- Use three representative samples by default: `KongQueZhuYu`, `XianKeLai1`, `CaoMei2`, matching existing cross-sample ForeSplat support in the experiment-four plan.
- Keep the execution script in dry-run mode by default; require `--execute` before launching long 2DGS training or evaluation jobs.
- Normalize raw masks into per-run `prepared_masks/mask_{image_stem}.png` symlinks so different upstream mask naming styles can be reused without changing COLMAP images.
- Write both status tables and final metric tables through the same summary script; before all runs finish, rows expose blockers such as `missing_masks` and `not_started`.
- RAP-FSAM3-v2 full-sequence masks are stored under `00-论文优化重构/数据管理/03-分割Mask/05-RAP-FSAM3掩膜/E4b_downstream/<Sample>/最终掩膜`.
- Final E4b comparison should be described cautiously: RAP-FSAM3-v2 is complete, but its downstream 2DGS reconstruction metrics are lower than SAM3/FSAM3-base on PSNR_fg, SSIM_fg, LPIPS_fg, outside ratio and leakage energy in this three-sample closed loop.

## Errors Encountered
- CodeGraph is not initialized for this repo, so local inspection is using targeted shell reads instead.
- Full-workspace search hits paper caches and third-party code; future searches should stay scoped to `2d-gaussian-splatting-main`, data-management, and experiment workspace directories.
- `git diff --stat` from `/data/fj/F2DMAS` reports that this workspace is not a git repository; use file-level checks instead of relying on git status here.
- Earlier mask inventory blocked full E4b execution because RAP-FSAM3-v2 had only small-subset masks. This is now resolved by generating full-sequence masks for `KongQueZhuYu`, `XianKeLai1`, and `CaoMei2`.

## Status
**Completed as of 2026-06-07 10:31 CST** - RAP-FSAM3-v2 masks are complete, RAP-FSAM3-v2 2DGS closed loop outputs exist for all three samples, the full manifest has 9/9 mask-complete rows, and the final summary/逐样本 tables have 9/9 `complete` rows. Current main-table means: SAM3 single prompt PSNR_fg 24.6598 / SSIM_fg 0.8454 / LPIPS_fg 0.0358; FSAM3-base 24.6602 / 0.8458 / 0.0357; RAP-FSAM3-v2 23.2179 / 0.8303 / 0.0397.
