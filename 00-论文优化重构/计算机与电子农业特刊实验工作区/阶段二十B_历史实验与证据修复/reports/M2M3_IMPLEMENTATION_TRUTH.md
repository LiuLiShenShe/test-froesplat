# M2M3-GO 实现真值

论文候选公式和代码不是同一实现。公式草稿位于 `修改内容.md`，退稿稿件只保留高层叙述；代码实际在 `2d-gaussian-splatting-m2m3-floor40/train.py`。
真实调用链：`training()` → `maybe_prune_gaussians()`（iteration达到 pruning_start 且满足 interval）→ 现有候选 prune mask → `m2m3_prune_scores()` → top-k removal budget → Floor40 global capacity gate（仅 m2m3_floor）→ `gaussians.prune_points()`。
标准 2DGS 的 `gaussian_model.densify_and_prune()` 没有被 M2M3 重写；M2M3 不执行论文所称的独立 densify/keep 微决策。
mask-to-model 只剩一个基于多视图 `hits/visible` 的 mask consistency score；model-to-mask deficit、boundary distance support、sigmoid utility、photometric residual 和 local block Floor40 均未真实执行。
Floor40 的真实机制是全局容量保护：`round(max_seen_count * 0.4)`，不是按植物前景局部块分别保留40%。
full pilot CSV 有 9 行：{'fg2dgs_baseline': 3, 'fg2dgs_m2m3': 3, 'fg2dgs_m2m3_floor40': 3}；这些是真实 M2M3 pilot 结果，但不是原论文20序列Table7平均，也不能证明未实现公式。
逐项证据、代码路径、参数和建议见 `tables/M2M3_FORMULA_CODE_MATRIX.csv`。
