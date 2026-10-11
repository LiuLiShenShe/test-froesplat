from __future__ import annotations

import csv
from pathlib import Path

from common import ROOT, WORK, markdown_table, read_csv, relative, report, write_csv

CODE = ROOT / '2d-gaussian-splatting-m2m3-floor40'
MANUSCRIPT_NOTE = ROOT / '修改内容.md'
PAPER = WORK / 'ForeSplat__Foreground_Object_2D_Gaussian_Splatting_via_Semantic_Mask_Guided_Task_Reformulation_for_Plant/ForeSplat.tex'
M2M3_ROOT = ROOT / '00-论文优化重构/m2m3实验'
PILOT = M2M3_ROOT / '实验输出/reports/full_evaluation_summary_20260610_151144_full.csv'


def main():
    rows = [
        {'formula_or_module': 'Mask-to-model support S^m2g', 'paper_source': '修改内容.md:75-87', 'code_status': 'PARTIAL_MISMATCH', 'real_execution': 'mask_consistency_scores: project xyz into up to pruning_mask_max_views cameras; nearest-sample gt_alpha_mask; hits/visible', 'actual_parameters': 'max_views=12; threshold=0.45 in pilot; gt_alpha_mask; no q_i, eta_ij, projected footprint area average', 'experiment_evidence': 'M2M3 pilot rows with pruning_mode=mask/topology and foreground region', 'recommendation': 'REWRITE_DESCRIPTION'},
        {'formula_or_module': 'Model-to-mask deficit S^g2m', 'paper_source': '修改内容.md:89-107', 'code_status': 'NOT_IMPLEMENTED', 'real_execution': 'No per-Gaussian rendered alpha deficit H_i=M_i(1-A_i) aggregation found', 'actual_parameters': 'No tau_h or rendered-alpha primitive deficit parameter', 'experiment_evidence': 'NONE', 'recommendation': 'REMOVE_UNSUPPORTED_CLAIM'},
        {'formula_or_module': 'Boundary support B_j', 'paper_source': '修改内容.md:109-126', 'code_status': 'NOT_IMPLEMENTED', 'real_execution': 'No mask-boundary distance or sigma_b computation in M2M3 training path', 'actual_parameters': 'No sigma_b', 'experiment_evidence': 'NONE', 'recommendation': 'REMOVE_UNSUPPORTED_CLAIM'},
        {'formula_or_module': 'Gaussian utility U_j sigmoid', 'paper_source': '修改内容.md:128-145', 'code_status': 'MISMATCH', 'real_execution': 'm2m3_prune_scores sums normalized positive penalties for opacity, brightness, mask, scale and screen radius', 'actual_parameters': 'weights: opacity=1, brightness=1, mask=3, scale=0.5, view=0.5; topology mode; no sigmoid, residual R_j or explicit view-count term', 'experiment_evidence': 'M2M3 pilot', 'recommendation': 'REWRITE_DESCRIPTION'},
        {'formula_or_module': 'Keep decision', 'paper_source': '修改内容.md:151-160', 'code_status': 'IMPLICIT_ONLY', 'real_execution': 'Candidates not selected for prune remain; no explicit keep utility threshold', 'actual_parameters': 'prune mask and top-k budget determine survival', 'experiment_evidence': 'M2M3 pilot pruning reports', 'recommendation': 'REWRITE_DESCRIPTION'},
        {'formula_or_module': 'Densify decision', 'paper_source': '修改内容.md:151-160', 'code_status': 'NOT_IMPLEMENTED', 'real_execution': 'Standard gaussian_model.densify_and_prune remains unchanged; M2M3 wrapper runs after optimizer/densification', 'actual_parameters': 'densification standard 2DGS parameters', 'experiment_evidence': 'M2M3 training runs, but no M2M3 densify controller', 'recommendation': 'REMOVE_UNSUPPORTED_CLAIM'},
        {'formula_or_module': 'Prune decision', 'paper_source': '修改内容.md:151-160', 'code_status': 'IMPLEMENTED_WITH_DIFFERENT_FORMULA', 'real_execution': 'Existing opacity/brightness/mask candidate mask; m2m3 score top-k limits removal; gaussians.prune_points', 'actual_parameters': 'm2m3_max_remove_ratio=0.03; pruning interval=3000; pruning start=18000 in pilot; region=foreground; threshold mask=0.45', 'experiment_evidence': '9 full pilot rows across 3 scenes and 3 methods', 'recommendation': 'REWRITE_DESCRIPTION'},
        {'formula_or_module': 'Floor40 local block capacity', 'paper_source': '修改内容.md:162-176 and ForeSplat.tex:809', 'code_status': 'MISMATCH', 'real_execution': 'Global floor before prune: floor_count=round(reference_count*ratio); reference max_seen in pilot; no local blocks', 'actual_parameters': 'ratio=0.4; reference=max_seen; start=18000; end=iterations; blocked_by_floor reported', 'experiment_evidence': 'fg2dgs_m2m3_floor40 pilot rows and capacity reports', 'recommendation': 'REWRITE_DESCRIPTION'},
        {'formula_or_module': 'Foreground mask region gate', 'paper_source': 'M2M3 design notes and runner', 'code_status': 'IMPLEMENTED', 'real_execution': 'm2m3_region_mode=foreground gates mask penalty when mask_scores is available', 'actual_parameters': 'foreground; mask weight=3.0', 'experiment_evidence': 'M2M3 pilot', 'recommendation': 'KEEP_AS_IMPLEMENTED'},
        {'formula_or_module': 'M2M3 code path invocation', 'paper_source': 'ForeSplat.tex:805-809', 'code_status': 'IMPLEMENTED', 'real_execution': 'training -> maybe_prune_gaussians -> m2m3_prune_scores when capacity_control_mode is m2m3/m2m3_floor', 'actual_parameters': 'opt-in capacity_control_mode; default none', 'experiment_evidence': relative(PILOT), 'recommendation': 'KEEP_AS_IMPLEMENTED'},
    ]
    write_csv('M2M3_FORMULA_CODE_MATRIX.csv', rows)
    pilot = read_csv(PILOT)
    method_counts = {}
    for row in pilot:
        method_counts[row['method']] = method_counts.get(row['method'], 0) + 1
    report('M2M3_IMPLEMENTATION_TRUTH.md', f'''# M2M3-GO 实现真值

论文候选公式和代码不是同一实现。公式草稿位于 `修改内容.md`，退稿稿件只保留高层叙述；代码实际在 `2d-gaussian-splatting-m2m3-floor40/train.py`。
真实调用链：`training()` → `maybe_prune_gaussians()`（iteration达到 pruning_start 且满足 interval）→ 现有候选 prune mask → `m2m3_prune_scores()` → top-k removal budget → Floor40 global capacity gate（仅 m2m3_floor）→ `gaussians.prune_points()`。
标准 2DGS 的 `gaussian_model.densify_and_prune()` 没有被 M2M3 重写；M2M3 不执行论文所称的独立 densify/keep 微决策。
mask-to-model 只剩一个基于多视图 `hits/visible` 的 mask consistency score；model-to-mask deficit、boundary distance support、sigmoid utility、photometric residual 和 local block Floor40 均未真实执行。
Floor40 的真实机制是全局容量保护：`round(max_seen_count * 0.4)`，不是按植物前景局部块分别保留40%。
full pilot CSV 有 {len(pilot)} 行：{method_counts}；这些是真实 M2M3 pilot 结果，但不是原论文20序列Table7平均，也不能证明未实现公式。
逐项证据、代码路径、参数和建议见 `tables/M2M3_FORMULA_CODE_MATRIX.csv`。
''')
    report('M2M3_DECISION_OPTIONS.md', '''# M2M3-GO 决策选项

## 方案 A：按真实实现修改论文

- 把 M2M3-GO 描述为可选的、基于多视图 mask consistency 与 opacity/brightness/scale/screen-size penalty 的容量裁剪控制器。
- 明确它只包装 prune candidate path，不实现独立 primitive-level densification，也没有已执行的 model-to-mask deficit、boundary-support 或 sigmoid utility。
- 把 Floor40 改成全局 `max_seen` 容量下限，避免“local block”表述。
- 保留真实 pilot 结果为 M2M3 pilot/implementation evidence，不把它扩展成原 Table 7 的20序列平均。

## 方案 B：补实现并重新实验

- 实现并单元测试 S^m2g、S^g2m、B_j、U_j 以及显式 keep/densify/prune 规则。
- 增加 per-block capacity accounting，使 Floor40 真正按局部块执行。
- 冻结新参数和数据划分后，重新跑三场景 smoke，再跑预先注册的全量实验；不能把现有 pilot 结果回填为新公式结果。

本阶段不选择 A/B，不修改 M2M3 生产代码，也不进入 Phase 20E；最终选择依赖用户对论文叙述与后续研发范围的确认。
''')
    print('m2m3 audit complete', len(rows), flush=True)


if __name__ == '__main__':
    main()
