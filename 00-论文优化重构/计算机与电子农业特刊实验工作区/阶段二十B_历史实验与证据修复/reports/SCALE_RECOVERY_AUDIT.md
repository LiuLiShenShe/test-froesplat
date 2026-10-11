# 尺度恢复审计

`植株数据.xlsx` 是真实工作簿：1 个 sheet、21 个植株行、17 列；字段为人工株高/冠幅/三片叶的人工与虚拟数值。
工作簿没有盆径、半径、scale marker 或 scale factor 字段，不能由它恢复物理尺度。
退稿稿件在 Table 8 叙述中依赖物理尺度，但仓库中没有逐样本盆径、标尺图、标定系数或传递记录。
`measure_traits_entry.py` 只要求 `--scale-marker`，输出外部工具的输入清单，并明确完整交互/半自动测量工具未随轻量发布包提供；这不是尺度恢复实现。
`06-MESH-new/*_mesh.log` 共 12 个日志，没有 pot/scale/trait 记录（详见 CSV）。
因此当前状态是 `USER_INPUT_REQUIRED`，不测量、不猜盆径、不把工作簿统计脚本当成三维测量算法。
