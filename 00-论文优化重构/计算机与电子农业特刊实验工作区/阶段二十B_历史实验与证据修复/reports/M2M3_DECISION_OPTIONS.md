# M2M3-GO 决策选项

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
