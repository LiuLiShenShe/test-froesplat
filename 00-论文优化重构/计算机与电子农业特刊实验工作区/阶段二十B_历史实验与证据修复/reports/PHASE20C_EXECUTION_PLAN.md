# Phase 20C 执行计划（只准备）

1. 在 V0–V3 配置 JSON 和命令锁定后，先运行 CPU dry-run/参数回显，不运行 SAM3。
2. 固定同一候选缓存、P6 prompt、输入帧、COLMAP 资产和输出命名；每组保存参数 JSON、运行日志、候选来源哈希。
3. 先执行 V0/V1，比较 base 与 R1 的 selected mask、primary score、oracle regret、空掩膜和候选选择变化。
4. 再执行 V2/V3，记录 SPNP、residual repair、corrective geometry 是否实际接受以及每阶段 mask 哈希。
5. 用 `potted_clean` GT 评价 DEV/安全集，不把 TEST32 无GT帧计算成准确率；报告 paired frame、sample_equal_mean、median、SD 和有效数。
6. 只有 V0–V3 完成且输入/版本一致后，才决定最小 2DGS/M2M3-GO GPU 重跑样本；不默认全量20序列。

停止条件：任何资产缺失、mask坐标域未闭合、refinement 参数未写入、运行中途失败或输出混入历史目录时，标记 BLOCKED 而非填数字。
