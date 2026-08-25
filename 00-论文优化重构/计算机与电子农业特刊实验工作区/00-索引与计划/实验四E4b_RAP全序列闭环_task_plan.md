# Task Plan: E4b RAP-FSAM3-v2 全序列闭环

## Goal
生成 `KongQueZhuYu`、`XianKeLai1`、`CaoMei2` 三个样本的 RAP-FSAM3-v2 全序列 mask，并只重跑 `RAP_FSAM3_v2` 的 E4b 2DGS 闭环，最终更新结果表和规划文档。

## Phases
- [x] Phase 1: 确认当前缺口和 GPU 状态
- [x] Phase 2: 从已有 RAP-FSAM3-v2 日志复用全序列生成参数
- [x] Phase 3: 生成三样本全序列 mask 到 `E4b_downstream/<样本名>/最终掩膜`
- [x] Phase 4: 核验 mask 数量、命名和图像有效性
- [ ] Phase 5: 只重跑 `RAP_FSAM3_v2` 的 E4b 训练、渲染、指标和 mesh
- [ ] Phase 6: 汇总结果并更新 `规划补充实验.md`

## Decisions Made
- 使用 GPU0：当前 GPU0 约 1.7 GB 占用，GPU1 约 31.6 GB 占用。
- E4b downstream 默认输出目录：`00-论文优化重构/数据管理/03-分割Mask/05-RAP-FSAM3掩膜/E4b_downstream/<样本名>/最终掩膜`。
- 只重跑 RAP：SAM3 与 FSAM3-base 已完成，不重复训练。
- RAP-FSAM3-v2 采用正式 A5c 口径：`P1-P5 + score_select + semantic_gate + sam3_if_supported SPNP + residual repair + reprompt/geometry logs + corrective_geometry colmap_tracks`，负向几何删除保持默认关闭。
- `XianKeLai1` 和 `CaoMei2` 输入帧数多于 E4b COLMAP 帧数；保留全输入生成，E4b 执行脚本按 COLMAP `images/` 精确匹配需要的 203 帧。
- `CaoMei2` 首次生成在 0084/0086-0089 出现空最终 mask；根因是 `score_select` 可能在存在非空候选时仍选择空候选。已给生成脚本增加保底逻辑：存在非空候选时只在非空候选中按分数选择。
- `KongQueZhuYu` 已完成 RAP 全序列 mask 生成：210/210，空 mask 0，E4b dry-run 匹配 210/210。
- `XianKeLai1` 已完成 RAP 全序列 mask 生成：208/208，空 mask 0，E4b dry-run 匹配 203/203。
- `CaoMei2` 修复版已完成 RAP 全序列 mask 生成：210/210，关键区间 0084-0089 非空，E4b dry-run 匹配 203/203。

## Errors Encountered
- `rg` 不可用：改用 `grep/find` 搜索历史命令和日志。
- `CaoMei2` 首次生成出现连续空 mask：已中止该进程，旧输出备份为 `E4b_downstream/CaoMei2_empty_select_bug_backup_20260606`，修复选择器后从头重跑 `CaoMei2`。

## Status
**Currently in Phase 5** - 三样本 RAP-FSAM3-v2 mask 已完成并通过严格匹配核验，正在只重跑 `RAP_FSAM3_v2` 的 E4b 2DGS 闭环。
