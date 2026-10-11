# TEST32 候选切换审计

状态：历史行为来源 VERIFIED；新的 R0/R1 执行及切换准确性判定 BLOCKED。

Phase17C 历史表共 32 帧，仅 2/32 = 6.25% 改变选择，另 30 帧不变。
完整 32 行均保存在 `TEST32_HISTORICAL_SELECTION_CHANGE.csv`，不筛除不变/不利帧。
本阶段只核验这些历史 ID 属于真实候选集，不重算分数，不修改历史排序。

| frame ID | historical R0 | historical R1 | 改善/退化/无明显差异 |
| --- | --- | --- | --- |
| CaoMei2_0146 | CaoMei2_0146_04 | CaoMei2_0146_03 | BLOCKED |
| WanNianQing1_0227 | WanNianQing1_0227_00 | WanNianQing1_0227_01 | BLOCKED |

两帧均为 NOT VERIFIED，缺少人工 potted_clean 和独立目标身份确认，
因此不能把切换、面积变大或非空解释为改善，也不能说“无明显差异”。
历史 raw-mask 相互 IoU 或历史 empty count 不是 GT 准确率/身份正确率。
批准后必须完整保留 32 帧，对以上两帧以及任何新发生切换、退化或严重失败做同等审查。

来源：本目录真实审计 CSV/JSON/日志；Phase16 冻结 manifest、Phase17A U00_Control、Phase17C 历史行为表及 Phase20C-FIX 决策/生产源。
