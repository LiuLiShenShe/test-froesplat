# TEST32 两次历史候选切换

状态：新 CPU 运行、GT 指标、Oracle 和真人目标身份结果均 **NOT VERIFIED**。

| frame_id | historical R0 ID | historical R1 ID | R0 F1/IoU | R1 F1/IoU | Oracle | identity | outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CaoMei2_0146 | CaoMei2_0146_04 | CaoMei2_0146_03 | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |
| WanNianQing1_0227 | WanNianQing1_0227_00 | WanNianQing1_0227_01 | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |

来源为原 TEST32_HISTORICAL_SELECTION_CHANGE.csv / Phase17C_test32_safety.csv，
不是新跑分。历史 2/32=6.25% 改变，30 帧未改变；所有 32 行均保留在带 BLOCKED
状态的输出中。没有排除退化帧，也不能把未切换、非空或原始 mask 间 IoU 当作正确。
两次切换均需批准的 potted_clean 和独立预测目标身份记录才能判断改善/退化/持平。
