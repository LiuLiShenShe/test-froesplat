# 冻结 R0/R1 独立比较 — BLOCKED

正式评价未执行，R0 F1、R1 F1、配对差、其他准确率指标和新 selection-change count
均 NOT VERIFIED；所有 *_EVALUATED.csv 是显式 BLOCKED 的覆盖/字段清单，
数值留空，不是结果表。历史候选 ID 单独命名，不伪装成新运行的 selected ID。

32 RGB / 8 序列 / 71 候选已通过原严格输入验证；不等同于 GT 或算法效果通过。
R0 只把 V1 ranking_policy 从 r1 改为 base，其余参数、score、候选与评价口径相同。
P6/per_instance/single，ensemble/SPNP/Residual/GCFC/A6/A7 全 OFF，权重和风险规则
完全来自 Phase20C-FIX。六位 confidence 为真实归档精度，不补猜原张量尾数。

Gate 通过后必须先跑 CaoMei2_0146 与 WanNianQing1_0227 两政策 CPU smoke，再全32。
复用生产算法入口和严格候选 metadata loader；不复制另一套 ranking。固定 GT 为
potted_plant AND NOT blue_cube。需报告 selected 与 final cleanup 层级以免混淆。

正式结果应重用 Phase20B/20C 的定义：F1、IoU_fg、背景/前景均值 mIoU、3px
Boundary-F1、双向边界距离 HD95、Leakage=FP/图像像素。Oracle 用同帧相同候选
对 GT 的最佳 F1，说明是原始候选层级；不要与清理后指标相混。Severe F1<0.5
仅为客观 proxy，真人目标身份/严重错误独立记录，不能由 F1 自动代签。

全32逐帧、8序列、frame-equal/sample-equal、median、SD(ddof=1)、配对差和
改善/退化均保留。若 bootstrap，单位为8个序列，说明小样本/宽区间，不把32
相关帧作为独立样本。无GT图像曾参与行为检查，不声称完全未见过的外部数据集。
现有 FIX/scripts/evaluate_independent_test.py 确实循环冻结32帧，但以 FIX 的
WORK/旧 review queue 寻址，缺少本次完整 Oracle/离散度/困难统计入口；不可把
其开发集表或只运行旧 CLI 当成 completion 正式全量结果。本次未声称实现了该
完整运行器；Gate 后仍需完成入口并按本规范实际执行。严禁 TEST32 结果驱动调参。
