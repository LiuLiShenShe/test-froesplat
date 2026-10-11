# 困难场景图件计划

DIFFICULT CASE LABELS: PARTIAL（0/32 人工核查；不是已完成部分图像判断）
图 A / 图 B: BLOCKED。没有生成图件或替代图片。

| code | 类别 | 人工确认帧数 | 当前覆盖 |
| --- | --- | --- | --- |
| D1 | Neighbor-plant adhesion | 0 | NOT COVERED |
| D3 | Foreground-background confusion | 0 | NOT COVERED |
| D6 | Small target | 0 | NOT COVERED |
| D11 | Cluttered background | 0 | NOT COVERED |
| D12 | Low contrast | 0 | NOT COVERED |

NOT COVERED 表示没有人工确认的可展示样本，**不表示真实图中一定不存在该类**。
历史绿色比例/primary D1/secondary D11 只保存在明确 NOT_GT 字段中，不作为选图依据。

## 完成 GT 后的完整生成要求

- 图 A：仅从双人签署的真实困难 RGB 选择，标注 frame ID、人工标签、视觉证据。
  某类无已确认样本继续 NOT COVERED，不拍脑袋挑图，也不生成合成场景。
- 图 B：真实 RGB | 人工 GT | 冻结 R0 | 冻结 R1，展示两次指定切换及任何退化/失败。
  非困难的切换案例不能冒充困难类别，允许诚实展示中性或退化结果。
- 用实际批准后指标标注 F1/IoU，保留真实全图与展示裁剪的坐标/缩放记录。
- 新图件脚本应拒绝未批准标签/GT/预测，不修改源图，不改变评价网格；排版缩略图
  仅用于展示，不反过来用于计算指标。
- 输出源追溯 CSV：frame ID、RGB/annotation/GT/R0/R1 路径与 SHA256、标签、证据、
  人工审核人、生成脚本 hash、配置、显示裁剪/缩放参数和最终图 hash。

现在不存在任何真实审核标签与独立预测，不能合法生成论文案例。
待合格人工输入后才实现/执行最终图件生成脚本；本阶段不冒充图件已交付。

来源：本目录真实审计 CSV/JSON/日志；Phase16 冻结 manifest、Phase17A U00_Control、Phase17C 历史行为表及 Phase20C-FIX 决策/生产源。
