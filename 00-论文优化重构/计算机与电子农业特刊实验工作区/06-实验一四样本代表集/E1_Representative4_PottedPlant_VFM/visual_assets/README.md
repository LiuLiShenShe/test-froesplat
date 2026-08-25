# 实验一四样本代表集 VFM 横向对比

## 口径

- 样本：`KongQueZhuYu`、`DouBanLv1`、`ChangShouHua2`、`CaoMei1`。
- 帧数：4 个样本 x 5 帧 = 20 帧 GT。
- GT 定义：project LabelMe GT foreground: union of closed target-object linestrips。
- 输出目录全部位于本工作区内，未覆盖旧 S23 六帧结果。

## 汇总结果

| Method | F1 | mIoU | HD95 px | Boundary F1 | Leakage energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| SEEM | 0.5439 | 0.6147 | 306.55 | 0.1584 | 0.005293 |
| CLIPSeg | 0.9221 | 0.9076 | 310.37 | 0.0874 | 0.018218 |
| Grounded-SAM | 0.9443 | 0.9335 | 246.36 | 0.4034 | 0.004374 |
| Grounded-SAM2 | 0.9238 | 0.9109 | 359.38 | 0.3018 | 0.004718 |
| SAM2 oracle | 0.8810 | 0.8656 | 250.19 | 0.2977 | 0.012240 |
| SAM3 single prompt | 0.9692 | 0.9624 | 193.71 | 0.4051 | 0.002272 |
| RAP-FSAM3-v2 | 0.9706 | 0.9641 | 183.42 | 0.4643 | 0.001739 |
| U-Net | 0.8622 | 0.8443 | 402.54 | 0.1501 | 0.028350 |
| DeepLabv3+ lite | 0.8758 | 0.8596 | 265.25 | 0.1532 | 0.017171 |
| SAM existing | 0.9837 | 0.9797 | 19.49 | 0.4283 | 0.001338 |

## 文件说明

- `manifest.csv`：四样本 20 帧图像、GT JSON 和 GT mask 索引。
- `gt_shape_summary.csv`：每个闭合 linestrip 的面积和 bbox，用于审计花盆/主体轮廓。
- `method_masks/`：各方法二值预测 mask。
- `metrics/summary_metrics.csv` 与 `metrics/frame_metrics.csv`：汇总与逐帧指标。
- `visual_assets/`：overlay、error map、contact sheet、指标柱状图和 source data。