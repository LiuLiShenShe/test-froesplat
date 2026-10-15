# 冻结输入直接查看记录

本记录来自当前模型直接查看本地真实输入对比图，未调用外部识图 API。图像由 `scripts/audit_changed_inputs.py` 读取冻结文件绘制；程序核对依据为 SHA256 和实际像素。显示图只用于定性核查，不生成正式渲染评价，不是人工前景标注。

已直接查看 `figures/INPUT_WanNianQing1_0037.png`、`INPUT_WanNianQing1_0038.png`、`INPUT_WanNianQing1_0165.png`、`INPUT_CaoMei2_0139.png` 以及 `CANDIDATES_CaoMei2_0139.png`。

- WanNianQing1 raw0037/0038：B_R0 黑背景图只保留偏右的一小块背景植株；C_R1 保留中央盆栽的主要叶片与花盆。
- WanNianQing1 raw0165：B_R0 保留画面上方背景植株；C_R1 保留中央目标叶片与部分花盆。
- CaoMei2 raw0139 对应注册图像 `crop_0124.png`：B_R0 保留左上方背景植株；C_R1 保留中央草莓盆栽。原始候选图的 01/R0 与 02/R1 白色区域与上述位置吻合。

这些观察说明四个冻结训练视图存在可见的候选选择差异，不能推出三维模型质量提高。具体选中 ID、像素改变比例和原文件路径/哈希见 `CHANGED_TRAINING_INPUT_AUDIT.csv`。所有 held-out 指标仍使用共同相机、canonical raw RGB 和固定 R0 代理 ROI，不用这些训练视图代替三维检验。
