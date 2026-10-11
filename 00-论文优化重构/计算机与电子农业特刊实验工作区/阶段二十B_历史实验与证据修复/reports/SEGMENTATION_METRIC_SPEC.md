# 分割指标规范

主统计口径为 sample_equal_mean：先对每个序列的有效帧指标取算术平均，再对序列均值等权平均。
Per-frame mean、per-sample mean、median、population SD(ddof=0) 分列为独立 aggregation 行。
4序列帧数相同，因此全有效时 sample_equal_mean=per_frame_mean；不把 pooled confusion F1 当帧均值。

F1=2TP/(2TP+FP+FN)；foreground IoU=TP/(TP+FP+FN)；mIoU=(foreground IoU+background IoU)/2。
分母为0时值为0，保持旧 safe_div 约定。
边界=mask减3×3方形腐蚀（OpenCV默认边界行为）；Boundary-F1 对双方边界用半径3 px、7×7椭圆膨胀判匹配，再取 precision/recall 调和平均。
双方空边界 BF1=1，单方空 BF1=0。
HD95 合并双向边界距离后取95百分位；cv2.distanceTransform(DIST_L2,3)，属于3×3近似欧氏距离，不换精确EDT。
单位为已有标注坐标系的原图像素；双方空边界0，单方空取图像对角线。
Leakage energy=FP/(H×W)，是旧脚本的掩膜泄漏比例，不是RGB辐射能量；新文稿应据实命名。
不与三维渲染脚本的RGB leakage energy混合。

权威旧实现：`03-扩展实验后续执行/实验二_四样本提示词鲁棒性汇总.py` 的 pixel_metrics/boundary_f1/hd95；
E1旧实现定义相同，逐函数回归测试验证。使用原函数，不执行其 main。
