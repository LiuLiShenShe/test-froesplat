# 旧2DGS/P6输入身份审计

旧掩膜目录样本数 19；有身份验证的P6匹配样本 6/19；唯一帧 23；
版本-帧对 104，有效 43，EXACT_MATCH 0。分布：{'DIFFERENT': 13, 'SIMILAR_BUT_DIFFERENT': 30, 'NOT_VERIFIED': 37, 'UNMATCHED': 24}。
19是SAM掩膜样本数量，不等于19个已证实训练完成的2DGS样本；BaiZhang不在该旧掩膜目录，单独UNMATCHED。
真实图像索引→原FFT RGB逐像素相同；old crop在mask内与RGB完全相同，核实mask/RGB配对，不按排序索引匹配。
两份掩膜必须同尺寸、同raw RGB坐标系才比较；没有任何缩放。
0.95只是SIMILAR_BUT_DIFFERENT描述阈值，不是3D等价阈值；IoU接近1也不能直接继承新方法结果。
图像尺寸、已注册COLMAP名和训练配置逐帧记录于CSV。COLMAP images是undistorted域，例如CaoMei1从2160×3840变2120×3800；
原掩膜比较不证明训练相机域一致。没有生成新P6+R1最终掩膜，本阶段的P6版本全部为旧运行。
Pot coverage只用已有split_v2真实pot标注，缺失写UNKNOWN。
