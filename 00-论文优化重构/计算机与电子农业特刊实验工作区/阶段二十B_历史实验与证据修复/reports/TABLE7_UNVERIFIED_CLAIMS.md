# Table 7 未验证声明

| method | paper_psnr | paper_ssim | paper_lpips | paper_train_time_sec | paper_mesh_time_sec | matching_artifact_rows | matching_samples | status | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COLMAP | 13.63 | 0.8745 | 0.1072 | 599.5 | 78.0 | 5 | 2 | UNVERIFIED | No source set proves same 20 sequence/sample split, view split, metric ROI, and runtime protocol |
| 3DGS_with_VFM | 30.17 | 0.9587 | 0.0386 | 5413.5 | 642.0 | 0 | 0 | UNVERIFIED | No source set proves same 20 sequence/sample split, view split, metric ROI, and runtime protocol |
| Standard_2DGS | 29.58 | 0.9574 | 0.0487 | 12913.7 | 157.9 | 11 | 11 | UNVERIFIED | No source set proves same 20 sequence/sample split, view split, metric ROI, and runtime protocol |
| ForeSplat | 31.09 | 0.9711 | 0.0365 | 5044.5 | 55.0 | 12 | 12 | UNVERIFIED | No source set proves same 20 sequence/sample split, view split, metric ROI, and runtime protocol |


禁止沿用“20序列平均”作为 VERIFIED；论文四行旧数字最多标记 `PAPER_REPORTED_UNVERIFIED`。
训练时间需在统一机器、迭代数、数据划分和计时边界下重测；网格时间需说明是否含后处理、写盘和失败重试。
