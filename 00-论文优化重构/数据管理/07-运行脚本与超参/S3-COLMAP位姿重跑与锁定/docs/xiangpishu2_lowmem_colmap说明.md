# XiangPiShu2 低内存 COLMAP 重跑说明

## 背景

`XiangPiShu2` 使用常规 COLMAP 重跑参数时，`sequential` 与 `exhaustive` 两次尝试均在 `feature_extractor` 阶段被系统以 `-9` 结束。该错误不是匹配失败或建图失败，而是 CPU SIFT 在高分辨率图像、全线程、affine shape 与 DSP 开启时产生过高内存峰值。

## 低内存策略

本次专门使用低内存参数重跑：

- `SiftExtraction.num_threads = 4`
- `SiftExtraction.max_image_size = 2000`
- `SiftExtraction.max_num_features = 6000`
- `SiftExtraction.estimate_affine_shape = 0`
- `SiftExtraction.domain_size_pooling = 0`
- `SiftMatching.num_threads = 4`

优先运行 `sequential_lowmem`；若注册率低于 70% 或失败，再运行 `exhaustive_lowmem`。

## 输出位置

- 脚本：`数据管理/07-运行脚本与超参/S3-COLMAP位姿重跑与锁定/scripts/rerun_xiangpishu2_lowmem_colmap.py`
- 参数：`数据管理/07-运行脚本与超参/S3-COLMAP位姿重跑与锁定/configs/xiangpishu2_lowmem_colmap.json`
- COLMAP 输出：`/data/fj/F2DMAS/04-COLMAP-rerun-original/XiangPiShu2/sequential_lowmem`
- 统一报告：`/data/fj/F2DMAS/04-COLMAP-rerun-original/rerun_report.json`

## 记录规则

该脚本不会删除原先 `sequential` 与 `exhaustive` 的失败记录。成功后会把 `sequential_lowmem` 或 `exhaustive_lowmem` 追加到 `rerun_report.json`，并将 `best` 指向成功的低内存结果。
