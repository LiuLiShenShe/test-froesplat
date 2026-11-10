# E13 原生训练稳定性：真实保存状态

Partial native training: latest sampled iteration 13,000 / 30,000。这是一份执行中数值诊断，不是正式终态几何排序。

![实际原生训练与Checkpoint数值状态](native_training_stability.png)

|保存迭代|原生基元数|非有限基元|比例|真实核验记录|
|---|---|---|---|---|
|7000|474,027|0|0.000000%|[fixed_checkpoint_validation_7000.json](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/recovery/fixed_checkpoint_validation_7000.json)|
|8000|642,879|4|0.000622%|[native_nonfinite_checkpoint_8000.json](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/recovery/native_nonfinite_checkpoint_8000.json)|
|9000|963,341|82|0.008512%|[rolling_checkpoint_validation_9000.json](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/recovery/rolling_checkpoint_validation_9000.json)|
|10000|509,543|43|0.008439%|[rolling_checkpoint_validation_10000.json](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/recovery/rolling_checkpoint_validation_10000.json)|
|11000|1,491,423|44|0.002950%|[rolling_checkpoint_validation_11000.json](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/recovery/rolling_checkpoint_validation_11000.json)|
|12000|5,081,006|47|0.000925%|[rolling_checkpoint_validation_12000.json](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/recovery/rolling_checkpoint_validation_12000.json)|
|13000|9,211,363|44|0.000478%|[rolling_checkpoint_validation_13000.json](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/recovery/rolling_checkpoint_validation_13000.json)|

所有原生参数及NaN payload原样保留。Rolling路径会更新，表中SHA和计数只描述记录时实际观察到的状态；未另行保存该迭代的完整模型时，不把核验JSON冒称可下载Checkpoint。未核验间隔不能认定参数全部有限。

原生opacity每3000步重置，下一次100步增密/删除使用0.05阈值；日志数量在当次增密/删除前采样。9.1K→9.2K等骤降属于原生优化的净变化，不能称为另一个Gaussian Cleaning收益。距离项为已经乘lambda_dist的实际日志值，非逐步完整峰值记录。

已测试的是官方base配置：depth_ratio=0、lambda_dist=50000、densify_grad_threshold=0.5。其他官方场景预设只列为未训练参考，不能将当前数值缺陷归结为所有QGS配置必然失败。恢复保留原生优化，但未保存原随机轨迹；不能单独把NaN归因于显存分配器。

来源：[固定QGS训练与增密源码](https://github.com/will-zzy/QGS/tree/74d05c945e99fcaef7afe5a8831903be71ad9b55)、实际train_metrics.csv及逐文件CPU核验。博查检索HTTP403不可用；不以训练数据补充实测结果。
