# E13 原生训练稳定性：真实保存状态

30K native training recorded。这是一份执行中数值诊断，不是正式终态几何排序。

![实际原生训练与Checkpoint数值状态](native_training_stability.png)

|保存迭代|原生基元数|非有限基元|比例|真实核验记录|
|---|---|---|---|---|
|1000|591,494|7|0.001183%|[rolling_checkpoint_validation_1000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_1000.json)|
|2000|401,083|13|0.003241%|[rolling_checkpoint_validation_2000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_2000.json)|
|3000|317,068|18|0.005677%|[rolling_checkpoint_validation_3000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_3000.json)|
|4000|204,224|5|0.002448%|[rolling_checkpoint_validation_4000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_4000.json)|
|5000|191,543|9|0.004699%|[rolling_checkpoint_validation_5000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_5000.json)|
|6000|184,065|11|0.005976%|[rolling_checkpoint_validation_6000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_6000.json)|
|7000|157,387|6|0.003812%|[fixed_checkpoint_validation_7000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/fixed_checkpoint_validation_7000.json)|
|8000|156,797|15|0.009567%|[rolling_checkpoint_validation_8000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_8000.json)|
|9000|158,771|25|0.015746%|[rolling_checkpoint_validation_9000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_9000.json)|
|10000|142,182|27|0.018990%|[rolling_checkpoint_validation_10000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_10000.json)|
|11000|152,152|31|0.020374%|[rolling_checkpoint_validation_11000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_11000.json)|
|12000|175,128|36|0.020556%|[rolling_checkpoint_validation_12000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_12000.json)|
|13000|172,539|128|0.074186%|[rolling_checkpoint_validation_13000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_13000.json)|
|14000|281,178|163|0.057970%|[rolling_checkpoint_validation_14000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_14000.json)|
|15000|532,735|360|0.067576%|[fixed_checkpoint_validation_15000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/fixed_checkpoint_validation_15000.json)|
|16000|532,735|403|0.075647%|[rolling_checkpoint_validation_16000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_16000.json)|
|17000|532,735|538|0.100988%|[rolling_checkpoint_validation_17000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_17000.json)|
|18000|532,735|643|0.120698%|[rolling_checkpoint_validation_18000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_18000.json)|
|19000|532,735|949|0.178137%|[rolling_checkpoint_validation_19000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_19000.json)|
|20000|532,735|1,094|0.205355%|[rolling_checkpoint_validation_20000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_20000.json)|
|21000|532,735|1,214|0.227881%|[rolling_checkpoint_validation_21000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_21000.json)|
|22000|532,735|1,341|0.251720%|[rolling_checkpoint_validation_22000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_22000.json)|
|23000|532,735|1,554|0.291702%|[rolling_checkpoint_validation_23000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_23000.json)|
|24000|532,735|1,763|0.330934%|[rolling_checkpoint_validation_24000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_24000.json)|
|25000|532,735|1,996|0.374670%|[rolling_checkpoint_validation_25000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_25000.json)|
|26000|532,735|2,169|0.407144%|[rolling_checkpoint_validation_26000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_26000.json)|
|27000|532,735|2,426|0.455386%|[rolling_checkpoint_validation_27000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_27000.json)|
|28000|532,735|2,840|0.533098%|[rolling_checkpoint_validation_28000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_28000.json)|
|29000|532,735|3,118|0.585282%|[rolling_checkpoint_validation_29000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/rolling_checkpoint_validation_29000.json)|
|30000|532,735|3,456|0.648728%|[fixed_checkpoint_validation_30000.json](/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/recovery/fixed_checkpoint_validation_30000.json)|

所有原生参数及NaN payload原样保留。Rolling路径会更新，表中SHA和计数只描述记录时实际观察到的状态；未另行保存该迭代的完整模型时，不把核验JSON冒称可下载Checkpoint。未核验间隔不能认定参数全部有限。

原生opacity每3000步重置，下一次100步增密/删除使用0.05阈值；日志数量在当次增密/删除前采样。相邻采样数量骤降属于原生优化的净变化，不能称为另一个Gaussian Cleaning收益。距离项为已经乘lambda_dist的实际日志值，图中竖线使用实际距离项起始配置，非逐步完整峰值记录。

当前实际Mip配置：depth_ratio=1.0、lambda_dist=10.0、densify_grad_threshold=1.0。base失败状态另行归档，不与当前曲线合并；其他预设未训练时不推广结论。恢复标志=False，原生捕获未保存RNG；不能单独把NaN归因于分配器。

来源：[固定QGS训练与增密源码](https://github.com/will-zzy/QGS/tree/74d05c945e99fcaef7afe5a8831903be71ad9b55)、实际train_metrics.csv及逐文件CPU核验。博查检索HTTP403不可用；不以训练数据补充实测结果。
