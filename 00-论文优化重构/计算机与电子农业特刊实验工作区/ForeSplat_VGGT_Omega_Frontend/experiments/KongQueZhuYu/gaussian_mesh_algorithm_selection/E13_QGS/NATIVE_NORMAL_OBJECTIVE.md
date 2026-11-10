# QGS当前Mip实际法线目标

直接读取固定官方train.py与本次actual_training_parameters.json。当前wo_image_weight=True，法线项在iteration>7000启用，lambda_normal=0.015；use_alpha=False。

官方实际表达式为 `lambda_normal * mean(detach(w) * sum_channels(1 - normal * depth_normal))`。三通道求和后是 `lambda_normal * mean(detach(w) * (3 - dot(normal, depth_normal)))`，不是标量 `1-dot`。对给定前向步骤，额外的 `2*lambda_normal*mean(detach(w))` 不贡献参数梯度；权重随下一步骤的曲率渲染变化，日志中的额外项不必保持同一个值。保留此原生表达式，未修正或替换核心优化。

当前分支权重来自渲染曲率：先nan_to_num、下限0.0001、log，再以curvature_clamp_threshold=-1截断上限，最后1−sigmoid并detach。这个阈值约束法线损失中的对数曲率权重，不是对Quadratic Primitive施加几何曲率上限。

此预设下理论权重范围为[0.731058579, 0.999900010]，对应额外标量范围[0.021931757, 0.029997000]。这些是公式界限，不是实际植物测量值。报告的normal_loss也包含原生方向幅值/Alpha等效应，不能直接换算法线角度。统一几何评价另外使用采样后归一化方向及真实Mesh面法线。

[配置、公式及源码SHA](native_normal_objective.json) · [原生配置](actual_config.yaml) · [固定官方源码](https://github.com/will-zzy/QGS/blob/74d05c945e99fcaef7afe5a8831903be71ad9b55/train.py)。博查HTTP403不可用；公式来自真实本地固定源码读取，未以训练数据补充算法行为。
