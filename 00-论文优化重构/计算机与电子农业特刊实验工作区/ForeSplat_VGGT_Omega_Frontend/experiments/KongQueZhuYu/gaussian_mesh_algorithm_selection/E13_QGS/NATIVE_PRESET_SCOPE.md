# QGS 原生配置与实际运行范围

当前 E13 主运行使用官方 `OmegaConf.merge(base.yaml, Mip.yaml)`，从同一 VGGT 初始化、208 帧训练 RGB 和修正相机独立运行 Seed 0 的 30K。逐项 CPU 核验由 NATIVE_PRESET_VALIDATION.json 保存，实际运行参数由 actual_training_parameters.json 保存。没有从 base Checkpoint 恢复，也没有额外深度/法线先验。

Mip 原生 depth_ratio=1、lambda_dist=10、lambda_normal=.015、densify_grad_threshold=1、percent_dense=.0001、增密500–15000/100步、opacity重置3000。距离/法线分别3000/7000后启用。原生多视角几何/NCC起始与结束均30000，严格条件在本次30K内不参与。curvature_clamp_threshold=-1保留官方值。

此前 base 配置实际发生两次 CUDA OOM：11700步与13301步。首次从Seed0开始，第二次使用真实7K原生捕获恢复及 expandable_segments。第二次失败实际保存17,228,608基元的原生PTH/六尺度PLY。失败模型不作为30K正式结果。实际文件复核发现其失败PTH的ZIP64记录偏移超出文件长度，Python和PyTorch原生读取器均失败，因此该PTH不可恢复；独立保存的六尺度PLY字节长度正确，继续作为失败时原生参数模型保留。实际交付状态见 failure_exports/20261109T084120Z_1535250/VALIDATED_DELIVERY_STATUS.json，原始PROVENANCE中的resume_compatible只描述捕获阶段，不能替代实际文件读取核验。原始文件及日志按同文件系统重命名保留在 runs/，实际当前路径映射见 runs/base_expandable_oom_13301/RETAINED_FILE_PATHS.json。两个base尝试的实际运行成本另列，不混入Mip模型的成功训练时间。

显存观测中存在其他GPU1进程，失败记录描述实际共享条件，不能推断独占A6000或所有QGS配置必然失败。Mip是官方预设的资源恢复选择，同时改变深度、正则和增密策略；不是只改变显存分配器的严格消融。DTU/TNT等仍为未训练参考。Mip已实际完成30K，532,735原生基元，统一208视角PSNR22.8956dB、真FG18.2986dB；3,456个非有限基元原样保留。终态实际路径与来源验证见training_complete.json/checkpoint_provenance_validation.json，不能将文件完整性等同几何质量。

新Mip Checkpoint及rolling状态存入独立NVMe子目录，工作区别名解析后的真实路径见 STORAGE_RELOCATION.json。旧HDD原文件与已存在的独立NVMe旧副本保留；旧副本不用于Mip初始化。

来源：[固定官方 QGS 源码和配置](https://github.com/will-zzy/QGS/tree/74d05c945e99fcaef7afe5a8831903be71ad9b55)，实际配置SHA、训练日志、失败捕获和路径核验。博查检索HTTP403不可用；未据此认定配额原因。
