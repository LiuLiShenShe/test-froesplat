# E13：base中断归档与当前Mip的实际范围

当前正式主运行是官方Mip预设的全新Seed0训练，从与E10字节一致的VGGT初始化、相同208RGB与修正相机开始。没有恢复base模型，也没有把base失败进度当作正式30K。

base两次实际训练尝试累计342.196185450分钟，另记失败导出时间；此成本与主Mip运行时间分别报告。第一轮在11700步OOM，第二轮从原生7K恢复后在13301步的loss.backward()再次OOM。两次均为物理GPU1与其他任务共享的实测环境，不推广为独占A6000或全部QGS配置的结论。

原生7K恢复使用expandable_segments:True，保留float32、Adam、二次表面相交、损失与增密原规则。官方capture不保存RNG/视角栈，不能声明恢复与未中断随机轨迹完全一致。原恢复记录原样归档，下面使用当前真实路径。

|实际保留文件|字节|
|---|---:|
|[归档base原生7K PLY](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/runs/base_expandable_oom_13301/checkpoints/point_cloud/iteration_7000/point_cloud.ply>)|123248620|
|[归档base原生7K Checkpoint](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/runs/base_expandable_oom_13301/checkpoints/chkpnt7000.pth>)|358372869|
|[归档base实际13K滚动Checkpoint](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/runs/base_expandable_oom_13301/recovery/latest_native_state.pth>)|6963799373|
|[base失败瞬态原生PLY](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/failure_exports/20261109T084120Z_1535250/native_failed_state.ply>)|4479439682|
|[不可读取的base失败PTH](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/failure_exports/20261109T084120Z_1535250/native_failed_state.pth>)|10163827261|
|[主Mip实际1K原生PLY](</home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/audited_snapshots/iteration_1000/native_quadratic.ply>)|153790040|
|[主Mip实际1K Checkpoint](</home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/audited_snapshots/iteration_1000/native_state.pth>)|447178053|

base失败PLY实际有17,228,608个六尺度基元，完整载荷长度与SHA已读取核验，45个非有限基元原样保留。伴随失败PTH在Python ZIP和原生PyTorch reader中均无法读取；ZIP64偏移超出真实文件长度，原因未确定。原捕获记录的resume_compatible标志不证明最终文件可读，当前交付状态明确为不可恢复的失败导出。

[失败文件实际读取状态](failure_exports/20261109T084120Z_1535250/VALIDATED_DELIVERY_STATUS.json) · [PLY全载荷数值核验](failure_exports/20261109T084120Z_1535250/PLY_NUMERICAL_VALIDATION.json)。

主Mip 1K被动模型有591,494个原生基元，7个非有限位置/旋转基元未清理；全部PLY字段逐位等于实际PTH，活跃SH阶数1。此模型是中间状态，不能冒充正式30K。后续原生7K/15K/30K固定保存与每1K的滚动状态分别记录；滚动路径会更新，不把核验JSON当成另存模型。

当前Mip checkpoints/recovery解析到独立NVMe路径，原大模型在原有归档位置保留。实际文件索引包括本地解析后的绝对路径；历史记录中的旧观察路径由[实际归档映射](runs/base_expandable_oom_13301/RETAINED_FILE_PATHS.json)解释。

[原恢复报告原文](runs/base_expandable_oom_13301/RECOVERY_REPORT_ORIGINAL.md) · [base实际执行记录](runs/base_expandable_oom_13301/training_execution_records.json) · [当前官方预设](actual_config.yaml) · [初始化字节身份](INITIAL_POINTCLOUD_IDENTITY.json) · [全部真实模型路径](../artifact_index/GAUSSIAN_PLY_INDEX.md)。

来源：[固定QGS官方源码](https://github.com/will-zzy/QGS/tree/74d05c945e99fcaef7afe5a8831903be71ad9b55)、[PyTorch实际版本序列化源码](https://github.com/pytorch/pytorch/tree/v2.5.1)、实际训练与逐文件读取核验。博查HTTP403不可用；不使用训练数据填充实验数值。
