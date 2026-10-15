# 时间和内存的实际观测口径

本文件说明现有观察代码的计时范围，不改变冻结训练或提取设置。

- 新训练的 `training_wall_seconds` 从观察包装计时开始，包含 seed 初始化、Scene/输入载入、alpha/SHA 核验、原生训练、观察日志、最终保存及 checkpoint/PLY GPU 重载核验；不包含该进程此前的 Python/扩展导入。外层进程起止、日志关闭及退出等待另见 `reports/TRAINING_EXECUTION.json`。这不是纯优化器计算时间。
- 训练的 `peak_cuda_allocated_bytes`、`peak_cuda_reserved_bytes` 分别是该可见 GPU 上 PyTorch 内存分配器的峰值。峰值统计重置在 seed/模型初始化前，结束于最终重载检查后，可能包含重载对象。不能改称不含观察开销的纯训练显存，也不能当成整个设备的连续精确显存峰值。CUDA context、扩展绕过 PyTorch 分配器的分配并非全部由 allocated 数值覆盖。
- `logs/RESOURCE_OBSERVATIONS.csv` 每约60秒只读采样指定 GPU 的设备显存、利用率，以及主机内存/I/O压力。采样有时间间隔，可能漏掉短暂峰值，不能替代连续设备峰值。没有占用或终止其他用户任务。
- Mesh 的 `wall_seconds` 从导入完成后的模型/Scene准备开始，包含冻结相机输入、checkpoint/PLY核验、浮点保存、原生TSDF、结构统计、文件写出及重载核验；不包括解释器最终退出等待。`timings` 进一步区分渲染与缓存、TSDF融合、后处理/验证/写出，但这些分项不覆盖外层所有开销。
- Mesh 的 `peak_process_RSS_KiB` 来自 Linux `resource.getrusage(RUSAGE_SELF).ru_maxrss`，是整个进程的CPU常驻内存高水位，单位KiB；包括导入、缓存和结构分析。敏感性脚本同一进程内执行两个参数，因此后一个结果的RSS高水位可能继承前一个，不能据此作独立参数内存排名。
- `M2M3_RENDER_TIMING_SUMMARY.csv` 是另行冻结的共同推理计时：每模型3次首视角预热、全部held-out相机2轮，CUDA events与同步wall-clock同时记录。范围为原生RGB/alpha/depth/normal渲染包，排除模型载入、GT指标计算、PNG/float序列化。allocated峰值包含已载入模型及相机状态；reserved可能继承缓存，二者分别保留。

历史M0训练记录与新增观察包装的范围和共享主机负载并非完全相同。训练耗时与显存数值按原口径保留，不由单次差值推出稳健加速或节省显存。共同推理计时仍只有固定顺序的一次会话，应称为三场景pilot。所有结果保留物理GPU UUID与实际环境来源。
