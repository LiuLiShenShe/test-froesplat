# E14 评价续接说明

恢复流程仅新增精修阶段渲染。此前四阶段各208视角的浮点PSNR、SSIM、Alex LPIPS、梯度及Alpha已经实际计算；复用前逐条核对208源ID、GT/RAP SHA256、基元数量与数值有限性。所有光度公式及LPIPS模型保持原实现。缓存不用于新增精修阶段，精修仍实际运行全部208视角评分。

原始评价记录保存在 `recovery_open3d019/evaluation_before_cache_reuse.json`，原计算成本仍在actual_commands.jsonl和完整日志；恢复记录的elapsed_seconds仅是恢复调用耗时。正式训练与Mesh算法不受该执行缓存影响。

来源：本地实际指标CSV、渲染索引及原始执行日志。
