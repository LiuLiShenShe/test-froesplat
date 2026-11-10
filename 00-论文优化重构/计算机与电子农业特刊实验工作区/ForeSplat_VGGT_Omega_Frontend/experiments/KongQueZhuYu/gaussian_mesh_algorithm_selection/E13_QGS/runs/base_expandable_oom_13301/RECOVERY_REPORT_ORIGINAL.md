# E13 QGS：真实中断与原生恢复记录

E13 尚未完成正式30K。本记录保留11.7K显存中断、真实7K状态及恢复的实现细节，不将中断时的临时基元数当作已导出的模型。

首轮原生训练在11,700迭代记录11,712,536个二次曲面基元，随后在原生CUDA反向传播请求2.46GiB时OOM。PyTorch已分配32.69GiB，保留未分配8.82GiB，设备空闲2.06GiB；另一个独立任务使用3.46GiB。物理GPU为1，日志中的CUDA GPU0是逻辑编号。未停止或修改其他任务。

实际保留的原生模型为7K、474,027个基元：

- [原生六尺度QGS PLY](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/checkpoints/point_cloud/iteration_7000/point_cloud.ply>)，123,248,620 bytes。
- [原生Gaussian/Adam Checkpoint](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/checkpoints/chkpnt7000.pth>)，358,372,869 bytes；SHA256 `cbb15ed13b9e94c13e16ca21225dd36fb2185a07fdd3f10fc177c21b4fe0ed16`。
- [7K PLY/Checkpoint全部字段逐位核验](partial_checkpoint_provenance_validation.json)。
- [首轮完整100步间隔日志](recovery/attempt_01_train_metrics.csv)与[原始终端错误](recovery/attempt_01_terminal_log.txt)。

恢复使用 `expandable_segments:True`，已在实际PyTorch 2.5.1+cu121、物理GPU1验证变长float32分配通过，且没有不支持该选项的警告。该设置改变显存分配方式，不修改float32精度、Adam、二次曲面相交、损失权重或增密规则。原先就是 `dataset.type=iterable`，RGB保留CPU、逐视角搬至CUDA；本次没有切换图像模式。

原生 `train.py`、`scene/gaussian_model.py`、`gaussian_renderer/__init__.py` 的实际SHA256均与首轮参数记录相同。208 RGB和K/W2C/P在恢复加载后再核验。目标仍为30,000，原生恢复从7,001开始，不再计算另一轮30K。

官方capture未保存随机数状态和视角栈；官方恢复会重设Seed0，不能宣称与未中断训练具有完全相同的随机轨迹。此外原生Scene.save发生在该迭代增密/optimizer.step之前，官方restore从saved_iteration+1继续，这一原生语义如实保留。没有通过修改损失/增密来强行压小基元数。

每1,000迭代增加被动滚动Checkpoint，仅保留 `recovery/latest_native_state.pth`，使用临时文件与原子替换；其迭代与保存阶段写在同名provenance JSON。原生7K、15K、30K的实际模型和Checkpoint另外保留。滚动状态不能标为正式30K，训练期间会更新，最终交付应优先使用固定终态路径。

首轮已消耗约186.880分钟，保留7K前缀约85.400分钟，其余约101.480分钟为丢弃进度。完成后实际总成本与有效模型前缀+恢复段分别写入training_complete.json和比较CSV，不隐藏恢复成本。活动日志仅保留首轮≤7K及新的恢复轨迹；中断轮完整日志独立保留。

后续原生渲染、208评价、TSDF、原生Viewer验证和E14依赖队列等待实际结果。当前没有30K QGS Mesh，未填推测路径。

来源：[固定官方QGS](https://github.com/will-zzy/QGS/tree/74d05c945e99fcaef7afe5a8831903be71ad9b55)、[已安装PyTorch版本CUDA分配器文档](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/cuda.rst)、Context7读取的[当前文档与兼容别名](https://github.com/pytorch/pytorch/blob/v2.10.0/docs/source/notes/cuda.rst)，以及实际CUDA错误、配置、模型和进程记录。博查检索HTTP403不可用，不以训练数据补充实测结果。
