# 执行偏差、失败与证据限制

本记录区分执行准备问题、真实模型/提取失败与科研负结果。原始日志保留，不能用后续成功覆盖早期证据。最终训练、提取及指标状态以各自 run manifest 和 `reports/FINAL_ACCEPTANCE.json` 为准。

- 首次 CPU 预检查尝试对全部 Git 跟踪文件做内容哈希，包含约41 GB无关历史文件；在生成配置和首次 GPU 工作前中止。改为保存全部 Git index/status，并对本阶段相关生产源码、历史报告、冻结输入、相机、参考模型及配置共5868文件作内容哈希。记录见 `logs/FREEZE_FULL_SCAN_INTERRUPTED.txt`、`configs/FREEZE.json` 和 `reports/BASELINE_GIT_INDEX_POINTER.json`。没有实验结果用于此范围选择。
- 首次协议测试发生于配置生成前，配对配置测试因文件尚不存在报 FileNotFoundError；保留 `logs/TEST_INITIAL.txt`。冻结配置后完整协议测试通过，见 `logs/TEST_FROZEN.txt`；正式训练前新增相机轴深度测试后7项通过，见 `logs/TEST_BEFORE_FORMAL.txt`。没有通过放宽算法或评价条件消除该准备顺序错误。
- HongZhang M0 Mesh 主函数已输出成功后，进程退出和共享磁盘回收较慢。只读进程诊断工具返回权限不足，见 `logs/HongZhang_M0_TEARDOWN_DIAGNOSTIC.txt`；没有提权、杀死训练、修改提取参数或替换模型，原进程随后正常退出。主函数提取计时与控制器进程时间分开保留。
- 三个 M0 原始/后处理 Mesh 均可重载，但存在片状外部结构、开边及分离组件。固定 top-1 后处理会删去许多组件，不能认定它们全部是噪声或真实 floating Gaussian；也不能用后处理剩一组件证明植物结构完整。真实图片与逐项统计保留于 `MESH_REFERENCE_RESULTS.md`、`MESH_VISUAL_INSPECTION.md`、`figures/` 与外部 Mesh。
- 历史 M0 初始 RNG 状态哈希缺失。复用依据是禁用 M2M3 时的源码路径、训练参数、输入与硬件等价，而不是拥有历史初始化状态证据。M1/M2 的初始 RNG、参数、alpha、相机序列逐项实测对照；单次 CUDA 浮点训练仍未被证明逐位确定。
- 所有场景只有一个 Seed0，本次属于机制 pilot。验证视角不作为独立训练重复。没有独立三维 GT、真实物理尺度或新的人工前景标注；面积和深度保留 COLMAP 任意单位。共同 R0 ROI、Mesh/Gaussian 深度一致性及法向/轮廓代理不能被改称真实几何精度。
- 训练、提取和计时共享主机与磁盘；其他用户任务在另一 GPU 上运行。仅使用最初冻结且空闲的物理 GPU1，不切换 GPU，不触碰其他任务。逐步观察、校验和序列化有额外开销，历史训练 wall-clock 与本次不能构成严格因果加速对照。

正式训练故障、TSDF 敏感性失败或数值异常若发生，必须保留原目录、参数、日志与 run 状态。不得换 seed、降低设置、按验证质量重跑或把补充敏感性结果替换主提取协议。
