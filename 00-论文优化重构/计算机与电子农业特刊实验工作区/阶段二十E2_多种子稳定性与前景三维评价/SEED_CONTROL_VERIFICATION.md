# Seed 控制验证与继承限制

## 真实设置路径

Phase20E-1 使用冻结生产代码 `2d-gaussian-splatting-main/utils/general_utils.py:safe_state(False)`，依次设置 Python random、NumPy、`torch.manual_seed(0)`，并选择逻辑 CUDA 0。原入口没有独立 seed 参数。当前阶段包装位于 `scripts/protocol.py:set_seed()`：先调用原路径；Seed 1/2 在模型初始化前以指定 seed 重设 Python、NumPy、PyTorch CPU 及全部可见 CUDA RNG。Seed 0 重放直接走原路径。

包装在原生 `training()` 调用前完成设置。原生函数内部依次初始化 GaussianModel、Scene、相机列表和 optimizer。本阶段在 Scene 返回和 optimizer 创建前后读取随机状态并做 SHA256，不抽取随机数；相机选择包装只调用原 `randint` 一次，再记录所选图像。原训练数学、优化器、损失、增密/剪枝和 CUDA kernel 均未改动。未开启新的 deterministic 算法选项，原 CUDA 扩展仍存在可能的非确定性。

## 已执行的五个独立进程验证

`scripts/verify_seed_control.py` 在同一物理 GPU UUID 上执行 legacy0、stage0、stage1a、stage1b、stage2 五个全新进程。

`reports/SEED_CONTROL_VERIFICATION.json` 的实际检查包括：

- 原 `safe_state(0)` 与包装 Seed 0 的全部初始 RNG hash、后续 Python/NumPy/CPU/CUDA 抽样及模拟相机顺序完全相同。
- 两次 Seed 1 的状态及抽样一致。
- Seed 1 与 Seed 2 的 Python、NumPy、CPU 和 CUDA 初始 hash 均不同，后续真实抽样也不同。
- 在当前安装 PyTorch 中，`torch.manual_seed` 后显式重设所有 CUDA seed 不改变预期状态；这一点经过实际状态比较。
- 全部非 seed/路径训练参数相同；历史记录明确 seed=0。

实际 JSON 原始证据为 `reports/RNG_legacy0.json`、`RNG_stage0.json`、`RNG_stage1a.json`、`RNG_stage1b.json`、`RNG_stage2.json`。SHA 与状态摘要来自实际运行，不是构造的随机数演示。

## 新训练的逐模型证据

每次真正启动保存 `*_INITIALIZATION.json`：初始四类 RNG、实际有序训练/验证相机列表、列表 hash、继承 sparse PLY hash、六个实际 Gaussian 初始张量 hash，以及 Scene 后、optimizer 前/后的状态。C 在第一次优化前与同 seed 的 B 比较这些字段，只有全部相同才继续。

配置和输入验证覆盖冻结相机、视图集合、输入图像/候选/掩膜和 sparse points；训练 seed 不改变相机成员或继承点。不同 seed 可以改变原生随机 rotation 初始化与相机采样顺序，这是本阶段要测量的训练随机性；只要求同 seed 的 B/C 六项初始张量一致，不把跨 seed 的 rotation 强制写成相同。每次训练曲线还保存 30,000 个实际相机选择，最终验收按 B/C 同 seed 比较该序列。

最终逐对检查与结果保存于 `reports/ACTUAL_TRAINING_SEED_VERIFICATION.json`；该文件仅在真实训练完成后生成。完整参数、命令、GPU、时间、峰值内存和最终模型 hash 见 `MULTISEED_TRAINING_MANIFEST.csv`。

## Seed 0 可比性的明确边界

历史 Seed 0 运行没有逐进程保存最初 RNG state hash；不能事后补造或把当前重放的 hash 填进旧记录。本阶段归档将其标为 `HISTORICAL_NOT_CAPTURED`。历史源码 hash、safe_state 调用、seed=0 运行记录、相同环境和物理 GPU、原始输入及历史 xyz/scale/rotation/opacity 四项初始参数断言，以及现在的原路径重放共同支持**源码路径等价**。

因此三种子表是“历史原路径 Seed 0 + 源码等价包装 Seed 1/2”的描述性比较，不能称为三个均拥有相同现场 RNG 取证的严格同包装运行。历史现场状态无法恢复，未作逐位确定性或严格跨版本复现声明。新增 Seed 1/2 的四组 B/C 配对拥有实际逐模型初始化证据，可单独核验；最终统计另行保留这两个新增 seed 的描述，避免三种子均值掩盖继承限制。

若有人要求比源码等价更强的严格三种子证据，应另立最小重放：同冻结输入/原算法/同 GPU，以统一包装重新执行 Seed 0 的 B/C 配对并另存，不覆盖旧模型。本阶段不追加这四组训练，也不将该未执行方案当作已验证结果。三种子统计不提供 p 值、置信显著性或独立几何精度结论。

## 零步启动故障

WanNianQing1/B_R0/Seed 1 的 attempt01、attempt02 均因阶段记录代码的 NumPy 标量 JSON 序列化错误退出，observed_iteration=0、optimizer_steps=0。源脚本、异常日志和生成资产分别完整存档。冻结计划未改写；查看任何新增优化结果前另存 `reports/PREOPTIMIZATION_REMEDIATION.json`，披露原 retry trigger 未覆盖此类记录故障，以及将这两次零步启动单独计数的补充规则。真正 full training 的最大尝试数仍为 2，seed、配置、GPU 和数据均未变更。没有依据图像质量或指标重跑。

## 文档与检索来源

当前安装环境和原生产源码是本验证的首要证据。Context7 已检索 `/pytorch/pytorch` 和 `/pytorch/pytorch/v2.5.1`；版本文档原文保存在 `logs/CONTEXT7_PYTORCH_2_5_1_DOCS.txt`。官方 reproducibility 文档入口为 https://pytorch.org/docs/stable/notes/randomness.html ，但网络工具未取得可用正文，未将其作为实验证据。博查搜索返回 HTTP 403（额度不足），因此外部搜索不可用；本文件所有实验性断言均来自仓库源码和实际测量，没有用训练知识补造结果。
