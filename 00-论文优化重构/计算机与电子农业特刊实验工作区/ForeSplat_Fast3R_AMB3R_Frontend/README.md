# ForeSplat 几何前端实验

只研究 `KongQueZhuYu`，输入为既有 `/data/fj/F2DMAS/03-SAM/KongQueZhuYu`。所有入口强制 `CUDA_VISIBLE_DEVICES=1`，进程中的 `cuda:0` 是物理 GPU1。依赖只通过 uv 管理，两个隔离环境分别为 `.venv-fast3r`、`.venv-amb3r`。

先阅读 `DEVELOPMENT_PLAN.md`、`docs/*SOURCE_AUDIT.md`。实际结果与错误均保存 `experiments/KongQueZhuYu`；不要覆盖已有 Bundle 或失败实验。

## 可复现入口

```bash
# 不改动源数据；清单记录所有210帧，可读208帧
CUDA_VISIBLE_DEVICES=1 uv run --no-project --python .venv-fast3r/bin/python -m frontends.common

# 官方Fast3R，全208帧一次全局交互前向；保持长宽比
CUDA_VISIBLE_DEVICES=1 uv run --no-project --python .venv-fast3r/bin/python frontends/fast3r/run.py

# 官方显式FP32协议的独立全量精度对照，保留BF16主输出
CUDA_VISIBLE_DEVICES=1 uv run --no-project --python .venv-fast3r/bin/python frontends/fast3r/run.py --precision float32 --output experiments/KongQueZhuYu/Fast3R_FP32

# 官方AMB3R固定相同输入的OFF/ON；若需要另一次运行请加--suffix
CUDA_VISIBLE_DEVICES=1 uv run --no-project --python .venv-amb3r/bin/python -m frontends.amb3r.base --mode both

# 官方原生SfM / Fast3R自定义基础模型
CUDA_VISIBLE_DEVICES=1 uv run --no-project --python .venv-amb3r/bin/python frontends/sfm/run.py --model amb3r
CUDA_VISIBLE_DEVICES=1 uv run --no-project --python .venv-fast3r/bin/python frontends/sfm/run.py --model fast3r

# 仅CPU评估与科学绘图；读取已存在结果，不启动训练
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=1 uv run --no-project --python .venv-fast3r/bin/python frontends/evaluate.py --reuse-metrics
```

冒烟可用 Fast3R `--views 3 --output <新目录>` 或 SfM `--views 32 --output <新目录>`；只有 `--views 0`/默认0覆盖全部可用帧。AMB3R显存不足时 `--representative 32 --suffix _representative32` 会保留同一组固定均匀间隔帧，并明确标记子集。

SfM `--descriptor-fp32` 是单独的接口精度适配，只将原有 descriptor 值精确扩展到 FP32 后交给未修改的 whitening。默认保持原模型 dtype；若实际数值失败，用新目录及 `--retry-of` 记录重试关联，不覆盖原错误。

实际安装冻结版本：`docs/fast3r-requirements.lock.txt`、`frontends/amb3r/requirements.lock.txt`。hybrid新增依赖记录为 `docs/FAST3R_SFM_ENVIRONMENT.json`。CUDA扩展安装路径和版本兼容性见 `docs/AMB3R_SOURCE_AUDIT.md`。不安装 Fast3R cuROPE；不升级全局 PyTorch。

也可通过CPU统一入口 `python3 -m frontends.launch fast3r-sfm` 选择隔离后端，其他选项见 `--help`。后端进程返回后统一读取Bundle，无需把两个不兼容CUDA依赖集装入同一环境。

三个后端入口均接受 `--manifest <CSV>`，输入适配与模型环境分离。Fast3R/SfM 用 `--output` 指定新目录，Base 用 `--suffix` 保留独立 OFF/ON 目录。本阶段仅使用本株清单；已有输出会被保护。

## 输出语义

`frontends.common.FrontendBundle` 为后续3DGS/Mesh提供统一CPU读取接口，详见 `docs/FRONTEND_BUNDLE_INTERFACE.md`。

Fast3R的Global点图在每次输入的第一相机参考系，Local点图在各相机系。AMB3R原始 `world_points` 与 `pose` 仍是归一化尺度；`depth_metric` 为模型预测的公制深度，额外缩放几何明确带 `_metric_derived` 标签。SfM仅使用原生注册/地图融合，其最终尺度不视为已验证公制。

`conf>1` 为官方有效性判断，并不是严格高质量过滤；q85对照另存。所有前景点云完整保留；可视化采样是单独副本。没有新的输出点云平均融合、曝光修改、扫描配准或训练。

来源：[Fast3R官方源码](https://github.com/facebookresearch/fast3r)、[AMB3R官方源码](https://github.com/HengyiWang/amb3r)及本目录真实运行记录。

## 实际交付（2026-11-08）

全部五套新前端已真实处理KongQueZhuYu的208个可读帧；原始210帧中0044/0206损坏，清单保留。两条SfM均208注册/0未映射，原生73KF、混合35KF。全部结果在本项目目录持久化；七份Bundle v2审计通过。

- [运行总报告](experiments/KongQueZhuYu/RUN_REPORT.md)、[七问技术决策](experiments/KongQueZhuYu/final_analysis.md)。
- [定量CSV](experiments/KongQueZhuYu/comparison.csv)、[对照说明](experiments/KongQueZhuYu/comparison.md)。
- [整株正侧顶图](experiments/KongQueZhuYu/visualizations/whole_plant_comparison.png)、[叶片/叶缘图](experiments/KongQueZhuYu/visualizations/leaf_and_edge_geometry_uniform_gray.png)、[五视图人工2D GT](experiments/KongQueZhuYu/visualizations/manual_GT_five_view_geometry_comparison.png)。
- [统一输出接口](docs/FRONTEND_BUNDLE_INTERFACE.md)、[完整审计汇总](experiments/KongQueZhuYu/shared_config/bundle_audit_summary.json)。

当前继续使用Ω深度前端为工程基线；原生AMB3R-SfM保留为大规模注册候选。Fast3R、AMB3R Base及混合方案未给出综合替换收益，暂不进入EO。模型预测公制深度不是本植物已验证真实尺度。
