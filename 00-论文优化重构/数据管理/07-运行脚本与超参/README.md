# 运行脚本与超参管理规范

本目录用于存放 Plant-aware 2DGS 新版执行中每一个阶段的运行脚本、命令、配置/超参文档和日志入口。

根目录：

```text
/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参
```

## 1. 阶段划分

```text
S0-环境与输出规范/
S1-数据盘点与样本冻结/
S2-抽帧与基础质量筛选/
S3-COLMAP位姿重跑与锁定/
S4-2DGS-baseline回归/
S5-M2-FSAM3-Mask生成与对齐/
S6-M1-H-VQG视图质量门控/
S7-M3-Mask-constrained-2DGS/
S8-M4-Topology-aware-pruning/
S9-M5-Edge-aware-meshing/
S10-主实验指标汇总与图表/
```

## 2. 每个阶段内部结构

每个阶段固定包含：

```text
scripts/   # Python / shell 脚本，或指向实际脚本的软链接
configs/   # yaml/json/txt 超参配置
docs/      # 阶段说明、命令记录、参数解释、失败记录
logs/      # 运行日志，或指向实际 log 的软链接
```

## 3. 命名规则

脚本命名：

```text
<stage>_<purpose>.py
<stage>_<purpose>.sh
```

配置命名：

```text
<stage>_<sample_or_all>_<method_tag>.yaml
```

命令记录：

```text
docs/commands_<YYYYMMDD>.md
```

日志命名：

```text
logs/<sample>_<method_tag>_<YYYYMMDD_HHMMSS>.log
```

## 4. 当前已纳入脚本

S1：

- `build_data_management_index.py`：生成新版 `数据管理` 的软链接和 `dataset_index.csv/json`。

S3：

- `rerun_failed_original_linux.py`：用 FFT 保留后的原始 RGB 帧非破坏重跑失败 COLMAP 样本。

## 5. 使用规则

- 所有后续新增 Python 脚本都要复制或软链接到对应阶段的 `scripts/`。
- 所有关键超参必须保存到对应阶段的 `configs/`。
- 一次运行的完整命令必须写入对应阶段的 `docs/commands_*.md` 或实验输出的 `command.txt`。
- 如果日志体积很大，可以只在 `logs/` 中放软链接。
- 旧目录中的脚本可以保留，但新版执行文档必须能从这里找到入口。
