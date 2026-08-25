# 新版数据管理目录规范

本目录是 Plant-aware 2DGS 重构项目的新版执行数据存放地。后续代码开发、实验重跑和论文统计优先从这里读取数据索引。

根目录：

```text
/data/fj/F2DMAS/00-论文优化重构/数据管理
```

## 1. 设计原则

- 不直接破坏旧目录结果。
- 大体量图像、COLMAP、Gaussian、mesh 默认使用软链接集中管理。
- 新版实验输出统一写入本目录下的 `06-实验输出/`。
- 每个样本使用原始样本名作为目录名，例如 `CaoMei1`、`KongQueZhuYu`。
- 每个方法或阶段使用两位数字前缀，保证排序稳定。
- 任何可用于论文表格的数据都必须能追溯来源路径、生成命令和状态。

## 2. 目录结构

```text
数据管理/
├── 00-规范与索引/
│   ├── README.md
│   └── dataset_index.csv
├── 01-输入图像/
│   ├── 01-raw_frames/
│   ├── 02-fft_frames/
│   └── 03-pr_iqa_frames/
├── 02-位姿COLMAP/
│   ├── 01-current_ok/
│   ├── 02-rerun_original_candidates/
│   └── 03-final_locked/
├── 03-分割Mask/
│   ├── 01-gt_masks/
│   ├── 02-sam_masks/
│   ├── 03-seem_masks/
│   └── 04-fsam3_masks/
├── 04-重建结果/
│   ├── 01-2dgs_gaussians_existing/
│   ├── 02-2dgs_mesh_existing/
│   ├── 03-3dgs_sugar_gaussians_existing/
│   └── 04-3dgs_sugar_mesh_existing/
├── 05-评测结果/
│   └── 01-existing_metrics/
├── 06-实验输出/
└── 07-运行脚本与超参/
```

## 3. 命名规则

样本目录：

```text
<SampleName>/
```

阶段目录：

```text
<two_digit>-<stage_name>/
```

实验输出目录：

```text
06-实验输出/<sample>/<method_tag>/
```

`method_tag` 示例：

```text
E2_2dgs_baseline
E3_fsam3_preprocess
E5_pr_iqa_fsam3
E6_mask_constrained
E7_mask_pruning
E8_full_plant_aware
M1_A4_pr_iqa_full
M3_A4_mask_constrained
M4_A4_topology_pruning
M5_A2_edge_tsdf
```

## 4. COLMAP 数据规则

COLMAP 分三层：

- `01-current_ok/`：旧 `04-COLMAP` 中当前可用结果。
- `02-rerun_original_candidates/`：使用原图或 FFT 保留原图重新跑出的候选结果。
- `03-final_locked/`：当前确认或阶段性选用的最佳位姿版本，训练入口优先读取这里。

当前规则：

- 若某样本有成功的非破坏重跑结果，`03-final_locked/` 优先链接该结果。
- 若没有重跑结果但旧 `04-COLMAP` 可用，`03-final_locked/` 链接旧可用结果。
- 若旧结果失败且重跑仍未成功，该样本暂不进入 `03-final_locked/`。
- `03-final_locked/` 是软链接层，不覆盖旧 `04-COLMAP`。

## 5. 软链接说明

本目录默认使用软链接集中旧资源：

- 节省空间。
- 保留旧目录作为原始证据。
- 后续代码只需要读取新版统一入口。

如需生成可搬迁归档包，再单独做实体复制版本。

## 6. 索引文件

`dataset_index.csv` 记录每个样本当前可用数据状态，至少包括：

- sample
- raw_frames
- fft_frames
- has_gt
- current_colmap_status
- rerun_colmap_status
- selected_colmap_path
- notes

索引用脚本生成，不能只靠手工表格。

## 7. 运行脚本与超参

每一个阶段运行过的 Python 脚本、shell 命令、config/超参和日志入口统一放在：

```text
数据管理/07-运行脚本与超参/
```

阶段命名：

```text
S0-环境与输出规范
S1-数据盘点与样本冻结
S2-抽帧与基础质量筛选
S3-COLMAP位姿重跑与锁定
S4-2DGS-baseline回归
S5-M2-FSAM3-Mask生成与对齐
S6-M1-PR-IQA视图质量诊断
S7-M3-Mask-constrained-2DGS
S8-M4-Topology-aware-pruning
S9-M5-Edge-aware-meshing
S10-主实验指标汇总与图表
```

每个阶段固定包含：

```text
scripts/
configs/
docs/
logs/
```

后续任何新增脚本和关键超参，都必须同步到这里，保证从 `数据管理` 目录就能复现每个阶段。
