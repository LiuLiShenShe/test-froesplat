# configs/ — 真实运行配置快照

本目录**不手写**任何数值。每个文件都由 `scripts/build_configs.py` 从下述来源生成，
重新生成即可覆盖（`python scripts/build_configs.py`）。

| 文件 | 来源 | 含义 |
|---|---|---|
| `R6_RAP_FULL_real.json` | `runs/R6_RAP_FULL_real/参数.json` | 5 帧全链路 RAP-FSAM3 的完整 138 项参数（含 SG-MPS/SB-FBPR/GCFC 全部开关） |
| `KongQueZhuYu_RAP_FULL_210.json` | `runs/KongQueZhuYu_RAP_FULL_210/参数.json` | 全场景 210 帧完整 RAP 的参数（与 5 帧一致，输入目录与 GT 不同） |
| `downstream_recipe_actual.json` | `runs/downstream/KongQueZhuYu_C_R1_baseline/run_record.json` | C_R1 基线训练/评测的**实际**命令行、耗时、掩膜清单哈希 |
| `dev_ablation_rungs.json` | `scripts/run_dev_ablation.py` 的 `CONFIGS` / `R6` 常量 | 跨场景 4 场景 16 帧消融的 10 个档位旗标 |
| `m2m3_go_flags.json` | `2d-gaussian-splatting-m2m3-floor40/arguments/__init__.py` | M2M3-GO 全部 `--m2m3_go_*` 默认值（论文 §2.7 公式的代码侧默认） |

## 为什么可以直接引用

- RAP 侧：生产脚本在**开始推理之前**把 `argparse` 的完整命名空间写成 `参数.json`，
  因此该文件就是"那一次运行到底用了什么"，不是事后复述。
- 下游侧：`run_downstream_2dgs.py` 把**实际执行的命令行字符串**（含解释器绝对路径）
  写入 `run_record.json`，并同时记录 `train_returncode` / `eval_returncode` / `train_wall_sec`。
- M2M3-GO：直接从参数解析器源码解析，避免"论文写 quantile、代码默认 fixed"这类漂移
  （本次核查确认：源码默认 `--m2m3-go-tau-mode fixed`，而论文 §2.7 与阶段二十一的
  实验档位都要求 `quantile`，运行时**必须显式传**，`07_M2M3_GO_INTEGRATION.md` §5 已据此写死）。

## 与日志的交叉核对

| 核对面 | 位置 |
|---|---|
| 5 帧全链路逐帧参数 | `runs/R6_RAP_FULL_real/参数.json` + `逐帧报告.csv` |
| 每档训练命令 | `runs/downstream/*/run_record.json` → `train_cmd` |
| 消融各档命令 | `logs/dev_<scene>_<config>.log` 首行是完整 `command` 数组 |
| 交付物索引 | `run_manifest.json`（33 个 run） |
