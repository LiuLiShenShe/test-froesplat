# 恢复、命令与大文件管理

仓库基线为 `694743e7201e1fd79b4d7cff89ad72a4db8a9e33`。所有新代码、指标、报告和科学展示图位于本阶段目录。历史 E0/E1、生产算法和论文不修改。

## 实体目录

- 项目：`/data/fj/F2DMAS`
- 冻结输入及原模型：`/data/fj/F2DMAS_phase20e0_data/f53bf8f3_v1/`
- 原 8 个实体模型：上述目录的 `phase20e1_outputs/`
- 新受管理根目录：`/data/fj/F2DMAS_phase20e2_data/694743e7_v1/`
- 新训练：`phase20e2_outputs/<scene>/<B_R0|C_R1>/seed_<1|2>/attempt_<NN>/`
- 新真实浮点渲染：`evaluation/<scene>/<group>/seed_<0|1|2>/<camera>_FLOAT_RENDER.pt`

每个成功目录保存 `chkpnt30000.pth`、`point_cloud/iteration_30000/point_cloud.ply`、`FULL_FROZEN_TRAIN_CONFIG.json`、继承 input.ply、相机 JSON 和原生日志资产。失败目录同样保留。正式模型只采用最终 30000 迭代，不从中间保存点挑选。

完整路径、大小和 SHA256 见 `reports/LARGE_ARTIFACT_MANIFEST.csv`；每个最终模型与其原配置、GPU、环境、命令和完整日志关联见 `MULTISEED_TRAINING_MANIFEST.csv`。新模型不上传 Git，原模型资产不覆盖。受管理根目录为本机实体存储，不假称远程备份；迁移/恢复必须同时复制该根目录和原 E0 数据根，不能只克隆轻量 Git 后假称已有 checkpoint。

## 运行环境

固定物理 GPU index 1、UUID `GPU-6836f890-98b2-5891-07ef-675c25199c65`。原与新实验均使用 `2d-gaussian-splatting-main/venv/bin/python`，完整 PyTorch/CUDA/扩展 `.so` hash、版本和机器信息在 `reports/PRETRAINING_ENVIRONMENT.json`。训练启动环境为 `CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1`。

进入项目根目录后，可用下列 Bash 变量简化命令（不含任何凭据）：

```bash
TASK_STAGE='/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十E2_多种子稳定性与前景三维评价'
TASK_PYTHON='/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python'
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
```

## 原命令和重算顺序

冻结命令 `preflight.py --freeze` 已执行且不可重新定义现有计划。原 `configs/EXPERIMENT_PLAN.json` 的 SHA256 保存在启动故障补充和验收记录中。只读重验：

```bash
"$TASK_PYTHON" "$TASK_STAGE/scripts/preflight.py" --label RECOVERY
"$TASK_PYTHON" -m unittest discover -s "$TASK_STAGE/tests" -v
```

真实训练的完整 argv 逐组保存在 `*_PROCESS.json` 和最终 manifest。矩阵控制器 `run_matrix.py` 按固定顺序执行，已成功且通过重载的模型直接跳过；不得为了指标再次运行同一组或覆盖其目录。原生 checkpoint 不含完整 RNG/相机栈，故中断训练不从该点恢复，按冻结失败规则从头以同 seed/config/GPU 开始。

```bash
"$TASK_PYTHON" "$TASK_STAGE/scripts/run_matrix.py"
"$TASK_PYTHON" "$TASK_STAGE/scripts/evaluate.py" --all
"$TASK_PYTHON" "$TASK_STAGE/scripts/summarize.py"
"$TASK_PYTHON" "$TASK_STAGE/scripts/recompute_metrics.py"
"$TASK_PYTHON" "$TASK_STAGE/scripts/make_figures.py"
"$TASK_PYTHON" "$TASK_STAGE/scripts/acceptance.py"
"$TASK_PYTHON" "$TASK_STAGE/scripts/write_final_reports.py"
```

本次后处理由 `finalize.py --wait-for-training` 顺序启动上述六个评价步骤。它只在矩阵控制器报告八组已完成且可重载后开始，不与训练并发使用 GPU；不会启动额外训练。实际逐步 argv、源文件 SHA256、时间、退出码和日志索引在 `reports/FINALIZATION_PROGRESS.json` 与 `logs/FINALIZE_*.txt`。任一步失败就停止并保留证据，按以上单步命令显式恢复，不覆盖失败日志或成功模型。

评价脚本跳过已完成评价，拒绝覆盖已有浮点目录。评价代码故障时保留不完整目录和日志并显式归档，修复后只重复评价，不重训已成功模型。原 Seed 0 RGB/alpha 正式分数直接用原 E1 float tensor，新 depth/normal replay 只写入新目录；渲染差异单独记录。

`recompute_metrics.py` 使用已保存 float tensor 重算全图、前景及背景/几何代理；同 checkpoint 的 HongZhang B/C alias 不重复当作独立模型。归档的显示 PNG 仅用于查看，不能当作正式指标输入。

## 本次终结恢复与图像刷新记录

首次 `finalize.py` 在 HongZhang 共享模型别名校验处停止；原 `reports/FINALIZATION_PROGRESS.json` 和 `logs/FINALIZE_recompute_metrics.txt` 保持失败原样。修复仅针对相同 checkpoint/float render 的别名身份及指标等同性断言，并增加回归检查。本次已经执行的恢复入口为：

```bash
"$TASK_PYTHON" "$TASK_STAGE/scripts/recover_finalization.py"
```

该入口重算全部唯一 float 文件，再生成图、验收和报告，不训练或重新渲染。实际命令、脚本 SHA256、退出码、耗时、日志哈希及原始失败证据哈希见 `reports/FINALIZATION_RECOVERY.json`；原失败不改写为成功。

随后直接查看真实图片时发现差值图色条标签过长，缩短文字后执行了以下显示层刷新流程：

```bash
"$TASK_PYTHON" "$TASK_STAGE/scripts/refresh_figures.py"
# 直接查看真实生成图像，并写入 reports/FIGURE_VISUAL_QA.md 后执行：
"$TASK_PYTHON" "$TASK_STAGE/scripts/finalize_figure_refresh.py"
```

第二个入口重新运行验收与最终报告生成。真实 float、数值指标和共同色阶不变。57 张索引图的 SHA256、执行命令与日志见 `reports/FIGURE_REFRESH_PROGRESS.json`。这些一次性恢复入口拒绝覆盖已有进度记录，以上命令是本次实际执行流程的归档，不表示应在现有成功交付目录中再次覆盖运行。常规独立复核仍使用前节的只读验收与 float 重算入口。

## 启动故障恢复限制

前两次 WanNianQing1/B_R0/Seed 1 启动在第一个优化前因阶段记录 JSON 类型错误退出；不伪造训练完成记录。`reports/PREOPTIMIZATION_REMEDIATION.json` 是查看新训练结果前公开记录的协议补充，保留冻结计划原文。`logs/failed_attempts/` 保存每次对应记录代码。只对这两次证据明确的零步记录故障作例外，正式完整训练仍最多两次同配置尝试，不扩展到质量不佳的运行。

## 来源与检索说明

本阶段结论仅来自真实训练、冻结资产、源码与浮点重算。Context7 的 PyTorch 2.5.1 文档已拉取并存档于 `logs/CONTEXT7_PYTORCH_2_5_1_DOCS.txt`。博查 Web Search API 返回 HTTP 403（额度不足），外部搜索不可用，没有把搜索失败隐藏为已联网查证。`/home/test/.codex/RTK.md` 在当前环境不存在，未假称读取；没有 `.codegraph/` 索引，因此未自行建立索引。
