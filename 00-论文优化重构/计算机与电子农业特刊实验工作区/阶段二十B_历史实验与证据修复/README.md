# Phase 20B：历史实验与证据修复

本目录对应 `c119498c` 之后的 Phase 20B。目标是恢复可追溯的历史证据、统一分割评价口径、核对旧 2DGS 输入、核查 Table 7 和 M2M3-GO，并只准备 Phase 20C 的 V0–V3 配置；本阶段不执行 GPU 推理、2DGS 训练、COLMAP 或 Phase 20C。

## 结论摘要

- Easy20 的现有 GT 已在内存中核对为 `potted_clean = potted_plant AND NOT blue_cube`；31/31 个可用 GT 记录通过核对，未修改 GT 文件。
- Table 3、4、5 已用真实预测重算，主口径为 `sample_equal_mean`；可复算数字见 `tables/TABLE3_RECALCULATED.csv`、`TABLE4_RECALCULATED.csv`、`TABLE5_RECALCULATED.csv`。
- 旧 19 个 SAM 目录中，只有 6 个样本存在有身份验证的 P6 交叉帧；43 个有效版本-帧比较中 0 个 `EXACT_MATCH`。这不支持把旧 2DGS 直接命名为 P6+R1 结果。
- `植株数据.xlsx` 只有人工/虚拟表型记录，没有逐样本盆径、尺度系数、Mesh 版本或点选记录；尺度恢复为 `USER_INPUT_REQUIRED`。
- Table 7 的 20 序列平均、统一视图划分和四种方法的 wall-clock 时间没有闭合证据；M2M3 仅有 3 场景 pilot 记录。
- R1 离线归档公式复核为 `182/182 PASS`；0.25 差异来自 `base_total` 与含风险惩罚的 `total_score` 混比，不修改 `ranking_policy.py`。
- V0–V3 已生成 20 个配置行，真实输入、候选缓存、COLMAP sparse、SAM3 repo/checkpoint 预检为 `20/20 READY_FOR_PHASE20C`；没有启动命令。

## 目录

- `reports/`：审计报告、执行命令、V0–V3 配置 JSON 和最终准备计划。
- `tables/`：所有可重算 CSV；每张表保留来源、状态、聚合方式或哈希字段。
- `scripts/`：CPU 审计、指标重算、来源盘点、R1 复核和 V0–V3 预检脚本。
- `tests/`：不启动 GPU 的一致性测试。
- `PHASE20B_FINAL_REPORT.md`：最终状态、14 个验收问题和下一阶段阻断项。

## 可重复运行

在仓库根目录执行：

```bash
STAGE='00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十B_历史实验与证据修复'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$STAGE/scripts" python "$STAGE/scripts/segmentation.py"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$STAGE/scripts" python "$STAGE/scripts/mask_identity.py"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$STAGE/scripts" python "$STAGE/scripts/evidence_inventory.py"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$STAGE/scripts" python "$STAGE/scripts/table7_audit.py"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$STAGE/scripts" python "$STAGE/scripts/m2m3_audit.py"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$STAGE/scripts" python "$STAGE/scripts/r1_recheck.py"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$STAGE/scripts" python "$STAGE/scripts/prepare_v0_v3.py"
PYTHONDONTWRITEBYTECODE=1 python -m pytest "$STAGE/tests" -q
```

上述脚本只读取真实仓库数据并写入本阶段目录；`prepare_v0_v3.py` 只创建阶段内的输入别名和候选缓存符号链接，不复制视频、模型权重或原始 GT。

## 证据状态约定

- `VERIFIED`：文件、身份和计算路径可由本阶段脚本重建；不等于原论文叙述全部正确。
- `OFFLINE_REPLAY`：基于已归档候选离线重放，不是生产推理运行。
- `INTERMEDIATE_OUTPUT_NOT_INDEPENDENT_RUN`：来自某次流水线中间产物，不能充当独立消融。
- `UNVERIFIED` / `NOT_VERIFIED`：缺少完整身份、版本、划分或调用链证据。
- `USER_INPUT_REQUIRED`：需要用户补充真实资料，禁止用估计值替代。

## 外部检索说明

本阶段优先使用仓库文件、Git 历史和真实产物。按任务要求尝试 Bocha Web Search API 时返回 HTTP 403（余额/套餐配额不足），因此没有把网络搜索结果当作证据；本阶段结论均以本地文件和脚本输出为准。
