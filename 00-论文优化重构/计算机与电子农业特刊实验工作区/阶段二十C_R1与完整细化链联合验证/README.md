# Phase 20C — R1 × RAP-FSAM3 Refinement Integration

本目录只承载 Phase 20C 的输入映射、配置、运行日志、指标、图件和判定证据；不修改 2DGS、M2M3-GO、GT、R1 权重或论文正文。

## 冻结因素

| group | ranking | SPNP | residual repair | GCFC |
| --- | --- | --- | --- | --- |
| V0 | base | OFF | OFF | OFF |
| V1 | r1 | OFF | OFF | OFF |
| V2 | base | ON | ON | ON |
| V3 | r1 | ON | ON | ON |

所有组固定 P6、`per_instance`、`single`、无 prompt ensemble、A6/A7 OFF、同一候选别名、同一评分权重与 GT。Easy21 为开发回归集，GT10 为历史困难开发诊断集；两者不能被称为独立泛化测试集。

## 资产与可复现性

`scripts/prepare_phase20c.py` 只创建新目录内的符号链接、CSV、JSON 和命令，不复制大体积候选掩膜。每个输入、GT 和候选均在 `configs/PHASE20C_INPUT_MANIFEST.csv` 中记录路径、尺寸和 SHA256。原始候选及其逐实例评分 CSV 保留在 Phase 12/17A 目录。

当前生产复用入口 `生成RAP-FSAM3掩膜.py::load_existing_candidates` 读取 raw mask，但不会读取历史候选评分 CSV，因而将复用候选的 `sam_score` 置为 `0.0`。这不是本阶段偷偷修复的算法行为；正式报告必须单独披露其对“生产复放等价性”的影响。

## 运行顺序

```bash
python scripts/prepare_phase20c.py
python scripts/run_phase20c.py --dataset Easy21 --group V0
python scripts/run_phase20c.py --dataset Easy21 --group V1
python scripts/run_phase20c.py --dataset GT10 --group V0
python scripts/run_phase20c.py --dataset GT10 --group V1
python scripts/run_phase20c.py --dataset Easy21 --group V2
python scripts/run_phase20c.py --dataset Easy21 --group V3
python scripts/run_phase20c.py --dataset GT10 --group V2
python scripts/run_phase20c.py --dataset GT10 --group V3
```

V2/V3 必须在 V0/V1 smoke 和输入审计通过后运行。若任一 GT10 几何映射、SAM3 refinement 或候选元数据等价性未闭合，报告状态只能为 `PARTIAL` 或 `BLOCKED`。
