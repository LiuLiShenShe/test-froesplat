# ForeSplat Phase20D — 独立 GT 与困难场景验证

本目录仅完成能可靠完成的前置审计。缺失人工 GT 时不运行 R0/R1、不重新评分、
不推理、不生成 GT、不计算任何 TEST32 分割准确率或绘制虚假困难案例。

- 最终状态及证据：`PHASE20D_FINAL_REPORT.md`、`INPUT_AUDIT_STATUS.json`。
- 真实输入：`TEST32_INPUT_MANIFEST.csv`（32 行）、`TEST32_CANDIDATE_PROVENANCE.csv`
  （71 行），原始文件来源和 SHA256 为 `SOURCE_HASHES.csv`。
- 人工待办：`TEST32_ANNOTATION_REVIEW.csv`、`DIFFICULTY_HUMAN_VERIFICATION.csv`。
  先阅读 `annotation_package/ANNOTATION_INSTRUCTIONS.md`；所有人工字段当前空白。
- `TEST32_PER_FRAME_RESULTS.csv`、`TEST32_SAMPLE_SUMMARY.csv` 只包含 BLOCKED
  状态和空的指标列；它们是结果模式，不是实验结果，空值不是零。
- `configs/FROZEN_V1_PARAMETERS.json` 与 `FROZEN_R0_PARAMETERS.json` 是 Phase20C-FIX
  完整生产参数的可追溯快照，不是已执行的 TEST32 配置（历史 DEV 路径保留）。
- `metadata/TEST32/` 是真实 U00_Control score/box/ID/hash 的严格 loader 快照；
  保留历史六位小数，未恢复不存在的张量原始位数，未重算总分或选择。
- Git 只保存小型脚本/配置/CSV/报告/日志，RGB、候选、上下文、GT JSON 本地保留。
  没有人工批准，本阶段没有任何论文可用的独立准确率表或困难场景图。

## 可复现前置审计

```bash
STAGE='00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十D_独立GT与困难场景验证'
/home/test/biosoft/enter/bin/python "$STAGE/scripts/audit_inputs.py" \
  --output-dir /tmp/foresplat-phase20d-new-audit
/home/test/biosoft/enter/bin/python "$STAGE/scripts/validate_annotations.py" \
  --output "$STAGE/logs/annotation_gate_new_attempt.json"
/home/test/biosoft/enter/bin/python -I -m pytest "$STAGE/tests" -q
```

审计必须恢复准确的原始文件；缺失/不匹配候选时该帧 BLOCKED，禁止生成替代候选。
首次审计拒绝覆盖既有表格、审核表和日志；重跑请使用新的目录。
完整恢复说明见前置 Phase20C-FIX `ARTIFACT_RECOVERY.md` 与本目录 `SOURCE_HASHES.csv`。
预检通过仅说明输入记录/几何关卡通过，不自动锁定方法，也不自动进入评价或 Phase20E。

来源：本地实际文件、冻结 manifest、用户 Phase20D 规范。博查检索 HTTP403；外部
公开仓库检索不是本实验原始数据或人工审核证据，未用于推断任何 TEST32 结论。
