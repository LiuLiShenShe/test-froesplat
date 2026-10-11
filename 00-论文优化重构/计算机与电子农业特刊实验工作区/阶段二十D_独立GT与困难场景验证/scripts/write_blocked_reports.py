from __future__ import annotations

import json
from collections import defaultdict

from phase20d_common import BASELINE, DIFFICULTIES, METRICS, WORK, read_csv, write_csv, write_json, write_text


STATUS = """PHASE20D STATUS: PARTIAL
HUMAN GT: USER_ANNOTATION_REQUIRED
TEST32 EVALUATION: BLOCKED
R1 INDEPENDENT VALIDATION: NOT VERIFIED
DIFFICULT CASE LABELS: PARTIAL
FOREGROUND METHOD: PROVISIONAL
PHASE20E READY: NO"""


def main():
    audit = json.loads((WORK / "INPUT_AUDIT_STATUS.json").read_text())
    gate = json.loads((WORK / "logs/annotation_gate.json").read_text())
    if audit["input_audit"] != "PASS" or gate["evaluation_allowed"] or gate["annotation_files_found"] != 0:
        raise ValueError("This report generator is only for the observed complete-input/no-human-GT case")
    inputs = read_csv(WORK / "TEST32_INPUT_MANIFEST.csv")
    candidates = read_csv(WORK / "TEST32_CANDIDATE_PROVENANCE.csv")
    changes = read_csv(WORK / "TEST32_HISTORICAL_SELECTION_CHANGE.csv")
    by_sample = defaultdict(list)
    for row in inputs:
        by_sample[row["sample"]].append(row)
    per_frame = []
    for row in inputs:
        for method in ("R0", "R1"):
            per_frame.append({"frame_id": row["frame_id"], "sample": row["sample"], "frame": row["frame"],
                              "method": method, "status": "BLOCKED", "reason": "USER_ANNOTATION_REQUIRED",
                              "evaluation_executed": "NO", "selected_candidate_id": "", "annotation_sha256": "",
                              "prediction_sha256": "", **{metric: "" for metric in METRICS}})
    write_csv(WORK / "TEST32_PER_FRAME_RESULTS.csv", per_frame)
    summaries = []
    for sample, frames in by_sample.items():
        for method in ("R0", "R1", "PAIRED_R1_MINUS_R0"):
            summaries.append({"sample": sample, "method": method, "aggregation": "sequence_mean",
                              "expected_frames": len(frames), "evaluated_frames": 0, "status": "BLOCKED",
                              "selection_change_rate": "", **{metric: "" for metric in METRICS}})
    for method in ("R0", "R1", "PAIRED_R1_MINUS_R0"):
        for aggregation in ("sample_equal_mean", "frame_level_mean", "frame_median", "frame_sd_ddof1",
                            "sequence_mean_median", "sequence_mean_sd_ddof1"):
            summaries.append({"sample": "ALL_TEST32", "method": method, "aggregation": aggregation,
                              "expected_frames": 32, "evaluated_frames": 0, "status": "BLOCKED",
                              "selection_change_rate": "", **{metric: "" for metric in METRICS}})
    write_csv(WORK / "TEST32_SAMPLE_SUMMARY.csv", summaries)
    pending = [{"frame_id": row["frame_id"], "rgb_path": row["rgb_path"], "rgb_sha256": row["rgb_sha256"],
                "annotation_json": row["annotation_json"], "annotation_status": "MISSING",
                "required_action": "MANUAL_POLYGONS_AND_DISTINCT_SECOND_HUMAN_REVIEW",
                "difficulty_review": "PENDING", "target_identity_review": "PENDING"} for row in inputs]
    write_csv(WORK / "TEST32_PENDING_ANNOTATION.csv", pending)
    identity_rows = [{"frame_id": row["frame_id"], "method": method, "selected_candidate_id": "",
                      "prediction_sha256": "", "target_identity_correct": "", "severe_human_error": "",
                      "human_visual_evidence": "", "reviewer": "", "second_reviewer": "",
                      "status": "BLOCKED_NO_PREDICTION_OR_GT"} for row in inputs for method in ("R0", "R1")]
    write_csv(WORK / "TEST32_PREDICTION_IDENTITY_REVIEW.csv", identity_rows)
    sample_table = "\n".join(f"| {sample} | {len(frames)} | {sum(int(row['candidate_count_verified']) for row in frames)} | 0 | 0 |"
                             for sample, frames in by_sample.items())
    mismatch_count = sum(row["phase16_annotation_target_matches_input"] == "0" for row in inputs)
    changed = [row for row in changes if row["historical_changed"] == "1"]
    change_table = "\n".join(f"| {row['frame_id']} | {row['historical_r0_selected']} | {row['historical_r1_selected']} | BLOCKED |"
                             for row in changed)
    frozen_hashes = audit["protected_code_sha256"]
    sources = "来源：本目录真实审计 CSV/JSON/日志；Phase16 冻结 manifest、Phase17A U00_Control、Phase17C 历史行为表及 Phase20C-FIX 决策/生产源。"
    write_text(WORK / "TEST32_GT_AUDIT.md", f"""# TEST32 GT 审计 — 2026-10-11

HUMAN GT: USER_ANNOTATION_REQUIRED
TEST32 EVALUATION: BLOCKED

## 输入与标注状态

| sequence ID | 原始 RGB | 原始 P6 候选 | 人工 JSON | 合格双人复核 |
| --- | --- | --- | --- | --- |
{sample_table}

32/32 原始 RGB 与 Phase16 哈希、Phase17A 原始生成输入、Phase20C-FIX 原图链接一致。
71/71 候选通过显式 ID、历史 score-list、box、阈值、mask hash、原始分辨率及二值像素
核验；strict metadata loader 接受。候选缺失时脚本停止受影响帧，禁止替代/合成。
32 帧均为宽 2160、高 3840 的 RGB；这里的“8 个”是冻结序列身份数，
并不是已经由人独立确认的 8 株物理个体。

Phase16 DEV26（BaiZhang、DouBanLv2）与 TEST 无 sample/sequence ID 交集；
Phase20C Easy21/GT10 的 31 个 DEV RGB 同样无序列交集。
检查文件 SHA256 与解码后 RGB SHA256：TEST 内及 TEST/上述 DEV 的完全重复匹配为 0。
邻近帧存在相关性；未人工验证近重复/跨序列物理身份，不能声称“完全排除所有泄漏”。
明细见 `DUPLICATE_FRAME_AUDIT.csv`。

`ANNOTATION_LOCATION_AUDIT.csv` 对每帧检查本阶段与 Phase20C-FIX `annotations/`、
03-GT/03-GT-区分/03-GT-区分_challenge、Phase16 target.json/annotations 共 7 个路径。
找到 TEST32 人工 JSON 0/32；原始及既有复核表均无合格双人审批，人工视觉证据 0。
预检 `logs/annotation_gate.json` 明确拒绝评价；无预测/准确率模块被读取或运行。

## 历史标注包差异

Phase16 `target.jpg` 副本有 {mismatch_count}/32 与原始 RGB 文件 hash 不同，
因此不作为冻结生成输入/新标注原图使用；记录其真实 hash，但不改写历史文件。
本阶段标注包直接链接经验证的原始 RGB。
旧 instruction card 的可选 plant/pot、linestrip、“身份不明删除帧”与新规范不一致；
新说明要求全部三个 label、polygon、保持 32 帧不变，并阻断不明确的身份。

## 人工待办

全部 32 帧列在 `TEST32_PENDING_ANNOTATION.csv`；每帧人工画 potted_plant/plant/pot、
存在时 blue_cube，记录目标身份与实际 D1/D3/D6/D11/D12（可 none）。
另一位真实独立审核者复核、签署并绑定最终 JSON SHA256。
主口径固定 `potted_clean = potted_plant AND NOT blue_cube`。
不允许 SAM3/R1/Agnes/AI 产生 GT 或替代第二审核人。

{sources}
""")
    write_text(WORK / "TEST32_SELECTION_CHANGE_AUDIT.md", f"""# TEST32 候选切换审计

状态：历史行为来源 VERIFIED；新的 R0/R1 执行及切换准确性判定 BLOCKED。

Phase17C 历史表共 32 帧，仅 2/32 = 6.25% 改变选择，另 30 帧不变。
完整 32 行均保存在 `TEST32_HISTORICAL_SELECTION_CHANGE.csv`，不筛除不变/不利帧。
本阶段只核验这些历史 ID 属于真实候选集，不重算分数，不修改历史排序。

| frame ID | historical R0 | historical R1 | 改善/退化/无明显差异 |
| --- | --- | --- | --- |
{change_table}

两帧均为 NOT VERIFIED，缺少人工 potted_clean 和独立目标身份确认，
因此不能把切换、面积变大或非空解释为改善，也不能说“无明显差异”。
历史 raw-mask 相互 IoU 或历史 empty count 不是 GT 准确率/身份正确率。
批准后必须完整保留 32 帧，对以上两帧以及任何新发生切换、退化或严重失败做同等审查。

{sources}
""")
    write_text(WORK / "R0_R1_INDEPENDENT_COMPARISON.md", f"""# R0/R1 independent comparison

TEST32 EVALUATION: BLOCKED
R1 INDEPENDENT VALIDATION: NOT VERIFIED

No new R0/R1 scoring, predictions or independent accuracy evaluation are executed.
There are no F1/IoU_fg/mIoU/HD95/Boundary-F1/Leakage/Oracle-regret values, failure counts
or human target-identity accuracy. Empty metric cells in the CSV schemas are missing,
not zero. Historical 2/32 (6.25%) selection churn is behavioral evidence only.

## Frozen comparison after all 32 human approvals

- Both policies use exactly the same 71 original P6 per-instance masks, original boxes,
  mask threshold and archived six-decimal SAM confidences, never new candidates.
- R0 uses ranking_policy=base; V1 uses ranking_policy=r1. All other production parameters,
  preprocessing, cleanup and selection rules come from Phase20C-FIX unchanged.
- P6/single/no ensemble; SPNP/Residual/GCFC/A6/A7 OFF; semantic gate and unaligned
  temporal features keep their original behavior. No GT-dependent runtime tuning.
- Reuse existing frozen metric definitions, including HD95/boundary tolerance 3 px,
  mIoU over foreground/background, Leakage=FP/(H*W). Oracle=max F1 among the original
  same-frame raw candidate masks; regret=Oracle F1 minus selected-raw F1. Keep this
  selection diagnosis distinct from final-cleaned-mask metrics.
- Report all 32 paired per-frame results, eight per-sequence means, primary sample-equal
  mean, frame mean/median/SD (ddof=1), median/SD of sequence means, paired R1-minus-R0
  differences and selection-change rate. Adjacent frames are not 32 independent plants.
- Count empty masks separately from F1<0.5 severe-accuracy proxy. Actual severe failures
  and wrong target identities require `TEST32_PREDICTION_IDENTITY_REVIEW.csv` to be
  independently signed by humans, not inferred from overlap or model confidence.
- Review CaoMei2_0146, WanNianQing1_0227, all degradations and any new failure, regardless
  of whether R1 helps. Do not change the evaluation set or retune the frozen method.
- No predeclared statistical significance test exists here; do not convert a small
  descriptive average into a significant/generalized benefit. Report absent benefit
  as absent and DEV GT10 rescues as DEV mechanism evidence only.

The old Phase20C-FIX evaluator is a verified starting point, not a complete drop-in for
this full R0/R1+Oracle+human-identity report (it is rooted at the old annotation package
and does not provide all requested summaries). It must not be called with pending GT.
This stage delivers only audit/preflight and blocked schemas; execution/figure scripts
remain BLOCKED until the human gate is passed in a subsequent explicit Phase20D action.

{sources}
""")
    difficulty_table = "\n".join(f"| {code} | {label} | 0 | NOT COVERED |" for code, label in DIFFICULTIES.items())
    write_text(WORK / "DIFFICULT_CASE_FIGURE_PLAN.md", f"""# 困难场景图件计划

DIFFICULT CASE LABELS: PARTIAL（0/32 人工核查；不是已完成部分图像判断）
图 A / 图 B: BLOCKED。没有生成图件或替代图片。

| code | 类别 | 人工确认帧数 | 当前覆盖 |
| --- | --- | --- | --- |
{difficulty_table}

NOT COVERED 表示没有人工确认的可展示样本，**不表示真实图中一定不存在该类**。
历史绿色比例/primary D1/secondary D11 只保存在明确 NOT_GT 字段中，不作为选图依据。

## 完成 GT 后的完整生成要求

- 图 A：仅从双人签署的真实困难 RGB 选择，标注 frame ID、人工标签、视觉证据。
  某类无已确认样本继续 NOT COVERED，不拍脑袋挑图，也不生成合成场景。
- 图 B：真实 RGB | 人工 GT | 冻结 R0 | 冻结 R1，展示两次指定切换及任何退化/失败。
  非困难的切换案例不能冒充困难类别，允许诚实展示中性或退化结果。
- 用实际批准后指标标注 F1/IoU，保留真实全图与展示裁剪的坐标/缩放记录。
- 新图件脚本应拒绝未批准标签/GT/预测，不修改源图，不改变评价网格；排版缩略图
  仅用于展示，不反过来用于计算指标。
- 输出源追溯 CSV：frame ID、RGB/annotation/GT/R0/R1 路径与 SHA256、标签、证据、
  人工审核人、生成脚本 hash、配置、显示裁剪/缩放参数和最终图 hash。

现在不存在任何真实审核标签与独立预测，不能合法生成论文案例。
待合格人工输入后才实现/执行最终图件生成脚本；本阶段不冒充图件已交付。

{sources}
""")
    write_text(WORK / "FINAL_FOREGROUND_METHOD_LOCK.md", f"""# Foreground method lock decision

FOREGROUND METHOD: PROVISIONAL
R1 INDEPENDENT VALIDATION: NOT VERIFIED

V1 remains the frozen DEV-supported recommendation, NOT a TEST32-validated final lock.
Recipe: P6 / default_prompt_id=P6 / per_instance / single / no prompt ensemble /
ranking_policy=r1; SPNP=OFF, Residual=OFF, GCFC=OFF, A6=OFF, A7=OFF.
All weights, penalties, thresholds and first-occurrence tie rules stay byte-frozen.

## Immutable code and parameter sources

- Baseline commit: `{audit['baseline_commit']}`.
- ranking_policy.py SHA256: `{frozen_hashes['ranking_policy.py']}`.
- production SHA256: `{frozen_hashes['生成RAP-FSAM3掩膜.py']}`.
- metadata loader SHA256: `{frozen_hashes['candidate_metadata.py']}`.
- Complete source parameters: Phase20C-FIX small_run_outputs/V1_GT10_all/参数.json,
  snapshotted without tuning in configs/FROZEN_V1_PARAMETERS.json; original source
  path/hash retained. R0 snapshot differs only in ranking_policy=base.
- Candidates, score/detail tables, RGB, checkpoint and SAM3 processor sources are
  hashed in SOURCE_HASHES.csv. No full-precision tensor confidence is claimed:
  archival confidence is six-decimal, extra unsaved digits remain NOT_SAVED.

These are provenance/preparation hashes, NOT a final independent-validation code lock.
No test score replay/rounding certificate is executed before the human gate; the DEV
62/62 certificate cannot automatically be claimed for TEST32.

## Seven required independent conclusions

1. Correct rescue of difficult multi-instance targets: NOT VERIFIED.
2. TEST32 degradations: NOT VERIFIED, never reported as zero.
3. Wrong physical target selections: NOT VERIFIED; human confirmation required.
4. Gain concentrated in a few severe failure frames: NOT VERIFIED. Historical churn
   is only 2/32; it cannot establish either widespread gains or correctness.
5. Continued suitability of P6: DEV-supported recommendation; TEST32 NOT VERIFIED.
6. Evidence-supported five-type analysis: BLOCKED, all current categories NOT COVERED.
7. Final paper foreground module lock: NOT AUTHORIZED by the available evidence.

Do not retune R1/SPNP after looking at TEST32 and call it the same independent test.
GT10 three-frame rescue is DEV mechanism evidence only. No absent test benefit is
converted to benefit; no manuscript/2DGS/M2M3-GO changes are made.

{sources}
""")
    write_text(WORK / "PHASE20E_INPUT_REQUIREMENTS.md", f"""# Phase20E input requirements

PHASE20E READY: NO
Do not enter Phase20E from this delivery.

Required before any later transition:

1. Original immutable TEST32 32-frame/eight-sequence/71-candidate audit still passes,
   with original archived confidence precision and complete source recovery.
2. All 32 manual potted_plant/plant/pot/optional blue_cube JSONs, target identity,
   original RGB grid and final hashes have actual independent human second approval.
3. Human D1/D3/D6/D11/D12/none decisions have image-specific visual evidence; absent
   confirmed categories explicitly remain NOT COVERED rather than being fabricated.
4. Finish Phase20D, not Phase20E: frozen R0/R1 runs with identical candidates, score
   reproducibility check, all requested metrics/paired/sequence/dispersion statistics,
   both specified switch cases, all degradations and separate human identity/error review.
5. Traceable real figures, scripts, source manifests and honest independent conclusion
   establish whether V1 can be LOCKED; unresolved issues cannot be hidden as PASS.
6. Preserve hashes and record final validated version, limitations and public materials,
   without large caches, force push or overwritten historical evidence.
7. Obtain explicit user authorization for Phase20E. This stage does not modify paper,
   2DGS or M2M3-GO even when annotations eventually become available.

{sources}
""")
    write_text(WORK / "PHASE20D_FINAL_REPORT.md", f"""# ForeSplat Phase20D final report — 2026-10-11

```text
{STATUS}
```

PARTIAL means the original-input audit and honest annotation preparation are complete;
the research objective of independent validation is BLOCKED. It does not mean partially
measured test accuracy. DIFFICULT CASE LABELS PARTIAL is the requested status vocabulary:
actual manual-confirmed labels are 0/32, and all five figure categories are NOT COVERED.

## Completed reliable work

- Start: clean main at `{BASELINE}`; no pre-existing edits. Read the requested Phase20C-FIX
  report/decision/provenance/plan/recovery, Phase16 DEV/TEST/spec, Phase17A original inputs,
  all 71 historical candidate/detail identities and current strict metadata/polygon gate.
- Re-audited 32 actual RGB and 71 actual binary P6 candidates at original 2160x3840;
  ID/order/score-list/box/threshold/area/RGB/mask hashes match original source records.
  Immutable production/R1/metadata/checkpoint/source hashes verified; no re-inference.
- Eight frozen sequence IDs are disjoint from DEV26 and Easy21/GT10 development IDs;
  exact file/decoded-RGB duplicates within TEST or against those DEV references: 0.
  Physical identity/near-duplicates remain human work; adjacent frames are correlated.
- Preserved original files; provided local original-RGB and context links, 32 strict
  archived-precision metadata snapshots, 32-frame pending queues and five-type evidence
  form. No candidate/GT synthesized, no human field signed by AI.
- All known TEST32 annotation locations checked: 0/32 manual JSONs. Current gate has
  {gate['verified_frames']}/32 verified frames and explicitly requires user annotation.
- Preserved all historical 32-frame selections: only two changed (2/32=6.25%).
  CaoMei2_0146: 04 -> 03; WanNianQing1_0227: 00 -> 01. Both benefit/degradation
  judgments are BLOCKED, not inferred from area, contrast, score or nonempty masks.
- Delivered blocked result schemas, statistical/metric definitions, manual identity
  review queue, figure plan and non-lock decision. All unmeasured metric values are blank.

| sequence | RGB | candidates | manual JSON | approved GT |
| --- | --- | --- | --- | --- |
{sample_table}

## Honest execution boundary

Step 1 complete. Step 2 is USER_ANNOTATION_REQUIRED. Steps 3–6 requiring approved
GT are stopped: no new R0/R1 scoring/replay, SAM3/prediction runs, accuracy evaluation,
Oracle evaluation, difficulty-image judgments, paper figures or final method lock.
Historical metadata precision is six-decimal, not original tensor exactness. No weights,
R1 formulas, risk penalties, candidate selection, evaluation set or thresholds changed.
Old DEV rounding certificates are not presented as a new TEST32 certificate.

First audit attempt mistakenly required the historical Phase16 target.jpg convenience
copy to share original-file hash, and stopped before all candidates. That validator
assumption was corrected, not original inputs or algorithm. Its logs and partial output
are preserved under logs/audit_attempt_01; final output comes from the successful second
audit. {mismatch_count}/32 convenience copies differ in file hash, so all new annotation
links point to verified originals; old instruction-card conflicts are explicitly superseded.

## Research conclusions currently possible

R1 independent rescue, regressions, severe errors, target identity correctness, benefit
concentration and P6 independent suitability are NOT VERIFIED. No valid claim of
significant benefit, broad generalization or zero regression is possible. GT10 three-frame
rescue remains development-set mechanism evidence only. Frozen V1 stays PROVISIONAL.
For each missing difficulty category: NOT COVERED (no confirmed human evidence).

## Validation and next required input

Actual final tests are recorded in logs/TEST_RESULTS_FINAL.txt (60 passed); original-boundary/hash checks in
logs/FINAL_BOUNDARY_AUDIT.json. See logs/annotation_gate.json for the expected exit-2
block before any prediction/metric. Git publication outcome is recorded separately.

The user must supply manual JSONs and distinct real-human review for all 32 frames,
fill both review/evidence tables, and rerun only the preflight. Synthetic tests do not
constitute GT and never write fixtures into TEST32 annotations. The later full evaluator
and figure generator are deliberately NOT claimed as complete while the required human
inputs are absent. PHASE20E READY remains NO; stop here, no manuscript changes.

## Sources and retrieval limitations

{sources}
Bocha search returned HTTP403 on 2026-10-11; external repository search supplied no
usable experimental evidence. Conclusions use observed local artifacts, not unsourced
training-data claims. `/home/test/.codex/RTK.md` is missing. No CodeGraph index exists,
so no index was created. No API credentials appear in published files.
""")
    write_json(WORK / "PHASE20D_STATUS.json", {"phase20d": "PARTIAL", "human_gt": "USER_ANNOTATION_REQUIRED",
                                               "test32_evaluation": "BLOCKED", "r1_independent_validation": "NOT VERIFIED",
                                               "difficulty_case_labels": "PARTIAL", "human_verified_difficulty_frames": 0,
                                               "foreground_method": "PROVISIONAL", "phase20e_ready": "NO",
                                               "steps_completed": [1], "steps_blocked": [2, 3, 4, 5, 6],
                                               "new_scoring": False, "new_predictions": False, "test_accuracy_computed": False})
    print(STATUS)


if __name__ == "__main__":
    main()
