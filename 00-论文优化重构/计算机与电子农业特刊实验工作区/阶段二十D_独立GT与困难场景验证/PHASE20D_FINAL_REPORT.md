# ForeSplat Phase20D final report — 2026-10-11

```text
PHASE20D STATUS: PARTIAL
HUMAN GT: USER_ANNOTATION_REQUIRED
TEST32 EVALUATION: BLOCKED
R1 INDEPENDENT VALIDATION: NOT VERIFIED
DIFFICULT CASE LABELS: PARTIAL
FOREGROUND METHOD: PROVISIONAL
PHASE20E READY: NO
```

PARTIAL means the original-input audit and honest annotation preparation are complete;
the research objective of independent validation is BLOCKED. It does not mean partially
measured test accuracy. DIFFICULT CASE LABELS PARTIAL is the requested status vocabulary:
actual manual-confirmed labels are 0/32, and all five figure categories are NOT COVERED.

## Completed reliable work

- Start: clean main at `898cec2a`; no pre-existing edits. Read the requested Phase20C-FIX
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
  0/32 verified frames and explicitly requires user annotation.
- Preserved all historical 32-frame selections: only two changed (2/32=6.25%).
  CaoMei2_0146: 04 -> 03; WanNianQing1_0227: 00 -> 01. Both benefit/degradation
  judgments are BLOCKED, not inferred from area, contrast, score or nonempty masks.
- Delivered blocked result schemas, statistical/metric definitions, manual identity
  review queue, figure plan and non-lock decision. All unmeasured metric values are blank.

| sequence | RGB | candidates | manual JSON | approved GT |
| --- | --- | --- | --- | --- |
| CaoMei2 | 4 | 15 | 0 | 0 |
| ChangShouHua3 | 4 | 4 | 0 | 0 |
| DouBanLv3 | 4 | 4 | 0 | 0 |
| HongZhang | 4 | 4 | 0 | 0 |
| WanNianQing1 | 4 | 6 | 0 | 0 |
| WangWenCao2 | 4 | 6 | 0 | 0 |
| XianKeLai3 | 4 | 7 | 0 | 0 |
| XiangPiShu2 | 4 | 25 | 0 | 0 |

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
audit. 32/32 convenience copies differ in file hash, so all new annotation
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

来源：本目录真实审计 CSV/JSON/日志；Phase16 冻结 manifest、Phase17A U00_Control、Phase17C 历史行为表及 Phase20C-FIX 决策/生产源。
Bocha search returned HTTP403 on 2026-10-11; external repository search supplied no
usable experimental evidence. Conclusions use observed local artifacts, not unsourced
training-data claims. `/home/test/.codex/RTK.md` is missing. No CodeGraph index exists,
so no index was created. No API credentials appear in published files.
