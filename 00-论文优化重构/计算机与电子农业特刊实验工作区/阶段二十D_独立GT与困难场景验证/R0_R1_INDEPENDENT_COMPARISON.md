# R0/R1 independent comparison

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

来源：本目录真实审计 CSV/JSON/日志；Phase16 冻结 manifest、Phase17A U00_Control、Phase17C 历史行为表及 Phase20C-FIX 决策/生产源。
