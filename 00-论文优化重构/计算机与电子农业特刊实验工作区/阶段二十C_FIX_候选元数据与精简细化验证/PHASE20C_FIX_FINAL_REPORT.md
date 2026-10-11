# Phase20C-FIX final report

```text
PHASE20C_FIX STATUS: PARTIAL
CANDIDATE SCORE PROVENANCE: PARTIAL
R1 HARD-CASE RESCUE: PASS
SPNP-ONLY VALIDATION: PASS
RECOMMENDED METHOD: V1
GCFC STATUS: OPTIONAL
READY FOR INDEPENDENT TEST: NO
NEXT PHASE: REMEDIATION_REQUIRED
```

PARTIAL is deliberate: real original confidences are recovered at archived six-decimal
precision for 55/55 candidates, but pre-serialization tensor digits were not preserved;
the three GT10 COLMAP frames are genuinely unregistered; TEST32 human GT is absent.
SPNP-ONLY PASS means both real groups completed and were evaluated, not that every
refinement metric improved. No missing geometry or GT is synthesized.

## Completed work

- Audited Phase20C commit 970082da, clean main start, production loader/call path,
  all frozen parameters, candidate format, metadata, original masks and source RGB.
- Minimal backward-compatible metadata fix with strict mode and identity/hash checks;
  frozen R1 source SHA256 c80442a2161c0560bc7bf53a7714b6d9da5a996e2a824716a0b0dc2092071905.
- 55 matched candidates / 31 frames; historical score error ≤ 6.81443809158e-08.
  All 62 R0/R1 selections certified even under any rounding-consistent confidence.
- New CPU V0/V1 production runs on 31 frames each; four GPU smoke frames plus
  V4/V5 on 31 frames each. All outputs isolated; no old artifact overwritten.
- Comprehensive overlap/boundary/distance/leakage, oracle, identity, acceptance,
  changed pixels, per-stage, paired and runtime evidence; no cherry-picked final metric.
- Verified missing registration in both original/distorted models; coordinate caveat
  for historical GCFC reported; no COLMAP reconstruction or core algorithm change.
- TEST32 package: 32 RGB / 8 disjoint samples / 71 complete original candidates,
  no verified GT; human labels and distinct second review remain user work.
- Tests: `189 passed in 16.83s`; expected annotation gate blocks before test accuracy.

## Easy21

### sample_equal_mean — primary aggregate

| group | frames | selected_f1 | final_f1 | final_iou | final_hd95 | final_boundary_f1 | final_leakage | oracle_regret | empty_count | severe_accuracy_failure_proxy_count | spnp_accepted_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V0 | 21 | 0.983030 | 0.983030 | 0.966645 | 22.535186 | 0.408455 | 0.000942 | 0.000000 | 0 | 0 | 0 |
| V1 | 21 | 0.983030 | 0.983030 | 0.966645 | 22.535186 | 0.408455 | 0.000942 | 0.000000 | 0 | 0 | 0 |
| V4 | 21 | 0.983030 | 0.985841 | 0.972094 | 23.770994 | 0.497618 | 0.001323 | 0.000000 | 0 | 0 | 21 |
| V5 | 21 | 0.983030 | 0.985841 | 0.972094 | 23.770994 | 0.497618 | 0.001323 | 0.000000 | 0 | 0 | 21 |
| V3_HISTORY | 21 | 0.983030 | 0.985033 | 0.970527 | 24.467354 | 0.487302 | 0.001768 | 0.000000 | 0 | 0 | 21 |

### frame-level paired differences — different weights

| effect | frames | final_f1 | final_iou | final_hd95 | final_boundary_f1 | final_leakage | positive_f1_frames | negative_f1_frames |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 21 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| SPNP_R1 | 21 | 0.002981 | 0.005783 | -5.966761 | 0.081818 | 0.000398 | 20.000000 | 1.000000 |
| SPNP_R0 | 21 | 0.002981 | 0.005783 | -5.966761 | 0.081818 | 0.000398 | 20.000000 | 1.000000 |
| R1_WITH_SPNP | 21 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| FULL_CHAIN_MINUS_SPNP_DESCRIPTIVE | 21 | -0.000787 | -0.001527 | 1.027891 | -0.008271 | 0.000444 | 1.000000 | 20.000000 |

### sample-equal paired differences

| effect | frames | final_f1 | final_iou | final_hd95 | final_boundary_f1 | final_leakage |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | 21 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| SPNP_R1 | 21 | 0.002811 | 0.005449 | 1.235808 | 0.089164 | 0.000381 |
| SPNP_R0 | 21 | 0.002811 | 0.005449 | 1.235808 | 0.089164 | 0.000381 |
| R1_WITH_SPNP | 21 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| FULL_CHAIN_MINUS_SPNP_DESCRIPTIVE | 21 | -0.000808 | -0.001567 | 0.696360 | -0.010316 | 0.000445 |
## GT10

### sample_equal_mean — primary aggregate

| group | frames | selected_f1 | final_f1 | final_iou | final_hd95 | final_boundary_f1 | final_leakage | oracle_regret | empty_count | severe_accuracy_failure_proxy_count | spnp_accepted_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V0 | 10 | 0.676668 | 0.676668 | 0.655080 | 405.607321 | 0.227573 | 0.012606 | 0.296130 | 0 | 3 | 0 |
| V1 | 10 | 0.972799 | 0.972799 | 0.947439 | 56.926381 | 0.348940 | 0.001966 | 0.000000 | 0 | 0 | 0 |
| V4 | 10 | 0.972799 | 0.976853 | 0.955030 | 60.284435 | 0.406393 | 0.002727 | 0.000000 | 0 | 0 | 10 |
| V5 | 10 | 0.676668 | 0.680332 | 0.661909 | 409.318585 | 0.268059 | 0.013202 | 0.296130 | 0 | 3 | 7 |

### frame-level paired differences — different weights

| effect | frames | final_f1 | final_iou | final_hd95 | final_boundary_f1 | final_leakage | positive_f1_frames | negative_f1_frames |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 10 | 0.296130 | 0.292360 | -348.680940 | 0.121367 | -0.010640 | 3.000000 | 0.000000 |
| SPNP_R1 | 10 | 0.004054 | 0.007590 | 3.358054 | 0.057454 | 0.000761 | 10.000000 | 0.000000 |
| SPNP_R0 | 10 | 0.003664 | 0.006829 | 3.711264 | 0.040486 | 0.000596 | 7.000000 | 0.000000 |
| R1_WITH_SPNP | 10 | 0.296520 | 0.293121 | -349.034150 | 0.138335 | -0.010474 | 3.000000 | 0.000000 |

### sample-equal paired differences

| effect | frames | final_f1 | final_iou | final_hd95 | final_boundary_f1 | final_leakage |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | 10 | 0.296130 | 0.292360 | -348.680940 | 0.121367 | -0.010640 |
| SPNP_R1 | 10 | 0.004054 | 0.007590 | 3.358054 | 0.057454 | 0.000761 |
| SPNP_R0 | 10 | 0.003664 | 0.006829 | 3.711264 | 0.040486 | 0.000596 |
| R1_WITH_SPNP | 10 | 0.296520 | 0.293121 | -349.034150 | 0.138335 | -0.010474 |


## Decision and remaining work

Recommend V1; see `FINAL_FOREGROUND_METHOD_DECISION.md` for multi-metric reasons
and optional-module policy. The archive-precision limit is documented, not repaired
with guessed confidence or changed ranking. Independent validation can proceed only
after human target/plant/pot/cube annotations and manual difficulty/second review.
Those actions are in `PHASE20D_INDEPENDENT_VALIDATION_PLAN.md`. Do not retune on TEST32.
No claim of independent validation, full GT10 V3, universal refinement effectiveness,
paper rewrite, 2DGS or M2M3-GO experiment is made.

## Evidence and reproducibility

Requested deliverables are in this directory. `small_run_outputs/` archives true small
production parameters, candidate scores, selections, runtime and failure summaries.
`logs/` contains original stdout/stderr as tracked text; GPU masks/RGB remain local
with paths, hashes and configurations in `artifact_hashes/` and execution records.
`ARTIFACT_RECOVERY.md` explains ignored data recovery/reproduction; missing source
archives are explicitly required, not replaced with inferred masks.

The external Bocha search failed HTTP403 (quota exhausted), and the requested GitHub
fetch yielded no usable page. All experiment conclusions here come from actual local
frozen files and new runs, not training-data assertions. No secrets are in artifacts.
Final boundary: commit/push these verified small files and stop; do not enter Phase20D
or alter manuscript text. Git submission status is recorded separately.
