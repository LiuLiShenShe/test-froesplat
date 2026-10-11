# SPNP-only ablation report

V4 = P6/per_instance/single/R1/SPNP ON/Residual OFF/GCFC OFF.
V5 differs only in ranking_policy=base. Both use the original candidate masks,
real archived confidences, RGB/GT/checkpoint/weights/AMP/preprocessing, and independent
output directories. `FROZEN_PARAMETER_AUDIT.csv` verifies every non-factor setting.
Two-frame smoke tests run on CaoMei1_0000 and DouBanLv2_0039 for both groups before
the complete runs. V5 deliberately keeps R0's wrong instance; its failed target is
not hidden or treated as a technical smoke failure. The smoke checks execution,
identity consistency and nonempty same-resolution masks; V4 also retains its target.

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


## Hard cases and attribution

| frame | group | selected_candidate_id | selected_f1 | final_f1 | oracle_regret | spnp_accepted | hard_case_target_identity_preserved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 39.000000 | V0 | DouBanLv2_0039_00 | 0.000000 | 0.000000 | 0.986660 | 0.000000 | 0.000000 |
| 40.000000 | V0 | DouBanLv2_0040_01 | 0.000000 | 0.000000 | 0.988160 | 0.000000 | 0.000000 |
| 41.000000 | V0 | DouBanLv2_0041_00 | 0.000000 | 0.000000 | 0.986485 | 0.000000 | 0.000000 |
| 39.000000 | V1 | DouBanLv2_0039_01 | 0.986660 | 0.986660 | 0.000000 | 0.000000 | 1.000000 |
| 40.000000 | V1 | DouBanLv2_0040_00 | 0.988160 | 0.988160 | 0.000000 | 0.000000 | 1.000000 |
| 41.000000 | V1 | DouBanLv2_0041_03 | 0.986485 | 0.986485 | 0.000000 | 0.000000 | 1.000000 |
| 39.000000 | V4 | DouBanLv2_0039_01 | 0.986660 | 0.987331 | 0.000000 | 1.000000 | 1.000000 |
| 40.000000 | V4 | DouBanLv2_0040_00 | 0.988160 | 0.989307 | 0.000000 | 1.000000 | 1.000000 |
| 41.000000 | V4 | DouBanLv2_0041_03 | 0.986485 | 0.988567 | 0.000000 | 1.000000 | 1.000000 |
| 39.000000 | V5 | DouBanLv2_0039_00 | 0.000000 | 0.000000 | 0.986660 | 0.000000 | 0.000000 |
| 40.000000 | V5 | DouBanLv2_0040_01 | 0.000000 | 0.000000 | 0.988160 | 0.000000 | 0.000000 |
| 41.000000 | V5 | DouBanLv2_0041_00 | 0.000000 | 0.000000 | 0.986485 | 0.000000 | 0.000000 |

## Every negative V4 F1 frame (not excluded)

| dataset | sample | frame | selected_f1 | final_f1 | delta_f1 | final_leakage |
| --- | --- | --- | --- | --- | --- | --- |
| Easy21 | DouBanLv1 | 100.000000 | 0.984890 | 0.982669 | -0.002221 | 0.002346 |

V4-V1 isolates SPNP under R1; V5-V0 isolates SPNP under R0; V4-V5 isolates ranking
under fixed SPNP. Do not attribute all V4-V0 gain to SPNP. Selected F1 and oracle
regret refer to pre-refinement candidates; final F1 is a different measurement.
SPNP acceptance comes from the real production `SPNP接受` field, independently
of nonzero pixel change. Changed-pixel ratio uses full image area; a separate
selected-mask-normalized ratio is also saved. Severe F1<0.5 is an accuracy proxy,
not a new manual identity annotation. Hard-case target IDs are known original
candidate identities and validated against frozen GT before and after refinement.

## Acceptance / changed pixels (real production decisions)

| dataset | group | frames | spnp_acceptance_rate | spnp_changed_frames | changed_pixel_ratio_per_image |
| --- | --- | --- | --- | --- | --- |
| Easy21 | V4 | 21 | 1.000000 | 21 | 0.002734 |
| Easy21 | V5 | 21 | 1.000000 | 21 | 0.002734 |
| GT10 | V4 | 10 | 1.000000 | 10 | 0.005244 |
| GT10 | V5 | 10 | 0.700000 | 7 | 0.004788 |

V5 GT10 accepts only 7/10 SPNP refinements; the three wrong-target R0 frames
remain failures rather than being rescued by SPNP. Guarded rejection is logged,
not counted as accepted merely because the module is enabled.

## HD95 aggregation is not interchangeable

Easy21 has sample sizes 5/5/5/5/1. sample_equal_mean weights every sample at 1/5,
while a frame-level mean gives the single-frame XianKeLai1 sample 1/21.
An improvement in frame-level mean ΔHD95 does not override sample-equal worsening.
GT10 has equal 5/5 frame counts, so these two means coincide. Both values and
per-frame deltas are retained; no favorable metric alone determines the method.

- Easy21: frame mean ΔHD95 = -5.966761px; sample-equal ΔHD95 = +1.235808px.
- GT10: frame mean ΔHD95 = +3.358054px; sample-equal ΔHD95 = +3.358054px.

具体权重解释：XianKeLai1 只有 0000 一帧，其 ΔHD95 为 +39.049294px；
sample-equal 中权重 1/5，frame mean 中权重 1/21。这一距离指标损失在两种
统计权重下贡献不同，因此 Easy21 出现逐帧均值改善但 sample-equal 退化。
这不是通过更换统计口径证明方法变好；主决策保持预设 sample-equal 权重。

## Full-chain historical comparison

V3_HISTORY is the saved Phase20C Easy21 chain, not a new restored-score V3 run.
Its zero-score input ranking is not labelled SCORE_EQUIVALENT. Actual original
RGB, candidate identities, selected mask pixels and SPNP parameters match; exact
historical SPNP-mask equality is recorded per new frame. These allow a matched
saved-mask performance comparison, but not a claim that historical scoring or
GCFC coordinate mapping was valid. GT10 has no full V3 group; no fabricated
seven-frame or full-chain result is substituted. Geometry audit explains absence.
Historical residual masks were unchanged; GCFC had mostly negative adjacent-stage
F1 effects on Easy21. No stable cross-dataset GCFC benefit is established.

## Runtime

| dataset | group | wall_sec_including_load_and_io | pipeline_elapsed_sec | gpu_used |
| --- | --- | --- | --- | --- |
| Easy21 | V0 | 328.518779 | 327.704000 | 0.000000 |
| Easy21 | V1 | 329.027658 | 328.491000 | 0.000000 |
| Easy21 | V4 | 551.080117 | 504.010000 | 1.000000 |
| Easy21 | V5 | 462.758808 | 445.007000 | 1.000000 |
| GT10 | V0 | 338.227387 | 337.743000 | 0.000000 |
| GT10 | V1 | 178.624956 | 178.075000 | 0.000000 |
| GT10 | V4 | 412.124574 | 395.223000 | 1.000000 |
| GT10 | V5 | 272.305442 | 250.095000 | 1.000000 |

GPU timings are real production wall/pipeline times including model startup, scoring,
mask I/O and shared-machine load, not isolated CUDA kernel times. V0/V1 are CPU replay;
their wall time is not directly comparable as GPU compute cost. SPNP adds a SAM3
checkpoint load and per-frame text/geometric prompt calls; V1 requires no SAM3 GPU
when verified candidates are already cached.

Sources: V4/V5 real run logs, `V4_V5_PER_FRAME.csv`, `V4_V5_METRICS.csv`,
`FRAME_PAIRED_DIFFERENCES.csv`, archived Phase20C masks and `RUN_TIMES.csv`.
