# R0/R1 score revalidation

All 55 candidates restore independently matched original archived confidences and boxes.
Maximum historical total-score difference: 6.81443809158e-08;
explicit archive tolerance: 1e-07. All scores include real q_sam.
`SCORE_EQUIVALENT` is numerical archived-precision equivalence, not unrounded tensor bytes.
`ROUNDING_SELECTION_CERTIFICATE.csv` separately certifies 62/62 historical R0/R1
selections for every confidence consistent with six-decimal serialization; no confidence
value is inferred. No Phase20C selection changes after restoration.

Easy21: R1 leaves all 21 correct selections unchanged. GT10: R0 still has three
severe wrong-target selections; R1 selects original correct candidates 0039_01,
0040_00, 0041_03, with zero oracle regret. This is DEV/regression, not independent TEST.

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

Selected candidate ID, primary, risk penalty, q_sam, F1/IoU, oracle regret and change
reason for historical R0, Phase17C offline R1, Phase20C zero-score R0/R1 and restored
R0/R1 are all in `R0_R1_SELECTION_REPLAY.csv`. Per-candidate counterparts and hashes
are in `CANDIDATE_SCORE_REPLAY.csv`. New V0/V1 production masks are validated against
the real raw masks, not just against labels in a CSV.

Sources: this delivery's replay CSVs and small archived production logs; frozen
Phase12/P17A original metadata and Phase17C benchmark. ranking_policy.py is unchanged.
