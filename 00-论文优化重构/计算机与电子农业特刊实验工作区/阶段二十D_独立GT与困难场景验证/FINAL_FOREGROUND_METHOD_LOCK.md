# Foreground method lock decision

FOREGROUND METHOD: PROVISIONAL
R1 INDEPENDENT VALIDATION: NOT VERIFIED

V1 remains the frozen DEV-supported recommendation, NOT a TEST32-validated final lock.
Recipe: P6 / default_prompt_id=P6 / per_instance / single / no prompt ensemble /
ranking_policy=r1; SPNP=OFF, Residual=OFF, GCFC=OFF, A6=OFF, A7=OFF.
All weights, penalties, thresholds and first-occurrence tie rules stay byte-frozen.

## Immutable code and parameter sources

- Baseline commit: `898cec2af3aa121511716c3fb288b00ede4468b3`.
- ranking_policy.py SHA256: `c80442a2161c0560bc7bf53a7714b6d9da5a996e2a824716a0b0dc2092071905`.
- production SHA256: `3056f3fd5a95926b295671d586bc77dbe69c0014a77439daf336ab2ab03e2be8`.
- metadata loader SHA256: `96fa57b293fe057a5b2f9b696d1e77c3387ddf41ba16a9ad4d52000a629f6d49`.
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

来源：本目录真实审计 CSV/JSON/日志；Phase16 冻结 manifest、Phase17A U00_Control、Phase17C 历史行为表及 Phase20C-FIX 决策/生产源。
