# Foreground method decision — ENGINEERING_LOCKED

Frozen engineering recipe: **V1=P6+per_instance+single+R1**; original score weights area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5; archived confidence precision6decimals; SPNP/Residual/GCFC/A6/A7/semantic gate/temporal alignment/repromptOFF. Full unchanged parameters: logs/full_r1_runtime_parameters.json. Frozen ranking codeSHA256=c80442a2161c0560bc7bf53a7714b6d9da5a996e2a824716a0b0dc2092071905.

## Seven requested answers

1. GT10's0039/0040/0041 repairs remain supported: read-only recomputation of original candidate identities and masks reproduces R0F1=0 and R1F1>0.98. These are development regressions, not independent TEST evidence.
2. Both TEST32 switches improve under OpenCV and Pillow; both R1IDs reach the raw candidateF1 oracle.
3. No new empty or severeF1<0.5 frame; severe count drops3→1. One old severe error remains, listed below.
4. Full32 sample-equalF1 0.888872591→0.950448054; delta=+0.061575463. F1 andIoU improve on2, tie on30, regress on0; not32 improvements. Means ofBoundary-F1/HD95/leakage also improve, but boundary agreement remains modest and long-distance outliers remain. No claim of universal correct-target selection.
5. The13-frame ambiguity subset retains a positive pairedF1/IoU effect with both implementations; this supports engineering ranking stability only, not an ambiguity-freeGT conclusion.
6. Both rasterizers give the sameF1/IoU method ordering and switch direction; STABLE is limited to that sensitivity check.
7. ENGINEERING_LOCKED is adequate to freeze matched3D engineering treatments without tuning. It does not require or assert original dual-signatureGatePASS, and does not promise segmentation accuracy on every view. A/B/C matching and coordinate/data readiness remain conditional.

| frame_id | method | candidate_id | f1 | iou_fg | dataset |
| --- | --- | --- | --- | --- | --- |
| DouBanLv2_0039 | R0 | DouBanLv2_0039_00 | 0.0 | 0.0 | GT10_DEVELOPMENT_ONLY_NOT_TEST32 |
| DouBanLv2_0039 | R1 | DouBanLv2_0039_01 | 0.9866595443157061 | 0.9736703383162864 | GT10_DEVELOPMENT_ONLY_NOT_TEST32 |
| DouBanLv2_0040 | R0 | DouBanLv2_0040_01 | 0.0 | 0.0 | GT10_DEVELOPMENT_ONLY_NOT_TEST32 |
| DouBanLv2_0040 | R1 | DouBanLv2_0040_00 | 0.988159827640328 | 0.9765967537500315 | GT10_DEVELOPMENT_ONLY_NOT_TEST32 |
| DouBanLv2_0041 | R0 | DouBanLv2_0041_00 | 0.0 | 0.0 | GT10_DEVELOPMENT_ONLY_NOT_TEST32 |
| DouBanLv2_0041 | R1 | DouBanLv2_0041_03 | 0.9864847961891434 | 0.9733300422923329 | GT10_DEVELOPMENT_ONLY_NOT_TEST32 |

## Retained serious error

| frame | R0_ID | R1_ID | oracle_ID | R1_F1 | oracle_F1 | FP | FN | R1_regret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WanNianQing1_0223 | WanNianQing1_0223_00 | WanNianQing1_0223_00 | WanNianQing1_0223_01 | 0.00000000 | 0.98718147 | 142322 | 6703835 | 0.98718147 |

No new label, approval, R1 formula, weight, mask cleanup, TEST-trained threshold or runtime oracle is introduced. Oracle is evaluation-only. Continue this error in Phase20E analysis if the corresponding view is registered; do not silently replace it with oracle01.

Sources: Phase20C-FIX FINAL_FOREGROUND_METHOD_DECISION.md and GT10_DEVELOPMENT_REGRESSION.csv; new complete32results, both rasterized summaries and immutable production logs.
