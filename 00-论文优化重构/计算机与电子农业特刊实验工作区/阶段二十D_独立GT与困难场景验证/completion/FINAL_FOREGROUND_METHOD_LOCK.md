# Final foreground method lock — NOT VERIFIED

```text
FOREGROUND METHOD: PROVISIONAL
INDEPENDENT LOCK: BLOCKED
```

V1 remains the development-supported recommendation from Phase20C-FIX, not an
independently confirmed method. The observed GT gate is blocked; no accuracy or
selection-identity result can be inferred from the existence of annotations.

## Required decision questions

| question | evidence-based answer |
| --- | --- |
| Is R1 better than R0 on TEST32? | NOT VERIFIED; no approved-GT evaluation |
| Which TEST32 frames regress? | NOT VERIFIED; no favorable-only selection |
| What happened to both candidate switches? | CaoMei2_0146 04→03, WanNianQing1_0227 00→01 are historical IDs only; improvement/regression/Oracle/identity NOT VERIFIED |
| Any new severe errors? | NOT VERIFIED; no new predictions or human identity audit |
| Are the three DEV rescues still supported? | Phase20C-FIX archived DEV evidence still supports DouBanLv2_0039–0041 rescue; sources unchanged, not rerun or independently generalized here |
| Is V1 ready as the final ForeSplat module? | PROVISIONAL, independent lock blocked |
| What limits remain? | Invalid export schema/components and missing review provenance/evidence; 8 correlated sequences; six-decimal archived confidences; prior no-GT image behavior checks |
| What masks/config/version enter 3DGS? | No masks released for 3D yet; future hash-bound V1 final-cleanup output from strict P6 cache, with selected raw masks separately indexed |

## Frozen recipe and sources

P6 + `per_instance` + `single` + `ranking_policy=r1`, ensemble OFF; SPNP, Residual,
GCFC, A6/A7 OFF. R0 differs only by `ranking_policy=base`. Full parameters are in
the parent `configs/FROZEN_V1_PARAMETERS.json` and `FROZEN_R0_PARAMETERS.json`,
including `fsam3_basic` cleanup, 15-pixel closing, hole fill and largest-component
selection. Parent archived paths are DEV paths, not executed TEST32 commands.

Production baseline: `898cec2a`; completion input baseline: `22073ca5`.
Frozen R1 source SHA256: `c80442a2161c0560bc7bf53a7714b6d9da5a996e2a824716a0b0dc2092071905`.
Frozen loader SHA256: `96fa57b293fe057a5b2f9b696d1e77c3387ddf41ba16a9ad4d52000a629f6d49`.
All source/candidate identities must remain identical. Historical confidences are
six-decimal archived observations, not reconstructed pre-serialization tensors.

Future 3D training uses the frozen **final** V1 output, not a GT-filled or hand-corrected
mask; publish an explicit raw-selection vs final-cleanup stage column, mask SHA256,
full parameter snapshot, code commit and raw/undistorted camera-domain mapping first.
No method, prompt, penalty, weight, dataset frame or source candidate was changed here.

## Release condition

Archive actual records and final compatible exports, pass the unchanged gate, smoke
both cases under both policies, run all32, calculate full metrics and review identities.
Then write a new result-bound lock decision. If R1 is neutral or worse, report it; do
not retune on TEST32 and still call it the same independent confirmation. Stop Phase20D
here while blocked; no Phase20E or manuscript edits.

Source: unchanged Phase20C-FIX decision/results and actual completion gate/hash logs.
