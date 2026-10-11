# GT10 Results

GT10 is the Phase 17 diagnostic development set, not an independent generalization test. V0/V1 were formally run on all 10 frames. V2/V3 were not run as complete groups because COLMAP observations are absent for `DouBanLv2_0037`, `DouBanLv2_0039`, and `DouBanLv2_0041`; the missing geometry is recorded in `configs/COLMAP_OBSERVATION_AUDIT.csv`.

| Group | Status | Frames | F1 | IoU_fg | mIoU | HD95 px | Boundary-F1 | Leakage | Empty | Severe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V0 | COMPLETE | 10 | 0.6767 | 0.6551 | 0.7867 | 405.61 | 0.2276 | 0.012606 | 0 | 3 |
| V1 | COMPLETE | 10 | 0.9728 | 0.9474 | 0.9634 | 56.93 | 0.3489 | 0.001966 | 0 | 0 |
| V2 | BLOCKED | 0 | — | — | — | — | — | — |  |  |
| V3 | BLOCKED | 0 | — | — | — | — | — | — |  |  |

## Core hard cases

| Frame | Group | Selected candidate | Selected F1 | Oracle regret | Historical replay match |
| --- | --- | --- | --- | --- | --- |
| 0039 | V0 | DouBanLv2_0039_00 | 0.0000 | 0.9867 | 1 |
| 0039 | V1 | DouBanLv2_0039_01 | 0.9867 | 0.0000 | 1 |
| 0040 | V0 | DouBanLv2_0040_01 | 0.0000 | 0.9882 | 1 |
| 0040 | V1 | DouBanLv2_0040_00 | 0.9882 | 0.0000 | 1 |
| 0041 | V0 | DouBanLv2_0041_00 | 0.0000 | 0.9865 | 1 |
| 0041 | V1 | DouBanLv2_0041_03 | 0.9865 | 0.0000 | 1 |

The historical values in Phase 17C are used only as replay references; all F1 values above come from the new Phase 20C masks.
