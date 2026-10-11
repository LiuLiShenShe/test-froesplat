# Easy21 Results

Easy21 是开发/历史回归集，不是独立泛化测试集。统计主列为 sample-equal mean；同时保留 frame-level、median、population SD 于 `metrics/summary.csv`。

| Group | Status | Frames | F1 | IoU_fg | mIoU | HD95 px | Boundary-F1 | Leakage | Empty | Severe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V0 | COMPLETE | 21 | 0.9830 | 0.9666 | 0.9791 | 22.54 | 0.4085 | 0.000942 | 0 | 0 |
| V1 | COMPLETE | 21 | 0.9830 | 0.9666 | 0.9791 | 22.54 | 0.4085 | 0.000942 | 0 | 0 |
| V2 | COMPLETE | 21 | 0.9850 | 0.9705 | 0.9816 | 24.47 | 0.4873 | 0.001768 | 0 | 0 |
| V3 | COMPLETE | 21 | 0.9850 | 0.9705 | 0.9816 | 24.47 | 0.4873 | 0.001768 | 0 | 0 |

## Paired effects

| Effect | N | ΔF1 | ΔIoU_fg | ΔmIoU | ΔHD95 | ΔBoundary-F1 | ΔLeakage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ranking_V1_minus_V0 | 21 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.0000 | 0.000000 |
| refinement_R0_V2_minus_V0 | 21 | 0.0022 | 0.0043 | 0.0027 | -4.94 | 0.0735 | 0.000843 |
| refinement_R1_V3_minus_V1 | 21 | 0.0022 | 0.0043 | 0.0027 | -4.94 | 0.0735 | 0.000843 |
| interaction_(V3-V2)-(V1-V0) | 21 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.0000 | 0.000000 |

V0/V1 share the same final masks on this cache. V2/V3 are the real full-chain runs; the actual module-level change counts and acceptance rates are in `MODULE_EXECUTION_TRUTH.md`.
