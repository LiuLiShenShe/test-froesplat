# Two real switch cases

| frame | rasterizer | R0_ID | R1_ID | oracle_ID | R0_F1 | R1_F1 | delta_F1 | R0_IoU | R1_IoU | R0_regret | R1_regret | effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CaoMei2_0146 | opencv | CaoMei2_0146_04 | CaoMei2_0146_03 | CaoMei2_0146_03 | 0.00010177 | 0.98550661 | 0.98540484 | 0.00005089 | 0.97142734 | 0.98540484 | 0.00000000 | IMPROVED |
| WanNianQing1_0227 | opencv | WanNianQing1_0227_00 | WanNianQing1_0227_01 | WanNianQing1_0227_01 | 0.00000000 | 0.98500998 | 0.98500998 | 0.00000000 | 0.97046272 | 0.98500998 | 0.00000000 | IMPROVED |
| CaoMei2_0146 | pillow | CaoMei2_0146_04 | CaoMei2_0146_03 | CaoMei2_0146_03 | 0.00008125 | 0.98554231 | 0.98546106 | 0.00004063 | 0.97149672 | 0.98546106 | 0.00000000 | IMPROVED |
| WanNianQing1_0227 | pillow | WanNianQing1_0227_00 | WanNianQing1_0227_01 | WanNianQing1_0227_01 | 0.00000000 | 0.98501998 | 0.98501998 | 0.00000000 | 0.97048215 | 0.98501998 | 0.00000000 | IMPROVED |

Both switches improve F1/IoU with both fixed rasterizers, and both R1 selections are the F1 oracle. Both frames have GEOMETRY_AMBIGUITY warnings, retained without repair. These are measured contour-overlap improvements, not fabricated human target approvals. All30 unchanged selections were actually evaluated; WanNianQing1_0223 remains severe(F1=0) and selects00 instead of oracle01. Candidate stability does not mean correctness.

Figures: figures/comparison_CaoMei2_0146.png and figures/comparison_WanNianQing1_0227.png.
Sources: R0_R1_TEST32_RESULTS.csv, SECONDARY_RESULTS.csv, ORACLE_ALL_CANDIDATES.csv, real production logs.
