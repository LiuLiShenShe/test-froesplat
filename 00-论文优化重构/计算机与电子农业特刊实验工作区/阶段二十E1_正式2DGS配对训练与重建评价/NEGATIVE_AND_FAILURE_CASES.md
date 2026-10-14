# Negative, mixed and failed cases

Final-attempt process/evaluation failures: **0**.
Preserved preliminary compatibility-failed attempts: **2**,
actual extra wall cost **1206.402529 seconds**.
They remain in logs/failed_attempts/ and reports/FAILED_ATTEMPT_COSTS.csv, not
hidden by subsequent successful retries.
Mixed or regressed paired view comparisons: **94** of 136.
These are all retained, not discarded, in metrics/NEGATIVE_OR_MIXED_VIEW_CASES.csv.

| sample | comparison | PSNR | SSIM | LPIPS | Gaussians | primary_direction |
| --- | --- | --- | --- | --- | --- | --- |
| WanNianQing1 | A_raw_RGB_control_v1->B_R0 | -20.908479 | -0.654928 | 0.257149 | -316639 | REGRESSED |
| WanNianQing1 | B_R0->C_R1 | 0.001684 | 0.000790 | -0.000728 | 4978 | IMPROVED |
| CaoMei2 | A_raw_RGB_control_v1->B_R0 | -20.168983 | -0.785136 | 0.328381 | -191287 | REGRESSED |
| CaoMei2 | B_R0->C_R1 | -0.025290 | -0.004376 | 0.002335 | 13763 | REGRESSED |
| HongZhang | A_raw_RGB_control_v1->B_R0 | -19.178793 | -0.598614 | 0.233966 | -375982 | REGRESSED |
| HongZhang | B_R0->C_R1 | 0.000000 | 0.000000 | 0.000000 | 0 | NO_CHANGE |

PSNR improvement with SSIM decrease or LPIPS increase is explicitly classified
as MIXED, not an unconditional improvement. Original full raw references also
reward background reconstruction: intentional black-background suppression can
lower full-image metrics while lowering descriptive contamination proxies.
Neither outcome is hidden by replacing the reference image.

OOM/NaN/interrupted final attempts, if any, remain in reports/FAILURES.csv and original
stdout/stderr/process records. Partial runs never enter full-training averages.

Three scenes and one fixed seed form a feasibility pilot, not broad generalization.
Held-out views are correlated and are not independent experimental repetitions.
No independent 3D GT exists: no truth Chamfer/F-score, physical dimensions or mesh.
All 68 frozen validation views lack matching existing manual fixed contours.
Frozen R0 proxy ROIs are predicted labels, not GT; fixed central rectangles are
method-independent descriptive spatial ROIs, not guaranteed plant-only regions.
CaoMei2 uses historically foreground-conditioned COLMAP cameras, shared by A/B/C.
CaoMei2 TEST32 raw0146 and WanNianQing1 raw0227 are not registered improvement cases.
Actual treatment switches: WanNianQing1 raw0037/raw0038/raw0165; CaoMei2 raw0139.
HongZhang B/C uses the same checkpoint; it cannot establish an R1-versus-R0 benefit.
Input treatment is explicit black-RGB precomposition only; no alpha-weighted RGB
loss or mask-guided sparse initialization is enabled. A is not historical ForeSplat.
CUDA rasterization/densification may not be bitwise deterministic across repeated
runs; one-seed differences cannot estimate this variability.
