# Foreground/background descriptive reconstruction analysis

| sample | group | R0_proxy_foreground_psnr | R0_proxy_background_rgb_energy | R0_proxy_missing_alpha_fraction | R0_proxy_silhouette_iou |
| --- | --- | --- | --- | --- | --- |
| WanNianQing1 | A_raw_RGB_control_v1 | 27.240882 | 0.439385 | 0.000022 | 0.305089 |
| WanNianQing1 | B_R0 | 23.082850 | 0.001203 | 0.004621 | 0.498823 |
| WanNianQing1 | C_R1 | 23.410994 | 0.001233 | 0.004452 | 0.525042 |
| CaoMei2 | A_raw_RGB_control_v1 | 22.856390 | 0.373447 | 0.000000 | 0.142325 |
| CaoMei2 | B_R0 | 19.999550 | 0.000188 | 0.000007 | 0.143712 |
| CaoMei2 | C_R1 | 18.898690 | 0.000181 | 0.000000 | 0.142359 |
| HongZhang | A_raw_RGB_control_v1 | 24.991257 | 0.480014 | 0.000217 | 0.368076 |
| HongZhang | B_R0 | 20.322178 | 0.001626 | 0.002829 | 0.419604 |
| HongZhang | C_R1 | 20.322178 | 0.001626 | 0.002829 | 0.419604 |

The shared frozen B_R0 mask defines a predicted ROI for **all** groups. This is
not independent segmentation truth or 3D geometry accuracy. Outside-ROI energy
quantifies appearance contamination relative to that proxy; missing alpha is an
appearance-coverage proxy, not a validated physical hole. Components outside
that proxy may include boundary misalignment, not confirmed floating geometry.

A fixed central spatial crop also has PSNR/SSIM/LPIPS, independently of method.
It is not asserted to exclude all background. No independent silhouette IoU or
Boundary-F1 is reported because none of the held-out raw identities has GT.

Visual observations belong in VISUAL_RECONSTRUCTION_COMPARISON.md after direct
image inspection; no automatic proxy is promoted to geometric truth.

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
