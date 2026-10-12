# Phase20E minimum 3D protocol — plan only

READY FOR PHASE20E: **NO**. This file prepares the requested plan, not an execution.
Require completed independent Phase20D results and an evidence-based lock/provisional
decision plus explicit Phase20E authorization first. No training is launched here.

## Actual plant samples proposed

- **CaoMei2**: include the true target sequence and its difficult multi-instance view
  `CaoMei2_0146`; planned case remains even if its R1 switch regresses.
- **WanNianQing1**: include `WanNianQing1_0227`, the other historical switch; current
  manual D columns say none, so do not mislabel this sequence as a D1 case.
- **HongZhang**: ordinary/mixed-difficulty control, including `HongZhang_0134` and
  recorded none views; human identity and D3/D11 evidence must be approved first.

These are real frozen TEST sequence IDs, not synthetic replacement examples. Their
original/distorted sparse directories were found in `04-COLMAP/<sample>/sparse/0` and
`distorted/sparse/0`; directory existence is not a registration/quality certificate.
Use the sequences' full registered view sets for training, not only four TEST frames.
If one sequence is unregistered/ineligible, report and resolve it; do not silently
replace it based on favorable results. DouBanLv2_0039–0041 is a separate DEV mechanism
reference, excluded from independent TEST32 statistics and not a required fourth run.

## Frozen mask source and original 2DGS control

Three foreground treatments per plant: original 2DGS input handling, frozen R0, frozen
V1/R1. Use exact Phase20D final-cleanup mask outputs, index raw selected candidate IDs
separately, and record RGB/candidate/metadata/output/config/commit SHA256s. Full-sequence
masks are not delivered by the32-frame test; generate those later under exactly the
locked parameters and known candidate provenance, not with a GT-based runtime selector.
Missing full-view candidate archives must be declared; any later new inference set is
versioned separately, never presented as restoration of the TEST32 71 candidates.

Original control is `2d-gaussian-splatting-main`, unmodified. Baseline inspected at
22073ca5: train.py SHA256 `e9ecc56a33c799818c814e6d135b9b140044d92de69c2d276bcc940933c8ce1a`,
render.py SHA256 `054c280506f1546b92e8df96b26b76bea1b5c6db2ac807a56a89ad3bcf3a7c11`.
At execution, record the full tree/submodule revision and original training configs;
two file hashes alone do not establish a complete immutable 2DGS release. Do not change
the 2DGS or M2M3-GO algorithms; this minimal plan does not add a M2M3-GO experiment.

## Minimum reruns and quality

Initial budget: **3 plants × 3 input treatments = 9 matched trainings**, one fixed
seed each as a feasibility pilot, not a variance/significance claim. Reuse the exact
existing model only if its input/seed/config/camera/checkpoint provenance matches; else
rerun that control. Same registered views, held-out views, cameras/poses, iterations,
initial point cloud, crop/resolution, optimizer and hardware. Additional seeds only
in a separately documented robustness extension after this pilot, not cherry-picked.

Report held-out PSNR/SSIM/LPIPS on both full image and a fixed independent target ROI,
foreground silhouette IoU/Boundary-F1, leakage and holes/floaters, training/render time,
GPU peak memory, empty/failed render counts. No available reference mesh means no claim
of true 3D geometry accuracy; Chamfer/F-score requires a real independent scan. Retain
matched views and bad cases; summarize by plant, not correlated image pseudo-replicates.

## Prevent mask/COLMAP domain mismatch

1. Join source RGB, registered image name/ID, camera ID/model/intrinsics and mask by
   exact identity, not directory order. Hash cameras.bin/images.bin/points3D.bin.
2. Determine whether training consumes raw distorted or undistorted pinhole RGB.
   Raw TEST32 is2160×3840; historical geometry records include2119×3779 undistorted
   grids. Inspect actual sizes per plant; do not assume every camera shares this size.
3. Never resize/crop/pad a raw mask to fit an undistorted grid. Apply the **same explicit
   distortion/undistortion mapping** to RGB and mask (nearest-neighbor for labels),
   then use matching intrinsics. Record mapping, source/result hashes and validity ROI.
4. Require exact name/ID/dimension agreement and actual registration for every input
   view. Mask/RGB edge overlays and projected known sparse tracks must agree; abort
   mismatch rather than falling back to an unverified camera or identity.
5. Freeze train/validation split, camera bundle and transformed-mask manifest before
   launching models; no per-treatment camera optimization or GT-derived mask changes.

Source: frozen local manifests, actual root COLMAP directory availability, unchanged
training source hashes and Phase20C-FIX coordinate warnings. Plan only; no 3D result.
