# Mask/RGB and loader alignment

| sample | registered | candidate_views | new_instances | R0_R1_switches | train_mask_differences | ready | training_runs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WanNianQing1 | 213 | 213 | 339 | 3 | 3 | True | 3 |
| CaoMei2 | 106 | 106 | 111 | 1 | 1 | True | 3 |
| HongZhang | 213 | 213 | 472 | 0 | 0 | True | 2 |

Native RGB roundtrip tolerance is maximum one intensity unit, mean <=0.1; measured
values are retained rather than called exact. Sparse observation conversion max
must be <=1e-6px. Masks are binary and map invalid pixels to background. All target
dimensions must equal cameras. Alignment figures show exact-domain masks and RGB;
native model viewing is required and machine QA is not a human signature.

Real production loading additionally checks all registered views, frozen split,
camera transforms, common RGB identity, pre-composited RGB/alpha pixels, declared
resolution=2 behavior and initial inherited target sparse point identity. All groups use
one camera symlink bundle. Each group receives one real forward render without
`training_setup`, optimizer, backward or optimization step. Reports in
`reports/*_PREFLIGHT.json` include actual GPU/CUDA/extension paths and memory.
`reports/ACTUAL_LOADER_CAMERA_PROVENANCE.csv` also checks all 1596 group-view
camera transforms and FoV exactly against frozen intrinsics/poses. The production
`CameraInfo.uid` is a camera_id, NOT image_id; the unique registered name performs
the join, and the true registered image_id is retained independently in manifests.

Current `utils/camera_utils.py:44` loads alpha; `scene/cameras.py:47` only stores it.
Default RGB loss uses all RGB pixels and plant-aware alpha losses are disabled;
   default sparse initialization uses inherited model points without extra filtering. Consequently our treatment
is black RGB pre-composition, not an assumed alpha loss or mask-guided initialization.
The base unmasked RGB is identical across groups; treatment composites intentionally
differ. PIL RGB/alpha channel scaling at resolution=2 is explicitly frozen and
verified, never mistaken for a camera-domain mask resize.
