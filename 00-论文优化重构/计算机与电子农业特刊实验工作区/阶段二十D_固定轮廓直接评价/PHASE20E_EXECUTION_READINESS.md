# Phase20E executable preparation — CONDITIONAL, training not started

The foreground configuration is engineering-locked. TEST32 is not a full-sequence training input. Use3plants×3matchedtreatments×1fixedseed=9trainings as a feasibility pilot, not significance or variance evidence.

| sample | raw_rgb_count | registered_views | train_views | validation_views | model_undistorted_dimensions | historical_rgba_exists | historical_rgba_matching_grid | undistorted_rgb_matching_grid | undistorted_rgb_unreadable | colmap_input_unreadable | verified_frozen_candidate_views | missing_full_view_candidate_views | test32_registered_views | test32_unregistered_views |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CaoMei2 | 250 | 106 | 92 | 14 | ["2109x3777"] | 106 | 2 | 106 | 0 | 2 | 1 | 105 | CaoMei2_0146 | CaoMei2_0007;CaoMei2_0008;CaoMei2_0009 |
| WanNianQing1 | 252 | 213 | 186 | 27 | ["2121x3801"] | 193 | 193 | 213 | 0 | 4 | 2 | 211 | WanNianQing1_0223;WanNianQing1_0225 | WanNianQing1_0224;WanNianQing1_0227 |
| HongZhang | 250 | 213 | 186 | 27 | ["2122x3784"] | 185 | 181 | 209 | 4 | 0 | 3 | 210 | HongZhang_0134;HongZhang_0202;HongZhang_0229 | HongZhang_0200 |

## Actual findings, not assumptions

- FullRGB:01-FFmepg/{sample}; registered/camera models:04-COLMAP/{sample}/sparse/0 and distorted/sparse/0. View and camera manifests include exactIDs, names, hashes, raw inputs and real dimensions.
- Historical05-2DGS-new cfg_args uses images_rgba, resolution=2, eval=False, sh_degree=3, white_background=False and old Windows paths. Cannot reuse as an already matched held-out control.
- CaoMei2 registration names are crop_0095.png..crop_0200.png on previously foreground-processedRGB, not raw fullRGB. This is a segmentation-conditioned pose selection and a limitation of reusing that model. Its0007/0008/0009 test views are not in this model;0146 is. Other two sequence test registration status is explicitly enumerated in the table, never assumed.
- All examined full-view metadata archives are indexed inPHASE20E_CANDIDATE_ARCHIVE_SEARCH.csv. The frozenU00 set supplies only the listed registered views, not full sequences. Existing03-SAM foreground masks are not originalP6 per-instance SAM3 candidates.
- Actual unreadable/mismatchedRGB and incompleteRGBA are recorded, not silently skipped. Historical names/extensions can differ; joins are explicit, not positional. No resizing workaround.

## Minimum future execution checklist

1. Preserve all existingmodels. Resolve unreadable images from originalRGB and provenance; forCaoMei2 either reuse and document historical mask-conditioned poses identically forA/B/C or build a separately versioned rawRGB-onlyCOLMAP model before splitting. Do not pretend these are already raw-matched poses.
2. Freeze exact registered viewIDs and sample-specificCameraOPENCV/PINHOLE models. Current proposed split is saved inPHASE20E_VIEW_MANIFEST.csv: sort exactregistered stems, every eighth(index%8==0) held-out; all other views train. Seed=0, iterations=30000, resolution=2, sh_degree=3, blackbackground. If a new model is necessary, create a new full-viewmanifest/split before any treatment, not a per-treatment model.
3. A=original historicalinput handling, B=frozenR0, C=frozenR1. Use identical viewset/poses/intrinsics/points3D/seed/iterations/optimizer/hardware. IfA cannot be provenance-completely recovered, report it and rebuild a versionedA control; do not claim the partial existingRGBA collection represents a matchedA treatment.
4. Future completeP6 generation: single SAM3 archive per sequence on the originalRGB grid, per_instanceP6 with savedfullmetadata, IDs/boxes/order/confidence/model/checkpoint/RGB/maskSHA256. Then replay the SAME archive under frozenbase andr1. Any new archive isNEW_CANDIDATE_SET, never a restored71candidateTEST32set. Do not regenerate or tune onTEST32 during this phase.
5. Map rawmasks into the exactundistorted domain, not2160×3840→2119×3779resize. Actualcamera dimensions are sample-specific in the table. For each pixelcenter in theundistortedPINHOLE grid, unproject using thatK, then project using thepaired originalOPENCV distortion model; remap thebinarymask with nearest-neighbor andconstant0. ResolveCOLMAP pixelcenter convention using its camera implementation; do not assume anOpenCV origin convention without anRGB roundtrip. Save mapSHA256/validROI/source/output and test actualRGBroundtrip and sparse-trackreprojection. No camera-number-only joins.
6. The same mapping applies to all treatmentRGB/alpha. Use mapped binaryalpha/RGBA without changing 2DGS/M2M3-GO. Existing 2DGS camera loader handles undistortedPINHOLE, not arbitraryOPENCV. Validate exactresolution handling, name/extensioncoverage andmaskalignment before launching.
7. Report held-outPSNR/SSIM/LPIPS plus fixedindependentROI/silhouette/leakage/outlier/emptyrender stats, time/peakmemory and all failures, perplant. No reference3Dscan means no truegeometryaccuracy claim. Reuse a historicalcheckpoint only if all inputs/config/camera/split/seed provenance truly match; otherwise rerunA. Phase20E still requires separate userauthorization.

Current block is technicalfull-viewdata/coordinate preparation, NOT another manual-label/signature loop. No 2DGS command is executed. Sources: actual binarymodels/images/cfg_args, hashed manifests and unchanged2DGS dataset_readers.py. COLMAP documentation retrieved viaContext7: https://github.com/colmap/colmap/blob/main/doc/faq.rst and doc/cameras.rst.
