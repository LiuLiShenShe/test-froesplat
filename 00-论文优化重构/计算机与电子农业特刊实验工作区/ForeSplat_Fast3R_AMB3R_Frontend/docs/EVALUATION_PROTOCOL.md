# ForeSplat predicted-geometry evaluation protocol

Only saved real KongQueZhuYu predictions are evaluated. No PSNR, exposure optimization, 3DGS training, mesh extraction, point averaging, scan alignment or scene-scale fitting is performed. The original manifest retains 210 RGB/mask entries, with two unreadable RGB files and 208 usable frames.

`frontends/evaluate.py` runs on CPU. The scripts set `CUDA_VISIBLE_DEVICES=1`; no GPU0 process is launched. Environments remain uv-managed.

To avoid repeated random reads under shared-disk contention, evaluation may recover float32 XYZ/RGB from the Bundle's complete `global_raw.ply`. This is allowed only when the header/schema/byte count match, frame order matches export records, and total finite raw points equal every unpadded pixel. Row-major rectangles are reconstructed from original padding metadata; three original XYZ/RGB pixels are checked for exact equality and `bulk_raw_reconstruction_audit.json` records the proof. Confidence and masks remain their original saved files; padding is invalid. The original NPZs are never overwritten. Unmapped/partial-finite bundles fall back to direct NPZ reads. Whole-cloud visualization may similarly use the complete per-frame concatenation PLY after verifying every frame offset, with identical per-view seeded random sampling. Neither the legacy fused cloud nor capped display PLY is used.

## Outputs and interpretation

- Foreground/edge coverage is the count of finite point predictions with confidence >1 divided by the corresponding input-mask pixels. A one-pixel inner mask boundary is computed for **evaluation only**; exported masks are never eroded. Since confidence heads often use 1+exp, nearly complete coverage is possible without accurate geometry. A separate per-viewq85 confidence control is reported for foreground and edges; the threshold is computed over finite geometry and unpadded image pixels, matching the separate q85 PLY exports. The q85 control is intersected with the baselineconfidence>1 criterion. Ties at q85 and numericalconfidencefloor1 are reported explicitly; q85=1 is not described as retaining onlythetop15%, and it never reaccepts baseline-invalidpoints.
- Point count is reported together with model foreground pixel count and points/pixel. It is not an independent point-density accuracy measure. The existing VGGT-Ω run has higher input resolution than the new 512/518-long-edge frontends.
- Self-reprojection evaluates saved world points through saved W2C and K against their source pixels. This is meaningful as pointmap/camera agreement. It is algebraically zero for depth-unprojected points and is not GT geometry accuracy or SfM track residual.
- Neighboring available views are evaluated bidirectionally. At most 6000 deterministic foreground pixels per direction are projected into the target; target mask, confidence and bounds are checked. Predicted source points lying >5% behind the target-predicted surface are treated as occluded and excluded. Relative depth agreement uses a 5% target-depth tolerance. Report both overlap and agreement, because visibility filtering can make a small accepted set appear consistent. Frame gaps are retained in pair CSV names; no claim is made that missing source frames were inferred.
- Pointmap distance is the median or p90 source–target world distance at accepted projected target pixels, divided by the method's robust foreground scene extent. This uses predicted geometry and cannot establish scanned-GT accuracy. A perspective sample discretization and occlusion boundaries affect results.
- A separate pose-independent symmetric world-cloud nearest-neighbor distance uses at most6000 raw foreground points per view, sampled by a fixed original-frame-number RNG, divided by the same scene extent. Both directions of the nearest-neighbor query are pooled. It remains available when head/camera projection overlap is absent. No real surface correspondences or visibility are known: unequal density, occlusions, duplicate sheets and collapsed planes can create misleadingly small values. It is a geometric proximity proxy and never a sole accuracy or winner criterion. The OFF/ON shared-valid and mask+finite controls apply the same pixel selections in both methods.
- Depth discontinuity is the fraction of valid neighboring foreground model pixels with >5% relative depth change. Thin leaf folds, actual occlusion edges, pixel scale and noise all contribute. It must be inspected alongside RGB and geometric figures.
- A local-plane residual proxy uses 12 nearest neighbors in up to1500 deterministic per-view predicted foreground samples. It reports median sqrt(smallest covariance eigenvalue), normalized by scene extent. This mixes curvature, multi-surface neighborhoods and noise; it is not an isolated noise estimate.
- Camera translation increments are divided by scene extent; rotation increments are degrees. Determinant and orthogonality check that rotations are proper. PnP success is only reported when actual solver status exists.
- Confidence means/quantiles are descriptive. Low-confidence frames have foreground median `(c-1)/c <0.2` (a numerical diagnostic, not a calibrated probability). Sampled nearest spacing uses the deterministic at-most1000-points/view foreground sample; spacing <0.001 scene extent defines a close-point/redundancy proxy. This is not full-cloud density or duplicate-surface GT error. Sampled spacing above median+6MAD is reported as an isolation/outlier proxy; real edges, unequal surface sampling and thin structures also trigger it, so this is not a count of proven erroneous points. The plotting transform `(c-1)/c` uses a common [0,1] color scale but does not make different model confidence heads statistically calibrated.

## Shared leaf and edge ROIs

Source RGB `crop_0000.png` was directly viewed. Original-pixel ROIs (left,top,right,bottom) are:

| ROI | Original pixels |
| --- | --- |
| Visible upper-left red leaf | 540,570,1080,1230 |
| Its left-edge neighborhood | 510,630,720,1110 |

These rectangular evaluation windows are not a new segmentation algorithm or leaf-instance annotations. Every method maps model pixels to original pixels using its saved inverse pixel matrix. Figures show the same source-pixel support windows. They do not claim that different reconstructed sheets have been registered to measured 3D surfaces. The main `edge_q85_coverage` field covers the complete foreground-mask boundary, including the pot and blue reference object. ROI q85 coverage describes retained foreground pixels inside a shared window; it is not a measured leaf-edge accuracy score.

## Plotting

All geometry figures use original RGB colors, a black background, the same point size0.5, equal axis limits[-0.9,0.9], front(x,-y), side(z,-y), top(x,z). Each method contributes at most250000 raw predicted foreground points, divided equally among its available views using uniform sampling seeded by original frame number, **only for visualization**. Complete pointmaps and PLYs are retained separately.

Coordinates are transformed into the earliest source camera that is finite and registered in every available method, then centered by median foreground coordinates. `display_anchors.json` records the shared source filename. If no common camera exists, each method uses its earliest finite camera and the limitation is explicit; empty/unmapped methods have no camera anchor. A display-only extent equals the maximum axis range of the2.5–97.5% foreground quantiles, with each frame contributing at most1000 raw points. This normalization is recorded in each method's `visualizations/normalization.json`. It is not written into exported predictions or used as real metric scale; no measured GT is used.

Whole-sequence and common-source-frame figures are separate and labeled. Local/global diagnostic panels have independent display scales, explicitly labeled, because local pointmap scale may differ; they do not perform local-to-global alignment or averaging.

The existing `/data/fj/F2DMAS/03-GT/KongQueZhuYu` supplies manual binary masks, original photos and LabelMe JSON for source indices0000,0025,0050,0075,0100. The five-view comparison uses these actual masks and photos in its GT column. Model cloud panels use the same original-view camera projections, point size, RGB coloring and black background. This GT is2D only; no ground-truth cloud is fabricated. `shared_config/ground_truth_interfaces.json` records source hashes and the unprovided scan interface.

Manual-GT support recall maps every original GT foreground pixel through the saved pixel transform and tests the corresponding finite, confidence-valid RAP foreground prediction. A nearest pixel footprint is used consistently, with GT one-original-pixel inner-edge support recorded separately. This narrow boundary is sensitive to the existing RAP/GT mask disagreement and model-pixel quantization; a low edge-support score is not isolated evidence of an inaccurate3D leaf edge. This measures saved2D support and registration coverage, not3D accuracy, unseen surface completeness or calibrated depth quality. Masks are unchanged.

The full208-frame Fast3R_FP32 control passes official inference precision string `32`, disabling the outer autocast branch while preserving the official internal FlashAttention BF16 operation. It is an additional row and separate precision figure; the original BF16 experiment remains the primary result. No weights or network operations are replaced.

AMB3R direct point-head geometry, depth-unprojected world geometry and pose have separate prediction gauges. Head/camera reprojection and point-head-to-unprojection norm ratios are internal agreement diagnostics, not measured camera error. Official `depth_metric` is a learned metre prediction without external measurement. A depth-head metric factor transferred to direct pointmaps is conditional and explicitly gauge-unvalidated; it is not established physical-scale geometry.

Timing columns retain their measured scope. Fast3R model time excludes PnP, with a separately recorded model+PnP total excluding export. AMB3R Base timing covers its official forward. SfM pipeline timing includes registration and local serialization when recorded; legacy VGGT has no verified combined total. Shared storage I/O affected wall time during this session, so pipeline wall-time differences are not isolated algorithm-speed comparisons.

## Reusing completed measurements

`--reuse-metrics` reuses actual saved per-frame and pair measurements. Completed OFF/ON same-pixel controls additionally record original raw/RGB/mask file path, size and mtime identities, camera and pixel-transform SHA256, input order and seeds in `amb3r_off_on_pair_cache.json`. A changed identity causes pixel verification and controls to be recomputed. This avoids rereading unchanged full predictions under shared storage contention; it does not replace real inference or claim a new measurement.

The automatic evidence summary is written to `evaluation_analysis.md`. An existing `final_analysis.md` is preserved so a reviewed technical decision remains intact when measurements are refreshed. Final figures include an extra identical constant-gray leaf/edge version. Original RGB, exposure, masks and pointmaps are unchanged.

## Legacy per-frame export completion

Every bundle must contain actual `pointcloud/global_per_frame/{stem}_mask.ply` and `{stem}_mask_confidence.ply` files. The existing VGGT-Ω import originally lacked these416 files; its failed v2 audit is preserved as `bundle_audit_before_legacy_per_frame_export.json`. They were completed from the full saved binary PLY concatenations, using the exact original input/CSV order and verified point counts. Only identical original15-byte XYZ/RGB records are copied. Byte-identical branches may share a hardlink; the original complete PLYs remain unchanged.

`lossless_split_audit.json` records source SHA256, per-frame counts/SHA256, file identity checks and complete foreground XYZ/RGB byte comparisons for the first, middle and final raw frames. Three original prediction archives were read for these spot checks; the208-frame geometry evaluation was not repeated. The importer now enforces the real per-frame file/count postcondition on new exports and can repair missing derived files of an existing import without rerunning a model. Its `run_parameters.method` matches `bundle.json.method`, while original legacy labels and known source parameters are preserved.

## Legacy VGGT-Ω reference

`frontends/vggt_reference.py` reconstructs saved depth backprojection from `per_frame/depth_*.npy`, raw W2C and K, saved confidence, RGB, masks and pixel matrices. It imports the raw model frame and does not apply the old COLMAP display alignment. The old fused PLY and any capped point export are excluded. Saved point-head predictions are absent from that legacy run, so the reference is explicitly depth-based and is not called a raw point-head result.

Sources: the actual local bundle files and legacy VGGT-Ω artifact. Network discovery was attempted by the parent via the user-specified Bocha API, which returned403 quota exhaustion; these diagnostics are computed from files, not from training-data assertions.
