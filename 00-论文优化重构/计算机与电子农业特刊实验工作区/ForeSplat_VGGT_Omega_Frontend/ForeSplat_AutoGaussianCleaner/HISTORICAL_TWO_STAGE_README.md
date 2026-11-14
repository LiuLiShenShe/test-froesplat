# ForeSplat Two-Stage Gaussian Artifact Suppression

The default public CLI strategy is now `historical_two_stage`. Explicit
`--method C5-*` retains the previous implementation for additional comparisons.
No C5-cleaned checkpoint is used as a new pipeline input.

The current development source is version8. Version7 Kong completed full-camera
and actual browser review, but version7 ChangShouHua1 rolled back on raw thin
coverage. Version8 keeps numeric limits and records separately alpha-only tails
with neither original photo/depth support nor worsened observed RGB. All supported
structure, RGB worsening and geometry checks remain active. If final checks still
fail, one bounded original-row restoration is allowed before rollback. The205-view
ChangShouHua1 replay passed with black pixels1,767,250→661,170; this diagnostic
does not count as a completed two-stage experiment. The four development plants
completed in `v8/`. Kong, ChangShouHua1 and XiangPiShu1 improve with residual
artifacts; WangWenCao1 is a recorded severe visual failure. The shared protocol
is frozen for the remaining15 plants; heldout success remains unproven.

From `ForeSplat_VGGT_Omega_Frontend/`, use the existing uv environment:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-project --python .venv/bin/python3 python \
  ForeSplat_AutoGaussianCleaner/auto_clean_gaussians.py \
  --scene ForeSplat_AutoGaussianCleaner/configs/scenes/KongQueZhuYu.json \
  --strategy historical_two_stage \
  --output ForeSplat_AutoGaussianCleaner/experiments/Historical_TwoStage_Cleaner/v8/KongQueZhuYu
```

One command runs the development comparisons and, after their measured gate
passes, the remaining available GGGS models using frozen shared parameters:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-project --python .venv/bin/python3 python \
  ForeSplat_AutoGaussianCleaner/run_historical_two_stage.py \
  --output ForeSplat_AutoGaussianCleaner/experiments/Historical_TwoStage_Cleaner/v8 \
  --editor-source ForeSplat_AutoGaussianCleaner/experiments/Historical_TwoStage_Cleaner/viewer_compatibility/supersplat_alpha_compatible
```

The scene interface is unchanged: `model`, `gggs_repo`, COLMAP `dataset`
(images, masks, sparse/0), `geometry_cameras`, `geometry_dir`, and `plant`.
No ROI, point ID, plant-specific threshold or manual round count is required.
The historical `baseline_models.R5`, when supplied, is used for the independent
Stage2 comparison; otherwise Stage2 automatically consumes Stage1 output.
Native row lineage is checked before accepting an historical reference.

Batch scheduling keeps one native GPU1 child and overlaps the preceding plant's
CPU trial retirement, comparison export and software-editor captures. At most
one delivery job is outstanding, and all delivery failures propagate. Use
`--serial-postprocessing` for the original wait-between-plants schedule.
`ACTIVE_JOB.json` records `gpu` and `postprocessing` independently;
`batch_status.json`/`BATCH_PROGRESS.md` publish native completion immediately.
`POSTPROCESS_COMPLETE.json` records delivery time separately from cleaning.
Captures still require direct image inspection before visual selection.
Interrupted trial retirement recovers its per-file journals on resume.
Progress updates preserve `FINAL_TWO_STAGE_CLEANER_REPORT.md`.

Stage1 restores the actual native color-Jacobian compositor contributions,
same-normalizer foreground/background harm ratios, conservative Round1 then
Round2 pruning, and repeated current-model reattribution. It keeps the input
foreground contribution as an envelope. The contribution floor, spatial
distances, representative cameras and free cameras derive from input data.
Round3 whole-model shrinkage, fixed P3 mesh silhouettes and marker boxes are
excluded from cleaning. Historical fixed8 cameras remain evaluation-only.

Version7 retains the historical Stage1 acceptance choice. Dark-region
mean RGB error may increase within the configured foreground PSNR loss budget
only when foreground alpha/coverage guards pass; all boundary/thin RGB, per-view, depth,
normal and full-camera structural checks remain active. This is a measured
photometric tradeoff, recorded in the report, not a claim of unchanged pot-rim
appearance. Stage2 retains the strict dark-region RGB guard. A Stage1 model is
not recommended until matched actual SuperSplat comparisons are reviewed.

Boundary coverage now uses the measured support inside the RAP foreground,
without falling back to the complete exterior boundary band when the qualified
subset is less than half the band. This prevents unsupported black tails from
being treated as required leaf coverage. Raw thin/skeleton checks and immutable
RGB-worsening/depth-supported alpha-loss guards still protect true edges.

Stage2 identifies detached components with local 3D connectivity, visibility
and depth/mask evidence. V3 brightness/KNN/scale priors only discover candidates.
Mixed candidates receive bounded anisotropic axis proposals in their quaternion
frame. Version3 bakes the filter only for selected rows before changing an actual
filtered covariance axis, preserving effective center opacity even where the
old raw-opacity compensation was infeasible. Both local surface-normal and
longest-axis proposals are evaluated. XYZ, SH, rotation and all non-candidate
parameters stay exact. Filter/scale/opacity field changes are recorded.

Attribution uses up to1920px on the long side, with one original64-view
foreground envelope and24-view current-model reattribution. Quality evaluation
uses960px. This distinction matters because rasterizer minimum pixel footprint
makes a simple pixel-area correction insufficient for small splats at different
resolutions. Automatic camera selection remains the same deterministic nested
sampling. After a broad prune fails foreground guards, one masked native
reattribution identifies the Gaussian rows contributing to worsening dark RGB,
edge errors or depth-supported coverage loss. Two fixed whole-round protected
proposals are tested; this does not invoke recursive point-group evaluation.

Version4 also applies measured-damage protection to the strongest longest-axis
covariance proposal: harmful candidate rows are restored exactly, while the
remaining selected rows keep their local covariance edit. These corrected
whole-round proposals still receive24/64-view checks and the strict Stage2
dark RGB guard. No hand-selected pixels or Gaussian IDs enter the algorithm.

Automatic free cameras in version4 use4:3 landscape framing at the same vertical
FoV. The earlier portrait framing covered only37.6° horizontally in KongQueZhuYu,
whereas the matching official editor viewport covered77.8°; it missed many
visible detached fragments. Training cameras and historical fixed8 evaluation
are unchanged. Automatic scores from the landscape protocol must not be mixed
with earlier portrait scores; both compared models are remeasured together.

Every whole-round proposal gets 24-camera evaluation, then 64-camera confirmation
only if promising. Selected stage outputs receive all observed-camera
PSNR/SSIM/masked LPIPS, boundary, thin/skeleton, dark foreground, depth and normal
evaluation. A failed full-view check rolls back the stage. Automatic free-view
black pixels, high-alpha area and connected components influence selection;
point count is not an objective. Input RGB/mask/depth/camera caches are reusable;
contribution caches require exact current-model and protocol identities.

Each plant exports `stage1_cleaned_gaussians.ply`, `stage1_supersplat.ply`,
`stage2_cleaned_gaussians.ply`, `stage2_supersplat.ply`, `removed_stage1.ply`,
`removed_stage2.ply`, `cleanup_metrics.json`, and `CLEANING_REPORT.md`.
Stage subdirectories retain actual parameter changes, original IDs, rejected
and accepted proposals, timings, native/viewer parity and comparison images.
`cleaned_gaussians_supersplat.ply` is the final viewer entry. If no accepted
Stage2 change exists, it represents the unchanged Stage2 input, not a claimed
new improvement. Read `model_index.json` for real absolute paths.

External depth projection silhouettes are diagnostic support proxies, not
ground truth. Native image comparison and parity do not substitute for an
interactive SuperSplat visual review. Incomplete runs and unmet development
gates remain explicitly pending. No training, mesh edits or other reconstruction
experiments are performed. Current experimental evidence comes from local
files; the requested Bocha lookup returned HTTP403 quota exhaustion.

Development status: version8 KongQueZhuYu completed both full208-camera checks
without rollback. Stage1 removed62,501 rows (576,396→513,895), with automatic
free8 black pixels995,307→2,487. Historical best scores3,063 under that same
automatic protocol. The independent historical fixed8 count is70,392 versus
68,362, a2.97% increase; the recovered result is close, not identical to history.
Stage2 starts from historical510,248, removes107 and locally edits1,035,
with automatic black pixels3,063→992 and historical fixed8 68,362→50,683.
The measured whole-program time is2,513.29seconds, excluding browser review.
Actual compatible-editor white/black top, orbit, oblique, near and far images
were directly inspected. B3 is selected for Kong; sparse dots and streaks remain.
The selected native/viewer entries are `v8/KongQueZhuYu/selected_cleaned_gaussians.ply` and `v8/KongQueZhuYu/selected_supersplat.ply` beneath the experiment root. Native lineage, source/input integrity, actual screenshot hashes and selection passed the delivery audit. Other plants and heldout success remain pending until their actual runs finish.

The pristine local official SuperSplat3.8.0 build with installed
splat-transform3.11.1 applied sigmoid twice to opacity. Its earlier screenshots
exaggerated nearly transparent splats. Earlier visual rejections based on those
captures are invalid for algorithm selection; their native metrics remain valid.
See `experiments/Historical_TwoStage_Cleaner/viewer_compatibility/VISUAL_REVIEW_ERRATUM.md`.
The original build/captures are preserved. A separate one-line compatibility
patch passes known-alpha0.004/0.02/0.5 actual browser tests. Exported Gaussian
files retain standard opacity logits. Records include the exact build SHA,
model SHA, screenshot SHA, software adapter, camera-position error and direct
inspection observations. Batch eligibility requires this verified provenance.

Farthest-point representative views have unequal trajectory populations; equal
means overstated foreground support for several historically removed background
splats. Original64-view envelopes and current24-view attribution use the same
data-derived trajectory-population quadrature weights. Cross-plant and heldout
success must not be inferred from the CLI or unit tests.
