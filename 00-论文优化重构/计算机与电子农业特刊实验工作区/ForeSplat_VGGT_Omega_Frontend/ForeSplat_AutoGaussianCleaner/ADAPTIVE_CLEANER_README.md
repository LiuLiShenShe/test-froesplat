# GGGS Cleaner comparison and adaptive development

Run in the existing frontend uv environment; use GPU1. No conda or new frontend inference.
The native PLY must contain three log-scales, quaternion, opacity logit, all SH and filter_3D.

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python auto_clean_gaussians.py \
  --scene configs/scenes/KongQueZhuYu.json --method C5 \
  --output experiments/GGGS_Cleaner_Comparison/C5_Fused_GGGS/new_run
```

The root's default configuration is still the earlier development configuration;
it has not been promoted to the cross-plant selected method. The matched current
experiment uses the immutable `development_variants/v12/ForeSplat_AutoGaussianCleaner/`
source and its `configs/adaptive_root_coverage.yaml`, with production variant C5-C.
Its six-plant development validation and thirteen independent tests are pending.
C5-A/B/C/D are available explicitly.
An output directory is never reused with different configuration or input; `--resume`
only accepts an identical complete run. Source snapshots accompany all new C5 runs.

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python auto_clean_batch.py \
  --manifest experiments/GGGS_Cleaner_Comparison/cross_plant_validation/plant_manifest.json \
  --method C5 --group development --output-root experiments/GGGS_Cleaner_Comparison/cross_plant_validation/development_run
```

Before **independent tests**, the experiment manifest must refer to the actual
development decision, and source/config/manifest hashes are frozen. Test results
cannot be used for tuning and then relabeled as independent. Missing GGGS models
remain explicit missing rows; the established model-preparation batch supplies
30K GGGS models and is scheduled separately from cleaning.

A scene JSON provides `plant`, native `model`, corrected GGGS `gggs_repo`,
`dataset` (images/masks/COLMAP sparse cameras), `geometry_cameras` and
`geometry_dir` (already inferred depth/confidence). No Gaussian IDs, ROI,
per-plant brightness threshold, crop box or cleaning-round parameter is needed.
There is no 2DGS/3DGS/SuGaR training path in this cleaner.

Appearance priors are DC brightness, effective three-axis scale, effective Alpha,
KNN and normalized spacing. They never independently certify deletion in C5-B/C/D.
Native contribution, depth, contour and thin structure protect real dark leaves.
Groups are automatic spatial components separated by foreground risk; rejected
groups split spatially or by the rejected camera's native visibility. Background
benefit is measured by rerendering, and every round reattributes exposed layers.
Mixed groups independently optimize constrained covariance with compensated
effective Alpha. No global transparency adjustment is made.

The current frozen experiment configuration is the v12 clone's
`configs/adaptive_root_coverage.yaml`. It retains full native validation of corrective depth changes against the existing
confidence-filtered VGGT depth. A depth change is excused only with enough
reference coverage, predominantly smaller reference error, bounded harmful
changes, and no aggregate foreground-photo regression. Alpha, thin structure,
normals, holes and other photo rules remain active. VGGT depth is an auxiliary
reference, not botanical ground truth. A dark-MSE-only exception has a fixed
flagged-view RMSE increment cap of 1/255 and a cumulative aggregate budget.
Neither exception permits accepting incomplete preflight evaluations.

Pending spatial groups and split children are jointly ranked by predicted native
background contribution. Every round records untried groups and their predicted
benefit. A difficult region cannot reserve the entire trial budget through
recursive split-first scheduling. Initial radii=0 footprints can be reused only
for unchanged members; groups touching parameter-modified members rerender all views.
After the initial trial budget, v12 tests each remaining root group once, with no
extra descendant expansion. Its stopping decision also includes accepted local
benefits from the current round. This scheduling change retains the quality and
candidate thresholds of v11. Every accepted edit requires complete native-view
evaluation, including proven unchanged views.

`prepare_development_decision.py` records a decision only after the matched
four main ablations and all requested development outputs have actually finished,
their native artifacts and paired measurements verify, and both contact sheets
have been directly reviewed. It rejects observed conspicuous leaf loss and a
single-plant-only modification. It does not modify the manifest, promote sources
or execute independent tests. Promotion must preserve the frozen core source,
batch entrypoint and configuration bytes.
`verify_development_decision.py` supplies a read-only pre/post-test certificate:
the recorded native models, cleanup reports, actual visual reviews, summary,
four matched main ablations, exact split and promoted source/configuration must
still match the selection record. This supplements the frozen batch checks
without changing its cleaner implementation.

Independent legacy cleaners are also available, without any reconstruction:

```bash
uv run --no-sync python clean_legacy_gggs.py --input /absolute/native_gggs.ply \
  --variant C1 --thresholds original --output /absolute/new_output
```

Use C2 for V3 and `--thresholds adaptive` for the scale/spacing-normalized variant.
They export full native/baked/removed models and IDs. `RULES_COMPLETE.json` records
execution only; these appearance rules require native multiview evaluation and
are not automatically certified plant-preserving cleanup.

The original global Boundary F1 is still reported. The foreground safety gate
uses the target-local contour F1 plus boundary coverage, because a measured safe
7084-point deletion changed the global contour metric while all per-camera FG
MSE/Alpha/thin/dark-MSE/depth and FG-local F1 metrics stayed exactly unchanged.
See comparisons/GLOBAL_BOUNDARY_FALSE_REJECTION_WITNESS.json.

Outputs: native PLY, baked SuperSplat PLY, native/baked removed PLY, original
IDs, parameters, actual candidate metrics/reasons, source snapshots, paired
black/white/Alpha images, runtime/memory and COMPLETE only after final validation.
An unchanged result is `no_accepted_modification`; it is never labeled cleaning success.
Binary mask/depth/free-view proxies do not certify all botanical semantics.

Actual paths are indexed in `experiments/GGGS_Cleaner_Comparison/artifact_index/`.
`MODEL_READY.json` denotes an exported model awaiting final evaluation, not a complete experiment.
Wall time includes I/O and concurrent workstation load; it is not an isolated GPU throughput benchmark.

Local native code is authoritative for GGGS parameter encoding. PyTorch docs
were read through Context7 for the installed 2.10 version:
https://github.com/pytorch/pytorch/blob/v2.10.0/docs/source/notes/autograd.rst .
Bocha returned HTTP403; no web result is used to invent experimental conclusions.
