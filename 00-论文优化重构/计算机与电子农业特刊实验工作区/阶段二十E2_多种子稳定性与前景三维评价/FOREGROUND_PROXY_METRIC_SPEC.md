# Fixed proxy foreground and structural metric specification

Frozen before new metric observations. Canonical source: E0 `reports/EVALUATION_REFERENCE_MANIFEST.csv`.
The ROI source is E0 `MATCHED_2DGS_DATASET_MANIFEST.csv` **B_R0 mask_path**, including held-out views,
with the original mask SHA256 verified. Nearest-neighbour resizing at the actual frozen camera render size,
then uint8 >127. Every model uses the identical resized boolean ROI tensor and hash for a given view.
The proxy was produced by R0: it is method-conditioned and may favor R0. It is not independent GT.
There are no newly annotated or synthesized manual foreground masks.

The rendered RGB is clamped to [0,1], exactly as E1. Reference resizing is channelwise Pillow default
BICUBIC, float32 /255, exactly as E1. Background is black [0,0,0]. Original and resized reference hashes
are recorded. E1 SSIM uses its unchanged 11px Gaussian window, sigma 1.5, constants 0.01² and 0.03²;
LPIPS uses package 0.1.4, VGG version 0.1, normalize=True with original weight hashes.

For boolean ROI M with N foreground pixels and three RGB channels:

- Foreground MSE = sum over foreground RGB channels of (prediction-reference)² / (3N).
- Foreground PSNR = -10 log10(MSE), with intensity maximum 1. Outside pixels do not enter this denominator.
- Absolute colour error = sum over foreground RGB channels of |prediction-reference| / (3N).
- Coverage proxy = fraction of ROI pixels with rendered alpha >0.5. Missing-alpha fraction = 1-coverage.
- Fixed-mask composited SSIM = native SSIM(M*prediction, M*reference).
- Fixed-mask composited LPIPS = native LPIPS(M*prediction, M*reference).

The latter two are **composited proxies**, not standard independently labelled object metrics.
Shared black pixels contribute to the score. SSIM convolution and LPIPS receptive fields cross ROI
boundaries, so changes near the boundary influence neighboring windows. No claim of an independent
foreground truth follows from a higher score. Empty ROI metrics are undefined and raise an explicit error.

Background suppression is measured independently: mean squared RGB across three channels outside ROI,
mean alpha outside ROI, alpha sum outside ROI, and alpha>0.5 connected components with 8-neighbour
connectivity and size >=16 pixels. Largest component size is also retained. These components may reflect
mask error, real unmasked structure or boundary discrepancies; they are not confirmed floating Gaussians.
Proxy silhouette IoU uses alpha>0.5 versus this predicted R0 mask and is explicitly not true silhouette IoU.

Depth comes from the unchanged renderer `surf_depth` (depth_ratio=0, expected depth), in arbitrary inherited
COLMAP units. Quantiles p01/p05/p50/p95/p99 use ROI pixels with alpha>0.5 and finite positive depth.
Valid pixel count/fraction are reported. Rendered normals and depth-derived `surf_normal` are world-space
vectors from the same renderer. Normal proxy is mean signed (and supplementary absolute) cosine after
vector normalization on this support, excluding nonfinite vectors or norms <=1e-6. It measures internal
agreement of model-derived quantities, not independent surface accuracy or temporal/multiview consistency.

Gaussian count and quantiles of activated scale, opacity, and XYZ distance from the inherited sparse-point
centroid are descriptive. Two scale axes are present for 2DGS surfels. A larger/smaller count or narrower
scale/depth distribution alone is not a validated geometric improvement. No Chamfer/F-score is computed.

Inherited Seed 0 RGB/alpha metrics consume the original E1 float tensors, with hash checks and unchanged
full-reference scores. Depth and normal tensors are newly rendered from the exact frozen Seed 0 PLY into
E2 storage; RGB/alpha replay discrepancies are recorded separately. Their structural proxies describe that
replay realization. They never overwrite E1 evidence. Every E2 metric can be recomputed from stored tensors.

Four independent reporting layers are used: A full raw-RGB fidelity; B fixed-proxy foreground fidelity;
C background suppression; D descriptive geometry. No combined score or universal ranking is defined.
Scene aggregates are arithmetic means across its fixed held-out cameras; a seed-pair is the repeat unit.
Undefined structural quantities are not imputed. `reports/METRIC_VALID_VIEW_COUNTS.csv` records the
defined-view count for each endpoint; `reports/METRIC_OMISSIONS.csv` records each undefined view and
its support-based reason. Available-view descriptive means remain labelled by these counts. A paired
seed effect is omitted if either model lacks that metric on any fixed held-out view, so no effect is
computed from unequal camera subsets. Such omissions are listed in `reports/PAIRED_METRIC_OMISSIONS.csv`.
