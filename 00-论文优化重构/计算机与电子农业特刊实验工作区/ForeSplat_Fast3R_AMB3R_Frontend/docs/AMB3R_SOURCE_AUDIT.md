# AMB3R official source and paper audit

Audit date: 2026-11-07. Official checkout: `92c4081f910f98e683503092b85301861519175e`. Core files are unmodified. The checkpoint is the official AMB3R Google Drive artifact already downloaded into `third_party/amb3r/checkpoints/amb3r.pt` (3.9 GiB), SHA256 `2d9342282cce0a20e1074ad1513d9a18c3d100dc06c734a4a11f2d1baeea5239`.

Read directly: [arXiv paper v1](https://arxiv.org/pdf/2511.20343), [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_AMB3R_Accurate_Feed-forward_Metric-scale_3D_Reconstruction_with_Backend_CVPR_2026_paper.pdf), [official repository](https://github.com/HengyiWang/amb3r). The PDFs were downloaded and their full text inspected, including sections 3 and 4.1–4.5. PDF SHA256s: arXiv `4283f7c102f8d9da841034e095e59c7678940a73b6bce100a7ac604267afffc8`; CVPR `658532f7bbb9c6cf95d47a56d191449738ec61e5cb88ad634ba6c4e790fcbc3b`.

Bocha search was attempted by the parent task and returned quota-related HTTP 403. This audit uses directly fetched official primary sources. Browser CDP was checked using the web-access skill and is not configured; no login/browser operation was needed for these public artifacts. Context7 was used for current FlashAttention installation documentation, resolving `/dao-ailab/flash-attention` then fetching the detailed CUDA/PyTorch compatibility question.

## Frontend and backend execution

[`amb3r/model.py`](https://github.com/HengyiWang/amb3r/blob/92c4081f910f98e683503092b85301861519175e/amb3r/model.py) constructs the VGGT frontend, a sparse backend, and a metric scale head. `forward(frames, iters=0)` encodes images once and decodes the frontend heads, returning a one-element stage list. `iters=1` returns that stage and the backend-refined stage. The independent adapter calls these exact methods; it does not call the adaptive SfM/VO model method, whose early stopping or blending would confound this OFF/ON ablation.

[`amb3r/frontend.py`](https://github.com/HengyiWang/amb3r/blob/92c4081f910f98e683503092b85301861519175e/amb3r/frontend.py) accepts `(B,T,3,H,W)` images in `[-1,1]`, converts them to `[0,1]`, and uses VGGT's patch encoder. Decoding uses the official `mem_eff=True`, retaining decoder feature layers 4, 11, 17 and 23 for prediction heads. Camera, depth, confidence, and point heads run outside autocast; the aggregator uses the official BF16 preparation. Official attention uses PyTorch scaled-dot-product attention; `XFORMERS_AVAILABLE` is explicitly false in the vendored attention code, so no additional xformers-based alteration is necessary.

[`get_voxel_feat`](https://github.com/HengyiWang/amb3r/blob/92c4081f910f98e683503092b85301861519175e/amb3r/model.py#L99) concatenates 1024-channel encoder and 2048-channel decoder features, detaches geometry/features, reshapes to `(H/14,W/14)`, applies the backend aligner, and resizes points/features to `(H/7,W/7)` before voxel construction. The inline comment suggesting a quarter resolution is inconsistent with the executed `H//7,W//7`; the adapter preserves executed official code.

[`amb3r/backend.py`](https://github.com/HengyiWang/amb3r/blob/92c4081f910f98e683503092b85301861519175e/amb3r/backend.py) builds voxel indices from the joint pointmap bounding box, hashes voxel coordinates, and averages **features within voxels** using torch-scatter. This is the official learned backend, not a new output point-cloud averaging algorithm. Default voxel size `0.01` is in the frontend's normalized coordinates, as described in paper section 4.2. The sparse Point Transformer V3 serializes voxel coordinates, processes them through its U-Net architecture and FlashAttention, then returns per-voxel features.

`voxel_to_point_interpolation` computes PyTorch3D KNN with `K=16`. Its `knn.dists` are squared distances, so the executed weights `1/(dists+1e-8)` are inverse squared-distance weights, normalized over neighbors. The gather/reduction uses 50,000-point chunks; KNN itself still receives the complete point set. The features are downsampled and injected via trained zero-convolution blocks into patch tokens and 48 alternating frontend attention layers. These operations are kept unchanged.

The checkpoint's `backend.interp_v2` flag controls whether the bounding-box minimum is restored to voxel coordinate centers before KNN. `load_weights` reads the flag, supplies false only for older missing-key checkpoints and then loads strictly. The adapter never overwrites the checkpoint flag; it records its actual value in runtime metadata. This flag is important because the legacy branch computes voxel coordinates without that bounding-box offset; do not silently "fix" the official model when benchmarking.

Actual CPU strict loading passed on this checkpoint (model initialization 12.200 s, strict load and BF16 preparation completed 16.310 s after initialization start). The loaded `backend.interp_v2` value is **false**. This audited official branch is retained for the comparison; no correction is introduced into the network.

## Camera and metric scale conventions

In `frontend.py` lines 152–161, `pose_encoding_to_extri_intri` supplies world-to-camera `(3,4)` extrinsics, and `closed_form_inverse_se3` supplies camera-to-world `(4,4)` poses. `pts3d_by_unprojection` uses the normalized `depth`, **not** `depth_metric`. Raw outputs declare normalized first-input-camera coordinates; agreement between the independently predicted point head and pose/depth geometry is evaluated separately and is not guaranteed by that declaration. Actual reprojection checks in the adapter validate depth-unprojected points, pose inversion, rotation matrices and first-camera identity. Direct point-head reprojection is reported separately as a consistency diagnostic; it is not a feature-track error or ground-truth accuracy measure.

`ScaleProjector` uses per-frame encoder patch/class/register features and decoder depth features to predict the log metric depth at the pixel with median predicted depth. The official scene scale is:

`s = torch.median_t(exp(median_metric_z_log[t]) / (torch.median_pixels(depth[t]) + 1e-8))`

Then `depth_metric = depth * s`. The adapter preserves all official outputs and labels metric pointmaps and scaled camera translations as **derived** arrays. NumPy lower medians are used to match `torch.median` for even array sizes. Fast3R geometry remains scale-ambiguous in its own bundle. No measured scan alignment, ground-truth scale correction, or guaranteed centimeter accuracy is claimed here. Black RAP foreground backgrounds are included in the official scale predictor, because masking that predictor would modify the evaluated inference. Metric scale reliability on this plant requires actual measured distances unavailable to this phase.

There is no native AMB3R local pointmap head. The `pts3d_by_unprojection` field is still globally expressed. It must not be named a local pointmap or concatenated as if different SfM clusters shared a reference camera.

## Confidence compatibility and evaluation limits

The vendored VGGT head [`head_act.py`](https://github.com/HengyiWang/amb3r/blob/92c4081f910f98e683503092b85301861519175e/thirdparty/vggt/heads/head_act.py) uses `confidence = 1 + exp(logit)`, so its raw range is greater than 1 (up to floating-point saturation). The SfM transform `(confidence-1)/confidence` therefore lies in `[0,1)` for finite valid values. This matches the mathematical range of Fast3R's `exp`-with-minimum-one confidence; distribution calibration across models must still be inspected on the real images. No hand-made confidence amplification is applied in the Base adapter.

`confidence>1` is a model validity baseline, not a selective high-quality filter for an expp1 head. Per-frame q85 export provides a separate quantile-threshold comparison intersected with that baseline. Ties and floating-point floor saturation can prevent this threshold from selecting a nominal top 15%; actual retained counts and tie/floor fractions are reported. Raw masks are preserved; evaluating a one-pixel mask boundary does not modify segmentation and measures the whole foreground silhouette, including stem/pot if present. No per-leaf annotated ground truth has been supplied, so leaf-only accuracy should not be fabricated.

OFF/ON uses identical transformed RGB/mask arrays, original order, input size, seed, shared-input digest and explicit stage-0 pointmap fingerprints. The full 208-frame sequence is attempted first. If memory prevents either stage, both stages must use the exact same explicitly labeled representative subset. Such results do not establish full-sequence performance. SfM adaptive refinement remains a distinct experiment.

## Isolated uv environment

The upstream README suggests Conda/Python 3.9/Torch CUDA 11.8. The authorized task requires uv. The existing isolated `.venv-amb3r` uses Python 3.11.9, Torch 2.5.0+cu121 and torchvision 0.20.0+cu121; global environments are untouched. PyTorch3D v0.7.8 source commit `75ebeeaea0908c5527e7b1e305fbc7681382db47` and torch-scatter `2.1.2+pt25cu121` are already installed and import successfully. spconv-cu120 `2.3.6` imports; GPU functionality is validated in the actual Base ON run, not inferred from import.

The inherited FlashAttention 2.7.3 extension required `libcudart.so.11.0` while Torch uses CUDA 12.1. It was replaced with the official [v2.7.3 CUDA12/Torch2.5/CPython3.11/ABI-FALSE wheel](https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl), installed using `uv pip install --python .venv-amb3r/bin/python --no-deps --reinstall`. Wheel SHA256 `b1243e9b86687348a5ab03a073abacdf8e3d5e9e4b7e5326a183f47348c5dfba`. The [official setup.py](https://github.com/Dao-AILab/flash-attention/blob/v2.7.3/setup.py#L429) selects these exact CUDA-major/Torch-major-minor/ABI tags. No CUDA11-runtime symlink was used.

NumPy is pinned back to official `1.26.4`; inherited latest pandas and plyfile were incompatible and are pinned to `2.2.3` and `1.1.2` respectively. The complete environment is recorded in `frontends/amb3r/requirements.lock.txt`. All Python subprocesses set `CUDA_VISIBLE_DEVICES=1`; only physical GPU1 is authorized. No conda, global Torch update, training, exposure compensation, mesh extraction, or segmentation rerun occurs.

At this checkout there is no root AMB3R LICENSE file. Licenses exist in several vendored components; this audit records that absence rather than inventing a top-level license. Distribution/reuse decisions require checking the official project terms independently.

## Actual full 208-frame OFF/ON results and head-gauge diagnostic

Both original full-sequence forwards completed on physical GPU1, preserving all 208 usable inputs at 294×518 (width×height), seed 42, original order, complete view interaction and unmodified network weights. OFF inference took 43.6516 s with 23.6710 GiB peak allocated memory; ON took 120.1082 s with 30.8256 GiB peak allocated / 38.6016 GiB peak reserved memory. Shared-input SHA256 and stage-0 world-point SHA256 are exactly identical between the two runs. No representative subset was needed.

Pixel-weighted foreground confidence validity increased from 20.6765% to 94.8171%; mean per-view foreground silhouette boundary validity increased from 8.2876% to 90.6600%. These are confidence-valid support counts, not measured 3D accuracy. Many ON confidence values still lie close to the floating-point floor. The separate q85 derived clouds were repaired to intersect baseline `conf>1`, while preserving previous PLYs and every raw prediction.

Actual CPU inspection also found substantial point-head / camera-head disagreement. Median across views of per-view foreground direct-point reprojection medians was OFF 273.6383 px and ON 231.9382 px. The first view was OFF 47.3984 px and ON 20.1026 px. Depth-unprojection reprojection is an algebraic consistency check and remains approximately subpixel after the original BF16 inference arithmetic; this does not prove camera accuracy.

The read-only `head_consistency_diagnostic.json` files show median pixelwise world-coordinate norm ratios `norm(world_points)/norm(pts3d_by_unprojection)`:

| Scope | OFF | ON |
| --- | ---: | ---: |
| First input camera foreground | 0.4937 | 0.9839 |
| Median across all view foreground medians | 1.6315 | 2.2963 |

These norm ratios are **not fitted scales**. Their variation across views does not establish one global correction factor and cannot identify whether point predictions, depth, poses or multiple heads are inaccurate without ground truth/tracks. Raw arrays, primary cameras and primary clouds remain untouched.

The official [training implementation](https://github.com/HengyiWang/amb3r/blob/92c4081f910f98e683503092b85301861519175e/amb3r/training.py#L198) applies a sequence-level scale-invariant alignment to predicted pointmaps, separately applies per-view scale-invariant alignment to depth, and multiplies each predicted camera translation by that depth scale before the multitask loss (test lines 198–215; train lines 404–419). The [multitask loss](https://github.com/HengyiWang/amb3r/blob/92c4081f910f98e683503092b85301861519175e/amb3r/loss.py#L124) then supervises camera, depth and point heads separately. These supervision-time alignments are not applied by raw `AMB3R.forward` at inference. Independent head gauge freedom is a plausible explanation for the observed disagreement, not a proven single root cause.

Consequently, `world_points_metric_derived` and `global_mask_confidence_metric_derived.ply` are explicitly marked `gauge_validated=false`: they transfer the depth-derived scene scale to the direct point head under an unverified shared-gauge assumption, and must not be advertised as guaranteed metre coordinates. Their original names and values are preserved for provenance. The additional `global_depth_unprojection_mask_confidence_metric_predicted.ply` uses `pts3d_by_unprojection_metric_derived` with the same scale as `cameras_metric_derived.npz`; this pair is algebraically consistent by construction and remains a learned metric prediction without measured plant accuracy. It uses unchanged foreground masks and the same primary confidence filter, with no fitting, alignment or point averaging.

Reproduce the supplementary CPU inspection with `.venv-amb3r/bin/python -m frontends.amb3r.diagnose_heads`; it only reads saved predictions and writes diagnostics.
