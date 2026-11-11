# ForeSplat Automatic Gaussian Artifact Suppression

Standalone post-processing module for native GGGS SH Gaussian models. Keeps original E10/P2/R5 and all previous assets unchanged. GPU1 and the existing uv environment are mandatory for this experiment.

From `ForeSplat_VGGT_Omega_Frontend/`:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-project --python .venv/bin/python python \
  ForeSplat_AutoGaussianCleaner/scripts/auto_clean_gaussians.py \
  --scene ForeSplat_AutoGaussianCleaner/configs/scenes/KongQueZhuYu.json \
  --output ForeSplat_AutoGaussianCleaner/experiments/development/KongQueZhuYu/C

CUDA_VISIBLE_DEVICES=1 uv run --no-project --python .venv/bin/python python \
  ForeSplat_AutoGaussianCleaner/scripts/auto_clean_batch.py \
  --manifest ForeSplat_AutoGaussianCleaner/configs/plant_manifest.json \
  --output-root ForeSplat_AutoGaussianCleaner/experiments/all_19_plants_v3 \
  --prepare-missing-models --ablations --resume
```

One global `configs/default.yaml` applies to all plants. `--variant A_once/A/B_once/B/C` switches only ablation factors. No ROI, Gaussian ID or plant-specific thresholds are accepted. Camera sampling, mask bands, local spacing and depth units derive from scene inputs. The automatic loop stops when no feasible candidate improves background alpha, the gain plateaus, or its fixed safety limit is reached.

Scene JSON requires `plant`, `dataset` (images/masks/sparse/0 COLMAP), `model` (native PLY), `gggs_repo` (compatible native renderer), and optionally `geometry_cameras`/`geometry_dir` (existing front-end depth/confidence and pixel transforms). Camera names map the inputs. `geometry_cameras` has `frames` with `image_name`, `intrinsic_model`, `extrinsic_model_w2c`, optional `stem`, and optional top-level `scene_alignment_matrix`. Poses are validated before external geometry is used. Raw RGB location is retained in scene provenance; training RGB supplies foreground photometric targets. Distorted cameras must first be undistorted by the upstream reconstruction pipeline. Input SG degree must be zero for equivalent SuperSplat output; unsupported models fail explicitly.

Actual contribution = native color Jacobian = sum(T×alpha) for foreground, background, boundary, thin structures and confidence-valid depth moments. This accounts for projected ellipses and occlusion. Color never determines invalidity. Geometric depth moments are auxiliary footprint evidence, not ground truth. A2 ranked candidates are pruned within an explicit foreground budget. A3 uses bounded, anisotropic log-scale/quaternion optimization, frozen XYZ/SH and **constant effective opacity**, followed by whole-camera hard quality constraints and rollback. A4 remains uncertain when evidence cannot certify pruning. No low-density or world-box clipping rule is used. All edits trigger reattribution of the current model.

All-camera FG PSNR, masked SSIM/LPIPS, Alpha, boundary F1, thin skeleton coverage, dark-FG Alpha/MSE, depth and normal changes protect the initial representation as well as current best. Native evaluation uses scaled camera intrinsics with the half-pixel resize convention. Default maximum side 960 is reported and may miss subpixel structures; binary RAP masks do not provide independent leaf/pot/petiole labels. Thin coverage is a narrow skeleton proxy, not a guarantee of every petiole. Free views use only external front-end points for support, not the model being cleaned; their projected support and conservative AABB are explicitly proxies. Full upstream training uses resolution2, 30K, sh3/sg0, original locked GGGS core and corrected camera interface. Existing Omega bundles are reused; only the historical Kong baseline uses Standard VGGT.

Each output includes native and baked SuperSplat PLY, original removed rows, edit manifest, input references, all-camera metrics, comparison images, evidence and accepted/rejected candidates. Baking uses `sigma_eff=hypot(sigma,filter_3D)` and opacity compensation; native/viewer render parity is measured on four automatically selected cameras. The viewer model represents the same effective native Gaussians.

`prepare_manifest.py` records the 19-bundle inventory and prepares symlinks without inference. Development plants are frozen before testing; test-triggered changes require a new evaluation version and untouched plants. `auto_clean_batch.py` freezes config/source hashes, runs one GPU job at a time, resumes matching completed outputs, and writes real measured rows after each stage. It prioritizes C on all plants, then runs A/A_once/B/B_once on the four development plants and the first two alphabetically fixed heldout plants, plus historical P2/R5 comparison and automatic R5 cleaning. These evaluation groups do not alter cleaning rules. Pending/failed runs retain empty metrics. An interrupted full training is reported, not passed off as a complete checkpoint. `reports/` under the selected output root contains comparison CSVs, ablations and the continuously updated final status report.

Validation:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-project --python .venv/bin/python python -m pytest \
  ForeSplat_AutoGaussianCleaner/tests -q
```

Mechanism correctness is separate from cross-plant success. Only completed independent plant measurements and C-versus-B ablations can establish generalization or refinement benefit. Reports explicitly retain no-gain outcomes and limitations. Search was attempted through the specified Bocha API (HTTP403 quota); PyTorch2.10 autograd was verified with Context7 and [official source documentation](https://github.com/pytorch/pytorch/blob/v2.10.0/docs/source/notes/autograd.rst). Experimental conclusions come from local files and native rendering.
