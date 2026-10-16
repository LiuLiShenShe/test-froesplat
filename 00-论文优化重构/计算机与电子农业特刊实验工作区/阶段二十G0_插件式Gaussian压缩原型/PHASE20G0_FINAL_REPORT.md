# ForeSplat Phase 20G-0 Final Report

## Scope

Scene `WanNianQing1` / `B_R0` / training seed 0 / M0 was selected from the frozen Phase20E-1 source. Every compression run starts from the same PLY `436ec90adae95941f2228789825a3f21c4300a3a605796ea8da387e574c633ef`. Ratios and order were frozen in `configs/EXPERIMENT_PLAN.json` before metrics. No training, SAM3, R1, COLMAP, split, mesh source, or paper TeX was modified.

## Reproducibility

- Baseline commit: `723c02c20e281b150d38144e61c912e5128849b4`
- Plan SHA256: `568fa560562f0778441413192e82018244b6cd6a40b9313269d4e00980bd2c69`
- Source PLY bytes: `31342576`; source SHA256: `436ec90adae95941f2228789825a3f21c4300a3a605796ea8da387e574c633ef`
- Frozen held-out cameras: `27` per run
- Physical GPU: index 1, UUID `GPU-6836f890-98b2-5891-07ef-675c25199c65`, `NVIDIA RTX A6000`
- Protected historical files hashed: `4241`, historical hashes verified: `4174`
- Final input recheck: see `reports/FINAL_INPUT_RECHECK.json`

## Real compression results

| method | removed ratio | Gaussians | PLY bytes | raw PSNR | raw SSIM | raw LPIPS | ROI PSNR proxy | FPS mean | peak reserved VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0 | 128447 | 31342576 | 5.2126 | 0.26046 | 0.60688 | 23.0828 | 167.937 | 0.922 |
| random | 0.2 | 102758 | 25074460 | 5.2089 | 0.25262 | 0.61366 | 22.5072 | 180.380 | 0.807 |
| random | 0.4 | 77069 | 18806343 | 5.2021 | 0.23946 | 0.62369 | 21.5909 | 197.378 | 0.643 |
| random | 0.6 | 51379 | 12537983 | 5.1851 | 0.21428 | 0.63853 | 19.8133 | 221.874 | 0.520 |
| opacity | 0 | 128447 | 31342576 | 5.2126 | 0.26046 | 0.60688 | 23.0828 | 167.029 | 0.922 |
| opacity | 0.2 | 102758 | 25074460 | 5.2114 | 0.25888 | 0.60848 | 22.9348 | 182.086 | 0.836 |
| opacity | 0.4 | 77069 | 18806343 | 5.1992 | 0.23799 | 0.62307 | 21.1871 | 200.739 | 0.668 |
| opacity | 0.6 | 51379 | 12537983 | 5.1771 | 0.20980 | 0.63790 | 19.1557 | 220.900 | 0.562 |

PLY byte reduction is measured separately from Gaussian-count reduction; no quantization or SH compression was applied. The 0% random and opacity outputs preserve the source PLY SHA256. All eight models loaded through the untouched native `GaussianModel.load_ply`, and the independent native model-directory validation is `PASS`.

## Evaluation controls

All runs use the same 27 held-out cameras, 1060x1900 resolution, SH degree 3, black background, and frozen E1/E2 metric definitions. LPIPS is VGG v0.1 with `normalize=True`; RGB is clamped only at metric input, and raw render ranges are retained. The ROI is the frozen B_R0 proxy mask and is not independent ground truth. Edge values are a 3-pixel eroded-inner-boundary diagnostic, also a proxy. Offline pruning time is not included in FPS. FPS uses seven full-camera rounds per model, two warmup sweeps, explicit CUDA synchronization, and 189 timed frames per model.

Unpruned replay deltas against the historical E1 render are retained in `reports/UNPRUNED_E1_REPLAY_DELTA.csv`; the new evaluation does not silently replace it with the old render.

## Method status

- Generic plugin interface: **PASS**
- Random and opacity baselines: **PASS**
- PUP: **BLOCKED**, not approximated. Native ABI has no Fisher buffer; 2DGS has five spatial parameters and surfel covariance rather than the official 3D six-parameter path.
- LightGaussian: **BLOCKED**, not approximated. Native ABI has no count-render output; official pinned kernel counts accepted hits and opacity, while its volume correction assumes three activated scales.
- Tiny exact Fisher evidence: `PASS`, with 5x5 Fisher and square-after-sum relative error `8.786549`.

## Mesh

The predeclared Opacity 40% model passed the frozen E3-3 TSDF extraction and reload protocol: `PASS`. This demonstrates renderer/depth/alpha/TSDF compatibility only; it does not demonstrate geometric accuracy improvement.

## Acceptance status

```text
PHASE20G0 STATUS: PARTIAL
PLUGIN_INTERFACE: PASS
BASIC_PRUNING: PASS
PUP_2DGS: BLOCKED
LIGHTGAUSSIAN_2DGS: BLOCKED
MODEL_LOAD: PASS
RENDER_EVALUATION: PASS
MESH_COMPATIBILITY: PASS
FOREGROUND_AWARE_EXTENSION: RECOMMENDED
NEXT_PHASE: REMEDIATION_REQUIRED
```

Overall status is PARTIAL because the required independent compressor and real baseline evaluation are complete, but neither conference-derived scorer is yet a faithful 2DGS adaptation.
