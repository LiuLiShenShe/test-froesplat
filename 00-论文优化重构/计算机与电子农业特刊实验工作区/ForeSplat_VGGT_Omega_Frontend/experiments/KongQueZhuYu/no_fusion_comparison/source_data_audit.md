# Source data audit — KongQueZhuYu no-fusion initialization comparison

Purpose: before building E2 (raw VGGT-Ω depth back-projection) and E3 (official-style 100K
COLMAP init), verify that every per-frame input already exists on disk so that **VGGT-Ω never
has to be re-run**. E1 reuses its existing GGGS dataset and checkpoint verbatim.

## A. Paths

- Plant batch root: `runs/full_sequence_batch/KongQueZhuYu/`
- E1 GGGS dataset: `runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset/`
- E1 training run: `experiments/KongQueZhuYu/E1_GGGS_Baseline/`
- Comparison root (this work): `experiments/KongQueZhuYu/no_fusion_comparison/`

## B. Inputs located (all present, no VGGT-Ω re-inference required)

### 1. RGB (208/208) — `GGGS_Dataset/images/crop_XXXX.png`
Original-resolution colour images, **2160 × 3840** (W × H, portrait), 208 files.

### 2. Foreground masks (208/208) — `GGGS_Dataset/masks/crop_XXXX.png`
RAP-FSAM3 binary foreground masks, composited original resolution.

### 3. VGGT-Ω per-frame raw depth — `GGGS_Dataset/per_frame/depth_crop_XXXX.npy`
208 files, shape **(688, 384)** float32, camera-frame **Z** depth. Verified frame 0:
finite everywhere, depth range ≈ 0.587–1.034.

### 4. VGGT-Ω per-frame depth confidence — `GGGS_Dataset/per_frame/confidence_crop_XXXX.npy`
208 files, (688, 384) float32, confidence map. Frame 0 range ≈ 1.00–6.46.

### 5/6. Camera extrinsics + intrinsics — `GGGS_Dataset/per_frame/cameras.json`
Per-frame `intrinsic_model` (3×3, **model** resolution) and `extrinsic_model_w2c`
(3×4, **cam-from-world**), plus `scene_alignment_matrix` `T_align` applied exactly once
(`X' = T_align @ X`) to move the raw VGGT-Ω world into the exported COLMAP/GGGS frame.
`num_frames = 208`, `model_height = 688`, `model_width = 384`.

### 7. Per-frame preprocessing transforms — `GGGS_Dataset/metadata/preprocess_transforms.json`
208 records; each has `pixel_transform` `A` (3×3, `p_model = A p_original`, PIL
pixel-centre: crop → bicubic resize → pad), `pixel_transform_inverse`, original/model
sizes. Used for `K_original = A⁻¹ K_model` and `p_original = A⁻¹ p_model`.

### 8. Per-frame foreground observations / model RGB
- `GGGS_Dataset/per_frame/model_mask_crop_XXXX.npy` — (688, 384) uint8 0/1 foreground mask at model resolution (208).
- `GGGS_Dataset/per_frame/model_rgb_crop_XXXX.npy` — (688, 384, 3) uint8 model-resolution RGB (208).
- Frame 0 foreground pixels = 60,178 / 264,192.

### 9. Current fused point cloud (E1 initialisation source)
- Fused cloud / GGGS init cloud: `GGGS_Dataset/sparse/0/points3D.ply` — **1,252,383** points
  (aligned frame; GGGS `readColmapSceneInfo` prefers `points3D.ply` over `points3D.bin`).
- Fusion provenance: `GGGS_Dataset/metadata/fusion_metadata.json` —
  `num_candidates = 13,640,967`, `num_fused_points = 1,252,383`,
  confidence **percentile-20 threshold = 1.2527** (computed over foreground pixels).
- The COLMAP `points3D.bin` holds a separate **1,000,000**-point raw single-observation cap
  (`max_points=1e6`); it is *not* what GGGS E1 initialised from — the fused `points3D.ply` is.

### 10. E1 GGGS dataset + checkpoint
- Dataset: `GGGS_Dataset/` (images/ + masks/ + sparse/0/{cameras,images,points3D}.{bin,txt}).
- Checkpoint: `experiments/KongQueZhuYu/E1_GGGS_Baseline/checkpoints/final_gaussians/point_cloud.ply`
  (600,458 final Gaussians). **E1 is reused; not retrained.**

## C. Quarantine / view-set consistency

- Source plant had 212 frames; `metadata/quarantined_frames.csv` records 4 drops
  (3 corrupt PNG, 1 missing mask). Net **208 valid views** used everywhere.
- E1 dataset images and masks = the same 208 `crop_XXXX.png` that the per-frame depth /
  confidence / mask arrays index. E2/E3 reuse **exactly** these 208 views (by copying /
  hard-linking the E1 `images/` + `masks/` and the E1 `cameras.bin`/`images.bin` camera and
  pose parameters), so all three experiments train on the identical view set.

## D. Coordinate conventions (verified, reused verbatim — not re-derived)

- `world_points_from_depth = Rᵀ (cam_points − t)`, `cam_points = ((x−cx)/fx·Z, (y−cy)/fy·Z, Z)`
  with **integer pixel indices** and camera-frame Z depth (`inference.unproject_depth_map_to_point_map`).
- Alignment applied once: points `X → T_align X`, poses `E → E T_align⁻¹` (identical to E1).
- Confidence filter: `valid = foreground & finite_geometry & (conf ≥ percentile-20 within
  foreground) & (depth > 0)`; **no** depth-edge filter, no added filtering (matches E1).
- `N_E1 = 1,252,383` (len of E1 init cloud) → E2 targets the same count (per-view stratified,
  seed 0); E3 targets 100,000 (global, seed 0).
