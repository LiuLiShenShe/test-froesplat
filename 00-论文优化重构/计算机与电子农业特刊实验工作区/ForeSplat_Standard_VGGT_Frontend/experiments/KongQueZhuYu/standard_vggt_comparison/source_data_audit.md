# Source Data Audit — Standard VGGT Frontend (KongQueZhuYu)

## 1. Input manifest

- **Dataset directory**: `ForeSplat_VGGT_Omega_Frontend/runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset`
  (same directory the VGGT-Ω frontend exported for E1)
- **Views read**: **208** image/mask pairs
- **File-naming convention**: RGB = `images/crop_XXXX.png`, Mask = `masks/crop_XXXX.png`; identical 0-padded
  index ordering as E1 (no sorting drift, no sub-selection, no frame sampling)
- **Per-image provenance**: SHA-256 of every source RGB + mask recorded in
  `Standard_VGGT_Frontend/metadata/input_manifest.json` (one entry per frame, incl. its affine518 matrix)
- **Source-of-truth check (hard-link inodes)**: `crop_0000.png` shares the same inode (242615120) as the
  corresponding E1 image → identical bytes, zero RGB pixel changes

## 2. Geometry

- **Image resolution**: 2160 × 3840 (portrait), verified via pycolmap `Camera.width/height`
- **Affine `A`** (original px → 518 model px):
  `s = 518 / 3840 = 0.13489583333333333`
  `A = [[s, 0, pad_left·s], [0, s, pad_top·s], [0, 0, 1]]` with
  `pad_left = pad_top = (3840 − 2160)/2 = 840` (both original and 1024-pad step)
- **Affine round-trip error**: `4.55e-13 px` (float64 exact)
- **Recovered principal point**: `K_orig[0,2], K_orig[1,2] = (1080, 1920)` = image center ✓
- **pycolmap export**: 208 cameras `PINHOLE 2160 3840 … 1080 1920`; principal point matches the
  center pixel of every 2160×3840 source image

## 3. Foreground + confidence gate

- Foreground mask derived from RAP-FSAM3 masks (read from E1 `masks/`, unmodified; pad 0 on the black square)
- Per-view 518×518 mask = nearest-neighbor resize of the original mask, background=0
- Foreground pixel gate: `fg ∧ depth > 0 ∧ conf ≥ p20(in-foreground)`
- **Aggregate counts**:
  | stage | count |
  |---|---|
  | 208-view foreground pixels | 9,743,236 |
  | after conf p20 (p20 = 1.0265, computed only on foreground pixels) | 7,794,590 |
  | final initialization points written (deterministic seed 0) | **1,252,383** |
- Per-view validity: min 2,281 / max 72,846 foreground pixels / view; **0 views dropped** by confidence filtering
- Confidence p20 value `1.0265` is **recomputed here**, not inherited from VGGT-Ω's own per-frame threshold;
  the number coincidentally sits in the same [1, 6.5] range because the VGGT-Ω and standard VGGT depth
  heads use the same expp1 activation and the same normalized 0–1 depth scale

## 4. Sampling

- **Target**: E1's 1,252,383 points; **seed 0**
- Per-view stratified allocation: `alloc_i = round(target · valid_i / Σvalid)`, largest-fraction remainder first
- Points taken uniformly at random **without replacement** within each view's candidate set; no point moved,
  no voxel-averaging, no normal fusion, no sub-pixel refinement, no replication, no interpolation
- Result: exactly 1,252,383 points, each a raw single-observation depth back-projection
- `points3D.ply` carries one observation per point (image_id, point2D index) in COLMAP track; no multi-view fusion

## 5. Model + inference

- **Checkpoint**: `checkpoints/model.pt` = official VGGT-1B (`facebookresearch/vggt`, sha256 `d15bf50a…53afe0`)
- **git commit of standard VGGT code**: `a288dd0f14786c93483e45524328726ab7b1b4ce` (recorded in `model_metadata.json`)
- **PyTorch / CUDA / Python**: torch 2.10.0+cu128 / CUDA 12.8 / Python 3.12 (uv-managed venv)
- **GPU**: NVIDIA RTX A6000 48 GB
- **Inference config**: aggregator + camera_head + depth_head only (no Point Head, no Track Head, no BA);
  518×518 square model input; BF16 autocast; `model.eval()` + `torch.inference_mode()`
- **Autocast boundary fix** (commit `d23ddb98`, this repo):
  only the aggregator runs under `torch.cuda.amp.autocast`; camera_head, pose→extrinsic/intrinsic conversion
  and depth_head run in fp32, matching the official `demo_colmap.py::run_VGGT`. Running the pose conversion
  inside bf16 autocast produced `|RR^T − I| ≈ 1.6e-3` → measured 2.93 px reprojection error. After the fix,
  **208-view reprojection: mean 0.0004 px, p95 0.0011 px, ok=True**.

## 6. Reference (E1, VGGT-Ω) — recorded for comparison, not overwritten

- VGGT-Ω RUN_REPORT: 211 paired frames → **208 processed**, 3 quarantined (see
  `ForeSplat_VGGT_Omega_Frontend/runs/full_sequence_batch/KongQueZhuYu/metadata/quarantined_frames.csv`)
- VGGT-Ω preprocess: **balanced@512** → model input `[688, 384]` (aspect-preserving crop-and-resize, NOT the
  square padding used here)
- VGGT-Ω fusion: 13.64 M candidates → 1,252,383 fused points via voxel fusion
  (`core voxel 0.00957`, `edge voxel 0.00478`), mean observations/point `10.89`, edge representatives `120,737`
- Standard-VGGT frontend here: **no fusion** — one raw observation per point; same 1,252,383 final count
- VGGT-Ω's depth was written at its own 688×384 model resolution; standard-VGGT depth here is 518×518 square.
  Cross-method comparisons map both to original 2160×3840 coordinates via each frontend's affine transform

## 7. Deliverables written by this run

```
experiments/KongQueZhuYu/standard_vggt_comparison/
├── Standard_VGGT_Frontend/
│   ├── depth/crop_XXXX.npy                208 × (518,518,1) float32
│   ├── confidence/crop_XXXX.npy           208 × (518,518) float32
│   ├── predictions/world_points_sampled.npy   (1,252,383, 3)
│   ├── predictions/sampled_provenance.npz     frame, px_model, py_model, conf, depth, xyz per point
│   ├── visualizations/                    (populated by compare scripts)
│   └── metadata.json, metadata/input_manifest.json
└── Standard_VGGT_Dataset/
    ├── images/  (208 hard links to E1 images)
    ├── masks/   (208 hard links to E1 masks)
    └── sparse/0/{cameras.bin|txt, images.bin|txt, points3D.bin|txt, points3D.ply}
```

## 8. Audit sign-off

- Image / mask filenames + count match E1: **✓ 208/208**
- RGB pixels byte-identical to E1 (hard-link inode equality on crop_0000): **✓**
- Foreground mask source: RAP-FSAM3 (E1 `masks/`): **✓**
- Background color used by VGGT: **black** (matches VGGT-Ω convention, NOT white)
- Coordinate system: standard VGGT OpenCV W2C, square-padded affine → original 2160×3840 px, principal point verified
- Sampling determinism: seed 0, no point modification, target count met: **✓**
