# RUN REPORT - DouBanLv3

- Source frames: /data/fj/F2DMAS/03-SAM/DouBanLv3
- Total paired frames: 210  |  processed: 208  |  quarantined: 2
- Image resolution: 2160x3840
- Preprocess: balanced @ 512 (model input [688, 384])
- Inference time: 57.57 s
- Peak VRAM: allocated 22202.5 MiB / reserved 31088.0 MiB

## Fusion

- Candidates (per-view foreground pixels): 9416554
- Fused points: 1369270 (target 1500000, core voxel 0.007653141614404197, edge voxel 0.0038265708072020986)
- Meaningful observations per fused point: 6.877061499923316
- Edge representatives: 129957 (thinned 0)
- Raw COLMAP points (per-view, single observation): 1000000 (cap 1000000; 0 = unlimited)

## GGGS verification

- readColmapSceneInfo: ok (cameras: 208, ply points: 1369270)
- GaussianModel.create_from_pcd: ok

## OOM ladder attempts

- balanced@512: ok (22202.5 MiB allocated, 57.57 s)

## Notes

- All views of the sequence are in `GGGS_Dataset/sparse/0` (no frame sampling).
- `points3D.ply` is the multi-view fused cloud; per-view raw points stay in
  `sparse/0/points3D.bin` (single-observation tracks, exact) and `debug/raw_pointcloud.ply`.
- Provenance: `debug/observations.npz` maps every fused point to its source images,
  pixels, confidences and fusion weights.
- Per-frame depth/confidence/model RGB/model masks are in `GGGS_Dataset/per_frame/`.
- Reprojection error is NOT a geometric accuracy measure (points are depth
  back-projections, not SfM triangulations).
- `images/` and `masks/` are hard links to the source directory (same inodes); do not delete or edit the source plant folder without re-exporting.
