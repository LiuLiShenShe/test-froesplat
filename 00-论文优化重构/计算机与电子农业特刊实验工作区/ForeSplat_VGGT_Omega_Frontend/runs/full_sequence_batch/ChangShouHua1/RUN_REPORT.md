# RUN REPORT - ChangShouHua1

- Source frames: /data/fj/F2DMAS/03-SAM/ChangShouHua1
- Total paired frames: 213  |  processed: 205  |  quarantined: 8
- Image resolution: 2160x3840
- Preprocess: balanced @ 512 (model input [688, 384])
- Inference time: 54.657 s
- Peak VRAM: allocated 21968.8 MiB / reserved 30716.0 MiB

## Fusion

- Candidates (per-view foreground pixels): 13176218
- Fused points: 1239199 (target 1500000, core voxel 0.008670530039911408, edge voxel 0.004335265019955704)
- Meaningful observations per fused point: 10.632850736645205
- Edge representatives: 55404 (thinned 0)
- Raw COLMAP points (per-view, single observation): 1000000 (cap 1000000; 0 = unlimited)

## GGGS verification

- readColmapSceneInfo: ok (cameras: 205, ply points: 1239199)
- GaussianModel.create_from_pcd: ok

## OOM ladder attempts

- balanced@512: ok (21968.8 MiB allocated, 54.657 s)

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
