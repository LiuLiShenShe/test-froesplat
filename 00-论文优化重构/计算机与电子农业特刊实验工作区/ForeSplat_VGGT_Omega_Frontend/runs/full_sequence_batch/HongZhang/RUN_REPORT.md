# RUN REPORT - HongZhang

- Source frames: /data/fj/F2DMAS/03-SAM/HongZhang
- Total paired frames: 213  |  processed: 199  |  quarantined: 14
- Image resolution: 2160x3840
- Preprocess: balanced @ 512 (model input [688, 384])
- Inference time: 51.725 s
- Peak VRAM: allocated 21507.2 MiB / reserved 29960.0 MiB

## Fusion

- Candidates (per-view foreground pixels): 15770302
- Fused points: 1229530 (target 1500000, core voxel 0.006708717369562531, edge voxel 0.0033543586847812655)
- Meaningful observations per fused point: 12.826284840548828
- Edge representatives: 20502 (thinned 0)
- Raw COLMAP points (per-view, single observation): 1000000 (cap 1000000; 0 = unlimited)

## GGGS verification

- readColmapSceneInfo: ok (cameras: 199, ply points: 1229530)
- GaussianModel.create_from_pcd: ok

## OOM ladder attempts

- balanced@512: ok (21507.2 MiB allocated, 51.725 s)

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
