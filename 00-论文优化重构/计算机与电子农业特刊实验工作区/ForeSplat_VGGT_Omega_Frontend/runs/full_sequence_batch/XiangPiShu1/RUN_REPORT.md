# RUN REPORT - XiangPiShu1

- Source frames: /data/fj/F2DMAS/03-SAM/XiangPiShu1
- Total paired frames: 208  |  processed: 204  |  quarantined: 4
- Image resolution: 2160x3840
- Preprocess: balanced @ 512 (model input [688, 384])
- Inference time: 54.223 s
- Peak VRAM: allocated 21892.0 MiB / reserved 30578.0 MiB

## Fusion

- Candidates (per-view foreground pixels): 13236578
- Fused points: 1199409 (target 1500000, core voxel 0.007245974733793936, edge voxel 0.003622987366896968)
- Meaningful observations per fused point: 11.035916855718108
- Edge representatives: 29839 (thinned 0)
- Raw COLMAP points (per-view, single observation): 1000000 (cap 1000000; 0 = unlimited)

## GGGS verification

- readColmapSceneInfo: ok (cameras: 204, ply points: 1199409)
- GaussianModel.create_from_pcd: ok

## OOM ladder attempts

- balanced@512: ok (21892.0 MiB allocated, 54.223 s)

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
