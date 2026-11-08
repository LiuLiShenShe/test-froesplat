# RUN REPORT - XiangPiShu2

- Source frames: /data/fj/F2DMAS/03-SAM/XiangPiShu2
- Total paired frames: 211  |  processed: 207  |  quarantined: 4
- Image resolution: 2160x3840
- Preprocess: balanced @ 512 (model input [688, 384])
- Inference time: 55.257 s
- Peak VRAM: allocated 22124.0 MiB / reserved 30962.0 MiB

## Fusion

- Candidates (per-view foreground pixels): 15484694
- Fused points: 1174423 (target 1500000, core voxel 0.006394896272810518, edge voxel 0.003197448136405259)
- Meaningful observations per fused point: 13.18493762468889
- Edge representatives: 19139 (thinned 0)
- Raw COLMAP points (per-view, single observation): 1000000 (cap 1000000; 0 = unlimited)

## GGGS verification

- readColmapSceneInfo: ok (cameras: 207, ply points: 1174423)
- GaussianModel.create_from_pcd: ok

## OOM ladder attempts

- balanced@512: ok (22124.0 MiB allocated, 55.257 s)

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
