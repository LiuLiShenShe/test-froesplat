# RUN REPORT - WanNianQing1

- Source frames: /data/fj/F2DMAS/03-SAM/WanNianQing1
- Total paired frames: 213  |  processed: 211  |  quarantined: 2
- Image resolution: 2160x3840
- Preprocess: balanced @ 512 (model input [688, 384])
- Inference time: 57.43 s
- Peak VRAM: allocated 22436.9 MiB / reserved 31252.0 MiB

## Fusion

- Candidates (per-view foreground pixels): 13582867
- Fused points: 1212162 (target 1500000, core voxel 0.008129053281631844, edge voxel 0.004064526640815922)
- Meaningful observations per fused point: 11.205488210321723
- Edge representatives: 47459 (thinned 0)
- Raw COLMAP points (per-view, single observation): 1000000 (cap 0; 0 = unlimited)

## GGGS verification

- readColmapSceneInfo: ok (cameras: 211, ply points: 1212162)
- GaussianModel.create_from_pcd: ok

## OOM ladder attempts

- balanced@512: ok (22436.9 MiB allocated, 57.43 s)

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
