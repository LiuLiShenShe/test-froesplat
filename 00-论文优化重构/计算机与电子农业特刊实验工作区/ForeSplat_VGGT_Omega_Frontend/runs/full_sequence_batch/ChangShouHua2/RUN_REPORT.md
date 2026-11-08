# RUN REPORT - ChangShouHua2

- Source frames: /data/fj/F2DMAS/03-SAM/ChangShouHua2
- Total paired frames: 212  |  processed: 211  |  quarantined: 1
- Image resolution: 2160x3840
- Preprocess: balanced @ 512 (model input [688, 384])
- Inference time: 85.329 s
- Peak VRAM: allocated 22436.9 MiB / reserved 31252.0 MiB

## Fusion

- Candidates (per-view foreground pixels): 13317095
- Fused points: 1289801 (target 1500000, core voxel 0.00876660933096373, edge voxel 0.004383304665481865)
- Meaningful observations per fused point: 10.324922216683039
- Edge representatives: 62958 (thinned 0)
- Raw COLMAP points (per-view, single observation): 1000000 (cap 0; 0 = unlimited)

## GGGS verification

- readColmapSceneInfo: ok (cameras: 211, ply points: 1289801)
- GaussianModel.create_from_pcd: ok

## OOM ladder attempts

- balanced@512: ok (22436.9 MiB allocated, 85.329 s)

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
