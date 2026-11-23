# TomatoPGT independent COLMAP run

This run starts with the original TomatoPGT JPEGs for each of the 42 local time-point scans across three cultivars. It reproduces the benchmark `prepare_colmap.py` recipe: resize to a 1600 px long edge, mask the turntable background using color distance to the image border, run single-camera SIMPLE_RADIAL COLMAP with CPU SIFT, match all pairs, and select the mapper model with the most registered views. No benchmark sparse model or camera poses are loaded. Each scan's `colmap/independent_sfm.json` records the input and model provenance; `colmap/logs/` contains the COLMAP logs. Camera intrinsics and feature coordinates are mapped back to the original image dimensions, with the radial distortion coefficient retained for undistortion.

COLMAP supplies relative geometry. The May 8 Beefsteak scan uses documented target-board baselines and archived 2D observations for scale, without importing Metashape camera poses or geometry. Other scans estimate scale from target-board correspondences in their own images and COLMAP cameras. Centimetre outputs require passing the scale checks. Scans without accepted scale retain model-unit outputs.

The serial queue then runs SAM3, GGGS at 30K iterations, bounded TSDF, plant/pot separation, individual leaf mesh extraction, and phenotype reporting. Physical GPU 1 is reserved for this queue's GPU steps. Height is the visible plant mesh extent along the estimated vertical. SAM3 leaflet and compound-leaf candidates are reported separately; reference leaf traits require matched identity and compatible trait definitions.

- [Batch index](summary/batch_index.html)
- [Per-scan results](summary/batch_results.csv)
- [Queue status](summary/batch_progress.json)
- [Current stage and log path](summary/progress.json)
- [May 8 scale calibration](BFS_05082025/calibration/scale_calibration.json)

`../ForeSplat_TomatoPGT_Pilot` is the earlier camera-reuse experiment. Its outputs are separate and are not used by this run. Restart this run using the existing uv environment and `run_tomatopgt_pilot.py --all` after checking that the current queue is stopped.
