# P3: E10 GGGS -> official GS2Mesh

Preparation verified. Actual execution status and completion records are in STATUS.json, render_complete.json, stereo_complete.json and fusion_complete.json when those files exist. GPU work follows P1/P2 per the parent schedule.

Official source: https://github.com/yanivw12/gs2mesh/tree/560f3e8a2349aaad24fe9444ef1ca2770aeec5ab . Stereo model: native DLNR_Middlebury with10 recurrent inference iterations and native3px bidirectional consistency. Full model and weight SHA are in PROVENANCE.json. Context7 had no GS2Mesh entry; official source was read directly. Bocha returned HTTP403 insufficient package quota; no search result or training-data evidence is claimed.

The renderer adapter loads the actual E10 30K GGGS PLY with its corrected native renderer. All208 source cameras, K and the noncentral principal point are preserved at1080×1920. Novel right cameras use native7% median camera-radius baseline (0.037776436900048636) along each left camera's positive X axis. DLNR flow is negative for this arrangement; native disparity=-flow and depth=fx*baseline/disparity. CAMERA_STEREO_VALIDATION.json contains the actual CPU checks for all208 pairs.

Adaptation records:
- Replaces GS2Mesh's ordinary3DGS loader/render interface with the existing E10 GGGS interface; no Gaussian training.
- Fixes GS2Mesh's LR/RL loop padding for width1080: both directions use original1080×1920 inputs padded once to1088×1920 and identically unpadded.
- Uses original RAP masks, without recomputing SAM or applying the native10px erosion that can remove thin edges.
- Keeps native DLNR architecture/weights/iterations/occlusion logic and native Open3D ScalableTSDF integration. Uses shared historical voxel/truncation scale.
- Primary TSDF uses native4–20 baseline depth range. An additional separately stored fusion uses the shared historical1.5692457556724548 max depth to diagnose range clipping.
- The real E10 median-depth range audit found49.71% of valid foreground pixels outside the native range (maximum0.75553), or55.80% as a mean across views. This is a reference-only diagnosis, not a P3 stereo or mesh result; both fusion variants must be inspected.
- Confidence is exp(-native LR reprojection residual/3px), a consistency score rather than calibrated network probability.
- Native raw mesh retained; uniform post-processing removes only components with fewer than50 triangles. TSDF vertex colors are distinguished from UV-textured OBJ.

Execution from this directory:

```bash
.venv/bin/python scripts/run_p3_driver.py --smoke-only
.venv/bin/python scripts/run_p3_driver.py
```

The second command resumes the smoke frame and runs full stereo, both fusion routes,208-view raw/post evaluation and actual fixed-eight-view panels. Each stage has a log, exact argv, wall time, sampled child GPU memory and PyTorch allocated/reserved memory. CPU stages can continue after stereo releases GPU. The uv-created local .venv reads parent torch/Open3D packages without modifying their environment; supplementary dependencies are installed only locally. No conda was used.

The active local environment overrides Open3D with0.19.0 and NumPy2.4.2, matching the successful historical E6/M5 runtime. The initial parent Open3D0.20 runtime returned an empty mesh for a known plane; the unchanged native TSDF succeeds with17,161 vertices/33,800 faces and maximum depth error2.03e-6 in the local fixed runtime. Both validation records are retained; synthetic test meshes were never exported as experiment artifacts.

Outputs once actually produced:
- stereo/000–207/left.png, right.png, GGGS auxiliary depth/alpha NPZ, reused RAP mask;
- stereo/*/out_DLNR_Middlebury/depth_raw.npy, depth.npy, both disparity directions, confidence.npy, occlusion_mask.npy;
- meshes/raw.ply and post.ply; meshes_shared_depth_range/raw.ply and post.ply;
- each mesh directory's evaluation contains uniform208 views and topology; visualizations contains source IDs0,26,53,79,105,131,157,183.

These paths are a workflow description only until execution verifies actual files; the global delivery index must include actual existing paths and real file sizes after execution.

