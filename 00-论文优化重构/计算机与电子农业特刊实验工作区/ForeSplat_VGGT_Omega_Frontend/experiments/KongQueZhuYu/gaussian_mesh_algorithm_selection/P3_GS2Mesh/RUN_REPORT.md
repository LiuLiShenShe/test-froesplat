# P3 E10 GGGS → official GS2Mesh: actual extraction report

Actual208-view stereo and both TSDF extractions completed. Uniform208 mesh evaluation and full fixed-view mesh visualization are owned by the parent CPU queue; no final geometry ranking is claimed here.

## Source and invariant input

Official GS2Mesh commit:560f3e8a2349aaad24fe9444ef1ca2770aeec5ab (https://github.com/yanivw12/gs2mesh). Official bundled DLNR Middlebury weights SHA256:57e46b8fed66b685f732b730ebffd71998dcabea49f8443ca77e015419f6ae18. Native DLNR model57,378,681 parameters; mixed precision,10 recurrent iterations,3px LR consistency; model internals unchanged.

E10 original576,396-Gaussian30K model, seed0, physicalGPU1;208 original source IDs0–209 excluding44/206. Exact original K/noncentral principal point and W2C; right rectified +X baseline0.037776436900048636 (native7% median camera radius). All8 representative left renders match historicalE10float RGB exactly (max error0.0). No Gaussian retraining, ordinary3DGS substitute, new frontend or new segmentation.

## Actual runtime

208 stereo rendering:281.84s; peak allocated0.657GiB/reserved5.787GiB.
208 native bidirectional DLNR inference:418.09s; peak allocated3.188GiB/reserved4.162GiB.

Parent-authorized E10/P2 shared-free-view subprocesses ran before DLNR and are recorded separately in ../additional_visual_commands.jsonl; they are excluded from native DLNR time. Command-level wall includes import/startup/teardown and is retained in actual_commands.jsonl.

## Actual mesh files

|Variant|Stage|Absolute path|Bytes|Vertices|Faces|Components|Boundary edges|Nonmanifold edges|
|---|---|---|---:|---:|---:|---:|---:|---:|
|native4–20baseline|raw|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P3_GS2Mesh/meshes/raw.ply|38837036|522597|937251|8023|98651|0|
|native4–20baseline|post|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P3_GS2Mesh/meshes/post.ply|36087672|478560|898522|158|66468|0|

native4–20baseline actual native fusion:22.67s; depth max0.755529.
|shared depth range|raw|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P3_GS2Mesh/meshes_shared_depth_range/raw.ply|63927059|851566|1576682|10877|111988|0|
|shared depth range|post|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P3_GS2Mesh/meshes_shared_depth_range/post.ply|59856016|787660|1514233|251|72307|0|

shared depth range actual native fusion:28.61s; depth max1.569246.

Both routes preserve raw mesh, all component/removal archives and post mesh (only components<50 triangles removed). Vertex RGB is present; no UV-textured OBJ is claimed. Open leaf boundaries are not counted as physical holes.

## Depth/fusion stage diagnosis

Actual mean view valid depth fraction:1.000000; Stereo+RAP support:0.921890.
Native4–20baseline depth maximum0.755529 reduces mean support to0.391169 and creates0.604699 missing interior fraction (>5px from RAP boundary) at the fusion-input stage. Shared max1.569246 keeps0.921890 support and0.072978 interior missing.
These are input/projected proxy gaps, not confirmed3D holes: other views may recover missing surface. Actual native-range source0 post mesh preliminary IoU0.618438, coverage0.667743, boundary Chamfer46.98px; limited to one view and not a208 ranking. Direct gray/normal/color view shows missing pot/marker and central leaf breaks.

Any benefit relies on an external pretrained DLNR stereo prior, rather than improved GGGS optimization. Confidence=exp(-native LR reprojection residual/3px), a consistency score and not calibrated network probability.

## Compatibility and preserved evidence

The original local Open3D0.20/NumPy2.5.4 legacy TSDF produced zero geometry for a known plane, reproducing historicalM5 runtime failure. Independent uv environment now overrides Open3D0.19.0/NumPy2.4.2, matching successful historicalE6/M5. Unchanged native TSDF known-plane validation:17,161vertices/33,800faces, depth error2.03e-6; synthetic meshes were not exported. Failed/successful validation JSONs retained.
Outer stereo orchestration fixes repeated LR/RL padding at width1080→1088; both directions unpad to original shape. RAP mask is reused without10px erosion. Native model/weights/warping/iteration logic unchanged. Shared historical voxel0.0017389331 and truncation0.0086946655 used.
All actual1876 assets with real resolved paths/sizes are in ARTIFACTS_P3.csv; stereo camera validation, source/weight SHA, patch and environment versions are stored locally. Fixed8 Stereo RGB/depth/confidence panels exist in visualizations/stereo/.

Sources: local actual checkpoints/render/stereo/fusion logs and completion records; official source https://github.com/yanivw12/gs2mesh/tree/560f3e8a2349aaad24fe9444ef1ca2770aeec5ab ; official DLNR weights https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network/releases/download/v1.0.0/DLNR_Middlebury.pth . Bocha search unavailable (HTTP403 insufficient quota); Context7GS2Mesh not indexed, official source read directly. Open3D/uv documentation fetched via Context7.
