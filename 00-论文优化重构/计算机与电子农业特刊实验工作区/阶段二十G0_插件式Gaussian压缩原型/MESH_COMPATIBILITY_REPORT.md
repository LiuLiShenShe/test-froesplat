# Mesh Compatibility Report

The predeclared `opacity_p40` model was passed through the unchanged Phase20E-3 TSDF protocol. The run is an engineering compatibility check, not a geometric-accuracy claim.

- Status: **PASS**
- Input PLY SHA256: `9c19986f58e830bb9c0a765cd7524b492402cc9fa150611287267998002aa1f6`
- Protocol SHA256: `e7f86af93eaad85a485ffbc1b32b7906469b89e25eb387419fb426fbd5093bb3`
- Integration cameras: `186`
- Depth/alpha finite and nonempty: `True` / `True`
- Output directory: `/data/fj/ForeSplat_Phase20G0_assets/723c02c2_v1/mesh/opacity_p40`
- Mesh variants: raw=1359449 vertices/2539101 triangles, SHA256 14aec8862c83ee8a4c8601c9569aa9fdaaf711cc9b58e90dfd3988041b2ab523, post=1199176 vertices/2369993 triangles, SHA256 4208d58b16d38ec5ce72dd659a48dfe528f9687a76d6bc820f797fde5fd98cdd

Masking, voxel size, SDF truncation, depth truncation, camera focus/radius, and post-processing are inherited from `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十E3_M2M3真实消融与Mesh验证/configs/WanNianQing1_MESH.json`. No predicted-alpha threshold or new geometric target was introduced.
