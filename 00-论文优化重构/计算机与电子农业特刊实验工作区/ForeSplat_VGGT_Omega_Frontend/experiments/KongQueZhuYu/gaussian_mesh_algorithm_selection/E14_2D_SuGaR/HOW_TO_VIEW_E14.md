# E14 actual models and viewing

Direct native browser viewer after the experiment finishes (binds only127.0.0.1, GPU1):

`/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/.venv/bin/python /data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/scripts/view_native_web.py --stage final --port 8864`

Open http://127.0.0.1:8864; on a remote server forward that port through SSH. Use --stage pruned/before/initial/refined to inspect actual corresponding models. Refined depth/normal are explicitly auxiliary. No server is started by the experiment driver.

Use --ply /actual/absolute/native2DGS.ply --label E11 for an existing E11 native2D surfel PLY. The viewer preserves its representation; E11 training losses/priors remain E11, and this is not an E14 model conversion.

Initial/pruned/30K PLYs are native2D surfels with two scales. SuperSplat3DGS import does not provide an equivalent rendering. Use the official2DGS renderer/network viewer; native fixed/freeview images are in visual_views/surfel/.

The refinement export has three Gaussian scales and is compatible with3DGS/SuperSplat. The .pt is the original mesh-bound SuGaR representation; Gaussian PLY alone does not carry mesh binding.

Native2DGS rendering command for the actual30K model (requires a native network viewer to connect):

`/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/.venv/bin/python /data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/official/gaussian_splatting_2d/view.py -s /data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/viewer_dataset -m /data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/training -r 2 --iteration 30000 --depth_ratio 1.0 --initialization_prior ""`

For actual7000 post-pruning PLY, use viewer_models/pruned as -m and --iteration 7000. The lightweight viewer dataset keeps corrected cameras/RGB and omits dense prior loading. The GUI camera adapter preserves supplied projection/pixel centres.

Load the native OBJ together with its sibling MTL and PNG. Use geometry-only PLY for gray/wireframe/normal comparison; initial Mesh has vertex colors rather than UV texture.

Actual fixed8 Mesh gray/normal/vertex-color/visible-face-edge wireframes are in visual_views/mesh/. Only refined_native_textured_OBJ provides native UV texture; missing vertex color/texture is explicitly unavailable.

Every file retains standardVGGT/COLMAP arbitrary world gauge. If the model appears upside down, orbit the view180° aboutX and keep the original geometry unchanged. This view rotation does not establish metric scale.

E14 inspection order: initialization → Gaussian iteration1 → iteration7000 before/after plusremoved →30000 → initialraw/post Mesh → refinementGaussian/PT → refinedgrayMesh → native texturedOBJ.

RefinedGaussian nativeDepth/Normal are unavailable from the original3DGS API. Centre-Z expected-depth/axis-normal maps are explicitly auxiliary. Geometry evaluation uses actual refinedMesh raycastDepth/Normal.

| Artifact | Stage | Actual absolute path | Bytes | Viewer |
|---|---|---|---:|---|
|Point Cloud PLY|initialization|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/training/input.ply|34361515|MeshLab/Open3D|
|Gaussian PLY|initial_native|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/training/point_cloud/iteration_1/point_cloud.ply|310525669|Native2DGS renderer|
|Gaussian PLY|cluster_before|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/training/point_cloud/iteration_7000/point_cloud.ply|20256679|Native2DGS renderer|
|Gaussian PLY|cluster_after|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/cluster_pruning/after.ply|10720915|Native2DGS renderer|
|Checkpoint|cluster_before_state|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/cluster_pruning/before_pruning_checkpoint.pth|58782162|Native2DGS|
|Checkpoint|cluster_after_state|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/cluster_pruning/after_pruning_checkpoint.pth|31112722|Native2DGS|
|Gaussian PLY|cluster_removed|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/cluster_pruning/removed.ply|9537271|Native2DGS renderer|
|Gaussian PLY|final_30000|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/training/point_cloud/iteration_30000/point_cloud.ply|125392620|Native2DGS renderer|
|Checkpoint|native_30000|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/training/chkpnt30000.pth|363848658|Native2DGS|
|Mesh PLY|initial_raw|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/initial_mesh/raw.ply|67768736|Blender/MeshLab/Open3D|
|Mesh PLY|initial_post|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/initial_mesh/post.ply|63312256|Blender/MeshLab/Open3D|
|Mesh PLY|initial_native_largest_cluster|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/initial_mesh/native_largest_cluster.ply|42421192|Blender/MeshLab/Open3D|
|Gaussian Model|refined_native|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/refinement/sugarfine_post_normalconsistency01_gaussperface1/15000.pt|1059192021|NativeSuGaR|
|Gaussian PLY|refined_3d_export|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/refinement/refined_ply/sugarfine_post_normalconsistency01_gaussperface1.ply|393841156|SuperSplat|
|Mesh PLY|refined_raw|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/refinement/refined_raw.ply|134985481|Blender/MeshLab/Open3D|
|Mesh PLY|refined_post|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/refinement/mesh_post/post.ply|60802353|Blender/MeshLab/Open3D|
|Textured OBJ|refined_textured|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/refinement/sugarfine_post_normalconsistency01_gaussperface1.obj|198639950|Blender/MeshLab|
|Mesh PLY|cluster_before_raw|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/cluster_pruning/cluster_before_mesh/raw.ply|89021149|Blender/MeshLab/Open3D|
|Mesh PLY|cluster_before_post|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/cluster_pruning/cluster_before_mesh/post.ply|80262309|Blender/MeshLab/Open3D|
|Mesh PLY|cluster_after_raw|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/cluster_pruning/cluster_after_mesh/raw.ply|54752707|Blender/MeshLab/Open3D|
|Mesh PLY|cluster_after_post|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/cluster_pruning/cluster_after_mesh/post.ply|49331115|Blender/MeshLab/Open3D|
|Texture Material|refined_textured|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/refinement/sugarfine_post_normalconsistency01_gaussperface1.mtl|72|Blender/MeshLab together withOBJ|
|Texture Image|refined_textured|/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/refinement/sugarfine_post_normalconsistency01_gaussperface1.png|44500805|Blender/MeshLab together withOBJ|

Sources: actual existing outputs; [official2D-SuGaR](https://github.com/prajwalcr/2d-sugar/tree/e62770ec6ab50076ebb6c0e3b4ac39f777f14d27), actual view.py.
