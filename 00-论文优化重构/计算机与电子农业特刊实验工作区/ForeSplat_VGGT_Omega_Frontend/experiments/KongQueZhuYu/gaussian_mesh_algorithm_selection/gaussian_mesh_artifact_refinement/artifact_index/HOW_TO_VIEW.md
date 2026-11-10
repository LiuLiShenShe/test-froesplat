# 最终查看文件

SuperSplat 打开：`/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/task0_gaussian_cleaning/best_cleaned/cleaned_gaussians_supersplat.ply`（510248 Gaussian，filter_3D 已烘焙到标准 scale/opacity）。

Blender / MeshLab / Open3D 打开：`/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/task2_cleaned_gs2mesh/G2_BestCleaned_to_GS2Mesh/local_repair/G2_targeted_cleaned_mesh.ply`（1492364 triangles，G2 官方融合后另行移除17449个非目标标记三角面）。

原生 GGGS Renderer 和今后 GS2Mesh 输入使用：`/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/task0_gaussian_cleaning/best_cleaned/cleaned_gaussians.ply`。不要把未烘焙的原生 filter_3D 文件与忽略该字段的查看器画面混为一谈。

所有模型保持修正后的标准 VGGT / E10 OpenCV 世界坐标，−Y 近似为植株向上，尺度未作物理标定。查看器需要向上轴时可仅调整对象显示旋转：Y-up 软件绕X轴180°；Blender Z-up 可绕X轴−90°。不要将查看旋转写回分析模型或沿用旋转后的 Triangle ID。

本次使用原生 GGGS Rasterizer 和 Open3D Raycast 生成真实 RGB/Alpha/Depth/Normal/三角面ID，并由当前模型直接读取比较图片。未启动交互式 SuperSplat、Blender 或 MeshLab GUI。Gaussian PLY 原生字段已加载验证；标准兼容版本与原生有效尺度、不透明度及四视角渲染核对，RGB/Alpha最大差小于1/255（非位级相同）。Mesh 使用 Open3D 实际读取和逐像素 raycast。GUI 自由拖动仍应视为用户最后的视觉验收。

Plant-only 试验文件：`/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/task0_gaussian_cleaning/best_cleaned/plant_only_geometry_v2.ply`。它已去除花盆主体，但盆沿/基部混合Gaussian和微小标记残留未消失，根与叶基完整性未认证；不用于表型测量。v1裁切保留作失败对照。

同姿态逐轮图：`/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/comparisons/before_after_images/free_free_0053_-15_all_rounds.png`；每轮完整8个冻结自由视角位于该版本 preview/free。Mesh 8自由视角：`/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/comparisons/mesh_comparison`。局部灰色/法线/线框/颜色/组件图和精确三角面ID位于各版本 visualizations；ID在 raycast.npz。

各轮标准SuperSplat导出：round_01/02/03与additional_rounds/*/gaussians_supersplat.ply；原E10/P2的等效查看导出在task0_gaussian_cleaning/diagnosis/E10_gaussians_supersplat.ply和P2_gaussians_supersplat.ply；最终废点标准导出为best_cleaned/removed_gaussians_supersplat.ply。所有绝对路径和SHA见ALL_GAUSSIAN_FILES.md，逐版本公式、输入/输出SHA见task0_gaussian_cleaning/VIEWER_BAKING_MANIFEST.json。
