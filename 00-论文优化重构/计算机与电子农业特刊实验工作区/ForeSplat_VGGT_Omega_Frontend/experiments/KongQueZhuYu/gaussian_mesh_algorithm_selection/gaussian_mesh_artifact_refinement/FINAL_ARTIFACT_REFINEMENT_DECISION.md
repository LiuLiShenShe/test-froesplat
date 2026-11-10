# ForeSplat伪影处理最终判定

**当前模型尚未达到完整、干净三维资产使用目标。** 本轮真实完成多轮Gaussian处理、P3已定位组件编辑、G1/G2完整208视角官方GS2Mesh与统一评价。质量标记逐项如下，Improved不等于Resolved。

| 验收问题 | 结论 | 实际证据 |
|---|---|---|
| 1 原P2为何不干净 | 已诊断 | 中心/邻域保护忽略大footprint背景危害，前层删除又露出黑层 |
| 2 新清理删了哪些 | 可追溯 | 相对E10删66148，含P2已有32322、新背景30234、语义标记3592；全部原ID和原字段保留 |
| 3 大面积黑云 | Improved | 固定8自由视角2682781→68362px，减少97.45%；仍有小云/黑条 |
| 4 叶尖/交叉尖刺 | Improved | 大块周边尖刺显著减，混合叶缘和交叠黑层仍残留 |
| 5 叶片/照片副作用 | Improved，局部回归Unresolved | Plant+Pot FG19.496→19.539；叶片主体保留，但盆沿错误亮纹理和局部深度明显改变 |
| 6 source157漂浮环 | Resolved（指定范围） | P3删除829原三角面；G2直接融合环已消失，叶尖仍在 |
| 7 叶内错误连接 | Unresolved | 54实际ID候选部分覆盖真叶柄/卷叶，未盲目切除 |
| 8 P2→GS2Mesh | 小幅混合收益 | 缺面358.31→326.98；其他轮廓指标略退，环仍在 |
| 9 Best→GS2Mesh | Improved | IoU/BF/背景几何改善；原G2缺面397.78更差、标记又生成；另存局部修复版 |
| 10 当前最佳Gaussian | Improved查看候选 | 510248点原生R5+标记清除；SuperSplat导出与原生有效参数校验 |
| 11 当前最佳Mesh | Improved查看候选 | G2_targeted，1492364面；轮廓/背景最佳，局部面片错误仍在 |
| 12 打开哪两个文件 | 见下方 | 绝对路径及SHA在artifact_index/ALL_ARTIFACTS.csv |

SuperSplat：`/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/task0_gaussian_cleaning/best_cleaned/cleaned_gaussians_supersplat.ply`

Mesh：`/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/task2_cleaned_gs2mesh/G2_BestCleaned_to_GS2Mesh/local_repair/G2_targeted_cleaned_mesh.ply`

原生Gaussian输入：`/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/task0_gaussian_cleaning/best_cleaned/cleaned_gaussians.ply`；当前直接P3定点修复：`/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/task1_p3_local_repair/mesh_versions/P3_best_cleaned_mesh.ply`。

| model | stage | iou | boundary_f1 | boundary_chamfer_px | fg_coverage | bg_geometry | hole_px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M4_legacy | post | 0.882547 | 0.122550 | 35.816651 | 0.991885 | 0.060210 | 545.052885 |
| G0 | post | 0.915556 | 0.151903 | 22.294921 | 0.988083 | 0.036240 | 358.312500 |
| G1 | post | 0.914219 | 0.150062 | 22.937884 | 0.988187 | 0.037201 | 326.975962 |
| G2 | post | 0.918008 | 0.157959 | 19.565034 | 0.986937 | 0.033728 | 397.783654 |
| P3_local | v1 | 0.916275 | 0.153032 | 22.226719 | 0.988079 | 0.035925 | 358.740385 |
| P3_local | v2 | 0.913268 | 0.148717 | 22.928656 | 0.982363 | 0.034821 | 358.620192 |
| G2_targeted | post | 0.921293 | 0.163733 | 19.207052 | 0.982709 | 0.030164 | 347.894231 |

上述208指标全部沿用历史RAP，仍包含原标记目标；G2_targeted语义去标记导致FG coverage从.986937降至.982709。孔洞统计是>=9px、远离Mask边缘的封闭缺失投影，单位px，不是扫描真值或物理孔洞面积。标记删除后的孔洞减少不能归因叶片补面。

剩余根因可追溯：4011个残余黑贡献Gaussian，96.27%权重在前景/背景混合点；误删会牺牲已测前景。source37/38盆沿黑遮挡去除后由3135/290315/316725等点贡献错误亮纹理。source131固定ROI缺面2368，其中2193有有效source Stereo，说明仅清Gaussian仍不能修复多视角冲突或TSDF局部融合。还有近盆基部独立碎片和叶片表面不合理起伏。

后续若继续，需要对已定位混合Gaussian作局部受约束处理，或在已有GGGS路线训练中约束背景贡献/尺度/表面一致性；叶内部连接及孔洞需基于局部多视角一致性独立修复。本轮未执行任何重新训练、VGGT或RAP推理；不以可能的未来效果充当本轮验收。

真实查看：当前模型直接读取原生渲染与Open3D灰色/法线/线框/颜色图片；已读路径和逐图观察在comparisons/VISUAL_REVIEW.json。没有声称交互式SuperSplat/Blender GUI完成；查看器兼容通过PLY字段、原生渲染及Open3D实际读取验证。已实际加载55份唯一模型（64个路径/别名），所有Gaussian通过原GGGS GaussianModel读取，Mesh通过Open3D读取；见DELIVERY_VALIDATION.json。

完整绝对模型目录：artifact_index/ALL_GAUSSIAN_FILES.md、ALL_MESH_FILES.md；所有实验比较：comparisons/gaussian_cleaning_metrics.csv、mesh_artifact_metrics.csv、stereo_depth_comparison.csv（416行）、local_mesh_roi_metrics.csv。删除原因和原Triangle/Gaussian ID均在各版本目录。详细分析分task0、task1、task2报告。

资料来源：[真实ForeSplat项目](https://github.com/LiuLiShenShe/test-froesplat)、[固定官方GS2Mesh](https://github.com/yanivw12/gs2mesh/tree/560f3e8a2349aaad24fe9444ef1ca2770aeec5ab)与本地真实文件/实验。Bocha检索返回HTTP403额度不足，未据此编造搜索结果；官方源文件直接获取，ctx7对GS2Mesh无索引，uv官方文档已取。质量结论均来源本地测量和直接看图。
