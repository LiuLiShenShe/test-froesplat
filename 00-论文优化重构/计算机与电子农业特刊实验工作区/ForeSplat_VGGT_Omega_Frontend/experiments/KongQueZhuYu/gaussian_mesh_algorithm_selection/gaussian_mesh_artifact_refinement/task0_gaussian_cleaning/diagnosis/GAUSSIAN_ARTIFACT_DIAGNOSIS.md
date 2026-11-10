# GGGS 伪影诊断（真实模型与渲染证据）

结论：原P2未完成清理。中心投影/小范围尺度保护和KDTree邻近前景保护，使少量前景支持的背景大体元保留；前层去除后还会露出原先被遮挡的黑层。故仅一次原始模型贡献评分也不足。

原P2保留544074/576396，保护517673，异常尖刺仅删除5。旧保护点中8312只有≤1视角前景中心支持，42293有≥15视角背景中心支持。保护是候选关系，不是真实表面认证。

新证据以原生GGGS color Jacobian统计每Gaussian的 T×alpha，分别累加Mask内、边缘外背景、黑云ROI。模型参数detach，无优化或训练；各视角贡献总和与原生Alpha独立校验（相对误差低于0.03%）。完整208训练视角加8历史自由相机。高斯ID始终指原E10行号。

投影椭圆几何诊断、filtered scale、opacity、旋转、SH字段、实际raster radius/visible contributions均保留。`projected_ellipse_diagnostic.npz`为协方差一阶投影，不复现screen clamp/low-pass，不拿它代替真实Rasterizer贡献。覆盖大小本身不等同废点。

黑云不是只由5个尖刺组成：原E10测得黑云权重50%由2558个ID贡献，90%由15152个，99%由28835个。top250占21.60%，包括455628、569226、317467、575871等；完整ID/XYZ在dominant_cloud_gaussian_ids.json。P2保留的16510个 harm>.95 候选承载94.98%的原模型测得剩余黑权重，同时只承载0.5957%的前景贡献。后续需要重新归因可见层。

六个固定定位区和来源相机/ID/AABB在artifact_regions.json；95%贡献模型对应*_contributors.ply。叶片中心和盆边ROI同时含有效结构，未按ROI整片删除。每个Gaussian的原生所有字段及多视角贡献在gaussian_artifact_evidence.csv（576396行）；最近VGGT前景PointMap、GGGS可见采样表面、P3距离和k8密度在geometric_auxiliary_evidence.csv。VGGT只是辅助参考，不作为叶片真值。

实际3D残余贡献重新归因后，4011个Gaussian的黑ROI权重>.05；96.27%残余黑权重处于 harm≤.95 的混合贡献点上。示例ID186547、359655、406752、382988、470521、491751，完整列表及每视角关系在 best_cleaned/remaining_artifact_dependencies.csv。继续整点删除会切掉已测前景贡献，不能保证消掉伪影同时保住叶缘。

盆沿变亮回归已定位到3135、290315、316725、3627、124520等真实ID，六视角的95%贡献AABB、像素数量和原始参数在POT_BRIGHT_REGRESSION_DIAGNOSIS.json及pot_bright_artifact_dependencies.csv。source37变化区RGB MSE从0.0639升到0.1353，source38从0.0540升到0.1742；这是清理露出错误颜色，不能写作无副作用。

实际SH方向颜色加真实alpha贡献：8自由视角E10深色体元权重3333370.53→Best54136.36（98.38%下降），深色贡献Gaussian45729→5884；RGB阈值为Rec709 luminance<.12，所有视角并集统计，不是唯一废点认证。

`SPATIAL_COMPONENT_METRICS.json`另外报告原测 harm>.95 候选的半径.015连通图和AABB体积：P2有16510候选点/9个≥10点簇/37孤立中心，Best此旧候选集合为0。该0不表示真正所有漂浮点为0，新露出的混合点仍存在。

源代码：原脚本 `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/scripts/clean_gggs.py`；独立诊断/编辑脚本 `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/scripts`；所有原始文件未覆盖。
