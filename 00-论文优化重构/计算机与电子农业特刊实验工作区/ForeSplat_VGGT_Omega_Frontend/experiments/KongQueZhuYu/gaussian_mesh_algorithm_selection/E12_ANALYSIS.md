# E12 GGGS + EO：已完成30K的实测分析

本实验训练、208视角raw/corrected评价和TSDF/Tetrahedra提取均已完成。本报告记录E12独立证据；结合所有正式终态与直接视觉复核的综合选择见[最终算法选择报告](FINAL_RECONSTRUCTION_AND_MESH_SELECTION.md)。

|模型|迭代数|基元数|Raw PSNR|Corrected PSNR|Raw真FG PSNR|Corrected真FG PSNR|Raw SSIM|Corrected SSIM|Raw LPIPS|Corrected LPIPS|背景Alpha均值|
|---|---|---|---|---|---|---|---|---|---|---|---|
|E10|30000|576396|23.9005|未完成/无统一值|19.4605|未完成/无统一值|0.8776|未完成/无统一值|0.1171|未完成/无统一值|0.3780|
|E11|30000|248629|22.2615|未完成/无统一值|17.5925|未完成/无统一值|0.8632|未完成/无统一值|0.1373|未完成/无统一值|0.9931|
|E12|30000|495883|16.9736|24.7312|11.8204|20.4835|0.8260|0.7898|0.1552|0.1231|0.9102|
|P2|30000|544074|23.9130|未完成/无统一值|19.4748|未完成/无统一值|0.8776|未完成/无统一值|0.1170|未完成/无统一值|0.3317|

EO后真前景PSNR提高，但SSIM和LPIPS并未共同优于E10；不能只按24.7312dB选照片分支。背景Alpha均值0.3780→0.9102，统一灰背景自由视角直接查看也确认大片暗云仍存在。EO颜色矩阵独立于PLY，不能用raw模型冒充corrected外观。

这里的全图光度GT来自实际03-SAM训练RGB，其Mask外背景保存为黑色；几何前端的完整背景上下文来自02-FFT，二者不能混称。EO corrected前景SSIM映射均值0.652224→0.683271，前景梯度保留0.731265→0.744480，支持已观测叶片纹理拟合有收益。全图SSIM下降同时包含黑背景颜色偏差，不等同于前景SSIM下降；全图LPIPS不是单独前景LPIPS。FG SSIM是完整图像win7的SSIM map在真实RAP内取均值，仍受局部边界窗口影响。

由相同SSIM map的实际像素数量推导，背景SSIM均值E10 raw为0.982919，E12 corrected为0.841317；背景PSNR为30.6832/30.6124dB。前景对全图SSIM的平均贡献变化为+0.010382，背景为−0.098125，两者合计−0.087743。这里明确区分区域均值与像素加权贡献，不将全图SSIM下降解释为前景纹理下降，也不将数学分解解释为曝光偏移分量的因果实验。[逐视角前景/背景分解](comparison/EO_region_decomposition/per_view.csv)和[真实Mask/原始评分来源](comparison/EO_region_decomposition/PROTOCOL.json)完整保留。

光度目标实际为`0.8 L1(EO(render),GT) + 0.2 [1-SSIM(EO(render),GT)]`，并加上与E10完全相同的depth-normal、NCC和multi-view geometry项。旧EO实现仅L1使用校正RGB，SSIM仍使用raw；本次SSIM也改为校正RGB。`use_decoupled_appearance=1`是ExposureCompensator自身的3×4逐视角模式，未增加另一个外观网络；未启用前景RGB/Mask/背景Opacity Loss或训练Pruning。参数与核心源码审计见[E12 CONTROLLED_AUDIT](E12_GGGS_EO/CONTROLLED_AUDIT.json)，一行补丁见[EO_OBJECTIVE.patch](E12_GGGS_EO/EO_OBJECTIVE.patch)。

|Mesh|IoU|Chamfer px|外部几何比例|封闭缺失像素均值|相邻面1−cos|
|---|---|---|---|---|---|
|M4_E10_GGGS_TSDF|0.8825|35.8167|0.0602|545.0529|0.0582|
|P1|0.6284|68.3148|0.2753|1258.4760|0.0975|
|E12_TSDF|0.9026|18.6442|0.0472|446.3221|0.0560|
|E12_Tetrahedra|0.7340|65.9877|0.1555|1861.1779|0.1099|

E12 TSDF的轮廓/边界代理改善来自实际Mesh重投影，不是从PSNR推断；其封闭投影缺失面积也较M4降低。Tetrahedra仍有较多外部几何和缺失。灰色Mesh直接查看显示两条TSDF路线仍存在细尺度起伏，不能称叶片变成真实平滑表面。相邻面指标受曲率、三角形尺寸和拓扑影响，应与统一空间尺度PCA和法线证据一起解释。

|路线|上部叶区代理IoU|覆盖率|外部几何比例|内部缺失像素均值|叶区内部覆盖|
|---|---|---|---|---|---|
|M4_E10_GGGS_TSDF|0.890660|0.992162|0.102973|423.211538|0.995311|
|P3_shared|0.924507|0.988177|0.065151|326.774038|0.992969|
|E12_TSDF|0.916700|0.993665|0.077947|342.740385|0.996283|
|E12_Tetrahedra|0.870330|0.988517|0.120345|1395.894231|0.991792|

叶区采用同一RAP包围框上60%高度的近似区域，没有新增逐叶分割。E12的叶区轮廓相对M4也改善；P3共享范围的IoU/外部几何更好，但覆盖略低。不能仅根据整株盆体或单一轮廓指标宣布最佳薄叶Mesh。

旧训练RGB亮度口径有135个候选；旧法线采样在双线性插值后没有再次归一化，保留的角度中位5.1935°→4.2760°、P90 27.6621°→19.3252°属于该旧口径，统一法线指标使用下述v2。相对深度误差中位0.001674→0.001553，而P90 0.012095→0.013565未改善；邻近3D基元尺度比中位70.65→100.44也变大。对应真实三角面、像素和世界坐标见[反光Mesh对应CSV](comparison/reflection_mesh_correspondence/E12_TSDF_spatial_correspondence.csv)。亮度变化可能含阴影、曝光和匹配误差，并未确认为独立镜面反光。

法线v2重新归一化采样方向，零/非有限/无有效插值支持的法线不填成90°，另列有效数量；原始CSV与深度/匹配点保持不变：

|方法|旧固定候选数|有效法线数|v2中位°|v2 P90°|
|---|---|---|---|---|
|E10|135|135|5.058289819889746|27.26697472576634|
|E11|135|135|5.907803058624268|25.882572174072294|
|E12|135|135|4.19143670631685|19.22041235556263|
|E13|135|133|16.722794560034657|36.152868752452314|
|E14_30K|135|135|2.790146827697754|12.295926094055186|
|E14_refined_aux|135|129|14.073890686035156|37.10108947753906|
|P2|135|135|5.058601180045869|27.26697472576634|

实际02-FFT原图核验保留134个亮度候选，旧训练RGB候选中1个仅因Mask边界亮度变化超过阈值；旧集合没有删除。[原始/训练RGB与同像素审计](comparison/reflection_analysis/ORIGINAL_RGB_QUERY_VALIDATION.csv)和[原图候选几何指标](comparison/reflection_analysis/ORIGINAL_RGB_CANDIDATE_GEOMETRY.csv)分别保留。

真实30K Gaussian PLY：`/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E12_GGGS_EO/checkpoints/point_cloud/iteration_30000/point_cloud.ply`。

真实Checkpoint：`/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E12_GGGS_EO/checkpoints/chkpnt30000.pth`。

曝光参数：`/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E12_GGGS_EO/exposure_params/exposure_matrices.npz`。

[全部Mesh实际路径](artifact_index/MESH_FILE_INDEX.md) · [208逐视角配对](comparison/E10_P2_E12_paired_changes.csv) · [统一真实RGB及局部图](comparison/visualizations/gaussian/) · [有限直接视觉查看记录](comparison/visualizations/VISUAL_REVIEW.json)。

来源：本地完整训练、实际模型、208浮点渲染、RAP Mask与Mesh raycast；[固定官方GGGS源码](https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids/tree/23b418c2f0f19b30b251acad6fe13445dae968c7)。博查搜索HTTP403，当前实验数据来自真实文件，不以训练数据填充。

同一208源视角的描述性配对变化（对照减基线）：

|对照路线|实际阶段|指标|均值变化|差值P10|差值中位|差值P90|方向改善视角比例|
|---|---|---|---|---|---|---|---|
|E12|corrected|fg_mask_psnr|1.023050|0.381385|0.811269|1.997730|0.995192|
|E12|corrected|ssim|-0.087743|-0.206583|-0.081424|0.005720|0.187500|
|E12|corrected|lpips_alex|0.006008|-0.001542|0.006705|0.016964|0.134615|
|E12|corrected|background_alpha_mean|0.532152|0.304757|0.523786|0.768586|0.000000|
|E12|raw|fg_mask_psnr|-7.640037|-10.771748|-7.676747|-4.559931|0.000000|
|E12|raw|ssim|-0.051594|-0.085321|-0.050211|-0.021861|0.000000|
|E12|raw|lpips_alex|0.038163|0.017821|0.037615|0.063008|0.000000|
|E12|raw|background_alpha_mean|0.532152|0.304757|0.523786|0.768586|0.000000|

视角来自同株、同序列、Seed0；分位数和改善比例描述局部变化，不是独立植株/重复训练的显著性。孔洞减少可能伴随过填充，不能单独按“改善比例”选算法。


同一208源视角的描述性配对变化（对照减基线）：

|对照路线|实际阶段|指标|均值变化|差值P10|差值中位|差值P90|方向改善视角比例|
|---|---|---|---|---|---|---|---|
|E12_TSDF|post|silhouette_iou|0.020079|-0.002427|0.019131|0.046273|0.812500|
|E12_TSDF|post|hole_area_px|-98.730769|-427.900000|-51.500000|248.000000|0.663462|
|E12_Tetra|post|silhouette_iou|-0.148510|-0.247411|-0.137430|-0.055909|0.019231|
|E12_Tetra|post|hole_area_px|1316.125000|-386.400000|1005.500000|4247.800000|0.149038|

视角来自同株、同序列、Seed0；分位数和改善比例描述局部变化，不是独立植株/重复训练的显著性。孔洞减少可能伴随过填充，不能单独按“改善比例”选算法。

