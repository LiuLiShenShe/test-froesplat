# P1–P4：真实提取与同Checkpoint对照

各主提取和统一评价均已完成；正式模型、统一空间尺度证据与直接视觉复核共同支持的综合选择见[最终算法选择报告](FINAL_RECONSTRUCTION_AND_MESH_SELECTION.md)。

P1保留官方Marching Tetrahedra、10步Binary Search与原/post模型。Binary Search明显降低原始三角面的局部不稳定，但post IoU0.6284、外部几何0.2753弱于M4的0.8825/0.0602；来源0/53直接灰色查看仍有盆周大面积外部面。当前不支持P1优于TSDF，也不根据较少boundary edges认定薄叶无孔洞或无双层。

P2从576,396个Gaussian删除32,322个（5.61%），保留544,074个；依据208视角Mask支持、前/遮挡深度、局部邻域、尺度与拉长程度，保护叶缘，未按颜色/低Opacity单独删除。背景Alpha0.3780→0.3317，Raw真FG PSNR19.4605→19.4748。直接自由视角确认上方黑云明显减小，但标记牌射线和叶间黑块仍存在。

必须区分两个答案：黑云明显减小但未全部消失；Mesh整体仅有微小变化。P2 TSDF IoU0.8829、Chamfer34.63px相对M4的0.8825/35.82px变化很小，投影内部缺失545.05→544.86px几乎不变；P2 Tetrahedra post IoU0.6266略低于P1。不能宣称清理同时解决Mesh孔洞。

P3使用E10真实GGGS生成左右立体视图，官方DLNR权重/10次迭代/LR检查和TSDF保持；无重新3DGS训练。官方baseline4–20倍截断把深度上限设为0.755529，208输入支持只剩0.3912；补充共享上限1.569246保留0.9219支持。两条实际Mesh均保留，不能用共享范围结果替换原始失败倾向。额外DLNR先验的收益与GGGS表示收益分开。原生及补充详见[P3实际运行报告](P3_GS2Mesh/RUN_REPORT.md)。

|路线|IoU|Chamfer px|覆盖率|外部几何比例|封闭缺失像素均值|
|---|---|---|---|---|---|
|M4_E10_GGGS_TSDF|0.8825|35.8167|0.9919|0.0602|545.0529|
|M5_E11_2DGS_TSDF|0.6183|85.3810|0.9917|0.2899|400.5721|
|P1|0.6284|68.3148|0.9909|0.2753|1258.4760|
|P2_TSDF|0.8829|34.6327|0.9919|0.0600|544.8606|
|P2_Tetrahedra|0.6266|69.4349|0.9908|0.2767|1257.6106|
|P3_native|0.8325|41.7987|0.8905|0.0329|9806.8125|
|P3_shared|0.9156|22.2949|0.9881|0.0362|358.3125|
|P4_median_mask|0.6680|81.3174|0.9952|0.2297|211.1779|
|P4_median_mask_alpha_voxel075|0.6643|83.0539|0.9954|0.2333|254.9615|
|P4_median_mask_alpha|0.6680|81.3172|0.9952|0.2297|210.7500|
|P4_expected_mask_alpha|0.6183|85.3809|0.9917|0.2899|400.2163|
|P4_expected_mask|0.6183|85.3810|0.9917|0.2899|400.5721|
|P4_median_mask_alpha_voxel125|0.6717|79.5237|0.9951|0.2260|179.1010|

同一208视角叶区补充：

|路线|上部叶区代理IoU|覆盖率|外部几何比例|内部缺失像素均值|叶区内部覆盖|
|---|---|---|---|---|---|
|M1_Omega_NKSR|0.809800|0.997610|0.188740|310.524038|0.998052|
|M2_Standard_NKSR|0.882058|0.993378|0.112708|61.629808|0.996060|
|M3_E10_GGGS_NKSR|0.663772|0.995958|0.334233|54.673077|0.997904|
|M4_E10_GGGS_TSDF|0.890660|0.992162|0.102973|423.211538|0.995311|
|M5_E11_2DGS_TSDF|0.649731|0.993656|0.347450|267.865385|0.996302|
|P1|0.702123|0.989728|0.291902|895.639423|0.993086|
|P2_TSDF|0.890933|0.992158|0.102694|423.759615|0.995309|
|P2_Tetrahedra|0.705058|0.989712|0.288925|894.341346|0.993077|
|P3_native|0.899113|0.955183|0.062870|7028.254808|0.959414|
|P3_shared|0.924507|0.988177|0.065151|326.774038|0.992969|
|P4_median_mask|0.701569|0.995629|0.296143|153.120192|0.997693|
|P4_median_mask_alpha_voxel075|0.701738|0.995850|0.296111|177.769231|0.997803|
|P4_median_mask_alpha|0.701570|0.995631|0.296144|152.692308|0.997695|
|P4_expected_mask_alpha|0.649731|0.993657|0.347450|267.500000|0.996303|
|P4_expected_mask|0.649731|0.993656|0.347450|267.865385|0.996302|
|P4_median_mask_alpha_voxel125|0.702313|0.995549|0.295349|126.346154|0.997661|
|E12_TSDF|0.916700|0.993665|0.077947|342.740385|0.996283|
|E12_Tetrahedra|0.870330|0.988517|0.120345|1395.894231|0.991792|
|E13_QGS|0.877119|0.992458|0.116923|386.495192|0.995744|
|E14_cluster_before|0.536784|0.986168|0.458540|1067.894231|0.987337|
|E14_cluster_after|0.640887|0.937661|0.323716|895.610577|0.939712|
|E14_initial|0.537444|0.993075|0.460878|739.697115|0.994163|
|E14_refined|0.537824|0.993144|0.460521|695.889423|0.994210|

叶区是RAP包围框上60%高度的代理，不是逐叶语义真值。内部缺失连通区域先在完整RAP上识别，再按区域统计，切分线不会生成额外“孔洞”。P3共享范围上部IoU0.9245、内部覆盖0.9930；E12 TSDF为0.9167/0.9963，说明P3轮廓更紧同时存在覆盖代价。

[208可见叶区的统一空间尺度起伏](comparison/leaf_visible_surface/ALL_REGION_SURFACE_METRICS.csv)与[原图/训练RGB/法线采样口径审计](comparison/SURFACE_AND_REFLECTION_AUDIT.md)已补充。固定source131/157灰色窗口中P3的局部面较平顺，但不能代表全表面：至少2源ID组支持的D/64同向PCA起伏中位较低，P90仍高于M4。M4/P2在这些窗口很接近，E12仍有脊状起伏；这不认证其为真实叶脉或错误反光凹凸。

真实同射线薄表面补充：24,375条固定RGB匹配源射线来自69个源相机，源/目标并集195张照片；不是195个源视角，也不替代主208视角评价。

|路线|查询命中率|前方4voxel层数均值|命中中近层≥3比例|近同向配对比例|
|---|---|---|---|---|
|M4_E10_GGGS_TSDF|0.998277|1.032369|0.003822|0.001932|
|P1|0.995692|1.656492|0.133333|0.003708|
|P2_TSDF|0.998277|1.032205|0.003740|0.001890|
|P2_Tetrahedra|0.995692|1.660349|0.134116|0.004285|
|P3_shared|0.997456|1.044021|0.007116|0.002756|
|P4_median_mask_alpha|0.999262|1.073436|0.016176|0.004886|
|E12_TSDF|0.998892|1.032205|0.004230|0.002793|
|E12_Tetrahedra|0.994872|1.678646|0.144619|0.003918|

同一查询同时具有相反平行首面配对时的法线投影间距：

|对照路线|共同查询数|共同有效比例|M4间距voxel中位|对照间距voxel中位|逐射线差值中位|
|---|---|---|---|---|---|
|P1|4997|0.205005|4.996764|2.402641|-2.481200|
|P2_TSDF|9814|0.402626|5.080948|5.080855|0.000000|
|P2_Tetrahedra|4985|0.204513|4.996764|2.394486|-2.488238|
|P3_shared|7436|0.305067|4.981762|5.371226|0.255453|
|P4_median_mask_alpha|4029|0.165292|4.976316|5.731473|0.632611|
|E12_TSDF|7034|0.288574|4.945712|4.966596|0.030314|
|E12_Tetrahedra|4880|0.200205|4.994638|2.162765|-2.766691|
|E13_QGS|2870|0.117744|4.809964|5.043519|0.214851|
|E14_initial|1098|0.045046|5.421486|4.795176|-0.690173|
|E14_refined|1068|0.043815|5.370273|5.084922|-0.530210|

P1/P2 Tetrahedra局部近层较多，M4/P2 TSDF的相对前后间距也可能更大；不能把近层少单独等同于薄叶更准确。原图source0/79/157与实际有符号面交点保留为目的选取的局部示例，多层可来自卷曲和遮挡，不认证错误双层；首面间距也不是物理叶厚。[全部实际交点](comparison/same_ray_mesh_layers/PROTOCOL.json)、[共同查询配对CSV](comparison/same_ray_mesh_layers/MATCHED_FIRST_PAIR_SPACING.csv)和[直接实际射线图](comparison/visualizations/same_ray_layers/PROTOCOL.json)完整记录。

实际source157叶尖上方有1,441个距RAP>3px的额外表面投影像素，对应665面, 164面的两个原生组件，均与最大Mesh组件分离。原始02-FFT图保留其背景上下文；不是用置黑训练GT证明原图无几何。[世界坐标/实际面ID](comparison/P3_leaf_tip_spatial/source157_extra_surface_world_triangles.csv)及[原始背景窗口](comparison/P3_leaf_tip_spatial/source157_original_RGB_context.png)保留。这是观察后选定的局部诊断，不能当成全视角无偏发生率；原Mesh未修改。


P4 Expected换Median确有收益（基础post IoU0.6183→0.6680），同一Median的Alpha>=0.5几乎无变化，符合E11背景Alpha接近1，Alpha不能充分筛背景的事实。0.75/1/1.25倍voxel/truncation中较粗1.25倍代理更好，但须检查叶尖细节，不能只按较低缺失面积选择过度平滑。

|方案|区域|源深度有效率|有效深度但raw缺失|无效深度且raw缺失|仅post删除可见覆盖|
|---|---|---|---|---|---|
|expected_mask|foreground|0.999997|0.007375|0.000003|0.000921|
|expected_mask|leaf_interior3px_upper60pct|1.000000|0.003294|0.000000|0.000404|
|median_mask|foreground|0.999997|0.004313|0.000003|0.000476|
|median_mask|leaf_interior3px_upper60pct|1.000000|0.002033|0.000000|0.000273|
|expected_mask_alpha|foreground|0.999870|0.007257|0.000120|0.000921|
|expected_mask_alpha|leaf_interior3px_upper60pct|0.999920|0.003220|0.000074|0.000404|
|median_mask_alpha|foreground|0.999870|0.004198|0.000116|0.000475|
|median_mask_alpha|leaf_interior3px_upper60pct|0.999920|0.001963|0.000069|0.000273|
|median_mask_alpha_voxel075|foreground|0.999870|0.004000|0.000119|0.000442|
|median_mask_alpha_voxel075|leaf_interior3px_upper60pct|0.999920|0.001870|0.000073|0.000253|
|median_mask_alpha_voxel125|foreground|0.999870|0.004375|0.000111|0.000453|
|median_mask_alpha_voxel125|leaf_interior3px_upper60pct|0.999920|0.002024|0.000063|0.000251|

有效源深度仍可在融合/重投影后缺失：这里不能单独归咎于TSDF，因为有限正深度也可能几何错误。raw→post可见覆盖变化隔离组件删除造成的轮廓缺失，但仍可能删除前方表面而由更远几何保留轮廓；因此不能仅以覆盖几乎不变认定post几何影响很小。本次post不做平滑。上60%RAP内部是叶区近似，未声称为逐叶语义分割。


source53的固定叶片纹理窗口中有26,929个真实RAP前景像素满足GT亮度>0.15而E11亮度<0.01；其Alpha中位数0.999856，Expected/Median源深度均为有限正值。四份Expected/Median raw/post对这些黑像素覆盖均约99.9703%，所以这里主要是错误暗色/遮挡表面，不能把黑色区域当成孔洞。该区域Median与E10表面深度的相对差异中位41.12%，这是模型间不一致而非扫描误差。Expected/Median post分别有11.53%/25.69%共同命中像素比raw更远1%以上，说明组件删除会改变遮挡深度，即使轮廓近乎不变。见[同像素世界坐标CSV](comparison/P4_local_failure/source53_same_pixel_spatial.csv)和[24张原生RGB与raw/post同窗口图](comparison/visualizations/P4_stage_visuals/PROTOCOL.json)。

全部208视角的前方组件删除审计：

|方案|区域|轮廓仍在且post深度更远一voxel|post更远1%|仅轮廓新增缺失|
|---|---|---|---|---|
|expected_mask|foreground|0.009172|0.008732|0.000921|
|expected_mask|leaf_interior3px_upper60pct|0.009245|0.008782|0.000404|
|median_mask|foreground|0.006723|0.005967|0.000476|
|median_mask|leaf_interior3px_upper60pct|0.007447|0.006623|0.000273|
|expected_mask_alpha|foreground|0.009170|0.008730|0.000921|
|expected_mask_alpha|leaf_interior3px_upper60pct|0.009244|0.008781|0.000404|
|median_mask_alpha|foreground|0.006724|0.005967|0.000475|
|median_mask_alpha|leaf_interior3px_upper60pct|0.007447|0.006624|0.000273|
|median_mask_alpha_voxel075|foreground|0.007022|0.006072|0.000442|
|median_mask_alpha_voxel075|leaf_interior3px_upper60pct|0.007813|0.006765|0.000253|
|median_mask_alpha_voxel125|foreground|0.006575|0.005997|0.000453|
|median_mask_alpha_voxel125|leaf_interior3px_upper60pct|0.007326|0.006685|0.000251|

深度变远说明删除了原遮挡三角面，不说明删除正确；[原始逐视角证据](comparison/P4_post_depth/all208_per_view.csv)保留float32 raycast容差诊断。

E11 source53局部原生RGB也有黑色楔形/条带，说明该视角的问题不能全部归因于Mesh后处理。必须同时看RGB、Alpha、Expected/Median Depth及raw/post，不能以某一整株纹理截图认证原版2DGS更适合薄叶。

所有边界边数是拓扑边界；开放叶缘和RAP中的空白区域不计为本轮“封闭缺失投影”代理，空白区域尚未独立认证为真实物理孔洞。该代理是至少9像素且不接触RAP外缘3px带的缺失连通区域，单位为图像像素，绝不是物理3D孔洞面积。整株表外部几何字段沿用旧定义：Mask外被Mesh覆盖的像素/全部Mask外像素；叶区补充使用Mask外Mesh像素/该区域全部Mesh像素，两种分母明确区分。

[全部实际Mesh](artifact_index/MESH_FILE_INDEX.md) · [原始/清理/删除Gaussian](artifact_index/GAUSSIAN_PLY_INDEX.md) · [同像素反光Mesh对应](comparison/reflection_mesh_correspondence/) · [直接视觉查看范围](comparison/visualizations/VISUAL_REVIEW.json)。

来源：真实本地文件和统一208视角评价；[GGGS官方](https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids)、[GS2Mesh固定源码](https://github.com/yanivw12/gs2mesh/tree/560f3e8a2349aaad24fe9444ef1ca2770aeec5ab)、[2DGS官方](https://github.com/hbb1/2d-gaussian-splatting)。博查HTTP403不可用。

同一208源视角的描述性配对变化（对照减基线）：

|对照路线|实际阶段|指标|均值变化|差值P10|差值中位|差值P90|方向改善视角比例|
|---|---|---|---|---|---|---|---|
|P1|post|silhouette_iou|-0.254112|-0.325743|-0.267269|-0.169628|0.000000|
|P1|post|hole_area_px|713.423077|-612.500000|566.000000|2724.200000|0.278846|
|P2_TSDF|post|silhouette_iou|0.000319|-0.000270|0.000303|0.001048|0.745192|
|P2_TSDF|post|hole_area_px|-0.192308|-1.300000|0.000000|1.000000|0.134615|
|P2_Tetra|post|silhouette_iou|-0.255954|-0.331003|-0.269516|-0.169105|0.000000|
|P2_Tetra|post|hole_area_px|712.557692|-613.200000|564.000000|2690.800000|0.274038|
|P3_native|post|silhouette_iou|-0.050091|-0.285150|0.030905|0.052912|0.610577|
|P3_native|post|hole_area_px|9261.759615|1755.900000|7418.000000|18288.900000|0.000000|
|P3_shared|post|silhouette_iou|0.033010|-0.007948|0.044054|0.061993|0.774038|
|P3_shared|post|hole_area_px|-186.740385|-1050.900000|-53.000000|131.300000|0.644231|
|P4_median_mask_alpha|post|silhouette_iou|-0.214532|-0.289914|-0.214776|-0.138828|0.000000|
|P4_median_mask_alpha|post|hole_area_px|-334.302885|-1195.800000|-109.500000|109.600000|0.721154|

视角来自同株、同序列、Seed0；分位数和改善比例描述局部变化，不是独立植株/重复训练的显著性。孔洞减少可能伴随过填充，不能单独按“改善比例”选算法。

