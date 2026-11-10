# ForeSplat三维重建与Mesh最终选择

本株KongQueZhuYu的综合照片输出选择 **E10原生GGGS 30K → P2保守Gaussian Cleaning**；综合薄叶Mesh输出选择 **原始E10 → 官方GS2Mesh/DLNR → 共享VGGT深度范围TSDF（P3_shared）**。两条分支共用已训练E10，但几何分支实际使用原始E10，**没有把未执行的“P2→GS2Mesh”列为结果**。

E12 EO-corrected是本轮受控新增模型中已观测前景光度拟合最强的方案，适合需要逐视角曝光补偿的照片输出；它的raw PLY、背景Alpha和全图SSIM/LPIPS存在代价，因此不作为通用Gaussian默认。E12 TSDF是保留部分尖锐叶尖/较低整体粗糙度的备选Mesh。没有一个方案同时赢得所有纹理、叶缘、孔洞和起伏指标。

## 三套推荐与主要测量

|用途|实测方案|PSNR dB↑|RAP内PSNR dB↑|全图SSIM↑|全图LPIPS↓|背景Alpha↓|
|---|---|---|---|---|---|---|
|A 综合照片重建|E10 GGGS→P2 Cleaned|23.9130|19.4748|0.8776|0.1170|0.3317|
|A 曝光感知备选|E12 GGGS+EO corrected|24.7312|20.4835|0.7898|0.1231|0.9102|

|方案|IoU↑|边界F1↑|Chamfer px↓|前景覆盖↑|背景几何↓|封闭缺失投影px↓|相邻法线粗糙度↓|
|---|---|---|---|---|---|---|---|
|B 综合几何表面|0.9156|0.1519|22.2949|0.9881|0.0362|358.3125|0.0931|
|B 叶尖/粗糙度备选 E12 TSDF|0.9026|0.1368|18.6442|0.9936|0.0472|446.3221|0.0560|

**C 综合默认ForeSplat流水线：** 标准VGGT完整背景几何推理 → 已有RAP-SAM3在Point Map后过滤 → 修正K/W2C/投影/像素中心 → GGGS官方30K、Seed0；照片分支输出P2 Cleaned原生PLY和RGB，几何分支使用原始E10原生立体渲染、官方DLNR和P3共享深度范围TSDF。保留原始E10作为可追溯输入。低成本/不引入立体先验的几何备选为P2 TSDF；逐视角曝光需求才选择E12独立支路。上述默认仍有黑云、标记物和局部漂浮环，不声称已获得无伪影植物。

选择P2的理由是它几乎保留E10纹理并降低背景Alpha，SSIM/LPIPS也基本不变，而非32,322次删除就能解决所有问题。选择P3_shared的理由是轮廓、边界F1、背景几何与主要叶区连续性共同改善；它并非全部方案中单项孔洞代理最低，也不是所有叶尖/曲率都最佳。

## 比较条件和结论边界

本轮E12/E13/E14都实际完成原生30K，E14另完成SuGaR15K；P1–P4没有重训E10/E11。只使用首株固定208帧，标准VGGT前端与RAP Mask不重跑，不扩展19株。物理GPU1 RTX A6000 48GB、uv、Seed0；保留全部历史产物。官方核心表示/训练逻辑不由普通3DGS替代。

统一光度GT为实际03-SAM训练RGB与真实RAP Mask，Mask外大部分为黑色。前端几何推理使用02-FFT完整背景，反光诊断另使用02-FFT原始RGB，三者用途不同。208帧同时是训练输入，PSNR/SSIM/LPIPS表示已观测视角拟合，没有独立留出照片；自由视角只有实际渲染与几何诊断，没有自由视角照片真值。

E1–E14都用相同208图像ID和GT/Mask哈希重新整理光度评价，但历史前端、几何RGB、相机接口、训练条件保持原样，统一评价并不能使全部历史训练成为受控消融。E6缺失相机字段只从其保留FoV/P恢复到新评价元数据，原W2C/P/fullP/相机中心未改，未套用E10相机修正。Fast3R/AMB3R前端另表保留，不重新推理。

没有同株扫描。所有几何排序限于轮廓、跨视角一致性与局部表面代理，不代表毫米级三维距离精度。VGGT尺度未标定；Ω有独立gauge，不未经配准叠加。

hole_area是每视角RAP前景内至少9px、且不触及外缘3px带的封闭缺失投影面积均值；单位是图像px，不是3D物理孔洞面积。开放叶缘单独统计；原RAP空白区域排除在缺失前景中，未逐一人工认证为物理孔洞。不能把所有Boundary Edges计为孔洞。背景几何为Mask外Mesh覆盖像素/全部Mask外像素，叶区补充统计另有明确分母。低孔洞数也可能来自错误的大面片粘连。

相邻法线粗糙度是共同网格统计，仍受三角化、真实曲率和盆体影响。补充在上60%RAP叶区近似、至少两个ID分箱支持的可见表面上使用同一D/64半径和同朝向邻域PCA；这不是逐叶语义分割，也不是扫描误差。

## 十五项决策回答

1. **EO是否真正改善GGGS叶面？** 有部分可测收益：E10→E12 TSDF IoU 0.88255→0.90263，封闭缺失投影545.05→446.32px，相邻法线粗糙度0.05822→0.05597。叶区PCA中位局部RMS/D却从0.001310到0.001320，P90从0.002939到0.002928，不能称全面变平。部分固定视角叶片条带/边缘改善，大片背景Alpha与原始颜色却变差。结论是局部几何代理改善，不是PSNR增益等于真实几何增益。

2. **EO是否帮助反光几何错误？** 在原RGB核验的134个亮度变化候选（49个源视角）上，E10/E12均有134个有效原生几何查询。相对深度一致性误差中位0.001673→0.001549，而P90 0.012109→0.013670变差；跨视角法线绝对夹角中位4.987°→4.140°、P90 24.019°→16.739°改善；局部3px法线突变中位3.722°→3.160°改善。Gaussian面内形状尾部未共同改善。这些是实际像素→深度→世界点→跨视角对应，并不证明全部候选是纯镜面高光，也不能宣称已消除反光凹坑。QGS/E14各自有效数与不同原生形状定义单独报告，不用3D尺度比冒充QGS曲率。

3. **Cleaning是否消除黑云和尖刺？** 未消除。576,396→544,074，删除32,322（5.6076%）；背景Alpha 0.37804→0.33170，前景Alpha 0.992346→0.992326。多视角RAP/frustum/深度前后关系、局部邻域、尺度/异常拉长共同限制，517,673个候选受薄叶/叶缘保护；不是按颜色或低Opacity删点。自由53+15°仍有大片黑云、盆旁标记物与条状体元。这个保守结果证明部分背景改善，也证明尚未解决结构性伪影。

4. **Cleaning是否同时改善Mesh？** TSDF只微小改善：IoU 0.88255→0.88287，封闭缺失545.05→544.86px，粗糙度0.058225→0.058189；Tetrahedra IoU 0.62844→0.62659略差。反光候选深度基本相同。故黑云减少与Mesh变好分别回答，不能以清理点数替代Mesh证据。

5. **Tetrahedra是否优于现有TSDF？** 本株否。P1保留官方Marching Tetrahedra、10步Binary Search及所有raw/binary/post；post 15,334,802面，IoU0.62844，背景几何0.27529，封闭缺失1258.48px、粗糙度0.09754，均明显不及M4的联合表现。清理后Tetra也没有逆转。保留更多面数不能证明叶缘更好，背景表面与多层结构代价较大。

6. **GS2Mesh是否更连续？** P3_shared在主要叶区有实质连续性收益；整株IoU0.91556、边界F1 0.15190、背景几何0.03624。官方原native4–20预设在VGGT尺度中上界仅0.75553，前景支持约0.3912，覆盖0.89047、缺失9806.81px；共享上界1.56925将支持提高约0.9219，必须将native与shared独立展示。使用E10原生GGGS立体渲染和官方DLNR，无普通3DGS重训替代。Confidence是左右重投影残差exp(-residual/3px)，不是网络概率。共享版仍有source157漂浮环和部分叶尖变圆。

7. **2DGS为何平顺却Mesh有孔洞？** 正的Expected/Median深度不保证深度正确。E11源深度有效率约0.999997、Alpha>0.5约0.999870，仍有原生黑色楔形遮挡面。Expected raw/post覆盖0.992622/0.991701；Median raw/post为0.995685/0.995209，raw已缺失的部分说明融合与错误源深度都参与，post再增加少量轮廓缺失。post也可能移除近面并由远面维持轮廓，另存逐像素深度改变，不能只看覆盖。source53“GT亮而E11黑”的局部约99.9703%仍被raw/post Mesh覆盖，那里主要是错误暗面/遮挡，不是实体缺口。Median与稍大voxel减少投影缺失到179.10px，却仍有0.22595背景几何和0.67170 IoU，未修复错误粘连；Alpha0.5约束几乎没有改善。

8. **QGS是否比原2DGS更适合弯曲叶片？** 本次原生Mip30K照片FG PSNR18.2986高于E11 17.5925，官方top10 Mesh IoU0.89329也高于E11/M5 0.61833，但粗糙度0.13440、缺失1001.80px；固定叶缘仍多锯齿/面片，自由53+15°几乎黑屏，157-15°则可见合理叶片，不能泛化为所有自由视角失败。终态有3,456/532,735（0.6487%）非有限原生基元。QGS保留六尺度二次表面、原生渲染及官方Mip配置；该失败模式限制本配置结论，不否定全部QGS算法。它不作为本株精细曲面默认。

9. **2D-SuGaR先验、Pruning、Refinement的收益？** Metric3Dv2 giant2额外208深度/法线、逐图COLMAP轨迹中位尺度对齐、深度初始化和完整原生法线先验已记录。初始化1,272,640点，其中1,252,383源自VGGT、20,257为新增先验采样。7K原生DBSCAN 83,013→43,932，背景Alpha0.86368→0.41965，但前景Alpha0.94758→0.86477、Mesh覆盖0.97527→0.89669，清理有前景代价。30K→SuGaR15K照片FG PSNR11.61494→13.56167、LPIPS0.29546→0.22892；Mesh相邻法线粗糙度大幅降低0.10938→0.01466，但IoU仅0.48876→0.48899，背景几何0.46541→0.46517。它主要平滑了错误的大面片，未恢复正确叶片布局，低粗糙度不能单独证明收益。无单独prior-off正式消融，不能将收益唯一分配给先验、损失或表示；保留各实际阶段而非只展示精修终态。

10. **照片重建选什么？** 受控终态的综合默认是P2 Cleaned GGGS，纹理/SSIM/LPIPS和可独立查看PLY更稳。若只要求曝光感知的已观测前景拟合，E12 corrected最高：PSNR24.7312、FG20.4835；但全图SSIM0.78981和LPIPS0.12309不共同优于E10/P2。前景SSIM map均值0.652224→0.683271、梯度0.731265→0.744480确有改善；全图SSIM下降很大部分来自黑背景颜色偏差，不能误说前景SSIM也下降。E9略高的历史PSNR不能抹去其未修正接口与背景Alpha0.80742。

11. **精细三维几何选什么？** 联合轮廓/薄叶连续性选择原E10→P3_shared；更看重部分尖叶/较低整体相邻法线粗糙度时保留E12 TSDF。没有扫描证据支持其绝对距离更准确，也不将二次/二维表示名称直接等同于几何优越。

12. **哪种Mesh孔洞最少且不过度平滑？** 没有单项全胜者。M3 NKSR缺失仅85.66px但IoU0.65514、背景几何0.26398；P4粗voxel缺失179.10px但误表面仍多。因此P3_shared是约束轮廓、外部几何和叶区连续性后的推荐，而不是单项hole_area冠军。主叶区同向PCA RMS/D中位：M4 0.001310、P3 0.001254、E12 0.001320；P3的P90却0.003415差于M4 0.002939。P3整体粗糙度0.09307也高于M4 0.05822，局部漂浮/粗糙尾部必须保留展示，不能宣称所有区域平顺。

13. **叶片、叶尖、叶缘哪套最好？** 主要连续叶片和全局轮廓P3_shared更好；部分source157尖叶E12/M4更尖，P3局部环状漂浮、尖端变圆。固定8个相同叶缘窗口显示这一取舍。69源视角24,375同一查询射线上，近前面4voxel内≥3层比例M4/P3/E12为0.003815/0.007097/0.004226；P3并未消除所有多层。共同可用首个相反平行面间距配对中，P3相对M4中位差+0.25545voxel；这些可能涉及折叶/不同叶面，不能叫毫米叶片厚度。具体形态测量应使用相应局部人工复核。

14. **谁运行成本低？** 既有E10训练40.21min，P2无需重训（清理约5.39min CPU），P3新增原生立体渲染281.84s+DLNR418.09s+共享融合28.61s约12.14min、Stereo峰值分配3.19GiB。E11训练48.52min/2.62GiB；E12 70.44min/8.61GiB；QGS成功Mip30K 344.81min/3.46GiB，另归档失败base成本约342min，不混进成功模型训练分钟。E14先验12.92min/7.52GiB、原生训练80.20min/5.19GiB分配峰值（保留峰值46.72GiB），精修加恢复导出/拓扑适配71.18min；精修训练峰值未独立留存，4.55GiB仅恢复导出峰值，主表不冒称训练峰值。缓存输入融合分钟与包含全部渲染的历史Mesh分钟不能直接相减排名。

15. **最终默认路线？** 采用上述C双分支：E10/P2负责照片Gaussian，原E10/P3_shared负责表面Mesh；保留P2 TSDF作为低成本稳定备选，E12作为曝光/叶尖专项支路。QGS与2D-SuGaR本次正式模型全部交付并留作研究对照，不纳入本株默认。默认选择服务于本株已测条件与植物表型代理，不能推广到全部19株或扫描精度。

## 全历史Gaussian演进

|实验/阶段|前端/几何RGB/相机|表示|迭代|基元数|Raw PSNR|Corrected PSNR|Raw FG PSNR|Raw SSIM|Raw LPIPS|背景Alpha|
|---|---|---|---|---|---|---|---|---|---|---|
|E1|VGGT-Omega / foreground-only historical / historical centered interface|GGGS|30000|600458|17.8947|不适用|13.3128|0.7858|0.2406|0.8817|
|E2|VGGT-Omega / foreground-only historical / historical centered interface|GGGS|30000|607422|17.3925|18.2335|12.5949|0.7844|0.2492|0.9469|
|E3|VGGT-Omega / foreground-only historical / historical centered interface|GGGS|30000|1037020|15.1640|不适用|16.4045|0.6509|0.3795|0.1272|
|E4|VGGT-Omega / foreground-only historical / historical centered interface|GGGS|30000|814255|14.2161|14.0374|9.4051|0.6488|0.3547|0.0859|
|E5|standard VGGT / foreground-only historical / historical centered interface|GGGS|30000|5153165|13.1057|不适用|8.1050|0.7193|0.3546|0.9242|
|E6|standard VGGT / foreground-only historical / historical centered interface|2DGS|30000|119836|13.2018|不适用|8.1076|0.7134|0.3766|0.9884|
|E7|VGGT-Omega / full background context / historical centered interface|GGGS|30000|517932|22.4935|不适用|17.9227|0.8611|0.1289|0.7557|
|E8|VGGT-Omega / full background context / corrected|GGGS|30000|501507|23.2159|不适用|18.7214|0.8651|0.1244|0.2942|
|E9|standard VGGT / full background context / historical centered interface|GGGS|30000|541994|23.9683|不适用|19.5567|0.8780|0.1163|0.8074|
|E10|standard VGGT / full background context / corrected|GGGS|30000|576396|23.9005|不适用|19.4605|0.8776|0.1171|0.3780|
|E11|standard VGGT / full background context / corrected|2DGS|30000|248629|22.2615|不适用|17.5925|0.8632|0.1373|0.9931|
|E12|standard VGGT / full background context / corrected|GGGS|30000|495883|16.9736|24.7312|11.8204|0.8260|0.1552|0.9102|
|E13|standard VGGT / full background context / corrected|QGS|30000|532735|22.8956|不适用|18.2986|0.8586|0.1344|0.9993|
|E14|standard VGGT / full background context / corrected|2D-SuGaR|30000|513898|16.6208|不适用|11.6149|0.7635|0.2955|0.9988|
|P2|standard VGGT / full background context / corrected|GGGS|30000|544074|23.9130|不适用|19.4748|0.8776|0.1170|0.3317|
|E14_refined|standard VGGT / full background context / corrected|SuGaR mesh-bound refined3D Gaussian|15000|1588063|18.4298|不适用|13.5617|0.7925|0.2289|0.3474|
|E14_initial|standard VGGT / full background context / corrected|2D-SuGaR native2DGS|1|1272640|15.0096|不适用|14.1500|0.4167|0.4238|0.1972|
|E14_cluster_before|standard VGGT / full background context / corrected|2D-SuGaR native2DGS|7000|83013|17.4531|不适用|12.6560|0.7671|0.2811|0.8637|
|E14_cluster_after|standard VGGT / full background context / corrected|2D-SuGaR native2DGS|7000|43932|17.5748|不适用|12.7933|0.7679|0.2739|0.4196|

E12 corrected FG/SSIM/LPIPS另见首表及CSV独立字段；E2/E4旧EO数值也分列。E14_initial对应真实初始化保存iteration_1，不代表正式只训练1步。7K before/after不是两个独立30K。详细旧值/统一重算、Fast3R/AMB3R及其共同帧前端表见[完整历史报告](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/FULL_HISTORY_COMPARISON.md>)、[旧指标](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/comparison/LEGACY_GAUSSIAN_METRICS.csv>)、[前端共同帧表](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/comparison/FRONTEND_HISTORY_MATCHED.csv>)。

## 全Mesh路线的共同208评价

|路线/版本|面|IoU↑|F1↑|Chamfer px↓|覆盖↑|背景几何↓|封闭缺失px↓|粗糙度↓|
|---|---|---|---|---|---|---|---|---|
|M1_Omega_NKSR/post|8117768|0.8263|0.0475|27.86|0.9971|0.0930|369.39|0.0461|
|M2_Standard_NKSR/post|7851387|0.8750|0.1039|19.92|0.9864|0.0579|3660.36|0.0422|
|M3_E10_GGGS_NKSR/post|9001896|0.6551|0.0858|64.36|0.9954|0.2640|85.66|0.0294|
|M4_E10_GGGS_TSDF/post|1290765|0.8825|0.1225|35.82|0.9919|0.0602|545.05|0.0582|
|M5_E11_2DGS_TSDF/post|1998791|0.6183|0.0361|85.38|0.9917|0.2899|400.57|0.0941|
|P1/post|15334802|0.6284|0.0894|68.31|0.9909|0.2753|1258.48|0.0975|
|P2_TSDF/post|1290378|0.8829|0.1235|34.63|0.9919|0.0600|544.86|0.0582|
|P2_Tetrahedra/post|15315548|0.6266|0.0880|69.43|0.9908|0.2767|1257.61|0.0979|
|P3_native/post|898522|0.8325|0.1149|41.80|0.8905|0.0329|9806.81|0.0893|
|P3_shared/post|1514233|0.9156|0.1519|22.29|0.9881|0.0362|358.31|0.0931|
|P4_median_mask/post|2153414|0.6680|0.0331|81.32|0.9952|0.2297|211.18|0.1373|
|P4_median_mask_alpha_voxel075/post|4093421|0.6643|0.0308|83.05|0.9954|0.2333|254.96|0.1456|
|P4_median_mask_alpha/post|2153317|0.6680|0.0331|81.32|0.9952|0.2297|210.75|0.1373|
|P4_expected_mask_alpha/post|1998839|0.6183|0.0361|85.38|0.9917|0.2899|400.22|0.0941|
|P4_expected_mask/post|1998791|0.6183|0.0361|85.38|0.9917|0.2899|400.57|0.0941|
|P4_median_mask_alpha_voxel125/post|1324798|0.6717|0.0354|79.52|0.9951|0.2260|179.10|0.1321|
|E12_TSDF/post|1275509|0.9026|0.1368|18.64|0.9936|0.0472|446.32|0.0560|
|E12_Tetrahedra/post|13190948|0.7340|0.0765|65.99|0.9911|0.1555|1861.18|0.1099|
|E13_QGS/official_top10_post|2187817|0.8933|0.1482|28.95|0.9896|0.0476|1001.80|0.1344|
|E13_QGS/post|2338160|0.8287|0.0780|72.47|0.9916|0.0866|864.33|0.1356|
|E14_cluster_before/post|1991108|0.4402|0.0196|116.80|0.9753|0.5694|2051.85|0.1346|
|E14_cluster_after/post|1237026|0.4994|0.0306|103.85|0.8967|0.3801|1670.03|0.1224|
|E14_initial/post|1588063|0.4888|0.0165|116.45|0.9902|0.4654|998.00|0.1094|
|E14_refined/post|1588062|0.4890|0.0195|115.35|0.9905|0.4652|950.97|0.0147|

全部raw/post/binary/官方post拓扑、旧指标与统一重算分别保留在[全Mesh CSV](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/comparison/ALL_MESH_EXPERIMENTS.csv>)和[Mesh逐视角](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/comparison/mesh_metrics.csv>)；共同组件post与官方top10不是同一处理。P4 Expected Mask与M5同基线，六种P4全部保留。

逐视角配对而非仅均值：相对M4，P3_shared在161/208（77.40%）视角IoU改善，中位差+0.04405、P10/P90为−0.00795/+0.06199；E12 TSDF在169/208（81.25%）改善，中位差+0.01913、P10/P90为−0.00243/+0.04627。两者都不是全部视角获益。E12 corrected前景PSNR在207/208改善，中位+0.81127dB；P2前景PSNR差中位仅+0.000016dB，不能夸大清理的照片收益。全部相关视角作描述统计，不视为独立重复试验。

## 可直接打开的模型与Mesh

|模型阶段|实际表示|真实绝对PLY路径|字节|真实Checkpoint|
|---|---|---|---|---|
|E10|GGGS|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/geometry_rgb_dual_frontend/E10_Standard_Corrected_Interface/checkpoints/point_cloud/iteration_30000/point_cloud.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/geometry_rgb_dual_frontend/E10_Standard_Corrected_Interface/checkpoints/point_cloud/iteration_30000/point_cloud.ply>)|145253348|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/geometry_rgb_dual_frontend/E10_Standard_Corrected_Interface/checkpoints/chkpnt30000.pth](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/geometry_rgb_dual_frontend/E10_Standard_Corrected_Interface/checkpoints/chkpnt30000.pth>)|
|P2|GGGS|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P2_GGGS_Cleaning/cleaned_gaussians.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P2_GGGS_Cleaning/cleaned_gaussians.ply>)|137108204|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/geometry_rgb_dual_frontend/E10_Standard_Corrected_Interface/checkpoints/chkpnt30000.pth](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/geometry_rgb_dual_frontend/E10_Standard_Corrected_Interface/checkpoints/chkpnt30000.pth>)|
|E11|2DGS|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/geometry_mesh_final_selection/E11_2DGS_Corrected/checkpoints/point_cloud/iteration_30000/point_cloud.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/geometry_mesh_final_selection/E11_2DGS_Corrected/checkpoints/point_cloud/iteration_30000/point_cloud.ply>)|60666984|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/geometry_mesh_final_selection/E11_2DGS_Corrected/checkpoints/chkpnt30000.pth](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/geometry_mesh_final_selection/E11_2DGS_Corrected/checkpoints/chkpnt30000.pth>)|
|E12|GGGS|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E12_GGGS_EO/checkpoints/point_cloud/iteration_30000/point_cloud.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E12_GGGS_EO/checkpoints/point_cloud/iteration_30000/point_cloud.ply>)|124964072|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E12_GGGS_EO/checkpoints/chkpnt30000.pth](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E12_GGGS_EO/checkpoints/chkpnt30000.pth>)|
|E13|Quadratic six signed-scale primitive; curvature is not thickness|[/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/checkpoints/point_cloud/iteration_30000/point_cloud.ply](</home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/checkpoints/point_cloud/iteration_30000/point_cloud.ply>)|138512700|[/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/checkpoints/chkpnt30000.pth](</home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/checkpoints/chkpnt30000.pth>)|
|E14|2D-SuGaR|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/training/point_cloud/iteration_30000/point_cloud.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/training/point_cloud/iteration_30000/point_cloud.ply>)|125392620|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/training/chkpnt30000.pth](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/training/chkpnt30000.pth>)|
|E14_refined|SuGaR mesh-bound refined3D Gaussian|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/refinement/refined_ply/sugarfine_post_normalconsistency01_gaussperface1.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/refinement/refined_ply/sugarfine_post_normalconsistency01_gaussperface1.ply>)|393841156|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/refinement/sugarfine_post_normalconsistency01_gaussperface1/15000.pt](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/refinement/sugarfine_post_normalconsistency01_gaussperface1/15000.pt>)|

E12颜色矩阵：[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E12_GGGS_EO/exposure_params/exposure_matrices.npz](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E12_GGGS_EO/exposure_params/exposure_matrices.npz>)。直接PLY不包含逐帧EO。E13原生六尺度PLY不可用普通3DGS替代；E14原生两尺度与Refined三尺度分别列出。

|路线/版本|实际Mesh绝对路径|字节|顶点|面|
|---|---|---|---|---|
|P1/post|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P1_GGGS_Tetrahedra/recon_post.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P1_GGGS_Tetrahedra/recon_post.ply>)|566272327|7644159|15334802|
|P2_TSDF/post|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P2_GGGS_Cleaning/TSDF/post.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P2_GGGS_Cleaning/TSDF/post.ply>)|51070242|672451|1290378|
|P2_Tetrahedra/post|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P2_GGGS_Cleaning/Tetrahedra/recon_post.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P2_GGGS_Cleaning/Tetrahedra/recon_post.ply>)|565539433|7634105|15315548|
|P3_native/post|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P3_GS2Mesh/meshes/post.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P3_GS2Mesh/meshes/post.ply>)|36087672|478560|898522|
|P3_shared/post|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P3_GS2Mesh/meshes_shared_depth_range/post.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P3_GS2Mesh/meshes_shared_depth_range/post.ply>)|59856016|787660|1514233|
|P4_median_mask/post|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P4_2DGS_Depth_Mesh/median_mask/post.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P4_2DGS_Depth_Mesh/median_mask/post.ply>)|86836776|1153766|2153414|
|P4_median_mask_alpha_voxel075/post|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P4_2DGS_Depth_Mesh/median_mask_alpha_voxel075/post.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P4_2DGS_Depth_Mesh/median_mask_alpha_voxel075/post.ply>)|165850137|2208536|4093421|
|P4_median_mask_alpha/post|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P4_2DGS_Depth_Mesh/median_mask_alpha/post.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P4_2DGS_Depth_Mesh/median_mask_alpha/post.ply>)|86832404|1153705|2153317|
|P4_expected_mask_alpha/post|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P4_2DGS_Depth_Mesh/expected_mask_alpha/post.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P4_2DGS_Depth_Mesh/expected_mask_alpha/post.ply>)|80270400|1064415|1998839|
|P4_expected_mask/post|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P4_2DGS_Depth_Mesh/expected_mask/post.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P4_2DGS_Depth_Mesh/expected_mask/post.ply>)|80268552|1064391|1998791|
|P4_median_mask_alpha_voxel125/post|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P4_2DGS_Depth_Mesh/median_mask_alpha_voxel125/post.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/P4_2DGS_Depth_Mesh/median_mask_alpha_voxel125/post.ply>)|53221612|705861|1324798|
|E12_TSDF/post|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E12_GGGS_EO/TSDF/post.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E12_GGGS_EO/TSDF/post.ply>)|50432327|663733|1275509|
|E12_Tetrahedra/post|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E12_GGGS_EO/Tetrahedra/recon_post.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E12_GGGS_EO/Tetrahedra/recon_post.ply>)|486932881|6571881|13190948|
|E13_QGS/official_top10_post|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/meshes/native_TSDF/official_top10_post.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/meshes/native_TSDF/official_top10_post.ply>)|86834501|1144952|2187817|
|E13_QGS/post|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/meshes/native_TSDF/post.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/meshes/native_TSDF/post.ply>)|93814041|1243483|2338160|
|E14_cluster_before/post|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/cluster_pruning/cluster_before_mesh/post.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/cluster_pruning/cluster_before_mesh/post.ply>)|80262309|1066227|1991108|
|E14_cluster_after/post|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/cluster_pruning/cluster_after_mesh/post.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/cluster_pruning/cluster_after_mesh/post.ply>)|49331115|651950|1237026|
|E14_initial/post|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/initial_mesh/post.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/initial_mesh/post.ply>)|63312256|836610|1588063|
|E14_refined/post|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/refinement/mesh_post/post.ply](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/refinement/mesh_post/post.ply>)|60802353|836610|1588062|

E14原生UV资产：

|类型|实际OBJ/MTL/PNG|字节|
|---|---|---|
|Textured OBJ|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/refinement/sugarfine_post_normalconsistency01_gaussperface1.obj](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/refinement/sugarfine_post_normalconsistency01_gaussperface1.obj>)|198639950|
|Texture|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/refinement/sugarfine_post_normalconsistency01_gaussperface1.mtl](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/refinement/sugarfine_post_normalconsistency01_gaussperface1.mtl>)|72|
|Texture|[/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/refinement/sugarfine_post_normalconsistency01_gaussperface1.png](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/refinement/sugarfine_post_normalconsistency01_gaussperface1.png>)|44500805|

OBJ末尾换行导致当前Open3D遗漏一面，UV缝导入复制顶点；原生OBJ所有1,588,063面保留，已评价PLY1,588,062面。精确XYZ合并不移动三角面、不平滑或补洞，详情见[实际拓扑恢复证明](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/MESH_TOPOLOGY_RECOVERY.md>)。

完整历史/新增全部阶段Gaussian、Removed、Checkpoint、Mesh raw/post/binary/refined、Depth/Normal/Alpha与EO参数：[Gaussian索引](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/artifact_index/GAUSSIAN_PLY_INDEX.md>) · [Mesh索引](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/artifact_index/MESH_FILE_INDEX.md>) · [机器总索引](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/artifact_index/ALL_ARTIFACTS.csv>) · [实际查看指南](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/artifact_index/HOW_TO_VIEW_RESULTS.md>)。每项实际文件解析、字节/类型/头部计数与相对路径核验见[完整交付验证](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/artifact_index/FULL_DELIVERY_VALIDATION.json>)。

## 实际视觉对照与空间诊断

直接使用当前模型视觉读取图片，未调用外部识图API。已逐一查看八个统一叶缘gray/normal窗口、source53整株与RGB/Alpha/Depth/Normal、叶纹局部、以及53+15°和157−15°七模型原生自由视角；没有声称逐一人工查看全部208照片。所有8固定图、多ROI、俯视/斜俯视/自由视角和高亮候选图实际生成，完整入口见[视觉索引](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/comparison/visualizations/VISUAL_COMPARISON_INDEX.md>)和[直接视觉复核记录](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/comparison/visualizations/VISUAL_REVIEW.json>)。

![同相机整株灰色Mesh：E14大斜面无法靠纹理解释为正常叶片](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/comparison/visualizations/scientific_review/whole_0053_gray.png>)

同相机整株灰色Mesh：E14大斜面无法靠纹理解释为正常叶片。

![同叶尖窗口：P3局部漂浮环/圆钝尖端与E12尖端取舍](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/comparison/visualizations/scientific_review/leaf_edge_0157_gray_normal.png>)

同叶尖窗口：P3局部漂浮环/圆钝尖端与E12尖端取舍。

![真实自由53+15°：清理黑云未消失，QGS此视角黑屏，E14错误面](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/comparison/visualizations/scientific_review/free/free_0053_p15_RGB_Alpha.png>)

真实自由53+15°：清理黑云未消失，QGS此视角黑屏，E14错误面。

![真实自由157−15°：QGS可见叶片，不能从53泛化所有自由视角失败](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/comparison/visualizations/scientific_review/free/free_0157_m15_RGB_Alpha.png>)

真实自由157−15°：QGS可见叶片，不能从53泛化所有自由视角失败。

E14 Refined Gaussian深度/法线图明确标AUX_centerZ/AUX_axis，不能冒充原生表面Depth/Normal；实际Mesh深度/单位面法线另有真实raycast图。Gaussian原生RGB/Alpha/Depth/Normal八视角及所有固定局部窗口详见对应PROTOCOL。

原始RGB候选与各模型几何、原生基元关联、Mesh同一射线查询均保留源ID/像素/世界点/有效支持与缺失数。来源：[原RGB候选几何](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/comparison/reflection_analysis/ORIGINAL_RGB_CANDIDATE_GEOMETRY.csv>) · [原生形状独立定义](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/comparison/native_primitive_shapes/ALL_NATIVE_SHAPE_SUMMARY.csv>) · [共同射线多层](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/comparison/same_ray_mesh_layers/ALL_MESH_LAYER_METRICS.csv>) · [共同前后面间距](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/comparison/same_ray_mesh_layers/MATCHED_FIRST_PAIR_SPACING.csv>) · [可见叶区PCA](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/comparison/leaf_visible_surface/ALL_REGION_SURFACE_METRICS.csv>) · [208配对变化分布](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/comparison/paired_view_changes/PAIRED_VIEW_CHANGE_SUMMARY.csv>)。这些不是互相独立的扫描真值，相关视角只作描述统计。

## 官方源码、补丁与训练真实性

|方法|官方仓库固定commit|
|---|---|
|GGGS|[23b418c2f0f19b30b251acad6fe13445dae968c7](https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids/tree/23b418c2f0f19b30b251acad6fe13445dae968c7)|
|QGS|[74d05c945e99fcaef7afe5a8831903be71ad9b55](https://github.com/will-zzy/QGS/tree/74d05c945e99fcaef7afe5a8831903be71ad9b55)|
|2D-SuGaR|[e62770ec6ab50076ebb6c0e3b4ac39f777f14d27](https://github.com/prajwalcr/2d-sugar/tree/e62770ec6ab50076ebb6c0e3b4ac39f777f14d27)|
|GS2Mesh|[560f3e8a2349aaad24fe9444ef1ca2770aeec5ab](https://github.com/yanivw12/gs2mesh/tree/560f3e8a2349aaad24fe9444ef1ca2770aeec5ab)|

E12唯一训练基线E10：实际目标0.8 L1(EO(render),GT)+0.2[1−SSIM(EO(render),GT)]，几何正则/增密/初始化/208相机与Seed保持E10；旧EO只校正L1，本轮明确让SSIM也校正。没有额外FG RGB/Mask Loss或训练Pruning。E12 raw/corrected在评价时使用同一Gaussian和同一Alpha/Depth/Normal，颜色矩阵不改变几何；训练时EO可间接改变几何优化，必须用E10/E12各实际Mesh/空间指标判断。

QGS官方Mip原生depth_ratio=1、lambda_dist=10、lambda_normal=0.015、densify_grad_threshold=1、interval100、from500/until15000、percent_dense0.0001；dist从3000、normal从7000。30K训练中原生NCC/多视角从30000的严格后续条件未激活，曲率专项正则原配置为0/极晚启动，不添加自定义权重凑齐其他方法。二次表面基元与其CUDA渲染仍真实保留。base两次实际OOM归档；Mip从同一VGGT初始化、Seed0重新正式训练，不用失败base权重恢复成Mip。

E14完整原生流程：额外Metric3D prior、深度初始化、2DGS原生30K、7K DBSCAN、初始TSDF、SuGaR15K、Gaussian/UV纹理导出。完整法线图先验损失没有RAP/Alpha乘项，RAP另限制初始化/融合。15K训练已完成后texture extractor遇PyTorch2.10 weights_only默认加载问题，只对可信本机自产Checkpoint显式weights_only=False恢复导出，复用真实15K状态，没有重训或伪造终态。

[E12控制审计](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E12_GGGS_EO/CONTROLLED_AUDIT.json>) · [E12 EO补丁](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E12_GGGS_EO/EO_OBJECTIVE.patch>) · [QGS终态来源/非有限参数](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/checkpoint_provenance_validation.json>) · [E14真实精修完成](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E14_2D_SuGaR/refinement_complete.json>) · [官方直接联网README核验](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/source_manifest/upstream_verification_20261110/verification.json>)。源码commit、全部补丁与执行脚本SHA另保存在source_manifest。


实际适配补丁：[GGGS patch](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/source_manifest/GGGS_tracked_worktree.patch>) · [QGS patch](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/source_manifest/QGS_tracked_worktree.patch>) · [GS2Mesh patch](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/source_manifest/GS2Mesh_tracked_worktree.patch>) · [2D-SuGaR patch](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/source_manifest/2D_SuGaR_tracked_worktree.patch>) · [全部源码与执行脚本SHA清单](</data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/source_manifest/SOURCE_AND_SCRIPT_MANIFEST.json>)。

博查搜索实际返回HTTP403（余额/额度不足），未取得搜索结果；固定官方README直接HTTP核验200且与本地源码相同，技术接口文档按实际Context7结果与本地官方代码核查。上述实验结论以真实模型、日志、统一CSV和直接看图为来源，非论文排名或训练数据推测。
