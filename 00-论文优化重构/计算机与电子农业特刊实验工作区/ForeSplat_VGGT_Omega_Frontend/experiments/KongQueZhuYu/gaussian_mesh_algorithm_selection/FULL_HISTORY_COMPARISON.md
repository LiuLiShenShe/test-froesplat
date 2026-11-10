# 全历史比较：证据台账

历史E1–E11和M1–M5真实产物均保留；统一表以每行实际status、评价范围和文件为准。E1–E14及P2/E14各保存阶段均已使用实际208图像ID与相同GT/RAP文件哈希重算。E2/E3/E4/E6的旧数值/FG定义独立保留，历史原生相机与几何输入差异仍在表中记录，不能因评价帧统一而冒称严格受控训练比较。

E6旧camera_runtime缺少Fx/Fy/Cx/Cy字段，已依据其原存FoV与投影矩阵恢复到新的评价元数据；208个原W2C/P/fullP/相机中心逐项未改。没有将E10修正接口施加给E6，也没有改写旧模型或旧元数据。见[相机字段恢复证明](comparison/history_uniform/E6/CAMERA_FIELD_RECOVERY_AUDIT.json)。

|实验|前端|几何RGB|相机接口|步数|Raw PSNR|Corrected PSNR|评价范围|实际状态|
|---|---|---|---|---|---|---|---|---|
|E1|VGGT-Omega|foreground-only historical|historical centered interface|30000|17.894746642500206||208 actual frozen RGB/true RAP FG; native1080x1920 float clipped for metrics; historical K/background conditions retained per row|model_and208_metrics_verified|
|E2|VGGT-Omega|foreground-only historical|historical centered interface|30000|17.392539944080966|18.233461954081804|208 actual frozen RGB/true RAP FG; native1080x1920 float clipped for metrics; historical K/background conditions retained per row|model_and208_metrics_verified|
|E3|VGGT-Omega|foreground-only historical|historical centered interface|30000|15.164009876617657||208 actual frozen RGB/true RAP FG; native1080x1920 float clipped for metrics; historical K/background conditions retained per row|model_and208_metrics_verified|
|E4|VGGT-Omega|foreground-only historical|historical centered interface|30000|14.21607760100439|14.037425312818243|208 actual frozen RGB/true RAP FG; native1080x1920 float clipped for metrics; historical K/background conditions retained per row|model_and208_metrics_verified|
|E5|standard VGGT|foreground-only historical|historical centered interface|30000|13.105665606402756||208 actual frozen RGB/true RAP FG; native1080x1920 float clipped for metrics; historical K/background conditions retained per row|model_and208_metrics_verified|
|E6|standard VGGT|foreground-only historical|historical centered interface|30000|13.20182702202306||208 actual frozen RGB/true RAP FG; native1080x1920 float clipped for metrics; historical K/background conditions retained per row|model_and208_metrics_verified|
|E7|VGGT-Omega|full background context|historical centered interface|30000|22.49354059365734||208 actual frozen RGB/true RAP FG; native1080x1920 float clipped for metrics; historical K/background conditions retained per row|model_and208_metrics_verified|
|E8|VGGT-Omega|full background context|corrected|30000|23.215933315954604||208 actual frozen RGB/true RAP FG; native1080x1920 float clipped for metrics; historical K/background conditions retained per row|model_and208_metrics_verified|
|E9|standard VGGT|full background context|historical centered interface|30000|23.96829630456197||208 actual frozen RGB/true RAP FG; native1080x1920 float clipped for metrics; historical K/background conditions retained per row|model_and208_metrics_verified|
|E10|standard VGGT|full background context|corrected|30000|23.900519610857604||208 actual frozen RGB/true RAP FG; native1080x1920 float clipped for metrics; historical K/background conditions retained per row|model_and208_metrics_verified|
|E11|standard VGGT|full background context|corrected|30000|22.261471765286952||208 actual frozen RGB/true RAP FG; native1080x1920 float clipped for metrics; historical K/background conditions retained per row|model_and208_metrics_verified|
|E12|standard VGGT|full background context|corrected|30000|16.973571063501716|24.73118883045985|208 actual frozen RGB/true RAP FG; native1080x1920 float clipped for metrics; historical K/background conditions retained per row|model_and208_metrics_verified|
|E13|standard VGGT|full background context|corrected|30000|22.895602263200207||208 actual frozen RGB/true RAP FG; native1080x1920 float clipped for metrics; historical K/background conditions retained per row|model_and208_metrics_verified|
|E14|standard VGGT|full background context|corrected|30000|16.620825560776517||208 actual frozen RGB/true RAP FG; native1080x1920 float clipped for metrics; historical K/background conditions retained per row|model_and208_metrics_verified|
|P2|standard VGGT|full background context|corrected|30000|23.913001431552058||208 actual frozen RGB/true RAP FG; native1080x1920 float clipped for metrics; historical K/background conditions retained per row|model_and208_metrics_verified|
|E14_refined|standard VGGT|full background context|corrected|15000|18.429766996762357||208 actual frozen RGB/true RAP FG; native1080x1920 float clipped for metrics; historical K/background conditions retained per row|model_and208_metrics_verified|
|E14_initial|standard VGGT|full background context|corrected|1|15.009580150023139||208 actual frozen RGB/true RAP FG; native1080x1920 float clipped for metrics; historical K/background conditions retained per row|model_and208_metrics_verified|
|E14_cluster_before|standard VGGT|full background context|corrected|7000|17.45312251259555||208 actual frozen RGB/true RAP FG; native1080x1920 float clipped for metrics; historical K/background conditions retained per row|model_and208_metrics_verified|
|E14_cluster_after|standard VGGT|full background context|corrected|7000|17.574823050708183||208 actual frozen RGB/true RAP FG; native1080x1920 float clipped for metrics; historical K/background conditions retained per row|model_and208_metrics_verified|

E1–E4包含早期Ω/EO/Foreground组合；E5/E6是旧前景几何输入，E7–E10恢复背景并修正接口。新增均固定标准VGGT和修正208相机，但QGS/2D-SuGaR保留原生损失/训练阶段及额外先验，不能假定全部损失权重相同。没有同株扫描，因此全部几何排序是多视角一致性/轮廓代理。

统一光度评价使用03-SAM保存的训练RGB及真实RAP Mask；其背景大部分为黑色。前端几何RGB恢复02-FFT的背景上下文，新增反光诊断另用实际02-FFT原图核验候选。这三类RGB的来源与目的分开记录，不能把全图PSNR当成未分割场景的照片重建精度。[实际RGB源文件审计](comparison/reflection_analysis/ORIGINAL_RGB_SOURCE_FILES.csv)保留原图/训练图路径与测量差异。

[统一Gaussian CSV](comparison/ALL_GAUSSIAN_EXPERIMENTS.csv) · [保留旧Gaussian指标](comparison/LEGACY_GAUSSIAN_METRICS.csv) · [统一Mesh CSV](comparison/ALL_MESH_EXPERIMENTS.csv) · [逐视角渲染](comparison/rendering_metrics.csv) · [Mesh逐视角](comparison/mesh_metrics.csv) · [全部实际模型](artifact_index/ALL_ARTIFACTS.csv)。

Fast3R/AMB3R历史前端比较已在独立[完整历史前端表](comparison/FRONTEND_HISTORY.csv)与[共同帧表](comparison/FRONTEND_HISTORY_MATCHED.csv)保留，不重新推理。下面为旧条件的共同208帧诊断；Confidence/q85未跨模型校准，自重投影/邻帧一致性也不是扫描精度，不与恢复背景后的VGGT数值冒称同条件。

|历史前端|共同帧|状态|FG覆盖|叶缘覆盖|相邻深度支持|相邻投影重叠|
|---|---|---|---|---|---|---|
|VGGT_Omega_Existing|208|available|1.0|1.0|0.9605950742172171|0.9048910526390455|
|Fast3R|208|available|1.0|1.0|0.8234080948698118|0.8045486021990667|
|AMB3R_Base_OFF|208|available|0.2067651596663165|0.08106838575917587||0.0|
|AMB3R_Base_ON|208|available|0.9481712590962934|0.9063178269461272|0.8833518930957683|0.18492243405633957|
|AMB3R_SfM|208|available|1.0|1.0|0.8366653963014705|0.8533026278420669|
|Fast3R_AMB3R_SfM|208|available|1.0|1.0|0.6944014846891432|0.5441161271207317|
|Fast3R_FP32|208|available|1.0|1.0|0.7843603333955886|0.7804533529479655|

恢复背景后的标准/Ω独立表仍见[双前端真实208结果](../geometry_rgb_dual_frontend/frontend_comparison.csv)，不能将旧Fast3R/AMB3R几何RGB条件更名为新增实验条件。

来源：真实本地模型/CSV/训练日志；博查HTTP403，未取得联网搜索结果；官方固定源码直接读取成功。

[叶区与下部区域208分层统计](comparison/leaf_region_mesh_metrics.csv)及[逐视角证据](comparison/leaf_region_mesh_per_view.csv)避免只由盆体主导整株比较。原始208视角同时是本轮训练输入，没有独立留出照片；这些PSNR/SSIM/LPIPS衡量已观测视角拟合，自由视角仅有真实渲染外观和几何一致性证据。原始RAP中的空白区域排除在缺失前景代理之外，未逐一人工认证其是否为真实叶片孔洞。

[实际全视角配对变化汇总](comparison/paired_view_changes/PAIRED_VIEW_CHANGE_SUMMARY.csv)、[逐视角原值及差值](comparison/paired_view_changes/PAIRED_VIEW_METRIC_CHANGES.csv)和[来源/适用范围](comparison/paired_view_changes/PROTOCOL.json)补充均值、中位、P10/P90及方向改善比例。它们保留全部208视角，不把相关视角当成独立重复试验；Gaussian到各自Mesh的深度差不具有共同扫描真值，因此未按该差值跨模型排名。实际ECDF已从保存的终态模型结果重算，提供PNG和PDF。最终综合选择及适用范围见[算法选择报告](FINAL_RECONSTRUCTION_AND_MESH_SELECTION.md)。
