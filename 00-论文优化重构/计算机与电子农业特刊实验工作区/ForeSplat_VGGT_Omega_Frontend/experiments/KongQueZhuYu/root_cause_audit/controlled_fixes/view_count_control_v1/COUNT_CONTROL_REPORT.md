# 视角数量定向控制：共同源帧、共同图像对应、共同视角对

本次70→110严格共同38源帧控制未支持“视角更多必然导致退化”。共同rank近邻的两模型RGB相机证据和depth到真实RGB误差总体改善，但跨depth全query支持变化很小，部分pair反而更差。标准模型在当前黑前景输入下仍明显失配。208旧结果仅是历史参照，不能将它与当前70/110的差异全部归因数量。

此报告根据真实控制目录的cameras/depth/confidence/Mask、既有前景SIFT缓存和固定image-only RANSAC内点计算。评价仅CPU；不训练、不推理、不改原有03数值、历史Dataset或模型。所有像素误差在共同 **384×688** 网格，未使用1080×1920 GGGS训练像素，不比较不同模型原输入pixel单位。

## 主控制结果

| 条件 | 共同有效RGB pair | Sampson p50共享px | depth→真实RGB p50共享px | 5%depth全query支持 | 同pair相对旋转p90 deg |
| --- | --- | --- | --- | --- | --- |
| Omega_70 | 32/37 | 7.040 | 20.824 | 44.71% | 65.54 |
| Omega_110 | 32/37 | 5.101 | 16.343 | 44.83% | 48.21 |
| Standard_70 | 32/37 | 101.855 | 254.846 | 5.56% | 152.36 |
| Standard_110 | 32/37 | 61.596 | 235.656 | 5.91% | 121.17 |
| Omega_208_historical | 32/37 | 4.097 | 11.451 | 57.73% | 44.80 |
| Standard_208_historical | 32/37 | 66.003 | 179.562 | 9.36% | 121.72 |

Sampson p50是每pair的同一批独立前景RGB对应误差中位数再对pair取中位数；源depth投向真正匹配的目标RGB像素，不是向另一个预测depth找最近点。target可见性来自独立RGB匹配，预测depth正数/投影有效性和失败分母仍单独保存。depth5%支持使用相同原图前景query、双向投影和目标Mask/正depth资格，以所有共同queries为分母。它是跨预测几何支持比例，不是GT准确率或物理植物覆盖率。

## 严格共同的输入、帧和匹配

共同source IDs：`[0, 6, 9, 15, 21, 27, 30, 36, 42, 52, 58, 64, 67, 73, 79, 85, 88, 94, 100, 109, 115, 121, 124, 130, 136, 142, 145, 151, 157, 166, 172, 178, 181, 187, 193, 199, 202, 209]`。每帧仍按source名字定位相机/depth/K。新条件使用保存的真实pixel_transform_pixel_center映回原坐标；保留nominal旧A供审计，不改变模型预测或导出相机。所有模型/条件共享同一个原图→384×688参考网格。

固定共同38列表rank间隔1/2/4的37/36/34个unordered pair；共同image-only F有效32/18/10对，总60/107。每个pair的真实原sourceID gap存清单，rank邻居不是原gap1，也不是各自70/110子集的邻居。所有6个模型N结果使用相同pair集合，未删除某条件失败pair。

特征来自旧 `camera_projection_comparison/sift_features/*.npz` 中相同RAP前景RGB和Mask，只读、不重提背景条件特征。mutual Lowe0.75、image-only F RANSAC阈值1.5共享px、至少12内点、逐pair固定seed；原始内点坐标一次冻结在 [fixed_foreground_matches/](evaluation/fixed_foreground_matches/) 并用于所有条件。RANSAC不使用任何模型相机或depth决定内点，因此没有模型条件选择泄漏。远间隔少内点/重复纹理仍有不可靠风险，主要判断使用近邻共同pair。

输入/环境指纹实测复核：[actual_input_invariant_crosscheck.json](evaluation/actual_input_invariant_crosscheck.json)。两模型条件内checkpointSHA、seed0、Python、PyTorch、preprocessing、核心源码SHA、每帧Mask SHA、输入尺寸和nominal/真实A全部一致。70/110共同38帧float32模型RGB SHA逐帧全部相同。208历史无旧完整环境freeze，排除在严格环境一致结论之外。

## 同pair条件差分，避免只比较两个总体中位数

以下是逐同pair `target-baseline` 的差值分布；误差负数为改善，支持率正数为改善。这和两个条件各自总体中位数之差不同。pair存在时间/同植物相关性，不宣称独立总体因果置信区间。

| 模型 | 共同rank gap | 指标 | 同pair数 | 差值p10/p50/p90 | 改善pair数 |
| --- | --- | --- | --- | --- | --- |
| Omega | 1 | sampson_p50_px | 32 | -25.881/-1.653/3.224 | 20 |
| Omega | 1 | source_depth_to_RGB_error_p50_px | 32 | -78.230/-2.395/5.360 | 21 |
| Omega | 1 | bidirectional_all_query_depth_support5pct | 37 | -0.079/0.001/0.076 | 19 |
| Omega | 2 | sampson_p50_px | 18 | -42.916/-0.227/10.276 | 10 |
| Omega | 2 | source_depth_to_RGB_error_p50_px | 18 | -208.263/-7.714/5.367 | 12 |
| Omega | 2 | bidirectional_all_query_depth_support5pct | 36 | -0.075/0.009/0.102 | 21 |
| Omega | 4 | sampson_p50_px | 10 | -22.430/-2.597/2.967 | 7 |
| Omega | 4 | source_depth_to_RGB_error_p50_px | 10 | -61.278/-11.118/-5.331 | 9 |
| Omega | 4 | bidirectional_all_query_depth_support5pct | 34 | -0.058/0.021/0.137 | 22 |
| Standard | 1 | sampson_p50_px | 32 | -74.236/-19.439/79.325 | 22 |
| Standard | 1 | source_depth_to_RGB_error_p50_px | 32 | -205.343/-37.348/107.549 | 21 |
| Standard | 1 | bidirectional_all_query_depth_support5pct | 37 | -0.072/0.000/0.078 | 13 |
| Standard | 2 | sampson_p50_px | 18 | -90.836/-34.271/12.646 | 16 |
| Standard | 2 | source_depth_to_RGB_error_p50_px | 18 | -302.627/-129.719/169.353 | 14 |
| Standard | 2 | bidirectional_all_query_depth_support5pct | 36 | -0.015/0.000/0.026 | 16 |
| Standard | 4 | sampson_p50_px | 10 | -186.544/1.372/65.995 | 5 |
| Standard | 4 | source_depth_to_RGB_error_p50_px | 10 | -152.723/-58.670/28.919 | 8 |
| Standard | 4 | bidirectional_all_query_depth_support5pct | 34 | -0.015/0.000/0.017 | 11 |

原始每pair差分与scope见 [same_pair_condition_effects.csv](evaluation/same_pair_condition_effects.csv)。相对旋转比较在同camera pair上使用 `R_relative`，对global世界坐标gauge不敏感；相对平移只比较方向。camera中心仅在精确sourceID配对正尺度SO3 Sim3拟合后给出诊断，不直接相减未配准中心，也未应用该变换修改任何点图。

| 条件对reference | 留出中心p50 /reference范围 | 留出方向p50 deg | 可靠global Sim3 |
| --- | --- | --- | --- |
| Omega_110 / Omega_70 | 0.045 | 5.949 | True |
| Omega_208_historical / Omega_70 | 0.046 | 7.191 | True |
| Standard_110 / Standard_70 | 0.170 | 13.402 | False |
| Standard_208_historical / Standard_70 | 0.255 | 36.221 | False |

未通过global Sim3的中心残差不能当可靠注册后GT误差；背景改善判断主要依靠与depth无关的独立前景RGB极线证据、真正目标匹配投影，以及同pair的跨depth支持，不依靠中心差值大小。[source_matched_camera_Sim3.json](evaluation/source_matched_camera_Sim3.json)、[same_pair_relative_pose_changes.csv](evaluation/same_pair_relative_pose_changes.csv)。

![相同前景对应/帧pair的控制比较](evaluation/controlled_geometry_comparison.png)

图中count的横轴是共同source列表rank间隔，历史208使用虚线；背景图只有固定12near-pair同组。

## 数量效应的边界

70→110共同rank-near32image pair中，ΩSampson改善20/32、目标RGB-depth改善21/32；标准分别22/32、21/32。两者都存在变差pair，deep支持整体只小变：Ω44.71→44.83%，标准5.56→5.91%。不能宣称所有视角或所有geometry已改善，也没有证据把208旧黑前景退化全部归为N过大。较远rank gap4只10有效图像pair，标准Sampson逐pair差分中位还略增，主要证据仍是共同近邻。

当前严格单因素是70vs110；两模型各自输入前景RGB、Mask、A/尺寸、weights/source/seed/current环境指纹一致。208没有完整旧freeze并未重新推理，只是same-source/pair历史参照，报告中明确标识historical，不应做严格count-only归因。模型之间权重、预处理和分辨率不同，不做跨模型frame-count单因素结论。

背景定向结果另见 [BACKGROUND_CONTROL_REPORT.md](../background_control_v1/BACKGROUND_CONTROL_REPORT.md)，使用另一固定24source/12原gap1pair；不能把该12pair数字直接替代本38common pair数量实验的中位数。

## 文件与复现

CPU执行独立 `scripts/geometry_targeted_control_eval.py --scope count`，之后 `geometry_targeted_control_supplement.py` 与 `geometry_targeted_control_report.py` 相同scope；使用既有 uv `.venv-fast3r`，不占GPU。新增评价和报告只存本control目录，不改旧03数值。再次刷新复用condition_*_results.json与fixed_foreground_matches，避免重复大量原始数组读取。

historical Standard Mask从持久化Dataset原Mask按源SHA、相同square padding与nearest规则懒重放，仅读取被选sourceMask，不读取RGB，不依赖/tmp输入cache。共同38 source与已有临时bool模型Mask逐帧全等验证：[historical_standard_mask_replay_verification.json](evaluation/historical_standard_mask_replay_verification.json)。临时cache只作为一次验证reference，删除后不会破坏count评价入口。

[control_summary.csv](evaluation/control_summary.csv) / [JSON](evaluation/control_summary.json)、[cross_depth_pairs.csv](evaluation/cross_depth_pairs.csv)、[independent_RGB_pairs.csv](evaluation/independent_RGB_pairs.csv)、[relative_camera_pairs.csv](evaluation/relative_camera_pairs.csv)、[fixed_pair_manifest.csv](evaluation/fixed_pair_manifest.csv)、[evaluation_protocol.json](evaluation/evaluation_protocol.json)。来源为这些本地真实控制文件；本轮网络检索此前Bocha403，没有凭训练记忆生成实验结论。
