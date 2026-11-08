# 背景定向控制：保留Mask内原像素、只恢复Mask外RGB

仅恢复原图Mask外背景的24帧控制，在两模型同一批前景对应上产生强改善。两模型全部12/12 pair的独立RGB极线误差、源depth到真实目标RGB误差和双向depth支持均改善。因此先前208黑前景保存结果的失败不能直接当作模型固有的植物几何能力上限；当前证据优先指向几何输入背景处理。此结果不等于208帧背景恢复推理已经完成。

此报告根据真实控制目录的cameras/depth/confidence/Mask、既有前景SIFT缓存和固定image-only RANSAC内点计算。评价仅CPU；不训练、不推理、不改原有03数值、历史Dataset或模型。所有像素误差在共同 **384×688** 网格，未使用1080×1920 GGGS训练像素，不比较不同模型原输入pixel单位。

## 主控制结果

| 条件 | 共同有效RGB pair | Sampson p50共享px | depth→真实RGB p50共享px | 5%depth全query支持 | 同pair相对旋转p90 deg |
| --- | --- | --- | --- | --- | --- |
| Omega_RAP_baseline_24 | 12/12 | 9.802 | 23.300 | 52.69% | 23.41 |
| Omega_background_restored_24 | 12/12 | 0.548 | 1.170 | 92.03% | 10.27 |
| Standard_RAP_baseline_24 | 12/12 | 37.150 | 117.290 | 23.78% | 32.00 |
| Standard_background_restored_24 | 12/12 | 0.440 | 1.276 | 92.98% | 10.69 |

Sampson p50是每pair的同一批独立前景RGB对应误差中位数再对pair取中位数；源depth投向真正匹配的目标RGB像素，不是向另一个预测depth找最近点。target可见性来自独立RGB匹配，预测depth正数/投影有效性和失败分母仍单独保存。depth5%支持使用相同原图前景query、双向投影和目标Mask/正depth资格，以所有共同queries为分母。它是跨预测几何支持比例，不是GT准确率或物理植物覆盖率。

## 严格共同的输入、帧和匹配

共同source IDs：`[0, 1, 19, 20, 38, 39, 57, 58, 76, 77, 95, 96, 114, 115, 133, 134, 152, 153, 170, 171, 189, 190, 208, 209]`。每帧仍按source名字定位相机/depth/K。新条件使用保存的真实pixel_transform_pixel_center映回原坐标；保留nominal旧A供审计，不改变模型预测或导出相机。所有模型/条件共享同一个原图→384×688参考网格。

固定12近邻pair：(0,1)、(19,20)、(38,39)、(57,58)、(76,77)、(95,96)、(114,115)、(133,134)、(152,153)、(170,171)、(189,190)、(208,209)。全部12pair通过同前景image-only F资格。

特征来自旧 `camera_projection_comparison/sift_features/*.npz` 中相同RAP前景RGB和Mask，只读、不重提背景条件特征。mutual Lowe0.75、image-only F RANSAC阈值1.5共享px、至少12内点、逐pair固定seed；原始内点坐标一次冻结在 [fixed_foreground_matches/](evaluation/fixed_foreground_matches/) 并用于所有条件。RANSAC不使用任何模型相机或depth决定内点，因此没有模型条件选择泄漏。远间隔少内点/重复纹理仍有不可靠风险，主要判断使用近邻共同pair。

输入/环境指纹实测复核：[actual_input_invariant_crosscheck.json](evaluation/actual_input_invariant_crosscheck.json)。两模型条件内checkpointSHA、seed0、Python、PyTorch、preprocessing、核心源码SHA、每帧Mask SHA、输入尺寸和nominal/真实A全部一致。24帧模型RGB SHA全部变化，这是预定背景变量；Mask和几何变换逐帧全部相同。`prepared_sources.json`保存制备前后的来源与原尺寸逐像素断言：仅恢复Mask外原RGB，24帧Mask内RGB改动=0，PNG重读已核验。该图不冒充未经处理的original RGB；resize可在边界混入变化的外部像素，影响也是背景变量的一部分。

## 同pair条件差分，避免只比较两个总体中位数

以下是逐同pair `target-baseline` 的差值分布；误差负数为改善，支持率正数为改善。这和两个条件各自总体中位数之差不同。pair存在时间/同植物相关性，不宣称独立总体因果置信区间。

| 模型 | 共同rank gap | 指标 | 同pair数 | 差值p10/p50/p90 | 改善pair数 |
| --- | --- | --- | --- | --- | --- |
| Omega | 1 | sampson_p50_px | 12 | -34.549/-9.379/-2.197 | 12 |
| Omega | 1 | source_depth_to_RGB_error_p50_px | 12 | -58.578/-22.785/-11.895 | 12 |
| Omega | 1 | bidirectional_all_query_depth_support5pct | 12 | 0.186/0.361/0.628 | 12 |
| Standard | 1 | sampson_p50_px | 12 | -83.101/-36.679/-14.592 | 12 |
| Standard | 1 | source_depth_to_RGB_error_p50_px | 12 | -275.427/-116.221/-43.101 | 12 |
| Standard | 1 | bidirectional_all_query_depth_support5pct | 12 | 0.338/0.746/0.877 | 12 |

原始每pair差分与scope见 [same_pair_condition_effects.csv](evaluation/same_pair_condition_effects.csv)。相对旋转比较在同camera pair上使用 `R_relative`，对global世界坐标gauge不敏感；相对平移只比较方向。camera中心仅在精确sourceID配对正尺度SO3 Sim3拟合后给出诊断，不直接相减未配准中心，也未应用该变换修改任何点图。

| 条件对reference | 留出中心p50 /reference范围 | 留出方向p50 deg | 可靠global Sim3 |
| --- | --- | --- | --- |
| Omega_background_restored_24 / Omega_RAP_baseline_24 | 0.198 | 41.079 | False |
| Standard_background_restored_24 / Standard_RAP_baseline_24 | 0.271 | 123.339 | False |

未通过global Sim3的中心残差不能当可靠注册后GT误差；背景改善判断主要依靠与depth无关的独立前景RGB极线证据、真正目标匹配投影，以及同pair的跨depth支持，不依靠中心差值大小。[source_matched_camera_Sim3.json](evaluation/source_matched_camera_Sim3.json)、[same_pair_relative_pose_changes.csv](evaluation/same_pair_relative_pose_changes.csv)。

![相同前景对应/帧pair的控制比较](evaluation/controlled_geometry_comparison.png)

图中count的横轴是共同source列表rank间隔，历史208使用虚线；背景图只有固定12near-pair同组。

## 分母与固定局部区域

| 条件 | 同一前景内点总数 | 实际有效投影内点数 | 每pair有效比例最小值 |
| --- | --- | --- | --- |
| Omega_RAP_baseline_24 | 5098 | 5098 | 1.0 |
| Omega_background_restored_24 | 5098 | 5098 | 1.0 |
| Standard_RAP_baseline_24 | 5098 | 5098 | 1.0 |
| Standard_background_restored_24 | 5098 | 5098 | 1.0 |

四条件全部使用同5098个前景内点，全部5098/5098投影有效、每pair valid fraction=1，没有通过删除baseline失败内点制造差距。两模型全部12/12 pair的Sampson、真正目标RGB投影和双向depth支持均改善。这个12/12指匹配pair趋势，不是所有单像素都改善。

局部窗口沿用source0原图坐标：leaf=[540,570,1080,1230]、leaf_edge=[510,630,720,1110]、pot_rim=[800,1680,1700,2000]。它们是固定图像窗口，不是跨视图已跟踪叶片实例或3DGT。使用同SIFT内点再按相同坐标选子集，leaf同10pair、leaf_edge同8pair、盆边12pair有至少3内点；没有另提不同特征。

| 条件 | 固定region | 非空depth方向pair | 5%depth全query支持 | 同RGB pair数 | Sampson p50共享px | depth→RGB p50共享px |
| --- | --- | --- | --- | --- | --- | --- |
| Omega_RAP_baseline_24 | foreground_boundary_common_grid | 24 | 30.86% | 0 | unavailable | unavailable |
| Omega_RAP_baseline_24 | leaf_window | 21 | 61.33% | 10 | 10.763 | 29.162 |
| Omega_RAP_baseline_24 | leaf_edge_window | 19 | 62.37% | 8 | 10.500 | 25.062 |
| Omega_RAP_baseline_24 | pot_rim_window | 24 | 54.88% | 12 | 7.433 | 23.889 |
| Omega_background_restored_24 | foreground_boundary_common_grid | 24 | 52.79% | 0 | unavailable | unavailable |
| Omega_background_restored_24 | leaf_window | 21 | 96.98% | 10 | 0.369 | 0.830 |
| Omega_background_restored_24 | leaf_edge_window | 19 | 97.22% | 8 | 0.427 | 0.822 |
| Omega_background_restored_24 | pot_rim_window | 24 | 90.86% | 12 | 0.510 | 1.460 |
| Standard_RAP_baseline_24 | foreground_boundary_common_grid | 24 | 11.27% | 0 | unavailable | unavailable |
| Standard_RAP_baseline_24 | leaf_window | 21 | 23.74% | 10 | 35.343 | 109.333 |
| Standard_RAP_baseline_24 | leaf_edge_window | 19 | 27.71% | 8 | 22.933 | 134.061 |
| Standard_RAP_baseline_24 | pot_rim_window | 24 | 29.06% | 12 | 49.426 | 125.878 |
| Standard_background_restored_24 | foreground_boundary_common_grid | 24 | 52.41% | 0 | unavailable | unavailable |
| Standard_background_restored_24 | leaf_window | 21 | 98.02% | 10 | 0.257 | 0.744 |
| Standard_background_restored_24 | leaf_edge_window | 19 | 98.20% | 8 | 0.309 | 0.615 |
| Standard_background_restored_24 | pot_rim_window | 24 | 91.86% | 12 | 0.483 | 1.582 |

固定leaf/edge窗口中恢复背景后的目标RGB投影pair中位数约0.6–0.8共享px，有真实前景内点证据，不能仅解释为新背景特征独占统计。它仍不是稠密叶面/物理叶缘GT精度；窗口内样本数与空窗口公开。

共同前景一像素内边界在恢复后支持约52%；它包括盆和蓝色物体边界，受到真实跨视角可见性/自然遮挡、目标Mask剔除、离散栅格与depth预测共同影响。**52%不是GT准确率，不等同物理叶片缺损比例。** ROI补充使用另一个固定seed，在相同1500query框架下sampling与主whole-FG slightly不同，故补充whole-FG值与主表52.69/92.03等会有小差别，未覆盖主表。

全部局部原始记录：[ROI_cross_depth_pairs.csv](evaluation/ROI_cross_depth_pairs.csv)、[ROI_independent_RGB_pairs.csv](evaluation/ROI_independent_RGB_pairs.csv)、[ROI_summary.csv](evaluation/ROI_summary.csv)、[ROI_protocol.json](evaluation/ROI_protocol.json)。

## 技术解释与边界

Confirmed：在固定24源帧、相同权重/seed/尺寸/Mask且原Mask内RGB不动的条件下，恢复Mask外原背景使两模型的前景相机与depth目标对应明显改善。黑色前景输入的失败是本数据当前条件的重要原因；标准模型并非仅靠换名称、低点数或小主点修正解决。

Unresolved：全208背景恢复是否保持效果、背景动态/曝光/其他植物干扰的鲁棒性、绝对公制尺度、全部薄叶稠密结构正确性，以及这些更好相机/初始点转化到最终3DGS训练多少收益。本轮没有执行全208背景恢复推理或GGGS重训，不能伪造该交付。

## 文件与复现

CPU执行独立 `scripts/geometry_targeted_control_eval.py --scope background`，之后 `geometry_targeted_control_supplement.py` 与 `geometry_targeted_control_report.py` 相同scope；使用既有 uv `.venv-fast3r`，不占GPU。新增评价和报告只存本control目录，不改旧03数值。再次刷新复用condition_*_results.json与fixed_foreground_matches，避免重复大量原始数组读取。

背景评价完全不读取historical Standard模型Mask或任何/tmp input cache；只读取本24条件的持久化Mask/depth/camera和既有foreground SIFT缓存。

[control_summary.csv](evaluation/control_summary.csv) / [JSON](evaluation/control_summary.json)、[cross_depth_pairs.csv](evaluation/cross_depth_pairs.csv)、[independent_RGB_pairs.csv](evaluation/independent_RGB_pairs.csv)、[relative_camera_pairs.csv](evaluation/relative_camera_pairs.csv)、[fixed_pair_manifest.csv](evaluation/fixed_pair_manifest.csv)、[evaluation_protocol.json](evaluation/evaluation_protocol.json)。来源为这些本地真实控制文件；本轮网络检索此前Bocha403，没有凭训练记忆生成实验结论。
