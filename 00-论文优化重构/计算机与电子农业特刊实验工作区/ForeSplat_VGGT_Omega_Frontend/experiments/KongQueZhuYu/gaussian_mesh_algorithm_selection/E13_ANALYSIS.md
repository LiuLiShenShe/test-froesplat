# E13：原生训练/模型的实际证据

|实验|原生迭代数|原生基元数|Raw PSNR|真FG PSNR|SSIM|LPIPS|原生训练min|峰值GiB|
|---|---|---|---|---|---|---|---|---|
|E13|30000|532735|22.895602263200207|18.298579782991467|0.8585752750245425|0.13440697106461114|344.8097364664078|3.4587135314941406|

真实原生模型：`/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/checkpoints/point_cloud/iteration_30000/point_cloud.ply`。

相同修正208视角的终态外观比较：

|模型|Raw PSNR|真前景 PSNR|SSIM|LPIPS|背景Alpha|
|---|---|---|---|---|---|
|E10|23.900519610857604|19.460451747175807|0.877555976406886|0.11708615999668837|0.3780385998543352|
|E11|22.261471765286952|17.59254995546233|0.8632166551855894|0.13734949192103857|0.9931412817767034|
|E13|22.895602263200207|18.298579782991467|0.8585752750245425|0.13440697106461114|0.999343250806515|

QGS原生TSDF的raw、共同组件后处理和官方top10后处理分别保留，避免把后处理差异当成表示收益：

|阶段|顶点|面|IoU|Chamfer px|内部缺失像素|相邻面起伏代理|
|---|---|---|---|---|---|---|
|official_top10_post|1144952|2187817|0.8932850995327328|28.954452132900272|1001.8028846153846|0.1343968438843797|
|post|1243483|2338160|0.8286969913596589|72.46629005293518|864.3269230769231|0.13558917201220985|
|raw|1444364|2516788|0.8013712077532962|86.74391847722283|779.8317307692307|0.1512733896725012|

原生Quadratic Primitive保留六个有符号尺度/曲率参数，普通XYZRGB中心点云不能代替。当前主运行采用官方Mip预设：depth_ratio=1、lambda_dist=10、lambda_normal=.015、增密阈值1，距离/法线起始分别3000/7000，增密至15K。多视角几何/NCC起始和终止均30K，官方严格分支在本次30K内没有贡献；无额外深度法线先验。

[原生运行记录](E13_QGS/E13_RUN_SUMMARY.md)；[原生Viewer指南](E13_QGS/NATIVE_VIEWER_GUIDE.md)、[参数](E13_QGS/actual_config.yaml)、[相机和源码适配](E13_QGS/PREPARATION_REPORT.md)保留。终态几何排序使用统一Mesh和实际视角。

官方法线项逐通道计算sum(1−normal*depth_normal)，等价于带权3−dot；当前wo_image_weight=True使用detach后的对数曲率权重，curvature_clamp_threshold=-1约束该权重而非基元几何曲率。额外的2λmean(weight)不贡献同一步参数梯度，但会改变日志标量；日志normal_loss不能直接解释成法线角误差。[实际公式与源码SHA](E13_QGS/NATIVE_NORMAL_OBJECTIVE.md)及[真实终态208图上的标量分解](E13_QGS/native_normal_objective_decomposition/complete.json)分别记录。原生核心未替换。

[实际Checkpoint数值状态与训练稳定性图](E13_QGS/native_stability/NUMERICAL_STABILITY.md)；[首轮OOM与原生恢复记录](E13_QGS/RECOVERY_REPORT.md)。原生参数中的NaN原样保留，逐位来源核验通过不等于参数全部有限，也不等于几何质量正常。普通XYZRGB导出若排除无效中心，须另列原生索引映射，不用它代替六尺度QGS原文件。

base预设两次真实OOM发生在11700和13301步，失败Checkpoint/PLY和实际成本单独保留。主Mip运行从相同VGGT初始化与208相机/RGB重新开始Seed0的30K，没有恢复base参数。配置变化及限制见[官方预设](E13_QGS/OFFICIAL_NATIVE_PRESETS.csv)、[适用范围](E13_QGS/NATIVE_PRESET_SCOPE.md)和[归档路径](E13_QGS/runs/base_expandable_oom_13301/RETAINED_FILE_PATHS.json)。不推广为所有QGS配置结论。

[失败导出实际读取核验](E13_QGS/failure_exports/20261109T084120Z_1535250/VALIDATED_DELIVERY_STATUS.json)。该失败PTH的ZIP64偏移超过真实文件长度，Python和原生PyTorch均无法读取，不是可恢复Checkpoint；独立原生PLY的精确字节长度正确，作为失败状态模型保留。捕获阶段标志不替代文件读取核验。

主Mip实际1K被动捕获另保留原生PTH与六尺度PLY：591,494个基元，7个原生无效基元原样保留，所有PLY字段逐位等于实际PTH；SH活跃阶数为1。这是中间模型，不能冒充正式30K；实际绝对路径见[1K来源核验](E13_QGS/retained_native_states/iteration_1000/NATIVE_EXPORT_PROVENANCE.json)与[全部原生文件索引](artifact_index/GAUSSIAN_PLY_INDEX.md)。

[统一Gaussian指标](comparison/ALL_GAUSSIAN_EXPERIMENTS.csv) · [统一Mesh指标](comparison/ALL_MESH_EXPERIMENTS.csv) · [同像素空间对应](comparison/reflection_mesh_correspondence/) · [原生全尺寸RGB/Alpha/Depth/Normal与局部图](comparison/visualizations/gaussian/)。

没有同株扫描；几何结论限于轮廓与跨视角代理。正式终态、全部阶段Mesh和文件索引已交付；综合结论见[最终算法选择报告](FINAL_RECONSTRUCTION_AND_MESH_SELECTION.md)，本报告保留该方法各阶段的测量与限制。

来源：[固定官方源码](https://github.com/will-zzy/QGS/tree/74d05c945e99fcaef7afe5a8831903be71ad9b55)及真实本地训练/模型/评价记录。博查HTTP403不可用；Context7解析没有返回该具体方法的匹配官方项目，官方源码直接核对成功。

同一208源视角的描述性配对变化（对照减基线）：

|对照路线|实际阶段|指标|均值变化|差值P10|差值中位|差值P90|方向改善视角比例|
|---|---|---|---|---|---|---|---|
|E13|raw|fg_mask_psnr|-1.161872|-3.094971|-0.323329|0.139885|0.197115|
|E13|raw|ssim|-0.018981|-0.034410|-0.012642|-0.001126|0.057692|
|E13|raw|lpips_alex|0.017321|-0.003395|0.004715|0.032707|0.201923|
|E13|raw|background_alpha_mean|0.621305|0.362070|0.625738|0.842602|0.000000|

视角来自同株、同序列、Seed0；分位数和改善比例描述局部变化，不是独立植株/重复训练的显著性。孔洞减少可能伴随过填充，不能单独按“改善比例”选算法。

