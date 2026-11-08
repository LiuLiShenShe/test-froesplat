# E1 / E5 / E6 实际 checkpoint 渲染与评价审计

2026-11-08，KongQueZhuYu。三个既有 iteration30000 checkpoint 均已实际重渲染，使用物理 GPU1；没有新增训练、重建模型推理、曝光校正或历史实验覆盖。模糊与叶片纹理损失在当前 checkpoint 的真实浮点输出中可以复现。旧评价的主要问题是把前景包围框称为 FG，以及部分展示图的视角标签不对应相同 sourceID；PSNR/SSIM 的实现本身没有造成这里的模糊。

主实验固定 208 个可用帧中的 12 个均匀序号 `[0,19,38,56,75,94,113,132,151,169,188,207]`，对应原始 sourceID `[0,19,38,57,76,95,114,133,152,170,189,209]`。三个方法的文件名、GT 浮点 RGB 与二值 Mask 逐源 `array_equal` 全部通过。原训练图像为 RGB 2160×3840，实际 `resolution=2` Camera / GT / render 为 **1080×1920**；不是 camera JSON 中保存的原 CameraInfo 分辨率。所有 208 个实际 runtime Camera 的 FoV、R/T、world_view / projection / full_proj、camera_center 均已保存。

| 既有模型 | 实际 PLY Gaussian 数 | 12-view Full PSNR | 真 Mask FG PSNR | 5 px 双侧边缘 PSNR | SSIM | Alex LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| E1 Ω + GGGS | 600458 | 17.5003 | 13.0309 | 10.6844 | 0.7810 | 0.2452 |
| E5 standard VGGT + GGGS | 5153165 | 13.1380 | 8.2013 | 10.2543 | 0.7346 | 0.3413 |
| E6 standard VGGT + 2DGS | 119836 | 13.1820 | 8.2047 | 10.2590 | 0.7021 | 0.3835 |

这些是 12 个训练视角的每帧指标均值，不能替代旧全 208 帧均值或测试集性能。PSNR 以实际 float32 clamped renderer RGB 和同一 GT 计算，未从 PNG 反量化。真 FG 指标仅选外部二值 Mask 的像素；Mask 最近邻缩放。边缘带为原 Mask 的 5 个实际渲染像素 dilation XOR erosion，另存仅前景一侧边缘的指标；形态学操作只用于评价，不改变模型输入或训练数据。SSIM 对三个方法统一调用 scikit-image：`channel_axis=2, data_range=1, win_size=7, gaussian_weights=False, use_sample_covariance=True`。LPIPS 统一使用缓存 AlexNet 的同一个评价实例，完整 1080×1920，范围 `[-1,1]`，没有缩放。当前评价环境为 torch2.10.0、torchvision0.25.0、scikit-image0.26.0、lpips0.1.4、Pillow13.0.0；库文档通过 Context7 核对，并参见 [scikit-image 官方指标文档](https://scikit-image.org/docs/stable/api/skimage.metrics.html)。

| 既有模型 | 前景梯度能量保留 | 前景内侧边缘梯度保留 | FG alpha均值 | BG alpha均值 | BG alpha>.5比例 | alpha>.5 Mask IoU | BG RGB平均绝对误差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 | 0.4186 | 0.2315 | 0.9953 | 0.8943 | 91.94% | 0.3373 | 0.00976 |
| E5 | 0.2521 | 0.1147 | 0.9686 | 0.9549 | 95.55% | 0.3204 | 0.00733 |
| E6 | 0.0882 | 0.0602 | 0.9988 | 0.9935 | 99.54% | 0.3202 | 0.00585 |

梯度为 Rec709 灰度、reflect Sobel/8 的幅值，保留率为 render / GT 的区域均值。能量也可能来自噪声，因此同时保存梯度 MAE，不能将更高能量直接解释为正确叶脉。这里实际图像与原生 320×320 同位置放大均显示 E1 叶片保留较多而细纹理仍明显丢失，E5 大量错误颜色/残缺，E6 极度平滑。放大框只由 GT 上部前景边缘确定，三模型共享同一像素框；它是叶缘候选区域，未伪造叶片实例标注。

alpha 直接来自同一次实际 native rasterization：GGGS `render()['mask']` 是 `rendered_alpha`，2DGS `rend_alpha` 是 `allmap[1:2]`。这里没有用 RGB 推测 alpha。黑色 RGB 背景仍有大量高 opacity，说明低 RGB 背景误差不足以证明背景为空；训练可以通过黑色但不透明的 Gaussian 拟合黑背景。Mask IoU/precision/recall 与 BG>.1/.5/.9 全部保留于逐帧表。这个现象支持需要审计透明度/背景约束与初始化几何，但不能单独确认它是模糊的唯一原因。

[全视图 GT | E1 | E5 | E6](render_error_analysis/common_evaluation_v1/source_0000_GT_E1_E5_E6.png)、[原生像素叶缘局部](render_error_analysis/common_evaluation_v1/source_0000_GT_E1_E5_E6_leafedge_native320.png)、[同次 native alpha](render_error_analysis/common_evaluation_v1/source_0000_E1_E5_E6_alpha.png) 已直接查看。每个 sourceID 都有 RGB、绝对误差/真FG误差/边缘误差、alpha、depth、normal 与局部图。完整误差图统一数值范围0..0.5；概览270×480缩放仅用于展示，指标与局部图使用真实像素。

GGGS median depth 直接来自实际 renderer。2DGS 的旧 baseline 使用 `depth_ratio=0`，公开 `surf_depth` 为 expected depth；本审计另外通过只读 observer 保存同次 raster `allmap[5]` 的实际 median depth，没有修改 RGB 调用或返回值。GGGS 原生 normal 为 Camera 坐标；2DGS 原生 `rend_normal` 已转世界坐标，本审计用实际 world_view rotation 转回 Camera 坐标后统一展示，原生与转换后 float normal 均保留。展示归一化不改变原始数组。深度使用共同数值色标，但三个场景存在尺度 gauge；不是公制深度、GT depth 或跨模型尺度配准，不能从颜色差估算毫米误差。

旧评价审计有四个明确结论：

1. Ω `tools/evaluate_gggs_experiments.py:201` 与 standard `evaluate_e6.py` 的 `fg_psnr` / `fg_ssim` 使用整个 Mask 包围框，包含背景。新表保留 `legacy_fg_bbox_psnr` 对照，并新增真正 Mask 区域指标，不能将两列直接混比。
2. 旧全图 PSNR / SSIM 与当前同 source 的 checkpoint 重渲染逐帧几乎完全一致：E1/E5/E6 最大 PSNR差均≤3.6e-15，SSIM差为0，旧 bboxPSNR也匹配。E1/E5 LPIPS差0；E6换到统一评价环境后最大差0.000144。旧保存 CSV、旧208聚合与本次12视角真实结果分别归档，见 [同12视角复现表](render_error_analysis/common_evaluation_v1/historical_same12_vs_checkpoint_rerender.csv)。这排除了旧 PNG 量化、展示图陈旧或 PSNR实现不同作为当前模糊的解释。
3. 旧 `select_fixed_views()` 按各自估计的相机轨迹选择 front/right/top 等标签，跨 E1 与 E5 相同标签未必同 sourceID。旧 `render_e6_views.py` 的8个 VIEWS已包含4个LEAVES，合并后实际只有8个不同视角，不能称12视角。当前使用固定原 sourceID。
4. 官方2DGS `training_report` 的 train PSNR仅检查训练 Camera 列表的5个索引 `[5,10,15,20,25]`，旧最终评价为全208帧。训练15.27与最终13.20的差异首先是评价集合不同；相机列表还可能经过训练时shuffle，不能擅自将这5个索引当作固定原始sourceID。仅此差异无法证明最终渲染接口错误。

当前本地 E6 不是独立官方 Git checkout，父仓库 HEAD不能冒充 hbb1 官方 commit。已从官方源码逐文件获取并保留本地/官方 SHA 和完整 diff，参见 [官方源码对照](render_error_analysis/source_reference/E6_OFFICIAL_SOURCE_COMPARISON.json)、[hbb1 renderer](https://github.com/hbb1/2d-gaussian-splatting/blob/main/gaussian_renderer/__init__.py)、[hbb1 train.py](https://github.com/hbb1/2d-gaussian-splatting/blob/main/train.py)、[官方 surfel CUDA](https://github.com/hbb1/diff-surfel-rasterization/blob/main/cuda_rasterizer/forward.cu)。本地 renderer、Camera、GaussianModel、graphics_utils 与当前官方源码逐字一致。surfel Python接口及所有已对照的数值CUDA/config/auxiliary源码也一致；唯一 header差异是 `<cstddef>/<cstdint>` includes 与末尾换行，无函数、阈值、矩阵或算法改变。

E6 本地 train/Scene/reader 有植物扩展，实际 baseline cfg 的 mask/gate/init模式全none、列表为空；mask预处理不执行，Scene返回原pcd，RGB训练分支仍L1+DSSIM，view weight为1、mask/bg项0、额外pruning none早返回。默认仍执行的新增 tracks解析增加元数据/CPU开销，图像均存在时不删帧，也不改原pcd与FoV。该核对证明当前 baseline 的渲染数值路径与官方参考一致，不恢复历史完整训练命令、编译参数或独立官方commit。GitHub API取submodule pin遇到403；静态官方源码可读取，本报告明确使用“当前 main源码参考”。实际执行 `.so` SHA为 `cf435e6207d578d09175fa47754beb071c47045f6127b75ef93e77d8fc3cb576`。

E1/E5 执行 GGGS 官方派生仓库的当前既有本地代码，renderer SHA为 `5a9c34c87de59d0293c9f5bee99e815b3dacd79646c8cdbbb29a15dea56368c3`，参见 [GGGS官方 renderer](https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids/blob/23b418c2f0f19b30b251acad6fe13445dae968c7/gaussian_renderer/__init__.py)。为避免 GGGS `Scene(load_iteration=...)` 无条件覆写旧模型的 `multi_view.json`，本审计在全新 audit-owned model_path 下只读 symlink 到原 PLY。旧 multi_view/cameras/cfg 文件存在者前后SHA一致；没有调用 optimizer 或训练保存。

| 运行 | wall 秒 | rendering/export阶段峰值 allocated / reserved GiB | checkpoint SHA256 |
|---|---:|---:|---|
| E1 12 views | 146.135 | 0.593 / 1.213 | `19f61faedca7935b75fcedcbdd9d81ec9a1281ec8c484f4179b25c3abe30bab8` |
| E5 12 views | 161.066 | 3.189 / 3.514 | `98974a5d38fd3c79f6b399a291e8ee9d9f977e4c2e32114134ceab0d9fe71469` |
| E6 12 views | 236.638 | 0.649 / 1.008 | `3a2e48206f1fa92786bfe870ea51e2adbbe8551bb4e5c29b292432c74c60ba18` |

wall包含Scene加载、native渲染、CPU NPZ/PNG序列化与checkpoint哈希；不是纯GPU renderer测速。torch峰值从Scene已加载后起算，未记录checkpoint加载瞬时峰值。GGGS使用Ω既有uv环境 torch2.10.0+cu128；2DGS使用已有uv环境 torch2.5.1+cu121。只把GT tensor放CPU节省内存，Camera矩阵与renderer始终GPU1，实际GT逐像素验证一致。统一评价记录178.020秒覆盖主指标与RGB/error/alpha/normal图导出，不包含随后最终shared-depth图导出，缺失范围已在metadata说明，不推算未测总时间。

主点小位移的CPU敏感性诊断已经实际完成，见 [translation表](render_error_analysis/principal_translation_v1/summary_translation_metrics.csv)。GT/Mask固定，existing render按desired主点−runtime中心正向右/下做bilinear平移，边界填黑。Ω Δc=[1.40625,1.395348837]px：Full/真FG/边缘PSNR变化为 −0.1075/−0.1153/+0.0166dB。standard saved Δc=.25/.25px：E5变化 +0.00134/+0.00017/+0.01252dB，E6 +0.00222/+0.00074/+0.02361dB。另独立truePIL Δc=1.853281853px：E5 −0.01523/−0.01739/−0.04805dB，E6 +0.00241/−0.00082/+0.04301dB。这个图像域近似说明小主点平移不能救回目前模糊，不是shader修复，不证明重训收益，也不排除别的focal/pose/初始化问题。

额外复用既有 E3 final post-prune **1036619**点 PLY，对 source0/95/189进行了3视角实际诊断；它与旧 saved iteration30000的1037020点是不同artifact。新 [E1/E3同3视角表](render_error_analysis/e3_postprune_diagnostic_v1/comparison_with_E1/diagnostic.json) 显示 E1→E3：FullPSNR18.133→15.431，真FGPSNR13.791→16.005，边缘PSNR11.071→9.314；BGalpha .907→.102、BGalpha>.5比例92.71%→8.86%、IoU .3565→.8373，但BG RGB MAE .00714→.07382。已有前景条件减弱opaque背景，同时牺牲全图/边缘并出现背景颜色误差；不能因此直接推荐maskloss，3视角也不能替代全208条件评价。

E3同次native radii额外统计：source0/95/189的radii>0比例为99.88%/99.24%/54.01%，positive radii的median/p90/p99分别 `[6,14,28]`、`[7,16,31]`、`[12,25,51]`px，max219/276/638px，>20px比例2.89%/4.24%/17.53%。这是native frustum/radius有效性，包含遮挡点，不能当作真实可见surface比例。主E1/E5/E6这次未保存radii数组/统计，不额外重读大PLY重复baseline；该限制明确保留。

完整入口为 [render_checkpoint_views.py](scripts/render_checkpoint_views.py)、[统一评价](scripts/render_evaluate_common.py)、[主点平移](scripts/render_principal_translation_diagnostic.py)、[E3诊断](scripts/render_e3_postprune_diagnostic.py)。36个主视角的raw NPZ、全部208 runtime Camera与JSON运行参数在 `render_error_analysis/checkpoint_rerender_v1/{E1,E5,E6}`，可直接读取浮点RGB/GT/Mask/alpha/median-depth/native-normal/Camera-normal。主raw约2.5GB、派生评价697MB、E3约166MB。所有失败网络检索与源码pin限制已单独说明，没有借用别的方法冒充。

E5比E1多约8.6倍Gaussian却更差，E6比E5更平滑且同MaskFG/边缘PSNR近似，没有足够证据把更换renderer或增加点数当作修复。最终根因需结合相机/几何链路、滤波后有效opacity与训练/致密化审计。本文件的主36视角真实渲染与一致评价已完成；单因素effective-alpha删除脚本先作为五项审计后的控制准备，随后已按根代理调度实际执行。

五项审计完成后，根代理已调度并完成该固定删除规则的实际E1/E5三视角控制，见 [独立控制报告](controlled_fixes/PRUNE_EFFECTIVE_RENDER_REPORT.md)。两模型分别删除27.58%/64.94%的Gaussian，实际同相机同GT对照没有救回模糊或opaque背景，E5真FGPSNR下降0.09265dB。新控制捕获同次native radii与潜在可见率，但它们只覆盖删除后的三个训练视角，不补造主36视角的baseline radii，也不泛化为完整序列结论。原主baseline数据与表保持原样；GPU1已释放。

控制验证的SSIM/AlexLPIPS另已用纯CPU、缓存权重补齐，[新增感知协议与逐帧结果](controlled_fixes/prune_effective_render/perceptual_cpu_supplement/perceptual_supplement.json) 保持原6个floatRGB和baseline同source，未重渲染、下载或覆盖旧comparison。E1 ΔSSIM−.000772/ΔLPIPS+.000425，E5−.004146/+.002148，感知质量也未改善；CPU与原GPU LPIPS的微小舍入差异单独标注。
