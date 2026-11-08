# 既有 E1 / E5 effective-opacity 删除单因素控制

2026-11-08。根代理合并五项审计后调度，两个既有 checkpoint 均在物理GPU1实际执行 source0/95/189 三个共同训练视角。仅删除 native `get_opacity_with_3D_filter < 0.05` 的 Gaussian 行；所有残留 xyz / SH / raw opacity / raw scaling / rotation / SG / filter_3D 参数逐值保留，没有训练、optimizer或新PLY写入。完整keep mask、原始点索引、每个保留参数数组SHA256与原checkpoint SHA均归档。全部208个runtime Camera数组、三个GT和Mask与原baseline逐位相同。

| 既有模型 | 原点数 → 剩余 | 删除比例 | ΔFull PSNR | Δ真MaskFG PSNR | Δ双侧5px边缘 PSNR | Δ内侧边缘 PSNR |
|---|---:|---:|---:|---:|---:|---:|
| E1 | 600458 → 434858 | 27.58% | −0.00243 | −0.00744 | +0.00683 | −0.03185 |
| E5 | 5153165 → 1806635 | 64.94% | −0.08434 | −0.09265 | +0.00162 | −0.04315 |

差值为同三源实际float render的“删除后−baseline”每帧指标均值，未与主12视角或旧208视角聚合混比。E1删除165600点，其中104434是raw≥.05但effective<.05；E5删除3346530点，其中3055537是该子集。它们分别对应前期17.39%/59.29%的统计，当前控制按所有effective<.05删除，还包括raw本身低于阈值的点。

| 模型 | BG alpha均值 before → after | BG alpha>.5比例 before → after | alpha>.5 Mask IoU before → after |
|---|---:|---:|---:|
| E1 | .907245 → .907186 | 92.7127% → 92.7020% | .356499 → .356523 |
| E5 | .993350 → .993328 | 99.4340% → 99.4325% | .341517 → .341507 |

删除后opaque背景几乎没有减少，叶片/边缘清晰度没有实质收益，E5真FG和梯度保留略差。这条阈值删除规则未成为现有模糊的修复。不能预先把低effective-alpha点说成完全无贡献，也不能从这三训练视角推断重训或泛化效果；本对照只检验这一次固定删除规则。

Controlled Verification 的 SSIM / Alex LPIPS 已另外补齐：只读取删除后6个已有1080×1920 floatRGB，`CUDA_VISIBLE_DEVICES=''`，现有Ω uv环境单个CPU Alex实例，缓存backbone与LPIPS v0.1校准权重；权重加载只允许本地文件，没有下载、GPU作业、重渲染或训练。baseline直接复用主12表中相同source0/95/189，原RGB/alpha `comparison.json` SHA前后不变。SSIM与主流程使用同scikit-image参数；LPIPS保持原始分辨率、float32、范围[-1,1]，没有缩放。

| 模型 | SSIM baseline → 删除后 | ΔSSIM | LPIPS baseline → 删除后 | ΔLPIPS（越低越好） |
|---|---:|---:|---:|---:|
| E1 | .772213 → .771441 | −.000772 | .252241 → .252666 | +.000425 |
| E5 | .726237 → .722091 | −.004146 | .353087 → .355235 | +.002148 |

感知指标也没有改善。baseline LPIPS来自原GPU评价，补充为CPU评价，同架构/权重/输入协议仍可能有小量设备舍入差异，未宣称CPU/GPU逐位相同。新增独立 [逐帧感知CSV](prune_effective_render/perceptual_cpu_supplement/per_view_perceptual_metrics.csv)、[感知均值CSV](prune_effective_render/perceptual_cpu_supplement/summary_perceptual_metrics.csv)、[离线权重SHA/协议与运行JSON](prune_effective_render/perceptual_cpu_supplement/perceptual_supplement.json)；原比较数字与原始数据均保持。

同次native raster统计也已保存，不额外重复渲染。E1删除后source0/95/189的positive radii比例为94.26%/92.40%/59.74%，median/p90/p99 `[9,23,68]`、`[10,25,68]`、`[15,36,79]`px，最大1947/955/1468px，positive中的>20px比例12.62%/15.42%/33.17%。E5相应比例22.00%/44.24%/28.70%，分位数 `[12,43,134]`、`[7,29,100]`、`[11,41,126]`px，最大3148/2462/2469px，>20px比例27.84%/16.44%/26.09%。这是native radius/frustum有效性，包含遮挡Gaussian，不能当真实可见surface比例。旧baseline未捕获radii，因此这里不能计算删除前后radii的变化，也不能单独把大半径证明为唯一根因。

E1运行99.862秒、render/export阶段peak allocated/reserved0.443/0.715GiB；E5运行99.698秒、1.341/1.869GiB。wall含Scene加载、参数选择与SHA、渲染/CPU导出，且原主baseline是12视角，这些数据不能用于宣称纯GPU加速。峰值从实际删除后开始记录，未覆盖checkpoint加载/子集选择的瞬时分配。两模型GPU进程均正常退出，已立即把GPU1交回根代理；后续指标与报告全为CPU。

入口：[prune_effective_render.py](prune_effective_render.py)、[同相机CPU评价](evaluate_effective_prune.py)。结果：[E1原始参数选择](prune_effective_render/E1/effective_opacity_subset_control.json)、[E5原始参数选择](prune_effective_render/E5/effective_opacity_subset_control.json)、[E1逐帧/均值对照](prune_effective_render/E1/evaluation/comparison.json)、[E5逐帧/均值对照](prune_effective_render/E5/evaluation/comparison.json)。每种方法三个source均有GT|baseline|pruned、native-alpha、边缘误差图与完整raw RGB/Mask/alpha/depth/normal。数据来自实际 [GGGS原生renderer](https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids/blob/23b418c2f0f19b30b251acad6fe13445dae968c7/gaussian_renderer/__init__.py) 与冻结checkpoint，不是模拟或从旧PNG外推。
