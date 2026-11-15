# 三株 C1 → NKSR → 初步植物几何测量实测结果

三株均已实际完成。交付6份三角Mesh（Raw/Post各3份）、三张四行四视角比较图和模型单位几何测量。使用既有C1结果、原生GGGS渲染、已验证NKSR环境/权重/兼容模块；没有重新训练、清理或调整跨植物重建参数。原始RGB行来自02-FFT原始照片，保留实际背景；与相机对应的整幅画布只按评价分辨率缩放，不掩膜或改变曝光。原黑底训练参考另存training_reference，逐帧路径及前景像素一致性核验见visualizations/reference_rgb/sources.json。

[完整图像与模型索引](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/experiments/C1_NKSR_Phenotype/summary/index.html) · [三株总览](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/experiments/C1_NKSR_Phenotype/summary/mesh_overview.png) · [Mesh统计](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/experiments/C1_NKSR_Phenotype/summary/mesh_results.csv) · [表型几何](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/experiments/C1_NKSR_Phenotype/summary/phenotype_results.csv) · [阶段资源](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/experiments/C1_NKSR_Phenotype/summary/resources.csv) · [核验](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/experiments/C1_NKSR_Phenotype/summary/verification.json)

| 植物 | 相机数 | NKSR输入点 | Raw顶点/面 | Post顶点/面 | NKSR实际耗时(s) | NKSR进程峰值显存(GiB) |
|---|---:|---:|---|---|---:|---:|
| 豆瓣绿1 | 200 | 9,305,174 | 4,184,851 / 8,296,092 | 4,181,184 / 8,291,877 | 25.23 | 16.71 |
| 红掌 | 199 | 6,247,267 | 2,977,658 / 5,901,339 | 2,975,328 / 5,898,770 | 17.97 | 12.39 |
| 长寿花3 | 198 | 7,717,462 | 4,954,486 / 9,804,274 | 4,951,241 / 9,800,829 | 24.68 | 20.15 |

NKSR耗时计入模型初始化/权重加载、固定参数求解和dual mesh提取，排除表面生成及后续颜色/后处理导出；阶段资源表另列整个阶段及入口含启动耗时。GPU峰值为按本PID采样的物理GPU1使用量，Torch allocated/reserved另存。

| 植物 | Post顶点 | Post面 | 几何高度代理(模型单位) | 水平跨度代理(模型单位，两轴) | 物理株高/冠幅 | 叶片测量 |
|---|---:|---:|---:|---|---|---|
| 豆瓣绿1 | 4,181,184 | 8,291,877 | 0.552748 | 0.518843 / 0.379145 | PHYSICAL_SCALE_NOT_AVAILABLE | 缺Leaf ID、叶基叶尖和边界 |
| 红掌 | 2,975,328 | 5,898,770 | 0.785274 | 0.674715 / 0.536904 | PHYSICAL_SCALE_NOT_AVAILABLE | 缺Leaf ID、叶基叶尖和边界 |
| 长寿花3 | 4,951,241 | 9,800,829 | 0.639216 | 0.595302 / 0.498005 | PHYSICAL_SCALE_NOT_AVAILABLE | 缺Leaf ID、叶基叶尖和边界 |

**上述尺寸是全Mesh包含花盆、蓝色参考物及可能伪影的探索性PCA范围，不是正式植物株高或冠幅。** PCA主轴没有被当成重力方向。全Mesh表面积包含花盆、参考物、粗糙和可能双层表面，不能当成叶面积。XYZ包围盒、PCA三轴、表面积、原前景候选点范围及Raw/Post变化都记录在各株geometry.json及总CSV。物理值留空，不根据未知盆径或历史表格假设厘米尺度。

| 植物 | 阶段 | 活动耗时(s) | 入口总墙钟(s) | CPU(s) | 峰值RSS(GiB) | 峰值显存(GiB) | 读/写(MiB) | 最大合计I/O(MiB/s) |
|---|---|---:|---:|---:|---:|---:|---|---:|
| 豆瓣绿1 | GGGS表面提取 | 266.39 | 未单独计时 | 215.83 | 2.395 | 0.99 | 616.5 / 229.9 | 19.688 |
| 豆瓣绿1 | NKSR重建 | 136.07 | 304.61 | 155.75 | 5.821 | 16.71 | 478.7 / 612.9 | 19.333 |
| 豆瓣绿1 | Mesh评价 | 130.06 | 589.40 | 198.52 | 6.609 | 0.81 | 776.5 / 4.3 | 19.862 |
| 豆瓣绿1 | 表型计算 | 50.78 | 51.42 | 15.28 | 1.482 | — | 703.0 / 0.0 | 19.801 |
| 豆瓣绿1 | 原始RGB参考整理(CPU) | 4.44 | 未单独计时 | 3.08 | 0.157 | — | 11.1 / 4.1 | 4.721 |
| 红掌 | GGGS表面提取 | 270.93 | 273.60 | 227.74 | 2.100 | 0.87 | 544.5 / 155.8 | 19.688 |
| 红掌 | NKSR重建 | 98.53 | 102.20 | 110.13 | 4.553 | 12.39 | 404.0 / 436.1 | 19.319 |
| 红掌 | Mesh评价 | 82.37 | 85.37 | 137.08 | 5.036 | 0.62 | 532.6 / 6.6 | 19.826 |
| 红掌 | 表型计算 | 35.63 | 36.13 | 10.59 | 1.124 | — | 497.3 / 0.0 | 19.826 |
| 红掌 | 原始RGB参考整理(CPU) | 4.98 | 未单独计时 | 3.48 | 0.157 | — | 16.6 / 5.2 | 6.549 |
| 长寿花3 | GGGS表面提取 | 282.52 | 285.23 | 228.66 | 2.249 | 0.78 | 534.7 / 190.7 | 19.711 |
| 长寿花3 | NKSR重建 | 154.61 | 158.59 | 177.77 | 6.540 | 20.15 | 439.0 / 725.2 | 19.316 |
| 长寿花3 | Mesh评价 | 160.22 | 163.47 | 231.98 | 7.528 | 0.60 | 814.0 / 6.2 | 19.815 |
| 长寿花3 | 表型计算 | 57.08 | 57.68 | 16.84 | 1.703 | — | 799.6 / 0.0 | 19.815 |
| 长寿花3 | 原始RGB参考整理(CPU) | 4.74 | 未单独计时 | 3.22 | 0.157 | — | 15.4 / 4.9 | 6.596 |

活动时间覆盖阶段主体操作；独立入口时间含uv/Python/动态库冷启动及进程退出（首株表面入口未单独计时）。CPU记录为阶段主体进程时间，RSS为该阶段进程峰值；没有混入历史NKSR或C1清理耗时。大文件与图像流共用阶段20 MiB/s限速器，记录应用层字节而非设备物理峰值。

首株Mesh评价在结果写完后发生原生库解释器退出延迟，最终自行正常退出；未发送成功的终止信号，也未重算或修改几何。之后评价入口补齐与历史脚本一致的完成后直接退出处理。该额外墙钟时间保留在入口统计，不伪装成NKSR计算或删除此记录。详见首株visualizations/exit_deviation.json。

**豆瓣绿1：** 直接查看四视角四行图及crop_0000法向着色图：主要厚叶片、叶尖和花盆的大轮廓保留，叶片已形成三角面；但叶面明显起伏、碎屑和折叠状结构，叶缘较锯齿，盆口/下层叶片区域粗糙堆叠，叶柄连接不能据此确认为完整。Raw与Post视觉几乎一致，50面阈值并未修复这些较大结构。未见大面积外部黑云变成整块网格；下方蓝色方块在RGB参考和C1中就存在，被一同网格化，不能全部算作NKSR新增漂浮伪影。它也污染整Mesh尺寸，不能将全Mesh的PCA范围当作植物株高。仅凭四个投影视角，不能排除双层叶片或盆口错误连接。

[四行四视角图](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/experiments/C1_NKSR_Phenotype/DouBanLv1/visualizations/mesh_comparison.png) · [几何着色图](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/experiments/C1_NKSR_Phenotype/DouBanLv1/visualizations/geometry_comparison.png) · [Raw Mesh](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/experiments/C1_NKSR_Phenotype/DouBanLv1/mesh/raw.ply) · [Post Mesh](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/experiments/C1_NKSR_Phenotype/DouBanLv1/mesh/post.ply)

**红掌：** 直接查看四视角四行图及crop_0000法向着色图：大叶片和红色佛焰苞的大轮廓已形成三角面，多数可见叶尖存在；叶面起伏和叶缘缺口明显，茎叶密集区出现粗糙堆叠和连接成片的表面，部分叶柄外形可见但连续性、彼此独立性不能确认。盆口与下部植株区域混杂，不能据四个投影排除错误桥接或双层叶片。第三视角花盆边缘已有C1残留缺损，Mesh仍有坑洞。Raw与Post主体几乎相同，<50面后处理没有修复较大异常曲面。未见大面积外部黑云形成整体外壳；下方蓝色块在参考RGB/C1中已经存在，不是NKSR凭空新增的物体。

[四行四视角图](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/experiments/C1_NKSR_Phenotype/HongZhang/visualizations/mesh_comparison.png) · [几何着色图](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/experiments/C1_NKSR_Phenotype/HongZhang/visualizations/geometry_comparison.png) · [Raw Mesh](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/experiments/C1_NKSR_Phenotype/HongZhang/mesh/raw.ply) · [Post Mesh](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/experiments/C1_NKSR_Phenotype/HongZhang/mesh/post.ply)

**长寿花3：** 直接查看四视角四行图及crop_0000法向着色图：外侧厚叶的大轮廓和花盆存在，部分叶缘可辨；中央密集叶片/花蕾区形成明显粗糙、团块和折叠堆叠，不能将其视为独立完整器官。第二及第四视角上方细花柄在C1中已经断失，花蕾呈悬浮状，NKSR没有恢复其连接；其余细柄被遮挡或混入表面，未验证连续性。法向着色显示起伏并非只有颜色噪点，盆口与叶簇相接处存在桥接风险；不能排除局部双层或假封闭。Raw与Post视觉几乎不变，较大异常保留。没有看到大范围黑云形成外部壳体；下方蓝色块为原参考物，其几何较粗糙，不能算新生成物体。

[四行四视角图](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/experiments/C1_NKSR_Phenotype/ChangShouHua3/visualizations/mesh_comparison.png) · [几何着色图](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/experiments/C1_NKSR_Phenotype/ChangShouHua3/visualizations/geometry_comparison.png) · [Raw Mesh](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/experiments/C1_NKSR_Phenotype/ChangShouHua3/mesh/raw.ply) · [Post Mesh](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/experiments/C1_NKSR_Phenotype/ChangShouHua3/mesh/post.ply)

**下一步判断。**

1. C1清理后的GGGS可以沿现有Median Depth→有向表面点→官方ks流程进入NKSR，三株已完成；这是三株上的流程稳定性证据，不是对其余16株的质量保证。
2. 叶片、边缘及细结构的实际表现见上方逐株直接看图记录；成功生成三角面或减少连通分量不等于植物器官完整。四视角可见性也不能证实所有叶片都已完整恢复。
3. NKSR会对稠密观测进行隐式表面插值，可能保留或连接输入已有的错误深度；具体异常见逐株记录。Post只删<50面连通块，不能修复大块错误连接、双层叶面或植株/花盆粘连。
4. 现在可计算全Mesh XYZ尺寸、PCA范围、表面积、连通性、边界/非流形边，以及全前景候选点的探索性跨度。正式株高/冠幅仍缺植物专属区域、独立竖直方向和植物基准。
5. 厘米级测量需同株同次采集的真实已知长度、其两个三维对应点及单位/不确定度记录；再以真实长度除以模型距离得到比例。叶片测量还需要实例ID、叶基、叶尖、边界/面集合和测量方向。
6. 路径、相机数量及世界坐标适配已支持其他Scene；其余植物可以复用相同代码和公式。当前驱动限制为这三株，未自动启动其余16株。扩展前应保留逐株相机核验、资源检查和几何审查。

**固定协议与证据。** 全训练视角、训练分辨率二分之一；Alpha>=0.5，深度跳变0.05，既有差分法向朝向相机；采样robust bbox diagonal/1024，重建/512；官方ks与原兼容模块；迭代2000、tol1e-5、fused_mode、MISE1、trim、max_points1000000均固定。既有原始源码及GGGS/C1输入戳保持不变，3项适配回归与交付核验通过。原始及Post网格各自保存，未使用最大连通块策略。

信息来源：本轮真实运行记录、直接查看对比图、仓库历史源码及表型证据审计。博查HTTP403，Context7无NKSR匹配项；未用外部摘要代替本次实验结果。
