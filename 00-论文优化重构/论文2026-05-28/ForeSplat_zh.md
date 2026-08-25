# ForeSplat：面向低成本植物表型的前景感知 2D Gaussian Splatting

## 研究亮点

- ForeSplat 将 2DGS 重建目标与植物性状测量对象对齐。
- 前景 RGB 监督将泄漏能量比从 1.2201 降至 0.0190。
- 该流程完成了两类采集场景下的 20 个 RGB 多视角序列。
- 虚拟性状测量与人工测量高度一致，R² 最高达到 0.9879。
- 软视角加权在保留几何覆盖的同时减少 Gaussian 基元数量。

## 摘要

快速、非破坏性地测量植物结构性状，是智能育种、设施栽培和精准管理的重要基础。人工表型测量费时费力，二维图像又难以充分解析重叠叶片、遮挡器官和冠层几何。多视角三维重建为低成本表型获取提供了替代方案，但现有神经辐射场和 Gaussian Splatting 流程通常优化整场景外观，使花盆、基质、支架和背景进入重建模型。这种错位会限制后续网格提取和虚拟表型测量。本文提出 ForeSplat，一种前景感知 2D Gaussian Splatting 工作流，将植物掩膜从后处理线索前移到重建目标中。FSAM3 结合频域帧质量筛选、文本提示植物分割和前景精炼，生成多视角先验，并用于前景初始化、RGB 监督、透明度约束、视角加权、Gaussian 剪枝、TSDF 网格化和尺度恢复。在 20 个 RGB 序列和 21 株盆栽植物中，FSAM3 的 F1-score、mIoU 和 HD95 分别达到 98.3%、97.9% 和 41.4 px。消融分析显示，前景 RGB 监督将掩膜外非黑比例从 0.9908 降至 0.0294，将泄漏能量比从 1.2201 降至 0.0190。ForeSplat 达到 PSNR = 31.09 dB、SSIM = 0.9711 和 LPIPS = 0.0365，相比标准 2DGS 将训练时间和网格化时间分别降低 60.94% 和 65.17%。株高、冠幅、叶长和叶宽的虚拟测量与人工测量高度一致，R² 分别为 0.9878、0.9879、0.9738 和 0.8999。结果表明，普通 RGB 成像可作为室内和半受控复杂背景下低成本植物级三维表型监测的可复用途径。

**关键词：** 表型；重建；分割；Gaussian Splatting；RGB 成像；网格；性状

---

## 1. 引言

植物表型分析是连接基因型、环境响应和农艺表现的重要技术环节。株高、冠幅、叶长和叶宽等结构性状常用于育种筛选、栽培管理和生长状态评估，但人工测量通常依赖接触式操作，通量低且容易受到操作者经验影响。基于图像的高通量表型系统因此成为智能农业中的重要方向 [1-6]。与二维图像相比，三维表示能够记录器官空间位置、遮挡关系和冠层体量，更适合分析重叠叶片、复杂冠层和非平面结构 [7-12]。然而，植物并不是规则、刚性且纹理丰富的工程物体。薄叶、低纹理、重复纹理、局部遮挡、花叶混合和背景相似颜色都会使低成本三维重建和后续性状提取变得困难。

现有三维植物表型研究已从传统 SfM/MVS 和深度传感器逐渐扩展到神经渲染与显式 Gaussian 表示。LiDAR、结构光和深度相机能够提供高精度点云，但设备成本、校准复杂度和部署门槛限制了其在设施园艺和大规模育种中的普及 [13-15]。消费级 RGB 相机采集灵活、成本低，但 SfM/MVS 对视角覆盖、图像清晰度和叶片纹理高度敏感，常在薄叶和遮挡区域产生孔洞、噪声或边界模糊 [16-19]。NeRF 通过连续体积辐射场改善了植物重建质量，并已被用于田间和室内植物几何评估 [20-26]。3D Gaussian Splatting (3DGS) 进一步以显式 Gaussian primitives 表示场景，在渲染效率和可编辑性上具有优势 [27-29]。Plant3R、PlantGaussian、Cotton3DGaussians 和对象中心 3DGS 等近期研究表明，Gaussian 表示正在进入植物结构重建和性状分析场景 [30-33]。这些进展说明神经渲染正在成为低成本三维植物表型的重要工具，但也留下了一个面向农业应用的关键问题：重建出来的三维对象是否就是需要测量的植物对象。

这一问题在盆栽或设施场景中尤其突出。标准 NeRF、3DGS 和 2D Gaussian Splatting (2DGS) 通常优化整张图像的视觉重建，因此模型会同时学习植物、花盆、基质、桌面、支架和背景。对于新视角合成，这种整场景重建是合理目标；对于株高、冠幅和叶片尺寸测量，它却会使非植物几何进入网格提取和外包范围计算。LCR-GS 从 3DGS 场景中提取单株温室甜瓜，说明下游性状提取需要干净、可分析的植物表示 [34]。IPENS 将 SAM2 生成的二维掩膜提升到 NeRF 三维空间，用于水稻和小麦器官点云提取，说明可提示分割与辐射场可以降低标注负担 [35]。Gaussian Grouping 和 SAGA 等工作进一步说明，二维基础模型语义可以被蒸馏或提升到 Gaussian 空间，但这些方法主要面向通用场景分割和编辑，而非性状测量目标本身 [36,37]。这些研究共同强调了从场景级重建到植物级表示的必要性，但多数流程仍把对象分离放在重建之后。若背景在训练阶段已经获得稳定 Gaussian 容量，后处理掩膜剪枝往往难以完全消除其对植物网格和虚拟测量的影响。

本文的基本出发点是：农业表型测量中的掩膜不应只是后处理过滤器，而应参与定义三维优化问题。我们提出 ForeSplat，一种面向多物种盆栽植物的前景感知 2DGS 表型流程。与普通场景重建不同，ForeSplat 将初始化点、RGB 监督域、透明度约束、视角质量权重和 Gaussian 清理都绑定到由掩膜定义的植物前景。这样，训练阶段的主要光度梯度来自植物像素，模型容量优先分配给待测植株，而不是花盆、桌面和背景。2DGS 的平面 Gaussian 基元适合表达叶片这类薄表面结构 [28]，而 ForeSplat 进一步使这种表面表达服务于仅含植物的测量对象。

为实现这一目标，本文首先设计 FSAM3 作为重建导向的植物前景先验生成流程。该流程不是为了声明通用植物分割最优，而是将 FFT 频域帧质量筛选、SAM3 文本提示分割和 PCA 主前景精炼结合起来，生成跨视角对齐、边界较稳定、适合 2DGS 训练的掩膜。随后，ForeSplat 使用多视角掩膜一致性过滤 COLMAP 稀疏轨迹，在 2DGS 中引入前景 RGB 监督、alpha 掩膜损失和背景不透明度损失，并通过视角质量感知软损失加权保留低质量但可能具有几何覆盖价值的视角。训练后，掩膜引导多线索 Gaussian 剪枝和 TSDF 网格化将仅含植物的 Gaussian 表示转换为可测量网格。

本文旨在建立一个面向室内和半受控复杂背景的低成本、可复用、以表型测量为目标的三维 RGB 工作流，而不是无限外推的通用植物重建模型。本文贡献如下：

1. 提出 ForeSplat，将标准 2DGS 从整场景视觉重建改写为由掩膜定义的植物对象重建，使重建目标与株高、冠幅和叶片尺寸等农业表型对象对齐。

2. 提出 FSAM3 植物前景先验生成流程，将 FFT 频域质量筛选、SAM3 文本提示分割和 PCA 主前景精炼组合为重建导向掩膜管线，为多视角 2DGS 提供边界稳定、文件对齐和跨物种可用的前景约束。

3. 通过前景轨迹初始化、前景 RGB 监督、alpha/背景不透明度约束、软视角权重和掩膜引导 Gaussian 剪枝，构建从普通 RGB 序列到仅含植物网格的端到端表型流程。

4. 在 20 个多视角序列和 21 株植物上验证 ForeSplat，并通过外部基线、受控消融、紧凑化评估、网格结构评估和人工-虚拟测量比较分析其应用可行性与边界条件。

---

## 2. 材料与方法

### 2.1 研究设计与总体流程

ForeSplat 面向一个明确的农业表型任务：从普通多视角 RGB 图像生成可用于株高、冠幅和叶片尺寸测量的仅含植物网格。流程围绕“先定义待测植物对象，再进行三维重建”展开。第一，FSAM3 对原始帧进行质量筛选、植物前景分割和主组件精炼，得到与训练视图对齐的掩膜。第二，COLMAP 估计相机位姿和稀疏点轨迹，并通过多视角掩膜一致性过滤前景初始化点。第三，植物感知 2DGS 将 RGB 损失限制在植物前景区域，并用 alpha 掩膜损失和背景不透明度损失约束透明度场，使重建对象与表型测量对象一致。第四，视角质量感知软损失加权保留所有视角，仅调节不同视角对训练的贡献。第五，掩膜引导多线索 Gaussian 剪枝和 TSDF 网格提取输出紧凑、可测量的植物网格。总体流程及各模块之间的输入输出关系见图 1。

**图 1｜ForeSplat方法总览。** 原始多视角图像经过 FSAM3、COLMAP、前景对象 2DGS 优化、软视角加权、掩膜引导 Gaussian 剪枝、TSDF 网格化和表型测量，最终输出仅含植物的网格与虚拟表型值。

### 2.2 数据集、采集场景和样本用途

为覆盖盆栽植物表型测量中的常见结构差异，数据集包含 20 个多视角 RGB 序列，涵盖宽叶、低矮冠层、重叠叶、紧凑冠层、花叶混合、厚叶、细纹理和密集遮挡等条件。图像使用 iPhone 14 Pro Max 采集，视频分辨率为 1080 × 1920，帧率为 60 fps。采集场景包括固定装置辅助采集和复杂室内背景手持环绕采集，以同时评估半受控采集和更接近实际室内部署的复杂背景条件。20 个序列全部用于完整流程验证；表型统计以植株为单位，共 21 株植物，其中叶长和叶宽每株测量 3 片代表性叶片，因此叶片性状 n = 63。数据覆盖和端到端执行情况汇总于表 1。

**表 1｜数据覆盖与端到端工作流执行情况。** 表中汇总两类采集场景的数据规模、有效帧数、SfM 注册视角和流程成功率。

| 场景 | 物种/品种标签数 | 样本数 | 原始帧数 | 有效帧数 | SfM 注册视角 | 成功样本数 | 成功率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 固定装置辅助采集 | 8 | 10 | 2502 | 2104 | 2040 | 10 | 100% |
| 复杂背景采集 | 7 | 10 | 2500 | 2113 | 2089 | 10 | 100% |

样本覆盖宽叶、低矮冠层、重叠叶片、紧凑冠层、花叶混合、厚叶、细纹理和密集遮挡等结构条件，用于评估 ForeSplat 在不同冠层形态和背景复杂度下的适用性。

### 2.3 FSAM3：频域-空间植物前景先验

FSAM3 的目标是为三维表型重建提供稳定的植物前景先验，而不是作为独立的通用分割模型。其输入为每株植物的原始多视角 RGB 帧，输出与训练图像一一对齐的二值掩膜、RGBA alpha 图像和仅含前景的 RGB 图像。它由三步组成：FFT 帧筛选、SAM3 文本提示分割和 PCA 主组件精炼。

#### 2.3.1 FFT 帧质量筛选

模糊、失焦和低纹理帧会影响 SfM 位姿估计、SAM3 分割边界和后续 2DGS 优化。我们对每帧计算二维 FFT 幅值谱，并定义高频能量比：

\[
Q_{\text{FFT}}(I)=
\frac{\sum_{(u,v)\in H}|F(u,v)|}{\sum_{(u,v)\in \Omega}|F(u,v)|}.
\]

其中，\(F(u,v)\) 为频率 \((u,v)\) 的幅值，\(\Omega\) 表示完整频域，\(H\) 表示高频带。每个序列独立计算 \(Q_{\text{FFT}}\) 分布，并以第一四分位数作为样本自适应阈值。低于阈值的帧不进入后续分割和 COLMAP。

#### 2.3.2 SAM3 文本提示分割

质量筛选后的帧输入 SAM3。可提示分割模型已在自然图像和视频分割中显示出较强的通用前景定位能力 [38,39]，并被逐步用于植物器官、点云和三维场景分割任务 [35,40-44]。本文比较五个文本提示：P1 `green plant` 用于宽泛绿色植物区域，P2 `entire plant excluding pot` 用于不含花盆的完整植物体，P3 `leaves and stems` 用于叶片和茎等地上部器官，P4 `crop seedling` 用于小型或幼苗形态，P5 `plant body without background` 用于去除背景的完整植物前景。P2 被设为重建默认提示，因为花盆和土壤会污染仅含植物的 Gaussian 表示，且株高测量需要把容器与植物几何区分开。对每个视角 \(i\)，SAM3 输出二值掩膜 \(M_i\in\{0,1\}^{H\times W}\)。

#### 2.3.3 PCA 主前景精炼

SAM3 输出可能包含小碎片、孔洞或不稳定假阳性。FSAM3 首先使用 5×5 椭圆核进行形态学闭运算，然后执行 8 连通域分析并移除面积低于图像面积 0.5% 的组件。当仍存在多个大组件时，依据组件边界框在序列中的位置稳定性进行 PCA 选择，保留跨视角最稳定的主前景。该步骤的目标是获得与训练视图文件严格对齐、边界相对稳定的重建先验。

### 2.4 植物感知 2DGS：前景对象训练目标

如图 2 所示，ForeSplat 将 FSAM3 掩膜前移到 2DGS 的初始化、优化和后处理三个环节，使重建目标从整场景外观转向植物本体：前景轨迹初始化和前景 RGB 监督负责收束优化对象，alpha/背景不透明度约束与视角质量软加权负责稳定训练，mask-guided pruning 则清理冗余基元并导出仅含植物的 TSDF 网格。

**图 2｜ForeSplat 的 2DGS 算法改造示意。** 该图概述 COLMAP 相机、FSAM3 掩膜、前景 RGB 监督、alpha/背景不透明度约束、视角质量软加权、mask-guided pruning 和 TSDF 网格提取之间的关系，突出 ForeSplat 对训练目标和优化策略的改写。

#### 2.4.1 相机位姿估计与前景轨迹初始化

相机内外参和稀疏三维点轨迹由 COLMAP 增量 SfM 流程估计 [16]。标准 2DGS 从全部稀疏点初始化 Gaussian，因此背景点在优化开始前已经进入表示。ForeSplat 使用多视角掩膜一致性过滤稀疏点。设稀疏点 \(X_j\) 在视角集合 \(V_j\) 中可见，其在视角 \(i\) 的投影为 \(\pi_i(X_j)\)，则：

\[
\operatorname{Keep}(X_j)=1,\quad
\text{if}\quad
\frac{1}{|V_j|}\sum_{i\in V_j}M_i(\pi_i(X_j))\geq \tau_{\text{track}}.
\]

前景轨迹初始化要求稀疏点至少被 3 个视角观测，\(\tau_{\text{track}}=0.9\)，掩膜膨胀像素数为 0。

#### 2.4.2 前景 RGB 监督与透明度约束

标准 2DGS 在完整图像域 \(\Omega\) 上优化 RGB 重建 [28]。对于植物表型任务，这会使花盆、桌面和背景与待测植株竞争模型容量。ForeSplat 因此将光度监督重新定义到植物前景区域。标准整图 RGB 损失为：

\[
L_{\text{rgb-full}}=
\frac{1}{|\Omega|}\sum_{p\in\Omega}\|R(p)-I(p)\|_1 .
\]

ForeSplat 将 RGB 损失限制在植物前景：

\[
L_{\text{rgb-fg}}=
\frac{1}{|\Omega_{\text{fg}}|}
\sum_{p\in\Omega}M(p)\|R(p)-I(p)\|_1,\quad
\Omega_{\text{fg}}=\{p|M(p)=1\}.
\]

进一步加入 alpha 掩膜损失和背景不透明度损失：

\[
L_{\text{mask}}=\frac{1}{|\Omega|}\sum_{p\in\Omega}|A(p)-M(p)| ,
\]

\[
L_{\text{bg}}=
\frac{1}{|\Omega_{\text{bg}}|}
\sum_{p\in\Omega}(1-M(p))A(p),\quad
\Omega_{\text{bg}}=\{p|M(p)=0\}.
\]

完整优化目标为：

\[
L_{\text{core}}=
L_{\text{rgb-fg}}+\lambda_{\text{mask}}L_{\text{mask}}
+\lambda_{\text{bg}}L_{\text{bg}}+L_{\text{reg}},
\]

其中 \(L_{\text{reg}}\) 包括 2DGS 深度畸变损失和法向一致性损失。本文使用 \(\lambda_{\text{mask}}=0.08\)、\(\lambda_{\text{bg}}=0.02\)，掩膜损失类型为 `l1_dice`，忽略掩膜边界 2 px，掩膜损失从第 500 次迭代启动并在 1500 次迭代内预热。

### 2.5 视角质量感知软损失加权

多视角植物序列中，一些视角可能存在轻微模糊、反光或局部遮挡，但它们仍可能覆盖某些仅在狭窄角度可见的薄叶结构。直接删除低质量帧会破坏角度覆盖。ForeSplat 使用软加权：

\[
L_{\text{rgb-fg-soft}}=
\frac{\sum_i q_i L_{\text{rgb-fg}}(i)}{\sum_i q_i}.
\]

其中 \(q_i\) 为第 \(i\) 个视角质量权重。ForeSplat 读取 H-VQG 软权重文件，使用 `view_weight_mode=rgb_only`，权重范围限制在 0.6-1.0。质量分数综合掩膜覆盖率、掩膜边界锐度和前景 RGB 对比度。

### 2.6 掩膜引导多线索 Gaussian 剪枝

训练后期仍可能存在边界附近弱支撑 Gaussian 基元。对每个 Gaussian \(g_j\)，ForeSplat 计算：

\[
\operatorname{Score}(g_j)=
\alpha M_j+\beta O_j+\gamma V_j+\delta B_j+\eta C_j ,
\]

其中 \(M_j\)、\(O_j\)、\(V_j\)、\(B_j\) 和 \(C_j\) 分别表示掩膜投影一致性、透明度、可见视角数、颜色/亮度正常性和局部拓扑线索。剪枝模块使用 `pruning_mode=mask`，从 18,000 次迭代开始，每 3,000 次迭代执行一次，主要阈值包括不透明度阈值 0.005、亮度阈值 0.01、掩膜阈值 0.45、最大视角数 12、最大移除比例 0.03 和掩膜得分权重 3.0。

### 2.7 网格提取、尺度恢复与表型测量

为了将仅含植物的 Gaussian 表示转换为可交互测量的表型对象，本文通过深度渲染和 TSDF 式融合生成显式网格。TSDF 融合和 Marching Cubes 是从多视角深度或隐式场中提取显式表面的经典路线 [45,46]。对体素中心 \(x\)，融合距离为：

\[
D(x)=\frac{\sum_c w_c(x)d_c(x)}{\sum_c w_c(x)} ,
\]

其中 \(d_c(x)\) 为相机 \(c\) 下的截断有符号距离，\(w_c(x)\) 为融合权重。零水平集通过 Marching Cubes 提取。本文比较标准 TSDF、较小截断距离和边界后处理。尺度通过花盆直径这一已知物理参照恢复；尺度恢复后提取株高、冠幅、叶长和叶宽，并与人工测量比较。

### 2.8 评价指标

分割使用 F1-score、mIoU 和 HD95。前景重建使用 PSNR_fg、SSIM_fg、LPIPS_fg，以及两个背景泄漏指标：掩膜外非黑比例均值和泄漏能量比均值。PSNR、SSIM 和 LPIPS 分别反映像素误差、结构相似性和感知相似性，已广泛用于神经渲染和三维重建评价 [47-49]。仅含前景标准定义为：

\[
\text{outside}<0.05,\quad \text{leakage}<0.10.
\]

表型测量使用 MAE、RMSE、MAPE、偏差和 Pearson \(R^2\)。偏差定义为虚拟测量减人工测量。

---

## 3. 结果

### 3.1 FSAM3 提供稳定的多视角植物前景先验

FSAM3 在多物种、多结构盆栽植物上提供了稳定的重建前景先验。20 个样本均生成可用于重建的植物前景掩膜。P2 和 P5 在跨物种样本中最稳定；P1 容易包含绿色背景，P3 在部分粗茎或花叶混合样本中漏分，P4 对成熟植株欠分割。PCA 主前景精炼将平均组件数从每帧 12.4 个降至 4.1 个，降幅 67%，并在 98.2% 的帧中保留主导植物区域。数据覆盖、提示词差异、掩膜精炼和与 SEEM 的分割比较汇总于图 3。

**图 3｜数据覆盖与 FSAM3 前景先验质量。** a，固定装置辅助采集和复杂背景手持采集的代表性图像，并展示宽叶、紧凑冠层、花叶混合和密集遮挡等结构覆盖。b，代表性样本的原始视图、SAM3 初始掩膜、PCA 主前景精炼结果和仅前景 RGB。c，五种文本提示的典型成功与失败模式。d，FSAM3 与 SEEM 的分割指标比较：FSAM3 的 F1-score、mIoU 和 HD95 分别为 98.3%、97.9% 和 41.4 px，SEEM 分别为 95.1%、94.1% 和 281.9 px。e，精炼前后每帧连通组件数量和主前景保留率。

该结果说明 FSAM3 在本文采集条件下提供了更完整、更稳定的重建前景先验。

### 3.2 ForeSplat 改善应用级重建质量和处理效率

ForeSplat 能够从普通 RGB 采集稳定生成可测量三维表示。全部样本均完成从视频采集、FSAM3 掩膜、COLMAP、2DGS、TSDF 网格到表型测量的完整流程；20 个序列的流程成功率为 100%。在应用级重建对比中，ForeSplat 达到 PSNR = 31.09 dB、SSIM = 0.9711 和 LPIPS = 0.0365。四种流程的视觉重建、网格输出和效率趋势见图 4。

**图 4｜重建质量、几何输出与处理效率比较。** a，COLMAP、3DGS-FSAM3、标准 2DGS 和 ForeSplat 的代表性新视角渲染、仅前景渲染和 TSDF 网格输出，局部放大区域突出叶缘、花盆残留和背景泄漏差异。b，四种流程在 PSNR、SSIM、LPIPS、训练时间和网格时间上的归一化比较。c，FFT、SAM3/FSAM3 前景分割和二者联合引入后的模块消融曲线。d，固定装置和复杂背景两类采集场景中的成功样本数、有效帧数和注册视角数。

**表 2｜不同重建流程的重建质量与处理效率比较。**

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 训练时间 / s ↓ | 网格时间 / s ↓ |
|---|---:|---:|---:|---:|---:|
| COLMAP | 13.63 | 0.8745 | 0.1072 | 599.5 | 78 |
| 3DGS-FSAM3 | 30.17 | 0.9587 | 0.0386 | 5413.5 | 642 |
| Standard 2DGS | 29.58 | 0.9574 | 0.0487 | 12913.7 | 157.9 |
| ForeSplat | 31.09 | 0.9711 | 0.0365 | 5044.5 | 55.0 |

相较标准 2DGS，完整流程将 PSNR 提高 1.51 dB，并将训练时间和网格提取时间分别降低 60.94% 和 65.17%。相较 3DGS-FSAM3，完整流程在保持更高重建质量的同时将网格提取时间从 642 s 降至 55 s。模块消融进一步显示，FFT 主要减少低质量帧对位姿和优化的影响，SAM3/FSAM3 前景分割主要减少背景进入 Gaussian 表示和网格化过程；这一处理原则也与低质量帧、视角覆盖和重建稳定性之间的已知权衡一致 [18,19,50]。

### 3.3 前景 RGB 监督是抑制背景泄漏的关键

为明确仅含植物表示是否需要在训练目标中定义，我们在 KongQueZhuYu 主样本上比较整场景训练、输入域掩膜、透明度正则、前景 RGB 监督和后验掩膜剪枝等设置。无前景约束基线几乎重建全部背景，掩膜外非黑比例为 0.9908，泄漏能量比为 1.2201。输入域前景掩膜能压低背景，但 PSNR_fg 从 24.2090 dB 降至 20.7291 dB，前景质量显著下降。alpha 掩膜一致性、背景不透明度抑制及二者联合正则在整图 RGB 监督保持不变时仍无法阻止背景学习，泄漏能量比均约为 1.23。核心指标列于表 3，对应视觉证据见图 5。

**表 3｜KongQueZhuYu 上的前景对象重建目标消融。** 表中比较整场景训练、输入域掩膜、透明度约束、前景 RGB 监督和后验掩膜剪枝对背景泄漏和前景质量的影响。

| 方法设置 | 前景初始化 | 前景 RGB | alpha/背景 | PSNR_fg ↑ | 外部比例 ↓ | 泄漏 ↓ | Gaussian 数量 ↓ | 仅前景 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 无前景约束的整场景 2DGS | 否 | 否 | 否 | 24.2090 | 0.9908 | 1.2201 | 751,213 | 否 |
| 输入域前景掩膜约束 | 否 | 隐式 | 否 | 20.7291 | 0.0073 | 0.0042 | 263,108 | 是，质量下降 |
| alpha 掩膜一致性单项约束 | 否 | 否 | alpha | 24.3422 | 0.9898 | 1.2260 | 768,067 | 否 |
| 背景不透明度抑制单项约束 | 否 | 否 | 背景 | 24.7508 | 0.9900 | 1.2255 | 742,931 | 否 |
| alpha 与背景不透明度联合正则 | 否 | 否 | 联合 | 24.8126 | 0.9896 | 1.2266 | 763,266 | 否 |
| 前景 RGB 监督与透明度联合正则 | 否 | 是 | 联合 | 25.1055 | 0.0294 | 0.0190 | 592,900 | 是 |
| 前景对象重建目标完整配置 | 是 | 是 | 联合 | 25.0072 | 0.0294 | 0.0189 | 591,623 | 是 |
| 整场景训练后验掩膜剪枝 | 否 | 否 | 后验 | 24.6918 | 0.7509 | 0.7900 | — | 否 |

**图 5｜前景对象重建目标消融的视觉证据。** a，不同训练目标的渲染和网格可视化，显示背景、花盆和桌面是否进入最终表示。b，掩膜外非黑比例和泄漏能量比的柱状比较，并标出仅前景阈值。c，前景 RGB 监督前后局部区域放大，突出背景泄漏和叶缘保留差异。d，后验掩膜剪枝与训练期前景对象优化的对比。

只有启用前景 RGB 监督后，外部比例和泄漏才降至仅前景阈值以内。加入前景轨迹初始化后，完整配置保持相近的前景质量和背景抑制效果，PSNR_fg、SSIM_fg、LPIPS_fg、外部比例、泄漏和 Gaussian 数量分别为 25.0072、0.8548、0.0438、0.0294、0.0189 和 591,623。后验掩膜剪枝对照的外部比例和泄漏仍为 0.7509 和 0.7900，说明整场景训练后的处理并不等价于前景对象优化。

### 3.4 代表性结构样本验证前景对象重建的泛化性

三个结构差异明显的代表性样本用于评估前景对象重建目标的稳健性。KongQueZhuYu、XianKeLai1 和 CaoMei2 均满足仅前景标准，外部比例分别为 0.0294、0.0484 和 0.0147，泄漏能量比分别为 0.0189、0.0379 和 0.0081。XianKeLai1 最接近外部比例阈值，说明薄叶和细结构仍是边界敏感场景。三个代表性样本的输入、掩膜、渲染、网格和局部误差位置见图 6。

**图 6｜代表性结构样本上的仅前景重建。** a，复杂背景、薄叶细结构和密集遮挡样本的原始图像、前景掩膜、仅前景渲染和 TSDF 网格。b，三个样本的 PSNR_fg、外部比例、泄漏能量比和 Gaussian 数量。c，薄叶边界和遮挡区域的局部放大，展示剩余误差主要集中在叶缘、叶柄和局部遮挡处。

### 3.5 软视角加权保留几何覆盖并产生紧凑表示

为评估低质量视角在植物薄结构重建中的作用，我们比较了硬性视图剔除和软损失加权。该分析关注的不是单张图像质量，而是多视角几何覆盖能否被保留。质量阈值驱动的硬性剔除删除 27 个视角中的 10 个，使 PSNR_fg 从 25.0072 dB 降至 12.5478 dB；掩膜质量驱动剔除保留 24 个视角，但 PSNR_fg 仍仅为 13.4557 dB。软加权保留全部 27 个视角，仅使 PSNR_fg 下降 0.0506 dB，同时将 Gaussian 数量从 591,623 降至 532,264，减少 10.03%。不同视角策略和紧凑化配置的质量-规模权衡见图 7。

**图 7｜视角质量策略与表示紧凑性。** a，硬性视图剔除与软损失加权的视角覆盖示意，突出被删除视角对应的薄叶可见区域。b，三种视角质量策略在 PSNR_fg、SSIM_fg、LPIPS_fg、外部比例、泄漏和 Gaussian 数量上的比较。c，CaoMei2、XianKeLai1 和 KongQueZhuYu 中完整配置、软加权、掩膜引导剪枝和紧凑化配置的质量-紧凑性权衡。d，紧凑化前后 Gaussian 数量、泄漏指标和代表性渲染差异。

在三个代表性样本上，紧凑化配置将 Gaussian 总数从 1,216,294 降至 997,049，减少 18.03%。平均 PSNR_fg 下降 0.0657 dB，SSIM_fg 下降 0.0011，LPIPS_fg 增加 0.0003。其主要价值是表示紧凑性和导出清洁度，而不是显著提升前景渲染质量。相关的 Gaussian 压缩和精简研究也表明，表示紧凑性通常需要在渲染质量、存储和速度之间折中 [51-54]。

### 3.6 网格结构验证

为了将仅含植物的 Gaussian 表示转换为可测量对象，我们比较了不同 TSDF 网格化设置下的结构连通性、边界指标和计算时间。KongQueZhuYu 的标准 TSDF 网格包含 167,789 个顶点、8 个连通分量，最大分量比例为 0.9920；较小截断距离将顶点数降至 147,665，但连通分量增加到 20，最大分量比例降至 0.9350。XianKeLai1 也表现出相同趋势：较小截断距离将顶点数从 74,753 降至 66,138，但连通分量从 6 增加到 12。不同 TSDF 设置对网格形态、连通性和边界的影响见图 8。

**图 8｜TSDF 网格结构验证。** a，标准 TSDF、较小截断距离和边界后处理在 KongQueZhuYu 与 XianKeLai1 上的网格形态、连通分量和边界边分布。b，不同网格设置的顶点数、最大分量比例、边界边数和计算时间。c，叶缘、孔洞和薄叶边界的局部放大，展示较小截断距离带来的碎片化风险。

较小截断距离减少顶点数但增加碎片化风险；边界后处理保持连通分量数量不变，但增加网格生成耗时。这些结果支持结构和效率层面的评价，但不构成某一网格变体能够因果性降低表型误差的证据。

### 3.7 表型验证

该三维表示进一步用于农业表型测量。虚拟测量值与人工测量值在株高、冠幅、叶长和叶宽上进行比较，相关性和残差分布见图 9。

**图 9｜人工测量与虚拟表型测量的相关性。** a-d，株高、冠幅、叶长和叶宽的人工测量与虚拟测量散点图，并显示线性拟合和 1:1 参考线。e，四种性状的残差分布。f，MAE、RMSE、MAPE、偏差和 R² 的汇总可视化。

**表 4｜人工测量与虚拟表型测量一致性。**

| 性状 | n | MAE/cm ↓ | RMSE/cm ↓ | MAPE/% ↓ | 偏差/cm | R² ↑ |
|---|---:|---:|---:|---:|---:|---:|
| 株高 | 21 | 0.98 | 1.21 | 6.91 | 0.58 | 0.9878 |
| 冠幅 | 21 | 0.86 | 0.99 | 4.50 | 0.64 | 0.9879 |
| 叶长 | 63 | 0.51 | 0.64 | 7.45 | 0.31 | 0.9738 |
| 叶宽 | 63 | 0.45 | 0.64 | 9.73 | 0.38 | 0.8999 |

株高和冠幅一致性最高，叶长次之，叶宽误差最大。所有性状均存在轻微正偏差，说明虚拟测量相对人工测量略有高估。叶宽的较低 R² 与薄叶边界更敏感、TSDF 边界扩张和 landmark 放置误差有关。

---

## 4. 讨论

### 4.1 从整场景重建到前景对象重建

本研究最重要的发现是，面向性状测量的仅植物表示不能可靠地由整场景 2DGS 训练后的掩膜剪枝获得。标准 2DGS 在训练阶段根据整图 RGB 损失分配 Gaussian 容量，背景区域同样产生光度梯度，并通过增密形成稳定表示。后验剪枝只能删除已经形成的部分基元，不能改变训练过程中容量分配的方向。图 4 中 alpha 掩膜损失、背景不透明度损失和二者联合正则的失败说明，只约束透明度场不足以阻止背景学习；只有当 RGB 监督本身被限制到前景像素时，模型的主要优化压力才转向植物对象。

这一结论与 LCR-GS 和 IPENS 所强调的植物级可分析表示一致，但 ForeSplat 进一步表明，若目标是单株植物表型测量，对象边界最好在重建目标中尽早出现。换言之，掩膜不只是分割输出，而是农业表型三维优化问题的一部分。

### 4.2 FSAM3 是重建先验而非分割终点

FSAM3 的贡献在于为 2DGS 训练提供稳定掩膜，而不是替代三维优化。FFT 筛选减少模糊帧，SAM3 提示提供语义前景，PCA 精炼抑制假阳性碎片。人工标注基准说明其在本文数据条件下优于 SEEM。未来仍需在更大规模标注集上分析分割误差与重建误差的耦合关系。

### 4.3 视角质量应被调制，而不是被删除

硬性视图剔除的失败说明，多视角植物重建中的低质量帧不等同于无用帧。植物叶片常具有角度依赖可见性，少数视角即便质量较弱，也可能提供不可替代的几何覆盖。软加权将几何覆盖和光度可靠性分离：所有视角保留以维持三维约束，质量权重只调节其对 RGB 损失的贡献。这一原则对薄叶、遮挡和复杂冠层尤其重要。

### 4.4 表征紧凑性的价值

紧凑化配置的主要收益不是提高 PSNR，而是在几乎不损失前景质量的情况下减少 Gaussian 数量。对于高通量表型，这意味着更低存储、更快网格导出和更容易批量处理。掩膜引导剪枝比单纯不透明度阈值更适合植物，因为叶缘和孔洞附近的基元可能透明度不低但多视角掩膜支持不足。剪枝强度仍需保持保守，过强剪枝可能损伤细叶、叶柄和薄边界。

### 4.5 表型误差模式

株高和冠幅由整体外包范围决定，因此对局部边界误差不太敏感。叶长依赖单片叶片主轴，难度中等。叶宽依赖局部横截面边界，最容易受重建分辨率、Gaussian 支持域、TSDF 融合和标志点放置影响。这些结果支持 ForeSplat 用于结构性状自动测量，但叶宽仍是最需要改进的指标。后续研究可结合更高分辨率采集、边界感知网格精炼、叶缘不确定性建模和多操作者标志点重复性评估。

### 4.6 与参考研究的关系

与 Arshad 等对 NeRF 植物重建效率和精度的比较不同，ForeSplat 选择显式 2DGS 表示并把目标限定到植物前景。与 Plant3R 类似，本文也重视几何先验和 Gaussian 渲染的结合，但 Plant3R 侧重通过 MASt3R 改善小麦场景初始化，而 ForeSplat 侧重用前景掩膜改写 2DGS 训练目标。与 IPENS 和 LCR-GS 相比，ForeSplat 不是在重建后提取目标点云或单株子集，而是在训练期间直接生成仅含植物的 Gaussian 表示。三类路线并非互斥：未来可以将更强的特征匹配、SAM2/3 时序传播、LCR-GS 式多株分解与 ForeSplat 的前景对象目标结合，用于更复杂温室和田间场景。

---

## 5. 结论

本文提出 ForeSplat，一个从多视角图像质量控制到可用于表型测量的植物网格的前景感知 2D Gaussian Splatting 流程。通过 FSAM3 前景先验、前景轨迹初始化、前景 RGB 监督、alpha/背景不透明度约束、视角质量感知软损失加权和掩膜引导 Gaussian 剪枝，ForeSplat 将标准整场景 2DGS 改写为仅含植物的前景对象重建。结果表明，前景 RGB 监督是抑制背景泄漏的关键机制；硬性视图剔除会破坏薄结构角度覆盖，而软加权能在保持质量的同时降低 Gaussian 数量；紧凑化配置主要提升表示紧凑性和导出效率。跨 20 个序列和 21 株植物的验证表明，该流程可支持株高、冠幅、叶长和叶宽的测量。总体而言，ForeSplat 为室内或半受控条件下的低成本、非破坏性、多物种盆栽植物三维表型监测提供了一条可复现、可扩展的技术路径。

---

## 数据可用性

支持本研究的多视角图像、FSAM3 掩膜、表型测量表格、视角权重文件和主要运行配置可在合理请求下获得，并将在数据整理完成后通过项目仓库或数据存储库公开。

## 伦理声明

本研究仅涉及植物成像和测量，不涉及人类或动物受试对象。

## 利益冲突

作者声明不存在竞争性利益。

## AI 使用声明

在稿件准备过程中，作者使用 AI 辅助工具进行文献整理、结构重写、语言润色和中英互译。所有 AI 辅助内容均由作者审阅、核验和编辑；作者对论文内容、数据解释、引用准确性和发表作品完整性承担全部责任。

## 参考文献

[1] Araus, J. L., Cairns, J. E. Field high-throughput phenotyping: the new crop breeding frontier. Trends Plant Sci. 19, 52-61 (2014).  
[2] Araus, J. L., Kefauver, S. C., Zaman-Allah, M., Olsen, M. S., Cairns, J. E. Translating high-throughput phenotyping into genetic gain. Trends Plant Sci. 23, 451-466 (2018).  
[3] Zhao, C. et al. Crop phenomics: current status and perspectives. Front. Plant Sci. 10, 714 (2019).  
[4] Li, L., Zhang, Q., Huang, D. A review of imaging techniques for plant phenotyping. Sensors 14, 20078-20111 (2014).  
[5] Rebetzke, G. J., Jimenez-Berni, J., Fischer, R. A., Deery, D. M., Smith, D. J. High-throughput phenotyping to enhance the use of crop genetic resources. Plant Sci. 282, 40-48 (2019).  
[6] Wang, R.-F., Qu, H.-R., Su, W.-H. From sensors to insights: technological trends in image-based high-throughput plant phenotyping. Smart Agric. Technol. 12, 101257 (2025).  
[7] Paulus, S. Measuring crops in 3D: using geometry for plant phenotyping. Plant Methods 15, 103 (2019).  
[8] Paturkar, A., Gupta, G. S., Bailey, D. Making use of 3D models for plant physiognomic analysis: a review. Remote Sens. 13, 2232 (2021).  
[9] Akhtar, M. S., Zafar, Z., Nawaz, R., Fraz, M. M. Unlocking plant secrets: a systematic review of 3D imaging in plant phenotyping techniques. Comput. Electron. Agric. 222, 109033 (2024).  
[10] Harandi, N., Vandenberghe, B., Vankerschaver, J., Depuydt, S., Van Messem, A. How to make sense of 3D representations for plant phenotyping: a compendium of processing and analysis techniques. Plant Methods 19, 60 (2023).  
[11] Li, J. et al. A survey on 3D reconstruction techniques in plant phenotyping: from classical methods to neural radiance fields (NeRF), 3D Gaussian splatting (3DGS), and beyond. Plant Phenomics 26, 100137 (2025).  
[12] Qi, J. et al. Multiscale phenotyping of grain crops based on three-dimensional models: a comprehensive review of trait detection. Comput. Electron. Agric. 237, 110597 (2025).  
[13] Nguyen, T. T., Slaughter, D. C., Max, N., Maloof, J. N., Sinha, N. Structured light-based 3D reconstruction system for plants. Sensors 15, 18587-18612 (2015).  
[14] Debnath, S., Paul, M., Debnath, T. Applications of LiDAR in agriculture and future research directions. J. Imaging 9, 57 (2023).  
[15] Vazquez-Arellano, M., Reiser, D., Paraforos, D. S., Garrido-Izard, M., Griepentrog, H. W. 3-D reconstruction of maize plants using a time-of-flight camera. Comput. Electron. Agric. 145, 235-247 (2018).  
[16] Schonberger, J. L., Frahm, J.-M. Structure-from-motion revisited. Proc. IEEE Conf. Comput. Vis. Pattern Recognit. 4104-4113 (2016).  
[17] Furukawa, Y., Ponce, J. Accurate, dense, and robust multiview stereopsis. IEEE Trans. Pattern Anal. Mach. Intell. 32, 1362-1376 (2010).  
[18] Sheng, W. et al. MVS-Pheno: a portable and low-cost phenotyping platform for maize shoots using multiview stereo 3D reconstruction. Plant Phenomics 5, 0143 (2023).  
[19] Wang, Y., He, S., Ren, H., Yang, W., Zhai, R. 3DPhenoMVS: a low-cost 3D tomato phenotyping pipeline using 3D reconstruction point cloud based on multiview images. Agronomy 12, 1865 (2022).  
[20] Mildenhall, B. et al. NeRF: representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 99-106 (2022).  
[21] Muller, T., Evans, A., Schied, C., Keller, A. Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph. 41, 102 (2022).  
[22] Chen, A., Xu, Z., Geiger, A., Yu, J., Su, H. TensoRF: tensorial radiance fields. ECCV 333-350 (2022).  
[23] Tancik, M. et al. Nerfstudio: a modular framework for neural radiance field development. ACM SIGGRAPH Conf. Proc. 1-12 (2023).  
[24] Afful, J. et al. Evaluating Neural Radiance Fields for 3D Plant Geometry Reconstruction in Field Conditions. Plant Phenomics 6, Article 0235 (2024).  
[25] Jignasu, A. et al. Plant geometry reconstruction from field data using neural radiance fields. AAAI Workshop on AI for Agriculture and Food Systems (2023).  
[26] Choi, H.-B., Park, J.-K., Park, S. H., Lee, T. S. NeRF-based 3D reconstruction pipeline for acquisition and analysis of tomato crop morphology. Front. Plant Sci. 15, 1439086 (2024).  
[27] Kerbl, B., Kopanas, G., Leimkuhler, T., Drettakis, G. 3D Gaussian Splatting for real-time radiance field rendering. ACM Trans. Graph. 42, 139 (2023).  
[28] Huang, B. et al. 2D Gaussian Splatting for geometrically accurate radiance fields. ACM Trans. Graph. 43, 1-17 (2024).  
[29] Guedon, A., Lepetit, V. SuGaR: Surface-Aligned Gaussian Splatting for efficient 3D mesh reconstruction and high-quality mesh rendering. Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. 5354-5363 (2024).  
[30] Ma, J. et al. Plant3R: Fusing 3D feature learning with Gaussian splatting to enhance wheat plant 3D reconstruction precision. Plant Phenomics (2026). doi:10.1016/j.plaphe.2026.100200.  
[31] Shen, P., Jing, X., Deng, W., Jia, H., Wu, T. PlantGaussian: exploring 3D Gaussian splatting for cross-time, cross-scene, and realistic 3D plant visualization and beyond. Crop J. 13, 607-618 (2025).  
[32] Jiang, L., Sun, J., Chee, P. W., Li, C., Fu, L. Cotton3DGaussians: multiview 3D Gaussian splatting for boll mapping and plant architecture analysis. Comput. Electron. Agric. 234, 110293 (2025).  
[33] Li, J., Zhu, K., Zhang, Q., Chen, D., Sun, Q., Li, Z. Object-centric 3D Gaussian splatting for strawberry plant reconstruction and phenotyping. Smart Agric. Technol. 13, 101810 (2026).  
[34] Lin, J.-H., Lin, T.-T. From 3DGS scenes to plant traits: a scalable extraction and segmentation framework for muskmelon phenotyping. Front. Plant Sci. 17, 1783465 (2026).  
[35] Song, W. et al. IPENS: Interactive unsupervised framework for rapid plant phenotyping extraction via NeRF-SAM2 fusion. Plant Phenomics (2025). doi:10.1016/j.plaphe.2025.100106.  
[36] Ye, M., Danelljan, M., Yu, F., Ke, L. Gaussian grouping: segment and edit anything in 3D scenes. ECCV 162-179 (2024).  
[37] Cen, J. et al. Segment any 3D Gaussians. Proc. AAAI Conf. Artif. Intell. 39, 1971-1979 (2025).  
[38] Kirillov, A. et al. Segment Anything. Proc. IEEE/CVF Int. Conf. Comput. Vis. 3992-4003 (2023).  
[39] Ravi, N. et al. SAM 2: Segment Anything in images and videos. arXiv:2408.00714 (2024).  
[40] Cheng, T. et al. YOLO-World: real-time open-vocabulary object detection. Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. 16901-16911 (2024).  
[41] Shi, W., van de Zedde, R., Jiang, H., Kootstra, G. Plant-part segmentation using deep learning and multi-view vision. Biosyst. Eng. 187, 81-95 (2019).  
[42] Li, D. et al. PlantNet: a dual-function point cloud segmentation network for multiple plant species. ISPRS J. Photogramm. Remote Sens. 184, 243-263 (2022).  
[43] Du, R., Ma, Z., Xie, P., He, Y., Cen, H. PST: plant segmentation transformer for 3D point clouds of rapeseed plants at the podding stage. ISPRS J. Photogramm. Remote Sens. 195, 380-392 (2023).  
[44] Wu, X. et al. Point Transformer V3: simpler faster stronger. Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. 4840-4851 (2024).  
[45] Curless, B., Levoy, M. A volumetric method for building complex models from range images. SIGGRAPH 303-312 (1996).  
[46] Lorensen, W. E., Cline, H. E. Marching Cubes: a high resolution 3D surface construction algorithm. SIGGRAPH 163-169 (1987).  
[47] Hore, A., Ziou, D. Image quality metrics: PSNR vs. SSIM. Int. Conf. Pattern Recognit. 2366-2369 (2010).  
[48] Wang, Z., Bovik, A. C., Sheikh, H. R., Simoncelli, E. P. Image quality assessment: from error visibility to structural similarity. IEEE Trans. Image Process. 13, 600-612 (2004).  
[49] Zhang, R., Isola, P., Efros, A. A., Shechtman, E., Wang, O. The unreasonable effectiveness of deep features as a perceptual metric. Proc. IEEE Conf. Comput. Vis. Pattern Recognit. 586-595 (2018).  
[50] Pertuz, S., Puig, D., Garcia, M. A. Analysis of focus measure operators for shape-from-focus. Pattern Recognit. 46, 1415-1432 (2013).  
[51] Fan, Z. et al. LightGaussian: unbounded 3D Gaussian compression with 15x reduction and 200+ FPS. Adv. Neural Inf. Process. Syst. (2024).  
[52] Lee, J. C. et al. Compact 3D Gaussian representation for radiance field. Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (2024).  
[53] Liu, W. et al. EfficientGS: streamlining Gaussian Splatting for large-scale high-resolution scene representation. arXiv:2404.12778 (2024).  
[54] Fang, G., Wang, B. Mini-splatting: representing scenes with a constrained number of Gaussians. ECCV 165-181 (2024).  
