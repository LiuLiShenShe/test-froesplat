# ForeSplat: foreground-aware 2D Gaussian Splatting for automated multi-species plant phenotyping from multi-view images

---

**目标期刊：** Computers and Electronics in Agriculture (CompAg), Elsevier, IF 8.9 (2025)
**论文类型：** 原创研究论文（IMRaD）
**语言：** 中文
**字数：** 约 9,000 词（英文主文对应长度；投稿前以英文版最终统计为准）
**图：** 10（含分割、全数据集重建、外部 baseline 与表型验证图）
**表：** 13（含数据覆盖、分割 benchmark、外部 baseline 与表型验证表）

---

## 代号替换说明

早期实验蓝图和运行配置中使用过 `A0-A10` 与 `M1-M10` 代号。为避免论文正文呈现为工程蓝图，本文根据当前代码开关、运行配置和闭环结果，将这些代号统一替换为描述性方法名称；下表仅用于说明历史代号与本文表述的对应关系，正文、图表和讨论中不再把这些代号作为实验名称使用。

| 历史代号 | 论文正文采用表述 | 实现与证据依据 |
|---|---|---|
| A0 | 无前景约束的整场景 2DGS 重建基线 | `mask_mode=none`，不使用前景 mask 或透明度约束 |
| A1 | 输入域前景掩膜约束 | `mask_mode=preprocess`，将输入 RGB 背景置黑，但不改写 2DGS 目标函数 |
| A2 | alpha 掩膜一致性单项约束 | 整图 RGB 监督不变，仅启用 `use_mask_loss` |
| A3 | 背景不透明度抑制单项约束 | 整图 RGB 监督不变，仅启用 `use_bg_opacity_loss` |
| A4 | alpha 掩膜一致性与背景不透明度联合正则化 | 整图 RGB 监督不变，同时启用 alpha mask loss 和 background opacity loss |
| A5 | 前景限定 RGB 监督与透明度场联合正则化 | 启用 foreground RGB loss、alpha mask loss 和 background opacity loss，但不做前景轨迹初始化 |
| A6 | 前景对象重建目标的完整配置 | 在前景 RGB 监督和透明度约束基础上加入 foreground track initialization |
| A7-A10 | 未纳入本文闭环证据链的预留实验条件 | 当前代码与结果目录中没有进入本文主结果的正式配置，正文不保留这些代号 |
| M1 | 视角质量感知的视图选择与损失加权 | 对应 H-VQG retained-list、hard filtering、mask-quality reject-only filtering 和 soft view weighting |
| M2 | FSAM3 前景先验的训练接入与视图配准 | 对应 `mask_mode={preprocess,alpha}`、`mask_dir`、`mask_pattern` 等 mask 接入逻辑 |
| M3 | 掩膜约束的透明度场正则化 | 对应 alpha mask loss、background opacity loss、边界忽略和 warm-up 参数 |
| M4 | 掩膜引导的多线索 Gaussian 剪枝 | 对应 `pruning_mode=mask/topology` 及透明度、亮度、mask consistency、可见性等剪枝线索 |
| M5 | 边界感知 TSDF 网格化与轮廓一致性后处理 | 对应 `meshing_mode=standard/small_trunc/edge_aware/post_boundary` |
| M6-M10 | 未纳入本文实现路径与闭环证据链的预留模块 | 当前可复现配置未定义这些模块，正文不保留这些代号 |
| E7 | 整场景表征的后验掩膜剪枝对照 | 先训练无前景约束的整场景 2DGS，再按多视角 mask 投影删除背景 Gaussians |

---

## Highlights

- 前景 RGB 监督将 2DGS 从 full-scene 重建改写为 plant-only 重建。
- FSAM3 在人工标注 mask benchmark 上优于 SEEM，F1-score 和 mIoU 分别达到 98.3% 和 97.9%。
- ForeSplat/F2DMAS 在 20 个多视角序列上完成植物重建与表型测量流程验证。
- 与标准 2DGS、COLMAP、3DGS-FSAM3 和 SuGaR 对比，完整流程提高重建质量并改善 phenotype-ready mesh 可用性。
- 视角质量感知的软损失加权保留全部视角，避免硬性视图剔除破坏薄叶角度覆盖。
- 21 株植物验证显示全局性状精度高，叶宽仍是边界敏感瓶颈。

---

## 摘要

自动化三维植物表型需要从低成本多视角图像中恢复可测量的 plant-only 几何表示。标准 2D Gaussian Splatting（2DGS）以完整场景为重建目标，容易把模型容量分配给花盆、桌面和背景。本文提出 ForeSplat/F2DMAS，一个从智能手机视频采集、图像质量控制到 phenotype-ready 植物网格生成的 foreground-aware splatting 流程。该方法通过 FSAM3 生成重建导向前景先验，包括 FFT 频域筛选、SAM3 文本提示分割和 PCA 主前景精炼；随后通过 foreground track initialization、foreground RGB supervision、alpha mask loss 和 background opacity loss，将 2DGS 从 full-scene reconstruction 重新定义为 mask 约束的 foreground-object reconstruction。实验覆盖 20 个盆栽植物多视角序列和两类采集场景，并在 21 株植物上验证人工-虚拟表型一致性。人工标注分割 benchmark 显示，FSAM3 的 F1-score、mIoU 和 HD95 分别为 98.3%、97.9% 和 41.4 px，优于 SEEM 的 95.1%、94.1% 和 281.9 px。KongQueZhuYu 主样本消融显示，foreground RGB supervision 将 mask 外非黑比例从 0.9908 降至 0.0294，将泄漏能量比从 1.2201 降至 0.0190。应用级重建对比显示，完整流程达到 PSNR = 31.09 dB、SSIM = 0.9711 和 LPIPS = 0.0365，相比标准 2DGS 的 29.58 dB、0.9574 和 0.0487 有所改善，并将训练时间和 mesh 提取时间分别降低 60.94% 和 65.17%。相较于 3DGS-FSAM3，完整流程保持更高渲染质量，并将 mesh 提取时间从 642 s 降至 55 s；共有 11 个序列上的 Gaussian baseline 对比进一步纳入 SuGaR refined 作为 3DGS-to-mesh 参照。株高、冠幅、叶长和叶宽的人工-虚拟测量 R² 分别为 0.9878、0.9879、0.9738 和 0.8999。结果表明，ForeSplat/F2DMAS 可在室内或半受控复杂背景条件下支持低成本三维植物结构表型测量；叶宽等边界敏感性状仍是主要改进方向。

**关键词：** 植物表型；2D Gaussian Splatting；前景对象重建；SAM3 分割；视角质量感知；Gaussian 剪枝；TSDF 网格提取

---

## 1. 引言

设施园艺和精准农业需要在尽量少人工干预的条件下连续获取植物结构性状。株高、冠幅、叶长和叶宽等指标直接定义于物理空间，支撑生长监测、产量预测和育种筛选，但在薄叶、密集冠层和局部遮挡条件下难以仅由单幅二维图像稳定测量。因此，自动化植物表型不仅需要识别植物外观，还需要生成可重复、可解释、具有物理尺度的三维植物对象表示 [1]。

多视角三维重建和辐射场表示为这一目标提供了重要基础。SfM/MVS 能够从普通 RGB 图像估计相机位姿和稀疏/稠密几何，但在弱纹理、重复纹理和叶片遮挡区域容易出现匹配不稳定。NeRF 通过连续辐射场改善了新视角合成，3DGS 以显式 Gaussian 表示提高了渲染效率，而 2DGS 进一步用有方向的二维平面 Gaussian 替代体积椭球，使薄表面结构的法向一致性和网格提取更符合植物叶片的几何特征。这些进展使 2DGS 成为盆栽植物三维表型的有吸引力基础表示 [2,3]。

然而，现有 2DGS 的默认优化目标与表型测量所需的 plant-only 表示并不一致。标准 2DGS 以 full-scene reconstruction 为目标，训练损失奖励模型重建所有可见内容，包括花盆、土壤、桌面、背景布、支架和光照伪影。在半受控采集场景中，这些非植物结构常占据图像的大量区域，并在训练期间吸引 Gaussian 容量分配。训练完成后再进行 mask 剪枝或过滤，只能移除部分可见背景基元，难以恢复已分配到背景的模型容量，也难以处理与叶缘、盆沿或背景纹理投影重叠的边界 Gaussians。由此产生的背景泄漏会增加模型体积、复杂化网格提取，并在虚拟表型测量中引入系统误差 [4,5,6]。

这一瓶颈还受到输入数据质量和 mask 可靠性的共同制约。多视角植物序列常包含运动模糊、失焦或低纹理帧，直接影响 SfM 位姿估计、分割一致性和后续重建。与此同时，对数百帧图像逐帧人工标注植物 mask 成本过高，传统颜色阈值方法又容易受光照、叶色和背景干扰影响。已有质量评估、可提示分割和 Gaussian 剪枝方法分别处理了部分问题，但仍缺少一个面向表型测量的集成流程，能够同时完成质量筛选、跨物种植物前景 mask 生成，并将 2DGS 的训练目标从场景级重建重新定义为 mask 约束的前景对象重建 [7,8,9]。

为回应这一缺口，本文提出 ForeSplat（Foreground-aware 2D Gaussian Splatting for Multi-view Plant Phenotyping），一个从原始多视角图像到 phenotype-ready 植物网格的模块化流程。本研究的具体贡献如下：
1、ForeSplat/F2DMAS 首先通过 FSAM3 生成重建导向的多视角植物前景先验：FFT 频域筛选剔除低质量帧，SAM3 文本提示分割提取植物区域，PCA 引导的主成分精炼抑制不连通假阳性。FSAM3 被定位为重建先验而非通用分割 SOTA，并通过人工标注 mask 与 SEEM 对比验证其区域完整性和边界稳定性。

2、随后，ForeSplat 在 2DGS 中引入 foreground track initialization、foreground RGB supervision、alpha mask loss 和 background opacity loss，使优化目标直接围绕 mask 定义的植物前景展开；再通过视角质量感知的软损失加权保留所有视角但调制其损失贡献，并用掩膜引导的多线索 Gaussian 剪枝剪除弱支撑 Gaussians，最终经 TSDF 融合和尺度恢复输出可测量网格。

3、我们通过全流程应用验证、外部 baseline 对比和受控消融共同评估 ForeSplat 的作用边界。20 个多视角序列用于验证从视频采集、分割、SfM、2DGS 重建、TSDF 网格化到表型测量的完整流程；COLMAP、标准 2DGS、3DGS-FSAM3 和 SuGaR 对比用于检验重建质量、处理效率和 mesh 可用性；KongQueZhuYu 代表样本上的目标函数消融和整场景表征后验掩膜剪枝对照用于检验 foreground-object training 是否不同于 full-scene training 后剪枝；视角质量策略与紧凑化实验用于分析 soft weighting 和 mask-guided cleanup 的机制作用。结果表明，foreground RGB supervision 是将 full-scene 2DGS 转化为 plant-only reconstruction 的关键机制；完整流程在复杂背景下改善重建质量并显著降低 mesh 提取时间；跨 21 株植物的人工-虚拟性状对比进一步表明，该流程支持株高、冠幅、叶长和叶宽的测量。


---

## 2. 相关工作

### 2.1 三维植物表型

通过三维重建实现植物性状测量自动化，长期以来是精准农业和园艺科学中的重要研究方向。早期工作已经表明，当重建结果保留物理尺度和器官边界时，株高、叶面积、茎粗等几何性状可以从三维表示中可靠提取。传感器方案包括结构光、地面 LiDAR 和深度相机，它们在空间分辨率、成本和操作复杂度之间具有不同权衡。基于消费级 RGB 相机的多视角立体（MVS）作为一种低成本替代方案受到关注，尤其适用于温室和生长室等受控光照有利于图像匹配的部署场景。然而，经典 MVS 流程在低纹理或重复纹理叶片区域容易出现匹配失败，点云输出也通常需要大量后处理才能用于性状提取 [10,11,12]。

从经典 MVS 转向神经辐射场表示，标志着植物三维重建能力的重要变化。基于 NeRF 的方法已经应用于番茄和水稻穗表型重建 [13,14]，也被扩展到复杂农业场景建模 [15]，表明连续体积表示相比离散点云更能处理复杂光照和局部遮挡。3D Gaussian Splatting 进一步提升了计算效率，在保持有竞争力的重建质量的同时实现实时渲染。Li 等人 2025 年的综述首次系统覆盖了植物表型中的 NeRF 和 3DGS，确认此前综述尚未覆盖辐射场方法，并将薄叶重建、密集冠层处理和跨物种泛化识别为开放挑战 [2]。具体应用包括用于跨时间植物可视化的 PlantGaussian [16]、用于田间小麦穗表型分析的 Wheat3DGS [17]，以及用于桃园重建的 3DGS-Ag [18]。现有植物 3DGS/NeRF 工作的共同特征是使用 full-scene 训练目标：模型被优化为重建整张图像，包括非植物结构。我们的工作将 2DGS 优化目标从 full-scene 重建重新表述为植物 foreground-object 重建。

与重建方法进展并行，基于三维数据的自动性状提取也已显著成熟。Xiao 等人展示了一个从点云中提取 19 个小麦表型性状的集成流程（mIoU 92.3%）[19]。Reena 等人发布了 Wheat3D PartNet，这是第一个大规模带标注的三维小麦点云数据集（3 个品种、1,303 个模型），并 benchmark 了 PointNet++、3DGTN 和 GAPointNet 的器官分割性能 [20]。Gao 和 Su 使用基于学习的特征匹配（SuperPoint+LightGlue），从多视角水稻秧苗重建中实现株高 R² = 0.989、冠层面积 R² = 0.991 [21]。对于遮挡严重的作物，Jiang 等人使用深度强化学习修复被遮挡的番茄茎，实现茎粗 MAPE 9.7% [22]。这些进展说明，面向特定作物的三维数据自动性状提取正在接近生产可用水平，但跨物种泛化仍缺乏充分探索：每条流程通常针对单一物种，并依赖物种特定参数调优。

### 2.2 辐射场与 Gaussian Splatting

Neural Radiance Fields (NeRF) 将场景建模为连续函数 \(F_\theta: (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)\)，将三维位置 \(\mathbf{x}\) 和观察方向 \(\mathbf{d}\) 映射为发射颜色 \(\mathbf{c}\) 和体密度 \(\sigma\)。渲染通过可微体积光线步进完成，模型通过最小化训练视角中渲染像素颜色与真实像素颜色之间的光度误差进行优化。虽然 NeRF 能生成高质量新视角图像，但其隐式表示耦合了几何与外观，使显式表面提取较为困难，渲染计算代价也较高 [23,24]。

3D Gaussian Splatting (3DGS) 通过用显式各向异性三维 Gaussian 基元替代隐式 MLP 来同时缓解这两个问题。每个 Gaussian \(G_k\) 由位置 \(\boldsymbol{\mu}_k \in \mathbb{R}^3\)、协方差矩阵 \(\Sigma_k = R_k S_k S_k^T R_k^T\)（分解为旋转 \(R_k\) 和尺度 \(S_k\)）、透明度 \(\alpha_k \in [0,1]\)，以及用于视角相关颜色的球谐系数参数化。渲染时，Gaussians 通过相机投影映射为二维屏幕空间 splats，按深度排序，并以从前到后的 alpha compositing 在单次前向过程中合成。训练目标结合了 L1 光度损失和可微结构相似性（D-SSIM）项 [25]：

\[
L_{\text{3DGS}} = (1 - \lambda) L_1(I_{\text{render}}, I_{\text{gt}}) + \lambda L_{\text{D-SSIM}}(I_{\text{render}}, I_{\text{gt}})
\]

Gaussians 由稀疏 SfM 点云初始化，并在优化过程中基于位置梯度幅值和透明度阈值进行自适应 densification（split/clone）和 pruning [4,25]。

2D Gaussian Splatting (2DGS) 引入了一个关键几何修改：每个基元是一个平面二维 disk，而不是体积三维椭球。其协方差由两个张成 disk 平面的切向量 \(\mathbf{t}_u, \mathbf{t}_v\) 和法向量 \(\mathbf{n} = \mathbf{t}_u \times \mathbf{t}_v\) 构建，并将法向方向的尺度压缩到接近零。这种平面参数化为表面重建带来两个优势：(i) 二维 Gaussian 的渲染深度是观察光线与 disk 平面的交点，给出几何意义明确的表面点，而不是密度分布上的期望深度；(ii) 表面法向可直接由 disk 法向 \(\mathbf{n}\) 获得。2DGS 在 3DGS 损失之外加入两个几何项 [4]：

\[
L_{\text{2DGS}} = L_{\text{3DGS}} + \lambda_d L_d + \lambda_n L_n
\]

其中，\(L_d\) 为深度畸变损失，通过惩罚逐射线深度方差使 Gaussians 集中到表面；\(L_n\) 为法向一致性损失，使渲染法向与深度图梯度法向对齐。这些增加项使 2DGS 特别适合植物叶片这类薄表面对象。

Gaussian 到 mesh 的转换问题已由 SuGaR 等工作处理，SuGaR 在训练过程中引入额外正则，使 Gaussians 与底层表面对齐。Poisson surface reconstruction 和可微 iso-surface extraction 也已被探索。本文采用更简单的 TSDF 融合方法进行网格提取（Section 3.7），因为 2DGS 的平面几何相比 3DGS 已经提供了更好的表面对齐，而 TSDF 的简洁性避免在已被修改的 2DGS 优化中引入额外训练复杂度 [26,27,28]。

### 2.3 多视角重建的图像质量评估与视角选择

多视角三维重建的质量从根本上受输入图像质量限制。在 Structure-from-Motion 流程中，模糊或低纹理帧会引入异常特征匹配，从而降低相机位姿估计和稀疏重建质量。用于图像清晰度评估的频域方法在计算机视觉中已有较长历史：自然图像的功率谱通常遵循 \(1/f\) 衰减，而偏离这一特征，尤其是高频能量衰减，可以为失焦和运动模糊提供稳健指标。近期研究也将深度学习用于模糊检测和学习型感知质量指标，但这些方法通常需要任务特定训练数据 [7,11]。

在多视角重建文献中，视角选择主要用于提升 SfM 效率或 MVS 深度质量。主导范式是选择一个“最佳子集”，以最大化重建质量并最小化计算成本。然而，这些方法主要面向特征丰富度在视角间缓慢变化的一般场景。植物多视角序列具有不同挑战：只在狭窄角度范围内可见的薄叶，如果覆盖它们的少数视角被移除，就可能无法重建。该观察促使我们从子集选择范式转向 soft weighting 方法（Section 3.5），即所有视角都参与训练，但其梯度贡献由每视角质量调制。据我们所知，这是频域质量评估首次被专门设计用于多视角植物采集，也是首次证明 hard view filtering 与薄结构植物重建不兼容 [29,30]。

### 2.4 Promptable segmentation 与植物 mask 生成

植物前景图像分割传统上依赖颜色指数方法。Excess Green Index（ExG = 2G - R - B）、HSV thresholding 和 Otsu 自适应二值化利用植被偏绿的光谱特征。这些方法计算效率高，但存在明确失效模式：它们会将绿色非植物对象误认为植被，在光照和阴影变化下失效，并需要逐场景、逐物种参数调优。使用随机森林、SVM 或早期 CNN 的机器学习方法提高了鲁棒性，但需要大量数据集级标注 [31,32]。

Segment Anything Model (SAM) 代表了范式变化：一个在 1100 万张图像、超过 10 亿个 mask 上训练的视觉 Transformer，可以通过简单 prompt（点、框、文本）实现对多类对象的零样本泛化。SAM 2 通过基于记忆的时间传播机制将 promptable segmentation 扩展到视频，显著改善帧间一致性。在农业应用中，promptable segmentation 已被用于叶片实例分割、杂草检测和果实计数，研究发现 prompt 模态（点、框、文本）和 prompt 内容会显著影响不同物种和生长阶段的分割质量。SAM 系列用于植物表型分析的一个关键限制是缺少内置质量评估：模型总会输出 mask，但 mask 质量会随图像质量、植株姿态和背景复杂度显著变化 [8,33,34,35]。

我们的 FSAM3 流程通过将 SAM3 包装在质量感知框架中来处理这一限制：FFT screening（Stage 1）防止低质量帧进入分割，PCA-guided refinement（Stage 3）抑制 SAM3 在复杂背景中可能产生的假阳性碎片。这一三阶段设计将通用分割模型转化为面向重建的 mask 先验，使其能够支持跨物种的 2DGS foreground-object 优化。

### 2.5 Gaussian 剪枝与模型紧凑性

3DGS/2DGS 中的自适应 densification 机制可能产生冗余或支撑较弱的 Gaussians，尤其是在监督稀疏或噪声较大的区域。标准基于透明度的 pruning 会移除透明度低于固定阈值的 Gaussians，但这一标准本身无法判断 Gaussian 是否具有几何意义。已有多项工作提出了更复杂的剪枝策略。LightGaussian 使用可训练的重要性分数，并在训练后移除低分 Gaussians。Compact3D 对 Gaussian 参数应用向量量化。EfficientGS 基于训练迭代期间累积的视图空间位置梯度幅值进行剪枝。对于植物特异性重建，还存在一个额外剪枝线索：由于重建目标由 mask 定义，每个 Gaussian 与多视角 mask 的空间关系提供了直接的前景/背景信号。本文的掩膜引导多线索 Gaussian 剪枝模块（Section 3.6）利用这一点，将 mask 投影一致性、透明度、可见性和拓扑线索组合为多因素剪枝分数，从而在无需额外训练的情况下得到更紧凑的 plant-only Gaussian 集 [5,6,9]。

---

## 3. 材料与方法

### 3.1 方法框架与研究设计

本章描述 ForeSplat 从原始多视角图像到可测量植物网格的完整流程。该流程面向一个明确任务：将 2D Gaussian Splatting (2DGS) 从 full-scene reconstruction 重新定义为由植物 mask 约束的 foreground-object reconstruction。这一重定义使下游表型测量基于 plant-only 几何表示，而不是包含花盆、桌面和背景杂物的完整采集场景。

ForeSplat 包含五个顺序模块。整体工作流如图 1 所示。第一，FSAM3 对原始帧进行频域质量筛选，再执行文本 prompt 分割和主前景精炼，输出与训练图像一一对应的植物 mask。第二，COLMAP 估计相机位姿和稀疏三维点轨迹，进一步使用多视角 mask 一致性过滤初始化点。第三，前景对象优化将 RGB 监督限制在 mask 定义的植物像素，并用 alpha mask loss 与 background opacity loss 约束透明度场。第四，视角质量感知的软损失加权在保留全部视角的前提下调节逐视角损失权重。第五，掩膜引导的多线索 Gaussian 剪枝依据 mask 支持、透明度、可见性和拓扑线索清理弱支撑 Gaussians，并在训练后期参考梯度/可见性线索控制冗余表示。紧凑化前景对象表示随后导出为显式网格，用于虚拟表型特征提取，并对估计误差进行估算。


**图 1｜ForeSplat/F2DMAS 方法总览。** 多视角 RGB 图像首先经过 FSAM3 生成植物前景 mask，随后进入 COLMAP SfM、foreground track initialization、foreground-object 2DGS optimization、视角质量感知软损失加权、掩膜引导多线索 Gaussian 剪枝、TSDF mesh 提取和表型测量。

### 3.2 数据集、采集与样本用途

数据集包含 20 个盆栽植物多视角图像序列，按当前材料表覆盖 15 个物种或品种标签；表 2 为便于国际读者理解，以英文通用名归并展示为 10 类植物材料。样本包含宽叶、低矮冠层、重叠叶、紧凑冠层、密集叶、花叶混合、光滑叶、厚叶、细纹理和密集遮挡等形态条件。图像使用 iPhone 14 Pro Max 智能手机采集，视频分辨率为 1080 × 1920，帧率为 60 fps，镜头参数为 13 mm 和 f/2.2。采集包括两类场景：固定装置辅助采集和复杂背景手持环绕采集。固定装置由电动转台、黑色背景布、智能手机和尺度标记构成；复杂背景采集在普通室内或类温室环境中完成，背景包含桌面、支架、花盆、墙面、地面和颜色接近植物的杂物。20 个序列全部用于完整重建和表型流程验证；表型统计按植株个体计数，共包含 21 株可测植物，其中叶长和叶宽因每株 3 组代表性叶片测量而形成 n = 63 的叶片性状数据。20 个序列和 21 株植物分别对应多视角重建口径和植株级表型测量口径。

不同实验承担不同证据功能。完整 ForeSplat/F2DMAS 流程在全部 20 个序列上执行，用于验证从智能手机视频、FSAM3 分割、SfM 位姿估计、2DGS 重建、TSDF 网格化到虚拟表型测量的端到端可行性。固定装置辅助采集包含 10 个样本、2502 帧原始图像，经质量筛选后得到 2104 帧有效图像和 2040 个 SfM 注册视角，10 个样本完成完整流程，成功率为 100%；复杂背景采集包含 10 个样本、2500 帧原始图像，经质量筛选后得到 2113 帧有效图像和 2089 个 SfM 注册视角，10 个样本完成完整流程，成功率为 100%。KongQueZhuYu 用于前景对象目标函数消融和整场景表征后验掩膜剪枝对照，因为该序列包含明显复杂背景并能暴露 full-scene 训练的背景泄漏问题。KongQueZhuYu、XianKeLai1 和 CaoMei2 作为代表性可视化与机制分析样本，分别对应复杂背景、薄叶细结构和密集遮挡。

**表 1｜数据覆盖与 ForeSplat/F2DMAS 工作流执行情况。**

| 场景 | 物种数 | 样本数 | 原始帧数 | 有效帧数 | SfM 注册视角 | 成功样本数 | 成功率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 固定装置辅助采集 | 8 | 10 | 2502 | 2104 | 2040 | 10 | 100% |
| 复杂背景采集 | 7 | 10 | 2500 | 2113 | 2089 | 10 | 100% |

**表 2｜20 个多视角序列的材料与用途概要（英文通用名归并显示）。**

| Sample ID | Plant species | 结构/条件 | 场景 | 人工测量 |
|---|---|---|---|---|
| S01 | Peace lily | Broad leaves | Fixed | Yes |
| S02 | Strawberry | Low canopy | Fixed | Yes |
| S03 | Strawberry | Overlapping leaves | Fixed | Yes |
| S04 | Kalanchoe | Compact canopy | Fixed | Yes |
| S05 | Kalanchoe | Dense leaves | Complex | Yes |
| S06 | Kalanchoe | Flower-leaf mix | Complex | Yes |
| S07 | Peperomia | Smooth leaves | Fixed | Yes |
| S08 | Peperomia | Thick leaves | Complex | Yes |
| S09 | Peperomia | Dense small leaves | Complex | Yes |
| S10 | Anthurium | Glossy leaves | Complex | Yes |
| S11 | Calathea | Striped leaves | Complex | Yes |
| S12 | Fittonia | Fine texture | Fixed | Yes |
| S13 | Fittonia | Dense texture | Complex | Yes |
| S14 | Chinese evergreen | Broad variegated leaves | Fixed | Yes |
| S15 | Chinese evergreen | Partial occlusion | Complex | Yes |
| S16 | Rubber plant | Large leaves | Fixed | Yes |
| S17 | Rubber plant | Sparse canopy | Fixed | Yes |
| S18 | Cyclamen | Compact canopy | Fixed | Yes |
| S19 | Cyclamen | Flower-leaf mix | Complex | Yes |
| S20 | Cyclamen | Dense occlusion | Complex | Yes |

### 3.3 FSAM3：Frequency-Spatial 植物 mask 先验

图 2 展示了 FSAM3 的植物前景先验生成流程。其输入为每株植物的原始多视角 RGB 帧，通过给定的语义提示词，一次性输出与训练视角对齐的二值 mask、RGBA alpha 图像和 foreground-only RGB 图像。FSAM3 将频域清晰度评价与 promptable segmentation 结合，以减少低质量帧和背景误分割对重建的影响。FSAM3 不作为独立分割 benchmark 来定义贡献；它的主要功能是为 ForeSplat 提供可追踪的前景对象边界、透明度监督和后续 Gaussian 清理线索。

**图 2｜FSAM3 植物前景先验生成流程。** 原始帧经过 FFT screening、SAM3 text-prompted segmentation 和 PCA/main-component refinement 后，输出 binary masks、RGBA masks 和 foreground RGB images。

#### 3.3.1 FFT 帧质量筛选

多视角植物采集容易产生运动模糊、失焦和低纹理帧，尤其是在手动拍摄植株时、停止和叶片轻微摆动时。这些帧会干扰 SfM 特征匹配，也会使逐帧分割边界不稳定。因此，FSAM3 在分割和重建前先计算频域清晰度分数。频域高频能量和无参考清晰度指标常用于表征失焦或模糊程度 [7,36]，多尺度模糊估计也可作为清晰度评价的补充依据 [37]。

对每一帧 \(I\)，计算二维 Fast Fourier Transform (FFT) 幅值谱。高频能量比定义为：

\[
Q_{\text{FFT}}(I) = \frac{\sum_{(u,v) \in H} |F(u,v)|}{\sum_{(u,v) \in \Omega} |F(u,v)|}
\]

其中，\(F(u,v)\) 表示频率 \((u,v)\) 处的 FFT 幅值，\(\Omega\) 表示完整频域，\(H\) 表示频率范围上 50% 的高频带。对每个序列单独计算 \(Q_{\text{FFT}}\) 分布，并以第一四分位数作为样本特定阈值。低于该阈值的帧被排除在后续 FSAM3 和 COLMAP 输入之外。该设置剔除低质量帧，同时保留大多数视角以维持多视角覆盖。

#### 3.3.2 SAM3 文本 prompt 分割

质量筛选后的帧输入 SAM3 promptable segmentation 模块。Promptable segmentation 已在通用视觉分割中显示出跨类别迁移潜力 [8,33]，并被用于农业植物和杂草分割任务 [34,35]。植物表型场景中的 prompt 内容会影响叶片和植株区域的分割稳定性 [38]。本文使用文本 prompt 生成植物前景 mask，并比较五个 prompt：

| Prompt ID | Prompt text | 目标覆盖范围 |
|-----------|-------------|--------------|
| P1 | `green plant` | 宽泛植物区域 |
| P2 | `entire plant excluding pot` | 不含花盆的完整植物体 |
| P3 | `leaves and stems` | 地上部营养器官 |
| P4 | `crop seedling` | 小型或幼苗形态 |
| P5 | `plant body without background` | 去除背景的完整植物前景 |

P2 被设为所有重建实验的默认 prompt。该选择有两个方法学理由：花盆会污染 plant-only Gaussian 表示；株高测量需要从盆沿或植物基部参照向上计算，不能把容器作为植物几何的一部分。对每个保留视角 \(i\)，分割模块输出二值 mask \(M_i \in \{0,1\}^{H \times W}\)，其中 \(M_i(p)=1\) 表示像素 \(p\) 属于植物前景。当前重建运行中使用的 mask 二值化阈值为 0.5。

#### 3.3.3 主前景精炼与文件对齐

SAM3 输出可能包含小型假阳性碎片和叶片间孔洞。FSAM3 使用三步后处理得到训练可用 mask。首先，使用 5×5 椭圆核进行形态学闭运算，填补植物区域内的小孔洞。然后，执行 8 连通域分析，移除面积低于图像面积 0.5% 的小组件。最后，当序列中仍存在多个大组件时，对组件的边界框坐标进行 PCA，保留跨视角位置最稳定的主组件作为植物前景。

由于 mask 直接参与 RGB loss、alpha loss、foreground track initialization 和后续掩膜引导的多线索 Gaussian 剪枝，图像与 mask 的对齐是该流程的必要条件。

#### 3.3.4 分割 benchmark 协议

为验证 FSAM3 是否能提供适合重建的植物前景先验，本文使用人工标注的植物 mask 对其进行独立评价，并与 SEEM 进行比较。人工标注将植物地上部作为前景，排除花盆、土壤、桌面、支架和背景杂物。所有方法在相同图像和相同评价分辨率下生成二值 mask，并使用 F1-score、mean Intersection over Union (mIoU) 和 Hausdorff distance 95th percentile (HD95) 评价区域重叠与边界误差。该 benchmark 的目的不是声明 FSAM3 在通用分割任务上达到 SOTA，而是检验它在本文采集条件下是否比通用 promptable segmentation baseline 产生更适合三维重建的植物前景 mask。

### 3.4 Plant-aware 2DGS：前景对象重建

**图 3｜Foreground-object 2DGS objective 与标准 2DGS 的差异。** 标准 2DGS 在完整图像域上优化 RGB reconstruction；ForeSplat 将初始化、RGB supervision、alpha consistency 和 background opacity penalty 均绑定到 FSAM3 前景 mask，从训练阶段抑制非植物结构进入 Gaussian 表示。

#### 3.4.1 相机位姿估计与前景初始化

相机内外参和稀疏三维点轨迹由 COLMAP 的默认增量 SfM 流程估计 [11]。输入 FFT 筛选后的帧，而不是完整原始视频帧。标准 2DGS 会从全部 SfM 稀疏点初始化 Gaussian 基元，这会在优化开始前把背景点引入模型 [4]。ForeSplat 在初始化阶段使用多视角 mask 一致性过滤稀疏点。

设稀疏三维点 \(X_j\) 在视角集合 \(V_j\) 中可见，其在视角 \(i\) 中的投影为 \(\pi_i(X_j)\)。保留规则为：

\[
\operatorname{Keep}(X_j) = 1,\quad
\text{if}\quad
\frac{1}{|V_j|}\sum_{i \in V_j} M_i(\pi_i(X_j)) \geq \tau_{\text{track}} .
\]

其中，\(M_i\) 为第 \(i\) 个视角的 FSAM3 前景 mask。本文的前景对象重建目标完整配置和紧凑化前景对象重建配置均要求每个初始化点至少被 3 个视角观测，\(\tau_{\text{track}}=0.9\)，mask 膨胀像素数为 0。该设置使初始 Gaussian 集偏向植物区域，并减少后续优化对背景结构的容量分配。

#### 3.4.2 前景 RGB 监督与透明度约束

标准 3DGS/2DGS 通常在完整图像域 \(\Omega\) 上优化 RGB 重建 [4,25]：

\[
L_{\text{rgb-full}} =
\frac{1}{|\Omega|}
\sum_{p \in \Omega} \|R(p)-I(p)\|_1 ,
\]

其中，\(I(p)\) 为真实图像像素，\(R(p)\) 为渲染像素。该目标会奖励模型重建所有可见内容，包括非植物背景。为使训练目标与表型测量对象一致，本文将 RGB loss 限制在 mask 前景区域：

\[
L_{\text{rgb-fg}} =
\frac{1}{|\Omega_{\text{fg}}|}
\sum_{p \in \Omega}
M(p)\|R(p)-I(p)\|_1,\quad
\Omega_{\text{fg}}=\{p \mid M(p)=1\}.
\]

在本文中，foreground RGB loss 权重为 1.0，mask 外 RGB 权重为 0.0，并在前景裁剪时使用 12 px padding。该设置保留叶缘附近的局部上下文，但不让背景像素参与光度监督。

为约束透明度场，本文加入两个辅助损失。Alpha mask loss 约束渲染 alpha 图 \(A(p)\) 与前景 mask 一致：

\[
L_{\text{mask}} =
\frac{1}{|\Omega|}
\sum_{p \in \Omega} |A(p)-M(p)| .
\]

Background opacity loss 惩罚 mask 外非零透明度：

\[
L_{\text{bg}} =
\frac{1}{|\Omega_{\text{bg}}|}
\sum_{p \in \Omega}
(1-M(p))A(p),\quad
\Omega_{\text{bg}}=\{p \mid M(p)=0\}.
\]

前景对象重建目标完整配置的优化目标为：

\[
L_{\text{core}} =
L_{\text{rgb-fg}}
+\lambda_{\text{mask}}L_{\text{mask}}
+\lambda_{\text{bg}}L_{\text{bg}}
+L_{\text{reg}},
\]

其中，\(L_{\text{reg}}\) 包括 2DGS 的深度畸变损失和法向一致性损失。本文报告运行使用 \(\lambda_{\text{mask}}=0.08\)、\(\lambda_{\text{bg}}=0.02\)，mask loss 类型为 `l1_dice`，忽略 mask 边界 2 px，mask loss 从第 500 次迭代开始，并在 1500 次迭代内 warm-up。Foreground RGB supervision 是任务重定义的核心；alpha mask loss 与 background opacity loss 作为辅助约束，不单独承担 plant-only 重建目标。

### 3.5 视角质量感知的软损失加权

多视角植物序列的视角质量并不均一。模糊、遮挡、局部反光和视角边缘叶片运动会降低单帧质量。直接删除低质量视角是一种直观策略，但植物薄叶和遮挡结构可能只在少数角度可见。已有视角规划和 MVS 质量控制研究表明，视角覆盖会直接影响三维重建完整性 [29,30]。因此，删除视角会破坏角度覆盖，使前景对象重建缺少必要几何约束。

视角质量感知的软损失加权保留全部训练视角，只调节每个视角对前景 RGB loss 的贡献：

\[
L_{\text{rgb-fg-soft}} =
\frac{\sum_i q_i L_{\text{rgb-fg}}(i)}
{\sum_i q_i}.
\]

其中，\(q_i\) 是第 \(i\) 个视角的质量权重。当前实现读取预先计算的 H-VQG soft weight 文件，并使用 `view_weight_mode=rgb_only`，即权重仅作用于 RGB loss，不删除视角，也不改变 COLMAP 位姿。权重被限制在 0.6-1.0，默认值为 1.0。质量分数综合 mask 覆盖率、mask 边界锐度和前景 RGB 对比度三类线索，用于降低弱质量视角的梯度贡献，同时保留其几何覆盖。

### 3.6 紧凑前景清理

经过前景对象重建目标的完整配置或视角质量感知的软损失加权训练后，mask 边界和遮挡区域附近仍可能存在冗余或支撑较弱的 Gaussians。Gaussian 表示的压缩和剪枝通常依赖透明度或重要性分数 [5,6]，也可利用训练过程中的梯度和可见性线索 [9]。本文在训练后期执行 mask-guided pruning，用于得到更紧凑、便于导出的 plant-only 表示。对每个 Gaussian \(g_j\)，清理依据包括 mask 投影一致性 \(M_j\)、透明度 \(O_j\)、可见视角数 \(V_j\)、亮度或颜色正常性 \(B_j\)，以及局部连通或拓扑线索 \(C_j\)。可写为：

\[
\operatorname{Score}(g_j)=
\alpha M_j+\beta O_j+\gamma V_j+\delta B_j+\eta C_j .
\]

本文将掩膜引导的多线索 Gaussian 剪枝定位为 compactness 和 export cleanup 模块，而不是 foreground-object reconstruction 的根本来源。当前紧凑化前景对象重建配置使用 `pruning_mode=mask`，从第 18,000 次迭代开始，每 3,000 次迭代执行一次。主要阈值包括 opacity threshold = 0.005、brightness threshold = 0.01、mask threshold = 0.45、max views = 12、max remove ratio = 0.03 和 mask score weight = 3.0。每次剪枝均保存 pruning report，以便追踪被移除 Gaussians 的数量和依据。

### 3.7 网格提取、尺度恢复与表型测量

Plant-only Gaussian 表示通过 TSDF-style fusion 转换为显式网格。TSDF 融合和 Marching Cubes 是从深度观测提取显式表面的经典流程 [28,39]。Gaussian-to-mesh 方法也常利用显式或隐式表面对齐约束 [26]。首先从训练视角渲染深度图，然后在体素网格中融合 truncated signed distance field。对体素中心 \(x\)，融合距离定义为：

\[
D(x) =
\frac{\sum_c w_c(x)d_c(x)}
{\sum_c w_c(x)},
\]

其中，\(d_c(x)\) 是相机 \(c\) 下的局部截断有符号距离，\(w_c(x)\) 是对应融合权重。零水平集通过 Marching Cubes 提取。当前 mesh-only 评价比较三种变体：standard TSDF、smaller truncation TSDF 和 post-boundary cleanup。报告运行使用 iteration = 30000、voxel size = 0.02、depth truncation = 6.0、mesh resolution = 256 和 cluster number = 20。Standard 与 post-boundary 使用 \(sdf\_trunc=0.08\)，smaller truncation 使用等效 0.5 的 edge truncation scale，post-boundary cleanup 使用 `boundary_shrink_ratio=0.08`。

SfM 与 Gaussian 表示只确定到相似变换尺度。本文使用已知物理参照恢复绝对尺度，当前草稿采用花盆直径作为尺度参照。尺度恢复后提取四类虚拟表型：株高、冠幅、叶长和叶宽。株高定义为从盆沿或植物基部参照到最高可见植物点的垂直距离；冠幅定义为尺度恢复后植物在水平面的最大展开；叶长沿可见叶片中脉或主长轴测量；叶宽在近似垂直于叶长方向的最宽横截面处测量。每株植物贡献 1 个株高、1 个冠幅，以及 3 组代表性叶长/叶宽测量。虚拟测量与人工测量使用同一性状定义进行比较。

### 3.8 实验矩阵与验证设计

本文结合端到端应用验证、外部重建 baseline 对比和受控机制消融来验证方法。完整 ForeSplat/F2DMAS 流程在全部 20 个多视角序列上运行，用于评估复杂背景和不同植物结构下的工作流完成率、重建质量、mesh 提取效率和表型测量一致性。外部重建 baseline 包括 COLMAP、3DGS-FSAM3、标准 2DGS 和 SuGaR；这些方法用于比较传统 SfM/MVS、三维 Gaussian 表示、标准 2D surface Gaussian 表示和 3DGS-to-mesh 表面对齐流程在植物三维表型场景中的重建质量、处理时间和 mesh 可用性。比较时，方法共享相同采集序列和测试视角；需要前景输入的方法使用同一 FSAM3 输出或同一评价 mask；最终渲染质量和表型测量均使用相同评价脚本和性状定义。SuGaR 的跨方法定量结果采用 2DGS 与 SuGaR 共有的 11 个序列，并在代表性重叠叶区域进行 mesh 定性比较；因此相关结论限定为 Gaussian-based baseline comparison，而不扩大为所有三维重建范式的全面横评。

受控消融用于回答 ForeSplat 内部机制问题，而不承担全数据集泛化的唯一证据。前景对象目标函数消融用于分离 mask 预处理、透明度约束、前景 RGB 监督和前景初始化的作用：

- 无前景约束的整场景 2DGS 重建基线：不使用 mask 约束。
- 输入域前景掩膜约束：仅使用 foreground-only RGB 作为输入预处理，不改变 2DGS 目标函数。
- alpha 掩膜一致性单项约束：在 full-scene RGB supervision 下仅加入 alpha mask loss。
- 背景不透明度抑制单项约束：在 full-scene RGB supervision 下仅加入 background opacity loss。
- alpha 掩膜一致性与背景不透明度联合正则化：在 full-scene RGB supervision 下同时加入 alpha mask loss 和 background opacity loss。
- 前景限定 RGB 监督与透明度场联合正则化：使用 foreground RGB supervision、alpha mask loss 和 background opacity loss，但不使用 foreground track initialization。
- 前景对象重建目标的完整配置：在前景限定 RGB 监督与透明度场联合正则化基础上加入 foreground track initialization。

整场景表征的后验掩膜剪枝对照用于检验 full-scene training 后处理剪枝是否等价于 foreground-object training。该变体先训练无前景约束的整场景 2DGS 重建基线 30,000 次迭代，再剪除在超过 50% 训练视角中投影中心落在前景 mask 外的 Gaussians，并使用同一 foreground-only 指标评价剪枝后模型。视角质量实验比较前景对象重建目标的完整配置、质量阈值驱动的硬性视图剔除、掩膜质量驱动的视图剔除和视角质量感知的软损失加权。紧凑化前景对象重建配置实验比较前景对象重建目标的完整配置、仅加入视角质量感知的软损失加权、仅加入掩膜引导的多线索 Gaussian 剪枝，以及同时加入视角质量感知的软损失加权和掩膜引导的多线索 Gaussian 剪枝。网格实验评价结构、处理时间和代表性重叠叶区域的几何可用性，但不把某一 TSDF 变体写成已证明能提升所有表型精度的因果证据。

### 3.9 评价指标

分割质量在人工 mask 标注子集上使用 F1-score、mean Intersection over Union (mIoU) 和 Hausdorff distance 95th percentile (HD95)。这些指标用于描述 mask 区域重叠和边界偏差，在植物图像分割数据集和评测中较为常见 [40,41]。SAM 系列在农业植物和杂草分割中的评价也常使用区域重叠和边界相关指标 [34,35]。本文使用该 benchmark 验证 FSAM3 是否能提供稳定的重建先验，而不把 FSAM3 声明为通用植物分割 SOTA。

前景重建质量使用 mask 内指标和 mask 外泄漏指标共同评价。PSNR_fg 和 SSIM_fg 仅在 mask 前景像素上计算，其中 SSIM 用于度量结构相似性 [42]。LPIPS_fg 在将背景设为黑色后计算，用于比较前景视觉差异 [43]。outside_nonblack_ratio_mean 定义为 mask 外渲染 RGB 平均强度超过 \(\tau_{\text{black}}\) 的像素比例，本文使用归一化强度阈值 \(\tau_{\text{black}}=10/255\)，数值越低表示背景可见残留越少。leakage_energy_ratio_mean 定义为 mask 外渲染能量与 mask 内渲染能量之比，数值越低表示能量泄漏越少。本文采用以下操作性 foreground-only 标准：

\[
\text{outside\_nonblack\_ratio\_mean}<0.05,\quad
\text{leakage\_energy\_ratio\_mean}<0.10.
\]

模型紧凑性以 30,000 次训练迭代后的 Gaussian 总数衡量。网格结构使用顶点数、连通分量数、最大连通分量比例、边界边数量、边界一致性、mean/P95 displacement 和 mesh wall time 描述。表型测量精度使用 Mean Absolute Error (MAE)、Root Mean Square Error (RMSE)、Mean Absolute Percentage Error (MAPE)、Bias（平均有符号误差）和 Pearson \(R^2\)。所有方法变体使用相同的数据划分、mask、训练迭代数和评价脚本，以保证消融比较只反映被启用模块的差异。

\[
\operatorname{MSE}_{fg} =
\frac{\sum_p M(p)\|R(p)-I(p)\|_2^2}{\sum_p M(p)},\quad
\operatorname{PSNR}_{fg}=10\log_{10}\frac{\operatorname{MAX}_I^2}{\operatorname{MSE}_{fg}} .
\]

\[
\operatorname{outside} =
\frac{\sum_p(1-M(p))\mathbf{1}[\|R(p)\|_1/3>\tau_{\text{black}}]}
{\sum_p(1-M(p))},
\quad
\operatorname{leakage} =
\frac{\sum_p(1-M(p))\|R(p)\|_2^2}
{\sum_pM(p)\|R(p)\|_2^2+\epsilon}.
\]

\[
\operatorname{RMSE}=\sqrt{\frac{1}{n}\sum_i(\hat y_i-y_i)^2},\quad
\operatorname{MAE}=\frac{1}{n}\sum_i|\hat y_i-y_i|,\quad
\operatorname{MAPE}=\frac{100\%}{n}\sum_i\left|\frac{\hat y_i-y_i}{y_i}\right|.
\]

其中，\(\hat y_i\) 为虚拟测量值，\(y_i\) 为人工测量值；Bias 定义为 \(\frac{1}{n}\sum_i(\hat y_i-y_i)\)，即虚拟测量减人工测量。

### 3.10 实现细节与可复现性

所有 2DGS 实验均在单块 NVIDIA RTX 3090 GPU（24 GB）上运行。基础 2DGS 实现改编自 Huang 等人的官方代码库 [4]。训练运行 30,000 次迭代，保存 7,000 和 30,000 次迭代 checkpoint，评价使用 30,000 次迭代模型。训练分辨率参数为 4。Gaussian 位置初始学习率为 \(1.6 \times 10^{-4}\)，并在最终迭代衰减至 \(1.6 \times 10^{-6}\)。光度损失中的 D-SSIM 权重设为 0.2。深度畸变权重 \(\lambda_d\) 和法向一致性权重 \(\lambda_n\) 保持 2DGS 默认值，分别为 100 和 0.05。

FSAM3 使用项目环境中的 SAM3 为基础视觉模型进行修改得来。单株植物从原始图像到表型报告的处理时间约为 55 min，其中 COLMAP 约 15 min，FSAM3 mask generation 约 8 min，2DGS training 约 25 min，mesh extraction 约 5 min，measurement 约 2 min。代码将在发表时通过项目仓库提供；数据、mask、权重文件和运行配置应与代码一起归档，以保证前景对象目标函数消融、视角质量感知的软损失加权、掩膜引导的多线索 Gaussian 剪枝和 mesh-only 评价可复现。永久链接和 DOI 将在最终投稿版本的数据可用性声明中补齐。

---

## 4. 实验

本节从定量角度评估 ForeSplat/F2DMAS 的整体性能与关键机制。首先评估 FSAM3 是否能在人工标注 mask benchmark 上提供稳定植物前景先验；随后报告 20 个序列上的数据覆盖和端到端工作流执行情况；再通过前景对象目标函数消融与整场景表征的后验掩膜剪枝对照检验 foreground-object training 的必要性；之后与 COLMAP、标准 2DGS、3DGS-FSAM3 和 SuGaR 等外部重建流程比较重建质量、处理效率和 mesh 可用性；最后报告视角质量策略、紧凑化前景对象表示、网格结构和人工-虚拟表型验证。涉及 foreground-only 重建的消融结果采用 Section 3.9 中定义的标准，即 outside_nonblack_ratio_mean < 0.05 且 leakage_energy_ratio_mean < 0.10。

### 4.1 FSAM3 分割 benchmark 与重建先验分析

FSAM3 在 20 个样本中均生成了可用于重建的植物前景 mask。使用默认 prompt P2（"entire plant excluding pot"）时，所有序列均完成 mask 生成。固定装置辅助采集和复杂背景采集分别保留 2104 和 2113 帧有效图像，并分别获得 2040 和 2089 个 SfM 注册视角。该结果说明，FSAM3 输出能够支持后续 SfM 和 2DGS 重建流程。

五个 prompt 的敏感性分析显示，P2（"entire plant excluding pot"）和 P5（"plant body without background"）在跨物种样本中产生最一致的前景区域。P1（"green plant"）在个别序列中包含绿色背景对象；P3（"leaves and stems"）在木本或粗茎样本中漏分部分茎结构；P4（"crop seedling"）对成熟植株存在欠分割。PCA 主成分精炼将平均组件数从每帧 12.4 个降至 4.1 个，降幅为 67%，并在 98.2% 的帧中保留主导植物区域。

为量化 mask 质量，本文进一步使用人工标注的植物前景 mask 比较 FSAM3 与 SEEM。定性结果显示，FSAM3 在叶缘、重叠区域和细小结构处保留更完整，SEEM 在复杂背景、局部遮挡和背景颜色接近植物时更容易出现细结构漏分、局部叶片缺失和边界偏移。表 3 的定量结果与该观察一致：FSAM3 的 F1-score 为 98.3%，mIoU 为 97.9%，分别比 SEEM 高 3.2 和 3.8 个百分点；HD95 从 SEEM 的 281.9 px 降至 41.4 px。较低的边界误差对后续 plant-only 重建尤其重要，因为 mask 边界直接影响 RGB 监督域、透明度约束和 Gaussian 清理。

**表 3｜FSAM3 与 SEEM 的植物前景分割 benchmark。**

| Method | F1-score ↑ | mIoU ↑ | HD95 / px ↓ |
|---|---:|---:|---:|
| SEEM | 95.1 | 94.1 | 281.9 |
| FSAM3 | 98.3 | 97.9 | 41.4 |
| Improvement | +3.2 | +3.8 | -240.5 |

图 4 展示 FSAM3 和 SEEM 在复杂背景下的代表性对比。每行包含 RGB 图像、人工标注 mask、SEEM mask、FSAM3 mask 和误差图，其中 false positive 用红色、false negative 用蓝色标出。该图用于说明 FSAM3 的作用是提供更稳定的重建前景先验，而不是把 FSAM3 声明为通用分割 SOTA。

**图 4｜FSAM3 与 SEEM 的人工真值分割对比。** 代表性样本覆盖宽叶、薄叶、密集遮挡和复杂背景条件；误差图突出 SEEM 的细结构漏分、局部叶片缺失和边界偏移，以及 FSAM3 在叶缘和重叠区域的边界稳定性。

### 4.2 20 序列工作流执行与应用级重建结果

20 个多视角序列用于评估 ForeSplat/F2DMAS 从视频到 phenotype-ready mesh 的端到端执行情况。表 4 汇总了两类采集场景的数据覆盖和成功率。固定装置辅助采集和复杂背景采集各包含 10 个样本，均完成完整重建与表型流程，成功率为 100%。复杂背景场景包含手持运动、背景杂物和光照变化，因此该结果支持流程在非理想室内或类温室条件下的应用可行性。

**表 4｜不同采集场景下的数据覆盖与工作流执行情况。**

| Scene | Species number | Samples number | Frames | Valid frames | SfM views | Success samples | Success rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed | 8 | 10 | 2502 | 2104 | 2040 | 10 | 100% |
| Complex | 7 | 10 | 2500 | 2113 | 2089 | 10 | 100% |

在代表性复杂背景样本上，FSAM3 预处理后的序列获得 210 个有效相机视角和 24,226 个 SfM 稀疏空间特征点，说明频域筛选和植物前景分割能够为后续 2DGS 重建提供稳定输入。完整流程的应用级重建质量达到 PSNR = 31.09 dB、SSIM = 0.9711 和 LPIPS = 0.0365；这些指标用于评价完整工作流输出的渲染质量和后续可测量 mesh 的输入质量。



### 4.3 前景对象目标消融与整场景表征后验剪枝比较

Foreground RGB supervision 是将 full-scene 2DGS 转为 foreground-object reconstruction 的决定性算法。表 5 报告了 KongQueZhuYu 样本上的目标函数系统消融，并加入 full-scene 训练后的整场景表征后验掩膜剪枝对照。该样本包含复杂室内背景和 27 个评价视角，适合检验背景泄漏是否被真正抑制。

**表 5｜KongQueZhuYu 上的 foreground-object objective 消融。** outside 与 leakage 越低越好；foreground-only 标准为 outside < 0.05 且 leakage < 0.10。整场景表征的后验掩膜剪枝对照为 full-scene 训练后的剪枝结果；其 LPIPS 和 Gaussian 数量在当前统计口径下未纳入统一统计，表中以 n/a 标记。

| 方法设置 | fg init | fg RGB | alpha/bg | PSNR_fg ↑ | SSIM_fg ↑ | LPIPS_fg ↓ | outside ↓ | leakage ↓ | Gaussians ↓ | FG-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 无前景约束的整场景 2DGS 重建基线 | no | no | no | 24.2090 | 0.8514 | 0.0480 | 0.9908 | 1.2201 | 751,213 | no |
| 输入域前景掩膜约束 | no | implicit | no | 20.7291 | 0.7505 | 0.0696 | 0.0073 | 0.0042 | 263,108 | yes, 质量下降 |
| alpha 掩膜一致性单项约束 | no | no | alpha | 24.3422 | 0.8478 | 0.0491 | 0.9898 | 1.2260 | 768,067 | no |
| 背景不透明度抑制单项约束 | no | no | bg | 24.7508 | 0.8672 | 0.0451 | 0.9900 | 1.2255 | 742,931 | no |
| alpha 掩膜一致性与背景不透明度联合正则化 | no | no | both | 24.8126 | 0.8687 | 0.0445 | 0.9896 | 1.2266 | 763,266 | no |
| 前景限定 RGB 监督与透明度场联合正则化 | no | yes | both | 25.1055 | 0.8561 | 0.0437 | 0.0294 | 0.0190 | 592,900 | yes |
| 前景对象重建目标的完整配置 | yes | yes | both | 25.0072 | 0.8548 | 0.0438 | 0.0294 | 0.0189 | 591,623 | yes |
| 整场景表征的后验掩膜剪枝对照 | no | no | post-hoc | 21.34 | 0.79 | n/a | 0.31 | 0.28 | n/a | no |

无前景约束的整场景 2DGS 重建基线在 mask 内达到 PSNR_fg = 24.2090，但 outside = 0.9908、leakage = 1.2201，表明 full-scene 2DGS 同时重建了几乎全部背景。输入域前景掩膜约束将背景置黑后训练，outside 和 leakage 降至 0.0073 和 0.0042，但 PSNR_fg 降至 20.7291，SSIM_fg 降至 0.7505。该结果限定了简单输入域 mask 约束的作用：它能压低背景，但不能保持前景质量。

alpha 掩膜一致性单项约束、背景不透明度抑制单项约束以及二者的联合正则化，在整图 RGB supervision 不变的条件下均无法阻止背景学习，三个变体的 leakage 均约为 1.22，与无前景约束基线接近。相反，启用 foreground RGB supervision 后，outside 从 alpha 掩膜一致性与背景不透明度联合正则化的 0.9896 降至 0.0294，leakage 从 1.2266 降至 0.0190，PSNR_fg 同时升至 25.1055。加入 foreground track initialization 后，前景对象重建目标的完整配置结果与前景限定 RGB 监督与透明度场联合正则化接近（PSNR_fg = 25.0072, outside = 0.0294, leakage = 0.0189），并用于后续跨样本实验。

整场景表征的后验掩膜剪枝对照检验了另一条路线：先训练 full-scene 模型，再剪除在超过 50% 训练视角中投影中心落在前景 mask 外的 Gaussians。该对照的 outside = 0.31、leakage = 0.28，分别为 foreground-only 阈值的 6.2 倍和 2.8 倍；相对于前景对象重建目标的完整配置，其 outside 和 leakage 分别约高 10.5 倍和 14.8 倍。该对照将结论限定在训练目标层面：后验掩膜剪枝不能替代 foreground-object optimization。

图 5 展示无前景约束基线、输入域前景掩膜约束、前景限定 RGB 监督与透明度场联合正则化、前景对象重建目标完整配置和整场景表征后验掩膜剪枝对照的测试视角渲染与背景泄漏热力图。该图的核心信息不是复述表 5，而是展示三种失败模式：无前景约束基线背景泄漏严重，输入域前景掩膜约束前景质量下降，post-hoc pruning 仍残留背景结构；foreground RGB supervision 相关方法同时保持前景质量并抑制背景。

**图 5｜Foreground-object objective 消融可视化。** 每列对应一个方法设置，每行展示 RGB 渲染、mask 外泄漏热力图和局部叶缘放大图；图中重点比较 full-scene training、输入域 mask 预处理、foreground RGB supervision 和 post-hoc pruning 的背景泄漏差异。

### 4.4 与外部重建流程的质量和效率比较

为检验 ForeSplat/F2DMAS 的应用价值是否超出内部消融，本文将完整流程与 COLMAP、3DGS-FSAM3、标准 2DGS 和 SuGaR 进行比较。COLMAP 代表传统 SfM/MVS 几何重建流程；3DGS-FSAM3 代表使用同一植物前景先验的三维 Gaussian 表示；标准 2DGS 用于比较在缺少完整前景预处理和 foreground-object 训练时的 surface Gaussian 重建效果；SuGaR 代表从 3DGS 出发的 surface-aligned Gaussian-to-mesh baseline。表 6 显示，COLMAP 的 PSNR、SSIM 和 LPIPS 分别为 13.63 dB、0.8745 和 0.1072；3DGS-FSAM3 为 30.17 dB、0.9587 和 0.0386；完整流程为 31.09 dB、0.9711 和 0.0365。与 3DGS-FSAM3 相比，完整流程在保持更高重建质量的同时将 mesh 提取时间从 642 s 降至 55 s，降低约 91.4%。与标准 2DGS 相比，完整流程将 PSNR 从 29.58 dB 提高到 31.09 dB，并将训练时间和 mesh 提取时间分别从 12,913.7 s 和 157.9 s 降至 5,044.5 s 和 55.0 s。

**表 6｜不同重建流程的重建质量与处理效率比较。**

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Train time / s ↓ | Mesh time / s ↓ |
|---|---:|---:|---:|---:|---:|
| COLMAP | 13.63 | 0.8745 | 0.1072 | 599.5 | 78 |
| 3DGS-FSAM3 | 30.17 | 0.9587 | 0.0386 | 5413.5 | 642 |
| Standard 2DGS | 29.58 | 0.9574 | 0.0487 | 12913.7 | 157.9 |
| ForeSplat/F2DMAS | 31.09 | 0.9711 | 0.0365 | 5044.5 | 55.0 |

SuGaR 的定量结果来自与标准 2DGS 共有的 11 个序列，因此在表 7 单独报告跨序列 Gaussian baseline 对比，避免与表 6 的单一应用级流程时间统计混用。共有序列上，标准 2DGS 30k 的平均 PSNR、SSIM 和 LPIPS 分别为 25.62 dB、0.9282 和 0.1022；Vanilla 3DGS 7k 为 22.12 dB、0.8892 和 0.1356；SuGaR refined 为 25.74 dB、0.9172 和 0.1061。SuGaR 在 PSNR 上略高于标准 2DGS，但平均 SSIM 和 LPIPS 未超过标准 2DGS；其 refined 表示平均包含约 449,868 个 surface-aligned Gaussians，高于标准 2DGS 30k 的约 168,032 个。该结果说明，SuGaR 是强 Gaussian-to-mesh baseline，但其优势主要体现在 surface-aligned mesh 表示和部分样本的 PSNR，而不是在所有渲染指标和表示紧凑性上全面优于标准 2DGS。ForeSplat/F2DMAS 的核心差异仍在于训练目标与 plant-only 测量对象对齐。

**表 7｜共有 11 个序列上的 2DGS、3DGS 和 SuGaR Gaussian baseline 对比。**

| Method | n seq | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Representation size ↓ |
|---|---:|---:|---:|---:|---:|
| Standard 2DGS 30k | 11 | 25.62 | 0.9282 | 0.1022 | 168,032 |
| Vanilla 3DGS 7k | 11 | 22.12 | 0.8892 | 0.1356 | 154,653 |
| SuGaR refined | 11 | 25.74 | 0.9172 | 0.1061 | 449,868 |

表 8 进一步分离 FFT 质量筛选和植物分割两个前端模块的作用。单独加入 FFT 后，PSNR 从 29.58 dB 提升至 29.80 dB，SSIM 从 0.9574 提升至 0.9623，mesh 时间从 157.9 s 降至 144.3 s；单独加入 SAM3/FSAM3 植物分割后，PSNR 提升至 30.50 dB，LPIPS 降至 0.0397，训练时间和 mesh 时间分别降至 6541.3 s 和 73.6 s；二者组合后达到最佳综合结果。该结果说明，频域筛选主要缓解模糊帧对多视角几何估计和优化的影响，植物前景分割主要减少背景区域进入 Gaussian 表示和网格化过程。

**表 8｜ForeSplat/F2DMAS 工作流模块消融。**

| Method | FFT | Plant Seg. | 2DGS | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Training time / s ↓ | Mesh time / s ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | no | no | yes | 29.58 | 0.9574 | 0.0487 | 12913.7 | 157.9 |
| Base + FFT | yes | no | yes | 29.80 | 0.9623 | 0.0453 | 12510.3 | 144.3 |
| Base + SAM3 | no | yes | yes | 30.50 | 0.9687 | 0.0397 | 6541.3 | 73.6 |
| ForeSplat/F2DMAS | yes | yes | yes | 31.09 | 0.9711 | 0.0365 | 5044.5 | 55.0 |

图 6 展示原始图像、SuGaR 或 3DGS 派生 mesh 与 ForeSplat/F2DMAS mesh 的代表性对比，重点放在重叠叶区域、叶缘边界和背景残留。外部 baseline 能恢复整体形态，但在重叠叶区域容易出现边界模糊、局部粘连或背景残留；ForeSplat/F2DMAS 的优势在于前景分割、surface-oriented 2DGS 表示和显式 mesh 输出共同提高了表型测量可用性。

**图 6｜外部 baseline 的代表性重建和 mesh 对比。** 图中比较 COLMAP、3DGS-FSAM3、SuGaR、标准 2DGS 与 ForeSplat/F2DMAS 的 RGB 渲染、mesh 输出和重叠叶局部放大图，用于展示背景残留、叶片粘连和边界可测量性差异。

### 4.5 代表性结构样本上的 foreground-only 重建验证

前景对象重建目标的完整配置在三个代表性结构样本上均满足 foreground-only 标准。表 9 报告了复杂背景、薄叶细结构和密集遮挡三类样本的结果。该表不是全数据集泛化的唯一证据，而是用于展示不同结构难度下 foreground-object 目标的边界行为；全数据集工作流执行情况已在表 4 报告。

**表 9｜前景对象重建目标完整配置的代表性样本前景重建结果。**

| 样本 | 结构角色 | PSNR_fg ↑ | SSIM_fg ↑ | LPIPS_fg ↓ | outside ↓ | leakage ↓ | Gaussians ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| KongQueZhuYu | 复杂背景 / 主样本 | 25.0072 | 0.8548 | 0.0438 | 0.0294 | 0.0189 | 591,623 |
| XianKeLai1 | 薄叶 / 细结构 | 23.7276 | 0.8278 | 0.0309 | 0.0484 | 0.0379 | 253,827 |
| CaoMei2 | 密集叶 / 遮挡 | 25.0833 | 0.8121 | 0.0250 | 0.0147 | 0.0081 | 370,844 |

CaoMei2 获得最低 leakage（0.0081）和最高 PSNR_fg（25.0833）。XianKeLai1 是三者中最接近阈值的样本，outside = 0.0484，leakage = 0.0379，但仍满足 foreground-only 标准。该结果表明 ForeSplat 支持前景对象重建目标完整配置在代表性植物结构上的稳定性。

### 4.6 硬性视图剔除破坏角度覆盖，软损失加权保持重建稳定性

视角质量不适合通过 hard filtering 直接删除视角。表 10 比较了 KongQueZhuYu 上的前景对象重建目标完整配置、质量阈值驱动的硬性视图剔除、掩膜质量驱动的视图剔除和视角质量感知的软损失加权。质量阈值驱动的硬性视图剔除删除 27 个视角中的 10 个；掩膜质量驱动的视图剔除删除 3 个 mask 质量较差的视角；视角质量感知的软损失加权保留全部视角，只调节它们在 foreground RGB loss 中的权重。

**表 10｜KongQueZhuYu 上的视角质量策略比较。**

| 方法设置 | Eval views | PSNR_fg ↑ | SSIM_fg ↑ | LPIPS_fg ↓ | outside ↓ | leakage ↓ | Gaussians ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 前景对象重建目标的完整配置 | 27 | 25.0072 | 0.8548 | 0.0438 | 0.0294 | 0.0189 | 591,623 |
| 质量阈值驱动的硬性视图剔除 | 17 | 12.5478 | 0.6018 | 0.1179 | 0.1743 | 0.3020 | 597,116 |
| 掩膜质量驱动的视图剔除 | 24 | 13.4557 | 0.6244 | 0.1115 | 0.1450 | 0.2848 | 579,612 |
| 视角质量感知的软损失加权 | 27 | 24.9566 | 0.8543 | 0.0440 | 0.0284 | 0.0184 | 532,264 |

质量阈值驱动的硬性视图剔除使 PSNR_fg 从 25.0072 降至 12.5478，SSIM_fg 从 0.8548 降至 0.6018，outside 从 0.0294 升至 0.1743。掩膜质量驱动的视图剔除的退化较轻，但 PSNR_fg 仍只有 13.4557，leakage 为 0.2848，未达到 foreground-only 标准。与这两个 hard filtering 变体相比，视角质量感知的软损失加权在保留 27 个视角的条件下使 PSNR_fg 仅下降 0.0506 dB，SSIM_fg 仅下降 0.0005，并将 Gaussian 数量减少 59,359 个（10.03%）。该结果把视角质量控制的作用限定为 soft weighting：质量分数用于调节训练贡献，而不是删除多视角覆盖。

图 7 同时显示质量阈值驱动硬性视图剔除的视角覆盖缺口、视角质量感知软损失加权的视角权重分布，以及 PSNR_fg 和 Gaussian count 的并列柱状图。该图用于连接性能退化与视角覆盖破坏，而不只是展示指标差异。

**图 7｜视角质量感知策略比较。** 图中展示质量阈值驱动硬性视图剔除后的角度覆盖缺口、视角质量感知软损失加权的视角权重热图，以及前景对象重建目标完整配置、质量阈值驱动硬性视图剔除、掩膜质量驱动视图剔除和视角质量感知软损失加权的 PSNR_fg、outside/leakage 与 Gaussian count 对比。

### 4.7 紧凑化 plant-only 表征

紧凑化前景对象重建配置在小幅前景质量变化内减少了 Gaussian 数量和背景泄漏。该配置定义为前景对象重建目标完整配置加视角质量感知的软损失加权和掩膜引导的多线索 Gaussian 剪枝。表 11 报告了三个代表样本上的闭环结果，并保留中间变体以区分视角质量感知的软损失加权和掩膜引导的多线索 Gaussian 剪枝的作用。

**表 11｜紧凑化前景对象重建配置的代表性样本紧凑性结果。** KongQueZhuYu 的“前景对象重建目标完整配置 + 掩膜引导的多线索 Gaussian 剪枝”单独变体未纳入当前闭环表，因此不填充该行。

| Sample | 方法设置 | PSNR_fg ↑ | SSIM_fg ↑ | LPIPS_fg ↓ | outside ↓ | leakage ↓ | Gaussians ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| CaoMei2 | 前景对象重建目标的完整配置 | 25.0833 | 0.8121 | 0.0250 | 0.0147 | 0.0081 | 370,844 |
| CaoMei2 | 前景对象重建目标完整配置 + 视角质量感知软损失加权 | 25.0046 | 0.8107 | 0.0253 | 0.0140 | 0.0077 | 249,944 |
| CaoMei2 | 前景对象重建目标完整配置 + 掩膜引导的多线索 Gaussian 剪枝 | 25.0303 | 0.8108 | 0.0251 | 0.0144 | 0.0080 | 284,757 |
| CaoMei2 | 紧凑化前景对象重建配置 | 24.9718 | 0.8101 | 0.0252 | 0.0136 | 0.0076 | 246,452 |
| XianKeLai1 | 前景对象重建目标的完整配置 | 23.7276 | 0.8278 | 0.0309 | 0.0484 | 0.0379 | 253,827 |
| XianKeLai1 | 前景对象重建目标完整配置 + 视角质量感知软损失加权 | 23.6632 | 0.8274 | 0.0312 | 0.0478 | 0.0374 | 220,947 |
| XianKeLai1 | 前景对象重建目标完整配置 + 掩膜引导的多线索 Gaussian 剪枝 | 23.7256 | 0.8279 | 0.0310 | 0.0486 | 0.0376 | 251,047 |
| XianKeLai1 | 紧凑化前景对象重建配置 | 23.7070 | 0.8273 | 0.0312 | 0.0479 | 0.0373 | 219,661 |
| KongQueZhuYu | 前景对象重建目标的完整配置 | 25.0072 | 0.8548 | 0.0438 | 0.0294 | 0.0189 | 591,623 |
| KongQueZhuYu | 前景对象重建目标完整配置 + 视角质量感知软损失加权 | 24.9566 | 0.8543 | 0.0440 | 0.0284 | 0.0184 | 532,264 |
| KongQueZhuYu | 紧凑化前景对象重建配置 | 24.9423 | 0.8540 | 0.0441 | 0.0284 | 0.0182 | 530,936 |

三个样本合计，紧凑化前景对象重建配置将 Gaussian 总数从 1,216,294 降至 997,049，减少 219,245 个，降幅为 18.03%。平均 PSNR_fg 下降 0.0657 dB，SSIM_fg 下降 0.0011，LPIPS_fg 增加 0.0003；outside 和 leakage 分别平均下降 0.0009 和 0.0006。CaoMei2 上的压缩最明显，Gaussian 数量从 370,844 降至 246,452，减少 33.54%，PSNR_fg 下降 0.1115 dB。XianKeLai1 的薄叶结构可剪除冗余较少，Gaussian 数量减少 13.46%，PSNR_fg 下降 0.0206 dB，outside 仍低于 0.05。

紧凑化前景对象重建配置的主要收益是紧凑性和更干净的导出表示，而不是相对于前景对象重建目标完整配置的前景质量提升。在 CaoMei2 上，视角质量感知的软损失加权单独减少 32.6% Gaussians，掩膜引导的多线索 Gaussian 剪枝单独减少 23.2% Gaussians，二者组合减少 33.5% Gaussians。组合收益不是简单相加；按当前数据，视角质量感知的软损失加权是主要压缩来源，掩膜引导的多线索 Gaussian 剪枝在此基础上提供额外但较小的边界清理。

图 8 以前景对象重建目标完整配置与紧凑化前景对象重建配置的并列柱状图展示 Gaussian count、PSNR_fg、outside 和 leakage。该图突出紧凑化前景对象重建配置的主要 trade-off：Gaussian count 明显下降，前景质量指标变化很小。

**图 8｜前景对象重建目标完整配置与紧凑化前景对象重建配置的紧凑性权衡。** 图中按 KongQueZhuYu、XianKeLai1 和 CaoMei2 三个样本展示 Gaussian count、PSNR_fg、outside_nonblack 和 leakage_energy 的并列柱状图。

### 4.8 网格结构评估

网格实验提供结构和效率证据，但不直接证明表型精度提升。表 12 比较了 KongQueZhuYu 和 XianKeLai1 在 Standard TSDF、Smaller truncation TSDF 和 Post-boundary cleanup 下的网格结构。

**表 12｜紧凑化前景对象重建配置输出的网格结构与效率指标。** 位移指标只对 post-boundary cleanup 报告；mesh time 为 `render.py --skip_train --skip_test` 的 mesh-only wall time。

| Sample | Mesh variant | Vertices | Components | Largest comp. | Small comps | Boundary edges | Boundary consistency | Mean disp. | P95 disp. | Time/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KongQueZhuYu | Standard TSDF | 167,789 | 8 | 0.9920 | 5 | 12,088 | - | - | - | 53.33 |
| KongQueZhuYu | Smaller truncation | 147,665 | 20 | 0.9350 | 12 | 25,086 | - | - | - | 56.52 |
| KongQueZhuYu | Post-boundary | 167,789 | 8 | 0.9920 | 5 | 12,088 | 0.9631 | 0.0041 | 0.0222 | 58.26 |
| XianKeLai1 | Standard TSDF | 74,753 | 6 | 0.9488 | 0 | 6,956 | - | - | - | 78.15 |
| XianKeLai1 | Smaller truncation | 66,138 | 12 | 0.9487 | 5 | 9,763 | - | - | - | 78.57 |
| XianKeLai1 | Post-boundary | 74,753 | 6 | 0.9488 | 0 | 6,956 | 0.8278 | 0.0121 | 0.0376 | 97.10 |

Smaller truncation 在 KongQueZhuYu 和 XianKeLai1 上分别减少 12.0% 和 11.5% 顶点，但连通分量从 8 增至 20、从 6 增至 12，边界边数量也增加。该结果将 smaller truncation 的作用限定为更紧凑但更易碎片化的网格化策略。Post-boundary cleanup 保持两个样本的连通分量数量不变，并报告边界一致性和位移；XianKeLai1 的 boundary consistency 低于 KongQueZhuYu（0.8278 vs. 0.9631），mean displacement 更高（0.0121 vs. 0.0041）。该差异与薄叶边界更敏感的观察一致。

Post-boundary cleanup 增加 mesh wall time：KongQueZhuYu 从 53.33 s 增至 58.26 s，XianKeLai1 从 78.15 s 增至 97.10 s。当前结果只支持 mesh structural and efficiency evaluation，尚不能证明某一网格变体改善叶宽或整体表型精度；这一声明需要网格变体前后的表型误差对比。

图 9 展示两列样本、三行网格变体，并在叶缘或植株边界处加入局部放大图。图注同时报告碎片化风险和 post-boundary 的时间代价。

**图 9｜网格结构和边界处理可视化。** 图中两列展示 KongQueZhuYu 与 XianKeLai1，三行展示 Standard TSDF、Smaller truncation 和 Post-boundary cleanup，并加入叶缘或植株边界局部放大图。

### 4.9 表型验证

跨 20 个序列、21 株植物的人工-虚拟测量结果显示，ForeSplat/F2DMAS 对全局范围性状更稳定，对薄维度性状误差更高。表 13 中 Bias 定义为虚拟测量减人工测量；叶长和叶宽每株植物包含 3 组代表性叶片测量，因此 n = 63。

**表 13｜人工测量与虚拟表型测量的一致性。**

| Trait | n | MAE/cm ↓ | RMSE/cm ↓ | MAPE/% ↓ | Bias/cm | R² ↑ |
|---|---:|---:|---:|---:|---:|---:|
| 株高 | 21 | 0.98 | 1.21 | 6.91 | 0.58 | 0.9878 |
| 冠幅 | 21 | 0.86 | 0.99 | 4.50 | 0.64 | 0.9879 |
| 叶长 | 63 | 0.51 | 0.64 | 7.45 | 0.31 | 0.9738 |
| 叶宽 | 63 | 0.45 | 0.64 | 9.73 | 0.38 | 0.8999 |

株高和冠幅具有最高一致性，R² 分别为 0.9878 和 0.9879，MAPE 分别为 6.91% 和 4.50%。叶长达到 R² = 0.9738，MAPE = 7.45%。叶宽的 R² 为 0.8999，MAPE 为 9.73%，是四类性状中误差最高的指标。所有性状均为正偏差，范围为 0.31-0.64 cm，说明当前虚拟测量相对人工测量存在轻微高估。

该结果支持 ForeSplat/F2DMAS 用于株高、冠幅、叶长和叶宽的自动化测量，但边界条件清晰：叶宽等薄维度性状最容易受到重建分辨率和网格边界的影响。该边界与 Section 4.8 中 XianKeLai1 边界一致性较低、位移较高的结果相互对应。图 10 采用 2 x 2 散点图展示人工测量与虚拟测量关系，并在每个子图中标注 n、R² 和 y = x 参考线；叶宽子图加入 Bland-Altman inset，以直接显示 bias 和一致性范围。

**图 10｜人工测量与虚拟表型测量一致性。** 2 × 2 散点图分别展示株高、冠幅、叶长和叶宽，包含 y = x 参考线、R²、n 和误差统计；叶宽子图加入 Bland-Altman inset 以显示 bias 和一致性范围。

---

## 5. 讨论

本研究表明，面向植物表型的三维重建不应只追求完整场景的视觉还原，而应直接生成与测量目标一致的 plant-only 表示。ForeSplat 的主要价值在于把多视角图像质量控制、植物前景 mask 生成、foreground-object 2DGS 优化、视角质量感知的软损失加权和掩膜引导的 Gaussian 表征紧凑化连接成一条可验证的证据链。结果说明，背景泄漏主要来自训练目标与测量目标不一致；一旦 RGB 监督被限定在植物前景，2DGS 的容量分配就会从场景级重建转向植物对象重建。该发现为低成本、多物种、半受控环境下的自动化植物表型提供了一个更直接的技术路径。

### 5.1 Foreground-object reconstruction 是训练目标重定义

一个关键结论是，plant-only 表示不能可靠地通过 full-scene 2DGS 的后处理剪枝获得。原因并不只是剪枝阈值不够精细，而是标准 2DGS 在训练阶段已经根据整图 RGB 监督分配了 Gaussian 容量。背景、花盆、桌面和支架同样产生光度梯度，并在 densification 过程中吸引基元。训练完成后再删除 mask 外 Gaussians，只能作用于已经形成的表示，不能恢复原本分配给植物前景的容量。无前景约束基线、foreground RGB supervision 相关方法与整场景表征的后验掩膜剪枝对照支持这一解释，也与 Gaussian splatting 中基于梯度进行 densification 的训练机制一致。

这一点也解释了为什么 alpha mask loss 和 background opacity loss 只能作为辅助项。它们可以约束渲染 alpha 场，却不能在整图 RGB supervision 仍然存在时阻止模型学习背景外观。Foreground RGB supervision 改变的是优化问题本身：模型只因植物像素的重建误差而获得主要梯度。由此，foreground-object reconstruction 不是 mask 后处理的一个版本，而是对 2DGS 训练目标的重新表述。

这种目标一致性对植物表型尤其重要。表型测量关注株高、冠幅、叶长和叶宽等植物结构，而非采集台或背景布的几何。若训练目标仍奖励非植物区域，后续网格提取和尺度测量就必须处理额外结构。ForeSplat 将测量对象提前写入训练损失，使输出表示更接近 phenotype-ready mesh 的需求。

### 5.2 FSAM3 的作用是生成重建先验，而非替代三维优化

FSAM3 的贡献在于把通用分割能力转化为可用于三维重建的前景先验。FFT 筛选减少低质量帧对 SfM、分割和重建的共同干扰；SAM3 文本 prompt 提供跨物种植物区域的初始定位；PCA 主前景精炼进一步抑制不连通假阳性组件。人工标注 mask benchmark 显示，FSAM3 相比 SEEM 具有更高 F1-score 和 mIoU，并显著降低 HD95，说明它在本文采集条件下提供了更完整、更稳定的植物边界。该结果支持 FSAM3 作为重建前处理模块的有效性，但并不意味着它在任意农业图像分割任务上达到通用最优。

这种定位有助于避免过度解释 mask 的作用。Mask 并没有直接给出最终三维形状，它只定义哪些像素应参与前景对象学习。最终几何仍由多视角一致性、2DGS 表面正则和 TSDF 网格化共同决定。因此，FSAM3 的误差会影响重建边界，但不会把二维 mask 简单复制为三维模型。后续更大规模的分割标注仍有价值，尤其可用于分析不同物种、光照和遮挡条件下的 segmentation error 与 reconstruction error 的耦合关系。

### 5.3 视角质量应被调制，而不是被消除

视角质量实验说明，多视角植物重建中的低质量帧不能简单等同于可删除样本。植物叶片常在狭窄角度范围内可见，某些视角即使光照或边界质量较弱，也可能提供独特的几何覆盖。Hard view filtering 删除这些视角后，剩余图像无法恢复缺失的观察方向。结果中的明显退化说明，视角覆盖是三维重建的结构性条件，而不是普通训练样本数量问题。

视角质量感知软损失加权的意义在于把几何覆盖与光度可靠性分开处理。所有视角保留在训练过程中，以维持对叶片表面、遮挡边界和冠层侧面的角度覆盖；质量权重只调节各视角对 foreground RGB loss 的贡献。这样，低质量视角不会以同等强度影响颜色和边界拟合，却仍能提供必要的观察约束。该结果支持一个更一般的原则：在薄结构对象重建中，视角质量更适合用于梯度调制，而不是用于二元保留或删除。

### 5.4 表征紧凑性是紧凑化前景对象重建配置的主要实用价值

紧凑化前景对象重建配置的目标不是在前景对象重建目标完整配置基础上显著提高前景图像质量，而是在保持前景质量基本稳定的同时减少冗余 Gaussians。对高通量植物表型而言，这一点具有实际意义。更紧凑的 plant-only 表示可以降低存储和渲染开销，缩短网格导出时间，并使单批次处理更多样本成为可能。

掩膜引导的多线索 Gaussian 剪枝比单纯透明度阈值更适合植物对象。叶缘、叶间孔洞和遮挡边界处的 Gaussians 往往具有模糊状态：透明度可能不低，但多视角 mask 支持不足；颜色可能接近叶片，但可见性或拓扑支撑较弱。将 mask 一致性、透明度、可见性、颜色正常性和拓扑线索组合起来，可以更稳妥地识别弱支撑基元。这个过程仍需保持保守，因为过强剪枝可能伤及薄叶和细小叶柄。当前结果更适合支持 compact foreground cleanup 的结论，而不是把掩膜引导的多线索 Gaussian 剪枝写成重建质量提升的主要来源。

### 5.5 表型测量揭示了全局性状与薄维度性状的差异

人工与虚拟测量的一致性表明，ForeSplat 输出的网格已经能够承载基本结构性状测量。株高和冠幅这类全局范围性状更稳定，因为它们由整体外包范围决定，对局部叶缘误差不太敏感。叶长处于中间难度。叶宽最难，因为它依赖局部边界、网格分辨率和叶片姿态，任何轻微的边界扩张都会在窄维度上被放大。

叶宽的正偏差与网格边界分析相互吻合。2D Gaussian 的投影支持域可能略微超出真实叶缘，TSDF 融合又会把这种边界不确定性带入显式网格。对于宽叶样本，这类偏差在相对尺度上较小；对于薄叶样本，它会直接影响测量值。因此，当前证据支持自动化表型测量的可行性，也同时指出叶宽是最需要改进的性状。改进方向应集中在更高分辨率采集、边界感知网格精炼、叶缘不确定性建模和测量 landmark 协议标准化。

### 5.6 与三维植物表型研究的关系

已有三维植物表型研究已经证明，MVS、NeRF、3DGS、2DGS 和 Gaussian-to-mesh 方法可以为植物结构测量提供基础表示。ForeSplat 的增量并不在于否定这些表示，而在于把 2DGS 的训练目标改写为植物前景对象重建。对于薄叶和密集冠层，2DGS 的平面基元提供了合适的几何基础；但如果优化目标仍是 full-scene reconstruction，模型仍会学习大量非植物结构。本文与 COLMAP、标准 2DGS、3DGS-FSAM3 和 SuGaR 的对比表明，完整流程的收益不仅来自 Gaussian 表示本身，也来自前景先验、质量筛选、foreground-object 训练目标和面向测量的 mesh 输出之间的协同。

相较于物种特异性流程，ForeSplat 还提供了一个模块化跨物种方案。FFT screening、SAM3+PCA mask generation、foreground-object 2DGS、视角质量感知的软损失加权和掩膜引导的多线索 Gaussian 剪枝均可单独替换或改进。实践者可以只采用其中某一组件，例如在已有 3DGS 工作流中加入前景 RGB supervision，或在既有 mask 管线后加入 soft view weighting。这样的模块化设计降低了方法迁移成本，也便于后续研究定位误差来源。

### 5.7 边界条件与未来方向

这些结论应放在当前数据和采集条件内解释。本文已经在 20 个多视角序列上完成工作流验证，并在代表性样本上完成 foreground-object 目标函数、视角质量和紧凑化机制消融。全流程数据支持室内或半受控复杂背景下的多植物类型应用可行性；机制消融则用于解释为什么前景 RGB 监督、soft weighting 和 mask-guided cleanup 有效。当前采集仍主要发生在固定装置辅助和复杂室内背景条件下，田间风致运动、强光阴影、自然土壤背景和多株互相遮挡仍需单独验证。

下一步工作应围绕几个具体方向展开。第一，将验证场景扩展到真实田间和温室生产环境，并增加风致运动、强阴影和多株遮挡条件。第二，在更多生长阶段进行连续采集，评估时间序列表型的一致性和灵敏度。第三，进一步比较 NeRF、NeuS 和高质量 MVS 等非 Gaussian 流程在相同测量协议下的表现。第四，将不同网格变体与表型误差直接关联，特别是验证边界精炼是否能降低叶宽偏差。第五，引入多点尺度标定和多操作者 landmark 重复测量，以分离重建误差、尺度误差和人工测量误差。

### 5.8 小结

ForeSplat 的核心贡献是使训练目标、对象表示和表型测量保持一致。结果表明，前景 RGB 监督是抑制背景泄漏的关键机制，视角质量感知的软损失加权比硬性剔除更适合薄结构多视角重建，掩膜引导的 Gaussian 表征紧凑化为高通量部署提供了实用收益。该流程已经支持跨物种植物结构性状的自动化测量，但叶宽等边界敏感性状仍需要更精细的重建和测量协议。这个边界并不削弱方法贡献，反而明确了下一阶段最值得投入的技术方向。

---

## 6. 局限性

解释这些结果时需要考虑若干局限性。

**消融验证范围：** 完整工作流已在 20 个多视角序列上验证，但前景对象目标函数、视角质量和紧凑化配置的系统消融主要集中在代表性样本或少数结构类型上。该设计有利于隔离机制，但仍不能完全替代逐物种、逐结构类型的大规模消融。

**FSAM3 分割评估范围：** 本文使用人工标注 mask benchmark 比较了 FSAM3 和 SEEM，并报告 F1-score、mIoU 和 HD95。该评估支持 FSAM3 作为本文采集条件下的重建先验，但标注规模仍是代表性子集，不构成大规模通用植物分割数据集。复杂田间背景、极端遮挡和强阴影条件下的 mask 稳定性仍需进一步验证。

**受控室内环境：** 所有采集均在受控或半受控光照的室内环境中完成。田间部署会引入额外挑战，包括直射阳光、风致运动和复杂自然背景，这些尚未在当前研究中测试。

**网格与表型因果关系：** 当前网格和表型结果证明了可行性并刻画了误差模式，但没有建立特定网格变体带来改进的因果证据。“edge-aware meshing improves leaf width measurement accuracy” 这一表述不受当前证据支持，因为缺少特定网格精炼在表型指标上的前后对比。

**外部 baseline 范围：** 本文已经纳入 COLMAP、标准 2DGS、3DGS-FSAM3 和 SuGaR 等外部重建流程，并补充 SuGaR/mesh 代表性可视化比较。然而，NeRF、NeuS 和更多高质量 MVS 变体尚未在相同表型测量协议下系统比较；SuGaR 的跨方法定量结果也基于与标准 2DGS 共有的 11 个序列，而非全部 20 个序列。因此，本文结论主要针对 Gaussian-based 和应用型多视角流程，而不声称覆盖所有神经隐式或传统三维重建范式。

**物种分类学分辨率：** 当前数据集使用中文通用名和英文通用名进行物种或品种识别。面向国际期刊投稿需要经过验证的植物学命名。Supplementary Table S1 应提供从中文通用名、英文通用名到暂定拉丁双名的映射；最终分类学识别需要咨询植物学家或分类数据库。当前材料表按 15 个物种或品种标签记录，并在正文表格中归并为 10 类英文通用名；这一处理适合描述形态多样性，但不应被解读为严格的分类学采样设计。

**尺度恢复：** 绝对尺度使用单个已知物理尺寸（花盆直径，使用数字卡尺测量，精度 ±0.5 mm）恢复。该参照测量误差会线性传递到所有虚拟性状测量中。多点尺度校准（例如在多个深度放置棋盘格靶标）可以降低尺度不确定性，但当前采集协议未实施。

**测量协议：** 虚拟性状测量由一名操作者在提取网格上放置 landmark 完成。人工测量遵循标准园艺实践：株高从盆沿到最高光合组织，冠幅为最大水平范围，叶长/叶宽在每株植物三片完全展开叶片上使用软尺测量（±1 mm）。未评估操作者间变异。因此，报告的虚拟测量误差混合了重建误差与 landmark 放置误差；株高、冠幅和叶长具有较高 R²，但当前数据无法分离各误差来源的相对贡献。

---

## 7. 结论

本文提出 ForeSplat，一个覆盖多视角图像质量控制到 phenotype-ready 植物网格生成的 foreground-aware splatting 流程。其算法贡献从五个层面修改了标准 2DGS 框架：

1. **基于 FFT 的帧质量筛选**（FSAM3 Stage 1）通过排除高频能量不足的帧，自动化 SfM 输入帧选择。20 个多视角序列共包含 5002 帧原始图像，经质量筛选后保留 4217 帧有效图像，并获得 4129 个 SfM 注册视角。

2. **FSAM3 植物前景先验**结合 SAM3 文本提示、PCA 主前景精炼和形态学后处理，在人工标注 benchmark 上达到 F1-score = 98.3%、mIoU = 97.9% 和 HD95 = 41.4 px，优于 SEEM 的 95.1%、94.1% 和 281.9 px。

3. **前景对象优化**通过以下方式重写 2DGS 训练目标：(i) 通过多视角 mask 一致性过滤 COLMAP 稀疏点，实现前景偏置初始化；(ii) 将 RGB loss 计算限制在 mask 定义的前景像素；(iii) 添加 alpha mask loss 和 background opacity loss 作为透明度场辅助约束。系统消融表明，将 RGB supervision 按像素限制到前景是决定性算法修改，alpha 和 opacity regularization 单独无法阻止背景学习。

4. **视角质量感知的软损失加权**用前景 RGB loss 的逐视角质量调制替代标准子集选择范式。Hard filtering 通过移除 27 个视角中的 10 个并破坏角度覆盖，使重建灾难性退化（PSNR_fg: 25.01 -> 12.55 dB）。Soft weighting 保留全部视角，在仅 0.0506 dB PSNR_fg 损失下减少 10.03% 的 Gaussian 数量。

5. **掩膜引导的多线索 Gaussian 剪枝**基于 mask 一致性、透明度、可见性、颜色正常性和拓扑为每个 Gaussian 评分，剪除 mask 边界附近支撑较弱的 Gaussians。与前述模块组合后，完整 ForeSplat 流程在三个结构差异显著的样本上将 Gaussian 总数减少 18.03%，平均 PSNR_fg 下降 0.0657 dB。

应用级实验显示，ForeSplat/F2DMAS 在 20 个多视角序列和两类采集场景上完成从智能手机视频到表型测量的工作流验证。相较于标准 2DGS，完整流程将 PSNR 从 29.58 dB 提高到 31.09 dB，将 LPIPS 从 0.0487 降至 0.0365，并将训练时间和 mesh 提取时间分别降低 60.94% 和 65.17%。相较于 3DGS-FSAM3，完整流程将 mesh 提取时间从 642 s 降至 55 s。跨 21 株植物的虚拟表型测量在株高、冠幅和叶长上达到较高一致性（R² = 0.9878、0.9879 和 0.9738）；叶宽 R² = 0.8999，MAPE = 9.73%，表明边界敏感的薄维度测量是主要剩余挑战。

这些结果将 ForeSplat/F2DMAS 确立为一个模块化、可复现的自动化植物前景重建与 phenotype-ready 网格生成框架。流程的组件级模块化支持对单个阶段进行增量采用和独立改进，也为低成本室内或半受控条件下的三维植物结构表型提供了可部署路径。

---

## 数据可用性

支持本研究的多视角图像、FSAM3 mask、表型测量表格、视角权重文件和主要运行配置将在论文接收或数据整理完成后通过项目仓库或数据存储库公开。最终英文投稿版本将补充数据和代码的永久链接、访问许可、文件清单、版本标识、repository URL、DOI 或 accession number。如部分原始视频因存储体积或隐私化场景背景无法完全公开，将在最终版本中说明可获取范围和合理请求流程。

## 伦理声明

本研究仅涉及植物成像和测量。不涉及人类或动物受试对象。

## 作者贡献

CRediT 作者贡献将在投稿前根据作者实际分工补充，建议至少覆盖 Conceptualization、Methodology、Software、Validation、Formal analysis、Investigation、Data curation、Writing - original draft、Writing - review and editing、Visualization、Supervision、Project administration 和 Funding acquisition。

## 利益冲突

作者声明不存在竞争性利益。

## 资助

资助信息将在投稿前根据项目实际情况补充。若无专项资助，最终英文稿应写明：This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## AI 使用声明

在本稿件准备过程中，作者使用 Claude（Anthropic）作为 AI 辅助写作和研究工具，用于文献检索整理、草稿润色、双语翻译和格式检查。所有 AI 辅助生成或修改的内容均由作者审阅、核验和编辑；作者对论文内容、数据解释、引用准确性和发表作品完整性承担全部责任。

---

## 参考文献

[1] S. Paulus, "Measuring crops in 3D: using geometry for plant phenotyping," *Plant Methods, 15, 103*, 2019.

[2] J. Li, X. Qi, S. H. Nabaei, M. Liu, D. Chen, X. Zhang, X. Yin, Z. Li, "A survey on 3D reconstruction techniques in plant phenotyping: From classical methods to NeRF, 3DGS, and beyond," *Plant Phenomics, 7(4)*, 2025.

[3] S. Akhtar, M. F. Shahid, A. Raza, et al., "Unlocking plant secrets: A systematic review of 3D imaging in plant phenotyping techniques," *Comput. Electron. Agric., 222, 109033*, 2024.

[4] B. Huang, Y. Yu, D. Chen, et al., "2D Gaussian Splatting for geometrically accurate radiance fields," *Proc. SIGGRAPH*, 2024.

[5] Z. Fan, K. Wang, K. Wen, Z. Zhu, D. Xu, Z. Wang, "LightGaussian: Unbounded 3D Gaussian compression with 15x reduction and 200+ FPS," *Proc. NeurIPS*, 2024.

[6] J. C. Lee, D. Rho, X. Sun, J. H. Ko, E. Park, "Compact 3D Gaussian representation for radiance field," *Proc. CVPR*, 2024.

[7] S. Pertuz, D. Puig, M. A. Garcia, "Analysis of focus measure operators for shape-from-focus," *Pattern Recognit., 46(5), 1415–1432*, 2013.

[8] A. Kirillov, E. Mintun, N. Ravi, et al., "Segment Anything," *Proc. ICCV*, 2023.

[9] W. Liu, T. Guan, B. Zhu, et al., "EfficientGS: Streamlining Gaussian Splatting for large-scale high-resolution scene representation," *arXiv:2404.12778*, 2024.

[10] S. Paulus, S. Dupuis, A.-K. Mahlein, H. Kuhlmann, "Surface feature based classification of plant organs from 3D laserscanned point clouds for plant phenotyping," *BMC Bioinformatics, 14, 238*, 2013.

[11] J. L. Schonberger, J.-M. Frahm, "Structure-from-Motion revisited," *Proc. CVPR, 4104–4113*, 2016.

[12] Y. Furukawa, J. Ponce, "Accurate, dense, and robust multiview stereopsis," *IEEE Trans. Pattern Anal. Mach. Intell., 32(8), 1362–1376*, 2010.

[13] T. Choi, S. Lee, J. Park, et al., "NeRF-based 3D reconstruction pipeline for acquisition and analysis of tomato crop morphology," *Front. Plant Sci., 15, 1439086*, 2024.

[14] Z. Yang, L. Chen, J. Sun, et al., "PanicleNeRF: low-cost high-precision 3D reconstruction and phenotyping of rice panicles with smartphone," *Plant Phenomics, 6, 0279*, 2024.

[15] S. Chopra, R. Khosla, P. S. Thenkabail, "AgriNeRF: Neural Radiance Fields for agricultural scenes under challenging lighting," *arXiv:2409.15487*, 2024.

[16] Y. Shen, L. Chen, Y. Wang, et al., "PlantGaussian: 3D Gaussian Splatting for cross-time and cross-scene plant visualization," *The Crop Journal*, 2025.

[17] Y. Zhang, X. Liu, H. Wang, et al., "Wheat3DGS: In-field wheat head reconstruction and phenotyping with 3D Gaussian Splatting," *Proc. CVPR Workshop on Vision for Agriculture*, 2025.

[18] Y. Chen, H. Zhang, W. Li, "High-fidelity 3D reconstruction of peach orchards using a 3DGS-Ag model," *Comput. Electron. Agric.*, 2025.

[19] S. Xiao, J. Zhang, Y. Liu, et al., "ICFMNet: Automated segmentation and 3D phenotypic analysis pipeline for wheat," *Comput. Electron. Agric., 239*, 2025.

[20] R. Reena, J. H. Doonan, Y. H. Liu, "Wheat3D PartNet: Annotated 3D point cloud dataset for wheat organ segmentation," *Comput. Electron. Agric., 238*, 2025.

[21] Z. Gao, X. Su, "Three-dimensional reconstruction of densely planted rice seedlings based on multi-view images," *Plant Phenomics*, 2025.

[22] H. Jiang, X. Sun, S. Li, et al., "Plant stem occlusion inpainting with deep reinforcement learning for tomato 3D phenotyping," *Comput. Electron. Agric., 237*, 2025.

[23] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, R. Ng, "NeRF: Representing scenes as neural radiance fields for view synthesis," *Proc. ECCV, 405–421*, 2020.

[24] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, P. Hedman, "Mip-NeRF 360: Unbounded anti-aliased neural radiance fields," *Proc. CVPR, 5470–5479*, 2022.

[25] B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis, "3D Gaussian Splatting for real-time radiance field rendering," *ACM Trans. Graph., 42(4), 1–14*, 2023.

[26] A. Guedon, V. Lepetit, "SuGaR: Surface-Aligned Gaussian Splatting for efficient 3D mesh reconstruction and high-quality mesh rendering," *Proc. CVPR*, 2024.

[27] B. Guillard, F. Stella, P. Fua, "MeshUDF: Fast and differentiable meshing of unsigned distance field networks," *Proc. ECCV*, 2022.

[28] B. Curless, M. Levoy, "A volumetric method for building complex models from range images," *Proc. SIGGRAPH, 303–312*, 1996.

[29] C. Mostegel, M. Rumpler, F. Fraundorfer, H. Bischof, "UAV-based autonomous image acquisition with multi-view stereo quality assurance," *Proc. CVPR Workshops*, 2016.

[30] S. Haner, A. Heyden, "Covariance propagation and next best view planning for 3D reconstruction," *Proc. ECCV*, 2012.

[31] D. M. Woebbecke, G. E. Meyer, K. Von Bargen, D. A. Mortensen, "Color indices for weed identification under various soil, residue, and lighting conditions," *Trans. ASAE, 38(1), 259–269*, 1995.

[32] N. Otsu, "A threshold selection method from gray-level histograms," *IEEE Trans. Syst. Man Cybern., 9(1), 62–66*, 1979.

[33] N. Ravi, V. Gabeur, Y.-T. Hu, et al., "SAM 2: Segment Anything in images and videos," *arXiv:2408.00714*, 2024.

[34] D. Cai, Y. Liu, Z. Chen, et al., "Performance evaluation of Segment Anything Model for weed detection in cotton fields," *Smart Agric. Technol., 7, 100416*, 2024.

[35] Y. Lu, S. Young, H. Wang, N. Wijewardane, "Robust plant segmentation of proximal aerial images by the Segment Anything Model," *Comput. Electron. Agric., 218, 108715*, 2024.

[36] R. Ferzli, L. J. Karam, "A no-reference objective image sharpness metric based on the notion of just noticeable blur (JNB)," *IEEE Trans. Image Process., 18(4), 717–728*, 2009.

[37] J. Park, Y.-W. Tai, D. Cho, I. S. Kweon, "A unified approach of multi-scale deep and hand-crafted features for defocus estimation," *Proc. CVPR, 1736–1745*, 2017.

[38] J. W. Abe, J. Ilao, G. Foliente, "Promptable leaf segmentation in plant phenotyping: Research perspectives and challenges," *Proc. 30th Int. Conf. M2VIP*, 2024.

[39] W. E. Lorensen, H. E. Cline, "Marching Cubes: A high resolution 3D surface construction algorithm," *Proc. SIGGRAPH, 163–169*, 1987.

[40] M. Minervini, A. Fischbach, H. Scharr, S. A. Tsaftaris, "Finely-grained annotated datasets for image-based plant phenotyping," *Pattern Recognit. Lett., 81, 80–89*, 2016.

[41] M. P. Pound, A. P. French, J. A. Atkinson, D. M. Wells, M. J. Bennett, T. P. Pridmore, "RootNav: Navigating images of complex root architectures," *Plant Physiol., 162(4), 1802–1814*, 2013.

[42] Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," *IEEE Trans. Image Process., 13(4), 600–612*, 2004.

[43] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," *Proc. CVPR, 586–595*, 2018.

[44] C. Y. Kuo, C. L. Chang, Y. C. Tsai, "Multi-view stereo for plant 3D reconstruction: a comparative study," *Biosyst. Eng., 216, 198–213*, 2022.
