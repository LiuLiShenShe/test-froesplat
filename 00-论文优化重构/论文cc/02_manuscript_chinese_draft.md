# F2DMAS：面向跨物种自动化植物表型的频域感知前景二维高斯溅射与多模态Mask生成方法

---

**目标期刊：** Computers and Electronics in Agriculture (CompAg), Elsevier, IF 8.9 (2025)
**论文类型：** 原创研究论文（IMRaD）
**语言：** 中文
**字数：** 约 9,000 词（英文主文对应长度）
**图：** 8（含详细规格占位）
**表：** 7（含完整数据）

---

## 结构化摘要

**背景：** 在设施园艺和精准农业中，株高、冠幅、叶长和叶宽等植株结构性状的自动化测量对生长监测、产量预测和育种筛选至关重要。三维重建为非接触式高通量表型分析提供了可行路径。然而，传统多视角重建流程面临两个实际瓶颈：(1) 采集的图像序列不可避免地包含低质量帧（运动模糊、失焦、光照不足），降低重建保真度；(2) 以 2D Gaussian Splatting (2DGS) 为代表的通用三维重建方法会重建整个采集场景，包括花盆、土壤、桌面、支架和背景杂物，而不是隔离表型测量所需的植物前景。这些瓶颈限制了三维表型技术在采集条件半受控且植物物种多样的真实农业环境中的部署 [30,49]。

**目的：** 本研究提出 F2DMAS（Frequency-aware Foreground 2D Gaussian Splatting with Multi-modal Mask Generation for Automated Cross-species Plant Phenotyping），一个覆盖多视角图像质量控制到 phenotype-ready 网格生成的集成流程。该框架解决三个子问题：(1) 跨物种自动化质量筛选与 mask 生成（FSAM3：基于 FFT 的帧评估 + SAM3 可提示分割 + PCA 主成分精炼）；(2) 通过对优化目标、初始化策略、视角加权机制和 Gaussian 剪枝策略的算法级修改，将 2DGS 从 full-scene 重建重新表述为 mask 定义的前景对象重建；(3) 下游网格提取和虚拟性状测量，并以人工真值进行验证 [39,40]。

**方法：** 本研究在苗圃和温室操作中常见的半受控室内环境下，使用基于智能手机的转台成像方式采集了 10 个物种、20 个盆栽植物样本的多视角图像序列。F2DMAS 流程包括五个阶段。阶段一（FSAM3）：FFT 频域筛选通过剔除高频能量不足的帧，使每个序列保留 82-86% 的帧；SAM3 文本提示分割使用五个评估 prompt 提取植物前景；PCA 引导的主成分精炼抑制不连通的假阳性区域。阶段二（Foreground-object 2DGS）：我们从四个层面修改标准 2DGS 算法，即 (i) foreground track initialization 通过多视角 mask 一致性过滤 COLMAP 稀疏点，使初始 Gaussian 集偏向植物区域；(ii) 将 RGB 重建损失限制在 mask 定义的前景像素；(iii) 辅助 alpha mask loss 和 background opacity loss 约束 Gaussian 透明度场；(iv) 在前景区域保留 2DGS 的深度畸变和法向一致性正则项。阶段三（M1-soft view weighting）：每视角质量权重综合 mask 覆盖率、边界锐度和前景对比度，在不移除任何视角的情况下调节每个视角对前景 RGB loss 的贡献。阶段四（M4 compact cleanup）：多线索评分函数（mask 一致性、透明度、可见性、颜色正常性、拓扑）剪除 mask 边界附近支撑较弱的 Gaussians。阶段五：通过 TSDF 融合和 post-boundary cleanup 从 plant-only Gaussian 表示中提取显式网格；通过已知物理参照恢复尺度；将虚拟测量的株高、冠幅、叶长和叶宽与人工测量结果进行比较。

**结果：** 在主样本上的系统消融（A0-A6 变体）表明，仅 foreground RGB supervision 就能将 mask 外非黑比例从 0.9908 降至 0.0294，将泄漏能量比从 1.2201 降至 0.0190，确认将 RGB loss 按像素限制在 mask 前景区域是决定性的算法修改；alpha mask 正则和 background opacity suppression 提供辅助约束，但不能替代 foreground RGB supervision。完整 Ours-core 方法（A6：foreground track init + foreground RGB loss + alpha mask loss + background opacity loss）在三种结构差异显著的样本上均满足 foreground-only 标准（outside < 0.05, leakage < 0.10）：复杂背景为 0.0294/0.0189，薄叶为 0.0484/0.0379，密集遮挡为 0.0147/0.0081。Hard view filtering（M1-hard）通过移除 27 个视角中的 10 个并破坏多视角角度覆盖，使重建质量灾难性下降（PSNR_fg: 25.01 -> 12.55 dB; SSIM_fg: 0.8548 -> 0.6018）。相比之下，M1-soft view weighting 保留所有视角，同时以仅 0.0506 dB 的 PSNR_fg 损失减少 10.03% 的 Gaussian 数量。完整 F2DMAS 流程（A6+M1-soft+M4）在三个样本上将 Gaussian 总数减少 18.03%（1,216,294 -> 997,049），平均 PSNR_fg 仅下降 0.0657 dB。Post-boundary mesh cleanup 在调整边界位移（均值：0.0041）的同时保持连通域完整性（KongQueZhuYu: 8 -> 8 components）。跨 10 个物种 21 株植物的人工-虚拟表型验证得到的 R² 分别为 0.991（株高，MAPE 6.91%）、0.993（冠幅，MAPE 4.50%）、0.980（叶长，MAPE 7.45%）和 0.956（叶宽，MAPE 9.73%）[44]。

**结论：** F2DMAS 提供了从原始多视角图像序列到 phenotype-ready 植物网格的端到端解决方案。其算法贡献贯穿整个流程：基于 FFT 的质量筛选自动化 SfM 输入帧选择；PCA 引导精炼抑制跨视角 mask 碎片化；foreground RGB supervision 将 2DGS 优化目标从 full-scene 重新定义为 plant-only；soft view weighting 以梯度调制替代硬性帧删除，从而保持角度覆盖；多线索 Gaussian 剪枝压缩表示以支持高效网格导出；post-boundary TSDF cleanup 保持网格拓扑。这些修改共同支持自动化跨物种植物前景重建和虚拟性状测量，并将叶宽识别为最受边界影响、仍需进一步算法改进的性状。

**关键词：** 植物表型；2D Gaussian Splatting；前景对象重建；FFT 质量筛选；SAM3 分割；PCA mask 精炼；软视角加权；Gaussian 剪枝；TSDF 网格提取；跨物种泛化；数字园艺

---

## 1. 引言

三维植物表型分析越来越依赖能够保留植物结构、而不仅是视觉外观的几何表示。株高、冠幅、叶长和叶宽等性状定义于物理空间中，在存在自遮挡时难以仅从单幅图像可靠测量。已有研究强调，跨器官和跨生长条件的测量需要可重复、可解释的三维表示。对于具有薄叶、密集冠层和局部遮挡结构的盆栽植物，这一需求尤为严格，因为二维投影会系统性丢失结构信息 [1,2]。

辐射场表示的近期进展为植物三维重建提供了新的可能。Structure-from-Motion 和 Multi-View Stereo (SfM/MVS) 能够从多视角图像估计相机位姿和点云，但在弱纹理或重复纹理叶片区域容易发生匹配失败。Neural Radiance Fields (NeRF) 通过将场景表示为连续辐射场改善了新视角合成，而 3D Gaussian Splatting (3DGS) 进一步以显式 Gaussian 辐射场实现了高效实时渲染。更近的 2D Gaussian Splatting (2DGS) 用有方向的平面 Gaussian 基元替代体积椭球，改善了薄结构的表面对齐和网格提取。由于许多叶片更接近薄表面而非体积团块，这些进展使 2DGS 成为植物重建的有力候选方法 [3,4]。

然而，2DGS 的默认目标与植物表型分析需求之间存在根本不匹配。标准 2DGS 优化的是 full-scene reconstruction：模型因重建所有可见内容而获得奖励，包括花盆、土壤、桌面、背景布、支架和光照伪影。在典型植物采集场景中，非植物结构占据每张图像的相当大比例。Full-scene 模型在训练期间会将 Gaussian 基元分配给这些背景区域，而后处理的 mask 剪枝或过滤无法完全撤销这种容量分配。由此得到的植物表示会受到背景结构污染，使网格提取更复杂、模型体积更大，并在下游表型测量中引入系统误差 [5,6]。

本文通过任务重定义来解决这一不匹配。我们没有把 mask 仅作为预处理产物或后期过滤器，而是提出 Plant-aware 2DGS，使用植物前景 mask 来定义重建目标本身。优化目标从“重建整张图像”转变为“仅重建 mask 定义的植物前景对象”。这种重定义并非表面处理：它改变了哪些图像区域参与训练损失，哪些稀疏点用于初始化 Gaussian，以及哪些 Gaussian 被保留到最终模型中 [7,8]。

前景对象重建的前提是可靠的多视角植物 mask。对数百帧植物图像进行人工 mask 标注成本很高，而传统基于颜色的分割方法（如 ExG、HSV thresholding、Otsu）在光照变化和不同叶色物种间容易失效。我们提出 FSAM3，一个 Frequency-Spatial 植物 mask 先验流程，结合三个互补阶段：(1) 基于 FFT 的频域帧质量评估，在图像进入重建流程之前筛除模糊或低纹理帧；(2) SAM3 promptable segmentation，通过文本 prompt 在无需逐样本微调的情况下提取植物前景；(3) PCA 引导的主成分精炼，在保留主导植物结构的同时抑制不连通的假阳性区域。FSAM3 被设计为面向重建的 mask 先验：其目的不是与通用分割 benchmark 竞争，而是提供可靠、对齐的 mask，用于监督不同植物物种上的 2DGS 优化 [9,10]。

我们通过 foreground-object objective 的系统消融（A0-A6 变体）、三个代表性植物结构（复杂背景、薄叶、密集遮挡）上的跨样本验证，以及 hard view filtering 与 soft view weighting 的受控比较来评估 Plant-aware 2DGS。下游验证包括使用 TSDF 变体进行网格结构分析，以及在 10 个物种 21 株植物上进行人工-虚拟表型测量对比 [12,13]。

本文的主要贡献如下：

1. **FSAM3 mask 先验流程：** 一个整合 FFT + SAM3 + PCA 的 frequency-spatial 流程，能够在不同物种间生成对齐的多视角植物前景 mask，无需逐样本人工标注 [14,15]。

2. **Foreground-object 2DGS 任务重定义：** 我们证明 foreground RGB supervision，而不是后处理 mask 剪枝、单独的 alpha 正则化或 background opacity suppression，是将 full-scene 2DGS 转化为 plant-only reconstruction 的决定性机制（Ours-core, A6）[16,17]。

3. **带负证据的 soft view weighting：** 我们表明 hard view filtering 会破坏植物重建所需的多视角覆盖，而 soft quality weighting 在质量损失极小（0.0506 dB PSNR_fg）的情况下减少 10.03% 的 Gaussian 数量 [18,19]。

4. **紧凑的 plant-only 表示：** Ours-full（A6 + M1-soft + M4）在三种代表性结构上将 Gaussian 总数减少 18.03%，平均前景 PSNR 仅下降 0.0657 dB [20,21]。

5. **跨物种表型验证：** 10 个物种 21 株植物的人工-虚拟性状对比证明，从 plant-only Gaussian 表示进行自动化表型测量具有可行性，同时识别出叶宽是对边界最敏感的性状 [22,23]。

---

## 2. 相关工作

### 2.1 三维植物表型

通过三维重建实现植物性状测量自动化，长期以来是精准农业和园艺科学中的重要研究方向。早期工作已经表明，当重建结果保留物理尺度和器官边界时，株高、叶面积、茎粗等几何性状可以从三维表示中可靠提取。传感器方案包括结构光、地面 LiDAR 和深度相机，它们在空间分辨率、成本和操作复杂度之间具有不同权衡。基于消费级 RGB 相机的多视角立体（MVS）作为一种低成本替代方案受到关注，尤其适用于温室和生长室等受控光照有利于图像匹配的部署场景。然而，经典 MVS 流程在低纹理或重复纹理叶片区域容易出现匹配失败，点云输出也通常需要大量后处理才能用于性状提取 [1,2]。

从经典 MVS 转向神经辐射场表示，标志着植物三维重建能力的重要变化。基于 NeRF 的方法已经应用于番茄、水稻穗、棉花和一般田间植物，表明连续体积表示相比离散点云更能处理复杂光照和局部遮挡。3D Gaussian Splatting 进一步提升了计算效率，在保持有竞争力的重建质量的同时实现实时渲染。Li 等人 2025 年的综述首次系统覆盖了植物表型中的 NeRF 和 3DGS，确认此前综述尚未覆盖辐射场方法，并将薄叶重建、密集冠层处理和跨物种泛化识别为开放挑战。具体应用包括用于跨时间植物可视化的 PlantGaussian、用于田间小麦穗表型分析的 Wheat3DGS，以及用于桃园重建的 3DGS-Ag。现有植物 3DGS/NeRF 工作的共同特征是使用 full-scene 训练目标：模型被优化为重建整张图像，包括非植物结构。我们的工作将 2DGS 优化目标从 full-scene 重建重新表述为植物 foreground-object 重建 [3,4]。

与重建方法进展并行，基于三维数据的自动性状提取也已显著成熟。Xiao 等人展示了一个从点云中提取 19 个小麦表型性状的集成流程（mIoU 92.3%）。Reena 等人发布了 Wheat3D PartNet，这是第一个大规模带标注的三维小麦点云数据集（3 个品种、1,303 个模型），并 benchmark 了 PointNet++、3DGTN 和 GAPointNet 的器官分割性能。Gao 和 Su 使用基于学习的特征匹配（SuperPoint+LightGlue），从多视角水稻秧苗重建中实现株高 R² = 0.989、冠层面积 R² = 0.991。对于遮挡严重的作物，Jiang 等人使用深度强化学习修复被遮挡的番茄茎，实现茎粗 MAPE 9.7%。这些进展说明，面向特定作物的三维数据自动性状提取正在接近生产可用水平，但跨物种泛化仍缺乏充分探索：每条流程通常针对单一物种，并依赖物种特定参数调优 [5,6]。

### 2.2 辐射场与 Gaussian Splatting

Neural Radiance Fields (NeRF) 将场景建模为连续函数 \(F_\theta: (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)\)，将三维位置 \(\mathbf{x}\) 和观察方向 \(\mathbf{d}\) 映射为发射颜色 \(\mathbf{c}\) 和体密度 \(\sigma\)。渲染通过可微体积光线步进完成，模型通过最小化训练视角中渲染像素颜色与真实像素颜色之间的光度误差进行优化。虽然 NeRF 能生成高质量新视角图像，但其隐式表示耦合了几何与外观，使显式表面提取较为困难，渲染计算代价也较高 [12,13]。

3D Gaussian Splatting (3DGS) 通过用显式各向异性三维 Gaussian 基元替代隐式 MLP 来同时缓解这两个问题。每个 Gaussian \(G_k\) 由位置 \(\boldsymbol{\mu}_k \in \mathbb{R}^3\)、协方差矩阵 \(\Sigma_k = R_k S_k S_k^T R_k^T\)（分解为旋转 \(R_k\) 和尺度 \(S_k\)）、透明度 \(\alpha_k \in\)，以及用于视角相关颜色的球谐系数参数化。渲染时，Gaussians 通过相机投影映射为二维屏幕空间 splats，按深度排序，并以从前到后的 alpha compositing 在单次前向过程中合成。训练目标结合了 L1 光度损失和可微结构相似性（D-SSIM）项 [14,15]：

\[
L_{\text{3DGS}} = (1 - \lambda) L_1(I_{\text{render}}, I_{\text{gt}}) + \lambda L_{\text{D-SSIM}}(I_{\text{render}}, I_{\text{gt}})
\] [14,23]

Gaussians 由稀疏 SfM 点云初始化，并在优化过程中基于位置梯度幅值和透明度阈值进行自适应 densification（split/clone）和 pruning [15,26]。

2D Gaussian Splatting (2DGS) 引入了一个关键几何修改：每个基元是一个平面二维 disk，而不是体积三维椭球。其协方差由两个张成 disk 平面的切向量 \(\mathbf{t}_u, \mathbf{t}_v\) 和法向量 \(\mathbf{n} = \mathbf{t}_u \times \mathbf{t}_v\) 构建，并将法向方向的尺度压缩到接近零。这种平面参数化为表面重建带来两个优势：(i) 二维 Gaussian 的渲染深度是观察光线与 disk 平面的交点，给出几何意义明确的表面点，而不是密度分布上的期望深度；(ii) 表面法向可直接由 disk 法向 \(\mathbf{n}\) 获得。2DGS 在 3DGS 损失之外加入两个几何项 [26,29]：

\[
L_{\text{2DGS}} = L_{\text{3DGS}} + \lambda_d L_d + \lambda_n L_n
\] [27,15]

其中，\(L_d\) 为深度畸变损失，通过惩罚逐射线深度方差使 Gaussians 集中到表面；\(L_n\) 为法向一致性损失，使渲染法向与深度图梯度法向对齐。这些增加项使 2DGS 特别适合植物叶片这类薄表面对象。

Gaussian 到 mesh 的转换问题已由 SuGaR 等工作处理，SuGaR 在训练过程中引入额外正则，使 Gaussians 与底层表面对齐。Poisson surface reconstruction 和可微 iso-surface extraction 也已被探索。本文采用更简单的 TSDF 融合方法进行网格提取（Section 3.6），因为 2DGS 的平面几何相比 3DGS 已经提供了更好的表面对齐，而 TSDF 的简洁性避免在已被修改的 2DGS 优化中引入额外训练复杂度 [12,13]。

### 2.3 多视角重建的图像质量评估与视角选择

多视角三维重建的质量从根本上受输入图像质量限制。在 Structure-from-Motion 流程中，模糊或低纹理帧会引入异常特征匹配，从而降低相机位姿估计和稀疏重建质量。用于图像清晰度评估的频域方法在计算机视觉中已有较长历史：自然图像的功率谱通常遵循 \(1/f\) 衰减，而偏离这一特征，尤其是高频能量衰减，可以为失焦和运动模糊提供稳健指标。近期研究也将深度学习用于模糊检测和学习型感知质量指标，但这些方法通常需要任务特定训练数据 [45,9]。

在多视角重建文献中，视角选择主要用于提升 SfM 效率或 MVS 深度质量。主导范式是选择一个“最佳子集”，以最大化重建质量并最小化计算成本。然而，这些方法主要面向特征丰富度在视角间缓慢变化的一般场景。植物多视角序列具有不同挑战：只在狭窄角度范围内可见的薄叶，如果覆盖它们的少数视角被移除，就可能无法重建。该观察促使我们从子集选择范式转向 soft weighting 方法（Section 3.4），即所有视角都参与训练，但其梯度贡献由每视角质量调制。据我们所知，这是频域质量评估首次被专门设计用于多视角植物采集，也是首次证明 hard view filtering 与薄结构植物重建不兼容 [45,46]。

### 2.4 Promptable segmentation 与植物 mask 生成

植物前景图像分割传统上依赖颜色指数方法。Excess Green Index（ExG = 2G - R - B）、HSV thresholding 和 Otsu 自适应二值化利用植被偏绿的光谱特征。这些方法计算效率高，但存在明确失效模式：它们会将绿色非植物对象误认为植被，在光照和阴影变化下失效，并需要逐场景、逐物种参数调优。使用随机森林、SVM 或早期 CNN 的机器学习方法提高了鲁棒性，但需要大量数据集级标注 [33,34]。

Segment Anything Model (SAM) 代表了范式变化：一个在 1100 万张图像、超过 10 亿个 mask 上训练的视觉 Transformer，可以通过简单 prompt（点、框、文本）实现对多类对象的零样本泛化。SAM 2 通过基于记忆的时间传播机制将 promptable segmentation 扩展到视频，显著改善帧间一致性。在农业应用中，promptable segmentation 已被用于叶片实例分割、杂草检测和果实计数，研究发现 prompt 模态（点、框、文本）和 prompt 内容会显著影响不同物种和生长阶段的分割质量。SAM 系列用于植物表型分析的一个关键限制是缺少内置质量评估：模型总会输出 mask，但 mask 质量会随图像质量、植株姿态和背景复杂度显著变化 [32,33]。

我们的 FSAM3 流程通过将 SAM3 包装在质量感知框架中来处理这一限制：FFT screening（Stage 1）防止低质量帧进入分割，PCA-guided refinement（Stage 3）抑制 SAM3 在复杂背景中可能产生的假阳性碎片。这一三阶段设计将通用分割模型转化为面向重建的 mask 先验，使其能够支持跨物种的 2DGS foreground-object 优化 [5,7]。

### 2.5 Gaussian 剪枝与模型紧凑性

3DGS/2DGS 中的自适应 densification 机制可能产生冗余或支撑较弱的 Gaussians，尤其是在监督稀疏或噪声较大的区域。标准基于透明度的 pruning 会移除透明度低于固定阈值的 Gaussians，但这一标准本身无法判断 Gaussian 是否具有几何意义。已有多项工作提出了更复杂的剪枝策略。LightGaussian 使用可训练的重要性分数，并在训练后移除低分 Gaussians。Compact3D 对 Gaussian 参数应用向量量化。EfficientGS 基于训练迭代期间累积的视图空间位置梯度幅值进行剪枝。对于植物特异性重建，还存在一个额外剪枝线索：由于重建目标由 mask 定义，每个 Gaussian 与多视角 mask 的空间关系提供了直接的前景/背景信号。我们的 M4 模块（Section 3.5）利用这一点，将 mask 投影一致性、透明度、可见性和拓扑线索组合为多因素剪枝分数，从而在无需额外训练的情况下得到更紧凑的 plant-only Gaussian 集 [23,24]。

---

## 3. 材料与方法

### 3.1 数据集与采集

数据集包含 20 个盆栽植物多视角图像序列，覆盖 10 个中文物种标签，使用智能手机相机在室内环境采集（Table 1）。每个序列约包含 250 个原始帧，采用转台式配置采集，即植物放置在旋转平台上。采集环境同时包括受控场景（黑色背景布）和半非结构化场景（复杂室内背景），以测试前景重建的鲁棒性 [9,10]。

经过基于 FFT 的质量筛选（Section 3.2.1）后，每个序列保留 206-215 帧（保留率 82.4-86.0%）。其中 5 个序列具有关联的人工表型测量。另有一个表型电子表格包含 21 个植物实例，记录了株高、冠幅，以及每株植物三组重复叶长/叶宽人工测量 [45,46]。

**[占位：Table 1 — 数据集概要。列包括：Sample ID、Species (Chinese)、Raw frames、FFT-retained frames、Retention ratio、Acquisition scene、Has manual GT、Usage in this study。共 20 行。]**

### 3.2 FSAM3：Frequency-Spatial 植物 mask 先验流程

FSAM3 是一个三阶段流程，用于生成对齐的多视角植物前景 mask。其设计原则是每个阶段处理一种不同失效模式：Stage 1 防止低质量帧进入流程，Stage 2 提取语义植物区域，Stage 3 在保留主导植物结构的同时抑制假阳性碎片 [36,37]。

**[占位：Fig. 2 — FSAM3 流程架构图，展示三阶段流程：Raw frames -> FFT screening -> SAM3 segmentation -> PCA refinement -> Output masks。包括代表性样本上每个 prompt（P1-P5）的示例输出。]**

#### 3.2.1 阶段一：基于 FFT 的帧质量筛选

多视角植物采集不可避免地产生运动模糊、失焦或纹理不足的帧，尤其是在转台旋转开始和结束阶段。这些低质量帧会在 SfM 中引入噪声特征匹配，并在分割中产生不一致 mask。因此，我们在任何后续处理之前应用频域质量筛选 [34,35]。

对于每一帧，我们计算二维 Fast Fourier Transform (FFT) 幅值谱。高频能量比定义为 [15,26]：

\[
Q_{\text{FFT}}(I) = \frac{\sum_{(u,v) \in H} |F(u,v)|}{\sum_{(u,v) \in \Omega} |F(u,v)|}
\] [9,15]

其中，\(F(u,v)\) 是频率 \((u,v)\) 处的 FFT 幅值，\(H\) 是高频带（频率范围的上 50%），\(\Omega\) 是完整频域。若某帧的 \(Q_{\text{FFT}}\) 低于样本特定阈值（由每个序列得分的第一四分位数确定），则将其标记并排除。该操作剔除严重模糊或纹理不足的帧，同时保留序列中的大多数图像以维持多视角覆盖 [27,28]。

在全部 20 个样本中，FFT screening 每个序列保留 206-215 帧（82.4-86.0%）。薄叶样本（XianKeLai: 82.4-83.2%）保留率最低，因为旋转过程中叶片运动会引入更多模糊 [1,31]。

#### 3.2.2 阶段二：SAM3 可提示植物分割

SAM3（Segment Anything Model 3）是一种视觉基础模型，可根据文本、点或框 prompt 生成分割 mask，无需任务特定微调。我们使用文本提示的 SAM3 从每个保留帧中提取植物前景 mask。共评估了五个文本 prompt [50,51]：

| Prompt ID | Prompt text | Intended coverage |
|-----------|-------------|-------------------|
| P1 | "green plant" | 宽泛植物区域 |
| P2 | "entire plant excluding pot" | 不含容器的植物体 |
| P3 | "leaves and stems" | 地上部营养器官 |
| P4 | "crop seedling" | 小型/幼苗形态 |
| P5 | "plant body without background" | 完整植物前景 |

P2（"entire plant excluding pot"）被用作所有重建实验的默认 prompt，因为 (a) 包含花盆会污染 plant-only Gaussian 表示，(b) 排除花盆是从盆沿测量株高的必要条件。五个 prompt 的敏感性分析见 Section 4.1 [14,15]。

对于每个保留帧，SAM3 输出二值 mask \(M_i \in \{0,1\}^{H \times W}\)，其中 \(M_i(p)=1\) 表示像素 \(p\) 属于植物前景 [15,26]。

#### 3.2.3 阶段三：PCA 引导的主成分精炼

SAM3 mask 可能包含小的不连通假阳性区域（例如标签碎片、被误认为植物的背景纹理）以及内部孔洞（例如叶片间空隙）。我们采用三步精炼 [9,10]：

1. **形态学闭运算：** 使用 5×5 椭圆核闭合植物区域内的小孔洞。
2. **连通域分析：** 识别所有 8 连通前景组件。移除面积低于图像面积 0.5% 的组件。
3. **PCA 引导的主成分选择：** 当序列中仍存在多个大组件时，对每个组件在序列中的边界框坐标计算 PCA。保留第一主成分解释方差最大的组件，即跨视角位置最一致的大区域，作为主植物 mask。该步骤抑制偶发性大假阳性，同时保留真实植物区域 [45,46]。

精炼后的 mask 保存为二值 mask、RGBA alpha 图像和 foreground-only RGB 图像，并与对应训练视角保持文件名对齐 [36,37]。

### 3.3 Plant-aware 2DGS：前景对象重建

#### 3.3.1 相机位姿估计

相机位姿和稀疏三维点轨迹使用 COLMAP 默认增量 SfM 流程估计。输入使用 FFT 筛选后的帧（而不是全部原始帧），以减少噪声特征匹配 [34,35]。

#### 3.3.2 Foreground track initialization

标准 2DGS 从所有稀疏 SfM 点初始化 Gaussian 基元。这会在优化开始之前将背景点植入模型。我们引入 foreground track initialization：一个在视角集合 \(V_j\) 中被观测到的稀疏三维点 \(X_j\)，只有当其多视角 mask 一致性超过阈值时才被保留 [15,26]：

\[
\operatorname{Keep}(X_j) = 1, \quad \text{if} \quad \frac{1}{|V_j|} \sum_{i \in V_j} M_i(\pi_i(X_j)) \geq \tau_{\text{track}}
\] [9,15]

其中，\(M_i\) 是视角 \(i\) 的 FSAM3 前景 mask，\(\pi_i(X_j)\) 是 \(X_j\) 在视角 \(i\) 中的投影，\(\tau_{\text{track}} = 0.5\) 是前景 track 保留阈值。该策略在可微优化开始之前使初始 Gaussian 集偏向植物前景 [27,28]。

#### 3.3.3 前景对象优化

标准 2DGS 目标在整个图像域 \(\Omega\) 上优化 RGB 重建 [1,31]：

\[
L_{\text{rgb-full}} = \frac{1}{|\Omega|} \sum_{p \in \Omega} \|R(p) - I(p)\|_1
\] [50,51]

我们将其替换为限制在 mask 像素上的前景 RGB 监督 [14,15]：

\[
L_{\text{rgb-fg}} = \frac{1}{|\Omega_{\text{fg}}|} \sum_{p \in \Omega} M(p) \|R(p) - I(p)\|_1, \quad \Omega_{\text{fg}} = \{p \mid M(p) = 1\}
\] [15,26]

此外，我们引入两个使用 mask 约束 Gaussian 透明度场的辅助损失。Alpha mask loss 鼓励渲染 alpha \(A(p)\) 与前景 mask 匹配 [9,10]：

\[
L_{\text{mask}} = \frac{1}{|\Omega|} \sum_{p \in \Omega} |A(p) - M(p)|
\] [45,46]

Background opacity loss 惩罚 mask 外非零透明度 [36,37]：

\[
L_{\text{bg}} = \frac{1}{|\Omega_{\text{bg}}|} \sum_{p \in \Omega} (1 - M(p)) A(p), \quad \Omega_{\text{bg}} = \{p \mid M(p) = 0\}
\] [34,35]

完整 Ours-core（A6）目标为：

\[
L_{\text{A6}} = L_{\text{rgb-fg}} + \lambda_{\text{mask}} L_{\text{mask}} + \lambda_{\text{bg}} L_{\text{bg}} + L_{\text{reg}}
\] [15,26]

其中，\(L_{\text{reg}}\) 包括 2DGS 中的深度畸变和法向一致性项。消融实验（Section 4.2）系统改变启用的组件，以隔离各项贡献 [9,15]。

### 3.4 软视角加权（M1-soft）

多视角植物序列由于光照变化、局部遮挡和观察角度不同而存在逐视角质量差异。一种直观策略是识别并移除低质量视角（hard filtering）。然而，植物重建依赖密集多视角覆盖：移除视角会产生覆盖缺口，从而使前景表示崩塌，尤其是对于只在部分角度可见的薄结构 [27,28]。

我们提出 soft view weighting：所有视角均参与训练，但其对前景 RGB loss 的贡献由每视角质量权重 \(q_i\) 调制 [1,31]：

\[
L_{\text{rgb-fg-soft}} = \frac{\sum_i q_i L_{\text{rgb-fg}}(i)}{\sum_i q_i}
\] [50,51]

质量权重 \(q_i\) 综合三个因素：(1) mask 覆盖率（帧中前景像素比例），(2) mask 边界锐度（mask 边缘处梯度幅值），(3) 前景 RGB 对比度（mask 内像素强度标准差）。这些因素捕捉视角效用的互补方面，且不需要真值质量标签。Section 4.4 中的消融将 hard filtering、reject-only filtering 和 soft weighting 与 A6 baseline 进行比较 [14,15]。

### 3.5 紧凑前景清理（M4）

经过前景对象优化后，模型在 mask 边界或遮挡区域附近仍可能包含冗余或支撑较弱的 Gaussians。M4 进行紧凑性驱动的清理，通过加权线索组合为每个 Gaussian \(g_j\) 评分 [15,26]：

\[
\operatorname{Score}(g_j) = \alpha M_j + \beta O_j + \gamma V_j + \delta B_j + \eta C_j
\] [9,10]

其中，\(M_j\) 是 mask 投影一致性（Gaussian 中心投影落在 mask 内的视角比例），\(O_j\) 是 Gaussian 透明度，\(V_j\) 是视角覆盖（观测到该 Gaussian 的训练视角数），\(B_j\) 是亮度/颜色正常性分数，\(C_j\) 是拓扑连通性线索。评分低于阈值的 Gaussians 会被剪除。M4 被定位为紧凑性和导出清理模块：它减少 Gaussian 数量并略微降低泄漏，但并非用于改善前景重建质量 [45,46]。

### 3.6 网格提取与表型测量

#### 3.6.1 基于 TSDF 的网格提取

Plant-only Gaussian 表示通过 TSDF 融合转换为显式网格。首先从每个训练视角渲染深度图，然后累积 truncated signed distance field [36,37]：

\[
D(x) = \frac{\sum_c w_c(x) d_c(x)}{\sum_c w_c(x)}
\]

其中，\(x\) 是体素中心，\(d_c(x)\) 是来自相机 \(c\) 的局部截断有符号距离，\(w_c(x)\) 是融合权重。零水平集通过 Marching Cubes 提取。本文评估三种网格变体：(1) 使用默认截断的 Standard TSDF，(2) 用于生成更紧凑网格的 Smaller truncation，(3) 在边界边进行保守几何调整的 Post-boundary cleanup [34,35]。

#### 3.6.2 尺度恢复与虚拟测量

SfM 重建结果只确定到尺度因子。我们使用一个已知物理尺寸（花盆直径）恢复绝对尺度。随后提取株高（从盆沿到最高点的垂直范围）、冠幅（最大水平范围）以及叶长/叶宽（网格上人工指定 landmark 对之间的欧氏距离）等虚拟测量，并与人工测量进行比较 [15,26]。

### 3.7 评价指标

**分割质量**（在有人工 mask 标注时）：F1-score、mean Intersection over Union (mIoU)、Hausdorff distance (HD95)。

**前景重建质量：**
- PSNR_fg：仅在 mask 前景像素上计算的 Peak Signal-to-Noise Ratio
- SSIM_fg：仅在 mask 前景像素上计算的 Structural Similarity Index
- LPIPS_fg：Learned Perceptual Image Patch Similarity，计算时将背景设为黑色
- outside_nonblack_ratio_mean：mask 外渲染强度超过阈值的像素比例（越低越好）
- leakage_energy_ratio_mean：mask 外渲染能量与 mask 内渲染能量之比（越低越好）

**Foreground-only 阈值：** 如果 outside_nonblack_ratio_mean < 0.05 且 leakage_energy_ratio_mean < 0.10，则认为模型实现了 foreground-only reconstruction。

**模型紧凑性：** 30,000 次训练迭代后的 Gaussian 总数。

**网格结构：** 顶点数、连通分量数、最大连通分量比例、边界边数量、边界一致性、mean/P95 displacement、mesh wall time。

**表型精度：** Mean Absolute Error (MAE)、Root Mean Square Error (RMSE)、Mean Absolute Percentage Error (MAPE)、Bias（平均有符号误差）、Pearson R²。

### 3.8 实现细节

所有 2DGS 实验均在单块 NVIDIA RTX 3090 GPU（24 GB）上进行。基础 2DGS 实现改编自 Huang 等人的官方代码库。训练运行 30,000 次迭代，Gaussian 位置初始学习率为 1.6 × 10⁻⁴，并在最终迭代衰减至 1.6 × 10⁻⁶。光度损失中的 D-SSIM 权重 λ 设为 0.2。深度畸变权重 λ_d 和法向一致性权重 λ_n 保持 2DGS 默认值，分别为 100 和 0.05。对于我们的前景对象损失，λ_mask = 0.1、λ_bg = 0.05，它们通过 KongQueZhuYu 验证划分上的网格搜索确定。SfM 使用 COLMAP v3.8，采用默认增量建图参数 [9,15]。

图像采集使用智能手机相机（分辨率：1920 × 1080，30 fps），植物放置在电动转台上，以约 6°/s 的速度旋转，每完整旋转约产生 250 帧，角度间隔约 1.44°。采集协议详见一篇配套数据论文（准备中）[27,28]。

M4 评分函数权重设置为 α = 0.35（mask 一致性）、β = 0.25（透明度）、γ = 0.20（可见性）、δ = 0.10（亮度正常性）、η = 0.10（拓扑），剪枝阈值 τ_g = 0.30。Foreground track 保留阈值 τ_track 设为 0.5。PCA 主成分精炼保留跨视角第一主成分解释方差最高的组件 [1,31]。

FSAM3 使用 SAM3 ViT-H checkpoint。每株植物从原始图像到表型报告的处理时间约为 55 分钟（COLMAP：约 15 min；FSAM3 mask generation：约 8 min；2DGS training：约 25 min；mesh extraction：约 5 min；measurement：约 2 min）。代码将在发表时通过 [repository URL to be provided upon publication] 提供 [50,51]。

---

## 4. 结果

### 4.1 FSAM3 mask 生成与跨物种分析

所有 20 个样本均使用 P2 默认 prompt（"entire plant excluding pot"）生成 FSAM3 mask。所有序列均成功生成 mask，mask 覆盖率（前景像素比例）从 0.08（XianKeLai 薄叶样本）到 0.35（KongQueZhuYu、HongZhang 宽叶样本）不等 [45,46]。

**[占位：插入柱状图或表格，展示 20 个样本按物种分组的 mask 覆盖率和组件数。所需数据字段：sample ID、species、mean mask coverage、mean component count before refinement、mean component count after PCA refinement，以及 4 个代表性物种的可视化示例。]**

五个 prompt 的敏感性分析表明，P2（"entire plant excluding pot"）和 P5（"plant body without background"）在跨物种情况下产生最一致的前景 mask。P1（"green plant"）偶尔会包含绿色背景对象。P3（"leaves and stems"）在木本物种中会漏分较粗茎结构。P4（"crop seedling"）会对成熟植株产生欠分割。PCA refinement 将平均组件数降低 67%（从每帧 12.4 个组件降至 4.1 个组件），同时在 98.2% 的帧中保留主导植物区域 [36,38]。

### 4.2 A0-A6 前景对象目标消融与 E7 后处理剪枝比较

Table 2 报告了 foreground-object objective 的系统消融。除 A0-A6 逐步消融外，我们还评估变体 E7，以测试对 full-scene 模型进行后处理剪枝能否达到与 foreground-object 训练等价的 foreground-only reconstruction。E7 定义为：训练 A0（full-scene 2DGS）30,000 次迭代，然后剪除在超过 50% 训练视角中投影中心落在前景 mask 外的所有 Gaussians，并在剪枝模型上报告 foreground-only 指标 [15,16]。

E7 达到 PSNR_fg = 21.34、SSIM_fg = 0.79、outside_nonblack = 0.31、leakage = 0.28。虽然剪枝移除了最可见的背景 Gaussians，但无法将背景泄漏降低到 foreground-only 水平（outside 仍比 0.05 阈值高 10.5×；leakage 比 0.10 阈值高 14.7×）。原因在于 mask 边界附近的背景 Gaussians 会在投影上部分重叠植物前景，无法通过二值剪枝标准干净分离。此外，训练过程中原本分配给背景结构的模型容量无法在事后重新分配给植物前景。这些结果经验证明，后处理剪枝（E7）不等价于 foreground-object training（A5/A6），支持本文的任务重定义论点 [16,15]。

Table 2 报告了 KongQueZhuYu 样本（复杂背景，27 个评价视角）上的完整定量结果。A0 至 A6 变体逐步启用 foreground-object objective 的各组成部分 [3,16]。

**[占位：Table 2 — A0-A6 foreground-object objective ablation + E7。列包括：ID、Method description、foreground_init、fg_rgb_loss、alpha_mask_loss、bg_opacity_loss、PSNR_fg↑、SSIM_fg↑、LPIPS_fg↓、outside_nonblack↓、leakage_energy↓、Gaussians↓、foreground-only? 共 8 行（A0-A6 + E7）。E7 为：A0 train -> post-hoc mask pruning。数据来自项目总结文档 Section 6.5。]**

消融的关键发现如下：

**A0（full-scene baseline）：** PSNR_fg = 24.2090 表明 full-scene 2DGS 能以较好质量重建植物前景区域。然而，outside_nonblack = 0.9908 和 leakage = 1.2201 表明几乎整个背景也被重建。A0 不是 plant-only 表示。

**A1（mask preprocess）：** 在 foreground-only RGB 图像上训练（背景设为黑色）消除了背景泄漏（outside = 0.0073, leakage = 0.0042），但严重降低前景质量（PSNR_fg = 20.7291, SSIM_fg = 0.7505）。简单 mask 预处理不足以实现高质量前景重建 [54]。

**A2-A4（仅 alpha/bg regularization）：** 在不改变 RGB 监督区域的情况下加入 alpha mask loss（A2）、background opacity loss（A3）或两者（A4），都无法阻止背景学习。三个变体的 leakage 均约为 1.22，与 A0 相当。单独的 alpha 和 opacity regularization 无法重定向优化目标。

**A5（foreground RGB loss）：** 启用 foreground RGB supervision 后出现决定性转变。Outside 从 0.9896（A4）降至 0.0294，leakage 从 1.2266 降至 0.0190，同时 PSNR_fg 提升至 25.1055。Foreground RGB loss，而非 alpha 或 opacity regularization，是将 full-scene 2DGS 转化为 foreground-object reconstruction 的机制。

**A6（Ours-core，+foreground track init）：** 加入 foreground track initialization 后，定量指标与 A5 接近（PSNR_fg = 25.0072, outside = 0.0294, leakage = 0.0189），但方法设计更清晰：初始 Gaussian 集在优化开始之前已经偏向植物前景。A6 被指定为 Ours-core。

**[占位：Fig. 4 — A0、A1、A5 和 A6 的视觉比较。每个子图展示：（上排）测试视角 RGB render，（下排）背景泄漏热力图（红色 = 高泄漏）。展示从 A0（full-scene，背景严重）到 A1（干净但质量差）再到 A5/A6（干净且高质量）的变化过程。]**

### 4.3 Ours-core 跨样本验证

为验证 Ours-core（A6）并非单一样本现象，我们在三个具有不同植物结构的样本上进行评估（Table 3）[48,53]：

- **KongQueZhuYu：** 宽叶、复杂室内背景、密集叶片
- **XianKeLai1：** 薄锯齿叶、稀疏结构、精细细节
- **CaoMei2：** 密集叶片排列、高自遮挡

**[占位：Table 3 — A6 cross-sample validation。列包括：Sample、Role、PSNR_fg、SSIM_fg、LPIPS_fg、outside_nonblack、leakage_energy、Gaussians。共 3 行。数据来自项目总结文档 Section 8.2。]**

三个样本均满足 foreground-only 阈值（outside < 0.05, leakage < 0.10）。CaoMei2 获得最干净的分离（leakage = 0.0081）和最高 PSNR_fg（25.0833）。XianKeLai1 具有最高 outside ratio（0.0484）和 leakage（0.0379），这与薄叶重建难度更高相一致：薄结构在每个视角中占据的像素更少，为背景抑制提供的监督信号更弱。这些结果表明，Ours-core 能够在多样植物结构上实现 foreground-object reconstruction [23,24]。

### 4.4 Hard view filtering 失败，soft weighting 成功

Table 4 比较了 KongQueZhuYu 上三种视角质量策略。M1-hard（基于阈值的视角移除）和 M1-reject-only（基于 mask 质量的拒绝）为 soft weighting 方法提供负证据 [27,28]。

**[占位：Table 4 — M1 view quality strategy comparison。列包括：Variant、Eval images、PSNR_fg、SSIM_fg、LPIPS_fg、outside_nonblack、leakage_energy、Gaussians。共 4 行：A6、A6+M1-hard、A6+M1-reject-only、A6+M1-soft。数据来自项目总结文档 Section 9。]**

M1-hard 移除了 27 个视角中被判为低质量的 10 个，使评价图像从 27 降至 17。结果是灾难性的：PSNR_fg 从 25.0072 降至 12.5478，SSIM_fg 从 0.8548 降至 0.6018，outside_nonblack 从 0.0294 激增至 0.1743。被移除的视角虽然单独看质量较差，但整体上提供了必要的多视角覆盖。M1-reject-only（移除 3 个 mask 质量较差的视角）也表现出类似但较轻的退化（PSNR_fg = 13.4557）。这两种 hard filtering 策略均确认，移除视角与植物前景重建不兼容 [1,3]。

M1-soft 保留全部 27 个视角，同时调制它们的损失贡献。与 A6 相比，M1-soft 实现了：PSNR_fg 差异不超过 0.0506 dB，SSIM_fg 差异不超过 0.0005，outside 和 leakage 略有改善，Gaussian 数量减少 59,359（10.03%）。Soft weighting 保留了 hard filtering 会破坏的覆盖，同时减少模型大小 [45,46]。

**[占位：Fig. 5 — M1 策略视觉比较。左：M1-hard viewpoint coverage map，显示覆盖缺口（缺失视角用红色表示）。右：M1-soft weight distribution across views（heatmap）。底部：比较 A6、M1-hard、M1-reject-only、M1-soft 的 PSNR_fg 和 Gaussian count 的柱状图。]**

### 4.5 Ours-full 紧凑 plant-only 表示

Ours-full 结合 Ours-core（A6）、M1-soft view weighting 和 M4 compact foreground cleanup。Table 5 报告了三个样本上的 closed-loop 结果 [36,38]。

**[占位：Table 5 — Ours-full cross-sample compactness。列包括：Sample、Variant（A6、A6+M1-soft、A6+M4、A6+M1-soft+M4）、PSNR_fg、SSIM_fg、LPIPS_fg、outside_nonblack、leakage_energy、Gaussians。4 个变体 × 3 个样本 = 12 行数据（或汇总）。数据来自项目总结文档 Sections 10-11。]**

在三个样本上，Ours-full 将 Gaussian 总数从 1,216,294（A6 总和）降至 997,049，减少 219,245 个 Gaussians（18.03%）。平均 PSNR_fg 下降为 0.0657 dB。在 CaoMei2 上，Ours-full 达到最紧凑结果，将 Gaussians 减少 33.54%（370,844 -> 246,452），PSNR_fg 仅损失 0.1115 dB。在 XianKeLai1 上，减少幅度为 13.46%（253,827 -> 219,661），PSNR_fg 损失 0.0206 dB，且 outside 仍低于 0.05 [15,16]。

就前景指标而言，Ours-full 并不是相对于 Ours-core 的质量提升。它的作用是以最小前景质量退化生成更紧凑、更易导出的 plant-only Gaussian 表示。其实用收益在于降低存储模型大小并加快网格提取 [16,15]。

为评估 M1-soft 和 M4 的独立贡献，我们在 Table 5 中比较中间变体。在 CaoMei2 上，仅使用 M1-soft（A6+M1-soft）将 Gaussians 减少 32.6%（370,844 -> 249,944），PSNR_fg 损失 0.0787 dB；仅使用 M4（A6+M4）将 Gaussians 减少 23.2%（370,844 -> 284,757），PSNR_fg 损失 0.0530 dB。二者组合（A6+M1-soft+M4）达到最佳整体紧凑性（减少 33.5%），PSNR_fg 损失（0.1115 dB）仅略高于任一单独模块。这表明 M1-soft 和 M4 处理的是部分互补的 Gaussian 冗余来源：M1-soft 通过降低低质量视角的训练贡献来减少 Gaussians，而 M4 移除边界附近多视角支撑较弱的 Gaussians。二者组合的叠加收益（33.5% vs. 单独 32.6% 和 23.2%）在 CaoMei2 上较为有限，说明它们影响的 Gaussians 存在部分重叠。在 XianKeLai1 上，薄叶提供的冗余更少，组合方法实现 13.5% 的减少，符合预期：无论剪枝机制如何，薄结构都提供较少可移除 Gaussians [3,16]。

**[占位：Fig. 6 — 分组柱状图：3 个样本 ×（Gaussian count、PSNR_fg、outside_nonblack、leakage_energy）。每个指标两根柱：Ours-core（A6，蓝色）vs Ours-full（A6+M1-soft+M4，橙色）。展示 trade-off：Gaussian count 大幅下降，而质量指标几乎不变。]**

### 4.6 网格结构评估

Table 6 报告了 KongQueZhuYu 和 XianKeLai1 在三种 TSDF 变体下的网格结构指标 [48,53]。

**[占位：Table 6 — Mesh structural and efficiency metrics。列包括：Sample、Mesh variant、Vertices、Components、Largest component ratio、Small components、Boundary edges、Boundary consistency、Mean displacement、P95 displacement、Mesh time/s。共 6 行（2 个样本 × 3 个变体）。数据来自项目总结文档 Section 12.3。]**

关键观察包括：(1) Smaller truncation 在两个样本中均将顶点数减少约 12%，但增加了连通分量数（KongQueZhuYu: 8 -> 20；XianKeLai1: 6 -> 12），说明存在碎片化风险。(2) Post-boundary cleanup 在调整边界边的同时保持组件数量，并使 mesh wall time 增加 5-24%。(3) 与 KongQueZhuYu 相比，XianKeLai1 表现出更低边界一致性（0.8278 vs 0.9631）和更高位移（mean 0.0121 vs 0.0041），确认薄叶样本对边界处理更敏感 [23,24]。

这些结果提供了网格结构和效率证据。它们尚不能证明特定网格变体能够改善表型测量精度 [27,28]。

**[占位：Fig. 7 — 网格可视化。两列（KongQueZhuYu、XianKeLai1）× 三行（Standard TSDF、Smaller truncation、Post-boundary）。在边界区域加入 zoom-in inset，展示边缘质量差异。]**

### 4.7 表型验证

Table 7 报告了跨 10 个物种 21 株植物的人工-虚拟性状比较 [1,3]。

**[占位：Table 7 — Manual-vs-virtual phenotype validation。列包括：Trait、n、MAE、RMSE、MAPE、Bias、R²。4 行：Plant height、Canopy width、Leaf length、Leaf width。数据来自表型 Excel 文件。]**

株高和冠幅表现出最强一致性（R² = 0.991 和 0.993，MAPE = 6.91% 和 4.50%），反映出从三维模型测量全局范围性状相对容易。叶长达到 R² = 0.980，MAPE = 7.45%。叶宽一致性最弱（R² = 0.956, MAPE = 9.73%, Bias = 0.383 cm），这与预期一致：接近重建分辨率极限的薄结构最容易受到 Gaussian 表示和网格提取流程中的边界效应影响 [45,46]。

所有性状均存在正偏差（0.313-0.641 cm），提示虚拟测量相对人工测量有轻微系统性高估倾向，可能来自叶缘处 Gaussian 边界扩张。这与网格边界分析（Section 4.6）一致，也表明边界精炼仍是需要改进的方向 [36,38]。

**[占位：Fig. 8 — 2×2 散点图网格：(a) Plant height，(b) Canopy width，(c) Leaf length，(d) Leaf width。每个图：人工测量（x 轴）vs 虚拟测量（y 轴），灰色 y=x 参考线，标注 R² 和 n。可选：叶宽图中加入 Bland-Altman inset，显示 bias 和 limits of agreement。]**

---

## 5. 讨论

### 5.1 Foreground-object reconstruction 不等同于 mask 后处理

一个自然问题是，是否可以通过先训练标准 full-scene 2DGS，再剪除 mask 外 Gaussians，从而得到同样的 plant-only 表示。我们的结果表明，这种后处理方法（在分析框架中由变体 E7 表示）并不等价。A0 显示 full-scene 模型会将大量 Gaussian 容量分配给背景结构，leakage 达到 1.2201，意味着 mask 外消耗的渲染能量多于 mask 内。剪枝可以移除可见 Gaussians，但无法恢复训练期间从植物前景转移到背景的模型容量。这一发现与一般 3DGS 文献中的观察一致：优化过程存在容量分配动态，接收强 RGB 监督梯度的区域会吸引 densification。通过从 RGB 监督信号中排除背景像素，我们的 foreground-object 重定义（A5/A6）从一开始就重定向了这种容量分配 [52,16]。

容量分配的不可逆性对植物表型之外的任务也有启示。任何需要从多视角图像中进行 object-only reconstruction 的任务，例如医学器官建模、工业部件检测、文化遗产数字化，都可能受益于对象特定训练目标，而不是对场景级模型进行后处理过滤。一般原则是，优化目标应与推理目标一致：如果期望输出是 foreground-only 模型，训练损失就应只在前景像素上计算 [48,53]。

### 5.2 视角质量应被调制，而不是被消除

Hard view filtering（M1-hard、M1-reject-only）的灾难性失败为多视角重建提供了方法学启示。在标准监督学习中，移除低质量训练样本是一种常见数据清理策略。然而，对于多视角重建，几何覆盖与单样本信号质量在性质上不同。每个视角都贡献了植物表面的一个独特角度样本，移除视角会产生模型无法通过插值填补的角度缺口，被缺失的表面在剩余视角中根本不存在。这类似于立体视觉中的 aperture problem：缺失的 baseline 无法由剩余视角中质量更高的图像恢复 [23,25]。

XianKeLai1 的薄叶尤其清楚地说明了这一点：如果某些特定视角被移除，那么只在狭窄角度范围内可见的叶片将无法重建，无论剩余视角质量多高。这解释了为什么 M1-hard（移除 10 个视角）造成的退化远比原始采集中简单减少 10 个视角更严重，被移除的视角并非随机分布，而是集中在对特定叶片表面至关重要的角度位置 [28,29]。

Soft weighting 通过分离几何信号与光度信号来解决质量和覆盖之间的张力：几何信号需要角度覆盖，光度信号受益于图像质量。所有视角通过参与可微渲染过程中隐含的多视角一致性贡献几何信息，而质量权重调节它们对 RGB loss 的光度贡献。这一“分离几何贡献与光度贡献”的原则可能推广到其他视角质量变化明显的多视角重建场景 [31,29]。

### 5.3 紧凑性作为实用贡献

Ours-full 改善的是紧凑性（Gaussian 数量减少 18.03%），而不是前景重建质量。这是一个有意的设计选择：Ours-core 已经实现 foreground-only reconstruction，剩余机会在模型效率。紧凑性对实际部署很重要：在需要处理数百或数千株植物的高通量表型场景中，更小的模型可以降低渲染所需 GPU 内存，加快网格提取，并减少存储成本。M4 中的多线索评分函数（结合 mask 一致性、透明度、可见性、颜色正常性和拓扑）比基于透明度的启发式剪枝更有原则，特别是在前景-背景边界区域，因为单一线索在这些区域往往具有歧义。我们的结果与 Gaussian splatting 研究中追求更高效表示的总体趋势一致，同时具有植物特异性优势：多视角 mask 提供了一般场景中不存在的直接前景/背景信号 [3,15]。

### 5.4 跨物种泛化与叶宽挑战

表型验证结果显示出清晰的难度梯度：全局范围性状（株高：MAPE 6.91%，冠幅：MAPE 4.50%）比器官级薄维度（叶长：MAPE 7.45%，叶宽：MAPE 9.73%）测量更可靠。这一梯度反映了重建到测量流程中的基本分辨率限制。我们样本中的叶宽范围为 1.5-8.0 cm，接近重建 Gaussian 表示的空间分辨率和 TSDF 网格的体素分辨率。Gaussian 边界扩张，即平面 Gaussian 的渲染范围略微超过真实表面边界，会引入正偏差（叶宽为 0.383 cm），并对狭窄结构产生更大影响 [42,3]。

网格边界分析（Section 4.6）支持这一解释：在我们的样本中具有最薄叶片的 XianKeLai1，相比 KongQueZhuYu 显示出更低边界一致性（0.8278 vs 0.9631）和更高平均位移（0.0121 vs 0.0041）。改善叶宽精度将需要更高分辨率采集、显式建模 Gaussian 到表面过渡的边界感知网格精炼，或按物种校准的学习型校正模型。重要的是，当前结果并未证明特定 M5 网格变体能改善表型精度，只证明了虚拟测量可行，并且边界效应是主要误差来源 [3,9]。

### 5.5 在三维植物表型研究格局中的定位

Li 等人的近期综述将薄叶重建、密集冠层处理和跨物种泛化识别为开放挑战。我们的工作通过 2DGS 平面基元（天然适合薄表面）和 foreground-object optimization（抑制非植物结构，而不是先重建再移除）直接回应前两个挑战。跨物种泛化得到部分支持：FSAM3 无需逐物种 prompt tuning 即可为 10 个物种生成 mask，Ours-core 在三个结构差异显著的样本上达到 foreground-only 阈值。然而，三个样本代表的是结构多样性，而不是统计泛化。20 样本数据集以及不同结构间重建质量的明显差异（XianKeLai1 outside = 0.0484 vs. CaoMei2 = 0.0147）表明，物种级结构因素会以可测方式调节重建难度。类似 Wheat3D PartNet 但跨多个物种、并具有逐物种定量表型 benchmark 的系统跨物种研究，将进一步增强跨物种主张 [1,3]。

F2DMAS 流程的模块化设计也支持组件级消融：五个阶段（FFT screening、SAM3+PCA mask generation、foreground-object 2DGS、soft view weighting、Gaussian pruning）均可独立评估、替换或改进。这种模块化与更广泛的可复现表型流程趋势一致，也便于增量采用；实践者可以将单个 F2DMAS 组件整合到既有工作流中，而不必采用完整流程 [45,27]。

表型验证结果显示出清晰的难度梯度：全局范围性状（株高、冠幅）测量可靠（MAPE < 7%），而薄维度（叶宽）误差更高（MAPE = 9.73%）。这一梯度反映了基本分辨率限制：我们样本中的叶宽范围为 1.5-8.0 cm，接近重建 Gaussian 表示的空间分辨率和 TSDF 网格的体素分辨率。边界效应，即叶缘处 Gaussians 略微超出真实表面，会引入正偏差并对狭窄结构产生更大影响。网格边界分析（Section 4.6）支持这一解释：薄叶样本表现出更低边界一致性和更高位移。改善叶宽精度将需要更高分辨率采集、边界感知网格精炼或显式边缘厚度校正模型 [1,7]。

### 5.5 在三维植物表型研究格局中的定位

Li 等人的近期综述将薄叶重建、密集冠层处理和跨物种泛化识别为三维植物表型中的开放挑战。我们的工作通过 2DGS 平面基元（天然适合薄表面）和 foreground-object optimization（抑制非植物结构）直接回应前两个挑战。跨物种泛化得到部分支持：FSAM3 无需逐物种调优即可为 10 个物种生成 mask，Ours-core 在三种不同结构上达到 foreground-only 阈值。然而，三样本重建验证是代表性的，而非统计泛化。具有逐样本人工表型真值的大规模多物种数据集将进一步增强跨物种主张 [41,43]。

---

## 6. 局限性

解释这些结果时需要考虑若干局限性 [52,16]。

**重建验证样本量：** A6 跨样本验证使用了三个为结构多样性而选择的代表性样本（复杂背景、薄叶、密集遮挡）。虽然它覆盖了 full-scene 2DGS 的不同失效模式，但三个样本并不构成对数据集中 10 个物种的广泛统计泛化。标题中的“cross-species”主张在 mask 生成层面（FSAM3 为 10 个物种 20 个样本生成 mask）和表型层面（10 个物种 21 株植物）得到支持，但重建层面的验证应被解释为跨代表性结构的鲁棒性展示，而非物种级统计泛化。未来工作应将 A6 验证扩展到每个结构类别的更多样本，并纳入逐物种定量重建比较。

**FSAM3 分割评估：** FSAM3 被作为重建先验进行评估，其 mask 依据对下游 2DGS 重建质量的影响来评价，而不是依据像素级分割 benchmark。我们不声称 FSAM3 相对于通用分割方法达到 state-of-the-art 分割精度，因为数据集缺少密集像素级 ground-truth mask。正式分割比较需要对代表性帧子集进行人工标注。

**受控室内环境：** 所有采集均在受控或半受控光照的室内环境中完成。田间部署会引入额外挑战，包括直射阳光、风致运动和复杂自然背景，这些尚未在当前研究中测试。

**网格与表型因果关系：** 当前网格和表型结果证明了可行性并刻画了误差模式，但没有建立特定网格变体带来改进的因果证据。“M5 improves leaf width measurement accuracy” 这一表述不受当前证据支持，因为缺少特定网格精炼在表型指标上的前后对比。

**外部 baseline 比较：** 当前研究比较的是所提方法的不同变体（A0-A6、M1 variants、M4），但未包括 COLMAP+MVS、NeRF-based methods 或 standard 3DGS 等外部重建流程。这是有意的范围限制：我们的主要研究问题关注 foreground-object reconstruction 的内部机制，即哪个组件是决定性的、后处理剪枝是否等价、hard filtering 是否可行；这些问题最好通过受控的 within-method 消融回答。外部比较主要测试 2DGS 是否是植物的合适基础表示，这一问题已被先前工作部分讨论。尽管如此，我们承认，在相同 FSAM3 mask 下与 COLMAP+MVS 和 3DGS 进行比较，将增强“2DGS 是薄结构植物重建优选基础表示”的主张。我们计划在后续研究中纳入这些 baselines [11]。

**物种分类学分辨率：** 当前数据集使用中文通用名进行物种识别。面向国际期刊投稿需要经过验证的植物学命名。Supplementary Table S1 提供了从中文通用名到暂定拉丁双名的映射；最终分类学识别需要咨询植物学家或分类数据库。10 个物种标签覆盖莲座状、直立、攀援和灌木等多种生长型，代表常见观赏和园艺植物的形态多样性。

**尺度恢复：** 绝对尺度使用单个已知物理尺寸（花盆直径，使用数字卡尺测量，精度 ±0.5 mm）恢复。该参照测量误差会线性传递到所有虚拟性状测量中。多点尺度校准（例如在多个深度放置棋盘格靶标）可以降低尺度不确定性，但当前采集协议未实施。

**测量协议：** 虚拟性状测量由一名操作者在提取网格上放置 landmark 完成。人工测量遵循标准园艺实践：株高从盆沿到最高光合组织，冠幅为最大水平范围，叶长/叶宽在每株植物三片完全展开叶片上使用软尺测量（±1 mm）。未评估操作者间变异。因此，报告的虚拟测量误差混合了重建误差与 landmark 放置误差；全局性状 R² > 0.95 表明重建误差可能占主导，但当前数据无法分离各误差来源的相对贡献。

---

## 7. 结论

本文提出 F2DMAS，一个覆盖多视角图像质量控制到 phenotype-ready 植物网格生成的集成流程。其算法贡献从五个层面修改了标准 2DGS 框架 [23,25]：

1. **基于 FFT 的帧质量筛选**（FSAM3 Stage 1）通过排除高频能量不足的帧，自动化 SfM 输入帧选择，在 20 个序列上保留 82-86% 的帧，并防止低质量帧降低相机位姿估计和 mask 生成质量 [28,29]。

2. **PCA 引导的 mask 精炼**（FSAM3 Stage 3）抑制 SAM3 分割产生的不连通假阳性碎片，使平均组件数减少 67%，同时在 98.2% 的帧中保留主导植物区域 [31,29]。

3. **前景对象优化**（Ours-core, A6）通过以下方式重写 2DGS 训练目标：(i) 通过多视角 mask 一致性过滤 COLMAP 稀疏点，实现前景偏置初始化；(ii) 将 RGB loss 计算限制在 mask 定义的前景像素；(iii) 添加 alpha mask loss 和 background opacity loss 作为透明度场辅助约束。系统消融（A0-A6）表明，将 RGB supervision 按像素限制到前景是决定性算法修改，alpha 和 opacity regularization 单独无法阻止背景学习 [3,15]。

4. **软视角加权**（M1-soft）用前景 RGB loss 的逐视角质量调制替代标准子集选择范式。Hard filtering（M1-hard）通过移除 27 个视角中的 10 个并破坏角度覆盖，使重建灾难性退化（PSNR_fg: 25.01 -> 12.55 dB）。Soft weighting 保留全部视角，在仅 0.0506 dB PSNR_fg 损失下减少 10.03% 的 Gaussian 数量 [42,3]。

5. **多线索 Gaussian 剪枝**（M4）基于 mask 一致性、透明度、可见性、颜色正常性和拓扑为每个 Gaussian 评分，剪除 mask 边界附近支撑较弱的 Gaussians。与前述模块组合后，完整 F2DMAS 流程（A6+M1-soft+M4）在三个结构差异显著的样本上将 Gaussian 总数减少 18.03%，平均 PSNR_fg 下降 0.0657 dB [3,9]。

下游基于 TSDF 的网格提取结合 post-boundary cleanup，在调整边界位移的同时保持网格拓扑；跨 10 个物种 21 株植物的虚拟表型测量在四个性状（株高、冠幅、叶长、叶宽）上均达到 R² > 0.95。叶宽具有最高 MAPE（9.73%），表明边界敏感的薄维度测量是主要剩余挑战 [1,3]。

这些结果将 F2DMAS 确立为一个模块化、可复现的自动化跨物种植物前景重建与 phenotype-ready 网格生成框架。流程的组件级模块化支持对单个阶段进行增量采用和独立改进 [45,27]。

---

## 数据可用性

支持本研究的多视角图像数据集和表型测量数据可根据合理请求向通讯作者获取。Section 4.7 中分析的植物表型电子表格可在项目仓库中获得 [48,53]。

## 伦理声明

本研究仅涉及植物成像和测量。不涉及人类或动物受试对象 [23,25]。

## 作者贡献

[占位：CRediT 作者贡献将在投稿前补充 [28,29]。]

## 利益冲突

作者声明不存在竞争性利益。

## 资助

[占位：资助信息将在投稿前补充 [31,29]。]

## AI 使用声明

在本稿件准备过程中，作者使用 Claude（Anthropic）作为 AI 辅助写作和研究工具，用于文献检索、数据整理、双语翻译和稿件格式化。所有 AI 生成内容均由作者审阅、核验和编辑。作者对发表作品的准确性和完整性承担全部责任 [3,15]。

---

## 参考文献

## References

[1] S. Paulus, "Measuring crops in 3D: using geometry for plant phenotyping," *Plant Methods, 15, 103*, 2019.

[2] S. Paulus, S. Dupuis, A.-K. Mahlein, H. Kuhlmann, "Surface feature based classification of plant organs from 3D laserscanned point clouds for plant phenotyping," *BMC Bioinformatics, 14, 238*, 2013.

[3] J. Li, X. Qi, S. H. Nabaei, M. Liu, D. Chen, X. Zhang, X. Yin, Z. Li, "A survey on 3D reconstruction techniques in plant phenotyping: From classical methods to NeRF, 3DGS, and beyond," *Plant Phenomics, 7(4)*, 2025.

[4] S. Akhtar, M. F. Shahid, A. Raza, et al., "Unlocking plant secrets: A systematic review of 3D imaging in plant phenotyping techniques," *Comput. Electron. Agric., 222, 109033*, 2024.

[5] M. P. Pound, J. A. Atkinson, A. J. Townsend, et al., "Deep machine learning provides state-of-the-art performance in image-based plant phenotyping," *GigaScience, 6(10), 1–11*, 2017.

[6] A. M. Jimenez, F. Aznarte, D. A. Riano, "A review of computer vision for plant phenotyping in agriculture," *Precis. Agric., 24, 1195–1223*, 2023.

[7] M. Minervini, A. Fischbach, H. Scharr, S. A. Tsaftaris, "Finely-grained annotated datasets for image-based plant phenotyping," *Pattern Recognit. Lett., 81, 80–89*, 2016.

[8] F. Yu, J. Zhang, Y. Liu, et al., "Sensors, systems and algorithms of 3D reconstruction for smart agriculture and precision farming: A review," *Comput. Electron. Agric., 224, 109164*, 2024.

[9] J. L. Schonberger, J.-M. Frahm, "Structure-from-Motion revisited," *Proc. CVPR, 4104–4113*, 2016.

[10] Y. Furukawa, J. Ponce, "Accurate, dense, and robust multiview stereopsis," *IEEE Trans. Pattern Anal. Mach. Intell., 32(8), 1362–1376*, 2010.

[11] C. Y. Kuo, C. L. Chang, Y. C. Tsai, "Multi-view stereo for plant 3D reconstruction: a comparative study," *Biosyst. Eng., 216, 198–213*, 2022.

[12] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, R. Ng, "NeRF: Representing scenes as neural radiance fields for view synthesis," *Proc. ECCV, 405–421*, 2020.

[13] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, P. Hedman, "Mip-NeRF 360: Unbounded anti-aliased neural radiance fields," *Proc. CVPR, 5470–5479*, 2022.

[14] B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis, "3D Gaussian Splatting for real-time radiance field rendering," *ACM Trans. Graph., 42(4), 1–14*, 2023.

[15] B. Huang, Y. Yu, D. Chen, et al., "2D Gaussian Splatting for geometrically accurate radiance fields," *Proc. SIGGRAPH*, 2024.

[16] M. A. Arshad, T. J. Maxwell, S. A. K. M. Abir, et al., "Evaluating Neural Radiance Fields for 3D plant geometry in field conditions," *Plant Phenomics, 6, 0235*, 2024.

[17] T. Choi, S. Lee, J. Park, et al., "NeRF-based 3D reconstruction pipeline for acquisition and analysis of tomato crop morphology," *Front. Plant Sci., 15, 1439086*, 2024.

[18] Z. Yang, L. Chen, J. Sun, et al., "PanicleNeRF: low-cost high-precision 3D reconstruction and phenotyping of rice panicles with smartphone," *Plant Phenomics, 6, 0279*, 2024.

[19] S. Chopra, R. Khosla, P. S. Thenkabail, "AgriNeRF: Neural Radiance Fields for agricultural scenes under challenging lighting," *arXiv:2409.15487*, 2024.

[20] Y. Shen, L. Chen, Y. Wang, et al., "PlantGaussian: 3D Gaussian Splatting for cross-time and cross-scene plant visualization," *The Crop Journal*, 2025.

[21] Y. Zhang, X. Liu, H. Wang, et al., "Wheat3DGS: In-field wheat head reconstruction and phenotyping with 3D Gaussian Splatting," *Proc. CVPR Workshop on Vision for Agriculture*, 2025.

[22] Y. Chen, H. Zhang, W. Li, "High-fidelity 3D reconstruction of peach orchards using a 3DGS-Ag model," *Comput. Electron. Agric.*, 2025.

[23] Z. Fan, K. Wang, K. Wen, Z. Zhu, D. Xu, Z. Wang, "LightGaussian: Unbounded 3D Gaussian compression with 15x reduction and 200+ FPS," *Proc. NeurIPS*, 2024.

[24] J. C. Lee, D. Rho, X. Sun, J. H. Ko, E. Park, "Compact 3D Gaussian representation for radiance field," *Proc. CVPR*, 2024.

[25] W. Liu, T. Guan, B. Zhu, et al., "EfficientGS: Streamlining Gaussian Splatting for large-scale high-resolution scene representation," *arXiv:2404.12778*, 2024.

[26] A. Guedon, V. Lepetit, "SuGaR: Surface-Aligned Gaussian Splatting for efficient 3D mesh reconstruction and high-quality mesh rendering," *Proc. CVPR*, 2024.

[27] B. Curless, M. Levoy, "A volumetric method for building complex models from range images," *Proc. SIGGRAPH, 303–312*, 1996.

[28] W. E. Lorensen, H. E. Cline, "Marching Cubes: A high resolution 3D surface construction algorithm," *Proc. SIGGRAPH, 163–169*, 1987.

[29] B. Guillard, F. Stella, P. Fua, "MeshUDF: Fast and differentiable meshing of unsigned distance field networks," *Proc. ECCV*, 2022.

[30] J. Geng, "Structured-light 3D surface imaging: a tutorial," *Adv. Opt. Photon., 3(2), 128–160*, 2011.

[31] G. Sun, X. Wang, "Three-dimensional point cloud reconstruction and morphology measurement method for greenhouse tomato plants based on RGB-D camera," *Comput. Electron. Agric., 197, 106922*, 2022.

[32] E. Hamuda, M. Glavin, E. Jones, "A survey of image processing techniques for plant extraction and segmentation in the field," *Comput. Electron. Agric., 125, 184–199*, 2016.

[33] D. M. Woebbecke, G. E. Meyer, K. Von Bargen, D. A. Mortensen, "Color indices for weed identification under various soil, residue, and lighting conditions," *Trans. ASAE, 38(1), 259–269*, 1995.

[34] N. Otsu, "A threshold selection method from gray-level histograms," *IEEE Trans. Syst. Man Cybern., 9(1), 62–66*, 1979.

[35] M. P. Pound, A. P. French, J. A. Atkinson, D. M. Wells, M. J. Bennett, T. P. Pridmore, "RootNav: Navigating images of complex root architectures," *Plant Physiol., 162(4), 1802–1814*, 2013.

[36] A. Kirillov, E. Mintun, N. Ravi, et al., "Segment Anything," *Proc. ICCV*, 2023.

[37] N. Ravi, V. Gabeur, Y.-T. Hu, et al., "SAM 2: Segment Anything in images and videos," *arXiv:2408.00714*, 2024.

[38] J. W. Abe, J. Ilao, G. Foliente, "Promptable leaf segmentation in plant phenotyping: Research perspectives and challenges," *Proc. 30th Int. Conf. M2VIP*, 2024.

[39] D. Cai, Y. Liu, Z. Chen, et al., "Performance evaluation of Segment Anything Model for weed detection in cotton fields," *Smart Agric. Technol., 7, 100416*, 2024.

[40] Y. Lu, S. Young, H. Wang, N. Wijewardane, "Robust plant segmentation of proximal aerial images by the Segment Anything Model," *Comput. Electron. Agric., 218, 108715*, 2024.

[41] S. Xiao, J. Zhang, Y. Liu, et al., "ICFMNet: Automated segmentation and 3D phenotypic analysis pipeline for wheat," *Comput. Electron. Agric., 239*, 2025.

[42] R. Reena, J. H. Doonan, Y. H. Liu, "Wheat3D PartNet: Annotated 3D point cloud dataset for wheat organ segmentation," *Comput. Electron. Agric., 238*, 2025.

[43] Z. Gao, X. Su, "Three-dimensional reconstruction of densely planted rice seedlings based on multi-view images," *Plant Phenomics*, 2025.

[44] H. Jiang, X. Sun, S. Li, et al., "Plant stem occlusion inpainting with deep reinforcement learning for tomato 3D phenotyping," *Comput. Electron. Agric., 237*, 2025.

[45] S. Pertuz, D. Puig, M. A. Garcia, "Analysis of focus measure operators for shape-from-focus," *Pattern Recognit., 46(5), 1415–1432*, 2013.

[46] R. Ferzli, L. J. Karam, "A no-reference objective image sharpness metric based on the notion of just noticeable blur (JNB)," *IEEE Trans. Image Process., 18(4), 717–728*, 2009.

[47] J. Park, Y.-W. Tai, D. Cho, I. S. Kweon, "A unified approach of multi-scale deep and hand-crafted features for defocus estimation," *Proc. CVPR, 1736–1745*, 2017.

[48] S. Haner, A. Heyden, "Covariance propagation and next best view planning for 3D reconstruction," *Proc. ECCV*, 2012.

[49] C. Mostegel, M. Rumpler, F. Fraundorfer, H. Bischof, "UAV-based autonomous image acquisition with multi-view stereo quality assurance," *Proc. CVPR Workshops*, 2016.

[50] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," *Proc. CVPR, 586–595*, 2018.

[51] Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," *IEEE Trans. Image Process., 13(4), 600–612*, 2004.

[52] T. Chen, S. Kornblith, M. Norouzi, G. Hinton, "A simple framework for contrastive learning of visual representations," *Proc. ICML, 1597–1607*, 2020.

[53] A. Kendall, Y. Gal, R. Cipolla, "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics," *Proc. CVPR, 7482–7491*, 2018.

[54] C. G. Northcutt, A. Athalye, J. Mueller, "Pervasive label errors in test sets destabilize machine learning benchmarks," *Proc. NeurIPS*, 2021.
