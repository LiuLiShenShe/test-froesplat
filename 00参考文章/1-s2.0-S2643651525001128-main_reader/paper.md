---
title: "IPENS: Interactive unsupervised framework for rapid plant phenotyping extraction via NeRF-SAM2 fusion"
authors: "Wentao Song; He Huang; Fang Qu; Jiaqi Zhang; Longhui Fang; Yuwei Hao; Chenyang Peng; Youqiang Sun"
journal: "Plant Phenomics 7 (2025) 100106"
doi: 10.1016/j.plaphe.2025.100106
source_pdf: /data/fj/F2DMAS/00参考文章/1-s2.0-S2643651525001128-main.pdf
generated: 2026-05-26
reader_type: bilingual_source_grounded_markdown
---

# IPENS: Interactive unsupervised framework for rapid plant phenotyping extraction via NeRF-SAM2 fusion

**中文题名：** IPENS：通过 NeRF-SAM2 融合实现快速植物表型提取的交互式无监督框架

**来源：** Plant Phenomics 7 (2025) 100106; DOI: 10.1016/j.plaphe.2025.100106

**说明：** 本文件为全文中英对照阅读稿。正文翻译为机器初译并经过领域术语规则校正；双栏公式、表格与参考文献的低置信区域已在 `translation_notes.md` 标注。

## 页面/章节索引

- [1. Introduction](#s007) — p.1
- [2. Materials and methods](#s018) — p.3
- [2.1. Overview of the method](#s020) — p.3
- [2.3. Dataset construction](#s022) — p.3
- [2.2. Data acquisition and reconstruction](#s024) — p.3
- [2.4. Neural Radiance Fields](#s026) — p.3
- [2.5. Pipeline of the interactive model in IPENS](#s039) — p.4
- [2.7. Auxiliary optimization strategy](#s054) — p.5
- [2.8. Phenotypic data extraction method](#s067) — p.7
- [2.8.2. Leaf surface area](#s081) — p.7
- [2.8.1. Voxel volumes of grains and panicles](#s085) — p.7
- [2.8.3. Leaf length and width](#s089) — p.7
- [2.9. Evaluation metrics](#s110) — p.8
- [2.10. Experimental details](#s116) — p.8
- [3. Results](#s135) — p.8
- [3.1. Quantitative experiment](#s139) — p.8
- [3.1.1. The effectiveness of IPENS](#s140) — p.8
- [3.1.2. Time performance analysis](#s165) — p.9
- [3.2. Phenotypic analysis](#s167) — p.9
- [3.2.1. Analysis of rice grain voxel volume](#s168) — p.9
- [3.2.2. Analysis of wheat panicle voxel volume](#s176) — p.10
- [3.2.3. Leaf phenotypic analysis](#s179) — p.11
- [4.2. Effectiveness of the proposed method](#s218) — p.11
- [4. Discussion](#s220) — p.11
- [4.1. Interpretation of 3D segmentation performance](#s221) — p.11
- [4.3. Limitation and future prospects](#s224) — p.11
- [5.1. Multi-species point cloud extraction visualization](#s229) — p.12
- [5. Conclusion](#s232) — p.12
- [5.2. Time consumption of 3D reconstruction models](#s233) — p.12
- [Author contributions](#s241) — p.13
- [Funding](#s243) — p.13
- [Data availability](#s245) — p.13
- [Declaration of competing interest](#s247) — p.13
- [Appendix A. Supplementary data](#s249) — p.13
- [References](#s251) — p.13

## 术语表

| English | 中文 |
| --- | --- |
| plant phenotyping | 植物表型/植物表型分析 |
| Neural Radiance Fields (NeRF) | 神经辐射场（NeRF） |
| Segment Anything Model 2 (SAM2) | Segment Anything Model 2（SAM2） |
| 3D instance segmentation | 三维实例分割 |
| point cloud extraction | 点云提取 |
| mask inverse rendering | 掩膜逆渲染 |
| multi-target collaborative optimization | 多目标协同优化 |
| voxel volume | 体素体积 |
| leaf surface area | 叶片表面积 |
| Structural Similarity Index Measure (SSIM) | 结构相似性指数（SSIM） |

## 全文中英对照


## Page 1

<a id="S001"></a>
**Source:** p.1 S001  
**Type:** body  
**Confidence:** high

**Original:** Research Article

**中文:** 研究论文

<a id="S002"></a>
**Source:** p.1 S002  
**Type:** body  
**Confidence:** high

**Original:** IPENS: Interactive unsupervised framework for rapid plant phenotyping extraction via NeRF-SAM2 fusion

**中文:** IPENS：通过 NeRF-SAM2 融合实现快速植物表型提取的交互式无监督框架

<a id="S003"></a>
**Source:** p.1 S003  
**Type:** body  
**Confidence:** high

**Original:** Wentao Song; He Huang; Fang Qu; Jiaqi Zhang; Longhui Fang; Yuwei Hao; Chenyang Peng; Youqiang Sun

**中文:** Wentao Song；He Huang；Fang Qu；Jiaqi Zhang；Longhui Fang；Yuwei Hao；Chenyang Peng；Youqiang Sun

<a id="S004"></a>
**Source:** p.1 S004  
**Type:** body  
**Confidence:** high

**Original:** Affiliations: Hefei Institutes of Physical Science, Chinese Academy of Sciences, Hefei, China; University of Science and Technology of China, Hefei, China; Anhui Agricultural University, Hefei, China; Anhui Jianzhu University, Hefei, China.

**中文:** 作者单位：中国科学院合肥物质科学研究院；中国科学技术大学；安徽农业大学；安徽建筑大学。

<a id="S005"></a>
**Source:** p.1 S005  
**Type:** body  
**Confidence:** high

**Original:** Keywords: Rice and wheat phenotype; NeRF; SAM2; 3D instance segmentation; Unsupervised

**中文:** 关键词：水稻和小麦表型；NeRF；SAM2；3D 实例分割；无监督

<a id="S006"></a>
**Source:** p.1 S006  
**Type:** body  
**Confidence:** medium

**Original:** Advanced plant phenotyping technologies are vital for trait improvement and accelerating intelligent breeding. Due to the species diversity of plants, existing methods heavily rely on large-scale high-precision manually annotated data. For self-occluded objects at the grain level, unsupervised methods often prove ineffective. This study proposes IPENS, an interactive unsupervised multi-target point cloud extraction method. It utilizes radiance field information to lift 2D masks, segmented by SAM2 (Segment Anything Model 2), into 3D space for target point cloud extraction. A multi-target collaborative optimization strategy addresses the challenge of segmenting multiple targets from a single interaction. On a rice dataset, IPENS achieves a grain-level segmentation mean Intersection over Union (mIoU) of 63.72%. For phenotypic trait estimation, it achieves a grain voxel volume coefficient of determination R2 = 0.7697 (Root Mean Square Error, RMSE = 0.0025), leaf surface area R2 = 0.84 (RMSE = 18.93), and leaf length and width prediction accuracies of R2 = 0.97 and R2 = 0.87 (RMSE = 1.49 and 0.21). On a wheat dataset, IPENS further improves segmentation performance to a mIoU of 89.68%, with exceptional phenotypic estimation results: panicle voxel volume R2 = 0.9956 (RMSE = 0.0055), leaf surface area R2 = 1.00 (RMSE = 0.67), and leaf length and width predictions reaching R2 = 0.99 and R2 = 0.92 (RMSE = 0.23 and 0.15). Without requiring annotated data, IPENS rapidly extracts grain-level point clouds for multiple targets within 3 min using single-round image interactions. These features make IPENS a high-quality, non-invasive phenotypic extraction solution for rice and wheat, offering significant potential to enhance intelligent breeding.

**中文:** 先进的植物表型技术对于性状改良和加速智能育种至关重要。由于植物物种多样性，现有方法高度依赖大规模、高精度的人工标注数据。对于籽粒级别存在自遮挡的目标，无监督方法往往效果有限。本文提出 IPENS，一种交互式无监督多目标点云提取方法。该方法利用辐射场信息，将 SAM2（Segment Anything Model 2）分割得到的 2D 掩膜提升到 3D 空间，从而提取目标点云。多目标协同优化策略解决了单次交互中分割多个目标的挑战。在水稻数据集上，IPENS 的籽粒级分割平均交并比（mIoU）达到 63.72%。在表型性状估计方面，籽粒体素体积的决定系数 R2 = 0.7697（RMSE = 0.0025），叶片表面积 R2 = 0.84（RMSE = 18.93），叶长和叶宽预测精度分别为 R2 = 0.97 与 R2 = 0.87（RMSE = 1.49 与 0.21）。在小麦数据集上，IPENS 将分割性能进一步提升到 mIoU = 89.68%，并取得优异的表型估计结果：穗部体素体积 R2 = 0.9956（RMSE = 0.0055），叶片表面积 R2 = 1.00（RMSE = 0.67），叶长和叶宽预测分别达到 R2 = 0.99 与 R2 = 0.92（RMSE = 0.23 与 0.15）。在不需要标注数据的情况下，IPENS 通过单轮图像交互，可在 3 分钟内快速提取多个目标的籽粒级点云。这些特性使 IPENS 成为面向水稻和小麦的高质量、非侵入式表型提取方案，并具有促进智能育种的重要潜力。

<a id="S007"></a>
### 1. Introduction
**Source:** p.1 S007  
**Type:** section  
**Confidence:** high

**Original:** 1. Introduction

**中文:** 1. 引言

<a id="S008"></a>
**Source:** p.1 S008  
**Type:** body  
**Confidence:** high

**Original:** Plant phenotypic analysis stands as a central direction in modern agricultural research. By elucidating the coordinated expression mechanisms of genomic information and environmental factors [1], it provides theoretical foundations for establishing genotype-phenotype association models [2]. This research framework holds significant application value for achieving targeted trait improvement in plants and accelerating intelligent breeding processes [3–5]. Current phenotyping technologies primarily focus on the visual interpretation of plant morphological features under multi-environment conditions and the dynamic monitoring of physiological traits, involving precise quantification and systematic evaluation of multidimensional phenotypic parameters [6]. Traditional phenotyping methods often rely on manual sampling and invasive detection, facing technical limitations such as limited measurement dimensions, low throughput, and non-reusability of samples, which severely constrain the efficiency of large-scale breeding studies. Consequently, the development of non-invasive phenotyping extraction technologies based on optical reconstruction and deep learning algorithms, along with the construction of automated phenotyping platforms, has emerged as a cutting-edge research hotspot in the interdisciplinary fields of plant genomics and intelligent agriculture [7]. Compared to traditional two-dimensional (2D) phenotyping methods [8], three-dimensional (3D) phenotyping technologies offer more comprehensive and accurate representations of plant structures. These

**中文:** 植物表型分析是现代农业研究的核心方向.通过阐明基因组信息和环境因素的协调表达机制 [1],它为建立基因型-表型协会模型提供了理论基础 [2].该研究框架具有显著的应用价值,以实现植物的目标特征改善和加速智能育种过程 [35].目前的表型技术主要集中在多环境条件下的植物形态特征的视觉解释和生理特征的动态监测,涉及精确量化和系统评估多维表型参数 [6]. 传统的表型方法通常依赖于手动采样和侵入性检测,面临技术局限性,如有限的测量尺寸,低吞吐量和样本的非重用性,这严重限制了大型育种研究的效率.因此,基于光学重建和深度学习算法的非侵入性表型提取技术的开发,以及自动化表型平台的建设,已成为植物基因组学和智能农业的跨学科研究热点 [7].与传统的二维 (2D) 表型方法 [8],三维 (3D) 表型技术提供更全面和更准确的植物表现.


## Page 2

<a id="S009"></a>
**Source:** p.2 S009  
**Type:** body  
**Confidence:** medium

**Original:** advancements enable researchers to analyze complex morphological features including shape, area, and angles with enhanced precision [9, 10]. In recent years, Neural Radiance Fields (NeRF) [11] and 3D Gaussian Splatting (3DGS) [12] have gained prominence for their ability to generate highly detailed and realistic 3D representations. Choi et al. [13] demonstrated the superiority of 3D phenotyping by automatically collecting tomato RGB images using unmanned greenhouse robots and reconstructing dense point clouds via NeRF. Their results showed significantly improved accuracy compared to traditional 2D approaches. Similarly, PanicleNeRF [14] achieved high correlations between 3D point cloud-derived panicle volume and grain traits in rice (indica: R2 = 0.85 for grain count, R2 = 0.80 for grain weight; japonica: R2 = 0.82 and R2 = 0.76, respectively) by integrating SAM [15] for 2D segmentation and NeRF for 3D reconstruction. Saeed et al. [16]applied NeRF to reconstruct peanut pods from 2D images, achieving a validation set IoU of 0.5 and precision of approximately 0.7, highlighting NeRF's potential for precise pod localization in legume crops. Shen et al. [17] combined 3DGS with SAM to reconstruct rapeseed plants using 36 oblique UAV images, enabling accurate biomass estimation. Jiang et al. [18] utilized 3DGS to extract point clouds of individual cotton plants. They applied YOLOv8x [19] to generate 2D masks, which were then integrated into SAGA [20] to extract 3D models of cotton bolls and the main stem. This approach enabled the estimation of boll count and main stem length. The results demonstrated mean absolute percentage errors (MAPE) of 11.43% for boll count and 10.45% for main stem length, validating the effectiveness of 3DGS in precise, non-destructive plant phenotyping. Although 3DGS exhibits higher reconstruction efficiency than lightweight models like Instant-NGP [21], its performance suffers in complex experimental environments, often resulting in low reconstruction accuracy or poor point cloud quality. In contrast, NeRF's data-driven nature ensures robust phenotypic trait analysis under diverse field conditions, making it a powerful tool for high-precision plant phenotyping in this study. Accurate 3D object point cloud segmentation constitutes an essential component in the extraction and analysis of 3D phenotypic data. Deep learning approaches have demonstrated great potential in the field of plant organ segmentation. Supervised methods, after being trained on large volumes of annotated data, can achieve high segmentation accuracy of organs[22–26], whereas unsupervised methods—relying on self-organizing and clustering techniques—effectively tackle segmentation tasks under annotation scarcity and complex backgrounds [27–29]. Interactive 3D segmentation methods based on the SAM are also rapidly evolving, thanks to their zero-shot capability across diverse scenarios. SA3D [30] leverages SAM's segmentation power by allowing the user to specify target points in a single view and then auto-generating prompts for the next image's mask, which are fused with NeRF information to extract the target's 3D point cloud. However, when applied to 360-degree scanned rice data, such point prompting strategies face challenges due to self-occlusion characteristics. This leads to prompt point drift, resulting in model failure and inability to export point clouds. Particularly for small targets like grains, SA3D fails to achieve extraction. SANeRF-HQ [31] enhances segmentation boundary accuracy in NeRF-based multi-view aggregation by leveraging density fields and RGB similarity. PointSAM [32], a widely adopted 3D annotation tool, utilizes SAM to generate extensive part-level and object-level pseudo-labels, extracting rich knowledge for fully supervised training in 3D instance segmentation. However, its single-instance-per-interaction and multi-round click optimization mechanism on point cloud data faces critical bottlenecks in interaction efficiency. Segment Anything Model 2 (SAM2) [33] is a significant upgrade by Meta over the SAM. In video segmentation tasks, the model demonstrates flexible utilization of interactive prompts for precise target boundary extraction. These advancements establish novel algorithmic design paradigms for breeding processes.

**中文:** 随着这些技术的进步,研究人员能够精确地分析包括形状,面积和角度在内的复杂的形态特征 [9, 10].近年来,神经辐射场 (Neural Radiance Fields, NeRF) [11]和3D Gaussian Splatting (3DGS) [12]都以其产生高度详细和现实的3D表示能力而闻名.Choi等人 [13]通过使用无人机温室机器人自动收集番茄RGB图像和NeRF来重建密度点云来证明了3D表型形形的优势.他们的结果显示了与传统的2D方法相比的显著提高了精度.同样,PanicleNeRF [14]也实现了3D点云衍生体积和穗部特征之间的高相关性 (指数:R285=水稻籽粒,R280=水稻籽粒; 果:R2 =0.82和R2 =0.76,分别) 通过将SAM [15]用于2D分类和NeRF用于3D重建.萨伊德等人 [16]应用NeRF从2D图像中重建花生,实现了0.5的验证组IoU和约0.7的精度,突出NeRF在豆类作物中精确的定位的潜力.沈等人 [17]将3DGS与SAM结合起来,使用36个斜面无人机图像重建油菜植株,使得生物质量精确估计.江等人 [18]利用3DGS来提取单个棉花植物的点云.他们应用YOLOv8x [19]生成2D掩膜,然后将其集成到SAGA [20]模型中,以提取3D棉花和主要干长度.这使得干和干的长度估计成为最准确的方法. 结果显示,对子数量来说,平均绝对百分比错误 (MAPE) 是11.43%和对主干长度为10.45%的,验证了3DGS在精确的,非破坏性植物 phenotyping中的有效性.虽然3DGS比像Instant-NGP这样的轻量级模型具有更高的重建效率,但其性能在复杂的实验环境中受到损害,通常导致重建精度低或点云质量差.相反,NeRF的数据驱动性确保了在各种场景条件下强大的表型分析,使其成为这个研究中的高精度表型的强大工具.精确的3D对象点分类是3D表型数据的提取和分析中重要组成部分. 深度学习方法在植物器官分类领域已经证明了很大的潜力.监督方法在大量标注数据上训练后,可以实现器官的高分类精度[2226],而基于自我组织和集群技术的非监督方法在标注短缺和复杂背景下有效地解决分类任务[2729].基于SAM的交互式3D分类方法也在迅速发展,由于它们在各种场景中具有零样本击能力.SA3D[30]利用SAM的分类能力,允许用户在单个视图中指定目标点,然后自动生成下一个图像的掩膜提示,这些信息与NeRF相结合,以提取3D目标点云. 然而,当应用到360度扫描的水稻数据时,这种点提示策略面临自遮挡特性所带来的挑战.这导致了提示点漂移,导致模型故障和无法出口点云.特别是对于像籽粒这样的小目标,SA3D未能实现提取.SANeRF-HQ [31]通过利用密度字段和RGB相似性来提高NeRF基于的多视图汇集中的分割边界精度,通过利用密度字段和RGB相似性.SAM PointSAM [32],一种广泛采用的3D标注工具,利用它来生成广泛的部分级和对象级伪标签,提取丰富的知识用于完全监督的3D实例分类培训.然而,其单次实例交互式和多点云优化机制在云中面临关键的数据交互口. 分割任何模型2 (SAM2) [33]是Meta对SAM的重大升级.在视频分割任务中,该模型展示了灵活的利用交互式提示来精确地提取目标边界.这些进步为育种过程建立了新的算法设计范式.

<a id="S010"></a>
**Source:** p.2 S010  
**Type:** body  
**Confidence:** high

**Original:** Although there has been considerable progress in extracting 3D organ point clouds of crops, several challenges remain:

**中文:** 虽然在3D器官点云的提取中取得了相当大的进展,但仍然存在几个挑战:

<a id="S011"></a>
**Source:** p.2 S011  
**Type:** body  
**Confidence:** high

**Original:** • 3D data annotation relies heavily on manual, fine-grained labeling, which incurs high labor costs, long turnaround times, and significant subjective bias. These constraints severely hinder the creation of large-scale, high-precision 3D phenotyping datasets and slow the progress of crop phenotypic analysis. In the breeding field, there is an urgent need for zero-shot, unsupervised 3D organ point-cloud extraction solutions with robust cross-variety adaptability.

**中文:** • 3D数据标注主要依赖于手动,细粒度的标签,这导致高劳动力成本,长期的转换时间和显著的主观偏见.这些限制严重阻碍了大规模,高精度的3D表型数据集的创建,并减缓了作物表型分析的进展.在育种领域,迫切需要具有强大的跨种类适应性的零样本,无监督的3D器官点云提取解决方案.

<a id="S012"></a>
**Source:** p.2 S012  
**Type:** body  
**Confidence:** high

**Original:** • For small, self-occluded rice grains, reconstruction-based interactive point cloud segmentation methods such as SA3D fail to extract the target grains. Therefore, it is necessary to develop a grain-level interactive point cloud extraction scheme that, with only minimal user prompts, can achieve high-quality extraction of individual grains.

**中文:** •对于小型,自遮挡的水稻籽粒,重建式交互式点云分割方法,如SA3D,无法提取目标粒.因此,需要开发一种籽粒级的交互式点云提取方案,只需最小的用户提示,可以实现单个籽粒的高质量提取.

<a id="S013"></a>
**Source:** p.2 S013  
**Type:** body  
**Confidence:** high

**Original:** • Mainstream interactive methods apply a single-instance, iterative correction strategy to point clouds, but their multi-click optimization mechanism creates an interaction-efficiency bottleneck. Existing approaches also exhibit significant shortcomings in triggering multitarget collaborative segmentation at the grain level with a single interaction, making them inadequate for the point cloud extraction demands of complex crops. To address these challenges, this study introduces NeRF and SAM2 models into the field of agricultural phenotyping and proposes an interactive unsupervised prompting framework to explore their potential for high-throughput biomass phenotype extraction in rice and wheat, while also discussing the feasibility of multi-species point cloud extraction. The specific contributions are as follows:

**中文:** •主流交互式方法应用单实例的反复纠正策略,但他们的多点击优化机制创造了交互式效率瓶.现有的方法也显示出引发单一交互式的籽粒级多目标协作分割化方面存在重大缺陷,使它们不适合复杂作物的点云提取需求.为了应对这些挑战,本研究将NeRF和SAM2模型引入农业表型分析领域,并提出了一个交互式的无监督提示框架,以探索它们在水稻和小麦中高吞吐量生物质表型提取的潜力,同时还讨论了多种点云提取的可行性.具体贡献如下:

<a id="S014"></a>
**Source:** p.2 S014  
**Type:** body  
**Confidence:** high

**Original:** • We propose IPENS, an interactive, unsupervised, multi-target phenotyping extraction and analysis framework based on NeRF and SAM2. Breaking the traditional reliance on crop annotation, we propose a single-round prompting solution: by supplying prompt points on just a few images, leveraging SAM2's video-propagation capability to semantically spread those prompts across the 2D image sequence, and combining NeRF's spatial representation strengths with differentiable rendering to lift the resulting 2D masks into 3D space. Ultimately, high-quality 3D point-cloud extraction for rice and wheat was achieved.

**中文:** •我们提出IPENS,一个基于NeRF和SAM2的交互式,无监督的多目标表型提取和分析框架.我们打破了传统对作物标注的依赖,我们提出了一个单轮提示解决方案:通过在几张图片上提供提示点,利用SAM2的视频传播能力,在2D图像序列中语义地传播这些提示,并将NeRF的空间表示强度与可微渲染结合起来,将结果的2D掩膜提升到3D空间.最终,为水稻和小麦实现了高质量的3D点云提取.

<a id="S015"></a>
**Source:** p.2 S015  
**Type:** body  
**Confidence:** high

**Original:** • We propose a multi-target collaborative extraction scheme tailored for breeding scenarios. To address the need for parallel analysis of multiple plant organs, we designed a multi-target training strategy that allows breeders to specify prompts for several organs at once, enabling a single interaction to synchronously extract 3D point clouds of organs such as rice grains, leaves, and stems. The framework supports both manual prompting and an automated prompting scheme based on YOLOv11 [34].

**中文:** • 我们提出了一个针对育种场景的多目标协作提取方案.为了满足多种植物器官的并行分析需求,我们设计了一种多目标培训策略,允许育种者同时指定几个器官的提示,使单个交互式可以同步提取3D点云的器官,如籽粒,叶子和茎.该框架支持手动提取和基于YOLOv11的自动提取方案. [34].

<a id="S016"></a>
**Source:** p.2 S016  
**Type:** body  
**Confidence:** high

**Original:** • We designed a two-stage post-processing optimization strategy. First, morphological inpainting is applied to SAM2's segmentation results to produce smoother, more coherent 2D masks, which in turn improves the accuracy of 3D mask extraction. Second, because self-occlusion of grains causes SAM2's performance to degrade when targets disappear, we need to rapidly relocate their reappearance and supply new prompts to refine the segmentation. To further simplify this process for users, we introduce an intelligent prompting algorithm based on the Structural Similarity Index Measure (SSIM) [35], which helps users quickly identify and annotate rear-view frames in the video, thereby enhancing the quality of 3D point cloud extraction.

**中文:** • 我们设计了一种两阶段后处理优化策略.首先,将 morfological inpainting 应用到SAM2 的分割结果产生更平稳,更一致的2D掩膜,这反过来提高了3D掩膜提取的精度.第二,由于颗粒的自我封闭导致SAM2 的性能在目标消失时降低,我们需要快速重新调整它们的出现,并提供新的提示来完善分割.为了进一步简化这一过程,我们引入了一个基于结构相似度指数测量 (SSIM) [35] 的智能云提示算法,这有助于用户快速识别和标注视频中的后视图框架,从而提高了3D提取的质量.

<a id="S017"></a>
**Source:** p.2 S017  
**Type:** body  
**Confidence:** high

**Original:** • Phenotypic methods were developed to compute the voxel volume of rice grains and wheat grains, leaf surface area, and leaf length and width. The effectiveness of model in point cloud extraction was demonstrated.

**中文:** • 开发了表型方法来计算大水稻籽粒和小麦粒的体素体积,叶片表面积,叶片长度和宽度.在点云采集中,该模型的有效性被证明.


## Page 3

<a id="S018"></a>
### 2. Materials and methods
**Source:** p.3 S018  
**Type:** section  
**Confidence:** high

**Original:** 2. Materials and methods

**中文:** 2. 材料与方法

<a id="S019"></a>
**Source:** p.3 S019  
**Type:** body  
**Confidence:** high

**Original:** film was removed for continued growth under a photoperiod of 22 h light/2 h dark, with light intensity maintained at 200 μmol/m2/s. To acquire high-quality 3D data, the first part of Fig. 1 shows the standardized data acquisition procedure: samples were randomly selected in a breeding accelerator growth chamber, then data capture was performed inside the breeding data acquisition cube. A backend control system precisely guided a robotic arm to execute a 360◦ orbit while synchronously triggering an Intel D435i camera to record 20 s videos at 1280 × 720 pixels and 30 fps; key frames were then extracted at 6 fps. Next, COLMAP estimated the external camera parameters for each image, and nerfacto [38] was used for 3D reconstruction to obtain the crop's NeRF field. Finally, the reconstructed point clouds were exported to facilitate ground-truth annotation.

**中文:** 为了获得高质量的3D数据,图1的第一部分显示了标准化数据采集程序:样本在育种加速器增长室中被随机选择,然后在育种数据采集立方体内部进行数据捕获.一个后端控制系统精确地引导了一个机器人臂执行360◦轨道,同时同步激活一个英特尔D435i摄像头以1280×720像素和30FPS录制20个视频;然后在6FPS中提取关键框.下一步,COLMAP估计了每个图片的外部参数,而NeRFacto [38]用于3D摄像头的重建,以获得NeRF的收获场景. 最后,重建的点云被出口,以促进地面真相标注.

<a id="F001"></a>
### Fig. 1. IPENS 的总体流程：数据准备、模型与方法，以及表型提取。
**Placed near:** p.3 S019  
**Source:** p.3 C002  
**Crop confidence:** high

![Fig. 1](assets/fig1.png)

**Original caption:** Fig. 1. Overall workflow of IPENS: Data preparation, model & method, and phenotyping extraction.

**中文图注:** 图 1. IPENS 的总体流程：数据准备、模型与方法，以及表型提取。

**Reading note:** 重点查看该图如何支撑相邻正文中的方法流程、实验比较或平台应用描述。

<a id="S020"></a>
### 2.1. Overview of the method
**Source:** p.3 S020  
**Type:** section  
**Confidence:** high

**Original:** 2.1. Overview of the method

**中文:** 2.1. 方法概览

<a id="S021"></a>
**Source:** p.3 S021  
**Type:** body  
**Confidence:** high

**Original:** sequence, camera poses are first estimated using COLMAP [36,37], and NeRF is then employed for 3D rendering to generate the crop point cloud. The framework provides two interaction schemes: manual prompting or YOLOv11-assisted prompting on key frames, which are then input to the interactive model. Two post-processing optimization strategies further enhance the 2D segmentation results and generate high-quality 3D segmentation masks. Finally, phenotypic traits are extracted from both the manually annotated 3D point cloud and the model-generated masks, and the resulting estimation errors are evaluated. The following sections describe, in detail, the data acquisition and reconstruction methods (Section 2.2), the dataset construction process (Section 2.3), and the 3D point cloud extraction pipeline (Section 2.5 and 2.6). We also introduce our prompting strategy and present two post-processing algorithms in Section 2.7. Finally, we outline the phenotypic methods for each organ in Section 2.8.

**中文:** 采用COLMAP [36,37],首先估计摄像头姿势,然后NeRF用于3D染,以生成产点云.该框架提供了两个交互方案:手动提示或YOLOv11辅助提示在关键框架上,然后输入到交互模型中.两个后处理优化策略进一步增强了2D分割结果,并产生了高质量的3D分割面膜.最后,从手动标注的3D点云和模型生成的面膜中提取了现象特征,并评估了导致的估值错误.下列部分详细描述了数据采集和重建方法 (第2.2,数据集构建过程 (第2.3,第2.5和第2.6). 我们还介绍了我们的提示策略,并在2.7节介绍了两个后处理算法.最后,我们在2.8节概述了每个器官的表型方法.

<a id="S022"></a>
### 2.3. Dataset construction
**Source:** p.3 S022  
**Type:** section  
**Confidence:** high

**Original:** 2.3. Dataset construction

**中文:** 2.3. 数据集构建

<a id="S023"></a>
**Source:** p.3 S023  
**Type:** body  
**Confidence:** high

**Original:** As shown in Fig. 2, each plant's point cloud consists of approximately 100,000 points, from which outliers were removed using a Gaussian distribution method. Subsequently, precise instance labeling was performed using the CloudCompare. The MMR and MMW datasets comprise multi-view images of crops, 2D instance segmentation results, raw XYZRGB data of point clouds, along with annotated labels for semantic and instance segmentation in the point clouds. Statistical information about the point cloud instances is presented in Table 1.

**中文:** 如图2所示,每个工厂的点云由约10万个点组成,从中使用高斯分布方法删除了异常值.随后,使用CloudCompare进行了精确的实例标记.MMR和MMW数据集包括作物多视图图,2D实例分割结果,点云的原始XYZRGB数据,以及点云中的语义和实例分割的标注标签.点云实例的统计信息在表1中呈现.

<a id="F002"></a>
### Fig. 2. 数据采集与实例分割。
**Placed near:** p.3 S023  
**Source:** p.4 C004  
**Crop confidence:** high

![Fig. 2](assets/fig2.png)

**Original caption:** Fig. 2. Data acquisition and instance segmentation.

**中文图注:** 图 2. 数据采集与实例分割。

**Reading note:** 重点查看该图如何支撑相邻正文中的方法流程、实验比较或平台应用描述。

<a id="S024"></a>
### 2.2. Data acquisition and reconstruction
**Source:** p.3 S024  
**Type:** section  
**Confidence:** high

**Original:** 2.2. Data acquisition and reconstruction

**中文:** 2.2. 数据采集与重建

<a id="S025"></a>
**Source:** p.3 S025  
**Type:** body  
**Confidence:** high

**Original:** Experiments were carried out on the multimodal and multitask rice (MMR) and wheat (MMW) datasets, both specifically designed for phenotyping research. This study selected the japonica rice model cultivar Nipponbare (Oryza sativa L. ssp. japonica). During the seedling stage, a 14-h light/10-h dark photoperiod was applied, while the vegetative and reproductive growth stages utilized an adjusted 10-h light/ 14-h dark cycle. Light intensity was consistently maintained at 600 μmol/m2/s (with light source positioned 30 cm above the plant canopy). For winter wheat variety Zhenmai 15 (Triticum aestivum L.), direct sowing in pots was employed. At germination: seeds were wrapped with plastic film to retain moisture and placed in a 22 ◦ C white-light incubator under complete darkness for sprouting; after emergence, thorough watering was conducted. Upon entering the seedling stage, the plastic

**中文:** 在多模式和多任务米 (MMR) 和小麦 (MMW) 数据集上进行了实验,这两种数据集都专门用于表型研究.本研究选择了日本米模型种植品种Nipponbare (Oryza sativa L. ssp. japonica).在苗种阶段,应用了14h光/10h的黑暗光周期,而在植被和生殖的生长阶段使用了调整的10h光/14h黑暗周期.光强度始终保持在600μmol/m2/s (光源位于植物顶30厘米以上).冬季米品种Zhenmai 15 (TriticuMAEsti L.),在中播种.在发芽时:种子被用塑料膜包裹,并放置在一个完全白色光下,米 进入种植阶段后,塑料

<a id="S026"></a>
### 2.4. Neural Radiance Fields
**Source:** p.3 S026  
**Type:** section  
**Confidence:** high

**Original:** 2.4. Neural Radiance Fields

**中文:** 2.4. 神经辐射场

<a id="S027"></a>
**Source:** p.3 S027  
**Type:** body  
**Confidence:** medium

**Original:** Neural Radiance Fields (NeRF) learn a continuous volumetric scene representation from a training dataset I of multi-view 2D images. The model approximates a function fθMask Representation and loss function 3: (x, d) → (c, σ) that maps 3D spatial coordinates x ∈ ℝ3 and viewing direction d ∈ 𝕊2 to emitted color c ∈ ℝ3 and volume density σ ∈ ℝ, where θ denotes the learnable parameters of f. To synthesize novel views Iθ, rendering involves casting rays through each pixel. For a camera positioned at origin xo with ray direction d, points along the ray are parameterized as r(t) = xo + td, where t

**中文:** 神经辐射场 (NeRF) 从训练数据集 I 中学习了多视图2D图像的连续体积场景表示.该模型接近了 fθMask 代表和损失函数 3: (x, d) → (c, σ) 的函数,该函数将3D空间坐标 x ∈ R3和视图方向 d ∈ S2映射到发射的颜色 c ∈ R3 和体积密度 σ ∈ R,其中 θ表示f 的可学习参数.为了合成小说视图 Iθ,染涉及通过每个像素射射射射线.对于一个以射线方向 d 位置的摄像头,沿线点被参数为 rt xo) = xo xo + td,其中 t xo,

<a id="C001"></a>
### Table 1
**Source:** p.3 C001  
**Type:** caption  
**Confidence:** high

**Original:** Table 1. Experiments were carried out on the multimodal and multitask rice (MMR) and wheat (MMW) datasets, both specifically designed for phenotyping research. This study selected the japonica rice model

**中文:** 实验在多模式和多任务米 (MMR) 和小麦 (MMW) 数据集上进行,这两种数据集都专门用于表型研究.本研究选择了日本米模型


## Page 4

<a id="S028"></a>
**Source:** p.4 S028  
**Type:** body  
**Confidence:** high

**Original:** box coordinates while extracting their center points and polygon vertices as prompt information. Compared to previous versions, the YOLOv11 model not only achieves higher segmentation accuracy but also has faster inference speed, capable of accurately locating the centers and boundaries of targets. Since SAM2 does not require highly precise prompt information, providing a general segmentation area is sufficient for prompts, so YOLOv11 is trained with very little data.

**中文:** 与之前的版本相比,YOLOv11模型不仅实现更高的分割准确性,而且具有更快的推断速度,能够准确地定位目标的中心和边界.由于SAM2不需要高度精确的快速信息,为提示提供一个一般的分割割域是足够的,因此YOLOv11是训练有素的数据很少.

<a id="S029"></a>
**Source:** p.4 S029  
**Type:** body  
**Confidence:** high

**Original:** The prompt information encompasses coordinates and label values of positive/negative samples generated through manual annotation or YOLO segmentation inference. SAM2 features an input interface accepting coordinates, labels, frame indices for prompts (with consistent IDs assigned to identical targets), producing per-frame masks for each object as output. Thanks to SAM2's streaming-memory mechanism, segmentation masks propagate continuously and spatiotemporally consistently, significantly reducing user interaction and improving realtime performance. With minimal intervention, high-quality 2D segmentation masks of the target across multiple views are obtained. Since the crop's NeRF field has already been constructed after data acquisition, we leverage it to introduce a multi-target collaborative optimization strategy. By applying mask inverse rendering, we project the 2D masks of multiple targets into 3D space frame by frame, forming an initial coarse 3D mask. This process is iteratively executed across all views. As the number of iterations increases, the 3D mask's accuracy progressively improves, ultimately yielding a high-quality 3D mask for the targets. Following this process, users can interact with IPENS in a single round, clicking on several different targets within the key frames to obtain the corresponding 3D masks.

**中文:** 提示信息包括通过手动标注或YOLO分割推理生成的正/负样本的坐标和标签值.SAM2具有接入界面,接受坐标,标签,框架索引的提示 (具有一致的ID分配给相同的目标),为每个对象产生每框面膜作为输出.由于SAM2的流媒体记忆机制,分割面膜不断地和空间时间不断传播,显著减少用户交互式并提高实时性能.通过最小的干预,高质量的目标2D分割面膜获得了多个视图.由于收获数据后已经构建了作物的NeRF领域,我们利用它引入了多目标合作优化策略. 通过使用掩膜反转染,我们将多个目标的2D掩膜投影到3D空间框架,形成一个初始的粗3D掩膜.这个过程在所有视图中都会进行反复执行.随着反复数量的增加,3D掩膜的精度会逐渐提高,最终为目标产生高质量的3D掩膜. 按照这个过程,用户可以在单个轮中与IPENS进行交互式,点击关键框架内的几个不同的目标,获得相应的3D掩膜.

<a id="S030"></a>
**Source:** p.4 S030  
**Type:** body  
**Confidence:** high

**Original:** 3D instance statistics results instance statistics results. Dataset

**中文:** 3D实例统计结果实例统计结果.数据集

<a id="S031"></a>
**Source:** p.4 S031  
**Type:** body  
**Confidence:** high

**Original:** Plants

**中文:** 植物

<a id="S032"></a>
**Source:** p.4 S032  
**Type:** body  
**Confidence:** high

**Original:** Class

**中文:** 类 类

<a id="S033"></a>
**Source:** p.4 S033  
**Type:** body  
**Confidence:** high

**Original:** Instance Seg

**中文:** 实例分类

<a id="S034"></a>
**Source:** p.4 S034  
**Type:** body  
**Confidence:** high

**Original:** MMR

**中文:** MMR

<a id="S035"></a>
**Source:** p.4 S035  
**Type:** body  
**Confidence:** high

**Original:** MMW

**中文:** MMW

<a id="S036"></a>
**Source:** p.4 S036  
**Type:** body  
**Confidence:** high

**Original:** Grain Leaf Stem Panicle Leaf Stem

**中文:** 籽粒叶子干穗部叶子干

<a id="S037"></a>
**Source:** p.4 S037  
**Type:** body  
**Confidence:** medium

**Original:** represents distance from the camera center. The RGB color Iθ(r) for ray r is computed via differentiable volume rendering: ∫ tf Iθ (r) = ω(r(t))c(r(t); d) dt; (1) tn

**中文:** 射线r的RGB颜色Iθ(r) 通过可分辨的体素体积染计算:∫ tf Iθ (r) = ω(r(t))c(r(t);d) dt; (1) tn

<a id="S038"></a>
**Source:** p.4 S038  
**Type:** body  
**Confidence:** medium

**Original:** (where the transmittance-weighted density is given by ω(r(t)) = exp −) ∫t tn σ(r(s)) ds ⋅σ(r(t)). The integration bounds tn and tf correspond to the near and far planes of the viewing frustum.

**中文:** (传输权重密度由 ω(r(t)) = exp −) ∫t tn σ(r(s)) ds ⋅σ(r(t)). 整合界限 tn 和 tf 与观测的近距离和远距离平面相符.

<a id="S039"></a>
### 2.5. Pipeline of the interactive model in IPENS
**Source:** p.4 S039  
**Type:** section  
**Confidence:** high

**Original:** 2.5. Pipeline of the interactive model in IPENS

**中文:** 2.5. IPENS 中交互模型的流程

<a id="S040"></a>
**Source:** p.4 S040  
**Type:** body  
**Confidence:** high

**Original:** To address interactive phenotyping requirements, as shown in the second part of Fig. 1, IPENS provides two types of prompts: (1) Interactive prompts, where users can select key frames from the video (with at least the first frame) and manually provide multiple targets, which can be prompt points or bounding box coordinates for objects such as grains, panicles, leaves, and stems; (2) To further reduce user interaction costs, this paper also proposes an automatic prompt method based on YOLOv11 to assist in generating target prompt information. Specifically, for each of the two crop species, ninety key frames were selected for annotation, with organ information meticulously and accurately labeled by human annotators. Using this dataset, we fine-tuned a pretrained YOLOv11x-Seg model for both crops with an input resolution of 640 × 640, using SGD as the optimizer, and an 8:2 train/validation split. The trained model performs 2D instance segmentation, outputting bounding

**中文:** 为了解决交互式表型分析要求,如图1的第二部分所示,IPENS提供了两种提示: (1) 交互式提示,用户可以从视频中选择关键框架 (至少是第一框) 并手动提供多个目标,这些可以是提示点或对像籽粒,子,叶片和茎等物体的边界框坐标; (2) 为了进一步降低用户交互式成本,本文还提出了基于YOLOv11的自动提示方法来帮助生成目标提示信息.具体来说,对于两个作物物种,选出了九十个关键框架进行标注,器官信息被人标注器精细准确标记. 使用这个数据集,我们为两种作物进行了预训练的YOLOv11x-Seg模型,输入分辨率为640 × 640,使用SGD作为优化器,并进行了 8:2的列车/验证分割.

<a id="S041"></a>
**Source:** p.4 S041  
**Type:** body  
**Confidence:** medium

**Original:** 2.6. 3D Mask Representation and loss function 3D Mask Representation: To represent the 3D mask, we utilize a voxel grid V ∈ ℝL×W×H, where each voxel is initialized with a soft mask confidence score of zero. For a 3D voxel grid containing multiple objects, its projection onto the 2D plane is formulated as:

**中文:** 3.6 3D掩膜表示和损失函数 3D掩膜表示:为了表示3D掩膜,我们使用一个语音格 V ∈ RL×W×H,每个语音格都被初始化为软掩膜的保证率为0.对于包含多个对象的3D语音格,其对2D平面的投影是这样的:

<a id="C003"></a>
### Table 1 box coordinates while extracting their center points and polygon 3D instance statistics results instance statistics results
**Source:** p.4 C003  
**Type:** caption  
**Confidence:** high

**Original:** Table 1 box coordinates while extracting their center points and polygon 3D instance statistics results instance statistics results. vertices as prompt information. Compared to previous versions, the Dataset Plants Class Instance Seg YOLOv11 model not only achieves higher segmentation accuracy but also has faster inference speed, capable of accurately locating the centers MMR 86 Grain 3316 and boundaries of targets. Since SAM2 does not require highly precise Leaf 94 Stem 128 prompt information, providing a general segmentation area is sufficient MMW 35 Panicle 35 for prompts, so YOLOv11 is trained with very little data. Leaf

**中文:** 与前版本相比,Dataset Plants Class Instance Seg YOLOv11模型不仅实现了更高的分割准确性,而且具有更快的推断速度,能够准确地定位中心MMR 86籽粒3316和目标边界.由于SAM2不需要高精度的Leaf 94 Stem 128提示信息,提供一般分割割域足够的MMW 35 Panicle 35提示,因此YOLOv11是训练有素的数据很少.

<a id="C005"></a>
### Table 1
**Source:** p.4 C005  
**Type:** caption  
**Confidence:** high

**Original:** Table 1. 3D instance statistics results instance statistics results.

**中文:** 图1.3D实例统计结果实例统计结果.


## Page 5

<a id="S042"></a>
**Source:** p.5 S042  
**Type:** body  
**Confidence:** high

**Original:** SAM2 to generate 2D masks for the image sequence. Based on the radiance field information, a mask inverse rendering process is performed iteratively, ultimately obtaining the 3D masks.

**中文:** 基于辐射场信息,一项掩膜反转染过程被重复执行,最终得到了3D掩膜.

<a id="F003"></a>
### Fig. 3. 交互模型的管道.鉴于在米或小麦上训练的辐射场,模型首先将手动输入或YOLO提示作为输入.然后使用SAM2来生成图像序列的2D掩膜.基于辐射场的信息,一次反面染过程被执行,最终得到3
**Placed near:** p.5  
**Source:** p.5 C006  
**Crop confidence:** high

![Fig. 3](assets/fig3.png)

**Original caption:** Fig. 3. Pipeline of the interactive model. Given a radiance field trained on rice or wheat, the model first takes manual inputs or YOLO prompts as input. It then uses SAM2 to generate 2D masks for the image sequence. Based on the radiance field information, a mask inverse rendering process is performed iteratively, ultimately obtaining the 3D masks.

**中文图注:** 图3.交互模型的管道.鉴于在米或小麦上训练的辐射场,模型首先将手动输入或YOLO提示作为输入.然后使用SAM2来生成图像序列的2D掩膜.基于辐射场的信息,一次反面染过程被执行,最终得到3D掩膜.

**Reading note:** 重点查看该图如何支撑相邻正文中的方法流程、实验比较或平台应用描述。

<a id="S043"></a>
**Source:** p.5 S043  
**Type:** body  
**Confidence:** medium

**Original:** ∫ tf M(r) =

**中文:** ∫ tf M(r) =

<a id="S044"></a>
**Source:** p.5 S044  
**Type:** body  
**Confidence:** high

**Original:** w(r(t)) G(r(t)) dt; tn

**中文:** w(r(t)) G(r(t)) dt; tn

<a id="S045"></a>
**Source:** p.5 S045  
**Type:** body  
**Confidence:** high

**Original:** negative samples, resulting in excessive encroachment of the background area on the target. This can be manually controlled during manual prompting. In the case of YOLO assistance, to automatically solve the problem of positive and negative sample imbalance and further improve the accuracy of target segmentation, this paper proposes a “Positive-Negative-Positive” strategy, which makes full use of negative samples while generating more positive prompt points to balance the ratio of positive and negative samples. Specifically, for each target instance, YOLO is first used to determine its center point (cX, cY). To suppress the background area and improve the precision of the target area, the algorithm selects five adjacent center points around the target center as negative sample prompt points. Through these negative sample prompt points, the model can distinguish between the target and the background, reducing the false detection rate. Then, uniformly distributed sampling points are constructed near the center point as positive prompt points. This method adopts fixed grid sampling, where the preset grid size is 3 and the sampling range (radius) is 1. Taking the y-axis direction as an example, for the upper and lower neighborhoods of the center point, by looping through the offsets from − radius to radius (with a step size of the grid size), the OpenCV function cv2.pointPolygonTest is used to judge whether the generated sampling points are inside the target polygon. Only points that meet the conditions are retained as positive prompt points. To enhance the quality of mask generation, we design a prompting strategy that balances the ratio of positive and negative samples, which may guide the SAM2 to focus more accurately on target shapes. Residual handling: Due to factors such as illumination, occlusion, and background noise, 2D organ segmentation results often suffer from incompleteness and discontinuities, affecting the accurate extraction of subsequent 3D data. To address this issue, this paper adopts a postprocessing method based on morphological operations. As shown in

**中文:** 在YOLO辅助的情况下,为了自动解决正面和负面样本不平衡的问题,并进一步提高目标分割的精度,本文提出了一个积极-负面-积极策略,充分利用负面样本,同时产生更多正面的提示点来平衡正面和负面样本的比例.具体来说,对于每个目标实例,YOLO首先用于确定其中心点 (cX,cY).为了压制背景区域并提高目标区域的精度,算法选择了目标中心周围的五个邻近的中心点作为负面提示点. 通过这些负面样本提示点,模型可以区分目标和背景,从而降低错误检测率.然后,在中心点附近,均分布的样本点被构建为正面提示点.这种方法采用固定格式样本,预设格式大小为3个,样本范围 (半径) 为1.采用y轴方向作为一个例子,对中心点的上下邻区进行了区间,通过从−半径到半径 (含格式大小的一步) 的偏移,OpenCV函数cv2.pointPolygonTest用于判断生成的样本点是否位于目标多角体内.只有符合条件的点保留为正面提示点. 为了提高掩膜生成的质量,我们设计了一个提示策略,平衡了正面和负面样品的比例,这可能会引导SAM2更准确地关注目标形状.残留处理:由于照明,遮蔽和背景噪音等因素,2D器官分割结果经常会遭受不完整和间断,影响了随后的3D数据的准确提取.为了解决这个问题,本文采用了基于形态操作的后处理方法.如图所示,

<a id="S046"></a>
**Source:** p.5 S046  
**Type:** body  
**Confidence:** high

**Original:** application of closing and opening operations to rice grains. Specifically, algorithm 1 first processes the binarized segmentation mask using closing. This process, which involves dilation followed by erosion,

**中文:** 具体来说,算法1首先处理二进制分割面膜使用关闭.这个过程涉及扩张,随后是侵蚀,

<a id="S047"></a>
**Source:** p.5 S047  
**Type:** body  
**Confidence:** high

**Original:** (2)

**中文:** (2) (2)

<a id="S048"></a>
**Source:** p.5 S048  
**Type:** body  
**Confidence:** medium

**Original:** where r(t) denotes the ray position at parameter t, w(r(t)) is the ray weight function, and G(r(t)) is defined as: ⎡ ⎤ Vo1 (r(t)) ⎢ Vo2 (r(t)) ⎥ ⎥: G(r(t)) = ⎢ (3) ⎣⋮ ⎦ Von (r(t)) Here Voi (r(t)) represents the mask score of the i-th object at r(t). Loss Formulation: The purpose of mask inverse rendering is to project the 2D masks of the image I into the 3D space. Let MSAM2 denote the mask generated by SAM2, and Mi represent the mask of the i-th target. The projection loss is defined as: ∑∑ Lproj = − MSAM2 (r)⋅Mi (r) i n r∈R(I)

**中文:** r(t) 表示在参数t的射线位置,w(r(t)) 是射线重量函数,而G(r(t)) 定义为: Vo1 (r(t)) Vo2 (r(t)):G(r(t)) = (3) Von (r(t)) 在这里 Voi (r(t)) 表示在r(t的第一个对象的掩膜分数. 损失格式:掩膜反转的目的是将图像I的2D掩膜投射到3D空间中.让MSAM2表示SA2生成的掩膜,而Mi表示i目标的i-th.投影损失被定义为: L = MSAM2 (r⋅r) 项 (MiR) ∈ R (MiR) ∈ I

<a id="S049"></a>
**Source:** p.5 S049  
**Type:** body  
**Confidence:** medium

**Original:** ∑ ∑ [(

**中文:** [(]

<a id="S050"></a>
**Source:** p.5 S050  
**Type:** body  
**Confidence:** high

**Original:** )] 1 − MSAM2 (r) ⋅Mi (r) i

**中文:** )] 1 − MSAM2 (r) ⋅Mi (r) i

<a id="S051"></a>
**Source:** p.5 S051  
**Type:** body  
**Confidence:** high

**Original:** (4)

**中文:** (4) (4)

<a id="S052"></a>
**Source:** p.5 S052  
**Type:** body  
**Confidence:** high

**Original:** n r∈R(I)

**中文:** n r∈R(I)

<a id="S053"></a>
**Source:** p.5 S053  
**Type:** body  
**Confidence:** high

**Original:** where R(I) denotes the ray set of the image I, MSAM2 (r) is the value of i the i-th mask generated by the SAM2 model at position r, Mi(r) is the mask value of the true target at position r, and λ is a balancing weight hyperparameter. Optimization: During training, Lproj continuously updates the mask score of M(r). The converged masks are fused with ID, mask score, and color attributes to enable the export of colored target point clouds.

**中文:** R(I) 表示图像的射线集合 I,MSAM2 (r) 是 SAM2模型在r位置生成的i-th掩膜的值,Mi(r) 是位置 r的真实目标的掩膜值, λ 是平衡权重的超参数.优化:在训练期间,Lproj不断更新M(r的掩膜分数.融合的掩膜与ID,掩膜分数和颜色属性合并,使得可输出有色的目标点云.

<a id="S054"></a>
### 2.7. Auxiliary optimization strategy
**Source:** p.5 S054  
**Type:** section  
**Confidence:** high

**Original:** 2.7. Auxiliary optimization strategy

**中文:** 2.7. 辅助优化策略

<a id="S055"></a>
**Source:** p.5 S055  
**Type:** body  
**Confidence:** high

**Original:** Prompting strategy: Before SAM2 segmentation, users often click on the center of the target to obtain a positive sample prompt point and select around the target as negative sample prompt points. However, this strategy of prompt points can lead to an imbalance between positive and

**中文:** 提示策略:在SAM2分割之前,用户经常点击目标中心以获得一个积极的样本提示点,然后选择围绕目标的负面样本提示点.然而,这种提示点策略可能导致正面和之间的不平衡.


## Page 6

<a id="S056"></a>
**Source:** p.6 S056  
**Type:** body  
**Confidence:** high

**Original:** effectively fills holes within the target area and repairs discontinuities caused by incompleteness; subsequently, opening is applied to the result of the closing operation to further remove noise. This operation, which involves erosion followed by dilation, can eliminate fine noise points that may be introduced by the closing operation, thereby obtaining a smoother and more coherent segmentation result. This post-processing scheme improves the 2D segmentation accuracy of crop organs, thereby enhancing the accuracy of subsequent 3D point cloud extraction.

**中文:** 后面,对关闭操作的结果进行了开放,以进一步消除噪音.这种操作,涉及侵蚀,随后扩张,可以有效地消除关闭操作可能引入的微噪点,从而获得更平滑和更一致的分割分结果.这种后处理方案提高了作物器官的2D分割精度,从而提高了随后的3D点云采集的精度.

<a id="F004"></a>
### Fig. 4. 残差处理与基于 SSIM 的快速提示帧检测过程示意图。
**Placed near:** p.6  
**Source:** p.6 C007  
**Crop confidence:** high

![Fig. 4](assets/fig4.png)

**Original caption:** Fig. 4. Illustration of residual handling and SSIM-Based fast prompt frame detection process.

**中文图注:** 图 4. 残差处理与基于 SSIM 的快速提示帧检测过程示意图。

**Reading note:** 重点查看该图如何支撑相邻正文中的方法流程、实验比较或平台应用描述。

<a id="S057"></a>
**Source:** p.6 S057  
**Type:** body  
**Confidence:** high

**Original:** a lower resolution, to reduce the complexity of subsequent SSIM calculations. Then, parallel processing is applied to the set of images to be detected. For each image, it is first preprocessed in the same way as the reference frame, and then its horizontal mirror image is generated. Subsequently, the SSIM value sraw between the preprocessed original image and the reference frame and the SSIM value smirror between the horizontal mirror image and the reference frame are calculated separately. If the difference smirror − sraw exceeds a preset threshold, the image is considered a rear frame. Finally, by counting the indices of all images detected as rear frames, the minimum and maximum indices are taken as the positions of the first and last occurrences of rear frames, respectively. This method significantly reduces computation time by utilizing downsampling and parallel processing techniques while maintaining high detection accuracy.

**中文:** 后面,SSIM值在预处理的原始图像和参考框架之间,以及SSIM值在水平镜像和参考框架之间,分别计算.如果差异smirror −sraw超过预设门值,则图像将被视为后框.最后,通过计算所有被视为后框的图像指数,将最小和最大指数作为后框的第一和最后一次发生的位置来计算. 这种方法通过利用下样和并行处理技术,显著缩短计算时间,同时保持高检测精度.

<a id="S058"></a>
**Source:** p.6 S058  
**Type:** body  
**Confidence:** medium

**Original:** Algorithm 1. Residual handling

**中文:** 算法 1. Residual handling

<a id="S059"></a>
**Source:** p.6 S059  
**Type:** body  
**Confidence:** medium

**Original:** Input:Segmentation mask M ∈ ℝH×W

**中文:** 输入:分割掩膜M ∈ RH×W

<a id="S060"></a>
**Source:** p.6 S060  
**Type:** body  
**Confidence:** medium

**Original:** ̃ ∈ ℝH×W Output: Post-processed binary mask M for each pixel p in M do if p > 0 then p ← 255 {Set as foreground} else p ← 0 {Set as background} end if end for K ←ones(5, 5) {Define a 5 × 5 structuring element} M ← MorphologicalClose(M, K) {Closing operation: fill holes within the target} ̃ M←MorphologicalOpen(M; K) {Opening operation: remove noise}

**中文:** 【机器初译待精修】̃ ∈ ℝH×W Output: Post-processed binary mask M for each pixel p in M do if p > 0 then p ← 255 {Set as foreground} else p ← 0 {Set as background} end if end for K ←ones(5, 5) {Define a 5 × 5 structuring element} M ← MorphologicalClose(M, K) {Closing operation: fill holes within the target} ̃ M←MorphologicalOpen(M; K) {Opening operation: remove noise}

<a id="S061"></a>
**Source:** p.6 S061  
**Type:** body  
**Confidence:** medium

**Original:** Algorithm 2.

**中文:** 算法 2.

<a id="S062"></a>
**Source:** p.6 S062  
**Type:** body  
**Confidence:** high

**Original:** Find Rear-view Frames

**中文:** 查找后视图框架

<a id="S063"></a>
**Source:** p.6 S063  
**Type:** body  
**Confidence:** medium

**Original:** Input:reference frame and images Output:first and last indices pre_ref ← Preprocess(reference) files ← SortedList(images) indices ←∅ for (idx, f) ∈ files in parallel do img ← ReadImage(f) proc ← Preprocess(img) mirror ← Preprocess(HorizontalFlip(img)) s_raw ← SSIM(pre_ref, proc) s_mirror ← SSIM(pre_ref, mirror) (s_mirror − s_raw) > threshold then Add idx to indices end if end for if indices ∕ = ∅ then first ← min(indices) last ← max(indices) end if

**中文:** 输入:参考框架和图像 输出:第一和最后指标 pre_ref ← 预处理(参考) 文件 ← 排列列列表(图像) 指标 ←为 (idx, f) ∈文件并行做 img ← 阅读图像(f) proc ← 预处理(img) 镜子 ← 预处理(水平Flip(img)) s_raw ← SSIM(pre_ref, proc) s_mirror ← SSIM(pre_ref, mirror) (s_mirror − s_raw) > 门然后添加 idx到指标结束如果结束如果指标 = ← 首先 → 首先 → 首先 → 最后 → 最后 → 最后 → 最后的指标) 结束如果

<a id="S064"></a>
**Source:** p.6 S064  
**Type:** body  
**Confidence:** medium

**Original:** ̃ M=255 ̃ M← {Normalize to [0, 1]} ̃ ̃ (1; H; W)) {Adjust output dimensions} M←Reshape(M;

**中文:** 【机器初译待精修】̃ M=255 ̃ M← {Normalize to [0, 1]} ̃ ̃ (1; H; W)) {Adjust output dimensions} M←Reshape(M;

<a id="S065"></a>
**Source:** p.6 S065  
**Type:** body  
**Confidence:** high

**Original:** SSIM-Based Fast Prompt Frame Detection: Due to the selfocclusion problem, SAM2 often experiences a decrease in segmentation accuracy for targets in subsequent images when the target is occluded, i.e., when the target disappears from view, as shown in

**中文:** 基于SSIM的快速快速检测框架:由于自闭问题,SAM2在后续图像中经常会出现目标的分割准确度下降,当目标被遮蔽时,即目标从视野中消失时,如图所示.

<a id="S066"></a>
**Source:** p.6 S066  
**Type:** body  
**Confidence:** medium

**Original:** the segmentation accuracy of the model, it is necessary to provide prompts for rear frames. Therefore, to identify rear frames and improve detection efficiency under a large amount of image data, this paper proposes a fast decision algorithm based on SSIM and parallel computing. Specifically, Algorithm 2 first preprocesses the front reference frame, including converting it to a grayscale image and scaling it to

**中文:** 因此,为了识别后框和提高大量图像数据的检测效率,本文提出了一个基于SSIM和平行计算的快速决策算法.具体来说,算法2首先预处理前框,包括将其转换为灰色图像,并将其扩展到.


## Page 7

<a id="S067"></a>
### 2.8. Phenotypic data extraction method
**Source:** p.7 S067  
**Type:** section  
**Confidence:** high

**Original:** 2.8. Phenotypic data extraction method

**中文:** 2.8. 表型数据提取方法

<a id="S068"></a>
**Source:** p.7 S068  
**Type:** body  
**Confidence:** medium

**Original:** between the vector formed by the next base point and bk and the principal component axis) ((qj − bk)⊤ v1 ⃦ θj = arccos ⃦ (11) ⃦qj − bk ‖ ⋅⃦v1 ‖ < θmax;

**中文:** 接下来的基点和bk和主要组件轴之间的向量) ((qj − bk) v1 θj = arccos (11) qj − bk ‖ ⋅v1 ‖ < θmax;

<a id="S069"></a>
**Source:** p.7 S069  
**Type:** body  
**Confidence:** high

**Original:** To accurately compute the phenotypic results of target organs, it is necessary to preprocess the point cloud extracted by the model. This specifically includes setting the voxel size to 0.01, performing voxel downsampling, and noise reduction; using the Alpha Shapes algorithm for initial mesh construction, followed by hole filling and loop subdivision on the mesh to generate multiple triangular facets. Given the sparse, irregular, or incomplete nature of organs, Alpha Shapes algorithm constructs a topologically consistent triangular mesh surface. Its key advantage lies in its ability to generate biologically plausible surfaces even in regions with significant variations in point density.

**中文:** 为了准确计算目标器官的表型结果,必须先处理模表型提取的点云.这包括设置体素为0.01,执行体素下样和噪音降低;使用Alpha Shapes算法进行初始网格构建,然后在网格上填孔和循环分割,产生多个三角形面.鉴于器官的稀疏,不规则或不完整性,Alpha Shapes算法构建了一个多学一致的三角形网格表面.其关键优势在于它能够产生生物可行的表面,即使在有明显的点密度变化地区.

<a id="S070"></a>
**Source:** p.7 S070  
**Type:** body  
**Confidence:** medium

**Original:** where θmax = π/2 is the maximum allowable deviation angle. If θj exceeds the maximum angle π/2, skip it; otherwise, select it. If no points meet the criteria, return the point with the smallest angle. s4: Termination conditions:

**中文:** 如果 θj超过最大的角度 π/2,则跳过;否则,选择它.如果没有点符合标准,则返回最小的角度的点. s4:终止条件:

<a id="S071"></a>
**Source:** p.7 S071  
**Type:** body  
**Confidence:** high

**Original:** where ϵ is the positional tolerance and Mmax is the maximum number of base points. If enough base points are found or the current base point coincides with the endpoint, the loop ends.

**中文:** 如果找到足够的基点或当前的基点与终点一致,循环结束.

<a id="S072"></a>
**Source:** p.7 S072  
**Type:** body  
**Confidence:** medium

**Original:** s5: Downsample the collected base point set B = {bm }M m=1 to obtain a set of base points along the curved path of the leaf. Summing the distances between consecutive points gives the total length of the leaf. M− 1 ∑

**中文:** 【机器初译待精修】s5: Downsample the collected base point set B = {bm }M m=1 to obtain a set of base points along the curved path of the leaf. Summing the distances between consecutive points gives the total length of the leaf. M− 1 ∑

<a id="S073"></a>
**Source:** p.7 S073  
**Type:** body  
**Confidence:** high

**Original:** (5)

**中文:** (5) (5)

<a id="S074"></a>
**Source:** p.7 S074  
**Type:** body  
**Confidence:** medium

**Original:** The leaf width is calculated based on a mesh parameterization method that flattens the leaf in the width direction, followed by 2D interpolation. The interval width is defined as the distance between boundary points of each row, and the length of the widest interval represents the leaf width. Specifically, define the triangular mesh: M = (V;E;F), where V = {vi ∈ ℝ3 } is the set of vertices, E is the set of edges, and F is the set of facets. s1: Mesh parameterization allows the transformation of the leaf's 3D mesh into a 2D space by minimizing energy loss. This paper employs the As-Rigid-As-Possible (ARAP) algorithm to create mapping connections between mesh vertices in both 3D and parameter spaces. The re-meshed 2D leaf grid maintains a one-to-one mapping relationship and shares the topological structure of the 3D mesh model vertices, enabling mutual retrieval based on the mapping relationship. To solve the mapping ϕ: V→ℝ2, minimize the energy: ∑ EARAP = wij ‖ϕ(vi) − ϕ(vj) − Ri (vi − vj)‖22; (14)

**中文:** 【机器初译待精修】The leaf width is calculated based on a mesh parameterization method that flattens the leaf in the width direction, followed by 2D interpolation. The interval width is defined as the distance between boundary points of each row, and the length of the widest interval represents the leaf width. Specifically, define the triangular mesh: M = (V;E;F), where V = {vi ∈ ℝ3 } is the set of vertices, E is the set of edges, and F is the set of facets. s1: Mesh parameterization allows the transformation of the leaf's 3D mesh into a 2D space by minimizing energy loss. This paper employs the As-Rigid-As-Possible (ARAP) algorithm to create mapping connections between mesh vertices in both 3D and parameter spaces. The re-meshed 2D leaf grid maintains a one-to-one mapping relationship and shares the topological structure of the 3D mesh model vertices, enabling mutual retrieval based on the mapping relationship. To solve the mapping ϕ: V→ℝ2, minimize the energy: ∑ EARAP = wij ‖ϕ(vi) − ϕ(vj) − Ri (vi − vj)‖22; (14)

<a id="S075"></a>
**Source:** p.7 S075  
**Type:** body  
**Confidence:** high

**Original:** (6)

**中文:** (6)

<a id="S076"></a>
**Source:** p.7 S076  
**Type:** body  
**Confidence:** medium

**Original:** b = v1 − v3:

**中文:** b = v1 − v3:

<a id="S077"></a>
**Source:** p.7 S077  
**Type:** body  
**Confidence:** medium

**Original:** Calculate the cross product of vectors a and b to obtain a vector c = a × b that is perpendicular to the plane of the triangle. Calculate half the magnitude of the cross product as the area of the triangle: (7)

**中文:** 计算a和b向量的交叉乘法,以获得向量c =a × b垂直于三角形的平面.计算半个交叉乘法的大小为三角形的面积: (7)

<a id="S078"></a>
**Source:** p.7 S078  
**Type:** body  
**Confidence:** medium

**Original:** A△ = 0:5 × ‖c‖:

**中文:** △ = 0:5 × ‖c‖: A

<a id="S079"></a>
**Source:** p.7 S079  
**Type:** body  
**Confidence:** high

**Original:** (13)

**中文:** (13) (13)

<a id="S080"></a>
**Source:** p.7 S080  
**Type:** body  
**Confidence:** medium

**Original:** ‖bm+1 − bm ‖2: m=1

**中文:** ‖bm+1 − bm ‖2:m=1

<a id="S081"></a>
### 2.8.2. Leaf surface area
**Source:** p.7 S081  
**Type:** section  
**Confidence:** high

**Original:** 2.8.2. Leaf surface area

**中文:** 2.8.2. 叶片表面积

<a id="S082"></a>
**Source:** p.7 S082  
**Type:** body  
**Confidence:** medium

**Original:** The mesh consists of a series of triangular facets, and the leaf area is approximated by calculating the sum of the areas of all triangles. The three vertices of each triangle are represented as vectors v1, v2, v3. The two edge vectors are a = v1 − v2;

**中文:** 网格由一系列三角形面积组成,叶片面积通过计算所有三角形面积的总和来进行近似计算.每个三角形的三个顶点都以向量v1,v2,v3表示.两个边缘向量是a =v1 −v2;

<a id="S083"></a>
**Source:** p.7 S083  
**Type:** body  
**Confidence:** high

**Original:** (12)

**中文:** (12) (12)

<a id="S084"></a>
**Source:** p.7 S084  
**Type:** body  
**Confidence:** high

**Original:** ‖bk − et ‖2 > ϵ and k < Mmax;

**中文:** ‖bk − 和 ‖2 > ε 和 k < Mmax;

<a id="S085"></a>
### 2.8.1. Voxel volumes of grains and panicles
**Source:** p.7 S085  
**Type:** section  
**Confidence:** high

**Original:** 2.8.1. Voxel volumes of grains and panicles

**中文:** 2.8.1. 籽粒与穗部的体素体积

<a id="S086"></a>
**Source:** p.7 S086  
**Type:** body  
**Confidence:** medium

**Original:** Calculate the voxel volume V using the following formula based on the voxel size and the number of voxel units num_voxels in the point cloud, to obtain the volumes of grains and panicles, facilitating subsequent indicator calculations. Voxels directly represent occupancy, enabling precise volume calculations without reconstruction ambiguity. Due to the absence of ground truth for real-world volumes, all volumetric analyses were performed within the voxel space. V = num_voxels*0:013

**中文:** 根据体素大小和点云中的体素单位num_voxels数量的公式计算出体素大小V,以获得颗粒和穗部的体积,从而促进随后的指标计算.体素直接代表占用量,从而使得没有重建模糊的精确的体积计算.由于现实世界卷积没有地面真相,所有的体积分析都是在体素空间内进行的.V = num_voxels*0:013

<a id="S087"></a>
**Source:** p.7 S087  
**Type:** body  
**Confidence:** medium

**Original:** Sum the areas of all triangles to obtain the surface area of the mesh: ∑ Amesh = A△: (8)

**中文:** 总算所有三角形的面积,以获得网格的表面面积: Amesh = A△: (8)

<a id="S088"></a>
**Source:** p.7 S088  
**Type:** body  
**Confidence:** high

**Original:** (i;j)∈E

**中文:** (i;j)∈E

<a id="S089"></a>
### 2.8.3. Leaf length and width
**Source:** p.7 S089  
**Type:** section  
**Confidence:** high

**Original:** 2.8.3. Leaf length and width

**中文:** 2.8.3. 叶长与叶宽

<a id="S090"></a>
**Source:** p.7 S090  
**Type:** body  
**Confidence:** high

**Original:** To calculate the length, a point set is extracted based on the nearest neighbor algorithm to fit the midrib of the leaf. The length of the curve formed by the point set represents the leaf length. Specifically, s1: Let the

**中文:** 为了计算长度,根据最近的邻居算法提取一个点集,以适应叶子的半叶.由点集形成的曲线长度代表叶子长度.具体来说,s1:让

<a id="S091"></a>
**Source:** p.7 S091  
**Type:** body  
**Confidence:** medium

**Original:** where Ri ∈ SO(2) is the local rotation matrix, and wij is the cotangent weight. s2: Let the parameterized coordinates be {ui = (ui; vi)⊤ }. Calculate the principal component direction of the parameterized coordinates:

**中文:** 里 ∈ SO(2) 是当地的旋转矩阵, wij 是对角质量. s2:让参数坐标是 {ui = (ui; vi) }.计算参数坐标的主要组件方向:

<a id="S092"></a>
**Source:** p.7 S092  
**Type:** body  
**Confidence:** medium

**Original:** preprocessed leaf point cloud be P = {pi ∈ ℝ3 }i=1, where pi =

**中文:** 预处理的叶片点云是P = {pi ∈ R3 }i=1,而 pi =

<a id="S093"></a>
**Source:** p.7 S093  
**Type:** body  
**Confidence:** medium

**Original:** (xi; yi; zi)⊤. Perform principal component analysis (PCA) on it and calculate the covariance matrix: C=

**中文:** 执行主要组件分析 (PCA) 在它上,并计算了对差矩阵:C=

<a id="S094"></a>
**Source:** p.7 S094  
**Type:** body  
**Confidence:** medium

**Original:** N 1 ∑ (p − p)(pi − p)⊤; N i=1 i

**中文:** (p − p) (pi − p);N i=1 i

<a id="S095"></a>
**Source:** p.7 S095  
**Type:** body  
**Confidence:** high

**Original:** (15)

**中文:** (15) (15)

<a id="S096"></a>
**Source:** p.7 S096  
**Type:** body  
**Confidence:** medium

**Original:** w = arg max Var({u⊤ i a}): ‖a‖=1

**中文:** w = arg max Var({u i a}): ‖a‖=1

<a id="S097"></a>
**Source:** p.7 S097  
**Type:** body  
**Confidence:** high

**Original:** s3: Perform 2D point interpolation. Define sampling intervals on the main axis direction w:

**中文:** 定义主要轴方向的样本取样间隔 w:

<a id="S098"></a>
**Source:** p.7 S098  
**Type:** body  
**Confidence:** high

**Original:** (9)

**中文:** (9) (9)

<a id="S099"></a>
**Source:** p.7 S099  
**Type:** body  
**Confidence:** medium

**Original:** k sk = smin + (smax − smin); n

**中文:** k = smin + (smax − smin);n

<a id="S100"></a>
**Source:** p.7 S100  
**Type:** body  
**Confidence:** medium

**Original:** where p is the centroid of the point cloud. Through eigenvalue decomposition C = VΛV⊤, the main direction v1 (corresponding to the largest eigenvalue λ1) is obtained. Treat the two farthest endpoints found on the first principal component axis as the starting and ending points for calculating the leaf length. { es = argminp∈P p⊤ v1 (10) et = argmaxp∈P p⊤ v1

**中文:** 通过自值分解C = VΛV,得到了主方向v1 (相应于最大的自值 λ1).将第一个主要组件轴上发现的两个最远的终点视为计算叶子长度的起点和结尾点. { es = argminp∈P p v1 (10) et = argmaxp∈P p v1.

<a id="S101"></a>
**Source:** p.7 S101  
**Type:** body  
**Confidence:** medium

**Original:** k = 0; 1; …; n

**中文:** k = 0; 1;...; n

<a id="S102"></a>
**Source:** p.7 S102  
**Type:** body  
**Confidence:** high

**Original:** (16)

**中文:** (16) (16)

<a id="S103"></a>
**Source:** p.7 S103  
**Type:** body  
**Confidence:** medium

**Original:** ⊤ where smin = mini u⊤ i w, smax = maxi ui w. For each sampling position sk,

**中文:** 在哪里smin = mini u i w, smax = maxi ui w.对于每一个采样位置sk,

<a id="S104"></a>
**Source:** p.7 S104  
**Type:** body  
**Confidence:** medium

**Original:** search for boundaries along the normal direction w⊥ = [− wy; wx]⊤: { l ⊥ tk = minu⊤i w=sk u⊤ i w (17) ⊥ trk = maxu⊤i w=sk u⊤ i w

**中文:** 寻找正常方向边界 w = [− wy; wx]: { l tk = minui w=sk u i w (17) trk = maxui w=sk u i w

<a id="S105"></a>
**Source:** p.7 S105  
**Type:** body  
**Confidence:** medium

**Original:** s4: Calculate the leaf width: For each row of interpolated boundary points, compute the Euclidean distance in the width direction Wk = trk − tlk. Take the maximum value across all rows as the leaf width

**中文:** s4:计算叶子宽度:对于每个连接边界点的行列,计算在宽度方向Wk = trk − tlk的尤克利德距离.将所有行列的最大值作为叶子宽度.

<a id="S106"></a>
**Source:** p.7 S106  
**Type:** body  
**Confidence:** medium

**Original:** s2: Build a KD-tree T = BuildKDTree(P). Starting from the starting point, perform K-nearest neighbor (KNN) search. Sort the K candidate points in a greedy manner based on their distance from the starting point, and execute the loop process until the preset conditions are met. s3: Let the current base point be bk, and the candidate point set be

**中文:** 建立一个KD树T = 建立KDTree(P).从起点开始,执行K-近邻 (KNN) 搜索.根据起点距离的贪方式排序K候选点,并执行循环过程,直到预设条件达到.

<a id="S107"></a>
**Source:** p.7 S107  
**Type:** body  
**Confidence:** medium

**Original:** W = max Wk: 1≤k≤n

**中文:** 澳大利亚: 1≤k≤n

<a id="S108"></a>
**Source:** p.7 S108  
**Type:** body  
**Confidence:** medium

**Original:** N k = {qj }Kj=1, satisfying N k = KNN(T; bk; K). Calculate the angle

**中文:** N k = {qj }Kj=1,满足N k = KNN(T; bk; K).计算角度

<a id="S109"></a>
**Source:** p.7 S109  
**Type:** body  
**Confidence:** high

**Original:** (18)

**中文:** (18) (18)


## Page 8

<a id="S110"></a>
### 2.9. Evaluation metrics
**Source:** p.8 S110  
**Type:** section  
**Confidence:** high

**Original:** 2.9. Evaluation metrics

**中文:** 2.9. 评价指标

<a id="S111"></a>
**Source:** p.8 S111  
**Type:** body  
**Confidence:** high

**Original:** y i is the predicted value, and n is the total where yi is the true value, ̂ number of samples. MAE directly calculates the average of the absolute differences between the predicted and true values. It is suitable for scenarios with noisy data or uniformly distributed errors. The formula is:

**中文:** 预测值为y,且总数为n,其中yi是真实值,样本数量.MAE直接计算了预测值和真实值之间的绝对差异的平均值.它适合有噪音数据或均分布错误的场景.公式是:

<a id="S112"></a>
**Source:** p.8 S112  
**Type:** body  
**Confidence:** medium

**Original:** Common 3D segmentation evaluation metrics were used: IoU, Precision, Recall, F1-score, and average inference time. For each instance, IoU is calculated by computing the intersection of the predicted region with the true region for that instance, then dividing it by their union. It measures the overlap between the predicted instance region and the true instance region, ranging from [0, 1], with higher values indicating closer alignment between prediction and truth. mIoU represents the ability to correctly segment 3D point clouds into various categories. The formulas are as follows: IoUi =

**中文:** 采用了常见的3D分类评估指标:IoU,精确性,回忆,F1分数和平均推断时间.对于每个实例,IoU是通过计算预测区域与实例的真实区域的交叉路口计算,然后将其分为它们的联盟来计算的.它测量了预测实例区域和真实实实例区域之间的重叠,从 [0, 1]开始,更高的值表明预测和真相之间更紧密的对应.mIoU代表了正确分类3D点云的能力.公式如下:IoUi =

<a id="S113"></a>
**Source:** p.8 S113  
**Type:** body  
**Confidence:** high

**Original:** |Pi ∩ Ti | |Pi ∪ Ti |

**中文:** 您的确可以看到.

<a id="S114"></a>
**Source:** p.8 S114  
**Type:** body  
**Confidence:** medium

**Original:** MAE =

**中文:** 马E =

<a id="S115"></a>
**Source:** p.8 S115  
**Type:** body  
**Confidence:** high

**Original:** (19)

**中文:** (19) (19)

<a id="S116"></a>
### 2.10. Experimental details
**Source:** p.8 S116  
**Type:** section  
**Confidence:** high

**Original:** 2.10. Experimental details

**中文:** 【标题暂译】2.10. Experimental details

<a id="S117"></a>
**Source:** p.8 S117  
**Type:** body  
**Confidence:** high

**Original:** (20)

**中文:** (20) (20)

<a id="S118"></a>
**Source:** p.8 S118  
**Type:** body  
**Confidence:** high

**Original:** To avoid the impact of different experimental conditions on the model, all experiments in this study were conducted under the same environment. The software and hardware configurations of the platform are shown in Table S4. Nerfacto iterates 30,000 steps to build the radiance field. During the segmentation process, the SGD optimizer is used with a learning rate of 0.1. The single-view mask is rendered in chunks. To evaluate rendering performance, we measured the inference time of the interactive model on a single-target scene with ray batch sizes ranging from 211 to 219(Table S5). The results demonstrate that increasing the ray batch size optimizes VRAM management mechanisms—by enhancing GPU parallel computing resource utilization, it significantly reduces per-ray processing overhead, thereby decreasing total inference time. However, when the batch size exceeds 214, resource contention caused by VRAM bandwidth saturation leads to performance degradation. Batches reaching 219 trigger Out-Of-Memory (OOM) errors. Consequently, we configure 214 rays per chunk to maximize computational efficiency while avoiding hardware limitations. To maximize the accuracy of point cloud extraction, this experiment employs the recently released sam2.1_hiera_large model. Given that the core objective of this paper is to reduce users’ time costs, and considering that data annotation and fine-tuning of large models are timeconsuming tasks, no fine-tuning was performed on the data in the experiment.

**中文:** 为了避免不同实验条件对模型的影响,本研究的所有实验都是在同一环境下进行的.平台的软件和硬件配置如图S4表.NeRFacto反复完成3万步的创建辐射场.在分割过程中,SGD优化器使用了学习率为0.1.单视掩膜被分成碎片.为了评估染性能,我们测量了交互式模型在单目标场景上推断时间,射线批量从211到219的射线批量.表S5表.结果表明,增加射线批量优化VRAM管理机制,通过提高 GPU平行计算资源利用,它显著减少了每射线处理的总推断时间,从而减少了总线处理时间. 然而,当批量大小超过214时,由于VRAM带宽和度量增加导致的资源纠纷导致性能下降.达到219的批量触发出内存 (OOM) 错误.因此,我们配置每块的214光线以最大限度地提高计算效率,同时避免硬件限制.为了最大限度地提高点云提取的精度,这项实验采用了最近发布的 SAM2.1_hiera_large模型.鉴于本文的目标是降低核心用户时间成本,并且考虑到数据标注和对大型模型进行细调是耗时间的任务,实验中没有进行细调的数据.

<a id="S119"></a>
**Source:** p.8 S119  
**Type:** body  
**Confidence:** medium

**Original:** where N is the total number of categories. Precision represents the proportion of true positive samples among those predicted as positive, indicating the reliability of the prediction results. Precisioni =

**中文:** 在此,N是类别的总数.精度表示预测中正确的正确样本比例,表明预测结果的可靠性.

<a id="S120"></a>
**Source:** p.8 S120  
**Type:** body  
**Confidence:** high

**Original:** |Pi ∩ Ti | |Pi |

**中文:** 您的确很难忘了.

<a id="S121"></a>
**Source:** p.8 S121  
**Type:** body  
**Confidence:** high

**Original:** (21)

**中文:** (21)

<a id="S122"></a>
**Source:** p.8 S122  
**Type:** body  
**Confidence:** medium

**Original:** Recall represents the proportion of the true region that is covered by the prediction. A higher value indicates fewer missed detections. Recalli =

**中文:** 召回表示预测所涵盖的真实区域的比例.较高的值表明更少的错过检测.召回 =

<a id="S123"></a>
**Source:** p.8 S123  
**Type:** body  
**Confidence:** high

**Original:** |Pi ∩ Ti | |Ti |

**中文:** 您的位置: 首页 首页 首页

<a id="S124"></a>
**Source:** p.8 S124  
**Type:** body  
**Confidence:** high

**Original:** (22)

**中文:** (22) (22)

<a id="S125"></a>
**Source:** p.8 S125  
**Type:** body  
**Confidence:** medium

**Original:** F1-score is the harmonic mean of precision and recall, providing a comprehensive measure of model performance. A higher value indicates better overall performance of the model. F1i = 2 ×

**中文:** 标准F1分数是精度和召回的和平均值,提供了模型性能的全面衡量量.一个更高的值表明模型的整体性能更好.F1i = 2 ×

<a id="S126"></a>
**Source:** p.8 S126  
**Type:** body  
**Confidence:** high

**Original:** Precisioni × Recalli Precisioni + Recalli

**中文:** 精确 × 复习精确 + 复习

<a id="S127"></a>
**Source:** p.8 S127  
**Type:** body  
**Confidence:** high

**Original:** (23)

**中文:** (23)

<a id="S128"></a>
**Source:** p.8 S128  
**Type:** body  
**Confidence:** medium

**Original:** Average inference time is a common speed metric used to evaluate model performance, representing the average time consumed by the model for one instance segmentation prediction. Assume that the model makes predictions for M samples. The formula for calculating the average inference time is: Tavg =

**中文:** 平均推断时间是用来评估模型性能的常见速度指标,它代表了模型对一个实例分割预测所花费的平均时间.假设模型对M样本进行了预测.计算平均推断时间的公式是:Tavg =

<a id="S129"></a>
**Source:** p.8 S129  
**Type:** body  
**Confidence:** medium

**Original:** M 1 ∑ Tj M j=1

**中文:** M 1 Tj M j=1

<a id="S130"></a>
**Source:** p.8 S130  
**Type:** body  
**Confidence:** high

**Original:** (26)

**中文:** (26)

<a id="S131"></a>
**Source:** p.8 S131  
**Type:** body  
**Confidence:** medium

**Original:** R2 evaluates the goodness of fit of the model by explaining the proportion of variance in the dependent variable. Its range is [0,1], and a value closer to 1 indicates better model fit. The definition is: ∑n (yi − ̂ y i)2 R2 = 1 − ∑i=1 (27)

**中文:** R2通过解释依赖变量中的差异比例来评估模型的合适性.它的范围为 [0,1],一个接近1的值表明模型更好.定义是:n (yi − ̂ y i) 2 R2 = 1 − i=1 (27)

<a id="S132"></a>
**Source:** p.8 S132  
**Type:** body  
**Confidence:** medium

**Original:** n i=1 (yi − y)

**中文:** n=1 (yi − y)

<a id="S133"></a>
**Source:** p.8 S133  
**Type:** body  
**Confidence:** medium

**Original:** where: Pi is the region predicted by the model for the ith instance, and Ti is its true region. N 1 ∑ mIoU = IoUi N i=1

**中文:** 在哪里:Pi是模型预测的第一个实例区域,Ti是其真实区域.N 1 mIoU = IoUi N i=1

<a id="S134"></a>
**Source:** p.8 S134  
**Type:** body  
**Confidence:** medium

**Original:** n 1∑ |yi − ̂ yi| n i=1

**中文:** 1 yi − n yi n i=1

<a id="S135"></a>
### 3. Results
**Source:** p.8 S135  
**Type:** section  
**Confidence:** high

**Original:** 3. Results

**中文:** 【标题暂译】3. Results

<a id="S136"></a>
**Source:** p.8 S136  
**Type:** body  
**Confidence:** high

**Original:** This section quantitatively evaluates the segmentation performance of IPENS on MMR and MMW, and analyzes the phenotypic results. It further demonstrates that IPENS can perform high-quality instance segmentation of multiple rice or wheat organs in an unsupervised interactive manner.

**中文:** 本节量化评估了IPENS在MMR和MMW上的分割性能,并分析了表型结果,进一步证明IPENS可以在无监督的交互式方式进行多种米或小麦器官的高质量实例分割.

<a id="S137"></a>
**Source:** p.8 S137  
**Type:** body  
**Confidence:** high

**Original:** (24)

**中文:** (24) 个月的时间

<a id="S138"></a>
**Source:** p.8 S138  
**Type:** body  
**Confidence:** medium

**Original:** where Tj is the inference time for the jth prediction. To evaluate the phenotypic quality of the segmented targets, we compared the predicted results with the ground truth. Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and R-squared (R2) were used. These metrics reflect the deviation between the model's predictions and the true values from different perspectives, as well as the goodness of fit of the model. They provide important references for model optimization and comparison. RMSE measures the overall prediction error of the model by calculating the square root of the average of the squared residuals between the predicted and true values. RMSE is sensitive to outliers, as larger errors are significantly magnified due to the squaring operation. The formula is: √̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅ n 1∑ RMSE = (yi − ̂ y i)2 (25) n i=1

**中文:** 为了评估分割目标的表型质量,我们将预测结果与基础真相进行了比较.使用了根平均正方形错误 (RMSE),平均绝对错误 (MAE) 和R-squared (R2).这些指标反映了模型的预测和实际值之间的偏差,以及模型的合适性.它们为模型优化和比较提供了重要的参考.RMSE通过计算预测和值之间的平均正方形残余的正方形根来衡量模型的整体预测错误.RMSE对外观值非常敏感,因为更大的错误因显著的正方形操作而被平方化. 公式是: 的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的

<a id="S139"></a>
### 3.1. Quantitative experiment
**Source:** p.8 S139  
**Type:** section  
**Confidence:** high

**Original:** 3.1. Quantitative experiment

**中文:** 3.1. 定量实验

<a id="S140"></a>
### 3.1.1. The effectiveness of IPENS
**Source:** p.8 S140  
**Type:** section  
**Confidence:** high

**Original:** 3.1.1. The effectiveness of IPENS

**中文:** 【标题暂译】3.1.1. The effectiveness of IPENS

<a id="S141"></a>
**Source:** p.8 S141  
**Type:** body  
**Confidence:** high

**Original:** In this study, to verify the effectiveness of IPENS, 30% of the data set was taken as the validation set, and the remaining part was provided as the training set for the comparison algorithm training. To demonstrate the segmentation capability of IPENS, this experiment will take the crop organs observable in the first frame for testing. Prompting is performed on the first frame and the rear frame of the video, manually prompting 2 positive samples and 2 negative sample prompt points, which are submitted to the model for 3D segmentation. The IoU, precision, recall, and F1 score indicators are quantitatively evaluated according to the ground truth in the data set. The results of the model on the MMR and MMW datasets are visualized in Fig. 5a and b, respectively. In terms of the IoU index, the segmentation accuracy on MMR for Grain, Leaf, and Stem is 61.48%,

**中文:** 在本研究中,为了验证IPENS的有效性,30%的数据集被作为验证集,其余部分被作为比较算法训练的训练集.为了证明IPENS的分割能力,该实验将在第一个框架中可观测的作物器官进行测试.在视频的第一框和后框上进行调试,手动调试2个正面样本和2个负面样本提示点,这些数据集被提交给模型进行3D分割.根据数据集中的地面进行量化评估,IoU,精度,召回和F1分数指标.MMR和MMW数据集的模型结果分别在图5a和b中可视化. 对于IoU指数来说,籽粒,叶子和茎的MMR的分割准确性为61.48%,

<a id="F005"></a>
### Fig. 5. (a) 水稻器官各评价指标的雷达图。(b) 小麦器官各评价指标的雷达图。
**Placed near:** p.8 S141  
**Source:** p.9 C010  
**Crop confidence:** high

![Fig. 5](assets/fig5.png)

**Original caption:** Fig. 5. (a) Radar chart of evaluation metrics across rice organs. (b) Radar chart of evaluation metrics across wheat organs.

**中文图注:** 图 5. (a) 水稻器官各评价指标的雷达图。(b) 小麦器官各评价指标的雷达图。

**Reading note:** 重点查看该图如何支撑相邻正文中的方法流程、实验比较或平台应用描述。


## Page 9

<a id="S142"></a>
**Source:** p.9 S142  
**Type:** body  
**Confidence:** high

**Original:** 69.54%, and 60.13%. On MMW, the IoU for Panicle, Leaf, and Stem is 92.82%, 86.47%, and 89.76%, respectively. Tables 2a and 2b presents the performance comparison results of mainstream deep learning-based instance segmentation algorithms. Specifically, The unsupervised method CrossPoint [39] is a representative algorithm for current unsupervised instance segmentation. The interactive method Agile3D [40] is the latest supervised SOTA model, where the number after the “@” symbol indicates the number of user clicks required for 3D point cloud segmentation (the accuracy of the model usually increases with the number of interactions). To maintain consistency in comparison benchmarks, this paper only compares its first interaction results. The fully supervised method oneformer3D [41] is currently the best fully supervised model, and its results can be regarded as the ideal upper limit for this task. This hierarchical comparison architecture clearly shows the performance differences of algorithms under different supervision paradigms. The results show that oneformer3D performs optimally on both data sets (mIoU: Rice 78.18%/Wheat 97.52%), reflecting its theoretical advantage of full supervision with sufficient labeled data. IPENS is significantly better than the unsupervised algorithm CrossPoint on both crops (mIoU: Rice 23.41%/Wheat 16.50%), demonstrating the effectiveness in the absence of labels. Compared with the first interaction results of Agile3D (mIoU: Rice 55.82%/Wheat 50.68%), our approach delivers better overall results, and it's not hard to see that Agile3D performs poorly on stems. This indicates that the unsupervised interactive algorithm IPENS can perform at the mainstream level on rice and wheat data sets, demonstrating the feasibility of applying unlabeled data in agricultural scenarios.

**中文:** 在MMW上,Panicle,Leaf和Stem的IoU分别为92.82%,86.47%,89.76%. 2a和2b表表述了主流基于深度学习的实例分割算法的性能比较结果.具体来说,未监督的方法CrossPoint [39]是目前未监督的实例分割算法的代表性算法. 交互式方法Agile3D [40]是最新的监督的SOTA模型,在@符号之后的数字表示需要3D点云分割的用户点击数量 (模型的准确性通常随着交互式数量增加).为了保持比较基准的一致性,本文仅将其第一次交互式结果进行比较. 完全监督方法 oneformer3D [41]目前是最佳完全监督模型,其结果可以被视为该任务的理想上限.这种等级比较架构清楚地显示了监督模式下的算法的性能差异.结果表明, oneformer3D在两个数据集上表现得最好 (mIoU:水稻 78.18%/麦 97.52%),反映了它对充分标签数据进行全面监督的理论优势.IPENS在两个作物上显著优于非监督的CrossPoint算法 (mIoU:水稻 23.41%/麦 16.50%),证明在没有标签的情况下效果. 与Agile3D的首次交互式结果相比 (mIoU:水稻 55.82%/小麦 50.68%),我们的方法总体带来了更好的结果,并且很容易看到Agile3D在茎上表现得不佳.这表明IPENS无监督的交互式算法可以在水稻和小麦数据集中表现得很好,证明了在农业场景中应用未标记数据的可行性.

<a id="S143"></a>
**Source:** p.9 S143  
**Type:** body  
**Confidence:** high

**Original:** Comparison of mainstream segmentation methods on the MMR and MMW datasets. (a) MMR Dataset Method CrossPoint Agile3D @1 oneformer3D IPENS

**中文:** 在MMR和MMW数据集中进行主流分割方法的比较. (a)MMR数据集方法CrossPoint Agile3D @1 oneformer3D IPENS

<a id="S144"></a>
**Source:** p.9 S144  
**Type:** body  
**Confidence:** high

**Original:** IoU Per Category (%)

**中文:** 按类别 (%) 的IoU

<a id="S145"></a>
**Source:** p.9 S145  
**Type:** body  
**Confidence:** high

**Original:** mIoU

**中文:** 现在我已经知道了.

<a id="S146"></a>
**Source:** p.9 S146  
**Type:** body  
**Confidence:** high

**Original:** Grain

**中文:** 籽粒

<a id="S147"></a>
**Source:** p.9 S147  
**Type:** body  
**Confidence:** high

**Original:** Leaf

**中文:** 叶

<a id="S148"></a>
**Source:** p.9 S148  
**Type:** body  
**Confidence:** high

**Original:** Stem

**中文:** 语音

<a id="S149"></a>
**Source:** p.9 S149  
**Type:** body  
**Confidence:** high

**Original:** 39.64 69.60 79.81 61.48

**中文:** 39.64 69.60 79.81 61.48

<a id="S150"></a>
**Source:** p.9 S150  
**Type:** body  
**Confidence:** high

**Original:** 18.13 70.60 77.55 69.54

**中文:** 18.13 70.60 77.55 69.54

<a id="S151"></a>
**Source:** p.9 S151  
**Type:** body  
**Confidence:** high

**Original:** 12.46 27.25 77.18 60.13

**中文:** 12.46 27.25 77.18 60.13

<a id="S152"></a>
**Source:** p.9 S152  
**Type:** body  
**Confidence:** high

**Original:** 23.41 55.82 78.18 63.72

**中文:** 23.41 55.82 78.18 63.72

<a id="S153"></a>
**Source:** p.9 S153  
**Type:** body  
**Confidence:** high

**Original:** (b) MMW Dataset Method CrossPoint Agile3D @1 oneformer3D IPENS

**中文:** (b) MMW数据集方法CrossPoint Agile3D @1 oneformer3D IPENS

<a id="S154"></a>
**Source:** p.9 S154  
**Type:** body  
**Confidence:** high

**Original:** IoU Per Category (%)

**中文:** 按类别 (%) 的IoU

<a id="S155"></a>
**Source:** p.9 S155  
**Type:** body  
**Confidence:** high

**Original:** mIoU

**中文:** 现在我已经知道了.

<a id="S156"></a>
**Source:** p.9 S156  
**Type:** body  
**Confidence:** high

**Original:** Panicle

**中文:** 穗部

<a id="S157"></a>
**Source:** p.9 S157  
**Type:** body  
**Confidence:** high

**Original:** Leaf

**中文:** 叶

<a id="S158"></a>
**Source:** p.9 S158  
**Type:** body  
**Confidence:** high

**Original:** Stem

**中文:** 语音

<a id="S159"></a>
**Source:** p.9 S159  
**Type:** body  
**Confidence:** high

**Original:** 27.79 50.51 99.26 92.82

**中文:** 27.79 50.51 99.26 92.82

<a id="S160"></a>
**Source:** p.9 S160  
**Type:** body  
**Confidence:** high

**Original:** 12.63 78.74 98.16 86.47

**中文:** 12.63 78.74 98.16 86.47

<a id="S161"></a>
**Source:** p.9 S161  
**Type:** body  
**Confidence:** high

**Original:** 9.09 22.79 95.15 89.76

**中文:** 9.09 22.79 95.15 89.76

<a id="S162"></a>
**Source:** p.9 S162  
**Type:** body  
**Confidence:** high

**Original:** 16.50 50.68 97.52 89.68

**中文:** 16.50 50.68 97.52 89.68

<a id="S163"></a>
**Source:** p.9 S163  
**Type:** body  
**Confidence:** high

**Original:** region); (3) When dealing with multi-target segmentation, as shown in (b)-(f), the inference time is positively correlated with the number of targets. This phenomenon is closely related to the computational cost of the multi-mask inverse mapping matrix. This finding provides a relatively effective solution for synchronous and efficient segmentation of multiple organs in breeding scenarios, which can effectively improve the timeliness of multi-target segmentation and thus shorten the overall cycle of phenotypic analysis in genomic selection breeding.

**中文:** (3) 在处理多目标分割时,如 (b) - ((f) 所示,推断时间与目标数量有积极相关性.这种现象与多面膜反向映射矩阵的计算成本密切相关.这一发现为育种场景中的多器官进行同步和高效分割的解决方案提供了相对有效的解决方案,这可以有效提高多目标分割的及时性,从而缩短基因组选择育种中的现象分析总周期.

<a id="S164"></a>
**Source:** p.9 S164  
**Type:** body  
**Confidence:** high

**Original:** data acquisition, COLMAP, NeRF reconstruction, 3D point cloud extraction, point cloud export, and trait extraction. The postreconstruction processes collectively take only 3 min, demonstrating the high efficiency of the module design. This paper also discusses in Section B the selection of reconstruction algorithms and their corresponding time consumption based on different application scenarios.

**中文:** 数据采集,COLMAP,NeRF重建,3D点云提取,点云出口和特征提取.后重建过程总共只需要3分钟,证明模块设计的高效性.本文还讨论了B节中重建算法的选择及其相应的时间消耗,基于不同的应用场景.

<a id="S165"></a>
### 3.1.2. Time performance analysis
**Source:** p.9 S165  
**Type:** section  
**Confidence:** high

**Original:** 3.1.2. Time performance analysis

**中文:** 3.1.2. 时间性能分析

<a id="S166"></a>
**Source:** p.9 S166  
**Type:** body  
**Confidence:** high

**Original:** This paper conducts a quantitative analysis of the correlation between model segmentation efficiency and the number of targets, as shown in Fig. 6. The experiment collected the average segmentation time of SA3D for single targets and IPENS for single and multiple targets. The experimental results indicate that: (1) Thanks to the customization of light parameters and the application of parallel algorithms, as compared in (a) and (b), IPENS has about a 3.3x acceleration over SA3D. (2) For single-organ segmentation tasks, such as the segmentation time for leaves in (b) and individual grains in (c) being 71.13s and 70.66s respectively, with a time difference of only 0.66%, indicating that the computational efficiency of the model is less affected by the morphological characteristics of the target organs (i.e., the area of the mask

**中文:** 本文对模型分割效率与目标数量的相关性进行了量化分析,如图6所示.实验收集了SA3D单目标的平均分割时间和IPENS单目标和多目标的平均分割时间.实验结果表明: (1) 由于定制光参数和应用平行算法,相比于 (a) 和 (b),IPENS比SA3D加速了3.3倍左右. (2) 单器官分割任务,如 (b) 中的叶子和 (c) 中的单粒的分割时间分别为71.13s和70.66s,时间差距仅为0.66%,表明模型的计算效率受到目标器官的形态特征的影响较小 (即掩膜掩膜面).

<a id="F006"></a>
### Fig. 6. 比较目标推断时间,举例来说,米. (a) SA3D单目标点云分割时间. (b) - c) IPENS分段时间不同器官. (d) - g) IPENS同时分段时间2至5多目标.
**Placed near:** p.9 S166  
**Source:** p.10 C012  
**Crop confidence:** high

![Fig. 6](assets/fig6.png)

**Original caption:** Fig. 6. Comparison of target inference time, taking rice as an example. (a) SA3D single-target point cloud segmentation time. (b)-(c) IPENS segmentation time for different organs. (d)-(g) IPENS simultaneous segmentation time for 2 to 5 multi-target.

**中文图注:** 图6.比较目标推断时间,举例来说,米. (a) SA3D单目标点云分割时间. (b) - c) IPENS分段时间不同器官. (d) - g) IPENS同时分段时间2至5多目标.

**Reading note:** 重点查看该图如何支撑相邻正文中的方法流程、实验比较或平台应用描述。

<a id="S167"></a>
### 3.2. Phenotypic analysis
**Source:** p.9 S167  
**Type:** section  
**Confidence:** high

**Original:** 3.2. Phenotypic analysis

**中文:** 3.2. 表型分析

<a id="S168"></a>
### 3.2.1. Analysis of rice grain voxel volume
**Source:** p.9 S168  
**Type:** section  
**Confidence:** high

**Original:** 3.2.1. Analysis of rice grain voxel volume

**中文:** 3.2.1. 水稻籽粒体素体积分析

<a id="S169"></a>
**Source:** p.9 S169  
**Type:** body  
**Confidence:** high

**Original:** Based on the voxel volume calculation algorithm, Fig. 7a shows the correlation between the labeled and predicted voxel volume of rice

**中文:** 根据体素体积计算算法,图7a显示了米标记和预测体素体积之间的相关性

<a id="F007"></a>
### Fig. 7. 7. (a) 标记和预测模表型提取的籽粒体素之间的相关性. (b) 标记和预测模表型提取的穗部语量之间的相关性.
**Placed near:** p.9 S169  
**Source:** p.10 C014  
**Crop confidence:** high

![Fig. 7](assets/fig7.png)

**Original caption:** Fig. 7. (a) Correlation between labeled and predicted grain voxel volume extracted by model. (b) Correlation between labeled and predicted panicle voxel volume extracted by model.

**中文图注:** 7. (a) 标记和预测模表型提取的籽粒体素之间的相关性. (b) 标记和预测模表型提取的穗部语量之间的相关性.

**Reading note:** 重点查看该图如何支撑相邻正文中的方法流程、实验比较或平台应用描述。

<a id="C008"></a>
### Table 3 presents the key time costs of the IPENS workflow, including data acquisition, COLMAP, NeRF reconstruction, 3D point cloud
**Source:** p.9 C008  
**Type:** caption  
**Confidence:** high

**Original:** Table 3 presents the key time costs of the IPENS workflow, including data acquisition, COLMAP, NeRF reconstruction, 3D point cloud

**中文:** 图3介绍了IPENS工作流程的关键时间成本,包括数据采集,COLMAP,NeRF重建,3D点云

<a id="C009"></a>
### Table 2
**Source:** p.9 C009  
**Type:** caption  
**Confidence:** high

**Original:** Table 2. Comparison of mainstream segmentation methods on the MMR and MMW datasets.

**中文:** 表2:对MMR和MMW数据集的主要分割方法进行比较.


## Page 10

<a id="S170"></a>
**Source:** p.10 S170  
**Type:** body  
**Confidence:** high

**Original:** different organs. (d)–(g) IPENS simultaneous segmentation time for 2 to 5 multi-target.

**中文:** (d) (g) IPENS同时分类时间为2到5个多目标.

<a id="S171"></a>
**Source:** p.10 S171  
**Type:** body  
**Confidence:** high

**Original:** grains, with an R2 of 0.7697 and an RMSE of 0.0025. These results validate the effectiveness of IPENS in extracting grain voxel volume and the accuracy of the 3D point cloud.

**中文:** 这些结果证实IPENS在提取籽粒语音体量中的有效性和3D点云的精度.

<a id="S172"></a>
**Source:** p.10 S172  
**Type:** body  
**Confidence:** high

**Original:** Time consumption in the IPENS workflow. Step

**中文:** 在IPENS工作流程中消耗时间.

<a id="S173"></a>
**Source:** p.10 S173  
**Type:** body  
**Confidence:** high

**Original:** Time

**中文:** 时间

<a id="S174"></a>
**Source:** p.10 S174  
**Type:** body  
**Confidence:** high

**Original:** Data Acquisition COLMAP Parameter Estimation NeRF Reconstruction 3D Point Cloud Extraction Point Cloud Export Trait Extraction

**中文:** 数据获取 COLMAP 参数估计 NeRF 重新构建 3D 云点 云点 云点 云点 输出 标志 输出

<a id="S175"></a>
**Source:** p.10 S175  
**Type:** body  
**Confidence:** high

**Original:** 3 min 8 min 8 min 2 min 0.5 min 0.5 min

**中文:** 3分8分8分2分0.5分0.5分

<a id="S176"></a>
### 3.2.2. Analysis of wheat panicle voxel volume
**Source:** p.10 S176  
**Type:** section  
**Confidence:** high

**Original:** 3.2.2. Analysis of wheat panicle voxel volume

**中文:** 3.2.2. 小麦穗体素体积分析

<a id="S177"></a>
**Source:** p.10 S177  
**Type:** body  
**Confidence:** high

**Original:** voxel volume of wheat panicles, with an R2 of 0.9956 and an RMSE of 0.0055. The fitting results nearly overlap with the reference fitting, indicating that the proposed method can achieve high-quality segmentation for wheat panicles.

**中文:** 面积音素的小麦,R2为0.9956和RMSE为0.0055.合结果几乎与参考相重叠,这表明,拟议的方法可以实现高质量的小麦的分割.

<a id="S178"></a>
**Source:** p.10 S178  
**Type:** body  
**Confidence:** high

**Original:** extracted by model.

**中文:** 由模表型提取.

<a id="C011"></a>
### Table 3 validate the effectiveness of IPENS in extracting grain voxel volume and Time consumption in the IPENS workflow
**Source:** p.10 C011  
**Type:** caption  
**Confidence:** high

**Original:** Table 3 validate the effectiveness of IPENS in extracting grain voxel volume and Time consumption in the IPENS workflow. the accuracy of the 3D point cloud. Step Time Data Acquisition 3 min

**中文:** 图3证实IPENS在IPENS工作流程中提取籽粒语音体量和时间消耗方面的有效性.3D点云的准确性.步骤时间数据采集3分

<a id="C013"></a>
### Table 3
**Source:** p.10 C013  
**Type:** caption  
**Confidence:** high

**Original:** Table 3. Time consumption in the IPENS workflow.

**中文:** 时间消耗在IPENS工作流程.


## Page 11

<a id="S179"></a>
### 3.2.3. Leaf phenotypic analysis
**Source:** p.11 S179  
**Type:** section  
**Confidence:** high

**Original:** 3.2.3. Leaf phenotypic analysis

**中文:** 3.2.3. 叶片表型分析

<a id="S180"></a>
**Source:** p.11 S180  
**Type:** body  
**Confidence:** high

**Original:** different stages of point cloud processing, and compares the results of Label and Pred. After preprocessing, the calculation of the convex hull area of the point cloud has a high RMSE (28.27) and MAE (18.24), and the fitting situation R2 is 0.68. This indicates that there is a large error in calculating the leaf area by convex hull, which is greatly affected by point cloud noise. The prediction error of the area after meshing the leaves has decreased, indicating that the repair treatment may have removed some noise and local abnormalities, improving the result stability. After subdividing the mesh, the RMSE and MAE are 18.93 and 13.21, and R2 has increased to 0.84. The error indicators at this stage have decreased compared with the previous two steps, and the subdivision algorithm further smooths the surface details, making the overall area estimation more accurate. Table 4b shows the surface area results of wheat leaves. It can be found that after subdivision, the RMSE and MAE reach 0.67 and 0.53, and R2 reaches 1.00, indicating the excellent ability of the model in estimating the area of wheat leaves. Overall, the gradual decrease in error at each stage indicates that the estimation of leaf surface area is gradually stabilizing and demonstrating higher accuracy.

**中文:** 在处理前后,计算点云的形体面积具有高的RMSE (28.27) 和MAE (18.24),而适配情况R2是0.68.这表明,在计算形体叶片面积时存在严重的错误,这受到点云噪音的影响.在叶网后,该区域的预测错误减少了,这表明修复处理可能消除了一些噪音和当地异常,从而改善了结果稳定性.在分网后,RMSE和MAE的不同数量为18.93和13.21,而R2则增加到0.84. 在这个阶段,错误指标与前两步相比较减少,分类算法进一步平滑了表面细节,使整体面积估算更加准确.表4b显示了小麦叶的表面积结果.可以发现,分类后,RMSE和MAE达到0.67和0.53和R2达到1.00,这表明模型在估算小麦叶的面积方面非常有能力.总的来说,每个阶段的误差逐渐减少表明,叶片表面积估计逐渐稳定并显示更高的准确性.

<a id="S181"></a>
**Source:** p.11 S181  
**Type:** body  
**Confidence:** high

**Original:** leaves. It indicates that the length and width measurement method exhibits high accuracy in estimating leaf size. Specifically, the predicted mean value for leaf length (24.32 cm) is close to the truth (24.77 cm), and the mean width value (2.38 cm) is also slightly lower than the truth (2.57 cm), indicating that the calculation method difference is at the millimeter level. Error analysis shows that the RMSE of length and width are 1.49 and 0.21, the MAE are 1.03 and 0.19, and R2 reaches 0.97 and 0.87, indicating that the deviation between the predicted value and the truth is small, which verifies the stability of the method in rice leaf estimation. Table 4d shows the statistical results of the length and width of wheat leaves. The RMSE of leaf length and width are 0.23 and 0.15, the MAE are 0.18 and 0.12 and the R2 reaches 0.99 and 0.92. The deviation is very small and the size difference is at the millimeter level. Combined with Table 2b, the leaf IoU indicates that a better segmentation result corresponds to excellent phenotypic results, which verifies the effectiveness of the method.

**中文:** 它表明,长度和宽度测量方法在估算叶子尺寸时具有高精度.具体来说,叶子长度预测平均值 (24.32厘米) 接近真相 (24.77厘米),平均宽度值 (2.38厘米) 也略低于真相 (2.57厘米),表明计算方法的差异在毫米水平.错误分析显示,长度和宽度的RMSE为1.49和0.21,MAE为1.03和0.19,R2达到0.97和0.87,这表明预测值和真相之间的偏差很小,这验证了叶子估计方法的稳定性.4d显示了小麦的长度和宽度的统计结果.表面的RMSE叶子和MAE叶子的长度和宽度为0.23和0.25,RMSE叶子和0.12和0.95,RMSE叶子和0.12和0.9和0.2是0.95,RMSE叶子和0.2和0.9和0.9的宽度为0.95. 偏差很小,尺寸差距在毫米水平.与表2b相结合,叶子IoU表明更好的分割结果与优秀的表型结果相匹配,这证明了该方法的有效性.

<a id="S182"></a>
**Source:** p.11 S182  
**Type:** body  
**Confidence:** high

**Original:** Comparison of rice and wheat leaf surface area, length, and width (in cm2 and cm). (a) Rice (Surface Area). Step

**中文:** 米和小麦叶表面积,长度和宽度 (在cm2和cm) 的比较. (a) 米 (表面积).

<a id="S183"></a>
**Source:** p.11 S183  
**Type:** body  
**Confidence:** high

**Original:** Label Mean

**中文:** 标签的意思是

<a id="S184"></a>
**Source:** p.11 S184  
**Type:** body  
**Confidence:** high

**Original:** Pred Mean

**中文:** 预言的意思是

<a id="S185"></a>
**Source:** p.11 S185  
**Type:** body  
**Confidence:** high

**Original:** RMSE

**中文:** RMSE

<a id="S186"></a>
**Source:** p.11 S186  
**Type:** body  
**Confidence:** high

**Original:** MAE

**中文:** ,我可以说是.

<a id="S187"></a>
**Source:** p.11 S187  
**Type:** body  
**Confidence:** high

**Original:** Cloud Convex Hull Area Repaired Mesh Area Subdivision Mesh Area

**中文:** 圆云圆体区域修复网格区域分类网格区域

<a id="S188"></a>
**Source:** p.11 S188  
**Type:** body  
**Confidence:** high

**Original:** 98.54 79.47 77.33

**中文:** 现在,我们在98.54 79.47 77.33

<a id="S189"></a>
**Source:** p.11 S189  
**Type:** body  
**Confidence:** high

**Original:** 109.49 89.29 86.80

**中文:** 109.49 89.29 86.80

<a id="S190"></a>
**Source:** p.11 S190  
**Type:** body  
**Confidence:** high

**Original:** 28.27 19.37 18.93

**中文:** 28.27 19.37 18.93 28.27 19.37 18.93

<a id="S191"></a>
**Source:** p.11 S191  
**Type:** body  
**Confidence:** high

**Original:** 18.24 13.58 13.21

**中文:** 18.24 13.58 13.21

<a id="S192"></a>
**Source:** p.11 S192  
**Type:** body  
**Confidence:** high

**Original:** 0.68 0.84 0.84

**中文:** 0.68 0.84 0.84

<a id="S193"></a>
**Source:** p.11 S193  
**Type:** body  
**Confidence:** high

**Original:** Step

**中文:** 步骤

<a id="S194"></a>
**Source:** p.11 S194  
**Type:** body  
**Confidence:** high

**Original:** Label Mean

**中文:** 标签的意思是

<a id="S195"></a>
**Source:** p.11 S195  
**Type:** body  
**Confidence:** high

**Original:** Pred Mean

**中文:** 预言的意思是

<a id="S196"></a>
**Source:** p.11 S196  
**Type:** body  
**Confidence:** high

**Original:** RMSE

**中文:** RMSE

<a id="S197"></a>
**Source:** p.11 S197  
**Type:** body  
**Confidence:** high

**Original:** MAE

**中文:** ,我可以说是.

<a id="S198"></a>
**Source:** p.11 S198  
**Type:** body  
**Confidence:** high

**Original:** Cloud Convex Hull Area Repaired Mesh Area Subdivision Mesh Area

**中文:** 圆云圆体区域修复网格区域分类网格区域

<a id="S199"></a>
**Source:** p.11 S199  
**Type:** body  
**Confidence:** high

**Original:** 22.98 16.32 15.87

**中文:** 22.98 16.32 15.87

<a id="S200"></a>
**Source:** p.11 S200  
**Type:** body  
**Confidence:** high

**Original:** 23.55 16.42 15.91

**中文:** 23.55 16.42 15.91

<a id="S201"></a>
**Source:** p.11 S201  
**Type:** body  
**Confidence:** high

**Original:** 1.72 0.78 0.67

**中文:** 七十二 0.78 0.67 1.67

<a id="S202"></a>
**Source:** p.11 S202  
**Type:** body  
**Confidence:** high

**Original:** 1.13 0.63 0.53

**中文:** 1.13 0.63 0.53

<a id="S203"></a>
**Source:** p.11 S203  
**Type:** body  
**Confidence:** high

**Original:** 0.99 1.00 1.00

**中文:** 0.99 1.00 1.00

<a id="S204"></a>
**Source:** p.11 S204  
**Type:** body  
**Confidence:** high

**Original:** (b) Wheat (Surface Area).

**中文:** (二) 麦子 (表面区域).

<a id="S205"></a>
**Source:** p.11 S205  
**Type:** body  
**Confidence:** high

**Original:** (c) Rice (Length & Width). Metric

**中文:** (三) 米 (长度和宽度).

<a id="S206"></a>
**Source:** p.11 S206  
**Type:** body  
**Confidence:** high

**Original:** Length

**中文:** 长度

<a id="S207"></a>
**Source:** p.11 S207  
**Type:** body  
**Confidence:** high

**Original:** Width

**中文:** 宽度

<a id="S208"></a>
**Source:** p.11 S208  
**Type:** body  
**Confidence:** high

**Original:** Label Mean Pred Mean RMSE MAE R2

**中文:** 标签: 预测 预测 RMSE MAE R2

<a id="S209"></a>
**Source:** p.11 S209  
**Type:** body  
**Confidence:** high

**Original:** 24.77 24.32 1.49 1.03 0.97

**中文:** 24.77 24.32 1.49 1.03 0.97

<a id="S210"></a>
**Source:** p.11 S210  
**Type:** body  
**Confidence:** high

**Original:** 2.57 2.38 0.21 0.19 0.87

**中文:** 2.57 2.38 0.21 0.19 0.87

<a id="S211"></a>
**Source:** p.11 S211  
**Type:** body  
**Confidence:** high

**Original:** Metric

**中文:** 测量

<a id="S212"></a>
**Source:** p.11 S212  
**Type:** body  
**Confidence:** high

**Original:** Length

**中文:** 长度

<a id="S213"></a>
**Source:** p.11 S213  
**Type:** body  
**Confidence:** high

**Original:** Width

**中文:** 宽度

<a id="S214"></a>
**Source:** p.11 S214  
**Type:** body  
**Confidence:** high

**Original:** Label Mean Pred Mean RMSE MAE R2

**中文:** 标签: 预测 预测 RMSE MAE R2

<a id="S215"></a>
**Source:** p.11 S215  
**Type:** body  
**Confidence:** high

**Original:** 10.23 10.08 0.23 0.18 0.99

**中文:** 10.23 10.08 0.23 0.18 0.99

<a id="S216"></a>
**Source:** p.11 S216  
**Type:** body  
**Confidence:** high

**Original:** 1.00 0.98 0.15 0.12 0.92

**中文:** 1.00 0.98 0.15 0.12 0.92

<a id="S217"></a>
**Source:** p.11 S217  
**Type:** body  
**Confidence:** high

**Original:** (d) Wheat (Length & Width).

**中文:** (三) 麦子 (长度和宽度).

<a id="S218"></a>
### 4.2. Effectiveness of the proposed method
**Source:** p.11 S218  
**Type:** section  
**Confidence:** high

**Original:** 4.2. Effectiveness of the proposed method

**中文:** 4.2. 所提方法的有效性

<a id="S219"></a>
**Source:** p.11 S219  
**Type:** body  
**Confidence:** high

**Original:** To better understand the working mechanism of IPENS, we provide more discussions. Firstly, the good construction of the NeRF field can derive high-quality point clouds, which is beneficial to the improvement of segmentation quality in subsequent processes. SAM2 is pre-trained on SA-V, the largest general object video segmentation dataset so far. This ensures that SAM2 performs well in various tasks and visual fields. Since the point cloud extraction capability of the model depends on the 2D segmentation and propagation capabilities of SAM2, more prompt information will optimize the accuracy of 2D target segmentation, and appropriate post-processing operations to fill holes and smooth edges will make the 3D mask more accurate.

**中文:** 为了更好地了解IPENS的工作机制,我们提供了更多的讨论.首先,NeRF领域的良好构建可以产生高质量的点云,这有利于后续过程中的分割质量改善.SAM2在SA-V上预训练,这是迄今为止最大的一般对象视频分割数据集.这确保SAM2在各种任务和视觉领域表现得很好.由于模型的点云提取能力取决于SAM2的2D分割和传播能力,因此更快速的信息将优化2D目标分割的精度,适当的后处理操作来填补漏洞和平滑边缘将使3D掩膜更准确.

<a id="S220"></a>
### 4. Discussion
**Source:** p.11 S220  
**Type:** section  
**Confidence:** high

**Original:** 4. Discussion

**中文:** 4. 讨论

<a id="S221"></a>
### 4.1. Interpretation of 3D segmentation performance
**Source:** p.11 S221  
**Type:** section  
**Confidence:** high

**Original:** 4.1. Interpretation of 3D segmentation performance

**中文:** 4.1. 3D 分割性能解释

<a id="S222"></a>
**Source:** p.11 S222  
**Type:** body  
**Confidence:** high

**Original:** Recall of grain are due to the regular shape and obvious color features of grain, which enable the model to accurately identify positive samples, but some occlusions or illumination changes lead to missed detections. The leaf segmentation effect is the best, which is related to the regular shape of the leaves and the high color contrast. The low precision indicates that there are cases where stems are mistakenly judged as leaves. The overall performance of the stem is weak, which may be due to the slender morphological characteristics of the stem increasing the difficulty of boundary localization or the blurred boundary with the leaf leading to feature confusion. It is necessary to optimize the model's ability to recognize the stem.

**中文:** 收获籽粒是由于籽粒的正规形状和明显的颜色特征,使模型能够准确识别正面样品,但某些罩或照明变化导致错误的检测.回忆叶片分割效应是最好的,这与叶子的正规形状和高颜色对比有关.低精度表明有时茎被误认为是叶子.茎的整体性能很弱,这可能是由于茎的微薄的形态特征增加了边界定位难度或叶子的模糊边界导致了功能混乱.必须优化模型识别茎的能力.

<a id="S223"></a>
**Source:** p.11 S223  
**Type:** body  
**Confidence:** high

**Original:** wheat panicle is the best, indicating that the model has a good segmentation effect on large targets with less occlusion. The segmentation effect of leaves is slightly worse than that of panicles and stems, which may be due to the irregular edges of leaves or their adhesion to stems, leading to boundary prediction errors. The results for stems indicate that the model can accurately capture the structure and boundaries of stems. Overall, the model can solve the task of high-precision wheat organ segmentation. Compared with the results of rice leaves and stems, it can be seen that the model achieves better results for wheat due to its relatively clear organ boundaries.

**中文:** 麦子是最好的,表明模型对较大的目标具有较少遮蔽性的好分割效应.叶子的分割效应略差于子和茎的分割效应,可能是由于叶子的不规则边缘或它们粘合到茎,导致边界预测错误. 茎的结果表明模型可以准确捕捉茎的结构和边界. 总体而言,模型可以解决高精度的麦子器官分割任务. 与子叶子和茎的结果相比,可以看到模型由于其相对清晰的器官边界,可以实现更好的结果.

<a id="S224"></a>
### 4.3. Limitation and future prospects
**Source:** p.11 S224  
**Type:** section  
**Confidence:** high

**Original:** 4.3. Limitation and future prospects

**中文:** 4.3. 局限性与未来展望

<a id="S225"></a>
**Source:** p.11 S225  
**Type:** body  
**Confidence:** high

**Original:** Due to the limitations of machine video memory and mask storage design, IPENS will encounter insufficient memory problems as the number of targets being segmented simultaneously increases, as shown in Table 5. This has become an obstacle to clicking on a target type such as rice grains and simultaneously segmenting all the grains. This requires gradual optimization of video memory usage design in subsequent work to enhance multi-target segmentation capabilities. The IPENS algorithm is seamlessly embeddable in high-throughput phenotyping platforms. As illustrated in Fig. 8a and b, our fully automated phenotyping chamber system transports potted crops via conveyor belts for continuous processing. Robotic arms equipped with a dual-camera perform multi-view imaging inside the chamber. IPENS processes image data in real-time to extract crop organ metrics, with breeding scientists accessing results through an interactive dashboard. Outdoor experiments reveal that NeRF reconstruction is highly susceptible to lighting (e.g., shadows, glare, uneven illumination) and wind disturbances, making it challenging to generate high-resolution 3D models. It limits IPENS’ application to real-world field conditions. To solve this problem, we are developing a field phenotyping vehicle that

**中文:** 由于机器视频内存和掩膜存储设计的局限性,IPENS将遇到不够的内存问题,因为同时分类目标的数量增加,如图5所示.这已经成为单击一目标类型,如粒和同时分类所有粒子的障碍.这需要逐步优化视频内存使用设计,以增强多目标分类能力.IPENS算法是无嵌入高吞吐量表型平台.如图8a和b所示,我们的全自动化型室系统通过传输带运输料的作物.机器人装备双摄像头进行多摄像头处理. IPENS实时处理图像数据以提取作物器官的测量,培养科学家通过交互式仪表板获取结果.户外实验显示NeRF重建对照明 (例如阴影,闪光,不均的照明) 和风力干扰高度敏感,使得产生高分辨率3D模型具有挑战性.它限制了IPENS应用到现实领域的现实环境.为了解决这个问题,我们正在开发一个现场表型车,该车将的表型号进行测试.

<a id="F008"></a>
### Fig. 8. 室内高通量表型舱系统与室外田间数据采集车。
**Placed near:** p.11 S225  
**Source:** p.12 C023  
**Crop confidence:** high

![Fig. 8](assets/fig8.png)

**Original caption:** Fig. 8. Indoor high-throughput phenotyping chamber system and outdoor field data acquisition vehicle.

**中文图注:** 图 8. 室内高通量表型舱系统与室外田间数据采集车。

**Reading note:** 重点查看该图如何支撑相邻正文中的方法流程、实验比较或平台应用描述。

<a id="C015"></a>
### Table 4b shows the surface area results (c) Rice (Length & Width)
**Source:** p.11 C015  
**Type:** caption  
**Confidence:** low

**Original:** Table 4b shows the surface area results (c) Rice (Length & Width). o M f A w E h r e e a a t c h le a 0 v. 6 e 7 s. a I n t d c a 0 n. 5 b 3 e, a f n ou d n R d 2 t r h e a a t c h a e ft s e 1 r. s 0 u 0 b, d in iv d i i s c io at n i, ng th t e he R M ex S c E ell a e n n d t M La e b t e r l i c Mean L 2 e 4 n.7 g 7 th W 2.5 id 7 th ability of the model in estimating the area of wheat leaves. Pred Mean 24.32 2.38 Overall, the gradual decrease in error at each stage indicates that the RMSE 1.49 0.21 estimation of leaf surface area is gradually stabilizing and demonstrating MAE 1.03 0.19 higher accuracy. R2 0.97 0.87

**中文:** 4b显示了表面积的结果 (c) 米 (长度和宽度). o M f A w E h r e e a t c h l a 0 v 6 e 7 s. a I n t d c a 0 n 5 b 3 e, a f n ou d n R d 2 t r h e a t c h a e ft s e 1 r s 0 u 0 b, d in iv d i s c io at n i, ng t e e e e e e e m ex S c E ell a e n d t m M La e b t e r l i c 平均L 2 e 4 n.7 g 7 秒 W 2.5 id 7 秒 模型在估算小麦叶的面积. 平均 24.32 2.38 总的来说,每一个阶段的逐步下降表明R E E E 面积的0.39 度率是1.03 度,表表现出了每一个小米的0.87 度的0.87 度,表现出了每一个小米的0.09 度的0.09 度.

<a id="C016"></a>
### Table 4c presents the statistical results of the length and width of rice (d) Wheat (Length & Width)
**Source:** p.11 C016  
**Type:** caption  
**Confidence:** high

**Original:** Table 4c presents the statistical results of the length and width of rice (d) Wheat (Length & Width). leaves. It indicates that the length and width measurement method ex- Metric Length Width hibits high accu racy in est imating leaf si ze. Sp ec ifica lly, th e pred icted Label Mean 10.23 1.00 mean value for leaf length (24.32 cm) is close to the truth (24.77 cm), Pred Mean 10.08 0.98 and the mean width value (2.38 cm) is also slightly lower than the truth RMSE 0.23 0.15 (2.57 cm), indicating that the calculation method difference is at the MAE 0.18 0.12 millimeter level. Error analysis shows that the RMSE of length and width R2 0.99 0.92 are 1.49 and 0.21, the MAE are 1.03 and 0.19, and R2 reaches 0.97 and 0.87, indicating that the deviation between the predicted value and the truth is small, which verifies the stability of the method in rice leaf

**中文:** 4c呈现了米 (d) 叶子的长度和宽度统计结果.它表明长度和宽度测量方法 ex- Metric Length Width hibits high accu racy in est imating leaf si ze. Sp ec ifica lly, th e pred icted Label 标签平均值 10.23 1.00 叶子长度 (24.32厘米) 接近真相 (24.77厘米),预值 10.08 0.98 和平均宽度值 (2.38厘米) 也略低于真相 RMSE 0.23 0.15 (2.57厘米),表明计算方法差距在MAE 0.18 0.12毫米水平. 错误分析显示,长度和宽度的RMSE R2 0.99 0.92为1.49和0.21,MAE为1.03和0.19,R2达到0.97和0.87,这表明预测值和真相之间的偏差很小,这验证了叶方法的稳定性

<a id="C017"></a>
### Table 4d shows the statistical results of the length and width of wheat leaves
**Source:** p.11 C017  
**Type:** caption  
**Confidence:** high

**Original:** Table 4d shows the statistical results of the length and width of wheat leaves. The RMSE of leaf length and width are 0.23 and 0.15, To better understand the working mechanism of IPENS, we provide the MAE are 0.18 and 0.12 and the R2 reaches 0.99 and

**中文:** 图4d显示了小麦叶的长度和宽度的统计结果.叶子长度和宽度的RMSE为0.23和0.15,为了更好地了解IPENS的工作机制,我们提供MAE为0.18和0.12和R2达到0.99和.

<a id="C018"></a>
### Table 2b, the leaf IoU indicates that a better segmen- of segmentation quality in subsequent processes
**Source:** p.11 C018  
**Type:** caption  
**Confidence:** high

**Original:** Table 2b, the leaf IoU indicates that a better segmen- of segmentation quality in subsequent processes. SAM2 is pre-trained on tation result corresponds to excellent phenotypic results, which verifies SA-V, the largest general object video segmentation dataset so far. This the effectiveness of the method. ensures that SAM2 performs well in various tasks and visual fields. Since the point cloud extraction capability of the model depends on the 2D 4. Discussion segmentation and propagation capabilities of SAM2, more prompt in- formation will optimize the accuracy of 2D target segmentation, and

**中文:** 2b,叶子IoU表示,在后续过程中更好的分段质量.SAM2预先训练在分段结果上,结果相应于出色的表型结果,这验证了SA-V,迄今为止最大的一般对象视频分段数据集.这使得方法的有效性.确保SAM2在各种任务和视觉领域表现良好.由于模型的点云提取能力取决于2D 4.讨论分段和传播能力SAM2,更快速的形成将优化2D目标分段的精度,并

<a id="C019"></a>
### Table 5
**Source:** p.11 C019  
**Type:** caption  
**Confidence:** high

**Original:** Table 5. This has become an obstacle to clicking on a target type such slender morphological characteristics of the stem increasing the diffi- as rice grains and simultaneously segmenting all the grains. This re- culty of boundary localization or the blurred boundary with the leaf quires gradual optimization of video memory usage design in subse- leading to feature confusion. It is necessary to optimize the model's quent work to enhance multi-target segmentation capabilities. ability to recognize the stem. The IPENS algorithm is seamlessly embeddable in high-throughput

**中文:** 5.这已经成为了对点击目标类型的障碍,例如增加子的微薄形态特性,增加籽粒的微小程度,同时分割所有粒子.这种边界定位的重组或与叶子的模糊边界需要逐步优化视频内存使用设计,从而导致功能混乱.必须优化模型的点工作,以增强多目标分割能力.识别子的能力.IPENS算法可以无嵌在高吞吐量中.

<a id="C020"></a>
### Table 4
**Source:** p.11 C020  
**Type:** caption  
**Confidence:** high

**Original:** Table 4. Comparison of rice and wheat leaf surface area, length, and width (in cm2 and cm).

**中文:** 表4:米和小麦叶表面面积,长度和宽度 (在cm2和cm) 的比较.


## Page 12

<a id="S226"></a>
**Source:** p.12 S226  
**Type:** body  
**Confidence:** high

**Original:** The IPENS method achieves unsupervised, interactive, and noninvasive extraction of grain-level point clouds. This can provide highquality phenotypic data for establishing genotype-phenotype association models, and its efficient extraction process has great potential in promoting intelligent breeding.

**中文:** IPENS方法实现了无监督,交互式和非侵入性的籽粒级点云的提取.这可以为建立基因型-基因型相关模型提供高质量的表型数据,其高效的提取过程有着促进智能育种的巨大潜力.

<a id="S227"></a>
**Source:** p.12 S227  
**Type:** body  
**Confidence:** high

**Original:** GPU memory consumption under varying numbers of targets. Target num

**中文:** 在不同数量的目标下使用GPU内存.

<a id="S228"></a>
**Source:** p.12 S228  
**Type:** body  
**Confidence:** high

**Original:** VRAM(MiB)

**中文:** VRAM(MiB)

<a id="S229"></a>
### 5.1. Multi-species point cloud extraction visualization
**Source:** p.12 S229  
**Type:** section  
**Confidence:** high

**Original:** 5.1. Multi-species point cloud extraction visualization

**中文:** 5.1. 多物种点云提取可视化

<a id="S230"></a>
**Source:** p.12 S230  
**Type:** body  
**Confidence:** high

**Original:** This section presents the visual results of instance segmentation on multiple species. As shown in Fig. S4, IPENS achieves good segmentation effects on the organs of different crops. It is worth noting that the model has not undergone specialized semantic segmentation training but can still adapt to various scenarios and species, showing strong generalization ability. This verifies the possibility of the model in extracting point clouds across species and varieties in the future.

**中文:** 本节介绍了对多种物种实例分类的视觉结果.如图S4所示,IPENS对不同作物器官产生了良好的分类效果.值得注意的是,该模型尚未接受专业的语义分类培训,但仍然可以适应各种场景和物种,显示出强大的概括能力.这验证了该模型在未来将物种和品种之间提取点云的可能性.

<a id="S231"></a>
**Source:** p.12 S231  
**Type:** body  
**Confidence:** high

**Original:** uses multiple cameras to expose simultaneously, achieving high-speed multi-view imaging of crops, as shown in Fig. 8c and d.

**中文:** 通过多台摄像头同时曝光,实现了高速度的多视图作物成像,如图8c和d所示.

<a id="S232"></a>
### 5. Conclusion
**Source:** p.12 S232  
**Type:** section  
**Confidence:** high

**Original:** 5. Conclusion

**中文:** 5. 结论

<a id="S233"></a>
### 5.2. Time consumption of 3D reconstruction models
**Source:** p.12 S233  
**Type:** section  
**Confidence:** high

**Original:** 5.2. Time consumption of 3D reconstruction models

**中文:** 5.2. 3D 重建模型耗时

<a id="S234"></a>
**Source:** p.12 S234  
**Type:** body  
**Confidence:** high

**Original:** This paper proposes an IPENS framework that combines the 2D segmentation and propagation capabilities of SAM2 with radiance field information to extract 3D target point clouds. A multi-target collaborative point cloud extraction scheme is designed. Two post-processing methods are proposed to optimize the 3D target point clouds based on the SAM2 segmentation results. In an unsupervised interactive manner, the IPENS method achieves grain-level segmentation on a rice dataset, with an mIoU of 63.72% for grains, leaves, and stems. The R2 for grain voxel volume is 0.7697, and the RMSE is 0.0025; the R2 for leaf surface area is 0.84, and the RMSE is 18.93; the R2 for leaf length and width are 0.97 and 0.87 respectively, with RMSE of 1.49 and 0.21. On the wheat dataset, the mIoU for panicles, leaves, and stems is 89.68%, the R2 for panicle voxel volume is 0.9956, and the RMSE is 0.0055; the R2 for leaf surface area is 1.0, and the RMSE is 0.67; the R2 for leaf length and width are 0.99 and 0.92 respectively, with RMSEs of 0.23 and 0.15. These results demonstrate the effectiveness of the IPENS model in phenotypic extraction of rice and wheat.

**中文:** 本文提出了IPENS框架,将SAM2的2D分割和传播能力与辐射场域信息结合起来,以提取3D目标点云.设计了一个多目标协作点云提取方案.建议采用两个后处理方法来优化3D目标点云,基于SAM2分割结果.以无监督的方式,IPENS方法实现了水稻数据集中的籽粒级分割,籽粒,叶子和茎的mIoU为63.72%;籽粒体素体积为0.7697,RMSE为0.0025;叶片面积为0.84,RMSE为18.93;叶片长度和宽度为0.877,RMS2为0.49和REM21.. 在小麦数据集中,穗部,叶子和茎的mIoU为89.68%,穗部体素体积为0.9956,RMSE为0.0055;叶子表面积为1.0,RMSE为0.67;叶子长度和宽度为0.99和0.92,RMSE为0.23和0.15.这些结果证明IPENS模型在米和小麦的异表型提取中有效.

<a id="S235"></a>
**Source:** p.12 S235  
**Type:** body  
**Confidence:** high

**Original:** Depending on different application scenarios, it is necessary to select an appropriate 3D reconstruction model. Table S6 shows the reconstruction performance and effects of Instant-NGP, Instant-NGP-bounded, Nerfacto, Nerfacto-big, and Mip-NeRF360. The performance of different 3D reconstruction methods is compared in terms of Training time, Peak Signal to Noise Ratio (PSNR), structural similarity index (SSIM), Learned Perceptual Image Patch Similarity (LPIPS), and Frames Per Second (FPS). Based on the time cost and performance metrics in the table, the models can be categorized into three groups:

**中文:** 根据不同应用场景,需要选择合适的3D重建模型.表 S6显示了即时NGP,即时NGP-限,NeRFacto,NeRFacto-big和Mip-NeRF360的重建性能和效果.根据训练时间,信号与噪音比率 (PSNR),结构相似度指数 (SSIM),学习的感知图像补丁相似度 (LPIPS) 和每秒框架 (FPS) 的不同3D重建方法的性能进行了比较.根据表中的时间成本和性能指标,模型可以分为三个组:

<a id="S236"></a>
**Source:** p.12 S236  
**Type:** body  
**Confidence:** high

**Original:** • Fast Models (≤10 min): This includes Instant-NGP, Instant-NGPbounded, and Nerfacto. These models are suitable for real-time monitoring, quick previews, or applications on mobile devices and edge computing. For example, in drone-based field inspections, they can instantly generate 3D models of crops to assess growth conditions; they are also ideal for processing data quickly on resource-

**中文:** •快速模型 (≤10分钟):包括即时NGP,即时NGPbounded和NeRFacto.这些模型适合实时监测,快速预览或移动设备和边缘计算上的应用.例如,在无人机的田地检查中,它们可以立即生成3D作物模型来评估生长条件;它们也非常适合快速处理资源上的数据.

<a id="C021"></a>
### Table 5 The IPENS method achieves unsupervised, interactive, and non- GPU memory consumption under varying numbers invasive extraction of grain-level point clouds
**Source:** p.12 C021  
**Type:** caption  
**Confidence:** high

**Original:** Table 5 The IPENS method achieves unsupervised, interactive, and non- GPU memory consumption under varying numbers invasive extraction of grain-level point clouds. This can provide high- of targets. quality phenotypic data for establishing genotype-phenotype associa- Target num VRAM(MiB) tion models, and its efficient extraction process has great potential in promoting intelligent breeding. 1 8736 2 10853 3

**中文:** 5 图表 5 IPENS 方法在不同数量的籽粒级点云中实现了无监督,交互式和非GPU内存消耗.这可以提供高质量的目标. 建立基因型-基因型协会-目标数 VRAM (MiB) 模型的质量表型数据,其高效的提取过程有着促进智能育种的巨大潜力. 1 8736 2 10853 3

<a id="C022"></a>
### Table 5
**Source:** p.12 C022  
**Type:** caption  
**Confidence:** high

**Original:** Table 5. GPU memory consumption under varying numbers of targets.

**中文:** 图5.在不同数量的目标下使用GPU内存消耗.


## Page 13

<a id="S237"></a>
**Source:** p.13 S237  
**Type:** body  
**Confidence:** high

**Original:** constrained devices such as agricultural robots or handheld terminals.

**中文:** 限制设备,如农业机器人或手持终端.

<a id="S238"></a>
**Source:** p.13 S238  
**Type:** body  
**Confidence:** high

**Original:** • Medium-Speed Models (≈1 h): Nerfacto-big falls into this category. It is suitable for high-precision agricultural analysis, where a balance between speed and accuracy is required. For instance, it can be used for dynamic monitoring of crop growth in greenhouses, generating high-precision models weekly to track development trends.

**中文:** • 中速模型 (≈1h):NeRFacto-big属于这个类别.它适合高精度农业分析,需要保持速度和精度之间的平衡.例如,它可以用于动态监测温室中作物增长,每周生成高精度模型来跟踪发展趋势.

<a id="S239"></a>
**Source:** p.13 S239  
**Type:** body  
**Confidence:** high

**Original:** • Long-Time Models (Several hours): Models like Mip-NeRF360 belong to this group. They are suitable for reconstructing extremely complex scenes or scenarios with high-fidelity requirements. For example, in the holographic modeling of ancient or famous trees for cultural heritage preservation, or for publishing research papers or generating standard datasets, these models maximize reconstruction quality despite the longer processing time.

**中文:** •长时间模型 (几个小时):像Mip-NeRF360这样的模型属于这个群体.它们适合重建具有高效度要求的极具复杂场景或场景.例如,在保护文化遗产的古老或著名树木的全息图模型中,或发表研究论文或生成标准数据集中,这些模型尽管需要更长的处理时间,但会最大限度地提高重建质量.

<a id="S240"></a>
**Source:** p.13 S240  
**Type:** body  
**Confidence:** high

**Original:** [9] K. Panjvani, A.V. Dinh, K.A. Wahid, LiDARPheno - a low-cost LiDAR-Based 3D scanning system for leaf morphological trait extraction, Front. Plant Sci. 10 (2019), 2019. [10] Q. Xu, L. Cao, L. Xue, B. Chen, F. An, T. Yun, Extraction of leaf biophysical attributes based on a computer graphic-based algorithm using terrestrial laser scanning data, Remote Sens. 11 (2019). [11] B. Mildenhall, P.P. Srinivasan, M. Tancik, J.T. Barron, R. Ramamoorthi, R. Ng, NeRF: representing scenes as neural radiance fields for view synthesis. Commun, ACM 65 (2021) 99–106. [12] B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis, 3D gaussian splatting for realtime radiance field rendering, ACM Trans. Graph. 42 (2023). [13] H.B. Choi, J.K. Park, S.H. Park, T.S. Lee, NeRF-based 3D reconstruction pipeline for acquisition and analysis of tomato crop morphology, Front. Plant Sci. 15 (2024). [14] X. Yang, X. Lu, P. Xie, et al., PanicleNeRF: low-cost, high-precision In-Field phenotyping of rice panicles with smartphone, Plant Phenomics 6 (2024) 279. [15] A. Kirillov, E. Mintun, N. Ravi, et al., Segment anything, arXiv: 2304.02643 [cs. CV]. url: https://arxiv.org/abs/2304.02643, 2023. [16] F. Saeed, J. Sun, P. Ozias-Akins, Y.J. Chu, C.C. Li, PeanutNeRF: 3D radiance field for peanuts, in: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), IEEE Computer Society, Los Alamitos, CA, USA, 2023, pp. 6254–6263, https://doi.org/10.1109/CVPRW59228.2023.00665, 10.1109/CVPRW59228.2023.00665. [17] Y. Shen, H. Zhou, X. Yang, et al., Biomass phenotyping of oilseed rape through UAV multi-view oblique imaging with 3DGS and SAM model, Comput. Electron. Agric. 235 (2025) 110320. [18] L. Jiang, J. Sun, P.W. Chee, C. Li, L. Fu, Cotton3DGaussians: multiview 3D gaussian splatting for boll mapping and plant architecture analysis, Comput. Electron. Agric. 234 (2025) 110293. [19] D. Reis, J. Kupec, J. Hong, A. Daoudi, Real-time flying object detection with YOLOv8, arXiv: 2305.09972 (2024) [cs.CV]. url: https://arxiv.org/abs/2 305.09972. [20] J. Cen, J. Fang, C. Yang, et al., Segment any 3D gaussians, arXiv: 2312.00860 [cs. CV]. url: https://arxiv.org/abs/2312.00860, 2025. [21] T. Müller, A. Evans, C. Schied, A. Keller, Instant neural graphics primitives with a multiresolution hash encoding, ACM Trans. Graph. 41 (2022) 102:1–102:15. [22] K.G. Liakos, P. Busato, D. Moshou, S. Pearson, D. Bochtis, Machine learning in agriculture: a review, Sensors 18 (2018). [23] Y. Li, W. Wen, T. Miao, et al., Automatic organ-level point cloud segmentation of maize shoots by integrating high-throughput data acquisition and deep learning, Comput. Electron. Agric. 193 (2022) 106702. [24] Z. Ao, F. Wu, S. Hu, et al., Automatic segmentation of stem and leaf components and individual maize plants in field terrestrial LiDAR data using convolutional neural networks, The Crop Journal 10 (2022) 1239–1250. Crop phenotyping studies with application to crop monitoring. [25] D. Li, G. Shi, J. Li, et al., PlantNet: a dual-function point cloud segmentation network for multiple plant species, ISPRS J. Photogrammetry Remote Sens. 184 (2022) 243–263. [26] M. Peng, Y. Liu, I.A. Qadri, et al., Advanced image segmentation for precision agriculture using CNN-GAT fusion and fuzzy C-means clustering, Comput. Electron. Agric. 226 (2024) 109431. [27] J. Yan, X. Wang, Unsupervised and semi-supervised learning: the next frontier in machine learning for plant systems biology, Plant J. 111 (2022) 1527–1538. [28] H. Zhu, X. Liu, H. Zheng, L. Yang, X. Li, Z. Han, Identifying strawberry appearance quality based on unsupervised deep learning, Precis. Agric. 25 (2024) 614–632. [29] Y. Huang, A.E. Hussein, X. Wang, A. Bais, S. Yao, T. Wilder, Unsupervised domain adaptation with self-training for weed segmentation, Intelligent Systems with Applications 25 (2025) 200468. [30] J. Cen, J. Fang, Z. Zhou, et al., Segment anything in 3D with radiance fields, arXiv: 2304.12308 [cs.CV]. url: https://arxiv.org/abs/2304.12308, 2024. [31] Y. Liu, B. Hu, C.K. Tang, Y.W. Tai, SANeRF-HQ: segment anything for NeRF in high quality, in: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 3216–3226, https://doi.org/10.1109/ CVPR52733.2024.00310. [32] Y. Zhou, J. Gu, T.Y. Chiang, F. Xiang, H. Su, Point-SAM: promptable 3D segmentation model for point clouds, arXiv: 2406.17741 (2024) [cs.CV]. url: htt ps://arxiv.org/abs/2406.17741. [33] Ravi N, Gabeur V, Hu YT, et al. Sam 2: segment anything in images and videos. arXiv preprint arXiv:2408.00714 2024. [34] R. Khanam, M. Hussain, YOLOv11: an overview of the key architectural enhancements, arXiv: 2410.17725 (2024) [cs.CV]. url: https://arxiv.org/abs/2 410.17725. [35] Z. Wang, A. Bovik, H. Sheikh, E. Simoncelli, Image quality assessment: from error visibility to structural similarity, IEEE Trans. Image Process. 13 (2004) 600–612. [36] J.L. Sch€ onberger, J.M. Frahm, Structure-from-Motion revisited, in: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 4104–4113, https://doi.org/10.1109/CVPR.2016.445. [37] J. Sch€ onberger, E. Zheng, M. Pollefeys, J.M. Frahm, Pixelwise view selection for unstructured multi-view stereo 9907 (2016), https://doi.org/10.1007/978-3-31946487-9_31. [38] M. Tancik, E. Weber, E. Ng, et al., Nerfstudio: a modular framework for neural radiance field development, in: ACM SIGGRAPH 2023 Conference Proceedings. SIGGRAPH ’23, Association for Computing Machinery, Los Angeles, CA, USA, 2023, https://doi.org/10.1145/3588432.3591516, 10.1145/3588432.3591516. [39] M. Afham, I. Dissanayake, D. Dissanayake, A. Dharmasiri, K. Thilakarathna, R. Rodrigo, CrossPoint: self-supervised cross-modal contrastive learning for 3D point cloud understanding, in: 2022 IEEE/CVF Conference on Computer Vision and

**中文:** 参考文献条目保留原文，未翻译。

<a id="S241"></a>
### Author contributions
**Source:** p.13 S241  
**Type:** section  
**Confidence:** high

**Original:** Author contributions

**中文:** 作者贡献

<a id="S242"></a>
**Source:** p.13 S242  
**Type:** body  
**Confidence:** high

**Original:** W. Song conceived the research idea, designed and performed the majority of the experiments, and drafted the manuscript. H. Huang and Y. Sun conducted the cultivation of rice and wheat and contributed to the critical review and revision of the manuscript. F. Qu, J. Zhang, L. Fang, Y. Hao, and C. Peng were responsible for data curation and contributed to the investigation aspects of the study.

**中文:** 设计了研究想法,设计和执行了大多数实验,并起草了手稿.黄和孙领导了米和小麦的种植,并为手稿的批判性审查和修订做了贡献.................................................................................................................................................

<a id="S243"></a>
### Funding
**Source:** p.13 S243  
**Type:** section  
**Confidence:** high

**Original:** Funding

**中文:** 基金资助

<a id="S244"></a>
**Source:** p.13 S244  
**Type:** body  
**Confidence:** high

**Original:** This research was supported by the National Key Research and Development Program of China (Grant Number 2023YFD1901003) and the Strategic Priority Research Program of the Chinese Academy of Sciences (Grant XDA28120402).

**中文:** 这项研究得到了中国国家关键研究和发展计划 (补贴号 2023YFD1901003) 和中国科学院战略优先研究计划 (补贴XDA28120402) 的支持.

<a id="S245"></a>
### Data availability
**Source:** p.13 S245  
**Type:** section  
**Confidence:** high

**Original:** Data availability

**中文:** 数据可用性

<a id="S246"></a>
**Source:** p.13 S246  
**Type:** body  
**Confidence:** high

**Original:** The data are freely available upon reasonable request. Code is available at https://github.com/Vincent-Songwentao/IPENS-Code.git.

**中文:** 根据合理的要求,数据可以自由获得.代码可在http://github.com/Vincent-Songwentao/IPENS-Code.git.上找到.

<a id="S247"></a>
### Declaration of competing interest
**Source:** p.13 S247  
**Type:** section  
**Confidence:** high

**Original:** Declaration of competing interest

**中文:** 利益冲突声明

<a id="S248"></a>
**Source:** p.13 S248  
**Type:** body  
**Confidence:** high

**Original:** The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

**中文:** 作者表示,他们没有任何已知的竞争性财务利益或个人关系,这些关系似乎会影响本文报告的工作.

<a id="S249"></a>
### Appendix A. Supplementary data
**Source:** p.13 S249  
**Type:** section  
**Confidence:** high

**Original:** Appendix A. Supplementary data

**中文:** 附录 A. 补充数据

<a id="S250"></a>
**Source:** p.13 S250  
**Type:** body  
**Confidence:** high

**Original:** Supplementary data to this article can be found online at https://doi. org/10.1016/j.plaphe.2025.100106.

**中文:** 对于这篇文章的补充数据可以在网上找到http://doi.org/10.1016/j.plaphe.2025.100106.

<a id="S251"></a>
### References
**Source:** p.13 S251  
**Type:** section  
**Confidence:** high

**Original:** References

**中文:** 参考文献

<a id="S252"></a>
**Source:** p.13 S252  
**Type:** reference  
**Confidence:** high

**Original:** [1] P. Ying-Hong, Analysis of concepts and categories of plant phenome and phenomics, Acta Agron. Sin. 41 (175) (2015) 175. [2] H.J. Liu, J. Yan, Crop genome-wide association study: a harvest of biological relevance, Plant J. 97 (2019) 8–18. [3] J.L. Araus, J.E. Cairns, Field high-throughput phenotyping: the new crop breeding frontier, Trends Plant Sci. 19 (2014) 52–61. [4] J.L. Araus, S.C. Kefauver, M. Zaman-Allah, M.S. Olsen, J.E. Cairns, Translating high-throughput phenotyping into genetic gain, Trends Plant Sci. 23 (2018) 451–466. [5] C. Zhao, Y. Zhang, J. Du, et al., Crop phenomics: current status and perspectives, Front. Plant Sci. 10 (2019), 2019. [6] M.S. Akhtar, Z. Zafar, R. Nawaz, M.M. Fraz, Unlocking plant secrets: a systematic review of 3D imaging in plant phenotyping techniques, Comput. Electron. Agric. 222 (2024) 109033. [7] Li J, Qi X, Nabaei SH, et al. A survey on 3D reconstruction techniques in plant phenotyping: from classical methods to neural radiance fields (NeRF), 3D gaussian splatting (3DGS), and beyond. 2025. arXiv: 2505.00737 [eess.IV]. url: htt ps://arxiv.org/abs/2505.00737. [8] Z. Li, R. Guo, M. Li, Y. Chen, G. Li, A review of computer vision technologies for plant phenotyping, Comput. Electron. Agric. 176 (2020) 105672.

**中文:** 参考文献条目保留原文，未做逐条翻译。


## Page 14

<a id="S253"></a>
**Source:** p.14 S253  
**Type:** reference  
**Confidence:** high

**Original:** Pattern Recognition (CVPR), 2022, pp. 9892–9902, https://doi.org/10.1109/ CVPR52688.2022.00967. [40] Y. Yue, S. Mahadevan, J. Schult, et al., AGILE3D: attention guided interactive multi-object 3D segmentation, arXiv: 2306.00977 (2024) [cs.CV]. url: htt ps://arxiv.org/abs/2306.00977.

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S254"></a>
**Source:** p.14 S254  
**Type:** reference  
**Confidence:** high

**Original:** [41] M. Kolodiazhnyi, A. Vorontsova, A. Konushin, D. Rukhovich, OneFormer3D: one transformer for unified point cloud segmentation, in: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 20943–20953,

**中文:** 参考文献条目保留原文，未做逐条翻译。


## 阅读提示

- IPENS 的核心思想是把 SAM2 在多视角图像序列上得到的 2D 掩膜，通过 NeRF 的辐射场表示和可微渲染约束提升到 3D 空间。
- 与完全监督 3D 分割相比，IPENS 的价值在于减少精细 3D 标注依赖；与单目标交互式方法相比，它强调一次交互中的多目标协同提取。
- 结果部分要重点对照 Table 2、Fig. 5、Fig. 7 和 Table 4：水稻籽粒分割相对困难，小麦穗部与叶片表型估计表现更强。
- 局限性主要来自显存随同步目标数增加而快速增长，以及野外光照/风扰对 NeRF 重建质量的影响。
