---
title: "From 3DGS scenes to plant traits: a scalable extraction and segmentation framework for muskmelon phenotyping"
authors: "Jing-Heng Lin"
journal: "Frontiers in Plant Science"
doi: 10.3389/fpls.2026.1783465
source_pdf: /data/fj/F2DMAS/00参考文章/fpls-17-1783465.pdf
generated: 2026-05-26
reader_type: bilingual_source_grounded_markdown
---

# From 3DGS scenes to plant traits: a scalable extraction and segmentation framework for muskmelon phenotyping

**作者：** Jing-Heng Lin

**来源：** Frontiers in Plant Science; DOI: 10.3389/fpls.2026.1783465

**说明：** 本文件为全文中英对照阅读稿。中文为机器初译并经过领域术语规则校正；双栏、公式、表格和复杂多子图区域的低置信点记录在 `translation_notes.md`。

## 页面/章节索引

- [1 Introduction](#s012) — p.1
- [2 Materials and methods](#s019) — p.3
- [2.1 Pipeline overview](#s020) — p.3
- [2.2 Data acquisition and 3DGS scene](#s022) — p.3
- [2.2.1 Data acquisition protocol and dataset](#s024) — p.3
- [2.2.2 Structure-from-Motion and 3DGS scene](#s042) — p.4
- [2.3 Plant gaussians extraction (LCR-GS)](#s044) — p.4
- [2.3.1 Plant extraction algorithm overview](#s046) — p.4
- [2.3.2 Input seeding for LCR-GS initialization](#s054) — p.5
- [2.3.4 Geometric clustering and retention filtering](#s063) — p.5
- [2.3.3 Lifting 2D cues into the 3DGS domain](#s067) — p.5
- [2.4 Organ-Level instance segmentation](#s089) — p.6
- [2.4.1 Input normalization and organ instance](#s090) — p.6
- [2.3.5 Chromatic refinement and instance](#s095) — p.6
- [2.5.2 Trait definition and computation](#s104) — p.7
- [2.4.2 Remapping segmentations to the 3DGS](#s106) — p.7
- [2.5 Phenotypic trait quantification](#s117) — p.7
- [2.5.1 Metric scaling and coordinate frame](#s118) — p.7
- [3 Results and discussion](#s121) — p.7
- [3.1 Optimizing 3DGS scene reconstruction](#s122) — p.7
- [3.1.1 Efficient frame selection for sTable 3DGS](#s123) — p.7
- [3.2 Plant instance extraction from 3DGS](#s128) — p.8
- [3.2.1 Effect of input cue count on plant](#s130) — p.8
- [3.1.2 Reconstruction quality comparison](#s132) — p.8
- [3.2.2 Ablation analysis of LCR-GS components](#s181) — p.9
- [3.2.3 Quantitative and qualitative evaluation of](#s190) — p.10
- [3.3 Organ-level instance segmentation](#s198) — p.11
- [3.4 Phenotypic trait validation and](#s219) — p.12
- [3.5 Scope, limitations, and future directions](#s260) — p.14
- [4 Conclusion](#s305) — p.15
- [Funding](#s310) — p.16
- [Acknowledgments](#s312) — p.16
- [Conflict of interest](#s314) — p.16
- [Author contributions](#s319) — p.16
- [Data availability statement](#s323) — p.16
- [References](#s324) — p.16

## 术语表

| English | 中文 |
| --- | --- |
| plant phenotyping | 植物表型/植物表型分析 |
| 3D Gaussian Splatting (3DGS) | 三维高斯泼溅（3DGS） |
| Neural Radiance Fields (NeRF) | 神经辐射场（NeRF） |
| Structure from Motion (SfM) | 运动恢复结构（SfM） |
| COLMAP | COLMAP |
| point cloud | 点云 |
| instance segmentation | 实例分割 |
| trait extraction | 性状提取 |
| PSNR / SSIM / LPIPS | PSNR / SSIM / LPIPS 指标 |

## 全文中英对照


## Page 1

<a id="S001"></a>
**Source:** p.1 S001  
**Type:** body  
**Confidence:** high

**Original:** Kai Huang, Jiangsu Academy of Agricultural Sciences, China

**中文:** 海黄,江苏农业科学院,中国

<a id="S002"></a>
**Source:** p.1 S002  
**Type:** body  
**Confidence:** high

**Original:** From 3DGS scenes to plant traits: a scalable extraction and segmentation framework for muskmelon phenotyping

**中文:** 从3DGS场景到植物特征:可扩展的桃子型的提取和分割框架

<a id="S003"></a>
**Source:** p.1 S003  
**Type:** body  
**Confidence:** high

**Original:** Koji Noshita, Kyushu University, Japan Adarsh Krishnamurthy, Iowa State University, United States

**中文:** 科吉·诺希塔,九州大学,日本阿达什·克里斯纳穆尔蒂,爱荷华州立大学,美国

<a id="S004"></a>
**Source:** p.1 S004  
**Type:** body  
**Confidence:** high

**Original:** Jing-Heng Lin and Ta-Te Lin* Department of Biomechatronics Engineering, National Taiwan University, Taipei, Taiwan

**中文:** 恒林和塔泰林* 生物天气工程系,国家台湾大学,台北,台湾

<a id="S005"></a>
**Source:** p.1 S005  
**Type:** body  
**Confidence:** high

**Original:** *CORRESPONDENCE

**中文:** *CORRESPONDENCE 对于这些问题,我们必须要做一些事情.

<a id="S006"></a>
**Source:** p.1 S006  
**Type:** body  
**Confidence:** high

**Original:** Ta-Te Lin m456@ntu.edu.tw RECEIVED 09 January 2026 REVISED 21 March 2026 ACCEPTED 27 March 2026

**中文:** 塔-泰林 m456@ntu.edu.tw 收到 09月2026日复制 21月2026日接受 27月2026

<a id="S007"></a>
**Source:** p.1 S007  
**Type:** body  
**Confidence:** high

**Original:** CITATION

**中文:** 引用

<a id="S008"></a>
**Source:** p.1 S008  
**Type:** body  
**Confidence:** high

**Original:** Lin J-H and Lin T-T (2026) From 3DGS scenes to plant traits: a scalable extraction and segmentation framework for muskmelon phenotyping. Front. Plant Sci. 17:1783465. doi: 10.3389/fpls.2026.1783465

**中文:** 林和林特 (2026) 从3DGS场景到植物特征:可扩展的桃体异型化提取和分割框架. 前面.植物科学 17:1783465. doi: 10.3389/fpls.2026.1783465

<a id="S009"></a>
**Source:** p.1 S009  
**Type:** body  
**Confidence:** high

**Original:** © 2026 Lin and Lin. This is an openaccess article distributed under the terms of the Creative Commons Attribution License (CC BY). The use, distribution or reproduction in other forums is permitted, provided the original author(s) and the copyright owner(s) are credited and that the original publication in this journal is cited, in accordance with accepted academic practice. No use, distribution or reproduction is permitted which does not comply with these terms.

**中文:** © 2026 林和林.这是一个开放版权文章,在Creative Commons Attribution License (CC BY) 条款下发行.在其他论坛中使用,发行或复制是允许的,只要原作者和版权所有者被认可,并且根据接受的学术做法引用本期刊的原始出版物.不符合这些条款的使用,发行或复制是允许的.

<a id="S010"></a>
**Source:** p.1 S010  
**Type:** body  
**Confidence:** medium

**Original:** Automated quantification of plant-level development from multi-plant greenhouse scenes requires separating individual plants from shared scene-level reconstructions and quantifying organ-level development, a challenge that single-plant acquisition workflows do not directly address. This study presents an end-to-end phenotyping pipeline built on 3D Gaussian Splatting (3DGS) and a post-reconstruction extraction framework, LCR-GS, designed to isolate plant instances from full greenhouse scenes without scene-specific model retraining. LCR-GS integrates zero-shot 2D cues with multi-view lifting, geometric clustering, and chromatic refinement to convert large scene-level reconstructions (~2M Gaussians) into compact per-plant subsets (~16K Gaussians). Experiments on greenhouse-grown muskmelon at the early vegetative stage demonstrate high plant-extraction precision (0.933) and strong organ-level instance segmentation (mean AP50 = 0.924). Plant height and leaf count are validated against manual measurements (height R² = 0.98, RMSE = 1.88 cm; leaf count R² = 0.86), whereas additional morphological traits, including leaf area, leaf area index, mean internode length, and stem node count, are reported as pipeline-derived descriptors for within-cohort comparison. By decoupling semantic inference from reconstruction, the pipeline reduces scene-scale data by over 99% and provides a practical route to derive compact per-plant 3D representations from multi-plant greenhouse imagery for downstream organ-level analysis. KEYWORDS

**中文:** 自动化量化植物级发展与多植物温室场景需要将单个植物分离于共享场景级重建和量化器官级发展,这是单植物收购工作流无法直接解决的挑战.本研究介绍了基于3D高斯人化 (3DGS) 的端到端型型化管道和后重建提取框架,LCR-GS,旨在将植物实例从完整的温室场景中隔离,而不需要重新设计场景特定模型.LCR-GS将零镜头2D线索与多视图升级,几何聚合和染色精炼整合起来,将大型场景级重建 (~2M高斯人) 转化为紧的每种子 (~16K高斯人). 温室培养的桃在早期植被阶段的实验表明,高植物提取精度 (0.933) 和强大的器官级实例分割 (平均AP50 = 0.924).植物高度和叶子数被验证了与手动测量 (高度R2 = 0.98,RMSE = 1.88厘米;叶子数R2 = 0.86),而包括叶片面积,叶片面积指数,平均内线路长度和干细胞数量的其他形态特征,被报告为内部协会比较的线路衍生描述.通过从重建中进行的语义推断,该管道将规模数据减少了99%以上,并提供了从温室中获得多层次分析的器官管道图像的复杂的每种植物3D表示的实际途径.

<a id="S011"></a>
**Source:** p.1 S011  
**Type:** body  
**Confidence:** high

**Original:** 3D gaussian splatting, 3D reconstruction, instance segmentation, plant phenotyping, trait extraction

**中文:** 3D高斯人,3D重建,实例分割,植物表型,特征提取

<a id="S012"></a>
### 1 Introduction
**Source:** p.1 S012  
**Type:** section  
**Confidence:** high

**Original:** 1 Introduction

**中文:** 【标题暂译】1 Introduction

<a id="S013"></a>
**Source:** p.1 S013  
**Type:** body  
**Confidence:** high

**Original:** Phenotypic characterization of high-value horticultural crops is essential for optimizing yield, quality, and management decisions in modern agriculture. Although traditional manual measurements remain accurate, they are labor-intensive and inherently limited in throughput, restricting their use in large-scale breeding or production programs (Araus et al., 2018). Image-based phenotyping systems have therefore been widely adopted as an effective alternative, enabling rapid, non-destructive trait quantification across development stages (Li et al., 2014; Wang et al., 2025). However, 2D imagery cannot fully recover the volumetric structure needed to resolve overlapping foliage, occluded organs, and plant-level geometry in shared multi-plant scenes. In greenhouse-grown muskmelon, neighboring leaves, vines, and support structures can overlap in image space even under controlled

**中文:** 高价值园林作物的型化特征对于优化现代农业产量,质量和管理决策至关重要.虽然传统的手动测量仍然是准确的,但它们需要劳动力和产量固然有限,限制了它们在大型育种或生产计划中使用 (Araus et al., 2018).因此,基于图像的型化系统已被广泛采用为有效的替代品,使得开发阶段的快速,非破坏性特征量化 (Li et al., 2014; Wang et al., 2025).然而,2D图像无法完全恢复解决重叠的树叶,被遮蔽的器官和植物级几何学所需的体积结构. 在温室培养的桃中,邻近的叶子,葡萄树和支结构可以在图像空间中重叠,即使在受控制的情况下


## Page 2

<a id="S014"></a>
**Source:** p.2 S014  
**Type:** body  
**Confidence:** high

**Original:** 10.3389/fpls.2026.1783465

**中文:** 10.3389/fpls.2026.1783465

<a id="S015"></a>
**Source:** p.2 S015  
**Type:** body  
**Confidence:** high

**Original:** operational complexity, and limited scalability make it unsuitable for large multi-plot phenotyping programs. In contrast, RGB-based reconstruction, typically performed using Structure from Motion (SfM) tools such as COLMAP (Schönberger and Frahm, 2016), offers a cost-effective and flexible alternative for deployment on UGVs and UAVs. Recent advances in neural rendering have further improved reconstruction quality: Neural Radiance Fields (NeRF) (Mildenhall et al., 2021) provides photorealistic novel view synthesis and improved geometric consistency in agricultural applications (Arshad et al., 2024; Choi et al., 2024). However, NeRF’s continuous volumetric representation lacks explicit geometry, creating a fundamental mismatch with downstream pipelines that require measurable, discrete structures for segmentation and trait extraction. 3D Gaussian Splatting (3DGS) addresses this gap by representing scenes as spatially explicit Gaussian primitives (Kerbl et al., 2023), offering NeRF-level fidelity while maintaining compatibility with point-based analysis. Recent plant-focused adaptations (Jiang et al., 2025; Shen et al., 2025) further highlight 3DGS’s suitability for phenotyping applications. In multi-plant phenotyping applications of 3D Gaussian Splatting (3DGS), two practical challenges arise. First, shared scene reconstructions contain neighboring plants and surrounding structures, making it difficult for downstream workflows, which typically assume clean, single-plant geometry, to operate reliably (Harandi et al., 2023). Second, and more specific to 3DGS, contemporary scene reconstructions consist of millions of Gaussian primitives (Fang and Wang, 2024; Li et al., 2025b). This high density, combined with noise introduced by sparse or uneven view coverage, makes direct point-based feature extraction computationally expensive and reduces geometric consistency. Recent work has attempted to address these issues by injecting 2D semantic cues during 3DGS optimization to obtain promptable or openworld segmentation. Gaussian Grouping lifts multi-view 2D features into the Gaussian space and learns instance groupings during reconstruction (Ye et al., 2024), whereas SAGA (Segment Any 3D Gaussians) distills 2D foundation models into Gaussian features to enable prompTable 3D segmentation at inference (Cen et al., 2025). While effective for general 3D scene editing, these training-integrated approaches couple semantic learning with reconstruction, limiting flexibility and increasing computational cost. These limitations motivate a post-reconstruction strategy that decouples reconstruction from semantic assignment so that a reconstructed scene can be reused for plant extraction without modifying the 3DGS model itself. The validated setting in this study occupies a middle regime between two relevant reference points. One recent object-centric 3DGS phenotyping workflow for strawberry isolates individual plants before reconstruction to obtain clean single-plant models (Li et al., 2026). At the other end, recent surveys of plant 3D reconstruction and 3DGS-based phenotyping note that dense, highly entangled canopies remain challenging for current pipelines (Li et al., 2025a). Our focus lies between these extremes: multiple neighboring greenhouse plants share one scene reconstruction, canopy entanglement remains limited, and downstream analysis still requires that the shared reconstruction be decomposed into individual plant units. This structured shared-scene extraction

**中文:** 运作复杂性和有限的可扩展性使其不适合大型多情节的型方案.相反,RGB基于的重建,通常使用结构从运动 (SfM) 工具 (如COLMAP,Schönberger and Frahm, 2016),为UGV和无人机部署提供了经济高效和灵活的替代方案.神经染的最新进展进一步改善了重建质量:神经辐射场 (Neural Radiance Fields) (Mildenhall et al., 2021) 提供了光学实质的新视图合成和改善了农业应用中的几何一致性 (Arshad et al., 2024; Choi et al., 2024).然而,NeRF的连续体积表现缺乏明确的几何,从而创造了基本的不匹配,需要测量,分辨和分辨的下游结构. 盖斯基斯分化 (3DGS) 通过将场景表示为空间明确的盖斯基原始物 (Kerbl et al., 2023),提供NeRF级忠诚性,同时保持与点分析的兼容性.最近的植物专注改造 (Jiang et al., 2025; Shen et al., 2025) 进一步突出了3DGS适用于型应用的适用性. 在3D盖斯基斯分化 (3DGS) 的多植物型应用中,出现了两个实际挑战.首先,共享场景重建包含邻近植物和周边结构,使下游工作流,通常假设清洁的单植物几何,难以可靠运行 (Harandi et al., 2023).第二,更具体地说,3DGS,包括数百万个盖斯基斯基原始的当代场景重建 (Fang Wang, 2024;;;;;). 这种高密度,加上稀疏或不均的视图覆盖带来的噪音,使直接点性功能提取比较昂贵,降低了几何一致性.最近的研究试图通过在3DGS优化过程中注入2D语义线索来解决这些问题,以获得即时或开放世界的分割.高斯集团将多视图2D功能将在高斯空间中提升,并在重建过程中学习实例组 (Ye et al., 2024),而SAGA (Segment Any 3D Gaussians) 将2D基础模型分化到高斯功能中,以实现即时的3D分割 (Cen et al., 2025). 虽然对于一般的3D场景编辑有效,但这些训练集成方法将语义学习与重建结合起来,限制了灵活性和计算成本的增加.这些限制激励了后重建策略,将重建与语义任务分离,以便重建的场景可以在不修改3DGS模型本身的情况下重建用于植物提取.本研究中的验证设置占据了两个相关参考点之间的中间状态.最近的一项对象中心的3DGS表型工作流程为草在重建之前将单个植物隔离,以获得清洁的单个植物模型 (Li等人, 2026). 另一方面,近期对工厂3D重建和3DGS基表型的调查指出,密集,高度纠的顶仍然是当前管道的挑战 (Li et al., 2025a).我们的重点在这些极端之间:多个邻近温室工厂共享一个场景重建,顶纠仍然有限,下游分析仍然要求共享重建被分解成单个工厂单位.

<a id="S016"></a>
**Source:** p.2 S016  
**Type:** body  
**Confidence:** high

**Original:** acquisition, making it difficult to separate individual plants and recover organ-level geometry from 2D views alone. Many important traits, including canopy architecture, biomass distribution, and organ-level geometry, therefore require three-dimensional reconstruction for accurate assessment (Akhtar et al., 2024). Recent progress in mobile platforms, including unmanned ground vehicles (UGVs) and unmanned aerial systems (UAS), has expanded access to large-scale 3D phenotyping (Feng et al., 2021; Rui et al., 2024). Achieving useful plant-level analysis from such acquisitions, however, still depends on 3D reconstruction and analysis pipelines that can process shared multi-plant scenes while preserving individualplant resolution. Extracting phenotypic traits from 3D plant reconstructions requires robust segmentation and geometric analysis. Recent progress in point-based deep learning has substantially advanced this capability. Foundational feature extractors such as PointNet/ PointNet++ (Qi et al., 2017a, 2017) and DGCNN (Wang et al., 2019) established methods for permutation-invariant feature learning, while Transformer-based backbones (Wu et al., 2024; Zhao et al., 2021) further improved the modeling of complex geometric relationships. Complementing these feature extractors, generalpurpose segmentation architectures, such as PointGroup’s offset clustering (Jiang et al., 2020), provided reliable mechanisms for separating discrete plant organs. Together, these technologies form the basis for many plant-specific 3D phenotyping applications. Building on these general-purpose architectures, plant-specific 3D analysis methods have evolved into three major functional categories. First, semantic segmentation classifies point clouds into organ types for species-level architectural analysis (Li et al., 2022a; Shi et al., 2019). Second, instance segmentation, which separates individual organs, is critical for tasks such as leaf counting, organ tracking, and per-organ trait measurement. This capability has been demonstrated across a wide range of crops, including maize (Li et al., 2022b), rapeseed (Du et al., 2023), and wheat (Ghahremani et al., 2021). Alongside these, other specialized techniques such as graph-based topological reasoning (Mirande et al., 2022) and multi-view fusion (Shi et al., 2019) have been developed to address occlusion and structural ambiguity in complex plant scenes. Third, geometric processing complements segmentation by directly extracting morphological traits: skeleton-based methods quantify branching topology and internode spacing (Cao et al., 2010), while mesh-based approaches estimate leaf area and surface curvature (Boukhana et al., 2022). Despite their diversity, these methods all rely on a critical but often implicit assumption: the availability of clean, pre-isolated, single-plant point clouds. While this condition is easily met in controlled environments, it becomes a major bottleneck in structured multi-plant greenhouse scenes, where neighboring plants and surrounding greenhouse structures are reconstructed together. This extraction step represents an intermediate bottleneck in phenotyping pipelines, where scene-level reconstructions must be decomposed into plant-level units before organ-level segmentation and trait computation can be applied. The practical effectiveness of 3D segmentation, therefore, depends critically on robust and accessible reconstruction workflows. Although LiDAR remains the precision benchmark, its cost,

**中文:** 因此,许多重要的特征,包括天花板架构,生物质分布和器官层次几何学,因此需要进行三维重建以进行精确评估 (Akhtar et al., 2024).移动平台的最新进展,包括无人机地面车 (UGV) 和无人机空系统 (UAS),已扩大了对大规模3D异型的访问 (Feng et al., 2021;Rui et al., 2024).从这些收购中获得有用的植物层次分析,然而,仍然取决于3D重建和分析管道,可以处理多种植物场景并保存个别的植物.从3D植物重建中提取异型特征需要强大的分割和几何分析. 最近的点基深度学习进展已经大幅提升了这一能力.像PointNet/PointNet++ (Qi et al., 2017a, 2017) 和 DGCNN (Wang et al., 2019) 等基础特征提取器建立了变量不变特征学习的方法,而基于变压器的背骨 (Wu et al., 2024; Zhao et al., 2021) 进一步改善了复杂几何关系的建模.补充这些特征提取器,通用分割结构,如PointGroups offset clustering (Jiang et al., 2020),为分离植物器官提供了可靠的离散机制.这些技术构成了许多植物特定的3D型应用的基础.基于这些通用结构,植物特定的3D分析方法已经演化为三个主要的功能类别. 首先,语义分类将点云分为物种级结构分析的器官类型 (Li et al., 2022a; Shi et al., 2019).第二,分类实例,分离个体器官,对于诸如叶子计数,器官跟踪和每个器官特征测量等任务至关重要.这种能力已被证明在各种种植中,包括玉米 (Li et al., 2022b),大黄蜂 (Du et al., 2023),小麦 (Ghahremani et al., 2021).除此之外,其他专业技术,如图形基础上的拓推理 (Mirande et al., 2022) 和多视图融合 (Shi et al., 2019) 已经开发出解决复杂植物场景中的和结构模糊性. 第三,几何处理通过直接提取形态特征来补充分割:骨式方法量化了分分拓和内部码间隔 (Cao et al., 2010),而网状方法估计了叶片面积和表面曲线 (Boukhana et al., 2022).尽管这些方法有多样性,但这些方法都依赖于一个关键但经常隐含的假设:清洁的,预隔离的,单植物点云的可用性.虽然这种条件在受控环境中很容易满足,但在结构化的多植物温室场景中,它成为一个主要的瓶,其中邻近植物和周围温室结构重建在一起. 这一步的提取代表了表型造型管道中一个中间瓶,在器官级分类和特征计算之前,场景级重建必须分解成植物级单元.因此,3D分类的实际效果,非常依赖于强大可访问的重建工作流程.尽管LiDAR仍然是精度基准,但它的成本,


## Page 3

<a id="S017"></a>
**Source:** p.3 S017  
**Type:** body  
**Confidence:** high

**Original:** 10.3389/fpls.2026.1783465

**中文:** 10.3389/fpls.2026.1783465

<a id="S018"></a>
**Source:** p.3 S018  
**Type:** body  
**Confidence:** high

**Original:** problem is narrower than dense-canopy phenotyping, but it is not addressed by object-centric single-plant workflows and remains underexplored in current 3DGS phenotyping studies. To address this shared-scene decomposition problem, we propose LCR-GS (Lift, Cluster, Refine for Gaussian Splat), a post-reconstruction framework that operates on fixed 3DGS reconstructions and introduces semantics via zero-shot 2D foundation models. The pipeline employs YOLO-World detections and SAM (Segment Anything Model) with minimal operator seeding on a small subset of views to initialize a reusable pool of multi-view cues (Cheng et al., 2024; Kirillov et al., 2023). The seeded cues are then lifted to 3D Gaussians with multi-view projection and consistency filtering, followed by geometric clustering that consolidates spatially coherent groups and isolates target plant instances while suppressing background structures. The resulting per-plant representations are then used for point-based organ-level instance segmentation and phenotypic trait estimation. In this study, our objective is to establish a bounded workflow that bridges scene-level 3D reconstruction and plant-level trait quantification for structured multi-plant greenhouse phenotyping. We make three main contributions toward this goal. First, we develop an end-to-end phenotyping pipeline based on three-dimensional Gaussian Splatting that reconstructs full greenhouse scenes while preserving per-plant spatial resolution for downstream organ-level analysis. Second, we introduce LCR-GS, a post-reconstruction extraction framework that decouples semantic inference from 3DGS optimization. LCR-GS requires only minimal operator seeding at initialization and then automatically performs multi-view lifting, geometric clustering, and refinement to convert structured multi-plant greenhouse scenes into clean, instance-level plant representations. This design enables reconstruction reuse, reduces annotation effort, and transforms scene-scale models into analysisready point sets. Third, we show that these extracted plant instances support point-based organ segmentation and bounded quantitative trait estimation for greenhouse-grown muskmelon. Validation on early-vegetative greenhouse muskmelon shows that the proposed framework provides a practical route from 3DGS scene reconstruction to organ-level trait quantification within this study setting.

**中文:** 为了解决这个共享场景分解问题,我们提出了LCR-GS (Lift, Cluster, Refine for Gaussian Splat),一个后修建框架,在固定的3DGS重建上运行,并通过零射2D基础模型引入语义学.该管道采用YOLO-World检测和SAM (Segment Anything Model) 进行,并且对一个小小小的视角小组进行最小操作,以启动可重复使用的多视线索 (Cheng et al., 2024; Kirillov et al., 2023). 然后,种子线索被通过多视图投影和一致性过将提高到3D高斯人,然后再进行几何聚合,将空间一致的组合集成并将目标植物实例分离起来,同时抑制背景结构.结果的植物性表示然后用于基于点的器官水平实例分类和现象特征估算.在本研究中,我们的目标是建立一个有界限的工作流程,可以实现场景级3D重建和结构化多植物温室表型的植物级定量化.我们为这一目标做出了三个主要贡献. 首先,我们开发了一个基于三维高斯式分化的基因型的终端型型的管道,重建了整个温室场景,同时保留了每种植物的空间分辨率,以便下游器官水平分析.第二,我们引入了LCR-GS,一个后重建提取框架,从3DGS优化中脱离了语义推理.LCR-GS需要在初始化时只进行最小的操作员种植,然后自动执行多视图升级,几何聚合和精炼,以将结构化多植物温室场景转化为清洁的实例级植物表示.这项设计使重建重新使用,减少了标注的努力,并将规模模型转化为准备的分析点. 第三,我们表明,这些提取的植物实例支持了基于点的器官分类和温室培养的桃的有限量化特征估计.早期植物性温室桃的验证表明,拟议的框架提供了从3DGS场景重建到在本研究环境中的器官水平特征量化的实际路径.

<a id="S019"></a>
### 2 Materials and methods
**Source:** p.3 S019  
**Type:** section  
**Confidence:** high

**Original:** 2 Materials and methods

**中文:** 【标题暂译】2 Materials and methods

<a id="S020"></a>
### 2.1 Pipeline overview
**Source:** p.3 S020  
**Type:** section  
**Confidence:** high

**Original:** 2.1 Pipeline overview

**中文:** 【标题暂译】2.1 Pipeline overview

<a id="S021"></a>
**Source:** p.3 S021  
**Type:** body  
**Confidence:** high

**Original:** pipeline for structured multi-plant greenhouse phenotyping, which consists of four stages: (1) multi-view RGB acquisition and Structure-from-motion, followed by 3D Gaussian Splatting to obtain a fixed scene reconstruction (Section 2.2); (2) the core LCR-GS post reconstruction extraction that lifts two dimensional cues to three-dimensional Gaussians, clusters, and refines the result to isolate individual plants (Section 2.3); (3) organ level instance segmentation on exported point clouds, with segmentation results remapped to the 3DGS domain (Section 2.4); and (4) trait computation on the segmented instances (Section 2.5). The following subsections first describe data acquisition and reconstruction, then present the LCR-GS extraction procedure, and finally detail the segmentation and trait computation modules.

**中文:** 结构化多植物温室表型的管道包括四个阶段: (1) 多视图RGB获取和 Structure-from-motion,随后是3D高斯人分光,以获得固定场景重建 (第二节); (2)核心LCR-GS后重建提取,将二维线索提高到三维高斯人,集群,并精细化结果以分离单个植物 (第二节); (3) 机器级实例分割化在出口点云上,将分割结果重新映射到3DGS域 (第二节);以及 (4) 分割化实例的特征计算 (第二节).下列部分首先描述了数据获取和重建,然后介绍了LCR-GS提取程序,最后介绍了分割化和特征计算模块.

<a id="S022"></a>
### 2.2 Data acquisition and 3DGS scene
**Source:** p.3 S022  
**Type:** section  
**Confidence:** high

**Original:** 2.2 Data acquisition and 3DGS scene

**中文:** 【标题暂译】2.2 Data acquisition and 3DGS scene

<a id="S023"></a>
**Source:** p.3 S023  
**Type:** body  
**Confidence:** high

**Original:** reconstruction

**中文:** 建设重建

<a id="S024"></a>
### 2.2.1 Data acquisition protocol and dataset
**Source:** p.3 S024  
**Type:** section  
**Confidence:** high

**Original:** 2.2.1 Data acquisition protocol and dataset

**中文:** 【标题暂译】2.2.1 Data acquisition protocol and dataset

<a id="S025"></a>
**Source:** p.3 S025  
**Type:** body  
**Confidence:** high

**Original:** Greenhouse data were collected at the National Taiwan University Experimental Farm in a muskmelon (Cucumis melo L.) greenhouse in August 2024. Video was recorded with an Insta360 One RS equipped with the Ultrawide 80 lens at 3840×2160 and 30 fps. Electronic stabilization was disabled to preserve geometric consistency. For each crop row, two passes were performed: an outbound pass at approximately 0.5 m camera height and a return pass at approximately 1.5 m, ensuring dense multi-view coverage of the canopy. Plants were arranged in a fixed grid layout with five plants per row, a within-row spacing of 72 cm, and a between-row spacing of approximately 1.0-1.2 m. The complete dataset is organized into two distinct subsets, as summarized in Table 1.

**中文:** 在2024年8月在台湾国立大学温室实验农场收集了温室数据,在一家瓜温室 (Cucumis melo L.) 中进行了录像.视频由配备了Ultrawide 80镜头的Insta360 One RS在3840×2160和30fps的速度上录制.电子稳定性被禁用以保持几何一致性.对于每个作物行,进行了两次通过:大约0.5米摄像头高度的出发通过和大约1.5米的返回通过,确保了大棚的密集多视图覆盖.植物在固定网格布局中配置,每行有五个植物,排间隔72厘米,排间间间间间距大约1.0-1.2米.完整数据集分为两个不同的子集,如表 1. 概述.

<a id="S026"></a>
**Source:** p.3 S026  
**Type:** body  
**Confidence:** high

**Original:** Overview of the end-to-end phenotyping pipeline based on 3D Gaussian Splatting (3DGS).

**中文:** 总体来看,基于3D高斯人分化 (3DGS) 的端到端表型管道.


## Page 4

<a id="S027"></a>
**Source:** p.4 S027  
**Type:** body  
**Confidence:** high

**Original:** 10.3389/fpls.2026.1783465

**中文:** 10.3389/fpls.2026.1783465

<a id="F002"></a>
### Fig. 2. 流程图展示了完整的数据流：从左侧的种子点准备，到右侧三阶段核心算法（提升、聚类、精修），最终导出三维植株模型。
**Placed near:** p.4  
**Source:** p.4 manual-layout  
**Crop confidence:** high

![Fig. 2](assets/fig2.png)

**Original caption:** FIGURE 2. The flowchart illustrates the complete data flow from seed preparation on the left through the three-stage core algorithm on the right (lifting, clustering, refinement), ultimately exporting a 3D plant model.

**中文图注:** 图 2. 流程图展示了完整的数据流：从左侧的种子点准备，到右侧三阶段核心算法（提升、聚类、精修），最终导出三维植株模型。

**Reading note:** 重点查看该图如何支撑相邻正文中的流程、比较、消融或性状提取结果。

<a id="S028"></a>
**Source:** p.4 S028  
**Type:** body  
**Confidence:** high

**Original:** The structure and characteristics of the muskmelon dataset.

**中文:** 肉数据集的结构和特性.

<a id="S029"></a>
**Source:** p.4 S029  
**Type:** body  
**Confidence:** high

**Original:** Subset name

**中文:** 亚组名字

<a id="S030"></a>
**Source:** p.4 S030  
**Type:** body  
**Confidence:** high

**Original:** Scenes

**中文:** 场景

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

**Original:** Biological stage

**中文:** 生物学的阶段

<a id="S033"></a>
**Source:** p.4 S033  
**Type:** body  
**Confidence:** high

**Original:** Annotation type

**中文:** 标注类型

<a id="S034"></a>
**Source:** p.4 S034  
**Type:** body  
**Confidence:** high

**Original:** Phenotyping Subset

**中文:** 类型的基因组

<a id="S035"></a>
**Source:** p.4 S035  
**Type:** body  
**Confidence:** high

**Original:** Early vegetative

**中文:** 早期植物性

<a id="S036"></a>
**Source:** p.4 S036  
**Type:** body  
**Confidence:** high

**Original:** Plant/background labels (Gaussian-level)

**中文:** 植物/背景标签 (高斯级)

<a id="S037"></a>
**Source:** p.4 S037  
**Type:** body  
**Confidence:** high

**Original:** Segmentation Training Subset

**中文:** 分类培训子组

<a id="S038"></a>
**Source:** p.4 S038  
**Type:** body  
**Confidence:** high

**Original:** Diverse stages/conditions

**中文:** 几种阶段/条件

<a id="S039"></a>
**Source:** p.4 S039  
**Type:** body  
**Confidence:** high

**Original:** 3D point-cloud organ labels (Point-level)

**中文:** 3D点云器官标签 (点级)

<a id="S040"></a>
**Source:** p.4 S040  
**Type:** body  
**Confidence:** medium

**Original:** motion yielded sufficient parallax. COLMAP was then used to estimate calibrated camera intrinsics and extrinsics and to reconstruct a sparse, geometrically consistent point cloud that served as the initialization scaffold for 3D Gaussian Splatting (3DGS). Building on this SfM output, the next step was to construct the full Gaussian scene by transforming each sparse point into a Gaussian primitive. Each sparse SfM point was converted into a Gaussian primitive defined by its mean mi, initial covariance Si (which becomes anisotropic during optimization), color ci, and opacity ai, forming the parameter set q = {mi, Si, ai, ci}. The 3DGS optimization learned q by minimizing a photometric reconstruction loss with opacity and scale regularization, using differentiable rendering through transmittance-weighted anisotropic Gaussian splats. Because anisotropic kernels naturally align with planar foliage, the representation effectively captures thin leaf structures. Each scene was optimized for 7, 000 iterations using the standard densification schedule, producing a fixed, high-fidelity Gaussian reconstruction used for subsequent plant extraction and organlevel segmentation.

**中文:** 然后,使用COLMAP来估计校准摄像头内在和外观,并重建一个稀有,几何一致的点云,作为3D高斯分光 (3DGS) 的初始化架构.基于此SfM输出,下一步是通过将每个稀有点转化为高斯原始的整体高斯场景构建.每个稀有SfM点被转化为由其平均mi,初始含量Si (在优化过程中变得异化),颜色Ci和异化,形成参数组 q = {mi,Si,ai,ai,ai}.3DGS优化q通过减少光学重复和异化损失,通过变化变化变化变化,通过变化变化变化变化变化变化的变化变化变化变化. 由于无性质核子自然地与平面叶子相结合,因此这幅画有效捕捉了薄叶结构.每个场景都通过标准密度化时间表进行了7,000次代,从而产生了固定的高效的高斯式重建,用于随后的植物提取和器官水平分割.

<a id="S041"></a>
**Source:** p.4 S041  
**Type:** body  
**Confidence:** high

**Original:** As indicated in Table 1, the phenotyping subset contains 6 greenhouse scenes (30 plants) captured 14–15 days after transplanting, during the early vegetative stage characterized by rapid leaf expansion and internode elongation. This subset is held out from segmentation network training and is used to validate the complete pipeline from plant extraction through trait estimation. For this subset, each Gaussian in the 3DGS reconstruction was manually labeled as plant or background using the SuperSplat Gaussian Splats Editor, which supports point-wise selection and carving within a 3DGS scene. The segmentation training subset contains 30 additional scenes (140 plants) collected at the same greenhouse facility using the same overall acquisition protocol and 3DGS reconstruction workflow, spanning diverse growth stages and illumination conditions. The two subsets do not overlap: the phenotyping subset was held out from segmentation-model training and reserved for end-to-end pipeline validation.

**中文:** 正如表1所示,表型子集包含6个温室场景 (30种植物) 在移植后1415天捕获,在早期的植被阶段,特征是快速的叶子扩张和内极延长.该子集从分割网络培训中进行了,并用于通过特征估计验证整个植物提取的管道.对于此子集,3DGS重建中的每个高斯人手动标注为植物或背景,使用SuperSplat Gaussian Splats Editor,该编辑器支持点式选择和雕刻在3DGS场景中. 分类培训子集包含30个额外的场景 (140个工厂) 采集在同一温室设施使用相同的总体采购协议和3DGS重建工作流程,跨越了多种增长阶段和照明条件.这两个子集并不重叠:表型的子集是从分类模型培训中进行的,并保留用于端到端管道验证.

<a id="S042"></a>
### 2.2.2 Structure-from-Motion and 3DGS scene
**Source:** p.4 S042  
**Type:** section  
**Confidence:** high

**Original:** 2.2.2 Structure-from-Motion and 3DGS scene

**中文:** 【标题暂译】2.2.2 Structure-from-Motion and 3DGS scene

<a id="S043"></a>
**Source:** p.4 S043  
**Type:** body  
**Confidence:** high

**Original:** initialization

**中文:** 开始的

<a id="S044"></a>
### 2.3 Plant gaussians extraction (LCR-GS)
**Source:** p.4 S044  
**Type:** section  
**Confidence:** high

**Original:** 2.3 Plant gaussians extraction (LCR-GS)

**中文:** 【标题暂译】2.3 Plant gaussians extraction (LCR-GS)

<a id="S045"></a>
**Source:** p.4 S045  
**Type:** body  
**Confidence:** high

**Original:** To balance pose accuracy with computational efficiency, the recorded videos were temporally subsampled before Structurefrom-Motion (SfM) processing. Frame quality was evaluated using image sharpness, measured by the variance of the Laplacian, and a temporal coverage constraint enforced uniform spatial progression along the acquisition path. Within each short window, the sharpest frame was retained, provided that inter-frame

**中文:** 为了平衡精度和计算效率,在 Structurefrom-Motion (SfM) 处理之前,记录的视频暂时被次样本化. 框架质量是通过图像敏度进行评估,测量在拉普拉西亚变量,时间覆盖限制在收购路径上强制执行统一的空间进展. 在每个短窗口内,最敏的框架被保留,只要是跨框架

<a id="S046"></a>
### 2.3.1 Plant extraction algorithm overview
**Source:** p.4 S046  
**Type:** section  
**Confidence:** high

**Original:** 2.3.1 Plant extraction algorithm overview

**中文:** 【标题暂译】2.3.1 Plant extraction algorithm overview

<a id="S047"></a>
**Source:** p.4 S047  
**Type:** body  
**Confidence:** high

**Original:** algorithm consists of a lightweight seeding stage followed by three automated stages - Lift, Cluster, and Refine - which progressively

**中文:** 算法由一个轻量级的种植阶段组成,随后由三个自动化阶段 - - 升级,集群和精炼 - 逐步

<a id="S048"></a>
**Source:** p.4 S048  
**Type:** body  
**Confidence:** high

**Original:** The flowchart illustrates the complete data flow from seed preparation on the left through the three-stage core algorithm on the right (lifting, clustering, refinement), ultimately exporting a 3D plant model.

**中文:** 流程图说明了从左边的种子准备到右边的三阶段核心算法 (提升,集群,精炼) 的完整数据流程,最终将3D植物模型出口.


## Page 5

<a id="S049"></a>
**Source:** p.5 S049  
**Type:** body  
**Confidence:** high

**Original:** 10.3389/fpls.2026.1783465

**中文:** 10.3389/fpls.2026.1783465

<a id="S050"></a>
**Source:** p.5 S050  
**Type:** body  
**Confidence:** high

**Original:** transform 2D semantic cues into a clean, plant-level 3D subset. This design decouples semantic inference from 3DGS optimization, enabling the reuse of fixed Gaussian reconstructions without retraining. The LCR-GS process begins with a lightweight seeding step, where a small number of manually selected views provide initial 2D semantic masks. These cues are projected into the 3DGS domain to identify Gaussians with multi-view support (Lift). Spatially coherent plant structures are then grouped through geometric clustering and thin-structure restoration (Cluster). A final refinement step uses chromatic filtering in CIELAB color space to remove background remnants and produce clean “Plant Gaussians, “ followed by export to both Gaussian and point-cloud formats (Refine).

**中文:** 2D语义线索将其分为清洁的,植物级的3D子集.这个设计将语义推断从3DGS优化中脱离,使得固定的高斯人重建能够在不再训练的情况下重复使用.LCR-GS过程开始于轻量级的种植步骤,其中少数手动选择的视图提供了初始的2D语义掩膜.这些线索被投射到3DGS领域,以识别具有多视图支持的高斯人 (Lift).空间一致的植物结构随后通过几何聚合和细结构恢复 (Cluster) 进行组合.最终的精炼使用CIELAB色空间中的染色过来删除清洁的背景残留物并产生植物变化Gaussians,然后将其出口到高斯人和基质转换 (Ref) 格式.

<a id="S051"></a>
**Source:** p.5 S051  
**Type:** body  
**Confidence:** high

**Original:** Each view contributes according to a geometryand qualitydependent weight (Equation 2):

**中文:** 每个视图都根据几何和质量依赖的权重 (方程2) 贡献:

<a id="S052"></a>
**Source:** p.5 S052  
**Type:** body  
**Confidence:** medium

**Original:** g,v xmax(cosqg,v, 0)rv, wv = a

**中文:** g,v xmax(cosqg,v,0)rv,wv = a

<a id="S053"></a>
**Source:** p.5 S053  
**Type:** body  
**Confidence:** medium

**Original:** g,v denotes the mean opacity over W(E g,v) as a visibility where a proxy, qg,v is the angle between the viewing ray in view v and a principal axis derived from Sg, and rv ∈ [0, 1] is the sharpness score of frame v, computed as the variance of the Laplacian and robustly normalized within each scene by clipping to the 5th-95th percentile range and rescaling to [0, 1]. The multi-view support score is obtained by aggregating the per-view contributions: Foreground Gaussians are selected by applying a scene-adaptive threshold. Let t = quantile(S, q) for a chosen percentile q. The foreground set is (Equations 3, 4):

**中文:** g,v表示W(E g,v) 上的平均度,即可见度,其中一个代理,qg,v是视觉射线v和从Sg中衍生的主轴之间的角度,rv ∈ [0, 1]是框架v的度分数,计算为拉普拉斯人的变量,并通过切割到第5-95个百分比范围并将其重新扩展到 [0, 1] 在每个场景内强正常化.多视觉支持分数通过汇集每视觉贡献得到:前面的高ussians是通过应用场景适应性门来选择的.让t =量子S,q) 为所选的百分比 q.前面的集合是 (方程 3,4):

<a id="S054"></a>
### 2.3.2 Input seeding for LCR-GS initialization
**Source:** p.5 S054  
**Type:** section  
**Confidence:** high

**Original:** 2.3.2 Input seeding for LCR-GS initialization

**中文:** 【标题暂译】2.3.2 Input seeding for LCR-GS initialization

<a id="S055"></a>
**Source:** p.5 S055  
**Type:** body  
**Confidence:** medium

**Original:** S(g) = ov wv

**中文:** 标签:S(g) = ov wv

<a id="S056"></a>
**Source:** p.5 S056  
**Type:** body  
**Confidence:** high

**Original:** Extraction begins with a one-time, minimal seeding step that initializes the 2D cue pool for each scene. An operator selects one to five representative views per scene that clearly capture the target plant. Open-vocabulary object detections are then applied to these views using YOLO-World, and the detected bounding boxes are passed to SAM to generate box-prompted binary masks. Each resulting cue is stored together with its associated metadata {image_id, view_index, mask_id, bbox} - where image_id identifies the source frame, view_index its position in the calibrated camera list, mask_id distinguishes multiple masks within the same view, and bbox is the bounding-box prompt passed to SAM. This metadata is cached for reuse throughout the extraction process. These cues provide the only manual input required by LCR-GS: they enable multi-view semantic lifting while avoiding any scenespecific retraining or additional labeling beyond the initial selection.

**中文:** 提取开始于一次性,最小的种植步骤,启动每个场景的2D标签池.一个操作员选择每场景的一个到五个代表性视图,清楚地捕捉目标植物.然后使用YOLO-World应用开放词汇对象检测到这些视图,然后检测到的边界框被传递到SAM,以生成框提示二进制掩膜.每个结果的标签被存储在一起与相关的元数据 {image_id,view_index,mask_id,bbox} - 在此,image_id识别了源框,view_index其位置在校准摄像头列表中,区分了同一视图内的多个掩膜,而bbox是向SAM传递的边界框.此类元数据被缓存为整个提取过程中重复使用. 这些线索提供了LCR-GS所需的唯一手动输入:它们使得多视图的语义升高能够实现,同时避免了任何场景特定的重训或超出最初选择的额外标签.

<a id="S057"></a>
**Source:** p.5 S057  
**Type:** body  
**Confidence:** high

**Original:** wv sg,v

**中文:** wv sg,v

<a id="S058"></a>
**Source:** p.5 S058  
**Type:** body  
**Confidence:** medium

**Original:** o kg,v (u)Mv (u), sg,v = ag~sg,v:

**中文:** ,v (u) Mv (u),sg,v = ag~sg,v:

<a id="S059"></a>
**Source:** p.5 S059  
**Type:** body  
**Confidence:** high

**Original:** (3)

**中文:** (3) (3)

<a id="S060"></a>
**Source:** p.5 S060  
**Type:** body  
**Confidence:** medium

**Original:** F = f g ∣ S(g) ≥ t g:

**中文:** S(g) ≥t g:

<a id="S061"></a>
**Source:** p.5 S061  
**Type:** body  
**Confidence:** high

**Original:** (4)

**中文:** (4) (4)

<a id="S062"></a>
**Source:** p.5 S062  
**Type:** body  
**Confidence:** high

**Original:** This foreground set is passed to geometric clustering stage for structural consolidation.

**中文:** 这一前景集被转移到结构整合的几何聚合阶段.

<a id="S063"></a>
### 2.3.4 Geometric clustering and retention filtering
**Source:** p.5 S063  
**Type:** section  
**Confidence:** high

**Original:** 2.3.4 Geometric clustering and retention filtering

**中文:** 【标题暂译】2.3.4 Geometric clustering and retention filtering

<a id="S064"></a>
**Source:** p.5 S064  
**Type:** body  
**Confidence:** medium

**Original:** Because the lifted foreground set may still contain background fragments or multi-plant overlap, geometric clustering is required to consolidate spatially coherent plant structures. After foreground selection, remaining Gaussians are grouped by OPTICS (Ordering Points To Identify the Clustering Structure) in a four-dimensional feature space that combines world-space coordinates with a viewagnostic depth cue (Ankerst et al., 1999). Let mg = (xg, yg, zg)⊤ denote the world coordinates of Gaussian g. To ensure that clustering responds to plant geometry rather than global scene layout, we first normalize all spatial features. To prevent distances in the feature space from being dominated by raw scene extent or outliers, a robust standardization operator is introduced and applied consistently to all scalar quantities (Equation 5):

**中文:** 由于提升的前景集可能仍然包含背景碎片或多植物重叠,因此需要几何聚合物来巩固空间一致的植物结构.在前景选择后,剩余的高西人被OPTICS (排序点来识别聚合结构) 组合成四维特征空间,将世界空间坐标结合视觉深度提示 (Ankerst et al., 1999).让mg = (xg, yang, zg) 表示高西亚线的世界坐标.为了确保聚合物响应植物几何学而不是全球布局,我们首先将所有空间特征进行正常化.为了防止特征空间中的距离被原始程度或异常规范所支配,一个强大的场景标准化操作员被应用到所有场景的比例度 (E 5):

<a id="S065"></a>
**Source:** p.5 S065  
**Type:** body  
**Confidence:** high

**Original:** This step lifts sparse 2D segmentations into the 3D domain by evaluating how consistently each Gaussian is supported across all selected viewpoints. Because each 2D mask provides only a partial and view-dependent observation of plant structure, multi-view lifting aggregates evidence across viewpoints to establish reliable semantic support. The lifting stage computes, for each Gaussian, how much of its projected footprint is covered by the 2D masks and how much confidence each view contributes. In 3DGS reconstruction, each Gaussian is associated with a center mg, covariance Sg, color cg, and opacity ag. The goal of the lifting stage is to aggregate 2D semantic cues from multiple views and assign a multi-view support score to each Gaussian, indicating how strongly it is supported as belonging to the target plant. Each Gaussian is projected into all seeded views using the calibrated intrinsics and extrinsics. For the Gaussian g in view v, projection of (mg, Sg) yields an ellipse E g,v with footprint W(E g,v). A separable kernel kg,v is normalized to sum to one over the footprint (E g,v). With a binary cue Mv ∈ f0, 1g, the masked footprint fraction and its opacity weighted score are defined as (Equation 1):

**中文:** 这一步通过评估每个高质量的视角在所有选定的视角中得到了多么一致的支持,将稀疏的2D分割分化带入3D领域.因为每个2D掩膜只提供了植物结构的部分和视角依赖的观察,多视角升高集了在各视角中的证据,以建立可靠的语义支持.升高阶段计算了每个高质量的预测足迹,以及每个视角的信心.在3DGS重建中,每个高质与一个中心mg,覆盖性 Sg,颜色 cg和度 ag有关.升高阶段的目标是从强大的视角中汇集2D语义线索,并将一个支持度分给Gausly,表明每个高质的视角是如何属于多视角的目标. 每个高斯人均被投射到所有种子视图中,使用校准的内在和外在视图.对于高斯人g在视图 v,投射 (mg,Sg) 产生了圆 E g,v,足迹 W(E g,v).一个可分离的内核kg,v被正常化为足迹 (E g,v) 上一个.通过二进制标注Mv ∈ f0,1g,掩盖足迹分数及其度权重分数被定义为 (方程1):

<a id="S066"></a>
**Source:** p.5 S066  
**Type:** body  
**Confidence:** high

**Original:** u∈W(E g,v)

**中文:** ∈W(E g,v)

<a id="S067"></a>
### 2.3.3 Lifting 2D cues into the 3DGS domain
**Source:** p.5 S067  
**Type:** section  
**Confidence:** high

**Original:** 2.3.3 Lifting 2D cues into the 3DGS domain

**中文:** 【标题暂译】2.3.3 Lifting 2D cues into the 3DGS domain

<a id="S068"></a>
**Source:** p.5 S068  
**Type:** body  
**Confidence:** medium

**Original:** ~sg,v =

**中文:** ~sg,v =

<a id="S069"></a>
**Source:** p.5 S069  
**Type:** body  
**Confidence:** high

**Original:** (2)

**中文:** (2) (2)

<a id="S070"></a>
**Source:** p.5 S070  
**Type:** body  
**Confidence:** medium

**Original:** Z(s ∣ S) =

**中文:** S) =

<a id="S071"></a>
**Source:** p.5 S071  
**Type:** body  
**Confidence:** high

**Original:** s − median(S), MAD(S)

**中文:** − 平均值 (s(S),MAD(S)

<a id="S072"></a>
**Source:** p.5 S072  
**Type:** body  
**Confidence:** high

**Original:** (5)

**中文:** (5) (5)

<a id="S073"></a>
**Source:** p.5 S073  
**Type:** body  
**Confidence:** medium

**Original:** where S is the set of values for a specific feature across all Gaussians, and MAD is the median absolute deviation. This operator centers each feature by its median and rescales by its intrinsic variability, producing standardized values that remain comparable across scenes with different scales or sampling densities. Applying Z(·) to each axis produces standardized coordinates used for shape similarity evaluation (Equation 6): xg0 = Z

**中文:** 在所有高西式中,S是特定特征的值集,MAD是中位绝对偏差.这个运算符将每个特征以其中位数和重度以其内在变化来中心,产生了可比较的标准值,在不同尺度或样本密度的场景中保持可比较的标准值.将Z(·) 应用到每个轴上,产生用于形状相似性评估的标准坐标 (方程 6):xg0 =Z

<a id="S074"></a>
**Source:** p.5 S074  
**Type:** body  
**Confidence:** medium

**Original:**    (xg xj), yg0 = Z(yg yj), zg0 = Z(zg zj),

**中文:** (xg xj),yang0 = Z(yg yj), zg0 = Z(zg zj),

<a id="S075"></a>
**Source:** p.5 S075  
**Type:** body  
**Confidence:** high

**Original:** (6)

**中文:** (6)

<a id="S076"></a>
**Source:** p.5 S076  
**Type:** body  
**Confidence:** high

**Original:** where {xj}, {yj}, {zj}represent the sets of coordinate values across all filtered Gaussians. To ensure depth is encoded consistently across viewpoints, a view-agnostic depth cue is computed. Let C(g) be the set of camera centers from which the Gaussian g is visible. The average camera distance rg is defined as (Equation 7):

**中文:** 【机器初译待精修】where {xj}, {yj}, {zj}represent the sets of coordinate values across all filtered Gaussians. To ensure depth is encoded consistently across viewpoints, a view-agnostic depth cue is computed. Let C(g) be the set of camera centers from which the Gaussian g is visible. The average camera distance rg is defined as (Equation 7):

<a id="S077"></a>
**Source:** p.5 S077  
**Type:** body  
**Confidence:** high

**Original:** (1)

**中文:** (1) (1)


## Page 6

<a id="S078"></a>
**Source:** p.6 S078  
**Type:** body  
**Confidence:** high

**Original:** 10.3389/fpls.2026.1783465

**中文:** 10.3389/fpls.2026.1783465

<a id="S079"></a>
**Source:** p.6 S079  
**Type:** body  
**Confidence:** medium

**Original:** rg =

**中文:** 现在,我们可以看到rg =.

<a id="S080"></a>
**Source:** p.6 S080  
**Type:** body  
**Confidence:** high

**Original:** o ∥ mg − c ∥, ∣ C(g) ∣ c∈C(g)

**中文:** o mg − c, C(g) c∈C(g)

<a id="S081"></a>
**Source:** p.6 S081  
**Type:** body  
**Confidence:** high

**Original:** D2M (x) ≤ t(z),

**中文:** ∆2M (x) ≤t(z),

<a id="S082"></a>
**Source:** p.6 S082  
**Type:** body  
**Confidence:** high

**Original:** (7)

**中文:** 现在,我们要做什么呢? (7)

<a id="S083"></a>
**Source:** p.6 S083  
**Type:** body  
**Confidence:** medium

**Original:** where t(z) is the z -quantile of the in-component Mahalanobis distance distribution. Unless otherwise noted, z = 0:80 is used. During training, a relaxed coverage z = 0:90 is used to increase recall for organ-level segmentation; the stricter default value is reinstated when remapping instance-level results back to the 3DGS representation. This refinement procedure is fully automatic and scene adaptive, requiring no manual threshold setting. The refined plant subset is exported in two complementary forms. First, a dense 3D point cloud is generated by sampling each Gaussian ellipsoid in its principal-axis frame, with the sample count proportional to opacity. Samples are mapped to world coordinates and inherit renderer colors, producing a clean and color-consistent point set for point-based learning (Stuart et al., 2025). Second, the corresponding Gaussian subset is preserved to support rendering and to remap organ-level segmentation results back into the original 3DGS representation for downstream phenotypic analysis.

**中文:** 在训练中,使用宽松的覆盖 z = 0:90 增加器官级分类的召回;重建了更严格的默认值,当将实例级结果重新映射到3DGS表示时.这种精炼程序是完全自动的,适应场景的,不需要手动设置门.精炼的植物子集出口在两个互补形式中.首先,通过采样每个高斯的圆形点云在其主轴中,样本数量与度相对,生成一个密集的3D点云,样本数量与度相对.样本被映射到世界坐标和继承色彩,产生一个清洁和一致的色彩点设置基于点学习 (Sart et al., 2025). 其次,相应的高斯子集被保存以支持染和重新映射器官级分割分结果,将其重新归纳到原始3DGS表示中,以便进行下游表型分析.

<a id="S084"></a>
**Source:** p.6 S084  
**Type:** body  
**Confidence:** medium

**Original:** where c represents a specific camera center vector. Standardizing rg produces a depth feature that is directly comparable to the three coordinate features (Equation 8):  dg0 = Z(rg ∣ rj):

**中文:** 在此,c 表示一个特定的摄像头中心向量.标准化rg产生了直接与三个坐标特征 (方程8) 相比的深度特征:dg0 =Z(rg rj):

<a id="S085"></a>
**Source:** p.6 S085  
**Type:** body  
**Confidence:** high

**Original:** (8)

**中文:** (8) (8)

<a id="S086"></a>
**Source:** p.6 S086  
**Type:** body  
**Confidence:** medium

**Original:** The complete clustering descriptor fg is then assembled by concatenating the four standardized components (Equation 9): fg = ½xg0, yg0, zg0, dg0 ,

**中文:** 然后通过连接四个标准化组件 (方程9) 组装了完整的聚合描述符fg:fg = 1⁄2xg0,yang0, zg0,dg0,

<a id="S087"></a>
**Source:** p.6 S087  
**Type:** body  
**Confidence:** high

**Original:** (9)

**中文:** (9) (9)

<a id="S088"></a>
**Source:** p.6 S088  
**Type:** body  
**Confidence:** medium

**Original:** so that Euclidean distance balances local geometric shape with  the depth cue. OPTICS is applied to the set fg using the Euclidean metric, and clusters are extracted using the steepnessbased x -method (x = 0.1), which identifies clusters as valleys in the reachability plot. To reduce over-segmentation, adjacent clusters are merged when their centroid distance falls within a cluster-adaptive threshold defined as the 90th percentile of point-to-centroid distances in the selected cluster, and when their median depth values differ by less than 15% relative to the selected cluster’s median depth. To recover thin foliage elements that may fall below the initial clustering threshold, a nearest-neighbor retain step restores any rejected Gaussian whose nearest kept neighbor lies within a global radius (Equation 10): rnr = b d~1NN,

**中文:** 为了减少过度划分,邻近的集群将合并,当他们的中心叶距离落入一个集群适应性门值,定义为选择集群中点-中心叶距离的90个百分点,当他们的中深值与选择的集群中中深度差异不到15%.为了恢复可能落在最初的集群门值以下的薄叶片元素,最接近的邻居保持任何被拒绝的邻居 (N1b) 之内,则保持一个近距离的小叶片:

<a id="S089"></a>
### 2.4 Organ-Level instance segmentation
**Source:** p.6 S089  
**Type:** section  
**Confidence:** high

**Original:** 2.4 Organ-Level instance segmentation

**中文:** 【标题暂译】2.4 Organ-Level instance segmentation

<a id="S090"></a>
### 2.4.1 Input normalization and organ instance
**Source:** p.6 S090  
**Type:** section  
**Confidence:** high

**Original:** 2.4.1 Input normalization and organ instance

**中文:** 【标题暂译】2.4.1 Input normalization and organ instance

<a id="S091"></a>
**Source:** p.6 S091  
**Type:** body  
**Confidence:** high

**Original:** segmentation model Per-plant point clouds exported from the 3DGS representation are segmented into stems and leaves using a transformer-based point cloud network with a PointGroup-style bottom up head. Because reconstructions derived from structure-from-motion lack a consistent metric scale, all inputs are normalized before learning to ensure uniform spatial density across scenes. Let xi denote a raw point, c the median spatial centroid, and s the median k-nearestneighbor spacing. Each point is normalized to a target spacing D by (Equation 12)

**中文:** 根据3DGS表示的每种植物点云被分成干部和叶子,使用一个基于变压器的点云网络,并使用一个像PointGroup这样的底部上头.由于从结构到运动中衍生的重建没有一致的度量尺度,因此在学习之前,所有输入都会正常化,以确保场景中均的空间密度.让 xi表示原点,c是中空间中心,s是中 k-近邻区间.每个点都会正常化到目标区间 D (方程 12) 之间.

<a id="S092"></a>
**Source:** p.6 S092  
**Type:** body  
**Confidence:** high

**Original:** (10)

**中文:** (10)

<a id="S093"></a>
**Source:** p.6 S093  
**Type:** body  
**Confidence:** medium

**Original:** where d~1NN denotes the median nearest-neighbor distance among the retained Gaussians and b is a fixed scale factor (default b = 2). This stage is purely geometric, acting as a robustness filter that consolidates a coherent per-plant subset before the refinement step.

**中文:** 在其中,d~1NN表示保留的高西人中最接近邻居的距离,b是固定尺度因子 (默认 b = 2).这个阶段是纯粹的几何,作为强度过器,在精炼阶段之前巩固一个连贯的植物子集.

<a id="S094"></a>
**Source:** p.6 S094  
**Type:** body  
**Confidence:** medium

**Original:** x0i = c + (xi − c) ·

**中文:** x0i = c + (xi − c) ·

<a id="S095"></a>
### 2.3.5 Chromatic refinement and instance
**Source:** p.6 S095  
**Type:** section  
**Confidence:** high

**Original:** 2.3.5 Chromatic refinement and instance

**中文:** 【标题暂译】2.3.5 Chromatic refinement and instance

<a id="S096"></a>
**Source:** p.6 S096  
**Type:** body  
**Confidence:** high

**Original:** exportation

**中文:** 导出

<a id="S097"></a>
**Source:** p.6 S097  
**Type:** body  
**Confidence:** high

**Original:** D, s

**中文:** ,D,s

<a id="S098"></a>
**Source:** p.6 S098  
**Type:** body  
**Confidence:** high

**Original:** (12)

**中文:** (12) (12)

<a id="S099"></a>
**Source:** p.6 S099  
**Type:** body  
**Confidence:** medium

**Original:** which stabilizes voxel resolution, attention coverage, and grouping radii across scenes. This normalization addresses variation in point cloud density across scenes and is distinct from the subsequent metric scale alignment step. The segmentation backbone follows Point Transformer V3 (PTv3), which preserves permutation invariance while capturing long-range context through attention on serialized local neighborhoods and radius-aware downsampling. The network outputs perpoint feature descriptors and semantic logits for stems and leaf classes. Training uses AdamW with mixed precision; the semantic head is trained with a base learning rate of and weight decay under a one-cycle schedule with warm-up, while the backbone is fine-tuned at a reduced learning rate of. The global batch size is 12, and training runs for 800 epochs. This organ-segmentation model is trained once for the target crop setting and subsequently applied to all extracted plant instances without scene-specific retraining or fine-tuning. Instance grouping follows a PointGroup-style bottom-up procedure. For each semantic class k, the subset Sk = fi ∣ y^ i = kg is partitioned into connected components using a Euclidean radius

**中文:** 这种规范化解决了场景间点云密度变化,并与随后的度量尺度调整步骤不同. 分类背骨遵循Point Transformer V3 (PTv3),通过通过关注串联的本地社区和半径意识下样本捕捉长距离背景,保持变量不变,同时保持变量不变.网络输出每个特征描述器和树干和叶片类的语义点逻辑.训练使用AdamW精度混合;语义背骨在一个单周期的学习率和体重衰减计划下训练,而背骨则以减少的速度调整.全球训练规模为800个,为12个时代. 这个器官分类模型是为了目标作物设置训练一次,然后在没有场景特定的重训或细调的情况下应用到所有提取的植物实例上.实例分类遵循PointGroup式的下游程序.对于每个语义类 k,子集 Sk = fi y^ i = kg 是通过尤克利德半径分成连接组件的.

<a id="S100"></a>
**Source:** p.6 S100  
**Type:** body  
**Confidence:** high

**Original:** The final refinement step converts the clustered candidate subset into a compact, plant-focused representation by suppressing residual background elements using chromatic filtering. Chromatic outliers are attenuated in the CIELAB (CIE L*a*b*) color space, which separates luminance and chromatic components and is widely used for plant-background separation and color-based analysis (Herná ndez-Herná ndez et al., 2016; Pape and Klukas, 2014). A full-covariance Gaussian mixture model (GMM) with up to four components is fitted in CIELAB space, with the number of components selected by the Bayesian Information Criterion (BIC). When the candidate point set exceeds 50, 000 points, a stratified subsample is drawn using quantile-cell sampling in CIELAB space to preserve the overall color distribution; smaller sets are used in full. The dominant component is selected by mixture weight, and an ellipsoidal chromaticity window is defined by a single coverage parameter z ∈ (0, 1). Gaussians whose Mahalanobis distance to the dominant component falls inside this window are retained (Equation 11):

**中文:** 最终的精炼步骤将聚合的候选子集转化为一个紧的,以植物为中心的表现,通过使用染色过来抑制残余背景元素.在CIELAB (CIE L*a*b*) 颜色空间中,染色异差被减弱,该空间分离了亮度和染色元件,并且广泛用于植物背景分离和基于颜色的分析 (Herná ndez-Herná ndez et al., 2016;Pape and Klukas, 2014).一个全化高质米混合模型 (GMM) 包含最多四个组件,在CIELAB空间中安装,由贝利斯信息标准 (BIC) 选择的组件数量较小.当候选点设置超过50,000个,则在CI中使用量化样本进行全细胞分布;用于保存整个颜色. 主要组件由混合物重量选择,而圆形色彩性窗口由单个覆盖参数 z ∈ (0,1) 定义.在这个窗口内,在其中占主成分的马哈拉诺比斯距离下降的高西斯人保持 (方程 11):

<a id="S101"></a>
**Source:** p.6 S101  
**Type:** body  
**Confidence:** high

**Original:** (11)

**中文:** (11)


## Page 7

<a id="S102"></a>
**Source:** p.7 S102  
**Type:** body  
**Confidence:** high

**Original:** 10.3389/fpls.2026.1783465

**中文:** 10.3389/fpls.2026.1783465

<a id="S103"></a>
**Source:** p.7 S103  
**Type:** body  
**Confidence:** medium

**Original:** expressed relative to density, r = g D, where g is a fixed scale factor. Two points in Sk are linked when ∥ x0i − x0j ∥ ≤ r. A breadth first search aggregates links into spatially contiguous regions representing individual organs. Instance confidence is computed as the average semantic logit across its member points.

**中文:** 相对于密度,r = g D,其中g是固定尺度因素. Sk 中的两个点是链接的,当 x0i − x0j ≤ r.一个宽度的首个搜索集体表达为空间连接的区域,代表个体器官.实例信心被计算为其成员国点的平均语义逻辑.

<a id="S104"></a>
### 2.5.2 Trait definition and computation
**Source:** p.7 S104  
**Type:** section  
**Confidence:** high

**Original:** 2.5.2 Trait definition and computation

**中文:** 【标题暂译】2.5.2 Trait definition and computation

<a id="S105"></a>
**Source:** p.7 S105  
**Type:** body  
**Confidence:** medium

**Original:** We extract six phenotypic traits relevant to muskmelon growth assessment: plant height, leaf surface area (LSA), leaf area index (LAI), leaf count, node count, and internode length. All measurements are computed in the aligned metric frame. Plant height is defined as the vertical extent from the ground plane (z = 0 in the metric-aligned frame) to the plant apex (Equation 14):

**中文:** 我们提取了六种对桃生长评估相关的现象特征:植物高度,叶片表面积 (LSA),叶片面积指数 (LAI),叶子数,节点数和内极长.所有测量都在对齐的测量框架中计算.植物高度被定义为从地面平面 (z = 0 在对齐的测量框架中) 到植物顶峰的垂直范围 (方程14):

<a id="S106"></a>
### 2.4.2 Remapping segmentations to the 3DGS
**Source:** p.7 S106  
**Type:** section  
**Confidence:** high

**Original:** 2.4.2 Remapping segmentations to the 3DGS

**中文:** 【标题暂译】2.4.2 Remapping segmentations to the 3DGS

<a id="S107"></a>
**Source:** p.7 S107  
**Type:** body  
**Confidence:** medium

**Original:** domain Per-point labels from the normalized cloud are transferred back to the original 3DGS reconstruction for visualization and downstream trait computation. Only visible splats are considered, as determined by their opacity parameter w using s (w) = 1=(1 + e−w) with a small threshold t. For each visible splat, the nearest labeled point from the segmented cloud is queried within a density-aware assignment radius (Equation 13)

**中文:** 从正常化云中每个域名标签被转移到原始的3DGS重构中进行可视化和下游特征计算.仅考虑可见的零点,根据其使用s (w) = 1=(1 + e−w) 的度参数w的决定,以小门t为基础.对于每个可见的零点,在密度意识的分配半径 (方程 13) 范围内查询了从分割云中最接近的标签点.

<a id="S108"></a>
**Source:** p.7 S108  
**Type:** body  
**Confidence:** medium

**Original:** rassign = admed,

**中文:** rassign = admed,

<a id="S109"></a>
**Source:** p.7 S109  
**Type:** body  
**Confidence:** medium

**Original:** H = s · maxzi0:

**中文:** 据说H = s · maxzi0:

<a id="S110"></a>
**Source:** p.7 S110  
**Type:** body  
**Confidence:** high

**Original:** A small morphological closing operation is applied on the height map to suppress isolated spurious peaks near the apex. The structuring element radius is set to twice the median nearestneighbor spacing for robustness across scenes. LSA is computed as the sum of one-sided areas, and LAI normalizes LSA by the fixed reference area Aplot. The reference area Aplot used for LAI normalization was fixed at 0.036 m2, derived from the mean canopy spread diameter (0.214 m, measured physically with a tape measure) using a circular-footprint approximation. For each leaf instance, we fit a PCA plane at its centroid to define local axes. Each Gaussian ellipsoid belonging to the leaf is orthogonally projected onto this plane, yielding a set of 2D ellipses. The leaf area is computed as the a-shape area of these ellipses, which captures fine leaf geometry while excluding void regions (Equation 15).

**中文:** 在高度地图上应用了小形态闭合操作,以压制顶点附近的孤立虚假峰值.结构元素半径设置为硬度的中隔两倍,以确保场景中的近邻A. LSA被计算为单边区域的总和,LAI通过固定参考区域Aplot来正常化LSA. LAI正常化所使用的参考区域Aplot是0.036m2,从平均天窗扩展直径 (0.214m,用磁带测量物理测量) 来获得,使用圆形足迹近似方法.对于每个叶片实例,我们将PCA放在其轴心的中心.叶片的高索圆形圆形直径为这个平面投射,产生2D圆形圆形. 叶片面积被计算为这些圆圆形的a形面积,这捕捉到细叶几何学,同时排除空隙区域 (方程 15).

<a id="S111"></a>
**Source:** p.7 S111  
**Type:** body  
**Confidence:** high

**Original:** (13)

**中文:** (13) (13)

<a id="S112"></a>
**Source:** p.7 S112  
**Type:** body  
**Confidence:** high

**Original:** where dmed is the median nearest-neighbor spacing among visible splats and a ≥ 1 is a fixed scale. Splats within this radius inherit the semantic and instance labels of the nearest reference point. Unassigned splats are retained for diagnostics. The mapped splats preserve their original Gaussian parameters (position, spherical harmonic coefficients, opacity, and covariance). Each organ instance is then exported as an individual PLY file containing its associated Gaussian splats, in a format compatible with standard 3DGS renderers. This representation enables perorgan manipulation, rendering, and analysis, while preserving the photometric fidelity of the original reconstruction.

**中文:** dmed是可见的点之间的近邻间隔中介,a ≥ 1是固定尺度.在这个半径内的点继承了最近参考点的语义和实例标签.未分配的点保留用于诊断.映射的点保留了其原始的高斯参数 (位置,球状和系数,度和覆盖率).然后,每个器官实例将作为包含其相关的高斯点的单个PLY文件出口,以兼容标准3DGS染器的格式.这种表示使器官操作,染和分析成为可能,同时保持原始重建的光学忠实性.

<a id="S113"></a>
**Source:** p.7 S113  
**Type:** body  
**Confidence:** medium

**Original:** LSA = s2 oAa (E ‘), LAI = ‘=1

**中文:** 美国的LSA = s2 oAa (E),LAI = =1.

<a id="S114"></a>
**Source:** p.7 S114  
**Type:** body  
**Confidence:** high

**Original:** LSA, Aplot

**中文:** 美国国家安全局,LSA,Aplot

<a id="S115"></a>
**Source:** p.7 S115  
**Type:** body  
**Confidence:** high

**Original:** (15)

**中文:** (15) (15)

<a id="S116"></a>
**Source:** p.7 S116  
**Type:** body  
**Confidence:** high

**Original:** Leaf count is the number of leaf instances after simple postprocessing that merges leaf fragments whose centroids lie within one median nearest-neighbor spacing and removes components smaller than 1 cm2 (in metric scale). This ensures biologically meaningful leaf instances while suppressing artifacts. A contracted centerline is extracted from the stem subset by Laplacian-based contraction with density-adaptive spacing. An orientation-aware minimum spanning tree (MST) is then constructed over the contracted nodes, and the main axis is selected as the path with maximum vertical span. Nodes are defined at topological junctions along this axis. Internode lengths are computed as geodesic distances between consecutive nodes along the centerline, which provides stable measurements even for curved stems.

**中文:** 叶子数量是简单后处理后的叶子实例数量,该实例数量是将叶片碎片合并,其中位体位于一个中位数内,在最接近邻居间隔内,并移除小于1厘米2的组件 (在度量尺度中).这确保了生物意义的叶片实例,同时抑制文物.通过基于拉普拉西亚的缩小和适应密度的缩小,从干部子集中提取一个合约的中线.然后在合约的节点上构建了一个定向意识的最小跨度树 (MST),主要轴被选为最大垂直跨度的中轴.节点在这个轴上定义在生物学交叉点上.内线长度被计算为沿线连续节点之间的地质距离,从而为稳定的节点提供曲线测量.

<a id="S117"></a>
### 2.5 Phenotypic trait quantification
**Source:** p.7 S117  
**Type:** section  
**Confidence:** high

**Original:** 2.5 Phenotypic trait quantification

**中文:** 【标题暂译】2.5 Phenotypic trait quantification

<a id="S118"></a>
### 2.5.1 Metric scaling and coordinate frame
**Source:** p.7 S118  
**Type:** section  
**Confidence:** high

**Original:** 2.5.1 Metric scaling and coordinate frame

**中文:** 【标题暂译】2.5.1 Metric scaling and coordinate frame

<a id="S119"></a>
**Source:** p.7 S119  
**Type:** body  
**Confidence:** medium

**Original:** alignment With organ-level segmentation complete and all leaf and stem instances mapped back into the 3DGS representation, the resulting labeled plant structures provide the foundation for computing biologically meaningful phenotypic traits. All trait measurements are reported in metric units and within a consistent world coordinate frame. A per-scene scale factor s is obtained from a visible calibration reference; the greenhouse rack width (42 cm) is measured in the SfM reconstruction and compared with its groundtruth dimension to compute the metric scale. To ensure geometric consistency across scenes, a Manhattanworld alignment is estimated from the rectilinear structural regularities of the greenhouse environment, following the approach of COLMAP’s model orientation aligner (Schönberger and Frahm, 2016; Coughlan and Yuille, 2000). This yields a similarity transform, where R is a rotation matrix aligning the vertical axis with gravity and flattening the ground plane to z = 0. Each Gaussian center is then transformed as. The resulting metric-aligned coordinates form the basis for all downstream trait computations.

**中文:** 配合 器官级分类完成,所有叶子和干部实例都被映射到3DGS表示中,结果标注的植物结构为计算具有生物意义的现象特征提供了基础.所有特征测量均以计分单位和一致的世界坐标框架内进行报告.一个每个场景尺度因子s从可见的校准参考中得到;温室架宽度 (42厘米) 在SfM重建中测量,并与其基础真相进行比较,以计算计分尺度.为了确保几何一致性,曼哈顿的世界配合是根据温室环境的直线结构规律来估计的,遵循LMAP的方法 (Schönger和Frahmen, 2016;Coughlan和Yu, 2000). 这产生了类似性转换,R是一个旋转矩阵,将垂直轴与重力相对应,并将地面平面平坦到z = 0.然后每个高斯中心都被转换为.

<a id="S120"></a>
**Source:** p.7 S120  
**Type:** body  
**Confidence:** high

**Original:** (14)

**中文:** 现在,我们必须要做什么? (14)

<a id="S121"></a>
### 3 Results and discussion
**Source:** p.7 S121  
**Type:** section  
**Confidence:** high

**Original:** 3 Results and discussion

**中文:** 【标题暂译】3 Results and discussion

<a id="S122"></a>
### 3.1 Optimizing 3DGS scene reconstruction
**Source:** p.7 S122  
**Type:** section  
**Confidence:** high

**Original:** 3.1 Optimizing 3DGS scene reconstruction

**中文:** 【标题暂译】3.1 Optimizing 3DGS scene reconstruction

<a id="S123"></a>
### 3.1.1 Efficient frame selection for sTable 3DGS
**Source:** p.7 S123  
**Type:** section  
**Confidence:** high

**Original:** 3.1.1 Efficient frame selection for sTable 3DGS

**中文:** 【标题暂译】3.1.1 Efficient frame selection for sTable 3DGS

<a id="S124"></a>
**Source:** p.7 S124  
**Type:** body  
**Confidence:** high

**Original:** reconstruction The reconstruction pipeline begins with dense video capture: a 30-second video at 30 frames per second (fps) yields ~900 frames

**中文:** 重建的重建管道从密集的视频捕捉开始:30秒的视频以每秒30 (fps) 速度产生 ~900.


## Page 8

<a id="S125"></a>
**Source:** p.8 S125  
**Type:** body  
**Confidence:** high

**Original:** 10.3389/fpls.2026.1783465

**中文:** 10.3389/fpls.2026.1783465

<a id="S126"></a>
**Source:** p.8 S126  
**Type:** body  
**Confidence:** high

**Original:** To quantify reconstruction fidelity, Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) were computed on a held-out view. Instant-NGP achieves 25.589 dB/0.7935 SSIM, whereas 3DGS attains higher fidelity at 26.281 dB/0.8297 SSIM. In addition to these numerical gains, 3DGS produces visibly sharper foliage and vine structures, preserving leaf shape and thin features that appear blurred or fragmented in the NeRF output. The inset zoom views in Figure 3 highlight these differences most clearly, showing improved edge integrity along leaf margins and better reconstruction of fine background textures. Overall, 3DGS provides superior geometric and photometric detail compared with NeRF under identical pose and training conditions, making it a more suitable reconstruction backbone for downstream plant extraction and phenotyping tasks.

**中文:** 为了量化重建效果,在一个持久的视图上计算了峰值信号与噪音比率 (PSNR) 和结构相似度指数 (SSIM).即时NGP达到25.589 dB/0.7935 SSIM,而3DGS在26.281 dB/0.8297 SSIM上达到更高的效果.除了这些数字增长之外,3DGS还产生了明显的更尖的叶片和葡萄结构,保留了叶片形状和薄面的特征,这些特征在NeRF输出中看起来模糊或碎片化.图3中的插入缩放视图突出了这些差异,最清楚地显示了这些差异,改善了叶片边缘的边缘完整性和更好的背景纹理重建. 总体而言,3DGS在相同的姿势和训练条件下提供了比NeRF更好的几何和光学细节,使其成为下游植物提取和表型工作的更适合重建骨干.

<a id="F003"></a>
### Fig. 3. (a) 稠密 MVS、(b) NeRF 和 (c) 3DGS 的重建结果比较。插图显示局部放大区域，用于突出几何保真度和光度保真度的差异。
**Placed near:** p.8 S126  
**Source:** p.9 manual-layout  
**Crop confidence:** high

![Fig. 3](assets/fig3.png)

**Original caption:** FIGURE 3. Comparison of reconstruction results from (a) dense MVS, (b) NeRF, and (c) 3DGS. Insets show zoomed regions highlighting differences in geometric and photometric fidelity.

**中文图注:** 图 3. (a) 稠密 MVS、(b) NeRF 和 (c) 3DGS 的重建结果比较。插图显示局部放大区域，用于突出几何保真度和光度保真度的差异。

**Reading note:** 重点查看该图如何支撑相邻正文中的流程、比较、消融或性状提取结果。

<a id="S127"></a>
**Source:** p.8 S127  
**Type:** body  
**Confidence:** high

**Original:** per greenhouse row, covering ~5 meters of crop area. Processing all frames through structure-from-motion (SfM) requires 6–8 hours per scene, making full-resolution reconstruction computationally impractical. To balance reconstruction quality and efficiency, we evaluated three frame-subsampling strategies: Uniform sampling (even temporal spacing), Sharpness-based sampling (ranked by Laplacian variance), and Coverage + Sharpness sampling (enforcing spatial coverage while prioritizing sharp frames). Reconstruction quality was assessed using two standard SfM geometric metrics from COLMAP: reprojection error (camera pose accuracy) and mean track length (multi-view feature consistency). These metrics directly affect the stability of subsequent 3DGS training, where inaccurate poses or weak correspondences can lead to geometric drift. The comparison of frame-selection strategies for 3DGS reconstruction is detailed in Table 2. As shown in Table 2, across six greenhouse scenes. At a 50% sampling rate, the Sharpness strategy achieves the lowest reprojection error (1.039 pixels). At a more aggressive 25% sampling rate, where computational savings are more substantial, the Coverage plus Sharpness maintains comparably low reprojection error (1.044 pixels) while producing substantially longer feature tracks (6.89 versus 4.21), indicating stronger multi-view geometric consistency. This suggests that when subsampling heavily, enforcing spatial coverage is crucial for maintaining reconstruction quality. Based on these results, we adopt Coverage + Sharpness at 25% sampling for all subsequent reconstructions. This configuration reduces SfM processing time to 1.5-2.0 hours per scene while retaining the geometric stability required for reliable 3DGS optimization. The selected frames serve as input to the plant extraction and phenotyping pipeline described in the following sections.

**中文:** 为了平衡重建质量和效率,我们评估了三个框架次采样策略:统一采样 (即时间距),基于度的采样 (由拉普拉西亚变量排名),以及覆盖率+度的采样 (强化空间覆盖率,同时优先考虑尖的图像).重建质量通过COLMAP的两个标准SfM几何指标来评估:射错误 (摄像头姿势精度) 和平均轨道长度 (多视图一致性).这些指标直接影响了3DGS的稳定性,不适应的训练或相应的几何. 3DGS重建框架选择策略的比较详细在表2中.如表2中所示,在六个温室场景中.在50%的采样率下,Sharpness策略实现了最低的重投错率 (1.039像素).在更积极的25%的采样率下,计算节省率更大,Coverage加 Sharpness保持了相对较低的重投错率 (1.044像素),同时产生了更长的功能轨道 (6.89比4.21),表明更强大的多视图几何一致性.这表明,在重投标时,强制执行空间覆盖率对于维持重建质量至关重要.基于这些结果,我们采用了Coverage + Sharpness在所有随后的重建中,以25%的采样率. 这种配置可以将SfM处理时间缩短到每场景1.5-2.0小时,同时保持可靠的3DGS优化所需的几何稳定性.所选的框架作为植物提取和表型的输入,如下部分所描述.

<a id="S128"></a>
### 3.2 Plant instance extraction from 3DGS
**Source:** p.8 S128  
**Type:** section  
**Confidence:** high

**Original:** 3.2 Plant instance extraction from 3DGS

**中文:** 【标题暂译】3.2 Plant instance extraction from 3DGS

<a id="S129"></a>
**Source:** p.8 S129  
**Type:** body  
**Confidence:** high

**Original:** scenes

**中文:** 场景

<a id="S130"></a>
### 3.2.1 Effect of input cue count on plant
**Source:** p.8 S130  
**Type:** section  
**Confidence:** high

**Original:** 3.2.1 Effect of input cue count on plant

**中文:** 【标题暂译】3.2.1 Effect of input cue count on plant

<a id="S131"></a>
**Source:** p.8 S131  
**Type:** body  
**Confidence:** medium

**Original:** separation quality The LCR-GS extraction pipeline allows users to adjust the number of seeded 2D cues per plant, trading annotation effort against separation quality. To quantify this effect, we evaluated three cue configurations on a representative greenhouse scene: one cue (N = 1), three cues (N = 3), and five cues (N = 5). This comparison serves as a single-scene diagnostic to motivate the cue-count setting; formal extraction benchmarking is reported on the full validation set. For each configuration, the per-splat lift scores were independently normalized to the range [0, 1] to allow fair comparison of spatial patterns across different N. Figure 4 visualizes the spatial distribution of lift scores under the percentile-based thresholds (top 10%, 15%, and 20%). Using percentile thresholds rather than absolute cutoffs enables direct comparison of the separation patterns produced by each cue configuration. The results reveal distinct separation trends across cue configurations: with one cue (N = 1), plant and background splats exhibit substantial overlap (plants ≈ 0.3-0.8; background ≈ 0.20.6), producing ambiguous boundaries and weak clustering performance. Using three cues (N = 3) substantially improves

**中文:** 液晶晶体管道 (LCR-GS) 提取管道允许用户对每种植物调整种植的2D线索数量,交易标注力与分离质量.为了量化这种效果,我们对一个代表性温室场景进行了三种线索配置:一个线索 (N = 1),三个线索 (N = 3) 和五个线索 (N = 5).这种比较作为一个单场诊断来激励线索计设置;正式的提取基准报告在完整的验证组上.对于每个配置,每个位数升级分数独立正常化到范围 [0, 1]以允许公平地比较不同N的空间模式.图4可视化在百分比率为基础的门值 (上方10%,15%,20%) 下面的升级分数的空间分布. 使用百分比值,而不是绝对的切断,可以直接比较每个标签配置产生的分离模式.结果揭示了不同标签配置的分离趋势:在一个标签 (N = 1),植物和背景点显示了显著的重叠 (植物 ≈ 0.3-0.8;背景 ≈ 0.20.6),产生了模糊的边界和弱的集群性能.使用三个标签 (N = 3) 显著改善

<a id="F004"></a>
### Fig. 4. 输入提示数量（行）和提升分数保留百分位（列）对植株-背景分离的影响。更多提示和更严格的百分位阈值能更好地隔离高置信植株区域。
**Placed near:** p.8 S131  
**Source:** p.9 manual-layout  
**Crop confidence:** medium

![Fig. 4](assets/fig4.png)

**Original caption:** FIGURE 4. Effect of input cue count (rows) and lift-score retention percentile (columns) on plant-background separation. Higher cue counts and stricter percentiles better isolate high-confidence plant regions.

**中文图注:** 图 4. 输入提示数量（行）和提升分数保留百分位（列）对植株-背景分离的影响。更多提示和更严格的百分位阈值能更好地隔离高置信植株区域。

**Reading note:** 重点查看该图如何支撑相邻正文中的流程、比较、消融或性状提取结果。

<a id="S132"></a>
### 3.1.2 Reconstruction quality comparison
**Source:** p.8 S132  
**Type:** section  
**Confidence:** high

**Original:** 3.1.2 Reconstruction quality comparison

**中文:** 【标题暂译】3.1.2 Reconstruction quality comparison

<a id="S133"></a>
**Source:** p.8 S133  
**Type:** body  
**Confidence:** high

**Original:** greenhouse scene using three methods: (a) COLMAP dense reconstruction (shown for qualitative reference), (b) Instant-NGP (NeRF model), and (c) 3D Gaussian Splatting (3DGS). Both neural rendering methods were trained with the same COLMAP-derived camera poses and the selected frame subset determined in Section 3.1.1.

**中文:** 使用三个方法的温室场景: (a) COLMAP密集重建 (显示为质量参考), (b) 即时NGP (NeRF模型),和 (c) 3D高斯式光 (3DGS).这两种神经染方法都采用相同的COLMAP衍生摄像头姿势和在 3.1.1节确定的选定的框架子集训练.

<a id="S134"></a>
**Source:** p.8 S134  
**Type:** body  
**Confidence:** high

**Original:** Comparison of frame-selection strategies for 3DGS reconstruction.

**中文:** 对于3DGS重建而言,相比较框架选择策略.

<a id="S135"></a>
**Source:** p.8 S135  
**Type:** body  
**Confidence:** high

**Original:** Sharp med. ↑

**中文:** ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

<a id="S136"></a>
**Source:** p.8 S136  
**Type:** body  
**Confidence:** high

**Original:** Policy

**中文:** 政策

<a id="S137"></a>
**Source:** p.8 S137  
**Type:** body  
**Confidence:** high

**Original:** Sharp s ↓

**中文:** 鱼的↓

<a id="S138"></a>
**Source:** p.8 S138  
**Type:** body  
**Confidence:** high

**Original:** TrackLen

**中文:** 追踪Len

<a id="S139"></a>
**Source:** p.8 S139  
**Type:** body  
**Confidence:** high

**Original:** Reproj (px)

**中文:** 雷普罗杰 (px)

<a id="S140"></a>
**Source:** p.8 S140  
**Type:** body  
**Confidence:** high

**Original:** All Frames

**中文:** 所有框架

<a id="S141"></a>
**Source:** p.8 S141  
**Type:** body  
**Confidence:** high

**Original:** 345.22

**中文:** 345.22 345.22

<a id="S142"></a>
**Source:** p.8 S142  
**Type:** body  
**Confidence:** high

**Original:** 59.64

**中文:** 59.64 59.64

<a id="S143"></a>
**Source:** p.8 S143  
**Type:** body  
**Confidence:** high

**Original:** 8.81

**中文:** 8.81 8.81

<a id="S144"></a>
**Source:** p.8 S144  
**Type:** body  
**Confidence:** high

**Original:** 1.266

**中文:** 1,266 1.266

<a id="S145"></a>
**Source:** p.8 S145  
**Type:** body  
**Confidence:** high

**Original:** Uniform

**中文:** 制服

<a id="S146"></a>
**Source:** p.8 S146  
**Type:** body  
**Confidence:** high

**Original:** 344.45

**中文:** 344.45 344.45

<a id="S147"></a>
**Source:** p.8 S147  
**Type:** body  
**Confidence:** high

**Original:** 58.44

**中文:** 58.44 58.44

<a id="S148"></a>
**Source:** p.8 S148  
**Type:** body  
**Confidence:** high

**Original:** 6.40

**中文:** 6.40 6.40

<a id="S149"></a>
**Source:** p.8 S149  
**Type:** body  
**Confidence:** high

**Original:** 1.174

**中文:** 1,174 1.174

<a id="S150"></a>
**Source:** p.8 S150  
**Type:** body  
**Confidence:** high

**Original:** Coverage+Sharpness

**中文:** 报道+度覆盖

<a id="S151"></a>
**Source:** p.8 S151  
**Type:** body  
**Confidence:** high

**Original:** 364.79

**中文:** 364.79 364.79

<a id="S152"></a>
**Source:** p.8 S152  
**Type:** body  
**Confidence:** high

**Original:** 56.59

**中文:** 56.59 56.59

<a id="S153"></a>
**Source:** p.8 S153  
**Type:** body  
**Confidence:** high

**Original:** 6.52

**中文:** 6.52 6.52

<a id="S154"></a>
**Source:** p.8 S154  
**Type:** body  
**Confidence:** high

**Original:** 1.093

**中文:** 1,093 1.093

<a id="S155"></a>
**Source:** p.8 S155  
**Type:** body  
**Confidence:** high

**Original:** 50%

**中文:** 50%的

<a id="S156"></a>
**Source:** p.8 S156  
**Type:** body  
**Confidence:** high

**Original:** 25%

**中文:** 现在,我们要做的是25%.

<a id="S157"></a>
**Source:** p.8 S157  
**Type:** body  
**Confidence:** high

**Original:** Sharpness

**中文:** 敏性

<a id="S158"></a>
**Source:** p.8 S158  
**Type:** body  
**Confidence:** high

**Original:** 382.39

**中文:** 382.39 382.39

<a id="S159"></a>
**Source:** p.8 S159  
**Type:** body  
**Confidence:** high

**Original:** 33.60

**中文:** 33.60 33.60

<a id="S160"></a>
**Source:** p.8 S160  
**Type:** body  
**Confidence:** high

**Original:** 7.74

**中文:** 7.74 7.74

<a id="S161"></a>
**Source:** p.8 S161  
**Type:** body  
**Confidence:** high

**Original:** 1.039

**中文:** 1,039

<a id="S162"></a>
**Source:** p.8 S162  
**Type:** body  
**Confidence:** high

**Original:** Uniform

**中文:** 制服

<a id="S163"></a>
**Source:** p.8 S163  
**Type:** body  
**Confidence:** high

**Original:** 324.36

**中文:** 324.36 324.36

<a id="S164"></a>
**Source:** p.8 S164  
**Type:** body  
**Confidence:** high

**Original:** 57.78

**中文:** 57.78

<a id="S165"></a>
**Source:** p.8 S165  
**Type:** body  
**Confidence:** high

**Original:** 4.99

**中文:** 4.99 4.99

<a id="S166"></a>
**Source:** p.8 S166  
**Type:** body  
**Confidence:** high

**Original:** 1.295

**中文:** 1,295 1.295

<a id="S167"></a>
**Source:** p.8 S167  
**Type:** body  
**Confidence:** high

**Original:** Coverage+Sharpness

**中文:** 报道+度覆盖

<a id="S168"></a>
**Source:** p.8 S168  
**Type:** body  
**Confidence:** high

**Original:** 389.33

**中文:** 389.33 389.33

<a id="S169"></a>
**Source:** p.8 S169  
**Type:** body  
**Confidence:** high

**Original:** 52.79

**中文:** 52.79 52.79

<a id="S170"></a>
**Source:** p.8 S170  
**Type:** body  
**Confidence:** high

**Original:** 6.89

**中文:** 6.89 6.89

<a id="S171"></a>
**Source:** p.8 S171  
**Type:** body  
**Confidence:** high

**Original:** 1.044

**中文:** 1,044

<a id="S172"></a>
**Source:** p.8 S172  
**Type:** body  
**Confidence:** high

**Original:** Sharpness

**中文:** 敏性

<a id="S173"></a>
**Source:** p.8 S173  
**Type:** body  
**Confidence:** high

**Original:** 404.89

**中文:** 404.89 404.89

<a id="S174"></a>
**Source:** p.8 S174  
**Type:** body  
**Confidence:** high

**Original:** 28.22

**中文:** 28.22 28.22

<a id="S175"></a>
**Source:** p.8 S175  
**Type:** body  
**Confidence:** high

**Original:** 4.21

**中文:** 4.21 4.21

<a id="S176"></a>
**Source:** p.8 S176  
**Type:** body  
**Confidence:** high

**Original:** 1.110

**中文:** 1.110


## Page 9

<a id="S177"></a>
**Source:** p.9 S177  
**Type:** body  
**Confidence:** high

**Original:** 10.3389/fpls.2026.1783465

**中文:** 10.3389/fpls.2026.1783465

<a id="S178"></a>
**Source:** p.9 S178  
**Type:** body  
**Confidence:** high

**Original:** Comparison of reconstruction results from (a) dense MVS, (b) NeRF, and (c) 3DGS. Insets show zoomed regions highlighting differences in geometric and photometric fidelity.

**中文:** 进行 (a) 密集MVS, (b) NeRF和 (c) 3DGS的重建结果进行比较.插件显示了缩小区域,突出了几何和光学忠诚度的差异.

<a id="S179"></a>
**Source:** p.9 S179  
**Type:** body  
**Confidence:** medium

**Original:** separation, with plant splats consistently scoring above 0.6 and background splats below 0.4, resulting in cleaner boundaries and stronger spatial consistency. Increasing to five cues (N = 5) further expands the high-score regions spatially but reduces peak sharpness, indicating that while additional cues enhance coverage, they yield diminishing returns relative to the increased annotation effort. For all subsequent experiments, we therefore adopt N = 3

**中文:** 植物间的分离,植物间的分离总是超过0.6分,背景间的分离总是低于0.4分,从而产生了更清洁的边界和更强大的空间一致性.增加到五个线索 (N = 5) 进一步扩大了高分数区域的空间,但降低了峰值的尖度,这表明,虽然额外的线索增强了覆盖率,但它们相对于增加的标注努力产生了减少的回报.因此,我们为所有随后的实验采用了N = 3.

<a id="S180"></a>
**Source:** p.9 S180  
**Type:** body  
**Confidence:** high

**Original:** cues per plant as a balanced configuration between effort and quality.

**中文:** 根据植物的指标,努力和质量之间有一个平衡的配置.

<a id="S181"></a>
### 3.2.2 Ablation analysis of LCR-GS components
**Source:** p.9 S181  
**Type:** section  
**Confidence:** high

**Original:** 3.2.2 Ablation analysis of LCR-GS components

**中文:** 【标题暂译】3.2.2 Ablation analysis of LCR-GS components

<a id="S182"></a>
**Source:** p.9 S182  
**Type:** body  
**Confidence:** high

**Original:** To quantify the contribution of each stage within the LCR-GS pipeline, we conducted a component-wise ablation study on a fixed

**中文:** 为了量化LCR-GS管道中的每个阶段的贡献,我们在固定上进行了组件式缩研究.

<a id="S183"></a>
**Source:** p.9 S183  
**Type:** body  
**Confidence:** high

**Original:** Effect of input cue count (rows) and lift-score retention percentile (columns) on plant-background separation. Higher cue counts and stricter percentiles better isolate high-confidence plant regions.

**中文:** 输入线索数 (row) 和升级点数保留率 (columns) 对植物背景分离的影响.较高的线索数和更严格的百分比更好地隔离高可靠的植物区域.


## Page 10

<a id="S184"></a>
**Source:** p.10 S184  
**Type:** body  
**Confidence:** high

**Original:** 10.3389/fpls.2026.1783465

**中文:** 10.3389/fpls.2026.1783465

<a id="S185"></a>
**Source:** p.10 S185  
**Type:** body  
**Confidence:** high

**Original:** Precision–recall curves showing the incremental effect of each LCR-GS component. Each curve adds one processing stage, illustrating improvements from lifting, geometric clustering, NN-retain, and final CIELAB refinement.

**中文:** 每个曲线都增加了一个处理阶段,说明了从升起,几何聚合,NN-保留和最终CIELAB精炼的改进.

<a id="S186"></a>
**Source:** p.10 S186  
**Type:** body  
**Confidence:** high

**Original:** five-plant diagnostic subset selected across the phenotyping scenes to span plant-size variation, with quantitative results shown in

**中文:** 在整个现象表现中选出的五种植物诊断子集,以跨越植物尺寸变化,数量结果显示在中.

<a id="S187"></a>
**Source:** p.10 S187  
**Type:** body  
**Confidence:** high

**Original:** the lift-score threshold, using percentiles ranging from the top 2% to 20%. The baseline, represented by the Lift score alone, establishes an initial level of separation (blue curve). While precision is high (0.86) under strict thresholds, it collapses to 0.19 as the threshold relaxes, indicating that lift alone cannot reliably distinguish the target plant from high-scoring background splats. Adding geometric clustering substantially stabilizes precision (orange curve), maintaining a narrower 0.72–0.89 range. This demonstrates that enforcing spatial coherence effectively suppresses distant noise clusters that the baseline approach fails to remove. Introducing the NN-retain stage improves recall, recovering thin or sparsely represented structures (e.g., thin leaves) that lie near the main cluster (green curve). This process increases the maximum recall to 0.996 while moderately reducing precision (0.596–0.792), reflecting the expected trade-off between completeness and purity. Finally, applying the CIELAB chromatic refinement yields the best overall performance (purple curve). By leveraging the highfidelity color representation inherent to 3DGS, this stage removes remaining non-plant splats that are geometrically close but chromatically distinct. The full pipeline achieves a precision range of 0.873– 0.971, with only a small reduction in maximum recall. Collectively, these results show that LCR-GS provides a well-balanced extraction strategy with both high precision and high recall.

**中文:** 起点门使用百分比从上 2% 到20%的分数,仅仅由起点分数所代表的基线建立了一个初步的分离水平 (蓝曲线).虽然在严格的门下精度高 (0.86),但随着门放松,它会降至0.19,这表明起点单独无法可靠地区分目标植物与高分的背景点.增加几何聚合物,保持较窄的 0.720.89 范围.这表明,执行空间连贯性结构有效地抑制了距离的噪音集群,而基线方法无法消除.引入NN-retain阶段,可以提高回忆,恢复或稀有地代表薄叶子 (例如,绿) 接近主要曲线 (绿). 这种过程将最大的召回率提高到0.996同时降低了度过度的精度 (0.5960.792),反映了完整性和纯度之间的预期交易.最后,应用CIELAB染色精炼率产生了最佳的整体性能 (紫曲线).通过利用3DGS固有的高真度颜色表示,这一阶段消除了其余的不植物的地址,这些地址是几何接近但染色分异的.整个管道实现了0.8730.971的精度范围,但只有最大的召回率减少了很小.总体而言,这些结果表明LCR-GS提供了高精度和高召回的平衡采摘策略.

<a id="S188"></a>
**Source:** p.10 S188  
**Type:** body  
**Confidence:** high

**Original:** process at a fixed operating point (top 12% of lift score), selected near the precision inflection point of the full pipeline. These visualizations confirm the quantitative trends: the initial Lifting stage separates the foreground but creates perforations (“holes”) on the leaf surfaces where low-confidence Gaussians were discarded; the Clustering stage effectively removes the non-cued, spatially distinct background; the NN-retain stage then visibly “patches”

**中文:** 在一个固定操作点 (上12%的升降分数),在整个管道的精确曲折点附近进行选择.这些可视化证实了数量趋势:最初的升降阶段将前景分开,但在低自信的高西亚人被丢弃的叶片表面上产生孔孔 (孔);集群阶段有效地消除了非,空间分化的背景;NN-保留阶段则可见地补.

<a id="S189"></a>
**Source:** p.10 S189  
**Type:** body  
**Confidence:** high

**Original:** the leaf perforations, corresponding to the recall gain in Figure 5, though it also incorrectly re-includes some non-plant elements near the roots. Before refinement, leakage occurs in two scenarios: (i) loose mask boundaries that include adjacent non-plant regions, and (ii) occluded greenhouse structures (e.g., trellises) in close spatial proximity to the plant. The CIELAB refinement successfully removes these last geometrically-proximate but chromatically-distinct artifacts, yielding a clean and geometrically complete plant instance.

**中文:** 叶子孔孔,相应于图5中的回忆收益,尽管它还错误地重新包含了一些非植物元素在根附近.在精炼之前,泄漏发生在两个场景中: (i) 宽松的面膜边界包括邻近的非植物区域,以及 (ii) 密闭的温室结构 (例如,拖) 接近植物的近距离.CIELAB精炼成功地移除了最后这些几何近似但染色分辨的文物,产生了一个清洁的和几何完整的植物实例.

<a id="F005"></a>
### Fig. 5. 精确率-召回率曲线展示各 LCR-GS 组件的递增效果。每条曲线增加一个处理阶段，说明提升、几何聚类、NN 保留以及最终 CIELAB 精修带来的改进。
**Placed near:** p.10 S189  
**Source:** p.10 manual-layout  
**Crop confidence:** high

![Fig. 5](assets/fig5.png)

**Original caption:** FIGURE 5. Precision-recall curves showing the incremental effect of each LCR-GS component. Each curve adds one processing stage, illustrating improvements from lifting, geometric clustering, NN-retain, and final CIELAB refinement.

**中文图注:** 图 5. 精确率-召回率曲线展示各 LCR-GS 组件的递增效果。每条曲线增加一个处理阶段，说明提升、几何聚类、NN 保留以及最终 CIELAB 精修带来的改进。

**Reading note:** 重点查看该图如何支撑相邻正文中的流程、比较、消融或性状提取结果。

<a id="S190"></a>
### 3.2.3 Quantitative and qualitative evaluation of
**Source:** p.10 S190  
**Type:** section  
**Confidence:** high

**Original:** 3.2.3 Quantitative and qualitative evaluation of

**中文:** 【标题暂译】3.2.3 Quantitative and qualitative evaluation of

<a id="S191"></a>
**Source:** p.10 S191  
**Type:** body  
**Confidence:** high

**Original:** plant-level extraction To evaluate the performance of the LCR-GS pipeline, we benchmark it against a training-based 2D-lifting approach adapted from the Gaussian Grouping framework (Ye et al., 2024). To ensure a fair comparison, we retrained Gaussian Grouping using the same 2D cues (YOLO-World detections and SAM masks) used by LCRGS. This isolates the methodological difference between trainingintegrated grouping (Gaussian Grouping) and post-reconstruction filtering (LCR-GS). The quantitative comparison is summarized in

**中文:** 为了评估LCR-GS管道的性能,我们将其与从高斯集团框架中改编的训练式2D升级方法进行比较 (Ye et al., 2024).为了确保公平的比较,我们使用LCRGS使用的相同的2D线索 (YOLO-World检测和SAM掩膜) 重新训练了高斯集团.这将训练集团和重建后过 (LCR-GS) 之间的方法差异隔离.数量比较总结在.

<a id="S192"></a>
**Source:** p.10 S192  
**Type:** body  
**Confidence:** high

**Original:** validation set (6 scenes, 30 plants) at the fixed top-12% lift-score threshold identified in the ablation study. From Table 3, the training-based Gaussian Grouping baseline achieves high recall (0.925) but suffers from very low precision (0.381). This imbalance suggests that the general-purpose segmentation architectures are sensitive to the quality of 2D masks, with the training process incorporating noise and clutter present in the input masks. In contrast, the LCR-GS achieves precision of 0.933 and mIoU of 0.890, demonstrating that its geometric and chromatic filtering stages effectively suppress non-plant Gaussians when the 2D inputs contain clutter or loose boundaries. In addition to improved accuracy, LCR-GS offers computational advantages. Because it operates after 3DGS reconstruction, it

**中文:** 从表3开始,基于训练的高斯基群组基线达到高回调 (0.925),但却受到非常低精度 (0.381).这种不平衡表明,一般用途的分割架构对2D掩膜的质量敏感,训练过程中包含了输入掩膜中的噪音和混乱.相反,LCR-GS达到0.933的精度和0.890的mIoU,证明其几何和染色过阶段有效地抑制了2DGS输入含有混乱或松的边界时的非植物加索人.除了精度之外,LCR-GS也提供了更好的操作优势.因为它在3D重建后,它可以实现更好的运行.


## Page 11

<a id="S193"></a>
**Source:** p.11 S193  
**Type:** body  
**Confidence:** high

**Original:** 10.3389/fpls.2026.1783465

**中文:** 10.3389/fpls.2026.1783465

<a id="F006"></a>
### Fig. 6. 两株植株上的消融结果。各列展示 LCR-GS 各阶段的递增效果，插图突出叶片完整性和背景去除方面的改进。
**Placed near:** p.11  
**Source:** p.11 manual-layout  
**Crop confidence:** high

![Fig. 6](assets/fig6.png)

**Original caption:** FIGURE 6. Ablation results on two plants. Columns show the incremental effects of each LCR-GS stage, with insets highlighting improvements in leaf completeness and background removal.

**中文图注:** 图 6. 两株植株上的消融结果。各列展示 LCR-GS 各阶段的递增效果，插图突出叶片完整性和背景去除方面的改进。

**Reading note:** 重点查看该图如何支撑相邻正文中的流程、比较、消融或性状提取结果。

<a id="S194"></a>
**Source:** p.11 S194  
**Type:** body  
**Confidence:** high

**Original:** Ablation results on two plants. Columns show the incremental effects of each LCR-GS stage, with insets highlighting improvements in leaf completeness and background removal.

**中文:** 两种植物的废除结果.列表显示了每个LCR-GS阶段的增量效果,插入突出了叶片完整性和背景移除的改善.

<a id="S195"></a>
**Source:** p.11 S195  
**Type:** body  
**Confidence:** high

**Original:** requires no per-scene retraining. The extraction process requires approximately 12 seconds per plant for geometric clustering (OPTICS + NN-retain) and maintains a peak memory footprint of 1.12 GB. The pipeline also performs substantial data reduction: dense greenhouse reconstructions contain an average of 2.1M Gaussians, whereas extracted plant instances contain only 15K21K Gaussians, a reduction of roughly 99%. This compact representation enables the subsequent organ-level segmentation to run efficiently within standard GPU memory constraints. On an NVIDIA RTX 4080 GPU (16 GB), end-to-end processing for a single greenhouse scene (five plants) requires approximately 2–2.5 h. SfM accounts for most of the runtime; the extraction, segmentation, and trait computation stages together account for the

**中文:** 采集过程需要每种工厂的几何聚合 (OPTICS + NN-retain) 约12秒,并且保持1.12GB的最高内存足迹.该管道还实现了大量的数据减少:密集温室重建中平均含有2.1M高斯,而采集的工厂实例中只含有15K21K高斯,减少了99%.这种紧的表示使随后的器官级分割能够在标准GPU内有效运行.在NVIDIA RTX 4080 GPU (16GB) 上,单个温室 (五个工厂) 的端到端处理需要大约22.5h.SfM占大部分运行时间;采集,分割和计算阶段一起占的特征.

<a id="S196"></a>
**Source:** p.11 S196  
**Type:** body  
**Confidence:** high

**Original:** remaining fraction. Structure-from-motion requires approximately 1.5–2.0 h, and 3DGS optimization requires 20–30 min.

**中文:** 结构从动作需要大约1.52.0h,3DGS优化需要2030分钟.

<a id="S197"></a>
**Source:** p.11 S197  
**Type:** body  
**Confidence:** high

**Original:** pipeline. From the original RGB video frames (Figure 7a), the 3DGS reconstruction produces a dense, photorealistic scene (Figure 7b). The LCR-GS pipeline isolates individual plant instances (Figure 7c), preserving fine leaf structure and stem geometry while cleanly separating the plant from the surrounding greenhouse structures. These results confirm that LCR-GS provides both high-quality plant isolation and compact per-plant representations suitable for downstream phenotypic analysis.

**中文:** 从原始的RGB视频框架 (图7a) 中,3DGS重建产生了密集的,光现实化的场景 (图7b).LCR-GS管道隔离了单个植物实例 (图7c),保持了细叶结构和干结构,同时清洁地将植物与周围的温室结构分离.这些结果证实LCR-GS提供了高质量的植物隔离和适合下游表型分析的紧的植物性表示.

<a id="F007"></a>
### Fig. 7. (a) 原始 RGB 帧、(b) 重建的 3DGS 场景以及 (c) 使用 LCR-GS 提取的植株实例的比较。
**Placed near:** p.11 S197  
**Source:** p.12 manual-layout  
**Crop confidence:** high

![Fig. 7](assets/fig7.png)

**Original caption:** FIGURE 7. Comparison of (a) original RGB frames, (b) the reconstructed 3DGS scene, and (c) extracted plant instances obtained using LCR-GS.

**中文图注:** 图 7. (a) 原始 RGB 帧、(b) 重建的 3DGS 场景以及 (c) 使用 LCR-GS 提取的植株实例的比较。

**Reading note:** 重点查看该图如何支撑相邻正文中的流程、比较、消融或性状提取结果。

<a id="S198"></a>
### 3.3 Organ-level instance segmentation
**Source:** p.11 S198  
**Type:** section  
**Confidence:** high

**Original:** 3.3 Organ-level instance segmentation

**中文:** 【标题暂译】3.3 Organ-level instance segmentation

<a id="S199"></a>
**Source:** p.11 S199  
**Type:** body  
**Confidence:** high

**Original:** performance

**中文:** 表演

<a id="S200"></a>
**Source:** p.11 S200  
**Type:** body  
**Confidence:** high

**Original:** and a training-based baseline.

**中文:** 培训基础的基础.

<a id="S201"></a>
**Source:** p.11 S201  
**Type:** body  
**Confidence:** high

**Original:** Configuration

**中文:** 配置

<a id="S202"></a>
**Source:** p.11 S202  
**Type:** body  
**Confidence:** high

**Original:** Precision

**中文:** 精确的

<a id="S203"></a>
**Source:** p.11 S203  
**Type:** body  
**Confidence:** high

**Original:** Recall

**中文:** 提醒

<a id="S204"></a>
**Source:** p.11 S204  
**Type:** body  
**Confidence:** high

**Original:** mIoU

**中文:** 现在我已经知道了.

<a id="S205"></a>
**Source:** p.11 S205  
**Type:** body  
**Confidence:** high

**Original:** Gaussian Grouping

**中文:** 盖斯人群集

<a id="S206"></a>
**Source:** p.11 S206  
**Type:** body  
**Confidence:** high

**Original:** 0.381

**中文:** 它们是0.381.

<a id="S207"></a>
**Source:** p.11 S207  
**Type:** body  
**Confidence:** high

**Original:** 0.925

**中文:** 九百二十五 0.925

<a id="S208"></a>
**Source:** p.11 S208  
**Type:** body  
**Confidence:** high

**Original:** 0.370

**中文:** 0.370 0.370

<a id="S209"></a>
**Source:** p.11 S209  
**Type:** body  
**Confidence:** high

**Original:** LCR-GS

**中文:** 美国LCR-GS

<a id="S210"></a>
**Source:** p.11 S210  
**Type:** body  
**Confidence:** high

**Original:** 0.933

**中文:** 九百三十三章 0.933

<a id="S211"></a>
**Source:** p.11 S211  
**Type:** body  
**Confidence:** high

**Original:** 0.961

**中文:** 0.961 0.961

<a id="S212"></a>
**Source:** p.11 S212  
**Type:** body  
**Confidence:** high

**Original:** 0.890

**中文:** 0.890 0.890

<a id="S213"></a>
**Source:** p.11 S213  
**Type:** body  
**Confidence:** high

**Original:** To evaluate organ-level segmentation performance, an additional dataset of 140 melon plants was collected using the same acquisition and extraction pipeline described in Section 2. The dataset was split into training (70%), validation (15%), and test (15%) subsets, yielding 98 training plants, 21 validation plants, and 21 test plants. To improve model generalization, training samples

**中文:** 为了评估器官级分类性能,使用第二节所述的相同采购和提取管道收集了140个桃植物的额外数据集.该数据集分为培训 (70%),验证 (15%) 和测试 (15%) 的子集,产生了98个培训厂,21个验证厂和21个测试厂.


## Page 12

<a id="S214"></a>
**Source:** p.12 S214  
**Type:** body  
**Confidence:** high

**Original:** 10.3389/fpls.2026.1783465

**中文:** 10.3389/fpls.2026.1783465

<a id="S215"></a>
**Source:** p.12 S215  
**Type:** body  
**Confidence:** high

**Original:** Comparison of (a) original RGB frames, (b) the reconstructed 3DGS scene, and (c) extracted plant instances obtained using LCR-GS.

**中文:** 进行 (a) 原始RGB框架的比较, (b) 复制的3DGS场景,以及 (c) 使用LCR-GS获得的提取植物实例.

<a id="S216"></a>
**Source:** p.12 S216  
**Type:** body  
**Confidence:** medium

**Original:** were generated using a relaxed CIELAB chromaticity filter (90% quantile), deliberately including a small proportion of non-plant Gaussians. This strategy, which increases plant recall, ensures that fine leaf details are preserved during training. Table 4 reports performance across three segmentation backbones on the test set, evaluated using Average Precision at IoU 0.5 (AP50) and Average Recall (AR50). As demonstrated in Table 4, the PTv3 backbone achieves the best overall results (mean AP50 = 0.924, AR50 = 0.952), outperforming both PointGroup and SoftGroup. Stem segmentation benefits substantially from PTv3’s long-range attention mechanism, which is essential for modeling the elongated, continuous geometry of vines. Leaf segmentation remains more challenging across all methods due to inter-leaf occlusion and variability in leaf size and curvature.

**中文:** 采用一种放松的CIELAB染色度过器 (90%量化),故意包括少量的非植物高西人.这种策略,增加了植物回忆,确保在训练中保持细叶细节.表4报告了测试组上的三个分割背骨的性能,使用IoU 0.5 (AP50) 和平均回忆 (AR50) 进行评估.如表4所示,PTv3背骨获得了最佳的总体结果 (平均AP50 =0.924,AR50 =0.952),超过了PointGroup和SoftGroup.干部分割取益于PTv3的长距离关注机制,这是对模拟长长,连续的葡萄形态至关重要的. 叶片分割仍然在所有方法中更加具有挑战性,因为叶片间的封闭和叶片尺寸和曲率的可变性.

<a id="S217"></a>
**Source:** p.12 S217  
**Type:** body  
**Confidence:** high

**Original:** with per-instance color coding. The middle row presents inference results from the PTv3 model, demonstrating that most organs are correctly segmented with clear boundaries. A small fraction of leaves exhibits imperfect separation from stems, particularly at attachment points where point density is high and geometric features are ambiguous (plant g). Additionally, in regions where non-plant elements are present, the model must distinguish plant tissue from a structurally similar background based on subtle geometric cues (plant a, c).

**中文:** 中间行呈现了PTv3模型的推断结果,证明大多数器官是正确的划分,有明确的边界.小部分叶子显示出不完美的分离从茎,特别是在连接点,点密度高,几何特征模糊 (植物 g).此外,在非植物元素存在的地区,模型必须根据微妙的几何线索 (植物 a, c) 区分植物组织与结构相似的背景.

<a id="S218"></a>
**Source:** p.12 S218  
**Type:** body  
**Confidence:** high

**Original:** The bottom row shows the final results after remapping segmented instances back to the 3DGS representation. Compared to the point cloud segmentation (middle row), the Gaussian-based rendering exhibits cleaner boundaries and reduced noise. This improvement is primarily due to the CIELAB color refinement being reapplied during the remapping stage. This step effectively removes residual non-plant Gaussians by using the strict extraction threshold (80% quantile) defined in the original extraction pipeline. Furthermore, this remapped 3DGS representation offers unique advantages for post-processing. Properties inherent to the 3DGS format, such as per-Gaussian opacity and explicit ellipsoidal geometry, allow for additional targeted filtering after segmentation. For instance, reconstruction artifacts that manifest as elongated, lowopacity Gaussians (e.g., elongated, low-opacity splats with high axis ratios) could be statistically identified from the final instances, demonstrating a clear advantage of operating in the Gaussian space over point-based methods.

**中文:** 下一行显示了重新映射分割实例后的最终结果,回到3DGS表示.与点云分割 (中行) 相比,基于高斯的染表现出更清洁的边界和减少噪音.这种改进主要是由于CIELAB颜色精炼在重新映射阶段重新应用.此步骤通过使用原始抽取管道所定义的严格的抽取门 (80%量化) 有效地消除残留非植入的高斯人.此外,这种重新映射的3DGS表示为后处理提供了独特的优势. 3DGS格式固有的特性,如每高斯的细度和明确的形形状,允许在分割后进行额外的化. 例如,从最后的实例中,可以统计地识别出长长的,低率高素质高素质高素质小块的重建物件,这表明在高素质空间运行比点式方法具有明显的优势.

<a id="S219"></a>
### 3.4 Phenotypic trait validation and
**Source:** p.12 S219  
**Type:** section  
**Confidence:** high

**Original:** 3.4 Phenotypic trait validation and

**中文:** 【标题暂译】3.4 Phenotypic trait validation and

<a id="S220"></a>
**Source:** p.12 S220  
**Type:** body  
**Confidence:** high

**Original:** population analysis To evaluate the accuracy of the proposed phenotyping pipeline, trait values computed from the organ-level 3DGS representations were compared against manual measurements. Plant height and leaf count were selected for validation because they are directly measurable and serve as key indicators of early vegetative growth.

**中文:** 为了评估拟议的表型结构的精确性,从器官级3DGS表示计算的特征值与手动测量进行了比较.植物高度和叶数被选为验证,因为它们是直接可测量的,并作为早期植物生长的关键指标.

<a id="S221"></a>
**Source:** p.12 S221  
**Type:** body  
**Confidence:** high

**Original:** Organ-level segmentation performance (AP50 and AR50) of different point-cloud instance segmentation models.

**中文:** 不同点云实例分割模型的器官级分割性能 (AP50和AR50).

<a id="S222"></a>
**Source:** p.12 S222  
**Type:** body  
**Confidence:** high

**Original:** AP50

**中文:** AP50

<a id="S223"></a>
**Source:** p.12 S223  
**Type:** body  
**Confidence:** high

**Original:** AR50

**中文:** 亚美50

<a id="S224"></a>
**Source:** p.12 S224  
**Type:** body  
**Confidence:** high

**Original:** Method PTv3 + PointGroup

**中文:** PTv3 +点组方法

<a id="S225"></a>
**Source:** p.12 S225  
**Type:** body  
**Confidence:** high

**Original:** Stem

**中文:** 语音

<a id="S226"></a>
**Source:** p.12 S226  
**Type:** body  
**Confidence:** high

**Original:** Leaf

**中文:** 叶

<a id="S227"></a>
**Source:** p.12 S227  
**Type:** body  
**Confidence:** high

**Original:** Mean

**中文:** 意思是

<a id="S228"></a>
**Source:** p.12 S228  
**Type:** body  
**Confidence:** high

**Original:** Stem

**中文:** 语音

<a id="S229"></a>
**Source:** p.12 S229  
**Type:** body  
**Confidence:** high

**Original:** Leaf

**中文:** 叶

<a id="S230"></a>
**Source:** p.12 S230  
**Type:** body  
**Confidence:** high

**Original:** Mean

**中文:** 意思是

<a id="S231"></a>
**Source:** p.12 S231  
**Type:** body  
**Confidence:** high

**Original:** 0.943

**中文:** 0.943

<a id="S232"></a>
**Source:** p.12 S232  
**Type:** body  
**Confidence:** high

**Original:** 0.905

**中文:** 0.905 0.905

<a id="S233"></a>
**Source:** p.12 S233  
**Type:** body  
**Confidence:** high

**Original:** 0.924

**中文:** 它们的数量为0.924.

<a id="S234"></a>
**Source:** p.12 S234  
**Type:** body  
**Confidence:** high

**Original:** 0.977

**中文:** 0.977

<a id="S235"></a>
**Source:** p.12 S235  
**Type:** body  
**Confidence:** high

**Original:** 0.927

**中文:** 0.927

<a id="S236"></a>
**Source:** p.12 S236  
**Type:** body  
**Confidence:** high

**Original:** 0.952

**中文:** 0.952

<a id="S237"></a>
**Source:** p.12 S237  
**Type:** body  
**Confidence:** high

**Original:** PointGroup

**中文:** 点组

<a id="S238"></a>
**Source:** p.12 S238  
**Type:** body  
**Confidence:** high

**Original:** 0.848

**中文:** 只有0.848.

<a id="S239"></a>
**Source:** p.12 S239  
**Type:** body  
**Confidence:** high

**Original:** 0.903

**中文:** 0.903 0.903

<a id="S240"></a>
**Source:** p.12 S240  
**Type:** body  
**Confidence:** high

**Original:** 0.876

**中文:** 0.876 0.876

<a id="S241"></a>
**Source:** p.12 S241  
**Type:** body  
**Confidence:** high

**Original:** 0.933

**中文:** 九百三十三章 0.933

<a id="S242"></a>
**Source:** p.12 S242  
**Type:** body  
**Confidence:** high

**Original:** 0.927

**中文:** 0.927

<a id="S243"></a>
**Source:** p.12 S243  
**Type:** body  
**Confidence:** high

**Original:** 0.930

**中文:** 0.930

<a id="S244"></a>
**Source:** p.12 S244  
**Type:** body  
**Confidence:** high

**Original:** SoftGroup

**中文:** 软组

<a id="S245"></a>
**Source:** p.12 S245  
**Type:** body  
**Confidence:** high

**Original:** 0.836

**中文:** 只有0.836.

<a id="S246"></a>
**Source:** p.12 S246  
**Type:** body  
**Confidence:** high

**Original:** 0.948

**中文:** 0.948

<a id="S247"></a>
**Source:** p.12 S247  
**Type:** body  
**Confidence:** high

**Original:** 0.892

**中文:** 0.892 0.892

<a id="S248"></a>
**Source:** p.12 S248  
**Type:** body  
**Confidence:** high

**Original:** 0.867

**中文:** 0.867 0.867

<a id="S249"></a>
**Source:** p.12 S249  
**Type:** body  
**Confidence:** high

**Original:** 0.948

**中文:** 0.948

<a id="S250"></a>
**Source:** p.12 S250  
**Type:** body  
**Confidence:** high

**Original:** 0.908

**中文:** 0.908 0.908


## Page 13

<a id="S251"></a>
**Source:** p.13 S251  
**Type:** body  
**Confidence:** high

**Original:** 10.3389/fpls.2026.1783465

**中文:** 10.3389/fpls.2026.1783465

<a id="F008"></a>
### Fig. 8. 七株代表性测试植株 (a-g) 的器官级分割结果比较。每株植株中，上排为真实标注，中排为点云空间中的 PTv3 预测结果，下排为 3DGS 表示中的重建器官实例。
**Placed near:** p.13  
**Source:** p.13 manual-layout  
**Crop confidence:** high

![Fig. 8](assets/fig8.png)

**Original caption:** FIGURE 8. Comparison of organ-level segmentation results across seven representative test plants (a-g). For each plant, the top row shows ground-truth annotations, the middle row shows PTv3 predictions in point-cloud space, and the bottom row shows reconstructed organ instances in the 3DGS representation.

**中文图注:** 图 8. 七株代表性测试植株 (a-g) 的器官级分割结果比较。每株植株中，上排为真实标注，中排为点云空间中的 PTv3 预测结果，下排为 3DGS 表示中的重建器官实例。

**Reading note:** 重点查看该图如何支撑相邻正文中的流程、比较、消融或性状提取结果。

<a id="S252"></a>
**Source:** p.13 S252  
**Type:** body  
**Confidence:** high

**Original:** Comparison of organ-level segmentation results across seven representative test plants (a–g). For each plant, the top row shows ground-truth annotations, the middle row shows PTv3 predictions in point-cloud space, and the bottom row shows reconstructed organ instances in the 3DGS representation.

**中文:** 对象七个代表性测试装置 (ag) 的器官级分类结果:对于每个器官,上行显示了地面真相标注,中行显示了点云空间中的PTv3预测,下行显示了3DGS表示中部器官重建实例.

<a id="S253"></a>
**Source:** p.13 S253  
**Type:** body  
**Confidence:** high

**Original:** Manual height was measured from the pot surface to the apical meristem using a ruler, and leaf count was obtained by visually enumerating fully expanded leaves (≥2 cm diameter). All validation results reported in this section are based on the early-vegetative muskmelon cohort (6 scenes, 30 plants).

**中文:** 从面到形位的手动高度通过一条规律来测量,并通过视觉计算完全扩大的叶子 (直径≥2厘米) 来获得叶子数量.本节报告的所有验证结果都是基于早期植物性桃群体 (6场景,30种植物).

<a id="S254"></a>
**Source:** p.13 S254  
**Type:** body  
**Confidence:** medium

**Original:** RMSE = 1.88 cm, MAPE = 6.4%), confirming that the 3DGS reconstruction accurately captures vertical plant structure. Statistical analysis reveals a near-constant systematic

**中文:** (RMSE = 1.88厘米,MAPE = 6.4%),证实3DGS重建精确捕捉垂直植物结构.统计分析显示,几乎是恒定的系统.

<a id="S255"></a>
**Source:** p.13 S255  
**Type:** body  
**Confidence:** medium

**Original:** overestimation (mean residual: +1.58 cm, median: +1.60 cm, p < 10 - 8), with 29 of 30 plants showing positive residuals. Approximately 70% of residuals fall within ±2 cm and 93% within ±3 cm of the ground-truth value. Calibration statistics (slope = 1.025, intercept = 0.852) confirm that this bias is primarily a constant offset rather than a scale distortion. A weak positive association exists between plant size and residual magnitude (Spearman r = 0.38, p = 0.039), indicating that taller plants tend to show slightly larger overestimation, though the effect is modest. This offset is attributed to partial inclusion of below-pot stem

**中文:** 校准统计 (斜率 = 1.025,截截取 = 0.852) 证实这种偏差主要是恒定偏移而不是尺度扭曲.植物大小和残留大小之间存在弱的积极关联 (斯皮尔曼r = 0.38,p = 0.039),这表明较高的植物往往表现出略有过高的估值,尽管影响较小.这种偏差归因于部分包含子根根以下的子.

<a id="S256"></a>
**Source:** p.13 S256  
**Type:** body  
**Confidence:** high

**Original:** Validation of computed phenotypic traits against manual measurements. Left: correlation between computed and measured plant height. Right: correlation between computed and measured leaf count.

**中文:** 验证计算的现象特征与手动测量.左:计算和测量的植物高度之间的相关性.右:计算和测量的叶子数量的相关性.


## Page 14

<a id="S257"></a>
**Source:** p.14 S257  
**Type:** body  
**Confidence:** high

**Original:** 10.3389/fpls.2026.1783465

**中文:** 10.3389/fpls.2026.1783465

<a id="S258"></a>
**Source:** p.14 S258  
**Type:** body  
**Confidence:** medium

**Original:** The trait estimates nevertheless permit examination of withincohort inter-trait relationships (Figure 10). Size-related traits display strong correlations - height with leaf count (r = 0.93), node count (r = 0.91), and LAI (r = 0.73) - indicating a coherent growth axis characteristic of early vegetative development. In contrast, average leaf area showed weaker associations with other traits (r = 0.61 with height, r = 0.37 with leaf count), indicating that while overall plant size (height, node count) represents a coordinated growth axis, individual leaf size is a partially independent trait. These trait distributions and correlations provide a basis for downstream phenotypic analyses, including growth modeling, quantitative trait locus mapping, and genotype-by-environment interaction studies.

**中文:** 品格估计不过允许对内科特之间的关系进行检查 (图 10).与尺寸相关的特征显示出强烈的相关性 - 高度与叶子数量 (r = 0.93),节点数量 (r = 0.91),LAI (r = 0.73) - 表明早期植物发展的连贯生长轴.相比之下,平均叶子面积与其他特征的关系较弱 (r = 0.61 高度,r = 0.37 叶子数量),表明,虽然整体植物大小 (高度,节点数量) 表示协调的生长轴,但单个叶子大小是一个部分独立的特征.这些分布和相关性为下游的生长现象分析提供了基础,包括增长调整,定量性基因特征和环境-环节相互作用研究.

<a id="F001"></a>
### Fig. 1. 基于三维高斯泼溅（3DGS）的端到端表型分析流程概览。
**Placed near:** p.14 S258  
**Source:** p.3 manual-layout  
**Crop confidence:** high

![Fig. 1](assets/fig1.png)

**Original caption:** FIGURE 1. Overview of the end-to-end phenotyping pipeline based on 3D Gaussian Splatting (3DGS).

**中文图注:** 图 1. 基于三维高斯泼溅（3DGS）的端到端表型分析流程概览。

**Reading note:** 重点查看该图如何支撑相邻正文中的流程、比较、消融或性状提取结果。

<a id="F010"></a>
### Fig. 10. 营养生长期器官级和植株级性状的 Pearson 相关矩阵，右侧面板标示显著性水平。
**Placed near:** p.14 S258  
**Source:** p.15 manual-layout  
**Crop confidence:** high

![Fig. 10](assets/fig10.png)

**Original caption:** FIGURE 10. Pearson correlation matrices of organ- and plant-level traits at the vegetative stage, with significance levels indicated on the right panel.

**中文图注:** 图 10. 营养生长期器官级和植株级性状的 Pearson 相关矩阵，右侧面板标示显著性水平。

**Reading note:** 重点查看该图如何支撑相邻正文中的流程、比较、消融或性状提取结果。

<a id="S259"></a>
**Source:** p.14 S259  
**Type:** body  
**Confidence:** medium

**Original:** segments in the reconstructed plant volume. Under the uniformcontainer conditions of this study, the near-constant offset does not affect inter-plant ranking or relative trait comparisons. Leaf count shows moderate but acceptable agreement (R² = 0.86, RMSE = 0.63 leaves, MAPE = 6.0%). The mean residual is not statistically significant (−0.20 leaves, t = −1.80, p = 0.083), though calibration analysis reveals a slight negative intercept (−0.986, p = 0.042) and a slope slightly above unity (1.111, p = 0.091), reflecting mild undercounting in sparse canopies and over-counting in dense ones. As shown in Figure 9 (right), 90% of predictions lie within ±1 leaf, providing sufficient accuracy for population-level phenotyping despite localized segmentation ambiguities (e.g., at leaf–stem junctions or under tight occlusion). Plant height and leaf count are the only traits directly validated against manual measurements in this study. The remaining traits reported in Table 5 (leaf area, LAI, mean internode length, and stem node count) are pipeline-derived descriptors computed from the segmented 3DGS representations without independent physical verification; they are reported for within-cohort comparison under the assumption of consistent estimation across plants. Population-level trait statistics are summarized in Table 5. As depicted in Table 5, substantial morphological diversity is observed across the 30-plant cohort: the coefficient of variation ranges from 17.1% (LAI) to 47.0% (stem node count). This variation may reflect genotypic differences, microenvironmental heterogeneity, and stochastic developmental processes. Height and node count show the widest absolute ranges (47.8 cm and 13 nodes), whereas LAI and mean internode length vary more modestly (1.13 m²/m² and 4.2 cm). Because leaf-area-related traits depend on the geometric suitability of the extracted leaf instances, we performed an internal robustness analysis for area estimation. Of the 224 extracted leaf instances, 204 (91.1%) satisfied geometric screening criteria combining minimum point support (lowest-support 5% excluded), spatial connectivity, and projection planarity; repeating the plantlevel area analysis after excluding the remaining 20 instances yielded highly consistent within-cohort rankings. The default ashape estimator also showed high rank agreement with alternative projected-area methods, and plant-level rankings remained stable across tested a parameter values above the default setting. These analyses support the internal consistency of the area estimator for within-cohort comparison but do not substitute for independent physical validation of the specific workflow used here.

**中文:** 在本研究的统一容器条件下,近常数的偏移不会影响植物间排名或相对特征的比较.叶子数量显示出温和但可接受的协议 (R2 = 0.86,RMSE = 0.63叶,MAPE = 6.0%).平均残余量不具有统计意义 (−0.20叶,t = −1.80,p = 0.083),尽管校准分析显示有轻微的负面截截 (−0.986,p = 0.042) 和斜率略高于单位 (1.111,p = 0.091),反映了稀有叶子的低数和密集的过度数.如图9 (右侧),90%的预测位于1叶内,尽管存在较为紧密的位数或位数 (位数). 植物高度和叶子数量是本研究中与手动测量直接验证的唯一特征.如表5所报告的剩余特征 (叶片面积,LAI,平均内极长度和干节数量) 是从3DGS分段表示计算的管道衍生描述器,而没有独立的物理验证;它们在植物间的一致估计下被报告为内队比较. 种群水平特征统计数据在表5中总结.如表5所示,在30种植物中观察到了大量的形态多样性:变化系数在17.1% (LAI) 到47.0% (节点数量) 之间.这种变化可能反映基因型差异,微环境异性和体育发展过程. 高度和节点数量显示了最大的绝对范围 (47.8厘米和13节点),而LAI和平均内节点长度更适度 (1.13m2/m2和4.2厘米).由于叶子面积相关的特征取决于提取的叶子实例的几何性质,因此我们进行了内部强度分析,以估计面积.在224个提取的叶子实例中,204个 (91.1%) 满足了结合最小点支持 (最低支持5%除外),空间连接性和投影平率的几何性质的几何选标准;在排除剩余20个实例后,重复植物水平面积分析,产生了高度一致的团队排名. 默认的ashape估计器也显示了与其他预测区域方法的高排名一致性,并且在测试的参数值上,工厂级排名保持稳定,高于默认设置.这些分析支持区间估计器内部一致性,但不会取代独立的物理验证,用于此处使用的具体工作流程.

<a id="F009"></a>
### Fig. 9. 计算得到的表型性状与人工测量结果的验证。左：计算株高与测量株高的相关性；右：计算叶片数与测量叶片数的相关性。
**Placed near:** p.14 S259  
**Source:** p.13 manual-layout  
**Crop confidence:** medium

![Fig. 9](assets/fig9.png)

**Original caption:** FIGURE 9. Validation of computed phenotypic traits against manual measurements. Left: correlation between computed and measured plant height. Right: correlation between computed and measured leaf count.

**中文图注:** 图 9. 计算得到的表型性状与人工测量结果的验证。左：计算株高与测量株高的相关性；右：计算叶片数与测量叶片数的相关性。

**Reading note:** 重点查看该图如何支撑相邻正文中的流程、比较、消融或性状提取结果。

<a id="S260"></a>
### 3.5 Scope, limitations, and future directions
**Source:** p.14 S260  
**Type:** section  
**Confidence:** high

**Original:** 3.5 Scope, limitations, and future directions

**中文:** 【标题暂译】3.5 Scope, limitations, and future directions

<a id="S261"></a>
**Source:** p.14 S261  
**Type:** body  
**Confidence:** high

**Original:** In practical phenotyping systems, scene reconstruction and trait extraction are often treated as separate stages, yet downstream analyses typically assume pre-isolated plant geometry. The proposed LCR-GS framework addresses this gap by introducing an intermediate extraction stage that converts shared-scene 3DGS reconstructions into analysis-ready plant-level units, improving interoperability with downstream organ-level segmentation and trait computation under structured greenhouse acquisition. In this study, quantitative validation in Section 3.4 is limited to early-vegetative greenhouse muskmelon. To examine how the extraction stage behaves beyond that validated regime, we applied LCR-GS to a mid-stage muskmelon scene, a later-stage scene with increased support-structure entanglement, and two additional potted crop settings (Figure 11). In the midstage muskmelon case, individual plant representations could still be visually distinguished. In the later-stage scene, however, plants extended beyond the trellis height and intertwined with support structures, leading to partially incomplete separation in entangled regions; this marks the current applicability boundary of the present workflow. Preliminary examples in sweet olive (Osmanthus fragrans) and peanut (Arachis hypogaea) further highlighted that residual noise patterns differed with plant morphology and cultivation setup. Potboundary residues, soil-associated points, and the retention or removal of woody stems depended on the intended phenotyping target, identifying task-specific filtering adjustment as one likely challenge in other crop settings. The validated trait results also require bounded interpretation. Plant height and leaf count remain the only directly validated traits

**中文:** 在实际的表型制造系统中,场景重建和特征提取通常被视为独立的阶段,但下游分析通常假设预先隔离的植物几何学.拟议的LCR-GS框架通过引入一个中间的提取阶段来解决这一差距,将共享场景3DGS重建转化为分析的植物级单元,提高了下游器官级分割和结构化绿色收购下的特征计算的互操作性.本研究中,3.4节的定量验证仅限于早期植物性温室桃.为了检查提取阶段如何在验证的制度之外表现,我们将LCR-GS应用于中阶段的桃场景,后阶段的支持结构纠场景和两个增加的农作物桃设置 (图 11). 在中阶段的瓜案例中,单个植物的表示仍然可以视觉区分.在后期的场景中,植物扩展到杆高度并与支结构交织,导致在纠的区域中部分不完整的分离;这标志着当前工作流程的当前适用性界限.甜蜜橄 (Osmanthus fragrans) 和花生 (Arachis hypogaea) 的初步例子进一步强调了剩余噪音模式与植物形态和种植设置不同.土相关的残留物,土壤相关点和木干的保留或移除取决于预期的异型目标,确定任务特定的过调整是其他种植设置中的可能挑战之一.结果也需要有效的界限解释. 植物高度和叶子数量仍然是唯一直接验证的特征

<a id="F011"></a>
### Fig. 11. 验证设置之外的 LCR-GS 提取定性示例。左：代表性 3DGS 场景渲染；右：对应提取的植株实例。(a) 温室甜瓜中期；(b) 温室甜瓜后期，箭头标示代表性残留或局部分离不完整区
**Placed near:** p.14 S261  
**Source:** p.15 manual-layout  
**Crop confidence:** medium

![Fig. 11](assets/fig11.png)

**Original caption:** FIGURE 11. Qualitative examples of LCR-GS extraction beyond the validated setting. Left: representative 3DGS scene renderings; right: corresponding extracted plant instances. (a) Mid-stage greenhouse muskmelon. (b) Late-stage greenhouse muskmelon, with arrows indicating representative residual or locally incomplete-separation regions. (c) Sweet olive and (d) peanut, with arrows indicating representative non-target residual regions.

**中文图注:** 图 11. 验证设置之外的 LCR-GS 提取定性示例。左：代表性 3DGS 场景渲染；右：对应提取的植株实例。(a) 温室甜瓜中期；(b) 温室甜瓜后期，箭头标示代表性残留或局部分离不完整区域；(c) 桂花和 (d) 花生，箭头标示代表性非目标残留区域。

**Reading note:** 重点查看该图如何支撑相邻正文中的流程、比较、消融或性状提取结果。

<a id="S262"></a>
**Source:** p.14 S262  
**Type:** body  
**Confidence:** high

**Original:** Trait

**中文:** 特征

<a id="S263"></a>
**Source:** p.14 S263  
**Type:** body  
**Confidence:** high

**Original:** Level

**中文:** 级别 级别

<a id="S264"></a>
**Source:** p.14 S264  
**Type:** body  
**Confidence:** medium

**Original:** Mean ± STD

**中文:** 意思是 ± STD

<a id="S265"></a>
**Source:** p.14 S265  
**Type:** body  
**Confidence:** high

**Original:** Range

**中文:** 范围 范围

<a id="S266"></a>
**Source:** p.14 S266  
**Type:** body  
**Confidence:** high

**Original:** CV(%)

**中文:** CV(%)

<a id="S267"></a>
**Source:** p.14 S267  
**Type:** body  
**Confidence:** high

**Original:** Plant Height

**中文:** 植物高度

<a id="S268"></a>
**Source:** p.14 S268  
**Type:** body  
**Confidence:** high

**Original:** Plant

**中文:** 植物

<a id="S269"></a>
**Source:** p.14 S269  
**Type:** body  
**Confidence:** medium

**Original:** 31:4±13.3

**中文:** 31:4±13.3 31:4±13.3

<a id="S270"></a>
**Source:** p.14 S270  
**Type:** body  
**Confidence:** high

**Original:** 11.5-59.3

**中文:** 11.5-59.3

<a id="S271"></a>
**Source:** p.14 S271  
**Type:** body  
**Confidence:** high

**Original:** 42.5

**中文:** 42.5 42.5

<a id="S272"></a>
**Source:** p.14 S272  
**Type:** body  
**Confidence:** high

**Original:** Unit

**中文:** 单位

<a id="S273"></a>
**Source:** p.14 S273  
**Type:** body  
**Confidence:** high

**Original:** LAI

**中文:** 没有什么.

<a id="S274"></a>
**Source:** p.14 S274  
**Type:** body  
**Confidence:** high

**Original:** Plant

**中文:** 植物

<a id="S275"></a>
**Source:** p.14 S275  
**Type:** body  
**Confidence:** medium

**Original:** 1:45 ± 0:25

**中文:** ± 0:25 1:45

<a id="S276"></a>
**Source:** p.14 S276  
**Type:** body  
**Confidence:** high

**Original:** 0.91-2.04

**中文:** 0.91-2.04

<a id="S277"></a>
**Source:** p.14 S277  
**Type:** body  
**Confidence:** high

**Original:** 17.1

**中文:** 17.1 17.1

<a id="S278"></a>
**Source:** p.14 S278  
**Type:** body  
**Confidence:** medium

**Original:** m =m2

**中文:** m =m2

<a id="S279"></a>
**Source:** p.14 S279  
**Type:** body  
**Confidence:** high

**Original:** Avg. Leaf Area

**中文:** .叶片区

<a id="S280"></a>
**Source:** p.14 S280  
**Type:** body  
**Confidence:** high

**Original:** Leaf

**中文:** 叶

<a id="S281"></a>
**Source:** p.14 S281  
**Type:** body  
**Confidence:** medium

**Original:** 74.9 ± 22:8

**中文:** 74.9 ± 22:8

<a id="S282"></a>
**Source:** p.14 S282  
**Type:** body  
**Confidence:** high

**Original:** 39-122

**中文:** 39-122 39-122

<a id="S283"></a>
**Source:** p.14 S283  
**Type:** body  
**Confidence:** high

**Original:** 30.5

**中文:** 30.5 30.5

<a id="S284"></a>
**Source:** p.14 S284  
**Type:** body  
**Confidence:** high

**Original:** cm2

**中文:** 2cm2的

<a id="S285"></a>
**Source:** p.14 S285  
**Type:** body  
**Confidence:** high

**Original:** Leaf Count

**中文:** 叶子伯爵

<a id="S286"></a>
**Source:** p.14 S286  
**Type:** body  
**Confidence:** high

**Original:** Plant

**中文:** 植物

<a id="S287"></a>
**Source:** p.14 S287  
**Type:** body  
**Confidence:** medium

**Original:** 6:9 ± 2:1

**中文:** 6:9 ± 2:1

<a id="S288"></a>
**Source:** p.14 S288  
**Type:** body  
**Confidence:** high

**Original:** 4-11

**中文:** 4-11 4-11

<a id="S289"></a>
**Source:** p.14 S289  
**Type:** body  
**Confidence:** high

**Original:** 30.8

**中文:** 30.8 30.8

<a id="S290"></a>
**Source:** p.14 S290  
**Type:** body  
**Confidence:** high

**Original:** count

**中文:** 

<a id="S291"></a>
**Source:** p.14 S291  
**Type:** body  
**Confidence:** high

**Original:** Node Count

**中文:** 计算 Node

<a id="S292"></a>
**Source:** p.14 S292  
**Type:** body  
**Confidence:** high

**Original:** Stem

**中文:** 语音

<a id="S293"></a>
**Source:** p.14 S293  
**Type:** body  
**Confidence:** medium

**Original:** 6:9 ± 3:2

**中文:** 6:9 ± 3:2

<a id="S294"></a>
**Source:** p.14 S294  
**Type:** body  
**Confidence:** high

**Original:** 3-16

**中文:** 3-16 3-16

<a id="S295"></a>
**Source:** p.14 S295  
**Type:** body  
**Confidence:** high

**Original:** 47.0

**中文:** 47.0

<a id="S296"></a>
**Source:** p.14 S296  
**Type:** body  
**Confidence:** high

**Original:** count

**中文:** 

<a id="S297"></a>
**Source:** p.14 S297  
**Type:** body  
**Confidence:** high

**Original:** Avg. Internode Length

**中文:** . 国际极长度

<a id="S298"></a>
**Source:** p.14 S298  
**Type:** body  
**Confidence:** high

**Original:** Stem

**中文:** 语音

<a id="S299"></a>
**Source:** p.14 S299  
**Type:** body  
**Confidence:** medium

**Original:** 3:7±0.9

**中文:** 时间为 3:7±0.9

<a id="S300"></a>
**Source:** p.14 S300  
**Type:** body  
**Confidence:** high

**Original:** 1.9-6.1

**中文:** 1.9-6.1

<a id="S301"></a>
**Source:** p.14 S301  
**Type:** body  
**Confidence:** high

**Original:** 23.6

**中文:** 23.6 23.6


## Page 15

<a id="S302"></a>
**Source:** p.15 S302  
**Type:** body  
**Confidence:** high

**Original:** 10.3389/fpls.2026.1783465

**中文:** 10.3389/fpls.2026.1783465

<a id="S303"></a>
**Source:** p.15 S303  
**Type:** body  
**Confidence:** high

**Original:** Pearson correlation matrices of organand plant-level traits at the vegetative stage, with significance levels indicated on the right panel.

**中文:** 植物级素质在植被阶段的皮尔森相关矩阵,右侧面板上显示了显著性水平.

<a id="S304"></a>
**Source:** p.15 S304  
**Type:** body  
**Confidence:** high

**Original:** in this study, whereas leaf area and LAI should be read as bounded pipeline-derived descriptors whose values have not been verified against independent physical measurements. Within the uniformcontainer muskmelon setting studied here, the height residual behaves largely as a near-constant positive offset and is therefore absorbed in relative comparisons. This interpretation is scopeconditional, because changes in container configuration or basal occlusion would make the offset non-constant and require explicit pot-boundary detection or comparable container-aware correction for absolute comparisons. Independent physical validation of leafarea-related traits remains an important future priority and would require destructive or otherwise independent reference measurements, as in Usenko et al. (2025). Beyond the structured greenhouse regime studied here, field deployment would require handling uncontrolled illumination, the absence of rectilinear structural cues currently used for scene alignment, and denser plant interactions. Longitudinal use would additionally require reliable temporal registration across time points under canopy change. Broader cross-crop use would likewise require further study of task-specific filtering refinement, as suggested by the distinct residual noise patterns in Figures 11c, d.

**中文:** 在本研究中,叶片面积和LAI应被读为有界限的管道衍生描述符,其值尚未与独立的物理测量进行验证.在这里研究的均容器果设置中,高度残余主要作为一个近恒的正面抵消,因此被相对比较中吸收.这种解释是范围条件的,因为容器配置或基层封闭的变化将使抵消变化不变,并且需要明确的边界检测或可比较的容器意识对比较进行更正确的对比.独立的物理优先权验证的叶片面积相关特征仍然是一个重要的未来,并且需要破坏性或其他独立的参考测量,如Usenko等 (2025). 除了研究的结构化温室制度之外,现场部署将需要处理不受控制的照明,目前用于场景对齐的直线结构线索的缺失和密集的植物相互作用.长度使用还需要在天窗变化的时间点中可靠的时间登记.更广泛的交叉作物使用也需要进一步研究任务特定的过精炼,正如图11c,d中的明显残留噪音模式所示.

<a id="S305"></a>
### 4 Conclusion
**Source:** p.15 S305  
**Type:** section  
**Confidence:** high

**Original:** 4 Conclusion

**中文:** 【标题暂译】4 Conclusion

<a id="S306"></a>
**Source:** p.15 S306  
**Type:** body  
**Confidence:** high

**Original:** This study addressed the challenge of deriving plant-level phenotypic information directly from dense 3D reconstructions of multi-plant greenhouse scenes, a challenge not directly addressed by single-plant acquisition workflows. We introduced an end-toend pipeline built on 3D Gaussian Splatting (3DGS) and contributed three main advances. First, we established a reconstruction-toanalysis workflow that preserves per-plant geometric fidelity in structured multi-plant greenhouse environments under moderate inter-plant proximity and view-dependent occlusion. Second, we proposed LCR-GS, a post-reconstruction extraction framework that decouples semantics from 3DGS optimization and requires only minimal operator seeding; its Lift-Cluster-Refine stages reliably convert dense scene-level 3DGS models into clean, analysis-ready plant instances. Third, we demonstrated that these extracted plants support robust organ-level segmentation and trait computation, with plant height and leaf count validated against manual measurements and additional morphological traits reported as pipelinederived descriptors for within-cohort comparison. Quantitative evaluation on early-vegetative greenhouse-grown muskmelon plants confirmed high extraction precision (0.933),

**中文:** 这项研究解决了直接从多植物温室场景的密集3D重建中获得植物级现象信息的挑战,这项挑战并非直接解决的单植物收购工作流程.我们引入了一个基于3D高斯人分光 (3DGS) 的终端管道,并贡献了三个主要进展.首先,我们建立了一个重建分析工作流程,以保持结构化多植物温室环境中的每个植物的几何忠诚度,在适度的植物间接近和视觉依赖的位下.第二,我们提出了LCR-GS,一个后重建提取框架,从3DGS优化中脱离语义并只需要最小的操作员种植;其 Lift-Cluster-Refine阶段可靠地将密度准备的3DGS植物级模型转化为清洁的场景分析. 第三,我们证明这些提取的植物支持强大的器官水平分类和特征计算,植物高度和叶子数量与手动测量进行验证,并报告了作为管道衍生描述器进行内部组合比较的额外形态特征.早期植物性温室培养的桃植物的量化评估证实了高提取精度 (0.933),

<a id="S307"></a>
**Source:** p.15 S307  
**Type:** body  
**Confidence:** high

**Original:** Qualitative examples of LCR-GS extraction beyond the validated setting. Left: representative 3DGS scene renderings; right: corresponding extracted plant instances. (a) Mid-stage greenhouse muskmelon. (b) Late-stage greenhouse muskmelon, with arrows indicating representative residual or locally incomplete-separation regions. (c) Sweet olive and (d) peanut, with arrows indicating representative non-target residual regions.

**中文:** 证实设置之外的LCR-GS提取的质量例子.左:代表性3DGS场景染;右:相应的提取植物实例. (一) 温室桃中期. (二) 晚期温室桃,箭头表示代表性残留或局部不完整的分离区域. (三) 甜蜜橄和 (三) 桃,箭头表示代表性非目标残留区域.


## Page 16

<a id="S308"></a>
**Source:** p.16 S308  
**Type:** body  
**Confidence:** high

**Original:** 10.3389/fpls.2026.1783465

**中文:** 10.3389/fpls.2026.1783465

<a id="S309"></a>
**Source:** p.16 S309  
**Type:** body  
**Confidence:** medium

**Original:** strong segmentation performance (AP50 = 0.924), and close agreement between 3DGS-derived and manually measured plant height (R² = 0.98) and leaf count (90% of estimates within ±1 leaf). These results show that the proposed pipeline achieves accuracy suitable for population-level comparison of the validated traits within the demonstrated cohort while reducing data scale by more than 99% to support efficient downstream learning. The approach operates within several important constraints. Validation is limited to early-vegetative muskmelon under structured greenhouse conditions; extension to later growth stages, denser canopy architectures, or other crop species requires further study. Leaf area, LAI, and internode-related traits are reported as pipeline-derived descriptors and have not been verified against independent physical measurements. Additional practical limitations include the need for limited manual seeding to initialize multiview cues and sensitivity to the quality of the underlying 3D reconstruction under challenging lighting or occlusion conditions. Future work will focus on automated cue generation, improved handling of denser canopy conditions, and broader cross-crop applicability with task-specific filtering adjustment. Despite these constraints, the retained per-plant 3DGS representations open new opportunities for longitudinal monitoring and growth modeling within controlled settings. Because 3DGS encodes explicit geometry and photometric detail, the retained representations provide a useful basis for further downstream analysis. Overall, this work provides a practical path from scene-level 3D reconstruction to organ-level trait characterization for structured greenhouse phenotyping.

**中文:** 分割性能 (AP50 = 0.924),以及3DGS衍生和手动测量植物高度 (R2 = 0.98) 和叶子数量 (90%的估计在±1叶内).这些结果表明,拟议的管道实现了适合人口水平对照验证的特征在被证明的队列中的准确性,同时将数据规模降低了99%以上以支持有效下游学习.这种方法运行在几个重要的限制范围内.验证仅限于结构化温室条件下的早期植物性;延长到后期生长阶段,更密集的架构或其他作物物种需要进一步研究. 叶片面积,LAI和内部码相关的特征被报告为管道衍生的描述符,并未被独立的物理测量验证.另外的实际限制包括需要有限的手动种植,以启动多视图线索和对在具有挑战性的照明或封闭条件下底层3D重建的质量敏感性.未来的工作将集中在自动化标注生成,更密集的天窗条件的更好处理以及更广泛的跨作物可用性,具体的任务过调整.尽管存在这些限制,但保留的每种植的3DGS表示开辟了对控制环境内的纵向监测和增长建模的新机会. 由于3DGS编码了明确的几何学和光学细节,因此保留的表示为进一步的下游分析提供了有用的基础.总的来说,这项工作提供了从场景层面的3D重建到器官层面的特征特征化结构化温室表型的实际路径.

<a id="S310"></a>
### Funding
**Source:** p.16 S310  
**Type:** section  
**Confidence:** high

**Original:** Funding

**中文:** 基金资助

<a id="S311"></a>
**Source:** p.16 S311  
**Type:** body  
**Confidence:** high

**Original:** The author(s) declared that financial support was not received for this work and/or its publication.

**中文:** 作者表示,没有获得这本书和/或其出版的资金支持.

<a id="S312"></a>
### Acknowledgments
**Source:** p.16 S312  
**Type:** section  
**Confidence:** high

**Original:** Acknowledgments

**中文:** 【标题暂译】Acknowledgments

<a id="S313"></a>
**Source:** p.16 S313  
**Type:** body  
**Confidence:** high

**Original:** The authors would like to thank Mr. Lien-Chieh Cheng and the staff members of the National Taiwan University Experimental Farm for their assistance in this research.

**中文:** 作者们想感谢Lien-Chieh Cheng先生和台湾国立大学实验农场的员工,他们为这项研究提供了帮助.

<a id="S314"></a>
### Conflict of interest
**Source:** p.16 S314  
**Type:** section  
**Confidence:** high

**Original:** Conflict of interest

**中文:** 利益冲突

<a id="S315"></a>
**Source:** p.16 S315  
**Type:** body  
**Confidence:** high

**Original:** The author(s) declared that this work was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest.

**中文:** 作者 (s) 宣称,这项工作是在没有商业或金融关系的情况下进行的,这可以被视为潜在的利益冲突.

<a id="S316"></a>
**Source:** p.16 S316  
**Type:** body  
**Confidence:** high

**Original:** Generative AI statement

**中文:** 创建人工智能声明

<a id="S317"></a>
**Source:** p.16 S317  
**Type:** body  
**Confidence:** high

**Original:** The datasets presented in this study can be found in online repositories. The names of the repository/repositories and accession number(s) can be found below: https://github.com/bblabNTU/ 3dgs-muskmelon-phenotyping-dataset.

**中文:** 本研究中提出的数据集可以在网上存储库中找到.存储库/存储库和接入号码的名称(s) 可以在下面找到: https://github.com/bblabNTU/ 3DGS-muskmelon-phenotyping-dataset.

<a id="S318"></a>
**Source:** p.16 S318  
**Type:** body  
**Confidence:** high

**Original:** The author(s) declared that generative AI was used in the creation of this manuscript. During the preparation of this manuscript, ChatGPT (OpenAI) was used solely to assist with language editing for clarity and grammar. The authors carefully reviewed and edited all content generated with the assistance of this tool and take full responsibility for the accuracy, integrity, and originality of the final manuscript. Any alternative text (alt text) provided alongside figures in this article has been generated by Frontiers with the support of artificial intelligence and reasonable efforts have been made to ensure accuracy, including review by the authors wherever possible. If you identify any issues, please contact us.

**中文:** 作者 (s) 宣布,创建人工智能在创建这本手稿中被使用.在编写这本手稿时,ChatGPT (OpenAI) 仅用于帮助语言编辑,以提高清晰度和语法.作者仔细审查和编辑使用该工具生成的所有内容,并承担最终手稿的准确性,完整性和原创性的全部责任.与本文中的数字一起提供的任何替代文本 (alt文本) 都由边界公司在人工智能的支持下创建,并已作出合理努力确保准确性,包括作者在可能的情况下进行审查.如果您发现任何问题,请联系我们.

<a id="S319"></a>
### Author contributions
**Source:** p.16 S319  
**Type:** section  
**Confidence:** high

**Original:** Author contributions

**中文:** 作者贡献

<a id="S320"></a>
**Source:** p.16 S320  
**Type:** body  
**Confidence:** high

**Original:** Publisher’s note

**中文:** 出版商的注意事项

<a id="S321"></a>
**Source:** p.16 S321  
**Type:** body  
**Confidence:** high

**Original:** JL: Conceptualization, Formal analysis, Investigation, Methodology, Validation, Writing – original draft, Writing – review & editing, Data curation, Software, Visualization. TL: Conceptualization, Formal analysis, Investigation, Methodology, Project administration, Resources, Supervision, Validation, Writing – original draft, Writing – review & editing.

**中文:** 简介:概念化,正式分析,调查,方法,验证,写作 原稿,写作 回顾和编辑,数据策划,软件,视觉化. TL:概念化,正式分析,调查,方法,项目管理,资源,监督,验证,写作 原稿,写作 回顾和编辑.

<a id="S322"></a>
**Source:** p.16 S322  
**Type:** body  
**Confidence:** high

**Original:** All claims expressed in this article are solely those of the authors and do not necessarily represent those of their affiliated organizations, or those of the publisher, the editors and the reviewers. Any product that may be evaluated in this article, or claim that may be made by its manufacturer, is not guaranteed or endorsed by the publisher.

**中文:** 本文所述的所有声明都是作者提出的,不一定代表其附属组织的声明,或者出版商,编辑和评论员的声明.任何产品或产品的评价或声明可能由其制造商做出的声明,都没有出版商的保证或认可.

<a id="S323"></a>
### Data availability statement
**Source:** p.16 S323  
**Type:** section  
**Confidence:** high

**Original:** Data availability statement

**中文:** 数据可用性声明

<a id="S324"></a>
### References
**Source:** p.16 S324  
**Type:** section  
**Confidence:** high

**Original:** References

**中文:** 参考文献

<a id="S325"></a>
**Source:** p.16 S325  
**Type:** reference  
**Confidence:** high

**Original:** Akhtar, M. S., Zafar, Z., Nawaz, R., and Fraz, M. M. (2024). Unlocking plant secrets: a systematic review of 3D imaging in plant phenotyping techniques. Comput. Electron. Agric. 222, 109033. doi: 10.1016/j.compag.2024.109033

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S326"></a>
**Source:** p.16 S326  
**Type:** reference  
**Confidence:** high

**Original:** Araus, J. L., Kefauver, S. C., Zaman-Allah, M., Olsen, M. S., and Cairns, J. E. (2018). Translating high-throughput phenotyping into genetic gain. Trends Plant Sci. 23, 451– 466. doi: 10.1016/j.tplants.2018.02.001

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S327"></a>
**Source:** p.16 S327  
**Type:** reference  
**Confidence:** high

**Original:** Ankerst, M., Breunig, M. M., Kriegel, H.-P., and Sander, J. (1999). OPTICS: ordering points to identify the clustering structure. ACM SIGMOD Rec. 28, 49–60. doi: 10.1145/ 304182.304187

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S328"></a>
**Source:** p.16 S328  
**Type:** reference  
**Confidence:** high

**Original:** Arshad, M. A., Jubery, T., Afful, J., Jignasu, A., Balu, A., Ganapathysubramanian, B., et al. (2024). Evaluating neural radiance fields for 3D plant geometry reconstruction in

**中文:** 参考文献条目保留原文，未做逐条翻译。


## Page 17

<a id="S329"></a>
**Source:** p.17 S329  
**Type:** reference  
**Confidence:** high

**Original:** 10.3389/fpls.2026.1783465

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S330"></a>
**Source:** p.17 S330  
**Type:** reference  
**Confidence:** high

**Original:** Boukhana, M., Ravaglia, J., Hé troy-Wheeler, F., and De Solan, B. (2022). Geometric models for plant leaf area estimation from 3D point clouds: a comparative study. Graphics Visual Comput. 7, 200057. doi: 10.1016/j.gvc.2022.200057

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S331"></a>
**Source:** p.17 S331  
**Type:** reference  
**Confidence:** high

**Original:** Li, L., Zhang, Q., and Huang, D. (2014). A review of imaging techniques for plant phenotyping. Sensors 14, 20078–20111. doi: 10.3390/s141120078 Li, Y., Ma, Q., Yang, R., Li, H., Ma, M., Ren, B., et al. (2025b). “SceneSplat: Gaussian splatting-based scene understanding with vision-language pretraining”, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), (Piscataway: IEEE), 4961–4972.

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S332"></a>
**Source:** p.17 S332  
**Type:** reference  
**Confidence:** high

**Original:** Cao, J., Tagliasacchi, A., Olson, M., Zhang, H., and Su, Z. (2010). “Point cloud skeletons via Laplacian based contraction”, in: 2010 Shape Modeling International Conference, (Piscataway: IEEE), 187–197. doi: 10.1109/SMI.2010.25 Cen, J., Fang, J., Yang, C., Xie, L., Zhang, X., Shen, W., et al. (2025). Segment any 3D gaussians. Proc. AAAI Conf. Artif. Intell. 39, 1971–1979. doi: 10.1609/aaai.v39i2.32193

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S333"></a>
**Source:** p.17 S333  
**Type:** reference  
**Confidence:** high

**Original:** Li, Y., Wen, W., Miao, T., Wu, S., Yu, Z., Wang, X., et al. (2022b). Automatic organlevel point cloud segmentation of maize shoots by integrating high-throughput data acquisition and deep learning. Comput. Electron. Agric. 193, 106702. doi: 10.1016/ j.compag.2022.106702

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S334"></a>
**Source:** p.17 S334  
**Type:** reference  
**Confidence:** high

**Original:** Cheng, T., Song, L., Ge, Y., Liu, W., Wang, X., and Shan, Y. (2024). “YOLO-World: real-time open-vocabulary object detection”, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), (Piscataway: IEEE), 16901–16911. doi: 10.1109/CVPR52733.2024.01599

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S335"></a>
**Source:** p.17 S335  
**Type:** reference  
**Confidence:** high

**Original:** Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., and Ng, R. (2021). NeRF: representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 99–106. doi: 10.1145/3503250

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S336"></a>
**Source:** p.17 S336  
**Type:** reference  
**Confidence:** high

**Original:** Choi, H.-B., Park, J.-K., Park, S. H., and Lee, T. S. (2024). NeRF-based 3D reconstruction pipeline for acquisition and analysis of tomato crop morphology. Front. Plant Sci. 15. doi: 10.3389/fpls.2024.1439086

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S337"></a>
**Source:** p.17 S337  
**Type:** reference  
**Confidence:** high

**Original:** Mirande, K., Godin, C., Tisserand, M., Charlaix, J., Besnard, F., and Hé troy-Wheeler, F. (2022). A graph-based approach for simultaneous semantic and instance segmentation of plant 3D point clouds. Front. Plant Sci. 13. doi: 10.3389/fpls.2022.1012669

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S338"></a>
**Source:** p.17 S338  
**Type:** reference  
**Confidence:** high

**Original:** Coughlan, J. M., and Yuille, A. L. (2000). “The Manhattan world assumption: regularities in scene statistics which enable Bayesian inference”, in: Advances in Neural Information Processing Systems 13, (Cambridge: The MIT Press), 845–851.

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S339"></a>
**Source:** p.17 S339  
**Type:** reference  
**Confidence:** high

**Original:** Pape, J.-M., and Klukas, C. (2014). “3-D histogram-based segmentation and leaf detection for rosette plants”, in: Computer Vision – ECCV 2014 Workshops, (Cham: Springer), 61–74. doi: 10.1007/978-3-319-16220-1_5

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S340"></a>
**Source:** p.17 S340  
**Type:** reference  
**Confidence:** high

**Original:** Du, R., Ma, Z., Xie, P., He, Y., and Cen, H. (2023). PST: plant segmentation transformer for 3D point clouds of rapeseed plants at the podding stage. ISPRS J. Photogramm. Remote Sens. 195, 380–392. doi: 10.1016/j.isprsjprs.2022.11.022

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S341"></a>
**Source:** p.17 S341  
**Type:** reference  
**Confidence:** high

**Original:** Qi, C. R., Su, H., Mo, K., and Guibas, L. J. (2017a). “PointNet: deep learning on point sets for 3D classification and segmentation”, in: 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), (Piscataway: IEEE), 77–85. doi: 10.1109/CVPR.2017.16

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S342"></a>
**Source:** p.17 S342  
**Type:** reference  
**Confidence:** high

**Original:** Fang, G., and Wang, B. (2024). “Mini-splatting: representing scenes with a constrained number of Gaussians”, in: Computer Vision – ECCV 2024, (Cham: Springer), 165–181. doi: 10.1007/978-3-031-72980-5_10

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S343"></a>
**Source:** p.17 S343  
**Type:** reference  
**Confidence:** high

**Original:** Qi, C. R., Yi, L., Su, H., and Guibas, L. J. (2017b). “PointNet++: deep hierarchical feature learning on point sets in a metric space”, in: Advances in Neural Information Processing Systems 30, (Red Hook: Curran Associates Inc.), 5099–5108.

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S344"></a>
**Source:** p.17 S344  
**Type:** reference  
**Confidence:** high

**Original:** Feng, L., Chen, S., Zhang, C., Zhang, Y., and He, Y. (2021). A comprehensive review on recent applications of unmanned aerial vehicle remote sensing with various sensors for high-throughput plant phenotyping. Comput. Electron. Agric. 182, 106033. doi: 10.1016/j.compag.2021.106033

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S345"></a>
**Source:** p.17 S345  
**Type:** reference  
**Confidence:** high

**Original:** Rui, Z., Zhang, Z., Zhang, M., Azizi, A., Igathinathane, C., Cen, H., et al. (2024). Highthroughput proximal ground crop phenotyping systems – a comprehensive review. Comput. Electron. Agric. 224, 109108. doi: 10.1016/j.compag.2024.109108 Schönberger, J. L., and Frahm, J.-M. (2016). “Structure-from-motion revisited”, in: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), (Piscataway: IEEE), 4104–4113. doi: 10.1109/CVPR.2016.445

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S346"></a>
**Source:** p.17 S346  
**Type:** reference  
**Confidence:** high

**Original:** Ghahremani, M., Williams, K., Corke, F. M., Tiddeman, B., Liu, Y., and Doonan, J. H. (2021). Deep segmentation of point clouds of wheat. Front. Plant Sci. 12. doi: 10.3389/fpls.2021.608732 Harandi, N., Vandenberghe, B., Vankerschaver, J., Depuydt, S., and Van Messem, A. (2023). How to make sense of 3D representations for plant phenotyping: a compendium of processing and analysis techniques. Plant Methods 19, 60. doi: 10.1186/s13007-023-01031-z

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S347"></a>
**Source:** p.17 S347  
**Type:** reference  
**Confidence:** high

**Original:** Shen, P., Jing, X., Deng, W., Jia, H., and Wu, T. (2025). PlantGaussian: exploring 3D Gaussian splatting for cross-time, cross-scene, and realistic 3D plant visualization and beyond. Crop J. 13, 607–618. doi: 10.1016/j.cj.2025.01.011

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S348"></a>
**Source:** p.17 S348  
**Type:** reference  
**Confidence:** high

**Original:** Herná ndez-Herná ndez, J. L., Garcı́a-Mateos, G., Gonzá lez-Esquiva, J. M., EscarabajalHenarejos, D., Ruiz-Canales, A., and Molina-Martı́nez, J. M. (2016). Optimal color space selection method for plant/soil segmentation in agriculture. Comput. Electron. Agric. 122, 124–132. doi: 10.1016/j.compag.2016.01.020

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S349"></a>
**Source:** p.17 S349  
**Type:** reference  
**Confidence:** high

**Original:** Shi, W., van de Zedde, R., Jiang, H., and Kootstra, G. (2019). Plant-part segmentation using deep learning and multi-view vision. Biosyst. Eng. 187, 81–95. doi: 10.1016/ j.biosystemseng.2019.08.014

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S350"></a>
**Source:** p.17 S350  
**Type:** reference  
**Confidence:** high

**Original:** Jiang, L., Sun, J., Chee, P. W., Li, C., and Fu, L. (2025). Cotton3DGaussians: multiview 3D Gaussian splatting for boll mapping and plant architecture analysis. Comput. Electron. Agric. 234, 110293. doi: 10.1016/j.compag.2025.110293

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S351"></a>
**Source:** p.17 S351  
**Type:** reference  
**Confidence:** high

**Original:** Stuart, L. A. G., Morton, A., Stavness, I., and Pound, M. P. (2025). “3DGS-to-PC: 3D Gaussian splatting to dense point clouds”, in: Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), (Piscataway: IEEE), 3730–3739.

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S352"></a>
**Source:** p.17 S352  
**Type:** reference  
**Confidence:** high

**Original:** Jiang, L., Zhao, H., Shi, S., Liu, S., Fu, C.-W., and Jia, J. (2020). “PointGroup: dual-set point grouping for 3D instance segmentation”, in: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), (Piscataway: IEEE), 4866–4875. doi: 10.1109/CVPR42600.2020.00492

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S353"></a>
**Source:** p.17 S353  
**Type:** reference  
**Confidence:** high

**Original:** Usenko, D., Helman, D., and Giladi, C. (2025). Using 3D reconstruction from image motion to predict total leaf area in dwarf tomato plants. Comput. Electron. Agric. 237, 110627. doi: 10.1016/j.compag.2025.110627

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S354"></a>
**Source:** p.17 S354  
**Type:** reference  
**Confidence:** high

**Original:** Kerbl, B., Kopanas, G., Leimkühler, T., and Drettakis, G. (2023). 3D Gaussian splatting for realtime radiance field rendering. ACM Trans. Graphics 42, 139:1–139:14. doi: 10.1145/3592433

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S355"></a>
**Source:** p.17 S355  
**Type:** reference  
**Confidence:** high

**Original:** Wang, R.-F., Qu, H.-R., and Su, W.-H. (2025). From sensors to insights: technological trends in image-based high-throughput plant phenotyping. Smart Agric. Technol. 12, 101257. doi: 10.1016/j.atech.2025.101257

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S356"></a>
**Source:** p.17 S356  
**Type:** reference  
**Confidence:** high

**Original:** Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., et al. (2023). “Segment anything”, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), (Piscataway: IEEE), 3992–4003. doi: 10.1109/ICCV51070.2023.00371

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S357"></a>
**Source:** p.17 S357  
**Type:** reference  
**Confidence:** high

**Original:** Wang, Y., Sun, Y., Liu, Z., Sarma, S. E., Bronstein, M. M., and Solomon, J. M. (2019). Dynamic graph CNN for learning on point clouds. ACM Trans. Graphics 38, 146:1–146:12. doi: 10.1145/3326362

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S358"></a>
**Source:** p.17 S358  
**Type:** reference  
**Confidence:** high

**Original:** Li, D., Shi, G., Li, J., Chen, Y., Zhang, S., Xiang, S., et al. (2022a). PlantNet: a dualfunction point cloud segmentation network for multiple plant species. ISPRS J. Photogramm. Remote Sens. 184, 243–263. doi: 10.1016/j.isprsjprs.2022.01.007

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S359"></a>
**Source:** p.17 S359  
**Type:** reference  
**Confidence:** high

**Original:** Wu, X., Jiang, L., Wang, P.-S., Liu, Z., Liu, X., Qiao, Y., et al. (2024). “Point transformer V3: simpler faster stronger”, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), (Piscataway: IEEE), 4840–4851. doi: 10.1109/ CVPR52733.2024.00463

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S360"></a>
**Source:** p.17 S360  
**Type:** reference  
**Confidence:** high

**Original:** Li, J., Qi, X., Nabaei, S. H., Liu, M., Chen, D., Zhang, X., et al. (2025a). A survey on 3D reconstruction techniques in plant phenotyping: from classical methods to neural radiance fields (NeRF), 3D Gaussian splatting (3DGS), and beyond. Plant Phenomics. 26, 100137. doi: 10.1016/j.plaphe.2025.100137

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S361"></a>
**Source:** p.17 S361  
**Type:** reference  
**Confidence:** high

**Original:** Ye, M., Danelljan, M., Yu, F., and Ke, L. (2024). “Gaussian grouping: segment and edit anything in 3D scenes”, in: Computer Vision – ECCV 2024, (Cham: Springer), 162–179. doi: 10.1007/978-3-031-73397-0_10

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S362"></a>
**Source:** p.17 S362  
**Type:** reference  
**Confidence:** high

**Original:** Li, J., Zhu, K., Zhang, Q., Chen, D., Sun, Q., and Li, Z. (2026). Object-centric 3D Gaussian splatting for strawberry plant reconstruction and phenotyping. Smart Agric. Technol. 13, 101810. doi: 10.1016/j.atech.2026.101810

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S363"></a>
**Source:** p.17 S363  
**Type:** reference  
**Confidence:** high

**Original:** Zhao, H., Jiang, L., Jia, J., Torr, P. H. S., and Koltun, V. (2021). “Point transformer”, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), (Piscataway: IEEE), 16239–16248. doi: 10.1109/ICCV48922.2021.01595

**中文:** 参考文献条目保留原文，未做逐条翻译。


## 阅读提示

- 先读摘要、方法流程图和结果图表，再回到方法细节，可更快抓住论文贡献。
- 对图像重建/分割论文，重点核对数据采集方式、3D 表示、分割或性状提取流程、评价指标和失败案例。
- 公式、表格和复杂多子图页面已经保留原文锚点；若要精校中文，优先处理 `translation_notes.md` 中标为 low/medium 的块。
