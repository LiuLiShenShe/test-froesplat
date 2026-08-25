---
title: "Plant3R: Fusing 3D feature learning with Gaussian splatting to enhance wheat plant 3D reconstruction precision"
authors: "Jiateng Ma"
journal: "Plant Phenomics, 8 (2026) 100200"
doi: 10.1016/j.plaphe.2026.100200
source_pdf: /data/fj/F2DMAS/00参考文章/1-s2.0-S2643651526000373-main-2.pdf
generated: 2026-05-26
reader_type: bilingual_source_grounded_markdown
---

# Plant3R: Fusing 3D feature learning with Gaussian splatting to enhance wheat plant 3D reconstruction precision

**作者：** Jiateng Ma

**来源：** Plant Phenomics, 8 (2026) 100200; DOI: 10.1016/j.plaphe.2026.100200

**说明：** 本文件为全文中英对照阅读稿。中文为机器初译并经过领域术语规则校正；双栏、公式、表格和复杂多子图区域的低置信点记录在 `translation_notes.md`。

## 页面/章节索引

- [1. Introduction](#s008) — p.1
- [2. Materials and methods](#s016) — p.2
- [2.1. The pipeline of Plant3R](#s017) — p.2
- [2.2. Image data acquisition](#s019) — p.2
- [2.3. Sparse point cloud and camera pose estimation via MASt3R](#s026) — p.4
- [2.4. High-fidelity rendering via 3DGS](#s034) — p.4
- [2.4.1. Sparse point cloud initialization](#s036) — p.4
- [2.5. Geometric computation based on Gaussian rendering](#s054) — p.5
- [2.4.2. Gaussian distribution optimization and visual rendering in plant](#s057) — p.5
- [3.2. Accuracy analysis of camera pose estimation and initial point cloud](#s064) — p.6
- [2.6. Implementation settings and model evaluate methods](#s067) — p.6
- [3. Results](#s069) — p.6
- [3.1. High-fidelity reconstruction efficiency](#s070) — p.6
- [3.3. Gaussian rendering results of wheat](#s123) — p.7
- [3.6. Crop phenotyping extraction and validation](#s160) — p.9
- [4. Discussion](#s165) — p.10
- [4.2. Limitations and future potential](#s205) — p.10
- [Author contributions](#s214) — p.11
- [5. Conclusion](#s216) — p.11
- [Funding](#s217) — p.11
- [Declaration of competing interest](#s220) — p.11
- [Data availability](#s227) — p.12
- [References](#s229) — p.12

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

**Original:** Research Article

**中文:** 研究文章

<a id="S002"></a>
**Source:** p.1 S002  
**Type:** body  
**Confidence:** high

**Original:** Plant3R: Fusing 3D feature learning with Gaussian splatting to enhance wheat plant 3D reconstruction precision Jiateng Ma a, Xiaolong Hu a, *, Liangsheng Shi a, b, Yufan Zhang a, Yixiang Jiang a, Hao Zhang a, Shuo Duan a a b

**中文:** 植物3R:将3D功能学习与加斯人喷融合,以提高小麦植物3D重建精度Jiateng Ma a, Xiaolong Hu a, *,Liangsheng Shi a, b, Yufan Zhang a, Yixiang Jiang a, Hao Zhang a, Shuo Duan a a b

<a id="S003"></a>
**Source:** p.1 S003  
**Type:** body  
**Confidence:** high

**Original:** State Key Laboratory of Water Resources Engineering and Management, Wuhan University, Wuhan, China School of Artificial Intelligence, Wuhan University, Wuhan, China

**中文:** 水资源工程和管理的关键实验室,武汉大学,武汉,中国人工智能学校,武汉大学,武汉,中国

<a id="S004"></a>
**Source:** p.1 S004  
**Type:** body  
**Confidence:** high

**Original:** A R T I C L E I N F O

**中文:** 现在,我们必须要做一个.

<a id="S005"></a>
**Source:** p.1 S005  
**Type:** body  
**Confidence:** high

**Original:** A B S T R A C T

**中文:** ,我认为这是一个很好的方法.

<a id="S006"></a>
**Source:** p.1 S006  
**Type:** body  
**Confidence:** high

**Original:** Keywords: Plant 3D reconstruction Wheat phenotyping MASt3R 3D Gaussian splatting

**中文:** 关键词:植物3D重建 麦子表型 MASt3R 3D 高斯人喷

<a id="S007"></a>
**Source:** p.1 S007  
**Type:** body  
**Confidence:** high

**Original:** Precise reconstruction of plant phenotypes is crucial for smart agriculture. Conventional methods struggle with low efficiency and strong dependency on high-quality data, especially for low-texture and structurally complex crops like wheat. We propose a novel 3D reconstruction framework—Plant3R—that fuses deep feature learning with 3D Gaussian Splatting (3DGS). It innovatively uses the Matching and Stereo 3D Reconstruction (MASt3R) model for sparse point cloud reconstruction and camera pose estimation via its 3D feature matching capabilities, which substantially improve image matching rates and the quality of sparse point clouds. Subsequently, 3DGS is employed for rendering and optimization, enabling end-to-end, high-fidelity, and high-robust 3D reconstruction of wheat plants. Validated on potted wheat at multiple growth stages using handheld images, our experimental results demonstrate that Plant3R performs well in feature extraction and matching, and the reconstructed point cloud provides a good geometric prior for the subsequent rendering stage. In most scenes, its key rendering metrics—Peak Signal-to-Noise Ratio (PSNR) > 34, Structural Similarity Index Measure (SSIM) of 0.94, and Learned Perceptual Image Patch Similarity (LPIPS) < 0.26—surpassed Neural Radiance Fields (NeRF) and the original 3DGS. Moreover, extracted phenotypic traits such as plant height, leaf length, and width showed high correlation with manual measurements (R2 > 0.94), confirming its utility for accurate and quantitative phenotype analysis. Overall, Plant3R not only improves the rendering quality and geometric precision of 3D modeling, but also provides a reliable tool for accurate phenotypic parameter extraction and high-throughput crop phenotyping in precision agriculture.

**中文:** 【机器初译待精修】Precise reconstruction of plant phenotypes is crucial for smart agriculture. Conventional methods struggle with low efficiency and strong dependency on high-quality data, especially for low-texture and structurally complex crops like wheat. We propose a novel 3D reconstruction framework—Plant3R—that fuses deep feature learning with 3D Gaussian Splatting (3DGS). It innovatively uses the Matching and Stereo 3D Reconstruction (MASt3R) model for sparse point cloud reconstruction and camera pose estimation via its 3D feature matching capabilities, which substantially improve image matching rates and the quality of sparse point clouds. Subsequently, 3DGS is employed for rendering and optimization, enabling end-to-end, high-fidelity, and high-robust 3D reconstruction of wheat plants. Validated on potted wheat at multiple growth stages using handheld images, our experimental results demonstrate that Plant3R performs well in feature extraction and matching, and the reconstructed point cloud provides a good geometric prior for the subsequent rendering stage. In most scenes, its key rendering metrics—Peak Signal-to-Noise Ratio (PSNR) > 34, Structural Similarity Index Measure (SSIM) of 0.94, and Learned Perceptual Image Patch Similarity (LPIPS) < 0.26—surpassed Neural Radiance Fields (NeRF) and the original 3DGS. Moreover, extracted phenotypic traits such as plant height, leaf length, and width showed high correlation with manual measurements (R2 > 0.94), confirming its utility for accurate and quantitative phenotype analysis. Overall, Plant3R not only improves the rendering quality and geometric precision of 3D modeling, but also provides a reliable tool for accurate phenotypic parameter extraction and high-throughput crop phenotyping in precision agriculture.

<a id="S008"></a>
### 1. Introduction
**Source:** p.1 S008  
**Type:** section  
**Confidence:** high

**Original:** 1. Introduction

**中文:** 1. 引言

<a id="S009"></a>
**Source:** p.1 S009  
**Type:** body  
**Confidence:** high

**Original:** Plant phenotyping serves as a critical bridge between genotypes and agronomic performance, providing the basis for understanding how plants respond to genetic and environmental factors [1]. Against the backdrop of global food security and climate change, accurate and scalable phenotyping has become essential for breeding resilient crop varieties and guiding precision agriculture [2,3]. Meanwhile, the rapid development of consumer-grade sensing technologies and artificial in­ telligence is significantly accelerating the modernization of smart agri­ culture, leading to a notable increase in agricultural production efficiency [4]. In this process, the rapid and accurate acquisition of crop phenotypic information is considered a powerful tool for dynamic monitoring in crop breeding, and a foundation for intelligent agricul­ tural management [5]. Traditional phenotyping methods typically rely

**中文:** 植物型样式构成基因型和农业性能之间的关键桥梁,为了解植物如何应对遗传和环境因素提供了基础 [1].在全球粮食安全和气候变化背景下,精确和可扩展的植物型样式构成对于育种具有弹性作物品种和指导精密农业至关重要 [2,3].同时,消费者级传感技术和人工智能的快速发展正在显著加速智能农业的现代化,导致农业生产效率显著提高 [4]. 在这个过程中,快速准确地获取作物表型信息被认为是动态监测作物育种的强大工具,也是智能农业管理的基础 [5].传统的表表型方法通常依赖于.

<a id="S010"></a>
**Source:** p.1 S010  
**Type:** body  
**Confidence:** high

**Original:** on manual measurements, which are labor-intensive, inefficient, and somewhat destructive, while also being susceptible to subjective in­ fluences [6,7]. Although 2D image-based phenotyping methods mitigate these issues to some extent, they are limited by spatiotemporal di­ mensions, making it difficult to accurately characterize the complex 3D morphological changes during crop growth [8]. In contrast, crop 3D reconstruction technology non-invasively acquires the plant's volu­ metric structural information [9]. This not only allows for a complete record of key phenotypic features such as plant architecture and organ spatial topological relationships [10]—providing a data foundation for Functional Structural Plant Models (FSPM) [11], but also enables the revelation of the dynamic response of crop phenotypic plasticity to environmental stress through 3D data analysis [12]. This capability of­ fers crucial guidance for crop variety improvement and optimizing cultivation management strategies.

**中文:** 手动测量,是劳动密集的,不高效的,有点破坏性的,同时也容易受到主观影响 [6,7].虽然基于2D图像的表型方法在一定程度上减轻了这些问题,但它们受到空间时间的限制,使得难以准确地描述作物生长过程中的复杂的3D形态变化 [8].相反,作物3D重建技术不侵入地获得了植物的气体结构信息 [9]. 这不仅可以完全记录植物建筑和器官空间拓关系等关键的现象特征, [10]为功能结构植物模型 (FSPM) [11]提供数据基础,还通过3D数据分析,通过环境压力揭示了农作物表型塑性动态反应的可能性 [12].这种能力为提高农作物品种和优化种植管理策略提供了关键的指导.

<a id="S011"></a>
**Source:** p.1 S011  
**Type:** body  
**Confidence:** high

**Original:** * Corresponding author. State Key Laboratory of Water Resources Engineering and Management, Wuhan University, Wuhan, Hubei, 430072, China. E-mail address: xlhu@whu.edu.cn (X. Hu).

**中文:** 相关作者.国家水资源工程和管理关键实验室,武汉大学,武汉,湖北,430072,中国.电子邮件地址:xlhu@whu.edu.cn (X. Hu). *

<a id="S012"></a>
**Source:** p.1 S012  
**Type:** body  
**Confidence:** high

**Original:** 2643-6515/© 2026 The Authors. Published by Elsevier B.V. on behalf of Nanjing Agricultural University. This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/).

**中文:** 2643-6515/© 2026 作者.由 Elsevier B.V. 代表南京农业大学出版.这是一个CC BY许可证下开放访问文章 (http://creativecommons.org/licenses/by/4.0/).


## Page 2

<a id="S013"></a>
**Source:** p.2 S013  
**Type:** body  
**Confidence:** high

**Original:** J. Ma et al.

**中文:** 詹姆斯·马等人

<a id="S014"></a>
**Source:** p.2 S014  
**Type:** body  
**Confidence:** high

**Original:** integrated 3DGS with the SAM model to achieve precise 3D recon­ struction and measurement of wheat spikes under field conditions [33]; furthermore, Song et al. combined 3DGS with an improved YOLOv8 model, proposing the concept of “digital cousins” to boost the model's detection performance to over 90.7% [34]. Despite these significant advancements, existing technologies still face the following challenges in meeting the demands of crop phenotyping research for fast, accurate, and low-cost 3D reconstruction: I) Traditional 3D reconstruction methods often struggle with incomplete or distorted geometry and cu­ mulative noise, especially in regions with complex or low-texture structures. II) Recent approaches like NeRF and 3DGS, which rely on SfM preprocessing, are highly sensitive to input quality, making them prone to failure when data is incomplete or captured under uncon­ strained conditions. To address these challenges, we propose Plant3R, an unconstrained highly robust reconstruction pipeline that integrates the deep feature learning of MASt3R [35] with high-fidelity rendering of 3DGS. Plant3R leverages MASt3R's robust feature matching and stereo reconstruction capabilities to provide reliable initialization, while adaptive Gaussian adjustment enables the efficient optimization of the complex anisotropic structures of wheat. This design ensures both computational feasibility and reconstruction fidelity across different growth stages. To our knowledge, this is the first application of a combined MASt3R and 3DGS model in the field of crop reconstruction. The main contributions of this study are as follows: 1) We propose a novel 3D reconstruction frame­ work, Plant3R, which, for the first time, fuses deep feature learning from the MASt3R model with the efficient differentiable rendering of 3DGS. 2) The hybrid paradigm leverages adaptive Gaussian primitive adjust­ ment to simulate the anisotropic 3D structure of wheat, achieving high-fidelity reconstruction while ensuring computational feasibility. 3) The framework enables efficient high-fidelity reconstruction from a small number of unconstrained images, significantly reducing compu­ tational cost and improving the practicality of 3DGS-based plant modeling.

**中文:** 另外,Song等人将3DGS与改进的YOLOv8模型结合起来,提出了"数字表兄弟"的概念,以提高模型的检测性能到90.7%以上[34].尽管有了这些显著的进步,现有的技术仍然面临以下挑战:满足农作物型研究的要求,快速,准确,低成本的3D重建:I) 传统的3D重建方法往往与不完整或扭曲的几何学和形噪音而斗争,特别是在复杂或低文本结构的地区. (二) 基于SfM预处理的NeRF和3DGS等近期方法对输入质量非常敏感,使它们容易失败,当数据不完整或在不紧张的情况下捕获.为了解决这些挑战,我们提出了Plant3R,一个无限制的高强度重建管道,将MAST3R的深度学习功能 [35]与3DGS的高真度染结合起来.Plant3R利用MAST3R的强大的功能匹配和立体重建能力来提供可靠的初始化,而适应性高斯式调整使得小麦的复杂的 anisotropic结构能够高效地优化.这种设计确保了计算可行性和重建真度在不同增长阶段. 据我们所知,这是结合MAST3R和3DGS模型的应用在作物重建领域的第一个应用.本研究的主要贡献是: 1) 我们提出了一个新的3D重建框架,Plant3R,该框架首次将从MAST3R模型中深度学习的功能与3DGS的高效可分化染融合在一起. 2) 混合范式利用适应性高斯的原始调整来模拟小麦的异质3D结构,实现高效重建,同时确保计算可行性. 3) 该框架可以从少数数量的无限制图像中高效高效重建,显著降低计算成本,提高3DGS植物模型的实用性.

<a id="S015"></a>
**Source:** p.2 S015  
**Type:** body  
**Confidence:** high

**Original:** In recent years, 3D reconstruction technology has been widely applied in crop phenotyping research [13]. Active 3D reconstruction techniques, such as TOF cameras [14,15], Lidar [16,17] and laser scanners [18], are highly favored for their ability to provide high-precision 3D models. For example, V� azquez-Arellano et al. used a TOF camera to reconstruct 3D models of maize and extracted plant height information, controlling the average error to within 8.7 mm [19]; Zheng et al. utilized a 3D digitizer to scan wheat plants for digital visualization, extracting phenotypic parameters such as leaf length, leaf width, and leaf inclination angle, with corresponding coefficients of determination (R2) reaching 0.93, 0.98, and 0.85 [20]; Nadeem et al. used UAV LiDAR to obtain multi-temporal point cloud data of crop canopies and validated the effect of different flight paths on extracting canopy heights of various crops [21]. However, these methods generally face challenges of high equipment cost and complex operation, often requiring multiple calibrations and repeated scanning to complete point cloud matching and stitching, which severely limits their widespread adoption in practice [22]. To overcome these limitations, many researchers have turned to passive imaging methods, relying on passive sensors like cameras to generate 3D crop models by processing images from different view­ points. Among numerous techniques, the SfM-MVS algorithm [23] has received significant attention for plant model reconstruction due to its advantages such as low hardware cost and high color fidelity. It esti­ mates camera parameters and 3D point clouds through feature matching across multiple overlapping images. However, its reconstruction process has high requirements for image quality and viewpoint coverage, in­ volves considerable computational complexity, and requires a long processing time. This is especially prone to errors and reconstruction failures when dealing with complex crop structures like overlapping wheat leaves. Thus, researchers began integrating deep learning into 3D reconstruction networks. For instance, Liu et al. proposed ReC-MVSNet, which integrates a reparameterization structure into a point cloud 3D reconstruction network to enhance the model's complex feature extraction capabilities, improving its accuracy by nearly 43.3% [24]. Similarly, He et al. combined GRNN with SfM-MVS to extract trait pa­ rameters of soybean plants, achieving a MAPE as low as 2.7% for plant height extraction [25]. Additionally, Wang et al. utilized thermal infrared images and stereo vision to acquire 3D data of potatoes with temperature information, used for analyzing the 3D distribution of the Crop Water Stress Index (CWSI) [26]. While these methods improved performance by optimizing networks, they did not fundamentally change the underlying principles of reconstruction. The proposal of Neural Radiance Fields (NeRF) [27] in 2020 marked a revolutionary breakthrough. It has achieved high-quality reconstruc­ tion through deep learning and demonstrated immense application po­ tential in botany. Hu et al. were the first to introduce NeRF into crop phenotyping analysis, achieving high-fidelity reconstruction of various crop scenes and releasing a related dataset, which laid the foundation for subsequent research [28]. Yang et al. developed PanicleNeRF to address the phenotyping of rice panicles, successfully extracting panicle inflo­ rescence traits by combining it with the SAM and YOLOv8 models, with an R2 of up to 0.8 compared to ground truth values [29]. However, the implicit representation of NeRF still poses a major challenge for quan­ titative analysis: it is difficult to directly extract and quantify phenotypic parameters from the model, complicating cross-scene analysis. The emergence of 3DGS [30] has perfectly addressed this problem. Compared to NeRF's implicit representation, 3DGS achieves a significant breakthrough in rendering efficiency through explicit Gaussian distri­ bution optimization; its training results can be saved and edited, providing a natural data foundation for the extraction and calculation of phenotypic parameters. Currently, 3DGS has been applied to crop 3D phenotyping extraction and analysis with great success [31]. For example, Shen et al. proposed a foreground-segmentation-based 3D Gaussian splatting method-PlantGaussian, to achieve high-fidelity plant reconstruction across space, time, and scenes [32]; Zhang et al.

**中文:** 近年来,3D重建技术已广泛应用于作物型研究 [13].ThoF摄像头 [14,15],Lidar [16,17]和激光扫描仪 [18]等3D重建技术,因其提供高精度3D模型的能力而受到高度青.例如,V azquez-Arellano等人使用了TOF摄像头重建玉米的3D模型和提取的植物高度信息,控制平均错误在8.7毫米内 [19];Zheng等人使用了3D数字化器扫描小麦植物进行数字化可视化,提取了叶子长度,叶子宽度和叶子倾斜角等型参数,并达到0.93,0.98,0.85的定位效率 (R2) [20];N和 al. 无人机LiDAR以获取种植顶的多时间点云数据,并验证了不同飞行路径对不同种植顶高度的提取的影响 [21].然而,这些方法通常面临高设备成本和复杂操作的挑战,通常需要多次校准和重复扫描才能完成点云匹配和接,这严重限制了它们在实践中广泛的采用 [22].为了克服这些限制,许多研究人员转向了被动成像方法,依赖像摄像头这样的被动传感器来生成3D种植视图模型,通过从不同点处理图像. 它通过通过多个重叠图像的功能匹配来评估摄像头参数和3D点云.然而,其重建过程对图像质量和视角覆盖率有很高的要求,涉及相当的计算复杂性,并需要长时间处理.这在处理重叠的小麦叶等复杂作物结构时尤其容易出现错误和重建失败.因此,研究人员开始将深度学习整合到3D重建网络中.例如,等人提出了ReC-MVSNet,该系统将重组结构整合到一个点云3D重建网络中,以提高模型的复杂功能提取能力,提高其准确度近43.3% [24].同样,他等人. 另外,王等人利用热红外图像和立体视觉获取土豆的3D数据,并获得温度信息,用于分析农作物水压力指数 (CWSI) 的3D分布.虽然这些方法通过优化网络提高了性能,但它们没有根本改变重建的基本原则.2020年推出的神经辐射场 (Neural Radiance Fields) [27]标志着革命性的突破.它通过深度学习实现了高质量的重建,并在植物学中证明了巨大应用潜力. Hu et al. 扬等人是第一个将NeRF引入作物 phenotyping分析,实现了高真度重建各种作物场景,并发布相关数据集,为后续研究奠定了基础 [28].等人开发了 PanicleNeRF来解决米穗部的 phenotyping,通过与SAM和YOLOv8模型结合的穗部流动性特征成功提取,R2高达0.8与地面真相值相比,但NeRF的隐含表示仍然构成了量子分析的一个重大挑战:很难直接从模型中提取和量化现象参数,通过跨场分析. 3DGS的出现已经完美地解决了这个问题. 与NeRF隐含的表示相比,3DGS通过明确的高斯式分布优化实现了显著的效率表现突破;其训练结果可以保存和编辑,为提取和计算现象参数提供了自然数据基础.目前,3DGS已被应用到作物3D现象表现式提取和分析方面,取得了很大的成功 [31].例如,Shen等人提出了基于前景分割的3D高斯式喷方法-PlantGaussian,以实现高效率的植物重建在空间,时间和场景中 [32];张等人.

<a id="S016"></a>
### 2. Materials and methods
**Source:** p.2 S016  
**Type:** section  
**Confidence:** high

**Original:** 2. Materials and methods

**中文:** 2. 材料与方法

<a id="S017"></a>
### 2.1. The pipeline of Plant3R
**Source:** p.2 S017  
**Type:** section  
**Confidence:** high

**Original:** 2.1. The pipeline of Plant3R

**中文:** 【标题暂译】2.1. The pipeline of Plant3R

<a id="S018"></a>
**Source:** p.2 S018  
**Type:** body  
**Confidence:** high

**Original:** This study proposes the Plant3R algorithm, which fuses 3D feature learning with the 3D Gaussian Splatting model, selecting potted wheat as the research object to reconstruct 3D model of wheat plants across multiple growth stages. As illustrated in Fig. 1, the entire 3D recon­ struction pipeline of Plant3R primarily consists of three steps: I) wheat image acquisition and data preprocessing; II) sparse reconstruction and camera pose estimation; III) high-fidelity plant rendering and highquality geometric model extraction. In the preprocessing stage, to ensure the algorithm's universality for datasets with inconsistent view­ points or missing metadata, this study innovatively applies initial scene construction and camera pose estimation based on 3D point map regression to the wheat reconstruction process, outputting SfM format data to provide high-quality geometric priors for subsequent rendering. Subsequently, the sparse point cloud and camera poses are input into the 3DGS model, combined with adaptive density control and a tile-based rasterizer to achieve high-fidelity 3D plant rendering. For convenient extraction and measurement of 3D phenotypic parameters, the 3D Gaussian Splatting scene undergoes additional optimization, and finally, poisson surface reconstruction is employed to generate a high-quality geometric model.

**中文:** 本研究提出了Plant3R算法,该算法将3D功能学习与3D高斯斯派特模型融合在一起,选择麦作为重建多个成长阶段的小麦植物的3D模型的研究对象.如图1,Plant3R的整个3D重建管道主要由三个步骤组成:I) 收获小麦图像和数据预处理;II) 稀有重建和摄像头姿势估计;III) 高效的植物染和高质量的几何模表型提取. 在预处理阶段,为了确保算法对数据集具有不一致的视图点或缺失的元数据的普遍性,本研究创新地将基于3D点地图回归的初步场景构建和摄像头姿势估算应用到小麦重建过程中,输出了SfM格式数据,以提供高质量的几何预测.随后,稀有点云和摄像头姿势被输入到3DGS模型中,结合适应性密度控制和基于的缩机来实现高效率3D染.为了方便地提取和测量3D现象形状参数,3D Gaussian Splatting场景进行额外优化,最后,生成鱼类的表面重建被用于高质量的化模型.

<a id="F001"></a>
### Fig. 1. Plant3R 模型流程概览。
**Placed near:** p.2 S018  
**Source:** p.3 manual-layout  
**Crop confidence:** high

![Fig. 1](assets/fig1.png)

**Original caption:** Fig. 1. Overview of the Plant3R model's pipeline.

**中文图注:** 图 1. Plant3R 模型流程概览。

**Reading note:** 重点查看该图如何支撑相邻正文中的流程、比较、消融或性状提取结果。

<a id="S019"></a>
### 2.2. Image data acquisition
**Source:** p.2 S019  
**Type:** section  
**Confidence:** high

**Original:** 2.2. Image data acquisition

**中文:** 【标题暂译】2.2. Image data acquisition

<a id="S020"></a>
**Source:** p.2 S020  
**Type:** body  
**Confidence:** high

**Original:** The experimental subject selected for this study was wheat, and potted plant experiments were conducted at the Irrigation and Drainage Experimental Field of Wuhan University (30.54◦ N, 114.36◦ E). Wheat plants were cultivated in pots with a diameter of 24 cm and a depth of

**中文:** 对于这项研究而言,小麦是选择的实验主体,在武汉大学灌和排水实验场 (30.54◦N, 114.36◦E) 进行了盆植物实验.小麦植物被种植在直径24厘米,深度的盆中.


## Page 3

<a id="S021"></a>
**Source:** p.3 S021  
**Type:** body  
**Confidence:** high

**Original:** J. Ma et al.

**中文:** 詹姆斯·马等人

<a id="S022"></a>
**Source:** p.3 S022  
**Type:** body  
**Confidence:** high

**Original:** overlap between adjacent views were captured around the plant (covering approximately 360◦) using an iPhone14 mobile device with a resolution of 4032 x 3024 pixels, a 1.0x (26 mm equivalent) focal length, and an aperture of f/1.5. Detailed camera parameters and acquisition settings are provided in Supplementary Table S1. To demonstrate the stability of the Plant3R model proposed in this study when processing sparse data or datasets lacking metadata, image data were collected in an unconstrained manner. The robustness and universality of the model were validated by reconstructing 3D point clouds from images of wheat potted plants collected at key growth stages.

**中文:** 通过iPhone14移动设备,以4032 x 3024像素的分辨率,1.0x (26毫米相当) 的焦距,和f/1.5的光圈,捕获了相邻的视图 (覆盖大约360◦) 在工厂周围. 详细的摄像头参数和收购设置在补充表S1中提供.为了证明本研究提出的Plant3R模型在处理稀缺数据或缺乏元数据的数据集时的稳定性,图像数据被无限制地收集.通过从重建在关键的增长阶段收集的小麦盆栽图像中3D点云来验证了该模型的强度和普遍性.

<a id="S023"></a>
**Source:** p.3 S023  
**Type:** body  
**Confidence:** high

**Original:** 24 cm, filled with 9 kg of air-dried soil. We used a standard pot culti­ vation method, with 5-10 seeds sown per pot. After seedlings reached the three-leaf stage, thinning was performed, retaining two to three seedlings with similar growth vigor per pot. Supplementary Fig. S1 shows the schematic diagram of the initial placement of potted plants in the experimental field. Ample water and nutrients were provided throughout the entire growth cycle of the wheat. Potted plants with significant morphological differences at various growth stages, including tillering, jointing, grain filling, and maturity, were used for data acquisition. At each growth stage, twelve pots of sample data were collected. For each pot, 30 RGB images with approximately 75-80%

**中文:** 补充图S1显示了最初放置植物在实验场的图案图.在整个小麦生长周期中提供了足够的水和营养素.在各种生长阶段,包括,结合,粮食填充和成熟等不同阶段的显著形态差异的植物用于数据采集.在每个阶段,收集了十二个的生长数据.每个子,约有30个RGB图像,75-80%.


## Page 4

<a id="S024"></a>
**Source:** p.4 S024  
**Type:** body  
**Confidence:** high

**Original:** J. Ma et al.

**中文:** 詹姆斯·马等人

<a id="S025"></a>
**Source:** p.4 S025  
**Type:** body  
**Confidence:** medium

**Original:** camera parameter estimation, a hierarchical optimization strategy is employed to ensure global consistency. In this progress, we initially take advantage of pixel correspondences to achieve coarse alignment, and gradient descent is used to minimize a 3D matching loss: ∑ ∑ ⃒⃒ ⃒⃒λ σ*; P* = argmin qc ⃒⃒Xcn − Xcm ⃒⃒ 1; (4) σ;P

**中文:** 在这种进展中,我们最初利用像素相应度来实现粗的对应,并使用梯度下降来最大限度地减少3D匹配损失:λ σ*;P* = argmin qc Xcn − Xcm 1; (4) σ;P σ*;

<a id="S026"></a>
### 2.3. Sparse point cloud and camera pose estimation via MASt3R
**Source:** p.4 S026  
**Type:** section  
**Confidence:** high

**Original:** 2.3. Sparse point cloud and camera pose estimation via MASt3R

**中文:** 【标题暂译】2.3. Sparse point cloud and camera pose estimation via MASt3R

<a id="S027"></a>
**Source:** p.4 S027  
**Type:** body  
**Confidence:** high

**Original:** The preprocessing stage of image data, which involves estimating sparse point clouds and camera parameters, provides the data founda­ tion for the dense reconstruction and high-fidelity rendering of the 3D Gaussian Splatting model, significantly impacting the accuracy and precision of the reconstruction outcomes. However, the sparse recon­ struction process in traditional SfM algorithms is segmented into mul­ tiple subtasks, where reconstruction errors accumulate throughout the pipeline, often resulting in reconstruction failure. To overcome these limitations, we chose to fuse the matching module with the MASt3R model with the ASMK retrieval pipeline. Unlike conventional methods that approach image matching as a 2D problem, we treat it as a 3D task, leveraging point map regression to achieve initial scene reconstruction and global camera alignment.

**中文:** 图像数据的预处理阶段,包括估计稀有点云和摄像头参数,为3D高斯派特模型的密集重建和高真度染提供了数据基础,显著影响了重建结果的准确性和精度.然而,传统的SfM算法中的稀有重建过程被分为多个子任务,重建错误在整个管道中积累,通常导致重建失败.为了克服这些局限性,我们选择将匹配模块与MASMK检索管道的MASMK模型和模块合并. 与传统方法不同,我们把图像匹配视为2D问题,我们把它视为3D任务,利用点地图回归来实现初始场景重建和全球相机配合.

<a id="S028"></a>
**Source:** p.4 S028  
**Type:** body  
**Confidence:** high

**Original:** (n;m)ϵE (i;j)ϵMn;m

**中文:** (n;m) εE (i;j)εMn;m

<a id="S029"></a>
**Source:** p.4 S029  
**Type:** body  
**Confidence:** medium

**Original:** and this is performed iteratively using the Adam optimizer [38]. We σʹ reparameterize σ as σ = min σ to ensure that the minimum value of σ is 1, which helps avoid degenerate solutions. Subsequently, the results from this coarse alignment undergo a second-stage global optimization. Local Bundle Adjustment (BA) is executed for each sub-scene to minimize a weighted 2D pixel reprojection error: ∑ ∑ [ (()) (())] qc ρ ync − πn yX mc + ρ ymc − πm yX nc; L2= (5) (n;m)ϵE cϵMn;m

**中文:** 这是在使用亚当优化器进行的,以循环执行.我们将 σ 作为 σ = min σ 重组 σ 确保 σ 的最小值为 1,这有助于避免退化解决方案.随后,这种粗的排列结果会进行第二阶段的全球优化.对于每个子场景,执行本地捆绑调整 (BA) 以最大限度减少权重的2D像素排放错误: [ (()) (()))] qc ρ ync − πn yX mc + ρ ymc − πm yX nc; L2= (5) (n;m) εE cεMn;m

<a id="S030"></a>
**Source:** p.4 S030  
**Type:** body  
**Confidence:** high

**Original:** (1) Sparse reconstruction and camera pose estimation

**中文:** (1) 缩重建和摄像头姿势估计

<a id="S031"></a>
**Source:** p.4 S031  
**Type:** body  
**Confidence:** high

**Original:** which optimizes the camera extrinsic and intrinsic parameters along with the point cloud coordinates. After completing the above calcula­ tions, and in a manner similar to traditional SfM methods, anchor points are created and each pixel is rigidly connected to its nearest anchor point to form pseudo-trajectories. This process effectively reduces the number of optimization variables and enhances the model's optimization efficiency.

**中文:** 完成以上计算后,并以类似于传统SfM方法的方式进行了结点计算,每个像素都被紧密连接到其最近的结点,形成伪轨迹. 这种过程有效地减少了优化变量数量,并提高了模型的优化效率.

<a id="S032"></a>
**Source:** p.4 S032  
**Type:** body  
**Confidence:** medium

**Original:** First, we perform image matching on the input images and complete local pairwise reconstruction. An effective and scalable MASt3R encoder integrated with Aggregated Selective Match Kernels (ASMK) [36]are employed to achieve efficient image retrieval and generate a similarity matrix Sϵ[0; 1]N×N. To obtain a small number of pairs, we select a fixed number Na = 20 of key images using farthest point sampling (FPS), and the remaining images are connected to their nearest keyframe as well as their k (k = 10) nearest neighbors, forming a visibility graph G for subsequent optimization, where edges e = (n; m) link potentially over­ lapping image pairs (In; Im).For each edge e = (n; m) ϵ E in the graph, a lightweight ViT network is utilized to conduct bidirectional feature matching and employ the union operation of f(In; Im) and f(Im; In) op­ erations to eliminate dependency on the order of input images. Based on the geometric features from the encoder, four point map types: Xn;n, Xn;m, Xm;n, Xm;m are generated through implicit neural field regression, where Xn;m ϵRH×W×3 represents a 2D-to-3D mapping [35] from image In to 3D points in the coordinate system of image Im, maintaining robust­ ness to photometric and geometric variations. Sparse correspondences are then extracted using Fast Nearest Neighbor (FastNN) search [35]. A weighted-average-based canonical point map generation mecha­ nism is designed to mitigate single-edge estimation uncertainty: For an image In, its connected edge set ℇn = {e|e ϵ E ⋀n ϵ e} is defined as all image pairs sharing scene overlap with In. Noise suppression is achieved through confidence maps. By computing each pixel position(i, j) in the image In, the weighted canonical point map is computed by aggregating estimates from all relevant edges: ∑ X nC ~ n;i;j = eϵℇ ∑ n:e:i:j n;e;i;j; X (1) eϵℇn Cn:e:i:j

**中文:** 【机器初译待精修】First, we perform image matching on the input images and complete local pairwise reconstruction. An effective and scalable MASt3R encoder integrated with Aggregated Selective Match Kernels (ASMK) [36]are employed to achieve efficient image retrieval and generate a similarity matrix Sϵ[0; 1]N×N. To obtain a small number of pairs, we select a fixed number Na = 20 of key images using farthest point SAMpling (FPS), and the remaining images are connected to their nearest keyframe as well as their k (k = 10) nearest neighbors, forming a visibility graph G for subsequent optimization, where edges e = (n; m) link potentially over­ lapping image pairs (In; Im).For each edge e = (n; m) ϵ E in the graph, a lightweight ViT network is utilized to conduct bidirectional feature matching and employ the union operation of f(In; Im) and f(Im; In) op­ erations to eliminate dependency on the order of input images. Based on the geometric features from the encoder, four point map types: Xn;n, Xn;m, Xm;n, Xm;m are generated through implicit neural field regression, where Xn;m ϵRH×W×3 represents a 2D-to-3D mapping [35] from image In to 3D points in the coordinate system of image Im, maintaining robust­ ness to photometric and geometric variations. Sparse correspondences are then extracted using Fast Nearest Neighbor (FastNN) search [35]. A weighted-average-based canonical point map generation mecha­ nism is designed to mitigate single-edge estimation uncertainty: For an image In, its connected edge set ℇn = {e|e ϵ E ⋀n ϵ e} is defined as all image pairs sharing scene overlap with In. Noise suppression is achieved through confidence maps. By computing each pixel position(i, j) in the image In, the weighted canonical point map is computed by aggregating estimates from all relevant edges: ∑ X nC ~ n;i;j = eϵℇ ∑ n:e:i:j n;e;i;j; X (1) eϵℇn Cn:e:i:j

<a id="S033"></a>
**Source:** p.4 S033  
**Type:** body  
**Confidence:** high

**Original:** (2) Data format transformation To utilize the output of the MASt3R model as the initialization object for 3DGS, it must be transformed into standard COLMAP-compatible SfM format: Extract valid 3D points from constraint point maps through merging and deduplication, then saved as points3D.txt, while storing camera intrinsic and extrinsic parameters following COLMAP specifications, where cameras.txt records focal length, principal point, and distortion coefficients; images.txt records the pose matrix and cor­ responding point observations for each image. These formatted outputs can be directly integrated into the 3DGS workflow to achieve high-fidelity 3D reconstruction of plants.

**中文:** (2) 数据格式转换 为了利用MAST3R模型的输出作为3DGS的初始化对象,必须转换为标准的COLMAP兼容 SfM格式:通过合并和排版从限制点地图中提取有效的3D点,然后保存为3D.txt,同时存储相机内在和外在参数,遵循COLMAP规格,其中相机.txt记录焦距,主点和扭曲系数;图像.txt记录每个图像的矩阵和回应点观测.这些格式输出可以直接集成到3DGS工作流中,以实现高效的3D复制植物.

<a id="S034"></a>
### 2.4. High-fidelity rendering via 3DGS
**Source:** p.4 S034  
**Type:** section  
**Confidence:** high

**Original:** 2.4. High-fidelity rendering via 3DGS

**中文:** 【标题暂译】2.4. High-fidelity rendering via 3DGS

<a id="S035"></a>
**Source:** p.4 S035  
**Type:** body  
**Confidence:** high

**Original:** To obtain high-fidelity 3D models of wheat plants, 3D Gaussian Splatting is used for reconstruction via rendering. The whole process is mainly divided into three stages: sparse point cloud initialization, Gaussian splat optimization, and visualization rendering.

**中文:** 为了获得高效的小麦植物3D模型,使用3D高斯派特式割用于通过染进行重建.整个过程主要分为三个阶段:稀点云初始化,高斯派特式割优化和可视化割.

<a id="S036"></a>
### 2.4.1. Sparse point cloud initialization
**Source:** p.4 S036  
**Type:** section  
**Confidence:** high

**Original:** 2.4.1. Sparse point cloud initialization

**中文:** 【标题暂译】2.4.1. Sparse point cloud initialization

<a id="S037"></a>
**Source:** p.4 S037  
**Type:** body  
**Confidence:** medium

**Original:** The initialization process of 3DGS is straightforward, beginning with sparse point clouds in SFM format reconstructed from multi-view im­ ages, serving as the initial positions of 3D Gaussian functions, each Gaussian point is represented by the formula: (∑− 1 ∑)

**中文:** 3DGS的初始化过程很简单,从多视觉时代重建的SFM格式稀点云开始,作为3D高证函数的初始位置,每个高证点都以公式表示: (−1)

<a id="S038"></a>
**Source:** p.4 S038  
**Type:** body  
**Confidence:** medium

**Original:** (x− μ) e− 2(x− μ); G x; μ; = (6) 3 ∑ 1 (2π)2 | |2

**中文:** (x−μ) e−2(x−μ);G x;μ; = (6) 3 1 (2π) 2

<a id="S039"></a>
**Source:** p.4 S039  
**Type:** body  
**Confidence:** medium

**Original:** where Xn;e;i;j represents the obtained estimate value of Xn;n from edge e. The canonical depth map is then extracted from the canonical point map ~n = X ~ n;:;:;3, and the Weiszfeld algorithm [37] is applied to optimize Z focal length: ⃒⃒) ∑⃒⃒⃒⃒(W ~ n;i;j;1:2 ⃒⃒ X H ⃒⃒;;j − − f f * = argmin ⃒⃒⃒⃒ i − (2) ~ n;i;j;3 ⃒⃒2 f

**中文:** ;e;i;j表示从边缘e获得的Xn;n的估值值.加нони范围地图随后从加нони范围地图中提取 ~n = X ~ n;:;:;:;3,并应用了韦斯菲尔德算法 [37]来优化Z焦距:) (W ~ n;i;j;1:2 X H;;j − − − f * = argmin i − (2) ~ n;i;j;3 2 f

<a id="S040"></a>
**Source:** p.4 S040  
**Type:** body  
**Confidence:** high

**Original:** X i;j

**中文:** 现在,我知道,我知道,我知道,我知道.

<a id="S041"></a>
**Source:** p.4 S041  
**Type:** body  
**Confidence:** medium

**Original:** where x is the position of any point in space, and μ represents the mean of the initialized Gaussian distribution, indicating the center position of ∑ each Gaussian point, 3×3 is the covariance matrix, which determines the shape and orientation of the Gaussian distribution, parameterized by scaling factors s and quaternion q. Additionally, the opacity of the Gaussian splat is controlled by the parameter α, with a range of [0,1); Spherical Harmonic (SH) coefficients are used to control the color of the Gaussian distribution and can represent complex lighting effects. All parameters are updated during iterative optimization, and are rapidly splatted onto the rendered image through opacity blending projection. To obtain high-fidelity 3D wheat model in the subsequent rendering process, the initialization stage begins from the positions of sparse

**中文:** x是空间中的任何点的位置,μ是初始化高斯分布的平均值,指的是每一个高斯分点的中位置,3×3是覆盖矩阵,它决定了高斯分布的形状和方向,通过缩小因子s和四 q进行参数化.此外,高斯积分的度由参数α控制,范围为 [0,1);用于控制高斯分布的颜色,可代表复杂的照明效应的球体和系数.所有参数都在循环优化过程中更新,并通过度混合投影快速地被射到呈现的图像上. 为了在后续染过程中获得高效度的3D小麦模型,初始化阶段从稀的位置开始.

<a id="S042"></a>
**Source:** p.4 S042  
**Type:** body  
**Confidence:** medium

**Original:** assuming the pinhole model with central principal point and square pixels. To ensure the 3D point cloud strictly adheres to the pinhole camera model and precisely corresponds to pixel coordinates, a visibility-dependent constrained point map is constructed by defining camera extrinsics Pn = [Rn =tn], intrinsics Kn, and scale factors σ n, with the inverse projection formula for 3D points: Xn;i;j =

**中文:** 为了确保3D点云严格遵守孔摄像头模型,并精确地与像素坐标相符,通过定义摄像头外观 Pn = [Rn =tn],内观 Kn 和尺度因子 σ n,构建了一个视力依赖的限制点地图,并为3D点推测公式:Xn;i;j =.

<a id="S043"></a>
**Source:** p.4 S043  
**Type:** body  
**Confidence:** high

**Original:** 1 −1 −1 P K Zn;i;j [i; j; 1]⊺: σn n n

**中文:** 1 −1 −1 P K Zn;i;j [i;j; 1]: σn n n

<a id="S044"></a>
**Source:** p.4 S044  
**Type:** body  
**Confidence:** high

**Original:** (3)

**中文:** (3) (3)

<a id="S045"></a>
**Source:** p.4 S045  
**Type:** body  
**Confidence:** high

**Original:** Based on the constructed constrained point maps and the result of

**中文:** 基于构建的限制点地图和的结果.


## Page 5

<a id="S046"></a>
**Source:** p.5 S046  
**Type:** body  
**Confidence:** high

**Original:** J. Ma et al.

**中文:** 詹姆斯·马等人

<a id="S047"></a>
**Source:** p.5 S047  
**Type:** body  
**Confidence:** medium

**Original:** Gaussian projection transformation, instantiation, and global sorting operations. By adjusting the shape and orientations of Gaussians, anisotropic variance is rendered; the final pixel color is obtained through blending according to the opacity α of the Gaussians:) ∑ ∏i− 1 (C= ci αʹi j=1 1 − αʹj; (11)

**中文:** 通过调整高士人的形状和方向,变异性变异得到了;最终的像素颜色通过混合根据高士人的度α得到了: (-i−1 (C=ci αʹi j=1 1 − αʹj; (11))

<a id="S048"></a>
**Source:** p.5 S048  
**Type:** body  
**Confidence:** medium

**Original:** points, using K-Means clustering to initialize Gaussian points' means μ based on the input sparse points. Assuming the input sparse point set P = p1; p2, p3 …, pn,where pi is a point of the cloud, and μi represents the cluster's center, it is updated iteratively by the following formula: 1 ∑ μj = ⃒⃒ ⃒⃒ p ϵC pi; j i Cj

**中文:** 假设输入稀疏点集合P = p1;p2,p3...,pn,在pi是云中的一个点,而μi代表集群的中心,它会被以下公式不断更新: 1 μj = p εC pi; j i Cj

<a id="S049"></a>
**Source:** p.5 S049  
**Type:** body  
**Confidence:** high

**Original:** (7)

**中文:** 现在,我们要做什么呢? (7)

<a id="S050"></a>
**Source:** p.5 S050  
**Type:** body  
**Confidence:** high

**Original:** iϵN

**中文:** 没有什么可做.

<a id="S051"></a>
**Source:** p.5 S051  
**Type:** body  
**Confidence:** medium

**Original:** ⃒ ⃒ where Cj represents the set of all points in the j-th cluster, ⃒Cj ⃒ represents ∑ the number of points in the j-th cluster, and pi ϵCj pi represents the sum

**中文:** ,Cj表示j集中的所有点,Cj表示j集中的点数, pi εCj表示pi的总和.

<a id="S052"></a>
**Source:** p.5 S052  
**Type:** body  
**Confidence:** high

**Original:** where ci is the color of each point and αʹi is given by evaluating a 2D Gaussian with covariance Σ multiplied with a learned per-point opacity. Finally, we need to ensure the geometric positional relationship between the foreground objects and the background objects to guarantee physi­ cally plausible depth rendering.

**中文:** 在此,ci是每个点的颜色,α′i是通过评估2D高斯式的变量Σ乘以学习的点均度乘以给出的.最后,我们需要确保前景对象和背景对象之间的几何定位关系,以确保物理可行的深度染.

<a id="S053"></a>
**Source:** p.5 S053  
**Type:** body  
**Confidence:** medium

**Original:** of position vector of all points in the cluster Cj. The covariance matrix controls the scaling and rotation of the Gaussian splat, mathematically represented as ∑ = R(q)S(s)S(s)T R(q)T; (8) 3×3

**中文:** 聚合物中所有点的位置向量Cj. 兼差矩阵控制了高斯平面的扩展和旋转,数学上表示为 = R(q) S(s) S(s) T R(q) T; (8) 3×3

<a id="S054"></a>
### 2.5. Geometric computation based on Gaussian rendering
**Source:** p.5 S054  
**Type:** section  
**Confidence:** high

**Original:** 2.5. Geometric computation based on Gaussian rendering

**中文:** 【标题暂译】2.5. Geometric computation based on Gaussian rendering

<a id="S055"></a>
**Source:** p.5 S055  
**Type:** body  
**Confidence:** high

**Original:** where R(q) represents the rotation transformation matrix q derived from quaternion, and S(s) represents the scaling transformation. For the convenience of the model's learning, the initial covariance matrix is set to be isotropic, and the axis length of the Gaussian splat is equal to the average distance of the nearest three points, which ensures that the initialization size of the Gaussian function matches the geo­ metric structure of the scene, avoiding excessively large or small initialization.

**中文:** 为了使模型学习方便,初始变量矩阵设定为同位素,而高斯的轴长度等于最近三个点的平均距离,这确保高斯函数的初始化大小匹配场景的地表结构,避免过度大的或小的初始化.

<a id="S056"></a>
**Source:** p.5 S056  
**Type:** body  
**Confidence:** high

**Original:** While 3DGS is excellent for high-fidelity rendering, the results usu­ ally need specialized renderers for visualization. Compared to that, point clouds, compatible with most 3D software, are extensively utilized in crop 3D phenotyping. Hence, this study employs the 3DGS-to-PC method [39] to quickly extract high-quality plant point clouds from the rendered 3D Gaussian scene, supporting further phenotypic analysis. The process mainly includes regularization, color rendering, point sampling, and mesh generation, enabling robust extraction of highquality point cloud representations from Gaussian scenes. First, a reg­ ularization term is introduced to process the Gaussian functions, ensuring that each covariance matrix is positive definite. Then, Gaussian filtering reduces the number of Gaussians in the scene by filtering large Gaussians and removing low-opacity Gaussians to produce accurate and plant-structure-consistent complex 3D Gaussians. In the color rendering step, considering the limitations of the traditional method that directly uses the Gaussian's own color and ignores the light change effect caused by the change of view angle, we simulate the rendering contribution of the Gaussian in the scene, calculate the color of each sampling point, use the Gaussian renderer for rendering, and perform reverse mapping ac­ cording to the pixel color to ensure that the color is consistent with the real rendering result, improving the color authenticity and view-angle consistency of the plant point cloud. During the sampling process, use probability sampling to randomly sample from the multivariate normal distribution corresponding to the Gaussians, dynamically allocate the number of sampling points according to the volume of the Gaussian points to ensure that Gaussians with larger volumes can be allocated more points, and then use the Mahalanobis distance threshold to filter out abnormal points to ensure that the point cloud can accurately represent the plant structure. Subsequently, Open3D [40] was used to screen surface Gaussian points. Specifically, a statistical outlier removal filter(remove_statistical_outlier) was applied with optimized parameters (nb_neighbors and std_ratio) to eliminate isolated noise points generated by reflection and reconstruction artifacts. This parameter configuration was empirically optimized to balance noise removal and preservation of fine leaf details. The same processing pipeline was consistently applied to all samples across different growth stages to ensure data uniformity and geometric reliability. To prepare the point cloud for analysis, background elements (e.g., surrounding environment) were removed using an interactive cropping step. In this step, a 3D axis-aligned bounding box was first defined around each plant to roughly separate the plant canopy from the background. Points within the bounding box was retained, while points outside it were discarded. Subsequently, fine-grained manual editing was performed to remove residual back­ ground points adhering to the pot rim ensuring that only plant regions were retained for subsequent quantitative analysis. For NeRF-based methods, since the geometry is represented implic­ itly as a density field, we utilized the Marching Cubes algorithm [41] to extract geometric structures. Specially, we evaluated the density field on a uniform voxel grid and extracted the isosurface with a predefined density threshold τ. The vertices of the extracted mesh were then treated

**中文:** 虽然3DGS对于高效率染非常好,但结果通常需要专门的染器来进行可视化.相比之下,与大多数3D软件兼容的点云在作物3D型化中得到广泛应用.因此,本研究采用3DGS-to-PC方法 [39]来快速从染的3D高效率的植物点云中提取高质量的植物点云,支持进一步的 feno型分析.该过程主要包括规范化,色彩染,点样本和网格生成,使高质量的云表示从高效率的高效率的高效率的高效率的云生成能够得到强大的提取.首先,一个规范化点术语被引入处理高效率的高效率的高效率,确保每个正面的覆盖矩阵是确的. 然后,高斯式过通过过大高斯式和删除低度高斯式来减少现场高斯人的数量,从而产生精确和与植物结构一致的复杂3D高斯式.在颜色染阶段,考虑到传统方法的局限性,直接使用高斯式的颜色,忽略了视角变化的光变效应,我们模拟了场景中的高斯式的染贡献,计算了每个样本点的颜色,使用高斯式染器进行染,并根据像素颜色进行反向映射,以确保颜色与实际染结果一致,提高了植物云点的颜色真实性和视角一致性. 在采样过程中,使用概率采样,从高西亚人的多变量正常分布中随机采样,根据高西亚人的积分数量动态分配采样点数量,以确保较大的积分的高西亚人可以分配更多点,然后使用马哈拉诺比斯距离门来过异常点,以确保点云能够准确地表示植物结构.随后,Open3D [40]被用于查高西亚的表面点.具体来说,用优化参数 (nb_neighbors和d_ratio) 消除反射和重建物件产生的隔离噪音点,应用了一个统计外层取除过. 这种参数配置经验上优化,以平衡噪音除去和细叶细节的保存.同一处理管道始终应用于不同生长阶段的所有样品,以确保数据均性和几何可靠性.为了准备点云进行分析,使用交互式剪除步骤去除背景元素 (例如周围环境).在这一步中,每个植物周围首先定义了一个3D轴线的边界框,以大致分离植物天窗和背景.边界框内的点保留,而其外的点被丢弃.随后,进行了精细的手动编辑,以删除粘贴到边的后面点,确保只有植物区域保留后面的量化分析. 对于基于NeRF的方法,由于几何学被暗示为密度场所,我们利用马歇斯基算法[41]提取几何结构.特别是,我们评估了密度场在一模一样的体素格格上,并提取了具有预定义密度门的异表面.然后,提取的网格的顶部被处理.

<a id="S057"></a>
### 2.4.2. Gaussian distribution optimization and visual rendering in plant
**Source:** p.5 S057  
**Type:** section  
**Confidence:** high

**Original:** 2.4.2. Gaussian distribution optimization and visual rendering in plant

**中文:** 【标题暂译】2.4.2. Gaussian distribution optimization and visual rendering in plant

<a id="S058"></a>
**Source:** p.5 S058  
**Type:** body  
**Confidence:** medium

**Original:** reconstruction The core of our approach is the optimization step, aiming to better fit the reconstructed model by adjusting the parameters of the Gaussian distribution, so that the rendered image is as consistent as possible with the input image. During the training process, each point in the space is expanded and projected onto a multi-view image for rendering. The quality of the training is judged by calculating the difference between the rendering result and the input image. We use Stochastic Gradient Descent to optimize the parameters of the Gaussian distribution, including position, covariance matrix, color and opacity. The loss function is L 1 combined with a D-SSIM term: L = (1 − λ)L 1 + λL D− SSIM;

**中文:** 我们的方法的核心是优化步骤,旨在通过调整高斯分布的参数来更好地适应重建模型,以便使转载的图像尽可能与输入图像一致.在训练过程中,空间中的每个点都会扩大并投射到复制的多视图像上.训练的质量通过计算转载结果和输入图像之间的差异来判断.我们使用斯托卡斯式渐进降低来优化高斯分布的参数,包括位置,覆盖矩阵,颜色和度.损失函数是L1结合D-SSIM术语:L = (1 − λ) 1 + λL D− SSIM;

<a id="S059"></a>
**Source:** p.5 S059  
**Type:** body  
**Confidence:** high

**Original:** (9)

**中文:** (9) (9)

<a id="S060"></a>
**Source:** p.5 S060  
**Type:** body  
**Confidence:** medium

**Original:** and λ = 0.2 in our experiments, employing L 1 loss to ensure pixel-level precision in rendering processes, and D-SSIM loss to ensure the struc­ tural similarity of the rendering results. The contribution of both is balanced through weighting to achieve the parameter optimization objective. After optimization warm-up, we conduct adaptive density control every 100 iterations, dynamically adjust the Gaussian distribution to optimize the fine-grained geometry of wheat organs, which ensures the integrity of wheat plant reconstruction. Gaussians are strategically added in high-detail regions, such as leaf structures, and reduced in areas of excessive reconstruction, with oversized Gaussians periodically removed from the spatial domain to optimize the balance between ac­ curacy and computational efficiency. In the rendering process, 3D Gaussians in the world coordinate sys­ tem are projected to 2D rasterized plane in the camera coordinate sys­ tem to enable effective interaction with camera parameters. This ∑ method defines the projected covariance matrix ʹ2×2 in camera coor­ dinate as follows: ∑ʹ ∑ (10) = JW WT JT; 2×2 where J is the Jacobian matrix of the projection transformation and W is the view translation matrix relative to the initial camera pose (R1; t1), performing the transformation from the world coordinate system to the camera coordinate system. A tile-based rasterizer approach is employed for rendering to achieve real-time rendering. Each pixel tile undergoes rasterization via 3D

**中文:** 【机器初译待精修】and λ = 0.2 in our experiments, employing L 1 loss to ensure pixel-level precision in rendering processes, and D-SSIM loss to ensure the struc­ tural similarity of the rendering results. The contribution of both is balanced through weighting to achieve the parameter optimization objective. After optimization warm-up, we conduct adaptive density control every 100 iterations, dynamically adjust the Gaussian distribution to optimize the fine-grained geometry of wheat organs, which ensures the integrity of wheat plant reconstruction. Gaussians are strategically added in high-detail regions, such as leaf structures, and reduced in areas of excessive reconstruction, with oversized Gaussians periodically removed from the spatial domain to optimize the balance between ac­ curacy and computational efficiency. In the rendering process, 3D Gaussians in the world coordinate sys­ tem are projected to 2D rasterized plane in the camera coordinate sys­ tem to enable effective interaction with camera parameters. This ∑ method defines the projected covariance matrix ʹ2×2 in camera coor­ dinate as follows: ∑ʹ ∑ (10) = JW WT JT; 2×2 where J is the Jacobian matrix of the projection transformation and W is the view translation matrix relative to the initial camera pose (R1; t1), performing the transformation from the world coordinate system to the camera coordinate system. A tile-based rasterizer approach is employed for rendering to achieve real-time rendering. Each pixel tile undergoes rasterization via 3D


## Page 6

<a id="S061"></a>
**Source:** p.6 S061  
**Type:** body  
**Confidence:** high

**Original:** J. Ma et al.

**中文:** 詹姆斯·马等人

<a id="S062"></a>
**Source:** p.6 S062  
**Type:** body  
**Confidence:** high

**Original:** as the representative point cloud for the subsequent quantitative com­ parisons against the point cloud generated by our Plant3R pipeline. To evaluate the accuracy of the model - generated results, phenotypic parameters such as the plant height and volume (to verify the biomass) of the crops are extracted for verification. The height and volume of the plants are measured based on the point - cloud results and compared with the ground - truth measurements. To accurately compare the reconstruction results with the phenotypic parameters, the scale - re­ covery ratio of the model is established, using a standard checkerboard as a reference. After segmenting the plants, measure the plant height and volume. The plant height is determined by subtracting the z-coordinate of the pot edge from the z-coordinate of the highest point of the plant point cloud. The volume is calculated using the grid method, similar to integral calculus.

**中文:** 对于我们Plant3R管道所产生的点云的后续数量比较,它是代表性的点云.为了评估模型生成结果的准确性,采用标准的测试板来确定模型的基因形状参数,例如植物高度和体积 (以验证生物质量).根据点云结果和地面真相测量,测量植物的高度和体积.为了精确地将重建结果与基因形状参数进行比较,建立了模型的规模-覆盖率,使用标准的测试板作为参考. 分割植物后,测量植物的高度和体积. 植物高度由从植物点云最高点的z坐标中减去边的z坐标,计算出了这个体积,使用网格方法计算,类似于整体计算.

<a id="S063"></a>
**Source:** p.6 S063  
**Type:** body  
**Confidence:** high

**Original:** converge more efficiently without requiring additional views or longer optimization.

**中文:** 它们可以更有效地融合,而不需要额外的视图或更长的优化.

<a id="S064"></a>
### 3.2. Accuracy analysis of camera pose estimation and initial point cloud
**Source:** p.6 S064  
**Type:** section  
**Confidence:** high

**Original:** 3.2. Accuracy analysis of camera pose estimation and initial point cloud

**中文:** 【标题暂译】3.2. Accuracy analysis of camera pose estimation and initial point cloud

<a id="S065"></a>
**Source:** p.6 S065  
**Type:** body  
**Confidence:** high

**Original:** extraction The accuracy of camera pose estimation and initial point cloud directly impact the detail integrity of subsequent 3DGS rendering and the quality of dense reconstruction. To comprehensively and objectively evaluate the performance of the Plant3R framework against the tradi­ tional SfM algorithm, we conducted a quantitative assessment across four key wheat growth stages. The evaluation focuses on two di­ mensions: global statistic performance (Table 1) and plant-specific geometric fidelity (Table 2). As presented in Table 1, the Plant3R model demonstrates a signifi­ cant advantage in feature extraction and matching capabilities across the four key stages. The mean match rate of Plant3R consistently out­ performs SfM, peaking at 48.35% during the Tillering stage – approxi­ mately 14% higher than SfM. Furthermore, the number of reconstructed sparse points generated by the Plant3R model was consistently several times greater than that of SfM. For instance, during the tillering stage, Plant3R reconstructed 48,028 points, whereas SfM produced only 10,608 points, and similar trends were maintained throughout the jointing, grain filling, and maturity stages. The mean observations per image for Plant3R are 3 to 5 times higher than those of SfM (e.g., 6995.92 vs 1604.46 at Tillering). This indicates that Plant3R establishes much richer connectivity between 2D images and 3D space. Regarding reprojection error, Plant3R has a higher mean reprojection error (~1.45px) compared to SfM (~1.12px). However, this apparent preci­ sion of SfM is achieved by aggressively filtering out feature points in low-texture regions, resulting in extremely sparse outputs. In contrast, Plant3R intentionally retains a larger number of challenging feature points, accepting a slight increase in pixel-level variance to preserve geometric completeness and ensure broader scene coverage. To further analyze the geometric basis that is crucial for downstream 3DGS reconstruction is the plant rather than the background, an addi­ tional quantitative analysis focusing specifically on the plant region was performed (Table 2). While Table 1 establishes that Plant3R produces a denser and more feature-rich sparse reconstruction, Table 2 evaluates how much of this reconstruction geometry corresponds to the actual plant canopy. The results reveal that Plant3R consistently achieved a substantially higher number of valid plant points, plant point ratio (PPR), and canopy coverage index (CCI) than the SfM algorithm across all growth stages. For example, during the tillering stage, Plant3R pro­ duced 3197 plant points, accounting for 6.65% of all reconstructed points, while SfM yielded only 43 plant points (0.41%). At the maturity stage, Plant3R achieved 9.92% PPR and a canopy coverage index of 18.4%, whereas SfM achieved only 3.88% PPR and 2.4% CCI. The CCI, which quantifies the proportion of the canopy surface covered by the reconstructed points, clearly indicates that the Plant3R model not only generates more points overall but also captures a far greater proportion of points located on the biologically meaningful plant structures. This reflects the ability of Plant3R to recover more complete and continuous canopy geometry, especially in low-texture and geometrically complex regions that are challenging for traditional feature-based SfM methods. In summary, the combined quantitative results in Tables 1 and 2 demonstrate that, during critical winter wheat growth stages, the Plant3R model significantly outperforms the traditional SfM algorithm in terms of feature extraction, matching, and the density of generated sparse point clouds. Although SfM holds a slight advantage in repro­ jection error, Plant3R's comprehensive point cloud generation capability is crucial for achieving high-quality dense reconstruction and detailed 3DGS rendering, which are key to accurate crop phenotyping analysis.

**中文:** 摄像头姿势估计的准确性和初始点云直接影响了后续3DGS染的细节完整性和密集重建的质量.为了全面和客观地评估Plant3R框架的性能与传统的SfM算法相比,我们对四个关键的小麦生长阶段进行了量化评估.评估侧重于两个方面:全球统计性能 (表 1) 和植物特定的几何忠实性 (表 2).如表 1中所示,Plant3R模型在四个关键阶段中表现出了显著优势,在功能提取和匹配能力方面.Plant3R的匹配率持续地表现出SfM,在缩阶段达到48.35%左右,比SfM大约14%. 此外,Plant3R模型所生成的重建稀疏点数量是SfM的数量数倍的.例如,在造阶段,Plant3R重建了48,028个点,而SfM仅产生了10,608个点,并保持了相似的趋势在结合,料填充和成熟阶段.Plant3R的每张图像平均观察量比SfM高3到5倍 (例如,Tillering的6995.92与1604.46).这表明Plant3R建立了2D图像和3D空间之间的更丰富的连接性.关于错误投射错误,Plant3R的平均重新投射率 (~1.45px) 比SfM (~1.12px) 更高. 然而,SfM的这种显而易见精度通过在低质量区域的特征点进行激进过,从而产生极少的输出来实现.相反,Plant3R故意保留了更多的具有挑战性的特征点,接受了微小的像素级变化增加,以保持几何完整性和确保更广泛的场景覆盖.为了进一步分析下游3DGS重建至关重要的几何基础是植物而不是背景,进行了专注于该区域的量化分析 (表 2).而表 1 确立了Plant3R产生更密集和更丰富的稀疏重建,表 2 评估了这种重建几何质量与实际植物相匹配的程度. 结果显示,Plant3R在所有增长阶段都取得了比SfM算法更高的有效植物点,植物点比率 (PPR) 和天花板覆盖率指数 (CCI) 数量.例如,在工阶段,Plant3R获得了3197个植物点,占所有重建点的6.65%;SfM只获得了43个植物点 (0.41%).在成熟阶段,Plant3R获得了9.92%的PPR和18.4%的天花板覆盖率指数,而SfM仅获得3.88%的PPR和2.4%的CCI.CCI量化量化量化量化量化量化量化量化量化量化量,表明Plant3R点不仅产生了更全面的重建,而且更有意义的生物模型也占据了位于植物上结构的比例. 这反映了Plant3R能够恢复更完整,更连续的天窗几何学,特别是低纹理和几何复杂的区域,这些地区对传统基于特征的SfM方法具有挑战性.总结来说,表1和2中的数量结果表明,在冬季的关键小麦生长阶段,Plant3R模型在特征提取,匹配和生成稀有点云密度方面显著优于传统SfM算法.尽管SfM在排放错误方面有所略有优势,但Plant3R的全面云生成能力对于实现高质量的密集重建和详细的3DGS染至关重要,这些都是精确的作物异型分析的关键.

<a id="S066"></a>
**Source:** p.6 S066  
**Type:** body  
**Confidence:** high

**Original:** winter wheat potted plants obtained by both methods, directly con­ firming the quantitative differences mentioned above. As shown in

**中文:** 冬季小麦植物采用两种方法得到了,直接确认了上述数量差异.

<a id="S067"></a>
### 2.6. Implementation settings and model evaluate methods
**Source:** p.6 S067  
**Type:** section  
**Confidence:** high

**Original:** 2.6. Implementation settings and model evaluate methods

**中文:** 【标题暂译】2.6. Implementation settings and model evaluate methods

<a id="S068"></a>
**Source:** p.6 S068  
**Type:** body  
**Confidence:** high

**Original:** All model training and evaluation experiments were conducted on Windows 10 operating system, using an Intel® Core™ i7-10700 CPU and an Nvidia GeForce RTX 4080 GPU. More information about key hardware components was shown in Table S2. Visual Studio Code was used as the programming environment, with Python 3.10 as the pro­ gramming language. All other comparison algorithms were executed under this setting. To quantitatively evaluate the rendering quality of our model, we employed several objective metrics to measure the differences between rendered and real images, including Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS). PSNR represents the ratio between the maximum possible signal power of an image and the power of cor­ rupting noise by calculating the mean square error (MSE) between the real and rendered images, and then converted to a logarithmic scale. As a crucial indicator of image quality, a higher PSNR value signifies better image quality. SSIM assesses the similarity between images by comprehensively comparing their luminance, contrast, and structural information; a value closer to 1 indicates higher structural similarity and better visual quality. LPIPS evaluates the perceptual similarity between images by extracting deep perceptual features using a pre-trained con­ volutional neural network. In this study, the AlexNet network [42] was selected as the feature extractor to derive features from rendered and actual images and evaluate their discrepancies. These metrics will collectively provide a comprehensive quantitative basis for our model's rendering performance.

**中文:** 所有模型培训和评估实验都在Windows 10操作系统上进行,使用了Intel® CoreTM i7-10700 CPU和Nvidia GeForce RTX 4080 GPU.有关关键硬件组件的更多信息在表S2中显示.视觉工作室代码被用于编程环境,Python 3.10是编程语言.所有其他比较算法都在此设置下执行.为了量化评估我们模型的染质量,我们使用了几个客观的指标来测量染图像和真实图像之间的差异,包括峰值信号与噪音比率 (PSNR),结构相似图像测量指数 (SSIM) 和学习的感知补丁相似性 (LPIPS). PSNR表示图像最大信号功率和破坏噪音功率之间的比率,通过计算真实和呈现图像之间的平均平方错误 (MSE),然后转换为一个逻辑尺度.作为图像质量的关键指标,更高的 PSNR值意味着更好的图像质量.SSIM通过全面比较图像的亮度,对比和结构信息来评估图像之间的相似性;接近1的值表明更高的结构相似性和更好的视觉质量.LPIPS通过使用预训练的转变神经网络来提取深度的感知特征来评估图像之间的感知相似性. 在本研究中,亚历克斯网网络[42]被选为特征提取器,以从染和实际图像中提取特征,并评估它们的不一致性.这些指标将共同为我们的模型的染性能提供全面的量化基础.

<a id="S069"></a>
### 3. Results
**Source:** p.6 S069  
**Type:** section  
**Confidence:** high

**Original:** 3. Results

**中文:** 3. 结果

<a id="S070"></a>
### 3.1. High-fidelity reconstruction efficiency
**Source:** p.6 S070  
**Type:** section  
**Confidence:** high

**Original:** 3.1. High-fidelity reconstruction efficiency

**中文:** 【标题暂译】3.1. High-fidelity reconstruction efficiency

<a id="S071"></a>
**Source:** p.6 S071  
**Type:** body  
**Confidence:** high

**Original:** The Gaussian reconstruction of high-fidelity plants is divided into three parts, and the total time is approximately 50 min: among them, the acquisition of image data requires about 60 s; The time required for camera pose estimation and sparse point cloud reconstruction varies from 2 to 5 min depending on the complexity of the reconstructed scene. The initialization and rendering of plant point clouds are divided into 7000 training iterations and 30,000 training iterations. Among them, 7000 iterations take 3 to 5 min, and it takes approximately 20 min to complete 30,000 training iterations. After the 3D Gaussian rendering of the scene is completed, the time spent extracting the plant mesh depends on the structural complexity of the plants, basically ranging from 1 to 10 min. Compared with the traditional COLMAP, which takes several hours to complete dense reconstruction and Poisson reconstruction to extract the accurate three-dimensional mesh of plants, Plant3R achieves faster convergence and higher-quality reconstruction results. This improve­ ment mainly stems from the stronger initialization provided by the MASt3R-based sparse reconstruction, which allows the 3DGS stage to

**中文:** 高效度工厂的高效度重建被划分为三个部分,总时间约为50分钟:其中,图像数据的获取需要约60秒;摄像头姿势估计和稀点云重建所需的时间在复杂的重建场景上有所不同,从2分钟到5分钟.工厂点云的初始化和染分为7000次训练代和3万次训练代.其中,7000次训练代需要3分钟到5分钟,完成3万次训练代约需要20分钟.完成3D高效率的代后,代的时间基本上取决于工厂的结构复杂性,从1分钟到10分钟. 与传统的COLMAP相比,完成密集重建需要几个小时,以及Poisson重建以提取精确的三维植物网,Plant3R实现了更快的融合和更高质量的重建结果.这种改进主要是由于 MASt3R基于稀疏重建的更强的初始化,使得3DGS阶段能够


## Page 7

<a id="S072"></a>
**Source:** p.7 S072  
**Type:** body  
**Confidence:** high

**Original:** J. Ma et al.

**中文:** 詹姆斯·马等人

<a id="F002"></a>
### Fig. 2. 小麦不同生育阶段下 SfM 与 Plant3R 的比较。为直观展示 Plant3R 模型的特征提取与稀疏点云重建能力，作者选取其重建的稀疏点云与 SfM 算法进行可视化对比。
**Placed near:** p.7  
**Source:** p.7 manual-layout  
**Crop confidence:** high

![Fig. 2](assets/fig2.png)

**Original caption:** Fig. 2. Comparison between SfM and Plant3R at different growth stages of wheat. In order to intuitively demonstrate the feature extraction and sparse point cloud reconstruction capabilities of the Plant3R model, we selected its reconstructed sparse point cloud for visual comparison with the SfM algorithm.

**中文图注:** 图 2. 小麦不同生育阶段下 SfM 与 Plant3R 的比较。为直观展示 Plant3R 模型的特征提取与稀疏点云重建能力，作者选取其重建的稀疏点云与 SfM 算法进行可视化对比。

**Reading note:** 重点查看该图如何支撑相邻正文中的流程、比较、消融或性状提取结果。

<a id="S073"></a>
**Source:** p.7 S073  
**Type:** body  
**Confidence:** high

**Original:** Results of camera pose and sparse point cloud estimations. Method

**中文:** 摄像头姿势和稀有点云估计的结果.

<a id="S074"></a>
**Source:** p.7 S074  
**Type:** body  
**Confidence:** high

**Original:** Points

**中文:** 积分

<a id="S075"></a>
**Source:** p.7 S075  
**Type:** body  
**Confidence:** high

**Original:** Observations

**中文:** 观察

<a id="S076"></a>
**Source:** p.7 S076  
**Type:** body  
**Confidence:** high

**Original:** Mean Match Rate

**中文:** 平均匹配率

<a id="S077"></a>
**Source:** p.7 S077  
**Type:** body  
**Confidence:** high

**Original:** Mean Observations Per Image

**中文:** 每张图片平均观察量

<a id="S078"></a>
**Source:** p.7 S078  
**Type:** body  
**Confidence:** high

**Original:** Mean Reprojection Error

**中文:** 这意味着重新投放错误

<a id="S079"></a>
**Source:** p.7 S079  
**Type:** body  
**Confidence:** high

**Original:** Tillering

**中文:** 缩

<a id="S080"></a>
**Source:** p.7 S080  
**Type:** body  
**Confidence:** high

**Original:** Plant3R SfM

**中文:** 植物3R SfM

<a id="S081"></a>
**Source:** p.7 S081  
**Type:** body  
**Confidence:** high

**Original:** 48.35% 34.05%

**中文:** 48.35% 34.05%

<a id="S082"></a>
**Source:** p.7 S082  
**Type:** body  
**Confidence:** high

**Original:** 6955.92 1604.46

**中文:** 6955.92 1604.46

<a id="S083"></a>
**Source:** p.7 S083  
**Type:** body  
**Confidence:** high

**Original:** 1.484 1.118

**中文:** 484 1,118 1,118

<a id="S084"></a>
**Source:** p.7 S084  
**Type:** body  
**Confidence:** high

**Original:** Jointing

**中文:** 关联

<a id="S085"></a>
**Source:** p.7 S085  
**Type:** body  
**Confidence:** high

**Original:** Plant3R SfM

**中文:** 植物3R SfM

<a id="S086"></a>
**Source:** p.7 S086  
**Type:** body  
**Confidence:** high

**Original:** 41.09% 36.43%

**中文:** 百分之四十九,百分之三三.四十九,百分之三.四十九,百分之三.四十四.四十三.四十三.

<a id="S087"></a>
**Source:** p.7 S087  
**Type:** body  
**Confidence:** high

**Original:** 8768.27 2507.23

**中文:** 据悉,这次的比赛是27272507.23 8768.

<a id="S088"></a>
**Source:** p.7 S088  
**Type:** body  
**Confidence:** high

**Original:** 1.449 1.143

**中文:** 1,449 1,143

<a id="S089"></a>
**Source:** p.7 S089  
**Type:** body  
**Confidence:** high

**Original:** Grain Filling

**中文:** 粮食充

<a id="S090"></a>
**Source:** p.7 S090  
**Type:** body  
**Confidence:** high

**Original:** Plant3R SfM

**中文:** 植物3R SfM

<a id="S091"></a>
**Source:** p.7 S091  
**Type:** body  
**Confidence:** high

**Original:** 45.72% 32.37%

**中文:** 七十二分之二 32.37分之七 45.72%

<a id="S092"></a>
**Source:** p.7 S092  
**Type:** body  
**Confidence:** high

**Original:** 6113.5 1354.79

**中文:** 6113.5 1354.79

<a id="S093"></a>
**Source:** p.7 S093  
**Type:** body  
**Confidence:** high

**Original:** 1.475 1.116

**中文:** 1,116 1.475 个

<a id="S094"></a>
**Source:** p.7 S094  
**Type:** body  
**Confidence:** high

**Original:** Maturity

**中文:** 成熟度

<a id="S095"></a>
**Source:** p.7 S095  
**Type:** body  
**Confidence:** high

**Original:** Plant3R SfM

**中文:** 植物3R SfM

<a id="S096"></a>
**Source:** p.7 S096  
**Type:** body  
**Confidence:** high

**Original:** 43.04% 31.84%

**中文:** 43.04% 31.84%

<a id="S097"></a>
**Source:** p.7 S097  
**Type:** body  
**Confidence:** high

**Original:** 7232.13 1560.53

**中文:** 7232.13 1560.53

<a id="S098"></a>
**Source:** p.7 S098  
**Type:** body  
**Confidence:** high

**Original:** 1.457 1.137

**中文:** 457 1.137 1.137

<a id="S099"></a>
**Source:** p.7 S099  
**Type:** body  
**Confidence:** high

**Original:** Note: The term “Points” denote the number of points obtained in sparse reconstruction; “Observations” represent the total number of observations of all points across images; “Mean match rate” refers to the proportion of matched feature points among all observed points; “Mean observations per image” indicates the average number of observed points per image; “Mean reprojection error” describes the average reprojection error.

**中文:** 注意: 点表示在稀疏的重建中获得的点数; 观察表示图像中所有点的总数; 平均匹配率指所有观测点之间的匹配特征点比例; 平均观察点表示每图像的平均数量; 平均重投错描述平均重投错.

<a id="S100"></a>
**Source:** p.7 S100  
**Type:** body  
**Confidence:** high

**Original:** clouds generated by COLMAP appear sparse and uneven, with numerous significant holes in leaf and stem regions. This difference in results stems fundamentally from the choice of image matching and feature extraction strategies. Compared to the conservative feature selection of traditional SfM algorithms, the Transformer-based architecture of the MASt3R model learns and infers matching relationships from a broader image context through point graph regression and dense correspondence. This enables the extraction of significantly more feature points, thereby substantially increasing point cloud density and the feature point matching rate, and demon­ strating robustness and high coverage in complex scenes. This not only provides more accurate and richer geometric priors for the subsequent 3DGS model but also guarantees the stability of model convergence, particularly in low-texture plant reconstruction scenarios.

**中文:** 由于COLMAP生成的图像出现稀疏和不均,叶片和干部区域存在许多显著的洞穴.这种结果差异主要来自图像匹配和特征提取策略的选择.与传统的SfM算法的保守特征选择相比,MASTt3R模型的基于变压器的架构通过点图回归和密集相应,通过更广泛的图像环境学习和推断匹配关系.这使得更多的特征点得到提取,从而大幅增加点云密度和特征点匹配率,并使复杂场景中的强度和高覆盖率变化. 这不仅为后来的3DGS模型提供了更准确,更丰富的几何预测,而且还保证了模型融合的稳定性,特别是在低纹理的工厂重建场景中.

<a id="S101"></a>
**Source:** p.7 S101  
**Type:** body  
**Confidence:** high

**Original:** Quantitative evaluation of sparse point cloud quality for different initialization methods. Method

**中文:** 对于不同初始化方法的稀点云质量量量评估.

<a id="S102"></a>
**Source:** p.7 S102  
**Type:** body  
**Confidence:** high

**Original:** Total Points

**中文:** 总积分

<a id="S103"></a>
**Source:** p.7 S103  
**Type:** body  
**Confidence:** high

**Original:** Plant Points

**中文:** 植物点

<a id="S104"></a>
**Source:** p.7 S104  
**Type:** body  
**Confidence:** high

**Original:** Plant Point Ratio

**中文:** 植物点比率

<a id="S105"></a>
**Source:** p.7 S105  
**Type:** body  
**Confidence:** high

**Original:** Canopy Coverage Index

**中文:** 顶覆盖率指数

<a id="S106"></a>
**Source:** p.7 S106  
**Type:** body  
**Confidence:** high

**Original:** Tillering

**中文:** 缩

<a id="S107"></a>
**Source:** p.7 S107  
**Type:** body  
**Confidence:** high

**Original:** Plant3R SfM

**中文:** 植物3R SfM

<a id="S108"></a>
**Source:** p.7 S108  
**Type:** body  
**Confidence:** high

**Original:** 6.65% 0.41%

**中文:** 6.65% 0.41%

<a id="S109"></a>
**Source:** p.7 S109  
**Type:** body  
**Confidence:** high

**Original:** 12.1% 0.3%

**中文:** 12.1% 0.3%

<a id="S110"></a>
**Source:** p.7 S110  
**Type:** body  
**Confidence:** high

**Original:** Jointing

**中文:** 关联

<a id="S111"></a>
**Source:** p.7 S111  
**Type:** body  
**Confidence:** high

**Original:** Plant3R SfM

**中文:** 植物3R SfM

<a id="S112"></a>
**Source:** p.7 S112  
**Type:** body  
**Confidence:** high

**Original:** 3.54% 0.62%

**中文:** 对于3.54%的,这是0.62%的.

<a id="S113"></a>
**Source:** p.7 S113  
**Type:** body  
**Confidence:** high

**Original:** 7.5% 2.0%

**中文:** 七.5%的2.0%

<a id="S114"></a>
**Source:** p.7 S114  
**Type:** body  
**Confidence:** high

**Original:** Grain Filling

**中文:** 粮食充

<a id="S115"></a>
**Source:** p.7 S115  
**Type:** body  
**Confidence:** high

**Original:** Plant3R SfM

**中文:** 植物3R SfM

<a id="S116"></a>
**Source:** p.7 S116  
**Type:** body  
**Confidence:** high

**Original:** 11.58% 2.77%

**中文:** 11,58% 2.77%

<a id="S117"></a>
**Source:** p.7 S117  
**Type:** body  
**Confidence:** high

**Original:** 14.9% 0.1%

**中文:** 只有0.1%14.9%

<a id="S118"></a>
**Source:** p.7 S118  
**Type:** body  
**Confidence:** high

**Original:** Maturity

**中文:** 成熟度

<a id="S119"></a>
**Source:** p.7 S119  
**Type:** body  
**Confidence:** high

**Original:** Plant3R SfM

**中文:** 植物3R SfM

<a id="S120"></a>
**Source:** p.7 S120  
**Type:** body  
**Confidence:** high

**Original:** 9.92% 3.88%

**中文:** 9.92% 3.88%

<a id="S121"></a>
**Source:** p.7 S121  
**Type:** body  
**Confidence:** high

**Original:** 18.4% 2.4%

**中文:** 18.4% 2.4%

<a id="S122"></a>
**Source:** p.7 S122  
**Type:** body  
**Confidence:** medium

**Original:** Note: The term “Total Points” denote the number of points obtained in sparse reconstruction; “Plant Points” represent the number of the plant; “Plant Point Ratio” refers to the proportion of plant-region points to all reconstructed points () PPR = Nplant =Ntotal; “Canopy Coverage Index” represents the proportion of occupied grid cells on the XY projection plane to the total grid cells within the plant's bounding box (CCI = Nocc =Ntotal).

**中文:** 注意: 总点表示在稀疏重建中获得的点数; 植物点表示植物数; 植物点比率指植物区域点与所有重建点的比例 () PPR = Nplant =Ntotal; Canopy 覆盖率指数表示XY投影平面上占用的电网电池与植物界限框内的电网电池总数的比例 (CCI = Nocc =Ntotal).

<a id="S123"></a>
### 3.3. Gaussian rendering results of wheat
**Source:** p.7 S123  
**Type:** section  
**Confidence:** high

**Original:** 3.3. Gaussian rendering results of wheat

**中文:** 【标题暂译】3.3. Gaussian rendering results of wheat

<a id="S124"></a>
**Source:** p.7 S124  
**Type:** body  
**Confidence:** high

**Original:** This section aims to evaluate the Plant3R model's real-time rendering and geometric mesh extraction performance using our experimentally collected wheat dataset. To quantitatively analyze the strengths and weaknesses of different algorithms, a multi-dimensional evaluation system was adopted. This system introduced metrics such as PSNR, SSIM, and LPIPS, in addition to visual comparisons. Compared to traditional SfM-MVS, NeRF, and 3DGS algorithms, this model exhibited more significant advantages in agricultural scene applications.

**中文:** 本节旨在通过我们实验收集的小麦数据集来评估Plant3R模型的实时染和几何网格提取性能.为了量化分析不同算法的优点和缺点,采用了多维评估系统.该系统除了视觉比较之外,还引入了PSNR,SSIM和LPIPS等指标.与传统的SfM-MVS,NeRF和3DGS算法相比,该模型在农业领域的应用中表现出了更显著的优势.

<a id="S125"></a>
**Source:** p.7 S125  
**Type:** body  
**Confidence:** high

**Original:** higher coverage and uniformity across the entire winter wheat canopy, especially in regions that are challenging for traditional methods, such as low-texture and complex geometric areas. In contrast, the point

**中文:** 整个冬季的小麦顶的覆盖率和统一性更高,特别是在传统方法面临挑战的地区,如低纹理和复杂的几何区域.

<a id="S126"></a>
**Source:** p.7 S126  
**Type:** body  
**Confidence:** high

**Original:** 3.4. 2D image rendering comparison The following focus on thoroughly evaluating the 2D image rendering quality of the Plant3R model. We conducted a detailed anal­ ysis by comparing the rendering results of indoor wheat at different growth stages from the Plant3R model with those from Structure-fromMotion (SfM)-based NeRF and 3DGS algorithms. A comparison of the rendering results for NeRF, 3DGS, and Plant3R across different growth stages is presented in Fig. 3. From the visual­ izable results, our Plant3R model demonstrates significant advantages. Specifically, by integrating richer geometric prior information during the rendering initialization stage, Plant3R shows excellent results in handling artifacts and noise near the target object, effectively avoiding potential interference introduced by rendering noise. Furthermore, the Plant3R model is capable of capturing richer details with high fidelity when reconstructing wheat at different stages, including leaf textures, stem structures, and the microscopic morphology of wheat ears, with superior detail performance at high iteration counts. Although this model has exhibited superior rendering capabilities, we also note that it still inevitably shares some common issues inherent to 3DGS-based models. These limitations will be further discussed in subsequent sections. From the comparison of metrics across different growth stages in

**中文:** 3.4.2D图像染比较下面的重点是彻底评估Plant3R模型的2D图像染质量.我们通过将Plant3R模型不同成长阶段的室内小麦染结果与基于SfM的NeRF和3DGS算法进行比较进行了详分割析.在不同成长阶段的NeRF,3DGS和Plant3R染结果的比较,在图 3.从可视性结果中,我们的Plant3R模型显示出了显著的优势.具体来说,通过在染初始化过程中整合丰富的几何信息,Plant3R在处理目标物体和噪音附近的结果中表现出了卓越的成效,有效地通过染噪音的实现了染. 此外,Plant3R模型能够在重建小麦时以高度度度捕获更丰富的细节,包括不同阶段的叶纹,干结构和小麦耳的微观形态,在高代数时具有优越的细节性能.尽管这个模型表现出了优越的染能力,但我们还注意到它仍然不可避免地分享了基于3DGS的模型的一些固有的共同问题.这些局限性将进一步讨论在后续部分.从不同增长阶段的指标的比较中,我们将在中进行更多的分析.

<a id="F003"></a>
### Fig. 3. 三种不同算法在小麦四个生育阶段的二维图像渲染结果。
**Placed near:** p.7 S126  
**Source:** p.8 manual-layout  
**Crop confidence:** high

![Fig. 3](assets/fig3.png)

**Original caption:** Fig. 3. 2D image rendering results of wheat's four growth stages under three different algorithms.

**中文图注:** 图 3. 三种不同算法在小麦四个生育阶段的二维图像渲染结果。

**Reading note:** 重点查看该图如何支撑相邻正文中的流程、比较、消融或性状提取结果。

<a id="S127"></a>
**Source:** p.7 S127  
**Type:** body  
**Confidence:** high

**Original:** wheat. In order to intuitively demonstrate the feature extraction and sparse point cloud reconstruction capabilities of the Plant3R model, we selected its reconstructed sparse point cloud for visual comparison with the SfM algorithm.

**中文:** 为了直观地展示Plant3R模型的特征提取和稀点云重建能力,我们选择了其重建的稀点云,以便与SfM算法进行视觉比较.

<a id="C002"></a>
### Table 3, the Plant3R method generally demonstrates excellent rendering
**Source:** p.7 C002  
**Type:** caption  
**Confidence:** high

**Original:** Table 3, the Plant3R method generally demonstrates excellent rendering

**中文:** 在表3中,Plant3R方法一般显示出出了优秀的染.


## Page 8

<a id="S128"></a>
**Source:** p.8 S128  
**Type:** body  
**Confidence:** high

**Original:** J. Ma et al.

**中文:** 詹姆斯·马等人

<a id="S129"></a>
**Source:** p.8 S129  
**Type:** body  
**Confidence:** high

**Original:** Overall, the Plant3R method proposed in this study successfully combines the advantages of the MASt3R model in sparse point cloud generation with the characteristics of 3DGS in rendering efficiency and detail representation, thereby achieving higher quality rendering re­ sults. This method can more effectively preserve structural information in complex leaf textures and accurately reflect key agricultural pheno­ typic features, such as crop leaf edges and spikelet details at different growth stages, which is of great significance for crop 3D phenotyping research. 3.5. 3D geometry extraction of wheat at different growth stages High-quality 3D geometric results are crucial for accurate pheno­ typic analysis. They not only enable precise characterization of plant morphological features but also provide a structural basis for a deeper understanding of their physiological state. This is particularly important for the precise acquisition of information in complex and variable growth environments. To comprehensively evaluate our proposed Plant3R model, this section further compared its 3D geometric recon­ struction capabilities with several widely recognized algorithms (Col­ map, NeRF, and 3DGS). Fig. 4 visually presents the extracted geometric structures of wheat plants at different growth stages reconstructed by these various methods. From the overall visual effect of Fig. 4, our Plant3R model demon­ strates a clear advantage in 3D geometric reconstruction fidelity, structural completeness, and detail capture capability. The geometric results generated by Colmap exhibit significant incompleteness, partic­ ularly noticeable noise or omissions at the edges of plant leaves, failing to provide a smooth and continuous surface. In contrast, NeRF can achieve high-quality 2D image rendering, yet its geometric reconstruc­ tion results are sometimes accompanied by artifacts, surface blur, and insufficient representation of fine structures. The Plant3R model, how­ ever, is capable of reconstructing the complex 3D geometric morphology of wheat plants with higher precision and completeness. This includes clear and continuous stem forms, delicate spikelet structures, and even distinct texture details on the leaf surfaces, as shown in Fig. 5. This highfidelity 3D reconstruction capability enables Plant3R to provide more robust and refined 3D wheat models for downstream plant phenotyping analysis across various growth contexts. In summary, the Plant3R model's exceptional performance in 3D geometric reconstruction provides a solid data foundation for achieving high-precision, high-throughput wheat phenotyping analysis. This

**中文:** 总的来说,本研究提出的Plant3R方法成功地结合了MASTt3R模型在稀有点云生成中的优势,并与3DGS在效率和细节表示方面的特点,从而实现更高质量的染结果.这种方法可以更有效地保存复杂的叶纹理中的结构信息,并更准确地反映了农业现象的关键特征,如不同种植的叶片边缘和不同种植阶段的尖细节,这对于作物3D表型研究非常重要.3.5.不同种植阶段的小麦3D几何提取3D几何学结果是对准确的形分析至关重要的. 它们不仅能准确地描述植物形态特征,而且还提供了对它们生理状况的更深入理解的结构基础.这对于在复杂和变化的生长环境中准确地获取信息尤为重要.为了全面评估我们提出的Plant3R模型,本节进一步将其3D几何重建能力与几种广泛公认的算法 (COLMAP,NeRF和3DGS) 进行比较.图4视觉地呈现了小麦植物在不同生长阶段的抽取几何结构,这些方法重建了它们.从图4的整体视觉效应中,我们的Plant3R模型在3D几何重建忠实,结构完整性和细节捕获能力方面显著优势. 科尔玛普产生的几何结果显示出显著的不完整性,特别是植物叶边缘的明显的噪音或遗漏,无法提供平滑和连续的表面.相比之下,NeRF可以实现高质量的2D图像染,然而其几何重建结果有时伴随着文物,表面模糊,以及细结构的不足表示.Plant3R模型,无论如何,能够更高精度和完整度重建小麦植物的复杂3D几何形态.这包括清晰和连续的干茎形式,细微的尖结构,甚至在叶表面上的明显的纹理细节,如图 5. 这种高效的3D重建能力使Plant3R能够提供更强大,更精致的3D小麦模型,用于在各种生长环境中下游植物型分析.总结来说,Plant3R模型在3D几何重建方面的卓越性能为实现高精度,高吞吐量小麦型分析提供了坚实的数据基础.

<a id="F004"></a>
### Fig. 4. 四种不同算法在小麦四个生育阶段的三维点云提取结果。从不同模型的点云可视化结果看，Plant3R 模型的保真度显著优于 Colmap、NeRF 和原始 3DGS。
**Placed near:** p.8 S129  
**Source:** p.8 manual-layout  
**Crop confidence:** medium

![Fig. 4](assets/fig4.png)

**Original caption:** Fig. 4. 3D point cloud extraction results of wheat's four growth stages under four different algorithms. From the point cloud visualization results of different models, the fidelity of the Plant3R model is significantly better than Colmap, NeRF and original 3DGS.

**中文图注:** 图 4. 四种不同算法在小麦四个生育阶段的三维点云提取结果。从不同模型的点云可视化结果看，Plant3R 模型的保真度显著优于 Colmap、NeRF 和原始 3DGS。

**Reading note:** 重点查看该图如何支撑相邻正文中的流程、比较、消融或性状提取结果。

<a id="F005"></a>
### Fig. 5. NeRF、3DGS 与 Plant3R 在叶片、茎秆和穗部等模型表面细节上的比较。
**Placed near:** p.8 S129  
**Source:** p.9 manual-layout  
**Crop confidence:** high

![Fig. 5](assets/fig5.png)

**Original caption:** Fig. 5. Comparison between NeRF, 3DGS and Plant3R in model surface details such as leaf, stem and ear.

**中文图注:** 图 5. NeRF、3DGS 与 Plant3R 在叶片、茎秆和穗部等模型表面细节上的比较。

**Reading note:** 重点查看该图如何支撑相邻正文中的流程、比较、消融或性状提取结果。

<a id="S130"></a>
**Source:** p.8 S130  
**Type:** body  
**Confidence:** high

**Original:** different algorithms.

**中文:** 它们的算法不同.

<a id="S131"></a>
**Source:** p.8 S131  
**Type:** body  
**Confidence:** high

**Original:** capabilities. Particularly in the tillering, jointing, and grain-filling stages, PSNR and SSIM reached their highest values, indicating that this method has a significant advantage in maintaining reconstructed geometric accuracy and texture details, enabling accurate reproduction of the plant's true morphology. Although the PSNR in the maturity stage was slightly lower than that of 3DGS, the difference was very minimal (e.g., the PSNR for the Plant3R model at the maturity stage was 30.62, while for 3DGS it was 30.98). Furthermore, the Plant3R model exhibited better overall stability, maintaining a high level of visual quality. The 3DGS method performed relatively consistently across all stages, particularly excelling in the LPIPS perceptual quality metric, showcasing its advantages in detail capture and realistic perception. This could be related to the sensitivity of its point cloud-based rendering mechanism to local details. In contrast, NeRF's metrics were significantly lower than the other two methods across all growth stages. This was especially noticeable in later growth stages where crop morphology is more com­ plex, with a more pronounced decline in reconstruction accuracy and structural fidelity. This reflects the challenges NeRF may face when processing complex, non-rigid objects (such as plant leaves with fine textures and intricate structures), particularly when data volume or viewpoint coverage is insufficient.

**中文:** 特别是在削,结合和料填补阶段,PSNR和SSIM达到最高值,这表明这种方法在保持重建的几何精度和纹理细节方掩膜有显著优势,使植物真正的形态能够准确地复制.虽然成熟阶段的PSNR略低于3DGS,但差异非常小 (例如,成熟阶段的PSNR为Plant3R模型是30.62,而3DGS为30.98).此外,Plant3R模型表现出更好的整体稳定性,保持高水平的视觉质量.3DGS方法在所有阶段都表现得相对一致,特别是在LPIPS的细节质量中表现出卓越的感知优势,并实现了其现实感知优势. 这可能与其基于点云的染机制对本地细节的敏感性有关.相反,NeRF的指标在所有生长阶段都比其他两种方法要低得多.这尤其明显在后期的生长阶段,作物形态更复杂,重建精度和结构忠诚度更明显下降.这反映了NeRF在处理复杂的,非刚性物体 (如细纹理和复杂结构的植物叶片) 时可能面临的挑战,特别是当数据量或视点覆盖率不足时.

<a id="S132"></a>
**Source:** p.8 S132  
**Type:** body  
**Confidence:** high

**Original:** Evaluation results of different 3D reconstruction methods for wheat at different growth stages. Metrics | Methods

**中文:** 评估结果不同增长阶段的小麦不同3D重建方法.

<a id="S133"></a>
**Source:** p.8 S133  
**Type:** body  
**Confidence:** high

**Original:** NeRF

**中文:** 尼RF

<a id="S134"></a>
**Source:** p.8 S134  
**Type:** body  
**Confidence:** high

**Original:** 3DGS

**中文:** 3DGS

<a id="S135"></a>
**Source:** p.8 S135  
**Type:** body  
**Confidence:** high

**Original:** Plant3R

**中文:** 植物3R

<a id="S136"></a>
**Source:** p.8 S136  
**Type:** body  
**Confidence:** high

**Original:** Tillering

**中文:** 缩

<a id="S137"></a>
**Source:** p.8 S137  
**Type:** body  
**Confidence:** high

**Original:** PSNR SSIM LIPIS

**中文:** 据悉,PSNRIM的LIPIS是PSNRIM的LIPIS.

<a id="S138"></a>
**Source:** p.8 S138  
**Type:** body  
**Confidence:** high

**Original:** 27.44 0.89 0.11

**中文:** 44 0.89 0.11 27.44

<a id="S139"></a>
**Source:** p.8 S139  
**Type:** body  
**Confidence:** high

**Original:** 32.61 0.92 0.31

**中文:** 32.61 0.92 0.31

<a id="S140"></a>
**Source:** p.8 S140  
**Type:** body  
**Confidence:** high

**Original:** 34.03 0.92 0.28

**中文:** 34.03 0.92 0.28

<a id="S141"></a>
**Source:** p.8 S141  
**Type:** body  
**Confidence:** high

**Original:** Jointing

**中文:** 关联

<a id="S142"></a>
**Source:** p.8 S142  
**Type:** body  
**Confidence:** high

**Original:** PSNR SSIM LIPIS

**中文:** 据悉,PSNRIM的LIPIS是PSNRIM的LIPIS.

<a id="S143"></a>
**Source:** p.8 S143  
**Type:** body  
**Confidence:** high

**Original:** 21.58 0.76 0.27

**中文:** 五百八十八.七十六.七十七.二十七 21.

<a id="S144"></a>
**Source:** p.8 S144  
**Type:** body  
**Confidence:** high

**Original:** 29.46 0.92 0.30

**中文:** 29.46 0.92 0.30

<a id="S145"></a>
**Source:** p.8 S145  
**Type:** body  
**Confidence:** high

**Original:** 34.64 0.94 0.29

**中文:** 34.64 0.94 0.29

<a id="S146"></a>
**Source:** p.8 S146  
**Type:** body  
**Confidence:** high

**Original:** Grain Filling

**中文:** 粮食充

<a id="S147"></a>
**Source:** p.8 S147  
**Type:** body  
**Confidence:** high

**Original:** PSNR SSIM LIPIS

**中文:** 据悉,PSNRIM的LIPIS是PSNRIM的LIPIS.

<a id="S148"></a>
**Source:** p.8 S148  
**Type:** body  
**Confidence:** high

**Original:** 26.18 0.87 0.17

**中文:** 18.18 0.87 0.17 26.

<a id="S149"></a>
**Source:** p.8 S149  
**Type:** body  
**Confidence:** high

**Original:** 29.64 0.91 0.28

**中文:** 29.64 0.91 0.28

<a id="S150"></a>
**Source:** p.8 S150  
**Type:** body  
**Confidence:** high

**Original:** 34.02 0.94 0.26

**中文:** 34.02 0.94 0.26

<a id="S151"></a>
**Source:** p.8 S151  
**Type:** body  
**Confidence:** high

**Original:** Maturity

**中文:** 成熟度

<a id="S152"></a>
**Source:** p.8 S152  
**Type:** body  
**Confidence:** high

**Original:** PSNR SSIM LIPIS

**中文:** 据悉,PSNRIM的LIPIS是PSNRIM的LIPIS.

<a id="S153"></a>
**Source:** p.8 S153  
**Type:** body  
**Confidence:** high

**Original:** 20.95 0.74 0.27

**中文:** 20.95 0.74 0.27

<a id="S154"></a>
**Source:** p.8 S154  
**Type:** body  
**Confidence:** high

**Original:** 30.98 0.92 0.28

**中文:** 30.98 0.92 0.28

<a id="S155"></a>
**Source:** p.8 S155  
**Type:** body  
**Confidence:** high

**Original:** 30.62 0.92 0.27

**中文:** 30.62 0.92 0.27 30.62 0.92 0.27

<a id="S156"></a>
**Source:** p.8 S156  
**Type:** body  
**Confidence:** high

**Original:** four different algorithms. From the point cloud visualization results of different models, the fidelity of the Plant3R model is significantly better than Colmap, NeRF and original 3DGS.

**中文:** 从不同模型的点云可视化结果看,Plant3R模型的忠实性明显优于COLMAP,NeRF和原始3DGS.


## Page 9

<a id="S157"></a>
**Source:** p.9 S157  
**Type:** body  
**Confidence:** high

**Original:** J. Ma et al.

**中文:** 詹姆斯·马等人

<a id="S158"></a>
**Source:** p.9 S158  
**Type:** body  
**Confidence:** high

**Original:** enhancement in capability is expected to improve efficiency and accu­ racy in the field of crop breeding and holds significant importance for advancing modern agricultural phenotyping research.

**中文:** 能力增强预计将提高作物育种领域的效率和率,并且对于推进现代农业表型研究具有重大意义.

<a id="S159"></a>
**Source:** p.9 S159  
**Type:** body  
**Confidence:** high

**Original:** Plant height is an important indicator for evaluating the accuracy of reconstruction results. In this study, we defined plant height as the vertical distance between the soil plane and the highest point of the plant in the scale-recovered point cloud. To obtain a robust soil refer­ ence, the soil plane was estimated using RANSAC-based plane fitting from the lower region of the pot and soil points. The fitted plane was aligned to be parallel with OXZ plane to correct for potential tilt. The plant height(H) was then calculated as:

**中文:** 植物高度是评估重建结果的准确性的一个重要指标.在这项研究中,我们定义了植物高度为地面平面和植物在尺度恢复点云中最高点之间的垂直距离.为了获得强大的地面参考,使用RANSAC基于面积的面积从盆地和地面点的下部区域进行测量. 适应的面积被对齐并与OXZ平面进行平行调整,以纠正潜在倾斜. 植物高度(H) 然后被计算为:

<a id="S160"></a>
### 3.6. Crop phenotyping extraction and validation
**Source:** p.9 S160  
**Type:** section  
**Confidence:** high

**Original:** 3.6. Crop phenotyping extraction and validation

**中文:** 【标题暂译】3.6. Crop phenotyping extraction and validation

<a id="S161"></a>
**Source:** p.9 S161  
**Type:** body  
**Confidence:** high

**Original:** To quantitatively verify the accuracy of crop 3D models in agricul­ tural phenotyping, after completing the surface mesh extraction of wheat plants at different growth stages, this study further conducted a quantitative analysis of phenotypic parameters by comparing manually measured values of geometric phenotypic traits such as plant height, leaf length, and leaf width with model-calculated results. Considering the existence of a certain spatial scale relationship between the recon­ structed plant models and actual plants, we used a standard 6x9 checkerboard paper as a calibration target to perform spatial scale re­ covery, ensuring the accuracy of subsequent phenotypic parameter extraction and its consistency with measured data.

**中文:** 为了量化验证农业型化中的作物3D模型的准确性,在完成了不同生长阶段的小麦植物表面网格提取后,本研究进一步进行了对型参数的量化分析,通过与模型计算结果进行手动测量的几何型特征,如植物高度,叶长和叶宽等的几何型特征的值进行比较.考虑到重新结构化的植物模型和实际植物之间的某种空间尺度存在,我们使用了标准的6x9棋牌纸作为校准目标来进行空间尺度测量,确保了随后的型参数提取的准确性和与测量数据的一致性.

<a id="S162"></a>
**Source:** p.9 S162  
**Type:** body  
**Confidence:** medium

**Original:** H = max(yi) − yplane () where max yi denotes the Y-coordinate of the highest point on the plant, and yplane represents the mean Y-coordinate of the fitted soil plane. The comparison with manual measurement data is shown in Fig. 6, and the results indicate that the R2 between the model-estimated plant height and the measured values across different growth stages is as high

**中文:** 平面表示平面的平均Y坐标,而平面表示平面的平均Y坐标.与手动测量数据的比较如图 6,结果表明,模型估计的植物高度和不同生长阶段测量值之间的R2同样高.

<a id="F006"></a>
### Fig. 6. 小麦不同生育阶段模型推导株高与人工测量株高的比较。
**Placed near:** p.9 S162  
**Source:** p.9 manual-layout  
**Crop confidence:** medium

![Fig. 6](assets/fig6.png)

**Original caption:** Fig. 6. Comparison of model-derived and manually measured values of wheat plant height at different growth stages.

**中文图注:** 图 6. 小麦不同生育阶段模型推导株高与人工测量株高的比较。

**Reading note:** 重点查看该图如何支撑相邻正文中的流程、比较、消融或性状提取结果。


## Page 10

<a id="S163"></a>
**Source:** p.10 S163  
**Type:** body  
**Confidence:** high

**Original:** J. Ma et al.

**中文:** 詹姆斯·马等人

<a id="S164"></a>
**Source:** p.10 S164  
**Type:** body  
**Confidence:** high

**Original:** as 0.99. This demonstrates that the method proposed in this study can accurately extract wheat plant height information. Due to the complex geometric structure of wheat plants—charac­ terized by varying curvatures and mutual occlusions—extracting accu­ rate phenotypic traits remains a challenge. To address this, we developed a Graph-based Geodesic Skeletonization algorithm to robustly quantify leaf morphology from the segmented 3D point clouds. First, to isolate target leaves from the reconstructed plant models, we performed interactive segmentation using CloudCompare software. For the segmented leaf point clouds, we constructed a Riemannian graph structure where nodes represent 3D points and edges connect nearest neighbors (k-NN) weighted by Euclidean distance. Leaf Length Defini­ tion: Unlike simple Euclidean distance which underestimates curved structures, we defined leaf length as the geodesic distance along the leaf's central topological skeleton. We utilized Dijkstra's algorithm to compute the shortest path within the graph from the petiole base to the leaf tip, followed by spline interpolation to generate a smooth, contin­ uous central curve. Leaf Width Definition: Leaf width was calculated based on the local cross-sectional profile. For each node on the skeleton, we constructed a normal plane perpendicular to the tangent direction. The width was defined as the maximum span of the point cloud pro­ jection on this plane at the widest section of the leaf. This automated pipeline eliminates subjective errors associated with manual measure­ ments and ensures robustness against leaf curling and twisting. To validate the accuracy of the proposed Plant3R reconstruction and the skeleton-based extraction method, we compared the modelextracted values with manual measurements (ground truth) obtained during the experiment. As shown in Table 4 and Fig. 7, the regression analysis demonstrates a strong correlation between the two methods across different growth stages (Tillering, Jointing, and Grain Filling). Specifically, the coefficient of determination (R2) for leaf length excee­ ded 0.99 with a Mean Absolute Percentage Error (MAPE) below 1.1%, indicating that the skeleton extraction algorithm effectively captures the true curvature of the leaves. For leaf width, the method also achieved high precision (R2 > 0.94, RMSE <0.16 cm). All the above validation results indicate that the wheat plants reconstructed based on the Plant3R model possess highly accurate geometric structures and effectively preserve the detailed features of the leaves. This provides reliable data support for subsequent precise plant phenotyping analysis based on 3D models.

**中文:** 0.99.这表明本研究提出的方法可以准确地提取小麦植物高度信息.由于小麦植物的复杂几何结构因不同曲线和相互缩而特征化,提取率现象特征仍然是一个挑战.为了解决这一问题,我们开发了一种基于图的地质骨格化算法,以强度量化分段的3D点云的叶片形态.首先,为了从重建的植物模型中隔离目标叶片,我们使用云相比软件进行了交互式分段.对于分段的叶片点云,我们构建了一个里曼尼图结构,其中节点代表3D点和边缘连接 Euclidean 距离重量最接近的邻居 (k-N). 叶子长度定义:与低估曲线结构的简单的尤克利德距离不同,我们定义了叶子长度为叶子中部拓骨架的地质距离.我们利用迪克斯特拉算法计算了图中的从叶子底部到叶子尖端的最短路径,然后进行线插曲,产生一个平滑的,的中央曲线.叶子宽度定义:叶子宽度是根据本地横断形状计算的.对于骨架上的节点,我们构建了一个正常平面垂直于向方向.宽度被定义为这个平面上最宽的线块云投影的最大跨度. 这种自动化管道消除了与手动测量相关的主观错误,并确保了叶子卷和扭的强度.为了验证拟议的Plant3R重建和骨架式提取方法的准确性,我们将模型抽取的值与实验中获得的手动测量 (地面真相) 进行了比较.如图4和图7所示,回归分析显示了两种方法在不同生长阶段 (缩,关节和籽粒填充) 之间存在强烈的相关性.具体来说,叶子长度的确定系数 (R2) 超过0.99和平均绝对错误 (MAPE) 低于1.1%,这表明骨架提取算法有效捕获叶子的真正曲线. 对于叶子宽度,该方法也实现了高精度 (R2 > 0.94,RMSE <0.16厘米).所有以上验证结果表明,基于Plant3R模型重建的小麦植物具有高度精确的几何结构,有效地保存了叶子的详细特征.这为随后基于3D模型的精确植物 phenotyping分析提供可靠的数据支持.

<a id="F007"></a>
### Fig. 7. 不同生育阶段模型推导的小麦叶片尺寸与人工测量结果的验证。散点图展示三个生育阶段（分蘖期、拔节期和灌浆期）中，模型提取值与人工测量值在 (a-c) 叶长和 (d-f) 叶宽上的相关性
**Placed near:** p.10 S164  
**Source:** p.11 manual-layout  
**Crop confidence:** high

![Fig. 7](assets/fig7.png)

**Original caption:** Fig. 7. Validation of model-derived wheat leaf dimensions against manual measurements across different growth stages. Scatter plots showing the correlation between model-extracted and manually measured values for (a-c) leaf length and (d-f) leaf width across three growth stages (Tillering, Jointing, and Grain Filling). The high R2 values (>0.94) and low RMSE indicate a strong agreement between the model estimations and ground truth.

**中文图注:** 图 7. 不同生育阶段模型推导的小麦叶片尺寸与人工测量结果的验证。散点图展示三个生育阶段（分蘖期、拔节期和灌浆期）中，模型提取值与人工测量值在 (a-c) 叶长和 (d-f) 叶宽上的相关性。较高的 R2 值（>0.94）和较低的 RMSE 表明模型估计与真实值高度一致。

**Reading note:** 重点查看该图如何支撑相邻正文中的流程、比较、消融或性状提取结果。

<a id="S165"></a>
### 4. Discussion
**Source:** p.10 S165  
**Type:** section  
**Confidence:** high

**Original:** 4. Discussion

**中文:** 4. 讨论

<a id="S166"></a>
**Source:** p.10 S166  
**Type:** body  
**Confidence:** high

**Original:** Comparison of model-extracted parameters and manually measured values of wheat leaf geometric indices at different growth stages.

**中文:** 模表型提取参数和小麦叶几何指数手动测量值的比较在不同增长阶段.

<a id="S167"></a>
**Source:** p.10 S167  
**Type:** body  
**Confidence:** high

**Original:** 1) Adaptability to more complex environments

**中文:** 1) 适应更复杂的环境

<a id="S168"></a>
**Source:** p.10 S168  
**Type:** body  
**Confidence:** high

**Original:** Tillering

**中文:** 缩

<a id="S169"></a>
**Source:** p.10 S169  
**Type:** body  
**Confidence:** high

**Original:** Jointing

**中文:** 关联

<a id="S170"></a>
**Source:** p.10 S170  
**Type:** body  
**Confidence:** high

**Original:** Grain filling

**中文:** 粮食填充

<a id="S171"></a>
**Source:** p.10 S171  
**Type:** body  
**Confidence:** high

**Original:** Maturity

**中文:** 成熟度

<a id="S172"></a>
**Source:** p.10 S172  
**Type:** body  
**Confidence:** high

**Original:** Scale Factor

**中文:** 规模因素

<a id="S173"></a>
**Source:** p.10 S173  
**Type:** body  
**Confidence:** high

**Original:** Plant Height

**中文:** 植物高度

<a id="S174"></a>
**Source:** p.10 S174  
**Type:** body  
**Confidence:** high

**Original:** Maximum Leaf Length

**中文:** 最长的叶子长度

<a id="S175"></a>
**Source:** p.10 S175  
**Type:** body  
**Confidence:** high

**Original:** Maximum Leaf Width

**中文:** 最多叶片宽

<a id="S176"></a>
**Source:** p.10 S176  
**Type:** body  
**Confidence:** high

**Original:** Extracted value Measured value

**中文:** 提取值 测量值

<a id="S177"></a>
**Source:** p.10 S177  
**Type:** body  
**Confidence:** high

**Original:** 12.80

**中文:** 12.80 12.80

<a id="S178"></a>
**Source:** p.10 S178  
**Type:** body  
**Confidence:** high

**Original:** 25.7

**中文:** 25.7 25.7

<a id="S179"></a>
**Source:** p.10 S179  
**Type:** body  
**Confidence:** high

**Original:** 18.08

**中文:** 18.08 18.08

<a id="S180"></a>
**Source:** p.10 S180  
**Type:** body  
**Confidence:** high

**Original:** 1.45

**中文:** 1.45 1.45 1.45

<a id="S181"></a>
**Source:** p.10 S181  
**Type:** body  
**Confidence:** high

**Original:** 25.6

**中文:** 25.6 25.6

<a id="S182"></a>
**Source:** p.10 S182  
**Type:** body  
**Confidence:** high

**Original:** 18.22

**中文:** 18.22 18.22

<a id="S183"></a>
**Source:** p.10 S183  
**Type:** body  
**Confidence:** high

**Original:** 1.40

**中文:** 1.40 1.40

<a id="S184"></a>
**Source:** p.10 S184  
**Type:** body  
**Confidence:** high

**Original:** Extracted value Measured value

**中文:** 提取值 测量值

<a id="S185"></a>
**Source:** p.10 S185  
**Type:** body  
**Confidence:** high

**Original:** 14.30

**中文:** 14.30

<a id="S186"></a>
**Source:** p.10 S186  
**Type:** body  
**Confidence:** high

**Original:** 35.7

**中文:** 35.7

<a id="S187"></a>
**Source:** p.10 S187  
**Type:** body  
**Confidence:** high

**Original:** 17.88

**中文:** 17.88 17.88

<a id="S188"></a>
**Source:** p.10 S188  
**Type:** body  
**Confidence:** high

**Original:** 1.52

**中文:** 1.52

<a id="S189"></a>
**Source:** p.10 S189  
**Type:** body  
**Confidence:** high

**Original:** 35.2

**中文:** 35.2 35.2

<a id="S190"></a>
**Source:** p.10 S190  
**Type:** body  
**Confidence:** high

**Original:** 18.03

**中文:** 18.03 时间

<a id="S191"></a>
**Source:** p.10 S191  
**Type:** body  
**Confidence:** high

**Original:** 1.50

**中文:** 1.50 1.50

<a id="S192"></a>
**Source:** p.10 S192  
**Type:** body  
**Confidence:** high

**Original:** Extracted value Measured value

**中文:** 提取值 测量值

<a id="S193"></a>
**Source:** p.10 S193  
**Type:** body  
**Confidence:** high

**Original:** 13.85

**中文:** 13,85 13.85

<a id="S194"></a>
**Source:** p.10 S194  
**Type:** body  
**Confidence:** high

**Original:** 50.9

**中文:** 50.9 50.9

<a id="S195"></a>
**Source:** p.10 S195  
**Type:** body  
**Confidence:** high

**Original:** 12.04

**中文:** 12.04 12.04

<a id="S196"></a>
**Source:** p.10 S196  
**Type:** body  
**Confidence:** high

**Original:** 1.61

**中文:** 1.61 1.61

<a id="S197"></a>
**Source:** p.10 S197  
**Type:** body  
**Confidence:** high

**Original:** 51.3

**中文:** 51.3 51.3

<a id="S198"></a>
**Source:** p.10 S198  
**Type:** body  
**Confidence:** high

**Original:** 12.20

**中文:** 12.20 12.20

<a id="S199"></a>
**Source:** p.10 S199  
**Type:** body  
**Confidence:** high

**Original:** 1.65

**中文:** 1.65 1.65 1.65

<a id="S200"></a>
**Source:** p.10 S200  
**Type:** body  
**Confidence:** high

**Original:** Extracted value Measured value

**中文:** 提取值 测量值

<a id="S201"></a>
**Source:** p.10 S201  
**Type:** body  
**Confidence:** high

**Original:** 14.75

**中文:** 14.75 14.75

<a id="S202"></a>
**Source:** p.10 S202  
**Type:** body  
**Confidence:** high

**Original:** 49.4

**中文:** 49.4 49.4

<a id="S203"></a>
**Source:** p.10 S203  
**Type:** body  
**Confidence:** high

**Original:** 49.6

**中文:** 49.6 49.6

<a id="S204"></a>
**Source:** p.10 S204  
**Type:** body  
**Confidence:** high

**Original:** 4.1. Plant3R's 3D reconstruction: enhancing accuracy through 3D feature learning and expanding agricultural applications Existing 3D reconstruction methods, whether traditional MVS or the recently emerging NeRF and 3DGS, are highly dependent on the quality of SfM-based camera pose estimation and sparse reconstructed point clouds. They impose high demands on image resolution, overlap, and quality. When using COLMAP for data processing, reconstruction often fails or suffers from missing point clouds for certain key structures due to difficulties in feature matching. Our results clearly indicate that with the same data input, the Plant3R model achieved significant improvements in both feature point matching and sparse point cloud reconstruction. This enhancement is attributed to the Plant3R model's innovative approach of treating 2D image matching as a 3D task and establishing a cross-view 3D feature space through a cross-attention mechanism, thereby enhancing its ability to recognize low-texture and highly re­ petitive structures. Such robust and high-density initialization process­ ing provides a more stable and richer data foundation for subsequent high-fidelity 3DGS rendering, effectively reducing the model's data requirements. Building on this foundation, Plant3R fully leverages the precise pose estimation and dense reconstruction capabilities to achieve higherfidelity plant 3D reconstruction and more accurate phenotypic feature extraction in conjunction with 3D Gaussian Splatting's rendering effi­ ciency. Our experimental results show that the Plant3R model's metrics, such as PSNR and SSIM, surpass those of NeRF and the original 3DGS across key growth stages of wheat, including tillering, jointing, and grain-filling. It is capable of accurately capturing and restoring key phenotypic features like wheat leaf edges and spikes. This high-fidelity reconstruction not only provides a solid foundation for accurate phenotypic parameter extraction but also offers high-resolution 3D data support for a deeper exploration of genotype-phenotype-environment interaction relationships.

**中文:** 4.1.3R的3D重建:通过3D功能学习提高精度和扩大农业应用现有的3D重建方法,无论是传统的MVS还是最近出现的NeRF和3DGS,都高度依赖于基于SfM的相机姿势估计和稀疏重建点云的质量.它们对图像分辨率,重叠和质量提出高要求.使用COLMAP来处理数据时,重建往往会因为功能匹配困难而失败或缺失某些关键结构的点云.我们的结果清楚表明,通过相同的数据输入,Plant3R模型实现了分点匹配和稀疏点云重建方面的显著改善. 这种增强归因于Plant3R模型的创新方法,即将2D图像匹配视为3D任务,并通过交叉注意力机制建立一个跨视图3D功能空间,从而提高了其识别低纹理和高度反感结构的能力.这种强和高密度的初始化过程为后续高 fidelity 3DGS染提供了更稳定和丰富的数据基础,有效降低了模型的数据需求.基于这一基础,Plant3R充分利用精确的估算和密度重建能力,以实现高 fidelity 植物3D重建和更精确的表表型提取功能,并与3D Gaussian Splatting的染效率技术结合. 我们的实验结果显示,Plant3R模型的测量标准,如PSNR和SSIM,在小麦的关键成长阶段,包括工,结合和粮食填充中超越NeRF和原始3DGS的测量标准.它能够精确捕获和恢复小麦叶边缘和尖等关键的现象特征.这种高 fide重建不仅为准确的现象特征抽取提供了坚实的基础,而且还提供了高分辨率的3D数据支持,以更深入地探索基因型-表型-环境关系.

<a id="S205"></a>
### 4.2. Limitations and future potential
**Source:** p.10 S205  
**Type:** section  
**Confidence:** high

**Original:** 4.2. Limitations and future potential

**中文:** 【标题暂译】4.2. Limitations and future potential

<a id="S206"></a>
**Source:** p.10 S206  
**Type:** body  
**Confidence:** high

**Original:** Despite the high reconstruction accuracy and robustness demon­ strated by Plant3R model across different growth stages of wheat, several limitations warrant further investigation and provide clear di­ rections for future research.

**中文:** 尽管Plant3R模型在不同种类的小麦生长阶段表现出了高重建精度和强度,但几个局限性需要进一步调查,并为未来的研究提供了明确的方向.

<a id="S207"></a>
**Source:** p.10 S207  
**Type:** body  
**Confidence:** high

**Original:** This study focused primarily on the reconstruction of potted wheat plants, where environmental control is relatively ideal. When the model is applied in open-field environments, its reconstruction accuracy and robustness may be affected by factors such as complex lighting changes, wind-induced plant motion, and background occlusion. Future research can explore how to integrate 3D reconstruction techniques for dynamic scenes (e.g., methods based on event cameras or dynamic Gaussian fields) with the Plant3R model to adapt to more challenging field environments. 2) Multimodal data fusion and deep phenotypic analysis Currently, the majority of research relies on RGB images for 3D reconstruction. However, certain physiological or pathological infor­ mation of wheat plants (e.g., nitrogen content, water stress) cannot be acquired solely through RGB images. Future work can consider inte­ grating multimodal data, such as by utilizing multispectral or hyper­ spectral imaging technology, to fuse spectral information with 3D structural information. This would enable a more comprehensive and indepth analysis of plant phenotypic features, providing richer data sup­ port for precision agriculture and crop stress diagnosis.

**中文:** 本研究主要集中在麦植物的重建,环境控制相对理想的环境中.当模型在开放场环境中应用时,重建的精度和强度可能会受到诸如复杂的照明变化,风引起的植物运动和背景结等因素的影响.未来的研究可以探讨如何将3D重建技术整合到动态场景 (例如基于事件摄像头或动态高斯场的方法) 与Plant3R模型来适应更具挑战性的场景环境. 2) 多模数据融合和深层表型分析 目前,大多数研究依赖RGB图像进行3D重建. 然而,某些生理或病理信息 (例如含量,水压) 无法仅仅通过RGB图像获得.未来的工作可能会考虑整合多模式数据,例如使用多谱或超谱成像技术,将光谱信息与3D结构信息融合在一起.这将使植物现象特征的更全面和详分割析成为可能,为精确的农业和作物压力诊断提供更丰富的数据支持.

<a id="S208"></a>
**Source:** p.10 S208  
**Type:** body  
**Confidence:** high

**Original:** Note: The term “scale factor” is the conversion ratio that links the reconstructed point cloud's dimensions to the plant's true physical size, and it is directly influenced by the camera's focal length.

**中文:** 注意:术语"尺度因素"是将重建点云的尺寸与工厂的真正物理尺寸联系在一起的转换比率,并且直接受到相机焦距的影响.

<a id="C006"></a>
### Table 4 1) Adaptability to more complex environments Comparison of model-extracted parameters and manually measured values of wheat leaf geometric indices at different growth stages
**Source:** p.10 C006  
**Type:** caption  
**Confidence:** high

**Original:** Table 4 1) Adaptability to more complex environments Comparison of model-extracted parameters and manually measured values of wheat leaf geometric indices at different growth stages. This study focused primarily on the reconstruction of potted wheat Scale Plant Maximum Maximum plants, where environmental control is relatively ideal. When the model Factor Height Leaf Length Leaf Width is applied in open-field environments, its reconstruction accuracy and robustness may be affected by factors such as complex lighting changes, Tillering Extracted 12.80 25.7 18.08 1.45 value wind-induced plant motion, and background occlusion. Future research Measured \ 25.6 18.22 1.40 can explore how to integrate 3D reconstruction techniques for dynamic value scenes (e.g., methods based on event cameras or dynamic Gaussian Jointing Extracted 14.30 35.7 17.88 1.52 fields) with the Plant3R model to adapt to more challenging field value environments. Measured \ 35.2 18.03 1.50 value 2) Multimodal data fusion and deep phenotypic analysis

**中文:** 4 1) 适应更复杂环境模表型提取参数和不同生长阶段测量小麦叶几何指数的手动值的比较.本研究主要集中在化小麦叶的重建上,环境控制相对理想.当模型高度高度叶长度叶宽应用于开放场环境中时,其重建精度和强度可能会受到复杂的照明变化,提取小麦叶的重建12.80 25.7 18.08 1.45 值风诱导植物运动和背景堵等因素的影响. 未来的研究测量 \ 25.6 18.22 1.40 可以探索如何将动态值场景的3D重建技术 (例如基于事件摄像头或动态高斯联合提取的14.30 35.7 17.88 1.52场景) 与Plant3R模型结合起来,以适应更具挑战性的场景值环境.测量 \ 35.2 18.03 1.50值 2) 多模数据融合和深度表型分析


## Page 11

<a id="S209"></a>
**Source:** p.11 S209  
**Type:** body  
**Confidence:** high

**Original:** J. Ma et al.

**中文:** 詹姆斯·马等人

<a id="S210"></a>
**Source:** p.11 S210  
**Type:** body  
**Confidence:** high

**Original:** between model-extracted and manually measured values for (a–c) leaf length and (d–f) leaf width across three growth stages (Tillering, Jointing, and Grain Filling). The high R2 values (>0.94) and low RMSE indicate a strong agreement between the model estimations and ground truth.

**中文:** 在三个增长阶段 (缩,关联和粮食填充) 中 (ac) 叶长度和 (df) 叶宽度的模表型提取和手动测量值之间的关系.高的R2值 (>0.94) 和低的RMSE表明模型估计和地面真相之间存在强烈的一致性.

<a id="S211"></a>
**Source:** p.11 S211  
**Type:** body  
**Confidence:** high

**Original:** 3) Applicability and generalization across plant scales and species

**中文:** 3) 适用性和通用性跨植物规模和物种

<a id="S212"></a>
**Source:** p.11 S212  
**Type:** body  
**Confidence:** high

**Original:** the Plant3R model's performance surpassed other methods, including the original 3DGS, achieving a PSNR over 34 and an SSIM of 0.94. The average relative error between phenotypes extracted from the recon­ structed 3D models and manual measurement results was within 6% which fully verifies the quantitative analysis accuracy of this method in the 3D model accuracy validation. Consequently, the Plant3R model offers significant practical utility, which is reflected not only in its ability to improve the fidelity of plant reconstruction but also in providing a new technical pathway and methodological reference for subsequent high-throughput phenotyping research. Moving forward, with the development of automated imaging systems and multimodal data acquisition methods, the Plant3R model holds promise for further expanding its application potential in crop breeding, smart agriculture, and precision management.

**中文:** 植物3R模型的性能超过了其他方法,包括原始的3DGS,实现了PSNR超过34和SSIM为0.94.从重建的3D模型中提取的表型和手动测量结果之间的平均相对错误在6%内,这充分验证了该方法在3D模型准确验证中的量化分析精确性.因此,Plant3R模型提供了显著的实用性,这不仅体现在其提高植物重建的忠实性的能力,而且在提供了新的技术参考路径和方法来后续高吞吐量表型研究. 随着自动化成像系统和多模式数据采集方法的发展,Plant3R模型有望进一步扩大其在作物种植,智能农业和精密管理领域的应用潜力.

<a id="S213"></a>
**Source:** p.11 S213  
**Type:** body  
**Confidence:** high

**Original:** The current version of Plant3R has been validated mainly on wheat plants ranging from approximately 20-60 cm, covering different key growth stages where the topology and geometry vary greatly and are similar to other cereal species (e.g., rice, barely and oat). These devel­ opmental differences already test the model's robustness to major morphological changes within one species. While additional plant spe­ cies were not included in this study, the dataand species-agnostic design of Plant3R -reconstructing plant geometry from plant 3D geo­ metric priors rather than crop-specific parameters-suggests good trans­ ferability to crops with comparable morphology. For taller and more complex crops like maize or sorghum, the framework is also adaptable, even though the additional vertical layers and self-occlusion pose present challenges for image coverage. To address this, we recommend a “multi-tier cylindrical” acquisition strategy: (1) Stratified Sampling: capturing images at multiple elevation levels to ensure uniform point density. (2) Upward Views: specifically adding upward-looking camera angles at the bottom tier to capture the stem base and the underside of leaves, which are often occluded in standard top-down views. This adaptive approach support scalable application of Plant3R across diverse crop architectures.

**中文:** 目前的Plant3R版本主要在小麦植物上进行了验证,大约20-60厘米的长度,涵盖了不同关键的生长阶段,其中的拓和几何学有很大的差异,并且与其他籽粒物种 (如,和麦) 相似.这些发展差异已经测试了模型对一种物种内重大形态变化的强度.虽然本研究没有包括其他植物物种,但Plant3R的数据和物种无知性设计 - - 从植物3D地质测量先驱而不是特定作物参数重建植物几何学 - 表明对具有相似形态的作物具有良好的可转化性. 对于比较高和复杂的作物,如玉米或,框架也可以适应,尽管额外的垂直层和自我封闭构成了图像覆盖面的挑战.为了解决这一问题,我们建议采用一个多层圆形获取策略: (1) 层面采样:以确保平衡点密度的多层高度捕获图像. (2) 上面视觉:特别增加下层上升的摄像头角度,以捕捉树干底和叶子下面,这些通常被封闭在标准的上下视图中.这种适应方法支持适应性应用 Plant3R 在各种作物架构中.

<a id="S214"></a>
### Author contributions
**Source:** p.11 S214  
**Type:** section  
**Confidence:** high

**Original:** Author contributions

**中文:** 作者贡献

<a id="S215"></a>
**Source:** p.11 S215  
**Type:** body  
**Confidence:** high

**Original:** X.H. proposed conceptualization. J.M. conducted the methodology. J.M. and Y.Z. wrote the original manuscript. J.M., Y.Z., X.H., and L.S. revised the manuscript. L.S., X.H. and Y.Z. applied for funding. L.S. and X.H. provided experiment resources. J.M. and Y.J. performed visuali­ zation. J.M., Y.J., H.Z., and S.D. acquired the data under Y.Z.’s super­ vision. J.M. and Y.J. conducted the data analysis.

**中文:** 简介:H提出了概念化.J.M.进行了方法论.J.M.和Y.Z.写了原稿.J.M.,Y.Z.,X.H.,和L.S.修改了原稿.L.S.,X.H.和Y.Z.申请了资金.L.S.和X.H.提供了实验资源.J.M.和Y.J.进行了视觉化.J.M.,Y.J.,H.Z.,和S.D.在Y.Z.的超级视觉下获得了数据.J.M.和Y.J.进行了数据分析.

<a id="S216"></a>
### 5. Conclusion
**Source:** p.11 S216  
**Type:** section  
**Confidence:** high

**Original:** 5. Conclusion

**中文:** 5. 结论

<a id="S217"></a>
### Funding
**Source:** p.11 S217  
**Type:** section  
**Confidence:** high

**Original:** Funding

**中文:** 基金资助

<a id="S218"></a>
**Source:** p.11 S218  
**Type:** body  
**Confidence:** high

**Original:** This study successfully proposed and validated an innovative 3D reconstruction method for potted wheat plants: the Plant3R model. This model effectively combines the MASt3R model's advantage of providing stable and accurate geometric priors with 3D Gaussian Splatting's highfidelity rendering capability. Our results demonstrated that the Plant3R model significantly outperforms the traditional SfM algorithm in terms of feature extraction, matching, and the density of generated sparse point clouds. This effectively enhances reconstruction accuracy and robustness when dealing with complex, low-texture, and highly repeti­ tive feature scenes like wheat plants. In the rendering results evaluation,

**中文:** 这项研究成功提出并验证了米植物的创新3D重建方法:Plant3R模型.该模型有效地结合了MASTt3R模型提供稳定和精确的几何前景的优势,并与3D高斯人光的高真度染能力.我们的结果表明,Plant3R模型在特征提取,匹配和生成稀有点云密度方面显著优于传统的SfM算法.这有效地提高了复杂,低纹理和高重复性特征场景,如小麦植物处理时重建精度和强度.在米结果评估中,

<a id="S219"></a>
**Source:** p.11 S219  
**Type:** body  
**Confidence:** high

**Original:** This work was supported by the National Natural Science Foundation of China (No. 52425901 and No. 52309058).

**中文:** 这项工作得到了中国国家自然科学基金会的支持 (第52425901号和第52309058号).

<a id="S220"></a>
### Declaration of competing interest
**Source:** p.11 S220  
**Type:** section  
**Confidence:** high

**Original:** Declaration of competing interest

**中文:** 利益冲突声明

<a id="S221"></a>
**Source:** p.11 S221  
**Type:** body  
**Confidence:** high

**Original:** The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

**中文:** 作者表示,他们没有任何已知的竞争性财务利益或个人关系,这些关系似乎会影响本文报告的工作.


## Page 12

<a id="S222"></a>
**Source:** p.12 S222  
**Type:** body  
**Confidence:** high

**Original:** J. Ma et al.

**中文:** 詹姆斯·马等人

<a id="S223"></a>
**Source:** p.12 S223  
**Type:** body  
**Confidence:** high

**Original:** Appendix A. Supplementary data

**中文:** 附件A. 补充数据

<a id="S224"></a>
**Source:** p.12 S224  
**Type:** body  
**Confidence:** high

**Original:** [20] C. Zheng, W. Wen, X. Lu, et al., Phenotypic traits extraction of wheat plants using 3D digitization, Smart Agricult. 4 (2) (2022) 150. [21] N. Fareed, A.K. Das, J.P. Flores, et al., UAS quality control and crop threedimensional characterization framework using multi-temporal lidar data, Remote Sens. 16 (4) (2024) 699. [22] S. Yu, D. Hu, D. Liu, et al., Vision-based optical 3D reconstruction and the application in crop information perception, Laser & Optoelectr. Progr. 61 (4) (2024) 0400004. [23] Y. Yao, Z. Luo, S. Li, et al., Mvsnet: depth inference for unstructured multi-view stereo. Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 767–783. [24] H. Liu, C. Xin, M. Lai, et al., RepC-MVSNet: a reparameterized self-supervised 3D reconstruction algorithm for wheat 3D reconstruction, Agronomy 13 (8) (2023) 1975. [25] W. He, Z. Ye, M. Li, et al., Extraction of soybean plant trait parameters based on SfM-MVS algorithm combined with GRNN, Front. Plant Sci. 14 (2023) 1181322. [26] L. Wang, Y. Miao, Y. Han, et al., Extraction of 3D distribution of potato plant CWSI based on thermal infrared image and binocular stereovision system, Front. Plant Sci. 13 (2023) 1104390. [27] B. Mildenhall, P.P. Srinivasan, M. Tancik, et al., Nerf: representing scenes as neural radiance fields for view synthesis, Commun. ACM 65 (1) (2021) 99–106. [28] K. Hu, W. Ying, Y. Pan, et al., High-fidelity 3D reconstruction of plants using neural radiance fields, Comput. Electron. Agric. 220 (2024) 108848. [29] X. Yang, X. Lu, P. Xie, et al., PanicleNeRF: low-cost, high-precision in-field phenotyping of rice panicles with smartphone, Plant Phenom. 6 (2024) 279. [30] B. Kerbl, G. Kopanas, T. Leimkühler, et al., 3D gaussian splatting for real-time radiance field rendering, ACM Trans. Graph. 42 (4) (2023) 139:1–139:14. [31] J. Li, X. Qi, S.H. Nabaei, et al., A survey on 3D reconstruction techniques in plant phenotyping: From classical methods to neural radiance fields (NeRF), 3D gaussian

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S225"></a>
**Source:** p.12 S225  
**Type:** body  
**Confidence:** high

**Original:** [32] P. Shen, X. Jing, W. Deng, et al., PlantGaussian: exploring 3d gaussian splatting for cross-time, cross-scene, and realistic 3d plant visualization and beyond, Crop J. 13 (2) (2025) 607–618. [33] D. Zhang, J. Gajardo, T. Medic, et al., Wheat3DGS: In-field 3D reconstruction, instance segmentation and phenotyping of wheat heads with gaussian splatting. Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5360–5370. [34] Y. Song, L. Yang, S. Li, et al., Improved YOLOv8 model for phenotype detection of horticultural seedling growth based on digital cousin, Agriculture 15 (1) (2024) 28. [35] V. Leroy, Y. Cabon, J. Revaud, Grounding image matching in 3d with mast3r. European Conference on Computer Vision, Springer Nature Switzerland, Cham, 2024, pp. 71–91. [36] G. Tolias, Y. Avrithis, H. J�egou, To aggregate or not to aggregate: selective match [36] Tolias G, Avrithis Y, J� egou H. To aggregate or not to aggregate: selective match kernels for image search, Proc. IEEE Int. Conf. Comp. Vis. (2013) 1401–1408. [37] S. Wang, V. Leroy, Y. Cabon, et al., Dust3r: geometric 3d vision made easy. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20697–20709. [38] D. Kingma, J. Ba, Adam: A method for stochastic optimization, in: Proceedings of the International Conference on Learning Representations (ICLR), 2015. [39] L.A.G. Stuart, A. Morton, I. Stavness, M.P. Pound, 3DGS-to-PC: Convert a 3D Gaussian splatting scene into a dense point cloud or mesh, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, 2025, pp. 3789–3798. [40] Q.Y. Zhou, J. Park, V. Koltun, Open3D: a Modern Library for 3D Data Processing, 2018 arXiv preprint arXiv:1801.09847. [41] W.E. Lorensen, H.E. Cline, Marching Cubes: a High Resolution 3D Surface Construction algorithm[M]//Seminal Graphics: Pioneering Efforts that Shaped the Field, 1998, pp. 347–353. [42] A. Krizhevsky, I. Sutskever, G.E. Hinton, Imagenet classification with deep convolutional neural networks, Adv. Neural Inf. Process. Syst. (2012) 25.

**中文:** 参考文献条目保留原文，未做逐条翻译。

<a id="S226"></a>
**Source:** p.12 S226  
**Type:** body  
**Confidence:** high

**Original:** Supplementary data to this article can be found online at https://doi. org/10.1016/j.plaphe.2026.100200.

**中文:** 对于这篇文章的补充数据可以在网上找到http://doi.org/10.1016/j.plaphe.2026.100200..

<a id="S227"></a>
### Data availability
**Source:** p.12 S227  
**Type:** section  
**Confidence:** high

**Original:** Data availability

**中文:** 数据可用性

<a id="S228"></a>
**Source:** p.12 S228  
**Type:** body  
**Confidence:** high

**Original:** The data that support this study are available upon reasonable request from the corresponding author. Code is available at Mlynnray/Plant3R.

**中文:** 支持这项研究的数据可根据相应作者合理要求获得.代码可在Mlynnray/Plant3R上找到.

<a id="S229"></a>
### References
**Source:** p.12 S229  
**Type:** section  
**Confidence:** high

**Original:** References

**中文:** 参考文献

<a id="S230"></a>
**Source:** p.12 S230  
**Type:** reference  
**Confidence:** high

**Original:** [1] G. Li, L. An, W. Yang, et al., Integrated biotechnological and AI innovations for crop improvement, Nature 643 (8073) (2025) 925–937. [2] M. Arif, M. Haroon, A.F. Nawaz, et al., Enhancing wheat resilience: biotechnological advances in combating heat stress and environmental challenges, Plant Mol. Biol. 115 (2) (2025) 41. [3] H. Mao, C. Jiang, C. Tang, et al., Wheat adaptation to environmental stresses under climate change: molecular basis and genetic improvement, Mol. Plant 16 (10) (2023) 1564–1589. [4] S. Yu, X. Liu, Q. Tan, et al., Sensors, systems and algorithms of 3D reconstruction for smart agriculture and precision farming: a review, Comput. Electron. Agric. 224 (2024) 109229. [5] J. Qi, F. Gao, Y. Wang, et al., Multiscale phenotyping of grain crops based on threedimensional models: a comprehensive review of trait detection, Comput. Electron. Agric. 237 (2025) 110597. [6] B. Xu, J. Zhang, Z. Tang, et al., Nighttime environment enables robust field-based high-throughput plant phenotyping: a system platform and a case study on rice, Comput. Electron. Agric. 235 (2025) 110337. [7] G.J. Rebetzke, J. Jimenez-Berni, R.A. Fischer, et al., High-throughput phenotyping to enhance the use of crop genetic resources, Plant Sci. 282 (2019) 40–48. [8] M.S. Akhtar, Z. Zafar, R. Nawaz, et al., Unlocking plant secrets: a systematic review of 3D imaging in plant phenotyping techniques, Comput. Electron. Agric. 222 (2024) 109033. [9] R. Zhu, S. Li, Y. Sun, et al., Research advances and prospects of crop 3D reconstruction technology, Smart Agricult. 3 (3) (2021) 94. [10] J.M. Davis, M. Gaillard, M.C. Tross, et al., 3D reconstruction enables highthroughput phenotyping and quantitative genetic analysis of phyllotaxy, Plant Phenom. 7 (1) (2025) 100023. [11] X. Wang, J. Hua, M. Kang, et al., Functional–structural plant model “GreenLab”: a state-of-the-art review, Plant Phenom. 6 (2024) 118. [12] Y. Zhang, Y. Zha, X. Jin, et al., Changes in vertical phenotypic traits of rice (Oryza sativa L.) response to water stress, Front. Plant Sci. 13 (2022) 942110. [13] L. Zhou, G. Wu, Y. Zuo, et al., A comprehensive review of vision-based 3d reconstruction methods, Sensors 24 (7) (2024) 2314. [14] M. Vazquez-Arellano, D. Reiser, D.S. Paraforos, et al., 3-D reconstruction of maize plants using a time-of-flight camera, Comput. Electron. Agric. 145 (2018) 235–247. [15] G. Sansoni, M. Trebeschi, F. Docchio, State-of-the-art and applications of 3D imaging sensors in industry, cultural heritage, medicine, and criminal investigation, Sensors 9 (1) (2009) 568–601. [16] S. Debnath, M. Paul, T. Debnath, Applications of LiDAR in agriculture and future research directions, J. Imag. 9 (3) (2023) 57. [17] H. Moreno, C. Valero, J.M. Bengochea-Guevara, et al., On-ground vineyard reconstruction using a LiDAR-based automated system, Sensors 20 (4) (2020) 1102. [18] N. Pfeifer, C. Briese, Geometrical aspects of airborne laser scanning and terrestrial laser scanning. International archives of photogrammetry, Rem. Sens. Spat. Inform. Sci. 36 (3/W52) (2007) 311–319. [19] M. V� azquez-Arellano, D.S. Paraforos, D. Reiser, et al., Determination of stem position and height of reconstructed maize plants using a time-of-flight camera, Comput. Electron. Agric. 154 (2018) 276–288.

**中文:** 参考文献条目保留原文，未做逐条翻译。


## 阅读提示

- 先读摘要、方法流程图和结果图表，再回到方法细节，可更快抓住论文贡献。
- 对图像重建/分割论文，重点核对数据采集方式、3D 表示、分割或性状提取流程、评价指标和失败案例。
- 公式、表格和复杂多子图页面已经保留原文锚点；若要精校中文，优先处理 `translation_notes.md` 中标为 low/medium 的块。
