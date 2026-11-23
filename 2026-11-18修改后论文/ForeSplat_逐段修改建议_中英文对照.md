# ForeSplat 逐段修改建议｜中英文对照

以同目录的 [ForeSplat.md](ForeSplat.md) 为底稿。以下行号指未修改的原稿；每处先说明怎么处理，再给英文替换段落及中文对照。英文是论文用稿，中文只供核对。`【待补：…】` 是暂留空位，不是拟投稿文字；锁定最终数据后再统一填写。

## 快速定位

| 原稿位置 | 处置 |
|---|---|
| 第 2–10 行：题目、摘要、关键词 | 重写；不沿用旧后端的渲染指标和测量误差 |
| 第 21–43 行：Science4Impact、引言 | 保留引言前两段背景；改写后四个论述单元与贡献列表 |
| 第 47–104 行：§2.1–2.2、Fig. 1、Tables 1–2 | 重写工作流；分别定义前端、单株选型和多株实验 |
| 第 105–225 行：§2.3、Algorithm 1、Fig. 2 | 保留算法、公式与图；只改后端用途及衔接 |
| 第 226–435 行：§2.4–2.9 | 删除旧 2DGS 损失、剪枝与 M2M3-GO；替换为 GGGS、公制标定、TSDF、叶片关联、表型及评价协议 |
| 第 439–533 行：§3.1–3.3、Figs. 3–4、Tables 3–5 | 保留前端实验；限定结论只对应原有二维前景评价 |
| 第 534–656 行：旧 §3.4–3.5、Figs. 5–9、Tables 6–8 | 整段替换；不继承旧实验的指标、样本数和结论 |
| 第 657–685 行：Discussion、Conclusion | 保留前端讨论；重写与旧后端相关的段落 |
| 第 691–693 行：Data Availability | 删除旧版配置清单；按本次实际归档内容填写 |

## 1. 题目、摘要和科学影响声明

### 题目与关键词｜第 2、5–10 行

**不合适：** 题目把前景目标 2DGS 写成方法本体，关键词也指向旧后端。如果九组标签同时含物种、品种或园艺类型，题目用 *diverse potted plants* 比 *multi-species* 稳妥；等物种表及学名确认后，再考虑 *multi-species*。

**EN:** ForeSplat: Vision-Foundation-Model-Guided Metric Gaussian-to-Mesh Reconstruction for Organ-Level Phenotyping of Diverse Potted Plants

**中文：** ForeSplat：视觉基础模型引导的公制高斯到网格重建，用于多种盆栽植物的器官级表型测量

**Keywords / 关键词：** vision foundation models（视觉基础模型）；Gaussian Splatting（高斯泼溅）；TSDF fusion（TSDF 融合）；metric scale recovery（公制尺度恢复）；leaf instance reconstruction（叶片实例重建）；plant phenotyping（植物表型测量）。

### 摘要｜第 4 行整段替换

**不合适：** 原段把 RAP-FSAM3 与旧 2DGS 联合训练捆在一起，并将旧的 `31.49 dB / 0.9711 / 0.0365` 和四项 `R²` 写成整套方法结果。保留原前端指标，但不转用旧后端指标。

**EN:** Multiview RGB imaging provides an accessible route to plant phenotyping, but image masks alone do not define a three-dimensional surface with a physical scale or identify individual organs across views. ForeSplat combines reconstruction-aware prompt adaptation with centimetre-scale surface reconstruction and multiview leaf association. On manually annotated frames from 21 RGB sequences, RAP-FSAM3 achieved an F1-score of 0.9706, an mIoU of 0.9641 and a Boundary-F1 of 0.4643. For three-dimensional reconstruction, COLMAP camera poses and sparse points initialise Geometry-Grounded Gaussian Splatting (GGGS). A 2-cm reference cube anchors the scale, and median depths screened by the foreground mask and rendering alpha are fused using a truncated signed distance function (TSDF). Visible mesh triangles are then labelled across views to separate the plant from the pot and extract individual leaf surfaces. On a separate reconstruction protocol covering 19 potted plants, 【待补：标定与网格结果句】. 【待补：同株同叶人工测量对照结果句】.

**中文：** 多视角 RGB 成像为植物表型测量提供了易于获取的数据，但二维掩膜本身既不能确定具有物理尺度的三维表面，也不能建立跨视角的器官身份。ForeSplat 将面向重建的提示适配、厘米尺度表面重建和多视角叶片关联结合起来。在来自 21 条 RGB 序列的人工标注帧上，RAP-FSAM3 的 F1、mIoU 和边界 F1 分别为 0.9706、0.9641 和 0.4643。三维重建由 COLMAP 相机位姿和稀疏点初始化 GGGS，以边长 2 cm 的参考立方体确定尺度，再用截断符号距离函数（TSDF）融合经前景掩膜及渲染 Alpha 筛选的中值深度。随后对可见网格面片进行跨视角标注，分离植物与花盆并提取独立叶面。在覆盖 19 株盆栽植物的另一套重建协议下，【待补：标定与网格结果句】。【待补：同株同叶人工测量对照结果句】。

### Science4Impact Statement｜第 23 行整段替换

**不合适：** 旧段直接把四项性状写成已验证测量，且没有区分去盆网格和有盆前景。

**EN:** ForeSplat converts multiview RGB images into centimetre-scale meshes of potted plants and exports separately inspectable leaf surfaces. The workflow keeps the link between each exported leaf and its supporting images, making it possible to check where a trait estimate comes from and to flag leaves with incompletely observed margins. 

**中文：** ForeSplat 将多视角 RGB 图像转换为盆栽植物的厘米尺度网格，并导出可单独检查的叶面。流程保留每片导出叶面与提供支持的图像之间的对应关系，因此可以追溯性状估计所用的区域，并标记叶缘观测不完整的个体。

## 2. 引言｜保留第 27、29 行，替换第 31–43 行

**第 31 行（研究缺口）：** 保留“渲染好不等于可测量”，删除“必须通过前景训练约束才能获得测量对象”的推断。

**EN:** A Gaussian representation fitted to multiview RGB images can reproduce appearance without uniquely defining the geometry needed for phenotyping. In potted plants, measurements depend on three additional requirements: a consistent physical scale, separation of the plant from its container, and association of each observed leaf across views. Thin margins and overlapping leaves make the last two requirements particularly difficult. Here, the potted-plant foreground is used as an initial image-space region of interest.

**中文：** 根据多视角 RGB 拟合的高斯表示可以重现图像外观，但不能仅凭外观确定表型测量所需的几何。盆栽植物的测量还需要满足三项条件：统一的物理尺度、植物与容器的分离，以及对各视角所观测叶片的跨视角关联。薄叶缘和叶片重叠尤其增加了后两项工作的难度。本文在图像空间中用盆栽前景限定初始感兴趣区域。

**第 33 行（VFM 的作用）：** 保留提示敏感性，删除“掩膜决定 Gaussian capacity”和基元级控制的叙述。

**EN:** Vision foundation models provide promptable segmentation priors for plant phenotyping [@ref55], but individual prompts can include background vegetation, omit mature structures or truncate leaf margins. RAP-FSAM3 adapts these predictions to the potted-plant foreground task. We use the resulting foreground envelope to screen rendered depth observations, and use separate semantic and instance predictions to label visible mesh surfaces. 

**中文：** 视觉基础模型可为植物表型测量提供由提示生成的分割先验，但单一提示可能纳入绿色背景、遗漏成熟器官或截断叶缘。RAP-FSAM3 将这些分割预测适配到盆栽前景任务。我们利用得到的前景范围筛选渲染深度，再采用独立的语义和实例预测为可见网格表面赋予标签。

**第 35–43 行（方法概述及贡献）：** 去掉“2DGS 训练注入与 M2M3-GO 创新”，改为三点；不要将使用现有 GGGS 或 TSDF 本身写成算法发明。

**EN:** We present ForeSplat, a multiview RGB workflow that links calibrated surface reconstruction to plant and leaf measurements. Its contributions are: (1) RAP-FSAM3, which adapts promptable foreground segmentation to potted plants through semantic-gated candidate selection and structure-balanced refinement; (2) a metric Gaussian-to-mesh workflow that combines COLMAP, native GGGS, cube-based scale recovery and TSDF fusion; and (3) a multiview mesh-labelling procedure that separates plant and pot surfaces and exports individual leaf meshes with links to their supporting images.

**中文：** 我们提出 ForeSplat，一种将经标定的表面重建与整株、叶片测量关联起来的多视角 RGB 工作流。主要贡献为：（1）RAP-FSAM3，通过语义门控的候选选择和结构平衡的提示精炼适配盆栽前景分割；（2）结合 COLMAP、原生 GGGS、立方体尺度恢复和 TSDF 融合，构建公制高斯到网格流程；（3）通过多视角网格标注分离植物和花盆表面，导出可追溯至支持图像的独立叶片网格。

## 3. 方法

### §2.1 工作流及 Fig. 1｜第 49、51 行

**不合适：** 原段和示意图把旧损失、视角软加权、剪枝和 M2M3-GO 画成现行步骤。新版 Fig. 1 应标出「COLMAP 注册 → 前景掩膜与去畸变视角对齐（训练前）→ 渲染深度筛选（训练后）」。去畸变 RGB 与 COLMAP 稀疏点另接 GGGS；不把掩膜画成 GGGS 的训练输入。

**EN:** COLMAP estimates camera poses and a sparse point cloud from multiview RGB images. After registration and before GGGS training, the potted-plant foreground masks are aligned with the corresponding undistorted views. Native GGGS is fitted using unmasked RGB images and the COLMAP sparse points. After scale recovery, the aligned masks screen the rendered GGGS depth before TSDF fusion. Separate SAM3 predictions support reference-cube detection, plant–pot separation and multiview leaf association. Whole-plant and leaf traits are computed from the corresponding centimetre-scale surfaces.

**中文：** COLMAP 根据多视角 RGB 图像估计相机位姿和稀疏点云。在视角注册后、GGGS 训练前，将盆栽前景掩膜对齐到对应的去畸变视角。原生 GGGS 使用未施加掩膜的 RGB 图像和 COLMAP 稀疏点进行拟合。恢复物理尺度后，使用已对齐的掩膜筛选 GGGS 渲染深度，再进行 TSDF 融合。独立的 SAM3 预测用于参考立方体检测、植物与花盆分离及多视角叶片关联。整株和叶片性状分别在对应的厘米尺度表面上计算。

**Fig. 1 EN:** Overview of ForeSplat. The foreground masks are aligned to undistorted COLMAP views before GGGS training and applied to the rendered depth before metric TSDF fusion. GGGS is trained on unmasked RGB images with COLMAP sparse-point initialisation. Mesh labelling separates the plant from the pot and associates leaves across views for trait extraction.

**Fig. 1 中文：** ForeSplat 工作流。前景掩膜在 GGGS 训练前对齐到 COLMAP 去畸变视角，在公制 TSDF 融合前用于筛选渲染深度。GGGS 使用未施加掩膜的 RGB 图像训练，并由 COLMAP 稀疏点初始化。随后通过网格标注分离植株与花盆、关联多视角叶片并提取性状。


### §2.3 RAP-FSAM3｜第 107、111 行及 Fig. 2 后

**不合适：** 算法不用重写，只改输出的用途。第 111 行的 `subsequent 2DGS optimisation` 改为 `downstream surface reconstruction`。区分 RAP-FSAM3 输出与原始帧的对应关系，以及 COLMAP 注册后将掩膜对齐到去畸变视角的步骤；不要把黑底 RGB 写成 GGGS 训练输入。

**EN（替换第 107 行输出句）：** RAP-FSAM3 produces a binary mask, an alpha image and a foreground RGB image aligned to each processed RGB frame. The binary mask supplies the potted-plant depth-screening envelope where an archived mask is available; missing envelopes are replaced by direct SAM3 potted-plant predictions. The masked RGB images are not used for GGGS photometric optimisation.

**中文：** RAP-FSAM3 为每张处理后的 RGB 帧生成对应的二值掩膜、Alpha 图和前景 RGB 图。有归档掩膜时，用二值掩膜限定盆栽深度筛选范围；缺失时，由 SAM3 直接预测盆栽前景。掩膜处理后的 RGB 图像不用于 GGGS 光度优化。

### §2.4 COLMAP 与 GGGS｜替换第 226–291 行

**删除：** 前景轨迹筛选公式、Foreground RGB Loss、Alpha/背景损失、损失热身与 2DGS 正则公式。保留 COLMAP 引用；给 GGGS 原论文另加 BibTeX，别引用 `ref28` 来说明 GGGS 的训练目标。

**EN:** Camera intrinsics, poses and sparse points were obtained from COLMAP structure from motion [@ref16]. After COLMAP registration and before GGGS training, we mapped the potted-plant foreground masks to the image coordinates of the corresponding undistorted RGB views and stored them separately. The masks were later used to screen GGGS-rendered depth before TSDF fusion.

**中文：** 相机内参、位姿和稀疏点来自 COLMAP 运动恢复结构。COLMAP 完成视角注册后、GGGS 训练前，我们将盆栽前景掩膜映射到对应去畸变 RGB 视角的图像坐标中，并单独保存。这些掩膜随后用于在 TSDF 融合前筛选 GGGS 渲染的深度。

**EN:** Native GGGS (Zhang et al., *Grounding Gaussian Splatting Geometry with Stochastic Solids*) was trained on the undistorted RGB images without masking and initialised with the COLMAP sparse points without foreground filtering. The run used 30,000 iterations, spherical harmonic degree 3, spherical Gaussian degree 0 and seed 0. The native photometric, depth–normal and multiview regularisation terms were retained. For meshing, we use the median camera-axis depth returned by the GGGS renderer.

**中文：** 原生 GGGS 使用未施加掩膜的去畸变 RGB 图像训练，以未按前景掩膜筛选的 COLMAP 稀疏点初始化。训练采用 30,000 次迭代、球谐阶数 3、球面高斯阶数 0、随机种子 0，保留 GGGS 原生的光度、深度–法线及多视角正则项网格提取使用 GGGS 渲染器给出的相机轴向中值深度。

### §2.5 公制尺度恢复｜替换第 292–306 行

**删除：** View-Quality-Aware Soft Loss Weighting。旧 §2.8 的“以花盆直径恢复尺度”一并删除。写清缩放发生在公制深度与网格生成前，而非对表型数值二次乘比例。

**EN:** A blue reference cube with a known edge length of 2 cm provides the scale. We fit the projections of a common three-dimensional cube to silhouettes in registered views. If the fitted edge length is $l_c$ in reconstruction units, the conversion is $s = 2\,\mathrm{cm}/l_c$. We multiply Gaussian centres, length-valued filter parameters and camera translations by $s$, and add $\log s$ to the stored Gaussian log-scales. Camera rotations and intrinsics remain unchanged. Metric mesh coordinates are subsequently used directly for trait extraction. 【待补：最终标定来源分组与例外处理】

**中文：** 边长为 2 cm 的蓝色参考立方体用于确定物理尺度。在注册视角中，用同一个三维立方体的投影拟合各视角轮廓。若重建坐标中拟合边长为 $l_c$，换算系数为 $s = 2\,\mathrm{cm}/l_c$。高斯中心、以长度为单位的滤波参数和相机平移乘以 $s$，高斯对数尺度加上 $\log s$；相机旋转和内参不变。随后直接在公制网格坐标中提取性状。【待补：最终标定来源分组与例外处理】

### §2.6 TSDF 网格重建｜替换第 307–326 行，删除第 419–425 行旧权重

**不合适：** 实现是在融合前筛有效深度，不是给融合体素施加 `visibility × normal × mask` 三权重。法线有效性若用于稳健范围采样，不等于法线加权融合。保留原稿经典 TSDF/Marching Cubes 引用（`ref45`、`ref46`），将 2DGS 对应代码仅列作实现来源。

**EN:** From each registered camera we render GGGS median depth and alpha. A depth is eligible for fusion only if it is finite, positive, has alpha at least 0.5 and lies within the potted-plant foreground mask. Only training views are integrated. Voxel length is set to $D_{\mathrm{scene}}/512$, where $D_{\mathrm{scene}}$ is the diagonal of the robust scene bounding box derived from the 0.5th and 99.5th percentiles along each coordinate axis. The TSDF truncation distance is five voxel lengths. We use standard bounded TSDF integration and extract a triangle mesh; post-processing removes connected components with fewer than 50 triangles, without smoothing or filling holes.

**中文：** 从注册相机渲染 GGGS 中值深度与 Alpha。仅当深度有限、为正、Alpha 不低于 0.5 且位于盆栽前景掩膜内时，才可参与融合。TSDF 只融合训练视角。体素边长为 $D_{\mathrm{scene}}/512$；$D_{\mathrm{scene}}$ 是根据各坐标轴第 0.5 和第 99.5 百分位计算的稳健场景包围盒对角线长度。TSDF 截断距离为 5 个体素边长。采用标准有界 TSDF 融合提取三角网格；后处理只删除少于 50 个三角面的连通分量，不执行平滑或补洞。


### §2.7 Plant-only 与叶片实例｜替换第 327–416 行

**删除：** 整段 M2M3-GO 及其基元效用、增密和剪枝公式。新的处理对象是 TSDF 网格三角面，不是高斯基元。两个过程分段，便于对应 Fig. 6。

**EN:** We assign semantic labels to the mesh by projecting SAM3 plant and pot masks through the registered cameras. Votes are restricted to the first intersected surface along each ray and screened for consistency with GGGS depth. A triangle needs at least three view votes and a class vote share of at least 0.6 to receive a semantic label; unsupported or inconsistent triangles are retained as unknown rather than reassigned to the plant. The plant-only mesh consists of plant-labelled faces.

**中文：** 利用注册相机将 SAM3 的植物和花盆掩膜投影到网格。每条视线只对首次相交的表面投票，并检查该表面与 GGGS 深度是否一致。赋予语义标签的三角面须至少获得三个视角的投票，且同一类别票数占比不低于 0.6；支持不足或标签冲突的面保持未知，不重新归入植物。去盆植物网格由标记为植物的三角面构成。

**EN:** SAM3 leaf proposals from different views are associated through their shared support on visible plant triangles. A candidate instance must be supported by at least three views. A plant triangle is assigned to an instance if at least two distinct views support the assignment and the instance receives at least 0.6 of that triangle's leaf votes. The assigned faces are exported as a centimetre-scale leaf mesh with a `Leaf ID`, a reference image and source-face indices. 

**中文：** SAM3 在各视角生成叶片候选，通过候选在可见植物三角面上的共同支持区域建立跨视角关联。每个候选实例至少需要三个视角支持。三角面归入某一实例须得到至少两个不同视角的支持，且该实例获得的叶片票数占比不低于 0.6。将已分配的面片导出为厘米尺度叶片网格，并保留 `Leaf ID`、参考图像和原始面片索引。

### §2.8 性状提取｜替换第 417–426 行旧段

**不合适：** 原文把“尺度恢复、性状计算、与人工测量吻合”压成一个结论句。新稿应分别定义测量对象、测量方法和质量标记，并明确叶长采用叶基到叶尖的三维直线距离，而非沿叶面测得的曲线长度。

**EN:** Plant height is estimated as the maximum projection of the plant-only mesh above the estimated base along the estimated vertical axis. Canopy width is the maximum span of the horizontally projected plant surface. Leaf length is measured as the three-dimensional Euclidean distance between the identified leaf base and tip on the mesh. Leaf width is the maximum accepted transverse path across the observed surface. Uncertain landmarks, incomplete surfaces and rejected transverse paths are flagged.

**中文：** 株高定义为去盆植物网格顶点沿估计竖直方向、相对于估计基部的最大投影高度。冠幅定义为植物表面水平投影的最大跨度。叶长定义为叶片网格上定位的叶基与叶尖之间的三维欧氏距离，即两点之间的直线距离。在观测叶面上的有效横向路径中取最大值作为叶宽。对不确定的标志点、不完整的表面及未通过检查的横向路径加以标记。

### §2.9 评价指标｜替换第 427–435 行

**不合适：** 保留二维前端指标，删除旧训练消融使用的 `outside < 0.05` 与 `leakage < 0.10` 判据。选型、流程完成与性状误差分别报告；误差协议可先写，结果留空。

**EN:** The foreground-prior evaluation retains the original F1, mIoU, HD95, Boundary-F1 and leakage-energy metrics. In the single-plant development experiment, held-out rendering quality and geometric proxies are reported separately; projected silhouette agreement and agreement with the renderer's depth are not equivalent to accuracy against a reference 3D scan. Trait agreement is assessed using references from the same specimen and capture session and, for leaf traits, the same leaf. The planned statistics are the eligible pair count, MAE, RMSE and signed bias for each trait. Bias is defined as the mesh-derived value minus the manual reference. The manual measurement convention for curved leaves is specified before agreement is calculated.

**中文：** 前景先验评价沿用 F1、mIoU、HD95、边界 F1 和泄漏能量。单株开发实验将留出视角渲染质量与几何代理指标分开报告；投影轮廓的一致性，以及与渲染深度的一致性，均不等同于根据三维参考扫描评价的准确度。表型一致性评价只采用同株、同采集批次的人工参考；叶片性状还须对应同一片叶。逐项报告有效配对数、MAE、RMSE 和带符号偏差。偏差定义为网格测量值减去人工参考值，计算前应明确弯曲叶片的人工量测口径。

## 4. 结果

### §3.1–3.3 前端评价｜第 439–533 行

**保留：** 原段落、Figs. 3–4 和 Tables 3–5。第 441 行的 `best overall` 可改成具体指标，避免让读者误以为所有指标均领先。在 §3.3 末段加一句明确证据边界。

**EN:** These results describe RAP-FSAM3 performance under the original 21-sequence foreground protocol. 

**中文：** 这些结果描述的是原有 21 序列协议下 RAP-FSAM3 的前景分割表现。

### §3.4 单株后端选型｜替换第 534–593 行；旧 Table 6 移出主文

**EN:** On the KongQueZhuYu development plant, MCMC-3DGS achieved higher foreground PSNR than GGGS on the 27 held-out views (25.42 versus 24.96 dB). GGGS nevertheless provided the strongest depth self-consistency in this comparison. In the subsequent mesh comparison, bounded TSDF using GGGS depth reached a silhouette IoU of 0.900, compared with 0.878 for 2DGS with bounded TSDF; 2DGS had slightly higher Boundary-F1 at a two-pixel tolerance. We selected GGGS with bounded TSDF for the multi-plant workflow on the basis of these geometric proxies, not because it delivered the highest image PSNR or established absolute surface accuracy.

**中文：** 在单株孔雀竹芋开发实验的 27 个留出视角上，MCMC-3DGS 的前景 PSNR 高于 GGGS（25.42 对 24.96 dB），但本次比较中 GGGS 的深度自一致性最高。后续网格比较中，GGGS 深度的有界 TSDF 网格轮廓 IoU 为 0.900，2DGS 的有界 TSDF 为 0.878；以 2 像素容差计算时，2DGS 的边界 F1 略高。多株流程因此选用 GGGS 与有界 TSDF，依据是本次几何代理指标，而不是最高图像 PSNR，也不是已验证的绝对表面精度。

### §3.5 多株公制网格结果｜替换旧第 594–627 行

**EN:** The metric reconstruction workflow was applied to 19 potted plants spanning nine recorded plant types. 【待补：最终标定结果、有效分母与尺度质控】. Figure 8 shows representative RGB images, Gaussian renderings, TSDF meshes and plant-only surfaces. Workflow completion, semantic separation and measurement accuracy are reported as distinct outcomes.

**中文：** 将公制重建流程应用于涵盖九类记录植物类型的 19 株盆栽植物。【待补：最终标定结果、有效分母与尺度质控】。Fig. 8 展示具代表性的 RGB 图像、高斯渲染、TSDF 网格和去盆植物表面。流程是否完成、语义分离与测量准确性分别报告。

### §3.6 多视角单叶实例｜在新 §3.5 后新增

**EN:** Multiview association yielded 798 leaf-surface candidates across 19 plants. For comparison with manual measurements, we selected three leaves per plant (57 leaves in total) and verified the correspondence between each physical leaf and its reconstructed instance.

**中文：** 多视角关联在 19 株植物上共生成 798 个叶面候选。为与人工测量进行对照，每株选取 3 片叶，共 57 片，并逐一确认实物叶片与重建实例的对应关系。

### §3.7 性状测量一致性｜替换旧第 628–656 行，含 Table 8、Fig. 9

**现在留空：** 删除旧植株/叶片样本数、旧表格及旧散点图，不预写“良好一致性”。后续仅回填株高、冠幅、叶长和叶宽的人工对照结果。

**EN:** 【待补：同株同叶参考测量；逐性状有效配对数、MAE、RMSE、bias 与结果解释；Table 8 和 Fig. 9】

**中文：** 【待补：同株同叶参考测量；逐性状有效配对数、MAE、RMSE、偏差和结果解释；Table 8 与 Fig. 9】

## 5. 讨论与结论

### Discussion｜第 659–681 行

**第 659–662 行：** 保留关于 SAM3 提示偏差的讨论；末句只落到“改善二维前景先验”，不延伸为已验证的 GGGS 性能增益。

**第 663–672 行：** 两节合并为 `Foreground Priors in Metric Surface Reconstruction`。删除“训练注入才必要”以及 M2M3-GO 对表型带来收益的陈述。

**EN:** RAP-FSAM3 affects the observations supplied to the depth-fusion stage rather than the parameters optimised by GGGS. A tighter envelope can exclude unwanted depth samples but can also remove thin margins when the mask is too conservative. The single-plant comparison shows why rendering PSNR and mesh-oriented criteria should be reported separately.

**中文：** RAP-FSAM3 影响输入深度融合阶段的观测，不改变 GGGS 所优化的参数。较紧的前景范围可以排除无关深度，但过于保守时也可能剔除薄叶缘。单株对比说明应分开报告渲染 PSNR 和面向网格的指标。

**第 673–678 行：** 用质量边界替换“R² 较低证明叶宽由叶缘误差主导”的旧结论。

**EN:** Metric scale does not correct local shape errors. Occlusion can obscure the leaf base or tip, missing margins can bias width estimates, and surface fusion can merge overlapping leaves. An exported instance may therefore have a reproducible identifier without being complete enough for anatomical measurement. 【待补：由最终人工对照数据支持的性状差异和误差解释】

**中文：** 公制尺度无法纠正局部形状误差。遮挡可能使叶基或叶尖难以定位，叶缘缺失可能使叶宽估计产生偏差，表面融合也可能将重叠叶片粘连。因而，导出的实例即使具备可追溯的编号，其表面仍可能不满足完整器官测量要求。【待补：由最终人工对照数据支持的性状差异和误差解释】

**第 679–681 行：** 保留已有文献引用，但删除 `ForeSplat does not extract … after reconstruction`。新版恰恰在重建后提取表面与器官，区别应落在网格标签、真实尺度与可追溯查看，不做无依据的优越性比较。

**EN:** Unlike a point-cloud-only output, ForeSplat retains source-triangle indices and supporting images for each exported leaf surface. This design allows leaf-length and leaf-width estimates to be checked against the mesh and the views that supported the instance. It complements prior work on point-based plant extraction and semantic lifting [@ref34; @ref35; @ref36; @ref37] .

**中文：** 与仅输出点云不同，ForeSplat 为每个导出叶面保存原始三角面索引和支持图像，使叶长和叶宽估计可以对照重建网格以及支持该实例的视角进行核查。这是对点云植物提取和语义提升方法的补充。

### Conclusion｜第 685 行整段替换

**EN:** ForeSplat links vision-foundation-model foreground priors to centimetre-scale surface reconstruction and leaf-level analysis. RAP-FSAM3 addresses prompt sensitivity in the original foreground benchmark. Native GGGS supplies rendered depth for TSDF fusion, and multiview mesh labelling separates plant and pot surfaces and exports traceable leaf instances. 【待补：最终标定与流程结果句】. 【待补：经人工对照支持的表型结论及适用范围】.

**中文：** ForeSplat 将视觉基础模型生成的前景先验与厘米尺度表面重建及叶片级分析相连接。RAP-FSAM3 针对原有前景基准中的提示敏感性问题。原生 GGGS 提供 TSDF 融合所需的渲染深度，多视角网格标注则分离植物与花盆表面，并导出可追溯的单叶实例。【待补：最终标定与流程结果句】。【待补：经人工对照支持的表型结论及适用范围】。

### Data Availability｜第 693 行替换

**EN:** 【待补：repository URL and version; released RGB data and annotations; split manifests; configuration and outputs for scale recovery, meshing and leaf association; and the availability of verified measurement references】.

**中文：** 【待补：仓库 URL 与版本，实际公开的 RGB 和标注、数据划分清单、标定与网格和叶片关联的配置及输出，以及经核验的参考测量数据获取方式】。

## 6. 图表换稿清单

图号先沿用原稿，排版时再统一调整交叉引用。不要给旧图换一段新图注后直接继续使用。

| 原图表 | 替换标题（EN / 中文） | 处理 |
|---|---|---|
| Fig. 1 | Workflow of ForeSplat / ForeSplat 总体流程 | 重画；标出 COLMAP 注册后、GGGS 训练前的掩膜对齐，掩膜接到训练后的深度筛选，不接 GGGS 训练 |
| Figs. 2–4、Tables 3–5 | RAP-FSAM3 foreground evaluation / RAP-FSAM3 前景评价 | 保留；限定为原前端协议 |
| Fig. 5 | Cube-based scale recovery and metric TSDF fusion / 蓝块尺度恢复与公制 TSDF 融合 | 用新版蓝块投影、深度筛选与网格结果替换 |
| Fig. 6 | Multiview leaf association on the plant mesh / 植株网格上的多视角叶片关联 | 用 RGB、SAM3 叶掩膜、共享三角面和导出的单叶网格替换 |
| Fig. 7、原 Table 6 | Single-plant Gaussian and mesh selection / 单株高斯模型与网格选型 | 替换旧训练消融；五种训练配置与七个网格分支分开列，以免被读成同一实验 |
| Fig. 8、原 Table 7 | Metric reconstruction across potted plants / 多类盆栽植物的公制重建 | 换成 19 株对应的新结果，标定统计栏 `【待补】` |
| Fig. 9、原 Table 8 | Agreement with manual trait measurements / 与人工表型测量的一致性 | `【待补】`；不放旧图或旧数值 |

## 本地核对依据

- [论文原稿](ForeSplat.md) 与用户提供的修改建议：定位旧段落和前端原始口径。
- [单株选型汇总](../experiments/COLMAP_Gaussian_Mesh_Selection/summary/final_selection.md)：用于 §3.4，避免混入历史 VGGT 或旧 2DGS 实验。
- [多株运行配置](../experiments/ForeSplat_MultiSpecies_Phenotyping/summary/run_configuration.json)、[多株结果清单](../experiments/ForeSplat_MultiSpecies_Phenotyping/summary/all19_results.csv)、[叶片匹配表](../experiments/ForeSplat_MultiSpecies_Phenotyping/summary/three_leaf_matches.csv) 与[误差汇总表](../experiments/ForeSplat_MultiSpecies_Phenotyping/summary/three_leaf_error_summary.csv)：用于划分实验口径和标记回填点。
- 跑批实现：项目根目录 `run_colmap_phenotyping_all19.py` 指向 `phenotyping/` 下的训练、标定、深度融合、面片标注和性状脚本。GGGS 方法名和作者按本地上游 README 核对；未把附件中的 GitHub 链接当作已核验的新版存档。
- 掩膜时序核查：`phenotyping/batch_prepare.py:164` 对齐并独立保存掩膜，`phenotyping/batch_train.py:22` 核对稀疏点初始化并用未加掩膜的 RGB 训练，`phenotyping/batch_mesh.py:37` 在融合前用掩膜筛选深度。
