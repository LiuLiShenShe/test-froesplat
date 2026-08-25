# ForeSplat: Foreground-Aware 2D Gaussian Splatting for Low-Cost Plant Phenotyping

## Research Highlights

- ForeSplat aligns the 2DGS reconstruction objective with plant-trait measurement.
- Foreground RGB supervision reduced the leakage energy ratio from 1.2201 to 0.0190.
- The workflow processed 20 RGB multiview sequences from two acquisition settings.
- Virtual trait measurements closely agreed with manual measurements, with \(R^2\) up to 0.9879.
- Soft view weighting reduced the number of Gaussian primitives while preserving geometric coverage.

## Abstract

Rapid and non-destructive measurement of plant structural traits is central to intelligent breeding, controlled-environment cultivation and precision crop management. Manual phenotyping is labour-intensive, whereas two-dimensional imaging cannot fully resolve overlapping leaves, occluded organs or canopy geometry. Multiview three-dimensional reconstruction offers a low-cost alternative, but existing neural radiance field and Gaussian Splatting pipelines usually optimise the appearance of the entire scene. As a result, pots, substrate, supports and background structures can enter the reconstructed model. This mismatch constrains subsequent mesh extraction and virtual trait measurement. Here we present ForeSplat, a foreground-aware 2D Gaussian Splatting workflow that moves plant masks from a post-processing cue into the reconstruction objective. FSAM3 combines frequency-domain frame-quality filtering, text-prompted plant segmentation and foreground refinement to generate multiview priors. These priors are then used for foreground initialisation, RGB supervision, opacity constraints, view weighting, Gaussian pruning, TSDF meshing and scale recovery. Across 20 RGB sequences and 21 potted plants, FSAM3 achieved an F1-score of 98.3%, an mIoU of 97.9% and an HD95 of 41.4 px. Ablation experiments showed that foreground RGB supervision reduced the outside-mask non-black ratio from 0.9908 to 0.0294 and the leakage energy ratio from 1.2201 to 0.0190. ForeSplat achieved PSNR = 31.09 dB, SSIM = 0.9711 and LPIPS = 0.0365, while reducing training time and meshing time by 60.94% and 65.17%, respectively, relative to standard 2DGS. Virtual measurements of plant height, canopy width, leaf length and leaf width closely agreed with manual measurements, with \(R^2\) values of 0.9878, 0.9879, 0.9738 and 0.8999. These results indicate that ordinary RGB imaging can support reusable, low-cost, plant-level 3D phenotyping under indoor and semi-controlled backgrounds.

**Keywords:** phenotyping; reconstruction; segmentation; Gaussian Splatting; RGB imaging; mesh; traits

---

## 1. Introduction

Plant phenotyping provides a technical bridge between genotype, environmental response and agronomic performance. Structural traits such as plant height, canopy width, leaf length and leaf width are widely used in breeding selection, cultivation management and growth assessment. However, manual measurement usually requires contact-based operation, has low throughput and is sensitive to operator experience. Image-based high-throughput phenotyping systems have therefore become an important direction in intelligent agriculture [1-6]. Compared with two-dimensional images, three-dimensional representations can record organ position, occlusion relationships and canopy volume. They are better suited to overlapping leaves, complex canopies and non-planar structures [7-12]. Yet plants are not regular, rigid or richly textured engineering objects. Thin leaves, weak texture, repeated texture, local occlusion, mixed flowers and leaves, and background colours similar to the target all make low-cost 3D reconstruction and trait extraction difficult.

Three-dimensional plant phenotyping has expanded from traditional structure from motion, multiview stereo and depth sensing to neural rendering and explicit Gaussian representations. LiDAR, structured light and depth cameras can provide accurate point clouds, but their cost, calibration complexity and deployment requirements limit broad use in horticultural facilities and large-scale breeding [13-15]. Consumer RGB cameras are flexible and inexpensive, but SfM/MVS is highly sensitive to view coverage, image sharpness and leaf texture. Thin leaves and occluded regions often produce holes, noise or blurred boundaries [16-19]. NeRF improves plant reconstruction through a continuous volumetric radiance field and has been used for field and indoor plant geometry assessment [20-26]. 3D Gaussian Splatting (3DGS) further represents scenes with explicit Gaussian primitives, with advantages in rendering efficiency and editability [27-29]. Recent studies, including Plant3R, PlantGaussian, Cotton3DGaussians and object-centric 3DGS, show that Gaussian representations are entering plant-structure reconstruction and trait-analysis workflows [30-33]. These advances indicate that neural rendering is becoming an important tool for low-cost 3D plant phenotyping. They also leave a practical agricultural question unresolved: is the reconstructed 3D object the plant object that needs to be measured?

This question is especially important in potted-plant and controlled-environment scenes. Standard NeRF, 3DGS and 2D Gaussian Splatting (2DGS) usually optimise the visual reconstruction of the full image. The model therefore learns plants together with pots, substrate, tables, supports and background structures. Full-scene reconstruction is a reasonable objective for novel-view synthesis. For plant height, canopy width and leaf-size measurement, however, non-plant geometry can enter mesh extraction and bounding-range computation. LCR-GS extracted individual greenhouse muskmelon plants from 3DGS scenes, showing that downstream trait extraction requires clean and analysable plant representations [34]. IPENS lifted SAM2-generated two-dimensional masks into a NeRF space to extract organ-level point clouds from rice and wheat, indicating that promptable segmentation can reduce annotation burden when combined with radiance fields [35]. Gaussian Grouping and SAGA further show that semantics from two-dimensional foundation models can be distilled or lifted into Gaussian spaces, although these methods mainly target general scene segmentation and editing rather than trait measurement itself [36,37]. Together, these studies highlight the need to move from scene-level reconstruction to plant-level representation. In most pipelines, however, object separation still occurs after reconstruction. Once the background has acquired stable Gaussian capacity during training, post hoc mask pruning may not fully remove its influence on plant meshes and virtual measurements.

The central premise of this study is that masks in agricultural phenotyping should not act only as post-processing filters. Instead, they should help define the 3D optimisation problem. We propose ForeSplat, a foreground-aware 2DGS phenotyping workflow for multiview potted plants. Unlike ordinary scene reconstruction, ForeSplat binds initialisation points, the RGB supervision domain, opacity constraints, view-quality weights and Gaussian cleaning to the plant foreground defined by masks. Thus, the dominant photometric gradients during training originate from plant pixels, and model capacity is preferentially allocated to the measured plant rather than to pots, tables or backgrounds. The planar Gaussian primitives used in 2DGS are well suited to leaf-like thin surfaces [28]. ForeSplat further directs this surface representation towards a foreground-only measurement object.

To achieve this objective, we first design FSAM3 as a reconstruction-oriented plant-foreground prior generator. The aim is not to claim universal optimality for plant segmentation. Instead, FFT-based frame-quality filtering, SAM3 text-prompted segmentation and PCA-based main-foreground refinement are combined to generate masks that are aligned across views, relatively stable at boundaries and suitable for 2DGS training. ForeSplat then filters COLMAP sparse tracks through multiview mask consistency, introduces foreground RGB supervision, alpha mask loss and background opacity loss into 2DGS, and retains lower-quality views through view-quality-aware soft loss weighting when those views may still provide geometric coverage. After training, mask-guided multi-cue Gaussian pruning and TSDF meshing convert the foreground-only Gaussian representation into a measurable plant mesh.

This study aims to establish a low-cost, reusable, RGB-based 3D workflow for indoor and semi-controlled complex backgrounds, with plant phenotyping as the reconstruction target. It is not intended as an unrestricted general plant reconstruction model. The contributions are as follows:

1. We propose ForeSplat, which reformulates standard 2DGS from full-scene visual reconstruction into mask-defined plant-object reconstruction. This aligns the reconstruction target with agricultural phenotyping objects such as plant height, canopy width and leaf size.

2. We propose FSAM3, a plant-foreground prior generation workflow that combines FFT-based frame-quality filtering, SAM3 text-prompted segmentation and PCA-based main-foreground refinement. It provides boundary-stable, file-aligned and cross-species foreground constraints for multiview 2DGS.

3. We construct an end-to-end phenotyping workflow from ordinary RGB sequences to foreground-only plant meshes through foreground trajectory initialisation, foreground RGB supervision, alpha and background opacity constraints, soft view weighting and mask-guided Gaussian pruning.

4. We validate ForeSplat on 20 multiview sequences and 21 plants, and analyse its feasibility and boundary conditions through external baselines, controlled ablations, compactness assessment, mesh-structure evaluation and manual-to-virtual measurement comparison.

---

## 2. Materials and Methods

### 2.1 Study Design and Overall Workflow

ForeSplat targets a specific agricultural phenotyping task: generating foreground-only plant meshes from ordinary multiview RGB images for plant height, canopy width and leaf-size measurement. The workflow follows the principle of defining the measured plant object before 3D reconstruction. First, FSAM3 performs quality filtering, plant-foreground segmentation and main-component refinement on raw frames, producing masks aligned with training views. Second, COLMAP estimates camera poses and sparse point tracks, and multiview mask consistency filters foreground initialisation points. Third, plant-aware 2DGS restricts RGB loss to the plant foreground and constrains the opacity field with alpha mask loss and background opacity loss, aligning the reconstructed object with the phenotyping target. Fourth, view-quality-aware soft loss weighting retains all views and modulates only their training contributions. Fifth, mask-guided multi-cue Gaussian pruning and TSDF mesh extraction produce compact and measurable plant meshes. The overall workflow and the input-output relationships among modules are shown in Fig. 1.

**Fig. 1 | Overview of the ForeSplat workflow.** Raw multiview images pass through FSAM3, COLMAP, foreground-object 2DGS optimisation, soft view weighting, mask-guided Gaussian pruning, TSDF meshing and phenotypic measurement. The workflow outputs foreground-only plant meshes and virtual trait values.

### 2.2 Dataset, Acquisition Settings and Sample Use

To cover common structural variation in potted-plant phenotyping, the dataset included 20 multiview RGB sequences. The sequences covered broad leaves, low canopies, overlapping leaves, compact canopies, mixed flowers and leaves, thick leaves, fine texture and dense occlusion. Images were acquired with an iPhone 14 Pro Max at a video resolution of 1080 x 1920 and a frame rate of 60 fps. Two acquisition settings were used: fixed-device-assisted capture and handheld circular capture in complex indoor backgrounds. This design allowed evaluation under both semi-controlled acquisition and more practical indoor deployment conditions. All 20 sequences were used for full-workflow validation. Phenotypic statistics were analysed at the plant level for 21 plants. Leaf length and leaf width were measured on three representative leaves per plant, giving \(n = 63\) for leaf traits. Dataset coverage and end-to-end workflow execution are summarised in Table 1.

**Table 1 | Dataset coverage and end-to-end workflow execution.** The table summarises the data scale, effective frame counts, SfM-registered views and workflow success rate for the two acquisition settings.

| Setting | Species/cultivar labels | Samples | Raw frames | Effective frames | SfM-registered views | Successful samples | Success rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed-device-assisted capture | 8 | 10 | 2502 | 2104 | 2040 | 10 | 100% |
| Complex-background capture | 7 | 10 | 2500 | 2113 | 2089 | 10 | 100% |

The samples covered broad leaves, low canopies, overlapping leaves, compact canopies, mixed flowers and leaves, thick leaves, fine texture and dense occlusion. They were used to evaluate the applicability of ForeSplat across canopy forms and background complexity.

### 2.3 FSAM3: Frequency-Spatial Plant-Foreground Priors

FSAM3 aims to provide stable plant-foreground priors for 3D phenotyping reconstruction, rather than to serve as an independent general segmentation model. Its input is the raw multiview RGB frames of each plant. Its outputs are binary masks, RGBA alpha images and foreground-only RGB images that are aligned one-to-one with the training images. FSAM3 consists of three steps: FFT frame filtering, SAM3 text-prompted segmentation and PCA main-foreground refinement.

#### 2.3.1 FFT-Based Frame-Quality Filtering

Blurred, defocused and weakly textured frames can affect SfM pose estimation, SAM3 segmentation boundaries and subsequent 2DGS optimisation. We computed the two-dimensional FFT magnitude spectrum for each frame and defined the high-frequency energy ratio as

\[
Q_{\text{FFT}}(I)=
\frac{\sum_{(u,v)\in H}|F(u,v)|}{\sum_{(u,v)\in \Omega}|F(u,v)|}.
\]

Here, \(F(u,v)\) denotes the magnitude at frequency \((u,v)\), \(\Omega\) denotes the full frequency domain and \(H\) denotes the high-frequency band. The distribution of \(Q_{\text{FFT}}\) was computed independently for each sequence, and the first quartile was used as a sample-adaptive threshold. Frames below this threshold were excluded from subsequent segmentation and COLMAP processing.

#### 2.3.2 SAM3 Text-Prompted Segmentation

Frames that passed quality filtering were input to SAM3. Promptable segmentation models have shown strong general foreground localisation in natural-image and video segmentation [38,39], and they have increasingly been used in plant-organ, point-cloud and 3D-scene segmentation tasks [35,40-44]. We compared five text prompts: P1, `green plant`, for broad green plant regions; P2, `entire plant excluding pot`, for the complete plant body without the pot; P3, `leaves and stems`, for above-ground organs such as leaves and stems; P4, `crop seedling`, for small or seedling-like forms; and P5, `plant body without background`, for the complete plant foreground excluding the background. P2 was selected as the default reconstruction prompt because pots and soil contaminate foreground-only Gaussian representations, and plant-height measurement requires separation between the container and plant geometry. For each view \(i\), SAM3 produced a binary mask \(M_i\in\{0,1\}^{H\times W}\).

#### 2.3.3 PCA Main-Foreground Refinement

SAM3 outputs may contain small fragments, holes or unstable false positives. FSAM3 first applies morphological closing with a 5 x 5 elliptical kernel. It then performs 8-connected component analysis and removes components smaller than 0.5% of the image area. When multiple large components remain, PCA-based selection is applied according to the positional stability of component bounding boxes across the sequence, retaining the most stable main foreground across views. This step is intended to produce reconstruction priors that are strictly aligned with training-view files and relatively stable at object boundaries.

### 2.4 Plant-Aware 2DGS: A Foreground-Object Training Objective

As shown in Fig. 2, ForeSplat moves FSAM3 masks into the initialisation, optimisation and post-processing stages of 2DGS. The reconstruction target is thereby shifted from full-scene appearance to the plant body. Foreground trajectory initialisation and foreground RGB supervision constrain the optimisation object. Alpha and background opacity constraints, together with view-quality soft weighting, stabilise training. Mask-guided pruning cleans redundant primitives and exports a foreground-only TSDF mesh.

**Fig. 2 | Algorithmic modification of 2DGS in ForeSplat.** The diagram summarises the relationships among COLMAP cameras, FSAM3 masks, foreground RGB supervision, alpha and background opacity constraints, view-quality soft weighting, mask-guided pruning and TSDF mesh extraction. It highlights the reformulation of the training objective and optimisation strategy in ForeSplat.

#### 2.4.1 Camera Pose Estimation and Foreground Trajectory Initialisation

Camera intrinsics, extrinsics and sparse 3D point tracks were estimated using the incremental SfM pipeline in COLMAP [16]. Standard 2DGS initialises Gaussian primitives from all sparse points, so background points enter the representation before optimisation begins. ForeSplat filters sparse points through multiview mask consistency. Let sparse point \(X_j\) be visible in a view set \(V_j\), and let its projection in view \(i\) be \(\pi_i(X_j)\). The retention criterion is

\[
\operatorname{Keep}(X_j)=1,\quad
\text{if}\quad
\frac{1}{|V_j|}\sum_{i\in V_j}M_i(\pi_i(X_j))\geq \tau_{\text{track}}.
\]

Foreground trajectory initialisation required each sparse point to be observed in at least 3 views. We set \(\tau_{\text{track}}=0.9\) and used no mask dilation.

#### 2.4.2 Foreground RGB Supervision and Opacity Constraints

Standard 2DGS optimises RGB reconstruction over the full image domain \(\Omega\) [28]. For plant phenotyping, this allows pots, tables and background structures to compete with the measured plant for model capacity. ForeSplat therefore redefines photometric supervision on the plant foreground. The standard full-image RGB loss is

\[
L_{\text{rgb-full}}=
\frac{1}{|\Omega|}\sum_{p\in\Omega}\|R(p)-I(p)\|_1 .
\]

ForeSplat restricts the RGB loss to the plant foreground:

\[
L_{\text{rgb-fg}}=
\frac{1}{|\Omega_{\text{fg}}|}
\sum_{p\in\Omega}M(p)\|R(p)-I(p)\|_1,\quad
\Omega_{\text{fg}}=\{p|M(p)=1\}.
\]

We further add an alpha mask loss and a background opacity loss:

\[
L_{\text{mask}}=\frac{1}{|\Omega|}\sum_{p\in\Omega}|A(p)-M(p)| ,
\]

\[
L_{\text{bg}}=
\frac{1}{|\Omega_{\text{bg}}|}
\sum_{p\in\Omega}(1-M(p))A(p),\quad
\Omega_{\text{bg}}=\{p|M(p)=0\}.
\]

The complete optimisation objective is

\[
L_{\text{core}}=
L_{\text{rgb-fg}}+\lambda_{\text{mask}}L_{\text{mask}}
+\lambda_{\text{bg}}L_{\text{bg}}+L_{\text{reg}},
\]

where \(L_{\text{reg}}\) includes the 2DGS depth-distortion loss and normal-consistency loss. We used \(\lambda_{\text{mask}}=0.08\) and \(\lambda_{\text{bg}}=0.02\). The mask loss type was `l1_dice`, mask boundaries of 2 px were ignored, and the mask loss was activated after 500 iterations with a 1500-iteration warm-up.

### 2.5 View-Quality-Aware Soft Loss Weighting

In multiview plant sequences, some views may contain mild blur, specular reflection or local occlusion. Nevertheless, they may still cover thin leaf structures visible only from narrow angles. Directly removing low-quality frames can therefore damage angular coverage. ForeSplat uses soft weighting:

\[
L_{\text{rgb-fg-soft}}=
\frac{\sum_i q_i L_{\text{rgb-fg}}(i)}{\sum_i q_i}.
\]

Here, \(q_i\) is the quality weight for view \(i\). ForeSplat reads H-VQG soft-weight files, uses `view_weight_mode=rgb_only` and constrains the weight range to 0.6-1.0. The quality score combines mask coverage, mask-boundary sharpness and foreground RGB contrast.

### 2.6 Mask-Guided Multi-Cue Gaussian Pruning

Weakly supported Gaussian primitives may remain near object boundaries in late training. For each Gaussian \(g_j\), ForeSplat computes

\[
\operatorname{Score}(g_j)=
\alpha M_j+\beta O_j+\gamma V_j+\delta B_j+\eta C_j ,
\]

where \(M_j\), \(O_j\), \(V_j\), \(B_j\) and \(C_j\) denote mask-projection consistency, opacity, visible-view count, colour/brightness normality and local topological cues, respectively. The pruning module uses `pruning_mode=mask`, starts at 18,000 iterations and is executed every 3,000 iterations. The main thresholds include an opacity threshold of 0.005, a brightness threshold of 0.01, a mask threshold of 0.45, a maximum view count of 12, a maximum removal ratio of 0.03 and a mask-score weight of 3.0.

### 2.7 Mesh Extraction, Scale Recovery and Phenotypic Measurement

To convert the foreground-only Gaussian representation into an interactive phenotyping object, we generated explicit meshes through depth rendering and TSDF-style fusion. TSDF fusion and Marching Cubes are classical routes for extracting explicit surfaces from multiview depth or implicit fields [45,46]. For a voxel centre \(x\), the fused distance is

\[
D(x)=\frac{\sum_c w_c(x)d_c(x)}{\sum_c w_c(x)} ,
\]

where \(d_c(x)\) is the truncated signed distance under camera \(c\), and \(w_c(x)\) is the fusion weight. The zero level set was extracted using Marching Cubes. We compared standard TSDF, a smaller truncation distance and boundary post-processing. Scale was recovered using the known physical diameter of the pot as a reference. After scale recovery, plant height, canopy width, leaf length and leaf width were extracted and compared with manual measurements.

### 2.8 Evaluation Metrics

Segmentation was evaluated using F1-score, mIoU and HD95. Foreground reconstruction was evaluated using PSNR_fg, SSIM_fg, LPIPS_fg and two background leakage metrics: mean outside-mask non-black ratio and mean leakage energy ratio. PSNR, SSIM and LPIPS reflect pixel error, structural similarity and perceptual similarity, respectively, and are widely used in neural rendering and 3D reconstruction evaluation [47-49]. The foreground-only criterion was defined as

\[
\text{outside}<0.05,\quad \text{leakage}<0.10.
\]

Phenotypic measurement was evaluated using MAE, RMSE, MAPE, bias and Pearson \(R^2\). Bias was defined as the virtual measurement minus the manual measurement.

---

## 3. Results

### 3.1 FSAM3 Provides Stable Multiview Plant-Foreground Priors

FSAM3 provided stable reconstruction priors for multiview potted plants across species and structural forms. All 20 samples produced plant-foreground masks suitable for reconstruction. P2 and P5 were the most stable prompts across species. P1 tended to include green background regions, P3 missed some thick stems or mixed flower-leaf structures, and P4 under-segmented mature plants. PCA main-foreground refinement reduced the mean number of connected components per frame from 12.4 to 4.1, a reduction of 67%, while retaining the dominant plant region in 98.2% of frames. Dataset coverage, prompt differences, mask refinement and the comparison with SEEM are summarised in Fig. 3.

**Fig. 3 | Dataset coverage and FSAM3 foreground-prior quality.** a, Representative images from fixed-device-assisted acquisition and complex-background handheld acquisition, showing broad leaves, compact canopies, mixed flowers and leaves, and dense occlusion. b, Raw views, initial SAM3 masks, PCA-refined main foregrounds and foreground-only RGB images for representative samples. c, Typical success and failure modes for the five text prompts. d, Segmentation metrics for FSAM3 and SEEM: FSAM3 achieved F1-score = 98.3%, mIoU = 97.9% and HD95 = 41.4 px, whereas SEEM achieved 95.1%, 94.1% and 281.9 px. e, Connected-component counts before and after refinement and the main-foreground retention rate.

These results show that FSAM3 provided a more complete and stable reconstruction foreground prior under the acquisition conditions of this study.

### 3.2 ForeSplat Improves Application-Level Reconstruction Quality and Processing Efficiency

ForeSplat stably generated measurable 3D representations from ordinary RGB acquisition. All samples completed the full workflow from video capture, FSAM3 masking, COLMAP, 2DGS and TSDF meshing to phenotypic measurement. The workflow success rate was 100% across 20 sequences. In the application-level reconstruction comparison, ForeSplat achieved PSNR = 31.09 dB, SSIM = 0.9711 and LPIPS = 0.0365. Visual reconstruction, mesh outputs and efficiency trends for the four workflows are shown in Fig. 4.

**Fig. 4 | Reconstruction quality, geometric output and processing efficiency.** a, Representative novel-view renderings, foreground-only renderings and TSDF mesh outputs from COLMAP, 3DGS-FSAM3, standard 2DGS and ForeSplat. Insets highlight differences in leaf edges, pot residue and background leakage. b, Normalised comparison of PSNR, SSIM, LPIPS, training time and meshing time for the four workflows. c, Module ablation curves after introducing FFT, SAM3/FSAM3 foreground segmentation and their combination. d, Successful sample counts, effective frame counts and registered-view counts in the fixed-device and complex-background acquisition settings.

**Table 2 | Reconstruction quality and processing efficiency of different workflows.**

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Training time / s ↓ | Meshing time / s ↓ |
|---|---:|---:|---:|---:|---:|
| COLMAP | 13.63 | 0.8745 | 0.1072 | 599.5 | 78 |
| 3DGS-FSAM3 | 30.17 | 0.9587 | 0.0386 | 5413.5 | 642 |
| Standard 2DGS | 29.58 | 0.9574 | 0.0487 | 12913.7 | 157.9 |
| ForeSplat | 31.09 | 0.9711 | 0.0365 | 5044.5 | 55.0 |

Compared with standard 2DGS, the complete workflow increased PSNR by 1.51 dB and reduced training time and mesh-extraction time by 60.94% and 65.17%, respectively. Compared with 3DGS-FSAM3, ForeSplat maintained higher reconstruction quality while reducing mesh-extraction time from 642 s to 55 s. Module ablations further showed that FFT mainly reduced the influence of low-quality frames on pose estimation and optimisation, whereas SAM3/FSAM3 foreground segmentation mainly reduced background entry into the Gaussian representation and meshing process. This processing principle is consistent with known trade-offs among low-quality frames, view coverage and reconstruction stability [18,19,50].

### 3.3 Foreground RGB Supervision Is Critical for Suppressing Background Leakage

To determine whether foreground-only plant representation must be defined during training, we compared full-scene training, input-domain masking, opacity regularisation, foreground RGB supervision and post hoc mask pruning on the KongQueZhuYu sample. The unconstrained full-scene baseline reconstructed nearly the entire background, with an outside-mask non-black ratio of 0.9908 and a leakage energy ratio of 1.2201. Input-domain foreground masking reduced background visibility but decreased PSNR_fg from 24.2090 dB to 20.7291 dB, indicating substantial foreground-quality loss. Alpha mask consistency, background opacity suppression and their joint regularisation did not prevent background learning when full-image RGB supervision was retained. Their leakage energy ratios remained around 1.23. The core metrics are listed in Table 3, and the corresponding visual evidence is shown in Fig. 5.

**Table 3 | Ablation of foreground-object reconstruction objectives on KongQueZhuYu.** The table compares full-scene training, input-domain masking, opacity constraints, foreground RGB supervision and post hoc mask pruning in terms of background leakage and foreground quality.

| Method setting | Foreground initialisation | Foreground RGB | Alpha/background | PSNR_fg ↑ | Outside ratio ↓ | Leakage ↓ | Gaussian count ↓ | Foreground only |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Full-scene 2DGS without foreground constraints | No | No | No | 24.2090 | 0.9908 | 1.2201 | 751,213 | No |
| Input-domain foreground mask constraint | No | Implicit | No | 20.7291 | 0.0073 | 0.0042 | 263,108 | Yes, with quality loss |
| Alpha mask consistency only | No | No | Alpha | 24.3422 | 0.9898 | 1.2260 | 768,067 | No |
| Background opacity suppression only | No | No | Background | 24.7508 | 0.9900 | 1.2255 | 742,931 | No |
| Joint alpha and background opacity regularisation | No | No | Joint | 24.8126 | 0.9896 | 1.2266 | 763,266 | No |
| Foreground RGB supervision with opacity regularisation | No | Yes | Joint | 25.1055 | 0.0294 | 0.0190 | 592,900 | Yes |
| Complete foreground-object reconstruction objective | Yes | Yes | Joint | 25.0072 | 0.0294 | 0.0189 | 591,623 | Yes |
| Post hoc mask pruning after full-scene training | No | No | Post hoc | 24.6918 | 0.7509 | 0.7900 | -- | No |

**Fig. 5 | Visual evidence for the foreground-object reconstruction ablation.** a, Renderings and mesh visualisations under different training objectives, showing whether the background, pot and table entered the final representation. b, Bar plots of outside-mask non-black ratio and leakage energy ratio, with foreground-only thresholds marked. c, Local magnifications before and after foreground RGB supervision, highlighting differences in background leakage and leaf-edge retention. d, Comparison between post hoc mask pruning and training-time foreground-object optimisation.

Only foreground RGB supervision reduced the outside ratio and leakage below the foreground-only thresholds. After adding foreground trajectory initialisation, the complete configuration maintained similar foreground quality and background suppression. Its PSNR_fg, SSIM_fg, LPIPS_fg, outside ratio, leakage and Gaussian count were 25.0072, 0.8548, 0.0438, 0.0294, 0.0189 and 591,623, respectively. In the post hoc mask-pruning control, the outside ratio and leakage remained 0.7509 and 0.7900. This indicates that post-processing after full-scene training is not equivalent to foreground-object optimisation.

### 3.4 Representative Structural Samples Validate the Generality of Foreground-Object Reconstruction

Three representative samples with distinct structures were used to assess the robustness of the foreground-object reconstruction objective. KongQueZhuYu, XianKeLai1 and CaoMei2 all satisfied the foreground-only criterion. Their outside ratios were 0.0294, 0.0484 and 0.0147, respectively, and their leakage energy ratios were 0.0189, 0.0379 and 0.0081. XianKeLai1 was closest to the outside-ratio threshold, indicating that thin leaves and fine structures remain boundary-sensitive cases. Inputs, masks, renderings, meshes and local error locations for the three representative samples are shown in Fig. 6.

**Fig. 6 | Foreground-only reconstruction on representative structural samples.** a, Raw images, foreground masks, foreground-only renderings and TSDF meshes from samples with complex backgrounds, thin fine structures and dense occlusion. b, PSNR_fg, outside ratio, leakage energy ratio and Gaussian count for the three samples. c, Local magnifications of thin leaf boundaries and occluded regions, showing that residual errors were concentrated mainly at leaf edges, petioles and local occlusions.

### 3.5 Soft View Weighting Preserves Geometric Coverage and Produces Compact Representations

To evaluate the role of lower-quality views in reconstructing thin plant structures, we compared hard view removal with soft loss weighting. This analysis focused not on single-image quality, but on whether multiview geometric coverage was preserved. Quality-threshold-based hard removal discarded 10 of 27 views and decreased PSNR_fg from 25.0072 dB to 12.5478 dB. Mask-quality-based removal retained 24 views, but PSNR_fg remained only 13.4557 dB. Soft weighting retained all 27 views, reduced PSNR_fg by only 0.0506 dB and decreased the Gaussian count from 591,623 to 532,264, a reduction of 10.03%. Quality-size trade-offs under different view strategies and compactness settings are shown in Fig. 7.

**Fig. 7 | View-quality strategies and representation compactness.** a, View-coverage schematic for hard view removal and soft loss weighting, highlighting thin leaf regions visible from deleted views. b, Comparison of PSNR_fg, SSIM_fg, LPIPS_fg, outside ratio, leakage and Gaussian count under three view-quality strategies. c, Quality-compactness trade-offs among the complete configuration, soft weighting, mask-guided pruning and compact configuration for CaoMei2, XianKeLai1 and KongQueZhuYu. d, Gaussian count, leakage metrics and representative rendering differences before and after compactness optimisation.

Across the three representative samples, the compact configuration reduced the total number of Gaussians from 1,216,294 to 997,049, a reduction of 18.03%. Mean PSNR_fg decreased by 0.0657 dB, SSIM_fg decreased by 0.0011 and LPIPS_fg increased by 0.0003. The main value of this configuration was compactness and export cleanliness rather than marked improvement in foreground rendering quality. Related studies on Gaussian compression and simplification also indicate that compact representation typically requires trade-offs among rendering quality, storage and speed [51-54].

### 3.6 Mesh-Structure Validation

To convert foreground-only Gaussian representations into measurable objects, we compared structural connectivity, boundary metrics and computational time under different TSDF meshing settings. The standard TSDF mesh for KongQueZhuYu contained 167,789 vertices and 8 connected components, with a largest-component ratio of 0.9920. A smaller truncation distance reduced the vertex count to 147,665 but increased the number of connected components to 20 and reduced the largest-component ratio to 0.9350. XianKeLai1 showed the same trend: the smaller truncation distance reduced the vertex count from 74,753 to 66,138 but increased the number of connected components from 6 to 12. The effects of different TSDF settings on mesh morphology, connectivity and boundaries are shown in Fig. 8.

**Fig. 8 | TSDF mesh-structure validation.** a, Mesh morphology, connected components and boundary-edge distributions under standard TSDF, smaller truncation distance and boundary post-processing for KongQueZhuYu and XianKeLai1. b, Vertex count, largest-component ratio, boundary-edge count and computation time under different meshing settings. c, Local magnifications of leaf edges, holes and thin leaf boundaries, showing the fragmentation risk introduced by a smaller truncation distance.

A smaller truncation distance reduced the vertex count but increased fragmentation risk. Boundary post-processing kept the number of connected components unchanged but increased mesh-generation time. These results support evaluation at the structural and efficiency levels, but they do not establish causal evidence that a particular mesh variant reduces phenotyping error.

### 3.7 Phenotypic Validation

The reconstructed 3D representations were further used for agricultural trait measurement. Virtual measurements were compared with manual measurements for plant height, canopy width, leaf length and leaf width. Correlations and residual distributions are shown in Fig. 9.

**Fig. 9 | Agreement between manual measurements and virtual phenotypic measurements.** a-d, Scatter plots of manual and virtual measurements for plant height, canopy width, leaf length and leaf width, with linear fits and 1:1 reference lines. e, Residual distributions for the four traits. f, Summary visualisation of MAE, RMSE, MAPE, bias and \(R^2\).

**Table 4 | Agreement between manual and virtual phenotypic measurements.**

| Trait | n | MAE/cm ↓ | RMSE/cm ↓ | MAPE/% ↓ | Bias/cm | \(R^2\) ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Plant height | 21 | 0.98 | 1.21 | 6.91 | 0.58 | 0.9878 |
| Canopy width | 21 | 0.86 | 0.99 | 4.50 | 0.64 | 0.9879 |
| Leaf length | 63 | 0.51 | 0.64 | 7.45 | 0.31 | 0.9738 |
| Leaf width | 63 | 0.45 | 0.64 | 9.73 | 0.38 | 0.8999 |

Agreement was highest for plant height and canopy width, followed by leaf length. Leaf width showed the largest error. All traits had a slight positive bias, indicating that virtual measurements modestly overestimated manual measurements. The lower \(R^2\) for leaf width is consistent with higher sensitivity to thin leaf boundaries, TSDF boundary expansion and landmark placement error.

---

## 4. Discussion

### 4.1 From Full-Scene Reconstruction to Foreground-Object Reconstruction

The central finding of this study is that foreground-only plant representations for trait measurement cannot be reliably obtained by mask pruning after full-scene 2DGS training. During training, standard 2DGS allocates Gaussian capacity according to full-image RGB loss. Background regions therefore generate photometric gradients and form stable representations through densification. Post hoc pruning can remove only some primitives after they have formed. It cannot redirect capacity allocation during training. The failure of alpha mask loss, background opacity loss and their joint regularisation in Fig. 5 shows that constraining the opacity field alone is insufficient to prevent background learning. Only when RGB supervision itself is restricted to foreground pixels does the main optimisation pressure shift towards the plant object.

This conclusion is consistent with the plant-level analysable representations emphasised by LCR-GS and IPENS. ForeSplat further shows that, when the target is single-plant phenotyping, object boundaries should appear as early as possible in the reconstruction objective. In this sense, masks are not merely segmentation outputs. They are part of the agricultural 3D optimisation problem.

### 4.2 FSAM3 Is a Reconstruction Prior Rather Than a Segmentation Endpoint

The contribution of FSAM3 is to provide stable masks for 2DGS training, not to replace 3D optimisation. FFT filtering reduces blurred frames, SAM3 prompts provide semantic foregrounds, and PCA refinement suppresses false-positive fragments. The manual-annotation benchmark indicates that FSAM3 outperformed SEEM under the data conditions of this study. Future work should analyse the coupling between segmentation error and reconstruction error on larger annotated datasets.

### 4.3 View Quality Should Be Modulated Rather Than Deleted

The failure of hard view removal shows that lower-quality frames in multiview plant reconstruction are not necessarily useless frames. Plant leaves often have angle-dependent visibility. A small number of weaker views may provide irreplaceable geometric coverage. Soft weighting separates geometric coverage from photometric reliability: all views are retained to maintain 3D constraints, whereas quality weights modulate only their contribution to RGB loss. This principle is particularly important for thin leaves, occlusions and complex canopies.

### 4.4 The Value of Representation Compactness

The main benefit of the compact configuration was not higher PSNR. Instead, it reduced the number of Gaussian primitives with minimal loss in foreground quality. For high-throughput phenotyping, this implies lower storage requirements, faster mesh export and easier batch processing. Mask-guided pruning is more suitable for plants than opacity-threshold pruning alone, because primitives near leaf edges and holes may have non-negligible opacity but insufficient multiview mask support. Pruning intensity should remain conservative, as excessive pruning may damage fine leaves, petioles and thin boundaries.

### 4.5 Phenotyping Error Patterns

Plant height and canopy width are determined by overall bounding ranges and are therefore less sensitive to local boundary errors. Leaf length depends on the main axis of individual leaves and has intermediate difficulty. Leaf width depends on local cross-sectional boundaries and is most affected by reconstruction resolution, Gaussian support domains, TSDF fusion and landmark placement. These results support the use of ForeSplat for automated structural-trait measurement, while identifying leaf width as the trait most in need of improvement. Future work could combine higher-resolution acquisition, boundary-aware mesh refinement, leaf-edge uncertainty modelling and repeated landmark assessment by multiple operators.

### 4.6 Relationship to Reference Studies

In contrast to comparisons of NeRF-based plant reconstruction efficiency and accuracy by Arshad and colleagues, ForeSplat uses an explicit 2DGS representation and restricts the reconstruction target to the plant foreground. Similar to Plant3R, this study values the combination of geometric priors and Gaussian rendering. Plant3R focuses on improving wheat-scene initialisation through MASt3R, whereas ForeSplat focuses on reformulating the 2DGS training objective with foreground masks. Compared with IPENS and LCR-GS, ForeSplat does not extract target point clouds or individual-plant subsets after reconstruction. Instead, it directly produces a foreground-only Gaussian representation during training. These routes are not mutually exclusive. Future systems could combine stronger feature matching, SAM2/3 temporal propagation, LCR-GS-style multi-plant decomposition and the ForeSplat foreground-object objective for more complex greenhouse and field scenes.

---

## 5. Conclusion

We present ForeSplat, a foreground-aware 2D Gaussian Splatting workflow that spans multiview image quality control, foreground-object reconstruction and plant mesh generation for phenotypic measurement. Through FSAM3 foreground priors, foreground trajectory initialisation, foreground RGB supervision, alpha and background opacity constraints, view-quality-aware soft loss weighting and mask-guided Gaussian pruning, ForeSplat reformulates standard full-scene 2DGS as foreground-only plant-object reconstruction. The results show that foreground RGB supervision is the key mechanism for suppressing background leakage. Hard view removal can damage angular coverage of thin structures, whereas soft weighting preserves quality while reducing the Gaussian count. Compactness optimisation mainly improves representation compactness and export efficiency. Validation across 20 sequences and 21 plants indicates that the workflow can support measurement of plant height, canopy width, leaf length and leaf width. Overall, ForeSplat provides a reproducible and extensible technical route for low-cost, non-destructive, multispecies 3D phenotyping of potted plants under indoor or semi-controlled conditions.

---

## Data Availability

The multiview images, FSAM3 masks, phenotypic measurement tables, view-weight files and main running configurations supporting this study are available from the authors upon reasonable request. They will be released through a project repository or data repository after data curation is complete.

## Ethics Statement

This study involved only plant imaging and measurement. It did not involve human or animal subjects.

## Competing Interests

The authors declare no competing interests.

## AI Use Statement

During manuscript preparation, the authors used AI-assisted tools for literature organisation, structural rewriting, language polishing and Chinese-English translation. All AI-assisted content was reviewed, verified and edited by the authors. The authors take full responsibility for the manuscript content, data interpretation, citation accuracy and integrity of the published work.

## References

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
