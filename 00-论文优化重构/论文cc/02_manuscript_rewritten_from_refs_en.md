# ForeSplat: foreground-aware 2D Gaussian Splatting for low-cost plant phenotyping

## Highlights

- ForeSplat aligns the 2DGS reconstruction objective with plant trait measurement.
- Foreground RGB supervision reduced the leakage energy ratio from 1.2201 to 0.0190.
- The pipeline completed 20 RGB multi-view sequences across two acquisition settings.
- Virtual trait measurements agreed closely with manual measurements, with R² up to 0.9879.
- Soft view weighting reduced Gaussian primitives while preserving geometric coverage.

## Abstract

Rapid and non-destructive measurement of plant structural traits is a key basis for intelligent breeding, protected cultivation and precision management. Manual phenotyping is labor-intensive, and two-dimensional images cannot fully resolve overlapping leaves, occluded organs and canopy geometry. Multi-view three-dimensional reconstruction offers a low-cost alternative for phenotyping, but existing Neural Radiance Field and Gaussian Splatting pipelines usually optimize full-scene appearance, allowing pots, substrates, supports and background structures to enter the reconstruction. This mismatch limits downstream mesh extraction and virtual trait measurement. Here we introduce ForeSplat/F2DMAS, a foreground-aware 2D Gaussian Splatting workflow that moves plant masks from a post-processing cue into the reconstruction objective. FSAM3 combines frequency-domain frame-quality screening, text-prompted plant segmentation and foreground refinement to generate multi-view priors, which are used for foreground initialization, RGB supervision, opacity constraints, view weighting, Gaussian pruning, TSDF meshing and scale recovery. Experiments on 20 RGB sequences and 21 potted plants showed that FSAM3 achieved an F1-score of 98.3%, mIoU of 97.9% and HD95 of 41.4 px. In ablation experiments, foreground RGB supervision reduced the outside-mask non-black ratio from 0.9908 to 0.0294 and the leakage energy ratio from 1.2201 to 0.0190. ForeSplat/F2DMAS achieved PSNR = 31.09 dB, SSIM = 0.9711 and LPIPS = 0.0365, while reducing training time and meshing time by 60.94% and 65.17% relative to standard 2DGS. Virtual measurements of plant height, canopy width, leaf length and leaf width agreed closely with manual measurements, with R² values of 0.9878, 0.9879, 0.9738 and 0.8999, respectively. These results indicate that ordinary RGB imaging can provide a reusable route for low-cost plant-level 3D phenotyping under indoor and semi-controlled complex-background conditions, although leaf-width measurement remains sensitive to boundary reconstruction.

**Keywords:** phenotyping; reconstruction; segmentation; Gaussian Splatting; RGB imaging; mesh; trait

---

## 1. Introduction

Plant phenotyping links genotype, environmental response and agronomic performance. Structural traits such as plant height, canopy width, leaf length and leaf width are widely used for breeding selection, cultivation management and growth-status assessment, but manual measurement is usually contact-based, low-throughput and affected by operator experience. Image-based high-throughput phenotyping systems have therefore become an important direction in intelligent agriculture [1-6]. Compared with two-dimensional images, three-dimensional representations can record organ position, occlusion relationships and canopy volume, making them more suitable for overlapping leaves, complex canopies and non-planar structures [7-12]. Plants, however, are not regular, rigid or richly textured engineering objects. Thin leaves, weak texture, repetitive texture, local occlusion, mixed flowers and leaves, and background colors similar to foliage all make low-cost 3D reconstruction and subsequent trait extraction difficult.

Existing 3D plant phenotyping studies have expanded from classical SfM/MVS and depth sensors to neural rendering and explicit Gaussian representations. LiDAR, structured light and depth cameras can provide high-precision point clouds, but equipment cost, calibration complexity and deployment requirements limit their adoption in protected horticulture and large-scale breeding [13-15]. Consumer RGB cameras are flexible and inexpensive, but SfM/MVS is highly sensitive to viewpoint coverage, image sharpness and leaf texture, and often produces holes, noise or blurred boundaries in thin leaves and occluded regions [16-19]. NeRF improves plant reconstruction through continuous volumetric radiance fields and has been used for field and indoor plant geometry evaluation [20-26]. 3D Gaussian Splatting further represents scenes with explicit Gaussian primitives, improving rendering efficiency and editability [27-29]. Recent studies including Plant3R, PlantGaussian, Cotton3DGaussians and object-centric 3DGS show that Gaussian representations are entering plant structural reconstruction and trait-analysis workflows [30-33]. These advances suggest that neural rendering is becoming an important tool for low-cost 3D plant phenotyping, but they also leave a key agricultural question: is the reconstructed 3D object the plant object that needs to be measured?

This problem is especially prominent in potted or protected-cultivation scenes. Standard NeRF, 3DGS and 2D Gaussian Splatting (2DGS) usually optimize visual reconstruction over the whole image, so the model learns plants, pots, substrates, tables, supports and backgrounds at the same time. For novel-view synthesis, full-scene reconstruction is a reasonable objective; for plant height, canopy width and leaf-size measurement, it can introduce non-plant geometry into mesh extraction and bounding-range calculation. LCR-GS extracts individual greenhouse muskmelon plants from 3DGS scenes, showing that downstream trait extraction requires clean, analysis-ready plant representations [34]. IPENS lifts SAM2-generated 2D masks into NeRF 3D space for rice and wheat organ point-cloud extraction, showing that promptable segmentation and radiance fields can reduce annotation burden [35]. Gaussian Grouping and SAGA further show that semantics from 2D foundation models can be lifted or distilled into Gaussian space, but these methods mainly target general scene segmentation and editing rather than trait-measurement objectives [36,37]. Together, these studies emphasize the need to move from scene-level reconstruction to plant-level representation, but most pipelines still separate the object after reconstruction. If the background has already received stable Gaussian capacity during training, post-processing mask pruning may not fully remove its effect on plant meshes and virtual measurements.

The basic premise of this paper is that masks in agricultural phenotyping should not be only post-processing filters; they should help define the 3D optimization problem. We propose ForeSplat/F2DMAS, a foreground-aware 2DGS phenotyping pipeline for multi-species potted plants. Unlike ordinary scene reconstruction, ForeSplat binds initialization points, the RGB supervision domain, opacity constraints, view-quality weights and Gaussian cleanup to the mask-defined plant foreground. The main photometric gradients during training therefore come from plant pixels, and model capacity is preferentially allocated to the target plant rather than the pot, table or background. The planar Gaussian primitives in 2DGS are suitable for thin surface structures such as leaves [28], and ForeSplat further makes this surface representation serve a plant-only measurement object.

To implement this objective, we first design FSAM3 as a reconstruction-oriented plant foreground prior. The aim is not to claim general-purpose plant segmentation superiority, but to combine FFT-based frame-quality screening, SAM3 text-prompted segmentation and PCA-guided main-foreground refinement into masks that are view-aligned, boundary-stable and suitable for 2DGS training. ForeSplat then filters COLMAP sparse tracks with multi-view mask consistency, introduces foreground RGB supervision, alpha mask loss and background opacity loss into 2DGS, and uses quality-aware soft loss weighting to retain low-quality views that may still provide valuable geometric coverage. After training, mask-guided multi-cue Gaussian pruning and TSDF meshing convert the plant-only Gaussian representation into a measurable mesh.

The goal of this work is not to build a universally extrapolatable plant reconstruction model, but to validate a low-cost, reusable RGB workflow for phenotype-oriented 3D reconstruction under indoor and semi-controlled complex-background conditions. The contributions are:

1. We propose ForeSplat/F2DMAS, which reformulates standard 2DGS from full-scene visual reconstruction into mask-defined plant-object reconstruction, aligning the reconstruction target with agricultural phenotype objects such as plant height, canopy width and leaf dimensions.

2. We propose FSAM3, a plant foreground-prior generation pipeline that combines FFT-based quality screening, SAM3 text-prompted segmentation and PCA-guided main-foreground refinement into a reconstruction-oriented mask workflow for boundary-stable, file-aligned and cross-species foreground constraints.

3. We build an end-to-end phenotyping workflow from ordinary RGB sequences to plant-only meshes through foreground track initialization, foreground RGB supervision, alpha/background opacity constraints, soft view weighting and mask-guided Gaussian pruning.

4. We validate ForeSplat/F2DMAS on 20 multi-view sequences and 21 plants, using external baselines, controlled ablations, compactness evaluation, mesh structural evaluation and manual-versus-virtual measurement comparison to analyze feasibility and boundary conditions.

---

## 2. Materials and Methods

### 2.1 Study design and pipeline overview

ForeSplat/F2DMAS targets a specific agricultural phenotyping task: generating plant-only meshes from ordinary multi-view RGB images for measuring plant height, canopy width and leaf dimensions. The workflow is organized around the principle of defining the target plant object before 3D reconstruction. First, FSAM3 performs frame-quality screening, plant foreground segmentation and main-component refinement, producing masks aligned with training views. Second, COLMAP estimates camera poses and sparse point tracks, and multi-view mask consistency filters foreground initialization points. Third, plant-aware 2DGS restricts RGB loss to the plant foreground and constrains the opacity field with alpha mask loss and background opacity loss, aligning the reconstructed object with the phenotyping object. Fourth, quality-aware soft loss weighting retains all views while modulating their contribution to training. Fifth, mask-guided multi-cue Gaussian pruning and TSDF mesh extraction output compact, measurable plant meshes. The overall workflow and module-level inputs and outputs are shown in Fig. 1.

**Fig. 1 | Overview of ForeSplat/F2DMAS.** Raw multi-view images are processed through FSAM3, COLMAP, foreground-object 2DGS optimization, soft view weighting, mask-guided Gaussian pruning, TSDF meshing and phenotype measurement to produce plant-only meshes and virtual trait values.

### 2.2 Dataset, acquisition settings and sample usage

To cover common structural variation in potted-plant phenotyping, the dataset contains 20 multi-view RGB sequences spanning broad leaves, low canopies, overlapping leaves, compact canopies, mixed flowers and leaves, thick leaves, fine texture and dense occlusion. Images were acquired with an iPhone 14 Pro Max at 1080 x 1920 resolution and 60 fps. Acquisition settings included fixed-device assisted acquisition and handheld orbiting under complex indoor backgrounds, allowing evaluation under both semi-controlled and more practical indoor deployment conditions. All 20 sequences were used for complete workflow validation. Phenotype statistics were computed at the plant level from 21 plants. For leaf length and leaf width, three representative leaves were measured per plant, giving n = 63 for leaf traits. Dataset coverage and end-to-end completion are summarized in Table 1.

**Table 1 | Data coverage and end-to-end workflow completion.** The main text retains the information needed to judge dataset scale and pipeline executability; per-sequence material, structural labels and usage are moved to Supplementary Table S1.

| Scene | Species/cultivar labels | Samples | Raw frames | Valid frames | SfM registered views | Successful samples | Success rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed-device assisted acquisition | 8 | 10 | 2502 | 2104 | 2040 | 10 | 100% |
| Complex-background acquisition | 7 | 10 | 2500 | 2113 | 2089 | 10 | 100% |

The samples cover broad leaves, low canopies, overlapping leaves, compact canopies, flower-leaf mixtures, thick leaves, fine texture and dense occlusion. Per-sample plant species, structural labels, acquisition settings and manual-measurement usage are provided in Supplementary Table S1.

### 2.3 FSAM3: frequency-spatial plant foreground prior

FSAM3 is designed to provide stable plant foreground priors for 3D phenotyping reconstruction, rather than to serve as an independent general-purpose segmentation model. Its inputs are raw multi-view RGB frames for each plant, and its outputs are binary masks, RGBA alpha images and foreground-only RGB images aligned one-to-one with the training images. It consists of three steps: FFT frame screening, SAM3 text-prompted segmentation and PCA-guided main-component refinement.

#### 2.3.1 FFT frame-quality screening

Blurred, defocused and low-texture frames can affect SfM pose estimation, SAM3 segmentation boundaries and later 2DGS optimization. For each frame, we compute the two-dimensional FFT magnitude spectrum and define the high-frequency energy ratio:

\[
Q_{\text{FFT}}(I)=
\frac{\sum_{(u,v)\in H}|F(u,v)|}{\sum_{(u,v)\in \Omega}|F(u,v)|}.
\]

Here, \(F(u,v)\) is the magnitude at frequency \((u,v)\), \(\Omega\) denotes the full frequency domain and \(H\) denotes the high-frequency band. Each sequence is processed independently, and the first quartile of the \(Q_{\text{FFT}}\) distribution is used as a sample-adaptive threshold. Frames below the threshold are excluded from subsequent segmentation and COLMAP processing.

#### 2.3.2 SAM3 text-prompted segmentation

Frames passing quality screening are input to SAM3. Promptable segmentation models have shown broad foreground-localization ability in natural images and videos [38,39] and are increasingly used for plant-organ, point-cloud and 3D-scene segmentation tasks [35,40-44]. We compare five text prompts: P1 `green plant` for a broad green plant region, P2 `entire plant excluding pot` for the complete plant body without the pot, P3 `leaves and stems` for above-ground leaves and stems, P4 `crop seedling` for small or seedling morphology, and P5 `plant body without background` for the complete plant foreground without background. P2 is used as the default reconstruction prompt because pots and soil contaminate plant-only Gaussian representations, and plant-height measurement requires the container to be separated from plant geometry. For each view \(i\), SAM3 outputs a binary mask \(M_i\in\{0,1\}^{H\times W}\).

#### 2.3.3 PCA-guided main-foreground refinement

SAM3 outputs may contain small fragments, holes or unstable false positives. FSAM3 first applies morphological closing with a 5 x 5 elliptical kernel, then performs 8-connected component analysis and removes components with areas below 0.5% of the image area. When multiple large components remain, PCA selection based on the temporal stability of component bounding-box positions is used to retain the most stable main foreground across views. The aim is to obtain a reconstruction prior that is strictly aligned with the training view files and relatively stable at the boundary.

### 2.4 Plant-aware 2DGS: foreground-object training objective

#### 2.4.1 Camera pose estimation and foreground track initialization

Camera intrinsics, extrinsics and sparse 3D point tracks are estimated with incremental SfM in COLMAP [16]. Standard 2DGS initializes Gaussians from all sparse points, allowing background points to enter the representation before optimization begins. ForeSplat filters sparse points using multi-view mask consistency. For a sparse point \(X_j\) visible in the view set \(V_j\), with projection \(\pi_i(X_j)\) in view \(i\), we define:

\[
\operatorname{Keep}(X_j)=1,\quad
\text{if}\quad
\frac{1}{|V_j|}\sum_{i\in V_j}M_i(\pi_i(X_j))\geq \tau_{\text{track}}.
\]

The complete configuration requires each initialization point to be observed by at least three views, uses \(\tau_{\text{track}}=0.9\), and applies no mask dilation.

#### 2.4.2 Foreground RGB supervision and opacity constraints

Standard 2DGS optimizes RGB reconstruction over the full image domain \(\Omega\) [28]. For plant phenotyping, this causes pots, tables and backgrounds to compete with the target plant for model capacity. ForeSplat therefore redefines photometric supervision on the plant foreground. The standard full-image RGB loss is:

\[
L_{\text{rgb-full}}=
\frac{1}{|\Omega|}\sum_{p\in\Omega}\|R(p)-I(p)\|_1 .
\]

ForeSplat restricts RGB loss to the plant foreground:

\[
L_{\text{rgb-fg}}=
\frac{1}{|\Omega_{\text{fg}}|}
\sum_{p\in\Omega}M(p)\|R(p)-I(p)\|_1,\quad
\Omega_{\text{fg}}=\{p|M(p)=1\}.
\]

We further add alpha mask loss and background opacity loss:

\[
L_{\text{mask}}=\frac{1}{|\Omega|}\sum_{p\in\Omega}|A(p)-M(p)| ,
\]

\[
L_{\text{bg}}=
\frac{1}{|\Omega_{\text{bg}}|}
\sum_{p\in\Omega}(1-M(p))A(p),\quad
\Omega_{\text{bg}}=\{p|M(p)=0\}.
\]

The complete objective is:

\[
L_{\text{core}}=
L_{\text{rgb-fg}}+\lambda_{\text{mask}}L_{\text{mask}}
+\lambda_{\text{bg}}L_{\text{bg}}+L_{\text{reg}},
\]

where \(L_{\text{reg}}\) includes the 2DGS depth-distortion loss and normal-consistency loss. We use \(\lambda_{\text{mask}}=0.08\), \(\lambda_{\text{bg}}=0.02\), a `l1_dice` mask-loss type, a 2-px ignored band around the mask boundary, and a mask-loss schedule that starts at iteration 500 and warms up over 1500 iterations.

### 2.5 Quality-aware soft loss weighting

In multi-view plant sequences, some views may contain slight blur, specular reflection or local occlusion, but they may still cover thin leaf structures visible only from narrow angles. Directly deleting low-quality frames can damage angular coverage. ForeSplat uses soft weighting:

\[
L_{\text{rgb-fg-soft}}=
\frac{\sum_i q_i L_{\text{rgb-fg}}(i)}{\sum_i q_i}.
\]

Here, \(q_i\) is the quality weight for the \(i\)-th view. The current implementation reads H-VQG soft-weight files, uses `view_weight_mode=rgb_only`, and constrains weights to the range 0.6-1.0. The quality score combines mask coverage, mask-boundary sharpness and foreground RGB contrast.

### 2.6 Mask-guided multi-cue Gaussian pruning

Weakly supported Gaussians may remain near boundaries late in training. For each Gaussian \(g_j\), ForeSplat computes:

\[
\operatorname{Score}(g_j)=
\alpha M_j+\beta O_j+\gamma V_j+\delta B_j+\eta C_j ,
\]

where \(M_j\), \(O_j\), \(V_j\), \(B_j\) and \(C_j\) denote mask-projection consistency, opacity, number of visible views, color/brightness normality and local topological cues, respectively. The current configuration uses `pruning_mode=mask`, starts at iteration 18,000, repeats every 3,000 iterations, and uses the following main thresholds: opacity threshold = 0.005, brightness threshold = 0.01, mask threshold = 0.45, max views = 12, max remove ratio = 0.03 and mask score weight = 3.0.

### 2.7 Mesh extraction, scale recovery and phenotype measurement

To convert the plant-only Gaussian representation into an interactive phenotyping object, we generate explicit meshes through depth rendering and TSDF-style fusion. TSDF fusion and Marching Cubes are classical routes for extracting explicit surfaces from multi-view depth or implicit fields [45,46]. For voxel center \(x\), the fused distance is:

\[
D(x)=\frac{\sum_c w_c(x)d_c(x)}{\sum_c w_c(x)} ,
\]

where \(d_c(x)\) is the truncated signed distance under camera \(c\), and \(w_c(x)\) is the fusion weight. The zero level set is extracted with Marching Cubes. We compare standard TSDF, smaller truncation and post-boundary cleanup. Scale is recovered using a known physical reference; the current draft uses pot diameter. After scale recovery, plant height, canopy width, leaf length and leaf width are extracted and compared with manual measurements.

### 2.8 Evaluation metrics

Segmentation is evaluated with F1-score, mIoU and HD95. Foreground reconstruction is evaluated with PSNR_fg, SSIM_fg and LPIPS_fg, together with two background-leakage metrics: outside_nonblack_ratio_mean and leakage_energy_ratio_mean. PSNR, SSIM and LPIPS measure pixel error, structural similarity and perceptual similarity, respectively, and are widely used in neural rendering and 3D reconstruction evaluation [47-49]. The foreground-only criterion is:

\[
\text{outside}<0.05,\quad \text{leakage}<0.10.
\]

Phenotype measurement is evaluated with MAE, RMSE, MAPE, bias and Pearson \(R^2\). Bias is defined as virtual measurement minus manual measurement.

---

## 3. Results

### 3.1 FSAM3 provides stable multi-view plant foreground priors

To test whether foreground masks are sufficient as priors for 3D reconstruction, we first evaluated FSAM3 segmentation stability across multi-species and multi-structure potted plants. FSAM3 generated reconstruction-ready plant foreground masks for all 20 samples. P2 and P5 were the most stable prompts across species; P1 tended to include green background, P3 missed parts of thick stems or mixed flower-leaf samples, and P4 under-segmented mature plants. PCA-guided main-foreground refinement reduced the mean number of components from 12.4 to 4.1 per frame, a 67% decrease, while retaining the dominant plant region in 98.2% of frames. Dataset coverage, prompt behavior, mask refinement and the comparison with SEEM are summarized in Fig. 2, with per-sample segmentation metrics provided in Supplementary Table S2.

**Fig. 2 | Dataset coverage and quality of the FSAM3 foreground prior.** a, Representative images from fixed-device assisted acquisition and handheld complex-background acquisition, showing broad leaves, compact canopies, flower-leaf mixtures and dense occlusion. b, Representative raw views, initial SAM3 masks, PCA-guided main-foreground refinements and foreground-only RGB images. c, Typical success and failure modes of the five text prompts. d, Segmentation comparison between FSAM3 and SEEM: FSAM3 achieved F1-score = 98.3%, mIoU = 97.9% and HD95 = 41.4 px, whereas SEEM achieved 95.1%, 94.1% and 281.9 px. e, Connected-component counts before and after refinement and the main-foreground retention rate. Per-sample segmentation metrics are provided in Supplementary Table S2.

These results indicate that FSAM3 provides a more complete and more stable reconstruction foreground prior under the acquisition conditions of this study. Because the benchmark remains a representative subset, we do not present FSAM3 as a general-purpose state-of-the-art plant segmentation model.

### 3.2 ForeSplat/F2DMAS improves application-level reconstruction quality and processing efficiency

To verify whether the workflow can reliably convert ordinary RGB acquisition into measurable 3D representations, we ran the complete pipeline from video acquisition, FSAM3 masks, COLMAP, 2DGS and TSDF meshing to phenotype measurement on all samples. All 20 sequences were completed successfully, with a 100% success rate in both fixed-device and complex-background settings. In application-level reconstruction comparison, ForeSplat/F2DMAS achieved PSNR = 31.09 dB, SSIM = 0.9711 and LPIPS = 0.0365. Visual reconstruction, mesh outputs and efficiency trends across the four workflows are shown in Fig. 3, with quantitative metrics reported in Table 2.

**Fig. 3 | Reconstruction quality, geometric output and processing efficiency.** a, Representative novel-view renderings, foreground-only renderings and TSDF mesh outputs from COLMAP, 3DGS-FSAM3, standard 2DGS and ForeSplat/F2DMAS, with zoomed regions highlighting leaf margins, pot residues and background leakage. b, Normalized comparison of the four workflows in PSNR, SSIM, LPIPS, training time and mesh time. c, Module-ablation trends after introducing FFT, SAM3/FSAM3 foreground segmentation and their combination. d, Successful samples, valid frames and registered views in fixed-device and complex-background acquisition settings.

**Table 2 | Reconstruction quality and processing efficiency across workflows.**

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Training time / s ↓ | Mesh time / s ↓ |
|---|---:|---:|---:|---:|---:|
| COLMAP | 13.63 | 0.8745 | 0.1072 | 599.5 | 78 |
| 3DGS-FSAM3 | 30.17 | 0.9587 | 0.0386 | 5413.5 | 642 |
| Standard 2DGS | 29.58 | 0.9574 | 0.0487 | 12913.7 | 157.9 |
| ForeSplat/F2DMAS | 31.09 | 0.9711 | 0.0365 | 5044.5 | 55.0 |

Compared with standard 2DGS, the complete workflow improved PSNR by 1.51 dB and reduced training time and mesh extraction time by 60.94% and 65.17%, respectively. Compared with 3DGS-FSAM3, it achieved higher reconstruction quality while reducing mesh extraction time from 642 s to 55 s. The module ablation further showed that FFT mainly reduced the influence of low-quality frames on pose estimation and optimization, whereas SAM3/FSAM3 foreground segmentation mainly reduced background entry into the Gaussian representation and meshing process; this principle is consistent with known trade-offs between frame quality, view coverage and reconstruction stability [18,19,50]. Per-configuration values are provided in Supplementary Table S3.

### 3.3 Foreground RGB supervision is the key to suppressing background leakage

To determine whether plant-only representation must be defined in the training objective, we compared full-scene training, input-domain masks, opacity regularization, foreground RGB supervision and post-hoc mask pruning on the main KongQueZhuYu sample. The unconstrained baseline reconstructed almost the entire background, with outside-mask non-black ratio = 0.9908 and leakage energy ratio = 1.2201. Input-domain foreground masking suppressed background, but PSNR_fg decreased from 24.2090 dB to 20.7291 dB, indicating substantial foreground-quality degradation. Alpha mask consistency, background opacity suppression and their combination could not prevent background learning when full-image RGB supervision remained unchanged, with leakage energy ratios all near 1.23. The mechanism-level quantitative comparison is given in Table 3, the corresponding visual evidence is shown in Fig. 4, and additional ablation configurations are provided in Supplementary Table S4.

**Table 3 | Foreground-object reconstruction objective ablation on KongQueZhuYu.** The table retains the core values needed to judge mechanism-level causality; SSIM_fg, LPIPS_fg and additional configurations are provided in Supplementary Table S4.

| Method setting | fg init | fg RGB | alpha/bg | PSNR_fg ↑ | outside ↓ | leakage ↓ | Gaussians ↓ | FG-only |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Full-scene 2DGS without foreground constraint | No | No | No | 24.2090 | 0.9908 | 1.2201 | 751,213 | No |
| Input-domain foreground mask constraint | No | Implicit | No | 20.7291 | 0.0073 | 0.0042 | 263,108 | Yes, degraded quality |
| Alpha mask consistency only | No | No | alpha | 24.3422 | 0.9898 | 1.2260 | 768,067 | No |
| Background opacity suppression only | No | No | bg | 24.7508 | 0.9900 | 1.2255 | 742,931 | No |
| Alpha + background opacity regularization | No | No | Both | 24.8126 | 0.9896 | 1.2266 | 763,266 | No |
| Foreground RGB supervision + opacity regularization | No | Yes | Both | 25.1055 | 0.0294 | 0.0190 | 592,900 | Yes |
| Full foreground-object objective | Yes | Yes | Both | 25.0072 | 0.0294 | 0.0189 | 591,623 | Yes |
| Post-hoc mask pruning after full-scene training | No | No | Post-hoc | 21.34 | 0.31 | 0.28 | n/a | No |

**Fig. 4 | Visual evidence for the foreground-object objective ablation.** a, Renderings and mesh visualizations under different training objectives, showing whether background, pot and table geometry enter the final representation. b, Bar comparison of outside-mask non-black ratio and leakage energy ratio, with the foreground-only thresholds marked. c, Local zooms before and after foreground RGB supervision, highlighting background leakage and leaf-margin preservation. d, Comparison between post-hoc mask pruning and training-time foreground-object optimization.

Only foreground RGB supervision brought outside and leakage below the foreground-only thresholds. After adding foreground track initialization, the full configuration maintained similar foreground quality and background suppression, with PSNR_fg, SSIM_fg, LPIPS_fg, outside, leakage and Gaussian count of 25.0072, 0.8548, 0.0438, 0.0294, 0.0189 and 591,623, respectively. The post-hoc mask-pruning control still had outside and leakage values of 0.31 and 0.28, showing that post-processing after full-scene training is not equivalent to foreground-object optimization.

### 3.4 Representative structural samples validate foreground-object reconstruction beyond one case

To test whether the foreground-object objective was effective beyond a single sample, we selected three representative structures for validation: complex background, thin leaves/fine structure and dense occlusion. KongQueZhuYu, XianKeLai1 and CaoMei2 all satisfied the foreground-only criterion, with outside values of 0.0294, 0.0484 and 0.0147 and leakage values of 0.0189, 0.0379 and 0.0081, respectively. XianKeLai1 was closest to the outside threshold, indicating that thin leaves and fine structures remain boundary-sensitive cases. Inputs, masks, foreground renderings, meshes and local error regions for the three representative samples are shown in Fig. 5, with full per-sample metrics provided in Supplementary Table S5.

**Fig. 5 | Foreground-only reconstruction on representative structural samples.** a, Raw images, foreground masks, foreground-only renderings and TSDF meshes for complex-background, thin-leaf/fine-structure and dense-occlusion samples. b, PSNR_fg, outside, leakage and Gaussian count for the three samples. c, Local zooms of thin leaf boundaries and occluded regions, showing that residual errors are concentrated near leaf margins, petioles and local occlusions. Full per-sample metrics are provided in Supplementary Table S5.

### 3.5 Soft view weighting preserves geometric coverage and produces compact representations

To evaluate the role of low-quality views in thin plant-structure reconstruction, we compared hard view filtering and soft loss weighting. This experiment focuses not on single-image quality, but on whether multi-view geometric coverage is preserved. Quality-threshold hard filtering removed 10 of 27 views and reduced PSNR_fg from 25.0072 dB to 12.5478 dB; mask-quality hard filtering retained 24 views but still achieved only 13.4557 dB. Soft weighting retained all 27 views, reduced PSNR_fg by only 0.0506 dB and reduced Gaussian count from 591,623 to 532,264, a 10.03% reduction. The quality-size trade-off across view strategies and compact configurations is shown in Fig. 6, with detailed per-configuration values provided in Supplementary Table S6.

**Fig. 6 | View-quality strategy and representation compactness.** a, Schematic view-coverage comparison between hard view filtering and soft loss weighting, highlighting thin-leaf regions visible in removed views. b, Comparison of the three view-quality strategies in PSNR_fg, SSIM_fg, LPIPS_fg, outside, leakage and Gaussian count. c, Quality-compactness trade-off among the full configuration, soft weighting, mask-guided pruning and compact configuration on CaoMei2, XianKeLai1 and KongQueZhuYu. d, Gaussian count, leakage metrics and representative rendering differences before and after compactification. Detailed values are provided in Supplementary Table S6.

Across the three representative samples, the compact configuration reduced the total Gaussian count from 1,216,294 to 997,049, a reduction of 18.03%. Mean PSNR_fg decreased by 0.0657 dB, SSIM_fg decreased by 0.0011 and LPIPS_fg increased by 0.0003. Its main value is representation compactness and cleaner export, rather than a substantial improvement in foreground rendering quality. Related work on Gaussian compression and pruning also shows that representation compactness usually requires balancing rendering quality, storage and speed [51-54].

### 3.6 Mesh structural validation

To convert the plant-only Gaussian representation into a measurable object, we compared structural connectivity, boundary metrics and computation time under different TSDF meshing settings. The standard TSDF mesh of KongQueZhuYu contained 167,789 vertices and 8 connected components, with a largest-component ratio of 0.9920; smaller truncation reduced the vertex count to 147,665 but increased connected components to 20 and reduced the largest-component ratio to 0.9350. XianKeLai1 showed the same trend: smaller truncation reduced vertices from 74,753 to 66,138 but increased connected components from 6 to 12. The effects of TSDF settings on mesh morphology, connectivity and boundary behavior are shown in Fig. 7, with full mesh-structure values provided in Supplementary Table S7.

**Fig. 7 | TSDF mesh structural validation.** a, Mesh morphology, connected components and boundary-edge distributions under standard TSDF, smaller truncation and post-boundary cleanup for KongQueZhuYu and XianKeLai1. b, Vertex count, largest-component ratio, boundary-edge count and computation time under different meshing settings. c, Local zooms of leaf margins, holes and thin boundaries, showing the fragmentation risk introduced by smaller truncation. Full mesh-structure values are provided in Supplementary Table S7.

Smaller truncation reduced vertex count but increased fragmentation risk. Post-boundary cleanup preserved the number of connected components but increased mesh generation time. The current evidence supports only structural and efficiency evaluation; it does not yet show that any specific mesh variant causally reduces phenotype-measurement error.

### 3.7 Phenotype validation

Finally, we tested whether the 3D representation supports agricultural phenotype measurement. Virtual measurements were compared with manual measurements for plant height, canopy width, leaf length and leaf width; correlations and residual distributions are shown in Fig. 8, and agreement metrics are summarized in Table 4.

**Fig. 8 | Correlation between manual and virtual phenotype measurements.** a-d, Scatter plots comparing manual and virtual measurements for plant height, canopy width, leaf length and leaf width, with linear fits and 1:1 reference lines. e, Residual distributions for the four traits. f, Summary visualization of MAE, RMSE, MAPE, bias and R².

**Table 4 | Agreement between manual and virtual phenotype measurements.**

| Trait | n | MAE/cm ↓ | RMSE/cm ↓ | MAPE/% ↓ | Bias/cm | R² ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Plant height | 21 | 0.98 | 1.21 | 6.91 | 0.58 | 0.9878 |
| Canopy width | 21 | 0.86 | 0.99 | 4.50 | 0.64 | 0.9879 |
| Leaf length | 63 | 0.51 | 0.64 | 7.45 | 0.31 | 0.9738 |
| Leaf width | 63 | 0.45 | 0.64 | 9.73 | 0.38 | 0.8999 |

Plant height and canopy width showed the strongest agreement, followed by leaf length, whereas leaf width had the largest error. All traits showed a slight positive bias, indicating that the current virtual measurements slightly overestimated manual measurements. The lower R² for leaf width is associated with higher sensitivity to thin leaf boundaries, TSDF boundary expansion and landmark-placement error.

---

## 4. Discussion

### 4.1 From full-scene reconstruction to foreground-object reconstruction

The most important finding of this study is that plant-only representations for trait measurement cannot be reliably obtained by applying mask pruning after full-scene 2DGS training. During training, standard 2DGS allocates Gaussian capacity according to full-image RGB loss; background regions also generate photometric gradients and become stable parts of the representation through densification. Post-hoc pruning can only delete part of an already formed representation and cannot redirect capacity allocation during training. The failure of alpha mask loss, background opacity loss and their combination in Fig. 4 shows that constraining the opacity field alone is insufficient to prevent background learning. The main RGB supervision itself must be restricted to foreground pixels before the optimization pressure shifts to the plant object.

This conclusion is consistent with the plant-level analysis-ready representation emphasized by LCR-GS and IPENS, but ForeSplat further shows that if the goal is single-plant phenotype measurement, the object boundary should appear early in the reconstruction objective. In other words, the mask is not only a segmentation output; it is part of the 3D optimization problem for agricultural phenotyping.

### 4.2 FSAM3 is a reconstruction prior, not a segmentation endpoint

The contribution of FSAM3 is to provide stable masks for 2DGS training rather than to replace 3D optimization. FFT screening reduces blurred frames, SAM3 prompts provide semantic foreground localization and PCA refinement suppresses false-positive fragments. The manually annotated benchmark shows that FSAM3 outperforms SEEM under the data conditions of this study, but this result should not be extrapolated as a universal plant-segmentation claim. Larger annotated datasets are still needed to analyze the coupling between segmentation error and reconstruction error.

### 4.3 View quality should be modulated, not deleted

The failure of hard view filtering shows that low-quality frames in multi-view plant reconstruction are not equivalent to useless frames. Plant leaves often have angle-dependent visibility, and even a small number of weaker views may provide irreplaceable geometric coverage. Soft weighting separates geometric coverage from photometric reliability: all views are retained to maintain 3D constraints, while quality weights only modulate their contribution to RGB loss. This principle is especially important for thin leaves, occlusion and complex canopies.

### 4.4 Value of compact representation

The main benefit of the compact configuration is not higher PSNR, but a reduction in Gaussian count with little loss of foreground quality. For high-throughput phenotyping, this means lower storage, faster mesh export and easier batch processing. Mask-guided pruning is more suitable for plants than a simple opacity threshold, because primitives near leaf edges and holes may have non-negligible opacity but insufficient multi-view mask support. Pruning should remain conservative, since overly aggressive pruning may damage thin leaves, petioles and fine boundaries.

### 4.5 Phenotyping error pattern

Plant height and canopy width are determined by global extents and are therefore less sensitive to local boundary error. Leaf length depends on the main axis of an individual leaf and is moderately difficult. Leaf width depends on local cross-sectional boundaries and is most affected by reconstruction resolution, Gaussian support, TSDF fusion and landmark placement. The current results support ForeSplat for automated structural trait measurement, while identifying leaf width as the main remaining target for improvement. Future work should focus on higher-resolution acquisition, boundary-aware mesh refinement, leaf-edge uncertainty modeling and multi-operator landmark-repeatability assessment.

### 4.6 Relation to the reference studies

Unlike Arshad et al., who compared NeRF reconstruction efficiency and accuracy for plants, ForeSplat adopts an explicit 2DGS representation and constrains the target to the plant foreground. Similar to Plant3R, this study values the combination of geometric priors and Gaussian rendering, but Plant3R focuses on MASt3R-based initialization for wheat, whereas ForeSplat focuses on using foreground masks to reformulate the 2DGS training objective. Compared with IPENS and LCR-GS, ForeSplat does not extract target point clouds or individual plant subsets after reconstruction; instead, it directly generates a plant-only Gaussian representation during training. These routes are not mutually exclusive: stronger feature matching, SAM2/3 temporal propagation, LCR-GS-style multi-plant decomposition and ForeSplat's foreground-object objective could be combined for more complex greenhouse and field scenarios.

---

## 5. Limitations

First, the complete workflow was validated on 20 sequences, but systematic ablations were concentrated on representative samples and cannot fully replace large-scale mechanism validation across species and structure types. Second, the manually annotated FSAM3 benchmark is a representative subset; it supports FSAM3 as a reconstruction prior in this study but does not constitute a general plant segmentation dataset. Third, acquisition scenes were mainly indoor or semi-controlled complex backgrounds; wind-induced field motion, strong sunlight and shadow, natural soil backgrounds and intermingled multi-plant canopies require separate validation. Fourth, the current mesh experiments evaluate only structure and efficiency, and have not established causal evidence that a specific mesh variant improves phenotype accuracy. Fifth, absolute scale is recovered from a single physical reference, so scale error propagates linearly to all virtual measurements. Sixth, virtual landmarks were placed by one operator, and inter-operator variation was not evaluated; the reported error therefore includes both reconstruction error and measurement-interaction error.

---

## 6. Conclusion

We propose ForeSplat/F2DMAS, a foreground-aware 2D Gaussian Splatting workflow from multi-view image quality control to phenotype-ready plant meshes. By combining FSAM3 foreground priors, foreground track initialization, foreground RGB supervision, alpha/background opacity constraints, quality-aware soft loss weighting and mask-guided Gaussian pruning, ForeSplat reformulates standard full-scene 2DGS as plant-only foreground-object reconstruction. Experiments show that foreground RGB supervision is the key mechanism for suppressing background leakage; hard view filtering damages angular coverage of thin structures, whereas soft weighting can reduce Gaussian count while preserving quality; and the compact configuration mainly improves representation compactness and export efficiency. Validation across 20 sequences and 21 plants further shows that the workflow supports automated measurement of plant height, canopy width, leaf length and leaf width, with leaf width remaining the main unresolved challenge. Overall, ForeSplat/F2DMAS provides a reproducible and extensible technical route for low-cost, non-destructive, multi-species 3D phenotyping of potted plants under indoor or semi-controlled conditions.

---

## CRediT author contribution statement

To be completed after the author list is finalized. Recommended roles include Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Resources, Data curation, Writing - original draft, Writing - review and editing, Visualization, Supervision, Project administration and Funding acquisition.

## Data Availability

The multi-view images, FSAM3 masks, phenotype measurement tables, view-weight files and main running configurations supporting this study will be released through a project repository or data repository after manuscript acceptance or data organization. The final submission will include permanent links, access permissions, file lists, version identifiers, repository URLs, DOIs or accession numbers.

## Ethics Statement

This study involves only plant imaging and measurement. No human or animal subjects were involved.

## Conflict of Interest

The authors declare no competing interests.

## AI Use Statement

During manuscript preparation, the authors used AI-assisted tools for literature organization, structural rewriting, language polishing and Chinese-English translation. All AI-assisted content was reviewed, verified and edited by the authors. The authors take full responsibility for the manuscript content, data interpretation, citation accuracy and integrity of the published work.

## Acknowledgements

Funding sources, platform support, data-acquisition assistance and non-author contributor information will be added before submission. According to the Computers and Electronics in Agriculture author guide, acknowledgements should appear as a separate section before the references.

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
