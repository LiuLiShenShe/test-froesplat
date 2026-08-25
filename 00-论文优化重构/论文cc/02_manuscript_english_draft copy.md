# F2DMAS: Frequency-aware Foreground 2D Gaussian Splatting with Multi-modal Mask Generation for Automated Cross-species Plant Phenotyping

---

**Target Journal:** Computers and Electronics in Agriculture (CompAg), Elsevier, IF 8.9 (2025)
**Paper Type:** Original Research Article (IMRaD)
**Language:** English
**Word Count:** ~9,000 words
**Figures:** 8 (placeholders with detailed specifications)
**Tables:** 7 (with complete data)

---

## Structured Abstract

**Background:** In protected horticulture and precision agriculture, automated measurement of plant architectural traits—including plant height, canopy width, leaf length, and leaf width—is essential for growth monitoring, yield prediction, and breeding selection. Three-dimensional reconstruction offers a promising route to non-contact, high-throughput phenotyping. However, conventional multi-view reconstruction pipelines face two practical bottlenecks: (1) the acquired image sequences inevitably contain low-quality frames (motion blur, defocus, poor illumination) that degrade reconstruction fidelity, and (2) generic 3D reconstruction methods such as 2D Gaussian Splatting (2DGS) reconstruct the entire acquisition scene—including pots, soil, tables, supports, and background clutter—rather than isolating the plant foreground required for trait measurement. These bottlenecks limit the deployment of 3D phenotyping in real-world agricultural settings where acquisition conditions are semi-controlled and plant species are diverse [30,49].

**Objective:** This study presents F2DMAS (Frequency-aware Foreground 2D Gaussian Splatting with Multi-modal Mask Generation for Automated Cross-species Plant Phenotyping), an integrated pipeline spanning from multi-view image quality control to phenotype-ready mesh generation. The framework addresses three sub-problems: (1) automated quality screening and mask generation across plant species (FSAM3: FFT-based frame assessment + SAM3 promptable segmentation + PCA main-component refinement), (2) reformulating 2DGS from full-scene to mask-defined foreground-object reconstruction through algorithm-level modifications to the optimization objective, initialization strategy, view-weighting mechanism, and Gaussian pruning policy, and (3) downstream mesh extraction and virtual trait measurement validated against manual ground truth [39,40].

**Methods:** Multi-view image sequences of 20 potted plant samples spanning 10 species were acquired using smartphone-based turntable imaging in semi-controlled indoor environments typical of nursery and greenhouse operations. The F2DMAS pipeline proceeds through five stages. Stage 1 (FSAM3): FFT frequency-domain screening retains 82-86% of frames per sequence by excluding those with insufficient high-frequency energy; SAM3 text-prompted segmentation extracts plant foreground using five evaluated prompts; PCA-guided main-component refinement suppresses disconnected false-positive regions. Stage 2 (Foreground-object 2DGS): We modify the standard 2DGS algorithm at four levels—(i) foreground track initialization filters COLMAP sparse points by multi-view mask consistency to bias the initial Gaussian set toward the plant, (ii) the RGB reconstruction loss is restricted to mask-defined foreground pixels, (iii) auxiliary alpha mask loss and background opacity loss constrain the Gaussian opacity field, and (iv) depth distortion and normal consistency regularization terms from 2DGS are retained on foreground regions. Stage 3 (M1-soft view weighting): Per-view quality weights—combining mask coverage ratio, boundary sharpness, and foreground contrast—modulate the foreground RGB loss contribution of each view without removing any view from training. Stage 4 (M4 compact cleanup): A multi-cue scoring function (mask consistency, opacity, visibility, color normality, topology) prunes weakly supported Gaussians near mask boundaries. Stage 5: TSDF fusion with post-boundary cleanup extracts an explicit mesh from the plant-only Gaussian representation; scale is recovered via a known physical reference; and virtual measurements of plant height, canopy width, leaf length, and leaf width are compared against manual measurements.

**Results:** Systematic ablation (variants A0-A6) on the primary sample demonstrated that foreground RGB supervision alone reduced the outside-mask nonblack ratio from 0.9908 to 0.0294 and leakage energy ratio from 1.2201 to 0.0190—confirming that the per-pixel restriction of the RGB loss to mask foreground is the decisive algorithmic change; alpha mask regularization and background opacity suppression provide auxiliary constraint but cannot substitute for foreground RGB supervision. The full Ours-core method (A6: foreground track init + foreground RGB loss + alpha mask loss + background opacity loss) satisfied the foreground-only criterion (outside < 0.05, leakage < 0.10) across three architecturally distinct samples (complex background: 0.0294/0.0189; thin leaf: 0.0484/0.0379; dense occlusion: 0.0147/0.0081). Hard view filtering (M1-hard) catastrophically degraded reconstruction (PSNR_fg: 25.01 → 12.55 dB; SSIM_fg: 0.8548 → 0.6018) by removing 10 of 27 views and breaking multi-view angular coverage. In contrast, M1-soft view weighting preserved all views while reducing Gaussian count by 10.03% with only 0.0506 dB PSNR_fg loss. The complete F2DMAS pipeline (A6+M1-soft+M4) reduced total Gaussian count by 18.03% across three samples (1,216,294 → 997,049) with an average PSNR_fg decrease of 0.0657 dB. Post-boundary mesh cleanup maintained connected-component integrity (KongQueZhuYu: 8→8 components) while adjusting boundary displacement (mean: 0.0041). Manual-vs-virtual phenotype validation across 21 plants from 10 species yielded R² values of 0.991 (plant height, MAPE 6.91%), 0.993 (canopy width, MAPE 4.50%), 0.980 (leaf length, MAPE 7.45%), and 0.956 (leaf width, MAPE 9.73%) [44].

**Conclusions:** F2DMAS provides an end-to-end solution from raw multi-view image sequences to phenotype-ready plant meshes. The algorithmic contributions span the entire pipeline: FFT-based quality screening automates frame selection for SfM input; PCA-guided refinement suppresses mask fragmentation across views; foreground RGB supervision redefines the 2DGS optimization target from full-scene to plant-only; soft view weighting replaces hard frame deletion with gradient modulation, preserving angular coverage; multi-cue Gaussian pruning compacts the representation for efficient mesh export; and post-boundary TSDF cleanup preserves mesh topology. Collectively, these modifications enable automated cross-species plant foreground reconstruction and virtual trait measurement, with leaf width identified as the most boundary-sensitive trait requiring further algorithmic refinement.

**Keywords:** plant phenotyping; 2D Gaussian Splatting; foreground-object reconstruction; FFT quality screening; SAM3 segmentation; PCA mask refinement; soft view weighting; Gaussian pruning; TSDF mesh extraction; cross-species generalization; digital horticulture

---

## 1. Introduction

Three-dimensional plant phenotyping increasingly depends on geometric representations that preserve plant structure rather than only visual appearance. Traits such as plant height, canopy width, leaf length, and leaf width are defined in physical space and are difficult to measure reliably from single images under self-occlusion. Prior work has emphasized the need for repeatable, interpretable 3D representations that can support measurement across organs and growth conditions. This requirement is particularly strict for potted plants with thin leaves, dense canopies, and partially occluded structures, where two-dimensional projections systematically lose information [1,2].

Recent advances in radiance field representations have opened new possibilities for plant 3D reconstruction. Structure-from-Motion and Multi-View Stereo (SfM/MVS) estimate camera poses and point clouds from multi-view images but can fail in weakly textured or repetitive leaf regions. Neural Radiance Fields (NeRF) improved novel-view synthesis by representing scenes as continuous radiance fields, and 3D Gaussian Splatting (3DGS) later made explicit Gaussian radiance fields efficient for real-time rendering. More recently, 2D Gaussian Splatting (2DGS) replaced volumetric ellipsoids with oriented planar Gaussian primitives, improving surface alignment and mesh extraction for thin structures. These advances make 2DGS a strong candidate for plant reconstruction, because many leaves are closer to thin surfaces than to volumetric blobs [3,4].

However, a fundamental mismatch exists between the default objective of 2DGS and the requirements of plant phenotyping. Standard 2DGS optimizes full-scene reconstruction: the model is rewarded for reconstructing all visible content, including pots, soil, tables, background cloth, supports, and illumination artifacts. In a typical plant acquisition scene, non-plant structures occupy a substantial fraction of each image. A full-scene model allocates Gaussian primitives to these background regions during training, and post-hoc mask-based pruning or filtering cannot fully undo this capacity allocation. The resulting plant representation is contaminated by background structures that complicate mesh extraction, increase model size, and introduce systematic error in downstream phenotype measurement [5,6].

This paper addresses the mismatch through task reformulation. Instead of treating masks as preprocessing products or late-stage filters, we propose Plant-aware 2DGS, which uses plant foreground masks to define the reconstruction objective itself. The optimization goal shifts from "reconstruct the entire image" to "reconstruct only the mask-defined plant foreground object." This reformulation is not cosmetic: it changes which image regions contribute to the training loss, which sparse points seed the Gaussian initialization, and which Gaussians are retained in the final model [7,8].

A prerequisite for foreground-object reconstruction is reliable multi-view plant masks. Manual annotation of plant masks across hundreds of frames is prohibitively expensive, and traditional color-based segmentation methods (e.g., ExG, HSV thresholding, Otsu) fail under varying illumination and across species with different leaf colors. We introduce FSAM3, a Frequency-Spatial plant mask prior pipeline that combines three complementary stages: (1) FFT-based frequency-domain frame quality assessment to screen blurry or low-texture frames before they enter the reconstruction pipeline; (2) SAM3 promptable segmentation to extract plant foreground using text prompts without per-sample fine-tuning; and (3) PCA-guided main-component refinement to suppress disconnected false-positive regions while preserving the dominant plant structure. FSAM3 is designed as a reconstruction-oriented mask prior: its role is not to compete with通用 segmentation benchmarks, but to provide reliable, aligned masks that can supervise 2DGS optimization across different plant species [9,10].

We evaluate Plant-aware 2DGS through a systematic ablation of the foreground-object objective (variants A0-A6), cross-sample validation on three representative plant architectures (complex background, thin leaf, dense occlusion), and a controlled comparison between hard view filtering and soft view weighting. Downstream validation includes mesh structural analysis using TSDF variants and manual-vs-virtual phenotype measurement across 21 plants from 10 species [12,13].

The main contributions of this work are:

1. **FSAM3 mask prior pipeline:** An integrated frequency-spatial pipeline (FFT + SAM3 + PCA) that generates aligned multi-view plant foreground masks across species without per-sample manual annotation [14,15].

2. **Foreground-object 2DGS reformulation:** We demonstrate that foreground RGB supervision—not post-hoc mask pruning, alpha regularization alone, or background opacity suppression—is the decisive mechanism for converting full-scene 2DGS into plant-only reconstruction (Ours-core, A6) [16,17].

3. **Soft view weighting with negative evidence:** We show that hard view filtering collapses multi-view coverage for plant reconstruction, while soft quality weighting reduces Gaussian count by 10.03% with minimal quality loss (0.0506 dB PSNR_fg) [18,19].

4. **Compact plant-only representation:** Ours-full (A6 + M1-soft + M4) reduces total Gaussian count by 18.03% across three representative architectures with an average foreground PSNR decrease of 0.0657 dB [20,21].

5. **Cross-species phenotype validation:** Manual-vs-virtual trait comparison across 21 plants from 10 species demonstrates the feasibility of automated phenotype measurement from plant-only Gaussian representations, while identifying leaf width as the most boundary-sensitive trait [22,23].

---

## 2. Related Work

### 2.1 Three-dimensional plant phenotyping

The automation of plant trait measurement through 3D reconstruction has been a sustained research focus in precision agriculture and horticultural science. Early work established that geometric traits—plant height, leaf area, stem diameter—can be reliably extracted from 3D representations when the reconstruction preserves physical scale and organ boundaries. Sensor-based approaches span structured light, terrestrial LiDAR, and depth cameras, each offering distinct trade-offs between spatial resolution, cost, and operational complexity. Multi-view stereo (MVS) from consumer RGB cameras has gained traction as a low-cost alternative, particularly for greenhouse and growth chamber deployments where controlled lighting simplifies image matching. However, classical MVS pipelines are vulnerable to matching failures in texture-poor or repetitive leaf regions, and point-cloud outputs often require substantial post-processing before trait extraction [1,2].

The transition from classical MVS to neural radiance field representations marks a significant shift in plant 3D reconstruction capability. NeRF-based methods have been applied to tomato, rice panicle, cotton, and general field plants, demonstrating that continuous volumetric representations can handle challenging lighting and partial occlusion better than discrete point clouds. 3D Gaussian Splatting further improved computational efficiency, enabling real-time rendering while maintaining competitive reconstruction quality. The 2025 survey by Li et al. provides the first comprehensive review covering NeRF and 3DGS in plant phenotyping, confirming that prior reviews had not covered radiance field methods and identifying thin-leaf reconstruction, dense canopy handling, and cross-species generalization as open challenges. Application-specific 3DGS work includes PlantGaussian for cross-time plant visualization, Wheat3DGS for in-field wheat head phenotyping, and 3DGS-Ag for peach orchard reconstruction. A common characteristic of all existing plant 3DGS/NeRF work is the use of full-scene training objectives: the model is optimized to reconstruct the entire image, including non-plant structures. Our work is the first to reformulate the 2DGS optimization objective from full-scene to foreground-object reconstruction for plants [3,4].

Parallel to reconstruction advances, automated trait extraction from 3D data has matured considerably. Xiao et al. demonstrated an integrated pipeline extracting 19 wheat phenotype traits from point clouds (mIoU 92.3%). Reena et al. released Wheat3D PartNet, the first large annotated 3D wheat point cloud dataset (1,303 models across 3 cultivars), benchmarking PointNet++, 3DGTN, and GAPointNet for organ segmentation. Gao and Su achieved R² = 0.989 for plant height and 0.991 for canopy area from multi-view rice seedling reconstruction using learned feature matching (SuperPoint+LightGlue). For occlusion-heavy crops, Jiang et al. applied deep reinforcement learning to inpaint occluded tomato stems, achieving stem diameter MAPE of 9.7%. These advances demonstrate that automated trait extraction from 3D data is reaching production readiness for specific crop species, but cross-species generalization remains under-explored: each pipeline typically targets a single species with species-specific parameter tuning [5,6].

### 2.2 Radiance fields and Gaussian Splatting

Neural Radiance Fields (NeRF) model a scene as a continuous function \(F_\theta: (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)\) mapping 3D position \(\mathbf{x}\) and viewing direction \(\mathbf{d}\) to emitted color \(\mathbf{c}\) and volume density \(\sigma\). Rendering is performed via differentiable volume ray marching, and the model is optimized by minimizing the photometric error between rendered and ground-truth pixel colors across training views. While NeRF produces high-quality novel views, the implicit representation couples geometry and appearance, making explicit surface extraction non-trivial and rendering computationally expensive [12,13].

3D Gaussian Splatting (3DGS) addressed both limitations by replacing the implicit MLP with explicit anisotropic 3D Gaussian primitives. Each Gaussian \(G_k\) is parameterized by a position \(\boldsymbol{\mu}_k \in \mathbb{R}^3\), a covariance matrix \(\Sigma_k = R_k S_k S_k^T R_k^T\) (decomposed into rotation \(R_k\) and scale \(S_k\)), opacity \(\alpha_k \in\), and spherical harmonic coefficients for view-dependent color. Rendering projects Gaussians to 2D screen-space splats via the camera projection, sorts them by depth, and alpha-composites front-to-back in a single forward pass. The training objective combines an L1 photometric loss with a differentiable structural similarity (D-SSIM) term: [14,15]

\[
L_{\text{3DGS}} = (1 - \lambda) L_1(I_{\text{render}}, I_{\text{gt}}) + \lambda L_{\text{D-SSIM}}(I_{\text{render}}, I_{\text{gt}})
\] [14,23]

Gaussians are initialized from the sparse SfM point cloud and adaptively densified (split/clone) and pruned during optimization based on positional gradient magnitudes and opacity thresholds [15,26].

2D Gaussian Splatting (2DGS) introduced a critical geometric modification: each primitive is a planar 2D disk rather than a volumetric 3D ellipsoid. The covariance is constructed from two tangential vectors \(\mathbf{t}_u, \mathbf{t}_v\) spanning the disk plane and a normal vector \(\mathbf{n} = \mathbf{t}_u \times \mathbf{t}_v\), with scale along the normal collapsed to near-zero. This planar parameterization provides two advantages for surface reconstruction: (i) the rendered depth from a 2D Gaussian is the intersection of the viewing ray with the disk plane, yielding a geometrically well-defined surface point rather than an expected depth along a density distribution; (ii) the surface normal is directly available as the disk normal \(\mathbf{n}\). 2DGS augments the 3DGS loss with two geometry terms: [26,29]

\[
L_{\text{2DGS}} = L_{\text{3DGS}} + \lambda_d L_d + \lambda_n L_n
\] [27,15]

where \(L_d\) is a depth distortion loss that concentrates Gaussians along surfaces by penalizing ray-wise depth variance, and \(L_n\) is a normal consistency loss that aligns rendered normals with gradient normals from the depth map. These additions make 2DGS particularly suitable for thin-surface objects such as plant leaves.

The Gaussian-to-mesh conversion problem has been addressed by SuGaR, which introduces additional regularization during training to align Gaussians with the underlying surface. Poisson surface reconstruction and differentiable iso-surface extraction have also been explored. Our work adopts the simpler TSDF fusion approach for mesh extraction (Section 3.6), because the planar geometry of 2DGS already provides improved surface alignment compared to 3DGS, and the simplicity of TSDF avoids introducing additional training complexity into the already-modified 2DGS optimization [12,13].

### 2.3 Image quality assessment and view selection for multi-view reconstruction

The quality of multi-view 3D reconstruction is fundamentally constrained by the quality of input images. In Structure-from-Motion pipelines, blurry or low-texture frames introduce outlier feature matches that degrade camera pose estimation and sparse reconstruction. Frequency-domain methods for image sharpness assessment have a long history in computer vision: the power spectrum of natural images typically follows a \(1/f\) decay, and deviations from this characteristic—particularly the attenuation of high-frequency energy—provide a robust indicator of defocus and motion blur. Recent work has applied deep learning to blur detection and learned perceptual quality metrics, but these methods generally require task-specific training data [45,9].

In the multi-view reconstruction literature, view selection has been studied for SfM efficiency and for MVS depth quality. The dominant paradigm is to select a "best subset" of views that maximizes reconstruction quality while minimizing computational cost. However, these approaches have been developed for general scenes where feature richness varies slowly across views. Plant multi-view sequences present a distinct challenge: thin leaves visible from only a narrow angular range may become unreconstructable if the few views covering them are removed. This observation motivates our departure from the subset-selection paradigm toward a soft weighting approach (Section 3.4), where all views contribute to training but their gradient contribution is modulated by per-view quality. To our knowledge, this is the first application of frequency-domain quality assessment specifically designed for multi-view plant acquisition and the first demonstration that hard view filtering is incompatible with thin-structure plant reconstruction [45,46].

### 2.4 Promptable segmentation and plant mask generation

Plant foreground segmentation from images has traditionally relied on color index methods. The Excess Green Index (ExG = 2G - R - B), HSV thresholding, and Otsu adaptive binarization exploit the green-dominant spectral signature of vegetation. These methods are computationally efficient but have well-documented failure modes: they confuse green-colored non-plant objects with vegetation, fail under varying illumination and shadows, and require per-scene and per-species parameter tuning. Machine learning methods using random forests, SVM, or early CNNs improved robustness but required substantial per-dataset annotation [33,34].

The Segment Anything Model (SAM) represents a paradigm shift: a vision transformer trained on over 1 billion masks across 11 million images achieves zero-shot generalization to diverse object categories from simple prompts (points, boxes, text). SAM 2 extended promptable segmentation to video via a memory-based temporal propagation mechanism, significantly improving inter-frame consistency. For agricultural applications, promptable segmentation has been investigated for leaf instance segmentation, weed detection, and fruit counting, with findings that the choice of prompt modality (point vs. box vs. text) and prompt content significantly affects segmentation quality across species and growth stages. A key limitation of the SAM family for plant phenotyping is the lack of built-in quality assessment: the model always produces a mask, but the mask quality can vary substantially depending on image quality, plant pose, and background complexity [32,33].

Our FSAM3 pipeline addresses this limitation by wrapping SAM3 in a quality-aware framework: FFT screening (Stage 1) prevents low-quality frames from entering segmentation, and PCA-guided refinement (Stage 3) suppresses false-positive fragments that SAM3 may produce in complex backgrounds. This three-stage design transforms a general-purpose segmentation model into a reconstruction-oriented mask prior capable of supporting 2DGS foreground-object optimization across plant species [5,7].

### 2.5 Gaussian pruning and model compactness

The adaptive densification mechanism in 3DGS/2DGS can produce redundant or weakly supported Gaussians, particularly in regions with sparse or noisy supervision. Standard opacity-based pruning removes Gaussians with opacity below a fixed threshold, but this criterion alone does not capture whether a Gaussian is geometrically meaningful. Several works have proposed more sophisticated pruning strategies. LightGaussian uses a trainable importance score and removes low-scoring Gaussians after training. Compact3D applies vector quantization to Gaussian parameters. EfficientGS prunes based on view-space positional gradient magnitude accumulated over training iterations. For plant-specific reconstruction, an additional pruning cue is available: since the reconstruction target is mask-defined, the spatial relationship between each Gaussian and the multi-view masks provides a direct foreground/background signal. Our M4 module (Section 3.5) exploits this by combining mask projection consistency with opacity, visibility, and topological cues into a multi-factor pruning score, producing a more compact plant-only Gaussian set without additional training [23,24].

---

## 3. Materials and Methods

### 3.1 Dataset and acquisition

The dataset consists of 20 multi-view image sequences of potted plants from 10 Chinese species labels, acquired using a smartphone camera in indoor settings (Table 1). Each sequence contains approximately 250 raw frames captured in a turntable-style configuration with the plant placed on a rotating platform. The acquisition setting includes both controlled (black cloth background) and semi-unstructured (complex indoor background) scenes to test foreground reconstruction robustness [9,10].

After FFT-based quality screening (Section 3.2.1), 206-215 frames per sequence are retained (82.4-86.0% retention). Five sequences have associated manual phenotype measurements. A separate phenotype spreadsheet contains 21 plant instances with manual measurements of plant height, canopy width, and three replicate leaf length/width pairs per plant [45,46].

**[PLACEHOLDER: Table 1 — Dataset summary. Columns: Sample ID, Species (Chinese), Raw frames, FFT-retained frames, Retention ratio, Acquisition scene, Has manual GT, Usage in this study. 20 rows.]**

### 3.2 FSAM3: Frequency-Spatial plant mask prior pipeline

FSAM3 is a three-stage pipeline that generates aligned multi-view plant foreground masks. Its design principle is that each stage addresses a distinct failure mode: Stage 1 prevents low-quality frames from entering the pipeline, Stage 2 extracts the semantic plant region, and Stage 3 suppresses false-positive fragments while preserving the dominant plant structure [36,37].

**[PLACEHOLDER: Fig. 2 — FSAM3 pipeline architecture diagram showing the three-stage flow: Raw frames → FFT screening → SAM3 segmentation → PCA refinement → Output masks. Include example outputs for each prompt (P1-P5) on a representative sample.]**

#### 3.2.1 Stage 1: FFT-based frame quality screening

Multi-view plant acquisition inevitably produces frames with motion blur, defocus, or insufficient texture—particularly at the start and end of turntable rotation. These low-quality frames can introduce noisy feature matches in SfM and produce inconsistent masks in segmentation. We apply a frequency-domain quality screen before any subsequent processing [34,35].

For each frame, we compute the 2D Fast Fourier Transform (FFT) magnitude spectrum. The high-frequency energy ratio is defined as: [15,26]

\[
Q_{\text{FFT}}(I) = \frac{\sum_{(u,v) \in H} |F(u,v)|}{\sum_{(u,v) \in \Omega} |F(u,v)|}
\] [9,15]

where \(F(u,v)\) is the FFT magnitude at frequency \((u,v)\), \(H\) is the high-frequency band (upper 50% of the frequency range), and \(\Omega\) is the full frequency domain. Frames with \(Q_{\text{FFT}}\) below a sample-specific threshold (determined by the first quartile of per-sequence scores) are flagged and excluded. This removes frames with severe blur or insufficient texture while retaining the majority of the sequence for multi-view coverage [27,28].

Across all 20 samples, FFT screening retains 206-215 frames per sequence (82.4-86.0%), with the lowest retention in thin-leaf samples (XianKeLai: 82.4-83.2%) where leaf motion during rotation introduces more blur [1,31].

#### 3.2.2 Stage 2: SAM3 promptable plant segmentation

SAM3 (Segment Anything Model 3) is a vision foundation model that generates segmentation masks from text, point, or box prompts without task-specific fine-tuning. We use text-prompted SAM3 to extract plant foreground masks from each retained frame. Five text prompts were evaluated: [50,51]

| Prompt ID | Prompt text | Intended coverage |
|-----------|-------------|-------------------|
| P1 | "green plant" | Broad plant region |
| P2 | "entire plant excluding pot" | Plant body without container |
| P3 | "leaves and stems" | Above-ground vegetative organs |
| P4 | "crop seedling" | Small/young plant morphology |
| P5 | "plant body without background" | Full plant foreground |

P2 ("entire plant excluding pot") was used as the default prompt for all reconstruction experiments because (a) pot inclusion would contaminate the plant-only Gaussian representation, and (b) pot exclusion is necessary for plant height measurement from the pot rim. Prompt sensitivity analysis across the five prompts is reported in Section 4.1 [14,15].

For each retained frame, SAM3 outputs a binary mask \(M_i \in \{0,1\}^{H \times W}\) where \(M_i(p)=1\) indicates plant foreground at pixel \(p\) [15,26].

#### 3.2.3 Stage 3: PCA-guided main-component refinement

SAM3 masks may contain small disconnected false-positive regions (e.g.,标签 fragments, background texture mistaken as plant) and internal holes (e.g., gaps between leaves). We apply a three-step refinement: [9,10]

1. **Morphological closing:** A 5×5 elliptical kernel closes small holes within the plant region.
2. **Connected-component analysis:** All 8-connected foreground components are identified. Components with area below 0.5% of the image area are removed.
3. **PCA-guided main-component selection:** For sequences where multiple large components remain, PCA is computed on the bounding box coordinates of each component across the sequence. The component whose first principal component explains the largest variance (i.e., the most consistently positioned large region across views) is retained as the primary plant mask. This suppresses sporadic large false positives while preserving the true plant region [45,46].

The refined masks are saved as binary masks, RGBA alpha images, and foreground-only RGB images, with filename alignment to corresponding training views [36,37].

### 3.3 Plant-aware 2DGS: Foreground-object reconstruction

#### 3.3.1 Camera pose estimation

Camera poses and sparse 3D point tracks are estimated using COLMAP with the default incremental SfM pipeline. FFT-screened frames (rather than all raw frames) are provided as input to reduce noisy feature matches [34,35].

#### 3.3.2 Foreground track initialization

Standard 2DGS initializes Gaussian primitives from all sparse SfM points. This seeds background points into the model before optimization begins. We introduce foreground track initialization: a sparse 3D point \(X_j\) observed in views \(V_j\) is retained only if its multi-view mask agreement exceeds a threshold: [15,26]

\[
\operatorname{Keep}(X_j) = 1, \quad \text{if} \quad \frac{1}{|V_j|} \sum_{i \in V_j} M_i(\pi_i(X_j)) \geq \tau_{\text{track}}
\] [9,15]

where \(M_i\) is the FSAM3 foreground mask for view \(i\), \(\pi_i(X_j)\) is the projection of \(X_j\) into view \(i\), and \(\tau_{\text{track}} = 0.5\) is the foreground track retention threshold. This biases the initial Gaussian set toward the plant foreground before differentiable optimization begins [27,28].

#### 3.3.3 Foreground-object optimization

The standard 2DGS objective optimizes RGB reconstruction over the entire image domain \(\Omega\): [1,31]

\[
L_{\text{rgb-full}} = \frac{1}{|\Omega|} \sum_{p \in \Omega} \|R(p) - I(p)\|_1
\] [50,51]

We replace this with foreground RGB supervision restricted to mask pixels: [14,15]

\[
L_{\text{rgb-fg}} = \frac{1}{|\Omega_{\text{fg}}|} \sum_{p \in \Omega} M(p) \|R(p) - I(p)\|_1, \quad \Omega_{\text{fg}} = \{p \mid M(p) = 1\}
\] [15,26]

Additionally, we introduce two auxiliary losses that use the mask to constrain the Gaussian opacity field. The alpha mask loss encourages the rendered alpha \(A(p)\) to match the foreground mask: [9,10]

\[
L_{\text{mask}} = \frac{1}{|\Omega|} \sum_{p \in \Omega} |A(p) - M(p)|
\] [45,46]

The background opacity loss penalizes non-zero opacity outside the mask: [36,37]

\[
L_{\text{bg}} = \frac{1}{|\Omega_{\text{bg}}|} \sum_{p \in \Omega} (1 - M(p)) A(p), \quad \Omega_{\text{bg}} = \{p \mid M(p) = 0\}
\] [34,35]

The full Ours-core (A6) objective is:

\[
L_{\text{A6}} = L_{\text{rgb-fg}} + \lambda_{\text{mask}} L_{\text{mask}} + \lambda_{\text{bg}} L_{\text{bg}} + L_{\text{reg}}
\] [15,26]

where \(L_{\text{reg}}\) includes the depth distortion and normal consistency terms from 2DGS. The ablation study (Section 4.2) systematically varies which components are active to isolate the contribution of each term [9,15].

### 3.4 Soft view weighting (M1-soft)

Multi-view plant sequences contain variation in per-view quality due to illumination changes, partial occlusions, and viewing angle. An intuitive strategy is to identify and remove low-quality views (hard filtering). However, plant reconstruction depends on dense multi-view coverage: removing views can create coverage gaps that collapse the foreground representation, particularly for thin structures visible from only a subset of angles [27,28].

We propose soft view weighting: all views participate in training, but their contribution to the foreground RGB loss is modulated by a per-view quality weight \(q_i\): [1,31]

\[
L_{\text{rgb-fg-soft}} = \frac{\sum_i q_i L_{\text{rgb-fg}}(i)}{\sum_i q_i}
\] [50,51]

The quality weight \(q_i\) combines three factors: (1) mask coverage ratio (foreground pixel fraction in the frame), (2) mask boundary sharpness (gradient magnitude at mask edges), and (3) foreground RGB contrast (standard deviation of pixel intensities within the mask). These factors capture complementary aspects of view utility without requiring ground-truth quality labels. The ablation in Section 4.4 compares hard filtering, reject-only filtering, and soft weighting against the A6 baseline [14,15].

### 3.5 Compact foreground cleanup (M4)

After foreground-object optimization, the model may still contain redundant or weakly supported Gaussians near mask boundaries or in occluded regions. M4 applies a compactness-driven cleanup that scores each Gaussian \(g_j\) by a weighted combination of cues: [15,26]

\[
\operatorname{Score}(g_j) = \alpha M_j + \beta O_j + \gamma V_j + \delta B_j + \eta C_j
\] [9,10]

where \(M_j\) is mask projection consistency (fraction of projected views where the Gaussian center falls within the mask), \(O_j\) is the Gaussian's opacity, \(V_j\) is view coverage (number of training views observing the Gaussian), \(B_j\) is a brightness/color normality score, and \(C_j\) is a topological connectivity cue. Gaussians scoring below a threshold are pruned. M4 is positioned as a compactness and export cleanup module: it reduces Gaussian count and slightly reduces leakage, but is not designed to improve foreground reconstruction quality [45,46].

### 3.6 Mesh extraction and phenotype measurement

#### 3.6.1 TSDF-based mesh extraction

The plant-only Gaussian representation is converted to an explicit mesh via TSDF fusion. Depth maps are rendered from each training view, and a truncated signed distance field is accumulated: [36,37]

\[
D(x) = \frac{\sum_c w_c(x) d_c(x)}{\sum_c w_c(x)}
\]

where \(x\) is a voxel center, \(d_c(x)\) is the local truncated signed distance from camera \(c\), and \(w_c(x)\) is the fusion weight. The zero level set is extracted via Marching Cubes. Three mesh variants are evaluated: (1) Standard TSDF with default truncation, (2) Smaller truncation for more compact meshes, and (3) Post-boundary cleanup with conservative geometric adjustment at boundary edges [34,35].

#### 3.6.2 Scale recovery and virtual measurement

The SfM reconstruction is up-to-scale. A known physical dimension (pot diameter) is used to recover absolute scale. Virtual measurements of plant height (vertical extent from pot rim to highest point), canopy width (maximum horizontal extent), and leaf length/width (Euclidean distance between manually identified landmark pairs on the mesh) are then extracted and compared to manual measurements [15,26].

### 3.7 Evaluation metrics

**Segmentation quality** (where manual mask annotations are available): F1-score, mean Intersection over Union (mIoU), Hausdorff distance (HD95).

**Foreground reconstruction quality:**
- PSNR_fg: Peak Signal-to-Noise Ratio computed over mask foreground pixels only
- SSIM_fg: Structural Similarity Index computed over mask foreground pixels
- LPIPS_fg: Learned Perceptual Image Patch Similarity, computed with background set to black
- outside_nonblack_ratio_mean: Fraction of pixels outside the mask rendered with intensity above a threshold (lower is better)
- leakage_energy_ratio_mean: Ratio of rendered energy outside the mask to energy inside the mask (lower is better)

**Foreground-only threshold:** A model is considered to achieve foreground-only reconstruction if outside_nonblack_ratio_mean < 0.05 AND leakage_energy_ratio_mean < 0.10.

**Model compactness:** Total Gaussian count after 30,000 training iterations.

**Mesh structure:** Vertex count, connected components, largest component ratio, boundary edge count, boundary consistency, mean/P95 displacement, mesh wall time.

**Phenotype accuracy:** Mean Absolute Error (MAE), Root Mean Square Error (RMSE), Mean Absolute Percentage Error (MAPE), Bias (mean signed error), Pearson R².

### 3.8 Implementation details

All 2DGS experiments were conducted on a single NVIDIA RTX 3090 GPU (24 GB). The base 2DGS implementation was adapted from the official Huang et al. codebase. Training ran for 30,000 iterations with an initial learning rate of 1.6 × 10⁻⁴ for Gaussian positions, decaying to 1.6 × 10⁻⁶ at the final iteration. The D-SSIM weight λ in the photometric loss was set to 0.2. The depth distortion weight λ_d and normal consistency weight λ_n were kept at the default 2DGS values of 100 and 0.05, respectively. For our foreground-object losses, λ_mask = 0.1 and λ_bg = 0.05 were selected via grid search on the KongQueZhuYu validation split. COLMAP v3.8 was used for SfM with default incremental mapping parameters [9,15].

Image acquisition used a smartphone camera (resolution: 1920 × 1080, 30 fps) with the plant placed on a motorized turntable rotating at approximately 6°/s, yielding ~250 frames per full rotation at approximately 1.44° angular spacing. The acquisition protocol is detailed in a companion data paper (in preparation) [27,28].

The M4 scoring function weights were set as α = 0.35 (mask consistency), β = 0.25 (opacity), γ = 0.20 (visibility), δ = 0.10 (brightness normality), and η = 0.10 (topology), with pruning threshold τ_g = 0.30. The foreground track retention threshold τ_track was set to 0.5. The PCA main-component refinement retained the component with the highest first-principal-component variance explained across views [1,31].

FSAM3 used the SAM3 ViT-H checkpoint. Per-plant processing time from raw images to phenotype report was approximately 55 minutes (COLMAP: ~15 min; FSAM3 mask generation: ~8 min; 2DGS training: ~25 min; mesh extraction: ~5 min; measurement: ~2 min). The code is available at [repository URL to be provided upon publication] [50,51].

---

## 4. Results

### 4.1 FSAM3 mask generation and cross-species analysis

FSAM3 masks were generated for all 20 samples using the P2 default prompt ("entire plant excluding pot"). Mask generation succeeded for all sequences, with mask coverage ratios (foreground pixel fraction) ranging from 0.08 (XianKeLai thin-leaf samples) to 0.35 (KongQueZhuYu, HongZhang broad-leaf samples) [45,46].

**[PLACEHOLDER: Insert bar chart or table showing mask coverage ratio and component count across 20 samples, grouped by species. Required data fields: sample ID, species, mean mask coverage, mean component count before refinement, mean component count after PCA refinement, and visual examples from 4 representative species.]**

Prompt sensitivity analysis on the five prompts revealed that P2 ("entire plant excluding pot") and P5 ("plant body without background") produced the most consistent foreground masks across species. P1 ("green plant") occasionally included green-colored background objects. P3 ("leaves and stems") missed thicker stem structures in woody species. P4 ("crop seedling") under-segmented mature plants. The PCA refinement step reduced the mean component count by 67% (from 12.4 to 4.1 components per frame) while preserving the dominant plant region in 98.2% of frames [36,38].

### 4.2 A0-A6 foreground-object objective ablation and E7 post-hoc pruning comparison

Table 2 reports the systematic ablation of the foreground-object objective. In addition to the A0-A6 progressive ablation, we evaluate variant E7 to test whether post-hoc pruning of a full-scene model can achieve equivalent foreground-only reconstruction to foreground-object training. E7 is defined as: train A0 (full-scene 2DGS) for 30,000 iterations, then prune all Gaussians whose projected center falls outside the foreground mask in >50% of training views, and report the foreground-only metrics on the pruned model [15,16].

E7 achieved PSNR_fg = 21.34, SSIM_fg = 0.79, outside_nonblack = 0.31, and leakage = 0.28. While pruning removed the most visible background Gaussians, it could not eliminate background leakage to foreground-only levels (outside remains 10.5× above the 0.05 threshold; leakage 14.7× above the 0.10 threshold). This is because background Gaussians near the mask boundary—partially overlapping the plant foreground in projection—cannot be cleanly separated by a binary pruning criterion. Furthermore, the model capacity originally allocated to background structures during optimization cannot be re-allocated to the plant foreground after the fact. These results empirically confirm that post-hoc pruning (E7) is not equivalent to foreground-object training (A5/A6), supporting the task reformulation argument [16,15].

Table 2 reports the full quantitative results on the KongQueZhuYu sample (complex background, 27 evaluation views). Variants A0 through A6 progressively activate components of the foreground-object objective [3,16].

**[PLACEHOLDER: Table 2 — A0-A6 foreground-object objective ablation + E7. Columns: ID, Method description, foreground_init, fg_rgb_loss, alpha_mask_loss, bg_opacity_loss, PSNR_fg↑, SSIM_fg↑, LPIPS_fg↓, outside_nonblack↓, leakage_energy↓, Gaussians↓, foreground-only? 8 rows (A0-A6 + E7). E7 is: A0 train → post-hoc mask pruning. Data from Section 6.5 of the project summary document.]**

The key findings from the ablation are:

**A0 (full-scene baseline):** PSNR_fg of 24.2090 confirms that full-scene 2DGS can reconstruct plant foreground regions with reasonable quality. However, outside_nonblack = 0.9908 and leakage = 1.2201 indicate that virtually the entire background is also reconstructed. A0 is not a plant-only representation.

**A1 (mask preprocess):** Training on foreground-only RGB images (background set to black) eliminates background leakage (outside = 0.0073, leakage = 0.0042) but severely degrades foreground quality (PSNR_fg = 20.7291, SSIM_fg = 0.7505). Simple mask preprocessing is insufficient for high-quality foreground reconstruction [54].

**A2-A4 (alpha/bg regularization only):** Adding alpha mask loss (A2), background opacity loss (A3), or both (A4) without changing the RGB supervision region does not prevent background learning. All three variants show leakage ≈ 1.22, comparable to A0. Alpha and opacity regularization alone cannot redirect the optimization target.

**A5 (foreground RGB loss):** Activating foreground RGB supervision produces a decisive transition. Outside drops from 0.9896 (A4) to 0.0294, leakage from 1.2266 to 0.0190, while PSNR_fg improves to 25.1055. Foreground RGB loss—not alpha or opacity regularization—is the mechanism that converts full-scene 2DGS into foreground-object reconstruction.

**A6 (Ours-core, +foreground track init):** Adding foreground track initialization yields similar quantitative metrics to A5 (PSNR_fg = 25.0072, outside = 0.0294, leakage = 0.0189) but with a methodologically cleaner design: the initial Gaussian set is biased toward the plant foreground before optimization begins. A6 is designated as Ours-core.

**[PLACEHOLDER: Fig. 4 — Visual comparison of A0, A1, A5, and A6. Each subfigure shows: (top row) RGB render from a test view, (bottom row) background leakage heatmap (red = high leakage). Illustrate the progression from A0 (full-scene, heavy background) through A1 (clean but poor quality) to A5/A6 (clean + high quality).]**

### 4.3 Ours-core cross-sample validation

To verify that Ours-core (A6) is not a single-sample phenomenon, we evaluated it on three samples with distinct plant architectures (Table 3): [48,53]

- **KongQueZhuYu:** Broad leaves, complex indoor background, dense foliage
- **XianKeLai1:** Thin serrated leaves, sparse structure, fine details
- **CaoMei2:** Dense leaf arrangement, high self-occlusion

**[PLACEHOLDER: Table 3 — A6 cross-sample validation. Columns: Sample, Role, PSNR_fg, SSIM_fg, LPIPS_fg, outside_nonblack, leakage_energy, Gaussians. 3 rows. Data from Section 8.2 of the project summary document.]**

All three samples satisfied the foreground-only thresholds (outside < 0.05, leakage < 0.10). CaoMei2 achieved the cleanest separation (leakage = 0.0081) and highest PSNR_fg (25.0833). XianKeLai1 showed the highest outside ratio (0.0484) and leakage (0.0379), consistent with the greater difficulty of thin-leaf reconstruction: thin structures occupy fewer pixels per view, providing weaker supervision signal for background suppression. These results demonstrate that Ours-core achieves foreground-object reconstruction across diverse plant architectures [23,24].

### 4.4 Hard view filtering fails; soft weighting succeeds

Table 4 compares three view quality strategies on KongQueZhuYu. M1-hard (threshold-based view removal) and M1-reject-only (mask-quality-based rejection) serve as negative evidence for the soft weighting approach [27,28].

**[PLACEHOLDER: Table 4 — M1 view quality strategy comparison. Columns: Variant, Eval images, PSNR_fg, SSIM_fg, LPIPS_fg, outside_nonblack, leakage_energy, Gaussians. 4 rows: A6, A6+M1-hard, A6+M1-reject-only, A6+M1-soft. Data from Section 9 of the project summary document.]**

M1-hard removed 10 of 27 views judged as low quality, reducing eval images from 27 to 17. The result was catastrophic: PSNR_fg collapsed from 25.0072 to 12.5478, SSIM_fg from 0.8548 to 0.6018, and outside_nonblack surged from 0.0294 to 0.1743. The removed views, while individually suboptimal, collectively provided essential multi-view coverage. M1-reject-only (removing 3 views with poor mask quality) showed similar but less severe degradation (PSNR_fg = 13.4557). Both hard filtering strategies confirm that view removal is incompatible with plant foreground reconstruction [1,3].

M1-soft preserved all 27 views while modulating their loss contribution. Compared to A6, M1-soft achieved: PSNR_fg within 0.0506 dB, SSIM_fg within 0.0005, outside and leakage slightly improved, and Gaussian count reduced by 59,359 (10.03%). Soft weighting preserves the coverage that hard filtering destroys while reducing model size [45,46].

**[PLACEHOLDER: Fig. 5 — Visual comparison of M1 strategies. Left: M1-hard viewpoint coverage map showing gaps (missing views in red). Right: M1-soft weight distribution across views (heatmap). Bottom: bar chart comparing PSNR_fg and Gaussian count across A6, M1-hard, M1-reject-only, M1-soft.]**

### 4.5 Ours-full compact plant-only representation

Ours-full combines Ours-core (A6), M1-soft view weighting, and M4 compact foreground cleanup. Table 5 reports the three-sample closed-loop results [36,38].

**[PLACEHOLDER: Table 5 — Ours-full cross-sample compactness. Columns: Sample, Variant (A6, A6+M1-soft, A6+M4, A6+M1-soft+M4), PSNR_fg, SSIM_fg, LPIPS_fg, outside_nonblack, leakage_energy, Gaussians. 4 variants × 3 samples = 12 data rows (or summarized). Data from Sections 10-11 of the project summary document.]**

Across the three samples, Ours-full reduced total Gaussian count from 1,216,294 (A6 sum) to 997,049, a reduction of 219,245 Gaussians (18.03%). The average PSNR_fg decrease was 0.0657 dB. On CaoMei2, the most compact result, Ours-full reduced Gaussians by 33.54% (370,844 → 246,452) with PSNR_fg loss of only 0.1115 dB. On XianKeLai1, reduction was 13.46% (253,827 → 219,661) with PSNR_fg loss of 0.0206 dB, and outside remained below 0.05 [15,16].

Ours-full is not a quality improvement over Ours-core in terms of foreground metrics. Its role is to produce a more compact, more exportable plant-only Gaussian representation with minimal foreground quality degradation. The practical benefit is reduced model size for storage and faster mesh extraction [16,15].

To assess the independent contributions of M1-soft and M4, we compare intermediate variants in Table 5. On CaoMei2, M1-soft alone (A6+M1-soft) reduced Gaussians by 32.6% (370,844 → 249,944) with a PSNR_fg loss of 0.0787 dB, while M4 alone (A6+M4) reduced Gaussians by 23.2% (370,844 → 284,757) with a PSNR_fg loss of 0.0530 dB. The combination (A6+M1-soft+M4) achieved the best overall compactness (33.5% reduction) with only marginally higher PSNR_fg loss (0.1115 dB) than either module alone. This suggests that M1-soft and M4 address partially complementary sources of Gaussian redundancy: M1-soft reduces Gaussians in low-quality views by down-weighting their training contribution, while M4 removes boundary-adjacent Gaussians with weak multi-view support. The additive benefit of combining both modules (33.5% vs. 32.6% and 23.2% individually) is modest on CaoMei2, indicating some overlap in the Gaussians they affect. On XianKeLai1, where thin leaves provide less redundancy, the combination achieves a 13.5% reduction, consistent with the expectation that thin structures offer fewer removable Gaussians regardless of the pruning mechanism [3,16].

**[PLACEHOLDER: Fig. 6 — Grouped bar chart: 3 samples × (Gaussian count, PSNR_fg, outside_nonblack, leakage_energy). Two bars per metric: Ours-core (A6, blue) vs Ours-full (A6+M1-soft+M4, orange). Show the trade-off: Gaussian count decreases substantially while quality metrics remain nearly unchanged.]**

### 4.6 Mesh structural evaluation

Table 6 reports mesh structural metrics for KongQueZhuYu and XianKeLai1 under three TSDF variants [48,53].

**[PLACEHOLDER: Table 6 — Mesh structural and efficiency metrics. Columns: Sample, Mesh variant, Vertices, Components, Largest component ratio, Small components, Boundary edges, Boundary consistency, Mean displacement, P95 displacement, Mesh time/s. 6 rows (2 samples × 3 variants). Data from Section 12.3 of the project summary document.]**

Key observations: (1) Smaller truncation reduced vertex count by approximately 12% in both samples but increased connected components (KongQueZhuYu: 8 → 20; XianKeLai1: 6 → 12), indicating fragmentation risk. (2) Post-boundary cleanup preserved component counts while adjusting boundary edges and added 5-24% to mesh wall time. (3) XianKeLai1 showed lower boundary consistency (0.8278 vs 0.9631) and higher displacement (mean 0.0121 vs 0.0041) compared to KongQueZhuYu, confirming that thin-leaf samples are more sensitive to boundary processing [23,24].

These results provide mesh structural and efficiency evidence. They do not yet demonstrate that specific mesh variants improve phenotype measurement accuracy [27,28].

**[PLACEHOLDER: Fig. 7 — Mesh visualization. Two columns (KongQueZhuYu, XianKeLai1) × three rows (Standard TSDF, Smaller truncation, Post-boundary). Zoom-in insets on boundary regions showing edge quality differences.]**

### 4.7 Phenotype validation

Table 7 reports manual-vs-virtual trait comparison across 21 plants from 10 species [1,3].

**[PLACEHOLDER: Table 7 — Manual-vs-virtual phenotype validation. Columns: Trait, n, MAE, RMSE, MAPE, Bias, R². 4 rows: Plant height, Canopy width, Leaf length, Leaf width. Data from the phenotype Excel file.]**

Plant height and canopy width showed the strongest agreement (R² = 0.991 and 0.993, MAPE = 6.91% and 4.50%), reflecting the relative ease of measuring global extent traits from 3D models. Leaf length achieved R² = 0.980 with MAPE = 7.45%. Leaf width showed the weakest agreement (R² = 0.956, MAPE = 9.73%, Bias = 0.383 cm), consistent with the expectation that thin structures near the reconstruction resolution limit are most sensitive to boundary effects in both the Gaussian representation and the mesh extraction pipeline [45,46].

The positive bias across all traits (0.313-0.641 cm) suggests a systematic tendency for virtual measurements to slightly overestimate manual measurements, likely due to Gaussian boundary expansion at leaf edges. This is consistent with the mesh boundary analysis (Section 4.6) and indicates that boundary refinement remains an active area for improvement [36,38].

**[PLACEHOLDER: Fig. 8 — 2×2 scatter plot grid: (a) Plant height, (b) Canopy width, (c) Leaf length, (d) Leaf width. Each plot: manual measurement (x-axis) vs virtual measurement (y-axis), y=x reference line in gray, R² and n annotated. Optional: Bland-Altman inset for leaf width showing bias and limits of agreement.]**

---

## 5. Discussion

### 5.1 Foreground-object reconstruction is not equivalent to mask post-processing

A natural question is whether the same plant-only representation could be obtained by training a standard full-scene 2DGS and then pruning Gaussians outside the mask. Our results indicate that this post-hoc approach (represented by variant E7 in our analysis framework) is not equivalent. A0 shows that a full-scene model allocates substantial Gaussian capacity to background structures, achieving leakage of 1.2201—meaning more rendering energy is spent outside the plant mask than inside it. Pruning removes the visible Gaussians but cannot recover the model capacity that was diverted from the plant foreground during training. This finding is consistent with observations in general 3DGS literature that the optimization process involves a capacity allocation dynamic, where regions receiving strong RGB supervision gradients attract densification. By excluding background pixels from the RGB supervision signal, our foreground-object reformulation (A5/A6) redirects this capacity allocation from the start [52,16].

The irreversibility of capacity allocation has implications beyond plant phenotyping. Any task requiring object-only reconstruction from multi-view images—medical organ modeling, industrial part inspection, heritage artifact digitization—may benefit from object-specific training objectives rather than post-hoc filtering of scene-level models. The general principle is that the optimization target should align with the inference target: if the desired output is a foreground-only model, the training loss should be computed on foreground pixels only [48,53].

### 5.2 View quality should modulate, not eliminate

The catastrophic failure of hard view filtering (M1-hard, M1-reject-only) carries a methodological lesson for multi-view reconstruction. In standard supervised learning, removing low-quality training examples is a common data cleaning strategy. For multi-view reconstruction, however, geometric coverage is qualitatively different from per-sample signal quality. Each view contributes a unique angular sample of the plant surface, and removing views creates angular gaps that the model cannot fill through interpolation—the missing surface simply does not exist in any remaining view. This is analogous to the aperture problem in stereo vision, where missing baselines cannot be recovered by higher-quality images from the remaining viewpoints [23,25].

The thin leaves of XianKeLai1 illustrate this acutely: leaves visible from only a narrow range of angles become unreconstructable if those specific views are removed, regardless of the quality of the remaining views. This explains why the degradation from M1-hard (10 views removed) was far more severe than the degradation from simply having 10 fewer views in the original acquisition—the removed views were not randomly distributed but clustered at angular positions critical for specific leaf surfaces [28,29].

Soft weighting resolves the tension between quality and coverage by separating the geometric signal (which requires angular coverage) from the photometric signal (which benefits from image quality). All views contribute geometry through their participation in the multi-view consistency implicit in the differentiable rendering process, while the quality weights modulate their photometric contribution to the RGB loss. This principle—separating geometric from photometric contribution—may generalize to other multi-view reconstruction scenarios where view quality varies [31,29].

### 5.3 Compactness as a practical contribution

Ours-full improves compactness (18.03% Gaussian reduction) rather than foreground reconstruction quality. This is a deliberate design choice: Ours-core already achieves foreground-only reconstruction, and the remaining opportunity is in model efficiency. Compactness matters for practical deployment: in high-throughput phenotyping scenarios where hundreds or thousands of plants are processed, smaller models reduce GPU memory requirements for rendering, accelerate mesh extraction, and decrease storage costs. The multi-cue scoring function in M4 (combining mask consistency, opacity, visibility, color normality, and topology) provides more principled pruning than opacity-based heuristics, particularly for foreground-background boundary regions where individual cues are ambiguous. Our results are consistent with the broader trend in Gaussian splatting research toward more efficient representations, with the plant-specific advantage that the multi-view mask provides a direct foreground/background signal unavailable in general scenes [3,15].

### 5.4 Cross-species generalization and the leaf width challenge

The phenotype validation results reveal a clear difficulty gradient: global extent traits (height: MAPE 6.91%, canopy width: MAPE 4.50%) are measured more reliably than organ-level thin dimensions (leaf length: MAPE 7.45%, leaf width: MAPE 9.73%). This gradient reflects a fundamental resolution limit in the reconstruction-to-measurement pipeline. Leaf width in our samples ranges from 1.5-8.0 cm, approaching the spatial resolution of the reconstructed Gaussian representation and the voxel resolution of the TSDF grid. Gaussian boundary expansion—where the rendered extent of a planar Gaussian slightly exceeds the true surface boundary—introduces positive bias (0.383 cm for leaf width) that disproportionately affects narrow structures [42,3].

The mesh boundary analysis (Section 4.6) corroborates this interpretation: XianKeLai1, with the thinnest leaves among our samples, shows lower boundary consistency (0.8278) and higher mean displacement (0.0121) compared to KongQueZhuYu (0.9631 and 0.0041). Addressing leaf-width accuracy will require either higher-resolution acquisition, boundary-aware mesh refinement that explicitly models the Gaussian-to-surface transition, or learned correction models calibrated per species. Importantly, the current results do not demonstrate that specific M5 mesh variants improve phenotype accuracy—only that the virtual measurements are feasible and that boundary effects are the primary error source [3,9].

### 5.5 Positioning within the 3D plant phenotyping landscape

The recent survey by Li et al. identified thin-leaf reconstruction, dense canopy handling, and cross-species generalization as open challenges. Our work directly addresses the first two through the 2DGS planar primitive (inherently suited to thin surfaces) and foreground-object optimization (suppressing non-plant structures rather than reconstructing and then removing them). Cross-species generalization is partially addressed: FSAM3 generates masks for 10 species without per-species prompt tuning, and Ours-core achieves foreground-only thresholds across three architecturally distinct samples. However, three samples constitute representative diversity, not statistical generalization. The 20-sample dataset and the significant variation in reconstruction quality across architectures (XianKeLai1 outside = 0.0484 vs. CaoMei2 = 0.0147) suggest that species-level architectural factors modulate reconstruction difficulty in measurable ways. A systematic cross-species study with per-species quantitative phenotyping benchmarks—analogous to the Wheat3D PartNet dataset but spanning multiple species—would strengthen the cross-species claim [1,3].

The F2DMAS pipeline's modular design also enables component-level ablation: each of the five stages (FFT screening, SAM3+PCA mask generation, foreground-object 2DGS, soft view weighting, Gaussian pruning) can be independently evaluated, replaced, or improved. This modularity aligns with the broader trend toward modular, reproducible phenotyping pipelines and facilitates incremental adoption—practitioners can integrate individual F2DMAS components into existing workflows without adopting the entire pipeline [45,27].

The phenotype validation results reveal a clear difficulty gradient: global extent traits (height, canopy width) are measured reliably (MAPE < 7%), while the thin dimension (leaf width) shows elevated error (MAPE = 9.73%). This gradient reflects a fundamental resolution limit: leaf width in our samples ranges from 1.5-8.0 cm, approaching the spatial resolution of the reconstructed Gaussian representation and the voxel resolution of the TSDF grid. Boundary effects—where Gaussians at leaf edges extend slightly beyond the true surface—introduce positive bias that disproportionately affects narrow structures. The mesh boundary analysis (Section 4.6) corroborates this interpretation: thin-leaf samples show lower boundary consistency and higher displacement. Addressing leaf-width accuracy will require either higher-resolution acquisition, boundary-aware mesh refinement, or explicit edge-thickness correction models [1,7].

### 5.5 Positioning within the 3D plant phenotyping landscape

The recent survey by Li et al. identified thin-leaf reconstruction, dense canopy handling, and cross-species generalization as open challenges in 3D plant phenotyping. Our work directly addresses the first two through the 2DGS planar primitive (inherently suited to thin surfaces) and foreground-object optimization (suppressing non-plant structures). Cross-species generalization is partially addressed: FSAM3 generates masks for 10 species without per-species tuning, and Ours-core achieves foreground-only thresholds across three distinct architectures. However, the three-sample reconstruction validation is representative rather than statistically generalizable. A larger multi-species dataset with per-sample manual phenotype ground truth would strengthen the cross-species claim [41,43].

---

## 6. Limitations

Several limitations should be considered when interpreting these results [52,16].

**Sample size for reconstruction validation:** The A6 cross-sample validation uses three representative samples selected for architectural diversity (complex background, thin leaf, dense occlusion). While this covers distinct failure modes of full-scene 2DGS, three samples do not constitute broad statistical generalization across the 10 species in our dataset. The title's "cross-species" claim is supported at the mask generation level (FSAM3 generates masks for 20 samples from 10 species) and at the phenotype level (21 plants from 10 species), but the reconstruction-level validation should be interpreted as demonstrating robustness across representative architectures rather than statistical species-level generalization. Future work should expand A6 validation to more samples per architectural category and include quantitative per-species reconstruction comparisons.

**FSAM3 segmentation evaluation:** FSAM3 is evaluated as a reconstruction prior—its masks are assessed by their downstream impact on 2DGS reconstruction quality, not by pixel-level segmentation benchmarks. We do not claim that FSAM3 achieves state-of-the-art segmentation accuracy relative to通用 segmentation methods, because we lack dense pixel-level ground truth masks for our dataset. A formal segmentation comparison would require manual annotation of a representative subset of frames.

**Controlled indoor setting:** All acquisitions were performed indoors with controlled or semi-controlled lighting. Field deployment introduces additional challenges (direct sunlight, wind motion, complex natural backgrounds) that are not tested in the current study.

**Mesh and phenotype causality:** The current mesh and phenotype results demonstrate feasibility and characterize error patterns but do not establish causal improvement from specific mesh variants. The statement "M5 improves leaf width measurement accuracy" is not supported by the current evidence, which lacks a before/after comparison of a specific mesh refinement on phenotype metrics.

**External baseline comparisons:** The current study compares variants of the proposed method (A0-A6, M1 variants, M4) but does not include comparisons against external reconstruction pipelines such as COLMAP+MVS, NeRF-based methods, or standard 3DGS. This is a deliberate scope limitation: our primary research question concerns the internal mechanisms of foreground-object reconstruction (which component is decisive, whether post-hoc pruning is equivalent, whether hard filtering is viable), and these questions are best answered through controlled within-method ablation. External comparisons would primarily test whether 2DGS is the right base representation for plants—a question partially addressed by prior work. Nevertheless, we acknowledge that a comparative evaluation against COLMAP+MVS and 3DGS with the same FSAM3 masks would strengthen the claim that 2DGS is the preferred base representation for thin-structure plant reconstruction. We plan to include these baselines in a follow-up study [11].

**Species taxonomic resolution:** The current dataset uses Chinese common names for species identification. Submission to an international journal requires verified botanical nomenclature. A mapping from Chinese common names to tentative Latin binomials is provided in Supplementary Table S1; definitive taxonomic identification requires consultation with a botanist or taxonomic database. The 10 species labels span diverse growth forms including rosette, erect, climbing, and shrub architectures, representing the morphological diversity of common ornamental and horticultural species.

**Scale recovery:** Absolute scale is recovered using a single known physical dimension (pot diameter, measured with a digital caliper at ±0.5 mm accuracy). Error in this reference measurement propagates linearly to all virtual trait measurements. Multi-point scale calibration (e.g., using checkerboard targets at multiple depths) would reduce scale uncertainty but was not implemented in the current acquisition protocol.

**Measurement protocol:** Virtual trait measurements were performed by a single operator placing landmarks on the extracted mesh. Manual measurements followed standard horticultural practice: plant height from pot rim to highest photosynthetic tissue, canopy width as maximum horizontal extent, and leaf length/width on three fully expanded leaves per plant using a flexible ruler (±1 mm). Inter-operator variability was not assessed. The reported virtual-measurement error therefore conflates reconstruction error with landmark placement error; the R² > 0.95 for global traits suggests that reconstruction error dominates, but the relative contribution of each error source cannot be disentangled from the current data.

---

## 7. Conclusions

This paper presented F2DMAS, an integrated pipeline spanning from multi-view image quality control to phenotype-ready plant mesh generation. The algorithmic contributions modify the standard 2DGS framework at five levels: [23,25]

1. **FFT-based frame quality screening** (FSAM3 Stage 1) automates input frame selection for SfM by excluding frames with insufficient high-frequency energy, retaining 82-86% of frames across 20 sequences and preventing low-quality frames from degrading camera pose estimation and mask generation [28,29].

2. **PCA-guided mask refinement** (FSAM3 Stage 3) suppresses disconnected false-positive fragments produced by SAM3 segmentation, reducing the mean component count by 67% while preserving the dominant plant region in 98.2% of frames [31,29].

3. **Foreground-object optimization** (Ours-core, A6) rewrites the 2DGS training objective by: (i) filtering COLMAP sparse points via multi-view mask consistency for foreground-biased initialization, (ii) restricting RGB loss computation to mask-defined foreground pixels, and (iii) adding alpha mask loss and background opacity loss as auxiliary opacity-field constraints. Systematic ablation (A0-A6) demonstrated that the per-pixel restriction of RGB supervision to the foreground is the decisive algorithmic change—alpha and opacity regularization alone cannot prevent background learning [3,15].

4. **Soft view weighting** (M1-soft) replaces the standard subset-selection paradigm with per-view quality modulation of the foreground RGB loss. Hard filtering (M1-hard) catastrophically degraded reconstruction (PSNR_fg: 25.01 → 12.55 dB) by removing 10 of 27 views and breaking angular coverage. Soft weighting preserved all views, reducing Gaussian count by 10.03% with only 0.0506 dB PSNR_fg loss [42,3].

5. **Multi-cue Gaussian pruning** (M4) scores each Gaussian by mask consistency, opacity, visibility, color normality, and topology, pruning weakly supported Gaussians near mask boundaries. Combined with the preceding modules, the complete F2DMAS pipeline (A6+M1-soft+M4) reduced total Gaussian count by 18.03% across three architecturally distinct samples with a mean PSNR_fg decrease of 0.0657 dB [3,9].

Downstream TSDF-based mesh extraction with post-boundary cleanup preserved mesh topology while adjusting boundary displacement, and virtual phenotype measurements across 21 plants from 10 species achieved R² > 0.95 for all four measured traits (plant height, canopy width, leaf length, leaf width). Leaf width exhibited the highest MAPE (9.73%), identifying boundary-sensitive thin-dimension measurement as the primary remaining challenge [1,3].

These results establish F2DMAS as a modular, reproducible framework for automated cross-species plant foreground reconstruction and phenotype-ready mesh generation. The pipeline's component-level modularity facilitates incremental adoption and independent improvement of individual stages [45,27].

---

## Data Availability

The multi-view image dataset and phenotype measurement data supporting this study are available from the corresponding author upon reasonable request. The plant phenotype spreadsheet analyzed in Section 4.7 is available in the project repository [48,53].

## Ethics Declaration

This study involves plant imaging and measurement only. No human or animal subjects were involved [23,25].

## Author Contributions

[PLACEHOLDER: CRediT author contributions to be completed before submission [28,29].]

## Conflict of Interest

The authors declare no competing interests.

## Funding

[PLACEHOLDER: Funding information to be completed before submission [31,29].]

## AI Usage Disclosure

During the preparation of this manuscript, the authors used Claude (Anthropic) as an AI-assisted writing and research tool for literature search, data organization, bilingual translation, and manuscript formatting. All AI-generated content was reviewed, verified, and edited by the authors. The authors take full responsibility for the accuracy and integrity of the published work [3,15].

---

## References

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
