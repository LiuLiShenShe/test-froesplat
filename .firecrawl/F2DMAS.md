F2DMAS: A Deployable Digital Twin Framework for High-Fidelity Plant Phenotyping Across Complex Environments

Jian Fanga,b,c,d, Nengfu Xiea,c,d,\*, Yane Duanb,\*,Jingchao Fana,c,d, Xiaoli Wanga,c,d, Hailong Liua,c,d, Huoguo Zhenga,c,d,Hao Wua,c,d, Zhibo Menga,c,d, Xin Wanga,c,d, Rui Mana,c,d

aAgricultural Information Institute, Chinese Academy of Agricultural Sciences, Beijing 100081, China

b College of Intelligent Science and Engineering, Beijing University of Agriculture, Beijing 102206, China

c National Agricultural Science Data Center, Beijing 100081, China

dData Hub, Chinese Agrosystem Long-Term Observation Network, Beijing 100081, China

Abstract

Plant phenomics requires accurate and high-throughput 3D structural data. However, acquiring high-fidelity 3D plant models in unstructured environments remains a critical technical bottleneck. Traditional photogrammetry and standard volume-rendering methods (e.g., 3D Gaussian Splatting) often fail when processing non-rigid, thin-walled organs like leaves, resulting in geometric distortion and topological expansion. Furthermore, the intrinsic mismatch between front-end environmental noise and back-end rendering mechanisms exacerbates reconstruction collapse. To address this deployability gap, we propose F2DMAS, an end-to-end plant digital twin framework driven by system-level integration, designed specifically for complex greenhouses and unstructured indoor environments. Relying solely on consumer-grade multi-view RGB image sequences, F2DMAS introduces a robust frequency-spatial preprocessing pipeline (FSAM3) to autonomously filter motion blur and decouple complex background noise without requiring domain-annotated data. Crucially, by employing explicit 2D geometric Gaussians, the framework mathematically enforces the zero-thickness property of plant leaves as a geometric prior, fundamentally eliminating the topological volumetric expansion inherent in standard 3DGS. Extensive experiments demonstrate that F2DMAS reconstructs state-of-the-art high-fidelity plant geometry (PSNR 31.09) under non-ideal backgrounds while compressing meshing extraction time by 91.4% compared to baseline 3DGS. The virtually extracted phenotypic traits exhibit an consistency with manual ground truths. This framework provides an accessible, computationally efficient, and reliable digital twin solution for high-throughput crop phenotyping in complex agricultural scenarios.

Keywords: Plant phenomics; 3D reconstruction; 2D Gaussian Splatting; Digital twin; Unstructured environments

Introduction

Plant phenomics serves as a core driver for modern agricultural science and crop production systems, acting as a critical bridge for precisely analyzing genotype-environment interactions1. High-throughput and high-precision phenotypic data collection quantifies plant growth dynamics, stress resistance mechanisms, and germplasm characteristics across all dimensions. This comprehensive quantification provides an indispensable data foundation for crop genetic improvement, smart agricultural decision-making, and ecosystem monitoring2.

However, current plant phenomics remains constrained by the dual limitations of biophysical characteristics and data dimensionality. The morphological plasticity and self-occlusion effects of non-rigid plant organs, particularly leaves and stems, severely hinder the acquisition of complete point cloud data. This obstruction frequently leads to geometric distortion and feature loss during 3D reconstruction3. Furthermore, crop phenotypic analysis possesses high multidimensional complexity. Taking wheat as an example, precise breeding relies not only on a single trait but also requires multi-source data fusion and collaborative prediction for yield components such as effective panicle number, environmental responses, and cross-regional growth heterogeneity. The increase in data dimensionality and coupling difficulty imposes strict requirements on the generalization capability of algorithmic models4.

The spatial heterogeneity of the plant growth environment constitutes a core barrier to phenotypic analysis. Complex greenhouses and unstructured indoor settings exhibit significant differences in illumination fluctuations, background noise, and environmental disturbances. These environmental discrepancies cause drastic distribution shifts in phenotypic data collected under different scenes, severely weakening the cross-scene generalization ability of existing algorithmic models5. To overcome this application bottleneck from controlled laboratories to unstructured environments, current research urgently requires the construction of a robust and adaptable phenotypic analysis framework. This framework must possess strong environmental robustness to achieve high-throughput non-destructive monitoring and cross-domain feature alignment for multiple species and traits in unstructured environments, thereby supporting robust intelligent breeding decisions in non-ideal backgrounds6.

Early plant phenotypic analysis heavily relied on manual measurements using tools including tape measures and calipers. This contact-based method presented intrinsic defects of low efficiency, high subjective error, and destructiveness. Physical contact often induced thigmomorphogenesis in plants, thereby interfering with their natural growth trajectories7. With breakthroughs in computer vision, automated analysis based on 2D digital imaging gradually replaces manual measurement to become the mainstream paradigm for non-destructive and high-throughput phenotypic analysis. This transition significantly improves the accuracy and repeatability of trait extraction and resolves the bottleneck of traditional methods in large-scale breeding screening 8.

Although high-resolution optical imaging combined with computer vision algorithms achieves automated extraction of 2D planar features such as projected area and color indices, this technical path has inherent dimensionality reduction limitations. 2D images cannot reconstruct the volumetric topology of plants, leading to the loss of key 3D morphological parameters including stem diameter, spatial curvature of leaves, and internode spacing9. A more severe challenge lies in the occlusion effect caused by the complex spatial structure of the plant canopy. Under a 2D perspective, mutual occlusion and self-occlusion phenomena result in numerous invisible areas, making it impossible to precisely quantify the traits of occluded organs. This technical bottleneck severely restricts the completeness of phenotypic analysis and hinders the high-throughput acquisition of full-dimensional plant morphological parameters 10.

With the continuous iteration of computer vision technology, plant phenotypic analysis accelerates its evolution from 2D planes to 3D spatial dimensions. 3D point clouds serve as the core medium for analyzing the complex geometric structures of plants 11.

Currently, data acquisition technologies are primarily divided into active and passive categories. Light Detection and Ranging (LiDAR), representing active remote sensing, accurately captures canopy structures and penetrates partial occlusions through its high-precision ranging capability. However, high hardware costs, particularly for multi-line LiDAR, and massive data processing requirements create an insurmountable barrier for large-scale field applications10. Although RGB-D depth cameras lower the adoption threshold, their imaging principles based on structured light or time-of-flight are highly susceptible to infrared spectral interference in strong outdoor light environments. This susceptibility leads to missing depth data and severe signal-to-noise ratio degradation, which fails to meet all-weather monitoring requirements. Given these limitations, passive 3D reconstruction technologies based on Multi-View Stereo (MVS) and Structure from Motion (SfM) emerge as the current mainstream alternatives. This technology resolves high-density point clouds using only standard RGB images, combining low hardware costs, acquisition flexibility, and cross-scale adaptability to provide a scalable general path for high-throughput phenotypic analysis in complex agricultural scenes.

Structure from Motion and Multi-View Stereo (SfM-MVS) technology generates high-precision 3D models through a pipeline progressing from sparse point clouds to dense reconstruction by analyzing feature matching relationships among multi-view images. This technology is widely applied to the whole-growth-stage phenotypic analysis of field crops such as corn and wheat due to its low hardware threshold and high throughput advantages12.

However, this technology highly depends on object surface textures and exhibits significant texture blind spots. In weak-texture regions including plant leaves, feature point extraction algorithms such as SIFT and ORB struggle to capture effective descriptors, causing matching failures and point cloud holes. Furthermore, bundle adjustment in large-scale scenes involves massive non-linear optimization calculations. The high time costs and computational power requirements severely restrict the efficiency of real-time phenotypic analysis 9.

As a paradigm-shifting technology in computer vision, Neural Radiance Fields (NeRF) achieve high-fidelity novel view synthesis of plant details under complex lighting conditions through the combination of volume rendering and implicit neural networks. Compared to traditional explicit modeling, improved NeRF architectures incorporating acceleration operators such as hash encoding (e.g., Instant-NGP) significantly break through the training efficiency bottlenecks of early models, enabling the rapid generation of photorealistic plant digital twins13. However, the implicit continuous scene representation of NeRF leads to an endogenous contradiction between visual fidelity and geometric accuracy. Converting implicit density fields into explicit geometric meshes often produces discontinuous topological structures and surface artifacts due to density threshold sensitivity. This lack of geometric fidelity directly restricts its application in extracting phenotypic parameters that require millimeter-level accuracy, including leaf area and stem diameter 10.

Although 3D Gaussian Splatting (3DGS) revolutionizes rendering speed through explicit ellipsoids, its volumetric geometric representation has endogenous defects when handling thin-walled structures such as plant leaves. The spatial overlap of ellipsoids frequently causes reconstructed surface thickening and normal vector noise, which fails to meet the requirements of high-precision phenotypic measurement. Therefore, 2D Gaussian Splatting emerges as the latest paradigm evolution. By flattening primitives from 3D ellipsoids to 2D oriented disks, it mathematically forces physical alignment between primitives and object surfaces14. 2DGS not only inherits the real-time rendering and topological editing advantages of explicit representations but also significantly enhances the geometric fidelity of explicit mesh extraction by introducing depth distortion and normal consistency regularization terms. This characteristic gives it a decisive advantage in plant phenotypic analysis. It precisely restores leaf edges and texture details, overcoming the blur artifacts common in traditional 3DGS during complex canopy reconstruction 15. However, existing Gaussian splatting technologies still face background segmentation challenges in unstructured agricultural scenes. The models inevitably couple background noise including soil and potted support frames. Consequently, the subsequent point cloud cleaning and trait extraction processes remain highly labor-intensive, highlighting an urgent need to develop end-to-end semantic removal algorithms.

To eliminate geometric interference from unstructured environments including soil, supports, and lighting artifacts on 3D reconstruction, precisely extracting the region of interest mask from multi-view sequences is a critical preprocessing step that determines the signal-to-noise ratio of the reconstruction. Traditional photogrammetry software such as 3DF Zephyr and Agisoft Metashape mainly rely on color space conversion and manual threshold segmentation. These hard-coded algorithms operate adequately in controlled laboratories. However, in unstructured agricultural scenes with severe illumination fluctuations such as shadows and highlights, and color overlaps such as green backgrounds, their feature recognition capabilities suffer catastrophic degradation, leading to fragmented mask edges or erroneous subject removal 16. Segmentation schemes based on fully supervised deep learning including Mask R-CNN and U-Net significantly improve automation levels, but their performance highly depends on large-scale, pixel-level annotated datasets. This data-hungry characteristic imposes a high cost barrier. More critically, supervised models exhibit severe domain shift problems. Models trained for specific species or growth environments experience a sharp decline in generalization ability when transferred to new scenes, failing to meet the requirements of multi-variety and cross-period general phenotypic analysis17.

Vision Foundation Models (VFMs) represented by Grounding DINO, SAM, and Depth Anything establish a new paradigm in computer vision. Pre-trained on massive multimodal datasets, these models demonstrate unprecedented zero-shot inference and scene generalization capabilities, providing a theoretically optimal solution to the high dependence of traditional models on annotated data 18. However, excellent performance in the general domain does not directly equate to plug-and-play capability in the vertical agricultural domain. Agricultural scenes present extreme complexity. High-frequency occlusion of plant leaves, subtle texture similarities, and dynamic changes in unstructured environments pose severe challenges to the boundary segmentation accuracy and small object recognition capabilities of VFMs. Currently, the actual effectiveness of VFMs in cross-scene plant phenotypic analysis still lacks systematic quantitative evaluation and verification, representing a key missing link in transforming these models into agricultural productivity 19.

Rather than relying on isolated algorithmic improvements, addressing the intrinsic mismatch between high-frequency physical noise at the front-end and implicit rendering features at the back-end requires system-level integration. We demonstrate that establishing an end-to-end physical alignment pipeline is crucial, proving the engineering necessity of frequency-spatial feature decoupling as a prerequisite prior for 2DGS.

To resolve these challenges, this study proposes a cross-scene multi-plant 3D reconstruction framework integrating vision foundation models and 2D Gaussian Splatting. The framework develops an image processing pipeline, FSAM3, which leverages vision foundation models to automate the processing of multi-view image sequences via text prompts. This design overcomes the reliance on annotated data and the limited scene transferability inherent in traditional methods. The framework employs 2DGS to perform high-fidelity 3D plant modeling and evaluates its applicability for capturing 3D plant structures by analyzing reconstruction efficiency and accuracy, ultimately supporting comprehensive plant phenotypic analysis.

Materials and methods

Overall System Architecture

This study presents a comprehensive system architecture consisting of dataset construction, data preprocessing, 3D reconstruction, and phenotypic trait extraction. The workflow initiates by collecting multi-view image sequences of diverse potted plant species across two distinct scenes. To process these raw visual inputs, the automated FSAM3 framework integrates the SAM3 vision foundation model to perform zero-shot semantic segmentation and tracking. This integration enables high-precision mask extraction guided by specific text prompts. Following preprocessing, the pipeline employs 2D Gaussian Splatting to construct high-fidelity 3D plant models. A Truncated Signed Distance Function surface optimization algorithm subsequently refines the extracted geometry to generate high-quality 3D plant meshes. The system ultimately extracts key phenotypic traits, including plant height, canopy spread, leaf length, and leaf width, from the reconstructed models. These computational measurements are systematically compared against ground truth data to evaluate the overall reconstruction accuracy. Fig. 1 illustrates the complete technical framework.

Fig. 1. Overall Architecture: (a) Two Data Collection Methods, (b) FSAM3 Framework Workflow, (c) 2DGS Reconstruction Process, (d) Scale Recovery, Feature Extraction, and Evaluation

Experimental Materials and Data Acquisition

The plant dataset originates from two distinct acquisition methods including fixed equipment data collection and complex environment collection. This dataset comprises fifteen plant species. For the fixed equipment method, this study constructs a low-cost and portable plant phenotypic data collection system as shown in Fig. 2(a), which consists of an electric turntable, a black cloth background, and a mobile device. The system records target strawberry seedlings by allowing the electric turntable to rotate the plants instead of utilizing a tripod to fix the recording device. Although a black cloth serves as the background, the natural folds of the fabric are intentionally preserved. These irregular wrinkles generate random specular reflections and shadow variations during dynamic filming, creating optical noise similar to complex field lighting environments. This design evaluates the segmentation robustness of the algorithm under uncontrolled lighting and background interference.In the complex environment acquisition method, plants are placed in intricate indoor settings containing various irrelevant backgrounds and clutter with similar colors. During data collection, the camera revolves around the plant twice to ensure complete coverage, as illustrated in Fig. 2(b). The mobile device employed is a standard consumer-grade smartphone, specifically an iPhone 14 Pro Max. It captures video at 60 FPS with a resolution of 1080 1920 pixels using an ultra-wide 13mm f/2.2 lens. Since no tripod is used to fix the shooting angle or height, the acquisition process remains flexible. The target strawberry seedlings are maintained within the field of view for over 70% of the recording duration. The electric turntable has a diameter of 15cm, a maximum rotation angle of , and a positioning accuracy of . This experiment constructs a non-ideal background environment to simulate unavoidable environmental interference inherent in actual field operations.

Fig. 2. (a)Schematic of the data acquisition setup. 1. Target strawberry seedling; 2. Electric turntable; 3. Smartphone; 4. Wrinkled black background. (b) Schematic Diagram of Data Collection in a Complex Environment. (c) illustrates strawberry seedlings captured via the fixed equipment data acquisition method. (d) displays cyclamen plants obtained through complex environment acquisition.

The data acquisition protocol utilizes handheld mobile devices to record video sequences for efficient multi-view image extraction without reliance on tripods. During the fixed equipment collection process, the target plant is positioned on an electric turntable where a black card and a red calibration marker are placed to streamline subsequent image processing. Recording commences simultaneously with the rotation of the turntable. For complex environment acquisition, seedlings are placed on stationary objects to facilitate manual recording. Both strategies involve continuous recording across two distinct elevations. The lower viewpoint provides a comprehensive profile of the plant, whereas the higher viewpoint utilizes an elevated position and a calculated camera tilt to capture a top-down perspective. To ensure geometric completeness, the target plant remains centered within the frame throughout the process. The sequence progresses from the lower viewpoint to the higher elevation, with the plant rotating at each level to capture the full morphology from all orientations.

FSAM3: Frequency-Spatial Dual-Domain Collaborative Processing Framework

Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) impose rigorous requirements on the photometric consistency and spatial purity of multi-view sequences. In unstructured agricultural environments, motion blur induced by stochastic factors often overlaps with intricate agronomic backgrounds, which severely undermines multi-view geometric constraints and leads to significant topological interference in the underlying 3D reconstruction. Although existing literature attempts to mitigate these issues using segmentation models20 for background removal or joint optimization strategies21 for deblurring, achieving efficient synergy between frequency-domain quality and spatial-domain semantics during the preprocessing stage remains a challenge. To address these bottlenecks, this study proposes FSAM3, a frequency-spatial dual-domain collaborative processing framework. The framework innovatively deploys the Fast Fourier Transform (FFT) as a frequency-domain quality gateway to precisely filter blurred artifacts. Simultaneously, it integrates the powerful spatial-domain semantic decoupling capabilities of the Segment Anything Model 3 (SAM3), supplemented by an adaptive target selection mechanism. Consequently, this research constructs an automated preprocessing pipeline spanning from unsupervised blur elimination to high-frequency feature preservation, ensuring the robustness of 3D reconstruction in complex agricultural scenarios at the data source level.

- 2.3.1. Frequency-Domain Quality Filtering Mechanism

Structure from Motion (SfM) within the 3D reconstruction pipeline relies heavily on precise pixel-level feature matching. In uncontrolled environments, motion blur in the plant canopy caused by environmental drafts or camera displacement destroys high-frequency texture features, which directly leads to camera pose drift and sparse point cloud divergence. To address this, this study implements a frequency-domain quality filtering mechanism based on no-reference image assessment theory. The mechanism utilizes a high-frequency energy detection algorithm derived from the 2D Discrete Fourier Transform (2D DFT) to enforce a rigid quality gate on the input sequence. For a grayscale image with a spatial resolution of, the 2D DFT spectrum is defined as follows:

(1)

To quantify image sharpness, a low-frequency centralization operation shifts the zero-frequency component to the center of the spectrum. The amplitude spectrum is then extracted and log-transformed to compress the dynamic range:

_(2)_

Edges and texture details in sharp images map to high-frequency components located far from the center in the frequency domain. By defining a central low-frequency mask with radius to shield the core energy that dominates global illumination, the mean amplitude of the remaining high-frequency region is calculated as the sharpness evaluation metric:

_(3)_

In this equation, represents the total number of pixels within the high-frequency region. The system establishes a predefined sharpness threshold. When an observed frame satisfies, the algorithm identifies severe motion blur and automatically excludes the frame from the multi-view sequence. This process ensures that the video frames entering the subsequent 3D reconstruction stages maintain absolute optical sharpness at the physical feature level.

- 2.3.2. Spatial-Domain Semantic Decoupling Mechanism

Following the elimination of blurred frames, the remaining high-fidelity image sequences still contain complex backgrounds including greenhouse supports and reflective grounds. Directly inputting these full-element images into the 3D reconstruction pipeline causes the background regions to preempt a massive amount of Gaussian primitives. This preemption dissipates computational resources and induces irreversible physical adhesion between the plant base and environmental structures during mesh extraction. To overcome this bottleneck, the current stage introduces the FSAM3 architecture to execute fully automated unsupervised semantic segmentation. Compared to traditional image segmentation paradigms that heavily rely on domain-specific annotated data and lack temporal consistency, FSAM3 employs a unified Transformer architecture to process image and video streams synchronously. Given an input image, the data first flows through a hierarchical Vision Transformer encoder to extract high-dimensional spatial features. Simultaneously, a lightweight text encoder maps a natural language or spatial grid prompt into a multimodal prompt embedding. A memory attention module retrieves the historical spatio-temporal context from a memory bank to ensure multi-view consistency. Subsequently, these features are jointly fed into a mask decoder, which aggregates the features through a bidirectional attention mechanism and outputs an initial binary mask. The core mathematical expression is:

_(4)_

To eliminate tiny internal holes within the foreground region caused by reflections or local texture absence, the system applies a morphological closing operation to the initial mask. Defining the morphological structuring element as, this operation is a cascade of morphological dilation and erosion:

_(5)_

In this equation, and represent the dilation and erosion operators, respectively. Finally, a pixel-wise Hadamard product between the closed mask and the original image in the spatial dimension generates a high-purity plant image with an Alpha transparency channel:

_(6)_

This decoupling mechanism physically severs the photometric interference of environmental textures on the Truncated Signed Distance Function regularization optimization. It provides the subsequent 2D Gaussian Splatting pipeline with a high-purity input stream containing only plant topological features, constituting a rigid prerequisite for successfully extracting watertight meshes of thin-walled organs.

Traditional image segmentation models such as UNet, YOLOv8, and Mask R-CNN exhibit a high dependence on domain-annotated data and restricted generalization capabilities. Early iterations of the Segment Anything Model lacked temporal consistency across continuous frames. In contrast, the FSAM3 architecture introduced in this study demonstrates significant advantages in cross-scene generalization and multi-view processing22. The model innovatively integrates a lightweight text encoder to map natural language prompts into high-dimensional semantic embeddings. This integration breaks the locality limitations of traditional sparse point prompts and utilizes global semantic context to achieve precise extraction of specific biological components under zero human interaction conditions. Operationally, FSAM3 employs a unified Transformer architecture to parse image and video streams synchronously, as illustrated in Fig.3. To overcome bottlenecks in processing multi-view and temporal data, the architecture introduces a streaming memory mechanism whose core information flow is collaboratively constructed by four key components. First, the image encoder relies on a hierarchical Vision Transformer to efficiently extract high-dimensional spatio-temporal features from the input sequence. Subsequently, the memory attention module serves as the central hub for multi-view consistency. It utilizes an internal memory encoder and memory bank to continuously store and retrieve target interaction information and mask priors from historical frames, physically ensuring the topological continuity of the segmented target under complex viewpoint transformations. Concurrently, the prompt encoder deeply fuses natural language features to generate multimodal prompt embeddings while maintaining compatibility with traditional geometric sparse prompts including points and bounding boxes. Ultimately, the mask decoder utilizes a bidirectional attention mechanism to globally aggregate the image spatio-temporal features, the multimodal prompt embeddings, and the spatio-temporal context retrieved from the memory bank, thereby decoding and outputting high-fidelity continuous segmentation masks.

The text-driven segmentation pipeline based on FSAM 3 facilitates the transition from interactive to automated processing. Instead of relying on manually annotated points or bounding boxes in the initial frame, the system processes a descriptive text prompt indicating the entire green plant excluding the pot. The text encoder within FSAM 3 parses this instruction and integrates it with image features to automatically generate an initial mask that covers the complete plant morphology. This approach eliminates the subjective errors associated with manual annotation and ensures the objectivity of feature extraction. Upon generating the semantic mask for the initial frame, the memory engine of FSAM 3 activates. The model stores the semantic and geometric features of the plant within a memory bank.

During subsequent rotated views, the model continuously tracks and segments the target based on the semantic consistency maintained in the memory bank, even under conditions of self-occlusion or illumination variations. This continuous tracking operates without requiring repeated text inputs for each frame. This semantic-driven mask sequence exhibits high semantic continuity, effectively preventing the misidentification of non-target objects common in traditional methods. Consequently, it provides highly pure geometric constraints for the subsequent 2D Gaussian Splatting pipeline.

Fig.3 Seedlings image segmentation based on FSAM3

Manifold-Isomorphic 3D Reconstruction and Phenotyping Pipeline Based on 2DGS

The pipeline for acquiring high-fidelity 3D plant structures comprises four core execution nodes including FSAM3 semantic prior decoupling, Structure from Motion sparse reconstruction, 2D Gaussian Splatting 2D manifold optimization, and spatial artifact filtering with explicit meshing.

Fig.4 illustrates the flow of the reconstruction pipeline.The system initially extracts a complete multi-view image set from the captured video through frame interval sampling. To address motion blur caused by handheld recording in uncontrolled agronomic environments, the pipeline deploys a high-frequency energy detection algorithm based on the Fast Fourier Transform. This algorithm centrally masks low-frequency components and performs rigid elimination of blurred frames by averaging high-frequency amplitudes. Subsequently, the framework utilizes the FSAM3 semantic decoupling model to forcibly isolate the greenhouse background. This critical step generates an absolutely pure Alpha mask that severs the trajectory of underlying 2DGS patches erroneously aligning their depth with the complex background. Consequently, this mechanism forces the computational resources to converge 100% on the plant body.

The processed high-fidelity image array enters the Structure from Motion algorithm to extract camera intrinsic and extrinsic parameters alongside sparse point clouds. Using this sparse topology as a prior, the system initializes the 2D Gaussian Splatting scene representation. It utilizes 2D elliptical patches equipped with explicit normal constraints to tightly approximate the zero-thickness physical attributes of plant leaves. Ultimately, after eliminating photometric redundant artifacts, the pipeline executes joint extraction of depth and normals to output an explicit manifold mesh that perfectly conforms to the high-frequency physical boundaries of the plant. This mesh provides a fundamental baseline for the absolute quantification of downstream phenotypic parameters.

Fig.4. 3D Reconstruction Pipeline for Seedlings.

- 2.4.1. Sparse Topology Reconstruction and Camera Pose Estimation

The pipeline utilizes the Structure from Motion (SfM) algorithm to perform sparse reconstruction of the plants. SfM operates as an offline 3D reconstruction technology that overcomes viewpoint limitations by analyzing multi-view geometric relationships within unordered image sequences to jointly solve for the 3D topological structure of the scene and camera motion trajectories. To address the complex canopy occlusions and repetitive leaf textures of strawberry seedlings, this pipeline includes four core stages comprising feature association, epipolar constraints, pose estimation, and non-linear optimization.Feature detection algorithms extract keypoints and high-dimensional descriptors from image sequences across different viewpoints. To eliminate outliers in unordered matching, the system introduces epipolar geometry for spatial constraints. Assuming and represent a pair of homogeneous pixel coordinates in adjacent views, they strictly satisfy the equation:

_(7)_

In this equation, denotes the fundamental matrix, which restricts candidate matching points to converge on their corresponding epipolar lines.

Based on the pre-calibrated camera intrinsic matrix, the fundamental matrix is dimensionally upgraded to the essential matrix:

_(8)_

By performing Singular Value Decomposition (SVD) on the essential matrix, the system decouples and extracts the rotation matrix and the translation vector for the camera in the current view. This process constructs the camera projection matrix for the -th viewpoint:

(9)

After acquiring precise multi-view camera poses and pixel feature associations, a triangulation mechanism back-projects the 2D image plane coordinates into 3D physical space. Defining an unknown 3D physical point in the scene as in homogeneous coordinates, its mapping relationship with the image observation point is defined as:

(10)

In this context, represents the projective depth scale factor. Solving this system of linear equations via SVD recovers the 3D discrete coordinates on the surface of the strawberry seedling, instantiating the initial sparse 3D point cloud.

The SfM pipeline is highly susceptible to error accumulation during continuous pose deduction. Bundle adjustment applies global non-linear constraints to the camera extrinsic parameters and 3D point coordinates to mitigate this accumulation. The system defines a reprojection error cost function:

(11)

Here, denotes the total set of camera views, represents the total set of 3D points, and acts as an observation visibility indicator variable. The term represents a non-linear mapping function that projects the 3D point onto the -th image plane based on and. The system utilizes the Levenberg-Marquardt algorithm to minimize this cost function, ultimately outputting a high-fidelity sparse geometric topology of the plant alongside globally aligned camera pose sequences.

- 2.4.2. 2D Manifold Isomorphic Reconstruction Algorithm Based on 2DGS

This study discards the volumetric sphere assumption of traditional 3DGS and adopts 2D Gaussian Splatting (2DGS) as the core scene representation. Plant leaves typically act as zero-thickness 2D manifolds in physical space. The volumetric distribution of 3DGS induces topological expansion at perspective intersections. Conversely, 2DGS utilizes explicit 2D surfels to perfectly construct the thin-walled morphology of plant leaves through normal consistency constraints.The 2DGS scene is discretized into millions of 2D Gaussian surfels with explicit geometric boundaries. Each surfel is parameterized by a 3D center coordinate, a 2D scaling matrix, and a rotation matrix defining the orientation of the local tangent plane. The first two column vectors of matrix, denoted as and, constitute the orthogonal basis of the tangent plane. The third column vector serves as the explicit normal of the surfel. Its probability density distribution function on the local 2D tangent plane is defined as:

(12)

Here, represents the local coordinates on the tangent plane. During the rasterization rendering stage, 2DGS abandons the volumetric approximation of 3DGS integrating along camera rays and executes rigorous ray-surfel intersection tests. Given a ray equation emitted from the camera, the exact intersection depth with the plane containing the 2D Gaussian surfel is calculated as:

(13)

After determining the intersection point, the system projects it onto the local tangent plane of the surfel to obtain the 2D coordinates. It then evaluates the final spatial opacity by combining the base opacity and the Gaussian attenuation factor. This dimensionality reduction representation based on ray intersection fundamentally eliminates rendering blur and depth divergence caused by viewpoint shifts. It ensures the high-frequency sharpness of plant edges, including serrated leaf margins.

The rendered color and depth are solved through -blending accumulation via depth sorting along the ray:

_(14)_

During the training process, alongside standard photometric errors such as and D-SSIM loss, 2DGS introduces a depth-based normal consistency regularization. The system forces the geometric normal calculated from the rendered depth map to remain aligned with the explicit normal of the 2D surfel. This alignment ensures that the Gaussian surfels extend smoothly in space without physical self-intersection.

- 2.4.3. Photometry-Based Spatial Artifact Filtering

During the scene optimization process of 2DGS, viewpoint blind spots, canopy occlusion boundaries, and non-ideal lighting conditions in the training set frequently generate numerous abnormally dark Gaussian primitives in concealed areas. These spatial artifacts manifest as high-frequency black patches in the rendering results. Essentially, they result from local overfitting caused by insufficient multi-view geometric constraints during gradient descent optimization. Analysis indicates that these redundant primitives share a significant photometric commonality where their isotropic base color radiance approaches pure black. Based on these photometric characteristics, this study proposes an automated post-processing filtering algorithm based on the Human Visual System to eliminate invalid redundant data from the underlying topology.

The core implementation pipeline of this filtering mechanism comprises three stages including feature extraction, brightness perception, and hard threshold pruning. 2DGS relies on Spherical Harmonics to perform high-dimensional encoding of the view-dependent color radiance for each Gaussian primitive.

This module first extracts the 0-order Spherical Harmonics coefficients for all Gaussian primitives from the scene model. As the base constant of the Spherical Harmonics basis, the 0-order component operates independently of the observation viewpoint and directly represents the base RGB color value of the Gaussian primitive under isotropic conditions. Defining the extracted 0-order coefficient as, its mapping relationship to the base color vector is defined as:

(15)

In this equation, represents the 0-order Spherical Harmonics basis constant which is approximately 0.28209, and denotes a clipping activation function ensuring the color values converge within the interval.

After acquiring the standardized RGB color components, the system calculates the perceived brightness of each Gaussian primitive according to the BT.601 color space conversion standard published by the International Telecommunication Union.

This standard incorporates the unequal sensitivity characteristics of the human visual system to different spectral wavelengths by assigning specific psychophysical weights to the red, green, and blue color channels. The perceived brightness of a Gaussian primitive is calculated as follows:

(16)

Upon obtaining the global perceived brightness distribution of the primitives, the system introduces a hard brightness pruning mechanism. Setting a global brightness perception threshold, the system classifies Gaussian primitives satisfying as photometric artifacts and permanently strips them from the scene point cloud topology.

Such dark primitives provide extremely low visual contributions to the effective scene structure and occupy additional memory overhead. Based on multiple rounds of ablation experiment tuning on the dataset, setting the pruning threshold to 0.15 enables the algorithm to achieve an optimal photometric balance between thoroughly eliminating dark artifact noise and preserving effective plant canopy shadow details. This configuration significantly enhances the spatial purity and rendering signal-to-noise ratio of the 2DGS reconstruction model.

- 2.4.4. Joint Depth-Normal Explicit Meshing

After acquiring the purified 2DGS model, it is necessary to convert it into a continuous explicit polygon mesh to execute phenotypic quantification. Because 2DGS employs planar primitives instead of volumetric distributions, the Signed Distance Function regularization commonly used in implicit fields destroys its manifold properties. Therefore, this study adopts a meshing strategy based on Truncated Signed Distance Function fusion that combines depth and normals.

The underlying mathematical architecture of 2DGS allows for the lossless rendering of high-precision depth maps and normal maps from arbitrary viewpoints. The system defines a uniform voxel grid within the bounding box of the target object and traverses all camera poses in the training set to back-project the depth and normals rendered from each viewpoint into the global space.

For any voxel point in 3D space, its truncated signed distance value relative to camera is calculated based on the consistency of the rendered depth and normals. The globally fused distance is computed via a weighted average:

_(17)_

The weight is dynamically determined by the angle between the incident ray and the normal, which eliminates unreliable depth observations located at edges with large inclination angles. After the global TSDF converges, the Marching Cubes algorithm extracts the isosurface.

Because the underlying data originates from zero-thickness 2DGS surfels rather than inflated volumetric Gaussians, the extracted triangular mesh perfectly preserves the thin-walled morphology of plant leaves. Finally, the system executes connected component filtering to strip discrete fragments and outputs a highly smooth and high-fidelity polygonal phenotypic twin. This process establishes an absolute geometric and physical baseline for downstream tasks including surface area integration and extreme distance calculation.

Scale Recovery and Phenotypic Measurement Verification Based on Digital Twins

Although 2D Gaussian Splatting reconstructs high-fidelity plant geometry, models generated by Structure from Motion and Gaussian splatting inherently lack absolute scale information. To imbue the models with true physical attributes and verify their measurement accuracy, this study adopts a physical-digital dual validation strategy. This strategy utilizes the open-source point cloud processing software CloudCompare to conduct virtual measurements. The reconstructed mesh model exists in a relative coordinate system. To map this model to a real-world coordinate system, the experiment utilizes a pre-positioned calibration marker with a known physical size within the scene as a geometric reference. The system extracts the mesh vertices corresponding to the calibration marker in CloudCompare and calculates their Euclidean distance in the virtual space. The global scaling factor is computed as follows:

_(18)_

Subsequently, the system applies a scaling transformation matrix parameterized by to the entire plant mesh model to unify the spatial units into centimeters:

(19)

Using the scale-corrected digital twin model, the pipeline simulates standard agronomic measurement processes to extract precise phenotypic parameters, as illustrated in Fig.5.

Fig.5. (a) Original State of the Mesh; (b) Calibration Result via Scale Factor; (c) Extraction of the Region of Interest; (d) Measurement Example for Plant Height and Canopy Spread; (e) Measurement Method Example for Leaf Length and Width

Evaluation Metrics

To comprehensively quantify the performance of the algorithm in object detection, image segmentation, and 3D reconstruction tasks, this study selects the F1-Score, mean Intersection over Union (mIoU), Peak Signal-to-Noise Ratio (PSNR), Structural Similarity (SSIM), Learned Perceptual Image Patch Similarity (LPIPS), and the Coefficient of Determination () as core evaluation metrics.The F1-Score comprehensively evaluates the precision () and recall () of the model. Precision indicates the proportion of actual positive samples among those predicted as positive, and recall indicates the proportion of correctly predicted positive samples among all actual positive samples. The formulas are defined as:

(20)

(21)

(22)

In these equations, represents true positives, represents false positives, and represents false negatives. A higher F1-Score indicates a more balanced overall classification performance of the model. The mean Intersection over Union (mIoU) serves as the core metric for image semantic segmentation by calculating the average Intersection over Union across all classes.

It measures the spatial overlap between the predicted region and the ground truth label region:

(23)

Here, denotes the number of target categories excluding the background. An mIoU value closer to 1 signifies higher spatial precision in segmentation.

The Peak Signal-to-Noise Ratio (PSNR) evaluates the degree of physical distortion in the reconstructed image based on pixel-level errors. The Mean Squared Error (MSE) is first calculated:

(24)

(25)

In these expressions, and represent the ground truth and reconstructed images respectively, is the total number of pixels, and is the maximum possible pixel value, which is 255 for 8-bit images. A higher PSNR value indicates a smaller pixel-level error.

Structural Similarity (SSIM) measures the perceptual similarity between images from three dimensions including luminance, contrast, and structure:

(26)

Here, and represent the feature activation values of the -th layer of the network, and are the spatial dimensions of the feature map, and denotes the channel weight. A lower LPIPS value indicates higher perceptual similarity in the human visual system.The Coefficient of Determination () is utilized for regression tasks such as the fitting of plant phenotypic parameters to evaluate the extent to which the model's predicted values explain the variance in the true values:

_(27)_

In this equation, represents the true observed value, represents the model's predicted value, and is the sample mean of the true values. A value closer to 1 indicates a higher goodness of fit for the regression model.

Result

Segmentation Performance of FSAM3 and SSEM

Qualitative analysis demonstrates that FSAM3 successfully captures the complete morphology of the target plant. It effectively identifies and preserves complex components including soil regions and fine-grained plant structures. Conversely, the SEEM baseline exhibits structural deficiencies characterized by the prominent absence of soil regions and fine-grained plant parts.As shown in Fig.6.

This morphological completeness indicates that FSAM3 possesses enhanced sensitivity to complex semantic boundaries. Quantitative evaluations strictly align with these qualitative observations of structural preservation. FSAM3 achieves an F1-score of 98.3 and an mIoU of 97.9%, outperforming SEEM by absolute margins of 3.2 and 3.8%, respectively. The HD95 metric substantially decreases from 281.9 pixels to 41.4 pixels. This massive reduction of 240.5 pixels in the HD95 metric directly quantifies the elimination of severe boundary deviations. This result mathematically verifies the observation that FSAM3 resolves the missing component issues inherent in the SEEM method.

Fig.6. Comparison of Original Images, FSAM3 Segmentation, and SEEM Segmentation (First Row: Original Image Segmentation Results; Second Row: FSAM3 Segmentation Results; Third Row: SEEM Segmentation Results)

Comparison of 3D Plant Reconstruction Based on F2DMAS and 2DGS

This study performs high-precision camera pose estimation based on the Structure from Motion (SfM) algorithm by executing feature matching and bundle adjustment on image sequences preprocessed by FSAM3. The results indicate that the estimated camera poses exhibit high spatial alignment with the physical shooting trajectories, while the generated sparse point clouds precisely capture the complex 3D topological contours of the target plants. The system reconstructs a sparse point cloud model containing 24,226 spatial feature points based on 210 valid camera viewpoints. This quantitative outcome validates the robustness of the pose estimation pipeline in uncontrolled environments. It confirms that the preprocessing mechanism provides a reliable and feature-rich prior for subsequent high-fidelity 3D reconstruction through rigorous geometric constraints.

Table 1 presents a comparison of the 3D reconstruction results obtained using the F2DMAS method and the standard 2DGS method on the same dataset.

Quantitative evaluation indicates that the proposed F2DMAS method consistently outperforms the baseline 2DGS across all fidelity metrics. Specifically, F2DMAS achieves a Peak Signal-to-Noise Ratio (PSNR) of 31.09 dB, a significant improvement of 1.51 dB over the 29.58 dB recorded by 2DGS. The Structural Similarity Index (SSIM) increased from 0.9574 to 0.9711, while the Learned Perceptual Image Patch Similarity (LPIPS) score decreased from 0.0487 to 0.0365. These improvements demonstrate that integrating semantic segmentation results as a prior allows the model to concentrate its representational capacity effectively on the target plant structure, thereby enhancing the reconstruction of fine-grained textures.

Regarding operational overhead, F2DMAS shows a substantial reduction in both optimization and post-processing durations. The total training time was compressed from 12913.7 s for the standard 2DGS pipeline to 5044.5 s, representing an approximately 2.56-fold acceleration. A similar trend is observed in the mesh generation phase, where the duration decreased from 157.9 s to 55 s. This efficiency gain is directly attributable to the automated exclusion of non-target background elements, which prevents the redundant allocation and optimization of Gaussian primitives on irrelevant scene components such as greenhouse supports or reflective surfaces.

Beyond quantitative gains, F2DMAS significantly improves the spatial purity of the reconstructed scene. While standard 2DGS is often hindered by white Gaussian noise and floating artifacts at the scene periphery, the proposed semantic decoupling mechanism ensures that Gaussian primitives are strictly constrained within the plant's topological boundaries. By mitigating resource dissipation on environmental elements, the framework produces a smoother and more refined 3D representation. This approach effectively prevents physical adhesion between the plant base and environmental structures, ensuring high purity of the input stream for subsequent mesh extraction.

Table 1. Comparison of 3D Reconstruction Results between F2DMAS and the Standard 2DGS

|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| Metric | PSNR | SSIM | LPIPS | Train time(s) | Mesh time(s) |
| 2DGS | 29.58 | 0.9574 | 0.0487 | 12913.7 | 157.9 |
| F2DMAS | 31.09 | 0.9711 | 0.0365 | 5044.5 | 55 |

Comparative Analysis with COLMAP and 3DGS

To systematically evaluate the reconstruction paradigm shift, we established a rigorous baseline covering the core evolution of 3D vision frameworks: COLMAP represents the upper bound of traditional explicit multi-view geometry; standard 3DGS serves as the benchmark for volumetric rendering; and SuGaR represents the state-of-the-art in Gaussian-based explicit mesh extraction. This baseline selection ensures a comprehensive evaluation of the core rendering paradigms rather than limiting the comparison to task-specific finetuned models. Table 2 presents the quantitative evaluation results of different methods on the test set.

As demonstrated in the table, F2DMAS establishes the highest performance across all visual quality metrics, achieving a PSNR of 31.09, an SSIM of 0.9711, and the lowest LPIPS of 0.0365. Compared to the 3DGS baseline, F2DMAS attains a strict numerical advantage in rendering quality. The performance gap is particularly significant when compared to the traditional COLMAP method, resulting in an absolute increase of 17.46 in PSNR and a reduction of 0.0707 in LPIPS. F2DMAS overcomes the traditional trade-off between rendering fidelity and processing speed. The most critical advantage is observed during the mesh generation phase, where F2DMAS requires only 55 seconds. This represents a 91.4% reduction in meshing time compared to the 642 seconds required by 3DGS, even surpassing the traditionally faster COLMAP baseline of 78 seconds. Concurrently, F2DMAS reduces total training time from 5413.5 seconds to 5044.5 seconds relative to 3DGS. This dual optimization of visual metrics and extraction speed validates the architectural efficiency of the method. To demonstrate reconstruction quality qualitatively, Fig.7(a) presents a visual comparison of each method under test viewpoints.Both approaches effectively eliminate background noise through the FSAM3 preprocessing step, confirming the utility of the FSAM3 framework across diverse 3D reconstruction pipelines.

Table 2. Comparative Analysis with COLMAP and 3DGS

|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| Method | PSNR (↑) | SSIM (↑) | LPIPS(↓) | Train time(s) | Mesh time(s) |
| COLMAP | 13.63 | 0.8745 | 0.1072 | 599.5 | 78 |
| 3DGS | 30.17 | 0.9587 | 0.0386 | 5413.5 | 642 |
| F2DMAS | 31.09 | 0.9711 | 0.0365 | 5044.5 | 55 |

To facilitate a visual comparison of reconstruction quality, Fig.7(a) illustrates the results generated by different methodologies under test viewpoints. Both approaches effectively eliminate background noise via the FSAM3 preprocessing pipeline, confirming the practical utility and versatility of the FSAM3 framework across diverse 3D reconstruction tasks.

Fig.7. (a)Qualitative Comparison of Gaussian Point Clouds Across Original Images, SuGaR, and F2DMAS.(b) Comparison of Surface Meshes for Phenotypic Analysis

Fig.7(a) presents a qualitative comparison between the original input images and the Gaussian point clouds generated by SuGaR and F2DMAS. While the SuGaR baseline often retains background artifacts and exhibits less precise topological alignment with the plant surface, the F2DMAS framework effectively isolates the plant structure and produces a cleaner, manifold-isomorphic representation. This visualization highlights the efficacy of the FSAM3 preprocessing pipeline in suppressing environmental noise and ensuring high-fidelity geometric reconstruction in uncontrolled agricultural environments.

The automated measurement of plant phenotypes relies significantly on the surface smoothness of the mesh. To demonstrate the geometric recovery advantages of 2DGS, a comparative analysis is conducted among the original imagery, the mesh produced by the SuGaR method, and the mesh generated by the F2DMAS framework. As illustrated in Fig.7(b), leaf overlapping is observed in the original images.

In the SuGaR-generated mesh, these overlapping regions exhibit blurred boundaries and physical adhesion. In contrast, F2DMAS effectively resolves these issues by maintaining distinct boundaries between overlapping leaves, which provides a high-fidelity geometric representation for downstream phenotypic analysis.

To quantitatively evaluate the measurement accuracy, we extract four key agronomic indicators including leaf length, leaf width, plant height, and canopy width on the scale-restored digital twin model. We then conduct regression analysis between these extractions and manually measured physical ground truths. Fig.8 illustrates these comparisons, where subfigures (a) through (d) represent plant height, canopy width, leaf length, and leaf width, respectively.

The virtual extraction method demonstrates robust accuracy and serves as an excellent alternative to manual measurement for various plant phenotypic traits. Linear regression analysis reveals that the coefficients of determination () for all four evaluated dimensions exceed 0.95. This result indicates that the digitized data successfully capture and explain over 95% of the variance in the manual baseline measurements. Such high consistency confirms the fundamental reliability of the virtual extraction pipeline for phenotypic analysis.

A comparative evaluation across different phenotypic dimensions shows that the extraction accuracy of macroscopic morphological features is significantly higher than that of microscopic leaf features. The linear fit for canopy width achieves the highest performance with an of 0.993, a root mean square error (RMSE) of 0.99 cm, and an optimal regression slope of 1.01. Plant height closely follows with an of 0.991. Conversely, leaf width measurement exhibits the lowest consistency among all tested indicators with an of 0.956. This discrepancy suggests that the current algorithm is more effective in capturing the overall canopy and plant structure than the local fine-grained leaf geometry.

Despite the strong overall linear correlation, an analysis of the regression equations reveals specific systematic biases within the extraction algorithm. For plant height, the regression equation indicates a systematic positive baseline shift of approximately 1 cm, which is relatively independent of the absolute plant size. More critically, the leaf width measurement presents a clear amplification bias. Given the steepest slope, the virtual extraction method proportionally overestimates the actual leaf width by approximately 9% as the physical size increases. This proportional deviation represents the most significant departure from the ideal zero-error fit () among all evaluated metrics.

Fig.8. Comparative Results of Phenotypic Analysis

Discussion

This work proposes F2DMAS, a plant 3D reconstruction framework designed for cross-species and cross-scene applications, aiming to achieve integrated reconstruction from uncontrolled multi-view images to high-quality mesh models through a multi-model fusion strategy. Experimental results indicate that F2DMAS demonstrates significant advantages in multiple fidelity metrics and computational efficiency compared to baseline models such as single 2DGS, 3DGS, and the SuGaR workflow. Specifically, compared to the single 2DGS model, this framework improves PSNR by 5.10%, increases SSIM by 1.43%, and reduces LPIPS by 25.05%. Regarding computational overhead, it reduces training time by 60.94% and meshing reconstruction time by 65.17%. Furthermore, compared to the mesh models generated by SuGaR, F2DMAS achieves substantial improvements in the geometric reconstruction quality of plant edge regions and severely occluded leaves, presenting sharper boundary features. The overall architecture of F2DMAS consists of two core modules including front-end image processing and back-end 3D representation. The following sections discuss its core technical mechanisms and research limitations in depth.

Addressing challenges such as severe occlusion and drastic illumination changes common in complex agricultural environments23, strong cross-scene and cross-species generalization capabilities are prerequisites for the practical application of 3D reconstruction models. Traditional 3D reconstruction platforms are often limited by high hardware costs, lengthy processing times, cumbersome workflows, and restricted scene generalization capabilities, making it difficult to achieve normalized large-scale applications in real large-scale farmlands or uncontrolled greenhouse scenes24. Therefore, we construct a comprehensive evaluation dataset encompassing two modalities of fixed-instrument acquisition and complex-environment acquisition to verify the robustness of the proposed framework. The performance advantage of the F2DMAS framework highly depends on the refined processing of input views, a process driven by the front-end foundational vision module FSAM3. Based on the SAM3 large model 25, FSAM3 introduces a degraded image hard distinction mechanism based on the fast Fourier transform and a semantic prompt filtering post-processing strategy. Without additional manual intervention, this module achieves continuous and robust segmentation of target plants in complex backgrounds relying solely on text prompts. Compared to the general semantic segmentation model SEEM26, FSAM3 performs better in maintaining semantic consistency and preserving background information such as soil. Quantitative evaluation shows that the F1 score and mIoU of FSAM3 reach 98.3 and 97.9% respectively, increasing by 3.2 and 3.8 percentage points over SEEM. Meanwhile, its HD95 metric drops significantly from 281.9 pixels to 41.4 pixels. This error reduction of 240.5 pixels directly quantifies the elimination effect of this module on severe boundary deviations, mathematically confirming that FSAM3 effectively resolves the missing plant part issue present in the SEEM method and provides strong support for efficient and precise smart agriculture.

At the 3D representation level, early agricultural 3D reconstruction relied on active vision technologies such as LiDAR or passive optical methods such as structure from motion27. However, the former suffers from high costs and massive data processing loads28, while the latter easily experiences feature matching failures when handling slender structures like rice leaves and wheat stalks or highly reflective areas, resulting in missing point clouds and surface holes29. Neural rendering paradigms represented by neural radiance fields30 and 3D Gaussian splatting31 reshape the technical path of 3D reconstruction. Although neural radiance fields implicitly resolve complex volume densities and synthesize high-quality novel views, their massive computational overhead limits downstream applications32. While 3DGS achieves a balance between fidelity and rendering efficiency, directly converting it into a mesh model still faces bottlenecks including long processing times and high computational resource consumption33. We explore the application potential of 2DGS in agricultural plant 3D reconstruction to address these issues. Experimental results demonstrate that on the self-built dataset, the F2DMAS framework achieves photorealistic rendering quality with a PSNR of 31.09, significantly outperforming the baseline 3DGS (30.17) and the traditional COLMAP (13.63). After performing plant mesh modeling via the TSDF algorithm (as shown in Fig.7(b)), the model extracts core phenotypic parameters such as plant height, crown width, leaf length, and leaf width with high precision, showing high consistency with measured data (). These results prove that 2DGS effectively overcomes the limitations of traditional methods in reconstructing complex plant geometric structures and possesses the potential to expand to complex field environments, providing a cost-reducing and efficiency-enhancing 3D digitization solution for precision agriculture.

Despite these advantages, limitations in marginal depth optimization persist. The observed 9% amplification bias in leaf width extraction stems fundamentally from the inherent depth ambiguity of the 2DGS ray-surfel intersection mechanism when rasterizing high-frequency physical boundaries (e.g., serrated leaf margins) at glancing angles. Furthermore, the Truncated Signed Distance Function (TSDF) utilized in the meshing phase exhibits extreme sensitivity to this marginal depth variance, leading to a physical outward expansion of the extracted isosurface. This marginal dilation remains a common structural challenge for explicit Gaussian representations at microscopic physical boundaries, which constitutes a primary direction for our future optimization.

Regarding practical application and popularization, while F2DMAS significantly lowers the data acquisition barrier by relying solely on consumer-grade smartphones, the underlying FSAM3 inference and 2DGS optimization still demand considerable GPU resources. To bridge the gap between laboratory algorithms and widespread agricultural deployment, the operational paradigm of this framework is designed for a cloud-edge collaborative architecture. End-users only need to perform hardware-light data collection by capturing multi-view RGB sequences via mobile devices. The computationally intensive 3D reconstruction and automated phenotypic trait extraction are subsequently executed on cloud-based servers. Coupled with the zero-shot semantic decoupling of FSAM3—which eliminates the need for end-users to perform tedious manual annotations or train specific models for new crop varieties—this deployment strategy fundamentally democratizes high-throughput 3D phenotyping. It transforms complex computer vision pipelines into an accessible, plug-and-play daily management tool for scalable precision agriculture.

Although F2DMAS demonstrates excellent performance in cross-scene plant reconstruction, this study still has certain limitations that point the direction for subsequent work. First, the FSAM3 module serving as the core of multi-view image preprocessing highly depends on prompts to generate segmentation masks. In extremely complex large-scale natural scenes, the effectiveness of a single prompt might be limited and requires repeated iterative optimization, which increases the time cost and operational threshold for practical deployment. Future research focuses on targeted fine-tuning of FSAM3 to enhance its scene adaptability under weak or prompt-free conditions. Second, while the current evaluation dataset covers two acquisition environments, it lacks real field scenes and full life-cycle data of plants. Subsequent work needs to construct a larger-scale and more diverse agricultural dataset to further validate and optimize the framework. Finally, the plant phenotypic metrics extracted by the current system are limited to basic geometric parameters such as plant height, crown width, leaf length, and leaf width. Future work aims to expand the algorithm boundaries to extract deep-level agricultural phenotypic data such as leaf area, leaf inclination angle, and stalk morphology, comprehensively improving the universality and practical value of this framework in precision agriculture applications.

Conclusion

In this study, we developed F2DMAS, a highly deployable plant digital twin framework tailored for high-fidelity phenotyping in unstructured environments. By overcoming the limitations of standard 3D reconstruction techniques on non-rigid thin-walled structures, F2DMAS successfully mitigates the topological expansion issue that traditionally plagues canopy and leaf morphology estimations. The integration of zero-shot background decoupling ensures its robustness across varying agricultural scenarios, from controlled laboratories to complex greenhouses and unstructured indoor environments, using consumer-grade multi-view RGB image sequences.

Experimental results comprehensively verify the significant advantages of this framework in core processing stages. In the front-end segmentation task, the F1 score and mIoU of the FSAM3 module reach 98.3 and 97.9%, respectively, mitigating severe boundary deviations. In the back-end reconstruction task, compared to the standard 2DGS model, the F2DMAS workflow improves PSNR and SSIM by 5.10% and 1.43%, respectively. Furthermore, model training and meshing reconstruction times decrease substantially by 60.94% and 65.17%. Compared to traditional 3DGS and the cutting-edge SuGaR method, the proposed scheme effectively overcomes volumetric artifacts and improves data processing efficiency while ensuring geometric fidelity.

Virtual measurement data extracted from the generated 3D mesh models highly match the manual measurement ground truth, confirming extreme reliability in downstream phenotyping tasks. While minor marginal depth expansion persists at microscopic high-frequency boundaries, without altering the underlying hardware requirements, this framework significantly improves the physical accuracy of macroscopic 3D plant models. It establishes a reliable, computationally efficient geometric foundation for automated and non-destructive crop phenotypic analysis.

Declarations

**Ethics approval and consent to participate**

Not applicable.

**Consent for publication**

Not applicable.

**Data Availability**

The data that support this study are available upon reasonable request from the corresponding author. Code is available at [https://github.com/LiuLiShenShe/F2DMAS.git](https://github.com/LiuLiShenShe/F2DMAS.git).

**Competing interests**

The authors declare no competing interests.

**Funding**

This research was supported by the Basic Research Funds of Chinese Academy of Agricultural Sciences (Y2026JC11), Chinese Academy of Agricultural Sciences Innovation Project (No. CAAS-ASTIP-2026-AII)&the National Key R&D Program of China (2022ZD0119500).

**Authors' contributions**

Jian Fang, Jingchao Fan, and Xiaoli Wang conceived the calibration pipeline, experimental design, algorithm development, and data analysis strategy.Jian Fang and Nengfu Xie wrote the original draft of the manuscript.Jian Fang, Yane Duan, and Hailong Liu performed data collection and algorithm implementation.Huoguo Zheng, Hao Wu, Zhibo Meng, Xin Wang, and Rui Man contributed to the data analysis.All authors reviewed and edited the manuscript.Nengfu Xie and Yane Duan conceived the overall project and acquired the research funding.

**Acknowledgements**

This research was supported by the Basic Research Funds of Chinese Academy of Agricultural Sciences (Y2026JC11), Chinese Academy of Agricultural Sciences Innovation Project (No. CAAS-ASTIP-2026-AII)&the National Key R&D Program of China (2022ZD0119500).

**References**

1\. Nguyen HT, Khan MAR, Nguyen TT, et al. Advancing Crop Resilience Through High-Throughput Phenotyping for Crop Improvement in the Face of Climate Change. _Plants_. 2025;14(6). doi:10.3390/plants14060907

2\. Walsh JJ, Mangina E, Negrão S. Advancements in Imaging Sensors and AI for Plant Stress Detection: A Systematic Literature Review. _Plant Phenomics_. 2024;6:0153. doi:10.34133/plantphenomics.0153

3\. Chen H, Liu S, Wang C, et al. Point Cloud Completion of Plant Leaves under Occlusion Conditions Based on Deep Learning. _Plant Phenomics_. 2023;5:0117. doi:10.34133/plantphenomics.0117

4\. Qin Y, Tauqir M, Yu X, et al. Predicting Multiple Traits of Rice and Cotton Across Varieties and Regions Using Multi-Source Data and a Meta-Hybrid Regression Ensemble. _Sensors_. 2026;26(2). Accessed February 2, 2026. https://www.mdpi.com/1424-8220/26/2/375

5\. Yang S, Feng Q, Zhang J, Yang W, Zhou W, Yan W. From laboratory to field: cross-domain few-shot learning for crop disease identification in the field. _Front Plant Sci_. 2024;15:1434222. doi:10.3389/fpls.2024.1434222

6\. Wang T, Tong R, Xu T, Li Y, Chen Y. Artificial intelligence in plant science: from image-based phenotyping to yield and trait prediction. _Front Plant Sci_. 2026;16:1732979. doi:10.3389/fpls.2025.1732979

7\. Zhu X, Huang Z, Li B. Three-Dimensional Phenotyping Pipeline of Potted Plants Based on Neural Radiation Fields and Path Segmentation. _Plants_. 2024;13(23). doi:10.3390/plants13233368

8\. Andeer PF, Zwart PH, Ushizima D, et al. Frontiers \| EcoBOT: an AI/ML enabled automated phenotyping capability for model plants. doi:10.3389/fpls.2025.1633557

9\. Li S, Cui Z, Yang J, Wang B. A Review of Optical-Based Three-Dimensional Reconstruction and Multi-Source Fusion for Plant Phenotyping. _Sensors (Basel)_. 2025;25(11):3401. doi:10.3390/s25113401

10\. Li Y, Liang Z, Liu B, et al. Applications of 3D Reconstruction Techniques in Crop Canopy Phenotyping: A Review. _Agronomy_. 2025;15(11):2518. doi:10.3390/agronomy15112518

11\. Harandi N, Vandenberghe B, Vankerschaver J, Depuydt S, Van Messem A. How to make sense of 3D representations for plant phenotyping: a compendium of processing and analysis techniques. _Plant Methods_. 2023;19(1):60. doi:10.1186/s13007-023-01031-z

12\. Wen W, Wang J, Zhao Y, et al. 3D Morphological Feature Quantification and Analysis of Corn Leaves. _Plant Phenomics_. 2024;6:0225. doi:10.34133/plantphenomics.0225

13\. Zheng X, AI X, Qin H, et al. Tomato-Nerf: Advancing Tomato Model Reconstruction With Improved Neural Radiance Fields. _IEEE Access_. 2024;12:184206-184215. doi:10.1109/ACCESS.2024.3424908

14\. Huang B, Yu Z, Chen A, Geiger A, Gao S. 2D Gaussian Splatting for Geometrically Accurate Radiance Fields. In: _Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers_. ACM; 2024:1-11. doi:10.1145/3641519.3657428

15\. Yu Z, Chen A, Huang B, Sattler T, Geiger A. Mip-Splatting: Alias-Free 3D Gaussian Splatting. In: _2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_. IEEE; 2024:19447-19456. doi:10.1109/CVPR52733.2024.01839

16\. Liu Q, Wang C, Jiang J, et al. Multi-source data fusion improved the potential of proximal fluorescence sensors in predicting nitrogen nutrition status across winter wheat growth stages. _Computers and Electronics in Agriculture_. 2024;219:108786. doi:10.1016/j.compag.2024.108786

17\. Colovic M, Stellacci AM, Mzid N, et al. Comparative Performance of Aerial RGB vs. Ground Hyperspectral Indices for Evaluating Water and Nitrogen Status in Sweet Maize. _Agronomy_. 2024;14(3). doi:10.3390/agronomy14030562

18\. Singh R, Bidese R, Dhakal K, Sornapudi S. Few-Shot Adaptation of Grounding DINO for Agricultural Domain. doi:10.1109/CVPRW67362.2025.00530

19\. Zhu H, Qin S, Su M, Lin C, Li A, Gao J. Frontiers \| Harnessing large vision and language models in agriculture: a review. doi:10.3389/fpls.2025.1579355

20\. Chen J, Jiao Y, Jin F, et al. Plant Sam Gaussian Reconstruction (PSGR): A High-Precision and Accelerated Strategy for Plant 3D Reconstruction. _Electronics_. 2025;14(11):2291. doi:10.3390/electronics14112291

21\. Cannici M, Scaramuzza D. Mitigating Motion Blur in Neural Radiance Fields with Events and Frames. In: _2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_. IEEE; 2024:9286-9296. doi:10.1109/CVPR52733.2024.00887

22\. Ravi N, Gabeur V, Hu YT, et al. SAM 2: Segment Anything in Images and Videos.

23\. Rincón MG, Mendez D, Colorado JD. Four-Dimensional Plant Phenotyping Model Integrating Low-Density LiDAR Data and Multispectral Images. _Remote Sensing_. 2022;14(2):356. doi:10.3390/rs14020356

24\. Wu S, Hu C, Tian B, et al. A 3D reconstruction platform for complex plants using OB-NeRF. _Front Plant Sci_. 2025;16:1449626. doi:10.3389/fpls.2025.1449626

25\. Carion N, Gustafson L, Hu YT, et al. SAM 3: Segment Anything with Concepts. _arXiv_. Preprint posted online November 20, 2025:arXiv:2511.16719. doi:10.48550/arXiv.2511.16719

26\. Zou X, Yang J, Zhang H, et al. Segment Everything Everywhere All at Once. _arXiv_. Preprint posted online July 11, 2023:arXiv:2304.06718. doi:10.48550/arXiv.2304.06718

27\. Zhu Y, Sun G, Ding G, et al. Large-scale field phenotyping using backpack LiDAR and CropQuant-3D to measure structural variation in wheat. _Plant Physiol_. 2021;187(2):716-738. doi:10.1093/plphys/kiab324

28\. Jin S, Sun X, Wu F, et al. Lidar sheds new light on plant phenomics for plant breeding and management: Recent advances and future prospects. _ISPRS Journal of Photogrammetry and Remote Sensing_. 2021;171:202-223. doi:10.1016/j.isprsjprs.2020.11.006

29\. Choi HB, Park JK, Park SH, Lee TS. Frontiers \| NeRF-based 3D reconstruction pipeline for acquisition and analysis of tomato crop morphology. doi:10.3389/fpls.2024.1439086

30\. Mildenhall B, Srinivasan PP, Tancik M, Barron JT, Ramamoorthi R, Ng R. NeRF: representing scenes as neural radiance fields for view synthesis. _Commun ACM_. 2022;65(1):99-106. doi:10.1145/3503250

31\. Kerbl B, Kopanas G, Leimkuehler T, Drettakis G. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. _ACM Trans Graph_. 2023;42(4):1-14. doi:10.1145/3592433

32\. Zhao J, Ying W, Pan Y, et al. Exploring Accurate 3D Phenotyping in Greenhouse through Neural Radiance Fields. _arXiv_. Preprint posted online March 28, 2024:arXiv:2403.15981. doi:10.48550/arXiv.2403.15981

33\. Park J, Suh JW, Ban Y. Dual-Dimensional Gaussian Splatting Integrating 2D and 3D Gaussians for Surface Reconstruction. _Applied Sciences_. 2025;15(12):6769. doi:10.3390/app15126769