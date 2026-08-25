# Translation and Extraction Notes

- Source PDF: `/data/fj/F2DMAS/00参考文章/1-s2.0-S2643651526000373-main-2.pdf`
- PDF type: selectable-text PDF.
- Paper type: 3D reconstruction / plant phenotyping methods paper.
- Translation method: NLLB machine translation with domain-term post-processing.
- Draft-mode caveat: equations, tables, references, and complex multi-panel figure pages may need human polishing.
- Text/caption blocks: 236; figure crops: 7.

## Low/Medium Confidence Blocks

- `S025` p.4 (body, medium): camera parameter estimation, a hierarchical optimization strategy is employed to ensure global consistency. In this progress, we initially take advantage of pixel correspondences t
- `S029` p.4 (body, medium): and this is performed iteratively using the Adam optimizer [38]. We σʹ reparameterize σ as σ = min σ to ensure that the minimum value of σ is 1, which helps avoid degenerate soluti
- `S032` p.4 (body, medium): First, we perform image matching on the input images and complete local pairwise reconstruction. An effective and scalable MASt3R encoder integrated with Aggregated Selective Match
- `S037` p.4 (body, medium): The initialization process of 3DGS is straightforward, beginning with sparse point clouds in SFM format reconstructed from multi-view im­ ages, serving as the initial positions of 
- `S038` p.4 (body, medium): (x− μ) e− 2(x− μ); G x; μ; = (6) 3 ∑ 1 (2π)2 | |2
- `S039` p.4 (body, medium): where Xn;e;i;j represents the obtained estimate value of Xn;n from edge e. The canonical depth map is then extracted from the canonical point map ~n = X ~ n;:;:;3, and the Weiszfel
- `S041` p.4 (body, medium): where x is the position of any point in space, and μ represents the mean of the initialized Gaussian distribution, indicating the center position of ∑ each Gaussian point, 3×3 is t
- `S042` p.4 (body, medium): assuming the pinhole model with central principal point and square pixels. To ensure the 3D point cloud strictly adheres to the pinhole camera model and precisely corresponds to pi
- `S047` p.5 (body, medium): Gaussian projection transformation, instantiation, and global sorting operations. By adjusting the shape and orientations of Gaussians, anisotropic variance is rendered; the final 
- `S048` p.5 (body, medium): points, using K-Means clustering to initialize Gaussian points' means μ based on the input sparse points. Assuming the input sparse point set P = p1; p2, p3 …, pn,where pi is a poi
- `S051` p.5 (body, medium): ⃒ ⃒ where Cj represents the set of all points in the j-th cluster, ⃒Cj ⃒ represents ∑ the number of points in the j-th cluster, and pi ϵCj pi represents the sum
- `S053` p.5 (body, medium): of position vector of all points in the cluster Cj. The covariance matrix controls the scaling and rotation of the Gaussian splat, mathematically represented as ∑ = R(q)S(s)S(s)T R
- `S058` p.5 (body, medium): reconstruction The core of our approach is the optimization step, aiming to better fit the reconstructed model by adjusting the parameters of the Gaussian distribution, so that the
- `S060` p.5 (body, medium): and λ = 0.2 in our experiments, employing L 1 loss to ensure pixel-level precision in rendering processes, and D-SSIM loss to ensure the struc­ tural similarity of the rendering re
- `S122` p.7 (body, medium): Note: The term “Total Points” denote the number of points obtained in sparse reconstruction; “Plant Points” represent the number of the plant; “Plant Point Ratio” refers to the pro
- `S162` p.9 (body, medium): H = max(yi) − yplane () where max yi denotes the Y-coordinate of the highest point on the plant, and yplane represents the mean Y-coordinate of the fitted soil plane. The compariso

## Figure Crop Notes

- `F001` p.3 `assets/fig1.png` bbox=[81.47, 52.43, 513.86, 592.1] caption=manual-layout
- `F002` p.7 `assets/fig2.png` bbox=[39.07, 500.84, 287.27, 700.38] caption=manual-layout
- `F003` p.8 `assets/fig3.png` bbox=[53.24, 52.43, 273.08, 306.25] caption=manual-layout
- `F004` p.8 `assets/fig4.png` bbox=[343.51, 505.45, 520.72, 700.38] caption=manual-layout
- `F005` p.9 `assets/fig5.png` bbox=[123.94, 52.39, 471.36, 171.72] caption=manual-layout
- `F006` p.9 `assets/fig6.png` bbox=[67.19, 390.86, 528.08, 729.07] caption=manual-layout
- `F007` p.11 `assets/fig7.png` bbox=[67.19, 52.43, 528.08, 329.16] caption=manual-layout

## Known Limitations

- Tables are represented as caption/source blocks and nearby prose; exact cell-level table reconstruction is not guaranteed.
- Figure crops use PDF image-object clustering. For pages composed of many subimages, crops may cover the whole visual panel instead of individual subpanels.
- References are retained as source blocks when extractable but are not translated.
