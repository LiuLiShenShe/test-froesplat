# Translation and Extraction Notes

- Source PDF: `/data/fj/F2DMAS/00参考文章/plantphenomics.0235.pdf`
- PDF type: selectable-text PDF.
- Paper type: 3D reconstruction / plant phenotyping methods paper.
- Translation method: NLLB machine translation with domain-term post-processing.
- Draft-mode caveat: equations, tables, references, and complex multi-panel figure pages may need human polishing.
- Text/caption blocks: 253; figure crops: 6.

## Low/Medium Confidence Blocks

- `S079` p.5 (body, medium): NeRFs model a scene as a continuous function mapping a 3D position x = (x, y, z) and a 2D viewing direction d = (θ, ϕ) to a color c = (r, g, b) and density σ. The function is param
- `S082` p.5 (body, medium): C(r) =
- `S083` p.5 (body, medium): ∫ tn
- `S087` p.5 (body, medium): ) (t where T(t) = exp − ∫t 𝜎(r(s))ds represents the accumulated n transmittance along the ray r(t) = o + td, with o being the ray origin and [tn, tf] the near and far bounds. In ou
- `S112` p.7 (body, medium): To assess the similarity between the ground truth (obtained from TLS) and the reconstructed 3D point cloud, the following metrics are employed: 1. Precision/Accuracy. Given a recon
- `S113` p.7 (body, medium): Its value ranges from 0 to 100, with higher values indicating better performance. Both the above 2 metrics are extensively utilized in recent studies [43,48]. 3. F-score. The F-sco
- `S116` p.7 (body, medium): The harmonic nature of the F-score ensures that if either P(d) or R(d) approaches zero, the F-score will also tend toward zero, providing a more robust summary statistic than the a
- `S119` p.7 (body, medium): (PSNR = 10 ⋅ log10
- `S122` p.7 (body, medium): where MAXI is the maximum possible pixel value of the image, and MSE is the mean squared error between the reference and the reconstructed image. The MSE is given by: MSE =
- `S123` p.7 (body, medium): n m ())2 1 ∑ ∑(() I i, j − K i, j, mn i=1 j=1
- `S128` p.7 (body, medium): () SSIM x, y = ()(), 𝜇2x + 𝜇2y + C1 𝜎 2x + 𝜎 2y + C2
- `S132` p.7 (body, medium): where 𝕀(⋅) is an indicator function. Precision values ranges from 0 to 100, with higher values indicating better performance. 2. Recall/Completeness. Conversely, the recall metric 
- `S138` p.8 (body, medium): The significant negative correlation between LPIPS and the F1 score (−0.82), PSNR (−0.81), and SSIM (−0.69) underscore the impact of LPIPS on the quality of 3D reconstruction (see 

## Figure Crop Notes

- `F001` p.2 `assets/fig1.png` bbox=[49.29, 394.37, 548.63, 697.52] caption=manual-layout
- `F002` p.4 `assets/fig2.png` bbox=[42.45, 72.74, 550.07, 705.05] caption=manual-layout
- `F003` p.9 `assets/fig3.png` bbox=[117.61, 77.85, 548.67, 155.76] caption=manual-layout
- `F004` p.9 `assets/fig4.png` bbox=[49.21, 249.26, 525.03, 353.45] caption=manual-layout
- `F005` p.11 `assets/fig5.png` bbox=[75.85, 63.1, 557.55, 621.49] caption=manual-layout
- `F006` p.13 `assets/fig6.png` bbox=[77.65, 63.21, 557.55, 672.83] caption=manual-layout

## Known Limitations

- Tables are represented as caption/source blocks and nearby prose; exact cell-level table reconstruction is not guaranteed.
- Figure crops use PDF image-object clustering. For pages composed of many subimages, crops may cover the whole visual panel instead of individual subpanels.
- References are retained as source blocks when extractable but are not translated.
