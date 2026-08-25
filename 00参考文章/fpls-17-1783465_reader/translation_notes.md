# Translation and Extraction Notes

- Source PDF: `/data/fj/F2DMAS/00参考文章/fpls-17-1783465.pdf`
- PDF type: selectable-text PDF.
- Paper type: 3D reconstruction / plant phenotyping methods paper.
- Translation method: NLLB machine translation with domain-term post-processing.
- Draft-mode caveat: equations, tables, references, and complex multi-panel figure pages may need human polishing.
- Text/caption blocks: 364; figure crops: 11.

## Low/Medium Confidence Blocks

- `S010` p.1 (body, medium): Automated quantification of plant-level development from multi-plant greenhouse scenes requires separating individual plants from shared scene-level reconstructions and quantifying
- `S040` p.4 (body, medium): motion yielded sufficient parallax. COLMAP was then used to estimate calibrated camera intrinsics and extrinsics and to reconstruct a sparse, geometrically consistent point cloud t
- `S052` p.5 (body, medium): g,v xmax(cosqg,v, 0)rv, wv = a
- `S053` p.5 (body, medium): g,v denotes the mean opacity over W(E g,v) as a visibility where a proxy, qg,v is the angle between the viewing ray in view v and a principal axis derived from Sg, and rv ∈ [0, 1]
- `S055` p.5 (body, medium): S(g) = ov wv
- `S058` p.5 (body, medium): o kg,v (u)Mv (u), sg,v = ag~sg,v:
- `S060` p.5 (body, medium): F = f g ∣ S(g) ≥ t g:
- `S064` p.5 (body, medium): Because the lifted foreground set may still contain background fragments or multi-plant overlap, geometric clustering is required to consolidate spatially coherent plant structures
- `S068` p.5 (body, medium): ~sg,v =
- `S070` p.5 (body, medium): Z(s ∣ S) =
- `S073` p.5 (body, medium): where S is the set of values for a specific feature across all Gaussians, and MAD is the median absolute deviation. This operator centers each feature by its median and rescales by
- `S074` p.5 (body, medium):    (xg xj), yg0 = Z(yg yj), zg0 = Z(zg zj),
- `S079` p.6 (body, medium): rg =
- `S083` p.6 (body, medium): where t(z) is the z -quantile of the in-component Mahalanobis distance distribution. Unless otherwise noted, z = 0:80 is used. During training, a relaxed coverage z = 0:90 is used 
- `S084` p.6 (body, medium): where c represents a specific camera center vector. Standardizing rg produces a depth feature that is directly comparable to the three coordinate features (Equation 8):  dg0 = Z(
- `S086` p.6 (body, medium): The complete clustering descriptor fg is then assembled by concatenating the four standardized components (Equation 9): fg = ½xg0, yg0, zg0, dg0 ,
- `S088` p.6 (body, medium): so that Euclidean distance balances local geometric shape with  the depth cue. OPTICS is applied to the set fg using the Euclidean metric, and clusters are extracted using the ste
- `S093` p.6 (body, medium): where d~1NN denotes the median nearest-neighbor distance among the retained Gaussians and b is a fixed scale factor (default b = 2). This stage is purely geometric, acting as a rob
- `S094` p.6 (body, medium): x0i = c + (xi − c) ·
- `S099` p.6 (body, medium): which stabilizes voxel resolution, attention coverage, and grouping radii across scenes. This normalization addresses variation in point cloud density across scenes and is distinct
- `S103` p.7 (body, medium): expressed relative to density, r = g D, where g is a fixed scale factor. Two points in Sk are linked when ∥ x0i − x0j ∥ ≤ r. A breadth first search aggregates links into spatially 
- `S105` p.7 (body, medium): We extract six phenotypic traits relevant to muskmelon growth assessment: plant height, leaf surface area (LSA), leaf area index (LAI), leaf count, node count, and internode length
- `S107` p.7 (body, medium): domain Per-point labels from the normalized cloud are transferred back to the original 3DGS reconstruction for visualization and downstream trait computation. Only visible splats a
- `S108` p.7 (body, medium): rassign = admed,
- `S109` p.7 (body, medium): H = s · maxzi0:
- `S113` p.7 (body, medium): LSA = s2 oAa (E ‘), LAI = ‘=1
- `S119` p.7 (body, medium): alignment With organ-level segmentation complete and all leaf and stem instances mapped back into the 3DGS representation, the resulting labeled plant structures provide the founda
- `S131` p.8 (body, medium): separation quality The LCR-GS extraction pipeline allows users to adjust the number of seeded 2D cues per plant, trading annotation effort against separation quality. To quantify t
- `S179` p.9 (body, medium): separation, with plant splats consistently scoring above 0.6 and background splats below 0.4, resulting in cleaner boundaries and stronger spatial consistency. Increasing to five c
- `S216` p.12 (body, medium): were generated using a relaxed CIELAB chromaticity filter (90% quantile), deliberately including a small proportion of non-plant Gaussians. This strategy, which increases plant rec
- `S254` p.13 (body, medium): RMSE = 1.88 cm, MAPE = 6.4%), confirming that the 3DGS reconstruction accurately captures vertical plant structure. Statistical analysis reveals a near-constant systematic
- `S255` p.13 (body, medium): overestimation (mean residual: +1.58 cm, median: +1.60 cm, p < 10 - 8), with 29 of 30 plants showing positive residuals. Approximately 70% of residuals fall within ±2 cm and 93% wi
- `S258` p.14 (body, medium): The trait estimates nevertheless permit examination of withincohort inter-trait relationships (Figure 10). Size-related traits display strong correlations - height with leaf count 
- `S259` p.14 (body, medium): segments in the reconstructed plant volume. Under the uniformcontainer conditions of this study, the near-constant offset does not affect inter-plant ranking or relative trait comp
- `S264` p.14 (body, medium): Mean ± STD
- `S269` p.14 (body, medium): 31:4±13.3
- `S275` p.14 (body, medium): 1:45 ± 0:25
- `S278` p.14 (body, medium): m =m2
- `S281` p.14 (body, medium): 74.9 ± 22:8
- `S287` p.14 (body, medium): 6:9 ± 2:1
- `S293` p.14 (body, medium): 6:9 ± 3:2
- `S299` p.14 (body, medium): 3:7±0.9
- `S309` p.16 (body, medium): strong segmentation performance (AP50 = 0.924), and close agreement between 3DGS-derived and manually measured plant height (R² = 0.98) and leaf count (90% of estimates within ±1 l

## Figure Crop Notes

- `F001` p.3 `assets/fig1.png` bbox=[60.38, 574.59, 534.1, 749.36] caption=manual-layout
- `F002` p.4 `assets/fig2.png` bbox=[60.38, 523.39, 534.1, 740.35] caption=manual-layout
- `F003` p.9 `assets/fig3.png` bbox=[60.38, 80.0, 534.1, 244.57] caption=manual-layout
- `F004` p.9 `assets/fig4.png` bbox=[67.47, 410.69, 527.01, 740.35] caption=manual-layout
- `F005` p.10 `assets/fig5.png` bbox=[152.51, 80.0, 441.97, 281.59] caption=manual-layout
- `F006` p.11 `assets/fig6.png` bbox=[60.38, 80.0, 534.1, 470.83] caption=manual-layout
- `F007` p.12 `assets/fig7.png` bbox=[60.38, 80.0, 534.04, 295.37] caption=manual-layout
- `F008` p.13 `assets/fig8.png` bbox=[60.33, 80.0, 534.04, 333.24] caption=manual-layout
- `F009` p.13 `assets/fig9.png` bbox=[109.93, 551.74, 484.44, 740.4] caption=manual-layout
- `F010` p.15 `assets/fig10.png` bbox=[109.99, 80.0, 484.44, 249.33] caption=manual-layout
- `F011` p.15 `assets/fig11.png` bbox=[67.47, 602.14, 527.01, 731.33] caption=manual-layout

## Known Limitations

- Tables are represented as caption/source blocks and nearby prose; exact cell-level table reconstruction is not guaranteed.
- Figure crops use PDF image-object clustering. For pages composed of many subimages, crops may cover the whole visual panel instead of individual subpanels.
- References are retained as source blocks when extractable but are not translated.
