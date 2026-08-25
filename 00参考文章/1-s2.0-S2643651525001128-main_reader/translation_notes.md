# Translation and Extraction Notes

- Source PDF: `/data/fj/F2DMAS/00参考文章/1-s2.0-S2643651525001128-main.pdf`
- PDF type: selectable-text PDF with two-column Elsevier layout.
- Paper type: methods / algorithm paper for interactive unsupervised 3D plant phenotyping.
- Translation method: NLLB machine translation with domain-term post-processing; section titles and recurring captions are manually stabilized.
- Draft-mode caveat: equations, algorithms, dense tables, and references were preserved with lower translation confidence where layout extraction was noisy.
- Text/caption blocks: 277; figure crops: 8.

## Low/Medium Confidence Blocks

- `S006` p.1 (body, medium): Advanced plant phenotyping technologies are vital for trait improvement and accelerating intelligent breeding. Due to the species diversity of plants, existing methods heavily rely
- `S009` p.2 (body, medium): advancements enable researchers to analyze complex morphological features including shape, area, and angles with enhanced precision [9, 10]. In recent years, Neural Radiance Fields
- `S027` p.3 (body, medium): Neural Radiance Fields (NeRF) learn a continuous volumetric scene representation from a training dataset I of multi-view 2D images. The model approximates a function fθMask Represe
- `S037` p.4 (body, medium): represents distance from the camera center. The RGB color Iθ(r) for ray r is computed via differentiable volume rendering: ∫ tf Iθ (r) = ω(r(t))c(r(t); d) dt; (1) tn
- `S038` p.4 (body, medium): (where the transmittance-weighted density is given by ω(r(t)) = exp −) ∫t tn σ(r(s)) ds ⋅σ(r(t)). The integration bounds tn and tf correspond to the near and far planes of the view
- `S041` p.4 (body, medium): 2.6. 3D Mask Representation and loss function 3D Mask Representation: To represent the 3D mask, we utilize a voxel grid V ∈ ℝL×W×H, where each voxel is initialized with a soft mask
- `S043` p.5 (body, medium): ∫ tf M(r) =
- `S048` p.5 (body, medium): where r(t) denotes the ray position at parameter t, w(r(t)) is the ray weight function, and G(r(t)) is defined as: ⎡ ⎤ Vo1 (r(t)) ⎢ Vo2 (r(t)) ⎥ ⎥: G(r(t)) = ⎢ (3) ⎣⋮ ⎦ Von (r(t)) 
- `S049` p.5 (body, medium): ∑ ∑ [(
- `S058` p.6 (body, medium): Algorithm 1. Residual handling
- `S059` p.6 (body, medium): Input:Segmentation mask M ∈ ℝH×W
- `S060` p.6 (body, medium): ̃ ∈ ℝH×W Output: Post-processed binary mask M for each pixel p in M do if p > 0 then p ← 255 {Set as foreground} else p ← 0 {Set as background} end if end for K ←ones(5, 5) {Define
- `S061` p.6 (body, medium): Algorithm 2.
- `S063` p.6 (body, medium): Input:reference frame and images Output:first and last indices pre_ref ← Preprocess(reference) files ← SortedList(images) indices ←∅ for (idx, f) ∈ files in parallel do img ← ReadI
- `S064` p.6 (body, medium): ̃ M=255 ̃ M← {Normalize to [0, 1]} ̃ ̃ (1; H; W)) {Adjust output dimensions} M←Reshape(M;
- `S066` p.6 (body, medium): the segmentation accuracy of the model, it is necessary to provide prompts for rear frames. Therefore, to identify rear frames and improve detection efficiency under a large amount
- `S068` p.7 (body, medium): between the vector formed by the next base point and bk and the principal component axis) ((qj − bk)⊤ v1 ⃦ θj = arccos ⃦ (11) ⃦qj − bk ‖ ⋅⃦v1 ‖ < θmax;
- `S070` p.7 (body, medium): where θmax = π/2 is the maximum allowable deviation angle. If θj exceeds the maximum angle π/2, skip it; otherwise, select it. If no points meet the criteria, return the point with
- `S072` p.7 (body, medium): s5: Downsample the collected base point set B = {bm }M m=1 to obtain a set of base points along the curved path of the leaf. Summing the distances between consecutive points gives 
- `S074` p.7 (body, medium): The leaf width is calculated based on a mesh parameterization method that flattens the leaf in the width direction, followed by 2D interpolation. The interval width is defined as t
- `S076` p.7 (body, medium): b = v1 − v3:
- `S077` p.7 (body, medium): Calculate the cross product of vectors a and b to obtain a vector c = a × b that is perpendicular to the plane of the triangle. Calculate half the magnitude of the cross product as
- `S078` p.7 (body, medium): A△ = 0:5 × ‖c‖:
- `S080` p.7 (body, medium): ‖bm+1 − bm ‖2: m=1
- `S082` p.7 (body, medium): The mesh consists of a series of triangular facets, and the leaf area is approximated by calculating the sum of the areas of all triangles. The three vertices of each triangle are 
- `S086` p.7 (body, medium): Calculate the voxel volume V using the following formula based on the voxel size and the number of voxel units num_voxels in the point cloud, to obtain the volumes of grains and pa
- `S087` p.7 (body, medium): Sum the areas of all triangles to obtain the surface area of the mesh: ∑ Amesh = A△: (8)
- `S091` p.7 (body, medium): where Ri ∈ SO(2) is the local rotation matrix, and wij is the cotangent weight. s2: Let the parameterized coordinates be {ui = (ui; vi)⊤ }. Calculate the principal component direct
- `S092` p.7 (body, medium): preprocessed leaf point cloud be P = {pi ∈ ℝ3 }i=1, where pi =
- `S093` p.7 (body, medium): (xi; yi; zi)⊤. Perform principal component analysis (PCA) on it and calculate the covariance matrix: C=
- `S094` p.7 (body, medium): N 1 ∑ (p − p)(pi − p)⊤; N i=1 i
- `S096` p.7 (body, medium): w = arg max Var({u⊤ i a}): ‖a‖=1
- `S099` p.7 (body, medium): k sk = smin + (smax − smin); n
- `S100` p.7 (body, medium): where p is the centroid of the point cloud. Through eigenvalue decomposition C = VΛV⊤, the main direction v1 (corresponding to the largest eigenvalue λ1) is obtained. Treat the two
- `S101` p.7 (body, medium): k = 0; 1; …; n
- `S103` p.7 (body, medium): ⊤ where smin = mini u⊤ i w, smax = maxi ui w. For each sampling position sk,
- `S104` p.7 (body, medium): search for boundaries along the normal direction w⊥ = [− wy; wx]⊤: { l ⊥ tk = minu⊤i w=sk u⊤ i w (17) ⊥ trk = maxu⊤i w=sk u⊤ i w
- `S105` p.7 (body, medium): s4: Calculate the leaf width: For each row of interpolated boundary points, compute the Euclidean distance in the width direction Wk = trk − tlk. Take the maximum value across all 
- `S106` p.7 (body, medium): s2: Build a KD-tree T = BuildKDTree(P). Starting from the starting point, perform K-nearest neighbor (KNN) search. Sort the K candidate points in a greedy manner based on their dis
- `S107` p.7 (body, medium): W = max Wk: 1≤k≤n
- `S108` p.7 (body, medium): N k = {qj }Kj=1, satisfying N k = KNN(T; bk; K). Calculate the angle
- `S112` p.8 (body, medium): Common 3D segmentation evaluation metrics were used: IoU, Precision, Recall, F1-score, and average inference time. For each instance, IoU is calculated by computing the intersectio
- `S114` p.8 (body, medium): MAE =
- `S119` p.8 (body, medium): where N is the total number of categories. Precision represents the proportion of true positive samples among those predicted as positive, indicating the reliability of the predict
- `S122` p.8 (body, medium): Recall represents the proportion of the true region that is covered by the prediction. A higher value indicates fewer missed detections. Recalli =
- `S125` p.8 (body, medium): F1-score is the harmonic mean of precision and recall, providing a comprehensive measure of model performance. A higher value indicates better overall performance of the model. F1i
- `S128` p.8 (body, medium): Average inference time is a common speed metric used to evaluate model performance, representing the average time consumed by the model for one instance segmentation prediction. As
- `S129` p.8 (body, medium): M 1 ∑ Tj M j=1
- `S131` p.8 (body, medium): R2 evaluates the goodness of fit of the model by explaining the proportion of variance in the dependent variable. Its range is [0,1], and a value closer to 1 indicates better model
- `S132` p.8 (body, medium): n i=1 (yi − y)
- `S133` p.8 (body, medium): where: Pi is the region predicted by the model for the ith instance, and Ti is its true region. N 1 ∑ mIoU = IoUi N i=1
- `S134` p.8 (body, medium): n 1∑ |yi − ̂ yi| n i=1
- `S138` p.8 (body, medium): where Tj is the inference time for the jth prediction. To evaluate the phenotypic quality of the segmented targets, we compared the predicted results with the ground truth. Root Me
- `C015` p.11 (caption, low): Table 4b shows the surface area results (c) Rice (Length & Width). o M f A w E h r e e a a t c h le a 0 v. 6 e 7 s. a I n t d c a 0 n. 5 b 3 e, a f n ou d n R d 2 t r h e a a t c h

## Figure Crop Notes

- `F001` p.3 `assets/fig1.png` bbox=[75.39, 456.23, 519.88, 728.07] caption=C002
- `F002` p.4 `assets/fig2.png` bbox=[124.99, 53.39, 470.27, 321.92] caption=C004
- `F003` p.5 `assets/fig3.png` bbox=[75.39, 53.43, 519.88, 324.7] caption=C006
- `F004` p.6 `assets/fig4.png` bbox=[124.99, 53.41, 470.27, 355.2] caption=C007
- `F005` p.9 `assets/fig5.png` bbox=[75.33, 515.32, 519.97, 728.12] caption=C010
- `F006` p.10 `assets/fig6.png` bbox=[124.93, 53.42, 470.36, 350.89] caption=C012
- `F007` p.10 `assets/fig7.png` bbox=[75.39, 521.96, 519.88, 718.49] caption=C014
- `F008` p.12 `assets/fig8.png` bbox=[124.93, 447.3, 470.36, 728.07] caption=C023

## Known Limitations

- Tables are represented as caption/source blocks and nearby prose; exact cell-level table recreation was not guaranteed for all tables because the PDF interleaves table columns with body text.
- References are retained as source blocks when extractable but are not fully translated, to avoid low-value noisy translation.
- Formula-heavy paragraphs may preserve original symbols with only partial Chinese explanation.
