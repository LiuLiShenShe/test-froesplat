# Phase 19A — Reviewer Comments Extraction

**Generated:** 2026-09-14 | **Source streams:** 2 | **Total comments:** 32

---

> **Stream A** = real reviews from F2DMAS application-version submission (may target Frontiers).
> **Stream B** = simulated self-review for the CompAg ForeSplat paper (5-reviewer panel: EIC + R1 + R2 + R3 + Devil's Advocate).
>
> Both streams are audit inputs; the new paper must address both.

---

## Stream A — Real Reviews (F2DMAS Application-Version Submission)

### Reviewer 1 (A1) — English

**A1-C01** — Terminology (full names at first occurrence)

> The authors integrated several existing tools (FFT, SAM, 2DGS, TSDF). While this is good and valuable, it would have been helpful for readers to better understand them if the authors had explicitly stated and provided complete naming information when they first appeared in the paper's abstract or introduction part.

**A1-C02** — Table 6 fairness / SAM3 vs FSAM3 contribution

> The main comparison results in the table 6 shows an improvement, but standard 2DGS uses the original image with the background. What if standard 2DGS and your method used the blur picture? Please add a simple additional experiment to clarify this. If F2DMAS still leads, the claim is stronger. And for the term of Plant Seg., is the improvement simply from using SAM3, or from the proposed FSAM3 components? This is unclear.

**A1-C03** — Leaf width error source / point-cloud alternative

> The leaf-width results (R2 value is about 0.90, MAPE is 9.73%) are noticeably weaker than other traits. The authors mention this in the discussion, but I would encourage a slightly deeper look. Two non-fatal issues. As we know, leaf width measurement is very hard. Is the bias from segmentation, 2DGS surface dilation, or TSDF meshing? If you change the data into point cloud format whether the leaf width could be measured to avoid mesh-induced errors?

**A1-C04** — GPU configuration and training environment

> Please specify GPU configuration and training environment (Linux, Pytorch .etc ) for reproducibility.

**A1-C05** — Author affiliations repeated

> I do not know why the Author affiliations on page 1 appear repeatedly. This looks like a formatting glitch. Please contact the frontiers panel to correct this ASAP.

---

### Reviewer 2 (A2) — English

**A2-C01** — R² precision in Abstract

> Abstract: were the measurements carried out with four-digit accuracy? Apparently no (Fig. 8), so it is enough to retain two digits in r2.

**A2-C02** — RMSE without context

> Absolute values of RMSE are non-informative in the absence of corresponding averages; either provide the latter or present a relative error metric (%).

**A2-C03** — Section 2.1 structure

> Section 2.1: the second paragraph of this section is more appropriate for introduction.

**A2-C04** — Acquisition time and throughput

> It is interesting to know how much time it typically takes to acquire raw data for a single plant in a typical fixed or complex scenario. It would be interesting to estimate the throughput of the suggested pipeline (plants processed per unit time).

**A2-C05** — Software implementation details / reproducibility

> It would be desirable to have more specific information on the details of software implementation of the algorithms and models employed by the authors (frameworks, programming libraries, etc.). Moreover, the GIT repository contains a lot of scripts with the names which are not always self-explanative, so it is desirable to include clarifications mapping all steps/operation to specific software modules/scripts from the repository. Including some test datasets would be helpful as well.

**A2-C06** — Discussion lacks references

> It is strange to see the Discussion section completely lacking references to the literature. Normally. This section is dedicated to comparison of the findings obtained with those previously accumulated hence the references to previous works are expected. At the same time, the Methods section contains a lot of references. Please consider restructuring these sections.

---

## Stream B — Simulated 5-Reviewer Self-Review (CompAg ForeSplat Paper)

### EIC — Prof. David Rousseau

**B-EIC-C01** — Cross-species evidence gap

> The paper claims "cross-species" in the title and throughout, but the reconstruction validation uses only 3 samples. While 20 samples are available for mask generation, the reconstruction claim—which is the paper's core contribution—rests on 3 samples. This is a significant gap between the framing and the evidence.

**B-EIC-C02** — FSAM3 segmentation evaluation

> The paper explicitly declines to benchmark FSAM3 against existing segmentation methods. While I accept the rationale (no pixel-level GT), a simple qualitative comparison against ExG/Otsu on a few representative frames would substantially strengthen the mask generation contribution.

**B-EIC-C03** — Missing ablation: M1-soft vs M4 independent contribution

> Table 5 reports the four-way combination (A6, A6+M1-soft, A6+M4, A6+M1-soft+M4) but the Results text does not explicitly discuss the independent contribution of M1-soft vs M4. The reader needs to know which module is doing what work.

**B-EIC-C04** — Scale recovery validation

> The Methods mention pot diameter as the scale reference but provide no validation of scale recovery accuracy. A systematic error in scale would propagate to all phenotype measurements.

---

### Reviewer 1 (B-R1) — Dr. Maria Chen, Plant Phenotyping

**B-R1-C01** — Botanical nomenclature (Latin binomials)

> The paper uses 10 Chinese species labels. For an international journal, verified botanical nomenclature is essential. At minimum, the species should be identified with accepted Latin binomials. Common Chinese names are ambiguous: "KongQueZhuYu" could refer to multiple Calathea species, and "XianKeLai" likely corresponds to a Cyclamen or related genus. Without taxonomic resolution, the "cross-species" claim is scientifically imprecise. I strongly recommend providing a table with: Chinese name, tentative Latin binomial, family, and growth form (rosette/erect/climbing).

**B-R1-C02** — Phenotype measurement protocol

> The Methods mention "manually identified landmark pairs on the mesh" for leaf length/width. This is insufficiently specified. How many operators performed the manual measurement? What was the intra-operator repeatability? For the virtual measurement, were landmarks placed on the mesh by a human operator or automatically detected? If manual landmark placement was used for virtual measurement, then the reported R² values conflate reconstruction error with landmark placement error. This needs explicit discussion.

**B-R1-C03** — Trait selection (leaf area, stem diameter excluded)

> The four traits (height, canopy width, leaf length, leaf width) are reasonable but are all linear dimensions. Why were leaf area and stem diameter not included, given that these are standard traits in 3D phenotyping literature [Paulus 2019]? If the mesh does not support reliable area computation (due to holes or incomplete surfaces), this should be stated as a limitation.

**B-R1-C04** — Growth stage not reported

> The paper does not report plant growth stage or developmental age. For cross-species comparison, developmental stage can be as important as species identity—a seedling of one species may be architecturally more similar to a seedling of another species than to a mature plant of its own species.

---

### Reviewer 2 (B-R2) — Dr. Thomas Müller, 3D Vision & Gaussian Splatting

**B-R2-C01** — Technical novelty assessment

> The individual components (foreground track initialization, per-pixel mask on RGB loss, alpha mask loss, opacity regularization, soft view weighting, Gaussian pruning) are not individually novel—each has precedents in the 3DGS/2DGS literature or in general computer vision. The contribution lies in (a) their specific combination for the plant phenotyping application, (b) the systematic ablation that identifies which components are load-bearing, and (c) the negative evidence for hard view filtering. This is a legitimate engineering contribution, but the paper should be more explicit about which components are novel combinations vs. novel mechanisms.

**B-R2-C02** — Missing external baselines

> The paper compares variants of its own method (A0-A6, M1 variants, M4) but does not compare against any external method. For a CompAg submission, I would expect at least a comparison against: (1) A traditional MVS pipeline (e.g., COLMAP + OpenMVS) as a non-learning baseline; (2) A NeRF-based method (e.g., Instant-NGP or Zip-NeRF) with the same foreground masks; (3) Standard 3DGS with the same masks. Without external baselines, it's impossible to assess whether 2DGS was the right choice of base representation.

**B-R2-C03** — Mesh evaluation not connected to phenotype

> The mesh comparison (Standard TSDF vs. Smaller truncation vs. Post-boundary) reports structural metrics but never evaluates which variant produces better phenotype measurements. This is a missed opportunity—even a simple leaf width comparison across the three mesh variants on 2-3 samples would connect the mesh analysis to the phenotype validation and strengthen the downstream utility argument.

**B-R2-C04** — Hyperparameter values not reported

> The method description uses clean mathematical notation but some terms are under-defined. For example, in the M4 scoring function (Section 3.5), the weights α, β, γ, δ, η are listed but their values are never reported. Similarly, the pruning threshold τ_g is mentioned but not specified.

---

### Reviewer 3 (B-R3) — Dr. Amara Patel, Agricultural AI

**B-R3-C01** — Practical impact / absolute metrics for compactness

> The 18.03% Gaussian reduction is presented as a practical benefit, but the absolute numbers matter more for deployment. At ~1M Gaussians per plant (Ours-core) vs. ~800K (Ours-full), what is the actual GPU memory saving in MB? What is the rendering FPS difference? Reporting these application-level metrics would make the compactness contribution more concrete for practitioners.

**B-R3-C02** — Reproducibility (hardware, software, hyperparams, code)

> The paper lacks several details needed for reproduction: (1) What smartphone model was used? What resolution? What frame rate? (2) What was the turntable rotation speed? How many degrees per frame? (3) What COLMAP version and parameters were used? (4) What SAM3 checkpoint/version was used? (5) What are the training hyperparameters (learning rate, iterations, λ values)? (6) Code availability is not mentioned. For a method with this many components, a code repository (even if cleaned post-hoc) is essential for reproducibility.

**B-R3-C03** — FSAM3 prompt design rationale

> The five prompts (P1-P5) are listed, but the rationale for choosing these specific prompts is not explained. Were they derived from prior work, pilot experiments, or domain expertise? A short justification would help readers design prompts for their own species.

**B-R3-C04** — Cost analysis / per-plant processing time

> For agricultural deployment, cost matters. What is the approximate per-plant processing time (wall clock) from raw images to phenotype measurements? What hardware was used? Even a rough estimate (e.g., "approximately 45 minutes per plant on a single NVIDIA RTX 3090") would help readers assess deployment feasibility.

---

### Devil's Advocate (B-DA)

**B-DA-C01** — E7 CRITICAL: Missing post-hoc pruning comparison

> The paper's central claim is that "foreground-object reconstruction" is fundamentally different from and superior to "full-scene reconstruction + post-hoc mask pruning." The A0-A6 ablation convincingly shows that A5/A6 produce lower leakage than A0-A4. However, the critical missing comparison is A0 + post-hoc pruning vs. A6. The paper mentions this as variant "E7" in the Discussion (Section 5.1) but never reports E7 results quantitatively. The reasoning that "pruning cannot recover diverted capacity" is a mechanistic hypothesis, not an empirical finding. To test it, one would need to: (a) train A0, (b) prune all Gaussians whose centers project outside the multi-view mask, (c) optionally fine-tune the pruned model briefly with foreground-only loss, and (d) report the same metrics as Table 2. If the pruned+fine-tuned A0 achieves metrics comparable to A6, the "capacity diversion" argument collapses. If it does not, the paper gains its strongest empirical argument. Either way, this comparison belongs in the main results, not as a brief mention in Discussion.

**B-DA-C02** — Ablation replication (single sample)

> A0-A6 ablation uses one sample. The ablation's internal validity is strong, but without replication on at least one other sample, we cannot rule out sample-specific effects (e.g., the particular background of KongQueZhuYu may be especially easy to separate).

**B-DA-C03** — FSAM3 mask evaluation (no quantitative metrics)

> FSAM3 is a named method with a dedicated Results subsection, but no quantitative mask metrics are reported. "Generating masks for 20 samples" is a feasibility claim, not a quality claim.

**B-DA-C04** — FFT threshold (per-sample quartile vs fixed)

> The FFT threshold uses the "first quartile of per-sequence scores." This means the threshold is sample-dependent: a high-quality sequence loses 25% of its frames to an internal threshold, while a blurry sequence may retain frames that a fixed threshold would exclude. Why not a fixed threshold calibrated on a held-out set?

**B-DA-C05** — Naming inconsistency

> The term "F2DMAS" appears in the title and abstract but is almost never used in the main text, which uses "Plant-aware 2DGS," "Ours-core," "Ours-full," and "FSAM3." Inconsistent naming weakens the brand.

---

## Comment ID Summary

| Stream | Reviewer | Comment IDs | Count |
|--------|----------|-------------|-------|
| A | Reviewer 1 | A1-C01 -- A1-C05 | 5 |
| A | Reviewer 2 | A2-C01 -- A2-C06 | 6 |
| B | EIC | B-EIC-C01 -- B-EIC-C04 | 4 |
| B | Reviewer 1 | B-R1-C01 -- B-R1-C04 | 4 |
| B | Reviewer 2 | B-R2-C01 -- B-R2-C04 | 4 |
| B | Reviewer 3 | B-R3-C01 -- B-R3-C04 | 4 |
| B | Devil's Advocate | B-DA-C01 -- B-DA-C05 | 5 |
| **Total** | | | **32** |
