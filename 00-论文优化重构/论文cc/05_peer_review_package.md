# Stage 3: Peer Review Package — F2DMAS Manuscript

**Journal:** Computers and Electronics in Agriculture (CompAg)
**Review Mode:** Full (5-reviewer panel)
**Date:** 2026-05-24

---

## Phase 0: Reviewer Configuration Card

| Role | Identity | Expertise | Review Focus |
|------|----------|-----------|-------------|
| **EIC** | Prof. David Rousseau, Senior Editor, CompAg | Plant phenotyping, computer vision, precision agriculture | Overall quality, journal fit, significance, editorial decision |
| **R1** | Dr. Maria Chen, Plant Physiologist, Wageningen University | 3D plant phenotyping, trait measurement, experimental design | Biological validity, cross-species claims, phenotype metrics |
| **R2** | Dr. Thomas Müller, Computer Vision Researcher, ETH Zurich | 3D reconstruction, Gaussian Splatting, NeRF, mesh extraction | Technical novelty, method rigor, comparison to SOTA |
| **R3** | Dr. Amara Patel, Agricultural AI, UC Davis | ML for agriculture, automated phenotyping, deployment | Practical utility, modularity, reproducibility, agricultural relevance |
| **DA** | Devil's Advocate | Logical fallacy detection, counter-argument construction | Core argument stress test, overclaim detection |

---

## Phase 1: Independent Reviews

---

### EIC Review — Prof. David Rousseau

**Overall Assessment:**

This manuscript presents F2DMAS, an integrated pipeline for automated cross-species plant 3D reconstruction and phenotype measurement. The paper addresses a genuine and well-motivated problem in agricultural automation: how to obtain plant-only 3D representations suitable for trait measurement from multi-view images acquired in semi-controlled settings. The task reformulation—from full-scene to foreground-object 2D Gaussian Splatting—is conceptually clean and practically relevant.

**Strengths:**
1. The A0-A6 ablation is systematic and convincing. The demonstration that alpha mask and background opacity regularization alone cannot prevent background learning (A2-A4), while foreground RGB supervision (A5) produces a decisive transition, is a clean and well-controlled experimental result.
2. The M1-hard negative evidence is valuable. Too few papers report negative results with the same rigor as positive results. Showing that hard view filtering catastrophically degrades reconstruction (PSNR_fg 25→12 dB) while soft weighting preserves quality is a strong methodological contribution.
3. The claim calibration is unusually disciplined for a methods paper. The clear distinction between "mesh structural evidence" and "phenotype accuracy improvement," and the explicit list of what cannot be claimed, builds reviewer trust.
4. The 51-reference bibliography covers the relevant landscape from classical SfM through NeRF, 3DGS, 2DGS, SAM, and plant-specific applications.

**Weaknesses:**
1. **Cross-species evidence gap**: The paper claims "cross-species" in the title and throughout, but the reconstruction validation uses only 3 samples. While 20 samples are available for mask generation, the reconstruction claim—which is the paper's core contribution—rests on 3 samples. This is a significant gap between the framing and the evidence.
2. **FSAM3 segmentation evaluation**: The paper explicitly declines to benchmark FSAM3 against existing segmentation methods. While I accept the rationale (no pixel-level GT), a simple qualitative comparison against ExG/Otsu on a few representative frames would substantially strengthen the mask generation contribution.
3. **Missing ablation: M1-soft alone vs M4 alone**: Table 5 reports the four-way combination (A6, A6+M1-soft, A6+M4, A6+M1-soft+M4) but the Results text does not explicitly discuss the independent contribution of M1-soft vs M4. The reader needs to know which module is doing what work.
4. **Scale recovery details**: The Methods mention pot diameter as the scale reference but provide no validation of scale recovery accuracy. A systematic error in scale would propagate to all phenotype measurements.

**Editorial Recommendation:** MAJOR REVISION

**Key Revision Priorities:**
1. Either (a) expand reconstruction validation to more samples (ideally 5+), or (b) reframe the cross-species claim to "representative architecture diversity" with explicit limitation
2. Add qualitative mask comparison (FSAM3 vs ExG/Otsu) on 3-5 representative frames
3. Add independent contribution discussion for M1-soft vs M4
4. Report scale recovery validation

---

### Reviewer 1 (R1) — Dr. Maria Chen, Plant Phenotyping

**Review Focus:** Biological validity, cross-species claims, phenotype measurement

**Summary:**

This paper tackles a practical bottleneck in 3D plant phenotyping: obtaining clean plant-only models without manual segmentation. The core idea—changing what the model optimizes rather than post-processing the output—is biologically sensible because leaves, stems, and background structures are interleaved in the image, and separating them after the fact is fundamentally harder than not learning the background in the first place.

**Detailed Comments:**

**Biological relevance (Major):** The paper uses 10 Chinese species labels. For an international journal, verified botanical nomenclature is essential. At minimum, the species should be identified with accepted Latin binomials. Common Chinese names are ambiguous: "KongQueZhuYu" could refer to multiple Calathea species, and "XianKeLai" likely corresponds to a Cyclamen or related genus. Without taxonomic resolution, the "cross-species" claim is scientifically imprecise. I strongly recommend providing a table with: Chinese name, tentative Latin binomial, family, and growth form (rosette/erect/climbing).

**Phenotype measurement protocol (Major):** The Methods mention "manually identified landmark pairs on the mesh" for leaf length/width. This is insufficiently specified. How many operators performed the manual measurement? What was the intra-operator repeatability? For the virtual measurement, were landmarks placed on the mesh by a human operator or automatically detected? If manual landmark placement was used for virtual measurement, then the reported R² values conflate reconstruction error with landmark placement error. This needs explicit discussion.

**Trait selection (Minor):** The four traits (height, canopy width, leaf length, leaf width) are reasonable but are all linear dimensions. Why were leaf area and stem diameter not included, given that these are standard traits in 3D phenotyping literature [Paulus 2019]? If the mesh does not support reliable area computation (due to holes or incomplete surfaces), this should be stated as a limitation.

**Growth stage (Minor):** The paper does not report plant growth stage or developmental age. For cross-species comparison, developmental stage can be as important as species identity—a seedling of one species may be architecturally more similar to a seedling of another species than to a mature plant of its own species.

**Recommendation:** MAJOR REVISION

**Key Requests:**
1. Add botanical nomenclature table (Latin binomials, family, growth form)
2. Specify manual and virtual measurement protocols in detail (operators, repeatability, landmark placement)
3. Discuss why leaf area and stem diameter were excluded
4. Report growth stage information

---

### Reviewer 2 (R2) — Dr. Thomas Müller, 3D Vision & Gaussian Splatting

**Review Focus:** Technical novelty, method rigor, SOTA comparison

**Summary:**

The paper proposes several algorithm-level modifications to the standard 2DGS pipeline for the specific application of plant foreground reconstruction. The core technical insight—that restricting RGB loss computation to foreground pixels changes the optimization dynamics in a way that post-hoc pruning cannot replicate—is well-motivated and convincingly demonstrated through the A0-A6 ablation. The negative M1-hard result is, in my view, the most interesting finding: it challenges the conventional wisdom that "removing bad data improves models" and provides a clear counterexample specific to multi-view geometry.

**Detailed Comments:**

**Technical novelty assessment (Major):** The individual components (foreground track initialization, per-pixel mask on RGB loss, alpha mask loss, opacity regularization, soft view weighting, Gaussian pruning) are not individually novel—each has precedents in the 3DGS/2DGS literature or in general computer vision. The contribution lies in (a) their specific combination for the plant phenotyping application, (b) the systematic ablation that identifies which components are load-bearing, and (c) the negative evidence for hard view filtering. This is a legitimate engineering contribution, but the paper should be more explicit about which components are novel combinations vs. novel mechanisms. Specifically:

- Foreground track initialization: Has this been used in any prior 3DGS/2DGS work? If not, it may be individually novel.
- Foreground RGB loss (mask-gated L1): Similar to masked losses in image inpainting literature. Worth citing.
- Alpha mask loss: Standard in NeRF/3DGS with mask supervision.
- M1-soft: The closest analogue is uncertainty-weighted losses in multi-task learning [Kendall et al., CVPR 2018]. Worth discussing.

**Comparison to SOTA (Major):** The paper compares variants of its own method (A0-A6, M1 variants, M4) but does not compare against any external method. For a CompAg submission, I would expect at least a comparison against:
1. A traditional MVS pipeline (e.g., COLMAP + OpenMVS) as a non-learning baseline
2. A NeRF-based method (e.g., Instant-NGP or Zip-NeRF) with the same foreground masks
3. Standard 3DGS with the same masks

Without external baselines, it's impossible to assess whether 2DGS was the right choice of base representation. The paper's argument is entirely internal (which variant of our method works best), but never external (does our method work better than existing alternatives).

**Mesh evaluation (Major):** The mesh comparison (Standard TSDF vs. Smaller truncation vs. Post-boundary) reports structural metrics but never evaluates which variant produces better phenotype measurements. This is a missed opportunity—even a simple leaf width comparison across the three mesh variants on 2-3 samples would connect the mesh analysis to the phenotype validation and strengthen the downstream utility argument.

**Mathematical notation (Minor):** The method description uses clean mathematical notation but some terms are under-defined. For example, in the M4 scoring function (Section 3.5), the weights α, β, γ, δ, η are listed but their values are never reported. Similarly, the pruning threshold τ_g is mentioned but not specified.

**Recommendation:** MAJOR REVISION

**Key Requests:**
1. Clarify which components are individually novel vs. novel in combination
2. Add at least one external baseline comparison (MVS or NeRF or 3DGS)
3. Connect mesh variant evaluation to phenotype measurement
4. Report all hyperparameter values (α-η weights, τ_g, τ_track)

---

### Reviewer 3 (R3) — Dr. Amara Patel, Agricultural AI

**Review Focus:** Practical utility, modularity, deployment feasibility

**Summary:**

This paper presents a modular pipeline that is practically motivated and clearly structured. As someone who works on deploying computer vision systems in agricultural settings, I appreciate several design choices: (1) the FFT-based quality screening is computationally cheap and doesn't require GPU inference, (2) the modular architecture allows individual components to be adopted independently, and (3) the use of smartphone-based acquisition lowers the barrier to adoption. The paper's emphasis on "what the method cannot do" (e.g., not claiming phenotype accuracy improvement from M5) is refreshing and builds credibility.

**Detailed Comments:**

**Practical impact (Minor):** The 18.03% Gaussian reduction is presented as a practical benefit, but the absolute numbers matter more for deployment. At ~1M Gaussians per plant (Ours-core) vs. ~800K (Ours-full), what is the actual GPU memory saving in MB? What is the rendering FPS difference? Reporting these application-level metrics would make the compactness contribution more concrete for practitioners.

**Reproducibility (Major):** The paper lacks several details needed for reproduction:
1. What smartphone model was used? What resolution? What frame rate?
2. What was the turntable rotation speed? How many degrees per frame?
3. What COLMAP version and parameters were used?
4. What SAM3 checkpoint/version was used?
5. What are the training hyperparameters (learning rate, iterations, λ values)?
6. Code availability is not mentioned. For a method with this many components, a code repository (even if cleaned post-hoc) is essential for reproducibility.

**FSAM3 prompt design (Minor):** The five prompts (P1-P5) are listed, but the rationale for choosing these specific prompts is not explained. Were they derived from prior work, pilot experiments, or domain expertise? A short justification would help readers design prompts for their own species.

**Cost analysis (Minor):** For agricultural deployment, cost matters. What is the approximate per-plant processing time (wall clock) from raw images to phenotype measurements? What hardware was used? Even a rough estimate (e.g., "approximately 45 minutes per plant on a single NVIDIA RTX 3090") would help readers assess deployment feasibility.

**Recommendation:** MINOR REVISION

**Key Requests:**
1. Report absolute memory/rendering metrics for compactness
2. Add detailed reproducibility information (hardware, software versions, hyperparameters)
3. Estimate per-plant processing time and hardware requirements
4. Justify prompt design rationale

---

### Devil's Advocate Review

**Strongest Counter-Argument (300 words):**

The paper's central claim is that "foreground-object reconstruction" is fundamentally different from and superior to "full-scene reconstruction + post-hoc mask pruning." The A0-A6 ablation convincingly shows that A5/A6 produce lower leakage than A0-A4. However, the critical missing comparison is A0 + post-hoc pruning vs. A6. The paper mentions this as variant "E7" in the Discussion (Section 5.1) but never reports E7 results quantitatively. The reasoning that "pruning cannot recover diverted capacity" is a mechanistic hypothesis, not an empirical finding. To test it, one would need to: (a) train A0, (b) prune all Gaussians whose centers project outside the multi-view mask, (c) optionally fine-tune the pruned model briefly with foreground-only loss, and (d) report the same metrics as Table 2. If the pruned+fine-tuned A0 achieves metrics comparable to A6, the "capacity diversion" argument collapses. If it does not, the paper gains its strongest empirical argument. Either way, this comparison belongs in the main results, not as a brief mention in Discussion.

**Issue List:**

| Severity | Dimension | Location | Issue |
|----------|-----------|----------|-------|
| CRITICAL | Evidence | §5.1, missing from §4 | E7 (A0 + post-hoc pruning) is argued conceptually but never tested empirically. This is the most direct test of the core claim. |
| MAJOR | Rigor | §4.2 | A0-A6 ablation uses one sample. The ablation's internal validity is strong, but without replication on at least one other sample, we cannot rule out sample-specific effects (e.g., the particular background of KongQueZhuYu may be especially easy to separate). |
| MAJOR | Evidence | §4.1 | FSAM3 is a named method with a dedicated Results subsection, but no quantitative mask metrics are reported. "Generating masks for 20 samples" is a feasibility claim, not a quality claim. |
| MAJOR | Rigor | §3.2.1 | The FFT threshold uses the "first quartile of per-sequence scores." This means the threshold is sample-dependent: a high-quality sequence loses 25% of its frames to an internal threshold, while a blurry sequence may retain frames that a fixed threshold would exclude. Why not a fixed threshold calibrated on a held-out set? |
| MINOR | Argument | Throughout | The term "F2DMAS" appears in the title and abstract but is almost never used in the main text, which uses "Plant-aware 2DGS," "Ours-core," "Ours-full," and "FSAM3." Inconsistent naming weakens the brand. |

**Ignored Alternative Explanations:**
1. The improvement from A4 to A5 could be partially explained by the effective increase in per-pixel supervision weight (since L_rgb_fg is normalized by |Ω_fg| rather than |Ω|), rather than by the spatial restriction per se. A variant that keeps full-image RGB loss but up-weights foreground pixels would test this.
2. The M1-soft improvement could be explained by the implicit regularization from reduced learning rate on low-quality views, rather than by the quality weighting mechanism specifically. A uniform weight reduction (same weight for all views) control would test this.

**Missing Stakeholder Perspectives:**
- The paper is written for computer vision researchers who work on agriculture. It does not address the plant biologist or breeder who would be the end user. What would a breeder need to trust these virtual measurements? What confidence intervals are acceptable for breeding decisions?

**"So What?" Test:** If the F2DMAS pipeline were available tomorrow as a software tool, what decision would a breeder or grower make differently? The paper does not connect the technical metrics to any actionable agricultural outcome.

**DA Verdict:** The core claim is plausible and internally consistent, but the most direct empirical test (E7) is missing, and the single-sample ablation limits generalizability. The paper's strongest evidence (A0-A6, M1-hard) is on one sample. **Overall: The core argument is well-structured but needs the E7 comparison and at least partial ablation replication to be fully convincing.**

---

## Phase 2: Editorial Synthesis

### Cross-Reviewer Consensus Matrix

| Issue | EIC | R1 | R2 | R3 | Consensus |
|-------|-----|----|----|-----|-----------|
| 3-sample validation insufficient for "cross-species" | MAJOR | MAJOR | — | — | CONSENSUS (MAJOR) |
| Missing external baselines (MVS, NeRF, 3DGS) | — | — | MAJOR | — | SINGLE (MAJOR) |
| Missing E7 (A0+prune) comparison | — | — | — | CRITICAL (DA) | SINGLE (CRITICAL) |
| FSAM3 lacks quantitative mask evaluation | MAJOR | — | — | — | SINGLE (MAJOR) |
| Botanical nomenclature needed | — | MAJOR | — | — | SINGLE (MAJOR) |
| Insufficient reproducibility details | — | — | — | MAJOR | SINGLE (MAJOR) |
| M1-soft vs M4 independent contribution unclear | MAJOR | — | — | — | SINGLE (MAJOR) |
| Hyperparameter values not reported | — | — | MAJOR | — | SINGLE (MAJOR) |
| Scale recovery validation missing | MAJOR | — | — | — | SINGLE (MAJOR) |
| Measurement protocol underspecified | — | MAJOR | — | — | SINGLE (MAJOR) |

### Editorial Decision: MAJOR REVISION

The manuscript presents a well-motivated, carefully ablated contribution to automated plant 3D reconstruction. The reviewers are in broad agreement on the paper's strengths: the systematic A0-A6 ablation, the M1-hard negative evidence, the claim calibration discipline, and the practical motivation are all strong. However, several substantive issues must be addressed before the paper is ready for publication:

1. **Consensus issue — Cross-species evidence:** Both EIC and R1 flag the gap between the "cross-species" framing and the 3-sample reconstruction validation. The authors should either expand the validation or reframe the claim.

2. **Devil's Advocate CRITICAL — Missing E7 comparison:** The most direct empirical test of the core claim (training full-scene then pruning vs. foreground-object training) is argued conceptually but not shown empirically. This MUST be addressed.

3. **Missing baselines and mask evaluation:** R2's request for external baselines and the need for at least qualitative FSAM3 evaluation are well-justified.

4. **Reproducibility:** R3's request for detailed methods information is standard for CompAg and should be straightforward to address.

### Scores

| Dimension | EIC | R1 | R2 | R3 | Average |
|-----------|-----|----|----|-----|---------|
| Originality (20%) | 7/10 | 7/10 | 6/10 | 7/10 | **6.8/10** |
| Methodological Rigor (25%) | 7/10 | 6/10 | 6/10 | 6/10 | **6.3/10** |
| Evidence Sufficiency (25%) | 5/10 | 5/10 | 5/10 | 6/10 | **5.3/10** |
| Argument Coherence (15%) | 8/10 | 7/10 | 8/10 | 8/10 | **7.8/10** |
| Writing Quality (15%) | 7/10 | 7/10 | 7/10 | 8/10 | **7.3/10** |
| **Overall** | **6.7/10** | **6.2/10** | **6.2/10** | **6.8/10** | **6.5/10** |

---

## Revision Roadmap (Prioritized)

### P0 — Must Address (blocking for acceptance)

| # | Issue | Source | Suggested Action |
|---|-------|--------|-----------------|
| P0-1 | Missing E7 empirical comparison | DA (CRITICAL) | Add E7 results (A0 train → prune → fine-tune with fg loss) as a row in Table 2 or as a separate comparison table. This is the single most important addition. |
| P0-2 | Cross-species evidence gap | EIC, R1 | Option A: Expand A6 validation to 5+ samples. Option B: Reframe as "architectural diversity" with explicit limitation statement. Recommend Option A for CompAg. |
| P0-3 | Missing external baselines | R2 | Add at least 1 external comparison (COLMAP+MVS or 3DGS with same masks). Run on KongQueZhuYu only if resources are limited. |
| P0-4 | FSAM3 mask evaluation | EIC, DA | Add qualitative comparison figure (FSAM3 vs ExG vs Otsu on 3-5 frames) + component count reduction table. |

### P1 — Strongly Recommended

| # | Issue | Source | Suggested Action |
|---|-------|--------|-----------------|
| P1-1 | Reproducibility details | R3 | Add appendix or supplemental table with: smartphone model, resolution, COLMAP version/params, SAM3 checkpoint, 2DGS training hyperparams (lr, iterations, all λ values, τ values), hardware specs. |
| P1-2 | Botanical nomenclature | R1 | Add table mapping Chinese names → Latin binomials → family → growth form. Consult a botanist or taxonomic database. |
| P1-3 | M1-soft vs M4 contribution | EIC | Add discussion paragraph explicitly comparing independent contributions. The data exists in Table 5—just needs interpretation. |
| P1-4 | Hyperparameter values | R2 | Report α, β, γ, δ, η in M4 scoring function; τ_track for foreground track init; τ_g for M4 pruning. |
| P1-5 | Scale recovery validation | EIC | Report pot diameter measurement error and its propagation to trait measurements. Add 1-2 sentences to Methods. |

### P2 — Would Improve

| # | Issue | Source | Suggested Action |
|---|-------|--------|-----------------|
| P2-1 | Measurement protocol details | R1 | Specify # operators, intra-operator repeatability, landmark placement method for virtual measurements. |
| P2-2 | Ablation replication | DA | Replicate key ablation rows (A0, A5, A6) on at least 1 additional sample. |
| P2-3 | Per-plant processing time | R3 | Report approximate wall clock time from raw images to phenotype report. |
| P2-4 | FFT threshold justification | DA | Discuss choice of per-sample quartile vs. fixed threshold. Report sensitivity to threshold choice. |
| P2-5 | Naming consistency | DA | Use "F2DMAS" consistently or rename. If F2DMAS = the full pipeline, use it throughout instead of "Plant-aware 2DGS." |
| P2-6 | Prompt design rationale | R3 | Add 1-2 sentences explaining how the 5 prompts were selected. |

---

*Review package completed 2026-05-24. 5 reviewers, 4 MAJOR REVISION decisions (with DA CRITICAL flag), 1 MINOR REVISION.*
