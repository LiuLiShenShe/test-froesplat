# Phase 19A — Reviewer Comment Resolution Status

**Generated:** 2026-09-14 | **Total comments:** 32 | **Resolution evidence sources:** Phase 12 (GT QA), Phase 14/14.1 (A6/A7 factorial), Phase 17B (failure attribution), Phase 17C (ranking rescue), Phase 18 (evidence registry)

---

## Summary by Stream

| Stream | Reviewer | RESOLVED | PARTIALLY_RESOLVED | NOT_RESOLVED | NO_LONGER_RELEVANT |
|--------|----------|----------|---------------------|--------------|---------------------|
| A | Reviewer 1 | 4 | 0 | 1 | 0 |
| A | Reviewer 2 | 6 | 0 | 0 | 0 |
| B | EIC | 3 | 1 | 0 | 0 |
| B | Reviewer 1 (Plant Phenotyping) | 3 | 0 | 1 | 0 |
| B | Reviewer 2 (3D Vision) | 3 | 0 | 1 | 0 |
| B | Reviewer 3 (Ag AI) | 4 | 0 | 0 | 0 |
| B | Devil's Advocate | 2 | 1 | 2 | 0 |
| **Total** | | **25** | **2** | **5** | **0** |

---

## Notable Resolution Decisions

### RESOLVED — R1.2 Table 6 Ablation (A1-C02)

The most complex Stream A comment. R1.2 asked two questions: (1) whether F2DMAS still leads when standard 2DGS uses the same blurred input, and (2) whether the "Plant Seg." improvement comes from SAM3 alone or from FSAM3 components. Phase 17B proved that SAM3 generates high-quality candidates (oracle F1 = 0.9867--0.9882 on all three failure frames), ruling out candidate generation as the bottleneck. Phase 17C then showed that the ranking signal (specifically q_contrast) was the failure cause, not the segmentation backbone. The new paper must revise Table 6 as a fair-input ablation with five controlled settings (Base, Base+FFT, Base+SAM3, Base+FSAM3, F2DMAS) and add a component-level segmentation ablation table. The revision directly resolves both questions: FSAM3 improves over SAM3-only (temporal propagation + morphological closing), and F2DMAS leads even without FFT filtering.

### RESOLVED — P0-4 / FSAM3 Mask Evaluation (B-EIC-C02, B-DA-C03)

Phase 12 now provides quantitative mask metrics on 21 GT frames from 5 species: F1 = 0.9835 (median 0.9839), IoU = 0.9676, Precision = 0.9943, Recall = 0.9731. Cube overlap is zero. This directly answers the feasibility-versus-quality concern. The new paper must report these metrics in Section 4.1 and add a qualitative comparison figure (FSAM3 vs ExG/Otsu on 3--5 frames). The pixel-level GT limitation for ExG/Otsu benchmarking is acknowledged.

### PARTIALLY_RESOLVED — E7 Post-Hoc Pruning (B-DA-C01, CRITICAL)

The DA's strongest argument: A0 + post-hoc pruning (E7) was argued conceptually but never tested empirically. The simulated review response added E7 to Section 4.2 with quantitative results (PSNR_fg = 21.34, outside = 0.31, leakage = 0.28), which fails the foreground-only threshold and supports the capacity-diversion hypothesis. However, E7 was tested on a single sample (KongQueZhuYu) only. More importantly, Phase 17B and Phase 17C provide related but NOT identical evidence: the ranking-failure diagnosis (oracle F1 = 0.987 but wrong instance selected) and the ranking correction (R1 removes q_contrast, rescues 3/3 failures) address a structurally similar problem -- small high-contrast fragments outranking correct candidates -- but through a different mechanism (ranking vs. pruning). The new paper must include E7 results in Table 2 while noting the single-sample limitation, and must not conflate ranking-failure rescue with post-hoc pruning equivalence.

### PARTIALLY_RESOLVED — Cross-Species Evidence (B-EIC-C01)

The EIC flagged a gap between the "cross-species" title and the 3-sample reconstruction validation. Phase 12 provides segmentation evidence on 21 GT frames from 5 species, and the phenotype validation covers 21 plants from 10 species. However, reconstruction validation remains limited to 3 architecturally representative samples. The new paper must reframe the cross-species claim as "architectural diversity" with an explicit limitation statement in Section 6, while retaining cross-species applicability at the mask and phenotype levels. The title qualification and Contribution 4 language must be revised accordingly.

### NOT_RESOLVED — R1.3 Leaf Width Error Source (A1-C03)

R1.3 asked whether leaf-width error comes from segmentation, 2DGS surface dilation, or TSDF meshing, and whether point-cloud measurement could help. Phase 11-18 work is entirely within the segmentation pipeline scope; leaf-width measurement is a downstream 2DGS/phenotyping issue that cannot be resolved by segmentation-level evidence. The new paper must expand the Limitations section to discuss error sources and note point-cloud-based measurement as a future direction, but cannot claim the issue is resolved.

### NOT_RESOLVED — External Baselines (B-R2-C02)

R2 requested comparison against at least one external method (COLMAP+MVS, NeRF, or 3DGS with same masks). No external baselines have been added. The new paper must include a dedicated limitation paragraph in Section 6 explaining the rationale for within-method ablation (internal mechanism questions are best answered by controlled component variation) and commit to external baselines in a follow-up study. The paper must NOT claim "outperforms SOTA" or "best reconstruction" -- these claims remain unsupported.

### NOT_RESOLVED — Botanical Nomenclature (B-R1-C01)

R1 requested Latin binomials, family, and growth form for all 10 species. Supplementary Table S1 was created as a placeholder but definitive taxonomic identification requires botanist consultation and may be non-trivial (e.g., "KongQueZhuYu" maps to multiple Calathea species). This is a non-blocking limitation that can be addressed at proof stage.

### NOT_RESOLVED — Ablation Replication (B-DA-C02)

The DA noted that A0-A6 ablation runs on a single sample (KongQueZhuYu). No replication on an additional sample has been performed. The new paper must acknowledge this as a limitation and state that findings are development-set results, not generalizable ablation conclusions.

### NOT_RESOLVED — FFT Threshold (B-DA-C04)

The DA questioned the per-sample quartile FFT threshold as sample-dependent. The current approach was retained with the rationale that it adapts to per-sequence quality variation without requiring a manually labeled quality dataset. A fixed threshold calibrated on a held-out set was not implemented. The new paper must note this as a limitation and commit to threshold sensitivity analysis in follow-up work.

---

## Resolution Evidence Mapping

| Comment | Primary Phase Evidence | Key Finding |
|---------|----------------------|-------------|
| A1-C02 (Table 6 ablation) | Phase 17B + Phase 17C | Oracle F1=0.987 proves candidate generation succeeds; R1 ranking correction eliminates failure mode |
| B-EIC-C01 (Cross-species) | Phase 12 + Phase 14.1 | Segmentation evidence on 21 GT frames / 5 species; reconstruction remains 3 samples |
| B-EIC-C02 (FSAM3 metrics) | Phase 12 | F1=0.9835 IoU=0.9676 on 21 GT frames |
| B-EIC-C03 (M1-soft vs M4) | Phase 18 (re-analysis) | Data in Table 5; interpretation paragraph added |
| B-DA-C01 (E7 CRITICAL) | Response + Phase 17B/17C | E7 results added; ranking failure evidence is related but not identical |
| B-DA-C03 (Mask metrics) | Phase 12 | Same as B-EIC-C02 |
| B-R2-C02 (External baselines) | None | No external baselines added; acknowledged as limitation |
| A1-C03 (Leaf width) | None | Downstream issue; not addressed by Phase 11-18 |
| B-R1-C01 (Nomenclature) | Supplementary Table S1 | Placeholder only; pending botanist consultation |
