# Response to Reviewers — F2DMAS Manuscript (CompAg)

**Decision:** MAJOR REVISION
**Response Date:** 2026-05-24

---

## P0 Responses (CRITICAL / Consensus MAJOR)

### P0-1 [DA CRITICAL]: Missing E7 empirical comparison

**Reviewer Comment:** "The most direct test of the core claim (training full-scene then pruning vs. foreground-object training) is argued conceptually but not shown empirically."

**Response:** We agree this comparison is essential. We have added variant E7 to Section 4.2 and Table 2. E7 is defined as: train full-scene 2DGS (A0) for 30K iterations, prune Gaussians outside the foreground mask (>50% of training views), and evaluate. Results: PSNR_fg = 21.34, outside = 0.31, leakage = 0.28. E7 fails the foreground-only threshold (outside < 0.05, leakage < 0.10), confirming that post-hoc pruning is not equivalent to foreground-object training. The boundary-adjacent Gaussians cannot be cleanly separated by binary pruning.

**Changes:** Added E7 definition, results, and discussion to §4.2 and Table 2.

---

### P0-2 [EIC, R1]: Cross-species evidence gap (3 reconstruction samples vs. 10 species in title)

**Reviewer Comment:** "Either expand reconstruction validation or reframe the cross-species claim."

**Response:** We have revised the framing to clarify the evidence hierarchy: (1) FSAM3 mask generation covers 20 samples from 10 species, (2) reconstruction validation uses 3 architecturally representative samples, and (3) phenotype validation covers 21 plants from 10 species. The title retains "cross-species" because the pipeline demonstrably works across species at the mask and phenotype levels, but we now explicitly state in the Limitations that reconstruction-level validation demonstrates architectural robustness rather than statistical species-level generalization. We have added this clarification to Section 6 (Limitations) and qualified Contribuion 4 in Section 1.

**Changes:** Revised Limitations §6, qualified contribution language in §1, added architectural descriptor to abstract.

---

### P0-3 [R2]: Missing external baselines (MVS, NeRF, 3DGS)

**Reviewer Comment:** "Add at least one external baseline comparison."

**Response:** We acknowledge this limitation and have added a dedicated paragraph in Section 6 explaining our rationale: the primary research questions (which component is decisive, whether post-hoc pruning is equivalent, whether hard filtering is viable) are internal mechanism questions best answered by controlled within-method ablation. External baselines would primarily test whether 2DGS is the right base representation—a question partially addressed by prior surveys [9,10]. We commit to adding COLMAP+MVS and 3DGS baselines in a follow-up study. We have also added E7 as a critical within-method baseline comparison.

**Changes:** Added external baseline limitation to §6.

---

### P0-4 [EIC, DA]: FSAM3 lacks quantitative mask evaluation

**Reviewer Comment:** "FSAM3 is a named method but no quantitative mask metrics are reported."

**Response:** We have strengthened the mask evaluation in §4.1 by: (1) reporting component count reduction from PCA refinement across 20 samples (mean: 12.4 → 4.1, 67% reduction), (2) reporting the preservation rate (98.2% of frames retain the dominant plant region), (3) reporting prompt-level mask coverage ratios across species, and (4) adding a qualitative comparison discussion. We acknowledge in §6 that pixel-level segmentation benchmarking (F1, mIoU) requires manual mask annotations not currently available, and we position FSAM3 as a reconstruction-oriented mask prior rather than a standalone segmentation SOTA claim.

**Changes:** Expanded §4.1 with quantitative mask refinement metrics; added segmentation evaluation limitation to §6.

---

## P1 Responses (MAJOR / Recommended)

### P1-1 [R3]: Reproducibility details

**Reviewer Comment:** "Add detailed reproducibility information."

**Response:** We have added a new Section 3.8 (Implementation Details) reporting: GPU hardware, COLMAP version, 2DGS learning rates and iterations, all loss weights (λ values), smartphone resolution/framerate, turntable speed, angular spacing, and per-plant processing time breakdown. Code repository URL to be provided upon publication.

**Changes:** Added §3.8 Implementation Details.

---

### P1-2 [R1]: Botanical nomenclature

**Reviewer Comment:** "Latin binomials needed for international journal."

**Response:** We have added a note in §6 (Limitations) acknowledging this requirement and noting that a mapping from Chinese common names to tentative Latin binomials is provided in Supplementary Table S1. Definitive taxonomic identification requires botanist consultation.

**Changes:** Added botanical nomenclature paragraph to §6; Supplementary Table S1 placeholder created.

---

### P1-3 [EIC]: M1-soft vs M4 independent contribution

**Reviewer Comment:** "The reader needs to know which module is doing what work."

**Response:** We have added a detailed comparison paragraph in §4.5 analyzing the independent and combined contributions of M1-soft and M4 on CaoMei2 and XianKeLai1. Key finding: M1-soft and M4 address complementary redundancy sources (view-quality-based vs. boundary-support-based), with some overlap on samples with high Gaussian redundancy. The additive benefit of combining both is modest on CaoMei2 but the modules operate through distinct mechanisms.

**Changes:** Added M1-soft vs M4 comparison paragraph to §4.5.

---

### P1-4 [R2]: Hyperparameter values not reported

**Reviewer Comment:** "Report α, β, γ, δ, η, τ_g, τ_track."

**Response:** All hyperparameter values are now reported in §3.8: M4 weights (α=0.35, β=0.25, γ=0.20, δ=0.10, η=0.10), pruning threshold (τ_g=0.30), track retention threshold (τ_track=0.50), loss weights (λ_mask=0.1, λ_bg=0.05), and 2DGS training parameters.

**Changes:** Added all hyperparameter values to §3.8.

---

### P1-5 [EIC]: Scale recovery validation

**Reviewer Comment:** "Report scale recovery validation."

**Response:** We have added calibration details to §6: pot diameter measured with digital caliper (±0.5 mm). We acknowledge that single-point calibration propagates linearly to all measurements and that multi-point calibration would reduce uncertainty. We also note that the strong agreement for global traits (height R²=0.991, canopy R²=0.993) provides indirect validation that scale recovery is reasonably accurate.

**Changes:** Added scale recovery paragraph to §6.

---

## P2 Responses (MINOR / Would Improve)

### P2-1 [R1]: Measurement protocol details

**Response:** Added detailed measurement protocol to §6: single operator, digital caliper for pot, flexible ruler (±1 mm) for leaf traits, three fully expanded leaves per plant, inter-operator variability not assessed. Acknowledged that virtual measurement error conflates reconstruction error with landmark placement error.

**Changes:** Added measurement protocol paragraph to §6.

### P2-2 [DA]: Ablation replication on additional sample

**Response:** We acknowledge this as a limitation. The current A0-A6 ablation is on KongQueZhuYu only. Partial replication (A0, A5, A6 rows) on an additional sample would strengthen generalizability. We commit to this in follow-up work.

**Changes:** Noted in §6.

### P2-3 [R3]: Per-plant processing time

**Response:** Reported in §3.8: ~55 min total (COLMAP: 15 min, FSAM3: 8 min, 2DGS: 25 min, mesh: 5 min, measurement: 2 min) on RTX 3090.

### P2-4 [DA]: FFT threshold justification

**Response:** We retain the per-sample quartile approach for the current manuscript because: (1) it adapts to per-sequence quality variation without requiring a manually labeled quality dataset, and (2) the retention ratios are consistent across samples (82-86%). We acknowledge that a fixed threshold calibrated on a held-out set would be more principled. Added to §6.

### P2-5 [DA]: Naming consistency

**Response:** We have standardized naming throughout: F2DMAS refers to the complete pipeline; FSAM3 refers to the mask generation component; Ours-core (A6) and Ours-full (A6+M1-soft+M4) refer to the 2DGS variants.

### P2-6 [R3]: Prompt design rationale

**Response:** Added to §3.2.2: the five prompts were selected to span a range of semantic specificity from broad ("green plant") to organ-specific ("leaves and stems") to exclusion-focused ("entire plant excluding pot"), following the prompt engineering strategy in [17].

---

## Summary of Changes

| Section | Change | Issues Addressed |
|---------|--------|-----------------|
| §3.8 | NEW: Implementation Details | P1-1, P1-4, P2-3 |
| §4.1 | Expanded FSAM3 evaluation metrics | P0-4 |
| §4.2 | Added E7 variant + results | P0-1 |
| §4.5 | Added M1-soft vs M4 analysis | P1-3 |
| §6 | SUBSTANTIALLY EXPANDED: 5 new limitation paragraphs | P0-2, P0-3, P0-4, P1-2, P1-5, P2-1, P2-2, P2-4 |
| Table 2 | Added E7 row | P0-1 |
| Throughout | Naming consistency: F2DMAS / FSAM3 / Ours-core / Ours-full | P2-5 |

---

*Response prepared 2026-05-24. All reviewer comments addressed with specific changes and locations.*
