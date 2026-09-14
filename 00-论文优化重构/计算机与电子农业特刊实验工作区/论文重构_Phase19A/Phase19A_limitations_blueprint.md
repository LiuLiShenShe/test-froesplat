# Phase 19A — Limitations Section Blueprint (ForeSplat Rewrite)

**Generated:** 2026-09-14 | **Gate:** G12 (limitations must be explicit; all NOT_RESOLVED and PARTIALLY_RESOLVED reviewer comments that cannot be addressed by results must appear here)
**Source inputs:** `Phase19A_claim_evidence_matrix.csv`, `Phase19A_claim_audit.csv`, `Phase19A_resolution_status.md`, `Phase19A_reviewer_issue_matrix.csv`, `Phase18_report.md`

---

## Blueprint conventions

- Each limitation subsection below gives: the limitation in **draft paragraph form**, the **evidence/comment anchors**, **allowed** and **forbidden wording**, **manuscript placement**, and a **follow-up action** (concrete future-work commitment required by reviewers).
- The G12 gate requires that every reviewer comment classified as NOT_RESOLVED in `Phase19A_resolution_status.md` has a corresponding limitation paragraph or explicit statement in the manuscript. This blueprint explicitly maps those.
- Global forbidden tokens apply: no `significant(ly)`, `outperform(s)`, `superior`, `SOTA`, `generalize(s)`, `strong agreement`, `robust` (except "R1 robust instance ranking" as a policy name).

---

## L1. Dataset composition, roles, and annotation scope

**Reviewer comment addressed.** Internal (Phase 18 §18 limitation 1); underpins all subsequent limitations.

**Draft paragraph.**
> This study uses several nested evaluation subsets with distinct roles. The frozen P6 evaluation set (21 annotated frames from 5 species) is the paper's primary masked-segmentation baseline. The GT10 diagnostic subset (10 annotated frames from 2 sequences) was used during failure-mode diagnosis and ranking-policy development; it was **not** held out and therefore cannot confirm generalization to new data. The TEST32 behavioral-audit set (32 unlabeled frames from 8 samples) supports only a selection-churn and behavioral-safety check (no ground-truth accuracy claims). Of the 58 frozen challenge targets, only 10 frames carry pixel-level ground truth; the remaining 48 target frames are unlabeled. Results on GT10 are therefore descriptive of the diagnosed failure mode, not of the method's overall accuracy distribution.

**Evidence anchors.** Phase 18 §5 (Dataset Roles); Phase 18 §18 (Limitations); P16-A06 (GT10 manifest); P16-A07 (TEST32 manifest).

**Allowed wording.** "GT10 ... participated in failure-mode diagnosis and fix development; not a held-out confirmation set"; "TEST32 = unlabeled behavioral safety only; no ground truth; cannot support accuracy claims"; "descriptive of the diagnosed failure mode".

**Forbidden wording.** Any claim that GT10 results constitute independent validation; "TEST32 confirms accuracy"; "58-target set demonstrates general performance".

**Manuscript placement.** Section 6 (Limitations), first paragraph — sets the context for all subsequent limitations.

**Follow-up.** Annotating a portion of the remaining 48 target frames for held-out evaluation of ranking policy is natural follow-up.

---

## L2. R1 ranking policy: not independently validated on held-out data

**Reviewer comment addressed.** Phase 18 §18 limitation 2; maps to C05 (NOT_SUPPORTED / FORBIDDEN).

**Draft paragraph.**
> The R1 robust instance-ranking policy was developed and selected using the GT10 diagnostic subset (10 frames) and then assessed for behavioral safety on the unlabeled TEST32 set (32 frames). No independent, ground-truth-validated held-out frame set was available to confirm that the identified ranking failure — and its resolution — generalize beyond the diagnosed sequence (DouBanLv2). R1 is therefore reported as the method that eliminated the identified failure mode with zero material regressions on the diagnostic and historical subsets, not as a method with a demonstrated general improvement in segmentation accuracy. The selection rationale (failure rescue, no regression, low churn, structural simplicity) is not evidence of global superiority.

**Evidence anchors.** C05 (FORBIDDEN), Phase 18 §18 (limitation 2), Phase 17C (R1 selection rationale), C04 (SUPPORTED — rescue on DEV set only).

**Allowed wording.** "R1 was selected for failure-rescue, zero material regression, low selection churn, and structural simplicity — not because its mean F1 is significantly higher"; "N=10 mean-F1 comparisons are descriptive only"; "descriptive development/pilot evidence".

**Forbidden wording.** "R1 significantly improves general segmentation" (C05 FORBIDDEN); "R1 proven superior globally"; "R1 improves/outperforms the baseline in general"; "general improves".

**Manuscript placement.** Section 6 (Limitations), second paragraph, immediately after L1.

**Follow-up.** Independent GT-based validation on a held-out set that was unseen during ranking-policy development.

---

## L3. A6/A7: conclusions limited to the evaluated datasets and settings

**Reviewer comment addressed.** C06, C07, C08, P08 (all NOT_SUPPORTED/DEMOTED/FALSE); EIC cross-module concerns partially resolved; G11 gate requires that A6/A7 do not appear in main Method.

**Draft paragraph.**
> The cross-view consensus (A6) and memory propagation (A7) mechanisms were evaluated on Easy21 (21 frames, 5 samples) and GT10 (10 frames, 2 sequences) under a factorial protocol with a corrected execution environment (Phase 14.1). Under these evaluated conditions, A6 did not provide consistent measurable benefit (E_A6 ≈ −0.003; one material regression ΔF1 = −0.058 on ChangShouHua2_0075) and A7 was empirically near-neutral (E_A7 ≈ 0; 13 propagated / 9 selected of 21 Easy21 frames; one marginal GT10 improvement ΔF1 = +0.006). Both are disabled by default. These findings are limited to the specific sequences, species, and acquisition conditions studied here; no broad claim is made about the utility of cross-view or temporal consistency mechanisms for VFM-based plant segmentation in general. Future work may re-evaluate these under different temporal densities, frame counts, or species architectures.

**Evidence anchors.** C06 (E_A6), C07 (E_A7, Phase14.1 corrected exposure), C08 (FALSE — defaults OFF), P08 (consolidated negative ablation), Phase 14/14.1 factorial data.

**Allowed wording.** "A6 did not provide consistent measurable benefit under the evaluated settings"; "A7 was empirically near-neutral under valid execution exposure"; "both are disabled by default"; "no broad claim is made".

**Forbidden wording.** "A6 improves / benefits segmentation"; "A7 improves segmentation"; "A6/A7 retained as default"; "A7 neutral based on the invalid 0-propagation observation" — must always cite Phase 14.1 corrected data; A6/A7 as main innovation; adding A6/A7 to main Method chapter (G11 violation).

**Manuscript placement.** Section 6 (Limitations), third paragraph. Must NOT appear in main Method chapter (G11 PASS confirmed in `Phase19A_method_component_audit.md`).

**Follow-up.** Re-evaluation of A6/A7 under conditions where temporal redundancy is higher (more frames per sequence, more sequential acquisitions) or architectural diversity is greater.

---

## L4. Reconstruction and phenotyping validation scope

**Reviewer comments addressed.** B-EIC-C01 (PARTIALLY_RESOLVED — cross-species), B-R2-C02 (NOT_RESOLVED — external baselines), B-R1-C02 (RESOLVED with limitation — no repeated measurements, single operator, no inter-rater reliability), B-DA-C02 (NOT_RESOLVED — single-sample A0-A6 ablation, no replication), A1-C03 (NOT_RESOLVED — leaf-width error source unresolved), B-EIC-C04 (RESOLVED with limitation — single-point scale calibration), P06 (single-sample injection ablation).

**Draft paragraph.**
> The foreground-object reconstruction (ForeSplat) is validated on 20 sequences (workflow-level metrics: PSNR = 31.09 dB, SSIM = 0.9711, LPIPS = 0.0365; +1.51 dB over standard 2DGS), but full-scene reconstruction ablation (A0–A6) and the foreground-object loss injection ablation (E7–E9 in Table 2) are each based on a single representative sample (KongQueZhuYu); replication on additional samples is not yet available, and these ablation findings are treated as development-set results. Reconstruction is reported on 3 architecturally representative samples from the 20-sequence set, which is not a statistical species-level generalization study; cross-species applicability is supported at the mask level (F1 = 0.9835, IoU = 0.9676 on 21 GT frames from 5 species, Phase 12) but reconstruction-level cross-species claims remain an open extension. No external reconstruction baselines (e.g., COLMAP+MVS, NeRF-family, standard 3DGS with the same masks) were included; whether 2DGS is the optimal base representation is therefore not established, and within-method ablation was chosen because internal mechanism questions are best answered by controlled component variation. Trait measurements from reconstructed meshes were comparable to manual measurements (R² 0.988, 0.988, 0.974, 0.900 on n = 21 plants / n = 63 leaves, descriptive only). Leaf width showed the weakest agreement (R² = 0.900, MAPE = 9.73%); the error source — segmentation boundary shifts, 2DGS surface expansion, or TSDF boundary smoothing — is not attributed and point-cloud-based measurement is noted as a future direction. The manual measurement protocol was carried out by a single operator using digital calipers (± 0.5 mm) with three leaves per plant; inter-operator repeatability and landmark-placement variability are not assessed. Scale recovery used a single-pot-diameter calibration (± 0.5 mm caliper accuracy); no multi-point validation was performed, though the plant-height agreement (R² ≈ 0.99) provides indirect supporting evidence. No repeated measurements were collected on the virtual or manual side, and no Bland–Altman or inter-rater reliability analysis was conducted.

**Evidence anchors.** P05 (20-sequence workflow), P06 (single-sample injection ablation), P07 (21 plants / 63 leaves), B-EIC-C01 (PARTIALLY_RESOLVED — 3-sample reconstruction), B-R2-C02 (NOT_RESOLVED — no external baselines), B-DA-C02 (NOT_RESOLVED — single-sample A0–A6), A1-C03 (NOT_RESOLVED — leaf-width error), B-R1-C02 (single operator), B-EIC-C04 (single-point scale), Phase 12 FSAM3 mask metrics.

**Allowed wording.** "comparable to manual measurements (R² 0.988, 0.988, 0.974, 0.900) on the 21-plant / 63-leaf set (descriptive, no repeated measurements collected)"; "architecturally representative samples, not a statistical species-level generalization study"; "the error pattern aligns with the trends reported in recent 3D plant phenotyping studies"; "point-cloud-based measurement is noted as a future direction"; "multi-point calibration would reduce scale uncertainty"; "leaf-area and stem-diameter were excluded because the mesh does not support reliable area computation at this stage" (ties to B-R1-C03 — RESOLVED).

**Forbidden wording.** "strong agreement" (claim_audit row 9); "virtual measurements are accurate/superior"; "R² proves measurement accuracy"; "cross-species generalization at the reconstruction level" (B-EIC-C01 — 3 samples only); "outperforms SOTA"; "best reconstruction"; "post hoc pruning is useless" (P06 — "insufficient" is honest).

**Manuscript placement.** Section 6 (Limitations), fourth paragraph — the longest paragraph, covering reconstruction scope, external baselines, phenotyping validation, and measurement protocol.

**Follow-up actions (explicit reviewer commitments).**
1. Replicate A0–A6 ablation on at least one additional sample (B-DA-C02).
2. Add external baselines: COLMAP+MVS, NeRF-family, standard 3DGS with same masks (B-R2-C02).
3. Attribute leaf-width error to a specific pipeline stage; evaluate point-cloud-based measurement (A1-C03).
4. Collect repeated measurements and conduct Bland–Altman / inter-rater reliability analysis (B-R1-C02).
5. Extend reconstruction validation beyond 3 architecturally representative samples (B-EIC-C01).

---

## L5. FFT threshold: sample-dependent heuristic

**Reviewer comment addressed.** B-DA-C04 (NOT_RESOLVED).

**Draft paragraph.**
> Frame-quality filtering uses a fast Fourier transform (FFT) threshold set to the first quartile of per-sequence frequency scores. This is a heuristic choice: the threshold adapts to per-sequence quality variation, but it is sample-dependent — a high-quality sequence retains all frames while a lower-quality sequence may exclude 25% of its frames by internal threshold, and conversely a fixed threshold calibrated on a held-out set was not implemented. Sensitivity of downstream reconstruction and phenotyping metrics to the FFT threshold is therefore unreported. A threshold-sensitivity analysis using a fixed or optimized threshold is identified as future work.

**Evidence anchors.** B-DA-C04 (NOT_RESOLVED), `Phase19A_reviewer_issue_matrix.csv`.

**Allowed wording.** "per-sample quartile threshold"; "a heuristic choice"; "sample-dependent"; "threshold sensitivity is unreported".

**Forbidden wording.** "optimal threshold"; "the FFT threshold is validated"; "the threshold generalises across species/conditions"; claiming the threshold is well-calibrated.

**Manuscript placement.** Section 6 (Limitations), fifth paragraph. Also note in Methods (Section 3.2.1) the rationale for the per-sample quartile approach, per resolution_status (B-DA-C04) action.

**Follow-up.** Threshold-sensitivity analysis; fixed-threshold calibration on a held-out set.

---

## L6. No statistical significance tests — descriptive distributions only

**Reviewer comments addressed.** A2-C01 (RESOLVED — R² reported to 2 decimal places in Abstract), A2-C02 (RESOLVED — MAPE added), A2-C01 in the claim_audit: no instance of "significantly" as a statistical claim exists in the manuscript, and no significance test is conducted.

**Draft paragraph.**
> All numerical results in this study are descriptive statistics (means, medians, R², MAE, MAPE) computed on finite, non-independent sample sets without repeated measurements. No statistical significance tests (t-tests, ANOVA, non-parametric equivalents) are reported or used to support any claim. R² values should therefore be interpreted as descriptive agreement metrics, not as evidence of a statistically significant association. Two-digit precision is used for R² throughout, consistent with the measurement accuracy of the manual protocol.

**Evidence anchors.** A2-C01 (R² precision constraint), A2-C02 (RMSE context), Phase 18 §18 ("No significance test performed on any result; all distributions are descriptive"), claim_audit row 16 ("significantly / significant: no instances with a reported test").

**Allowed wording.** "descriptive statistics"; "descriptive agreement metrics"; "no statistical significance test is reported"; "no repeated measurements were collected".

**Forbidden wording.** "significantly" / "significant" as a statistical claim without a reported test; "significant improvement" / "significant difference" / "p < 0.05" (none of these appear, and must not be added without a test).

**Manuscript placement.** Section 6 (Limitations), sixth paragraph. Also serves as a standing constraint that should appear at the end of Section 3 (Methods) as a framing statement for all Results tables: "All values are descriptive; no significance test is applied."

**Follow-up.** The paper may optionally add a brief statement in the Abstract (per A2-C01: "descriptive, no significance test") and in Section 7 (Conclusion): "reported results are descriptive and do not imply statistical significance."

---

## L7 (supplementary). Botanical nomenclature pending

**Reviewer comment addressed.** B-R1-C01 (NOT_RESOLVED — placeholder table only; definitive identification requires botanist consultation).

**Draft paragraph.**
> Common Chinese species names are used throughout the manuscript. Verified botanical nomenclature (accepted Latin binomials, family, growth form) is provided in Supplementary Table S1; definitive taxonomic identification for some species (e.g., KongQueZhuYu may correspond to multiple *Calathea* species; XianKeLai may correspond to *Cyclamen* or a related genus) may require specialist consultation and is noted as pending at proof stage. Readers should consider the species names as provisional pending confirmation.

**Evidence anchors.** B-R1-C01 (NOT_RESOLVED), Supplementary Table S1 placeholder.

**Allowed wording.** "provisional pending confirmation"; "requires specialist consultation"; "provided in Supplementary Table S1".

**Forbidden wording.** "verified botanical nomenclature" as established; claiming cross-species taxonomic precision; the species name as the basis for generalization claims.

**Manuscript placement.** Section 6 (Limitations), brief seventh paragraph, or a footnote in Supplementary Table S1 cross-referenced from Section 6.

**Follow-up.** Botanist consultation at proof stage; update Supplementary Table S1 with confirmed nomenclature.

---

## Summary: G12 gate checklist

| Limitation | NOT_RESOLVED comment addressed | Covered? |
|---|---|---|
| L1 Dataset | Internal (Phase 18 §18) | Yes |
| L2 R1 validation | C05 (Phase 18 §18) | Yes |
| L3 A6/A7 scope | C06-C08, P08 | Yes |
| L4 Reconstruction/phenotyping | B-EIC-C01, B-R2-C02, B-DA-C02, A1-C03, B-R1-C02, B-EIC-C04 | Yes |
| L5 FFT threshold | B-DA-C04 | Yes |
| L6 Descriptive statistics | A2-C01, A2-C02 | Yes |
| L7 Botanical nomenclature | B-R1-C01 | Yes |

All NOT_RESOLVED reviewer comments have a corresponding limitation paragraph. The Discussion blueprint (Phase19A_discussion_blueprint.md) covers the remaining PARTIALLY_RESOLVED and RESOLVED comments that require honest framing.
