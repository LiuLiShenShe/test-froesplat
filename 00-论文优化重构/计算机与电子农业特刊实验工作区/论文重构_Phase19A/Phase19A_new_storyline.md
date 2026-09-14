# Phase 19A — New Storyline (ForeSplat Rewrite)

**Generated:** 2026-09-14 | **Purpose:** ≤15-step narrative chain that makes the ranking-failure + R1 insight the scientific spine WITHOUT hiding the phenotyping core (CompAg needs tangible ag value)
**Source inputs:** `Phase19A_claim_evidence_matrix.csv`, `Phase19A_claim_audit.csv`, `Phase19A_resolution_status.md`, `Phase18_report.md`

---

## Narrative spine in one sentence

> Direct VFM prompting is fragile for potted plants (prompt problem); even when a high-quality candidate mask exists, a contrast-biased ranking can select a distractor and collapse F1 to zero (ranking problem); RAP-FSAM3 generates per-instance candidates and R1 ranks them without the contrast term, rescuing all identified catastrophic failures with zero regressions; the selected masks drive ForeSplat foreground-object 2DGS, giving reconstruction quality and time gains that support downstream trait measurement comparable to manual values — with the validation scope stated explicitly.

---

## The 15 steps

### 1. Agricultural problem → low-cost 3D plant phenotyping need
**Narrative.** Image-based, three-dimensional plant phenotyping is needed for breeding and management decisions, but conventional acquisition and measurement pipelines are labor-intensive and costly. A low-cost route (smartphone/turntable acquisition, open-source reconstruction) would broaden access for agricultural practice.
**Evidence anchors.** P07 (downstream value: traits comparable to manual), P05 (20-sequence workflow feasibility), A2-C04 (acquisition-time / throughput framing).
**Manuscript section.** Introduction §1 (paragraph 1).
**Forbidden wording.** Unverifiable urgency claims; "high-throughput field phenotyping at scale" (no field evidence).

### 2. VFM promise → flexible segmentation for plants
**Narrative.** Vision foundation models (SAM family and text/point-promptable segmenters) promise task- and domain-flexible segmentation without per-species training, which is attractive for diverse ornamental and crop plants.
**Evidence anchors.** P01/P03 context (VFM used as the mask generator).
**Manuscript section.** Introduction §1 (paragraph 2).
**Forbidden wording.** "VFM segmentation is robust across plants" — no such claim exists.
**Citation slots.** `[SLOT]` SAM-family foundation models; `[SLOT]` VFM applications in agriculture (special-issue scope).

### 3. Prompting fails on plants → semantic fragility
**Narrative.** Direct single-prompt VFM segmentation is semantically fragile in potted-plant scenes: the same image gives very different masks depending on prompt wording. Per-prompt mean F1 ranges from 0.08 (seedling-oriented prompt on mature plants) to 0.97 (plant-instance prompt); green-region and background-excluding prompts fail in characteristic ways (background inclusion, under-recall, over-conservatism).
**Evidence anchors.** P03 (PROVEN, per-prompt F1 report: 0.0768 / 0.7972 / 0.6471 / 0.9692 / 0.4051).
**Manuscript section.** Introduction §1 (problem motivation) + Results §3.2 (prompt-robustness table).
**Forbidden wording.** "single-prompt VFM segmentation is robust"; "the prompt set generalises to all scenes"; "P6 prompt failed" (attribute to ranking, not prompt, on hard frames).

### 4. Foreground-object reconstruction paradigm
**Narrative.** For phenotyping, the object of interest is the foreground plant, not the full scene. Full-scene reconstruction followed by post-hoc mask pruning is insufficient: reconstruction capacity is diverted to background, and pruning cannot recover it. This motivates foreground-object reconstruction as the paradigm, and motivates the question: where do the foreground masks come from?
**Evidence anchors.** P06 (injection ablation: full-scene outside-mask non-black ratio 0.9908 / leakage 1.2201; post hoc pruning leaves outside 0.7509 / leakage 0.7900), B-DA-C01 (E7 framing).
**Manuscript section.** Introduction §1–§2 (paradigm) + Results §3.4 (injection ablation) + Discussion D4/D5.
**Forbidden wording.** "post hoc pruning is useless" (P06: "insufficient"); "post hoc pruning is equivalent to foreground-object injection" (B-DA-C01 — ranking evidence is related but not identical).

### 5. RAP-FSAM3: reconstruction-aware prompt adaptation
**Narrative.** RAP-FSAM3 is the segmentation front-end: it adapts prompting to the reconstruction need — per-instance candidate generation, semantic-gated multi-prompt selection, structure-balanced foreground–background prompt refinement (SB-FBPR), and geometry-consistency foreground correction (GCFC). The full pipeline reaches F1 = 0.9706 / mIoU = 0.9641 on the frozen 21-frame evaluation set.
**Evidence anchors.** P01 (F1=0.9706, mIoU=0.9641, HD95=183.42 px, Boundary-F1=0.4643, leakage energy=0.001739), P04 (module ablation), P02 (delta concentration).
**Manuscript section.** Methods §2 (RAP-FSAM3) + Results §3.1 (VFM comparison) + Results §3.3 (module ablation).
**Forbidden wording.** "significantly outperforms single-prompt SAM3" (P02 — use the boundary/leakage delta phrasing); "state-of-the-art segmentation".

### 6. Candidate generation + scoring
**Narrative.** RAP-FSAM3 generates per-instance candidate masks (production default prompt list P2) and scores each candidate with interpretable features (area, compactness, edge quality, temporal consistency, contrast, leakage, side, and SAM confidence). The instance with the highest primary score is selected. Candidate generation and scoring are explicit, auditable stages.
**Evidence anchors.** C02/C04 context (ranking stage), P04 (SG-MPS), method_component_audit §3 (R1 addition), Phase 18 §2 (feature list: q_area, q_comp, q_edge, q_temp, q_contrast, q_leak, q_side, sam_score).
**Manuscript section.** Methods §2.3 (candidate generation + scoring) — this section must describe the production R1 primary score (q_contrast excluded), with the superseded selector (q_contrast included) described as the historical/diagnostic variant (G11 rewrite required per method_component_audit).
**Forbidden wording.** Describing q_contrast as part of the production primary score (method_component_audit G11 violation); leaving R1 out of the Method entirely.

### 7. Catastrophic failures on hard frames
**Narrative.** On a hard diagnostic subset (DouBanLv2 frames 0039/0040/0041), the pipeline's final F1 collapsed to 0.0000 under the original ranking; overall diagnostic mean F1 was 0.6767 (3 catastrophic failures). These frames are the visible symptom of a deeper problem.
**Evidence anchors.** C02 (PROVEN), Phase 18 §7 (R0: mean F1 0.6767, 3 catastrophic, oracle regret 0.2961), Phase 18 §9 (R0 GT10 mean 0.6767, min 0.0000).
**Manuscript section.** Results §3.3 (ranking-failure analysis) — supported by Phase18 Figure A (5-panel rows) and failure-rescue table.
**Forbidden wording.** "P6 could not detect the plant"; "candidate generation failed" (Phase17A→17B correction).

### 8. Candidate-level oracle: generation was fine
**Narrative.** An oracle audit re-selects the best candidate from the generated set. On the same three failure frames, the oracle candidate scored raw F1 = 0.9867 / 0.9882 / 0.9865. The correct whole-plant mask existed in the candidate set on every failure frame.
**Evidence anchors.** C01 (PROVEN), P17B-A09/P17B-A10 (oracle audit, best-instance F1).
**Manuscript section.** Results §3.3 (oracle audit) + Phase18 Figure A.
**Forbidden wording.** "no correct candidate existed"; "generation failed"; "P6/prompt failed to detect DouBanLv2 0039–0041".

### 9. Root cause: ranking, not segmentation
**Narrative.** Diagnosis class C3_RANKING_FAILURE (wrong instance selected): the selected candidate was a small, high-contrast distractor. On failure frames, the contrast signal (q_contrast) showed a strong negative association with candidate quality (Spearman rho = −0.7187, p = 0.0038) — q_contrast can favor small, high-contrast distractors over larger whole-plant candidates. This evidence is correlational on the identified failure subset.
**Evidence anchors.** C02 (PROVEN diagnosis), C03 (SUPPORTED mechanism), Phase 18 §6 (root-cause), Phase18 Figure B (candidate feature comparison).
**Manuscript section.** Results §3.3 (mechanism) + Discussion D2.
**Forbidden wording.** "q_contrast is the proven/global/sole cause"; "q_contrast always favors distractors" (SUPPORTED, not PROVEN).

### 10. R1 ranking fix rescues + no regressions
**Narrative.** R1 removes q_contrast from the primary score (retained for diagnostics) and selects by the corrected primary score. R1 rescues all 3 identified failures to F1 ≥ 0.95, mean diagnostic F1 rises to 0.9728, oracle regret drops to 0.0000, with zero material regressions on Easy21 (21 frames, 5 samples) and 0 empty selections; selection churn on the 32-frame behavioral audit is 6.2% (2/32). R1 was selected by a pre-registered decision order (rescue → no regression → oracle regret → churn → simplicity), not because its mean F1 is statistically higher.
**Evidence anchors.** C04 (SUPPORTED), C05 (FORBIDDEN — no general-improvement claim), Phase 18 §6/§7/§8 (decision order, rescue table, Easy21 safety), P17C-A13 (churn 6.2%).
**Manuscript section.** Results §3.3 (R1 rescue) + Discussion D1/D2 + Tables (failure rescue, ranking ablation, Easy21 1b).
**Forbidden wording.** "R1 significantly improves general segmentation"; "R1 is proven superior globally"; "R1 improves segmentation in general" (C05).

### 11. ForeSplat foreground-object 2DGS injection
**Narrative.** The R1-selected masks are injected into ForeSplat, a foreground-object 2D Gaussian Splatting reconstruction: foreground track initialization, foreground RGB supervision with mask and background-opacity constraints, view-quality-aware soft loss weighting, and mask-guided Gaussian pruning. Foreground-object loss injection is necessary — full-scene 2DGS leaves outside-mask content and leakage; post hoc mask pruning does not recover them.
**Evidence anchors.** P06 (injection ablation, single representative sample), P05 (workflow), method_component_audit §1 (production-consistent components).
**Manuscript section.** Methods §3 (ForeSplat) + Results §3.4 (injection ablation) + Table 6 (E7 row; note single-sample limitation per B-DA-C01).
**Forbidden wording.** "input masking or post hoc pruning is sufficient"; "foreground-object injection is not necessary" (P06 says necessary, single-sample); "significant test implied".

### 12. Reconstruction quality + time gains
**Narrative.** Under the unified 20-sequence protocol, ForeSplat achieved PSNR = 31.09 dB, SSIM = 0.9711, LPIPS = 0.0365, +1.51 dB over standard 2DGS (29.58 dB), with reduced training/reconstruction time (60.94%) and meshing time (65.17%); meshing 55 s vs 642 s for 3DGS+preproc. Reported as workflow-level descriptive means.
**Evidence anchors.** P05 (DESCRIPTIVE), B-R3-C04 (per-plant time ~55 min RTX 3090 — from resolution_status), B-R3-C01 (absolute memory/FPS).
**Manuscript section.** Results §3.5 (workflow validation) + Methods §3.8 (implementation details: hardware, timing breakdown).
**Forbidden wording.** "significantly improves reconstruction quality"; "state-of-the-art reconstruction"; "outperforms standard 2DGS" (use "+1.51 dB compared with").

### 13. Phenotypic agreement (comparable)
**Narrative.** Mesh-based trait measurements were comparable to manual measurements: plant height R² = 0.988 (MAE 0.98 cm), canopy width R² = 0.988 (MAE 0.86 cm), leaf length R² = 0.974 (MAE 0.51 cm, n = 63), leaf width R² = 0.900 (MAE 0.45 cm). The weakest trait is the boundary-sensitive organ-level one (leaf width); this pattern aligns with the trends reported in recent 3D plant phenotyping studies. Descriptive only — no repeated measurements, no Bland–Altman, no inter-rater analysis.
**Evidence anchors.** P07 (DESCRIPTIVE, "comparable to"), A1-C03 (leaf-width error source NOT_RESOLVED), claim_audit rows 9 and 14.
**Manuscript section.** Results §3.5 (phenotyping) + Discussion D5 + Limitations L4.
**Forbidden wording.** "strong agreement"; "virtual measurements are accurate/superior"; "R² proves measurement accuracy"; "statistically validated agreement".

### 14. A6/A7 evaluated, not retained (design decision)
**Narrative.** Two candidate enhancements — A6 cross-view consensus and A7 memory propagation — were implemented and evaluated under corrected execution. Neither provided consistent measurable benefit under the evaluated settings (A6: E_A6 ≈ −0.003, one material regression ΔF1 = −0.058; A7: E_A7 ≈ 0 under valid exposure, one marginal GT10 gain ΔF1 = +0.006). Both are disabled by default. This is an evidence-based design decision — not a degradation, not a headline innovation — and it is reported as a negative result in the ablation section.
**Evidence anchors.** C06, C07, C08 (FALSE), P08 (consolidated negative ablation), Phase14.1 corrected A7 exposure, Phase18 Figure D.
**Manuscript section.** Results (A6/A7 negative ablation) + Discussion D3 + Limitations L3. NOT in main Method (G11 PASS).
**Forbidden wording.** "A6/A7 improve segmentation"; "A6/A7 retained as default"; "A7 neutral based on the invalid 0-propagation observation"; A6/A7 as main innovation; conflating ranking rescue with post-hoc pruning equivalence (B-DA-C01).

### 15. Limitations + ag-relevant future work
**Narrative.** The paper closes with explicit scope: GT10 is a diagnostic (not held-out) subset; TEST32 is unlabeled behavioral safety only; R1 has no independent held-out GT validation; reconstruction validation is 3 architecturally representative samples (no cross-species reconstruction claim); no external reconstruction baselines; single-sample A0–A6/injection ablations; FFT threshold is a sample-dependent heuristic; all statistics are descriptive; botanical nomenclature pending. Ag-relevant future work: external baselines (COLMAP+MVS, NeRF, 3DGS with same masks), threshold-sensitivity analysis, replicated ablations, repeated-measurement / Bland–Altman / inter-rater phenotyping validation, point-cloud-based leaf-width measurement, scale-marker repeatability, held-out R1 validation.
**Evidence anchors.** Phase 18 §18 (limitations), resolution_status NOT_RESOLVED set (B-R2-C02, B-DA-C02, B-DA-C04, A1-C03, B-R1-C01), B-EIC-C01 (PARTIAL), B-DA-C01 (PARTIAL).
**Manuscript section.** Section 6 (Limitations) + Section 7 (Conclusion / Future Work).
**Forbidden wording.** Any claim that turns a limitation into a strength ("despite the small reconstruction set, the method clearly generalizes"); "outperforms SOTA"; "generalizes to other crops/species".

---

## Section mapping summary

| Step | Manuscript section | Primary anchors |
|---|---|---|
| 1–2 | Introduction §1 | P07, P05 (motivation) |
| 3 | Intro §1 + Results §3.2 | P03 |
| 4 | Intro §2 + Results §3.4 | P06, B-DA-C01 |
| 5 | Methods §2 + Results §3.1/§3.3 | P01, P02, P04 |
| 6 | Methods §2.3 | C02/C04, method_component_audit |
| 7 | Results §3.3 | C02, Phase 18 §7/§9 |
| 8 | Results §3.3 | C01 |
| 9 | Results §3.3 + Discussion D2 | C02, C03 |
| 10 | Results §3.3 | C04, C05 (forbidden) |
| 11 | Methods §3 + Results §3.4 | P06 |
| 12 | Results §3.5 + Methods §3.8 | P05 |
| 13 | Results §3.5 + Discussion D5 | P07, A1-C03 |
| 14 | Ablation + Discussion D3 + Limitations L3 | C06, C07, C08, P08 |
| 15 | Section 6 + Section 7 | Phase 18 §18; NOT_RESOLVED set |

---

## "What must not appear" checklist (storyline-level)

1. No "cross-species" at the reconstruction level (B-EIC-C01).
2. No "P6/candidate generation failed" anywhere (Phase17A→17B correction).
3. No q_contrast in the production primary score description (G11 rewrite).
4. No A6/A7 in the main Method or as innovation (G11/P08).
5. No "significant / outperforms / SOTA / generalizes / strong agreement" (claim_audit).
6. No conflation of ranking-failure rescue with post-hoc-pruning equivalence (B-DA-C01).
7. No "robust" except inside "R1 robust instance ranking" (claim_audit row 12).
8. All F1 / R² / PSNR numbers in the story are the descriptive values from the claim_evidence_matrix — no hand-invented aggregates.
