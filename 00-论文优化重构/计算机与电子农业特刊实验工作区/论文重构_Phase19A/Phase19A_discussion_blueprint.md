# Phase 19A — Discussion Section Blueprint (ForeSplat Rewrite)

**Generated:** 2026-09-14 | **Source inputs:** `Phase19A_claim_evidence_matrix.csv`, `Phase19A_claim_audit.csv`, `Phase19A_resolution_status.md`, `Phase19A_reviewer_issue_matrix.csv`, `Phase18_report.md`
**Gate:** WP6C (Discussion/Abstract/Title/Story blueprints) — part of WP6.

---

## Blueprint conventions

- Each subsection below gives: **the claim being made**, **evidence anchors** (claim_id from `Phase19A_claim_evidence_matrix.csv`), **allowed wording** (verbatim, safe to lift), **forbidden wording** (must never appear in that subsection), **manuscript placement**, and **literature citation slots** (placeholders — must pass citation verification before use; marked `[SLOT]`).
- Global forbidden tokens (apply to all subsections unless a policy-name exception is noted): `significant(ly)` as statistical claim, `outperform(s)`, `superior`, `state-of-the-art` / `SOTA`, `generalize(s)`, `strong agreement`, `robust` as a general adjective (allowed only in the method name "R1 robust instance ranking").
- The Discussion must cite prior work (addresses **A2-C06**), and literature currently sitting in Methods must move to Discussion (addresses **A2-C03**).

---

## D1. Why candidate ranking matters for VFM-based segmentation

**Claim being made.** For vision-foundation-model (VFM) segmentation of plants, a correct high-quality candidate mask can exist in the candidate set and still be lost because the *instance-selection* (ranking) function picks a wrong candidate. Catastrophic failure is therefore a selection problem, not a detection/generation problem. This motivates treating "candidate generation" and "candidate ranking" as distinct, independently auditable pipeline stages.

**Evidence anchors.** C01 (PROVEN), C02 (PROVEN), C04 (SUPPORTED).

| Anchor | Value to cite |
|---|---|
| C01 | Raw oracle F1 = 0.9867 / 0.9882 / 0.9865 on DouBanLv2 0039/0040/0041 despite final F1 = 0 |
| C02 | Diagnosis class C3_RANKING_FAILURE (wrong_instance selected); best-instance F1 0.9867–0.9882 vs selected-instance F1 0.0 |
| C04 | R1 (q_contrast removed from primary score) rescues all 3 identified DEV-set failures to F1 ≥ 0.95; 0 material Easy21 regressions |

**Allowed wording.**
- "Baseline candidate generation produced high-quality whole-plant candidate masks (raw oracle F1 = 0.9867–0.9882) even on frames where final F1 = 0."
- "Catastrophic failures (DouBanLv2 0039–0041) were caused by candidate-ranking failure (selected distractor), not candidate generation (C3_RANKING_FAILURE)."
- "High-quality candidate masks were present, but the original candidate-ranking function selected distractor instances."
- "The failure mode was isolated by an oracle audit, which re-selects the best candidate from the generated set; this separates generation capacity from selection quality."

**Forbidden wording.** "P6/candidate generation failed"; "no correct candidate existed"; "P6 could not detect the plant"; "P6 prompt failed"; any phrasing that attributes the F1 = 0 frames to the segmentation backbone or the prompt rather than to the ranking function (Phase17A→17B historical correction, §4 of Phase18_report.md).

**Literature citation slots (verify before use):**
- `[SLOT]` SAM / SAM2 / SAM3 prompting behavior in dense-instance settings (regional/dense prompting).
- `[SLOT]` Instance-selection / mask-scoring literature showing selection quality is separable from proposal quality (e.g., Mask Scoring R-CNN family; objectness-scoring in two-stage detectors).
- `[SLOT]` Candidate-over-generation / proposal-quality vs. selection-bottleneck analyses in instance segmentation.
- `[SLOT]` Agricultural segmentation papers that report prompt/selection fragility (special-issue relevant).

---

## D2. Why q_contrast is misleading and how R1 fixes it

**Claim being made.** On the identified hard frames, the contrast/separability signal (q_contrast) in the superseded scoring function systematically favored small, high-contrast distractor masks over larger whole-plant candidates. This evidence is **correlational (SUPPORTED, not PROVEN)**: the mechanism is consistent with the observed failure subset, not established as a global property. R1 excludes q_contrast from the primary score and thereby eliminates the identified failure mode; R1 was chosen by a pre-registered decision order (rescue → no regression → oracle regret → churn → simplicity), **not** because its mean F1 is significantly higher.

**Evidence anchors.** C03 (SUPPORTED), C04 (SUPPORTED), C05 (NOT_SUPPORTED / FORBIDDEN).

| Anchor | Value to cite |
|---|---|
| C03 | Failure-frame Spearman rho = −0.7187, p = 0.0038; contrast-drop ablation rescued 3/3 identified failures |
| C04 | R1 rescues 0039/0040/0041 to F1 ≥ 0.95; 0 material regressions on Easy21; oracle regret 0.0000 |
| C05 | N=10 mean-F1 comparisons are **descriptive only**; do not claim significance or generalization |

**Allowed wording.**
- "q_contrast can favor small, high-contrast distractor masks over larger whole-plant candidates (failure-frame Spearman rho = −0.7187, p = 0.0038; contrast-drop ablation rescued 3/3 identified failures)."
- "Removing q_contrast from the primary ranking (R1) eliminated all identified DEV-set ranking failures (0039/0040/0041 rescued to F1 ≥ 0.95) with zero material regressions on Easy21."
- "R1 was selected for failure-rescue, zero material regression, low selection churn (6.2% on the 32-frame behavioral audit), and structural simplicity — not because its mean F1 is significantly higher (N=10 mean-F1 comparisons are descriptive only)."
- "We interpret this as consistent with a contrast-driven distractor bias on the identified hard frames; the mechanism remains correlational and is not asserted as a global property of the scoring space."

**Forbidden wording.** "q_contrast is the proven/global/sole cause of ranking failures"; "q_contrast always favors distractors"; "R1 significantly / statistically significantly improves / generalizes / outperforms the baseline" (C05 FORBIDDEN); "R1 proven superior globally"; "R1 robust segmentation" (general adjective use of "robust").

**Literature citation slots (verify before use):**
- `[SLOT]` Saliency/contrast priors and their documented bias toward small, high-contrast regions (saliency-detection literature).
- `[SLOT]` Weighting of hand-crafted scoring features in instance selection and its failure modes.
- `[SLOT]` Descriptive vs. inferential reporting norms for small-sample engineering evaluations.

---

## D3. Why A6/A7 are NOT retained (design decision, not degradation)

**Claim being made.** A6 (cross-view consensus) and A7 (memory propagation) were implemented, evaluated under corrected execution conditions, and **not retained** as production defaults because they did not provide consistent measurable benefit under the evaluated settings. This is an explicit, evidence-based design decision; it is **not** a core contribution, not a claim of degradation of the baseline, and it must not be presented as a headline innovation. Addresses **B-EIC-C03** (module roles clarified via a dedicated interpretation paragraph) and **B-DA-C01** (must not conflate ranking-failure rescue with post-hoc-pruning equivalence).

**Evidence anchors.** C06 (NOT_SUPPORTED/DEMOTED), C07 (NOT_SUPPORTED/DEMOTED), C08 (NOT_SUPPORTED/FALSE), P08 (consolidated negative ablation).

| Anchor | Value to cite |
|---|---|
| C06 | A6: E_A6 ≈ −0.003 on Easy21; one material regression ΔF1 = −0.058 (ChangShouHua2_0075); GT10 4/4 acted frames regressed |
| C07 | A7: E_A7 ≈ 0 under the corrected Phase 14.1 exposure (13 propagated / 9 selected of 21 frames on Easy21); one marginal GT10 improvement ΔF1 = +0.006 |
| C08 | Production defaults: `--use_cross_view_consensus False`, `--use_memory_propagation False`; A6/A7 are optional experimental branches |

**Allowed wording.**
- "A6 (cross-view consensus) did not provide consistent measurable benefit under the evaluated settings (E_A6 ≈ −0.003; one material regression ΔF1 = −0.058 on ChangShouHua2_0075)."
- "A7 (memory propagation) was empirically near-neutral (E_A7 ≈ 0) under valid execution exposure (13 propagated / 9 selected of 21 frames on Easy21); one marginal GT10 improvement (ΔF1 = +0.006)."
- "A6 and A7 are optional experimental branches, disabled by default; they are not retained as default production enhancements."
- "Retaining the simpler pipeline follows the pre-registered selection principle: an enhancement is retained only if it provides a consistent measurable benefit; neither A6 nor A7 met that criterion here."

**Forbidden wording.** "A6 improves / consistently benefits segmentation"; "A7 improves segmentation"; "A6/A7 retained as default production enhancements" (C08 is FALSE); "A6/A7 provide consistent benefit"; "A7 neutral based on the invalid 0-propagation observation" — must cite the **Phase 14.1 corrected exposure**, never the Phase 14 zero-propagation run (§4 historical corrections); presenting A6/A7 as part of the main Method (G11 — do NOT add them to the Method chapter); conflating the ranking-failure rescue with post-hoc-pruning equivalence (B-DA-C01 — the Phase 17B/17C evidence is related but **not identical** to E7 post-hoc pruning).

**Manuscript placement.** Results (negative-ablation subsection) + a short Discussion paragraph. A6/A7 must NOT appear as main-method stages.

**Literature citation slots (verify before use):**
- `[SLOT]` Cross-view / multi-view consistency mechanisms in multi-view segmentation (positioning A6's premise).
- `[SLOT]` Temporal memory / propagation in video and multi-view segmentation (positioning A7's premise).
- `[SLOT]` Reporting norms for negative results in systems/engineering papers.

---

## D4. Where the method sits relative to external SOTA (honest positioning)

**Claim being made.** All comparisons in this paper are **within-method ablations** on the paper's own pipeline; no external baseline (COLMAP+MVS, NeRF-family, standard 3DGS/2DGS with the same masks) is included. The rationale for within-method ablation is that internal mechanism questions are best answered by controlled component variation. Whether 2DGS is the right base representation relative to external alternatives remains an open question committed to follow-up work. The paper must NOT claim superiority over external methods. Addresses **B-R2-C02** (NOT_RESOLVED → framed as limitation + future work).

**Evidence anchors.** P05 (DESCRIPTIVE), P06 (DESCRIPTIVE), B-R2-C02.

| Anchor | Value to cite |
|---|---|
| P05 | PSNR = 31.09 dB, SSIM = 0.9711, LPIPS = 0.0365 on the unified 20-sequence protocol; +1.51 dB vs standard 2DGS (29.58 dB); train/recon time −60.94%; meshing time −65.17%; meshing 55 s vs 642 s for 3DGS+preproc |
| P06 | Single-sample injection ablation (values in File B L4); descriptive only |

**Allowed wording.**
- "ForeSplat achieved PSNR = 31.09 dB, SSIM = 0.9711, LPIPS = 0.0365 under the unified 20-sequence protocol; +1.51 dB compared with standard 2DGS (29.58 dB), with reduced training/reconstruction time (60.94%) and meshing time (65.17%)."
- "Comparisons in this study are within-method ablations; internal mechanism questions are best answered by controlled component variation."
- "Benchmarking against external reconstruction pipelines (e.g., COLMAP+MVS, NeRF-family, standard 3DGS/2DGS under the same foreground masks) is left to follow-up work and is required to position the choice of 2DGS as the base representation."
- "Among the tested methods on the frozen P6 evaluation set, RAP-FSAM3 achieved the highest boundary consistency and lowest leakage energy." (only if the table clearly scopes to tested methods)

**Forbidden wording.** "outperforms SOTA"; "state-of-the-art"; "superior"; "best reconstruction"; "significantly improves reconstruction quality"; "best [metric]" in running text unless qualified "among the tested methods on this evaluation set"; "2DGS was the right choice" asserted without external comparison.

**Manuscript placement.** Discussion (positioning paragraph) + a dedicated Section 6 limitation paragraph (B-R2-C02), with a concrete follow-up commitment.

**Literature citation slots (verify before use):**
- `[SLOT]` 2DGS (surfels/2D Gaussian representation), 3DGS.
- `[SLOT]` NeRF-family reconstruction (e.g., Instant-NGP, Zip-NeRF).
- `[SLOT]` COLMAP + MVS / OpenMVS non-learning baselines.
- `[SLOT]` Plant-specific reconstruction (PlantGaussian, Cotton3DGaussians, LCR-GS, IPENS) — also required to satisfy A2-C06 (Discussion must cite prior work).

---

## D5. Downstream phenotyping applicability and 3D-reconstruction trade-offs

**Claim being made.** Foreground-object 2DGS reconstruction supports trait measurement that is **comparable to** manual measurement on the tested set, with the strongest agreement on global architectural traits and the weakest on the boundary-sensitive organ-level trait (leaf width). This agreement is descriptive (no repeated measurements, no Bland-Altman, no inter-rater reliability) and must be worded accordingly. Reconstruction-side evidence for the foreground-object design comes from the injection ablation and the workflow-level quality/time numbers.

**Evidence anchors.** P05 (DESCRIPTIVE), P06 (DESCRIPTIVE), P07 (DESCRIPTIVE), A1-C03 (NOT_RESOLVED).

| Anchor | Value to cite |
|---|---|
| P07 | Plant height R² = 0.988 (MAE 0.98 cm), canopy width R² = 0.988 (MAE 0.86 cm), leaf length R² = 0.974 (MAE 0.51 cm, n=63), leaf width R² = 0.900 (MAE 0.45 cm); n=21 plant-level, n=63 leaf-level |
| P05 | PSNR = 31.09 dB, SSIM = 0.9711, LPIPS = 0.0365; time gains (see D4) |
| P06 | Full-scene outside-mask non-black ratio 0.9908 / leakage 1.2201; input masking PSNR_fg 24.21 → 20.73; foreground-object loss injection → outside 0.0294 / leakage 0.0190; post hoc pruning left outside 0.7509 / leakage 0.7900 (single representative sample) |
| A1-C03 | Leaf-width error source (segmentation boundary vs 2DGS surface dilation vs TSDF smoothing) **not attributed**; point-cloud-based measurement is future work |

**Allowed wording.**
- "Virtual measurements were comparable to manual measurements (R² 0.988, 0.988, 0.974, 0.900) on the 21-plant / 63-leaf set (descriptive, no repeated measurements collected)."
- "Leaf width showed the weakest agreement (R² = 0.900), a boundary-sensitive trait for which the underlying error source (segmentation boundary shifts, 2DGS surface expansion, or TSDF boundary smoothing) remains an open question."
- "This error pattern aligns with the trends reported in recent 3D plant phenotyping studies, in which global architectural traits are usually more stable than local organ-level traits under boundary uncertainty." (claim_audit row 14 — use "aligns with", not "is consistent with")
- "Foreground-object loss injection reduced outside-mask content (non-black ratio 0.9908 → 0.0294) and leakage (1.2201 → 0.0190) on the representative sample; post hoc mask pruning alone did not (outside 0.7509 / leakage 0.7900)." (single-sample, descriptive)

**Forbidden wording.** "strong agreement" (claim_audit row 9); "virtual measurements are accurate/superior"; "statistically validated agreement"; "R² proves measurement accuracy"; "significant improvement"; "post hoc pruning is useless" (P06 — must be "insufficient", not "useless"); claiming leaf-width accuracy is resolved or high-precision (A1-C03).

**Manuscript placement.** Discussion (phenotyping applicability + boundary-sensitive traits) + Limitations (error-source attribution, validation scope).

**Literature citation slots (verify before use):**
- `[SLOT]` 3D plant phenotyping reviews (e.g., Paulus 2019; plant-phenotyping survey literature).
- `[SLOT]` Organ-level vs architectural-trait stability under boundary uncertainty in 3D phenotyping (refs [57,58] in the old manuscript — re-verify).
- `[SLOT]` Leaf-width measurement difficulty; point-cloud-based measurement alternatives.
- `[SLOT]` Agreement statistics (Bland–Altman, Lin's concordance) as the future evaluation protocol.

---

## D6. Design lessons for agricultural VFM adaptation (field guidance, CompAg scope)

**Claim being made.** The study yields transferable, field-facing lessons for adapting vision foundation models to agricultural plant scenes: (a) direct single-prompt VFM segmentation is semantically fragile in potted-plant scenes, so prompt wording must be treated as a first-class experimental variable; (b) per-instance candidate generation decouples "what candidates exist" from "which candidate is selected," making selection failure diagnosable; (c) ranking signals carry morphology-specific biases (e.g., contrast) that warrant scrutiny before being trusted; (d) descriptive-evaluation discipline is required whenever no held-out ground truth exists. Repeated-measurement validation and scale-marker repeatability are stated as future work.

**Evidence anchors.** P03 (PROVEN per-prompt F1), P04 (DESCRIPTIVE ablation), P01/P02 (DESCRIPTIVE), C02/C04 (SUPPORTED/PROVEN).

| Anchor | Value to cite |
|---|---|
| P03 | Per-prompt F1: plant-instance 0.9692, green-region 0.7972, organ-level 0.6471, seedling-oriented 0.0768, background-excluding 0.4051 |
| P04 | Geometry-only multi-prompt selector degraded F1 to 0.8429; semantic gating recovered the stable plant-instance prompt (0.9692); full RAP-FSAM3 w/ GCFC 0.9706 |
| P01/P02 | F1 = 0.9706; boundary/leakage deltas vs single-prompt SAM3 (Boundary-F1 0.4643 vs 0.4051; leakage 0.001739 vs 0.002272) |

**Allowed wording.**
- "Single-prompt VFM segmentation is semantically fragile in potted-plant scenes (per-prompt F1 from 0.0768 to 0.9692 on the frozen evaluation set)."
- "Semantic gating recovered the stable plant-instance prompt (F1 = 0.9692), whereas the geometry-only selector degraded F1 to 0.8429 by admitting semantically inappropriate candidates."
- "Improvements over single-prompt SAM3 were concentrated in boundary consistency (Boundary-F1 0.4643 vs 0.4051) and leakage suppression (leakage energy 0.001739 vs 0.002272); region-overlap deltas (F1 0.9706 vs 0.9692) were small."
- "The method applies to the tested acquisition settings (fixed and complex-background potted-plant sequences)."

**Forbidden wording.** "generalizes" / "generalisation" (no independent held-out evidence — claim_audit row 6); "robust segmentation" as a general adjective; "significantly"; "outperforms"; "state-of-the-art"; "the prompt set generalises to all scenes".

**Manuscript placement.** Discussion (design-lessons paragraph) + Future Work (repeated-measurement study, Bland–Altman, inter-rater reliability, scale-marker repeatability, FFT threshold sensitivity, external baselines, point-cloud leaf width).

**Literature citation slots (verify before use):**
- `[SLOT]` Vision foundation models in agriculture (special-issue scope; SAM-family applications in agronomy).
- `[SLOT]` Prompt-engineering guidance for SAM-family models.
- `[SLOT]` Low-cost / smartphone-based phenotyping and acquisition-throughput estimates (ties to A2-C04 throughput framing).
- `[SLOT]` Evaluation practice for ag-AI without dense annotation.

---

## Cross-cutting Discussion constraints (checklist)

1. Discussion must contain literature citations throughout (A2-C06 RESOLVED) — move literature-style content out of Methods.
2. No `significant(ly)`, `outperform(s)`, `superior`, `SOTA`, `generalize(s)`, `strong agreement`, `robust` (except the R1 policy name).
3. A6/A7 appear only as negative ablation / design decision — never as main Method or as an innovation.
4. q_contrast / R1 section must stay correlational (SUPPORTED, not PROVEN).
5. Reconstruction positioning must not imply external-SOTA superiority (B-R2-C02 NOT_RESOLVED).
