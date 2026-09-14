# Phase 19A — Reviewer-to-Manuscript Action Mapping

**Generated:** 2026-09-14 | **Total comments:** 32 | **Input:** Phase19A_reviewer_issue_matrix.csv, Phase19A_resolution_status.md, Phase19A_reviewer_comments_raw.md

---

## Mapping Table

| Comment ID | Reviewer | Concern (verbatim/paraphrased) | Old-manuscript problem (sec/line ref) | Phase 11–18 evidence addressing it (artifact paths) | Manuscript action for Phase 19B | Status |
|---|---|---|---|---|---|---|
| A1-C01 | Reviewer 1 (Stream A) | Abbreviations (FFT, SAM, 2DGS, TSDF) introduced without full names at first occurrence. | Abstract and Introduction — terms first appear without expansion. | None needed (writing edit). | Add full names at first occurrence in Abstract and Introduction (FFT = fast Fourier transform; SAM = Segment Anything Model; 2DGS = two-dimensional Gaussian Splatting; TSDF = Truncated Signed Distance Function). | RESOLVED |
| A1-C02 | Reviewer 1 (Stream A) | Table 6 comparison unfair because standard 2DGS uses raw frames with background; unclear whether improvement comes from SAM3 alone or FSAM3 components. | Table 6 — baseline 2DGS uses original images; Plant Seg. contribution ambiguous (SAM3 vs FSAM3). | Phase 17B oracle audit (F1=0.987 candidate existed but not selected); Phase 17C ranking ablation R1 (q_contrast out) rescues 3/3 failures with zero regressions. *Phase17B_report.md, Phase17C_report.md, Phase18_evidence_registry.csv* | Revise Table 6 as fair-input ablation (Base, Base+FFT, Base+SAM3, Base+FSAM3, F2DMAS); add component-level segmentation ablation table (SAM3-only vs SAM3+temporal vs full FSAM3); clarify input conditions and foreground module naming. | RESOLVED |
| A1-C03 | Reviewer 1 (Stream A) | Leaf-width R2=0.90 MAPE=9.73% noticeably weaker; error source unclear (segmentation vs 2DGS vs TSDF); point-cloud alternative suggested. | Results (leaf width metrics), Discussion — R2=0.90 MAPE=9.73%; no error attribution. | None from Phase 11–18 (downstream of segmentation scope). | Expand Limitations and Future Work to discuss error sources (segmentation boundary shifts, 2DGS surface expansion, TSDF boundary smoothing); note point-cloud measurement as future direction. Do NOT claim leaf-width accuracy is resolved or high-precision. | NOT_RESOLVED |
| A1-C04 | Reviewer 1 (Stream A) | GPU configuration and training environment (Linux, PyTorch, etc.) not specified. | Materials and Methods — no GPU/hardware/software environment. | None needed (writing edit). | Add new subsection "Computational environment and implementation details" in Methods reporting OS, GPU, Python/PyTorch/CUDA, COLMAP version, and workflow-to-script mapping. | RESOLVED |
| A1-C05 | Reviewer 1 (Stream A) | Author affiliations appear repeated on page 1 (formatting glitch). | Title page (page 1) — affiliation labels repeated. | None needed. | Correct affiliation formatting; each affiliation listed once with superscript labels; check Frontiers submission preview during resubmission. | RESOLVED |
| A2-C01 | Reviewer 2 (Stream A) | R-squared reported with 4 decimal places (0.8999–0.9879); unrealistic precision for this measurement context. | Abstract — R2 with 4-digit precision. | None needed. | Report R-squared with 2 decimal places in Abstract and Table 7 (e.g., 0.90–0.99). | RESOLVED |
| A2-C02 | Reviewer 2 (Stream A) | RMSE absolute values non-informative without reference scale; should add mean values or relative error metric (%). | Abstract, Results, Table 7 — RMSE without context. | None needed. | Add MAPE alongside RMSE; add mean manual measurement values in Table 7 for each trait. | RESOLVED |
| A2-C03 | Reviewer 2 (Stream A) | Second paragraph of Section 2.1 is background/literature content more appropriate for Introduction. | Section 2.1 (second paragraph) — misplaced literature context. | None needed. | Move paragraph from Section 2.1 to Introduction as transition motivating integrated workflow. | RESOLVED |
| A2-C04 | Reviewer 2 (Stream A) | No data on acquisition time per plant or pipeline throughput (plants per unit time). | Materials and Methods — no throughput data. | None needed. | Add typical raw-data acquisition time (1.5–2.0 min per plant) and estimated throughput (30–40 plants/hour manual acquisition) in Methods. | RESOLVED |
| A2-C05 | Reviewer 2 (Stream A) | No frameworks, libraries, or script-to-workflow mapping; no test datasets specified. | Materials and Methods — no software implementation details; GIT repo scripts opaque. | None needed (writing edit). | Add Section 3.8 Implementation Details reporting GPU, COLMAP version, 2DGS params, Python/PyTorch/CUDA; clarify script-to-workflow mapping; note data availability statement. | RESOLVED |
| A2-C06 | Reviewer 2 (Stream A) | Discussion completely lacks references to prior work; Methods contains literature-oriented discussion instead. | Discussion — no citations; Methods — contains literature review. | None needed. | Add citations to Discussion (IPENS, PlantGaussian, Cotton3DGaussians, LCR-GS, etc.); move literature-style discussion from Methods to Discussion; keep only algorithm-identifying citations in Methods. | RESOLVED |
| B-EIC-C01 | EIC (Stream B) | Paper claims "cross-species" in title but reconstruction validation uses only 3 samples; significant gap between framing and evidence. | Title/Abstract/throughout — cross-species claim; reconstruction on 3 samples only. | Phase 12: 21 GT frames from 5 species (F1=0.9835, IoU=0.9676); Phase 14.1: A6/A7 factorial on Easy21 (21 frames, 5 species); phenotype validation covers 21 plants from 10 species. *Phase12 report, Phase14 report, Phase12 P6_全部21帧.csv* | Reframe as "architectural diversity" with explicit limitation in Section 6; keep cross-species at mask/phenotype level; qualify Contribution 4 in Section 1. Do NOT claim cross-species generalization at reconstruction level. | PARTIALLY_RESOLVED |
| B-EIC-C02 | EIC (Stream B) | FSAM3 declines benchmark against ExG/Otsu; a qualitative comparison would strengthen the mask contribution. | Section 4.1 — no quantitative mask metrics (F1, IoU, HD95). | Phase 12: F1=0.9835 IoU=0.9676 Precision=0.9943 Recall=0.9731 on 21 GT frames; component reduction 12.4→4.1 (67%); preservation rate 98.2%. *Phase12 report, Phase12 P6_全部21帧.csv* | Report F1/IoU/Precision/Recall in Section 4.1; add component reduction and preservation rate; add qualitative comparison figure (FSAM3 vs ExG/Otsu on 3–5 representative frames, Figure S1); note pixel-level GT limitation in Section 6. Position FSAM3 as reconstruction-oriented mask prior, not standalone SOTA segmentation. | RESOLVED |
| B-EIC-C03 | EIC (Stream B) | Table 5 four-way combination reported but independent contribution of M1-soft vs M4 not discussed. | Section 4.5, Table 5 — M1-soft and M4 contributions not independently analyzed. | Data exists in Table 5; Phase 18 re-analysis provides interpretation. | Add detailed comparison paragraph in Section 4.5 analyzing independent and combined contributions of M1-soft and M4 on CaoMei2 and XianKeLai1. | RESOLVED |
| B-EIC-C04 | EIC (Stream B) | Pot diameter used as scale reference but no validation of scale recovery accuracy; systematic error propagates to all measurements. | Methods — pot diameter as scale reference without accuracy validation. | None needed (writing edit + limitation acknowledgment). | Report pot caliper accuracy (±0.5 mm); add indirect validation from height R²=0.991; note as limitation in Section 6 (single-point calibration propagates linearly; multi-point calibration recommended for follow-up). | RESOLVED |
| B-R1-C01 | Reviewer 1 Stream B (Plant Phenotyping) | Chinese species names ambiguous; no Latin binomials, family, or growth form; cross-species claim scientifically imprecise. | Throughout — 10 Chinese common names without taxonomic resolution. | Supplementary Table S1 created (placeholder); definitive identification pending botanist consultation. *Response to reviewers* | Add mapping table in Supplementary Table S1 (Chinese name → Latin binomial → family → growth form); add note in Section 6. Definitive taxonomic identification requires botanist consultation (e.g., "KongQueZhuYu" maps to multiple Calathea species). | NOT_RESOLVED |
| B-R1-C02 | Reviewer 1 Stream B (Plant Phenotyping) | Manual landmark placement protocol underspecified: number of operators, repeatability, manual vs automatic virtual landmarks; R² conflates reconstruction and landmark error. | Methods/Results — no operator count, no intra-operator repeatability, no landmark protocol. | None needed (writing edit). | Add measurement protocol details: single operator, digital caliper (±0.5 mm), flexible ruler (±1 mm), 3 leaves per plant; note inter-operator variability not assessed. Do NOT claim high precision without noting protocol limitations. | RESOLVED |
| B-R1-C03 | Reviewer 1 Stream B (Plant Phenotyping) | Only linear dimensions; leaf area and stem diameter (standard in 3D phenotyping) excluded without justification. | Methods/Results — no leaf area or stem diameter. | None needed. | Discuss why leaf area and stem diameter were excluded; note as future work if mesh does not support reliable area computation. | RESOLVED |
| B-R1-C04 | Reviewer 1 Stream B (Plant Phenotyping) | Plant growth stage or developmental age not reported; important for cross-species comparison. | Methods — no growth stage or developmental age. | None needed. | Report growth stage for each species in Methods or Supplementary Table S1. | RESOLVED |
| B-R2-C01 | Reviewer 2 Stream B (3D Vision) | Individual components not individually novel; novelty lies in combination, systematic ablation, and negative evidence for hard view filtering. Paper should be explicit about novel combinations vs novel mechanisms. | Introduction/Methods — novelty framing unclear. | None needed. | Clarify in Introduction which components are novel in combination vs novel mechanisms; cite precedents for each component (masked losses in inpainting, uncertainty-weighted losses in multi-task learning, etc.). Frame as engineering contribution through systematic combination, not individual novelty. | RESOLVED |
| B-R2-C02 | Reviewer 2 Stream B (3D Vision) | No comparison against external methods (MVS, NeRF, 3DGS with same masks); impossible to assess whether 2DGS is the right base representation. | Results — no external baselines. | None — no external baselines added. | Add dedicated limitation paragraph in Section 6 explaining rationale (internal mechanism questions best answered by within-method ablation); commit to follow-up study with external baselines (COLMAP+MVS, Instant-NGP, standard 3DGS). Do NOT claim "outperforms SOTA" or "best reconstruction." | NOT_RESOLVED |
| B-R2-C03 | Reviewer 2 Stream B (3D Vision) | Mesh comparison reports structural metrics but never evaluates which variant produces better phenotype measurements; missed connection to downstream utility. | Results (mesh evaluation) — no link to phenotype measurement. | None needed (analysis of existing data). | Connect mesh variant evaluation to phenotype measurement; add leaf width comparison across mesh variants (Standard TSDF vs Smaller truncation vs Post-boundary) on 2–3 samples. | RESOLVED |
| B-R2-C04 | Reviewer 2 Stream B (3D Vision) | M4 scoring weights (α, β, γ, δ, η) and pruning threshold τ_g listed but values never reported. | Methods (Section 3.5) — hyperparameters under-defined. | None needed (writing edit). | Report all hyperparameter values in Section 3.8: α=0.35, β=0.25, γ=0.20, δ=0.10, η=0.10, τ_g=0.30, τ_track=0.50, λ_mask=0.1, λ_bg=0.05. | RESOLVED |
| B-R3-C01 | Reviewer 3 Stream B (Ag AI) | 18.03% Gaussian reduction presented without absolute GPU memory (MB) or rendering FPS difference; not concrete for practitioners. | Results — relative reduction without absolute deployment metrics. | None needed. | Report absolute GPU memory savings (MB) and rendering FPS difference in Section 3.8 or Results. | RESOLVED |
| B-R3-C02 | Reviewer 3 Stream B (Ag AI) | Missing reproducibility details: smartphone model, turntable speed, COLMAP version, SAM3 checkpoint, training hyperparameters, code availability. | Methods — no hardware/software/code specifics. | None needed (writing edit). | Add Section 3.8 with all reproducibility details (smartphone specs, turntable speed, COLMAP version, SAM3 version, learning rates, iterations, loss weights); note code URL to be provided upon publication. | RESOLVED |
| B-R3-C03 | Reviewer 3 Stream B (Ag AI) | Five prompts (P1–P5) listed without rationale for selection; prior work, pilot experiments, or domain expertise not stated. | Methods (Section 3.2.2) — no prompt selection rationale. | None needed. | Add 1–2 sentences explaining prompt selection rationale (span from broad green plant to organ-specific leaves and stems to exclusion-focused "entire plant excluding pot"). | RESOLVED |
| B-R3-C04 | Reviewer 3 Stream B (Ag AI) | No per-plant processing time (wall clock) or hardware requirements reported; deployment feasibility unclear. | Methods/Results — no throughput estimate. | None needed. | Report ~55 min total per plant on RTX 3090 with breakdown (COLMAP 15 min, FSAM3 8 min, 2DGS 25 min, mesh 5 min, measurement 2 min) in Section 3.8. | RESOLVED |
| B-DA-C01 | Devil's Advocate (Stream B) | Central claim: foreground reconstruction superior to full-scene + post-hoc pruning. E7 (A0 + post-hoc pruning) argued conceptually in Discussion but never tested empirically. Most direct test of core claim missing. | Section 5.1 (Discussion only) — E7 not in Results. | E7 added to Section 4.2 with quantitative results (PSNR_fg=21.34, outside=0.31, leakage=0.28 on KongQueZhuYu); Phase17B ranking failure + Phase17C ranking ablation provide related evidence (oracle F1=0.987; R1 rescues 3/3 failures). *Response to reviewers (Section 4.2 E7 results), Phase17B_report.md, Phase17C_report.md* | Add E7 to Section 4.2 and Table 2 with definition and results; note single-sample limitation in Section 6. Do NOT claim E7 definitively proves capacity diversion; report as consistent with the hypothesis. Do NOT conflate ranking-failure rescue with post-hoc pruning equivalence. | PARTIALLY_RESOLVED |
| B-DA-C02 | Devil's Advocate (Stream B) | A0-A6 ablation on single sample (KongQueZhuYu); without replication on additional sample, cannot rule out sample-specific effects. | Section 4.2 — single-sample ablation. | None — no replication performed. | Acknowledge as limitation in Section 6; commit to follow-up replication on additional sample(s). Do NOT claim generalizable ablation results; state as development-set finding only. | NOT_RESOLVED |
| B-DA-C03 | Devil's Advocate (Stream B) | FSAM3 named method with dedicated subsection but no quantitative mask metrics; "generating masks for 20 samples" is feasibility not quality. | Section 4.1 — no F1/IoU metrics on FSAM3 output. | Phase 12: F1=0.9835 IoU=0.9676 on 21 GT frames. *Phase12 report, Phase12 P6_全部21帧.csv* | Report F1/IoU/Precision/Recall in Section 4.1; add component count reduction and preservation rate; add qualitative comparison figure (Figure S1). | RESOLVED |
| B-DA-C04 | Devil's Advocate (Stream B) | FFT threshold uses per-sample quartile (sample-dependent); fixed threshold on held-out set not evaluated. | Section 3.2.1 — quartile-based threshold as heuristic. | None needed. | Discuss per-sample quartile rationale in Methods; note as limitation in Section 6; commit to threshold sensitivity analysis in follow-up work. Do NOT claim optimal threshold. | NOT_RESOLVED |
| B-DA-C05 | Devil's Advocate (Stream B) | F2DMAS in title/abstract but "Plant-aware 2DGS," "Ours-core," "Ours-full," "FSAM3" in main text; inconsistent naming weakens brand. | Throughout — naming inconsistency. | None needed. | Standardize naming: F2DMAS = complete pipeline; FSAM3 = mask component; Ours-core (A6) and Ours-full (A6+M1-soft+M4) = 2DGS variants. Use F2DMAS consistently in main text. | RESOLVED |

---

## Consolidated Threads

### Thread 1: Dataset and Linguistic Integrity

**Comments:** B-EIC-C01 (cross-species gap), B-R1-C01 (botanical nomenclature), B-R1-C04 (growth stage), A1-C05 (affiliation formatting).

**Manuscript response:** Add Supplementary Table S1 with Chinese name → Latin binomial → family → growth form mapping (pending botanist consultation); report growth stage per species in Methods; correct affiliation formatting; reframe cross-species claim as "architectural diversity" at reconstruction level while retaining cross-species applicability at mask and phenotype levels. Qualify Contribution 4 in Section 1.

### Thread 2: Ablation Completeness and Fair Comparison

**Comments:** A1-C02 (Table 6 fairness/SAM3 vs FSAM3), B-EIC-C03 (M1-soft vs M4 independent contribution), B-DA-C01 (E7 post-hoc pruning, CRITICAL), B-DA-C02 (ablation replication on single sample).

**Manuscript response:** Revise Table 6 as fair-input ablation (5 settings); add component-level segmentation ablation table; add M1-soft vs M4 interpretation paragraph in Section 4.5; add E7 to Table 2 with results and single-sample caveat. Acknowledge single-sample ablation as development-set finding in Section 6; commit to multi-sample replication in follow-up.

### Thread 3: Hardware Reproducibility and Implementation Details

**Comments:** A1-C04 (GPU config), A2-C05 (software details/script mapping), B-R1-C02 (measurement protocol), B-R2-C04 (hyperparameter values), B-R3-C01 (absolute GPU memory/FPS), B-R3-C02 (smartphone, turntable, COLMAP, SAM3, training, code), B-R3-C04 (per-plant processing time).

**Manuscript response:** Add Section 3.8 "Computational environment and implementation details" covering OS, GPU, Python/PyTorch/CUDA, COLMAP version, SAM3 version, smartphone specs, turntable speed, learning rates, iterations, loss weights, hyperparameter values (α, β, γ, δ, η, τ_g, τ_track, λ_mask, λ_bg), absolute GPU memory savings, rendering FPS, per-plant processing time breakdown (~55 min on RTX 3090); add measurement protocol details (single operator, caliper ±0.5 mm, 3 leaves per plant); note code URL to be provided upon publication.

### Thread 4: External Comparison Gap

**Comments:** B-R2-C02 (missing MVS/NeRF/3DGS baselines), B-DA-C01 (E7 post-hoc pruning as partial proxy).

**Manuscript response:** Add dedicated limitation paragraph in Section 6 explaining why within-method ablation was prioritized (internal mechanism questions require controlled component variation); commit to follow-up with COLMAP+MVS, Instant-NGP/Zip-NeRF, and standard 3DGS with same masks. Do NOT claim "outperforms SOTA" or "best reconstruction." E7 provides partial evidence (single sample) that post-hoc pruning underperforms foreground-first training, but is not a substitute for external baseline comparison.

### Thread 5: Statistical Rigor and Reporting Precision

**Comments:** A2-C01 (R² 4-digit precision), A2-C02 (RMSE without context), B-R2-C03 (mesh-to-phenotype connection).

**Manuscript response:** Report R² with 2 decimal places; add MAPE alongside RMSE; add mean manual values in Table 7; connect mesh variant evaluation to phenotype measurements (leaf width across mesh variants on 2–3 samples).

### Thread 6: Structural and Writing Quality

**Comments:** A1-C01 (abbreviations), A2-C03 (Section 2.1 misplaced paragraph), A2-C06 (Discussion without references), B-R2-C01 (novelty framing), B-R3-C03 (prompt rationale), B-DA-C05 (naming inconsistency).

**Manuscript response:** Expand abbreviations at first occurrence; move Section 2.1 paragraph to Introduction; add literature citations to Discussion (IPENS, PlantGaussian, Cotton3DGaussians, LCR-GS); clarify novelty framing (engineering contribution via systematic combination); add prompt selection rationale; standardize naming (F2DMAS = pipeline, FSAM3 = mask, Ours-core/Ours-full = 2DGS variants).

### Thread 7: Methodological Scope Boundaries

**Comments:** A1-C03 (leaf width error source), B-R1-C03 (trait selection), B-EIC-C04 (scale validation).

**Manuscript response:** Expand Limitations to discuss leaf-width error sources (segmentation boundary shifts, 2DGS surface expansion, TSDF boundary smoothing) and note point-cloud measurement as future direction. Discuss why leaf area/stem diameter excluded. Report pot caliper accuracy (±0.5 mm) and indirect scale validation (height R²=0.991); note single-point calibration as limitation.

### Thread 8: Threshold and Heuristic Choices

**Comments:** B-DA-C04 (FFT quartile threshold).

**Manuscript response:** Discuss per-sample quartile rationale in Methods (adapts to per-sequence quality without labeled quality dataset); note as limitation in Section 6; commit to threshold sensitivity analysis in follow-up.

---

## G3 Gate Verification

- **No blank manuscript actions:** All 32 comments have a non-empty manuscript action cell. Confirmed by inspection.
- **Status counts:**
  - RESOLVED: **25**
  - PARTIALLY_RESOLVED: **2**
  - NOT_RESOLVED: **5**
  - Total: **32**
- **NOT_RESOLVED items (must route to Limitations / Future Work; never claim resolved):**

| Comment ID | Issue | Required Limitation Language |
|---|---|---|
| A1-C03 | Leaf-width error source (segmentation vs 2DGS vs TSDF) | Discuss error sources; note point-cloud alternative as future work |
| B-R1-C01 | Botanical nomenclature (Latin binomials) | Placeholder table pending botanist consultation; may not be resolvable before publication |
| B-R2-C02 | External baselines (MVS, NeRF, 3DGS) | Within-method ablation chosen for internal mechanism questions; external comparison deferred to follow-up |
| B-DA-C02 | Ablation replication (single sample) | Development-set finding only; multi-sample replication in follow-up |
| B-DA-C04 | FFT threshold (per-sample quartile vs fixed) | Heuristic choice; threshold sensitivity analysis in follow-up |

These five NOT_RESOLVED items match the list in Phase19A_resolution_status.md (leaf width, external baselines, botanical nomenclature, ablation replication, FFT threshold).

---

WP7A SELF-CHECK: 32 rows — ALL PASS
