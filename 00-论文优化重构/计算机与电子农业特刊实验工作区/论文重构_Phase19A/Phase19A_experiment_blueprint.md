# Phase 19A — Experiments/Results Blueprint (FILE C: WP6B)

> Blueprint for the Experiments/Results chapter, reorganized by scientific QUESTION (not by Phase number).
> Every section ties to `Phase19A_claim_evidence_matrix.csv` allowed_wording and `paper_section`.
> Audit date: 2026-09-14.

---

## Cross-Check Rule (applies to ALL sections)

- Every reported number MUST trace to a CSV listed in the `notes` column of `Phase19A_claim_evidence_matrix.csv` (or the underlying phase evidence CSVs referenced there: `Phase18_evidence_registry.csv`, `Phase18_final_GT10_table.csv`, `Phase18_failure_rescue_table.csv`, `Phase18_ranking_ablation_table.csv`, `Phase18_score_mechanism_table.csv`, `Phase18_A6A7_ablation_table.csv`, `Phase18_final_easy21_table__*.csv`).
- **No hand-filled numbers.** If a number is not computationally verifiable from those CSVs, it must not enter the manuscript (Phase 18 integrity rule).
- Any delta between methods (e.g., F1 0.9706 vs 0.9692) must be described as a descriptive difference within annotation variance, not a significance claim.
- Where the claim table forbids wording, the `allowed_wording` string is the ceiling for that sentence.

---

## Section: Exp rationale (shared preamble)

**Purpose:** State that experiments are organised by question, each answered on the dataset class (frozen P6 / Easy21 / GT10-DEV / 20-sequence / single-sample) that carries the required evidence, at the stated evidence level. Results on the frozen P6 evaluation set are descriptive means over the 21 annotated frames.

---

## Q1 — Ordering: RAP-FSAM3 vs tested VFM baselines on the frozen P6 set

- **Claim rows:** P01, P02.
- **Evidence level:** DESCRIPTIVE (frozen-set means).
- **Dataset:** Frozen P6, 21 annotated frames, 5 samples.
- **Figure mapping:** Fig3 (current VFM comparison) and/or Fig_E1_metric_bars; supplementary Fig_E1_failure_cases.
- **Table mapping:** Table 3 (`tab:vfm-comparison`).
- **Exact metrics:** F1, mIoU, HD95, Boundary-F1, leakage energy (P01); delta-focused Boundary-F1 and leakage-energy differences vs single-prompt SAM3 (P02).
- **Numbers to use (from the claim table):**
  - RAP-FSAM3: F1=0.9706, mIoU=0.9641, HD95=183.42 px, Boundary-F1=0.4643, leakage energy=0.001739.
  - Single-prompt SAM3 comparison: F1 0.9692, mIoU 0.9624, Boundary-F1 0.4051, leakage energy 0.002272.
- **Wording restrictions:**
  - MUST phrase ordering claims as "achieved F1=0.9706 … **among the tested methods on this evaluation set**."
  - MUST NOT say "significantly outperforms" / "statistically significant" (no significance test; deltas within annotation variance).
  - MUST NOT use "outperforming [unqualified]" — replace with the specific metric deltas of P02.

---

## Q2 — Robustness: How sensitive is direct VFM prompting to prompt choice?

- **Claim rows:** P03.
- **Evidence level:** PROVEN (per-prompt F1 report) — but descriptive per-prompt means only.
- **Dataset:** Frozen P6, 21 frames.
- **Figure mapping:** Fig4 (prompt failure); Fig_E2_prompt_P2_vs_P4 board.
- **Table mapping:** Table 4 (`tab:prompt-robustness`).
- **Exact metrics:** per-prompt F1 means.
- **Numbers to use:** plant-instance 0.9692 (most stable); green-region 0.7972 (green-background inclusion); organ-level 0.6471 (under-recall); seedling-oriented 0.0768 (near-empty on mature plants); background-excluding 0.4051 (over-conservative).
- **Wording restrictions:**
  - MUST conclude with the allowed sentence: "Single-prompt VFM segmentation is semantically fragile in potted-plant scenes."
  - MUST NOT say the prompt set is robust or that single-prompt VFM generalises to all scenes.
  - Descriptive reporting per prompt only.

---

## Q3 — Mechanism: Which RAP-FSAM3 modules matter (incl. ranking)?

- **Claim rows:** P04 + MERGE the Phase17C ranking-failure row (C03 mechanism) as the diagnostic framing.
- **Evidence level:** DESCRIPTIVE (ablation table); C03 SUPPORTED (correlational — must stay correlational).
- **Dataset:** Frozen P6 for the ablation; DEV10 for the ranking-failure diagnostic.
- **Figure mapping:** Fig2 (mechanism) or Fig_E3/E3v2 panels; Fig B (ranking illustration, for C03).
- **Table mapping:** Table 5 (`tab:rap-ablation`).
- **Exact metrics:** F1 per ablation row.
- **Rows / numbers to use:**
  - Geometry-only multi-prompt selector: F1=0.8429 — MUST be framed as the ranking-failure diagnostic row (semantically inappropriate candidates), NOT as a general failure of all geometry features.
  - Semantic gating recovered the stable plant-instance prompt (0.9692).
  - SB-FBPR improved boundary/leakage.
  - Full RAP-FSAM3 w/ GCFC: 0.9706.
- **Wording restrictions:**
  - MUST NOT say the ablation "proves SG-MPS necessary in all settings" or that SB-FBPR/GCFC each give statistically significant gain.
  - MUST describe the geometry-only row as a diagnostic.
  - For C03: allowed wording is "q_contrast can favor small, high-contrast distractor masks over larger whole-plant candidates (failure-frame Spearman rho = −0.7187, p = 0.0038)". MUST NOT escalate to "proven global sole cause."

---

## Q4 — Ranking: What caused the catastrophic failures and does R1 fix them? (NEW subsection)

- **Claim rows:** C01, C02, C03, C04. (NOT C05 — C05 is FORBIDDEN except as the Discussion-only selection rationale.)
- **Evidence level:** C01/C02 PROVEN; C03/C04 SUPPORTED.
- **Dataset:** GT10 (DEV10) — 10 annotated frames (BaiZhang 0033-0037, DouBanLv2 0037-0041); N=10, 2 sequences, NOT held-out. MUST state this plainly.
- **Figure mapping:** Phase18 Figure_A (failure mechanism, 5-panel rows for DouBanLv2 0039/0040/0041) + Figure_B (ranking illustration, frame 0039).
- **Table mapping:** Phase18_failure_rescue_table.csv (failure-rescue table); Phase18_final_GT10_table.csv; Phase18_ranking_ablation_table.csv; Phase17C policy comparison.
- **Exact metrics:** raw oracle F1 vs selected F1; R0 vs R1 mean F1; Easy21 regression counts.
- **Numbers to use:**
  - C01: baseline candidate generator produced high-quality whole-plant candidate masks even when final F1=0 — raw oracle F1 = 0.9867–0.9882 on failure frames.
  - C02: failures were caused by candidate-ranking failure (selected distractor), diagnosis class C3_RANKING_FAILURE, NOT generation failure.
  - C03: contrast mechanism correlational evidence (Spearman rho = −0.7187, p=0.0038 on failure frames).
  - C04: R1 rescued 0039/0040/0041 to F1 >= 0.95; zero material regressions on Easy21 (R1 zero-inference rerank: 0 selection changes, 0 material regressions).
  - R0 vs R1: R0 mean F1=0.6767 (3 catastrophic, oracle regret 0.2961) → R1/R2/R3 all 0.9728, 0 catastrophic, regret 0.0 (Phase17C policy comparison; may appear descriptively).
- **Wording restrictions:**
  - MUST frame the R1 result as "rescue on the identified DEV set," NOT "general improvements."
  - MUST NOT claim "R1 significantly improves general segmentation performance" or "R1 proves superior globally" (C05 FORBIDDEN). The allowed C04 sentence is exactly: "Removing q_contrast from the primary ranking (R1) eliminated all identified DEV-set ranking failures (0039/0040/0041 rescued to F1 ≥ 0.95) with zero material regressions on Easy21."
  - C01 allowed wording: "Baseline candidate generation produced high-quality whole-plant candidate masks (raw oracle F1 = 0.9867–0.9882) even on frames where final F1 = 0." MUST NOT say "P6/candidate generation failed; no correct candidate existed."
  - C02 allowed wording: "Catastrophic failures (DouBanLv2 0039-0041) were caused by candidate-ranking failure (selected distractor), not candidate generation."
  - If the P6_raw_baseline run metric (f1_mean=0.983498) is cited anywhere, it must be labelled as the frozen Phase 12 run under production-valid config (default prompt_list=P2), never as P6-prompt-set output (`Phase19A_formula_audit.md` §4).

---

## Q5 — Reconstruction: Does foreground-object 2DGS + injection outperform the alternatives?

- **Claim rows:** P05, P06.
- **Evidence level:** DESCRIPTIVE (workflow-level means / single-sample representative ablation).
- **Dataset:** 20 RGB sequences for P05; representative broad-leaf sample (1 sequence, Partial GT) for P06.
- **Figure mapping:** Fig8 (reconstruction) + E4 injection panels for P05; Fig7 (injection-ablation) for P06.
- **Table mapping:** Table 7 (`tab:reconstruction`) for P05; Table 6 (`tab:injection-ablation`) for P06.
- **Exact metrics:** PSNR, SSIM, LPIPS (P05); outside-mask non-black ratio and leakage, PSNR_fg (P06).
- **Numbers to use:**
  - P05: ForeSplat PSNR=31.09 dB, SSIM=0.9711, LPIPS=0.0365; +1.51 dB vs standard 2DGS (29.58); reduced train/recon time 60.94% and meshing time 65.17% vs standard 2DGS; meshing 55 s vs 642 s for 3DGS+preproc.
  - P06: full-scene 2DGS outside-mask non-black ratio 0.9908 / leakage 1.2201; input masking dropped PSNR_fg 24.21→20.73; foreground-object loss injection → outside 0.0294 / leakage 0.0190; post hoc mask pruning left outside 0.7509 / leakage 0.7900.
- **Fair-input note (reviewer A1-C02):** Table 6/7 experiments must be reported with the fair-input comparison made explicit — input masking and post hoc pruning receive the same foreground-information as the injection baseline where applicable; any difference in input conditions to the optimiser must be stated so the ablation is not unfairly biased.
- **Wording restrictions:**
  - MUST NOT say "ForeSplat significantly improves reconstruction quality" or "statistically significant PSNR/SSIM gain" (descriptive workflow-level means; application-level comparison, not a pure loss-function ablation).
  - MUST NOT say "state-of-the-art reconstruction."
  - For P06 MUST NOT claim "input masking or post hoc pruning is sufficient" or imply a significance test; post hoc pruning must be described as insufficient, not "useless."

---

## Q6 — Phenotyping: How well do virtual traits agree with manual?

- **Claim rows:** P07.
- **Evidence level:** DESCRIPTIVE (agreement report).
- **Dataset:** 21 plants / 63 leaves (n=21 plant-level, n=63 leaf-level).
- **Figure mapping:** Fig9 (phenotyping) + E5 panels (Fig_E5_error_distribution, Fig_E5_trait_error_profile).
- **Table mapping:** Table 8 (`tab:phenotyping`).
- **Exact metrics:** R², MAE per trait.
- **Numbers to use:** plant height R2=0.988 (MAE 0.98 cm); canopy width R2=0.988 (MAE 0.86 cm); leaf length R2=0.974 (MAE 0.51 cm, n=63); leaf width R2=0.900 (MAE 0.45 cm).
- **Wording restrictions:**
  - MUST use "comparable to" — e.g., "Virtual measurements were comparable to manual measurements: …"
  - MUST NOT use "strong agreement," "virtual measurements are accurate/superior," "statistically validated agreement," or "R2 proves measurement accuracy."
  - No repeated measurements, no Bland-Altman claims — state the descriptive nature.

---

## Q7 — Negative ablation: What about A6/A7?

- **Claim rows:** P08 (consolidating C06, C07, C08 — all NOT_SUPPORTED/DEMOTED).
- **Evidence level:** NOT_SUPPORTED (negative/limitation result).
- **Dataset:** Easy21 + GT10 (21+10 frames, 7 samples).
- **Figure mapping:** Phase18 Figure_D (A6/A7 ablation bars).
- **Table mapping:** Phase18_A6A7_ablation_table.csv + Phase18_final_easy21_table__1a_historical_ablation.csv.
- **Exact metrics:** E_A6, E_A7, ΔF1 on regression frames, propagation/selection counts.
- **Numbers to use:**
  - A6: E_A6 ≈ −0.003; one material regression ΔF1 = −0.058 on ChangShouHua2_0075; on GT10 4/4 acted BaiZhang frames regress (mean ΔF1=−0.007).
  - A7: E_A7 ≈ 0 under corrected Phase 14.1 exposure (13 propagated / 9 selected of 21 frames on Easy21); one marginal GT10 improvement (ΔF1 = +0.006).
  - Both disabled by default (`--use_cross_view_consensus False`, `--use_memory_propagation False`).
- **Wording restrictions:**
  - THIS SECTION IS AN ABLATION/LIMITATION ONLY. INSTEAD of claiming benefit, show they were evaluated and not retained, with the allowed wording: "A6 and A7 **did not provide consistent measurable benefit under the evaluated settings**."
  - MUST NOT say A6/A7 "improve segmentation," "consistently benefit," or are "part of the production default pipeline."
  - A7 claim MUST cite the Phase 14.1 corrected exposure, NOT the invalid Phase 14 zero-propagation observation.
  - MUST NOT add A6/A7 stage language to the Methods chapter (G11 PASS note) — only here as negative result.

---

## Results-Chapter Table/Figure Summary

| Q | Claim | Table | Figure | Key numbers (allowed) |
|---|---|---|---|---|
| Q1 | P01, P02 | Table 3 | Fig3/E1 | F1=0.9706, Boundary-F1 0.4643, leak 0.001739 |
| Q2 | P03 | Table 4 | Fig4/E2 | 0.9692 / 0.7972 / 0.6471 / 0.0768 / 0.4051 |
| Q3 | P04 (+C03) | Table 5 | Fig2/E3, Fig_B | geometry-only 0.8429; full 0.9706 |
| Q4 | C01-C04 | failure_rescue, Fig_A/B | GT10 | oracle 0.9867–0.9882; R1 rescue ≥0.95; 0 Easy21 regressions |
| Q5 | P05, P06 | Tables 7, 6 | Fig8/E4, Fig7 | PSNR 31.09; injection outside 0.0294 |
| Q6 | P07 | Table 8 | Fig9/E5 | R² 0.988/0.988/0.974/0.900 |
| Q7 | P08 (C06-C08) | A6A7_ablation | Figure_D | E_A6≈−0.003; E_A7≈0; defaults off |

Note the fair-input caveat for Q5 (reviewer A1-C02) and the not-held-out caveat for Q4 (DEV10 descriptive).

---

## Cross-Check Section Template (to include in experiments chapter preface)

> All numbers in Sections Q1–Q7 are sourced from the Phase 18 evidence CSVs listed in the claim-evidence matrix `notes` column (`Phase18_final_GT10_table.csv`, `Phase18_failure_rescue_table.csv`, `Phase18_ranking_ablation_table.csv`, `Phase18_score_mechanism_table.csv`, `Phase18_A6A7_ablation_table.csv`, `Phase18_final_easy21_table__*.csv`, and underlying phase audits). No number is hand-filled; every figure/table trace is `claim → CSV → value`. Significance markers are absent by design; all P-row values are descriptive means on the evaluated set.

---

*FILE C of WP6B. Date: 2026-09-14.*