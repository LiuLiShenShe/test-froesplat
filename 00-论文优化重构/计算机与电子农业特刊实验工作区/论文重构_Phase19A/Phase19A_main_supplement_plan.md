# Phase 19A — Main-Text vs Supplementary Split Plan (WP7C)

**Date:** 2026-09-14 | **Target venue:** ~10-page CAS-SC article (Computers and Electronics in Agriculture special issue)
**Source of truth:** `Phase19A_experiment_asset_registry.csv` (21 assets) + `Phase19A_claim_evidence_matrix.csv` (C01–C08, P01–P08) + `Phase19A_experiment_blueprint.md` (Q1–Q7) + `Phase19A_table_plan.md` (Tables 1–8) + `Phase19A_figure_plan.md` (Figs 1–6) + `Phase19A_method_blueprint.md` + `Phase19A_discussion_blueprint.md` + `Phase19A_limitations_blueprint.md` + `Phase19A_abstract_blueprint.md` + `Phase19A_new_storyline.md`.
**Tail-2000 note:** `Phase19A_tail_2000.py_input` artifacts — searched Phase19A dir (25 files, no `*tail*`/`*2000*` match) and the wider workspace (`hybrid_review_v1`, `01-算法模块修改与验证`); no file matching `*tail_2000*` exists on disk. The split is therefore constrained only by the registry + claim-evidence matrix + blueprints listed above.

---

## 1. MAIN TEXT — concise inventory (each item load-bearing for a C/P claim or Q question)

### 1.0 Page-budget principle

Main text carries **one table row / one figure panel per claim**, never the full per-frame/per-candidate CSVs. Full CSVs live in Supplement S1–S2. Every number in main text must trace `claim → CSV → value` (Phase 18 integrity rule); no hand-filled numbers; no significance stars (all P-rows DESCRIPTIVE except P03 PROVEN-per-prompt-report, still descriptive means).

### 1.1 Introduction (6 paragraphs, per `Phase19A_introduction_blueprint.md`)

| Para | Content | Evidence / claim anchor | Asset |
|---|---|---|---|
| §1.1 | Ag need: low-cost 3D phenotyping; no results | motivation only, no claim | — |
| §1.2 | Dense potted-plant difficulty | motivation only | — |
| §1.3 | VFM prompt fragility (F1 0.0768–0.9692) | P03 (Table 4) | P12-A02 (frozen-P6 per-prompt means) |
| §1.4 | Method insight: generate-then-rank; q_contrast caution | C03 SUPPORTED, correlational only | P18-A15 (score_mechanism table, summary only) |
| §1.5 | Pipeline overview RAP-FSAM3 → R1 → ForeSplat → phenotyping; ranking-failure lesson; A6/A7 one-sentence negative flag | C01/C02 PROVEN; P08 negative flag | P18-A18 (architecture decision); P17B-A09/A10 (oracle); P18-A16 Fig A (referenced, shown in Results) |
| §1.6 | 4 contributions (segmentation / ForeSplat / ranking rescue / phenotyping) | P01, P05, C04, P07 with evidence levels | P18-A15/A16/A17 |

Fig 1 (overall pipeline, KEEP+UPDATE from `Fig1.png` + R1 annotation) sits here. A6/A7 labelled "optional (disabled by default)" in the diagram only — no stage description (G11).

### 1.2 Methods — production pipeline ONLY (per `Phase19A_method_blueprint.md` §§1–10)

| Subsection | Content | Asset / claim |
|---|---|---|
| M1 Workflow overview | 8-module ordered list, no A6/A7; R1 named as production selector | P18-A18 (C04/C06/C07/C08 decisions) |
| M2 Datasets & protocol | 5 datasets with roles: frozen P6 (21 frames, primary); 20-sequence workflow; GT10/DEV10 (10 frames, NOT held-out, descriptive); TEST32 (unlabeled, NO accuracy); Easy21 (21 frames, ablation/safety). Frozen-P6 naming guard (never P6-prompt-set) | P16-A06 (GT10 manifest, TRUE); registry P16-A07 cited only to state TEST32 has NO GT |
| M3 FFT filtering | Q_FFT + per-sequence first-quartile rule | method audit KEEP |
| M4 Candidate generation | per-instance mode; default `prompt_list=P2`; τ_mask=0.5, τ_conf=0.3, A_min=0.01/A_max=0.80 | formula audit §2.5 |
| M5 Feature scoring | q_area/q_comp/q_edge/q_temp/q_contrast/q_sam (+q_leak/q_side w=0); weights, denom D=5.5 frozen; B definition | formula audit §2.1–2.4 |
| M6 R1 ranking (NEW, prominent) | `F = B − q_contrast/5.5 − P`; P_v=0.25/P_t=0.15 subtracted; argmax(F, tie=0.0); old softmax SG-MPS described here as **superseded baseline** with C02/C03→C04 rationale; S=0 | P18-A18; C02/C03/C04 |
| M7 Foreground-object 2DGS | L_rgb-fg, L_mask (λ=0.08), L_bg (λ=0.02), χ(t) warm-up; q_i∈[0.6,1.0] rgb_only; foreground-track init (NOT A6 — terminology guard) | algorithm delta KEEP |
| M8 M2M3-GO + pruning | mask-guided pruning hyperparams; M2M3-GO S^m2g/S^g2m/B^∂/U_j; Floor40; refinement-layer framing, modest effect | algorithm delta KEEP |
| M9 Mesh/TSDF/scale/phenotyping | TSDF fusion + Marching Cubes; pot-diameter scale; 4 traits (n=21/n=63) | P07 |
| M10 Metrics | Segmentation (F1/mIoU/HD95/Boundary-F1/leakage); reconstruction (PSNR/SSIM/LPIPS); phenotyping (MAE/RMSE/MAPE/R²); descriptive-only, no stars | claim matrix P01/P03/P05/P07 |

Table 1 (notation, 28 symbols, NEW) + Table 2 (dataset protocol, KEEP+UPDATE) sit in Methods. **A6/A7 stage language is FORBIDDEN here** (G11 PASS).

### 1.3 Results Q1 — VFM comparison (frozen P6, 21 frames)

- **Claims:** P01 (F1=0.9706, mIoU=0.9641, HD95=183.42, Boundary-F1=0.4643, leak=0.001739), P02 (Boundary-F1 0.4643 vs 0.4051; leak 0.001739 vs 0.002272; F1/mIoU deltas small, within annotation variance).
- **Table:** Table 3 (`tab:vfm-comparison`). **Figure:** Fig 3-panel of figure-plan Fig 2 set / E1 metric bars (existing `Fig3`).
- **Assets:** P12-A02 (means; full 21-frame CSV → S1), P12-A01 (historical baseline anchor f1_mean=0.983498, labelled exactly as "frozen Phase 12 run, production-valid config, default prompt_list=P2" — never P6-prompt-set output), P18-A15 (final_easy21_1a historical row), P18-A17 (wording ceiling).
- **Guards:** "among the tested methods on this evaluation set"; NO "significantly/outperforming/SOTA/generalizes".

### 1.4 Results Q2 — Prompt robustness

- **Claim:** P03 (plant-instance 0.9692 most stable; green 0.7972; organ 0.6471; seedling 0.0768; bg-excl 0.4051). Closing sentence = allowed wording verbatim.
- **Table:** Table 4. **Figure:** figure-plan Fig 2 (existing `Fig4.png` + optional E2 P2-vs-P4 board).
- **Assets:** P12-A02 (per-prompt means; full table → S1).

### 1.5 Results Q3 — Module ablation incl. geometry-only diagnostic

- **Claims:** P04 (geometry-only 0.8429 = ranking-failure diagnostic row; semantic gating recovers 0.9692; SB-FBPR boundary/leakage; full 0.9706) + C03 mechanism framing (rho=−0.7187, p=0.0038, correlational).
- **Table:** Table 5 (module rows + ranking-policy sub-table R0–R3: 0.6767/0.9728/0.9728/0.9728, catastrophic 3/0/0/0). **Figure:** Fig2 mechanism / E3 panels + Figure_B inset (frame-0039 score bars).
- **Assets:** P18-A15 (A6A7/ranking tables — module rows), P18-A16 (Figure_B), P18-A17 (C03 SUPPORTED ceiling).
- **Guards:** geometry-only row framed as diagnostic, NOT general geometry failure; NO "proves SG-MPS necessary in all settings"; C03 stays correlational.

### 1.6 Results Q4 — Ranking-failure + R1 rescue [NEW] (GT10/DEV10, N=10, 2 seq, NOT held-out — stated plainly)

- **Claims:** C01 PROVEN (oracle 0.9867/0.9882/0.9865 despite final 0), C02 PROVEN (C3_RANKING_FAILURE, selected distractor), C03 SUPPORTED (contrast mechanism), C04 SUPPORTED (R1 rescues 0039/0040/0041 → F1≥0.95; 0 Easy21 regressions). C05 FORBIDDEN — appears only as Discussion selection-rationale sentence.
- **Tables:** Table 7 NEW (Part A: 3 failure frames oracle vs R0 vs R1; Part B: R0–R3 policy comparison + churn + Easy21 regressions). **Figures:** NEW composite Fig 3 (Phase18 Figure_A 3×5 panels + Figure_B inset) + NEW Fig 4 (Figure_C violin panel (a) + rescue bars panel (b) built from `Phase18_failure_rescue_table.csv`).
- **Assets:** P16-A06 (GT10 DEV evidence), P17B-A09 (oracle audit), P17B-A10 (diagnosis), P17C-A11 (R0 0.6767/regret 0.2961 → R1/R2/R3 0.9728/regret 0.0; decision order, NOT highest-mean), P17C-A14 (R1 Easy21 0 changes/0 regressions), P17C-A13 (TEST32 churn 6.2% = 2/32, threshold <30%; mean mask IoU 0.9375; 0 empty — rationale input), P18-A15 (failure_rescue + final_GT10 + ranking_ablation tables), P18-A16 (Figures A/B/C), P18-A17 (C01–C04 allowed/forbidden wording).
- **Guards:** "rescue on the identified DEV set", never "general improvement"; C05 sentence banned from Results; R1 rationale (rescue→non-regress→regret→churn→simplicity) stated.

### 1.7 Results Q5 — Reconstruction / injection (20-sequence workflow + 1-sample ablation)

- **Claims:** P05 (PSNR=31.09, SSIM=0.9711, LPIPS=0.0365; +1.51 dB vs 29.58; train/recon −60.94%, meshing −65.17%; 55 s vs 642 s 3DGS+preproc) + P06 (full-scene outside 0.9908/leak 1.2201; masking PSNR_fg 24.21→20.73; injection → 0.0294/0.0190; pruning leaves 0.7509/0.7900 = insufficient, single KongQueZhuYu sample).
- **Tables:** Table 7-workflow (plan Table 7→ renumbered per manuscript; source `tab:reconstruction`) + Table 6 (injection, with mandatory fair-input footnote: A0 raw RGB + background vs foreground variants masked RGB). **Figure:** figure-plan Fig 5 (Fig5/6/7 composite + optional E4 metrics chart).
- **Assets:** P18-A15 (paper tables), P12-A02 context only; P18-A17 wording ceiling.
- **Guards:** NO "significantly/SOTA"; P06 "insufficient" never "useless"; single-sample scope stated.

### 1.8 Results Q6 — Phenotyping (21 plants / 63 leaves)

- **Claim:** P07 (height R²=0.988/MAE 0.98; canopy 0.988/0.86; leaf-len 0.974/0.51 n=63; leaf-wid 0.900/0.45; "comparable to"; no repeats, no Bland-Altman).
- **Table:** Table 8 (R² to 2 decimals per A2-C01; +MAPE + mean-manual per A2-C02). **Figure:** figure-plan Fig 6 (Fig9 scatters + Fig8 reconstruction/mesh; optional E5 error panels).
- **Assets:** P18-A15, P18-A17.
- **Guards:** "comparable to" (never "strong agreement/accurate/superior/validated"); leaf-width weakest → Discussion/Limitations error-source-open.

### 1.9 Results Q7 — A6/A7 negative ablation (reviewer-forced negative result; LOGIC stays in main)

- **Claim:** P08 consolidating C06 (E_A6≈−0.003; ΔF1=−0.058 ChangShouHua2_0075; GT10 4/4 acted regress mean −0.007), C07 (E_A7≈0 under Phase14.1 corrected exposure 13 propagated/9 selected of 21; marginal GT10 +0.006), C08 FALSE (defaults OFF: `--use_cross_view_consensus False`, `--use_memory_propagation False`; optional branches, not production).
- **Table:** A6A7_ablation rows inside Table 5 / standalone A6A7 table (P18-A15). **Figure:** Phase18 Figure_D bars (descriptive).
- **Assets (summary in main):** P14-A04 (ABLATION_ONLY: V00 0.983498 / V10 0.9807 / V01+V11 superseded note), P14.1-A05 (ABLATION_ONLY: corrected lifecycle, E_A7≈0), P18-A15, P18-A16 (Figure_D), P18-A17, P18-A18 (DEMOTE/DEFAULT-OFF decisions).
- **Guards:** negative framing only ("did not provide consistent measurable benefit under the evaluated settings"); A7 MUST cite Phase14.1 corrected exposure, never Phase14 zero-propagation; NO Method-stage language. **FULL per-variant diagnostics → S2.**

### 1.10 Discussion (D1–D6, literature slots, per `Phase19A_discussion_blueprint.md`)

D1 ranking-matters (C01/C02/C04) · D2 q_contrast correlational + R1 decision order (C03/C04; C05 rationale sentence lives HERE, not Results) · D3 A6/A7 design decision, not degradation (C06–C08/P08; no B-DA-C01 conflation of rescue with E7 pruning) · D4 honest SOTA positioning, within-method ablation only (P05/P06; B-R2-C02 → follow-up commitment) · D5 phenotyping applicability + leaf-width open error (P07/A1-C03; "aligns with" not "consistent with") · D6 ag-VFM design lessons (P03/P04/P01/P02/C02/C04). Prior-work citations throughout (A2-C06); literature moved out of Methods (A2-C03). Global bans: significant(ly)/outperform/superior/SOTA/generalize/strong agreement/robust (except "R1 robust instance ranking" name).

### 1.11 Limitations (L1–L7, per `Phase19A_limitations_blueprint.md`)

L1 nested subsets, GT10 not held-out, TEST32 unlabeled, 10/58 GT · L2 R1 no held-out GT validation (C05) · L3 A6/A7 scope-limited (C06–C08/P08; no Method presence) · L4 reconstruction/phenotyping scope (single-sample A0–A6 + injection; 3-sample reconstruction, no cross-species recon claim; no external baselines; single-operator/no-repeat/no-Bland-Altman; single-point scale; leaf-width unattributed) + 5 follow-up commitments · L5 FFT quartile heuristic · L6 descriptive-only statistics · L7 botanical nomenclature provisional (Supp Table S1). Every NOT_RESOLVED comment maps to an L-paragraph (G12).

### 1.12 Conclusion

Recap: direct single-prompt VFM prompting is semantically fragile in potted-plant scenes; even with a high-quality whole-plant candidate present, a contrast-biased ranking collapsed final F1 to 0; R1 robust ranking rescued all identified failures (F1≥0.95, 0 regressions); RAP-FSAM3 reached F1=0.9706 on the frozen P6 set; ForeSplat achieved PSNR 31.09 dB; mesh traits comparable to manual (R² to 2 decimals); A6/A7 disabled by default. Scope sentence ("within the evaluated acquisition settings") + future-work pointer (external reconstruction baselines, held-out R1 validation, repeated-measurement phenotyping) + standard descriptive-statistics tail. No new numbers beyond Results; all claim_audit bans apply.

---

## 2. SUPPLEMENTARY MATERIAL

### S1 — Full per-frame / per-candidate score tables (diagnostic appendices, NO new claims)

**What goes in:**
- S1.1 `P6_全部21帧.csv` full per-frame table (precision/recall/F1/IoU per frame; means match P01/P03 main-text values). Readable appendix version of the frozen-P6 evidence.
- S1.2 `Phase17C_ranking_benchmark.csv` full 182-candidate table (157 DEV10 + 25 Easy21; q_* features, candidate F1/IoU, baseline-selected + oracle flags; header + 182 rows verified on disk).
- S1.3 `Phase17C_easy21_safety.csv` full per-frame R0–R3 rerank rows (21 frames × 4 policies; all ΔF1=0.0, 0 changes — the row-level backing of the main-text "0 regressions" sentence).
- S1.4 `Phase18_final_GT10_table.csv` + `Phase18_failure_rescue_table.csv` + `Phase18_ranking_ablation_table.csv` + `Phase18_score_mechanism_table.csv` full reproductions (the 2/3/4/6-row tables behind main Tables 5/7).
- S1.5 `Phase18_final_easy21_table__1a_historical_ablation.csv` (4 rows) + `__1b_ranking_policy.csv` (R0–R3 Easy21 rows) full reproductions.

**Feeding asset_ids:** P12-A02 (TRUE; means→main, full table→S1.1) · P18-A15 (TRUE; summarized tables→main, full CSVs→S1.4/S1.5) · P17C-A14 (TRUE; "0 regressions"→main Q4, row-level CSV→S1.3).
**Diagnostic-only archive (NO-CLAIM guard):** P17C-A12 (DEVELOPMENT_ONLY ranking benchmark) — the raw 182-row CSV is archived as S1.2 for reproducibility ONLY; it is not a paper table and supports no C/P claim (registry paper_section = "None (diagnostic benchmark)").
**Wording guards:** header note on each S1 table: "Diagnostic appendix — descriptive values on the stated evaluation set; no significance test; N=10 GT10 comparisons are descriptive only." C05 FORBIDDEN sentence must not appear in S1 captions.

### S2 — Complete A6/A7 diagnostics (negative-result home for full detail)

**What goes in:**
- S2.1 Phase14 factorial V00/V10/V01/V11 per-variant F1 table (with explicit supersede note: Phase14 V01/V11 numbers superseded by Phase14.1 corrected runs) + E_A6=−0.0028 derivation + ChangShouHua2_0075 ΔF1=−0.058 frame row.
- S2.2 Phase14.1 corrected runs (cuda_oom fix; session lifecycle; V01-R1 13 propagated/13 scored/9 selected of 21; E_A7≈0) + 13-new-test list (84/84 total with 71 historical).
- S2.3 Phase18 Figure_D bars (full-size) + `Phase18_A6A7_ablation_table.csv` full 3-row reproduction (A6 / A7-corrected / A6+A7, I_AB=−0.0001).
- S2.4 TEST32 selection-churn backing for A6/A7 context IF any A6/A7 churn rows exist (else omit; do not invent).
- S2.5 `Phase18_final_easy21_table__1a_historical_ablation.csv` A6/A7 rows with per-sample ΔF1 values.

**Feeding asset_ids:** P14-A04 (ABLATION_ONLY → summary in main Q7, full diagnostics in S2.1) · P14.1-A05 (ABLATION_ONLY → summary in main Q7, full corrected runs in S2.2).
**Wording guards:** every S2 table/figure caption carries the negative-result ceiling: "did not provide consistent measurable benefit under the evaluated settings"; A7 captions MUST cite "Phase14.1 corrected exposure" and MUST NOT reference the invalid Phase14 zero-propagation observation; defaults-OFF (`--use_cross_view_consensus False`, `--use_memory_propagation False`) restated; G11 restated ("A6/A7 are not part of the production Methods").

### S3 — TEST32 behavioral audit (unlabeled safety evidence ONLY — explicitly NO-GT)

**What goes in:**
- S3.1 TEST32 manifest summary (32 frames / 8 samples; unlabeled; SHA/metadata pointer to `Phase16_TEST_manifest.csv` + `Phase18_dataset_registry.csv`) — manifest only, no per-frame accuracy.
- S3.2 Selection-churn summary: R1 TEST32 churn 6.2% (2/32, <30% threshold); mean mask IoU 0.9375; 0 empty baseline/R1 (from `Phase17C_test32_summary.json`); DEV10 churn 30% (3/10) vs Easy21 0% (from `Phase17C_selection_churn.csv` + `Phase18_ranking_ablation_table.csv`) shown side-by-side to prevent misreading TEST32 churn as accuracy.
- S3.3 R1 selection-rationale restatement (rescue→non-regress→regret→churn→simplicity) with churn as one input among four — never standalone superiority.

**Feeding asset_ids:** P17C-A13 (TRUE; churn rationale → main Q4, full churn context → S3.2). P16-A07 (FALSE) is **excluded as evidence** — cited in S3.1 ONLY as the manifest pointer with the guard below; it supports zero accuracy claims (registry: "cannot support accuracy claims").
**Wording guards (mandatory header):** "TEST32 is UNLABELED — no ground truth, no accuracy/F1/IoU claims permitted from this section. Churn and empty-selection rates are behavioral-safety signals only." Any sentence of the form "TEST32 confirms accuracy/generalization" is FORBIDDEN.

### S4 — Implementation / reproducibility details

**What goes in:**
- S4.1 Environment versions (Python/CUDA/PyTorch/COLMAP/SAM3-commit snapshot; hardware incl. RTX 3090 per-plant ~55 min per B-R3-C04; memory/FPS absolutes per B-R3-C01 where recorded).
- S4.2 S20 hyperparams frozen set (prompt_list=P2 default; τ_mask=0.5/τ_conf=0.3; A_min=0.01/A_max=0.80; V_min=0.30/O_min=0.20; D=5.5; w_sam=0.5; P_v=0.25/P_t=0.15; ranking_policy=r1; A6/A7 defaults False; score_weights frozen; ScoreRecord fields; `提示词评分.csv` outputs).
- S4.3 COLMAP config (sparse-reconstruction settings; per-sequence poses; FFT-quartile exclusion list per sequence).
- S4.4 150-test suite summary (131 hist + 11 P17C + 8 P18 T1–T8 code-paper consistency: r1 default, A6/A7 False, weights frozen, GT10 0.9728 match, rescue 3/3) — summary table only, not the 150 logs.
- S4.5 Botanical nomenclature Supplementary Table S1 (provisional common→Latin mapping; "pending specialist confirmation at proof stage" — L7).

**Feeding asset_ids:** none directly claim-bearing; reproducibility backing for P18-A18 (architecture defaults) and Methods M3–M9. P18-A19 (DEVELOPMENT_ONLY consistency tests) is **excluded as narrative evidence**; only its summary counts (150/150) may appear in S4.4 with the guard "development-quality gate, not a performance claim."
**Wording guards:** "Development validation — not a performance claim" on S4.4; S4.5 header "provisional pending confirmation" (B-R1-C01).

### S5 — Expanded easy-vs-hard descriptor plots + per-sequence breakdowns

**What goes in:**
- S5.1 Phase18 Figure_C full-size violin (R0 vs R1 vs R2 vs R3 on Easy21 + DEV10) with N-labels (21 + 10) and "descriptive distribution, no inferential test" caption.
- S5.2 Per-sequence F1 breakdowns: Easy21 per-sample means (CaoMei1/ChangShouHua2/DouBanLv1/KongQueZhuYu/XianKeLai1) under R0–R3 (all Δ=0 from S1.3); GT10 per-frame R0 vs R1 bars (10 frames; 0039/0040/0041 rescue highlighted, other 7 non-regressed).
- S5.3 Failure-frame score-profile panel (Figure_B full-size: q_area/q_comp/q_edge/q_contrast/q_total-old/R1-primary bars for 0039) + Spearman note (rho=−0.7187, p=0.0038, failure-frame subset, correlational).

**Feeding asset_ids:** P17C-A11 (TRUE; means→main Q4, distributions→S5.1) · P18-A16 (TRUE; Figure_C + Figure_B → main Fig 4 / S5.1/S5.3) · P17C-A14 (TRUE; Easy21 zeros → S5.2).
**Wording guards:** every S5 caption: "descriptive only; N=10+21; no significance stars/p-values as performance evidence" (the p=0.0038 in S5.3 is the pre-registered mechanism correlation on failure frames, reported once with subset scope — not a general superiority test). C05 FORBIDDEN.

---

## 3. SPLIT RULES (decision rules used)

1. **Claim-dependence rule.** An experiment enters MAIN iff its numbers are the sole support of an allowed C/P-claim sentence (C01–C04 PROVEN/SUPPORTED; P01–P08 DESCRIPTIVE/PROVEN-report). Full row-level CSVs behind that sentence go to SUPP (S1/S2/S5). No MAIN sentence without a registry TRUE/ABLATION_ONLY-summary asset; no SUPP claim without a MAIN anchor.
2. **Reviewer-comment pressure rule.** Content forced by reviewers goes to the cheapest compliant home: (a) Q4 ranking-failure + R1 rescue is NEW MAIN (A1-C02 root-cause attribution demanded it); (b) Q7 A6/A7 negative ablation LOGIC stays in MAIN (EIC cross-module + G11 demanded an explicit evaluated-and-demoted verdict); (c) everything the reviewer asked to *see but not headline* (per-frame tables → B-R1-C02 protocol detail; FFT sensitivity → B-DA-C04; external baselines → B-R2-C02) goes to SUPP S1/S4 or Limitations L-paragraphs, never MAIN Results.
3. **Page-budget rule.** MAIN carries ≤1 table + ≤1 figure panel per Q (8 tables / 6 figures total per table/figure plans). Anything that would add a page without changing a claim verdict (182-row benchmark; 150-test logs; 84-row Easy21 safety reruns; per-sequence violins) is SUPP-only by default.
4. **Negative-result home rule.** The *verdict* of a negative result lives in MAIN (one paragraph + one table row + one figure bar set); the *diagnostics* live in SUPP. **Explicitly: A6/A7 negative-ablation LOGIC stays in main (Results Q7 + Discussion D3 + Limitations L3 — the Q7 negative result the reviewers forced), but FULL diagnostics (Phase14 factorial rows, Phase14.1 corrected run logs, Figure_D full-size, per-sample ΔF1) go to S2.** Same pattern for R1: rescue verdict in MAIN Q4, 182-candidate mechanism CSV in S1.2.
5. **Evidence-level quarantine rule.** DEVELOPMENT_ONLY / FALSE assets never support a MAIN claim sentence. They may appear in SUPP ONLY as explicitly labelled diagnostic/reproducibility appendices with a NO-CLAIM header (S1.2 benchmark; S4.4 test summary; S3.1 manifest pointer). ABLATION_ONLY assets are the single exception: their *summary verdict* appears in MAIN Q7, full tables in S2.
6. **Wording-ceiling inheritance rule.** Every MAIN and SUPP caption/text inherits the `allowed_wording`/`forbidden_wording` of its claim row (C05 FORBIDDEN everywhere except the Discussion rationale sentence; "comparable to" for P07; "insufficient" for P06; correlational-only for C03; corrected-exposure-only for C07).

---

## 4. GATE CHECK — orphan sweep (registry → MAIN/SUPP assignment)

| asset_id | paper_usable | Assigned home | Verdict |
|---|---|---|---|
| P12-A01 | TRUE | MAIN Q1 (historical baseline anchor, guarded label) | ASSIGNED |
| P12-A02 | TRUE | MAIN Q1/Q2 (means) + S1.1 (full 21-frame table) | ASSIGNED |
| P13-A03 | DEVELOPMENT_ONLY | EXCLUDED-by-policy (71-test dev gate; S4.4 counts context only, no claim) | EXCLUDED ✓ |
| P14-A04 | ABLATION_ONLY | MAIN Q7 summary + S2.1 full diagnostics (supersede note) | ASSIGNED (exception rule 5) |
| P14.1-A05 | ABLATION_ONLY | MAIN Q7 summary (corrected exposure) + S2.2 full corrected runs | ASSIGNED (exception rule 5) |
| P16-A06 | TRUE | MAIN Q4 (GT10 DEV evidence) | ASSIGNED |
| P16-A07 | FALSE | EXCLUDED-by-policy as evidence; manifest pointer in S3.1 with NO-GT guard only | EXCLUDED ✓ |
| P17A-A08 | DEVELOPMENT_ONLY | EXCLUDED-by-policy (121-test pilot; historical counts only) | EXCLUDED ✓ |
| P17B-A09 | TRUE | MAIN Q4 (oracle audit) | ASSIGNED |
| P17B-A10 | TRUE | MAIN Q4 (diagnosis) | ASSIGNED |
| P17C-A11 | TRUE | MAIN Q4 (policy comparison) + S5.1 (distributions) | ASSIGNED |
| P17C-A12 | DEVELOPMENT_ONLY | EXCLUDED as claim evidence; raw CSV archived S1.2 with NO-CLAIM guard | EXCLUDED ✓ (archive-only) |
| P17C-A13 | TRUE | MAIN Q4 (churn rationale) + S3.2 (full churn context) | ASSIGNED |
| P17C-A14 | TRUE | MAIN Q4 (Easy21 safety) + S1.3/S5.2 (row-level backing) | ASSIGNED |
| P18-A15 | TRUE | MAIN Tables 1a/1b/GT10/rescue/mechanism/ranking/A6A7 + S1.4/S1.5 full reproductions | ASSIGNED |
| P18-A16 | TRUE | MAIN Figs A–D (as manuscript Figs 3/4 + Table-5/ablation panels) + S2.3/S5.1/S5.3 full-size | ASSIGNED |
| P18-A17 | TRUE | MAIN Methods/Results/Discussion/Limitations wording ceilings | ASSIGNED |
| P18-A18 | TRUE | MAIN Methods architecture + S4.2 defaults | ASSIGNED |
| P18-A19 | DEVELOPMENT_ONLY | EXCLUDED-by-policy (8/8 consistency; S4.4 summary counts only, no claim) | EXCLUDED ✓ |
| P11-A20 | DEVELOPMENT_ONLY | EXCLUDED-by-policy (dev history; counts origin only) | EXCLUDED ✓ |
| P16-A21 | DEVELOPMENT_ONLY | EXCLUDED-by-policy (58-target manifest construction; S3.1 pointer only) | EXCLUDED ✓ |

**Orphan sweep result:** 12/12 TRUE assets assigned to MAIN (5 also mirrored in SUPP as full tables/plots) — **0 orphans**. 2/2 ABLATION_ONLY assigned (MAIN summary + S2 diagnostics) — **0 orphans**. 6/6 DEVELOPMENT_ONLY + 1/1 FALSE excluded from claim-bearing content (3 with explicitly labelled non-claim SUPP pointers: S1.2/S3.1/S4.4) — **0 violations**. Excluded-by-policy list verified against registry: P13-A03, P17A-A08, P17C-A12 (archive-only S1.2), P18-A19, P11-A20, P16-A21, P16-A07 (FALSE; manifest-pointer-only S3.1); ABLATION_ONLY exception: P14-A04 → S2.1, P14.1-A05 → S2.2 (both with MAIN Q7 summaries). No TRUE asset left without a MAIN or SUPP home.

**WP7C SELF-CHECK: ALL PASS**
