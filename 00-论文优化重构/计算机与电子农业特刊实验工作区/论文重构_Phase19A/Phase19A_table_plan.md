# Phase 19A — WP6A Table Plan (Phase 19B Manuscript)

Audit date: 2026-09-14. Defines 8 tables for the Phase 19B rewrite of ForeSplat.tex.
All paths absolute and verified present on disk.

---

## Notation

- `SRC_DIR` = `.../ForeSplat__Foreground_Object_2D_Gaussian_Splatting_via_Semantic_Mask_Guided_Task_Reformulation_for_Plant/`
- `D18` = `.../阶段十八_论文实验收束与最终审计/`
- `D17C` = `.../阶段十七C_候选排序修复开发/`
- `D12` = `.../阶段十二/`

---

## Table 1: Formal Notation and Symbol Reference -- NEW

**Table number:** Table 1
**Title:** "Notation reference"
**Purpose:** Provide the formal symbol table required by formula_audit; eliminates P2/P6 ambiguity and penalty sign confusion for readers.

**Columns:**

| Column | Description |
|---|---|
| Symbol | LaTeX notation (e.g., q_area, B, F, D, P, S) |
| Name | Plain-English name (e.g., "area quality score", "base total", "primary score", "denominator", "risk penalty", "semantic gate total") |
| Default value | Production default (e.g., 1.0, 5.5, 0.0, False) |
| Brief definition | One-sentence definition (e.g., "triangle-mapped quality of foreground-to-image area ratio") |

**Symbols to include (28 total):**

Feature scores (8): q_area, q_comp, q_edge, q_temp, q_contrast, q_leak, q_side, q_sam
Weights (6): w_area, w_comp, w_edge, w_temp, w_contrast, w_sam
Derived (3): D (denominator, default 5.5), B (base total), T (tie-break, default 0.0)
Penalty (3): P_v = 0.25 (vertical coverage), P_t = 0.15 (track overlap), P (total risk penalty, {0.0, 0.15, 0.25, 0.40})
Semantic (1): S (semantic gate total, default 0.0, disabled)
Composite (1): F (primary_score under R1 policy)
Hyperparams (6): A_min = 0.01, A_max = 0.80, V_min = 0.30, O_min = 0.20, tau_mask = 0.5, tau_conf = 0.3

**Source CSV:** None (constructed from formula_audit.md section 2 symbol tables)

**Supporting data file (exact path):**
`/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/论文重构_Phase19A/Phase19A_formula_audit.md`

**Claim IDs supported:** P04 (RAP-FSAM3 ablation; formal notation required for method section)

**Reviewer comment addressed:** B-R2-C04 (alpha/beta/gamma/delta/eta weights tau_g not reported); B-R1-C02 (measurement protocol underspecified -- notation reference sets the formal basis).

**Allowed wording:** Symbol definitions are descriptive/factual. No performance claims.

**Forbidden wording:** None applicable (pure reference table).

---

## Table 2: Dataset and Evaluation Protocol

**Table number:** Table 2
**Title:** "Dataset composition and evaluation protocol"
**Purpose:** Consolidate the 5-dataset evaluation framework: Frozen P6 (21 frames, 5 species, annotated), Easy21 (21 frames, 5 species, annotated, used for A6/A7 factorial + R1 safety), GT10/DEV10 (10 frames, 2 species, annotated, used for ranking-failure diagnosis + R1 rescue), TEST32 (32 frames, 8 species, unlabeled, behavioral safety only), and the 20-sequence workflow (21 plants, 10 species, full pipeline validation + phenotyping).

**Columns:**

| Column | Description |
|---|---|
| Dataset name | Frozen P6 / Easy21 / GT10 (DEV10) / TEST32 / 20-sequence workflow |
| Frames | 21 / 21 / 10 / 32 / all 20 sequences |
| Species | 5 / 5 / 2 / 8 / 10 |
| GT available | Yes / Yes / Yes / No / Partial (phenotype only) |
| Paper use | VFM comparison; RAP-FSAM3 ablation; prompt robustness / R1 safety check; A6/A7 factorial / R1 safety / ranking policy / failure-rescue evaluation / ranking-failure diagnosis / full workflow validation + phenotypic measurement |

**Source CSV:**
- Existing `Table 1` (tab:dataset-protocol) from ForeSplat.tex line 135
- `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十八_论文实验收束与最终审计/Phase18_dataset_registry.csv`

**Claim IDs supported:** P01, P03, P04, P05, P07 (all downstream claims depend on dataset definitions)

**Reviewer comment addressed:** B-EIC-C01 (cross-species framing vs. 3-reconstruction-sample gap -- table must explicitly state species per dataset); B-R1-C04 (growth stage not reported -- add note).

**Allowed wording:** Dataset descriptions are factual.

**Forbidden wording:** None applicable (protocol table).

---

## Table 3: VFM Comparison on Frozen P6

**Table number:** Table 3
**Title:** "VFM-based foreground-prior comparison on the frozen P6 evaluation set (21 annotated frames, 5 species)"
**Purpose:** Report F1, mIoU, HD95, Boundary-F1, and leakage energy for CLIPSeg, Grounded-SAM, single-prompt SAM3, and RAP-FSAM3.

**Columns:**

| Column | Description |
|---|---|
| Method | CLIPSeg / Grounded-SAM / SAM3 (single-prompt) / RAP-FSAM3 |
| F1 | Mean F1 on 21 frames |
| mIoU | Mean IoU |
| HD95 (px) | 95th percentile Hausdorff distance |
| Boundary-F1 | Boundary F1 score |
| Leakage energy | Leakage energy ratio |

**Source CSV:**
- Existing `Table 3` (tab:vfm-comparison) from ForeSplat.tex line 580
- `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十二/指标/P6_全部21帧.csv` (per-frame metrics)

**Claim IDs supported:** P01 (RAP-FSAM3 metrics on frozen P6), P02 (boundary/leakage improvements concentrated in Boundary-F1 and leakage energy)

**Reviewer comment addressed:** B-EIC-C02 (quantitative mask metrics now available for FSAM3/RAP-FSAM3); B-DA-C03 (feasibility vs. quality claim resolved).

**Allowed wording:** "RAP-FSAM3 achieved F1=0.9706, mIoU=0.9641, HD95=183.42 px, Boundary-F1=0.4643, leakage energy=0.001739 on the frozen P6 evaluation set (21 frames, 5 species). Improvements over single-prompt SAM3 were concentrated in boundary consistency (Boundary-F1 0.4643 vs. 0.4051) and leakage suppression (0.001739 vs. 0.002272); region-overlap deltas (F1 0.9706 vs. 0.9692, mIoU 0.9641 vs. 0.9624) were small."

**Forbidden wording:** "outperforming" (forbidden); "statistically significant" (forbidden -- no test); "best overall" (forbidden unless qualified as "highest on this evaluation set"); "generalizes" (forbidden).

---

## Table 4: Prompt Robustness (Semantic Bias Analysis)

**Table number:** Table 4
**Title:** "Per-prompt segmentation quality on the frozen P6 evaluation set (21 frames, 5 species)"
**Purpose:** Report per-prompt F1 for all 5 direct VFM prompts, demonstrating semantic fragility.

**Columns:**

| Column | Description |
|---|---|
| Prompt ID | P1 (green plant) / P2 (entire potted plant) / P3 (leaves and stems) / P4 (seedling) / P5 (plant without background) |
| Prompt text | Full English prompt text |
| F1 | Mean F1 on 21 frames |
| mIoU | Mean IoU |
| Boundary-F1 | Boundary F1 |

**Source CSV:**
- Existing `Table 4` (tab:prompt-robustness) from ForeSplat.tex line 610

**Claim IDs supported:** P03 (direct VFM prompting is semantically fragile in potted-plant scenes; plant-instance prompt F1=0.9692 most stable; green-region 0.7972, organ-level 0.6471, seedling 0.0768, background-excluding 0.4051).

**Reviewer comment addressed:** B-R3-C03 (prompt selection rationale).

**Allowed wording:** "The plant-instance prompt was the most stable direct prompt (F1=0.9692); green-region 0.7972 (green-background inclusion), organ-level 0.6471 (under-recall), seedling-oriented 0.0768 (near-empty on mature plants), background-excluding 0.4051 (over-conservative)."

**Forbidden wording:** "robust" as a general claim (forbidden); "generalises" (forbidden).

---

## Table 5: RAP-FSAM3 Module Ablation (Merged with R1 Ranking-Failure Row)

**Table number:** Table 5
**Title:** "RAP-FSAM3 module ablation on the frozen P6 evaluation set (21 frames) and ranking-policy ablation on GT10/DEV10 (10 frames)"
**Purpose:** Merge the current module-ablation rows with the new Phase 17C ranking-failure row, showing that the geometry-only selector failure IS the ranking-failure phenomenon, and R1 rescues the hard frames.

**Columns:**

| Column | Description |
|---|---|
| Variant | A0 (baseline) / A1 (semantic-gated selection only) / A2 (SB-FBPR only) / A5 (geometry-consistency only) / Full RAP-FSAM3 (A1+A2+A5+GCFC) |
| F1 | Mean F1 on frozen P6 |
| mIoU | Mean IoU |
| Boundary-F1 | Boundary F1 |
| Leakage energy | Leakage energy |
| HD95 (px) | 95th percentile Hausdorff |

**Plus ranking-policy sub-table (NEW rows):**

| Variant | Dataset | N frames | Mean F1 | Catastrophic failures (F1=0) |
|---|---|---|---|---|
| R0 (baseline scoring) | GT10 (DEV10) | 10 | 0.6767 | 3 (DouBanLv2 0039/0040/0041) |
| R1 (q_contrast excluded from primary) | GT10 (DEV10) | 10 | 0.9728 | 0 |
| R2 (R1 + q_contrast tie-break) | GT10 (DEV10) | 10 | 0.9728 | 0 |
| R3 (q_contrast area-gated) | GT10 (DEV10) | 10 | 0.9728 | 0 |

**Source CSV:**
- Module ablation: existing `Table 5` (tab:rap-ablation) from ForeSplat.tex line 641
- Ranking-policy rows: `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十八_论文实验收束与最终审计/Phase18_ranking_ablation_table.csv` (4 rows)
- R1 safety on Easy21: `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十七C_候选排序修复开发/Phase17C_easy21_safety.csv`

**Claim IDs supported:** P04 (module ablation: geometry-only F1=0.8429, semantic gating recovers 0.9692, full RAP-FSAM3 0.9706); C04 (R1 rescue: 3 failures rescued, 0 regressions on Easy21); C05 (R1 selection rationale, NOT "significantly improves").

**Reviewer comment addressed:** A1-C02 (Plant Seg. component contribution; ranking root cause); B-EIC-C03 (M1-soft vs M4 contributions independently discussed).

**Allowed wording:** "The geometry-only selector degraded F1 to 0.8429 (semantically inappropriate candidates); semantic gating recovered the stable plant-instance prompt (0.9692); full RAP-FSAM3 with GCFC achieved 0.9706. On the GT10 evaluation set, R0 baseline mean F1=0.6767 with 3 catastrophic failures (DouBanLv2 0039/0040/0041, selected distractor F1=0.0 despite oracle F1=0.9867-0.9882); R1 eliminated all identified ranking failures (0 catastrophic, F1>=0.95) with zero material regressions on Easy21."

**Forbidden wording:** "significantly improves" (C05 FORBIDDEN); "proven superior globally" (C05 FORBIDDEN); "significantly" (statistical claim forbidden without test).

---

## Table 6: Injection-Position Ablation (Fair-Input Note)

**Table number:** Table 6
**Title:** "Foreground-object prior injection ablation on a representative broad-leaf morphology (KongQueZhuYu)"
**Purpose:** Report the injection-position ablation with explicit input-condition framing to address reviewer A1-C02 (fair comparison: full-scene 2DGS uses raw RGB with background; foreground-object methods use foreground-masked RGB).

**Columns:**

| Column | Description |
|---|---|
| Variant | Standard 2DGS (A0) / Input masking / Foreground-object loss injection (A5) / Post-hoc mask pruning (E7) / M2M3-GO (A5+M2M3-GO) |
| Input condition | Raw RGB with background / Input mask on RGB / FG RGB supervision on masked regions / Same as A0 + post hoc pruning / Same as A5 + M2M3-GO |
| Outside-mask non-black ratio | 0.9908 / -- / 0.0294 / 0.7509 / -- |
| Leakage energy | 1.2201 / -- / 0.0190 / 0.7900 / -- |
| PSNR_fg (dB) | -- / 24.21->20.73 / -- / -- / -- |

**Source CSV:**
- Existing `Table 5` (tab:injection-ablation) from ForeSplat.tex line 679
- Supporting: `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/05-图件与论文映射/实验四_2DGS先验注入位置消融/figures/Fig_E4_prior_injection_B0_B5_metrics.png`

**Claim IDs supported:** P06 (foreground-object loss injection is necessary; post-hoc pruning insufficient; single-sample representative ablation).

**Reviewer comment addressed:** A1-C02 (fair-input comparison note: "standard 2DGS uses the original image with the background; foreground-object methods use foreground-masked RGB"); B-DA-C01 (E7 post-hoc pruning empirical result, single sample).

**Allowed wording:** "Full-scene 2DGS (standard input, background included) had outside-mask non-black ratio 0.9908 and leakage 1.2201; input masking reduced PSNR_fg from 24.21 to 20.73; foreground-object loss injection reduced outside ratio to 0.0294 and leakage to 0.0190; post hoc mask pruning left outside ratio at 0.7509 and leakage at 0.7900 (single representative sample, KongQueZhuYu)."

**Forbidden wording:** "post hoc pruning is useless" (forbidden -- say "insufficient" per claim_audit); "significant" (forbidden); "not equivalent" as a strong claim (hedge: "training-time injection is not replaceable by post hoc processing").

**Fair-input note (mandatory footnote):** "Standard 2DGS (A0) is evaluated on raw RGB frames with background; all foreground-object variants use foreground-masked RGB as input. The outside-ratio and leakage-energy comparisons therefore reflect the different input conditions, not solely the effect of loss design."

---

## Table 7: Ranking Policy / Failure-Rescue (NEW)

**Table number:** Table 7
**Title:** "Candidate-ranking failure and R1 rescue evidence on GT10/DEV10 (10 annotated frames, 2 species)"
**Purpose:** NEW table presenting the complete ranking-failure story: 3 failure frames with oracle F1 vs R0-selected F1; R1 rescue results; zero-regression safety check on Easy21; selection churn on TEST32.

**Columns (multi-part table):**

Part A -- Failure frames (3 rows):

| Column | Description |
|---|---|
| Frame | DouBanLv2 0039 / 0040 / 0041 |
| Candidate F1 (oracle) | Best-candidate raw F1 (0.9867 / 0.9882 / 0.9865) |
| R0-selected F1 | F1 of the mask selected by old scoring (0.0 for all 3) |
| R1-selected F1 | F1 of the mask selected by R1 (>=0.95 for all 3) |

Part B -- Policy comparison (4 rows):

| Column | Description |
|---|---|
| Policy | R0 (baseline) / R1 (q_contrast excluded) / R2 (R1 + q_contrast tie-break) / R3 (area-gated q_contrast) |
| Mean F1 (GT10) | 0.6767 / 0.9728 / 0.9728 / 0.9728 |
| Catastrophic failures (F1=0) | 3 / 0 / 0 / 0 |
| Selection churn (TEST32, 32 frames) | Reference 6.2% threshold: 2/32 changes (<30% threshold) |
| Easy21 regressions | -- / 0 / 0 / 0 |

**Source CSV:**
- `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十八_论文实验收束与最终审计/Phase18_failure_rescue_table.csv` (3 rescue rows)
- `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十八_论文实验收束与最终审计/Phase18_ranking_ablation_table.csv` (4 policy rows)
- `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十七C_候选排序修复开发/Phase17C_selection_churn.csv`
- `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十七C_候选排序修复开发/Phase17C_easy21_safety.csv`

**Claim IDs supported:** C01 (candidate generation quality), C02 (ranking root cause), C03 (q_contrast mechanism), C04 (R1 rescue evidence).

**Reviewer comment addressed:** A1-C02 (root-cause attribution: ranking not segmentation).

**Allowed wording:** "Removing q_contrast from the primary ranking (R1) eliminated all identified DEV-set ranking failures (DouBanLv2 0039/0040/0041 rescued to F1>=0.95) with zero material regressions on Easy21. R1 was selected for failure-rescue, zero material regression, low selection churn (6.2% on TEST32), and structural simplicity -- not because its mean F1 is significantly higher (N=10 mean-F1 comparisons are descriptive only)."

**Forbidden wording:** "R1 significantly improves general segmentation performance" (C05 FORBIDDEN); "R1 proven superior globally" (C05 FORBIDDEN); "statistically significant" (forbidden).

---

## Table 8: Phenotypic Agreement (Current Table 7)

**Table number:** Table 8
**Title:** "Virtual vs manual phenotypic measurement agreement (21 plants, 63 leaves, 10 species)"
**Purpose:** Report per-trait R^2, MAE, RMSE, MAPE, and mean manual measurement values, with appropriate hedging.

**Columns:**

| Column | Description |
|---|---|
| Trait | Plant height / Canopy width / Leaf length / Leaf width |
| n | 21 (plant-level) or 63 (leaf-level) |
| R^2 | 0.99 / 0.99 / 0.97 / 0.90 (2 decimal places per A2-C01) |
| MAE (cm) | 0.98 / 0.86 / 0.51 / 0.45 |
| MAPE (%) | Report alongside RMSE per A2-C02 |
| Mean manual (cm) | Report mean manual measurement per trait for reference scale |

**Source CSV:**
- Existing `Table 7` (tab:phenotyping) from ForeSplat.tex line 767

**Claim IDs supported:** P07 (virtual measurements comparable to manual measurements; R^2=0.988/0.988/0.974/0.900 on n=21/n=63; no repeated measurements, no Bland-Altman).

**Reviewer comment addressed:** A2-C01 (R^2 to 2 decimal places); A2-C02 (add MAPE and mean values); A1-C03 (leaf width R^2=0.90 -- discuss error sources as limitation); B-R1-C02 (measurement protocol details); B-R1-C03 (leaf area/stem diameter not measured -- future work).

**Allowed wording:** "Virtual measurements were comparable to manual measurements: plant height R^2=0.99 (MAE 0.98 cm), canopy width R^2=0.99 (MAE 0.86 cm), leaf length R^2=0.97 (MAE 0.51 cm, n=63), leaf width R^2=0.90 (MAE 0.45 cm). R^2 values are descriptive on the 21-plant / 63-leaf evaluation set; no repeated manual measurements were collected."

**Forbidden wording:** "strong agreement" (forbidden per claim_audit -- use "comparable to"); "accurate" / "high-precision" (forbidden); "statistically validated agreement" (forbidden).

---

## Table Inventory Summary

| New # | Status | New/Reuse | Carries Phase 17C Story | Key Claim IDs |
|---|---|---|---|---|
| Table 1 | **NEW** | **NEW** | No | P04 (notation) |
| Table 2 | KEEP + UPDATE | Reuse | No | P01, P03, P04, P05, P07 |
| Table 3 | KEEP + REWORD | Reuse | No | P01, P02 |
| Table 4 | KEEP | Reuse | No | P03 |
| Table 5 | KEEP + MERGE | Reuse + NEW rows | **Yes** (ranking-policy sub-table) | P04, C04, C05 |
| Table 6 | KEEP + REWORD | Reuse + fair-input note | No | P06 |
| Table 7 | **NEW** | **NEW from Phase17C/18 CSVs** | **Yes** | C01, C02, C03, C04 |
| Table 8 | KEEP + REWORD | Reuse | No | P07 |

**Total: 8 tables** (2 new, 4 retained with minor edits, 2 significant merges/rewrites).

**Critical rule for every table:** Allowed and forbidden wording from claim_evidence_matrix and claim_audit must be enforced at caption, running-text, and footnote levels. No table may use bolding implying "best" unless captioned as "highest on this evaluation set."

---

## Verification

- [x] 8 tables defined (>=6 required)
- [x] All fields filled for each table (number, title, columns, source CSVs, claim IDs, reviewer comment, allowed/forbidden wording)
- [x] Table 1 and Table 7 explicitly labelled NEW
- [x] All source CSV paths verified present on disk
- [x] Forbidden wording explicitly noted for every table
- [x] Tables 5 and 7 carry the Phase 17C ranking-failure story

**WP6A SELF-CHECK: TABLE_PLAN PASS**
