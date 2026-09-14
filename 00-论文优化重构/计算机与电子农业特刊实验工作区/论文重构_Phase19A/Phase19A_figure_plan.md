# Phase 19A — WP6A Figure Plan (Phase 19B Manuscript)

Audit date: 2026-09-14. Defines 6 figures for the Phase 19B rewrite of ForeSplat.tex.
All paths absolute and verified present on disk.

---

## Notation

- `SRC_DIR` = `.../ForeSplat__Foreground_Object_2D_Gaussian_Splatting_via_Semantic_Mask_Guided_Task_Reformulation_for_Plant/`
- `D18/figures` = `.../阶段十八_论文实验收束与最终审计/figures/`
- `D17B/vis` = `.../阶段十七B_上游失败归因与候选Oracle审计/visualizations/`
- `D05` = `.../05-图件与论文映射/`

---

## Figure 1: Overall Pipeline (RAP-FSAM3 + ForeSplat + M2M3-GO + R1 Ranking) -- KEEP + UPDATE

**Figure number:** Fig 1
**Working title:** "Overall workflow: RAP-FSAM3 generates per-instance foreground-prior candidates; R1 robust ranking selects the optimal mask; ForeSplat injects foreground-object priors into 2DGS with M2M3-GO primitive-level refinement; virtual phenotyping follows."

**Panels (none):** Single-column or full-width workflow diagram.

**Data source files:**
- Primary source: `SRC_DIR/Fig1.png` (491 KB, existing workflow figure)
- Supplementary: `D05/Fig1_总体框架素材包/` (01_input_acquisition, 02_rap_fsam3_vfm, 03_2dgs_optimization, 04_mesh_phenotyping)

**Reviewer comment addressed:** B-R2-C02 (no external baselines; structure/flow clarity); B-R2-C01 (clarify which components are novel in combination vs. novel mechanisms).

**Figure reuse strategy:** Keep existing `Fig1.png` base structure; add "R1 ranking" stage annotation between "Multi-prompt candidate generation" and "Semantic-gated selection" stages. Update caption to explicitly mention per-instance candidate generation and R1 robust instance ranking as the production pipeline.

**Quality/rendering notes:** Update to reflect production pipeline accurately: per-instance candidate generation, R1 ranking (q_contrast excluded from primary score), A6/A7 labelled "optional (disabled by default)." No TIF master available; work from PNG and material-pack drawio sources for rebuild.

**Status:** KEEP, requires minor annotation update to reflect R1 ranking stage.

---

## Figure 2: Prompt Robustness and Semantic Bias -- KEEP + EXTEND

**Figure number:** Fig 2
**Working title:** "Direct VFM prompting exhibits semantic bias in potted-plant scenes: single-prompt outputs vary drastically across semantically similar but morphologically diverse prompts (plant-instance prompt F1=0.9692; green-region F1=0.7972; seedling-oriented F1=0.0768 on mature plants)."

**Panels (none):** Grid of per-sample per-prompt segmentation results (existing Fig4 grid layout).

**Data source files:**
- Primary source: `SRC_DIR/Fig4.png` (272 KB, existing prompt-robustness figure)
- Supplementary panels: `D05/Fig4_Prompt鲁棒性与语义偏置分析素材包/01_grid_ordered_assets/` (~18 per-sample grid PNGs for CaoMei1, ChangShouHua2, DouBanLv1 with multiple prompts)
- Supporting panel: `D05/实验二三_RAP-FSAM3图板/figures/Fig_E2_prompt_P2_vs_P4.png` (5 MB, prompt P2 vs P4 diff board)

**Reviewer comment addressed:** B-R3-C03 (prompt selection rationale); P03 claim (proven: prompt sensitivity is the central motivation).

**Figure reuse strategy:** Keep existing `Fig4.png` as primary. Optionally swap in `Fig_E2_prompt_P2_vs_P4.png` as a supplementary highlight panel showing P2 vs P4 differences for a representative sample (KongQueZhuYu).

**Quality/rendering notes:** Descriptive only, no significance claims. Caption must state per-prompt F1 values as descriptive means on the frozen P6 evaluation set. No "robustness" framing in caption -- use "prompt sensitivity" or "semantic bias."

**Status:** KEEP; optionally swap in E2 panel.

---

## Figure 3: Candidate-Ranking Failure Mechanism -- NEW

**Figure number:** Fig 3 (NEW -- carries the Phase 17C ranking story currently missing from the manuscript)
**Working title:** "Candidate-ranking failure mechanism: on DouBanLv2 0039/0040/0041 the baseline candidate generator produced high-quality whole-plant masks (oracle F1=0.9867-0.9882) but the old contrast-based ranking selected high-contrast distractor masks (R0-selected F1=0.0); R1 robust ranking rescues all three failures."

**Panels (5-column layout, 3 rows):**
- Row 1: DouBanLv2 0039 (RGB / GT mask / R0 wrong selection / oracle candidate / R1-selected candidate)
- Row 2: DouBanLv2 0040 (same 5-column layout)
- Row 3: DouBanLv2 0041 (same 5-column layout)
- Sub-panel (small inset): bar chart of candidate feature scores for frame 0039 (q_area, q_comp, q_edge, q_contrast, q_total old, R1 primary) -- from Phase18 Figure_B

**Data source files:**
- Primary: `D18/figures/Figure_A_failure_mechanism.png` (986 KB) -- 3 frames x 5 panels, bilingual EN/labels
- Primary: `D18/figures/Figure_B_ranking_illustration.png` (102 KB) -- bar chart for frame 0039 candidate scores
- Supplementary: `D17B/vis/map_gt_DouBanLv2_0039.jpg`, `map_gt_DouBanLv2_0040.jpg`, `map_gt_DouBanLv2_0041.jpg` (GT masks)
- Supplementary: `D17B/vis/map_pred_DouBanLv2_0039.jpg`, `map_pred_DouBanLv2_0040.jpg`, `map_pred_DouBanLv2_0041.jpg` (R0 predicted wrong masks)
- Supporting evidence: `D18/Phase18_failure_rescue_table.csv` (3 rows: 0039/0040/0041 with R0 oracle regret and R1 rescue F1>=0.95)
- Supporting evidence: `D18/Phase18_score_mechanism_table.csv` (6 rows: q_* feature scores per failure frame)

**Reviewer comment addressed:** A1-C02 (Plant Seg. component contribution unclear -- candidate generation succeeds, ranking fails); C01 (candidate generation quality), C02 (ranking is root cause), C03 (q_contrast mechanism).

**Figure reuse strategy:** Embed Phase18 Figure_A as the main visual; embed Figure_B as a small inset (or side panel) showing the score-mechanism contrast bias. Both are paper-ready PNG+PDF.

**Quality/rendering notes:** Figure_A already has bilingual EN/labels; no rebuild needed. Combine Figures A+B into a single composite figure in LaTeX (subfigure/subcaption or includegraphics grid). Descriptive only -- no "significantly" or "statistically" in caption.

**Status:** READY (Phase18). NEW to manuscript.

---

## Figure 4: Ranking Policy Ablation and Failure Rescue -- NEW

**Figure number:** Fig 4 (NEW -- directly below Figure 3 in the ranking-failure subsection)
**Working title:** "R1 robust ranking (q_contrast removed from primary score) eliminates all identified DEV-set ranking failures; easy vs hard F1 distribution shows descriptive improvement in the hard subset."

**Panels (2-panel layout):**
- Panel (a): Easy vs Hard F1 violin plot (Phase18 Figure_C) -- R0 vs R1 vs R2 vs R3 on Easy21 and DEV10, descriptive only, no significance stars
- Panel (b): Rescue evidence -- bar chart showing DouBanLv2 0039/0040/0041 F1 under R0 (near zero) vs R1 (>=0.95) -- derived from `Phase18_failure_rescue_table.csv`

**Data source files:**
- Primary: `D18/figures/Figure_C_easy_vs_hard_distribution.png` (80 KB) -- violin plot, descriptive
- Primary data: `D18/Phase18_ranking_ablation_table.csv` (4 rows: R0/R1/R2/R3 ablation results)
- Rescue data: `D18/Phase18_failure_rescue_table.csv` (3 rescue rows)
- Policy comparison: `D17C/Phase17C_policy_comparison.csv` (R0-R3 full comparison)
- Safety evidence: `D17C/Phase17C_easy21_safety.csv` (zero regressions for R1 on Easy21)

**Reviewer comment addressed:** A1-C02 (ranking vs segmentation root cause); C04 (R1 rescue evidence); C05 (must NOT claim "significantly improves" -- only selection rationale).

**Figure reuse strategy:** Embed Phase18 Figure_C as the main panel. Add a new rescue bar chart panel (b) built from `Phase18_failure_rescue_table.csv` values.

**Quality/rendering notes:** Descriptive only throughout -- no significance stars, no p-values, no "significant" wording. Violin plot must be captioned as "descriptive distribution, N=10 DEV frames + 21 Easy frames, no inferential test." Rescue bars captioned as "R1 rescued all 3 identified failures to F1>=0.95 with zero material regressions on Easy21."

**Status:** READY (Phase18 Figure_C is paper-ready). Rescue bar chart needs minimal build from CSV.

---

## Figure 5: Foreground-Object 2DGS Constraints and Injection Ablation -- KEEP + EXTEND

**Figure number:** Fig 5
**Working title:** "Foreground-object 2DGS optimisation: training-time prior injection reduces outside-mask non-black ratio from 0.9908 to 0.0294 and leakage energy from 1.2201 to 0.0190; M2M3-GO provides additional primitive-level refinement; post hoc mask pruning is insufficient."

**Panels (3-subfigure layout, matching existing Fig5/6/7):**
- Panel (a): Foreground-track init + foreground RGB supervision + M2M3-GO refinement layer (existing `Fig5.png`)
- Panel (b): M2M3-GO micro-level mechanism: mask-to-model support, model-to-mask coverage deficit, boundary support; broad-leaf/compact-flowering/low-canopy examples; Floor40 gate (existing `Fig6.png`)
- Panel (c): Visual comparison of injection methods: standard 2DGS / input masking / foreground-object loss injection / post hoc pruning (existing `Fig7.png`)

**Data source files:**
- Primary: `SRC_DIR/Fig5.png` (456 KB) -- foreground-object optimisation + M2M3-GO
- Primary: `SRC_DIR/Fig6.png` (471 KB) -- M2M3-GO micro-level controller
- Primary: `SRC_DIR/Fig7.png` (540 KB) -- injection ablation visual evidence
- Supplementary: `D05/实验四_2DGS先验注入位置消融/figures/Fig_E4_prior_injection_B0_B5_metrics.png` (935 KB) -- quantitative injection metrics chart
- Supplementary: `D05/实验五_掩膜到表型误差敏感性/figures/Fig_E5_error_distribution.png` (395 KB) -- error distribution
- Supplementary: `D05/实验五_掩膜到表型误差敏感性/figures/Fig_E5_trait_error_profile.png` (772 KB) -- per-trait error profile

**Reviewer comment addressed:** A1-C02 (fair-input ablation -- injection ablation requires explicit note that full-scene 2DGS uses original frames with background while foreground-object uses foreground-masked RGB); B-DA-C01 (E7 post-hoc pruning empirical result now available); B-R2-C03 (mesh variant to phenotype connection).

**Figure reuse strategy:** Keep existing Fig5/Fig6/Fig7 as 3-subfigure composite. Optionally swap Fig7 with a combined visual+quantitative panel incorporating `Fig_E4_prior_injection_B0_B5_metrics.png`. No rebuild needed for core panels.

**Quality/rendering notes:** Descriptive only. Fig7 caption should specify "on a representative broad-leaf foreground morphology (KongQueZhuYu)" to match P06 scope qualifier. Mention that post-hoc pruning is "insufficient" (not "useless" per claim_audit).

**Status:** KEEP; optionally extend with E4 quantitative chart.

---

## Figure 6: Phenotyping Validation -- KEEP + QUALIFY

**Figure number:** Fig 6
**Working title:** "Virtual vs manual phenotypic measurements were comparable to manual measurements across 21 plants (height R^2=0.988, canopy R^2=0.988) and 63 leaves (leaf length R^2=0.974, leaf width R^2=0.900); no repeated manual measurements were collected."

**Panels (2-panel layout):**
- Panel (a): Scatter plots for the four traits (height, canopy width, leaf length, leaf width) with regression lines and R^2 values -- existing `Fig9.png` layout
- Panel (b): Representative reconstruction and mesh outputs demonstrating foreground-object isolation -- existing `Fig8.png` layout, or E5 trait error profile panel

**Data source files:**
- Primary: `SRC_DIR/Fig9.png` (919 KB) -- scatter plots with regression lines, residual distributions
- Primary: `SRC_DIR/Fig8.png` (691 KB) -- reconstruction and mesh examples
- Supplementary: `D05/实验五_掩膜到表型误差敏感性/figures/Fig_E5_trait_error_profile.png` (772 KB) -- per-trait error profile
- Supplementary: `D05/实验五_掩膜到表型误差敏感性/figures/Fig_E5_error_distribution.png` (395 KB) -- error distribution
- Supporting data: `09-PNG/20260306-pic/` (Plant_Height.pdf/png, Canopy_Width.pdf/png, Leaf_Length.pdf/png, Leaf_Width.pdf/png; raw trait data CSV)

**Reviewer comment addressed:** A2-C01 (R^2 to 2 decimal places); A2-C02 (add MAPE alongside RMSE); A1-C03 (leaf width limitation -- R^2=0.90 is weakest; discuss error sources); B-R2-C03 (mesh to phenotype connection); P07 claim (comparable to, no "strong agreement").

**Figure reuse strategy:** Keep existing `Fig8.png` and `Fig9.png` as two subfigures. Optionally replace Fig8 with a combined reconstruction+mesh example panel that includes `Fig_E5_trait_error_profile.png` for the quantitative error context.

**Quality/rendering notes:** Caption MUST say "comparable to manual measurements" -- NOT "strong agreement" (forbidden by claim_audit). Add caveat: "repeated manual measurements were not collected; descriptive R^2 on n=21 plants and n=63 leaves." R^2 values reported to 2 decimal places (0.99/0.99/0.97/0.90) per reviewer A2-C01.

**Status:** KEEP; caption wording revision mandatory.

---

## Figure Inventory Summary

| New # | Status | New/Reuse | Carries Phase 17C Story | Source |
|---|---|---|---|---|
| Fig 1 | KEEP + UPDATE | Reuse (annotate) | No | SRC_DIR/Fig1.png |
| Fig 2 | KEEP + EXTEND | Reuse (optional swap) | No | SRC_DIR/Fig4.png + E2 panel |
| Fig 3 | **NEW** | **NEW from Phase18 Figure_A + Figure_B** | **Yes** | D18/figures/Figure_A/B |
| Fig 4 | **NEW** | **NEW from Phase18 Figure_C + CSV data** | **Yes** | D18/figures/Figure_C + CSVs |
| Fig 5 | KEEP + EXTEND | Reuse (optional extend) | No | SRC_DIR/Fig5/Fig6/Fig7.png |
| Fig 6 | KEEP + QUALIFY | Reuse (caption fix) | No | SRC_DIR/Fig8/Fig9.png + E5 |

**Total: 6 figures** (4 retained with minor updates, 2 new carrying ranking-failure story).

**Critical rule for Figures 3 and 4:** No "significantly," "proven," "statistically," or significance stars in captions or labels. Descriptive-only wording throughout per Phase 18 conventions (C05 FORBIDDEN claim must not appear).

---

## Verification

- [x] 6 figures defined (>=5 required)
- [x] All fields filled for each figure (number, title, panels, source files, reviewer comment, reuse strategy, quality notes)
- [x] Figure 3 and Figure 4 explicitly labelled NEW
- [x] All source paths verified present on disk
- [x] No forbidden wording in figure plans ("outperforming," "strong agreement," "significantly," "SOTA")
- [x] Figures 3 and 4 carry the Phase 17C ranking-failure story

**WP6A SELF-CHECK: FIGURE_PLAN PASS**
