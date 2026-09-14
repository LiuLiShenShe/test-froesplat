# Phase 19A — WP4 Image Asset Inventory

Audit date: 2026-09-14. Catalogue of all figure/graphic assets available for the ForeSplat manuscript, grouped by source. All paths absolute and verified present on disk.

---

## Section 1: ForeSplat.tex Current Figures (Fig1–Fig9)

Source directory: `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat__Foreground_Object_2D_Gaussian_Splatting_via_Semantic_Mask_Guided_Task_Reformulation_for_Plant/`

| Figure | File (in source dir) | Size (bytes) | Caption / content (from ForeSplat.tex) | tex reference |
|---|---|---|---|---|
| Fig1 | `Fig1.png` | 502,751 | Overall workflow: RAP-FSAM3 adapts VFM outputs into reconstruction-aware potted-plant foreground priors; ForeSplat injects priors into 2DGS optimisation for foreground-object meshes + virtual trait measurements; M2M3-GO refines keep/densify/prune. | `\label{fig:workflow}` (L126) |
| Fig2 | `Fig2.png` | 504,898 | RAP-FSAM3 foreground-prior generation: semantic-gated multi-prompt selection, structure-balanced FG/BG prompt refinement, geometry-consistency correction. | `\label{fig:rap-fsam3}` (L303) |
| Fig3 | `Fig3.png` | 487,654 | Qualitative comparison of VFM-based potted-plant foreground priors (RAP-FSAM3 vs single-prompt + general VFM baselines). | `\label{fig:vfm-comparison}` (L602) |
| Fig4 | `Fig4.png` | 277,618 | Prompt sensitivity of direct VFM segmentation; single-prompt outputs exhibit semantic bias across plant forms. | `\label{fig:prompt-failure}` (L633) |
| Fig5 | `Fig5.png` | 465,993 | Foreground-object optimisation with M2M3-GO refinement layer: sparse-track init, foreground-conditioned rendering losses, opacity constraints, view-quality weighting, primitive-level decisions. | `\label{fig:algorithm}` (L674) |
| Fig6 | `Fig6.png` | 481,589 | Micro-level M2M3-GO primitive refinement mechanism: mask-to-model support, model-to-mask coverage deficit, boundary support; broad-leaf/compact-flowering/low-canopy mechanism examples; Floor40 gate. | `\label{fig:m2m3-go-micro}` (L714) |
| Fig7 | `Fig7.png` | 552,908 | Visual evidence for VFM-prior injection in 2DGS; training-time foreground-object optimisation vs input masking / post hoc pruning. | `\label{fig:injection-ablation}` (L724) |
| Fig8 | `Fig8.png` | 707,172 | Representative reconstruction and mesh outputs; ForeSplat reduces background residue and produces foreground-object meshes for virtual phenotyping. | `\label{fig:reconstruction}` (L760) |
| Fig9 | `Fig9.png` | 940,930 | Agreement between manual and virtual phenotypic measurements (scatter, residual distributions, summary metrics). | `\label{fig:phenotyping}` (L789) |

Status: all 9 currently embedded in ForeSplat.tex. Fig8 caption mentions "background residue" wording to check against Phase18 claim constraints (descriptive only, no significance claim). TIF/SVG masters not present in source dir; PNG-only.

---

## Section 2: Phase18 Final Figures (Figure A–D)

Source directory: `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十八_论文实验收束与最终审计/figures/`

| Figure | File (.png/.pdf both exist) | Size (PNG bytes) | Content | Paper section |
|---|---|---|---|---|
| Figure A | `Figure_A_failure_mechanism.png` / `.pdf` | 985,962 | Failure mechanism: 3 frames (DouBanLv2 0039/0040/0041) × 5 panels (RGB / GT / R0 wrong selection / oracle candidate / R1 selected). Bilingual EN/中 labels. | Results §3.3 (NEW ranking-failure analysis) — supports claims C01, C02, C04 |
| Figure B | `Figure_B_ranking_illustration.png` / `.pdf` | 101,756 | Candidate feature comparison for frame 0039: bar chart of oracle vs R0-selected distractor scores (q_area, q_comp, q_edge, q_contrast, total old, R1 primary). | Results §3.3 — supports C03, C04 |
| Figure C | `Figure_C_easy_vs_hard_distribution.png` / `.pdf` | 79,976 | Easy vs Hard F1 distribution (violin plot); descriptive only, no significance stars. | Results or Discussion — supports C04, C05 (descriptive) |
| Figure D | `Figure_D_A6A7_ablation.png` / `.pdf` | 88,283 | A6/A7 ablation bar chart (V00 Control / V10 A6 / V01-R1 A7 / V11-R1 A6+A7); descriptive. | Ablation (Results (Ablation) or Limitations) — supports C06, C07 |

Status: all four deterministic, paper-ready, PNG+PDF. These are the primary visual assets for the Phase 19B ranking-failure section and the A6/A7 negative-ablation section.

---

## Section 3: Phase17B / Phase17C Visualization Assets

Source directories:
- `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十七B_上游失败归因与候选Oracle审计/visualizations/`
- `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十七C_候选排序修复开发/` (no charts found)

| Asset | Path (relative to Phase17B visualizations/) | Size (bytes) | Content / use |
|---|---|---|---|
| instance_audit_DouBanLv2_0039.jpg | `instance_audit_DouBanLv2_0039.jpg` | 290,525 | Instance-level candidate audit for failure frame 0039 (raw instances + GT bbox + overlap check). Candidate supplementary panel. |
| instance_audit_DouBanLv2_0040.jpg | `instance_audit_DouBanLv2_0040.jpg` | 367,268 | Same audit for frame 0040. |
| instance_audit_DouBanLv2_0041.jpg | `instance_audit_DouBanLv2_0041.jpg` | 774,826 | Same audit for frame 0041 (9 raw instances). |
| map_gt_DouBanLv2_0039..0041.jpg | `map_gt_DouBanLv2_0039.jpg`, `map_gt_DouBanLv2_0040.jpg`, `map_gt_DouBanLv2_0041.jpg` | 203,156 / 202,793 / 190,324 | GT mask maps for the 3 failure frames. |
| map_pred_DouBanLv2_0039..0041.jpg | `map_pred_DouBanLv2_0039.jpg`, `map_pred_DouBanLv2_0040.jpg`, `map_pred_DouBanLv2_0041.jpg` | 210,605 / 210,986 / 195,698 | R0 predicted (wrong) masks for the 3 failure frames. |
| map_gt / map_pred BaiZhang & DouBanLv2 non-fail frames | `map_gt_BaiZhang_0033..0037.jpg`, `map_gt_DouBanLv2_0037/0038.jpg`, `map_pred_*` equivalents | 195,682–272,568 | GT/pred mask maps for all 10 DEV10 frames (context for failure-figure selection). |
| temporal_strip_DouBanLv2_0037-0041.jpg | `temporal_strip_DouBanLv2_0037-0041.jpg` | 402,071 | Temporal strip across DouBanLv2 0037→0041 (contextualising the sudden F1 dropout). Candidate supplementary/mechanism panel. |

Phase17C: **no policy-comparison charts were produced** (no PNG/JPG/PDF/SVG in 阶段十七C_候选排序修复开发/). The Phase18 Figure B/C cover the ranking-policy visual needs, or a comparator chart can be regenerated from `Phase17C_policy_comparison.csv`.

Related Phase17A gallery (behavioural panels, GT10/TEST32 executions) at `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十七A_10GT困难样本Pilot与无GT测试审计/07_gallery/trackA/` and `trackB/` (`trackA_contact_sheet.jpg`, `trackB_contact_sheet.jpg`): supplementary-only, descriptive.

---

## Section 4: Experimental Evidence Panels in `05-图件与论文映射/`

Base: `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/05-图件与论文映射/`

| Asset | Absolute path | Size (bytes) | Content / paper mapping |
|---|---|---|---|
| Fig_E1_metric_bars.png | `.../实验一_视觉基础模型横向对比/figures/Fig_E1_metric_bars.png` | 85,533 | VFM family metric bars (RAP-FSAM3 vs SAM3/CLIPSeg/Grounded-SAM et al.) on annotated frames. Maps to Results §3.1 / Table 3 (vfm-comparison). PDF/SVG also present. |
| Fig_E1_failure_cases.png | `.../实验一_视觉基础模型横向对比/figures/Fig_E1_failure_cases.png` | 4,449,661 | VFM failure-case panels. Supplementary for §3.1. PDF also present. |
| Fig_E2_prompt_P2_vs_P4.png | `.../实验二三_RAP-FSAM3图板/figures/Fig_E2_prompt_P2_vs_P4.png` | 5,047,204 | Prompt P2 vs P4 difference board (KongQueZhuYu), showing prompt-instability motivation. Maps to Results §3.2 / Table 4 (prompt robustness). PDF/SVG also present. |
| Fig_E3_ablation_A0_vs_A2A5.png | `.../实验二三_RAP-FSAM3图板/figures/Fig_E3_ablation_A0_vs_A2A5.png` | 4,367,470 | RAP-FSAM3 module ablation A0 vs A2/A5 panels. Maps to Results §3.3 / Table 5 (rap-ablation). PDF/SVG also present. |
| Fig_E3v2_A1s_A5c_mechanisms.png | `.../实验三_RAP-FSAM3v2新版图板/figures/Fig_E3v2_A1s_A5c_mechanisms.png` | 5,800,792 | A1s (semantic-gated selection) + A5c (geometry-consistency correction) mechanism panels. Maps to §3.3 / Table 5 (mechanism illustrations). PDF/SVG/TIF also present. |
| Fig_E5_error_distribution.png | `.../实验五_掩膜到表型误差敏感性/figures/Fig_E5_error_distribution.png` | 394,847 | Mask-to-phenotype error distribution. Maps to Results §3.5 / Discussion §4.4 (boundary-sensitive traits). PDF/SVG also present. |
| Fig_E5_trait_error_profile.png | `.../实验五_掩膜到表型误差敏感性/figures/Fig_E5_trait_error_profile.png` | 771,748 | Per-trait error profile (height/canopy/leaf-length/leaf-width). Maps to §3.5 / Table 8 (phenotyping) + Table 6 (injection-ablation). PDF/SVG also present. |

Also inventoried (material packs, for figure REBUILD of Fig1/Fig2/Fig4):
- `Fig1_总体框架素材包/`: 01_input_acquisition (raw multiview + COLMAP crop), 02_rap_fsam3_vfm (P1–P5/prompt candidates, selected candidate mask, semantic gate box, FFT quality), 03_2dgs_optimization (foreground track, RGB supervision GT, post-pruning), 04_mesh_phenotyping (TSDF mesh preview, trait panels), 99_optional_alternatives.
- `Fig2_RAP-FSAM3方法图素材包/`: 01_panel_a_multi_prompt_candidates (P1/P2/P4 candidate masks), 02_panel_b_semantic_gate_selection, 03_panel_c_posneg_prompt_refinement, 04_panel_d_geometry_correction_final_prior, 99_optional_alternatives, source_data.
- `Fig4_Prompt鲁棒性与语义偏置分析素材包/`: 01_grid_ordered_assets (~18 per-sample grids), 02_by_sample, 03_source_data, 04_text_and_labels.
- `实验四_2DGS先验注入位置消融/figures/Fig_E4_prior_injection_B0_B5_metrics.png` (935 KB PDF also present): prior-injection-position ablation, maps to Results §3.4 / Table 6 (injection-ablation).
- `仙客来_RAP-FSAM3_GT1机制诊断/`: GT1 mechanism diagnostic boards (legacy; supports the A1s diagnostic narrative in the paper mapping).

Note: `实验一_视觉基础模型横向对比/figures/` contains `Fig_E1_metric_bars` but no `Fig_E1_metric_bars_single` variant; nine-method full comparison (incl. SEEM, Grounded-SAM2, SAM2 oracle, U-Net, DeepLabv3+ lite) is stated in the manuscript as supplementary material and is supported by the experiment-1 figure assets.

---

## Section 5: Recommended Figure Plan (PROVISIONAL — figure_plan not yet written)

> Placeholder note: the authoritative figure layout plan lives in the upcoming `Phase19A_figure_plan.md` (WP6). This section gives the asset→figure mapping recommendation for that plan.

| New paper figure (proposed) | Source asset(s) | Paper section | Status |
|---|---|---|---|
| Fig X (NEW, ranking-failure mechanism) | `Phase18 Figure_A_failure_mechanism.png` (+ Phase17B map_gt/map_pred 0039–0041 as supplementary) | Results §3.3 | READY (Phase18) |
| Fig Y (NEW, R1 ranking illustration) | `Phase18 Figure_B_ranking_illustration.png` | Results §3.3 | READY (Phase18) |
| Fig Distribution (Easy vs Hard) | `Phase18 Figure_C_easy_vs_hard_distribution.png` | Results or Discussion | READY (Phase18) |
| Fig A6/A7 negative ablation | `Phase18 Figure_D_A6A7_ablation.png` | Ablation / Limitations | READY (Phase18) |
| Fig 5/6 keep (2DGS + M2M3-GO) | current `Fig5.png` / `Fig6.png` (rebuild panels from `Fig1_总体框架素材包/03_2dgs_optimization`) | Methods §3 | KEEP, optionally rebuild |
| Fig 4 keep (prompt sensitivity) | current `Fig4.png` (+ `实验四.../Fig_E2_prompt_P2_vs_P4.png`, `Fig4_Prompt鲁棒性...` grids) | Results §3.2 | KEEP, optionally swap/rebuild |
| Fig 1/2 keep (workflow + RAP-FSAM3 method) | current `Fig1.png`/`Fig2.png` (rebuild from `Fig1_总体框架素材包/`, `Fig2_RAP-FSAM3方法图素材包/`) | Methods §2 | KEEP, optionally rebuild |
| Fig reconstruction / phenotyping | current `Fig8.png` / `Fig9.png` (+ `Fig_E5_trait_error_profile.png`) | Results §3.5 | KEEP |

When the figure plan is written it should (a) reuse the four Phase18 figures as the core of the new ranking-failure and A6/A7-negative sections, (b) decide KEEP vs REBUILD for Fig1/2/4 using the material packs, and (c) keep all significance-restricted labels (no stars, descriptive-only per Phase18 conventions).

---

## Verification note

All paths in Sections 1–4 were confirmed present on disk (ls/find) at inventory time. Broken-path risk: Phase17C has no charts (noted above); Phase14 `visualizations/` is an empty directory (no Phase14 charts available — Figure D from Phase18 supersedes them); `Phase16 qa_visualizations/` is empty (GT QA views not exported as standalone panels).