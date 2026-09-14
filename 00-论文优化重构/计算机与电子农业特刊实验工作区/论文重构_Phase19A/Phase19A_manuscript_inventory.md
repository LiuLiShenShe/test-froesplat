# Phase 19A Manuscript Inventory

Generated: 2026-09-14 | Repo: /data/fj/F2DMAS

---

## 1. Primary Main Manuscript

**File:** `00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat__Foreground_Object_2D_Gaussian_Splatting_via_Semantic_Mask_Guided_Task_Reformulation_for_Plant/ForeSplat.tex`

| Field | Value |
|-------|-------|
| Size | 82,415 bytes (80.5 KB) |
| Last modified | 2026-08-25 12:25 |
| Document class | `cas-sc` (Elsevier CAS single-column) |
| Journal target | Computers and Electronics in Agriculture (CompAg), special issue on foundation models in agriculture |
| Title | *ForeSplat: Vision-Foundation-Model-Guided Foreground-Object 2D Gaussian Splatting for Low-Cost 3D Plant Phenotyping* |
| Short title | VFM-guided ForeSplat for plant phenotyping |
| Authors | Jian Fang [1,2,3,4], Nengfu Xie* [2,3,4], Yane Duan* [1], Jingchao Fan [2,3,4], Xiaoli Wang [2,3,4], Hailong Liu [2,3,4], Yue Wang [2,3,4], Hao Wu [2,3,4], Zhibo Meng [2,3,4], Xin Wang [2,3,4], Rui Man [2,3,4] |
| Affiliations | [1] College of Intelligent Science and Engineering, Beijing University of Agriculture; [2] Agricultural Information Institute, CAAS; [3] National Agricultural Science Data Center; [4] Data Hub, Chinese Agrosystem Long-Term Observation Network |
| Keywords | vision foundation model, 2D Gaussian Splatting, foreground-object reconstruction, plant phenotyping, RGB imaging |
| Figures in tex | 9 (`Fig1.png` through `Fig9.png`) |
| Key header comment | "Revised ForeSplat manuscript for the COMPAG special issue on foundation models in agriculture. Main revision: the manuscript has been repositioned from an FSAM3/engineering workflow paper to a VFM-guided foreground-prior adaptation and foreground-object 2DGS reconstruction paper." |
| Role | **MAIN** |

**CONFIRMED: PRIMARY TARGET FOR PHASE 19B REWRITE.** This is the current and only active primary manuscript. No other file in the repo shows evidence of being a more recent or more authoritative primary target.

---

## 2. All .tex Files

| # | Path | Size | Last modified | Role | Description |
|---|------|------|---------------|------|-------------|
| 1 | `f2dmas_elsarticle.tex` | 55,659 bytes | 2026-08-27 18:00 | **EARLIER_VERSION** | Early Chinese-language elsarticle draft. Title: "F2DMAS: Frequency-aware Foreground 2D Gaussian Splatting with Multi-modal Mask Generation for Automated Cross-species Plant Phenotyping." Uses `\documentclass[preprint,12pt]{elsarticle}` with CJKutf8/ctex. Journal field is TODO. Author fields are TODO. This is the earliest structured manuscript in the lineage. |
| 2 | `elsarticle-template-num.tex` | 8,462 bytes | -- | **TEMPLATE** | Bare Elsevier elsarticle numbered-style template. Standard example file from the elsarticle LaTeX bundle. Contains no manuscript content. |
| 3 | `00-论文优化重构/论文cc/F2DMAS_manuscript.tex` | 107,637 bytes | 2026-08-27 18:00 | **EARLIER_VERSION** | Bilingual F2DMAS manuscript (English primary + Chinese). Targets CompAg. Uses plain `article` class with `fontspec` (NotoSerifCJKtc). Section heading: "F2DMAS: Frequency-aware Foreground 2D Gaussian Splatting with Multi-modal Mask Generation for Automated Cross-species Plant Phenotyping." ~9,000 words, 8 figures, 7 tables. Superseded by ForeSplat.tex but contains the most complete middle-stage content. |
| 4 | `00-论文优化重构/论文cc/fontsetup.tex` | 235 bytes | 2026-08-27 18:00 | **SUPPLEMENTARY** | Companion to `F2DMAS_manuscript.tex`. Sets `NotoSerifCJKtc` main font and `DejaVuSansMono` mono font for XeLaTeX compilation. |
| 5 | `00-论文优化重构/论文2026-05-28/latex格式/cas-sc-template.tex` | 83,797 bytes | 2026-08-27 17:59 | **EARLIER_VERSION** | Revised ForeSplat manuscript built on the CAS-SC template. Same title, authors, and structure as the current `ForeSplat.tex` primary. Key differences from primary: affiliation numbering differs (author[1,3,4] vs author[2,3,4] for some members); uses `\usepackage{lmodern}` (added). This is the intermediate revision between the F2DMAS manuscript and the final ForeSplat primary. |
| 6 | `00-论文优化重构/论文2026-05-28/latex格式/cas-sc-template.original.tex` | 4,841 bytes | 2026-08-27 17:59 | **TEMPLATE** | Pristine Elsevier CAS-SC template (Copyright 2019-2024). Contains empty placeholder fields (`\title{}`, `\author[1]{}`, empty affiliation blocks). The "pre-revision baseline" that `cas-sc-template.tex` was built on top of. |
| 7 | `00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat__.../ForeSplat.tex` | 82,415 bytes | 2026-08-25 12:25 | **MAIN** | The primary manuscript. Full details in Section 1 above. |
| 8 | `00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat__.../ForeSplat-highlights.tex` | 693 bytes | 2026-08-25 12:25 | **SUPPLEMENTARY** | Highlights file for the ForeSplat manuscript. Five bullet points: RAP-FSAM3 adaptation, prompt robustness analysis, ForeSplat optimization, leakage energy reduction (1.2201 to 0.0190), virtual trait agreement (R^2 up to 0.988). Separate standalone .tex compilation. |

---

## 3. Bibliography

| # | Path | Size | Identity | Role |
|---|------|------|----------|------|
| 1 | `00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat__.../cas-refs.bib` | 21,713 bytes | Active bibliography for ForeSplat primary. Key-numbered entries (ref1, ref2, ...). | **BIBLIOGRAPHY** |
| 2 | `00-论文优化重构/论文2026-05-28/latex格式/cas-refs.bib` | 21,713 bytes | Byte-identical copy of the above (diff confirms "IDENTICAL"). Used by the intermediate revision. | **BIBLIOGRAPHY** |
| 3 | `instant-ngp/docs/assets/mueller2022instant.bib` | 708 bytes | Unrelated third-party bibliography bundled with the instant-ngp codebase. Not used by any manuscript. | **OTHER** |

---

## 4. PDF Manuscript Versions

| # | Path | Size | Last modified | Role | Description |
|---|------|------|---------------|------|-------------|
| 1 | `F2DMAS应用版/F2DMAS_revised_phenotypic_text.pdf` | 12.6 MB | 2026-08-27 18:00 | **EARLIER_VERSION** | Compiled PDF of the F2DMAS application manuscript (post-revision). Accompanied by `.docx` (17.2 MB) and parsed text extract in `.parsed/`. |
| 2 | `00-论文优化重构/论文2026-05-28/latex格式/cas-sc-template.original.pdf` | 100 KB | 2026-08-27 17:59 | **TEMPLATE** | Pre-compiled blank CAS-SC template PDF. Corresponds to `cas-sc-template.original.tex`. |
| 3 | `00-论文优化重构/论文cc/F2DMAS_manuscript.docx` | 55 KB | 2026-08-27 18:00 | **EARLIER_VERSION** | DOCX export of the bilingual F2DMAS manuscript from the 论文cc stage. |

---

## 5. Markdown Manuscript Drafts

All located in `00-论文优化重构/论文cc/`:

| # | File | Size | Role | Description |
|---|------|------|------|-------------|
| 1 | `02_manuscript_english_draft.md` | 81,585 bytes | **DRAFT** | English-only markdown draft of the F2DMAS manuscript. |
| 2 | `02_manuscript_english_draft copy.md` | 81,585 bytes | **DRAFT** | Exact duplicate of the English draft (identical size). |
| 3 | `02_manuscript_chinese_draft.md` | 70,671 bytes | **DRAFT** | Chinese-language markdown draft of the F2DMAS manuscript. |
| 4 | `02_manuscript_chinese_draft copy.md` | 95,942 bytes | **DRAFT** | Chinese draft copy; larger than the primary (25 KB more content). Likely a prior or expanded version. |
| 5 | `02_manuscript_bilingual_draft.md` | 100,183 bytes | **DRAFT** | Bilingual (English+Chinese) combined markdown draft. Largest of the markdown drafts. |
| 6 | `02_manuscript_rewritten_from_refs_en.md` | 53,430 bytes | **DRAFT** | English manuscript draft rewritten based on reference analysis. Shorter, more focused version. |
| 7 | `02_manuscript_rewritten_from_refs_zh.md` | 44,537 bytes | **DRAFT** | Chinese manuscript draft rewritten based on reference analysis. |

---

## 6. Figures

### 6a. ForeSplat Primary Figures (Fig1-Fig9)

All in `00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat__.../`:

| File | Size | Content (from tex reference) |
|------|------|------------------------------|
| `Fig1.png` | 491 KB | Overall framework of RAP-FSAM3-guided ForeSplat |
| `Fig2.png` | 494 KB | RAP-FSAM3: reconstruction-aware prompt-adaptive foreground-prior generation |
| `Fig3.png` | 477 KB | Qualitative comparison of VFM-based plant foreground priors |
| `Fig4.png` | 272 KB | Prompt robustness and semantic-bias analysis |
| `Fig5.png` | 456 KB | VFM-prior injection into foreground-object 2DGS optimisation |
| `Fig6.png` | 471 KB | Training-time VFM-prior injection is not equivalent to input masking or post hoc pruning |
| `Fig7.png` | 540 KB | Workflow-level reconstruction and mesh outputs |
| `Fig8.png` | 691 KB | Manual-virtual phenotypic measurement agreement |
| `Fig9.png` | 919 KB | Manual-virtual trait scatter plots (height, canopy, leaf) |

Role: **FIGURE** (all 9 primary manuscript figures)

### 6b. Intermediate Revision Figure Assets (论文2026-05-28)

Located in `00-论文优化重构/论文2026-05-28/latex格式/`:

Named figures with long descriptive names (predecessor versions of Fig1-Fig9):
- `Fig1-Overall framework of RAP-FSAM3-guided ForeSplat.png`
- `Fig2-RAP-FSAM3_ reconstruction-aware prompt-adaptive foreground-prior generation.png`
- `Fig3-Qualitative comparison of VFM-based plant foreground priors.png`
- `Fig4-Prompt robustness and semantic-bias analysis.png`
- `Fig5-VFM-prior injection into foreground-object 2DGS optimisation.png`
- `Fig6-Training-time VFM-prior injection is not equivalent to input masking or post hoc pruning.png`
- `Fig7-Workflow-level reconstruction and mesh outputs.png`
- `Fig8-Manual--virtual phenotypic measurement agreement.png`
- `fig1.png` through `fig9.png` (lowercase short-name copies)

Role: **FIGURE** (intermediate revision; predecessor versions)

Subdirectory figure asset panels:
- `fig3a_rendered_assets/` -- Rendered panels: ForeSplat, standard 2DGS, 3DGS+FSAM3, COLMAP, TSDF mesh (~10 PNGs + manifest)
- `fig4_assets/` -- Ablation panels: baseline, input_mask, fg_rgb_sup, fg_track_init, post_hoc_prune, alpha_bg, A6+M4 (~10 PNGs + CSV)
- `fig6_assets/` -- Metrics CSVs and manifest for injection-position figure
- `fig7_assets/` -- CSVs for view coverage, view quality strategies, quality-compactness tradeoff
- `fig9/fig9_panels/` -- Scatter plots (plant_height, canopy_width, leaf_length, leaf_width), residual distribution, combined draft PDF/PNG, summary metrics PNG

Role: **FIGURE** (component assets for intermediate revision figures)

### 6c. Figure Design Source Files (论文2026-05-28/figures/)

- `.drawio` source files for Fig1 (method overview), Fig2 (2DGS algorithm sketch), Fig5 (ablation evidence), Fig6 (representative samples), Fig7 (view quality strategy), Fig8 (TSDF mesh)
- `fig4_data_table_current.csv`, `fig4_candidate_renders_contact_sheet.png`, `fig1b_2dgs_algorithm_sketch.md`, `fig4_data_and_layout_plan.md`

Role: **FIGURE** (design sources and planning documents)

### 6d. Phase18 Audit Figures

Located in `00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十八_论文实验收束与最终审计/figures/`:

| File | Role |
|------|------|
| `Figure_A_failure_mechanism.pdf` | **FIGURE** |
| `Figure_A_failure_mechanism.png` | **FIGURE** |
| `Figure_B_ranking_illustration.pdf` | **FIGURE** |
| `Figure_B_ranking_illustration.png` | **FIGURE** |
| `Figure_C_easy_vs_hard_distribution.pdf` | **FIGURE** |
| `Figure_C_easy_vs_hard_distribution.png` | **FIGURE** |
| `Figure_D_A6A7_ablation.pdf` | **FIGURE** |
| `Figure_D_A6A7_ablation.png` | **FIGURE** |

### 6e. Experimental Figure Panels (05-图件与论文映射)

Located in `00-论文优化重构/计算机与电子农业特刊实验工作区/05-图件与论文映射/`:
- `实验一_视觉基础模型横向对比/figures/` -- Fig_E1_metric_bars.png, Fig_E1_failure_cases.png
- `实验二三_RAP-FSAM3图板/figures/` -- Fig_E2_prompt_P2_vs_P4.png, Fig_E3_ablation_A0_vs_A2A5.png
- `实验三_RAP-FSAM3v2新版图板/figures/` -- Fig_E3v2_A1s_A5c_mechanisms.png
- `实验五_掩膜到表型误差敏感性/figures/` -- Fig_E5_error_distribution.png, Fig_E5_trait_error_profile.png
- `Fig4_Prompt鲁棒性与语义偏置分析素材包/01_grid_ordered_assets/` -- ~18 per-sample grid images (CaoMei1, ChangShouHua2, DouBanLv1 with multiple prompts)

Role: **FIGURE** (experimental evidence panels, not directly in manuscript but supporting material)

### 6f. Trait Measurement PDFs (09-PNG/20260306-pic)

Located in `09-PNG/20260306-pic/`:

| File | Size | Role |
|------|------|------|
| `Plant_Height.pdf` | 31 KB | **FIGURE** |
| `Plant_Height.png` | 191 KB | **FIGURE** |
| `Canopy_Width.pdf` | 32 KB | **FIGURE** |
| `Canopy_Width.png` | 196 KB | **FIGURE** |
| `Leaf_Length.pdf` | 33 KB | **FIGURE** |
| `Leaf_Length.png` | 221 KB | **FIGURE** |
| `Leaf_Width.pdf` | 33 KB | **FIGURE** |
| `Leaf_Width.png` | 222 KB | **FIGURE** |
| `植株数据.xlsx - Sheet1.csv` | 2.2 KB | **SUPPLEMENTARY** (raw trait data) |

Role: **FIGURE** (manual-vs-virtual trait validation scatter plots, precursor to Fig9)

---

## 7. Supplementary / Review-Response Files

### 7a. Real Review Files (F2DMAS application review, CompAg)

Located in `F2DMAS应用版/审稿意见/`:

| # | File | Size | Role | Description |
|---|------|------|------|-------------|
| 1 | `审稿人1.md` | 2,408 bytes | **REVIEW_MATERIAL** | Reviewer 1 original comments |
| 2 | `审稿人1_中文翻译.md` | 12,946 bytes | **REVIEW_MATERIAL** | Reviewer 1 Chinese translation |
| 3 | `审稿人1_逐条回复与修稿清单.md` | 24,819 bytes | **REVIEW_MATERIAL** | Reviewer 1 point-by-point response and revision checklist |
| 4 | `审稿人2.md` | 1,949 bytes | **REVIEW_MATERIAL** | Reviewer 2 original comments |
| 5 | `审稿人2_中文翻译.md` | 9,889 bytes | **REVIEW_MATERIAL** | Reviewer 2 Chinese translation |
| 6 | `R1.2_Table6消融实验方案.md` | 10,734 bytes | **REVIEW_MATERIAL** | Ablation experiment plan for Table 6 (Reviewer 1, point 2) |
| 7 | `师兄给的模版-编辑回信.docx` | 35,410 bytes | **OTHER** | Template letter from colleague for editor response |
| 8 | `F2DMAS-审稿回复信_按模板替换.docx` | 19,076 bytes | **REVIEW_MATERIAL** | F2DMAS review response letter (filled template) |
| 9 | `F2DMAS-小修.docx` | 17,258,249 bytes | **EARLIER_VERSION** | F2DMAS minor revision manuscript (full DOCX with figures) |

### 7b. Simulated Review / Self-Review Files (论文cc)

Located in `00-论文优化重构/论文cc/`:

| # | File | Size | Role | Description |
|---|------|------|------|-------------|
| 1 | `05_peer_review_package.md` | 23,616 bytes | **REVIEW_MATERIAL** | Simulated peer review package |
| 2 | `06_response_to_reviewers.md` | 9,136 bytes | **REVIEW_MATERIAL** | Simulated reviewer response |
| 3 | `07_re_review_verification.md` | 3,925 bytes | **REVIEW_MATERIAL** | Simulated re-review verification checklist |
| 4 | `08_final_integrity_check.md` | 2,187 bytes | **REVIEW_MATERIAL** | Simulated final integrity check |

---

## 8. Manuscript Lineage Summary

The manuscript has evolved through four major stages, with five naming conventions reflecting progressive conceptual refinement:

1. **F2DMAS (early Chinese draft)** -- `f2dmas_elsarticle.tex` (55 KB elsarticle, Chinese, journal=TODO, authors=TODO). First structured manuscript, titled "F2DMAS: based on frequency-spatial feature decoupling and explicit geometric Gaussians." The project codename F2DMAS (Frequency-aware foreground 2DGS with Adaptive Mask Selection) originated here. Uses elsarticle preprint class with CJK/ctex for Chinese typesetting.

2. **F2DMAS application version** -- `F2DMAS应用版/` directory containing `F2DMAS_revised_phenotypic_text.pdf` (12.6 MB), `.docx` (17.2 MB), and the real CompAg reviewer comments (`审稿意见/`). Title: "F2DMAS: A Smartphone Video-Based 3D Phenotyping Workflow for Potted Plants." This version received actual peer review from two CompAg reviewers. The segmentation module within F2DMAS was called **FSAM3** (FFT screening + SAM3 segmentation). The reviewer response letter and ablation experiment plans in this directory represent the first real external feedback.

3. **F2DMAS bilingual manuscript (论文cc)** -- `F2DMAS_manuscript.tex` (108 KB, bilingual English+Chinese, `article` class) with 7 markdown drafts. Targeted CompAg directly. This was the most content-complete middle stage, containing both languages and detailed experimental descriptions. Accompanied by simulated self-review materials (files 05-08).

4. **ForeSplat CAS-SC manuscript (current primary)** -- The manuscript underwent a major repositioning: from a F2DMAS/FSAM3 engineering-workflow paper to a VFM-guided foreground-prior adaptation and foreground-object 2DGS reconstruction paper. The naming evolved: **F2DMAS** -> **FSAM3** (segmentation module retained) -> **RAP-FSAM3** (Reconstruction-Aware Prompt-adaptive FSAM3) -> **ForeSplat** (the new overall method name) -> **M2M3-GO** (a lightweight primitive-level refinement layer for 2DGS Gaussian keep/densify/prune decisions, nested within ForeSplat). An intermediate revision exists at `论文2026-05-28/latex格式/cas-sc-template.tex` (84 KB, `cas-sc`, same title/authors as the primary, with minor affiliation numbering differences and `lmodern` package). The final primary is `ForeSplat.tex` (82 KB, `cas-sc`) in the CompAg special-issue working area, dated 2026-08-25.

**Naming evolution chain:**
```
F2DMAS (project codename)
  -> FSAM3 (FFT + SAM3 segmentation module)
    -> RAP-FSAM3 (Reconstruction-Aware Prompt-adaptive foreground prior generator)
      -> ForeSplat (VFM-guided foreground-object 2DGS workflow)
        -> M2M3-GO (primitive-level Gaussian refinement layer, sub-component of ForeSplat)
```

---

## Confirmation

**PRIMARY TARGET FOR PHASE 19B REWRITE: `ForeSplat.tex` (CAS-SC, CompAg special issue on foundation models in agriculture)**

- Path: `/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat__Foreground_Object_2D_Gaussian_Splatting_via_Semantic_Mask_Guided_Task_Reformulation_for_Plant/ForeSplat.tex`
- Document class: `cas-sc`
- Journal: Computers and Electronics in Agriculture (CompAg), special issue
- 82,415 bytes | 9 figures | 11 authors | 4 affiliations | Complete manuscript with abstract, methods, results, discussion, and conclusion sections

No evidence was found that any other file supersedes this as the current primary manuscript target. The intermediate revision (`cas-sc-template.tex`) has slightly different affiliation numbering (minor) and identical conceptual content, confirming `ForeSplat.tex` as the authoritative latest version.
