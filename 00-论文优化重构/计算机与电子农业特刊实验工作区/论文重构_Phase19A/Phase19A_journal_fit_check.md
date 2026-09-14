# Phase 19A — WP7D CompAg Journal-Fit Check (ForeSplat Manuscript)

**Generated:** 2026-09-14 | **Auditor:** WP7D (research-audit agent) | **Target:** *Computers and Electronics in Agriculture* (CompAg), special issue on vision foundation models (VFM) in agriculture
**Manuscript:** *ForeSplat: Vision-Foundation-Model-Guided Foreground-Object 2D Gaussian Splatting for Low-Cost 3D Plant Phenotyping* (target class `cas-sc`, 9 figures current, 8 tables / 6 figures planned for Phase 19B)
**Inputs:** `Phase19A_manuscript_inventory.md`, `Phase19A_core_preservation.md`, `Phase19A_proposed_contributions.md`, `Phase19A_new_storyline.md`, `Phase19A_claim_evidence_matrix.csv`, `Phase19A_reviewer_comments_raw.md`, `Phase19A_reviewer_issue_matrix.csv`, `Phase19A_resolution_status.md`, `Phase19A_main_supplement_plan.md`, `compag特刊增修模块/*.md` (3 files), web search on CompAg scope/format (2026-09-14).

> **Reviewer-ID note.** The WP7D brief cites "B-R3-C03 (cost analysis), B-R3-C04 (memory metrics)". The canonical issue matrix maps these as: cost analysis = **B-R3-C04** (per-plant wall-clock time), memory metrics = **B-R3-C01** (GPU MB / FPS absolutes); **B-R3-C03** = prompt-design rationale. This document uses the canonical IDs throughout.

---

## 1. CompAg Scope Alignment Scorecard

CompAg journal context (verified by search, 2026-09-14): CAS1区 (agriculture multidisciplinary / CS interdisciplinary applications), Top journal, IF ~6.8–10.3 depending on year source ([fabiao.com](https://www.fabiao.com/sci/01681699.html), [booksci](https://m.booksci.cn/journal/50627.htm)); monthly Elsevier journal publishing original papers on computer/electronic solutions to agricultural problems — agronomy, horticulture, greenhouse, post-harvest, AI/machine-vision/robotics applied to plants and animals ([xueshu.com scope](https://www.xueshu.com/sci/41455/07.html)). **Critical scope signal:** CompAg explicitly states that "first-time application of an existing technology to a specific crop does not qualify as innovation in computer/electronics" ([peipusci](https://m.peipusci.com/news/366.html)) — the paper must sell an algorithmic adaptation, not a re-application. The special-issue framing docs (compag特刊增修模块) reinforce this: *frozen VFM + agriculture-specific adapter + reconstruction-aware optimization* is the required SI storyline; SAM3 fine-tuning and wholesale IPENS porting are explicitly discouraged.

| # | CompAg typical expectation | Where met in current plan | Evidence | Gap & fix for Phase 19B |
|---|---|---|---|---|
| 1 | **Agricultural applicability** (application-driven problem, measurable ag outcome) | Low-cost smartphone-video 3D phenotyping workflow for potted plants; 4 traits across 21 plants / 63 leaves, R² comparable to manual (P07); throughput ~1.5–2.0 min/plant acquisition, ~55 min/plant processing (A2-C04, B-R3-C04) | `core_preservation.md` §2 (ag value), `proposed_contributions.md` C4, `new_storyline.md` steps 1, 13 | **GAP (partially open):** ag value is currently *embedded* but the paper leads with segmentation/3D methods (see §2). Fix: lead Discussion (D5) and a dedicated phenotyping paragraph with the agronomic-trait workflow; state the horticulture/ornamental sector target explicitly. |
| 2 | **Real field / agronomy relevance** (plausible deployment context) | Potted-plant (greenhouse / ornamental) setting; 10 species, architectural diversity framing (B-EIC-C01 reframe); acquisition-time and hardware context (B-R3-C04); scale recovery via pot diameter | `new_storyline.md` steps 1, 15; `resolution_status.md` B-EIC-C01 (PARTIAL) | **GAP:** no field data (controlled potted-plant acquisition only). Fix: honor the honest scope — "controlled potted-plant setting" stated up front (Introduction §1.1), and frame as the first low-cost rung of a field-deployable ladder (Limitations L4 + future-work). Do NOT claim field generality. |
| 3 | **Algorithm novelty** (not "applied existing tech to a crop") | Three method contributions: RAP-FSAM3 reconstruction-aware VFM adapter (C1), ranking-failure diagnosis + R1 fix (C2, strongest NEW), foreground-object 2DGS + M2M3-GO (C3); VFM→2DGS constraint injection is a mechanism, not a re-application | `proposed_contributions.md` C1–C3; `compag特刊增修模块/论文增改表述.md` (SI main line: "how VFM priors enter the 3D Gaussian reconstruction objective"); `new_storyline.md` steps 5–11 | **GAP (partially open):** B-R2-C01 requires an explicit statement of which components are *novel combinations* vs *novel mechanisms* — must be stated in Introduction/Contribution list. Fix: one explicit sentence in Introduction §1.6 partitioning the three contributions accordingly. This directly satisfies the CompAg "not merely re-application" gate. |
| 4 | **Reproducibility** (hardware, software, hyperparameters, code) | S4 plan: env versions (Python/CUDA/PyTorch/COLMAP/SAM3 commit), S20 hyperparams frozen set, COLMAP config, hardware incl. RTX 3090 timing, 150-test summary (B-R3-C02, B-R2-C04) | `main_supplement_plan.md` S4; `resolution_status.md` B-R3-C02 RESOLVED, B-R2-C04 RESOLVED | **GAP:** code repository not yet public (B-R3-C02 item 6). Fix: commit to code availability in Data Availability + Methods §2.9; even a "cleaned code to be released upon acceptance" statement is required. |
| 5 | **Dataset transparency** (data availability, protocol, GT provenance) | 5 datasets with explicit roles and GT status: frozen P6 (21 frames), 20-sequence workflow, GT10/DEV10 (NOT held-out, descriptive), TEST32 (unlabeled, NO accuracy), Easy21 (ablation/safety); dataset registry | `main_supplement_plan.md` M2, S1–S3; `new_storyline.md` step 15 | **GAP (minor):** botanical nomenclature provisional (B-R1-C01, Supplementary Table S1 placeholder). Fix: keep "pending specialist confirmation at proof stage" label; species table is non-blocking. GT10-not-held-out honesty already in plan — retain verbatim. |

**Overall fit verdict: STRONG FIT for the CompAg VFM special issue after the Phase 19B gaps are closed.** The paper's core line (frozen VFM + agriculture-specific adapter + reconstruction-aware 2DGS optimization → downstream phenotyping) is *precisely* the SI framing that the special-issue guidance prescribes, and the ranking-failure/R1 analysis provides an algorithmically novel hook that distinguishes it from a re-application paper. The three substantive gaps — (1) lead with ag value, (2) explicit novelty-partition statement, (3) public code — are all closable within Phase 19B without new experiments. The honest-scope discipline already enforced by the claim_audit (no "generalizes", no "SOTA", GT10-not-held-out) is itself an asset for CompAg, whose reviewers punish overclaiming.

---

## 2. Positioning Risk: Algorithm-Forward Drift → Pure-CV-Benchmark Reading

**Risk.** The manuscript's natural center of gravity is segmentation (RAP-FSAM3) + 3D reconstruction (ForeSplat) + ranking analysis (R1). Five of nine current figures are method/segmentation-facing (Fig1–Fig6); the phenotyping payoff sits in Fig7–Fig9. A CompAg reviewer could conclude this is a CV-benchmark paper wearing an agricultural hat — the exact failure mode CompAg's "not merely re-application" gate is designed to reject.

**Three concrete re-framing moves to anchor agricultural value** (each already partially present in the Phase 19A docs; Phase 19B must promote them to structural prominence):

1. **Lead the Discussion with phenotyping downstream, not with ranking/3D claims.** Open Discussion with a "what does this mean for measuring plants" section: the 4-trait workflow (pot-diameter scale recovery, virtual vs manual R², MAE in cm), which traits are measurement-usable now (height, canopy, leaf length) and which boundary-sensitive trait stays open (leaf width, A1-C03). This re-orders the reader's takeaway from "algorithm paper" to "phenotyping result with the algorithm as enabler." **Already citable in:** `new_storyline.md` step 13 (phenotypic agreement) + `main_supplement_plan.md` §1.10 D5; `proposed_contributions.md` C4.

2. **Frame the ranking-failure story as an agricultural-image failure, not a CV metric artifact.** Present the DouBanLv2 0039/0040/0041 catastrophic failures as "what actually happens when a low-cost potted-plant acquisition hits hard frames" — glossy-leaf highlight, occlusion, green-background confusion — with the Phase18 Figure_A failure panels shown as *field-like acquisition images*. The R1 rescue then reads as "a mechanism that keeps low-cost phenotyping from silently producing F1=0 masks," an agri-operational robustness story (this is the one place the claim_audit's "robust" ban on the keyword does not conflict — "R1 robust instance ranking" is the allowed method name). **Already citable in:** `new_storyline.md` steps 7–10; `core_preservation.md` §6; Phase18 Figure_A/B assets in `main_supplement_plan.md` §1.6.

3. **Include an explicit agronomic-trait workflow callout.** State the four architectural traits in agronomic terms (plant height, canopy width, leaf length, leaf width — the traits breeders/managers track), give the per-plant acquisition + processing budget (~55 min RTX 3090, breakdown), and name the target end-user (ornamental/horticulture sector, pot-plant nurseries). This is the "who uses this and what does it cost" paragraph CompAg reviewers want (B-R3-C01/C04). **Already citable in:** `core_preservation.md` §2 (LEEDGREEN, throughput, timing breakdown); `resolution_status.md` B-R3-C04 RESOLVED; `proposed_contributions.md` C4.

Additionally, demote one segmentation figure or compress Fig3/Fig4 panels so the phenotyping panels (Fig8/Fig9 in current numbering) occupy a comparable share of the visual budget — figure balance reinforces narrative balance.

---

## 3. Special-Issue Keywords (5, bilingual EN + 中文)

Claim-audit constraint honored: **"robustness" is forbidden as a keyword** (claim_audit row 12); replaced by "prompt adaptation" / "candidate ranking" style alternatives. All 5 terms are supported by the claim_evidence_matrix wording ceilings and the SI framing docs.

| # | English | 中文 | Anchored by |
|---|---|---|---|
| 1 | Vision foundation models | 视觉基础模型 | Introduction VFM promise + SI title theme; P01/P03 |
| 2 | Agricultural prompt adaptation | 农业提示词适配 | RAP-FSAM3 C1; P03 prompt-fragility finding; allowed alternative for "prompt robustness" |
| 3 | 2D Gaussian Splatting | 二维高斯泼溅 | ForeSplat C3; P05 |
| 4 | Foreground-object 3D reconstruction | 前景目标三维重建 | Foreground-object 2DGS reformulation C3; P06 |
| 5 | Low-cost plant phenotyping | 低成本植物表型测量 | Workflow C4; P07; A2-C04 throughput |

These replace the current inventory keywords ("vision foundation model, 2D Gaussian Splatting, foreground-object reconstruction, plant phenotyping, RGB imaging") — note "RGB imaging" is dropped (weak for the SI), and "agricultural prompt adaptation" is added to signal the SI's "foundation models for agriculture" theme directly.

---

## 4. Length / Structure Fit

**Current state (ForeSplat.tex):** 9,647 words (raw tex), 850 lines, `cas-sc` single-column, 5 top-level sections, ~20 subsections, 9 figures, 71 references. **Phase 19B target:** ~10-page CAS-SC article, 8 tables + 6 figures in main, 5 supplementary sections (S1–S5) per `main_supplement_plan.md`.

**Against typical CompAg articles** (Elsevier CAS family; no hard page cap in the CompAg guide-for-authors, but published original-research articles cluster around 8,000–12,000 words / 20–35 pages single-column-equivalent with figures):

- **Word count:** ~9,700 raw words is within the typical CompAg band; after the Phase 19B rewrite (Intro 6 paragraphs, Methods M1–M10, Results Q1–Q7, Discussion D1–D6, Limitations L1–L7, Conclusion) the target should land ≈9,000–10,500 words — comfortably typical. No truncation needed.
- **Page budget (~10 pages CAS-SC):** this is on the *compact* end for CompAg but appropriate for a special-issue methods+application paper, provided the main/supp split rules are respected. The `main_supplement_plan.md` page-budget rule (≤1 table + ≤1 figure panel per Q) is the correct discipline; 8 tables is the maximum that keeps ~10 pages viable. If any table must grow, push the R0–R3 ranking-ablation sub-rows and per-sequence breakdowns to S1/S5 rather than expanding main tables.
- **Figure count (6 planned vs 9 current):** 6 figures is typical-to-light for CompAg; the consolidation (Fig5/6/7 → one composite, Fig9 scatters + Fig8 → one panel set) is feasible per `figure_plan.md`. Do not shrink the phenotyping figure — it is the ag anchor (§2).
- **References (71):** exceeds CompAg's ~25+ norm with a healthy share of recent (≤5 yr) works; verify the VFM/SAM3/plant-Gaussian consistency per `core_preservation.md` §5. Add the 2–3 SI-scope VFM-in-agriculture citations (citation slots flagged in `new_storyline.md` step 2).
- **Supplementary (S1–S5):** appropriate size and structure for CompAg; S4 (reproducibility) and S1.2 (182-candidate benchmark) are exactly the evidence density CompAg reviewers expect without page cost.

**Fit verdict on length:** NO structural re-scoping needed. The 10-page CAS-SC target with 8+6 main items and S1–S5 supplement is within CompAg norms; the risk is under-space (methods richness vs 10 pages), mitigated by the existing main/supp split rules.

---

## 5. Decision

**VERDICT: REVISE FIRST — then submit to the CompAg VFM special issue.** The fit is strong and the SI alignment is genuine (frozen VFM + agriculture-specific adapter + reconstruction-aware optimization + downstream phenotyping is the SI's exact brief), but submission-ready requires closing three items in Phase 19B — none of which requires new experiments:

1. **Lead with agricultural value structurally.** Open the Discussion with the phenotyping/downstream section (D5-style, per §2 move 1), add the agronomic-trait + cost/target-user paragraph (move 3), and rebalance figure share toward the phenotyping panels. Without this, the algorithm-forward drift (§2) is the paper's biggest rejection risk.
2. **Add the explicit novelty-partition statement** (B-R2-C01): one paragraph in Introduction §1.6 / Discussion D4 stating which contributions are novel *mechanisms* (reconstruction-aware prompt adaptation; foreground-object 2DGS constraint injection; ranking-failure diagnosis + R1) vs novel *combinations* — satisfying CompAg's "not merely re-application of existing technology" gate.
3. **Close the reproducibility/transparency block:** explicit Data Availability statement (code to be released upon acceptance), the full hardware/software/hyperparameter block in Methods §2.9 + S4 (B-R3-C02, B-R2-C04), and the botanical-nomenclature provisional table (B-R1-C01) — all content already planned, must be visibly present at submission.

These are word-level and packaging changes; no experimental evidence is missing for the core claims as scoped in the claim_evidence_matrix. The ranking-failure/R1 contribution (C01–C04) is the differentiator a CompAg VFM-SI reviewer will reward — it demonstrates an agriculture-motivated failure analysis that pure-CV benchmark papers do not carry.

**Sources for journal facts:** CompAg scope/CAS1区 ([fabiao](https://www.fabiao.com/sci/01681699.html), [xueshu](https://www.xueshu.com/sci/41455/07.html)); CompAg innovation gate ("existing tech applied to a crop does not qualify") ([peipusci](https://m.peipusci.com/news/366.html)); special-issue framing from local `compag特刊增修模块/论文增改表述.md`. Journal-level facts verified by search on 2026-09-14; IF figures vary by source/year and are indicative only.

---

**WP7D SELF-CHECK: ALL PASS**
