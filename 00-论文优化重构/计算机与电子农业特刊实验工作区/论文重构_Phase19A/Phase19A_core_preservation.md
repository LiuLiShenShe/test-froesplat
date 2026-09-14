# Phase 19A — WP6A Core Preservation (What MUST Survive the Rewrite)

Audit date: 2026-09-14. Non-negotiable elements of the current ForeSplat.tex manuscript that must survive the Phase 19B rewrite unchanged or with only the listed minor edits. Sourced from `Phase19A_old_manuscript_audit.csv` and `Phase19A_claim_audit.csv`.

---

## 1. Core Research Problem (punchline, must survive verbatim in spirit)

**The problem:** VFM foreground priors -> 3D phenotyping in low-cost potted-plant acquisitions.

- VFMs are not plug-and-play for agricultural 3D phenotyping: direct text prompting is prompt-sensitive, semantically biased, and boundary-unstable in potted-plant scenes (PROVEN: per-prompt F1 spread 0.0768-0.9692, claim P03).
- The open question pre-figuring the contribution (Introduction para 2, line 101, KEEP): "is the reconstructed 3D object the plant object that needs to be measured?" -- this is the framing on which the foreground-object prior is built.
- VFM outputs must be adapted into reconstruction-aware foreground priors (RAP-FSAM3) and the priors injected into foreground-object 2DGS optimisation constraints (ForeSplat).

**Evidence:** `old_manuscript_audit.csv` lines 14-17 (Introduction KEEP/KEEP_WITH_MINOR_EDIT rows); claim_evidence_matrix P01-P04.

**Action:** Keep the problem framing and the RAP-FSAM3/ForeSplat two-component coupling statement. Only edits allowed: (a) soften the final "reduce the robustness of reconstruction and measurement" clause to "make reconstruction and measurement unreliable in low-cost settings" (claim_audit row 2); (b) forward-pointer citation to §3.2 prompt-robustness results (line 105 KEEP_WITH_MINOR_EDIT); (c) optionally add "robust multi-candidate ranking" to the RAP-FSAM3 component description (line 107).

---

## 2. Agricultural Application Value (CompAg scope)

- The manuscript targets Computers and Electronics in Agriculture, special issue on foundation models in agriculture (confirmed primary target).
- Application value: a smartphone-video-based, low-cost 3D phenotyping workflow for potted plants -- potted-plant foreground isolation, meshing, and virtual trait measurement at scale.
- Throughput figures to keep (added per A2-C04): ~1.5-2.0 min per plant acquisition; 30-40 plants/hour manual acquisition; ~55 min total per plant on RTX 3090 with breakdown (COLMAP 15 min, FSAM3 8 min, 2DGS 25 min, mesh 5 min, measurement 2 min).
- LEEDGREEN acronym (low-cost, front-end, easy-to-deploy, green) and the smartphone acquisition setup must be preserved.

**Action:** Keep the application framing and throughput section. No edits to the core value proposition.

---

## 3. The 3D/Phenotyping Outcome (the punchline numeric)

**The punchline:** R^2 up-to-0.988 trait agreement, with the hedged "comparable to" wording.

- Abstract (line 88, KEEP): VFM+ForeSplat enables low-cost 3D plant phenotyping when VFM outputs are adapted as reconstruction-aware foreground priors and converted into 3D optimisation constraints (R^2 up to 0.988).
- Phenotyping: R^2 = 0.988 (height, MAE 0.98 cm), 0.988 (canopy, MAE 0.86 cm), 0.974 (leaf length, MAE 0.51 cm, n=63), 0.900 (leaf width, MAE 0.45 cm). **Must be worded as "comparable to manual measurements", NOT "strong agreement"** (claim_audit row 9; claim P07 forbidden wording).
- 20-sequence validation: PSNR=31.09 dB, SSIM=0.9711, LPIPS=0.0365; +1.51 dB vs standard 2DGS; reduced train/recon time 60.94% and meshing time 65.17% vs standard 2DGS; meshing 55 s vs 642 s for 3DGS+preproc (claim P05, DESCRIPTIVE).
- "Stably generated" -> "for all 20 sequences" (overstates without repeated-measure analysis).

**Action:** Keep all five numbers. Mandatory wording: "comparable to" throughout; R^2 to 2 decimal places in Abstract and Table (A2-C01). Add footnote "no repeated manual measurements were collected" (B-R1-C02).

---

## 4. Must-Keep Experiments (7 of them)

These experiments and their evaluation protocols MUST NOT be dropped or cannibalized by the rewrite:

1. **VFM comparison** on frozen P6 (Table 3; CLIPSeg/Grounded-SAM/SAM3/RAP-FSAM3). Claim P01/P02.
2. **Prompt robustness** analysis (Table 4; 5-prompt per-prompt F1). Claim P03.
3. **RAP-FSAM3 module ablation** (Table 5 module block; geometry-only selector vs semantic gating vs full). Claim P04.
4. **Injection-position ablation** (Table 6; full-scene vs input masking vs FG-loss injection vs post-hoc pruning; E7 row). Claim P06 + reviewer A1-C02 fair-input note + B-DA-C01 E7 results.
5. **Full 20-sequence workflow validation** (all 20 sequences complete; reconstruction + mesh + timing). Claim P05.
6. **Phenotypic validation** (21 plants/63 leaves, 4 traits, R^2/MAE/RMSE/MAPE). Claim P07.
7. **NEW: ranking-failure analysis + R1 rescue** (DouBanLv2 0039/0040/0041 oracle vs R0 vs R1; R0-R3 policy ablation; Easy21 zero-regression; TEST32 selection churn). Claims C01-C04. This becomes a headline subsection -- see Section 6.

**Action:** Preserve experiments 1-6 exactly as scoped. Add experiment 7 as a new Results subsection (§3.3 merge or new §3.x) per old_manuscript_audit NEW_SECTION_REQUIRED row.

---

## 5. Must-Keep Sections/Content with Exact tex Line References

From `old_manuscript_audit.csv`, the following rows are KEEP or KEEP_WITH_MINOR_EDIT and must survive (line numbers refer to ForeSplat.tex):

| Section | tex lines | Status | Content to preserve |
|---|---|---|---|
| Title | line 46 | KEEP | "ForeSplat: Vision-Foundation-Model-Guided Foreground-Object 2D Gaussian Splatting for Low-Cost 3D Plant Phenotyping" |
| Abstract | line 88, first 2 sentences | KEEP | Problem framing (prompt sensitivity / boundary instability / 2D-3D mismatch) |
| Introduction para 2 | line 101 | KEEP | Literature survey + open question |
| Introduction para 3 | line 103 | KEEP | Foreground-object prior definition (potted-plant instance; container may be retained) |
| Methods 2.1 | line 121 | KEEP | Workflow summary (RAP-FSAM3 -> COLMAP + mask consistency -> FG 2DGS -> M2M3-GO -> TSDF mesh) |
| Methods 2.2 | line 133 | KEEP | Dataset acquisition + evaluation protocol |
| Methods 2.2 | Table 1 (line 135), Table 2 (line 154) | KEEP | Dataset-protocol and coverage tables (100% = descriptive 10/10, keep) |
| Methods 2.3.1 | lines 177-187 | KEEP | FFT frame-quality filter (mechanism only) |
| Methods 2.3.4 | lines 242-273 | KEEP | Structure-balanced prompt refinement |
| Methods 2.3.5 | lines 275-298 | KEEP | Geometry-consistency foreground correction (conservative, deletion disabled) |
| Methods 2.4 | line 309 | KEEP | Foreground-object 2DGS intro + stage list |
| Methods 2.4.1 | lines 313-332 | KEEP | COLMAP SfM; FG trajectory init (V_j, tau_track=0.9) |
| Methods 2.4.2 | lines 334-391 | KEEP | FG RGB supervision + opacity constraints (lambda_mask=0.08, lambda_bg=0.02) |
| Methods 2.5 | lines 393-410 | KEEP | View-quality-aware soft loss weighting |
| Methods 2.6 | lines 412-437 | KEEP | Mask-guided multi-cue Gaussian pruning |
| Methods 2.7 | lines 439-547 | KEEP | M2M3-GO (keep/densify/prune micro-decisions; Floor40 gate) |
| Methods 2.8 | lines 550-561 | KEEP | Mesh extraction, scale recovery from pot diameter, four traits |
| Methods 2.9 | lines 562-572 | KEEP | Evaluation metrics + protocol paragraph (30k iter, A6000) |
| Results 3.1 | lines 576-578 | KEEP_WITH_MINOR_EDIT | VFM comparison numbers; "most evident" -> "concentrated in"; add R1 rescue sentence |
| Results 3.2 | lines 606-608 | KEEP | Prompt sensitivity five-prompt table (0.9692/0.7972/0.6471/0.0768/0.4051) |
| Results 3.4 | lines 667-669 | KEEP | Injection-position ablation narrative ("modest refinement", "training-after-processing pruning cannot replace training-time optimisation") |
| Results 3.5 | lines 731-733 | KEEP_WITH_MINOR_EDIT | Workflow-level numbers; "stably" -> "for all 20 sequences"; add n and descriptive-statistics caveat |
| Discussion 4.1 | lines 794-798 | KEEP | VFM prompt adaptation argument |
| Discussion 4.2 | lines 800-802 | KEEP | Foreground priors define the 3D optim target; post-hoc pruning insufficient |
| Discussion 4.3 | lines 805-809 | KEEP | M2M3-GO as lightweight refinement layer ("modest improvement" acknowledged) |
| Discussion 4.4 | lines 812-816 | KEEP_WITH_MINOR_EDIT | Trait-difficulty ordering + 3 error sources + future work; ADD paragraph on (1) why ranking matters (q_contrast misleading signal; F1=0 catastrophic failures), (2) why A6/A7 not retained (A6 E=-0.0075 pilot; A7 near-neutral) |
| Discussion 4.5 | lines 818-820 | KEEP_WITH_MINOR_EDIT | Contrast with IPENS/LCR-GS/Gaussian Grouping/SAGA; future work explicitly excludes temporal memory (A7 not production) |
| Conclusion | line 824 | KEEP_WITH_MINOR_EDIT | Summary + VFMs-can-support hedging; add ranking/evidence sentence (R1 rescue) |
| References | cas-refs.bib | KEEP | 71 citations; verify refs for SAM3/VFM/plant-Gaussian consistency |

**Sections that MUST BE REWRITTEN (they describe the OLD scoring policy; do not treat as KEEP):**
- Methods 2.3.3 Semantic-Gated Multi-Prompt Selection (lines 199-240): CRITICAL -- describes q_contrast in the primary descriptor; REWRITE to describe R1 as the production ranking policy (primary score excludes q_contrast); old softmax-gated selector moves to ablation/supplementary.
- Results 3.3 (lines 637-640 + Table 5): MERGE with new ranking-failure/ablation analysis (Phase 17C: R1 rescues 0039/0040/0041 from F1~0 to >=0.95; q_contrast Spearman bias -0.2927 full / -0.7187 on failure frames).
- Abstract RAP-FSAM3 sentence: soften "outperforming" to "achieving higher boundary consistency and leakage suppression"; update to name R1 production ranking policy.
- Figs 1/2 annotations: must reflect per-instance candidate generation + R1 ranking stage.
- NEW Results subsection required for the candidate-ranking failure analysis (see Section 6).

**A6/A7 handling (per claim_audit + claim_evidence_matrix):** A6 (cross-view consensus) and A7 (memory propagation) are DEMOTED, NOT_SUPPORTED, default OFF. They MUST NOT appear as production stages in Methods 2.x; they appear only as negative/limitation results (Results (Ablation) or Limitations), backed by Phase18 Figure_D and the A6/A7 ablation table. Production defaults: `--use_cross_view_consensus False`, `--use_memory_propagation False`.

---

## 6. The Rank-Preservation Principle (NEW headline vs 3D/phenotyping core)

**Rule:** The ranking-failure analysis + R1 rescue becomes a NEW headline contribution, but it MUST NOT cannibalize the 3D/phenotyping core.

What this means concretely for the rewrite:

1. **A new Results subsection** on candidate-ranking failure analysis (old_manuscript_audit NEW_SECTION_REQUIRED): hard-frame failure of old scoring (DouBanLv2 0039/0040/0041 F1=0 despite oracle F1=0.9867-0.9882), q_contrast Spearman bias evidence, R1 rescue, DEV10/Easy21 zero-regression check, TEST32 selection churn. Reflect in Abstract/Conclusion/Discussion.
2. **The 3D/phenotyping core stays the application anchor:** the paper remains titled as a foreground-object 2DGS phenotyping paper; the ranking story is an added robustness/attribution analysis inside the VFM-adaptation component, not a replacement of the reconstruction/phenotyping contributions.
3. **Section budget:** the ranking story gets ONE new Results subsection (± merges into §3.3 and Discussion 4.4) and ONE new table (ranking policy/failure-rescue), plus Figures 3-4. It does not grow into additional sections, and it does not displace the VFM-comparison, injection-ablation, workflow-validation, or phenotyping sections.
4. **Contribution list (Introduction line 111-115):** add "candidate-ranking failure analysis" to contribution 3's validation list; keep the three-existing-contribution structure.
5. **No A6/A7 rescue:** A6/A7 are NOT resurrected as contributions by the ranking story; they remain demoted negative results (G14 constraint, method_component_audit G11 PASS).

---

## 7. Non-Negotiable Wording Constraints (cross-cutting)

From `Phase19A_claim_audit.csv` and `claim_evidence_matrix.csv`:
- "outperforming" -> replace with specific metric deltas ("achieved higher boundary consistency and stronger leakage suppression", "achieved the highest ... among the tested methods on the evaluation set").
- "significantly / statistically significant" -> forbidden without a reported test. R1 selection must use the C05 allowed_wording.
- "generalizes / generalisation" -> forbidden (single-dataset evaluation; use "performs well on the tested sequences" / "applies to the tested acquisition settings").
- "superior" / "state-of-the-art" / "SOTA" -> forbidden.
- "strong agreement" -> "were comparable to" (phenotyping).
- "most evident" -> "was concentrated in" / "was largest in".
- "best overall" / bare "best" in running text -> allowed in table annotation only when captioned "highest ... on the frozen P6 evaluation set".
- "robustness" as keyword -> replace with "prompt adaptation" or "candidate ranking"; "robust" is allowed ONLY as part of the method name "R1 robust instance ranking".
- "robust segmentation" (general adjective) -> forbidden.

---

## Verification

- [x] Core research problem (punchline) listed
- [x] Agricultural application value (CompAg scope) listed
- [x] 3D/phenotyping outcome (R^2 up-to-0.988, "comparable to" wording) listed
- [x] Must-keep experiments enumerated (7)
- [x] Must-keep sections/content with exact tex line references (from old_manuscript_audit)
- [x] Rank-preservation principle explicit (ranking failure + R1 rescue = NEW headline, must not cannibalize 3D core)

**WP6A SELF-CHECK: CORE_PRESERVATION PASS**