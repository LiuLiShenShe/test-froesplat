# Phase 19A — WP6A Proposed Contributions (Phase 19B Manuscript)

Audit date: 2026-09-14. Proposes 4 contributions for the Phase 19B rewrite, each with draft wording, modules, evidence level (from claim_evidence_matrix), paper section, and risk line.

---

## Contribution 1: RAP-FSAM3 — Reconstruction-Aware Prompt-Adaptive VFM Adapter for Plant Foreground Priors

**Rank:** 1
**Contribution statement (draft wording):**

> We propose RAP-FSAM3, a reconstruction-aware prompt-adaptive adapter that turns a frozen vision foundation model (SAM3) into a foreground-prior generator for potted-plant scenes. RAP-FSAM3 generates per-instance multi-prompt candidates, applies semantic-gated selection and structure-balanced foreground/background prompt refinement, and closes with geometry-consistency correction, producing foreground priors directly usable as 2DGS optimisation constraints.

**Modules involved:** RAP-FSAM3 (multi-prompt candidate generation; semantic-gated multi-prompt selection; structure-balanced FG/BG prompt refinement; geometry-consistency correction; R1 robust instance ranking). ForeSplat (foreground-object 2DGS with M2M3-GO refinement layer).

**Evidence level (from claim_evidence_matrix):**
- P01 — DESCRIPTIVE (frozen-set means): "RAP-FSAM3 achieved F1=0.9706, mIoU=0.9641, HD95=183.42 px, Boundary-F1=0.4643, leakage energy=0.001739 on the manually annotated evaluation frames sampled from all 20 RGB sequences."
- P02 — DESCRIPTIVE (delta report): "Improvements over single-prompt SAM3 were concentrated in boundary consistency (Boundary-F1 0.4643 vs. 0.4051) and leakage suppression (leakage energy 0.001739 vs. 0.002272); region-overlap deltas (F1 0.9706 vs. 0.9692, mIoU 0.9641 vs. 0.9624) were small."
- P03 — PROVEN (per-prompt F1 report): direct prompting is semantically fragile in potted-plant scenes (per-prompt F1 0.0768-0.9692).

**Paper section where argued:** Results §3.1 (VFM comparison), §3.2 (prompt robustness); Methods §2.3 (RAP-FSAM3 architecture).

**RISK LINE:** "Outperforming"/"significantly"/"generalizes" wording FORBIDDEN for this contribution (claim_audit rows 4-8). Boundary/leakage gains are descriptive; region-overlap deltas are within annotation variance. All claims scoped to "among the tested methods on the frozen P6 evaluation set." No significance test reported.

---

## Contribution 2: Candidate-Ranking Failure Attribution + R1 Ranking Fix

**Rank:** 2 (strongest NEW contribution)
**Contribution statement (draft wording):**

> We diagnose a failure mode overlooked by prompt-adaptation-only pipelines: catastrophic segmentation failures on hard frames are caused by candidate RANKING, not candidate GENERATION. On DouBanLv2 0039/0040/0041 the candidate generator produced high-quality whole-plant masks (raw oracle F1 = 0.9867-0.9882) while the primary score selected a high-contrast distractor (final F1 = 0). We show the contrast quality term (q_contrast) can favor small, high-contrast distractors over larger whole-plant candidates, and introduce R1 robust instance ranking -- which removes q_contrast from the primary score -- eliminating all identified DEV-set ranking failures (0039/0040/0041 rescued to F1 >= 0.95) with zero material regressions on Easy21.

**Modules involved:** Candidate generation (per-instance), candidate ranking (R1 robust instance ranking, q_contrast excluded from primary score), R0 baseline scoring (diagnostic).

**Evidence level (from claim_evidence_matrix):**
- C01 — PROVEN: "Baseline candidate generation produced high-quality whole-plant candidate masks (raw oracle F1 = 0.9867-0.9882) even on frames where final F1 = 0."
- C02 — PROVEN: "Catastrophic failures (DouBanLv2 0039-0041) were caused by candidate-ranking failure (selected distractor), not candidate generation (C3_RANKING_FAILURE)."
- C03 — SUPPORTED: "q_contrast can favor small, high-contrast distractor masks over larger whole-plant candidates (failure-frame Spearman rho = -0.7187, p = 0.0038; contrast-drop ablation rescued 3/3 identified failures)." -- correlational wording only.
- C04 — SUPPORTED: "Removing q_contrast from the primary ranking (R1) eliminated all identified DEV-set ranking failures (0039/0040/0041 rescued to F1 >= 0.95) with zero material regressions on Easy21." -- reported as rescue on the identified DEV set; no global superiority claim.

**Paper section where argued:** NEW Results subsection (§3.3 merge or new §3.x) -- "Candidate-ranking failure analysis"; Discussion §4.4 (why ranking matters); Abstract/Conclusion one-sentence reflection.

**RISK LINE:** C05 FORBIDDEN — "R1 significantly improves general segmentation performance" must NOT appear. Only the selection rationale is allowed: "R1 was selected for failure-rescue, zero material regression, low selection churn, and structural simplicity -- not because its mean F1 is significantly higher (N=10 mean-F1 comparisons are descriptive only)." N=10 DEV frames; no generalization claim.

---

## Contribution 3: ForeSplat Foreground-Object 2DGS + M2M3-GO Primitive-Level Refinement

**Rank:** 3
**Contribution statement (draft wording):**

> We reformulate the plant 2D Gaussian Splatting target as a foreground-object optimisation: foreground-track initialization, foreground-RGB supervision, opacity constraints, view-quality weighting, and mask-guided multi-cue pruning, with M2M3-GO -- a lightweight primitive-level refinement layer that makes per-Gaussian keep/densify/prune micro-decisions from mask-to-model support and model-to-mask coverage deficits. Training-time foreground-object loss injection reduced the outside-mask non-black ratio from 0.9908 to 0.0294 and leakage energy from 1.2201 to 0.0190 on a representative broad-leaf morphology, while post hoc mask pruning was insufficient (outside ratio 0.7509, leakage 0.7900).

**Modules involved:** ForeSplat (foreground-object 2DGS: FG track init, FG RGB supervision, opacity constraints, view-quality weighting, mask-guided pruning), M2M3-GO (primitive-level keep/densify/prune), injection-position ablation variants.

**Evidence level (from claim_evidence_matrix):**
- P05 — DESCRIPTIVE (workflow-level means): PSNR=31.09 dB, SSIM=0.9711, LPIPS=0.0365 under the unified 20-sequence protocol; +1.51 dB vs standard 2DGS; time reductions 60.94%/65.17%; meshing 55 s vs 642 s.
- P06 — DESCRIPTIVE (ablation table): injection-position ablation numbers (single representative sample, KongQueZhuYu). Post hoc pruning described as insufficient, not "useless."
- B-DA-C01 (partial): E7 post-hoc pruning quantitative result now available (single sample).

**Paper section where argued:** Methods §2.4-2.7 (foreground-object 2DGS + M2M3-GO); Results §3.4 (injection ablation); Results §3.5 (workflow validation); Discussion §4.2-4.3.

**RISK LINE:** Injection ablation is single-sample representative -- "Do not claim generalizable ablation results; state as development-set finding only" (B-DA-C02 NOT_RESOLVED). No external baseline comparison (B-R2-C02 NOT_RESOLVED) -- "Do not claim outperforms SOTA or best reconstruction; must remain as explicit limitation." No significance test; descriptive only.

---

## Contribution 4: Low-Cost 3D Phenotyping Workflow + Validation Protocol

**Rank:** 4
**Contribution statement (draft wording):**

> We assemble and validate a low-cost smartphone-video-based 3D plant phenotyping workflow on potted plants: VFM-derived foreground priors, foreground-object 2D Gaussian Splatting, TSDF-style meshing with pot-diameter scale recovery, and virtual measurement of four architectural traits (height, canopy width, leaf length, leaf width) across 21 plants / 63 leaves. Virtual measurements were comparable to manual measurements (plant height R^2 = 0.99, canopy width R^2 = 0.99, leaf length R^2 = 0.97 with n=63, leaf width R^2 = 0.90). We contribute a validation protocol spanning frozen-set VFM comparison, prompt robustness, module ablation, injection-position ablation, a unified 20-sequence workflow evaluation, and a 10-frame DEV-set failure-rescue check.

**Modules involved:** Full pipeline (RAP-FSAM3 -> ForeSplat -> mesh extraction -> virtual phenotyping); validation protocol.

**Evidence level (from claim_evidence_matrix):**
- P05 — DESCRIPTIVE (workflow-level means): 20-sequence reconstruction metrics.
- P07 — DESCRIPTIVE (agreement report): "Virtual measurements were comparable to manual measurements: plant height R2=0.988 (MAE 0.98 cm), canopy width R2=0.988 (MAE 0.86 cm), leaf length R2=0.974 (MAE 0.51 cm, n=63), leaf width R2=0.900 (MAE 0.45 cm)."
- B-EIC-C01 (partial): cross-species framing reframed as architectural diversity; reconstruction validation limited to 3 architecturally representative samples -- explicit limitation.

**Paper section where argued:** Methods §2.8 (mesh extraction/scale recovery/measurement); Results §3.5 (workflow validation + phenotyping); Discussion §4.4-4.5; Conclusion.

**RISK LINE:** "strong agreement" FORBIDDEN -- use "comparable to" (claim_audit row 9). R^2 descriptive on n=21/n=63, no repeated measurements, no Bland-Altman, no inter-rater reliability. Do not claim cross-species GENERALIZATION at the reconstruction level (3 samples); use "cross-species applicability / architectural diversity." Do not claim high precision for leaf width (R^2=0.90, open limitation A1-C03). Report R^2 to 2 decimal places (A2-C01).

---

## Cross-Cutting Constraints (ALL contributions)

1. **G14 constraint:** A6 (cross-view consensus) and A7 (memory propagation) MUST NOT appear as main contributions. They are DEMOTED, default OFF, negative/limitation results only (Phase18 Figure_D + A6/A7 ablation table). Production defaults: `--use_cross_view_consensus False`, `--use_memory_propagation False` (claims C06/C07/C08/P08).
2. **Forbidden wording (claim_audit):** "outperforming", "generalizes/generalization", "superior", "state-of-the-art/SOTA", "strong agreement", "most evident", bare "best", "significantly" (statistical), "robustness" (keyword), "robust segmentation" (general adjective). "Robust" is allowed ONLY in the method name "R1 robust instance ranking."
3. **Allowed wording patterns:** "achieved [specific metric values] on the tested evaluation set", "achieved the highest [metric] among the tested methods", "was concentrated in", "were comparable to", specific metric deltas.
4. **No significance tests exist** anywhere in the evidence base (frozen P6 N=21, DEV10 N=10, Easy21 N=21, all descriptive). Any "statistically significant" phrasing is forbidden.

---

## Contribution Ranking and Rationale

| Rank | Contribution | Evidence strength | NEW? |
|---|---|---|---|
| 1 | RAP-FSAM3 VFM adapter | P01/P02/P03 (DESCRIPTIVE + PROVEN prompt fragility) | Retained (existing core) |
| 2 | Ranking-failure attribution + R1 fix | C01/C02 PROVEN; C03/C04 SUPPORTED | **NEW headline** |
| 3 | ForeSplat foreground-object 2DGS + M2M3-GO | P05/P06 DESCRIPTIVE | Retained (existing core) |
| 4 | Low-cost phenotyping workflow + protocol | P07/P05 DESCRIPTIVE | Retained (existing core) |

Rationale: Contribution 2 is ranked above the 3D-method contribution because C01/C02 are the only PROVEN claims in the evidence base and the ranking story is the Phase 17C headline currently missing from the manuscript. Contributions 1, 3, 4 are retained as the application core and must not be cannibalized (see core_preservation rank-preservation principle).

---

## Verification

- [x] 4 contributions proposed (within 3-4 range)
- [x] Each contribution has: statement, modules, evidence level, paper section, RISK LINE
- [x] Evidence levels drawn from claim_evidence_matrix allowed_wording
- [x] RISK LINEs explicitly forbid A6/A7 as contributions (G14) and forbid "outperforms"/"generalizes" wording
- [x] Contribution 2 flagged as strongest NEW contribution

**WP6A SELF-CHECK: CONTRIBUTIONS PASS**
