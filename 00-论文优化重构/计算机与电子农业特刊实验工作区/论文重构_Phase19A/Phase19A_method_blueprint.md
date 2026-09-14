# Phase 19A — Methods Blueprint (FILE B: WP6B)

> Blueprint for the Methods chapter reorganized around the PRODUCTION algorithm (NOT the old manuscript structure).
> Source of truth: `Phase19A_algorithm_delta.md` (WP4) + `Phase19A_formula_audit.md` (WP5) + `Phase19A_method_component_audit.md` (WP3).
> Audit date: 2026-09-14.

---

## Subsection 1 — Workflow Overview

**Must appear:**

- Naming of the two-stage workflow: **RAP-FSAM3** (foreground-prior generation) -> **ForeSplat** (foreground-object 2DGS) -> mesh + phenotyping.
- An ordered module list (top-level, no A6/A7):
  1. FFT frame-quality filtering
  2. Per-instance SAM3 candidate generation
  3. Candidate feature scoring (q_* features)
  4. R1 robust instance ranking (production selector)
  5. (mention) structure-balanced prompt refinement + geometry-consistency foreground correction as refinement stages applied to the R1-selected candidate
  6. Foreground-object 2DGS optimisation
  7. M2M3-GO + mask-guided pruning (refinement layer)
  8. Mesh extraction / TSDF / scale recovery / phenotyping
- One sentence placing the pipeline in the CompAg phenotyping context (single-foreground-plant reconstruction from multiview overhead RGB).
- A note that modules A6 (cross-view consensus) and A7 (memory propagation) are independent experimental branches, disabled by default, and not part of the described production pipeline (would link to Limitations/ablation only).

**Must NOT appear:**

- No A6/A7 stage descriptions in the Method flow.
- No old SG-MPS softmax selection as a production stage here (relegated to Subsection 6 as superseded baseline, or supplementary).

**Supporting evidence:** `Phase19A_algorithm_delta.md` Status Rollup (REWRITE/ADD rows); `Phase19A_method_component_audit.md` G11 gate (A6/A7 NOT_IN_MANUSCRIPT, correct).

---

## Subsection 2 — Image Acquisition / Dataset & Evaluation Protocol

**Must appear:**

- The 5 datasets with roles (per Phase18 dataset registry — as referenced by `claim_evidence_matrix` `dataset` column and `Phase19A_experiment_asset_registry.csv`):
  1. **Frozen P6 evaluation set** — 21 manually annotated frames sampled from all 20 RGB sequences, 5 samples. This is THE primary quantitative evaluation set (P01-P04, P07). MUST be named explicitly "frozen P6 evaluation set" — never conflated with the P6 *prompt set* (`Phase19A_formula_audit.md` §4).
  2. **20 RGB sequences** — workflow-level reconstruction validation (P05), no frame-level GT required.
  3. **GT10 (DEV10)** — 10 annotated GT frames (BaiZhang 0033-0037, DouBanLv2 0037-0041), 2 sequences; the DEV-set of the frozen 58-target challenge (26 DEV + 32 TEST). Used ONLY for the ranking-failure/R1-rescue analysis (C01-C04). MUST be labelled "NOT held-out; descriptive DEV evidence" (N=10).
  4. **TEST32** — 32 unlabeled TEST frames, 8 samples; behavioural safety audit only, NO accuracy claims (C05 rationale). Paper_usable=FALSE in asset registry for accuracy claims.
  5. **Easy21** — 21-frame annotated set used for A6/A7 negative ablation and R1 safety check (C05-C07, C08).
- Acquisition protocol: multiview RGB capture, COLMAP sparse reconstruction, per-sequence camera poses; pot diameter known from pots used (scale recovery basis).
- Evaluation protocol statement: metrics computed per-annotation-frame and averaged as descriptive means over the frozen set; no significance tests claimed (claim_evidence_matrix style).

**Must NOT appear:**

- No claim that the DEV10 (GT10) is a held-out test set; it is descriptive DEV evidence.
- No accuracy claims from TEST32 (unlabeled).
- No mixing of the P6 prompt set with the P6_raw_baseline run metrics (`f1_mean=0.983498` is the frozen Phase 12 run label, produced under default `prompt_list=P2` config — if used at all, describe exactly that way).

**Supporting evidence:** `Phase19A_formula_audit.md` §4 (P2 vs P6 resolution); `Phase19A_experiment_asset_registry.csv` P16-A06, P16-A07.

---

## Subsection 3 — FFT Frame-Quality Filtering (KEEP)

**Must appear:**

- Normalized high-frequency saliency index `Q_FFT(I)` computed per frame (ForeSplat.tex L177-187).
- Sample-adaptive threshold: per-sequence first-quartile of frame-quality scores; frames below threshold excluded from segmentation and COLMAP.
- Motivation: blurred frames corrupt candidate generation and foreground-track initialisation.

**Must NOT appear:**

- No change to the described mechanism (production-consistent KEEP).

**Supporting evidence:** `Phase19A_algorithm_delta.md` row #1 (KEEP); `Phase19A_method_component_audit.md` FFT row.

---

## Subsection 4 — Per-Instance SAM3 Candidate Generation

**Must appear:**

- Per-instance candidate generation (production `candidate_mode=per_instance`): each SAM3 instance mask becomes an independent Candidate; multiple candidates can come from one prompt.
- Production default prompt list: **P2** (`prompt_list="P2"`, `default_prompt_id=P2`), whose text is "entire plant excluding pot". The five-prompt catalogue (green plant / entire potted plant / leaves and stems / crop seedling / plant body without background) may be described as the AVAILABLE prompt set, not the deployed candidate set.
- Key hyperparameters: SAM3 mask threshold `τ_mask = 0.5`; SAM3 confidence threshold `τ_conf = 0.3`; areas bounded by `A_min=0.01`, `A_max=0.80`.
- Note that the prompt set is per-instance, not a 5-fixed-prompts-x1-frame union (correct the old "5 fixed prompts, K=5 per frame" phrasing).

**Must NOT appear:**

- The old framing "five fixed prompts with K=5 candidates per frame".
- Any statement implying production runs P6 (or P2+P6) prompt lists by default — default is P2 only; P6 participates only when `--prompt_list` explicitly includes it.

**Supporting evidence:** `Phase19A_algorithm_delta.md` row #2 (KEEP minor edit); `Phase19A_formula_audit.md` §2.5 (`candidate_mode`, prompt defaults); `Phase19A_method_component_audit.md` §0 finding 3.

---

## Subsection 5 — Candidate Feature Scoring (q_* features)

**Must appear:**

- Define the per-candidate feature vector (from `Phase19A_formula_audit.md` §2.1, symbol table):
  - `q_area` — area-ratio triangle score relative to `[A_min, A_max]`
  - `q_comp` — single-connected-domain completeness, `1/(1+max(comp_count−1,0))`
  - `q_edge` — boundary conformity (penalizes touching frame border, border_band=2px)
  - `q_temp` — temporal consistency; fixed neutral 0.5 in production default (`use_temporal_alignment=False`)
  - `q_contrast` — foreground-background L2 contrast normalized, `min(1, contrast/0.25)`
  - `q_sam` — SAM3 instance-confidence sigmoid, `1/(1+exp(−6(sam_score−0.5)))`
  - (production default also computes q_leak/q_side but with weight 0 — state they exist as available features, not contributing in the default config)
- Weights table (default): `w_area=w_comp=w_edge=w_temp=w_contrast=1.0`, `w_sam=0.5`, `w_leak=w_side=0.0`; **denom D = 5.5 frozen**.
- Define `base_total` `B = (Σ w_k·q_k + w_sam·q_sam)/5.5` (formula_audit §2.4).
- State the scoring is computed once per candidate mask.

**Must NOT appear:**

- Phrase q_contrast or q_temp as if they are in the PRODUCTION primary score (they are not — see Subsection 6).
- Field-name confusion: `ScoreRecord.leak_penalty` actually stores risk_penalty (see §6); do not translate the field name literally into prose.

**Supporting evidence:** `Phase19A_formula_audit.md` §2.1-§2.2; `Phase19A_method_component_audit.md` MOVE_TO_ABLATION rows.

---

## Subsection 6 — R1 Robust Instance Ranking (NEW — the production selector)

**Must appear (THE method, prominently):**

- R1 is the production default selector (`--ranking_policy r1`, `ranking_policy.py` single source of truth).
- Primary score formula (must keep explicit denom):

  ```
  F = B − q_contrast / 5.5 − P
  B = (q_area + q_comp + q_edge + q_temp + q_contrast + 0.5·q_sam) / 5.5
  P = 0.25·1[V < 0.30] + 0.15·1[track_mask valid ∧ IoU < 0.20]
  selection = argmax (F, tie_break),  tie_break = 0.0 under R1
  ```

- Explain the contrast term: `q_contrast` is subtracted at the normalized contribution `q_contrast/5.5` (not 1.0), because the weights sum `denom=5.5` stays frozen even after the exclusion (formula_audit §3.3).
- Risk penalty semantics: penalties are positive amounts subsequently SUBTRACTED from the primary score (`P_v=0.25` if vertical coverage < 0.30; `P_t=0.15` if valid track mask IoU < 0.20), clashing with intuitive "penalty = +0.25"; write explicitly so the sign is unambiguous (formula_audit §5).
- Selection rule: lexicographic argmax over `(primary_score, tie_break_score)`, `max` keeps first-seen ties.
- The OLD semantic-gated softmax selector (SG-MPS, `c_i^(k) = σ(w_φᵀφ − w_ψᵀψ − b)`, temperature softmax `a_i^(k)`) MUST be described **in this same subsection** as the superseded baseline (or a supplementary note), with the rationale: contrast-in-primary scoring was diagnosed as the cause of DEV-set ranking failures (C02/C03); R1 excludes contrast and rescues all identified failures (C04). Table 5 geometry-only row (F1=0.8429) stays as the ranking-failure diagnostic.
- Note `semantic_total = 0` (semantic gate disabled by default) so the S term vanishes.

**Must NOT appear:**

- The softmax-gated selector presented as the current production selection method.
- A formula written `F = B − q_contrast − P` (drops the /5.5 — WRONG).
- Claim that R1 "improves general segmentation performance" or "statistically significantly outperforms" baselines (C05 FORBIDDEN — selection rationale allowed: failure-rescue, zero material regression, low churn, structural simplicity).
- A6/A7 introduced here (no scoring-formula connection).

**Supporting evidence:** `Phase19A_algorithm_delta.md` rows #3 (REWRITE) + #4 (ADD); `Phase19A_formula_audit.md` §3, §5; `Phase19A_method_component_audit.md` headline finding 2.

---

## Subsection 7 — Foreground-Object 2DGS Optimisation (KEEP)

**Must appear (from ForeSplat.tex §3.3-3.4, delta KEEP rows):**

- Foreground RGB loss `L_rgb-fg` (boundary band 2px ignored, `r_b`), alpha mask L1-Dice loss `L_mask` (λ_mask=0.08), background opacity suppression `L_bg` (λ_bg=0.02), warm-up schedule `χ(t)` (active after 500 iters, 1500-iter warm-up).
- View-quality-aware soft loss weighting `q_i ∈ [0.6,1.0]`, `view_weight_mode=rgb_only` (§3.4).
- Foreground-track initialisation (multiview mask consistency): confidence-weighted mask consistency `Γ(X_j)`, keep criterion V_j ≥ 3 ∧ Γ ≥ τ_track=0.9; **terminology guard**: this point-level COLMAP track filtering is NOT the A6 cross-view consensus mechanism — keep the two distinct in prose.

**Must NOT appear:**

- No A6 mechanism language attached to the track initialisation.
- No significance claim on reconstruction gains in this Methods section (P05 descriptive).

**Supporting evidence:** `Phase19A_algorithm_delta.md` rows #10, #14 (both KEEP); `Phase19A_method_component_audit.md` KEEP rows.

---

## Subsection 8 — M2M3-GO + Mask-Guided Pruning (KEEP, refinement layer)

**Must appear:**

- Mask-guided multi-cue Gaussian pruning: descriptor `z_j = [M_j, log(O_j), log(1+V_j), B_j, C_j]`, survival score `P_keep`, `pruning_mode=mask`, starts at 18,000 iters every 3,000; opacity thr 0.005, brightness 0.01, mask thr 0.45, max view count 12, max removal ratio 0.03, mask-score weight 3.0.
- M2M3-GO (mask-to-model / model-to-mask micro-matching): mask-to-model support `S^m2g`, model-to-mask coverage deficit `S^g2m`, boundary support `B^∂`, utility `U_j`, thresholded micro-decisions (Prune/Densify/Keep), Floor40 optional safeguard.
- Framing: a lightweight primitive-level refinement layer applied after the foreground target is established — NOT a standalone compression module. Modest effect size per Table 5 evidence; keep this modest framing.

**Must NOT appear:**

- Claim that M2M3-GO provides large standalone gains (evidence: modest effect, scaling claims forbidden).

**Supporting evidence:** `Phase19A_algorithm_delta.md` rows #11, #12 (KEEP); `Phase19A_method_component_audit.md` M2M3-GO KEEP row.

---

## Subsection 9 — Mesh Extraction / TSDF / Scale Recovery / Phenotyping (KEEP)

**Must appear:**

- Depth rendering + TSDF-style fusion with visibility/normal/mask-aware weights `w_c(x)`, Marching Cubes zero-level extraction (ForeSplat.tex L550-561).
- Scale recovery from known pot diameter.
- Trait measurement: plant height, canopy width, leaf length, leaf width vs manual reference (n=21 plants, n=63 leaves for leaf traits).

**Must NOT appear:**

- "Cross-species" reconstruction claims (NOT_IN_MANUSCRIPT; 3-sample descriptive only) — `Phase19A_image_asset_inventory` / method_audit guard.

**Supporting evidence:** `Phase19A_algorithm_delta.md` row #13 (KEEP); `Phase19A_method_component_audit.md` mesh/TSDF row; P07 P07 evidence.

---

## Subsection 10 — Evaluation Metrics

**Must appear (exact metric lists from claim_evidence_matrix):**

- **Segmentation**: F1, mIoU, HD95 (px), Boundary-F1, leakage energy (mask-level). Used for frozen P6 (P01), prompt robustness (P03), module ablation (P04), ranking failure/R1 rescue (C01-C04), A6/A7 ablation (C06-C08).
- **Reconstruction**: PSNR_fg, SSIM_fg, LPIPS_fg (foreground focus). Workflow-level means over 20 sequences (P05).
- **Phenotyping**: MAE, RMSE, MAPE, R² between virtual and manual measurements (P07).

**Must NOT appear:**

- Significance-test notation (stars, p-values) attached to descriptive means — claim_evidence_matrix marks all P01-P08 rows as DESCRIPTIVE (except P03 PROVEN-per-prompt-report, still descriptive means).
- Any metric defined at a different aggregation level than stated (e.g., TEST32 accuracy without GT — not allowed).

**Supporting evidence:** `Phase19A_claim_evidence_matrix.csv` P01, P03, P05, P07 `allowed_wording` + `paper_table` columns.

---

## Method-Chapter Cross-Check (Phase 19B will follow)

| Subsection | Primary source row(s) | Production default |
|---|---|---|
| §1 Workflow | algorithm_delta Rollup | modules list, no A6/A7 |
| §2 Dataset | expt asset registry P16-A06/A07; formula_audit §4 | 5 datasets, frozen P6 named |
| §3 FFT | algorithm_delta #1 | KEEP |
| §4 Candidates | algorithm_delta #2; formula_audit §2.5 | per_instance, prompt_list P2, τ=0.5 |
| §5 Scoring | formula_audit §2.1-2.2 | denom=5.5, w_sam=0.5 |
| §6 R1 ranking | formula_audit §3, §5; algorithm_delta #3/#4 | r1 primary excludes q_contrast |
| §7 2DGS | algorithm_delta #10/#14 | λ_mask=0.08, λ_bg=0.02 |
| §8 M2M3-GO+prune | algorithm_delta #11/#12 | refinement layer framing |
| §9 Mesh/pheno | algorithm_delta #13 | TSDF, pot scale, 4 traits |
| §10 Metrics | claim_evidence_matrix | descriptive, no stars |

---

*FILE B of WP6B. Date: 2026-09-14.*