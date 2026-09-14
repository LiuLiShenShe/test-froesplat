# Phase 19A — WP7B Content Exclusion List (DO-NOT-SAY / DO-NOT-DO register for the Phase 19B rewrite)

**Generated:** 2026-09-14 | **WP7B deliverable** | **Inputs:** `Phase19A_claim_audit.csv` (19 rows), `Phase19A_claim_evidence_matrix.csv` (C01–C08, P01–P08), `Phase19A_method_component_audit.md` (G11), `Phase19A_algorithm_delta.md` (F1–F4), `Phase19A_formula_audit.md` (§3–§6), `Phase19A_experiment_asset_registry.csv`, `Phase19A_reviewer_comments_raw.md`, `Phase19A_reviewer_issue_matrix.csv`.

**Purpose:** This is a binding register for the Phase 19B rewrite. Every sentence of the new manuscript must pass all four gates below:
1. **Forbidden claims** (§1) — wording that must never appear.
2. **Forbidden experiment placement** (§2) — datasets/modules/jargon that must not occupy a given paper role.
3. **Forbidden framing** (§3) — narrative positions that are not supported.
4. **Always-mention checklist** (§4) — caveats that MUST appear so reviewers do not re-raise already-flagged issues.

Evidence levels follow `Phase19A_claim_audit.csv` (`NOT_SUPPORTED` / `SUPPORTED` / `DESCRIPTIVE` / `NA`) and `Phase19A_claim_evidence_matrix.csv` (`PROVEN` / `SUPPORTED` / `DESCRIPTIVE` / `NOT_SUPPORTED`). `FORBIDDEN` and `DEMOTED` markers are quoted verbatim from the evidence matrix.

---

## 1. Forbidden claims

### 1.1 "outperforms single-prompt SAM3 / other automatic VFM baselines" (unqualified)
- **Forbidden phrase/category:** "outperforms single-prompt SAM3", "outperforms other automatic VFM baselines", any unqualified "outperform(s)" against any baseline.
- **Why forbidden (evidence):** claim_audit row 4 (`SUPPORTED` **only with rewording**): the F1 delta (0.9706 vs 0.9692, Δ=0.0014, 0.1% relative) is within annotation-variance range on the 21-frame frozen P6 set; no significance test is reported; "outperforming" implies proven general superiority. Row 5 (`NOT_SUPPORTED`): extending the claim to "other automatic VFM baselines" without naming them and without a statistical test overstates descriptive evidence on the P6 annotated frames. Evidence matrix P01 (`DESCRIPTIVE` frozen-set means), P02 (`DESCRIPTIVE` delta report): same rationale.
- **Allowed replacement:** "achieved higher boundary consistency (Boundary-F1: 0.4643 vs 0.4051) and stronger leakage suppression (leakage energy 0.001739 vs 0.002272) compared with single-prompt SAM3; improvements in region-overlap metrics (F1, mIoU) were small". For the multi-baseline form: "achieved the highest boundary consistency and lowest leakage energy among the tested automatic VFM baselines (CLIPSeg, Grounded-SAM, single-prompt SAM3, RAP-FSAM3) on the frozen P6 evaluation set". Never drop the "on the frozen P6 evaluation set" qualifier.

### 1.2 "significantly / statistically significant" without a reported test
- **Forbidden phrase/category:** "significantly", "statistically significant", "significant improvement/gain/difference" used as a statistical claim, unless a reported test (with test name, statistic, and p-value) accompanies it.
- **Why forbidden (evidence):** claim_audit row 16: the manuscript currently has zero instances and any added instance is forbidden without a reported test. Evidence matrix P01: "no significance test"; F1/mIoU deltas vs SAM3 are descriptive. C05: "R1 significantly/statistically significantly improves …" is in the `forbidden_wording` column.
- **Allowed replacement:** report descriptive values only; where the absence of tests must be acknowledged (it must — see §4 item 3), write "all comparisons are descriptive; no statistical significance testing was performed on the frozen evaluation sets".

### 1.3 "generalizes / generalization / generalise"
- **Forbidden phrase/category:** "generalizes", "generalization", "generalise", "generalise to unseen…", "generalizes across…" in any form.
- **Why forbidden (evidence):** claim_audit row 6 (`NOT_SUPPORTED`): no independent held-out evidence exists; the 20-sequence / 21-plant evaluation is a single-dataset evaluation, not a generalization study. Reinforced by B-EIC-C01 (reconstruction rests on 3 samples) and method_component_audit (cross-species claims correctly `NOT_IN_MANUSCRIPT`).
- **Allowed replacement:** "performs well on the tested sequences" or "applies to the tested acquisition settings". No global-scope wording.

### 1.4 "superior", "state-of-the-art", "SOTA", "best overall" in running text
- **Forbidden phrase/category:** "superior", "superiority", "state-of-the-art", "SOTA", "best overall", and unqualified "best" in running prose.
- **Why forbidden (evidence):** claim_audit row 7 (`NOT_SUPPORTED`): "superior" is an absolute claim — always replace with the specific metric delta. Row 8 (`NOT_SUPPORTED`): no benchmark comparison against current SOTA on a standard benchmark exists; the paper compares internal baselines only on the P6 frozen set. Row 11: "best" is acceptable **only** as table bold-value annotation clearly labelled "on the frozen P6 evaluation set"; it is forbidden in running text unless qualified by "among the tested methods on this evaluation set". Issue matrix B-R2-C02 `claim_constraint`: "Do not claim outperforms SOTA or best reconstruction; must remain as explicit limitation."
- **Allowed replacement:** "achieved [specific metric values] on the tested evaluation set" / "achieved the highest [specific metric] among the tested methods on the frozen P6 evaluation set".

### 1.5 "strong agreement" (phenotyping)
- **Forbidden phrase/category:** "strong agreement", "excellent agreement" (virtual vs manual phenotype measurements).
- **Why forbidden (evidence):** claim_audit row 9 (`DESCRIPTIVE`, allowed=FALSE): R² reported on n=21 (plant-level) and n=63 (leaf-level) with no repeated manual measurements, no Bland-Altman analysis, no confidence intervals, no inter-rater reliability. Evidence matrix P07: "strong agreement" is in `forbidden_wording`; "comparable to" is the honest wording.
- **Allowed replacement:** "virtual measurements were comparable to manual measurements (R² = 0.988, 0.988, 0.974, 0.900) on the 21-plant / 63-leaf set (descriptive; no repeated measurements collected)".

### 1.6 "robust segmentation" as a general adjective
- **Forbidden phrase/category:** "robust segmentation", "robust pipeline", "robust method", "robust to [condition]" in running text as a general quality claim.
- **Why forbidden (evidence):** claim_audit row 12 (`DESCRIPTIVE`, allowed=TRUE **only as the method name**): the word "robust" is part of the R1 policy name ("R1 robust instance ranking", Policy R1), not a general algorithmic-robustness claim. method_component_audit headline finding 2 and algorithm_delta F1 confirm the policy name is the only sanctioned occurrence.
- **Allowed replacement:** keep "R1 robust instance ranking" as the policy name; everywhere else use the specific metric or mechanism ("reduced leakage", "higher boundary consistency") instead of "robust".

### 1.7 "A6/A7 improve segmentation / provide consistent benefit / retained as production default"
- **Forbidden phrase/category:** "A6 improves/cross-view consensus improves", "A7 improves/memory propagation improves", "A6/A7 provide consistent benefit", "A6/A7 retained as production default", "A6/A7 are part of the production pipeline".
- **Why forbidden (evidence):** evidence matrix C06 (`NOT_SUPPORTED`, DEMOTED): E_A6 ≈ −0.003, one material regression ΔF1 = −0.058 on ChangShouHua2_0075. C07 (`NOT_SUPPORTED`, DEMOTED): E_A7 ≈ 0 under valid execution exposure (13 propagated / 9 selected of 21 frames on Easy21; Phase14.1 corrected lifecycle — never cite the invalid Phase14 zero-propagation observation). C08 (`NOT_SUPPORTED`, FALSE): production defaults are `--use_cross_view_consensus False`, `--use_memory_propagation False`. P08: "do NOT add A6/A7 to the main Method (method_component_audit G11 PASS)". formula_audit §6: A6/A7 contribute no term to the scoring formula; they are post-selection experimental branches, default OFF.
- **Allowed replacement (negative-ablation / limitations framing only):** "A6 (cross-view consensus) and A7 (memory propagation) did not provide consistent measurable benefit under the evaluated settings; both are disabled by default (`--use_cross_view_consensus False`, `--use_memory_propagation False`)". A7 may additionally be described as future work in the Discussion (SAM2/3 temporal propagation), never as a realized stage.

### 1.8 C05 "R1 significantly improves general segmentation performance" — FORBIDDEN even as a caveated claim
- **Forbidden phrase/category:** "R1 significantly improves general segmentation performance", "R1 improves segmentation significantly", "R1 generalizes/outperforms", or any claim that R1's superiority (rather than its selection rationale) is the result.
- **Why forbidden (evidence):** evidence matrix C05 (`NOT_SUPPORTED`, **FORBIDDEN**, flagged verbatim: "C05 flagged FORBIDDEN in the Phase19B rewrite checklist"). N=10 mean-F1 comparisons (GT10) are descriptive only. This claim must not appear even with a caveat prefix; it is not a weakened version of an allowed claim — the claim itself is not allowed in any form.
- **Allowed replacement (selection rationale only):** "R1 was selected for failure-rescue (rescued the identified DEV-set failures DouBanLv2 0039/0040/0041 to F1 ≥ 0.95), zero material regression on Easy21, low selection churn (6.2% on TEST32), and structural simplicity — not because its mean F1 is significantly higher (N=10 mean-F1 comparisons are descriptive only)." The narrative is "rescue + zero regression + low churn", never "superiority".

---

## 2. Forbidden experiment placement

### 2.1 GT10 (DEV10) as a held-out test set; TEST32 as accuracy evidence
- **Forbidden placement:** labelling GT10 (DEV10) a "held-out test set" or "independent test set"; citing TEST32 metrics (F1/IoU or any GT-based accuracy) as evidence of correctness.
- **Why forbidden (evidence):** asset P16-A06: GT10 = 10 annotated GT frames, 2 sequences (BaiZhang 0033–0037, DouBanLv2 0037–0041) of the 58 frozen challenge targets (26 DEV + 32 TEST) — development-set subset, explicitly **NOT held-out**; it exists for failure diagnosis and R1 rescue demonstration. Asset P16-A07: TEST32 = 32 unlabeled frames from 8 samples; behavioral-safety audit only, **NO ground truth**; cannot support accuracy claims.
- **Allowed placement:** GT10 described as development-set diagnostic evidence (oracle audit, ranking-failure analysis, R1 rescue: C01, C02, C04); TEST32 described **only** for behavioral safety (0 empty masks under baseline and R1, selection churn 6.2% at threshold <30%, mean mask IoU 0.9375 — a behavior/safety statistic, not an accuracy claim).

### 2.2 A6/A7 in the main Method pipeline
- **Forbidden placement:** any description of A6 (cross-view consensus) or A7 (memory propagation) as a stage of the main method pipeline in the Method chapter.
- **Why forbidden (evidence):** method_component_audit G11 gate PASSED: A6/A7 are correctly absent from the manuscript Method and production defaults are OFF; the correct treatment is negative ablation (Phase18 Figure_D) and Limitations only. algorithm_delta F2 confirms no Method action needed; formula_audit §6 confirms they do not enter the scoring formula.
- **Allowed placement:** Results ablation section (descriptive negative result) and Limitations; A7 future-work mention in Discussion (§4.5-style "SAM2/3 temporal propagation") is fine and should be retained.

### 2.3 Phase numbers, git SHAs, internal workflow jargon in the main text
- **Forbidden placement:** "Phase 12 / 16 / 17 / 17A / 17B / 17C / 18", "S20", "DEV10 / DEV-set", "WPx", git commit SHAs (`d704faa8`, `26da29e6`, `d0a9970b`, `1103f6f9`, etc.), and workflow-run identifiers in the paper main text (Abstract, Introduction, Methods, Results, Discussion).
- **Why forbidden:** these are internal experiment-management identifiers from the rework pipeline, not scientific vocabulary; their presence signals provenance leakage and reads as workflow jargon to reviewers. The evidence they stand for is already expressed by asset/claim IDs in the provenance layer, which belongs in the response-to-reviewers / supplementary reproducibility notes, not the main text.
- **Allowed placement:** dataset/set names rendered in scientific form — "frozen P6 evaluation set" (21 annotated frames), "GT10 (10 annotated frames from two development sequences)", "TEST32 (32 unlabeled frames, behavioral safety only)", "R1 robust instance ranking". Commit-level provenance may appear only in the supplementary reproducibility appendix.

### 2.4 q_contrast described as part of the production primary score
- **Forbidden placement:** writing the production R1 primary score as if `q_contrast` is a positive contributor to selection; writing `F = B − q_contrast − P` (dropping the /5.5 denominator).
- **Why forbidden (evidence):** algorithm_delta F1 and formula_audit §3: production R1 primary score is `primary_score = base_total − q_contrast/5.5 − penalty` with `denom = 5.5` frozen (`ranking_policy.py`); `q_contrast` is excluded from ranking and retained only as a diagnostic feature. The `/5.5` must stay explicit because the weight sum stays frozen at 5.5 (writing `/1.0` breaks the derivation). formula_audit §3.3: `F = B − q_contrast − P` is an outright error.
- **Allowed placement:** R1 formula written exactly as `F = B − q_contrast/5.5 − P`, `B = (q_area + q_comp + q_edge + q_temp + q_contrast + 0.5·q_sam)/5.5`, `P = 0.25·1[V < 0.30] + 0.15·1[track_mask valid ∧ IoU < 0.20]`, `denom = 5.5` frozen, `semantic_total = 0` (gate off), selection = lexicographic argmax on `(primary_score, tie_break_score)` with tie_break = 0.0 under R1. `q_contrast` may be described as a diagnostic retained in the score record.

### 2.5 P6_raw_baseline mislabelled as "P6-prompt-set output"
- **Forbidden placement:** attributing the Phase 12 frozen baseline metrics (f1_mean = 0.983498) to the **P6 prompt set** ("whole potted plant including the pot").
- **Why forbidden (evidence):** formula_audit §4 and asset P12-A01 note: `P6_raw_baseline` is a historical frozen **run identifier**, not a prompt set; it was produced under production-valid config with the **default `prompt_list = "P2"`** ("entire plant excluding pot"). The token "P6" is ambiguous (prompt set vs run name) and the ambiguity is resolved by describing the run's actual configuration.
- **Allowed placement:** "Phase 12 frozen baseline (designated `P6_raw_baseline`, produced under the production default prompt list P2)" — state the config explicitly whenever this number is quoted. Any genuine P6-prompt-set ablation must use a fresh `--prompt_list P6` (or `P2,P6`) run, never the Phase 12 baseline numbers.

---

## 3. Forbidden framing

### 3.1 "cross-species" / "validated across species"
- **Forbidden framing:** "cross-species" in the title or abstract as a reconstruction-level claim; "validated across species", "multi-species validation" implying statistical coverage.
- **Why forbidden (evidence):** B-EIC-C01: the reconstruction validation — the paper's core claim — rests on **3 architecturally representative samples**; mask/phenotype-level evidence spans more species (10 species in phenotype set, 5 species in the 21-frame GT segmentation set), but that does not rescue a reconstruction-level species-generalization claim. method_component_audit: cross-species claims are `NOT_IN_MANUSCRIPT`; do not add them.
- **Allowed framing:** "architectural diversity" at the reconstruction level (descriptive on the 3 samples, with the explicit limitation that species-level generalization is not established); "cross-species applicability" **only** at the mask/phenotype level. Issue matrix B-EIC-C01 `claim_constraint`: "Do not claim cross-species generalization at reconstruction level; use cross-species applicability or architectural diversity".

### 3.2 Fabrication of reviewer comments / missing-comment invention
- **Forbidden framing:** inventing, renaming, or renumbering reviewer comment IDs; attributing a comment to a reviewer who did not make it; addressing a comment ID that does not exist in the issue matrix.
- **Why forbidden (evidence):** `Phase19A_reviewer_issue_matrix.csv` fixes exactly **32 comment IDs** (A1-C01–C05, A2-C01–C06, B-EIC-C01–C04, B-R1-C01–C04, B-R2-C01–C04, B-R3-C01–C04, B-DA-C01–C05). The response-to-reviewers document must cover every one of the 32 and no others. Stream A (real reviews) and Stream B (simulated panel) are audit inputs; neither may be silently dropped or supplemented.
- **Allowed framing:** each response maps 1:1 to an existing `comment_id`; the summary table in `Phase19A_reviewer_comments_raw.md` (32 comments) is the count of record.

### 3.3 Hand-filled approximation numbers
- **Forbidden framing:** writing any metric value that does not trace to a source artifact; "≈" or "about X" values entered from memory or by rounding without a source file; inventing deltas, p-values, or percentages.
- **Why forbidden (evidence):** Phase 18 integrity rule (asset P18-A15, P18-A17): every figure in the paper must trace to a source CSV / evidence-registry row. Each quoted number should map to a registry asset, e.g.: P12-A01/P12-A02 (frozen P6 baseline and 21-frame metrics), P14-A04 / P14.1-A05 (A6/A7 factorial — cite the **corrected Phase14.1** A7 exposure, never the invalid Phase14 zero-propagation), P16-A06/A07 (GT10/TEST32 manifests), P17B-A09/A10 (oracle + detection diagnosis), P17C-A11–A14 (policy comparison, ranking benchmark, churn, Easy21 safety), P18-A15/A16 (final tables/figures), P18-A17 (evidence registry C01–C08).
- **Allowed framing:** every number carries its source asset/CSV reference; any number not in the registry is treated as non-existent and must not appear.

---

## 4. Always-mention checklist (must appear so reviewers don't re-raise)

These items are **required** content in the Phase 19B rewrite — the absence of each one was already flagged (or is directly preemptable from flagged comments), so each must appear explicitly in the indicated location.

1. **GT10 N=10 / 2-sequences not-held-out caveat** — wherever GT10-based rescue/diagnostic results are reported (Results §3.3), state: 10 annotated GT frames from 2 development sequences, development-set evidence, **not** a held-out test set (preempts B-DA-C02 replication concern and B-R2 style generality complaints).
2. **TEST32 unlabeled status** — wherever TEST32 is mentioned: 32 frames, no ground truth, used for behavioral safety / selection-churn analysis only; no accuracy claim (preempts any reviewer asking for TEST32 accuracy).
3. **No significance tests** — Methods/Results must state that all comparisons are descriptive and no statistical significance testing was performed on the frozen evaluation sets (preempts C05-adjacent and P01-adjacent "significant" questions; see §1.2).
4. **Single-morphology injection ablation** — the foreground-object loss-injection ablation (P06) is a single representative broad-leaf sample; must be described as a descriptive, non-replicated development-set finding, not a general result (preempts B-DA-C02 single-sample replication concern applied to the injection ablation).
5. **Leaf-width downstream limitation** — leaf width (R² = 0.900, MAPE = 9.73%) is the weakest trait and its error attribution (segmentation boundary vs 2DGS surface dilation vs TSDF meshing) is **not resolved**; point-cloud measurement noted as future direction (A1-C03, `claim_constraint`: "Do not claim leaf width accuracy is resolved or high-precision; state as open limitation").
6. **No external reconstruction baselines** — explicitly acknowledge that no COLMAP+MVS, NeRF (Instant-NGP/Zip-NeRF), or external 3DGS baseline comparison was performed, and why internal mechanism questions were answered by within-method ablation; commit to follow-up external-baseline study (B-R2-C02, `claim_constraint`: "must remain as explicit limitation").
7. **FFT threshold sample-dependence** — the per-sequence first-quartile FFT threshold is a sample-dependent heuristic; no fixed threshold calibrated on a held-out set was evaluated; acknowledge as heuristic and note sensitivity analysis as future work (B-DA-C04, `claim_constraint`: "Do not claim optimal threshold; acknowledge as heuristic").

---

## 5. Quick-reference forbidden-token list

Apply a final mechanical scan for these tokens in the Phase 19B main text (each must be absent except where §1/§2/§3 explicitly allow a qualified form):

| Token | Status | Permitted exception |
|---|---|---|
| outperform / outperforming | Forbidden unqualified (§1.1) | Specific metric deltas + "on the frozen P6 evaluation set" |
| significant(ly) / statistically significant | Forbidden without a reported test (§1.2) | Descriptive-only statement of no tests (§4.3) |
| generaliz(e/ation/es) / generalise | Forbidden (§1.3) | "performs well on the tested sequences / tested acquisition settings" |
| superior / state-of-the-art / SOTA / best overall | Forbidden in running text (§1.4) | "highest [metric] among the tested methods on the frozen P6 evaluation set"; "best" only as qualified table annotation |
| strong agreement | Forbidden (§1.5) | "comparable to manual measurements (R² …)" |
| robust segmentation / robust pipeline | Forbidden as adjective (§1.6) | **only** "R1 robust instance ranking" (policy name) |
| A6/A7 improve / consistent benefit / production default | Forbidden (§1.7, §2.2) | Negative ablation + Limitations; "disabled by default (`--use_cross_view_consensus False`, `--use_memory_propagation False`)" |
| R1 significantly improves general segmentation | Forbidden in any form (§1.8) | Selection rationale: rescue + zero material regression + low churn + simplicity |
| held-out (applied to GT10) | Forbidden (§2.1) | "development-set diagnostic evidence, N=10, 2 sequences, not held out" |
| TEST32 accuracy / TEST32 F1 / TEST32 IoU | Forbidden (§2.1) | Behavioral safety: 0 empty outputs; selection churn 6.2% |
| Phase 12/16/17/17A/17B/17C/18, S20, DEV10, git SHAs | Forbidden in main text (§2.3) | Supplementary reproducibility appendix only |
| q_contrast in primary score; `F = B − q_contrast − P` | Forbidden (§2.4) | `F = B − q_contrast/5.5 − P`, denom = 5.5 frozen; q_contrast = diagnostic |
| P6_raw_baseline = P6-prompt-set output | Forbidden (§2.5) | "Phase 12 frozen baseline, produced under default prompt list P2" |
| cross-species (reconstruction-level) | Forbidden (§3.1) | "architectural diversity" (3-sample descriptive); "cross-species applicability" at mask/phenotype level only |
| invented comment IDs / fabricated reviewer text | Forbidden (§3.2) | Exactly the 32 issue-matrix IDs |
| hand-filled / memory-derived numbers | Forbidden (§3.3) | Every number traced to a registry asset CSV |

---

## WP7B SELF-CHECK: ALL PASS

WP7B exclusion list complete on **2026-09-14**. All four gates written (§1 forbidden claims incl. C05 FORBIDDEN; §2 forbidden placement incl. GT10/TEST32, A6/A7, jargon, q_contrast, P6_raw_baseline; §3 forbidden framing incl. cross-species, comment fabrication, hand-filled numbers; §4 always-mention checklist with all 7 caveats). Every entry carries evidence level and source traceability from claim_audit / evidence_matrix / method_component_audit / algorithm_delta / formula_audit / asset_registry / issue_matrix. No source file was modified.
