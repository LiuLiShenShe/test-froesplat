# Phase 18 — Paper Consolidation + Final Experiment Audit

**Protocol** | 阶段十八_论文实验收束与最终审计

Scope: evidence audit, historical consistency correction, final algorithm definition, paper-ready tables/figures, claim/evidence mapping, reproducibility package, final architecture decision. **No algorithm development.**

---

## §0 — Starting-State Audit

| Reference | SHA | Notes |
|---|---|---|
| Phase18 starting SHA | `d704faa8` | HEAD at Phase18 start |
| Phase17C final | `d704faa8` | Phase17C commit (amended from 260d81f0) |
| Stage 17B | `26da29e6` | Phase17B commit |
| Phase17A | `d656851e` | Phase17A follow-up commit |
| Phase14.1 | `79471fe7` | A7 validity closure |
| Frozen P6 | `b970913d` | 阶段十二 P6 正式验收 |
| Phase16 TEST manifest frozen | `1103f6f9` | 58-frame challenge freeze |

Working tree at Phase18 start: **clean**. Branch: `main`.

---

## §1 — No Algorithm Development (Freeze)

**ABSOLUTELY FORBIDDEN:** new ranking rules, re-tune q_contrast, modify score weights, modify thresholds, modify P6 prompt, re-enable A6/A7, add A8/A9 modules, adjust algorithm based on paper-table results.

**ALLOWED ONLY:** documentation correction, analysis-script correction, table generation, figure generation, reproducibility metadata, explicit reporting-bug fixes.

If a genuine production-algorithm bug is discovered: **stop Phase18 immediately, report separately.** Do not silently fix.

---

## §2 — Evidence Registry

**File:** `Phase18_evidence_registry.csv`
Each row = one paper-usable claim. Fields:
`claim_id, claim_text, module, phase, source_report, source_csv, source_commit, dataset, N_frames, N_samples, GT_available, evidence_type, evidence_level, allowed_wording, forbidden_wording, paper_section`

Evidence levels: `PROVEN / SUPPORTED / DESCRIPTIVE / BEHAVIOR_ONLY / NOT_SUPPORTED`

---

## §3 — Claim Matrix (C01–C08)

| ID | Claim | Level |
|---|---|---|
| C01 | Baseline candidate generator can produce high-quality whole-plant masks even when final F1=0 | PROVEN (identified failure frames) |
| C02 | Identified catastrophic failures caused by candidate ranking, not generation | PROVEN (DouBanLv2 0039–0041) |
| C03 | q_contrast can favor small, high-contrast distractor masks | SUPPORTED |
| C04 | Removing q_contrast from primary ranking eliminates identified DEV-set ranking failures | SUPPORTED |
| C05 | "R1 significantly improves general segmentation performance" | NOT SUPPORTED / FORBIDDEN |
| C06 | A6 cross-view consensus provides consistent improvements | NOT SUPPORTED |
| C07 | A7 memory propagation provides consistent improvements | NOT SUPPORTED |
| C08 | A6/A7 retained as default production enhancements | FALSE |

May be refined during evidence-registry build. C03 level decided by Phase17C stats (failure-frame Spearman −0.7187 intra-bin + ablation 3/3 rescue → SUPPORTED, not PROVEN globally).

---

## §4 — Historical Corrections Audit (Phase 11–17C)

Scan all historical `*.md / *.csv / *.json` for stale statements: prompt failure, candidate generation failure, A7 neutral/depends-on-A6, A6+A7 synergy, mild regression, A6/A7 improves, test counts (67/71/84/100/121/131/142), "HEAD remains / uncommitted / not pushed".

Classifications:
- **SUPERSEDED** — correct-at-the-time but superseded later
- **STALE_METADATA** — commit/SHA/test-count references no longer current
- **INCORRECT** — factually wrong

Write `Phase18_historical_corrections.md`. Never overwrite old reports.

---

## §5 — Phase17A → Phase17B Conclusion Evolution

Record explicitly in `Phase18_historical_corrections.md`:

- Phase17A (hypothesis): F1=0 may indicate upstream / prompt-candidate generation failure
- Phase17B (proved): candidate generation succeeded — raw oracle F1 ≈ 0.987; root cause = ranking failure
- Paper final wording: "High-quality candidate masks were present, but the original candidate-ranking function selected distractor instances." NOT "P6 failed to detect DouBanLv2 0039–0041."

---

## §6 — Phase17C Stale Metadata Correction

`Phase17C_report.md` §12 still says "HEAD remains at 26da29e6 ... uncommitted". Actual: HEAD = `d704faa8`, committed, tree clean. Add correction note in `Phase18_historical_corrections.md` — do not rewrite the historical report.

---

## §7 — Freeze Final Algorithm Definition

**File:** `Phase18_final_algorithm_spec.md`
Code-level language. q_contrast computed for diagnostics but excluded from R1 primary. A6/A7 optional experimental branches, disabled by default. Method flow must reflect real production default (R1 ranking, A5c refinement if enabled).

---

## §8 — Final Method Flow (§7 continued)

```
Image
 ↓
SAM3 multi-instance candidate generation
 ↓
Candidate feature extraction
 ↓
Structural/object plausibility scoring (R1 primary, q_contrast excluded)
 ↓
R1 robust instance ranking
 ↓
A5c / final refinement if enabled in actual default
 ↓
Selected plant mask
```

A6/A7: optional ablation branches only (or in ablation chapter).

---

## §9 — Dataset Registry

**File:** `Phase18_dataset_registry.csv`

| Dataset | Frames | Samples | GT | Role | Can tune? | Accuracy claim? |
|---|---|---:|---|---|---|---|
| Easy21 | 21 | 5 | YES | historical safety/easy anchor | NO | YES, historical |
| GT10 (DEV10) | 10 | 2 | YES | diagnostic DEV / pilot | YES historically used | descriptive only |
| TEST32 | 32 | 8 | NO | behavioral safety | NO | NO |
| Frozen P6 | 21 | 5 | YES | historical baseline | NO | YES, historical |
| Context P00 | 20 | 2 | NO | candidate audit context | — | NO |

GT10 is NOT held-out. TEST32 has no GT.

---

## §10 — Easy21 Final Performance Table

Two sub-tables (§10 protocol): **historical ablation** (V00/V10/V01/V11 with Phase14 vs Phase14.1-corrected numbers split by code-state) and **ranking-correction analysis** (R0/R1 baseline vs R1 winner on Easy21). Sources: Phase14/14.1 manifests, Phase17C policy comparison. **Auto-generated** in `build_paper_tables.py` — never hand-copied.

Frozen P6 baseline anchor: mean F1 = 0.9835 (Phase12 指标/P6_汇总.json).

---

## §11 — GT10 Final Table

| Variant/Policy | Mean F1 | Median F1 | Min F1 | Material regressions | Catastrophic failures |
|---|---:|---:|---:|---:|---:|

Original baseline vs R1. Label: N=10 frames from 2 sequences, DESCRIPTIVE DEVELOPMENT/PILOT EVIDENCE. No significance claims.

---

## §12 — Failure-Rescue Table

| Frame | Baseline F1 | Raw oracle F1 | R1 selected F1 | Oracle gap closure | Failure cause |
|---|---:|---:|---:|---:|---|
| DouBanLv2_0039 | 0 | 0.9867 | ≥0.95 | C3 ranking | q_contrast distractor |
| DouBanLv2_0040 | 0 | 0.9882 | ≥0.95 | C3 ranking | q_contrast distractor |
| DouBanLv2_0041 | 0 | 0.9865 | ≥0.95 | C3 ranking | q_contrast distractor |

Read automatically from CSV (Phase17C policy comparison + Phase17B oracle). No hand-fill.

---

## §13 — Score-Component Mechanism Table

Per failure frame: oracle whole-plant candidate vs wrong selected distractor — q_area/q_comp/q_edge/q_temp/q_contrast/total old/R1 score/F1. Source: `Phase17C_ranking_benchmark.csv`. Show small distractor saturates contrast and wins old ranking; final wording only to the extent Phase17C data supports.

---

## §14 — Ranking Ablation Table

| Policy | DEV10 mean F1 | Catastrophic failures | Oracle regret | Easy21 material regressions | Selection churn |
|---|---:|---:|---:|---:|---:|

R0/R1/R2/R3. Winner R1 chosen by §14 priority (rescue → non-regression → oracle regret → low churn → simplicity), NOT because R1 mean F1 is numerically highest. If R1/R2/R3 scores identical → R1 is structurally simplest.

---

## §15 — A6/A7 Negative Ablation Table

| Module | Easy-set effect | GT10 effect | Exposure | Failure/regression | Final status |
|---|---:|---|---|---|---|
| A6 | near-neutral (−0.003) | 4/4 acted frames regress | min-frame limited | 1 material regr. Easy21 | DEMOTED |
| A7 | near-neutral | 1 marginal improvement | valid propagation after 14.1 | 0 | DEMOTED |
| A6+A7 | near-neutral | 4/4 regress | — | 1 material regr. | DEMOTED |

Wording: "did not provide consistent measurable benefit under the evaluated settings" — not "failed completely".

---

## §16 — Final Figures (Figures A–D)

`figures/` (PDF+PNG). Bilingual EN/中 labels. Deterministic, no internet, no inference.

- **A — Failure mechanism** (0039/0040/0041): RGB / GT / wrong baseline / oracle / R1-selected mask, F1 labels.
- **B — Candidate ranking** (e.g. 0039): q_contrast + other components, old total rank vs R1 rank.
- **C — Easy vs GT10 distributions**: Easy21 baseline F1, GT10 baseline F1, GT10 R1 F1 (descriptive — no significance stars).
- **D — A6/A7 ablation**: Control/A6/A7/A6+A7 (descriptive).

---

## §17 — Recompute Every Paper Number From Source

`build_paper_tables.py` / `build_paper_figures.py`: deterministic, no internet, no inference, read-only historical evidence. No manual number copying.

---

## §18 — Paper-Number Provenance

**File:** `Phase18_paper_number_provenance.csv`: `paper_label, value, source_file, source_column, filter, aggregation, source_commit`.

---

## §19 — Statistical Restraint

Check all reports/paper-draft artifacts: no "significant / statistically significant / generalizes / robustly improves / outperforms / superior / state-of-the-art" without independent evidence. GT10: development-set / pilot / descriptive / identified failure mode. TEST32: behavioral safety / execution stability / selection churn. No accuracy claims for TEST32.

---

## §20 — Contribution Hierarchy

**Primary:** robust per-instance candidate ranking for SAM3-based plant segmentation; candidate-level failure attribution (failure can originate from ranking despite high-quality candidates).
**Secondary:** candidate scoring framework, hard-case diagnostic pipeline, weak-label workflow.
**Ablation/negative:** A6/A7 — evaluated but not retained.

---

## §21 — Architecture Decision Document

**File:** `Phase18_architecture_decision.md` — per-component RETAIN / DEMOTE / REMOVE / DEFAULT with reasons.

---

## §22 — Production Defaults vs Paper Consistency

Check: `--ranking_policy default == r1`, `A6 default == false`, `A7 default == false`. Add `tests/test_phase18_paper_code_consistency.py` (ranking policy default, A6/A7 defaults, q_contrast still in diagnostics).

---

## §23 — Reproducibility Manifest

**File:** `Phase18_reproducibility_manifest.json`: final git SHA, Python, CUDA, GPU, SAM3 checkpoint, ranking_policy, score config, A6/A7 defaults, dataset sources, Phase17B/17C commits, historical test count.

---

## §24 — Historical Test Audit

142/142 historical + Phase18 consistency tests → total PASS. No historical expectation edits.

---

## §25 — Artifact Completeness

Directory must contain: protocol, report, evidence registry, dataset registry, historical corrections, final algorithm spec, architecture decision, reproducibility manifest, 6 paper tables, provenance, figures/, tests/, build scripts.

---

## §26 — Final Paper-Ready Narrative

Fact order:
1. Easy-set performance near ceiling.
2. Hard-case pilot exposes catastrophic failures.
3. Candidate-level oracle analysis shows high-quality masks already exist.
4. Therefore failure lies in candidate ranking, not generation.
5. Score decomposition identifies q_contrast as dominant misleading component.
6. R1 removes q_contrast from primary ranking.
7. R1 eliminates identified DEV failures with no material regression on Easy21.
8. A6/A7 do not provide consistent benefit; disabled by default.

NOT "SAM3 cannot detect hard plants".

---

## §27 — Final Limitations Paragraph

GT10: only 10 frames from 2 sequences; participated in diagnosis/development. TEST32 unlabeled. R1 has not undergone independent GT-based held-out confirmation. A6/A7 conclusions limited to evaluated datasets/settings. No broad generalization claim.

---

## §28 — Final Decision

OPTION A (PAPER READY) if all numbers traceable, reports consistent, production defaults consistent, 142 historical tests pass, figures reproducible, claims restrained. Else OPTION C if evidence inconsistency. OPTION B (held-out GT validation) is an **optional** future note, non-blocking per user choice.

---

## §29 — Git Discipline

Commit only Phase18 new artifacts (docs/scripts/tables/figures/tests). Verify: no historical result overwritten, no GT modified, no Phase17B oracle modified, no Phase17C raw evidence modified, no production algorithm changed. **NO push.**

---

## §30 — Final Response Format

Mandatory output format per protocol §30 (Starting state / Final production algorithm / Evidence registry / Historical corrections / Dataset roles / Final ranking evidence / policy ablation / Easy21 safety / GT10 dev result / A6A7 status / tables / figures / provenance / code-paper consistency / regression tests / evidence integrity / final claims / limitations / architecture decision / final verdict).

---

## Execution Log

| # | Step | Verification | Status |
|---|---|---|------|
| 0 | Starting audit | BRAVED-SHA registry, clean tree | ✅ |
| 1–6 | Corrections + algorithm spec | historical corrections doc | ✅ |
| 7 | Paper tables | build_paper_tables.py → 6 .csv | ✅ |
| 8 | Figures A–D | build_paper_figures.py | ✅ |
| 9 | Provenance | paper_number_provenance.csv | ✅ |
| 13 | Consistency tests | tests/test_phase18_*.py | ✅ |
| 14 | Regression | 142/142 + Phase18 | ✅ |
| 23 | Reproducibility manifest | manifest .json | ✅ |
| 29–30 | Report + commit | Phase18_report.md, no push | ✅ |