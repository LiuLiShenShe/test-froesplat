# Phase 18 — Final Report

**阶段十八_论文实验收束与最终审计** | Paper Consolidation + Final Experiment Audit
Date: 2026-09-10 | HEAD: `d704faa8` (Phase 17C commit) | Working tree: clean

---

## §1 — Starting State

| Item | Value | Verified |
|---|---|---|
| HEAD SHA | `d704faa8` (Phase 17C, amended from `260d81f0`) | ✅ |
| Phase 17B SHA | `26da29e6` | ✅ |
| Phase 17A SHA | `d656851e` | ✅ |
| Phase 14.1 SHA | `79471fe7` | ✅ |
| Frozen P6 SHA | `b970913d` | ✅ |
| Phase 16 TEST manifest SHA | `1103f6f9` | ✅ |
| Working tree at start | clean | ✅ |

---

## §2 — Final Production Algorithm

Production code: `S20/.../生成RAP-FSAM3掩膜.py` + `ranking_policy.py`

```
Input: RGB image (or sequential frame pair)
 ↓
SAM3 candidate generation (per_instance, prompt_list=P2, sam3_mask_threshold=0.5)
 ↓
Per-candidate feature extraction:
  q_area, q_comp, q_edge, q_temp, q_contrast, q_leak, q_side, sam_score
 ↓
Candidate scoring (denom=5.5 frozen, full weights):
  base_total = (Σ w_i·q_i + 0.5·q_sam) / 5.5
  penalty = 0.25 if vertical_coverage < 0.30, else 0
  semantic_total = 0 (gate disabled)
 ↓
R1 robust instance ranking:
  primary_score = base_total − q_contrast/5.5 − penalty
  (q_contrast excluded from primary; retained for diagnostics)
  selection: argmax(primary_score) per instance branch
 ↓
Output: selected segmentation mask
```

Production defaults verified (8/8 consistency tests pass):
- `--ranking_policy`: default `r1`
- `--use_cross_view_consensus`: default `False` (store_true)
- `--use_memory_propagation`: default `False` (store_true)
- `--score_weights`: `area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5`
- `--prompt_list`: `P2`
- `--candidate_mode`: `per_instance`

---

## §3 — Evidence Registry

`Phase18_evidence_registry.csv`: 8 claims (C01–C08) mapped.

| Claim | Level | Paper section |
|---|---|---|
| C01: Baseline candidate generator produces high-quality masks even when final F1=0 | PROVEN | Results (oracle) |
| C02: Identified catastrophic failures caused by candidate ranking, not generation | PROVEN | Results |
| C03: q_contrast can favor small high-contrast distractor masks | SUPPORTED | Results / Discussion |
| C04: Removing q_contrast from primary ranking eliminates identified DEV-set failures | SUPPORTED | Results |
| C05: R1 significantly improves general segmentation | **NOT SUPPORTED** | **FORBIDDEN** |
| C06: A6 provides consistent improvements | **NOT SUPPORTED** | Limitations / Ablation |
| C07: A7 provides consistent improvements | **NOT SUPPORTED** | Limitations / Ablation |
| C08: A6/A7 retained as default production enhancements | **FALSE** | Architecture decision |

---

## §4 — Historical Corrections

4 active stale/corrected claims from Phase 11–17C, fully documented in `Phase18_historical_corrections.md`:

1. **A7 depends on A6** (Phase14_report.md L117): **SUPERSEDED** → A7 does NOT depend on A6; corrected in Phase14.1.
2. **A7 neutral** (Phase14_report.md L20): **INCORRECT** (invalid evidence) → original conclusion based on A7 failing to execute (cuda_oom); Phase14.1 corrected.
3. **1 mild regression** (Phase14_report.md L15): **CORRECTED** → ΔF1 = −0.058 = material regression, not mild.
4. **P6 prompt-candidate generation failure** (Phase17A L250): **REFUTED** → Phase17B proved candidate generation succeeded (oracle F1 ≈ 0.987).

**Phase17A → 17B evolution:** Paper must NOT write "P6 failed" or "candidate generation failed". Paper MUST write: "High-quality candidate masks were present, but the original candidate-ranking function selected distractor instances."

**Phase17C stale metadata:** 4 stale SHA/commit references in Phase17C_report/protocol/manifest corrected in `Phase18_historical_corrections.md` §6. Historical reports not rewritten.

**Historical test counts:** 71→84→100→121→131→142 confirmed temporally accurate.

---

## §5 — Dataset Roles

| Dataset | N | GT | Role | Accuracy claims |
|---|---|---|---|---|
| Easy21 | 21 frames / 5 samples | YES | Historical safety / easy anchor | YES (historical only) |
| GT10 (DEV10) | 10 frames / 2 samples | YES | Diagnostic DEV / pilot | Descriptive only, **not held-out** |
| TEST32 | 32 frames / 8 samples | NO | Behavioral safety | NO — no GT |
| Frozen P6 | 21 frames / 5 samples | YES | Historical baseline | YES (historical) |
| Context P00 | 20 frames / 2 samples | NO | Candidate audit context | NO |

---

## §6 — Final Ranking Evidence

R1 chosen by Phase 17C pre-registered decision order (§14):
1. Rescue all 3 identified failures: R1 = **3/3 rescued** (F1 ≥ 0.95)
2. Zero material DEV regressions (ΔF1 < −0.005): R1 = **0**
3. Zero material Easy21 regressions: R1 = **0**
4. Oracle regret: R1 = **0.0000** (oracle candidate selected in all 3 failure frames)
5. Selection churn: TEST32 behavioral audit = **6.2%** (2/32, below 30% threshold)
6. Simplest mechanism: R1 removes one term; R2 adds tiebreak; R3 adds area modulation

**Failure rescue mechanism (§26 narrative):**
- Baseline F1 = 0.0000 in DouBanLv2 0039/0040/0041
- Raw oracle F1 = 0.9867 / 0.9882 / 0.9865 (high-quality masks present)
- Root cause: q_contrast (contrast signal) favors small high-contrast distractor over whole-plant
- R1 removes q_contrast from primary ranking → selects correct whole-plant candidate → F1 = oracle F1

---

## §7 — Policy Ablation

| Policy | DEV10 mean F1 | Catastrophic | Oracle regret | Easy21 regressions | Churn (DEV/Easy) |
|---|---:|---:|---:|---:|---|
| R0 (baseline) | 0.6767 | 3 | 0.2961 | 0 | 0%/0% |
| **R1 (winner)** | **0.9728** | **0** | **0.0000** | **0** | 30%*/0% |
| R2 | 0.9728 | 0 | 0.0000 | 0 | 30%*/0% |
| R3 | 0.9728 | 0 | 0.0000 | 0 | 30%*/0% |

*R1/R2/R3 produce identical results on all 32 tested frames; 30% DEV churn = exactly the 3 rescued frames.

---

## §8 — Easy21 Safety

**Table 1a — Historical A6/A7 factorial:**

| Variant | A6 | A7 | F1 mean | F1 median | Regressions | Code state |
|---|---|---|---|---|---|---|
| V00 Control | OFF | OFF | 0.983498 | 0.983911 | 0 | Phase14 |
| V10 A6 | ON | OFF | 0.9807 | 0.9836 | 1 (ChangShouHua2_0075, ΔF1=−0.058) | Phase14 |
| V01-R1 A7 | OFF | ON | 0.9836 | 0.9850 | 0 | Phase14.1 corrected |
| V11-R1 A6+A7 | ON | ON | 0.9807 | 0.9833 | 1 (ChangShouHua2_0075) | Phase14.1 corrected |

**Table 1b — Ranking policy on Easy21 (zero-inference rerank):** R0–R3: 0 selection changes, 0 material regressions.

---

## §9 — GT10 Development Result

| Policy | Mean F1 | Median F1 | Min F1 | Catastrophic | Non-fail regressions |
|---|---|---|---|---|---|
| R0 (original ranking) | 0.6767 | 0.9575 | 0.0000 | 3 | 0 |
| R1 (winner) | 0.9728 | 0.9739 | 0.9536 | 0 | 0 |

Label: **DESCRIPTIVE DEVELOPMENT/PILOT EVIDENCE, N=10 frames from 2 sequences (not held-out)**. No significance claim.

---

## §10 — A6/A7 Negative Ablation

| Module | Easy21 ΔF1 | Easy21 regressions | GT10 ΔF1 | GT10 regressions | Exposure | Final status |
|---|---|---|---|---|---|---|
| A6 | −0.0028 | 1 | −0.0070 | 4/4 acted | min-frame limited | DEMOTED |
| A7 | +0.0001 | 0 | +0.0010 | 0 (1 marginal imp) | valid (Phase14.1 fixed) | DEMOTED |
| A6+A7 | −0.0028 | 1 | −0.0070 | 4/4 acted | — | DEMOTED |

"A6 and A7 did not provide consistent measurable benefit under the evaluated settings; both are disabled by default."

---

## §11 — Tables

6 paper tables auto-generated from source evidence (no hand-copied numbers):
- `Phase18_final_easy21_table__1a_historical_ablation.csv` (4 rows)
- `Phase18_final_easy21_table__1b_ranking_policy.csv` (4 rows)
- `Phase18_final_GT10_table.csv` (2 rows)
- `Phase18_failure_rescue_table.csv` (3 rows)
- `Phase18_score_mechanism_table.csv` (6 rows: 2 per failure frame)
- `Phase18_ranking_ablation_table.csv` (4 rows)
- `Phase18_A6A7_ablation_table.csv` (3 rows)

---

## §12 — Figures

4 paper-ready figures with bilingual EN/中 labels, deterministic PDF+PNG:
- **Figure A** — Failure mechanism: 3 frames × 5 panels (RGB/GT/R0 wrong/oracle/R1 selected)
- **Figure B** — Candidate feature comparison (0039): bar chart of oracle vs distractor scores
- **Figure C** — Easy vs Hard distribution: violin plot, descriptive only, no significance stars
- **Figure D** — A6/A7 ablation: bar chart (V00/V10/V01-R1/V11-R1), descriptive

---

## §13 — Provenance

72 unique paper numbers traced to source in `Phase18_paper_number_provenance.csv`.

---

## §14 — Code-Paper Consistency

8/8 tests pass (`test_phase18_paper_code_consistency.py`):
- T1: `--ranking_policy` default == `r1`
- T2: `--use_cross_view_consensus` default == `False`
- T3: `--use_memory_propagation` default == `False`
- T4: `--score_weights` default == `area=1,...,sam=0.5`
- T5: `ScoreRecord` fields present (ranking_policy, primary_score, tie_break_score, contrast_effective)
- T6: `提示词评分.csv` outputs 排名策略/主评分/对比度调制/平局决胜分
- T7: Paper GT10 table matches Phase17C evidence (mean 0.9728)
- T8: Failure rescue table: 3/3 frames rescued to F1 ≥ 0.95

---

## §15 — Regression Tests

| Suite | Count | Status |
|---|---|---|
| Historical (Phases 11/12/13/14.1/16/17A/17B) | 131 | ✅ PASS |
| Phase 17C (T1–T10, T21) | 11 | ✅ PASS |
| Phase 18 (consistency + table integrity) | 8 | ✅ PASS |
| **Total** | **150** | **✅ PASS** |

No historical expectation edits. No behavior changes to production code beyond what was committed in Phase17C.

---

## §16 — Evidence Integrity

- **Phase 17B report** preserved (3 × C3_RANKING_FAILURE references retained)
- **Phase 16 TEST manifest** immutable (SHA `1103f6f9`)
- **Phase 17C evidence** (rankings, benchmark, selected masks) unmodified
- **All Phase 18 artifacts are additive only** (new directory, no history overwritten)
- **Production algorithm code** unchanged from Phase 17C commit `d704faa8`

---

## §17 — Final Claims

**PROVEN:** Baseline candidate generation produces high-quality whole-plant masks on the failure frames (C01). The catastrophic failures (DouBanLv2 0039/0040/0041) originate from candidate-ranking function, not from candidate generation (C02).

**SUPPORTED:** The structural contrast signal (q_contrast) contributes to distractor selection in the identified failure subset (C03: intra-bin Spearman ρ = −0.7187). Removing q_contrast from the primary score eliminates the identified failure mode without material Easy21 regressions (C04).

**FORBIDDEN / NOT SUPPORTED:**
- C05: "R1 significantly improves general segmentation performance" — no independent held-out GT validation exists.
- C06/C07: A6/A7 consistent improvements — neither provided consistent measurable benefit.
- C08: A6/A7 as default — both are disabled by default.

---

## §18 — Limitations

- GT10 = 10 frames from 2 sequences; participated in the failure-mode diagnosis and fix development; not a held-out confirmation set.
- TEST32 = unlabeled behavioral audit only; no ground truth; cannot support accuracy claims.
- R1 has not undergone independent GT-based held-out confirmation.
- A6/A7 conclusions limited to the specific Easy21/GT10/TEST32 datasets and the evaluated settings; no broad generalization claim is made.
- No significance test performed on any result; all distributions are descriptive.

---

## §19 — Architecture Decision

Full decision record: `Phase18_architecture_decision.md`

| Component | Decision | Evidence |
|---|---|---|
| Candidate generation | RETAIN | Phase17B: oracle F1 ≈ 0.987 |
| P6 prompt | RETAIN | Phase17B: prompt succeeds when ranked correctly |
| R1 ranking | RETAIN / DEFAULT | Phase17C: 3/3 rescue, 0 regressions |
| q_contrast in primary | REMOVE FROM PRIMARY | Phase17C: intra-bin contrast bias |
| A6 | DEFAULT OFF | Phase14: E_A6=−0.0028; Phase17A: 4/4 regressions |
| A7 | DEFAULT OFF | Phase14.1: E_A7≈0; Phase17A: 1 marginal improvement |

---

## §20 — Final Verdict

### ✅ OPTION A — PAPER READY + B备注

All 8 paper-consistency gates are satisfied:
1. ✅ All paper numbers traceable to source (72 unique provenance records)
2. ✅ Phase reports internally consistent (4 stale claims corrected; evolution documented)
3. ✅ Production defaults consistent (8/8 code-paper consistency tests pass)
4. ✅ 142/142 historical tests pass (no expectation edits)
5. ✅ 150/150 total tests pass (131 hist + 11 P17C + 8 P18)
6. ✅ 4 paper-ready figures reproducible (deterministic, read-only scripts)
7. ✅ Claims restrained (C05 FORBIDDEN; no "significant/generalizes/outperforms")
8. ✅ Git clean, no push, no historical result overwritten

**B备注 (non-blocking future work):** Optional independent held-out R1 validation (new GT unseen during diagnosis) would strengthen generalization claims. This is a future note only; it does not block manuscript consolidation.

**Next step: MANUSCRIPT CONSOLIDATION (paper writing).**

---

*No algorithm was developed, modified, or tuned during Phase 18. All Phase 18 work is evidence audit, consistency correction, documentation, table/figure generation, and test verification only.*
