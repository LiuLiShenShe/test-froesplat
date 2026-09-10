# Phase 17C — Candidate Ranking Rescue Report

**Scope:** Ranking signal correction for RAP-FSAM3 agricultural plant segmentation.
**Method:** Zero-inference counterfactual reranking on frozen DEV10 + Easy21 candidate artifacts.
**Winner:** R1 (q_contrast excluded from primary score; simplest mechanism under §14 decision order).

---

## §1 — Objective

Fix the **C3 RANKING_FAILURE** failure mode diagnosed in Phase 17B [PROVEN]:
a structurally mis-specified contrast signal (`q_contrast`) causes the pipeline to select
a spurious tiny high-contrast distractor instance over the correct whole-plant candidate,
producing F1 = 0 in three DEV10 frames (DouBanLv2 0039/0040/0041) despite correct
candidates (F1 ≈ 0.987) existing in the same candidate set.

**Goal statement (verbatim):** "修复'小型高对比杂散实例压过完整植株候选'的排序缺陷，
同时不破坏现有正确帧" — fix the ranking defect while not damaging already-correct frames.

---

## §2 — Pre-Registered Policy Set

| Policy | Primary score | Tiebreak | Mechanistic question |
|--------|--------------|----------|---------------------|
| R0 (baseline) | `S/5.5 − penalty` = 总分 | 0 | Current production behavior |
| R1 | `(S − q_contrast)/5.5 − penalty` | 0 | Is contrast suitable as a ranking signal at all? |
| R2 | same as R1 | `q_contrast` | Is contrast useful only as tie-breaker at exact ties? |
| R3 | `(S − q_contrast·q_area)/5.5 − penalty` | 0 | Is contrast useful only when area plausibility supports it? |

Denominator frozen at 5.5 (frozen full-weights sum: area+comp+edge+temp+contrast=5.0, sam=0.5).

---

## §3 — Contrast Bias Audit (Step 4)

| Scope | Spearman(area_ratio, q_contrast) | Verdict |
|-------|----------------------------------|---------|
| Full 182 candidates | −0.2927 | NOT SUPPORTED (|ρ| < 0.3 threshold) |
| Failure frames only (0039/0040/0041) | −0.7187 | SUPPORTED (|ρ| > 0.3) |
| Oracle mean_q_contrast | 0.5518 vs 0.8442 (non-oracle) | Contrast discriminates wrong-way in failures |

**Finding:** `q_contrast` is NOT globally biased against large masks; the bias is
specific to failure frames where a tiny high-contrast distractor saturates the metric.
Intra-bin analysis (Step 5) confirms: ALL 31 oracle/selected candidates are in the
"large" bin (area_ratio > Q66 = 0.0276) — the failure is purely intra-bin.

---

## §4 — Component Ablation (Step 6)

| Dropped component | Failure rescue (0039/0040/0041) | DEV10 regression (ΔF1 < −0.005) | Easy21 regression |
|-------------------|--------------------------------|----------------------------------|-------------------|
| q_contrast | 3/3 | 0 | 0 |
| q_area | 0/3 | 1 DEV + 2 Easy21 | — |
| q_leak | 0/3 (weight=0, no-op) | 0 | 0 |
| q_side | 0/3 (weight=0, no-op) | 0 | 0 |

Contrast removal is the **minimal sufficient** intervention: 3/3 rescued, zero collateral.
Area removal is destructive (1 DEV + 2 Easy21 regressions) — proves area carries real signal.
Leak/side ablation confirm their zero weight: removing a weight-zero component is a literal
no-op (identical scores, 0/3 rescue = same as baseline).

---

## §5 — Policy Comparison (Step 8)

### DEV10 (10 frames, GT available)

| Policy | 0039/0040/0041 F1 ≥ 0.95 | ΔF1 < −0.005 (7 non-fail) | Mean F1 | Mean oracle regret |
|--------|--------------------------|---------------------------|---------|-------------------|
| R0 | 0/3 | 0 | 0.6767 | 0.2961 |
| **R1** | **3/3** | **0** | **0.9728** | **0.0000** |
| R2 | 3/3 | 0 | 0.9728 | 0.0000 |
| R3 | 3/3 | 0 | 0.9728 | 0.0000 |

### Easy21 (21 frames, GT available, safety set)

| Policy | Selection changed | ΔF1 < −0.005 | Material regressions |
|--------|-------------------|---------------|---------------------|
| R0 | — | — | 0 (baseline) |
| **R1** | 0 | 0 | **0** |
| R2 | 0 | 0 | 0 |
| R3 | 0 | 0 | 0 |

### Decision (§14 order: rescue > DEV reg > Easy21 reg > regret > churn > simplest)

R1/R2/R3 all pass gates 1–3 identically. Oracle regret = 0.0 for all three. R1 ≡ R2 in
all 32 frames (no exact primary-score ties exist; R2 tiebreak is never exercised). R3 is
mechanistically equivalent on these data.

**§14 tiebreaker #6 (simplest mechanism):** R1 removes one term from the primary score
with no conditional logic, no tiebreak, no area interaction — simplest possible intervention.
**Winner = R1.**

---

## §6 — Candidate Size Stratification (Step 5)

| Bin | Area range | N candidates | Oracle rate (GT-scope) | Selected rate (GT-scope) |
|-----|-----------|--------------|------------------------|-------------------------|
| Small | ≤ 0.0106 | 60 | 0/31 | 0/31 |
| Medium | 0.0106–0.0271 | 60 | 0/31 | 0/31 |
| Large | > 0.0271 | 62 | **31/31** | **31/31** |

Oracle candidates are exclusively in the large bin (31/31 GT-scope). Selected candidates
in the GT-scope are also exclusively in the large bin (31/31 GT-scope; 5 of 51 total
selected — including 20 non-GT context frames — fall in small/medium bins but are
irrelevant as they have no accuracy impact). The failure is NOT inter-bin (small selected
over large) but intra-bin: within the large bin, R0 prefers high-q_contrast; R1 removes
that signal.

---

## §7 — Production Replay Verification (Step 21)

Production `select_best_policy()` imported from `ranking_policy.py` (same module used in S20)
replayed against offline benchmark: **31/31 frames exact match** (selected ID, primary_score,
total_score all identical).

---

## §8 — Test Suite (Step 20)

| Test | Description | Status |
|------|-------------|--------|
| T1 | R0 baseline reproduces 提示词选择.csv for all 10 DEV10 frames | PASS |
| T2 | R1 primary formula = `(S − qc)/5.5 − penalty` | PASS |
| T3 | R2 cannot reverse unequal primaries | PASS |
| T4 | R2 breaks exact primary ties (constructed equal primaries) | PASS |
| T5 | R3 effective contrast = q_contrast × q_area | PASS |
| T6 | 0039/0040/0041 rescued F1 ≥ 0.95 under R1 | PASS |
| T7 | BaiZhang 0033–0037 no regression (ΔF1 ≥ −0.005) | PASS |
| T8 | Score diagnostics columns present | PASS |
| T9 | Empty candidate scored 0.0, never selected | PASS |
| T10 | Deterministic across identical calls | PASS |
| T21 | Production import matches offline recompute | PASS |

**Result: 11/11 PASS.**

---

## §9 — TEST32 Behavioral Safety (Step 26, conditional)

TEST32 (U00_Control, 8 samples × 4 = 32 frames, 71 raw candidates):
**NO accuracy claims** — behavioral audit only under winning policy R1.

| Metric | Result |
|--------|--------|
| Frames audited | 32 |
| Selection churn vs P00 baseline | 2/32 (6.2%) |
| Mean mask IoU (baseline-vs-R1 selected) | 0.9375 |
| Empty masks (baseline) | 0 |
| Empty masks (R1) | 0 |
| Churn risk (>30% → deployment flag) | **ok** |

Two churned frames (CaoMei2_0146, WanNianQing1_0227): selection changed to a different
candidate but no empty-mask or structural failures observed.

---

## §10 — Production Default Gate (Step 27)

Winner R1 confirmed → production `--ranking_policy` default changed from `base` to `r1`.
Changes applied: argparse default, ScoreRecord dataclass default, 3× getattr fallbacks.

Historical behavior reproducibility: T1 test uses `policy="base"` explicitly (never
depends on CLI default) — 131/131 historical tests pass post-change.

---

## §11 — Historical Regression (Step 29)

| Suite | Count | Status |
|-------|-------|--------|
| Historical (phase 11–17B) | 131 | ✅ PASS |
| Phase 17C (T1–T10, T21) | 11 | ✅ PASS |
| **Total** | **142** | **✅ PASS** |

---

## §12 — Evidence Integrity (Step 30)

- **Phase 17B report** (`Phase17B_report.md`): preserved — contains C3_RANKING_FAILURE
  diagnosis (3 references). Not modified by Phase 17C.
- **Phase 16 TEST manifest** (`Phase16_TEST_manifest.csv`): immutable — SHA
  `7504fcbb...` unchanged; git status shows zero modifications to phase 16 directory.
- **Old reports** (phase 11–15, 17A, 17B): all unchanged — `git status` shows 0
  modified files across all historical evidence directories.
- **Git repo:** HEAD remains at `26da29e6` (Phase 17B commit); all Phase 17C changes
  are uncommitted working-tree modifications pending the commit in Step 31.

---

## §13 — Implementation Gate Log

Per §17, production code was modified ONLY after benchmark winner was confirmed in Step 8.
Sequence: Step 8 (winner=R1) → Step 17 (production code extracted + modified) → Step 20
(tests) → Step 21 (replay) → Step 27 (default gate). No production code was written or
modified before Step 8's confirmation.

---

## §14 — Final Claim Language (§25)

> **The ranking correction eliminates the identified development-set failure mode.**

The C3 RANKING_FAILURE (F1=0 in DouBanLv2 0039/0040/0041 due to a high-contrast
distractor instance being ranked above the correct whole-plant candidate) is fully
resolved under policy R1: all three failure frames achieve F1 ≥ 0.95 with zero
material regressions on the remaining 7 DEV10 frames or 21 Easy21 safety frames.
The mechanism is structural (q_contrast removed from primary score) with no parameter
tuning, no SAM3 inference, and no modification of candidate generation.

**Evidence status:** SUPPORTED (development set + historical safety evidence; not
independent validation; §25 constraint observed).

---

## §15 — Test Data Sets

| Set | Source | Frames | Candidates | GT available | Role |
|-----|--------|--------|------------|-------------|------|
| DEV10 (GT10) | Phase17A P00_Control | 10 | 157 (10 GT + 20 context) | Yes | Development / ranking fix |
| Easy21 | Phase14 V00_Control | 21 | 25 | Yes | Safety / non-regression |
| TEST32 | Phase17A U00_Control | 32 | 71 (raw instances) | No | Behavioral safety only |

---

## §16 — Appendix: Files Modified/Created

| File | Status | Purpose |
|------|--------|---------|
| `S20/.../ranking_policy.py` | **NEW** | Single-source-of-truth ranking policy (base/r1/r2/r3) |
| `S20/.../生成RAP-FSAM3掩膜.py` | **MODIFIED** (6 edits) | Import ranking_policy, add --ranking_policy CLI, use primary_and_tiebreak + select_best_policy |
| `Phase17C_manifest.json` | **UPDATED** | Step 26–30 evidence record |
| `Phase17C_protocol.md` | **UPDATED** | Execution log rows 3–30 |
| `Phase17C_report.md` | **NEW** | This report |
| `Phase17C_test32_safety.csv` | **NEW** | TEST32 behavioral audit (32 rows) |
| `Phase17C_test32_summary.json` | **NEW** | TEST32 audit summary |
| `tests/test_phase17c.py` | **NEW** | T1–T10, T21 (11 tests) |
