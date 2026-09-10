# Phase 17C — Protocol (Abridged) + Gate Log

> 阶段十七C 候选排序修复开发 Candidate Ranking Rescue Development
> Protocol source: user's 16-section instruction. Verbatim constraints preserved below.
> Goal: "移除结构性误设的 ranking signal，不破坏既有正确帧" — NOT tuning a better score for 3 frames.

---

## 0. Starting State (Step 0 audit, PROVEN)

| Item | Value | Status |
|------|-------|--------|
| HEAD SHA | `26da29e6` (Phase17B) | ✅ |
| Phase17B SHA | `26da29e6` | ✅ |
| Phase17A SHA | `d656851e` | ✅ |
| Remote | `https://github.com/LiuLiShenShe/test-froesplat` | ✅ |
| Working tree | clean (only new Phase17C dir untracked) | ✅ |
| **不得自动 push** | commit allowed, push prohibited | locked |

### Artifact count + reconstruction exactness (PROVEN)
| Source | 提示词评分 rows | 候选评分明细 | raw_instance masks | reconstruction |
|--------|---------------|-------------|--------------------|----------------|
| P00_Control (DEV10 source) | 157 | 157 (30 files) | 157 | **157/157 exact** |
| Phase14 V00_Control (Easy21 source) | 25 | 25 (21 files) | 25 | **25/25 exact** |
| Phase12 P6_raw_baseline (cross-check) | — | 25 | 25 | V00==P12 ✓ |

- Score reconstruction formula **proven exact** from CSV columns:
  `base_total = (Σ w_i·q_i + 0.5·q_sam)/5.5`, `总分 = base_total − penalty`, `penalty ∈ {0, +0.25, +0.40}`.
  - 1 penalty found (BaiZhang_0042, vertical-coverage +0.25), all others 0 → **no hidden risk-penalty noise**.
  - `q_sam` must come from `候选评分明细.SAM3分数` (per-instance, positional join); the `提示词评分.csv SAM3原始分数` column is frame-level and identical across a frame's rows (NOT per-candidate).

---

## 1. Data-Role Freeze

| Set | Role | Prior use | Allowed use |
|-----|------|-----------|-------------|
| GT10 / DEV10 (P00: BaiZhang 0033–0037, DouBanLv2 0037–0041) | **DEVELOPMENT / DIAGNOSTIC SET** | Phase17B root-cause | design ranking fix, choose ranking rule, check rescue |
| Easy21 (V00: 5 samples, 21 frames) | **SAFETY / NON-REGRESSION SET** | Phase12/14/16/17A anchor (mean F1 0.9835) | non-regression development evidence |
| TEST32 (U00_Control: 8 samples × 4) | **BEHAVIORAL SAFETY ONLY** | Phase16/17A locked TEST (no GT) | selection churn / mask divergence only — NO accuracy claim |

Final claim language: "**development evidence + historical safety evidence**" — never "independent validation / confirmatory test / generalization proof".

Total candidates in benchmark: **182** (157 DEV10 + 25 Easy21).

---

## 2. 禁止重新推理 (Zero-Inference)

Phase 17C Part 1 forbids: SAM3 inference, prompt sweep, A6, A7, new candidate generation.
- Uses existing raw candidate masks + candidate metadata + SAM scores + score components + GT.
- All counterfactuals are pure CSV-column arithmetic on the exact reconstruction formula above.
- **No SAM3 inference in the entire phase** (user-confirmed: skip §22 real-inference smoke).
- Easy21 candidate regeneration NOT needed — Phase14 V00 has complete artifacts (25 raw masks + CSVs).

---

## 3. Pre-Registered Ranking Policies (structural only — no tuning)

| Policy | Primary score key | Tie-break | Mechanistic question |
|--------|-------------------|-----------|----------------------|
| R0 | `基础总分 = (Σ w_i q_i + 0.5 q_sam)/5.5 − penalty` (current) | 0 | Baseline |
| R1 | `(基线 Σ 去除 q_contrast + 0.5 q_sam)/5.5 − penalty` | 0 | contrast 是否应根本退出 primary 排序？ |
| R2 | same primary as R1 | **q_contrast** (tuple key) | contrast 是否只作精确平局时的 tie-break？ |
| R3 | replace `q_contrast` with `q_contrast_effective = q_contrast × q_area` | 0 | contrast 是否需由候选 plausibility (area) 调制？ |

- No `--alpha/beta/gamma`; no new threshold; denom fixed at 5.5 (frozen convention, matches leak/side weight-0 handling).
- A6 = OFF, A7 = OFF throughout (§23).
- No prompt sweep (generation already 0.987 — Phase17B gate unmet).

---

## 4. Decision Order (§14, strict)

1. Rescue all three C3 failures (0039/0040/0041, selected F1 ≥ 0.95)
2. Zero material regression on remaining DEV10 (ΔF1 < −0.005)
3. Zero material regression on Easy21 (ΔF1 < −0.005)
4. Lowest oracle regret
5. Lowest selection churn
6. Simplest mechanism

**NOT** highest mean F1. Winner ∈ {R1, R2, R3, NONE}. If all fail: stop, no weight tuning, conclude "q_contrast 不是唯一 ranking deficiency" → Phase 17D redesign.

---

## 5. Implementation Gate (§17)

- 只有 benchmark 得到明确 winner 后才允许修改生产代码 `生成RAP-FSAM3掩膜.py`。
- Modification minimal: extract pure-function ranking policy (e.g. `ranking_policy.py`) → `compute_candidate_primary_score(...)` + `select_best_policy(...)`; `score_candidate`/`select_mask` call it; policy selectable via `--ranking_policy {base,r1,r2,r3}`.
- Default = `base` (bitwise identical to current production).

---

## 6. Evidence & Diagnostics Preserved

- q_contrast & raw contrast stay recorded (diagnostic features — cannot drop columns, §18).
- Score CSV gains: `ranking_policy`, `primary_score`, `contrast_effective`, `final_rank` (+ `tie_break_score` for R2) (§19).
- Evidence integrity (§30): Phase17B `C3_RANKING_FAILURE` diagnosis preserved as historical record; old reports NOT overwritten; Phase16 TEST manifest immutable; historical 131/131 tests unchanged (no expectation edits to go green).
- Claim language (§25): "The ranking correction eliminates the identified development-set failure mode." — not "improves general segmentation performance".

---

## 7. Gates & Tests (§20, §27)

Production default adopted only if ALL: 3/3 rescued ∧ 0 material DEV regressions ∧ 0 material Easy21 regressions ∧ low churn ∧ replay exact ∧ tests pass. Otherwise keep baseline default.

New tests: T1–T10 (see `tests/test_phase17c.py`). Historical 131/131 must stay green.

---

## 8. Per-Section Execution Log

| § | Step | Deliverable | Status |
|---|------|-------------|--------|
| 0 | Starting audit + manifest | `Phase17C_manifest.json` | ✅ |
| 1 | Data-role freeze | this protocol §1 | ✅ |
| 3 | Ranking benchmark | `Phase17C_ranking_benchmark.csv` (182 rows) | ✅ |
| 4 | Contrast bias audit | `Phase17C_contrast_bias.csv` (Spearman) | ✅ |
| 5 | Size stratification | bins table (Q33/Q66) | ✅ |
| 6 | Component ablation | `Phase17C_component_ablation.csv` | ✅ |
| 8 | Policy comparison R0–R3 | `Phase17C_policy_comparison.csv` + winner | ✅ |
| 17 | Production modification | `ranking_policy.py` + minimal S20 diff | ✅ (after §14 winner=R1) |
| 20 | Tests T1–T10 | `tests/test_phase17c.py` (11 pass) | ✅ |
| 21 | Production replay | `Phase17C_production_replay.csv` exact-match (31/31) | ✅ |
| 22 | Real inference smoke | **SKIPPED** (user-confirmed zero-inference) | ✅ skipped |
| 26 | TEST32 behavioral audit | `Phase17C_test32_safety.csv` — 32 frames, churn 6.2%, mask-IoU 0.9375, 0 empty | ✅ PASS |
| 27 | Production default gate | S20 `--ranking_policy default=r1` (+ ScoreRecord, 3×getattr fallbacks) | ✅ |
| 29 | Historical regression | 131/131 historical + 11/11 Phase17C = 142/142 | ✅ |
| 30 | Evidence integrity | Phase17B C3_RANKING_FAILURE preserved; Phase16 TEST manifest immutable (git-clean); old reports unchanged | ✅ |
| 31 | Final report + commit (NO push) | `Phase17C_report.md` | ✅ commit `260d81f0`, no push |