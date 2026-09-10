# Phase 18 — Historical Corrections (Phase 11–17C Scan)

**阶段十八_论文实验收束与最终审计 · 历史修正清单**

| Field | Value |
|---|---|
| Phase | 18 (evidence audit, historical consistency correction) |
| Date | 2026-09-10 |
| HEAD at writing | `d704faa8` (Phase 17C commit, amended from `260d81f0`), working tree clean |
| Scope | Scan of Phase 11–17C reports / protocols / manifests for stale claims and stale metadata |
| Rule | Old reports are historical records — **never overwritten**. This document carries the corrections. |

---

## §4 — Historical Corrections: Four Active Stale/Corrected Claims

The Phase 11–17C scan surface contained many statements written at the time they were current.
Four statements remain **active** (referenced across later docs / eligible for paper citation) but are
stale or incorrect. Each is corrected below in a fixed format.

---

#### 1. "A7 depends on A6 consensus masks as seed base"

Original claim: **A7 neutral without A6**: A7 depends on A6 consensus masks as seed base. Without A6, A7 has no valid within-sample seed.

Source: `阶段十四_A6A7真实数据析因消融/Phase14_report.md` **L117** (§9 Known Limitations, item 1); same explanation at **L72–73** (§5.2 V01 A7).

Status: **SUPERSEDED**

Correct interpretation: A7 does **NOT** require A6. Code audit in `Phase14_correction_note.md` §D shows the A7 seed is selected per-sample by the top A1s-scored frame *within* the sample (`base_for_memory = selected_by_stem`, L2874–2876; per-frame seed selection L2908–2909); there is **no structural/algorithmic dependency** on A6 consensus masks and no cross-sample global-best fallback. The original V01 (A6 OFF) produced 0 propagations because of a **runtime resource failure** — GPU session leak (`cuda_oom_fallback`, 5 sessions started / 0 removed) — not because of a missing A6 seed.

Superseding evidence: `阶段十四点一_A7有效性封口/Phase14_correction_note.md` §D（"A7 depends on A6 is stale explanation"）and §E2（OOM root cause = session leak, empirically confirmed）; Phase 14.1 rerun **V01_R1** (A6 OFF, A7 ON) propagated **13/21** and selected **9/21** frames without A6（`Phase14_1_report.md` §D）.

Paper implication: State "A7 does not depend on A6 for operation." Do not present A6-seed-dependency as a design constraint or as the reason A7 under-performed.

---

#### 2. "A7 neutral" (based on 0-propagation)

Original claim: A7 main effect: +0.0000 (neutral) — supported by "A7 without A6 is completely neutral (identical to V00)" with V01 propagated 0/21 frames.

Source: `阶段十四_A6A7真实数据析因消融/Phase14_report.md` **L20** (§4 Factorial Effects) and **L72** (§5.2 V01 A7); invalid-evidence re-examination at `阶段十四点一_A7有效性封口/Phase14_1_report.md` **L115**.

Status: **INCORRECT** (invalid evidence, not merely a wrong number)

Correct interpretation: The original "A7 neutral" conclusion was built on an **invalid A7 observation**: A7 did not execute in the original V01 run (`状态=cuda_oom_fallback`, `记忆候选帧数=0`; session leak 5 started / 0 removed). "V01 == V00" therefore cannot be read as "A7 effect = 0." After the Phase 14.1 lifecycle fix, A7 runs correctly but remains empirically near-neutral on Easy21: **E_A7 ≈ 0 with valid exposure** (V01-R1: propagated **13/21**, selected **9/21**, 5/5 samples `ok`), plus exactly one marginal improvement on GT10 (ΔF1 = +0.006, BaiZhang). Governing principle: *"A module that falls back to baseline has not demonstrated a neutral effect. Failure to execute ≠ zero treatment effect."*

Superseding evidence: `阶段十四点一_A7有效性封口/Phase14_1_report.md` §A (principle, L19), §D (valid-exposure execution table, L55–78), §E–F (corrected factorial, E_A7 = +0.0000 on valid exposure); `Phase14_correction_note.md` §E2 (OOM root-cause confirmation).

Paper implication: **Do not cite the original "A7 neutral" claim as evidence.** Cite only the Phase 14.1 corrected result: "A7 is empirically near-neutral (E_A7 ≈ 0) under valid execution exposure (13 propagated / 9 selected of 21 Easy21 frames)."

---

#### 3. "1 mild regression" (ChangShouHua2_0075 under A6)

Original claim: "1 (mild)" regression under A6 — "1 regression: ChangShouHua2_0075 (0.9856→0.9278, delta=-0.058)".

Source: `阶段十四_A6A7真实数据析因消融/Phase14_report.md` **L15** (§1 summary table, V10 A6 row: `1 (mild)`) and **L64** (§5.1 V10 A6, per-frame delta); formal reclassification at `阶段十四点一_A7有效性封口/Phase14_1_report.md` **L99**.

Status: **CORRECTED**

Correct interpretation: ΔF1 = **−0.058** (0.9856 → 0.9278) exceeds the protocol's material-regression threshold (−0.005) by **≈11.6×**. Phase 14.1 explicitly reclassified this frame from "mild" to **material regression**, and it is the only measurable A6 effect on Easy21 (A6 main effect E_A6 = −0.0028, driven by this single frame).

Superseding evidence: `阶段十四点一_A7有效性封口/Phase14_1_report.md` **L99**（"按协议回归阈值（< -0.005），ΔF1 = -0.058 是 **material regression**，不是 mild"）; `Phase14_correction_note.md` §E2（corrected table last row: "非 mild：ΔF1 = -0.058，material regression"）.

Paper implication: Write "A6 produces one material regression on Easy21 (ΔF1 = −0.058)." The word "mild" must not be used for this frame.

---

#### 4. "P6 prompt-candidate generation failure" (Phase17A hypothesis)

Original claim: Hypothesis that "F1=0 may indicate upstream / prompt-candidate generation failure" —— "The hard-case problem is upstream (prompt/candidate generation), not in the A6/A7 mechanisms."

Source: `阶段十七A_10GT困难样本Pilot与无GT测试审计/Phase17A_report.md` **§10–§11**（L236–237, L250 区域, L256 — hypothesis of upstream/prompt-candidate cause）.

Status: **REFUTED**

Correct interpretation: Phase 17B's zero-inference candidate-oracle audit proved candidate **generation succeeded** — raw oracle F1 = **0.9867 / 0.9882 / 0.9865** for DouBanLv2 0039/0040/0041. Near-ceiling whole-plant candidates were present in the P00 candidate pool on every failing frame. The catastrophic F1=0 results were caused by the **candidate-ranking function** selecting a spurious instance over the correct candidate — **C3_RANKING_FAILURE** (cleanup was bitwise identity; mapping correct; prompt sweep gate not needed).

Superseding evidence: `阶段十七B_上游失败归因与候选Oracle审计/Phase17B_report.md` **L23**（"…is **REFUTED at candidate level**"）；`Phase17B_manifest.json` `prompt_sweep_gate`（0.9867/0.9882/0.9865; gate 0.1 not triggered, skip user-confirmed）; `Phase17B_candidate_oracle.csv`, `Phase17B_detection_diagnosis.csv`（`wrong_instance` for 0039–0041）.

Paper implication: Write "High-quality candidate masks were present, but the original candidate-ranking function selected distractor instances." **Do NOT write** "P6 failed to detect DouBanLv2 0039–0041" or "candidate generation failed."

---

## §5 — Phase17A → Phase17B Conclusion Evolution | 结论演化记录

The interpretation of the DouBanLv2 0039/0040/0041 catastrophic failures evolved across three phases, and the paper must reflect the final state, not the interim hypothesis.

**Phase17A (2026-08, hypothesis).** On the GT10 pilot, P00 (baseline) was run in 4 variants (P00/P01/P10/P11); the report hypothesized that "F1=0 may indicate upstream / prompt-candidate generation failure" — i.e., the P6 prompt or the candidate generator failed to produce any usable whole-plant mask. At the time this was a *reasonable* hypothesis: the rationale recorded in `Phase17A_report.md` L236–237 was "the hard-case problem is upstream (prompt/candidate generation)", zero-prediction frames had not yet been candidate-audited, and no inference-free attribution tool existed. (Note: an early draft figure "23/32 non-failing frames matched GT10 context" was **not found in any Phase 17A source** and is therefore **not used** in this document.)

**Phase17B (2026-09, proved).** A candidate-oracle audit (zero new SAM3/P6 inference) was run over all 157 P00 candidates of the DEV10 frames. For every F1=0 frame the raw oracle F1 was ≈ 0.987 (0.9867 / 0.9882 / 0.9865) — a high-quality whole-plant candidate **was generated** and was simply **not selected**. This **REFUTED** the generation-failure hypothesis and diagnosed the root cause as **C3_RANKING_FAILURE**: a small, high-contrast distractor instance outranked the correct plant.

**Phase17C (2026-09, fixed).** The ranking fix removed `q_contrast` from the primary candidate score (policy **R1**, commit `d704faa8`). R1 **rescued all 3 failures to F1 ≥ 0.95 with zero regressions** on Easy21 and zero new DEV-set regressions — confirming the Phase17B diagnosis by counterfactual replay.

**Paper must NOT write:** "P6 prompt failed" · "candidate generation failed" · "SAM3/P6 cannot detect the hard plants."
**Paper MUST write:** "High-quality candidate masks were present, but the original candidate-ranking function selected distractor instances."

---

## §6 — Phase17C Stale Metadata Correction | 阶段十七C 过时元数据修正

The following four references inside the Phase 17C artifacts are **stale metadata** (they describe the working tree *during* Phase 17C, before the Step-31 commit). All four resolve to the actual final state below.

| # | Stale reference (written in Phase 17C) | Location | Actual (Phase 18 verified) |
|---|---|---|---|
| 1 | "HEAD remains at `26da29e6` (Phase 17B commit); all Phase 17C changes are uncommitted … pending the commit in Step 31" | `Phase17C_report.md` **§12, L190–191** | HEAD = **`d704faa8`**（Phase 17C commit）, **committed**, working tree **clean** |
| 2 | "HEAD SHA — `26da29e6` (Phase17B)" | `Phase17C_protocol.md` **§0, L13** | HEAD = **`d704faa8`**（Phase17C） |
| 3 | "…✅ commit `260d81f0`, no push" | `Phase17C_protocol.md` **L130**（Step-31 execution log） | commit = **`d704faa8`**（amended from `260d81f0`） |
| 4 | `Phase17C_manifest.json` lacks a **`committed_sha`** field (records only `starting_sha` / `phase17b_sha` = `26da29e6`) | `Phase17C_manifest.json` | committed SHA = **`d704faa8`** |

**Note:** `Phase17C_report.md` is a **historical record** — it is NOT to be rewritten (per Phase 18 protocol §4/§29). This correction note is the authoritative record of the actual final state.

---

## §4-note — Numerical Test Counts Confirmed | 历史测试计数核验

All historical test counts were re-checked and are **temporally accurate within their own phase reports**; no stale count is left uncorrected:

| Phase | Reported count | Verified source | Status |
|---|---|---|---|
| Phase 13 | 71/71 | `阶段十三_A6A7增强分支强制验证/Phase13_report.md` L127 | ✅ accurate for Phase 13 |
| Phase 14.1 | 84/84 (71 historical + 13 new) | `阶段十四点一_A7有效性封口/Phase14_1_report.md` L129; `Phase14_1_manifest.json` `regression` | ✅ accurate for Phase 14.1 |
| Phase 16 | 100/100 | `阶段十六_HardCase_GT构建与锁定/Phase16_report.md` L313, L367 | ✅ accurate for Phase 16 |
| Phase 17A | 121/121 (phases 11, 12, 13, 14.1, 16 + Phase 17A T1–T12) | `阶段十七A_10GT困难样本Pilot与无GT测试审计/Phase17A_report.md` L274, L284 | ✅ accurate for Phase 17A |
| Phase 17B | 131/131 (121 historical + T1–T10) | `阶段十七B_上游失败归因与候选Oracle审计/Phase17B_report.md` L246, L257, L322 | ✅ accurate for Phase 17B |
| Phase 17C | 142/142 (131 historical + 11 Phase 17C) | `阶段十七C_候选排序修复开发/Phase17C_manifest.json` `steps_completed.29` (`131/131 + 11/11 = 142/142`); `Phase17C_report.md` §12/§17 | ✅ accurate for Phase 17C |

Each count is a monotone superset of the previous one (71 → 84 → 100 → 121 → 131 → 142), consistent with a strict no-edit history policy. No historical expectation was revised.

---