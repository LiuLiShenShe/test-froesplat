# Phase 17B — Protocol (Abridged) + Gate Log

> 阶段十七B 上游失败归因与候选Oracle审计
> Protocol source: user's 21-section instruction (verbatim constraints preserved below).
> This file records (a) the operational protocol and (b) a per-section execution log.

---

## 0. Mandate & Core Framing

> **F1 = 0 is a symptom, not a root-cause diagnosis. First ask: "Did a correct candidate ever exist?" Only then decide what to fix.**

Phase 17B answers that question at **candidate level**, using Phase 17A P00_Control artifacts only (**zero new inference**). The deliverable is a per-frame failure class + candidate-level evidence, not a tuned pipeline.

### Failure classes (protocol §5, deterministic priority order)
| Class | Meaning | Condition |
|-------|---------|-----------|
| **C5_DATA_MAPPING_FAILURE** | GT/candidate mapping is wrong | mapping audit fails (highest priority — check first) |
| **C1_GENERATION_FAILURE** | No correct candidate ever generated | raw oracle F1 < 0.10 |
| **C2_POSTPROCESS_FAILURE** | Cleanup destroyed a good candidate | raw oracle ≥ 0.80 but postcleanup oracle drop > 0.05 |
| **C3_RANKING_FAILURE** | Correct candidate exists but was not selected | raw oracle ≥ 0.80 but selected F1 < 0.5 |
| **C4_WEAK_GENERATION** | Only a partial candidate exists | 0.10 ≤ raw oracle < 0.80 |

### Verbatim constraints (must hold throughout)
- **不得自动 push / 禁止自动 push** — commit allowed, push prohibited.
- **A6/A7 frozen disabled** (§15) — `use_cross_view_consensus=False`, `use_memory_propagation=False` in P00; no new A6/A7 experiments.
- **No parameter tuning** (§16) — no modified score weights / SAM threshold / cleanup / prompt / candidate mode passed off as "improvement".
- **Evidence hierarchy** (§17) — claims labeled **PROVEN / SUPPORTED / HYPOTHESIS**; hypothesis never written as proven.
- **Pilot role** — the 10 GT frames are `DIAGNOSTIC_DEV` (downstream of Phase 17A pilot), never relabeled TEST; no add/remove.
- **Zero-inference preference** — Phase 17A artifacts suffice; no re-run of diagnostic inference. Only missing raw candidates would authorize re-inference (none were).
- **Phase16 TEST manifest SHA256 immutable** (`7504fcbb…`); historical 121/121 regression must stay green.

---

## 1. Gates & Prompt Sweep (§7) — Recorded Decision

| Gate | Value | Verdict |
|------|-------|---------|
| P6 raw oracle, DouBanLv2 0039 | 0.9867 | ≥ 0.10 → **gate not triggered** |
| P6 raw oracle, DouBanLv2 0040 | 0.9882 | ≥ 0.10 → **gate not triggered** |
| P6 raw oracle, DouBanLv2 0041 | 0.9865 | ≥ 0.10 → **gate not triggered** |
| Sweep threshold | 0.10 | — |
| **Decision** | — | **Skip P1–P5 sweep (user-confirmed).** A generation-selection sweep cannot explain F1=0 when generation already reaches 0.987. |

Recorded in `Phase17B_manifest.json` → `prompt_sweep_gate.gate_triggered = false`, `decision = "skip (user-confirmed)"`.

---

## 2. Per-Section Execution Log

| § | Protocol step | Output | Status |
|---|---------------|--------|--------|
| 0 | Starting audit (HEAD, clean tree, A6/A7 off) | `Phase17B_manifest.json` | ✅ |
| 1 | Freeze DEV10 = GT10, role=DIAGNOSTIC_DEV | `Phase17B_DEV10_manifest.csv` | ✅ |
| 2 | Mapping audit (C5 exclusion) | `Phase17B_mapping_audit.csv` + overlays | ✅ |
| 3 | Candidate inventory (all raw P6 candidates) | `Phase17B_candidate_inventory.csv` (157 rows) | ✅ |
| 4 | Candidate oracle (F1/IoU/P/R per candidate + per-frame oracle/selected) | `Phase17B_candidate_oracle.csv` | ✅ |
| 5 | Failure attribution (C5>C1>C2>C3>C4) | `Phase17B_failure_attribution.csv` | ✅ |
| 6 | Oracle table (10 rows) | report §6 | ✅ |
| 7 | Prompt gate | `Phase17B_manifest.json` (gate log above) | ✅ |
| 8 | Prompt oracle matrix | `Phase17B_prompt_oracle.csv` | ✅ |
| 9 | Prompt-failure decision | report §9 ("not supported") | ✅ |
| 10 | Detection-vs-mask | `Phase17B_detection_diagnosis.csv` | ✅ |
| 11 | Instance audit (failing frames) | `visualizations/instance_audit_*.jpg` | ✅ |
| 12 | Score decomposition (selected vs oracle) | `Phase17B_score_regret.csv` | ✅ |
| 13 | Cleanup trace (raw→selected→final) | `Phase17B_cleanup_trace.csv` | ✅ |
| 14 | Temporal diagnosis (0037–0041) | `Phase17B_temporal_diagnosis.csv` + strip | ✅ |
| 15 | A6/A7 frozen assertion | `脚本/phase17b_steps15_19.py` | ✅ |
| 16 | No-tuning assertion | `脚本/phase17b_steps15_19.py` | ✅ |
| 17 | Evidence hierarchy (PROVEN/SUPPORTED/HYPOTHESIS) | report §15 | ✅ |
| 18 | Decision tree → Phase 17C target | report §14 | ✅ |
| 19 | Outputs completeness | all §19 files present | ✅ |
| 20 | Tests T1–T10 + regression | `tests/test_phase17b.py`, 131/131 | ✅ |
| 21 | Final report + commit (no push) | `Phase17B_report.md` | ✅ |

---

## 3. Method Notes (protocol-critical)

- **Selection confirmation (§4/T2).** For each of the 10 GT frames, the instance chosen by the pipeline is confirmed three ways and must agree: (1) `argmax(total_score)` over P6 raw instances, (2) the `前景面积比例` in `提示词选择.csv`, (3) the `selected_instance_id` in `Phase17B_candidate_oracle.csv`. All 10 agree.
- **Cleanup identity (§13/T3).** `选择后掩膜 == 最终掩膜 == A5c_final_mask` bitwise for all 10 GT frames → postcleanup oracle F1 == raw oracle F1 → **C2 is excluded by construction**.
- **Score decomposition (§12).** The pipeline's own recorded `总分` (提示词评分.csv) is the selection criterion — no re-derivation. Component q-scores (面积/连通域/边界/时序/前背景对比/sam + leak/side) are compared for the oracle vs selected instance; driver = component weighted-gap most favoring the (wrong) selected instance.
- **Zero-inference (§T8).** All analysis reads pre-existing P00 artifacts whose mtimes predate the 17B scripts. No pipeline process is invoked by any 17B script (T6/T7).

---

## 4. Interpretation Guardrail

Per §17: labels are fixed before narrative. **PROVEN** = directly reproduced from saved artifacts with no modeling; **SUPPORTED** = consistent with multiple artifacts but narrower basis; **HYPOTHESIS** = extrapolates beyond directly measured frames. The report (§14–15) applies these strictly and states the expected decision — **Case B (ranking/selection)** — as the Phase 17C target, not as a fix executed here (no tuning allowed).