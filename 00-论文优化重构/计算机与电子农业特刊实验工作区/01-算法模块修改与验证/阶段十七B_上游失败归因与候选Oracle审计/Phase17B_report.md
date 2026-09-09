# Phase 17B — Upstream Failure Attribution + Candidate Oracle Audit

**Date:** 2026-09-09
**Branch:** main
**Starting SHA:** `d656851e` (post-Phase-17A)
**Python:** `/home/test/biosoft/enter/envs/sam3/bin/python`
**Mode:** zero-inference (no new pipeline runs; all evidence from Phase 17A P00_Control artifacts + Phase 16 GT)

---

## 1. Motivation & Scope

Phase 17A concluded "hard-case failure is upstream / P6 prompt-candidate generation" — but that was a **hypothesis without candidate-level evidence**. Phase 17B tests it directly: for each of the 10 GT frames, **did a correct candidate ever exist, and if so which stage destroyed or rejected it?**

| Question | Answer (one line) |
|----------|-------------------|
| Mapping correct? (C5) | ✅ Yes — all 10 frames map exactly (T9). |
| Did a correct candidate exist? (C1/C4) | ✅ Yes — raw oracle F1 = **0.987/0.988/0.987** on the 3 "F1=0" frames. |
| Did cleanup destroy it? (C2) | ❌ No — cleanup is bitwise identity (`选择后==最终==A5c`), T3. |
| Was the correct candidate selected? (C3) | ❌ **No — a spurious instance outranked it** in all 3 failing frames. |
| P6 prompt sweep needed? (§7) | ❌ No — gate unmet (0.987 ≥ 0.10), user-confirmed skip. |

**Verdict:** F1=0 on DouBanLv2 0039/0040/0041 is a **C3 RANKING_FAILURE** — the correct P6 candidate was generated at near-ceiling quality but **not selected**. The Phase 17A hypothesis "P6 prompt-candidate generation failure" is **REFUTED at candidate level**.

---

## 2. Manifest Immutability

| Manifest | SHA256 / ref | Status |
|----------|--------------|--------|
| Phase16 TEST manifest | `7504fcbb…` (locked ref) | ✅ T6 re-derived == manifest.json |
| Phase17A GT10 manifest | `f596b417…` (frozen) | ✅ re-derived == 17A |
| Phase17B DEV10 manifest | `89d150d7…` | ✅ frozen (exhaustive copy, role added only) |
| Phase17B starting SHA | `d656851e` | ✅ clean tree at start |

- `Phase17B_DEV10_manifest.csv` = exact 10 rows of Phase17A GT10, `role=DIAGNOSTIC_DEV`, no add/remove (§1).
- `no_pseudo_gt=true`, `zero_inference=true` in `Phase17B_manifest.json`.

---

## 3. Diagnostic Role Statement

The 10 GT frames are **DIAGNOSTIC_DEV** — a downstream diagnostic view of already-annotated pilot material. They are used **only after** predictions were frozen (T8: mask mtimes predate all 17B analysis). They never re-enter any pipeline.

---

## 4. Configuration Freeze

P00_Control was run under the identical frozen config as Phase 17A §4, **with A6 and A7 disabled**:

```
use_cross_view_consensus = False
use_memory_propagation   = False
score_weights = area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5   (leak/side weight 0)
use_semantic_gate = False
candidate_mode = per_instance        default_prompt = P6
sam3_mask_threshold = 0.5
```

**Phase 17B executes zero pipeline invocations** — it is analysis-only (T7). No weight, threshold, prompt, cleanup, or candidate-mode is modified (T10).

---

## 5. Execution Summary

| Step | Script | Input (read-only) | Output |
|------|--------|-------------------|--------|
| 0–1 | `phase17b_step0_1.py` | git HEAD, P00 参数.json, 17A GT9/manifest | `Phase17B_manifest.json`, `Phase17B_DEV10_manifest.csv` |
| 2–8+10 | `phase17b_core_analysis.py` | 提示词评分.csv, 候选评分明细, raw_instance_P6, 提示词选择.csv, GT | inventory, oracle, mapping, detection, prompt_oracle + overlays |
| 5+12–14 | `phase17b_attribution_analysis.py` | same + 选择后/最终/A5c masks | failure_attribution, score_regret, cleanup_trace, temporal_diagnosis |
| 11 | `phase17b_step11_instance_audit.py` | RGB + raw instances + GT | `instance_audit_*.jpg` (0039/0040/0041) |
| 14 | `phase17b_step14_temporal_strip.py` | RGB + GT + final masks | `temporal_strip_0037-0041.jpg` |
| 15–19 | `phase17b_steps15_19.py` | 参数.json + 17B scripts + outputs | assertions pass |

**All steps completed successfully. Zero-inference confirmed at every stage.**

---

## 6. Candidate Oracle Table (§4/§6)

Per frame: `selected_f1` (what the pipeline delivered), `raw_oracle_f1` (best raw P6 instance vs GT), `postcleanup_oracle_f1` (= raw, identity), `finalcandidate_oracle_f1` (best candidate after all stages = raw), `regret`.

| Sample | Frame | **selected F1** | raw oracle F1 | cleanup oracle | final oracle | **regret** | #inst | Class |
|--------|-------|----------------|---------------|----------------|--------------|-----------|-------|-------|
| BaiZhang | 0033 | 0.9593 | 0.9593 | 0.9593 | 0.9593 | 0.0000 | 3 | OK |
| BaiZhang | 0034 | 0.9557 | 0.9557 | 0.9557 | 0.9557 | 0.0000 | 4 | OK |
| BaiZhang | 0035 | 0.9605 | 0.9605 | 0.9605 | 0.9605 | 0.0000 | 4 | OK |
| BaiZhang | 0036 | 0.9623 | 0.9623 | 0.9623 | 0.9623 | 0.0000 | 1 | OK |
| BaiZhang | 0037 | 0.9536 | 0.9536 | 0.9536 | 0.9536 | 0.0000 | 2 | OK |
| DouBanLv2 | 0037 | 0.9855 | 0.9855 | 0.9855 | 0.9855 | 0.0000 | 1 | OK |
| DouBanLv2 | 0038 | 0.9898 | 0.9898 | 0.9898 | 0.9898 | 0.0000 | 1 | OK |
| **DouBanLv2** | **0039** | **0.0000** | **0.9867** | 0.9867 | 0.9867 | **0.9867** | 2 | **C3** |
| **DouBanLv2** | **0040** | **0.0000** | **0.9882** | 0.9882 | 0.9882 | **0.9882** | 3 | **C3** |
| **DouBanLv2** | **0041** | **0.0000** | **0.9865** | 0.9865 | 0.9865 | **0.9865** | 9 | **C3** |

- **7/10 frames selected correctly** (regret 0.0000): all BaiZhang (0.954–0.962) and DouBanLv2 0037/0038 (0.986/0.990).
- **3/10 frames (DouBanLv2 0039/0040/0041) lost everything**: a correct candidate with F1 ≈ 0.987 *existed* but the pipeline selected an instance that scores **F1 = 0.0**.
- Regret on failing frames ≈ the full oracle F1 — the entire performance is surrendered by selection. **This is the entire story of the F1=0 symptom.**

---

## 7. Prompt Sweep Gate (§7)

| Frame | P6 raw oracle | Threshold (0.10) | Gate triggered? |
|-------|---------------|------------------|-----------------|
| 0039 | 0.9867 | — | **No** |
| 0040 | 0.9882 | — | **No** |
| 0041 | 0.9865 | — | **No** |

**Decision: skip P1–P5 sweep (user-confirmed).** A prompt-selection sweep cannot contribute when P6 already generates at 0.987. Recorded in `Phase17B_manifest.json` `prompt_sweep_gate`.

---

## 8. Prompt Oracle Matrix (§8)

`Phase17B_prompt_oracle.csv`: each row is a GT10 frame; `P6` = raw oracle F1; `P1..P5` = `not_run(gate)`; `best_prompt = P6` for all 10 rows.

---

## 9. Prompt-Failure Decision (§9)

**P6-specific semantic prompt failure: NOT SUPPORTED.**
The raw P6 candidate achieves F1 = 0.9867/0.9882/0.9865 vs GT on exactly the frames that finally score 0.0000. Generation is not the bottleneck. **(PROVEN)**

---

## 10. Detection-vs-Mask Diagnosis (§10)

`Phase17B_detection_diagnosis.csv` — for every failing frame a mask-level target **was detected** (best instance F1 > 0.5), the GT box is present, and the best-mask instance **overlaps GT strongly, yet is not the selected instance**.

| Frame | n inst | GT box present | best inst F1 | selected inst | selected F1 | class |
|-------|--------|----------------|--------------|---------------|-------------|-------|
| 0039 | 2 | ✓ | 0.987 (i1) | i0 | 0.000 | **wrong_instance** |
| 0040 | 3 | ✓ | 0.988 (i0) | i1 | 0.000 | **wrong_instance** |
| 0041 | 9 | ✓ | 0.986 (i3) | i0 | 0.000 | **wrong_instance** |

**Diagnosis: the detection stage is fine; the selection stage picks the wrong instance.** (`wrong_instance`, not `no_detection`, not `box_good_mask_bad`.)

---

## 11. Instance Audit (§11)

`visualizations/instance_audit_DouBanLv2_0039/0040/0041.jpg` — per frame: RGB | GT | every raw P6 instance (each labeled with its instance id, F1 vs GT, SAM score, and SELECTED/ORACLE markers).

Key per-frame facts:

| Frame | selected inst | selected F1 | oracle inst | oracle F1 | selected area_ratio | oracle area_ratio |
|-------|---------------|-------------|-------------|-----------|---------------------|-------------------|
| 0039 | i_00 | 0.000 | i_01 | 0.9867 | 0.0358 | 0.1808 |
| 0040 | i_01 | 0.000 | i_00 | 0.9882 | 0.0453 | 0.1694 |
| 0041 | i_00 | 0.000 | i_03 | 0.9865 | 0.0291 | 0.1601 |

In all 3 frames the **oracle is a large, well-formed, GT-matching instance (area ratio ≈ GT)**, while the **selected instance is a tiny spurious fragment (area ratio 0.03–0.05)** that happens to sit on the plant and is nearly empty of true positives. Visually confirmed in the instance grids.

---

## 12. Score Decomposition (§12)

`Phase17B_score_regret.csv` compares the selected vs oracle instance across **all** score components recorded by the pipeline. The selected instance wins on total even though it matches GT badly, **entirely because of the 前背景对比得分 (foreground-background contrast component)**:

| Frame | oracle total | selected total | margin (sel−orc) | **driver** | oracle q_contrast | selected q_contrast | other divergences |
|-------|--------------|----------------|------------------|-----------|-------------------|---------------------|-------------------|
| 0039 | 0.6404 | 0.6788 | **+0.0384** | `contrast` | 0.1175 | **1.0000** | area favors oracle (0.43 vs 0.07) |
| 0040 | 0.6459 | 0.6933 | **+0.0474** | `contrast` | 0.1787 | **1.0000** | area favors oracle (0.40 vs 0.09) |
| 0041 | 0.6575 | 0.7198 | **+0.0623** | `contrast` | 0.2665 | **1.0000** | area favors oracle (0.38 vs 0.05) |

- **Driver per-frame: `contrast` (PROVEN).** The correct, GT-sized instance spans the full plant including low-contrast leaf edge regions → `q_contrast = contrast/0.25` lands at 0.12–0.27. The tiny spurious instance sits entirely inside a compact high-contrast blob → `q_contrast` saturates at 1.0.
- `area`, `comp`, `edge`, `temp`, `sam` components largely **favor the oracle** (larger, well-formed, better SAM scores) but cannot offset the contrast gap: the margin is razor-thin (−0.16 weighted contrast gap vs +0.10 area gap).
- `leak`/`side` weights are **0.0** (frozen config) so those columns are non-decisive.
- The pipeline's own recorded `总分` (提示词评分.csv) is the selection criterion; margins shown are on those exact values.

---

## 13. Cleanup Trace (§13)

`Phase17B_cleanup_trace.csv` — per-frame F1 at each stage: raw (best instance) → selected (选择后) → final (最终掩膜) → A5c.

- **All 10 frames:** `选择后掩膜 == 最终掩膜 == A5c_final_mask` bitwise (T3) → postcleanup F1 == raw F1 for every frame.
- On failing frames the collapse to 0.000 happens **at selection** (raw 0.987 → selected 0.000), not at cleanup.
- **C2_POSTPROCESS_FAILURE is impossible in this configuration** — there is no postprocess transformation to destroy anything.

---

## 14. Temporal Diagnosis (§14) — DouBanLv2 0037–0041

`Phase17B_temporal_diagnosis.csv` + `temporal_strip_DouBanLv2_0037-0041.jpg`:

| Frame | GT area | n raw inst | oracle area_ratio | oracle SAM | oracle contrast | oracle F1 | selected correct? |
|-------|---------|-----------|-------------------|------------|-----------------|-----------|-------------------|
| 0037 | 1 807 585 | 1 | 0.2128 | 0.9805 | 0.0297 | 0.9855 | ✅ |
| 0038 | 1 624 290 | 1 | 0.1939 | 0.9727 | 0.0242 | 0.9898 | ✅ |
| **0039** | 1 510 266 | **2** | 0.1808 | 0.9727 | 0.0294 | 0.9867 | ❌ |
| **0040** | 1 420 596 | **3** | 0.1694 | 0.9609 | 0.0447 | 0.9882 | ❌ |
| **0041** | 1 349 934 | **9** | 0.1601 | 0.9570 | 0.0666 | 0.9865 | ❌ |

- GT area shrinks smoothly (1.81M → 1.35M px) as the camera pulls back; the oracle instance tracks GT area ratio (0.21 → 0.16) and holds SAM ≈ 0.96–0.98 throughout.
- **The failure begins exactly at 0039, the first frame with 2+ raw instances.** Instance count: **1 → 1 → 2 → 3 → 9**. With a single instance there is no selection problem — the only candidate is picked. As soon as SAM3 emits an extra (spurious) instance, the ranker must choose, and it picks wrong.
- The selected spurious instances are always tiny fragments (area ratio 0.03–0.05) with perfect contrast — they outrank the correct instance even though they contain almost no true positives.
- **(SUPPORTED)** The temporal pattern is consistent across the strip: the trigger is the appearance of a competing spurious instance, not degradation of the target (which stays at 0.987 all the way through 0041).

---

## 15. Evidence Hierarchy (§17) & Decision Tree (§18)

### Evidence classification (each claim labeled once, nothing upgraded)

| Claim | Level | Basis |
|-------|-------|-------|
| Mapping is correct (C5 excluded) | **PROVEN** | T9: 10/10 stems/shapes/existence exact; overlays align |
| A correct candidate existed on all 3 failing frames | **PROVEN** | raw oracle 0.9867/0.9882/0.9865 measured vs GT |
| Cleanup is identity; no C2 | **PROVEN** | T3: 选择后==最终==A5c bitwise; cleanup_trace Δ=0 |
| Failing frames are C3 RANKING_FAILURE | **PROVEN** | oracle ≥ 0.80 (0.987) ∧ selected F1 = 0 < 0.5; wrong instance id |
| Driver is 前背景对比得分 (contrast) | **PROVEN** | score_regret.csv: contrast is the max negative weighted gap on all 3 frames; oracle 0.12–0.27 vs selected 1.0 |
| P6 prompt failure is NOT supported | **PROVEN** | generation reached 0.987; sweep gate unmet (0.987 ≥ 0.10) |
| A6/A7 are irrelevant here | **SUPPORTED** | disabled in P00; P00 artifacts only; mechanism neutral |
| Instance-count trigger (first spurious instance) | **SUPPORTED** | 0037–0041 strip: 1/1/2/3/9; failure begins at first extra instance |
| Generalization of this ranker defect to other dense-sequence frames | **HYPOTHESIS** | N=3 failing frames inside one sequence; other 5+ frame dense strips untested |

### Decision tree outcome

```
Did a correct candidate exist? ── YES (raw oracle ≥ 0.98)
   │
   ├─ Mapping wrong? ── NO (C5 excluded)
   ├─ Cleanup destroyed it? ── NO (identity; C2 excluded)
   ├─ Prompt gate? ── unmet (0.987 ≥ 0.10; sweep skipped)
   └─ Was it SELECTED? ── NO → C3_RANKING_FAILURE (3/3 failing frames)
```

**Expected (working) direction:** **Case B — ranking/selection.** Phase 17C should target **candidate scoring/selection**, specifically the contrast component's behavior on plant-sized masks vs tiny spurious fragments — **not** the P6 prompt, not cleanup, not A6/A7. (Determined here; **no tuning executed** per §16.)

---

## 16. Limitations

1. **N=10 frames, 2 sequences** — descriptive diagnostic only; no significance claim.
2. **3 failing frames within one sequence** (DouBanLv2) — the C3 finding is consistent and PROVEN per-frame but its breadth is 3 frames.
3. **Zero-inference scope** — conclusions depend on P00 per_instance candidates as executed; no re-run was allowed, so no new observations beyond the frozen artifacts.
4. **Score components beyond the recorded set** (risk penalties, semantic terms) are not separately modeled; the recorded per-component q-scores are sufficient to identify the driver at the margin shown.
5. **No GT on the 127 context candidates** — inventory F1 fields are empty for non-GT frames by design (no pseudo-GT).

---

## 17. Tests

**131/131 passed** (historical 121 + Phase 17B T1–T10). Zero failures, zero skips.

| Phase | Tests | Status |
|-------|-------|--------|
| Phase 11 | 9 | ✅ |
| Phase 12 | 19 | ✅ |
| Phase 13 | 25 | ✅ |
| Phase 14.1 | 12 | ✅ |
| Phase 16 | 15 | ✅ |
| Phase 17A | 21 | ✅ |
| **Phase 17B** | **10** | ✅ |
| **Total** | **131** | ✅ 131/131 |

Phase 17B tests: T1 oracle metric correctness (reference formula); T2 selection identity (argmax == 提示词选择 == oracle CSV); T3 cleanup identity (bitwise 3-way + postcleanup==raw); T4 regret formula (≥0, exact); T5 failure-class determinism (§5 C5>C1>C2>C3>C4; C3×3 PROVEN; C5 excluded); T6 prompt-sweep no-op (gate unmet, P1–P5 not_run, zero logs); T7 GT never enters inference; T8 zero-inference freeze (mask mtimes predate 17B); T9 frame/GT mapping exact; T10 A6/A7 disabled, scripts pass no a6/a7 flags.

---

## 18. Output Directory

```
阶段十七B_上游失败归因与候选Oracle审计/
├── Phase17B_protocol.md
├── Phase17B_report.md                 # this file
├── Phase17B_manifest.json
├── Phase17B_DEV10_manifest.csv
├── Phase17B_candidate_inventory.csv   # 157 rows (30 GT-scope with F1, 127 context no-GT)
├── Phase17B_candidate_oracle.csv      # 10 rows oracle/selected/regret
├── Phase17B_prompt_oracle.csv
├── Phase17B_score_regret.csv
├── Phase17B_cleanup_trace.csv
├── Phase17B_failure_attribution.csv
├── Phase17B_mapping_audit.csv
├── Phase17B_detection_diagnosis.csv
├── Phase17B_temporal_diagnosis.csv
├── visualizations/                   # map overlays (20) + instance audits (3) + temporal strip (1)
├── tests/test_phase17b.py            # T1–T10
└── 脚本/phase17b_*.py                # step scripts 0-1, core, attribution, step11, step14, steps15-19
```

---

## 19. Reproducibility

- Zero-inference: 17B scripts invoke **no pipeline** (T7); they read P00 artifacts whose mtimes predate the analysis (T8).
- All inputs are under git previous-commit or Phase-17A-frozen paths; SHAs recorded in `Phase17B_manifest.json`.
- Configuration freeze asserted by `脚本/phase17b_steps15_19.py` (Steps 15–16: A6/A7 off, no parameter change).
- Git commit locks state; **no push** per protocol.

---

## 20. Git Commit

```
<commit>  Phase17B: zero-inference candidate-oracle audit — refutes "generation failure";
          0039/0040/0041 = C3 RANKING_FAILURE (correct P6 candidate 0.987 not selected;
          driver = 前背景对比得分). Case B → ranking/selection for Phase 17C.
```

**Push: NOT executed** per user constraint "不得自动 push".

---

## 21. Acceptance Gates

| Gate | Criterion | Status |
|------|-----------|--------|
| G1 | Phase16/17A manifests immutable (T1/T6) | ✅ |
| G2 | DEV10 = GT10, role=DIAGNOSTIC_DEV, no add/remove | ✅ |
| G3 | Mapping audit exact for all 10 (C5 excluded) | ✅ |
| G4 | Candidate inventory 157 + oracle 10 + prompt oracle matrix | ✅ |
| G5 | Prompt-sweep gate unmet & documented; skip decision recorded | ✅ |
| G6 | Failure attribution deterministic per §5; F1=0 frames = C3 (PROVEN) | ✅ |
| G7 | Score decomposition identifies driver (contrast) per frame | ✅ |
| G8 | Cleanup trace identity verified (C2 excluded) | ✅ |
| G9 | Temporal diagnosis 0037–0041 with instance-count trigger | ✅ |
| G10 | Root-cause claim stated at correct evidence level (nothing hyped to proven) | ✅ |
| G11 | Tests T1–T10 + historical regression 121/121 → 131/131 | ✅ |
| G12 | Report §21 format complete; commit made; **no push** | ✅ |

**All 12 gates satisfied.**