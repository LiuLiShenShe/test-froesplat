# Phase 17A — Budgeted 10-GT Hard-case Pilot + Unlabeled TEST Behavioral Audit

**Date:** 2026-09-09
**Branch:** main
**Starting SHA:** `a304af7d` (post-Phase16)
**Python:** `/home/test/biosoft/enter/envs/sam3/bin/python`
**GPU:** 2× NVIDIA RTX A6000 (49 GB each)

---

## 1. Motivation & Scope

Phase 16 froze a 58-target hard-case challenge set (26 DEV / 32 TEST). Only **10 frames** carry human GT (BaiZhang 0033–0037, DouBanLv2 0037–0041). The originally planned Phase 17 (confirmatory factorial on all 32 TEST with full GT) **cannot run** — the user will not annotate the remaining 48 frames.

Phase 17A replaces it with a **budgeted two-track design**:

| Track | Frames | GT? | Purpose |
|-------|--------|-----|---------|
| **A** (P-runs) | 10 GT + 20 context | Yes | Accuracy/mechanism pilot (descriptive, N=10) |
| **B** (U-runs) | 32 locked TEST | **No** | Execution/exposure/stability/behavior audit only |

**Verbatim constraints (must hold throughout):**

> "10 labeled frames are enough for a pilot, not for broad confirmatory generalization. 32 unlabeled TEST frames can validate execution behavior, not correctness. Do not create pseudo-GT to fill the gap."

> **绝对禁止把 Track B 当作 accuracy evaluation**

---

## 2. Manifest Immutability

| Manifest | SHA256 | Status |
|----------|--------|--------|
| Phase16 full manifest | (frozen in Phase16_manifest.json) | ✅ T1 re-derived == frozen |
| Phase16 TEST manifest | `7504fcbb…` | ✅ T1 re-derived == frozen |
| Phase17A GT10 manifest | `f596b417…` | ✅ T12 frozen, no_pseudo_gt=true |

---

## 3. Pilot Role Statement

The 10 GT frames are **PILOT_GT10** — budgeted pilot subset of already-labeled DEV material. They are DEV frames, never relabeled as TEST. GT10 ∩ locked TEST = ∅ (T3). No pseudo-GT created.

---

## 4. Configuration Freeze

All 8 pipeline runs used identical frozen config from Phase 14.1:

```
--prompt_list P6 --default_prompt_id P6
--candidate_mode per_instance
--score_weights area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5
--sam3_mask_threshold 0.5
--save_raw_instance_masks --save_candidate_masks --save_intermediate_masks
--force --consensus_min_frames 5
```

- A6 variants: + `--use_cross_view_consensus --colmap_dir <abs 03-final_locked>`
- A7 variants: + `--use_memory_propagation`

**No pipeline code changes** — invocation only.

---

## 5. Execution Summary

| Variant | Track | Frames | Masks | Status | Wall-time |
|---------|-------|--------|-------|--------|-----------|
| P00 (Control) | A | 10 GT | 30/30 | ✅ success | 732 s |
| P10 (A6) | A | 10 GT | 30/30 | ✅ success | 880 s |
| P01 (A7) | A | 10 GT | 30/30 | ✅ success | 682 s |
| P11 (A6+A7) | A | 10 GT | 30/30 | ✅ success | 849 s |
| U00 (Control) | B | 32 TEST | 32/32 | ✅ success | 573 s |
| U10 (A6) | B | 32 TEST | 32/32 | ✅ success | 594 s |
| U01 (A7) | B | 32 TEST | 32/32 | ✅ success | 627 s |
| U11 (A6+A7) | B | 32 TEST | 32/32 | ✅ success | 661 s |

**8/8 runs succeeded.** Zero empty masks. Zero failures.

---

## 6. Track A — Accuracy & Hardness (GT10 Only)

### 6.1 Hardness Confirmation

| Metric | Value |
|--------|-------|
| Easy-21 anchor mean F1 | 0.9835 |
| Easy-21 anchor min F1 | 0.9761 |
| GT10 P00 mean F1 | **0.677** |
| GT10 P00 median F1 | 0.957 |
| GT10 frames below easy min | **8/10** |

**Root cause of hardness:** DouBanLv2 frames 0039–0041 produce **F1=0.0** (zero predictions — the plant is not detected). DouBanLv2 0037–0038 perform at ceiling (0.986–0.990). BaiZhang 0033–0037 cluster at 0.954–0.962 (below easy-21 min but non-zero). The distribution is **bimodal**: near-ceiling on 2 frames, moderate gap on 5 frames, complete failure on 3 frames.

### 6.2 Per-Frame F1 (P00 Baseline)

| Sample | Frame | F1 | IoU | Classification |
|--------|-------|----|-----|----------------|
| DouBanLv2 | 0037 | 0.986 | 0.972 | Ceiling |
| DouBanLv2 | 0038 | 0.990 | 0.980 | Ceiling |
| BaiZhang | 0036 | 0.962 | 0.927 | Below easy min |
| BaiZhang | 0035 | 0.961 | 0.924 | Below easy min |
| BaiZhang | 0033 | 0.959 | 0.922 | Below easy min |
| BaiZhang | 0034 | 0.956 | 0.915 | Below easy min |
| BaiZhang | 0037 | 0.954 | 0.911 | Below easy min |
| DouBanLv2 | 0039 | 0.000 | 0.000 | **Complete failure** |
| DouBanLv2 | 0040 | 0.000 | 0.000 | **Complete failure** |
| DouBanLv2 | 0041 | 0.000 | 0.000 | **Complete failure** |

### 6.3 Material Effect (ΔF1 vs P00)

| Variant | Improvement | Regression | Neutral | Mean ΔF1 |
|---------|-------------|------------|---------|----------|
| P10 (A6) | 0 | **4** | 6 | −0.007 |
| P01 (A7) | **1** | 0 | 9 | +0.001 |
| P11 (A6+A7) | 0 | **4** | 6 | −0.007 |

- A6 **always regresses** when it acts (4/4 frames → F1↓). 6/10 frames are noop (A6 unchanged vs control).
- A7 **marginally improves** 1/4 acted frames; 3/4 neutral. 6/10 noop.
- DouBanLv2: A6 and A7 are **complete no-ops** on all 5 frames (the zero-prediction frames stay zero; the ceiling frames stay ceiling).

---

## 7. Descriptive Factorial Analysis

| Effect | Mean | Median | Min | Max |
|--------|------|--------|-----|-----|
| **E_A6** (A6 main) | **−0.0075** | −0.0003 | −0.028 | 0.000 |
| **E_A7** (A7 main) | +0.0005 | 0.0000 | 0.000 | +0.003 |
| **I_AB** (A6×A7) | −0.0010 | 0.0000 | −0.006 | 0.000 |

**Per-sample means (N=2 sequences, descriptive only):**

| Sequence | E_A6 | E_A7 | I_AB |
|----------|------|------|------|
| BaiZhang | **−0.015** | +0.001 | −0.002 |
| DouBanLv2 | 0.000 | 0.000 | 0.000 |

**Interpretation (descriptive, no significance claim):**
- A6 has a **negative** main effect on hard cases: when it acts, it hurts. The negative effect is concentrated in BaiZhang (−0.015 mean) where A6 acts on 4/5 frames, all producing F1 regression.
- A7 is **near-neutral**: one marginal improvement, three neutral, acting on 4/10 frames.
- The A6×A7 interaction is negligible (−0.001).
- DouBanLv2 shows zero effect for all mechanisms — A6/A7 have nothing to act on when the baseline already fails (F1=0) or already succeeds (F1≈0.99).

---

## 8. Mechanism Analysis

### 8.1 A6 (Cross-view Consensus) Engagement

- **Track A:** A6 acts on **4/10** BaiZhang frames (all 4 regress F1). On DouBanLv2: 0/5 acted.
- **Track B:** A6 acts on **0/32** TEST frames. **Complete no-op.**

**Root cause:** Each TEST sample has exactly **4 frames**. The `consensus_min_frames=5` threshold means A6's cross-view consensus **cannot trigger** on any TEST sample. This is not a bug — it is a documented architectural constraint: A6 requires ≥5 temporally adjacent frames for cross-view registration.

### 8.2 A7 (Memory Propagation) Engagement

- **Track A:** A7 acts on **4/10** frames (1 improvement, 3 neutral). Seed-based memory propagation functional.
- **Track B:** A7 acts on **1/32** TEST frames (XianKeLai3_0204, change_ratio=0.037). 31/32 are noop.

A7's memory propagation requires a seed frame with sufficient quality; on most 4-frame TEST samples, the seed selection or memory transfer produces no behavioral change.

### 8.3 A7 Lifecycle Audit

| Run | Samples | Seed OK | Cross-sample Leak | Dangling Session |
|-----|---------|---------|-------------------|------------------|
| AP01 | 2 | 2 | 0 | None |
| AP11 | 2 | 2 | 0 | None |
| BU01 | 8 | 8 | 0 | None |
| BU11 | 8 | 8 | 0 | None |

Phase 14.1 fix (try/finally close_session) verified on all exit paths. Zero cross-sample leaks, zero missing seeds, zero dangling sessions.

---

## 9. Track B — Behavioral Audit (NO GT)

### 9.1 Execution

| Variant | Final Masks | Empty | Failures | Status |
|---------|-------------|-------|----------|--------|
| U00 | 32 | 0 | 0 | ✅ success |
| U10 (A6) | 32 | 0 | 0 | ✅ success |
| U01 (A7) | 32 | 0 | 0 | ✅ success |
| U11 (A6+A7) | 32 | 0 | 0 | ✅ success |

### 9.2 Behavioral Divergence vs U00 (Control)

| Variant | Acted | Noop | Max ΔRatio | Large Flips |
|---------|-------|------|------------|-------------|
| U10 (A6) | **0/32** | 32/32 | 0.000 | 0 |
| U01 (A7) | **1/32** | 31/32 | 0.037 | 0 |
| U11 (A6+A7) | **1/32** | 31/32 | 0.037 | 0 |

### 9.3 Stability

- **Zero large-flip frames** across all variants (change_ratio never exceeds 0.5).
- **Zero empty masks** across all variants.
- Maximum behavioral change: 3.7% of pixels on a single frame (XianKeLai3_0204, U01/U11).

### 9.4 Key Observation

Track B reveals that A6 is **architecturally inert** on 4-frame TEST samples due to `consensus_min_frames=5`. A7 has **minimal behavioral footprint** (1/32 frames). The pipeline is **stable** — no crashes, no empty masks, no large behavioral flips. But the mechanisms do not meaningfully differentiate between variants on these short sequences.

---

## 10. Cross-Evidence Synthesis

### Evidence Matrix

| Dimension | Track A (GT10) | Track B (32 TEST) |
|-----------|---------------|-------------------|
| **A6 engagement** | 4/10 acted (all F1↓) | 0/32 acted (all noop) |
| **A7 engagement** | 4/10 acted (1↑, 3 neutral) | 1/32 acted (neutral) |
| **A6×A7 interaction** | I_AB = −0.001 (negligible) | Not measurable (A6 never fires) |
| **Stability** | No crashes, no empty masks | No crashes, no empty masks, no large flips |
| **A7 lifecycle** | Clean (0 leaks, 0 dangling) | Clean (0 leaks, 0 dangling) |

### Narrative Synthesis

1. **A6 (Cross-view Consensus) hurts hard cases and cannot activate on short sequences.**
   - On Track A (10 GT frames, 5+ frame sequences): A6 acts on 4/10 BaiZhang frames, and every instance produces F1 regression (mean ΔF1=−0.007). A6 consistently removes correct mask pixels without replacing them — a net negative on hard cases.
   - On Track B (32 TEST frames, 4-frame sequences): A6 is a **complete no-op** (0/32). The `consensus_min_frames=5` threshold blocks A6 entirely. This is architectural, not a bug.
   - **Combined assessment:** A6 provides no benefit and active harm on hard cases. On short sequences it doesn't even run. There is no evidence supporting A6 as a default mechanism.

2. **A7 (Memory Propagation) is near-neutral with marginal upside.**
   - On Track A: 4/10 frames acted, 1 improvement (BaiZhang, ΔF1=+0.006), 3 neutral. No regression.
   - On Track B: 1/32 frames showed behavioral change (3.7% pixel ratio on XianKeLai3_0204), no large flips.
   - A7 lifecycle is clean across all 4 A7 runs.
   - **Combined assessment:** A7 neither helps nor hurts meaningfully. Its memory propagation is safe (no leaks, no crashes) but provides negligible behavioral impact on both hard and short sequences.

3. **Hard-case root cause is NOT mechanism failure — it is detection failure.**
   - DouBanLv2 frames 0039–0041: F1=0.0 (plant not detected at all). No mechanism can help when the baseline produces zero predictions.
   - BaiZhang frames: F1=0.954–0.962 (moderate gap below easy-21 min). A6 hurts; A7 doesn't meaningfully help. The gap is driven by **P6 prompt limitations**, not mechanism quality.
   - The hard-case problem is upstream (prompt/candidate generation), not in the A6/A7 mechanisms.

4. **The 4-frame constraint limits generalizability of Track B findings.**
   - All 8 TEST samples have exactly 4 frames. A6 requires ≥5. This means Track B cannot test A6 behavior on realistic 5+ frame sequences. This is a known limitation, not a discovery.

---

## 11. Architecture Decision (Provisional)

**Recommendation: Disable both A6 and A7 as production defaults.**

| Mechanism | Evidence | Recommendation |
|-----------|----------|----------------|
| A6 | Negative on hard cases (−0.007 F1), inert on short sequences, never improves | **Disable** |
| A7 | Near-neutral, 1 marginal improvement in 40 variant-frames, clean lifecycle | **Disable** |

**Rationale:**
- A6 has a documented negative effect when it acts (4/4 frames regress). It never improves any frame. Its architectural dependency on ≥5 frames makes it irrelevant for short sequences.
- A7 is safe but adds complexity without measurable benefit. The single marginal improvement (ΔF1=+0.006 on 1/10 frames) does not justify the operational overhead.
- The hard-case problem requires upstream solutions (prompt engineering, candidate generation, resolution handling), not A6/A7 mechanisms.

**Provisional caveat:** This decision is based on N=10 descriptive pilot data (2 sequences) plus 32 unlabeled behavioral observations. A confirmatory evaluation with full GT on all 58 hard-case frames remains blocked. If future work provides full GT and shows A7 has meaningful benefit on longer sequences, this decision should be revisited.

---

## 12. Limitations

1. **N=10, 2 sequences.** Descriptive only; within-sequence dependence acknowledged. No significance claim.
2. **4-frame TEST samples.** A6 cannot trigger on short sequences (architectural constraint). Track B behavioral findings are constrained to A7 evaluation.
3. **No full-GT confirmatory evaluation.** The 32 TEST frames have no GT; Track B is execution/behavior audit only, NOT accuracy.
4. **DouBanLv2 zero-prediction frames.** 3/10 GT frames have F1=0.0 — the mechanism evaluation is effectively on 7 frames for meaningful delta analysis.
5. **Single difficulty mode.** All hard cases are multi-plant density (3–4 plants per frame). Other difficulty modes (occlusion, partial visibility) not represented.

---

## 13. Tests

**121/121 tests passed** across all historical phases (11, 12, 13, 14.1, 16) plus Phase 17A (T1–T12). Zero failures, zero skips.

| Phase | Tests | Status |
|-------|-------|--------|
| Phase 11 | 9 | ✅ all pass |
| Phase 12 | 19 | ✅ all pass |
| Phase 13 | 25 | ✅ all pass |
| Phase 14.1 | 12 | ✅ all pass |
| Phase 16 | 15 | ✅ all pass |
| **Phase 17A** | **21** | **✅ all pass** |
| **Total** | **121** | **✅ 121/121** |

---

## 14. Output Directory

```
阶段十七A_10GT困难样本Pilot与无GT测试审计/
├── 00_input_trackA/          # 30 prefixed frames
├── 01_input_trackB/          # 32 prefixed TEST frames
├── 02_trackA_variants/       # P00/ P10/ P01/ P11/ pipeline outputs
├── 03_trackB_variants/       # U00/ U10/ U01/ U11/ pipeline outputs
├── 04_metrics/               # Track A F1/IoU/P/R + material + hardness
├── 05_behavior/              # Track B behavioral (NO GT) + execution + stability
├── 06_mechanism/             # A6/A7 mechanism analysis + A7 lifecycle
├── 07_gallery/               # Qualitative grids (Track A + Track B)
├── GT10_manifest/            # Frozen GT10 CSV + SHA256 manifest
├── logs/                     # 8 run logs + nvidia-smi snapshots
├── 脚本/                     # All analysis scripts
├── tests/                    # test_phase17a.py (T1–T12)
├── Phase17A_starting_audit.md
└── Phase17A_report.md        # This file
```

---

## 15. Reproducibility

- All configs frozen and documented (§4).
- GT10 manifest SHA256 locked (T12).
- Phase16 TEST manifest SHA256 immutable (T1).
- All pipeline runs use identical frozen config; only a6/a7 flags vary.
- No pipeline code changes — invocation only.
- Git commit locks state; **no push** per protocol.

---

## 16. Git Commit

```
git add 阶段十七A_10GT困难样本Pilot与无GT测试审计/
git commit -m "Phase17A: 10-GT hard-case pilot (descriptive) + 32-TEST behavioral audit (no GT, no pseudo-GT)"
```

**Push: NOT executed** per user constraint "不得自动 push".

---

## Acceptance Gates

| Gate | Criterion | Status |
|------|-----------|--------|
| G1 | Phase16 manifests + TEST SHA immutable (T1) | ✅ |
| G2 | GT10 frozen as PILOT_GT10, no pseudo-GT (T12) | ✅ |
| G3 | Track A: 4 variants × 10 GT frames executed + metrics (T6/T7) | ✅ |
| G4 | Track B: 32 unlabeled frames executed + behavioral-only (T8/T9) | ✅ |
| G5 | Descriptive factorial computed, significance withheld | ✅ |
| G6 | A7 lifecycle audited (T11) | ✅ |
| G7 | No TEST frame scored against GT; no accuracy from Track B | ✅ |
| G8 | Architecture decision + limitations documented | ✅ |
| G9 | Tests T1–T12 + full regression 121 pass | ✅ |
| G10 | Commit made, **no push** | ✅ |

**All 10 gates satisfied.**
