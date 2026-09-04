# Phase 14 — Real-data A6/A7 Factorial Ablation Report

日期：2026-09-01

## 1. 执行摘要

2×2 析因设计 (V00 Control, V10 A6, V01 A7, V11 A6+A7) 在21帧真实数据上执行。
实验过程中发现并修复了7个确定性代码 bug（详见 §8 Bug Fix Record）。

### 最终结论

| Variant | F1 mean | F1 median | Regressions | Improvements |
|---|---|---|---|---|
| V00 Control | 0.9835 | 0.9839 | — | — |
| V10 A6 | 0.9807 | 0.9836 | 1 (mild) | 0 |
| V01 A7 | 0.9835 | 0.9839 | 0 | 0 |
| V11 A6+A7 | 0.9807 | 0.9833 | 1 (mild) | 0 |

**A6 main effect: -0.0028 (negligible)**
**A7 main effect: +0.0000 (neutral)**
**Interaction: -0.0000 (none)**

## 2. Factorial Design

| Variant | A6 | A7 | Description |
|---|---|---|---|
| V00 Control | OFF | OFF | Baseline (frozen P6) |
| V10 A6 | ON | OFF | Cross-view consensus only |
| V01 A7 | OFF | ON | Memory propagation only |
| V11 A6+A7 | ON | ON | Both enabled |

## 3. Acceptance Gates

| Gate | Criterion | Status |
|---|---|---|
| G1 Control reproducibility | V00 F1 ≈ Frozen P6 (≤0.005 delta) | PASS (delta=0.0000) |
| G2 Real A6 | A6 trigger > 0 frames | PASS (20/21 enabled, 5/21 accepted) |
| G3 Real A7 | A7 propagated > 0 frames | PASS in V11 (2/21); 0/21 in V01 (expected: A7 needs A6 seed) |
| G4 Four variants complete | All 21 frames | PASS (4×21=84 frames) |
| G5 No hidden exclusions | All failures retained | PASS |
| G6 Full statistics | Paired delta, main effects, interaction | PASS (see §4) |
| G7 Frozen P6 integrity | Unchanged | PASS |

## 4. Factorial Effects

```
M00 (Control): 0.9835
M10 (A6):      0.9807
M01 (A7):      0.9835
M11 (A6+A7):   0.9807

E_A6  = -0.0028  (negligible negative)
E_A7  = +0.0000  (neutral)
I_AB  = -0.0000  (no interaction)
```

## 5. Per-variant Detail

### 5.1 V10 A6

- A6 enabled: 20/21 frames (1 frame had insufficient COLMAP data)
- A6 accepted (IoU ≥ 0.75): 5/21 frames
- A6 selected in Pass 2: 1/21 frames
- 1 regression: ChangShouHua2_0075 (0.9856→0.9278, delta=-0.058)
  - A6 accepted=1, IoU=0.8887, deleted 2.5% pixels
  - Borderline case: consensus deletion removed valid foreground

### 5.2 V01 A7

- A7 propagated: 0/21 frames
- A7 selected: 0/21 frames
- **A7 without A6 is完全 neutral** (identical to V00)
- 原因: A7 needs consensus masks as seed base; without A6, seed selection falls back to global best which may be from a different sample, producing no valid propagation

### 5.3 V11 A6+A7

- A6: same as V10 (20 enabled, 5 accepted, 1 selected)
- A7 propagated: 2/21 frames (A7 benefits from A6 consensus masks as seed)
- A7 selected: 2/21 frames
- Same 1 regression as V10 (ChangShouHua2_0075)

## 6. Hard-case Stratification

基于 V00 Control F1:
```
Easy:   21/21 frames (all F1 >= 0.9761)
Medium: 0
Hard:   0
```

All 21 frames are "easy" — the baseline F1 is already very high (min=0.9761).
This limits the headroom for A6/A7 improvements.

## 7. Statistical Tests

- Wilcoxon A6 vs V00: n_nonzero=1 (insufficient for test)
- Wilcoxon A7 vs V00: n_nonzero=0 (no difference)
- Wilcoxon A6+A7 vs V00: n_nonzero=3 (insufficient for test)

Due to the very high baseline quality, A6/A7 have minimal opportunity to improve,
and A6's single regression is the only measurable effect.

## 8. Bug Fix Record

| Bug | Description | Fix | Impact |
|---|---|---|---|
| #1 | `load_colmap_observations()` FileNotFoundError with per-sample COLMAP | Multi-sample fallback: iterate child dirs | V10, V11 enabled |
| #2 | COLMAP stem prefix mismatch (0000 vs CaoMei1_0000) | Detect sample prefix from dir path | V10, V11 enabled |
| #3 | Runner colmap_dir path resolution error | `.parent.parent.parent` (was `.parent.parent`) | V10, V11 enabled |
| #4 | A6 global consensus across unrelated samples | Per-sample A6 processing | 10 regressions → 1 |
| #5 | A6 rejected masks still enter Pass 2 | Guard: `共识接受==1` required | 1 regression → 1 (mild) |
| #6 | A7 global memory propagation across samples | Per-sample A7 processing | XianKeLai1_0000 -0.27 regression eliminated |
| #7 | A7 per-sample causes OOM from repeated SAM3 loads | Shared predictor parameter | V01, V11 functional |

## 9. Known Limitations

1. **A7 neutral without A6**: A7 depends on A6 consensus masks as seed base. Without A6, A7 has no valid within-sample seed.
2. **Easy dataset**: All 21 frames have F1 ≥ 0.976, leaving minimal headroom for improvement.
3. **Single A6 regression**: ChangShouHua2_0075 (F1 0.986→0.928) — consensus algorithm's IoU threshold (0.75) allows some corruption on borderline frames.
4. **Statistical power**: With n=21 and near-identical scores, Wilcoxon tests are underpowered.

## 10. Files

```
阶段十四_A6A7真实数据析因消融/
├── Phase14_protocol.md              (protocol + bug fix record)
├── Phase14_report.md                (this report)
├── run_phase14_factorial.py         (runner script)
├── V00_Control/                     (21 frames, A6 OFF, A7 OFF)
├── V10_A6/                          (21 frames, A6 ON, A7 OFF)
├── V01_A7/                          (21 frames, A6 OFF, A7 ON)
├── V11_A6+A7/                       (21 frames, A6 ON, A7 ON)
└── logs/                            (run logs)
```

## 11. Reproducibility

```
Frozen P6 SHA:     8085143950808cc58de8a6da64bd01230abd2633
Hotfix SHA:        b970913d895844b1bf8389a247a5fc571597f192
Bug fix commits:   load_colmap multi-sample + per-sample A6/A7 + acceptance guard
Python:            3.12 (sam3 env)
CUDA:              12.6
GPU:               NVIDIA RTX A6000 x2
SAM3 checkpoint:   /data/fj/F2DMAS/sam3/sam3.pt
```
