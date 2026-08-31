# Post-Phase-12 Hotfix Closure + Phase 13 Preflight

日期：2026-08-31

---

## 1. Code Audit

```
P0 Code Audit
PASS

reprompt_stems initialization:
  FIXED at L2640: reprompt_stems: set[str] = set()
  Previously: undeclared, would NameError on needs_reprompt=True

reprompt trigger condition:
  L1603-1605 (multi-candidate): sorted_recs[0].total_score - sorted_recs[1].total_score < args.reprompt_score_gap
  L1608-1609 (single-candidate): row is None OR total_score < reprompt_min_score OR empty_flag=True

reprompt output path:
  Pass 1: reprompt_stems (set, written but never read — dead variable)
  Pass 2: 重提示帧标记.csv (temporal consistency, independent concept)

mask_threshold:
  L885: _masks_scores_boxes(output, h, w, mask_threshold: float = 0.5)
  L911: m = logits[i, 0] > mask_threshold  (parameterized, not hardcoded)
  L968: call site passes args.sam3_mask_threshold

score_weights.sam:
  L559: default includes "sam=0.5"
  L1494: weights.get("sam", 0.0) * q_sam  (non-zero with default)

A6/A7 entry points:
  L673: use_cross_view_consensus (store_true, default False)
  L690: use_memory_propagation (store_true, default False)
  L2740-2816: A6 consensus pass (guarded by args.use_cross_view_consensus)
  L2780-2816: A7 memory pass (guarded by args.use_memory_propagation)

fallback paths:
  L506-508: torch.cuda.OutOfMemoryError → returns {}, info["cuda_oom_fallback"]
  L509-511: Exception → returns {}, info["unavailable: ..."]
  L2804-2809: memory propagation exception → degrades to per-frame mode
```

## 2. Reprompt Regression

```
Trigger verified: YES (3 trigger tests PASS)
needs_reprompt=True executed: YES (score_gap, low_score, empty_mask)
reprompt_stems verified: YES (initialization + bookkeeping tests)
output/log verified: YES (CSV columns + run log JSON + frozen baseline check)
control case verified: YES (2 control tests PASS)
```

## 3. Tests

```
Phase 11: 9/9 PASS
Phase 12: 8/8 PASS
Hotfix: 14/14 PASS
Total: 31/31 PASS
```

## 4. Frozen P6

```
Re-run: NO
Outputs modified: NO
Metrics modified: NO
Frozen commit: 8085143950808cc58de8a6da64bd01230abd2633 (Phase 12 deliverables in b970913d)
```

## 5. Hotfix

```
Current commit: b970913d895844b1bf8389a247a5fc571597f192
Files changed:
- 生成RAP-FSAM3掩膜.py L2640: +reprompt_stems: set[str] = set()
- tests/test_phase12_hotfix_reprompt.py: +14 regression tests
```

## 6. Evidence

```
Hotfix report: 阶段十二_GT_v2_QA与P6正式验收/Post-Phase-12_hotfix_reprompt_regression.md
New test: 阶段十二_GT_v2_QA与P6正式验收/tests/test_phase12_hotfix_reprompt.py
```

## 7. Phase 13 Preflight — Branch Coverage Matrix

| Branch | Trigger | Current test coverage | Runtime risk | Phase 13 action |
|---|---|---|---|---|
| **P6 baseline** | `--prompt_list P6 --default_prompt_id P6` | ✅ 21/21 frames, F1=0.9839 | low | **FROZEN** |
| **reprompt (Pass 1)** | top1-top2 gap < `reprompt_score_gap` | ✅ 3 trigger + 2 control tests | low (dead variable) | verify in A6+A7 multi-prompt runs |
| **reprompt (Pass 2)** | `--use_reprompt_detection` + temporal score > threshold | ⚠️ CSV structure tested, trigger NOT tested | medium | enable in Phase 13 with real frames |
| **A6 cross-view** | `--use_cross_view_consensus` | ⚠️ default-off test only | medium | forced validation with synthetic consensus |
| **A7 memory** | `--use_memory_propagation` | ⚠️ default-off test only | medium | forced validation with synthetic memory |
| **A6+A7 combined** | both flags ON | ❌ not tested | high | sequential enable, verify no interaction bug |
| **empty candidate** | all candidates have empty mask | ✅ tested (empty_mask_scores_zero) | low | covered |
| **OOM fallback** | `torch.cuda.OutOfMemoryError` in video predictor | ❌ not tested (requires GPU OOM) | high | synthetic OOM mock test |
| **model unavailable** | SAM3 load failure | ❌ not tested | medium | mock load_sam3 to raise |
| **fallback (general)** | any Exception in memory propagation | ✅ code path exists (L2804) | low | covered by exception handler |

### Coverage Summary

| Category | Tested | Untested |
|---|---|---|
| Core inference (P6) | ✅ | — |
| Score/selection logic | ✅ | — |
| Reprompt trigger | ✅ | — |
| A6 consensus | ⚠️ | trigger + voting |
| A7 memory | ⚠️ | propagation + fallback |
| OOM | — | ❌ |
| Model unavailable | — | ❌ |

## 8. Remaining Risks

1. **A6/A7 have no runtime test coverage** — default-off tests verify flags exist but not actual consensus/voting logic
2. **OOM fallback untested** — requires either real GPU OOM or heavy mocking
3. **`reprompt_stems` is dead code** — written but never read; Pass 1 `needs_reprompt` flag is returned but not consumed downstream
4. **Pass 2 reprompt detection** (`重提示帧标记.csv`) has structure tests but trigger logic (`reprompt_score()`) is untested
5. **`score_select` branch** in `select_mask` is only tested with synthetic data; ensemble mode not tested with real candidates

## 9. Final Verdict

```
Phase 11: CLOSED
Phase 12: CLOSED
Post-Phase-12 hotfix: CLOSED
Ready for Phase 13: YES (with caveats on A6/A7/OOM coverage)
```

Phase 13 recommended execution order:
1. A6 forced validation (synthetic cross-view data)
2. A7 forced validation (synthetic memory propagation)
3. A6+A7 combined (sequential enable)
4. OOM/unavailable mock tests
5. Full 21-frame A6 baseline comparison
6. Full 21-frame A6+A7 baseline comparison
