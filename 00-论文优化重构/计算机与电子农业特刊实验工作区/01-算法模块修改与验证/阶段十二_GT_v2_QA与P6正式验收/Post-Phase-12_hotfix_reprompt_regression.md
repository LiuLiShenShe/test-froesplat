# Post-Phase-12 Hotfix: Reprompt Regression Closure

日期：2026-08-31

## 1. Background

explore agent 在代码审计中发现：

```text
reprompt_stems could be referenced before initialization when needs_reprompt=True
```

具体位置：`生成RAP-FSAM3掩膜.py` L2734
```python
reprompt_stems.add(image_path.stem) if needs_reprompt else None
```

`reprompt_stems` 在 `main()` 中从未声明为 `set`，直接 `.add()` 会触发 `NameError`。

## 2. Root Cause

此前 Phase 12 P6 baseline 运行中，`needs_reprompt` 从未为 `True`：
- P6 baseline 使用 `per_instance` 模式，每个 prompt 通常只有一个实例
- 即使有多个实例，top1/top2 score gap 也大于 `reprompt_score_gap` (0.05)
- 因此 `reprompt_stems.add()` 从未执行，NameError 从未触发

```text
17/17 tests PASS
```

并不能证明 reprompt path 没问题 — 该分支从未被测试覆盖。

## 3. Fix

在 `main()` 函数中，Pass 1 循环前新增初始化：

```python
# 位置：生成RAP-FSAM3掩膜.py L2640
reprompt_stems: set[str] = set()
```

这确保 `reprompt_stems.add()` 在 `needs_reprompt=True` 时不会 NameError。

**注意**：`reprompt_stems` 当前是 dead code — 它被写入但从未被读取。
Pass 1 的 `needs_reprompt` 返回值仅用于填充该 set，不驱动任何下游逻辑。
Pass 2 的 `重提示帧标记.csv` 使用的是完全独立的 `reprompt_score()` 时序一致性检测。

## 4. Regression Test

新增 `tests/test_phase12_hotfix_reprompt.py`（14 项测试），覆盖 T1-T6：

### T1 — Trigger（3 tests）
- `test_score_gap_triggers_reprompt`: 构造 top1=0.71, top2=0.69, gap=0.02 < 0.05 → `needs_reprompt=True`
- `test_single_candidate_low_score_triggers`: 单候选 score=0.15 < min_score=0.2 → `needs_reprompt=True`
- `test_empty_mask_triggers`: 空掩膜 empty_flag=True → `needs_reprompt=True`

### T2 — No NameError（1 test）
- `test_reprompt_stems_no_nameerror`: 验证生产代码中 `reprompt_stems: set[str] = set()` 存在

### T3 — Stem Bookkeeping（2 tests）
- `test_reprompt_stems_is_set`: `needs_reprompt=True` → `stem in reprompt_stems`
- `test_no_reprompt_stem_not_added`: `needs_reprompt=False` → `stem not in reprompt_stems`

### T4 — Control（2 tests）
- `test_large_gap_no_reprompt`: gap=0.60 >> 0.05 → `needs_reprompt=False`
- `test_single_candidate_high_score_no_reprompt`: score=0.85 >= 0.2 → `needs_reprompt=False`

### T5 — Output/Log（3 tests）
- `test_reprompt_csv_columns`: `重提示帧标记.csv` 含 `图像/是否标记/重提示分数` 列
- `test_reprompt_log_recorded`: `运行日志.json` 含 `reprompt_marked_count` 和 `reprompt_marked`
- `test_reprompt_zero_in_frozen_baseline`: 冻结 baseline 中 `reprompt_marked_count=0`

### T6 — Candidate Validity（2 tests）
- `test_reprompt_selected_mask_valid`: reprompt 触发时，返回的 mask 仍是有效 ndarray
- `test_empty_candidate_mask_is_empty`: 空候选产生空 mask 但结构合法

### 额外：ScoreSelect Branch（1 test）
- `test_score_select_picks_correct_instance`: 验证修复后的 score_select 按 `(prompt_id, instance_id)` 匹配

## 5. Test Result

```text
Phase 11: 9/9 PASS
Phase 12: 8/8 PASS
Post-Phase-12 hotfix: 14/14 PASS
Total: 31/31 PASS
```

## 6. Frozen Baseline Statement

The frozen Phase-12 P6 baseline was produced before this hotfix.
Reprompt was disabled/not triggered in that frozen baseline.
Therefore this hotfix does not alter the frozen P6 predictions,
metrics, or conclusions.

冻结的 Phase-12 P6 基线在此 hotfix 之前产生。
Reprompt 在该冻结基线中未被启用/未被触发。
因此此 hotfix 不会改变冻结的 P6 预测、指标或结论。

## 7. Reproducibility

| 项目 | 值 |
| --- | --- |
| Frozen Phase 12 commit | `8085143950808cc58de8a6da64bd01230abd2633` |
| Hotfix commit (含 Phase 12 交付物) | `b970913d895844b1bf8389a247a5fc571597f192` |
| 当前 working tree | clean (无未提交修改) |
| Test command | `python3 -m pytest tests/test_phase11.py tests/test_phase12.py tests/test_phase12_hotfix_reprompt.py -v` |
| Test result | 31/31 PASS |
| git status | clean |

### Files Changed (hotfix)

| 文件 | 修改内容 |
| --- | --- |
| `生成RAP-FSAM3掩膜.py` L2640 | 新增 `reprompt_stems: set[str] = set()` |
| `tests/test_phase12_hotfix_reprompt.py` | 新增 14 项 reprompt 回归测试 |
