# Phase 13 — A6/A7 增强分支强制验证报告

日期：2026-08-31

## 1. 目标

证明 A6（跨视角共识）、A7（记忆传播）、失败路径在生产代码中**真实执行**。
本阶段只解决前两层：trigger + safety。不证明性能提升。

## 2. 测试矩阵

| 测试文件 | 测试数 | 通过 | 覆盖维度 |
|---|---|---|---|
| test_phase13_a6_runtime.py | 8 | 8 ✅ | A6 trigger, candidate, scoring, output, evidence |
| test_phase13_a7_runtime.py | 8 | 8 ✅ | A7 enable, memory write/read, propagation, output, lifecycle |
| test_phase13_a6a7_combined.py | 8 | 8 ✅ | A6+A7 combined, default-OFF baseline, unified scoring |
| test_phase13_failures_reprompt.py | 12 | 12 ✅ | F1-F5 failure injection, F6 reprompt detection |
| **Phase 13 总计** | **36** | **36 ✅** | |

## 3. 全量回归

| 阶段 | 测试数 | 通过 |
|---|---|---|
| Phase 11 | 9 | 9 ✅ |
| Phase 12 | 8 | 8 ✅ |
| Phase 12 hotfix | 14 | 14 ✅ |
| Phase 13 | 36 | 36 ✅ |
| **总计** | **67** | **67 ✅** |

## 4. A6 分支验证结果

| 检查项 | 结果 |
|---|---|
| flag defaults False | ✅ `use_cross_view_consensus` 默认关闭 |
| parser 存在 | ✅ `--use_cross_view_consensus` in source |
| sufficient frames → ConsensusResult | ✅ 3帧 → 非None, 有 per_frame_masks |
| insufficient frames → None | ✅ 1帧 < min_frames → None |
| per_frame_masks 非空 | ✅ 每帧 ndarray(bool), shape 正确 |
| per_frame_info 有共识指标 | ✅ 共识启用/共识接受/回退IoU/删除像素比例/补回像素比例 |
| ConsensusResult 结构完整 | ✅ reference_mask, per_frame_masks, per_frame_info, geo_support, center_band_mask |
| A6 无 try/except | ⚠️ 生产代码无 try/except 包裹 → 异常向上传播 |

## 5. A7 分支验证结果

| 检查项 | 结果 |
|---|---|
| flag defaults False | ✅ `use_memory_propagation` 默认关闭 |
| parser 存在 | ✅ `--use_memory_propagation` in source |
| seed frame → add_prompt (memory write) | ✅ mock 验证 add_prompt 被调用, frame_index=0, text 正确 |
| propagate_in_video (memory read) | ✅ mock 验证 propagate_in_video 被调用 |
| memory_masks 非空 | ✅ 输出 dict 有值, 每个 mask 是 ndarray(bool) |
| memory_info 结构 | ✅ 记忆后端/种子帧/传播方向/状态 |
| 多帧生命周期 | ✅ frame_0 write → frame_1 read, write+read 各调用一次 |
| A7 有 try/except | ✅ L2795-2809 捕获异常 → degrade to per-frame mode |

## 6. A6+A7 联合验证

| 检查项 | 结果 |
|---|---|
| A6+A7 同时成功 | ✅ 两者各自返回有效结果 |
| 统一评分 | ✅ A1s/A6/A7 候选均通过 score_candidate() |
| max() 选最佳变体 | ✅ highest sam_score → highest total_score |
| 默认全关 | ✅ 7个增强标志全部默认 False |
| guard 存在 | ✅ A6/A7/reprompt 均有 if guard |

## 7. 失败注入验证

| 编号 | 场景 | 结果 |
|---|---|---|
| F1 | A6 empty colmap | ✅ 不崩溃, 返回 None 或有效结果 |
| F1b | A6 insufficient frames | ✅ 返回 None |
| F2 | A7 SAM3 load failure | ✅ 返回 {}, status 含 "unavailable" |
| F3 | A7 CUDA OOM | ✅ 返回 {}, status="cuda_oom_fallback" |
| F4 | Empty mask → score=0 | ✅ total_score=0.0, empty_flag=True |
| F4b | Empty mask → needs_reprompt | ✅ select_mask 返回 needs_reprompt=True |
| F5 | Wrong shape mask | ✅ IndexError (文档化行为) |
| F5b | Valid memory mask | ✅ 正常评分, total_score>0 |
| F6 | reprompt_score() | ✅ 计算返回 dict, score>=0 |
| F6b | identical masks → low score | ✅ score < 0.5 |
| F6c | different masks → higher score | ✅ diff > same |
| F6d | flag exists in source | ✅ use_reprompt_detection 存在 |

## 8. 冻结 P6 基线状态

```
重新生成: NO
输出修改: NO
指标修改: NO
Frozen commit: 8085143950808cc58de8a6da64bd01230abd2633
Hotfix commit: b970913d895844b1bf8389a247a5fc571597f192
```

## 9. 已知限制

1. **A6 无 try/except 包裹** — 生产代码中 `apply_cross_view_consensus` 的任何异常会直接传播到 `main()`，无 graceful degradation
2. **A7 mock 测试无法验证 GPU 实际行为** — OOM 路径通过 mock `torch.cuda.OutOfMemoryError` 验证
3. **Pass-2 reprompt 检测** — `reprompt_score()` 函数已验证，但 `重提示帧标记.csv` 的完整端到端路径未测试（需真实帧序列）
4. **colmap 数据依赖** — A6 需要真实 COLMAP 3D 点数据，mock 测试验证了函数逻辑但未验证点云投影精度

## 10. 验收条件

| 条件 | 状态 |
|---|---|
| A6 真实执行 | ✅ forced validation 证明触发 |
| A7 真实执行 | ✅ forced validation 证明 memory write/read |
| 失败安全 | ✅ F1-F5 全部验证 |
| 不影响 frozen P6 | ✅ 输出未修改 |
| 全量回归通过 | ✅ 67/67 |

## 11. 文件清单

```
阶段十三_A6A7增强分支强制验证/
├── Phase13_branch_audit.md          (P0 生产代码审计)
├── Phase13_report.md                (本报告)
├── tests/
│   ├── test_phase13_a6_runtime.py   (8 tests — A6 forced validation)
│   ├── test_phase13_a7_runtime.py   (8 tests — A7 forced validation)
│   ├── test_phase13_a6a7_combined.py (8 tests — combined + baseline)
│   └── test_phase13_failures_reprompt.py (12 tests — failures + reprompt)
```
