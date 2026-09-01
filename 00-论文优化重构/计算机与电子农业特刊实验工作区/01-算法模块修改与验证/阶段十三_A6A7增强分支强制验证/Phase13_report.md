# Phase 13/13.1 — A6/A7 增强分支强制验证报告

日期：2026-09-01（13.1 hardening 更新）

## 1. 目标

证明 A6（跨视角共识）、A7（记忆传播）、失败路径在生产代码中**真实执行**。
本阶段只解决前两层：trigger + safety。不证明性能提升。

## 2. 测试矩阵

| 测试文件 | 测试数 | 通过 | 覆盖维度 |
|---|---|---|---|
| test_phase13_a6_runtime.py | 10 | 10 ✅ | A6 trigger, candidate, scoring, output, evidence, **graceful degradation** |
| test_phase13_a7_runtime.py | 8 | 8 ✅ | A7 enable, memory write/read, propagation, output, lifecycle |
| test_phase13_a6a7_combined.py | 8 | 8 ✅ | A6+A7 interface, default-OFF baseline, unified scoring |
| test_phase13_failures_reprompt.py | 12 | 12 ✅ | F1-F5 failure injection (**F2 strict, F5 shape guard**), F6 reprompt |
| test_phase13_integration.py | 2 | 2 ✅ | **Same-run A6→A7→Pass2 chain, shape guard in chain** |
| **Phase 13 总计** | **40** | **40 ✅** | |

## 3. 全量回归

| 阶段 | 测试数 | 通过 |
|---|---|---|
| Phase 11 | 9 | 9 ✅ |
| Phase 12 | 8 | 8 ✅ |
| Phase 12 hotfix | 14 | 14 ✅ |
| Phase 13 + 13.1 | 40 | 40 ✅ |
| **总计** | **71** | **71 ✅** |

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
| **A6 异常降级（13.1 新增）** | ✅ try/except 包裹 → consensus_result=None, 保留 selected_by_stem |
| A6 degradation 测试 | ✅ exception → None, selected_by_stem 不变 |

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
| **A7 形状守卫（13.1 新增）** | ✅ Pass 2 中 mask.shape != image.shape → skip, 不进入 variants |

## 6. A6+A7 联合验证

| 检查项 | 结果 |
|---|---|
| A6/A7 interface compatibility | ✅ 两者各自返回有效结果，候选格式兼容 |
| 统一评分 | ✅ A1s/A6/A7 候选均通过 score_candidate() |
| max() 选最佳变体 | ✅ highest sam_score → highest total_score |
| 默认全关 | ✅ 7个增强标志全部默认 False |
| guard 存在 | ✅ A6/A7/reprompt 均有 if guard |
| **Same-run pipeline（13.1 新增）** | ✅ A6 consensus → A7 seed from consensus → Pass 2 三变体 scoring |

### Same-run integration 验证细节

`test_a6_a7_pass2_chain` 执行完整链路：
1. 构造 3 帧 selected_by_stem
2. 调用 `apply_cross_view_consensus()` → ConsensusResult
3. 用 consensus_result.per_frame_masks 作为 A7 seed base
4. Mock SAM3 predictor → `propagate_memory_masks()` → memory_masks
5. 执行 Pass 2 variant creation（含形状守卫）
6. 验证：每帧 variants 包含 A1s + A6 + A7
7. 验证：score_candidate 对三者均返回有效 ScoreRecord
8. 验证：max() 选出最佳变体

## 7. 失败注入验证

| 编号 | 场景 | 结果 |
|---|---|---|
| F1 | A6 empty colmap | ✅ 不崩溃, 返回 None 或有效结果 |
| F1b | A6 insufficient frames | ✅ 返回 None |
| F2 | A7 SAM3 load failure | ✅ 返回 {}, status.startswith("unavailable:") **（严格断言，无 except pass）** |
| F3 | A7 CUDA OOM | ✅ 返回 {}, status="cuda_oom_fallback" |
| F4 | Empty mask → score=0 | ✅ total_score=0.0, empty_flag=True |
| F4b | Empty mask → needs_reprompt | ✅ select_mask 返回 needs_reprompt=True |
| F5 | Wrong shape A7 mask | ✅ **Pass 2 shape guard → skip, 不进入 variants（13.1 修正）** |
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

1. **A7 mock 测试无法验证 GPU 实际行为** — OOM 路径通过 mock `torch.cuda.OutOfMemoryError` 验证
2. **Pass-2 reprompt 检测** — `reprompt_score()` 函数已验证，但 `重提示帧标记.csv` 的完整端到端路径未测试（需真实帧序列）
3. **colmap 数据依赖** — A6 需要真实 COLMAP 3D 点数据，mock 测试验证了函数逻辑但未验证点云投影精度
4. **Pass-1 reprompt_stems 是 dead code** — 写入但从未读取；Pass-2 reprompt 只生成 CSV 不重新推理

## 10. 验收条件

| 条件 | 状态 |
|---|---|
| A6 真实执行 | ✅ forced validation 证明触发 |
| A7 真实执行 | ✅ forced validation 证明 memory write/read |
| A6 异常降级 | ✅ try/except → consensus_result=None, selected_by_stem 不变 |
| A7 形状守卫 | ✅ wrong shape → skip, 不进入 variants |
| 失败安全 | ✅ F1-F5（F2 严格断言、F5 形状守卫后验证 skip） |
| Same-run pipeline | ✅ A6→A7→Pass2 链式调用验证 |
| 不影响 frozen P6 | ✅ 输出未修改 |
| 全量回归通过 | ✅ 71/71 |

## 11. 文件清单

```
阶段十三_A6A7增强分支强制验证/
├── Phase13_branch_audit.md              (P0 生产代码审计)
├── Phase13_report.md                    (本报告)
├── tests/
│   ├── test_phase13_a6_runtime.py       (10 tests — A6 + degradation)
│   ├── test_phase13_a7_runtime.py       (8 tests — A7 lifecycle)
│   ├── test_phase13_a6a7_combined.py    (8 tests — interface + baseline)
│   ├── test_phase13_failures_reprompt.py (12 tests — failures + reprompt)
│   └── test_phase13_integration.py      (2 tests — same-run chain)
```

## 12. 13.1 修正记录

| 问题 | 修正 |
|---|---|
| A6 无 try/except → 异常中断实验 | 生产代码增加 try/except，异常时 consensus_result=None |
| F2 `except RuntimeError: pass` loophole | 改为严格断言：memory_masks=={} 且 status.startswith("unavailable:") |
| F5 wrong-shape crash 无守卫 | 生产代码 Pass 2 增加形状守卫：shape 不匹配 → skip A7 candidate |
| A6+A7 "combined" 测试未证明 same-run | 新建 integration test：A6→A7→Pass2 链式调用 |
| 报告措辞过强 | 修正所有相关表述 |
