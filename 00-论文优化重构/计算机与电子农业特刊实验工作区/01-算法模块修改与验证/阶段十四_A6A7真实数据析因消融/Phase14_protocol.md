# Phase 14 — Real-data A6/A7 Factorial Ablation Protocol

日期：2026-09-01

## 0. Reproducibility

```
Starting SHA:      5c6e4ab810df1a1eb96b431a775ca11dcb044287
Frozen P6 SHA:     8085143950808cc58de8a6da64bd01230abd2633
Python:            3.13.12 (sam3 env: 3.12)
CUDA:              12.6
GPU:               NVIDIA RTX A6000 ×2 (GPU 0: ~29GB free, GPU 1: ~33GB free)
SAM3 checkpoint:   /data/fj/F2DMAS/sam3/sam3.pt (3.45GB)
SAM3 repo:         /data/fj/F2DMAS/第三方源码/SAM3-latest
COLMAP loader:     /data/fj/F2DMAS/2d-gaussian-splatting-main/scene/colmap_loader.py
COLMAP data:       /data/fj/F2DMAS/00-论文优化重构/数据管理/02-位姿COLMAP/03-final_locked/
GT version:        v2 (potted_clean = potted_plant & ~blue_cube)
GT path:           /data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十二_GT_v2_QA与P6正式验收/GT_potted_clean/
Frame inputs:      /data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十二_GT_v2_QA与P6正式验收/frame_inputs/
Frame count:       21
Frozen P6 F1:      median=0.9839, mean=0.9835, min=0.9761, max=0.9891
```

## 1. Factorial Design

| Variant | A6 | A7 | Config |
|---|---|---|---|
| V00 Control | OFF | OFF | phase14_v00_control.json |
| V10 A6 | ON | OFF | phase14_v10_a6.json |
| V01 A7 | OFF | ON | phase14_v01_a7.json |
| V11 A6+A7 | ON | ON | phase14_v11_a6a7.json |

所有 variant 共享：
- 同一 21 帧 frame_inputs
- 同一 GT_potted_clean
- 同一 SAM3 model/checkpoint
- 同一 COLMAP data
- 同一 score_weights
- 同一 threshold 参数
- 同一 candidate_mode (per_instance)
- 同一 prompt_list (P6)
- 同一 image resolution
- 同一 random seed

## 2. Execution Protocol

### 2.1 严格顺序

1. V00 Control → 验证与 Frozen P6 一致
2. V10 A6
3. V01 A7
4. V11 A6+A7

每组完成后立即记录：
- 完成帧数 / 失败帧数
- A6 trigger rate
- A7 propagation rate
- wall-clock time
- fallback count

### 2.2 禁止事项

- 禁止根据中间结果调整任何参数
- 禁止跳过失败帧
- 禁止重新运行已完成的 variant
- 禁止修改 Frozen P6 产物
- 禁止修改 GT
- 禁止自动 push

### 2.3 Bug 处理

如发现确定性代码 bug：
1. STOP experiment
2. 记录 bug
3. 修复 bug
4. 重新运行受影响的 variant

### 2.4 Bug Fix Record (2026-09-01)

**Bug 1: `load_colmap_observations()` FileNotFoundError with per-sample COLMAP**
- 原因: `find_sparse_dir()` expects a single COLMAP model under `colmap_dir`. Per-sample data (e.g. `03-final_locked/CaoMei1/sparse/0/`) is not found at the parent level.
- 修复: 新增 `_load_colmap_from_sparse()` helper; `load_colmap_observations()` 增加 multi-sample fallback: 逐子目录调用 `find_sparse_dir` + 合并 observations。单模型路径向后兼容。

**Bug 2: COLMAP stem prefix mismatch**
- 原因: COLMAP stores image names as `0000.jpg` (frame number only), but mask stems are `CaoMei1_0000` (sample prefix + frame number). `colmap_mask_stem_candidates()` returns `['0000']` which never matches.
- 修复: `_load_colmap_from_sparse()` 检测 sparse_dir 的祖先目录推断 sample_prefix, 尝试 `{prefix}_{stem}` 匹配。

**Bug 3: Runner colmap_dir path resolution error**
- 原因: `PHASE14_DIR.parent.parent` resolves to `计算机与电子农业特刊实验工作区` (wrong), not `00-论文优化重构`.
- 修复: 改为 `PHASE14_DIR.parent.parent.parent`.

**验证:**
- Phase 11+12+13 regression: 71/71 pass
- Real-data COLMAP loading: 21/21 frames, all with points

**Bug 4: A6 global consensus across unrelated samples (2026-09-01 post-V10 diagnosis)**
- 原因: `apply_cross_view_consensus()` receives all 21 frames from 5 different samples. Cross-view consensus assumes frames share the same target; across unrelated samples it produces meaningless geometric support, causing severe F1 regressions (10/21 frames, mean F1 drops 0.9835→0.8662).
- 修复: A6 block now groups frames by sample name, runs `apply_cross_view_consensus` independently per sample, then merges results. `ConsensusResult` is reconstructed from merged per-sample outputs.
- 影响: V10, V11 must be re-run. V00, V01 unaffected (A6 OFF).

**Bug 5: A6 rejected masks still enter Pass 2 scoring (2026-09-01 post-V10 diagnosis)**
- 原因: `共识接受=0` 的 A6 mask 仍然作为 Candidate 进入 Pass 2 scoring, 可能因 image-based metrics 赢过 A1s baseline, 导致 corrupted mask 被选中。
- 修复: Pass 2 中 A6 Candidate 创建条件增加 `共识接受 == 1` 检查。rejected masks 不再进入 scoring。
- 影响: V10, V11 must be re-run again.

**Bug 6: A7 global memory propagation across unrelated samples (2026-09-01 post-V01 diagnosis)**
- 原因: `propagate_memory_masks()` 使用全局最佳帧作为 seed (可能来自不同 sample), 并向所有帧传播。XianKeLai1_0000 的 seed 是 ChangShouHua2_0100 (不同植物), 导致 F1=0.7118。
- 修复: A7 block now groups frames by sample, selects seed within each sample, runs `propagate_memory_masks` independently per sample, merges results.
- 影响: V01, V11 must be re-run.

**Bug 7: A7 per-sample fix causes OOM from repeated SAM3 loads (2026-09-01 post-V01 diagnosis)**
- 原因: per-sample A7 调用 `propagate_memory_masks` 5 次, 每次加载 SAM3 predictor (~4GB), 导致 CUDA OOM。
- 修复: `propagate_memory_masks` 增加可选 `predictor` 参数; A7 block 预加载一次 predictor, 传递给所有 per-sample 调用。
- 影响: V01, V11 must be re-run.

## 3. A6 Diagnosis Fields

每帧输出追加到 selection_rows CSV：
```
a6_enabled, a6_triggered, a6_status, a6_candidate_generated,
a6_candidate_count, a6_consensus_score, a6_selected, a6_fallback, a6_fallback_reason
```

## 4. A7 Diagnosis Fields

每帧输出追加到 selection_rows CSV：
```
a7_enabled, a7_status, a7_seed_frame, a7_memory_write, a7_memory_read,
a7_propagated, a7_mask_valid, a7_candidate_generated, a7_selected,
a7_fallback, a7_fallback_reason
```

## 5. Frame Ordering Audit

A7 有 temporal dependency。frame_inputs 按文件名排序：
```
CaoMei1_0000, CaoMei1_0025, CaoMei1_0050, CaoMei1_0075, CaoMei1_0100,
ChangShouHua2_0000, ..., DouBanLv1_0000, ..., KongQueZhuYu_0000, ...,
XianKeLai1_0000, ...
```

确认：每个 sample 内帧号递增，不同 sample 之间 memory reset。

## 6. Metrics

### 6.1 Per-frame

对每帧、每 variant 计算：
- F1
- IoU
- Precision
- Recall
- pred_area / gt_area ratio
- cube overlap pixels

### 6.2 Paired Delta

```
ΔA6_i  = F1(V10_i) - F1(V00_i)
ΔA7_i  = F1(V01_i) - F1(V00_i)
ΔA6A7_i = F1(V11_i) - F1(V00_i)
```

对 IoU 同理。

### 6.3 Factorial Effects

```
E_A6  = [(M10 + M11) - (M00 + M01)] / 2
E_A7  = [(M01 + M11) - (M00 + M10)] / 2
I_AB  = M11 - M10 - M01 + M00
```

### 6.4 Statistical Tests

- Wilcoxon signed-rank test (paired V10 vs V00, V01 vs V00, V11 vs V00)
- Effect size: rank-biserial correlation
- Report effective sample size (delta != 0)

## 7. Hard-case Stratification

基于 V00 Control F1：
```
Easy:   F1 >= 0.98
Medium: 0.95 <= F1 < 0.98
Hard:   F1 < 0.95
```

## 8. Trigger-conditioned Analysis

分别分析：
- All frames
- A6-triggered frames (a6_triggered=True)
- A7-propagated frames (a7_propagated=True)
- A6-or-A7 active frames
- Both-active frames

## 9. Regression Threshold

定义 regression：ΔF1 < -0.005

对每个 regression frame 输出：
- frame, variant, baseline F1, enhanced F1, delta
- winner source, A6/A7 trigger, candidate scores
- possible cause category

## 10. Acceptance Gates

| Gate | Criterion |
|---|---|
| G1 Control reproducibility | V00 F1 ≈ Frozen P6 F1 (investigate any >0.005 deviation) |
| G2 Real A6 | A6 trigger > 0 frames |
| G3 Real A7 | A7 propagated > 0 frames |
| G4 Four variants complete | V00, V10, V01, V11 all 21 frames |
| G5 No hidden exclusions | All failures retained |
| G6 Full statistics | Paired delta, main effects, interaction |
| G7 Frozen P6 integrity | Unchanged |
