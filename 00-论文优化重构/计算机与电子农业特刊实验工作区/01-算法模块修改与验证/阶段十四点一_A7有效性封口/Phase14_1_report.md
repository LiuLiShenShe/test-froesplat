# Phase 14.1 — Factorial Validity Closure Report (A7 Evidence Correction)

日期：2026-09-04
Freeze SHA：`f9f091d82122148ef011a8063d4949fbccc94828`（Phase 14 commit，V00/V10/V01/V11 产物 git-clean，未改动）
Scope：A7 生命周期/资源 bug 修复 + V01-R1/V11-R1 复跑 + 修正析因。**未做任何阈值/权重/超参调优，未重跑 V00/V10。**

---

## A. 为什么 Phase 14 需要修正

Phase14_report.md 原结论 "A7 main effect = +0.0000 (neutral)" 建立在一个 **无效的 A7 观测** 上：

| 项 | Phase 14 原报告声称 | 实际产物（`V01_A7/日志/记忆传播.json`） |
|---|---|---|
| V01 (A7-only) | A7 propagated 0/21，"A7 without A6 is completely neutral" | `状态=cuda_oom_fallback`，`记忆候选帧数=0` |
| V11 (A6+A7) | A7 propagated 2/21 | 汇总只反映最后一个 sample（XianKeLai1），`记忆候选帧数=8` |
| A7 依赖 A6 seed | §5.2 声称需要共识掩膜作 seed | 代码审计：A7 逐 sample 独立选 seed，无结构性 A6 依赖 |

**原则：** "A module that falls back to baseline has not demonstrated a neutral effect. Failure to execute ≠ zero treatment effect."

V01 是一个 **运行时失败观测**（cuda_oom_fallback），不是 "A7 effect = 0" 的证据。完整的不一致清单见 `Phase14_correction_note.md`。

## B. 根因：session 生命周期泄漏

原代码 `propagate_memory_masks` 只在成功路径和 `seed_empty` 路径调用 `close_session`；`torch.cuda.OutOfMemoryError` 和通用 `Exception` 路径直接 return，**不关闭 session**。

原始运行 STDERR 证据：

| Run | "started new session" | "removed session" | "propagation ended" |
|---|---|---|---|
| V01（原，A6 OFF） | **5** | **0** | **0** |
| V11（原，A6 ON） | **5** | **4** | **4** |

V01 的 5 个 session 全部泄漏 → 每个 sample 的 GPU 内存累积在已被泄漏的 predictor state 上 → 第 1 个 session 即 OOM，后续 4 个 sample 继续在累积的泄漏内存上建 session，进一步恶化。

**V11 也有 1 个泄漏 session**（DouBanLv1，字母序第 3），这解释了为什么原 V11 只有 8 个传播掩膜而 V11-R1 有 13 个：泄漏的 DouBanLv1 session 静默丢失了掩膜。原 V11 的 "状态=ok" 只对最后一个 sample（XianKeLai1）成立，不能推断 5 个 sample 全部成功。

## C. 修复（仅确定性资源/生命周期/报告 bug，无算法改动）

对 `生成RAP-FSAM3掩膜.py`：

1. **Session 生命周期（严格）**：`propagate_memory_masks` 改为 `try/finally`，`close_session` 在所有退出路径（成功 / seed_empty / OOM / 异常）被调用。
2. **CUDA 内存诊断**（无推理改动）：记录 predictor 加载前、start_session 前/后、add_prompt 前、propagate 前、close_session 后的 `memory_allocated`，写入 `info["memory"]`。
3. **逐 sample A7 诊断**：新增 `_merge_sampled_memory_info()` helper，`memory_info["samples"][sample]` 记录每个 sample 的 seed/状态/传播数/内存，汇总 状态 基于全部 sample（任一 OOM → `partial_oom`）。
4. **CSV 逐帧 seed**：`memory_seed_by_stem[stem]` 记录每帧的 seed，`记忆种子帧` 列写 per-frame 值，不再用 merged 单值（消除 XianKeLai1_0000 伪象）。
5. **独立计数**：`propagated_total`（有效传播帧）/ `candidate_total`（进入打分，=propagated）/ `selected_total`（最终采用，Pass-2 计数）分别统计。
6. **OOM 清理**：OOM/异常路径在 close_session 后 `gc.collect()` + `torch.cuda.empty_cache()`——仅由已确认的 session 泄漏证据支撑，非算法修复。

修复后全部 diff 仅触及 `propagate_memory_masks` 与 A7 块；Frozen P6 基线路径（A1s/P6 打分、候选生成、细化链）未改动。

## D. A7 执行表（逐 sample）

修复后复跑 `V01_A7_rerun`（A6 OFF，A7 ON）与 `V11_A6A7_rerun`（A6 ON，A7 ON），与 Phase 14 相同的 21 帧 / GT / 权重 / 阈值 / 分辨率 / seed。完整 CSV：`Phase14_1_a7_diagnostics.csv`。

| Variant | Sample | 状态 | seed 帧 | 帧数 | 传播掩膜数 |
|---|---|---|---|---|---|
| V01-R1 | CaoMei1 | ok | CaoMei1_0025 | 5 | 5 |
| V01-R1 | ChangShouHua2 | ok | ChangShouHua2_0100 | 5 | 1 |
| V01-R1 | DouBanLv1 | ok | DouBanLv1_0000 | 5 | 5 |
| V01-R1 | KongQueZhuYu | ok | KongQueZhuYu_0100 | 5 | 1 |
| V01-R1 | XianKeLai1 | ok | XianKeLai1_0000 | 1 | 1 |
| V11-R1 | CaoMei1 | ok | CaoMei1_0025 | 5 | 5 |
| V11-R1 | ChangShouHua2 | ok | ChangShouHua2_0100 | 5 | 1 |
| V11-R1 | DouBanLv1 | ok | DouBanLv1_0000 | 5 | 5 |
| V11-R1 | KongQueZhuYu | ok | KongQueZhuYu_0100 | 5 | 1 |
| V11-R1 | XianKeLai1 | ok | XianKeLai1_0000 | 1 | 1 |

| 汇总 | V01-R1 | V11-R1 |
|---|---|---|
| `samples_ok` | 5/5 | 5/5 |
| `samples_oom` | 0 | 0 |
| `propagated_total` | 13 | 13 |
| `candidate_total`（进入打分） | 13 | 13 |
| `selected_total`（Pass-2 采用） | 9 | 5 |
| session 泄漏 | 5 started / 5 closed | 5 started / 5 closed |

V01-R1 的 `记忆候选采用=1`（9 帧）：CaoMei1_0000/0025/0050/0100、DouBanLv1_0000/0025/0050/0075/0100。
V11-R1 的 `记忆候选采用=1`（5 帧）：CaoMei1_0000/0100、DouBanLv1_0050/0075/0100。

**内存轨迹（V01-R1，MiB allocated）**：每个 sample 在 `after_add_prompt` 短暂升至 ~8.4-8.6 GiB，`after_close_session` 回落到 predictor 基线 8239 MiB，跨 sample 无累积。这从运行时证据上确认了根因（session 泄漏）并验证了修复（生命周期闭环）。

## E. 修正后指标（21 帧，vs `GT_potted_clean`）

V00 / V10 复用 Phase 14 产物（未被触碰）；V01-R1 / V11-R1 为修复后复跑。

| Variant | F1 mean | F1 median | min | max |
|---|---|---|---|---|
| V00 Control | 0.9835 | 0.9839 | 0.9761 | 0.9891 |
| V10 A6 | 0.9807 | 0.9836 | 0.9278 | 0.9891 |
| V01-R1 A7 | 0.9836 | 0.9850 | 0.9754 | 0.9895 |
| V11-R1 A6+A7 | 0.9807 | 0.9833 | 0.9278 | 0.9891 |

| Delta vs V00 | mean | median | n_nonzero | regressions | improvements |
|---|---|---|---|---|---|
| V10 (A6) | -0.0028 | +0.0000 | 1 | **1**（ChangShouHua2_0075: 0.9856→0.9278, **ΔF1=-0.058**） | 0 |
| V01-R1 (A7) | +0.0001 | +0.0000 | 9 | 0 | 0 |
| V11-R1 (A6+A7) | -0.0027 | +0.0000 | 6 | **1**（ChangShouHua2_0075, ΔF1=-0.058） | 0 |

**A6 解释修正**：Phase 14 称 ChangShouHua2_0075 为 "1 mild regression"。按协议回归阈值（< -0.005），ΔF1 = -0.058 是 **material regression**，不是 mild。A6 的唯一可测效应是这一帧的显著回退。

## F. 修正后析因（mean F1）

```
M00 (Control): 0.9835
M10 (A6):      0.9807
M01 (A7-R1):   0.9836
M11 (A6+A7-R1):0.9807

E_A6  = (M10 + M11 - M00 - M01) / 2 = -0.0028  (material via single regression)
E_A7  = (M01 + M11 - M00 - M10) / 2 = +0.0000  (empirically neutral, valid exposure)
I_AB  = M11 - M10 - M01 + M00          = -0.0001  (no interaction)
```

**结论修正：**
- **A7 在此 21 帧 easy 集上经验中性（E_A7≈0）**，但这是**有效观测**：V01-R1 实际传播了 13 帧、采用了 9 帧，全部 5 个 sample 状态 ok。修正前 "A7 neutral" 建立在 0 传播的失败观测上，证据无效；修正后同一结论建立在真实执行的 A7 传播上。
- A6 与 Phase 14 一致（V10/V11 相同单帧回退），但回退幅度须按 material（ΔF1=-0.058）表述。
- 交互无（I_AB ≈ 0）。

## G. 证据状态与验收门

| Gate | 判据 | 状态 |
|---|---|---|
| G1 | V01-R1 无全局 cuda_oom_fallback | PASS（`状态=ok`，`samples_oom=0`） |
| G2 | V01-R1 有 ≥1 真实传播掩膜 | PASS（`propagated_total=13`） |
| G3 | propagated / scored / selected 独立统计 | PASS（13 / 13 / 9） |
| G4 | 逐 sample A7 状态可获得 | PASS（`samples[]` 10 行诊断） |
| G5 | V11-R1 诊断内部一致 | PASS（5/5 ok，propagated=13 ≥ selected=5，无泄漏 keys） |
| G6 | Phase 14 stale 解释已修正（不重写历史） | PASS（`Phase14_correction_note.md` §A–E, E2） |
| G7 | 全历史回归 71 + 新测试 13 = 84/84 | PASS |
| G8 | Frozen P6 未改动（未重跑、未修改） | PASS（Phase 14 目录 git-clean；V00/V10 产物复用） |

## H. 遗留限制

1. **Easy 集天花板**：21 帧基线 F1 均 ≥0.976，留给 A6/A7 的提升空间极小；A7 的 9 帧采用全部在 ±0.003 内。
2. **单帧 A6 回退**：ChangShouHua2_0075（ΔF1=-0.058）源于共识算法 IoU 阈值边缘删除有效前景，见 Phase 14 §8 Bug#5 后续。
3. **统计功效**：n=21 且分数高度接近，Wilcoxon 功效不足；本结论以均值/中位数 delta 与回归计数表述。
4. **XianKeLai1 只有 1 帧**（该 sample 在 21 帧输入集中仅 1 帧），A7 传播该 sample 只有 1 个目标帧。

## I. 文件

```
阶段十四点一_A7有效性封口/
├── Phase14_1_report.md            (本报告)
├── Phase14_1_manifest.json        (Gates + 变体指标 + 析因)
├── Phase14_1_a7_diagnostics.csv   (逐 sample A7 执行表)
├── Phase14_correction_note.md     (Phase 14 stale 解释修正清单，历史未改写)
├── V01_A7_rerun/                  (修复代码 V01 复跑，含日志/记忆传播.json/提示词选择.csv)
├── V11_A6A7_rerun/                (修复代码 V11 复跑)
└── tests/test_phase14_1_lifecycle.py  (13 tests: session 生命周期 + 逐 sample 合并)
```

## J. 复现性

```
Python / env：  /home/test/biosoft/enter/envs/sam3/bin/python
GPU：            NVIDIA RTX A6000（本复跑 pinned CUDA_VISIBLE_DEVICES=0，≥29 GiB free）
SAM3 检查点：   /data/fj/F2DMAS/sam3/sam3.pt
输入：          阶段十二 .../frame_inputs (21 帧)
GT：            GT_potted_clean
运行命令：      同 Phase 14 `build_cmd`，输出目录改为 *_rerun，仅 A7 生命周期/诊断代码修复
```
