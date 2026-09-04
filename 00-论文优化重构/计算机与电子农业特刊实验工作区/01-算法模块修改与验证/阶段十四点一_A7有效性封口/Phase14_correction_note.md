# Phase 14 Correction Note — A7 Evidence Inconsistencies

日期：2026-09-04
Freeze SHA：`f9f091d82122148ef011a8063d4949fbccc94828`（Phase 14 commit，worktree clean）
原文产物（未修改）：`阶段十四_A6A7真实数据析因消融/` 下 V00/V01/V10/V11、Phase14_report.md、Phase14_protocol.md

本文件是 **correction note**，不改写原报告，只指出原报告与已提交产物证据不一致之处，以及 Phase 14.1 将如何验证/修正。

---

## A. V01 (A7-only) 不是有效的 A7 观测

原报告 Phase14_report.md §5.2：

> A7 propagated: 0/21 frames · **A7 without A6 is completely neutral** (identical to V00)

实际产物 `V01_A7/日志/记忆传播.json`：

```json
{ "汇总": { "状态": "cuda_oom_fallback", "记忆候选帧数": 0, "种子帧": "" } }
```

运行日志 `logs/V01_run.log` STDERR：出现 **5 次 "started new session"，0 次 "removed session" / "propagation ended"**。
`V01_A7/提示词选择.csv`：全部 21 帧 `记忆候选采用=0`。

**结论：**
- V01 状态 = `cuda_oom_fallback`，记忆候选帧数 = 0。
- V01 **NOT** a valid A7-effect observation；它是一个 **A7 运行时失败/回退观测**。
- "V01 == V00" 不能被解释为 "A7 effect = 0"。

> 原则：A module that falls back to baseline has not demonstrated a neutral effect.
> Failure to execute ≠ zero treatment effect.

## B. V11 (A6+A7) 的汇总只反映最后一个 sample

实际产物 `V11_A6+A7/日志/记忆传播.json`：

```json
{ "汇总": { "状态": "ok", "记忆候选帧数": 8, "种子帧": "XianKeLai1_0000", "样本数": 5 } }
```

生产代码 `propagate_memory_masks` 逐 sample 调用后，主程序 L2921-2922：

```python
for k, v in sub_info.items():
    merged_memory_info[k] = v
```

每个 sample 的 sub_info 会**覆盖**同名 key，因此汇总只保留**最后一个** sample（按字母序 XianKeLai1）的状态/种子帧。其余 4 个 sample（CaoMei1、ChangShouHua2、DouBanLv1、KongQueZhuYu）的真实状态**丢失**。

`提示词选择.csv` 中全部 21 帧 `记忆种子帧=XianKeLai1_0000` 是合并覆写的**报告伪象（reporting artifact）**，不是跨样本传播。当前代码（L2899-2909）确实对每个 sample 独立选 seed（sample 内 A1s 最高分）。

**影响：**
- "状态=ok" 只能对最后一个 sample 成立；不能推断 5 个 sample 全部成功。
- Phase 14.1 将改为 `memory_info["samples"][sample] = {...}` 逐 sample 记录 + 汇总聚合。

## C. propagated / scored / selected 三个口径被混用

原报告 §5.3 写 "A7 propagated: 2/21，A7 selected: 2/21"，而产物 `记忆传播.json` 的 `记忆候选帧数=8`。

三个口径必须独立统计：
- **propagated**：SAM3 video predictor 为帧产出了有效 mask（V11 = 8）。
- **scored**：有效 A7 mask 进入 Pass-2 打分（= propagated，当前实现所有有效 mask 均进入打分）。
- **selected**：A7 候选赢得最终变体选择（`记忆候选采用=1`，V11 = 2 帧：CaoMei1_0000、CaoMei1_0100）。

原报告把 selected (2) 当 propagated 使用，口径不准确。

## D. "A7 depends on A6" 是 stale explanation

原报告 §5.2 与 §9 Known Limitations 1 声称 A7 需要 A6 共识掩膜作为 seed。

对当前生产代码的审计：
- A6 OFF 时 `base_for_memory = selected_by_stem`（L2874-2876），seed 由 **sample 内** A1s 最高分选择（L2908-2909），不存在跨 sample global-best fallback。
- 因此 **A7 没有结构上的 A6 依赖**（algorithmic dependency 不成立）。
- V01（A6 OFF）失败的真实原因是 **运行时 OOM**（runtime/resource failure），不是算法依赖。

必须区分：
```
algorithmic dependency   (not present in current code)
vs
runtime/resource failure (present: V01 = cuda_oom_fallback)
```

## E. OOM 处理的证据缺口

生产代码 `propagate_memory_masks` 的 session 生命周期：
- 成功路径调用 `close_session`（L505）；`seed_empty` 路径也调用（L482）。
- `torch.cuda.OutOfMemoryError`（L508）和通用 `Exception`（L511）路径**直接 return，不关闭 session**。

V01 STDERR 中 5 个 session 均无 "removed session"，印证 session 泄漏。后续 4 个 sample 会在已泄漏的 GPU state 上继续建 session，进一步恶化内存。

Phase 14.1 将修复为严格 `try/finally` 保证 close_session 在所有路径被调用，并增加逐 sample 的 CUDA memory 诊断。

---

## E2. OOM 根因已实证（2026-09-04 诊断性 V01 复跑）

修复后的诊断性 V01 复跑（`V01_A7_rerun`）在相同 21 帧 / 权重 / 阈值 / 分辨率 / seed 上成功：

- **Session 生命周期**：SAM3 video predictor STDERR 显示 **5 次 "started new session" + 5 次 "removed session"**（原 V01：5 次 started，0 次 removed）。会话泄漏已消除。
- **逐 sample 状态**：`samples_ok=5`，全部 `状态=ok`（原：汇总 `cuda_oom_fallback`）。
- **记忆传播**：`propagated_total=13`（原 V01：0），`selected_total=9`（原 V01：0）。
- **CUDA 内存轨迹**：每 sample 的 `after_close_session` 内存回落至预测器基线（~8.2 GiB），不再随 sample 数累积。
- **结论**：V01 的 OOM 根因 = **session 泄漏导致的 GPU 内存累积**（修复前 OOM/异常路径不调用 close_session）。A7 传播本身在该数据集上是可行的（13 帧有效传播），**不是 seed-box 过大导致的固有失败**。

---

## Phase 14.1 Corrected

| 判定 | 状态 |
|---|---|
| Phase14 原 A7 结论（A7 effect≈0/neutral）有效 | 结论定性保留，但证据基础修正：V01 从"失败观测（0 传播）"变为"有效观测（13 传播 / 9 采用）" |
| Phase14 原 V01 描述（A7 needs A6 seed） | **错误**，见 §D |
| Phase14 原 propagated=2 口径 | 错误，正确口径见 §C（V11 propagated=8） |
| A6 ChangShouHua2_0075 regression | 非 mild：ΔF1 = -0.058，material regression（按协议阈值 < -0.005） |
| Phase14.1 corrected factorial | 见 `Phase14_1_report.md`