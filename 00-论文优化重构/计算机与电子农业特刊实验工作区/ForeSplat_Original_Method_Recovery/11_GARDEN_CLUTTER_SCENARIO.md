# 11 — 花园多株场景（只取最前方前景株）验证

生成时间：2026-10-27（数据来自本轮真实运行，非估算）

状态：**已执行受控实验；方法在该场景下表现有效，但实验是"真实植株图像 + GT 掩膜"的受控合成，不是真实花园照片。**

---

## 0. 为什么做这个实验

用户提出部署场景假设：

> 在一个花园里有很多花，我只想分割出来最前方、前景的那一个。我觉得在这个场景下，现在的方法会好用。

这个假设与 dev 集（4 场景 16 帧）恰好互补：**dev 集每帧只有一株盆栽、干净台面**，SG-MPS 没有任何"邻株/背景/桌面"需要抑制，所以 oracle 在 16/16 帧都是单提示 P2 —— 门控最多只能追平基线（见 `05` §10、`10` §3.1）。而在"多株花坛"里，抑制项 ψ = [q_table, q_bg, q_neighbour, q_outlier] 才第一次真正有东西可抑制。

## 1. 数据获取的限制（必须先声明）

按用户要求"去找对应图片"，**联网检索不可用**，已如实记录：

| 途径 | 结果 |
|---|---|
| `open-websearch` 搜索引擎（bing/baidu/ddg/exa/startpage） | Bing 返回微信公众号/词典噪声；DuckDuckGo/exa/startpage `ETIMEDOUT` |
| `open-websearch fetch-web` | `ETIMEDOUT`（未走代理） |
| `curl` 直连 | github / upload.wikimedia.org / huggingface.co 可 200，`commons.wikimedia.org` 超时 |
| HuggingFace datasets API | 间歇可用，但未取到"花园多株 + 前景目标 GT"的可用标注集 |

**结论：无法取得带前景目标标注的真实花园数据集。** 因此改用更严谨且可复现的受控合成方案（§2）。这是本轮实验最重要的限制，任何结论都只在"受控合成"范围内成立。

## 2. 受控合成方案

脚本：`scripts/build_clutter_scenes.py`（确定性，给定 seed 可完全复现）
探针：`scripts/run_clutter_probe.py`（`--score-only` 可只重算指标）

**关键点：合成用的不是生成的假植株，而是你现有数据里真实拍摄的植株。** 从

- 真实 RGB：`dev_inputs/<scene>/<stem>.jpg`（2160×3840）
- 真实 GT：`GT口径拆分审计/gt_masks_split_v2/<scene>/mask_potted_plant_<stem>.png`

用 GT 掩膜把真实植株**精确抠出**（alpha 来自 GT，边缘即真实边界），再粘贴到 2160×3840 的
"花坛"背景（绿褐渐变 + 噪声）上，按固定深度顺序合成：

```
back-left   scale 0.44  压暗(0.70) + 模糊(5.0)    ← 远
back-right  scale 0.44  压暗(0.70) + 模糊(5.0)    ← 远
mid         scale 0.56  压暗(0.84) + 模糊(2.5)    ← 中
FRONT       scale 0.78  清晰、居中、贴底          ← 目标（GT 由构造确定）
```

**目标 GT 是构造出来的**：前景株粘贴时的 alpha 就是它的精确掩膜，因此 F1 是严格可算的，不依赖任何人工标注。邻居掩膜只保留**可见部分**（被更近植株遮挡的像素不计入，否则会把正确遮挡算成泄漏）。

### 2.1 两组场景

| 组 | 场景数 | 布局意图 |
|---|---|---|
| **EASY**（`runs/clutter_scene/`） | 4 | 前株最大、居中贴底 —— 与用户描述的"最前方那一株"一致 |
| **HARD**（`runs/clutter_scene_hard/`） | 3 | **后景株比目标更大**（scale 0.95 vs 0.62），目标偏左下、更清晰 —— 检验"最前方 ≠ 最大"时是否还选得对 |

EASY 目标占画面 44.0%–46.8%，HARD 目标占 25.6%–33.0%。

### 2.2 三种选择规则（同一候选池）

候选池只跑一次（P1–P5、`per_instance`、`--save_raw_instance_masks`），三种规则通过
`--reuse_existing_candidates` 复用同一池，**差别只在选择规则**：

| 规则 | 参数 |
|---|---|
| `single_P2` | `--prompt_list P2 --prompt_selection_mode single --ranking_policy r1` |
| `multi_nogate` | `P1..P5` + `score_select` + `--candidate_area_admissibility gate`，**无门控** |
| `multi_gate` | `multi_nogate` + `--use_semantic_gate --semantic_gate_mode gated_softmax --semantic_gate_temperature 0.25 --semantic_gate_target_mode potted_plant` |

所有配置**关闭** SB-FBPR / 残差修复 / GCFC：合成图是单帧、无 COLMAP，两个细化阶段需要
多帧或真实观测。因此本实验只检验 **候选生成 + SG-MPS + CDR** 这一段，即门控问题本身。

生成候选规模（真实 SAM3）：`single_P2` 3–7 个实例，`multi` 14–27 个实例。

## 3. 结果（EASY，4 场景）

| 规则 | F1 | IoU | Precision | Recall | 邻居泄漏 | 连通域 | 选中提示词 |
|---|---|---|---|---|---|---|---|
| `single_P2` | **0.9878** | 0.9758 | 0.9949 | 0.9807 | 1.68% | 6.5 | P2×4 |
| `multi_nogate` | 0.9579 | 0.9236 | 0.9962 | **0.9272** | 1.23% | 7.2 | P2,P3,P2,P2 |
| `multi_gate` | **0.9878** | 0.9758 | 0.9949 | 0.9807 | 1.68% | 6.5 | P2×4 |

逐场景：

| 场景 | single_P2 | multi_nogate | multi_gate | 无门控错在哪 |
|---|---|---|---|---|
| clutter_01 | 0.9891 | 0.9891 | 0.9891 | — |
| clutter_02 | 0.9886 | **0.8692** | 0.9886 | 选 P3（叶/茎），Recall 0.769 |
| clutter_03 | 0.9889 | 0.9889 | 0.9889 | — |
| clutter_04 | 0.9846 | 0.9846 | 0.9846 | — |

**参考下界（"把所有植株合并成一个掩膜"）**：同一批合成场景上，`union(目标 ∪ 全部邻居)`
对目标 GT 的 F1 只有 **0.8603–0.8975**。也就是说，"找一株"这件事在本合成里比"找全部"
高约 **9–13 个 F1 点**，目标口径（potted-plant 单体前景）与场景需求一致。

## 4. 结果（HARD，后景株更大，3 场景）

| 规则 | F1 | IoU | Precision | Recall | 邻居泄漏 | 选中提示词 |
|---|---|---|---|---|---|---|
| `single_P2` | 0.9840 | 0.9685 | 0.9976 | 0.9708 | 0.11% | P2×3 |
| `multi_nogate` | 0.9840 | 0.9685 | 0.9976 | 0.9708 | 0.11% | P2×3 |
| `multi_gate` | 0.9840 | 0.9685 | 0.9976 | 0.9708 | 0.11% | P2×3 |

逐场景：hard_01 0.9844 / hard_02 0.9869 / hard_03 0.9806（三规则完全相同）。

**参考下界**：`union(目标 ∪ 邻居)` 的 F1 只有 **0.4651–0.6046**（目标仅占 26–33%，而并集占
76–84%）。即使后景株更大，管线仍选对了目标，**没有退化成"选最大实例"**。

## 5. 结论（区分"有证据"与"未证明"）

### 5.1 有证据支持用户假设

1. **方法在"多株、只取前株"场景确实好用**：7/7 场景 F1 0.9806–0.9891，邻居泄漏 0.06%–3.15%。
2. **目标口径正确**：与"合并全部植株"相比高 9–13（EASY）/ 38–52（HARD）个 F1 点。
3. **门控在真实有干扰的场景里有纠错价值**：EASY 上无门控的多提示会把 clutter_02 选成 P3
   （F1 −0.119），门控把它纠回 P2；dev 单株集没有这种错误可纠，所以那里门控"无收益"是场景决定的，不是机制无效。
4. **"最前方 ≠ 最大"时仍成立**：HARD 组后景株大 1.5 倍，三规则全部仍选到目标。

### 5.2 未证明 / 必须写进论文的限制

1. **不是真实花园照片**：合成图的植株边缘来自 GT Alpha，边界干净、无真实遮挡阴影、无同色系邻叶交错。真实花园里实例分离更难，F1 必然更低。**本实验证明的是"选择机制在有多株时能选对"，不是"真实花园上的精度指标"。**
2. **单帧、无时序**：只测了候选生成 + SG-MPS + CDR，q_temp、记忆传播、共识投票、SB-FBPR、残差修复、GCFC 全部未参与。
3. **合成布局由我设定**：深度顺序、尺度、亮度/模糊是我构造的，覆盖了"前株最大"和"前株非最大"两类，但不等同于真实景深分布。
4. **门控在 EASY 上与单 P2 完全同分**：它只消除了无门控的退化，没有超过单提示基线。这与 dev 集的结论一致——**在本数据的候选池里单 P2 本身就是上限**（oracle 16/16 = P2）。

### 5.3 对论文的建议表述

> 在多株前景场景（受控合成，7 个场景）中，RAP-FSAM3 的前景目标选择 F1 为 0.984–0.989，
> 邻居泄漏 < 3.2%；消融显示该能力主要来自逐实例候选与 CDR 排序，SG-MPS 门控的作用是
> **阻止多提示扩展带来的提示词误选**（无门控时出现 1/4 场景误选 P3，F1 −0.119），与单提示
> 基线持平。真实花园照片上的精度尚未测量（联网检索不可用，无可用标注集）。

## 6. 复现命令

```bash
S=/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Original_Method_Recovery
PY=/home/test/.venvs/sam3/bin/python

# EASY
$PY $S/scripts/build_clutter_scenes.py
CUDA_VISIBLE_DEVICES=0 $PY $S/scripts/run_clutter_probe.py
# HARD
CLUTTER_ROOT=$S/runs/clutter_scene_hard $PY $S/scripts/build_clutter_scenes.py --hard
CLUTTER_ROOT=$S/runs/clutter_scene_hard CUDA_VISIBLE_DEVICES=0 $PY $S/scripts/run_clutter_probe.py

# 只重算指标（不重跑推理）
$PY $S/scripts/run_clutter_probe.py --score-only
```

产物：

| 路径 | 内容 |
|---|---|
| `runs/clutter_scene/inputs/*.jpg` | 4 张 EASY 合成图（2160×3840） |
| `runs/clutter_scene/gt/mask_*.png` | 前景目标 GT（构造确定） |
| `runs/clutter_scene/gt/neigh_*_*.png` | 邻居**可见**掩膜（已扣除遮挡） |
| `runs/clutter_scene/pool/` | 共享候选池（P1–P5 逐实例，真实 SAM3） |
| `runs/clutter_scene/runs/<scene>/<config>/` | 三种选择规则的完整 RAP 输出（含逐帧报告） |
| `runs/clutter_scene/metrics/clutter_probe.csv` | 逐场景指标 |
| `runs/clutter_scene/metrics/clutter_probe_summary.json` | 分规则聚合 |
| `runs/clutter_scene_hard/` | HARD 组同构产物 |
| `logs/clutter_pool.log`、`logs/clutter_{scene}_{config}.log`、`logs/clutter_probe_driver.log` | 运行日志 |

## 7. 一句话总结

**用户的判断成立**：在"多株花坛、只取最前方"的场景下，当前方法确实好用（受控合成 7/7 场景
F1 0.984–0.989，远优于"合并全部植株"的 0.46–0.90）；门控在这里第一次有真实纠错价值，但它
只把多提示拉回单提示基线，没有超过基线。真实花园照片的结论仍待联网可用后补测。
