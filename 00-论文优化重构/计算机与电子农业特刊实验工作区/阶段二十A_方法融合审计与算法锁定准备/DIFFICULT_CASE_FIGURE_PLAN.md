# 困难场景图件规划 / DIFFICULT_CASE_FIGURE_PLAN

**阶段：** Phase 20A
**目标：** 为新版论文准备 D1/D3/D6/D11/D12 真实困难场景展示图
**唯一证据源：** Phase 16 冻结清单 + Phase 17A/17B/17C 归档

---

## 0. 最重要的结论（必须先读）

> **D3、D6、D12 在冻结数据中各有 0 帧。**
> 58 帧挑战集中，53 帧携带**完全相同**的一对标签 `D1 + D11`，
> 另 5 帧携带 `D11 + D14`。
> "D1/D3/D6/D11/D12 五个可比难度类别"这一表述**不被冻结数据支持**。

因此本阶段**不能**给出"每类各 1–3 张真实代表图"的方案。
下文给出的是**诚实的可交付版本**：能给的给全，不能给的明确标为 BLOCKED 并给出最低成本补齐路径。

---

## 1. 难度标签的真实来源与口径

### 1.1 唯一产生逐帧标签的启发式

`阶段十六_HardCase_GT构建与锁定/脚本/phase16_split.py` L70-72、L149-153：

```python
D1  = "D1_neighbor_plant_adhesion"
D11 = "D11_cluttered_background"
D14 = "D14_other"
...
        if t["green_pct"] >= 20:
            prim, sec = D1, D11
        else:
            prim, sec = D11, D14
```

支撑特征（同文件 L86-98）：

```python
mask_g = (hsv[...,0] >= 30) & (hsv[...,0] <= 90) & (hsv[...,1] >= 30) & (hsv[...,2] >= 30)
green_ratio = mask_g.sum() / (h * w)
score = green_ratio * 100 * n_peaks
```

**性质判定：**

| 维度 | 结论 |
|---|---|
| 是否人工视觉核查 | **否**。DEV 26 帧是**用户手动选帧**（`selection_reason = user_manual_selection (multi-plant visible in background)`），但标签字段仍写 D1+D11 |
| 是否互斥类别 | **否**。D1/D11/D14 是**同一个 HSV 绿色比例阈值的两分支**，不是三种独立失效模式 |
| 是否可从绿色像素比例直接断定邻株交叠 | **否**。任务书已明确禁止此推断，审计确认该风险真实存在 |
| 是否已人工确认 | **否**。`Phase16_report.md` §6 自述："heuristic 标签仅用于标注引导，**不参与任何评分或过滤**" |

### 1.2 D3/D6/D12 的来源核查

```
$ grep -rl "D6_small_target\|D12_low_contrast\|D3_fg_bg" <workspace>
→ 唯一命中：阶段十六_HardCase_GT构建与锁定/Phase16_report.md
```

- D3 / D6 / D12 **仅以散文形式**出现在 `Phase16_report.md` §6 的表格中；
- 该表本身与冻结清单**不可调和**：表中 TEST 列 19+8+1 = 28 ≠ 32（清单为 32）；
  表中 D1 合计 16 ≠ 实际 53；
- 该表所描述的启发式依据（熵 / ExG 分离 / 绿占比 / 动态范围）对应 **Phase 15 的
  5 特征 `scout_difficulty.py`**，而该脚本**不在 Phase 16 脚本树中**（报告自身将其记为已知问题）。

**判定：D3/D6/D12 = `NOT VERIFIED`，无帧可选，其启发式不可从本仓复现。**

---

## 2. 各难度类别的真实帧清单与可用性

| 类别 | 冻结清单中的帧数 | 有 GT | 可用作论文定性图 | 可用作定量证据 |
|---|---|---|---|---|
| **D1** 邻株交叠（primary） | **53**（DEV 26 + TEST 27） | 10（GT10） | ✅ 是 | 仅 GT10 十帧 |
| **D11** 杂乱背景（primary） | **5**（green_pct<20 分支） | 0 | ✅ 是（须标注"无 GT"） | ❌ 否 |
| **D11**（secondary） | 58（全帧） | 10 | ✅ 是 | ❌ |
| **D14** other（secondary） | 5 | 0 | ✅ 是 | ❌ |
| **D3** 前景背景混淆 | **0** | — | ❌ **BLOCKED** | ❌ |
| **D6** 小目标 | **0** | — | ❌ **BLOCKED** | ❌ |
| **D12** 低对比 | **0** | — | ❌ **BLOCKED** | ❌ |

### 2.1 D1 / D11 primary 帧清单（27 TEST + 5）

**TEST32（D1 primary，27 帧，GT 均无）：**
CaoMei2 0007/0008/0009/0146 · ChangShouHua3 0189/0190/0191/0200 ·
DouBanLv3 0211/0220/0221/0222 · HongZhang 0134/0200/0202/0229 ·
WangWenCao2 0014/0015/0016/0017 · WanNianQing1 0223/0224/0225/0227 ·
XianKeLai3 0133/0141/0204

**D11 primary（5 帧，GT 均无）：**
XianKeLai3 0117 · XiangPiShu2 0133/0134/0135/0136

**DEV26（D1 primary，全为用户手选，10 帧有 GT）：**
BaiZhang 0033–0045（13 帧连续）· DouBanLv2 0029–0041（13 帧连续）

---

## 3. 推荐图件方案（诚实可交付版）

### 3.1 Panel A —— "候选排序失效"（**推荐，最高证据强度**）

**内容**：DouBanLv2 0039 / 0040 / 0041 三帧，展示 SAM3 已生成 F1≈0.987 的正确候选，
但原评分选中面积比 0.03–0.05 的高对比碎片（R1 修复后选中正确实例）。

| 属性 | 值 |
|---|---|
| 逐帧 GT | **有**（`GT_potted_clean_challenge`） |
| 定量证据 | **有**：Phase 17B/17C 逐帧 oracle 与 policy 对比 |
| 标签 | `D1 + D11`（**不是**独立的 D1/D3/D6/D12 类别） |
| 现成图件 | `visualizations/instance_audit_DouBanLv2_0039.jpg`（RGB \| GT \| 每个原始 P6 实例，标注 instance id / F1 / SAM score / SELECTED / ORACLE） |
| 现成图件 | `visualizations/temporal_strip_DouBanLv2_0037-0041.jpg`（注意实际文件名带样本名前缀） |
| 现成图件 | `visualizations/map_gt_*.jpg` / `map_pred_*.jpg`（RGB \| GT \| pred 三联，10 帧各一） |
| 图注建议 | "A single text prompt produces several SAM3 instances; geometric-plausibility features favour the large whole-plant instance, while a normalised foreground–background contrast term favours a small high-contrast fragment." |

**⚠️ 陷阱警告：** 不要把 DouBanLv2 0039–0041 的 **R0 掩膜**当作"困难场景下的输出"展示 ——
那是**错误选择**的产物（area ratio 0.03–0.05，几乎无真阳性）。
展示"输出差"时应同时展示 oracle 候选，否则读者会误读为"方法本身在该场景失败"。

### 3.2 Panel B —— "多株共存背景"（推荐，有 GT，定性可靠）

| 帧 | GT | 用途 |
|---|---|---|
| BaiZhang 0033–0037（5 帧） | ✅ 有 | R1 前后**选择不变**，F1 0.9536–0.9623 稳定。证明 R1 不破坏已正确帧 |
| DouBanLv2 0037 / 0038 | ✅ 有 | 同上，F1 0.9855 / 0.9898 |

现成图件：`阶段十七A…/07_gallery/trackA/trackA_BaiZhang_00XX.jpg`

### 3.3 Panel C —— "杂乱/稀疏背景"（D11 primary，**无 GT，仅定性**）

| 帧 | 序列 | 说明 |
|---|---|---|
| XiangPiShu2 0133 / 0134 / 0136 | XiangPiShu2 | `green_pct < 20` 分支 |
| XianKeLai3 0117 | XianKeLai3 | 同分支 |

现成图件：`阶段十七A…/07_gallery/trackB/trackB_<sample>_<frame>.jpg`
标注要求：**必须注明"行为审计帧，无人工标注，仅作定性展示"**。

### 3.4 Panel D —— "R1 行为变更的两帧"（CaoMei2 0146 / WanNianQing1 0227）

| 属性 | 值 |
|---|---|
| GT | **无** |
| 事实 | R1 使选择从 `…_04`→`…_03`（CaoMei2 0146）、`…_00`→`…_01`（WanNianQing1 0227） |
| 未观测到 | 空掩膜、结构性失败 |
| 证据文件 | `阶段十七C…/Phase17C_test32_safety.csv` |
| 允许的用途 | **只能**用于"部署前行为审计（churn 2/32 = 6.2%）"的叙述 |
| **禁止的用途** | ❌ 不得宣称这两帧"改进"或"变差"；❌ 不得给它们编造 F1 |

---

## 4. 无 "R0 掩膜 / R1 掩膜" 文件这一事实

Phase 17C 是 **zero-inference** 重排序，从未写出新掩膜。
```
find -iname "*r1*mask*" -o -iname "*R0*mask*"  →  0 命中
```
R1 的反事实选择**只以 CSV 数值形式存在**（`Phase17C_test32_safety.csv`、
`Phase17C_policy_comparison.csv`、`Phase17C_ranking_benchmark.csv`）。

**影响：** 若论文需要展示"R1 修复前 / 后的掩膜对比图"，必须：
- 方案 1：用 `Phase17C_ranking_benchmark.csv` 中的 `raw_instance_*` 候选掩膜（P00_Control 的 `候选掩膜/` 目录）
  离线重放 R1 选出掩膜 —— **需要一次离线掩膜选择重算，不需 GPU**；
- 方案 2：跑一次 V1（P6+R1）正式运行产出真实最终掩膜 —— **需要 GPU**。

Phase 20A 不实施二者，仅记录。

---

## 5. 推荐图片来源（最安全）

`阶段十六_HardCase_GT构建与锁定/annotation_package/HARD-<sample>-<frame>/` —— **58 个目录**，每个含：
- `target.jpg`（3840×2160 原始 RGB）
- `context_prev.jpg` / `context_next.jpg`（时序上下文）
- `instruction_card.md`

Phase 16 §9 已验证该包**无模型产物渗漏**（无 V10/V11/mask 文件）。
**投稿图件应从 annotation_package 取源图**，而非从任何模型输出目录取。

---

## 6. 证据限制声明（必须随图写入论文 limitations）

1. **D 类标签不是人工视觉核查结论**，而是 HSV 绿色比例阈值的二分支；53/58 帧标签完全相同。
2. **D3 / D6 / D12 无帧**；不得在论文中声称已覆盖这三类困难场景。
3. **无 GT 的帧只能作定性展示**，不得计算或暗示 F1 / mIoU 等任何准确率。
4. **GT10 是开发集（DEV），不是独立测试集**；其上的任何提升都是开发集内结果。
5. **TEST32 无 GT**，只有行为审计（churn 2/32），无准确率。
6. 数据为**室内盆栽场景**；不得宣称为真实温室或田间验证。
7. DouBanLv2 0039–0041 的失败是**候选排序**问题，不是检测/生成问题；
   若只展示失败输出而不展示 oracle 候选，会误导读者。

---

## 7. 最低成本补齐 D3 / D6 / D12 的路径（供 Phase 20B 决策，本阶段不执行）

| 类别 | 补齐难度 | 说明 |
|---|---|---|
| **D6 小目标** | **低** | 视觉上极易筛选，且 TEST 序列中大概率存在。建议：每序列各挑 1 帧小目标帧 → 人工标注 potted_clean → 8 帧，成本约等于 GT10 的一半 |
| **D3 前景背景混淆** | **中** | 需人工判别"前景与背景混淆"的具体形态；可复用 DouBanLv2 0039–0041（前景叶片与背景绿色物体混淆）作为 D3 的**近似**，但必须重新人工标注类别，且当前这三帧已标为 D1 |
| **D12 低对比** | **中** | 需人工判别低对比；Phase 16 `green_pct<20` 分支的 5 帧（XiangPiShu2 0133–0136 / XianKeLai3 0117）**可能**部分满足，但未经判别，不得直接宣称 |

**约束：** 任何补齐都必须
(a) 由人工视觉核查产生标签，不能沿用启发式；
(b) 记录 `label_origin = human_verified`；
(c) 保持 DEV/TEST 按**序列**隔离，不得为补齐某一类而跨 split 取帧。

---

## 8. `DIFFICULT_CASE_ASSET_MANIFEST.csv` 字段说明

| 字段 | 取值来源 |
|---|---|
| `sample_id` / `frame_id` | 冻结清单 |
| `source_image` | `annotation_package/HARD-*/target.jpg`（**优先**）或 `01-raw_frames/<sample>/<frame>.jpg` |
| `difficulty_label` | 冻结清单原值，**原样复制，不加工** |
| `label_origin` | `user_manual_frame_selection + heuristic_green_ratio`（DEV26）/ `heuristic_green_ratio`（TEST32） |
| `manually_verified` | **全部 `false`** —— 无任何人工类别确认记录 |
| `gt_available` / `gt_path` | 磁盘实测 |
| `r0_mask_path` | P00_Control / U00_Control 的 `选择后掩膜/mask_*.png` |
| `r1_mask_path` | **全部 `N/A`** —— 17C 为 zero-inference，无此文件 |
| `final_mask_path` | P00/U00 的 `最终掩膜/mask_*.png` |
| `recommended_figure_panel` | A / B / C / D |
| `evidence_limitations` | 逐行标注 |

完整逐帧清单见同目录 `DIFFICULT_CASE_ASSET_MANIFEST.csv`（58 行）。