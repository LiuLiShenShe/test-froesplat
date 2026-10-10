# 阶段二十A — 方法融合审计与算法锁定准备

**ForeSplat: Vision-Foundation-Model-Guided Foreground-Object 2D Gaussian Splatting for Low-Cost 3D Plant Phenotyping**

**基线 commit：** `6df0c3df`（Phase 19A）｜**分支：** `main`｜**日期：** 2026-10-10

---

## 本阶段做了什么

**不是**开发新算法，**而是**核实原论文 RAP-FSAM3、最新 R1 排序算法、已有细化模块、
Foreground-object 2DGS 与 M2M3-GO 之间的真实关系，并判断哪些既有实验可以继续使用。

**未修改任何生产代码、历史数据或论文正文。**

---

## 交付物

| 文件 | 内容 | 结论强度 |
|---|---|---|
| [`README.md`](README.md) | 本文件 | — |
| [`PHASE20A_FINAL_REPORT.md`](PHASE20A_FINAL_REPORT.md) | **总报告**：12 问直答、验收门、最终判定块 | 主入口 |
| [`METHOD_CONFIGURATION_AUDIT.md`](METHOD_CONFIGURATION_AUDIT.md) | 原稿 14 个模块 ↔ 代码入口 ↔ 历史实验状态逐条对照；论文–代码 5 处结构性差异 | 高 |
| [`METHOD_VERSION_MATRIX.csv`](METHOD_VERSION_MATRIX.csv) | **27 行**端到端组件矩阵（M1–M24，其中 M4 拆为 a/b/c/d 四条选择分支），逐项标注 `VERIFIED_ACTIVE` / `OPTIONAL` / `UNVERIFIED` | 高 |
| [`R1_CALL_PATH_AUDIT.md`](R1_CALL_PATH_AUDIT.md) | R1 在全部选择分支的实际作用；18 处 `total_score` + 12 处 `primary_score` 使用点穷举 | **决定性** |
| [`R1_FORMULA_VERIFICATION.md`](R1_FORMULA_VERIFICATION.md) | R1 公式逐行复核 + 182 条归档候选数值重算 | **PROVEN** |
| [`PROMPT_AND_RUN_PROVENANCE_MATRIX.csv`](PROMPT_AND_RUN_PROVENANCE_MATRIX.csv) | 全仓 **78 份** `参数.json` 逐份溯源（缺失值写 `UNKNOWN`） | 高 |
| [`REFINEMENT_CHAIN_AUDIT.md`](REFINEMENT_CHAIN_AUDIT.md) | SPNP → 残差修复 → GCFC 顺序/开关/守卫/fallback；21/21 帧逐字节验证 | **决定性** |
| [`METHOD_SELECTION_DECISION.md`](METHOD_SELECTION_DECISION.md) | 方案 A vs B 对比、推荐、函数级最小改动建议、风险 | 决策 |
| [`PLANT_POT_SCALE_CONSISTENCY.md`](PLANT_POT_SCALE_CONSISTENCY.md) | GT 类别定义、`potted_clean` 逐像素验证、P6 盆覆盖实测、尺度恢复 | 高 |
| [`RECONSTRUCTION_EVIDENCE_REUSE_MATRIX.csv`](RECONSTRUCTION_EVIDENCE_REUSE_MATRIX.csv) | Table 1–8 逐元素分类（`REUSE_DIRECTLY` / `REUSE_AS_LEGACY_BASELINE` / `TARGETED_RERUN_REQUIRED` / `UNVERIFIED`） | 高 |
| [`RECONSTRUCTION_EVIDENCE_REUSE_NOTES.md`](RECONSTRUCTION_EVIDENCE_REUSE_NOTES.md) | 上述矩阵的说明性详版 | 辅助 |
| [`DIFFICULT_CASE_FIGURE_PLAN.md`](DIFFICULT_CASE_FIGURE_PLAN.md) | D1/D3/D6/D11/D12 图件规划与证据限制 | 高 |
| [`DIFFICULT_CASE_ASSET_MANIFEST.csv`](DIFFICULT_CASE_ASSET_MANIFEST.csv) | **58 行**逐帧资产清单（源图/GT/R0掩膜/最终掩膜/面板/限制），全部路径经核实在盘 | 高 |
| [`FOLLOWUP_EXPERIMENT_PLAN.md`](FOLLOWUP_EXPERIMENT_PLAN.md) | V0–V3 链路对照 + 独立验证计划（只设计，不执行） | 计划 |
| [`FINAL_RAP_FSAM3_SPEC_DRAFT.md`](FINAL_RAP_FSAM3_SPEC_DRAFT.md) | 最终算法规格草稿，状态 **`DRAFT`**（附升级为 `LOCKED_CANDIDATE` 的 5 项前置条件） | 草稿 |
| [`P0_BLOCKERS_AND_NEXT_ACTIONS.md`](P0_BLOCKERS_AND_NEXT_ACTIONS.md) | 5 项 P0、9 项 P1、5 项 P2 阻断项 + 9 步行动清单 | 行动 |

---

## 五项最重要的发现

1. **R1 只在一条分支生效。** `single` + `per_instance` 下 R1 生效（`select_mask()` L1725）；
   `score_select`（L1688）、`weighted_fusion`（L1699）、A1s/A6/A7 变体选择（L3162）**全部使用旧 `total_score`**。

2. **R1 与所有细化模块零联合运行证据。** 78 份配置中，
   `single` + `use_spnp_refinement` = **0 例**；显式 `ranking_policy` + refinement = **0 例**。
   历史上开启 refinement 的 27 份运行 **100% 走 `score_select`**，即 R1 被绕过的分支。

3. **论文 Table 5 有两个数字全仓无出处**（`Boundary-F1=0.4543`、`leakage=0.001935`），
   各行还混用逐帧均值与样本等权均值；Table 3/4/5 的 F1 用 5 种 GT 变体**均无法复现**。

4. **花盆直径在全仓不存在**，尺度恢复无实现代码；且 2DGS 使用的掩膜目录无 pot 标注，
   与"用盆径恢复尺度"的方法叙述不闭合。

5. **Phase 19A 两处结论被本审计推翻**：`P6_raw_baseline` 实为 **P6**（非 P2，两份独立参数快照）；
   GCFC 在该基线上**未生效**（21/21 帧 refinement 链逐字节恒等）。

---

## 快速复现

```bash
# 非 GPU 全量测试（Phase 20A 实测 150 passed）
cd 00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证
python3 -I -m pytest $(find . -name "test_*.py" -path "*/tests/*") -q

# 全仓配置交叉扫描（78 份 参数.json）
python3 -I - <<'EOF'
import json, pathlib
n=sng=sc=0
for p in pathlib.Path('.').rglob('参数.json'):
    if '.git' in p.parts: continue
    d=json.loads(p.read_text(encoding='utf-8')); n+=1
    if d.get('use_spnp_refinement'): sng+=1
    if d.get('prompt_selection_mode')=='single' and d.get('use_spnp_refinement'): sc+=1
print(f'runs={n} spnp_on={sng} single_and_spnp={sc}')   # -> 78 / 26 / 0
EOF
```

---

## 状态

```text
PHASE20A VERDICT:        PARTIAL
FINAL METHOD STATUS:     DRAFT
CODE CHANGE REQUIRED:    NO（采纳方案 A 时）
GPU RERUN REQUIRED:      CONDITIONAL（V1/V2/V3；2DGS 待 P0-2 比对）
OLD 3D RESULTS REUSABLE: NOT VERIFIED
```

**本阶段到此为止，不进入 Phase 20B。**