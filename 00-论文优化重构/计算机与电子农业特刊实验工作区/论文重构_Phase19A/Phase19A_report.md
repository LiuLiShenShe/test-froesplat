# Phase 19A — 论文重构审计（Manuscript Reconstruction Audit）最终报告

**生成日期：** 2026-09-14
**阶段目标：** 对齐被拒稿件 + 审稿意见 + Phase 11-18 算法/实验 + 最终图表 + 证据等级，产出 Phase 19B 全文重写所需的全部设计文档。
**阶段性质：** AUDIT + RECONSTRUCTION DESIGN（审计 + 重构设计），**不包含正文写作**。

---

## 1. 起始状态（Starting State）

| 项 | 值 |
|---|---|
| HEAD | `9f72566e`（Phase 18 收束提交，含 §5 "23/32" 修正） |
| 分支 | `main` |
| 工作树 | 干净（仅新增未跟踪的 `论文重构_Phase19A/` 目录，无算法/稿件改动） |
| 主稿目标 | `ForeSplat.tex`（82,415 B，cas-sc 类，CompAg 特刊，9 图，11 作者） |
| 审稿流 | Stream A = 真实审稿（F2DMAS应用版/审稿意见/）；Stream B = 5 审稿人模拟审稿（论文cc/05_peer_review_package.md） |

## 2. 任务范围与禁止项（Scope & Forbidden Actions）

本阶段执行：① 旧稿审计、② 审稿意见审计、③ 新旧算法差异审计、④ 新实验资产整理、⑤ 图表资产整理、⑥ Claim–Evidence 映射、⑦ 审稿意见→稿件动作映射、⑧ 新稿蓝图。
**严格执行的禁止项**：未重写 `ForeSplat.tex` / `f2dmas_elsarticle.tex`；未删除旧稿；未覆盖旧图；未新增实验；未调参/改生产代码；未编造审稿意见；未将 DEV10 当 held-out、TEST32 当精度证据；未把 A6/A7 写回主方法创新；未自动 push。

## 3. 稿件清单（WP1, Step 1）

`Phase19A_manuscript_inventory.md`（18,759 B）—— 目录内全部 8 个 `.tex` 已编目并分类（主稿/早期版本/模板/补充）；确认 **ForeSplat.tex 为 Phase 19B 重写主目标**；`cas-refs.bib` 两份一致副本；识别 PDF 版本与图件资产。

## 4. 审稿意见组织（WP2, Steps 2-4）

- `Phase19A_reviewer_comments_raw.md`（14,375 B）—— **32 条意见，ID 编号 A1-C01…A2-C06 / B-EIC-C01…C04 / B-R1-C01…C04 / B-R2-C01…C04 / B-R3-C01…C04 / B-DA-C01…C05**（头注已修正 30→32）。
- `Phase19A_reviewer_issue_matrix.csv`（26,489 B）—— 32 行 × 16 列；`required_manuscript_action` 无空白；状态 **25 RESOLVED / 2 PARTIALLY_RESOLVED / 5 NOT_RESOLVED**。
- `Phase19A_resolution_status.md`（7,465 B）—— 分 stream 决议表；关键决策：R1.2 Table 6 消融 RESOLVED；E7 CRITICAL PARTIALLY_RESOLVED；叶片宽度、外部基线、植物学命名、消融复现、FFT 阈值 NOT_RESOLVED。

## 5. 旧稿审计（WP3, Steps 5-7）

- `Phase19A_old_manuscript_audit.csv`（23,844 B）—— **66 行 × 9 列**（修复脚本转写，CSV 解析回环验证无错行）；章节覆盖 Title/Abstract/Keywords/Highlights/Introduction(6)/Methods(19)/Results(6)/Discussion(5)/Conclusion/References/Figures(9)/Tables(7)。
- `Phase19A_method_component_audit.md`（10,562 B）—— **G11 关键发现**：A6/A7 缺席主方法（PASS）；**§2.3.3 描述旧版打分公式（q_contrast/q_temp 在主项）→ 必须重写为 R1**（REQUIRED REWRITE）；18 组件逐项状态表。
- `Phase19A_claim_audit.csv`（9,098 B）—— 19 行 × 6 列；**G13 通过**：10 FALSE / 1 "FALSE (statistical test absent)" / 4 TRUE / 4 NA；唯一 "robust" TRUE 为已记录的例外（仅限政策名 "R1 robust instance ranking"）。

## 6. 证据与资产（WP4, Steps 8-13）

- `Phase19A_algorithm_delta.md`（17,015 B）—— 核心发现：**F1 R1 缺席稿件（CRITICAL）**；F2 A6/A7 正确缺席（OK）；F3 per-instance 生成小改；F4 P2/P6 术语修复；全组件 Status 表。
- `Phase19A_experiment_asset_registry.csv`（8,344 B）—— 21 行 × 12 列；paper_usable：TRUE 12 / DEVELOPMENT_ONLY 6 / ABLATION_ONLY 2 / FALSE 1。
- `Phase19A_image_asset_inventory.md`（12,964 B）—— ForeSplat Fig1-9；Phase18 Figure A-D；Phase17B/C 可视化；05-图件映射面板；暂定图件计划。
- `Phase19A_claim_evidence_matrix.csv`（11,781 B）—— 16 行 × 14 列；Claims C01-C08 + P01-P08；**C05 FORBIDDEN 标记**；G6 孤儿检查 NONE。

## 7. 公式审计（WP5, Step 20）

`Phase19A_formula_audit.md`（19,309 B）—— 符号表；P2/P6 三重含义消歧（P2=生产默认提示集；P6=生产提示词典成员；P6_raw_baseline=Phase12 冻结运行标识）；罚项符号（+0.25 垂直覆盖 <0.30、+0.15 轨迹重叠 <0.20，正值**减去**）；**G9/G10 通过**：公式对照 `生成RAP-FSAM3掩膜.py` 行 1592-1614、1625-1643、1667 与 `ranking_policy.py:44-48`。
主公式：`primary_score = base_total − q_contrast/5.5 − penalty`（denom=5.5 冻结）。

## 8. 蓝图（WP6, Steps 14-27）

13/13 文件全部落地并验证（非空、结构完整）：

| 步骤 | 文件 | 字节 | 要点 |
|---|---|---|---|
| 14 | `Phase19A_figure_plan.md` | 14,272 | 6 图：Fig3（候选排序失败机制）与 Fig4（排序策略消融+救援）标记 NEW |
| 15 | `Phase19A_table_plan.md` | 19,095 | 8 表：Table1 符号表(NEW)、Table7 排序策略/救援(NEW)；表号对齐 claim matrix |
| 16 | `Phase19A_core_preservation.md` | 13,578 | 核心卖点（R² 至 0.988 "comparable to"）+ 7 项必留实验 + 排序故事不挤占 3D 核心原则 |
| 17 | `Phase19A_proposed_contributions.md` | 11,861 | 4 贡献，各带证据等级 + RISK LINE；G14 强制 A6/A7 非贡献 |
| 18 | `Phase19A_introduction_blueprint.md` | 13,843 | 6 段逻辑链，逐段引用槽 + 声明约束 |
| 19 | `Phase19A_method_blueprint.md` | 14,791 | 10 小节；§6 R1 为 THE 方法（旧 softmax 仅作被取代基线） |
| 21 | `Phase19A_experiment_blueprint.md` | 13,865 | Q1-Q7 科学问题组织；Q4 排序失败新小节；Q7 A6/A7 负结果 |
| 22 | `Phase19A_discussion_blueprint.md` | 19,810 | 6 小节，C05 仅限选择理由表述 |
| 23 | `Phase19A_limitations_blueprint.md` | 17,674 | G12：GT10=10/2 非 held-out、TEST32 无 GT、R1 未独立验证、无显著性检验等 7 项 |
| 24 | `Phase19A_abstract_blueprint.md` | 7,125 | 6 槽位抽象蓝图，MUST 含限制安全措辞 |
| 25 | `Phase19A_title_candidates.md` | 7,852 | 5 候选（EN+中）；无 cross-species 过度声明 |
| 26 | `Phase19A_new_storyline.md` | 16,172 | 15 步叙事链（排序失败+R1 修复为科学脊柱，不吞掉 3D 表型核心） |
| 27 | `Phase19A_old_to_new_mapping.csv` | 32,454 | **70 行 × 8 列**；KEEP 37 / KEEP_WITH_MINOR_EDIT 24 / REWRITE 3 / MERGE 1 / NEW_SECTION_REQUIRED 5；A6/A7 只路由到 Discussion/Limitations |

## 9. 排除与包装（WP7, Steps 28-32）

- `Phase19A_reviewer_to_manuscript_mapping.md`（22,202 B）—— 32 条意见逐条映射（ID/关注点/旧稿问题/新证据/稿件动作/状态）；8 条合并主题线；**G3 验证块：25/2/5 无空白动作**。
- `Phase19A_exclusion_list.md`（23,523 B）—— 禁止声明（C05 任何形式、outperform、significantly、generaliz*、strong agreement、A6/A7 获益）、禁止实验位置（GT10/TEST32、A6/A7 主方法、Phase 编号/SHA、q_contrast 主项、P6_raw_baseline 误标）、禁止框架（cross-species、编造意见、手填数字）、7 项必提清单。
- `Phase19A_main_supplement_plan.md`（25,773 B）—— 主文清单（逐节绑定 C/P claim 与 Q 问题）+ S1-S5 补充物 + 6 条拆分规则 + **孤儿扫描 0 孤儿**（12 TRUE 全分配、2 ABLATION_ONLY 经例外规则入 S2、7 政策排除）。
- `Phase19A_journal_fit_check.md`（15,840 B）—— CompAg 匹配记分卡（STRONG FIT after gaps）；定位风险与 3 项重框架动作；5 个双语关键词（无 "robustness"）；**裁决 REVISE FIRST**（3 项修改均为措辞/包装，无需新实验）；更正了 B-R3-C01/C03/C04 主题映射（与 issue matrix 核对一致）。

## 10. 接受门验证（Gates G1-G14, Step 34）

| 门 | 检查项 | 结果 |
|---|---|---|
| G1 | 旧稿完整审计（66 行、12 类章节全覆盖） | PASS |
| G2 | 审稿意见完整（32 条、双 stream） | PASS |
| G3 | 每条意见有稿件动作（0 空白） | PASS |
| G4 | 算法差异清晰（全组件 Status） | PASS |
| G5 | 实验资产已索引（21 行） | PASS |
| G6 | 主声明均有证据（16 claim、0 孤儿） | PASS |
| G7 | 图件+表格计划完成（6 图 + 8 表） | PASS |
| G8 | 核心保留已记录 | PASS |
| G9 | P2/P6 命名消歧 | PASS |
| G10 | 排序公式取自代码（S20 行 1592-1614/1625-1643/1667、ranking_policy.py:44-48） | PASS |
| G11 | A6/A7 移出主方法（方法组件审计 PASS） | PASS |
| G12 | 限制明确（limitations 蓝图 7 项） | PASS |
| G13 | 禁止声明已识别（FALSE/NA 占 15/19，无允许+禁词） | PASS |
| G14 | Phase 19B 蓝图完整（13/13 蓝图/映射文件） | PASS |

**G1-G14：14/14 PASS**（初检 G1/G13 的两处 FAIL 为校验脚本语义误差，修正后确认通过：G1 节名 '2 Materials and Methods' 而非 'Methods'；G13 唯一命中的 'robust' 为 R1 政策名文档化例外）。

## 11. Phase 19B 关键发现（Key Findings）

1. **§2.3.3 必须重写**：现文描述旧打分（q_contrast/q_temp 在主项），必须改为 R1（q_contrast 移出主项、denom=5.5 冻结）；旧语义门控 softmax 选择器降为消融/补充基线。
2. **排序失败 + R1 救援是新头条故事**（C01-C04；Figure A/C；0039/0040/0041 救援 F1≥0.95、Easy21 零回归）——但不挤占 3D 表型核心（rank-preservation 原则）。
3. **A6/A7 全程仅作负结果/限制**（C06-C08, P08；默认 OFF）；不进主方法，不作贡献。
4. **措辞天花板**：无 "outperforms"、"significantly"、"generalizes"、"strong agreement"、SOTA；以 "among the tested methods on this evaluation set" / "comparable to" 表述。
5. **P6_raw_baseline 不得误标**为 P6 提示集输出（实为默认 P2 提示集冻结运行）。

## 12. 禁止项合规检查（Forbidden Actions Compliance）

| 禁止项 | 状态 |
|---|---|
| 直接重写 ForeSplat.tex / f2dmas_elsarticle.tex | ✅ 未执行 |
| 删除旧稿件 | ✅ 未执行 |
| 覆盖旧图 | ✅ 未执行 |
| 新增实验 | ✅ 未执行 |
| 调参/改生产代码 | ✅ 未执行 |
| 编造审稿意见 | ✅ 未执行（32 条全部来自两 stream 源文件） |
| DEV10 当 held-out / TEST32 当精度证据 | ✅ 未执行（蓝图全部标注 NOT held-out / NO-GT） |
| A6/A7 写回主方法创新 | ✅ 未执行（G11/G14 PASS） |
| 自动 push | ✅ 未执行（本阶段仅本地 commit） |

## 13. 工作树状态（Working Tree）

- HEAD：`9f72566e`（未变）；分支 `main`。
- 新增：`论文重构_Phase19A/`（30 个文件：26 蓝图/审计/映射 + 本报告 + 3 验证辅助说明）。
- 无算法文件、无生产代码、无稿件 `.tex` 改动。

## 14. 输出物完整清单（Artifact Inventory, 30 项）

目录：`00-论文优化重构/计算机与电子农业特刊实验工作区/论文重构_Phase19A/`
（见 §4-§9 分节清单；全部 30 文件非空且经校验）

## 15. 对 Phase 19B 的交接物（Handover to Phase 19B）

1. 主稿目标：重写 `ForeSplat.tex`（正文），结构按 WP6 蓝图（intro 6 段 → methods 10 小节 → results Q1-Q7 → discussion 6 小节 → limitations 7 项）。
2. 图表：图件计划 6 图（Fig3/4 新增）+ 表格计划 8 表（Table1/7 新增）。
3. 主/补拆分：`main_supplement_plan.md`（主文 + S1-S5）。
4. 措辞控制：`exclusion_list.md` + `claim_evidence_matrix.csv`（allowed/forbidden wording 逐 claim）。
5. 审稿响应：`reviewer_to_manuscript_mapping.md`（32 条动作）。
6. 叙事线：`new_storyline.md` 15 步。
7. 公式权威：`formula_audit.md`（R1 公式 + S20 行引用）。
8. 提交建议：`journal_fit_check.md`（REVISE FIRST 后投 CompAg SI）。

## 16. 已知局限与遗留（Residual Gaps）

- 5 条 NOT_RESOLVED 审稿意见（叶片宽度 R²=0.900 归因、外部重建基线、植物学命名表、单样本消融复现、FFT 阈值）→ 全部路由至 Limitations/未来工作，不在 Phase 19B 主文中假装解决。
- 2 条 PARTIALLY_RESOLVED（B-EIC-C01 cross-species 措辞、B-DA-C01 E7 单样本证据）→ 按限定措辞写入。
- 无显著性检验：全部描述性；蓝图明示不得引入 "statistically significant"。

## 17. 一致性核查（Consistency Checks）

- 32 条意见 ID 与 issue matrix ↔ mapping 文件双向一致。
- 表号（Table 5/6/7/8 拆分重排）与 claim matrix P04-P07 目标一致。
- 资产注册表 21 行 ↔ main/supplement 分配 0 孤儿。
- R1 公式、罚项符号、P2/P6 消歧在 formula_audit / method_blueprint / experiment_blueprint / old_to_new_mapping 四处一致。
- B-R3-C01/C03/C04 主题映射在 issue matrix 与 journal_fit_check 中一致。

## 18. 统计摘要（Phase 11-18 证据面）

- 实验资产：21 项（TRUE 12 / DEVELOPMENT_ONLY 6 / ABLATION_ONLY 2 / FALSE 1）。
- Claim 面：16 条（8 旧 + 8 论文级）；证据等级 PROVEN 3 / SUPPORTED 3 / DESCRIPTIVE 6 / NOT_SUPPORTED 4（含 1 FORBIDDEN + 1 FALSE）。
- 审稿意见：32 条；25 已解决、2 部分解决、5 未解决。
- 测试套件：150/150（131 历史 + 11 P17C + 8 P18），T1-T8 一致性 8/8。

## 19. 结论（Verdict）

**Phase19A: CLOSED**

- 全部 26 项协议输出物 + 本报告（30 文件）已生成并验证。
- G1-G14 全部通过（14/14）。
- 无算法改动、无稿件改写、无 push；HEAD 保持 `9f72566e`，工作树仅新增 Phase19A 目录。

## 20. Phase 19B 就绪判定（Readiness for Phase 19B）

**Ready for Phase19B: YES**

- 蓝图完整（G14）、公式权威（G10）、审稿动作齐全（G3）、证据映射封闭（G6）、措辞天花板明确（G13）、排除清单与主/补拆分就绪（WP7）。
- Phase 19B 可直接依据本包开始全文重写 `ForeSplat.tex`，并按 `main_supplement_plan.md` 组织正文与补充材料。

---

**Phase19A 最终判定：CLOSED — Ready for Phase19B: YES**（2026-09-14）
