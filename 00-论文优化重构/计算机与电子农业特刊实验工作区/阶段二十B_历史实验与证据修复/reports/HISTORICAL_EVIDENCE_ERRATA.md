# 历史证据统一勘误

本文件只新增勘误，不覆盖 Phase 19A、Phase 20A 或论文历史文件。基线提交为 `c119498c`。

| 历史结论 | 统一勘误 | 证据 |
| --- | --- | --- |
| `P6_raw_baseline` 被描述成早期 P2 或其他默认提示词 | `P6_raw_baseline` 的实际配置是 P6；Phase 12 配置使用 `prompt_list=P6`、`candidate_mode=per_instance`、`prompt_selection_mode=single`。 | `阶段十二_GT_v2_QA与P6正式验收/P6_raw_baseline/参数.json`、`reports/V0_V3_PREFLIGHT_REPORT.md` |
| Phase 12 默认执行了 GCFC | Phase 12 的正式 P6 基线没有打开 GCFC；后续 GCFC/残差相关输出不能反写为默认 P6 基线。 | `阶段十二_GT_v2_QA与P6正式验收/P6_raw_baseline/参数.json`、`reports/TABLE_3_4_5_RECALCULATION_REPORT.md` |
| 多提示词 `score_select` 使用 R1 | `score_select` 分支使用旧选择路径，不调用 R1；多提示词结果不能标成 R1 正式运行。 | `ranking_policy.py`、Phase 20A `PROMPT_AND_RUN_PROVENANCE_MATRIX.csv` |
| R1 已在生产掩膜生成中执行 | 当前证据只支持 R1 离线归档重排序；本阶段复核 182 条候选为公式一致，不代表 SAM3 或 2DGS 已按 R1 重跑。 | `reports/R1_182_CANDIDATE_RECHECK.md`、`tables/R1_FORMULA_RECHECK.csv` |
| Table 3/4/5 可以把早期 GT 和 Phase 12 GT 混用 | 早期 E1 使用过 union 口径；本阶段统一使用 `potted_clean = potted_plant AND NOT blue_cube`，只报告可追溯 Easy20 真实预测。 | `reports/GT_DEFINITION_AUDIT.md`、`reports/TABLE_3_4_5_RECALCULATION_REPORT.md` |
| Table 5 的 `Boundary-F1=0.4543` 和 `Leakage=0.001935` 可直接继承 | 两个数字没有绑定到可追溯预测、GT、帧列表和聚合记录，不能写入新的 VERIFIED 表。 | `reports/TABLE_3_4_5_RECALCULATION_REPORT.md` |
| Table 7 的四行是已核验的 20 序列平均 | 本地只找到 11 个历史 2DGS `cfg_args` 模型、12 个 Mesh 日志和 2 个 M2M3 full-summary 文件（各为 3 场景 pilot）；20 序列共同划分、ROI、失败处理和计时边界未闭合。 | `reports/TABLE7_PROVENANCE_AUDIT.md`、`tables/TABLE7_CLAIM_STATUS.csv` |
| M2M3-GO 论文公式已由代码完整执行 | 实际代码只有多视图 mask consistency、加权 penalty pruning 和全局容量保护；model-to-mask deficit、boundary support、sigmoid utility、独立 densify 和 local Floor40 未执行。 | `tables/M2M3_FORMULA_CODE_MATRIX.csv`、`reports/M2M3_IMPLEMENTATION_TRUTH.md` |
| Floor40 是前景局部块保护 | Floor40 使用 `round(max_seen_count * 0.4)` 的全局容量下限，不是按前景局部块分别保护 40%。 | `tables/M2M3_FORMULA_CODE_MATRIX.csv`、`reports/M2M3_IMPLEMENTATION_TRUTH.md` |
| D3/D6/D12 已有可靠确认帧 | Phase 20A 的 D3/D6/D12 尚无可靠的已确认帧；本阶段不填入推测性帧号或指标。 | Phase 20A `P0_BLOCKERS_AND_NEXT_ACTIONS.md`、`tables/MASK_MATCH_COVERAGE.csv` |
| 论文 FFT 数学表达等同于代码 | 论文表达和实际 FFT 特征代码存在差异；应把代码实现单独作为实现事实，不能由公式反推执行细节。 | Phase 20A `FINAL_RAP_FSAM3_SPEC_DRAFT.md`、生产脚本 `生成RAP-FSAM3掩膜.py` |

## 使用规则

以上勘误只改变“可以怎样描述证据”，不修改原始 GT、预测、模型、历史 CSV、论文 `.tex`、RAP-FSAM3、2DGS、M2M3-GO 或 `ranking_policy.py`。
