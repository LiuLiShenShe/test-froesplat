# R1 182 条候选最终核查

182/182 PASS。
风险惩罚独立从原始实例掩膜纵向覆盖重算，不通过 base−archived_total 反推，避免循环验证。
含惩罚候选数：1。base_total 与 total_score 最大差 0.25000000。
total_score 最大误差 6.81443810269e-08；primary 最大误差 1.11022302463e-16；离线 primary 最大误差 6.81443811379e-08。
冻结分母5.5，weights=1,1,1,1,1,0.5；semantic_gate关闭，semantic_total=0；无track_mask实例，0.15分支不执行。
R1=base_total−q_contrast/5.5−risk_penalty，tie_break=0；q_temp未配准为0.5（CSV逐条保存）。
总分与原始提示词评分按图像身份+面积验证唯一来源；raw instance ID以文件名和候选明细映射，不靠目录排序。
数值容差1e-6处理归档SAM分数六位小数，不更改归档数据、参数或公式。
Phase20A所述0.25差值是比较未扣风险项的base_total与已扣项的total_score，见下表。

| candidate_id | base_total | risk_penalty | total_score | error_total |
|---|---|---|---|---|
| BaiZhang_0042_09 | 0.609126836742 | 0.25 | 0.359126836742 | 1.64952099757e-08 |

逐条判定与真实路径、哈希：`tables/R1_FORMULA_RECHECK.csv`；未运行SAM3、未修改ranking_policy.py。
