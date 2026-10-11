# 重建继承结论

没有任何样本获得REUSE_DIRECTLY（P6+R1）。有P6匹配 6/19；EXACT_MATCH版本-帧对 0/43。
逐样本决策见 `tables/RECONSTRUCTION_REUSE_DECISION.csv`；覆盖率见 `tables/MASK_MATCH_COVERAGE.csv`。
TARGETED_RERUN_REQUIRED是对新最终掩膜与已完成模型存在差异的有限样本建议，需Phase20C先产出新掩膜并再选受影响样本。
不得以此要求20序列全量重跑。旧模型仅能以其原输入/版本保留为legacy baseline，不能更名成R1结果。
即使个别帧EXACT_MATCH，也不能因局部覆盖或相同文件名判断整套训练输入/模型版本完全一致。
