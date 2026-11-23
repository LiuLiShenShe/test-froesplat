# TomatoPGT 复叶性状误差口径

- `final_leaf_trait_errors.csv`：逐性状误差与有效配对数。`n_scans` 为扫描数；同一扫描可有多片叶。
- `final_leaf_trait_pairs.csv`：19 个公制扫描中 70 片官方复叶的逐性状记录。筛 `status=included_reference_guided_landmarks` 即实际纳入的配对；其他行保留未纳入原因。包含 PLY 路径和两侧端点坐标。

作者通过对话确认已目视核对网页中的叶片对应。保留原始自动匹配记录、歧义和未匹配状态，不生成逐片人工端点签核。

叶长为复叶基部至顶端小叶尖的三维直线距离。叶宽为与参考相同中部位置两侧叶缘点的三维直线距离。参考侧使用 TomatoPGT 标注点云和 CloudGraph 骨架；模型侧由参考端点引导定位到对应复叶网格，再在原厘米坐标计算，未用配准尺度二次缩放。仅纳入已有两侧端点、通过原固定端点距离及尺度质控的样本。误差等于模型值减参考值，`r2_against_identity` 为 `1 - SSE/SST`（相对 1:1 线）。

两项结果适用于参考引导端点的已匹配子集，不能作为全部复叶的全自动性状精度。叶面积缺少独立、同口径的单侧面积参考，因此未计算误差。这里不统计株高和冠幅。

可从项目根目录运行 `python3 tools/summarize_tomatopgt_leaf_errors.py` 重新导出。脚本读取 `compound_auto_review/compound_match_suggestions.csv`，不改动原始匹配表、网格或标定。
