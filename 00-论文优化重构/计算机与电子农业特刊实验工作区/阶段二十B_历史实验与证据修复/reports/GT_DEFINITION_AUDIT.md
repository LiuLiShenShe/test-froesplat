# GT 定义审计

验证现有 GT：31/31。不写入或重新保存任何 GT。
Easy21 逐像素核对 Phase12 clean、split_v2 的 potted∧¬cube、真实 LabelMe 多边形（只在内存核对）。
Challenge10 核对现有含盆标注；无拆分 pot，因此盆区指标不能由该集恢复。
逐帧来源、哈希和提交见 `tables/GT_DEFINITION_AUDIT.csv`。
原 E1 manifest 指向早期 union GT；不得用当前脚本注释倒推历史执行。
Table3–5 统一到同一 Easy20：4 序列×5 帧；不是20序列，也不是独立测试集。
v1 启发式 GT 不参与任何统计。不缩放 GT 或预测。
