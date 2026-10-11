# 表型数据来源审计

人工与虚拟值可追溯到 `植株数据.xlsx`，并可重算 Table 8 的均值、MAE、RMSE、MAPE 和 R²；这些值是已记录结果，不是本仓库现场从 Mesh 提取的结果。
`生成实验一五当前结果.py` 读取工作簿并生成统计表，没有点云/Mesh读取、尺度变换、点选、叶片 landmark 或 trait geometry 函数。
`measure_traits_entry.py` 只写测量 manifest，要求外部 `mesh`、`scale-marker` 和可选 `landmarks`；发布版说明完整工具是交互/半自动且未提供。
因此四项性状的统计结果可以作为 `LEGACY_RECORDED_STATISTICS`，但不能标为 `REPRODUCIBLE_3D_EXTRACTION`；每个虚拟值对应的 Mesh、坐标系、尺度系数和点选记录未找到。
Table 8 的 n=21（株高/冠幅）和 n=63（叶长/叶宽）来自工作簿结构，但不证明所有样本都有可复核三维产物。
