# V0–V3 预检

配置数：20（4组×5样本）；资产行 READY：20/20。
候选缓存来自真实 Phase 12 `P6_raw_baseline/候选掩膜/raw_instance_P6`，不会覆盖历史运行；候选缓存不能证明 R1 已正式执行，只能减少候选生成成本。
输入、候选帧覆盖、COLMAP sparse、SAM3 repo/checkpoint 均逐样本记录在 `tables/V0_V3_INPUT_MANIFEST.csv`。
V0/V1 的候选选择/离线可在无GPU环境验证；若按正式生产流程执行，仍应保留运行日志和参数 JSON。
V2/V3 的 SPNP backend 会加载 SAM3，需要 GPU；本阶段未启动任何生成或细化。
若 manifest status 为 BLOCKED，不能进入 Phase20C；当前配置只表达可执行条件，不代表实验结果已经存在。
