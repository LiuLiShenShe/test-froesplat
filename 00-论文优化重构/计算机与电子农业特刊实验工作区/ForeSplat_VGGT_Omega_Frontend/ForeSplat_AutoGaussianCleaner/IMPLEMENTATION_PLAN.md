# ForeSplat Automatic Gaussian Artifact Suppression

## 范围与依赖

复用原生 GGGS T×alpha Jacobian、每轮重归因和 filter_3D 等价烘焙。禁止手工 ID/ROI、按株参数、暗色删除规则。Mesh 不修改。19 株 Bundle 已存在，完整 GGGS 仅 KongQueZhuYu；其它已发现的 WanNianQing 模型是 2DGS，不能混用。

## 顺序与验证

1. 数据清查与冻结分组（约 20 分钟）：configs/scenes、plant_manifest.json、DATA_INVENTORY.csv；检查模型 schema、RGB/Mask/Camera/Depth/Confidence 路径，冻结开发/独立测试分组。
2. 统一原生后处理（约 2 小时）：cleaner、evaluation、scripts；实际 Rasterizer 权重和 Alpha 守恒测试；尺度无关几何、边缘/骨架贡献；A1–A4分类；全参数冻结后只优化混合体元协方差，固定有效 opacity，逐候选质量约束回滚。
3. 原生导出与测试（约 30 分钟）：PLY原字段/原ID保留，filter_3D baking 参数及真实渲染等价验证；指标须支持逐视角最坏值与不可评估状态；CLI重复/断点运行拒绝不兼容输入。
4. 模型准备（约 12–18 小时）：优先开发样本、至少一个独立样本，随后其它植物；全部复用已完成前馈 Bundle，统一30K原GGGS配置与已锁定修正相机接口，不添加训练清理损失；GPU1/uv 严格串行，记录命令、源码/配置散列和耗时。
5. 开发/独立验证和消融（随训练顺序）：先固定统一参数，再运行 A/B/C、一次/多轮；保留所有候选及被拒绝原因；KongQueZhuYu R5作单独基线，不能混入原模型对照。
6. 全19株批处理与报告：原生/查看PLY绝对路径、实测评价；缺模型、失败、无收益、回归逐项报告，绝不将排队/初始化记作完成。

## 方法与风险

背景权重以实际合成贡献计算，颜色不进入分类。外部前端深度仅作有置信度的辅助证据，不作为真值；遮挡后的深度差不能独立触发删除。边缘及骨架保护按像素尺度归一。自由相机由实际轨迹/外部前景点推导；没有真实Mask的自由相机采用外部点投影的保守代理，标记代理覆盖局限。

候选同时满足前景MSE/Alpha/边缘/薄结构/深度/法线约束后才采用。每轮相对初始模型及当前最佳双重检查，防止累计变透明或结构损失。质量约束不能保证所有语义叶柄被检测：只有RAP二值Mask时报告骨架代理，盆/叶/柄细分类不可认证。

开发组固定为 KongQueZhuYu、XiangPiShu1、WangWenCao1、ChangShouHua1；其余15株独立测试。未读取测试清理结果前冻结算法/config哈希；任何根据测试升级都必须标记该组参与开发，不能继续声称独立。

## 外部资料

博查搜索 2026-11-10 HTTP403：余额/额度不足，无可用搜索结果。PyTorch2.10接口已通过ctx7核查：https://github.com/pytorch/pytorch/blob/v2.10.0/docs/source/notes/autograd.rst 。算法证据来自本地GGGS源码及历史诊断文件，后续以实测为准。
