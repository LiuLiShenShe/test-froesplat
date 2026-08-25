# Plant-aware 2DGS 实施拆分文档索引

本目录由 `整体实现方案.md`、`F2DMAS.docx` 和 `Claude-Code-Skills完整分析.md` 拆解生成，用于把原始 F2DMAS 应用型工作流论文，重构为方法型论文：

> Plant-aware 2D Gaussian Splatting for Thin-Leaf 3D Reconstruction and Phenotypic Measurement

核心主线：

> 针对植物薄叶片三维重建中的背景 Gaussian 竞争、漂浮伪影和边界膨胀问题，提出 Plant-aware 2DGS，通过 mask-constrained optimization、topology-aware pruning 和 edge-aware meshing，提高复杂背景下植物三维重建质量与表型测量准确性。

## 目录结构

```text
Plant-aware-2DGS实施拆分/
├── 00-项目总览/
│   ├── 01-方案拆解总览.md
│   ├── 02-开发约束与组合规则.md
│   └── 03-Skills调用与任务协作建议.md
├── 01-数据与资源/
│   ├── 01-资源盘点与数据总表任务.md
│   ├── 02-论文现有证据提取.md
│   └── 03-COLMAP失败排查与重跑记录.md
├── 02-算法模块/
│   ├── M1-PR-IQA植物重建导向视图质量诊断.md（H-VQG 分层视图质量门控）
│   ├── M1-PR-IQA方案判断记录.md（H-VQG 方案判断）
│   ├── M2-FSAM3植物Mask生成.md
│   ├── M3-Mask约束2DGS优化.md
│   ├── M4-Topology感知Gaussian剪枝.md
│   └── M5-Edge感知薄叶片网格化.md
├── 03-实验设计/
│   ├── 01-总实验矩阵与消融组合.md
│   ├── 02-指标体系与结果表格.md
│   └── 03-验收标准与风险清单.md
├── 04-论文写作与图表/
│   ├── 01-论文结构重写任务.md
│   └── 02-图表重绘任务.md
└── 05-执行管理/
    ├── 01-阶段路线图与任务清单.md
    └── 02-代码重跑整体流程.md
```

## 推荐阅读顺序

1. 先读 `00-项目总览/01-方案拆解总览.md`，确认论文主线和模块边界。
2. 再读 `00-项目总览/02-开发约束与组合规则.md`，这是后续所有代码开发必须遵守的硬约束。
3. 数据和 COLMAP 状态先读 `01-数据与资源/03-COLMAP失败排查与重跑记录.md`，确认哪些样本可用、哪些样本需要重跑。
4. 代码重新跑之前读 `05-执行管理/02-代码重跑整体流程.md`，按数据、COLMAP、baseline、M1-M5 的工程顺序执行。
5. 算法实现按 `M1` 到 `M5` 阅读，但实际开发必须保持单分支、可开关、可组合。
6. 实验执行前读 `03-实验设计/01-总实验矩阵与消融组合.md`。
7. 写论文和画图时读 `04-论文写作与图表/`。
