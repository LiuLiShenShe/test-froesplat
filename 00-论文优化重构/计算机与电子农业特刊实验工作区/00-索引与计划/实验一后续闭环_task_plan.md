# Task Plan: 实验一后续闭环

## Goal

把实验一从“已有分割结果和图件”推进到“可写入论文的横向分割主表 + 少量监督参照 + 至少一组下游传导证据”。

## Phases

- [x] Phase 1: 分割表与图件资产闭环。
- [x] Phase 2: 补 U-Net 少量监督参照。
- [x] Phase 3: 补 DeepLabv3+ 少量监督参照。
- [ ] Phase 4: 视资源补 SegFormer。
- [ ] Phase 5: 启动 3 方法统一 2DGS/网格/表型小闭环。
- [ ] Phase 6: 达到写作门槛后新建 `06-论文修订草稿/`。

## Decisions Made

- X-Decoder 和 OpenSeeD 本轮不再运行，也不再作为阶段十阻塞项；表格中保留为“暂停/未来扩展候选”。
- 实验一分割主表优先闭环已完成 S23 统一 6 帧 GT 子集：Florence-2、CLIPSeg、SAM2 oracle、SAM3 单提示词、Grounded-SAM、Grounded-SAM2、RAP-FSAM3-v2。
- 图件资产已对应到分割表备注：`05-图件与论文映射/实验一_视觉基础模型横向对比/figure_asset_index.csv` 可按 `method` 字段追踪 overlay、error_map、contact_sheet 和 source_data。
- 监督参照执行顺序为 U-Net、DeepLabv3+、SegFormer；前两项优先支撑审稿问题“少量 GT 训练是否超过通用 VFM”。
- U-Net 与 DeepLabv3+ lite 已按 S23 两折序列级 few-shot 口径完成；当前 torchvision C++ 扩展不可导入，DeepLabv3+ 使用项目内随机初始化轻量 ASPP encoder-decoder 参照。
- 首轮下游小闭环选 3 个方法：RAP-FSAM3-v2、SAM3 单提示词、Grounded-SAM；CLIPSeg 作为 Grounded-SAM 无法稳定进入下游时的替代第三方法。

## Deliverables

- 分割主表：`../04-结果表格模板/实验一_视觉基础模型横向对比分割表.csv`
- 下游状态表：`../04-结果表格模板/实验一_视觉基础模型横向对比下游重建表.csv`
- 图件索引：`../05-图件与论文映射/实验一_视觉基础模型横向对比/figure_asset_index.csv`
- 图件说明：`../05-图件与论文映射/实验一_视觉基础模型横向对比/图板说明.md`

## Status

**Currently in Phase 4/5 decision point** - S23 9 方法分割主表、少量监督参照和图件资产已闭环；下一步可选补 SegFormer，或直接启动 RAP-FSAM3-v2、SAM3 单提示词、Grounded-SAM/CLIPSeg 的统一下游 2DGS/网格/表型小闭环。
