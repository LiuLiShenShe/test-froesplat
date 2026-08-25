# Task Plan: 实验三多样本模块消融补强

## Goal
把实验三从单样本机制消融调整为多样本平均主表，并保留逐帧结果作为 supplementary。

## Phases
- [x] Phase 1: 梳理现有实验三计划、结果模板和可复用数据。
- [x] Phase 2: 修改实验三方案，明确 3 样本最低主表与 6 样本扩展口径。
- [x] Phase 3: 增补结果表模板，包括主表平均、逐样本汇总和逐帧 supplementary。
- [x] Phase 4: 更新总规划文件并交付修改摘要。

## Key Questions
1. 当前是否已有多样本 A0/A5c 可复用，是否已有 A1/A1s/A2 多样本输出？
2. 主文表应按样本均值还是帧均值汇总？

## Decisions Made
- 主文优先采用样本级等权平均，避免帧数不同导致某个样本权重过高。
- 若算力或标注有限，最低闭环采用 3 个样本 x 5 帧；完整逐帧结果放 supplementary。
- 实际已完成四样本代表集 4 x 5 = 20 帧主表，A0/A1/A1s/A2/A5c 五档均有数值。
- 旧 A1 使用保存评分离线重放：`old_score = 总分 - 语义门控总修正`，并保存选择记录。

## Errors Encountered
- 动态导入实验一评估脚本时，Python 3.13 `dataclass` 需要先把模块注册到 `sys.modules`；已修正后完成评估。

## Status
**Complete** - 多样本主表、逐样本表、逐帧 supplementary 和规划文档均已更新。
