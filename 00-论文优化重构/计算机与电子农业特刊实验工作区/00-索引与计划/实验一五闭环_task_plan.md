# Task Plan: 实验一与实验五闭环

## Goal
完成实验一视觉基础模型横向对比与实验五掩膜质量到表型误差敏感性实验的结果表、图件、说明文档和状态板闭环。

## Phases
- [x] Phase 1: 盘点现有计划、脚本、结果表与图件
- [x] Phase 2: 运行或修复实验一/实验五生成流程
- [x] Phase 3: 补齐结果表、source_data、图件说明和论文映射
- [x] Phase 4: 验证产物完整性并更新状态板

## Key Questions
1. 实验一当前缺口是外部基线真实运行、图件资产还是表格状态同步？
2. 实验五是否已有可复现脚本，是否需要基于现有实验一分割指标生成敏感性结果？
3. 哪些产物需要标注为真实结果、诊断结果或待真实重建替换？

## Decisions Made
- 使用现有工作区结构，不另建新的实验根目录。
- 实验一监督参照采用 S23 两序列 leave-one-sequence-out 口径：`KongQueZhuYu` GT5 与 `XianKeLai1` GT1 互为训练/测试序列，避免同序列相邻帧泄漏。
- 当前 PyTorch 和 GPU 可用，但 torchvision 导入失败；U-Net 与 DeepLabv3+ 参照均使用项目内随机初始化轻量网络，并在表格中标注无外部预训练。
- 实验五当前只可严谨闭环“人工-虚拟表型一致性”和“表型误差敏感性诊断图件”；严格多掩膜重建仍需单独补跑，不能写成已完成。
- 实验一下游 2DGS/网格/表型小闭环本轮未启动；下游表只更新状态，不填虚构重建指标。

## Errors Encountered
- `git status` 在 `/data/fj/F2DMAS` 返回非 Git 仓库；本轮按工作区文件产物推进。

## Status
**Completed current reproducible closure** - 实验一 S23 9 方法分割表、少量监督参照、图件资产与 source_data 已闭环；实验五当前表型误差图件、source_data 与状态说明已闭环。后续剩余项为实验一下游 2DGS 小闭环和实验五严格多掩膜统一重建。
