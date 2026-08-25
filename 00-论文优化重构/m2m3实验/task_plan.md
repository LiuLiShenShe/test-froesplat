# Task Plan: Foreground-Aware 2DGS + M2M3-Floor40

## Goal

在不破坏现有 `/data/fj/F2DMAS/2d-gaussian-splatting-main` 前景感知 2DGS 代码的前提下，建立独立副本仓库，并规划将 M2M3-Floor40 作为可选开关迁移到标准 200-view/多视图前景植物 2DGS 流程中的实验方案。

## Repository Isolation

- 原始仓库：`/data/fj/F2DMAS/2d-gaussian-splatting-main`
- M2M3 副本仓库：`/data/fj/F2DMAS/2d-gaussian-splatting-m2m3-floor40`
- 实验计划目录：`/data/fj/F2DMAS/00-论文优化重构/m2m3实验`
- 原则：原始仓库只读保留；所有 M2M3 代码修改、训练脚本和实验输出均进入副本仓库或本实验目录。

## Phases

- [x] Phase 1: 建立独立副本仓库
- [x] Phase 2: 审查当前前景感知 2DGS 的已有开关和代码挂点
- [x] Phase 3: 写出持久化实验计划文档
- [x] Phase 4: 在副本仓库中实现 M2M3-Floor40 独立开关
- [ ] Phase 5: 跑 3-scene pilot 实验
- [ ] Phase 6: 根据 pilot 决定是否扩展到正式多场景实验

## Key Questions

1. M2M3-Floor40 应该接在标准 2DGS 的哪个位置？
   - 初始建议：先接在现有 plant-aware pruning 之后/之内，作为候选删除集合的容量保护器；不要第一步就重写标准 densification。
2. 是否能和现有前景感知 loss、foreground track init、mask pruning 共存？
   - 可以。现有功能已有 `--xxx` 参数控制，M2M3 也应以默认关闭的新参数加入。
3. 预期收益是什么？
   - 不是再次证明“只重建前景”，而是在前景重建基础上减少冗余 Gaussian、降低废光/漂浮点、保护薄叶边界、减少 PLY 体积和 mesh 时间。
4. floor40 是否一定最优？
   - 不一定。标准 200-view/多视图 2DGS 比 sparse-view 更稳定，应先做 `floor30/floor40/floor50` 小扫或至少保留可配置参数。

## Decisions Made

- 副本仓库命名为 `2d-gaussian-splatting-m2m3-floor40`。
- 当前不修改原始仓库。
- M2M3-Floor40 采用独立开关模式，默认关闭。
- 实验优先在当前已经跑通过的植物场景中做 pilot，而不是直接全量重跑。
- 第一版实现以“保护现有前景感知 pruning / topology pruning 的容量边界”为主，降低破坏现有训练流程的风险。
- M2M3-Floor40 已在副本仓库中实现为默认关闭的可选开关；第一版只包裹现有后期 pruning 路径，不重写标准 2DGS densification。
- 三场景 pilot 脚本已生成在 `00-论文优化重构/m2m3实验/实验脚本`，所有训练输出将写入 `00-论文优化重构/m2m3实验/实验输出`。
- 三场景暂定为 `KongQueZhuYu`、`ChangShouHua2`、`CaoMei2`。孔雀竹芋和草莓使用 RAP-FSAM3 A5c 全帧 mask；长寿花当前只有 191/212 帧可匹配 mask，因此脚本会自动使用 matched-view gate list，并把长寿花初始化切换为 `foreground_mask`。
- 训练脚本默认 `--mode smoke`，只在显式指定 `--mode full` 时运行 30000 iteration。

## Errors Encountered

- 无阻塞错误。
- 注意：原仓库约 7.0G，副本也约 7.0G；后续训练输出需放入明确的实验输出目录，避免仓库继续膨胀且难以追踪。
- 语法验证使用 `python -m py_compile train.py arguments/__init__.py` 通过；验证产生的顶层 `__pycache__` 已清理。
- 三场景路径检查发现长寿花 `03-SAM/ChangShouHua2` 与 final locked source 有 21 帧缺 mask；处理方式是不伪造 mask，自动生成 `prepared_gate_lists/ChangShouHua2_mask_matched.txt`，训练只保留 191 个有 mask 的视角。

## Status

**Phase 5 prepared, not launched** - 已生成三场景 pilot 执行脚本、资产准备脚本、路径检查脚本和结果汇总脚本；已完成语法检查、路径检查和 dry-run。下一步可以启动 smoke pilot。
