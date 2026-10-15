# M2M3 方法与贡献决策

本阶段成功恢复三场景植物 Mesh，并完成 6/6 个新增 M1/M2 正式训练。M2M3 加权额外剪枝在 30/30 个预定事件中真实执行且产生删除，但外观、前景代理、Gaussian 容量和 Mesh 一致性方向跨场景不一致；因此加权剪枝判为 MIXED。Floor40 在 15/15 个 M2 事件中被计算和评估，但实际阻止删除数为 0，判为 NO_EFFECT。全部 9 个模型的 18 个 raw/post Mesh 可重载，结构代理差异为 MIXED；没有独立三维 GT，不能把这些差异写成真实几何精度变化。

```text
PHASE20E3 STATUS: PASS
REFERENCE_MESH: PASS
M2M3_WEIGHTED_PRUNING: MIXED
FLOOR40_ACTIVATION: NOT_TRIGGERED
FLOOR40_BENEFIT: NO_EFFECT
M2M3_MESH_EFFECT: MIXED
M2M3_MANUSCRIPT: REWRITE_TO_IMPLEMENTATION
TABLE7_EVIDENCE: PARTIAL
PHASE20F_READY: CONDITIONAL
```

## 加权剪枝

M1/M2 共记录 30 次真实额外剪枝事件，评分状态为 EVALUATED、冻结 alpha 支持可用且数值有限。实际额外删除总数为 WanNianQing1 M1/M2=9,017/9,256、CaoMei2=12,312/6,882、HongZhang=8,680/8,702。最终 Gaussian 数相对历史 M0 分别为 WanNianQing1 +4,500/-268、CaoMei2 -10,935/-13,501、HongZhang -14,864/-14,582；WanNianQing1 的 M1/M2 全图和前景 PSNR 均下降，而 CaoMei2 与 HongZhang 多数外观指标改善。M0 与新增运行在 18,000 iteration 之前已出现轨迹差异，因此最终容量差不能作为额外剪枝的纯因果估计。证据支持算法真实触发和场景依赖的容量/外观变化，不支持稳定正向收益或统一加速结论。

## 全局容量下限

15 个 M2 事件均处于冻结 Floor40 活动窗口并完成 floor 计算，累计 floor_blocked_count=0；WanNianQing1、CaoMei2、HongZhang 的阻止删除数均为 0。各场景五轮 3% 预算的理论容量下界均高于 0.4*max_seen_count，因而本次参数下没有实际删除被 Floor40 拦截。设置参数、进入活动窗口、计算下限与实际容量干预已分别记录；不能把本次 Floor40 写成已验证的容量保护收益。

## Mesh 对照

三个冻结 M0 参考模型和六个新增 M1/M2 模型均完成 bounded TSDF、raw/post Mesh 写出、重读和独立结构复算，共 18 行结构记录。后处理 Mesh 均为单一最大组件，但 raw Mesh 仍有大量分离组件；M1/M2 相对 M0 的顶点、面片、ROI 覆盖、代理轮廓和 Gaussian 深度一致性变化跨场景不一致。例：WanNianQing1 的 M1/M2 深度 MAE 由 0.0182 增至约 0.0518/0.0512，CaoMei2 M1 降至 0.0524 而 M2 升至 0.1576，HongZhang M1/M2 降至约 0.0637/0.0468。没有独立几何 GT 或物理尺度，故该结论仅是结构和重投影代理的 MIXED，不是几何精度证明。

## 可用于论文的贡献强度

论文方法必须改写到实现：固定 alpha 中心投影支持度、全局最大值归一化的加权额外 Gaussian pruning，以及可选的全局容量下限。删除或降级已实现证据不支持的 model-to-mask deficit、boundary-aware utility、bidirectional matching、local Floor40 和额外 densification/keep-prune 联合机制。M2M3 目前只能作为实现忠实的可选紧凑化/前景支持剪枝模块；在增加重复种子、正式 baseline 和独立几何评价前，不应作为已证实的强独立三维创新。

本次只有三场景各一个 Seed0；历史 M0 复用基于禁用分支的源码/配置等价及已归档初始参数证据，不是逐位相同训练轨迹。实际触发与正向收益分别判定。没有独立三维 GT，不宣称真实几何精度改善。

R1 不在本阶段调参或重训；前景参考保持冻结 R0。论文按 M2M3_MANUSCRIPT_CORRECTION_SPEC.md 改写到真实加权额外剪枝与可选全局 floor；本阶段没有修改论文正文。后续若要双向 deficit、边界效用或局部 Floor40，必须另立开发与实验，不当作既有贡献。

下一阶段先按 PHASE20F_HANDOFF.md 恢复物理尺度及同株同时间的独立表型记录；本阶段交付后停止，不自动进入 Phase20F。

## 实测来源绑定

- `M2M3_SCENE_SUMMARY.csv` SHA256 `592298e99345f485ad65e22b8dd6c7173094633e6c1243fe5fd0a7ffada9deac`
- `M2M3_PRUNING_EVENT_LOG.csv` SHA256 `72f8cdde652ed83412d82af0891cd19febcc4feb804a4f81f23e4c7ba19128a3`
- `M2M3_MESH_COMPARISON.csv` SHA256 `51d0dafe6b631f93253b150d55f176927b1d0425b14f0eb6680d36c793b8e681`
- `MESH_STRUCTURE_METRICS.csv` SHA256 `c6298decda2da8ec334eb059386fa191eb919f1d832a65a853fb70eef1ede1e2`
- `M2M3_TRAINING_RUNS.csv` SHA256 `f6117cbf8cb7c9dc877ff4106f2a6742f1c656706cdd3b5ec9dd6b4f37f6b764`
- `M2M3_HELDOUT_METRICS.csv` SHA256 `b06e5c8163696ac405da15622f4fc222aa4646967a9a9df953245b242db73322`
- `M2M3_BACKGROUND_GEOMETRY_PROXY_METRICS.csv` SHA256 `6621e9a7077808a73829d32a33a4eb7faccea0444c5fed154c30cc52eda681a5`
- `M2M3_FOREGROUND_PROXY_METRICS.csv` SHA256 `03cf2a23d2154995edf42333bfa14f2d36669cd05cd25272138c0cc8cb0770f3`
- `M2M3_RENDER_TIMING_SUMMARY.csv` SHA256 `b4cfd82830a633fa370f9c4c464c59ead5908f4f37ce76f5d06422f2523f8c18`
- `M2M3_CODE_FORMULA_MATRIX.csv` SHA256 `dadd891b49cedeb35fd7ea41a08f3463b344aab71fda3dfa62848ebe345b701a`
- `M2M3_CAPACITY_EFFECT_REPORT.md` SHA256 `356fce35f508b86225058cb438190fada691df7f2390f3da6d5eeed32e0ea8b9`
- `MESH_EXTRACTION_RUNS.csv` SHA256 `ed00875c78f9f3cb9aad79da93c16aaf79d9dd2096bc1f2f85036d6166438a54`
- `MESH_EXTRACTION_PROTOCOL.md` SHA256 `1f710d7983bae0ebedfa94d5b5444da22ae3ae35dee5a34595a20be064b3cd43`
- `M2M3_MANUSCRIPT_CORRECTION_SPEC.md` SHA256 `63679bc7925c26377dae3c77f92ec4fd5d2d03690e96560686839f9661029700`
- `TABLE7_EVIDENCE_UPDATE.md` SHA256 `871e3abfc30871683615fc80c2568d18e392089cffb9efcd735b63ccd5f11992`
