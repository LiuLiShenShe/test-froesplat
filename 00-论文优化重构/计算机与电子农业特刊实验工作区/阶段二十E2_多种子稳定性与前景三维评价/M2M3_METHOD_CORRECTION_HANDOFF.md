# M2M3 方法修正交接：以真实实现为下一阶段基线

本文件只作实施决策和证据交接。本阶段未修改 M2M3-GO、未运行新 M2M3 训练或 mesh 提取，也未修改论文。下一阶段建议选择 `REWRITE_TO_IMPLEMENTATION`：按已有加权剪枝与全局容量保护的真实实现修正方法叙述。未实现的双向匹配机制必须另立开发和实验任务。

## 1. 已复核的实现路径

生产实现为仓库根目录 `2d-gaussian-splatting-m2m3-floor40/train.py`，核心入口与链路为：

`training()` → `maybe_prune_gaussians()`（362 行，达到 pruning_start 且 iteration 满足 interval）→ 既有 opacity/brightness/mask 候选集合 → `m2m3_prune_scores()`（248 行）→ 每轮最大删除预算和 top-k → 仅 `m2m3_floor` 激活的全局 floor → `gaussians.prune_points()`（465 行）。

标准 `gaussian_model.densify_and_prune()` 仍是原训练流程的一部分；现有 M2M3 没有独立的 model-to-mask densify/keep 控制器。本文件行号来自本次只读审计，源码 SHA256 见 `reports/M2M3_FROZEN_MASK_LOADING_AUDIT.json`。

| 方法主张 | 真实实现 | 下一阶段应采用的表述 |
| --- | --- | --- |
| mask-to-model 支持 | `mask_consistency_scores()`（327 行）：将 Gaussian 中心投影到所选相机，用 nearest `grid_sample` 检查 mask，计算 hits/visible；无可见视图时 score 默认 1 | 所选多视图中心投影的前景支持比例；不能写成体积覆盖或足迹积分 |
| 视图质量加权 q、eta | 当前路径没有论文候选公式中的质量加权 | 删除未实现的权重说明 |
| model-to-mask deficit，例如 M(1−A) | 没有该双向误差图驱动的增密路径 | 不宣称已执行 |
| boundary-distance 支持、sigmoid utility | 未实现 | 删除对应已实现声明 |
| photometric residual、可见视图数量效用 | 未进入该加权剪枝 score | 不把 `m2m3_view_weight` 的名字当成视图数量证据 |
| Gaussian 加权分数 | opacity、brightness、mask 低于阈值的正惩罚，以及大于中位尺度的惩罚、`max_radii2D` 惩罚；各项用自身最大值归一化 | 明列实际分数、阈值和候选集合；屏幕半径项称屏幕半径项 |
| Floor40 | 全局数量下限，默认按已观测最大 Gaussian 数量乘 0.4 并 round；也有 initial/current 或固定 count 分支 | 全局容量保护，不是每个植物部位/局部块保留 40% |

旧 pilot 使用的权重、预算与时间表必须从旧完整配置逐项继承并冻结，不能把代码默认值与已运行参数混同。Phase20B 的 `tables/M2M3_FORMULA_CODE_MATRIX.csv` 保存了对应历史参数与证据路径。

## 2. Floor40 的作用边界

`resolve_capacity_floor_count()`（280 行）在 ratio 分支返回 `min(max(round(reference_count * ratio), 0), current_count)`。默认 reference 是 `max_seen_count`。floor 只限制当前额外剪枝允许删除的数量，不补回已被其他路径删除的 Gaussian，不对后续标准 densification/pruning 作全局数量保证。数量下限不能证明叶片、薄结构或拓扑被保留。

必须用每轮 capacity report 中的 requested、after_budget、floor_count、floor_blocked、removed 实证证明 gate 是否实际触发。仅设置 ratio=0.4 不能证明该机制在每个模型里产生影响。

## 3. 冻结输入中的 mask 如何进入后续实现

本次代表性文件检查确认 B/C 的 `images_rgba` 是 RGBA，alpha 含 0/255；`mask_mode=none` 不会阻止 loader 读取原始第四通道。`utils/camera_utils.py` 的 RGBA 分支会设置 `gt_alpha_mask`。主仓库与 floor40 分支均存在该入口。

因此后续无需伪造 mask，也不能仅凭 `mask_mode=none` 判断没有前景通道。进入后续训练前应逐视图核验相机 `gt_alpha_mask` 与被冻结输入 alpha 的一致性；保持额外 mask loss、背景 loss、sparse-point filtering 等开关为事先声明的状态。训练输入 alpha 与本阶段统一评价的 B_R0 heldout ROI 是不同用途，不能将 C 的训练 alpha 换作 C 的专属评价区域。

## 4. 旧 pilot 的复用范围

真实旧结果位于 `00-论文优化重构/m2m3实验/实验输出/reports/full_evaluation_summary_20260610_151144_full.csv`，共 9 行：三个场景各有 fg2dgs_baseline、fg2dgs_m2m3、fg2dgs_m2m3_floor40。可作为历史实际实现的 pilot，以及代码和 capacity 日志恢复入口。

不能把它当作本阶段冻结输入/相机/seed 下的匹配实验，不能证明未实现的数学公式，也不是原 Table 7 的 20 序列四方法平均。后续主实验应重新冻结统一输入、seed、起点、迭代数和计时边界；不挑选旧 pilot 的有利模型拼接成新结果。

## 5. 若另行实现论文双向机制

需另立 `IMPLEMENT_AND_VALIDATE` 任务，至少实现并测试：每视图可见性与遮挡、mask-to-model 支持及质量权重、model-to-mask deficit、边界距离和有效像素约束、从 deficit 到实际 Gaussian 增密的可复现规则、效用到 keep/prune/densify 的明确映射。若仍主张局部 Floor40，则需定义空间分块、参考容量、跨块规则及薄结构约束，不能继续借用全局 floor 的名字。

消融至少包括标准冻结 2DGS、真实现有加权剪枝、加权剪枝加全局 floor、双向机制各单向项与组合；只有确实新增局部机制时再加入局部 floor。必须保持共同 raw RGB 和固定 ROI，逐 seed 配对，报告额外计算与失败，不将新结果追认成旧论文已执行的实验。

## 6. 下一阶段 mesh 参考和结构评价

参考选择已在查看新增种子结果前冻结于 `configs/NEXT_PHASE_MESH_REFERENCE.json`：WanNianQing1、CaoMei2 均使用 Phase20E-1 B_R0/Seed 0；HongZhang 使用 B/C 共享的 Seed 0 实体 checkpoint。该选择是恢复和 mesh 工程参考，不表示它在任何指标上最好。本阶段没有生成 mesh。

无独立几何 GT 时，可真实报告顶点/面片数、连通分量及大小、边界与非流形边、退化面比例、同相机 mesh 与 Gaussian 的 alpha/depth 重投影差异、内部法向一致性、冻结 TSDF/提取参数的敏感性。重投影与模型间距离只能叫模型一致性或敏感性，不是几何真值精度。COLMAP 尺度仍是任意单位，不能报告真值 Chamfer、F-score 或物理面积/体积误差。

## 7. 物理尺度与表型缺项

Phase20B 审计的 `植株数据.xlsx` 有 21 个植株行、17 列人工/虚拟株高、冠幅和叶片数值，没有逐样本盆径、标尺、scale factor 或校准传递记录。`measure_traits_entry.py` 只要求外部 `--scale-marker` 并生成外部工具输入清单，未提供完整尺度恢复与测量实现。既有 12 个 mesh 日志没有补齐尺度证据。

后续物理表型仍缺：每场景真实长度及单位、明确的盆沿/标记对应定义、相机/点云/mesh 到真实长度的关联、尺度系数及不确定性、测量工具和人工参考的配对来源。本阶段不向用户追加标注要求、不猜测这些值。缺失时只报告任意单位结构描述，表型验证继续标记待补。

## 8. 原 Table 7 应降级的描述

Phase20B 已将四行旧数字标记为 `PAPER_REPORTED_UNVERIFIED`：COLMAP 13.63/0.8745/0.1072，3DGS+VFM 30.17/0.9587/0.0386，Standard 2DGS 29.58/0.9574/0.0487，ForeSplat 31.09/0.9711/0.0365。旧训练/mesh 时间也缺少共同 20 序列、划分、ROI 与计时协议的闭合证据。

后续论文修订应移除或明确降级“20 序列已验证平均”“R1 稳定提升三维重建”“M2M3 已实现原双向公式”“Floor40 局部结构保证”“统一条件下训练/mesh 加速”等表述。Phase20E-1 的三场景单种子 A/B/C 与 Phase20E-2 的两场景 B/C 三种子可以成为范围明确的新消融证据；不能直接替代原四方法 benchmark。本阶段不对 A→B 的随机性作结论，也没有新 M2M3、mesh 或物理表型结果。

## 9. 明确的实施决策

下一阶段优先执行 Phase20E3：以真实实现定义方法修正清单，先按冻结 B_R0/Seed 0 做小范围 mesh 工程恢复与结构描述。M2M3 若安排新对照，按真实加权剪枝/全局容量保护重新冻结实验，不追加未实现公式。任何双向新机制另立任务；本阶段交付后停止，不自动进入这些工作。

来源：Phase20A `REFINEMENT_CHAIN_AUDIT.md`；Phase20B `reports/M2M3_IMPLEMENTATION_TRUTH.md`、`tables/M2M3_FORMULA_CODE_MATRIX.csv`、`reports/SCALE_RECOVERY_AUDIT.md`、`reports/TABLE7_UNVERIFIED_CLAIMS.md`；本次真实源码只读复核与 `reports/M2M3_FROZEN_MASK_LOADING_AUDIT.json`。
