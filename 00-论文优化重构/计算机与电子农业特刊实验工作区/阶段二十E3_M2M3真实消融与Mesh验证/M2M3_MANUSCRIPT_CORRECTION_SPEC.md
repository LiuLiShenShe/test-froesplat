# M2M3 方法表述修正规格（本阶段不修改论文正文）

## 可保留的真实机制

本阶段依据 `2d-gaussian-splatting-m2m3-floor40/train.py`，不是依据旧论文公式反推实现。源码定位、SHA256 和每项映射见 `M2M3_CODE_FORMULA_MATRIX.csv`；实际调用与计数见 `M2M3_PRUNING_EVENT_LOG.csv`、各次初始化记录及外部模型目录的 `pruning/`、`capacity_control/`。

建议方法名称为“mask-supported weighted Gaussian pruning with optional global capacity floor”（掩膜支持度加权高斯剪枝与可选全局容量下限）。原名称若保留，必须明确它是此实现的标签，不能暗示双向匹配或局部容量保护。标准 2DGS 仍负责优化、常规增密与剪枝；本模块只增加离散剪枝。

## 与代码一致的数学定义

在一次实际额外剪枝调用中，当前 Gaussian 集为 $G=\{1,\ldots,N\}$。对由固定 stride 选取的最多12个训练相机，令 $v_{ic}$ 表示 Gaussian 中心在相机前方且投影严格落入 NDC $(-1,1)^2$。这里不进行遮挡检测，不能称为真实可见表面支持。令 $a_c(\pi_c(x_i))$ 为 `grid_sample(mode="nearest", align_corners=True)` 读取的冻结 alpha：

$$s_i=\begin{cases}
\frac{\sum_c v_{ic}\mathbf1[a_c(\pi_c(x_i))>0.5]}{\sum_c v_{ic}},&\sum_c v_{ic}>0,\\
1,&\sum_c v_{ic}=0.
\end{cases}$$

因此没有有效投影的 Gaussian 默认支持度为1，而不是0。训练 alpha 来自原 loader 缩放的 RGBA，与用于评价的最近邻二值 R0 ROI 是不同用途的固定数据表示。运行包装逐相机验证实际加载的 alpha 哈希，不更改掩膜。

对任一非负惩罚向量，原生归一化为：

$$\mathcal N(z)_i=\frac{[z_i]_+}{\max(\max_{j\in G}[z_j]_+,10^{-6})}.$$

最大值在全部当前 Gaussian 上计算，不能写成仅在候选集合或局部区域归一化。所有评分使用 detach 后的 float32，不反向传播评分梯度。

设 $\alpha_i=\mathrm{sigmoid}(\_opacity_i)$、$\ell_i=\max_k\exp(\_scaling_{ik})$、$r_i=\mathrm{max\_radii2D}_i$，SH DC 亮度为

$$b_i=\frac13\sum_{c=1}^3\mathrm{clip}(0.28209479177387814\,f^{DC}_{ic}+0.5,0,1).$$

冻结实验的实际剪枝分数为：

$$q_i=\mathcal N([0.005-\alpha_i]_+)_i+\mathcal N([0.01-b_i]_+)_i
+3\mathcal N([0.45-s_i]_+)_i
+0.5\mathcal N([\ell_i-\mathrm{median}_{torch}(\ell)]_+)_i
+0.5\mathcal N([r_i]_+)_i.$$

`torch.median` 在偶数长度时取较小的中位元素；屏幕半径是标准训练维护的 `max_radii2D` 状态，不是多视角重投影误差或边界项。上述权重为旧 pilot 实际命令与该版本源码默认值的可追溯组合，参数来源见 `configs/PILOT_PARAMETER_SOURCE.txt`，没有继承旧 pilot 不同的损失、分辨率或初始化。

本实验 `pruning_mode=mask`，实际候选集合为

$$C=\{i\in G:s_i<0.45\}.$$

尽管分数包含 opacity 与 brightness，二者在 mask 模式下不单独创建候选。不能把其他 `topology` 模式的候选并集写成本次正式算法。令 $B=\lfloor0.03N\rfloor$；当 $B>0$ 时，$C_B$ 是 $C$ 中分数最大的至多 $B$ 个 Gaussian。若候选数已小于预算，原生函数直接返回原集合；Top-k 仅在超预算时执行。代码对 $B=0$ 会跳过此预算限制，需如实保留该边界行为；本次实际模型规模不涉及这个边界。

M1 直接对 $C_B$ 调用 `prune_points`。M2 在冻结活动时间窗内进一步计算：

$$F_t=\min\{N,\max(0,\mathrm{round}(0.4N_{max\_seen}))\},\qquad A_t=\max(0,N-F_t),$$

并在 $C_B$ 中最多删除分数最大的 $A_t$ 个 Gaussian。`round` 是 Python 的原生舍入；`N_max_seen` 按实际 `update_capacity_state` 调用维护，包含初始容量及记录到的训练容量，不是人为指定的“目标完整性”容量。floor 只在 `capacity_control_mode=m2m3_floor` 且处于活动窗时参与这一额外剪枝路径。候选为空时原生函数提前返回，不计算评分或 floor。

本阶段额外剪枝在 18000、21000、24000、27000、30000 iterations 执行。调用顺序为：训练损失与反向传播 → 标准 2DGS densification/pruning（仅原有时间窗）→ 更新容量 → 优化器步（iteration<30000）→ 额外 M2M3 剪枝。最终 PLY 在末轮额外剪枝后重新保存。30000 次迭代对应原生 29999 个 optimizer steps，不能把这一继承行为改写为新模块修改。

## 必须删除或降级的历史表述

| 历史表述 | 真实证据 | 后续论文动作 |
|---|---|---|
| Mask-to-model support | Gaussian 中心在部分相机的 alpha 支持比例 | 降级为中心投影支持代理，补充无可见投影默认1和无遮挡检测限制 |
| Model-to-mask deficit | 没有独立 deficit 图、目标函数或增密入口 | 删除已实现和已验证的表述 |
| Boundary-aware utility | 实际评分无边界距离项 | 删除对应公式及因果解释 |
| Bidirectional matching | 仅单向 alpha 支持与额外剪枝 | 删除双向优化、双向收敛等描述 |
| Local Floor40 | 仅一个全局计数下限 | 改写为可选全局容量下限，限定作用路径 |
| Gaussian keep/prune/densify 联合机制 | 额外模块仅 prune；增密来自原生 2DGS | 拆清标准训练与额外剪枝，不归功于未实现机制 |
| Floor40 保证结构完整 | 计数保护不能保证叶片或局部结构 | 仅在实际 floor_blocked_count>0 时讨论容量干预，正向结构收益另需实验 |
| 三维真值精度提高 | 没有独立三维 GT 或物理尺度 | 仅报告外观、ROI 代理、结构及提取一致性，禁止真值 Chamfer/F-score |

## 贡献强度与修订边界

本阶段每个场景仅 Seed0，三场景构成机制 pilot。即使某项指标改善，也不能扩张为跨种子稳定收益或20序列结论。Floor 参数开启、逻辑进入、真正拦截删除以及正向收益是四个不同层次；最终结论必须由实际事件与共同评价分别支持。若本次不能支持强独立贡献，应将其降级为可选紧凑化模块或实现细节，并删除未经执行的创新表述。

准确结果与贡献判定在 `M2M3_METHOD_DECISION.md` 和 `M2M3_CAPACITY_EFFECT_REPORT.md` 中逐场景列出；本规格中的公式只描述真实实现。若未来确需双向 deficit、边界 utility 或局部容量保护，必须另立实现、测试、消融任务，不能将未来机制写成既有实验。

容量参考的精确定义：`max_seen_count` 是初始化后、每次训练迭代在标准 densify/prune 完成后调用 `update_capacity_state` 所观察到的最大 Gaussian 数，以及额外剪枝前/结束时的更新；不是标准增密算子内部任意瞬时容量峰值。额外剪枝从 18000 开始，而标准增密在 15000 前结束，所以本阶段五轮额外剪枝共享已观测的峰值参考。`M2M3_NATIVE_CAPACITY_SUMMARIES.csv` 和逐事件 `reference_max_seen_count` 提供真实运行来源；最终验收按 native summary 重算每轮 floor 和阻止删除数。
