# GGGS 优化与 E5 5.15M Gaussian 根因审计

**Confirmed：**E5 的5,153,165是实际最终 PLY/NPZ数量，初始点数与 E1 相同；主要数量增长发生7000–15000。GGGS 官方核心只按 raw opacity<.05 剪枝，传入的 `max_screen_size=20` 没有被该函数使用；E5 最终存在大量归一化尺度很小、filtered opacity 很低而 raw opacity 达到保留阈值的 Gaussian。**Unresolved：**历史 Q=0 是否真正触发，以及这些低 filtered-alpha 点是否造成 blur。没有用数学反例冒充历史事件，也没有运行不能回答后期增长的1000步训练。

依据：本轮直接读取并逐字节比对的 [官方 GaussianModel（固定 commit）](https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids/blob/23b418c2f0f19b30b251acad6fe13445dae968c7/scene/gaussian_model.py)、[官方 train.py](https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids/blob/23b418c2f0f19b30b251acad6fe13445dae968c7/train.py)，实际 `train_foresplat.py`、六项实验日志/PLY。[01 provenance](01_experiment_provenance.md) 记录输入、真实 SH3 和状态缺失。博查本轮403；直接官方源码与本地数值实测提供证据。本轮无训练或 GPU 工作。

## 真实数量增长与损失调度

| 记录步数 | E1 精确 CSV（更新前） | E5 日志 first→last count | 证据限制 |
|---:|---:|---:|---|
| 500 | 1,252,383 | 1,252,383 | 增密起点严格大于500，第一次600 |
| 600 | 1,252,383 | 1,252,383→620,187 | 显示刷新可跨该轮操作 |
| 1000 | 506,815 | 518,924→506,695 | 早期数量近似 |
| 3000 | 286,838 | 728,035→780,398 | 已有差异、随后的 opacity reset 可改变走势 |
| 5000 | 115,892 | 190,694 | 仍远小于最终数量 |
| 7000 | 92,566 | 141,164 | E5 实际保存7k PLY=**132,104**，优先于刷新标签 |
| 10000 | 397,727 | 1,187,099→1,271,158 | 正在7000后 regularization+增密阶段 |
| 14000 | 546,637 | 3,833,521→3,958,176 | 膨胀已显著 |
| 15000 | 600,458 | 5,153,165 | 增密严格 `<15000`，最后14900 |
| 30000/final | 600,458 | 5,153,165 | E1/E5无后期FG pruning，数量不再增长 |

完整原始日志回收是 [optimization_count_timeline.json](gaussian_analysis/optimization_count_timeline.json)。tqdm 先 set_postfix 再 update，且受刷新节流影响，重复标签可能包含不同轮次的 count；保留 first/last，不伪装每个显示迭代的唯一精确状态。E5 缺少 exact component CSV，Loss/normal/NCC 只剩4位小数 EMA。最终5.15M并非更大的初始点预算，也不是训练到了15k之后还在增密。

7000 启动 .05 depth-normal / .6 NCC / .02 geo，与仍在运行的增密重叠。E1–E4 7000–14900每100步 sampled instantaneous 加权均值：

| 实验 | RGB | weighted depth-normal | weighted NCC | weighted geo | 已加权 Mask+BG | total |
|---|---:|---:|---:|---:|---:|---:|
| E1 | .09311 | .00433 | .30508 | .00067 | 0 | .40319 |
| E2 | .10872 | .00411 | .30609 | .00066 | 0 | .41957 |
| E3 | .16956 | .00590 | .30041 | .00138 | .01117 | .48842 |
| E4 | .17269 | .00490 | .30308 | .00124 | .00839 | .49030 |

表中7000–14900 CSV 重建 total 的最大误差≤1.58e−6，符合6位小数舍入，见 [loss_pruning_summary.json](gaussian_analysis/loss_pruning_summary.json)。NCC 贡献大在所有 Ω 实验同样存在；不能把这个事实直接当 E5 特有缺陷。时间重叠仅支持需观察7000→15000增密事件的假设，不证明 regularizer 单独导致5.15M。E5 EMA 不具备可恢复各精确分量或反推梯度的能力。

15000后 E1 sampled total均值.40229，相比7000–14900的.40319几乎平台，RGB均值.09311→.08819而weighted NCC仍.31041。E5显示15k EMA Loss=.4543/NCC=.5012、30k=.4805/.5234，并非明显下降；这两个端点是EMA快照，不能替代整段平均或作为保证无改善的统计检验。几何约束不匹配与图像优化瓶颈仍需实际控制实验。

实际主相机按208帧stack随机pop、无放回循环；30k每帧144或145次（恰48帧145），具体次序没有保存。PatchMatch选择保存的 `nearest_id` 中一帧；Ω四项都是208帧无空邻居（205帧8邻居、3帧7），E5有3/208空邻居（1.44%）、201帧8邻居，其他4帧各1/5/6/6。空邻居返回NCC/geo零，因此不能说E5多视角约束完全未生效。源文件名索引gap：Ω中位3/p90=17，E5中位10/p90=106；图谱不同，但相邻编号本身不是真实重叠或准确位姿证明。[multiview_sampling.json](gaussian_analysis/multiview_sampling.json)。

nearest selection 的 `.01<camera-distance<1.5` 是**绝对native世界单位**，未乘extent；LR和clone-size则乘extent。场景gauge可影响图谱，应保留此机制差别，不过有201帧8邻居不足以单凭extent推定损失异常。E1–E4的实际weighted geo仅约.00067–.00138，明显小于NCC≈.30；E5 tqdm并**没有geo字段**（只Loss/normal/NCC/count），无CSV就不能编造它的实际geo活跃量级。

## 官方增密/剪枝实际机制

1. `grads=xyz_gradient_accum/denom`、`abs=xyz_gradient_accum_abs/denom`，NaN 改为0。normal gradients 达阈值 `.0002` 的比例为 `r`；`Q=quantile(abs,1−r)`。
2. clone 使用 normal norm≥阈值且最大 sigma≤`.01*extent`。官方 clone 会按当前 scale 随机采样位置，并不是保持坐标的严格重复。
3. split 使用 `(normal≥阈值 & max_sigma>.01*extent) OR (abs≥Q)`；abs 分支**不受大尺度条件限制**。生成 N=2 子点，新尺度为旧尺度/1.6，再删除父点，净增加每个选中父点1个。
4. clone 在 split 前运行；split 将新扩展数组的 abs 用0 padding。若 Q 真为0，`>=0` 可纳入零 absolute gradients，包括无观测 denom=0 被改为0的点，以及可能的新 clone padding。该边界并非凭张量形状推断，下面 CPU 反例验证了谓词。
5. 后续 prune 只有 `get_opacity < min_opacity`，即 raw alpha<.05。函数声明接受 `max_screen_size`，但函数体不读取它；没有执行 caller 想象的20像素 screen-radius cull，也没有 world-size cull。max_radii2D 虽在训练维护，未在此处成为剔除条件。这是官方固定 commit 的机制，不是自研 adapter 引入。

原始每事件函数返回 clone新增、split净新增、opacity删除，但历史训练调用忽略返回值。总点数差分不能分解这三个数量，更不能恢复 Q、有效观测数或梯度分布。

## NCC 实际参与条件：不是仅看 lambda

本轮另外逐行读取了 `gggs_community/utils/loss_utils.py:250–304` 与执行依赖的 `submodules/warp-patch-ncc/cuda_warp_patch_ncc/warp_patch_ncc_impl.cu:232–254`。Python先要求几何一致性 `d_mask` 非空，再只取内核valid patch且 `clamp(1−cc)<.9` 的样本；内核要求patch全部在图内、源/目标两侧variance均>.000005。无valid样本返回0。`gt_mask` 并未参与这个选择，但纯黑常量patch的variance=0，会被拒绝；不能把约.30的weighted NCC直接解释成黑背景常量patch造成的固定非零损失。

实际非零CSV/EMA说明当时有patch通过这些门限并参与loss，而不是只凭配置有lambda便声称生效。历史日志没有保存每门限有效点数、前景占比或各约束的梯度；因此仍不能把标量loss还原成有效FG像素比例，或把高NCC单独确认成E5增密主因。[官方固定commit的PatchMatch实现](https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids/blob/23b418c2f0f19b30b251acad6fe13445dae968c7/utils/loss_utils.py)。

## Q=0 的真实 CPU 函数反例及边界

`optimization_zero_selection.py` 提取实际 `GaussianModel` AST：完整运行 densify_and_prune，clone/split 只保留采样之前的**原始选点表达式**；device cuda字面量改CPU，无谓词、padding、quantile条件改动。用基数模拟代替几何采样，未调用随机函数、加载模型或训练。当前 audit torch2.10.0+cu128 的 CPU 实测：

| 合成输入 | Q | clone选点 | split选点 | 模拟最终计数 |
|---|---:|---:|---:|---:|
| 4点 normal/absolute accum全0，denom=[0,0,1,1]，含大小两类scale | 0 | 0/4 | **4/4** | 4→8 |
| 只有一个真实非零normal/absolute gradient .001 | .00025 | 0/4 | 1/4 | 4→5 |

证据 [zero_selection_counterexample.json](gaussian_analysis/zero_selection_counterexample.json) 包含官方源 SHA、CPU设备、完整输入、选点布尔值；`historical_E5_trigger_confirmed=false`。它确认边界数学行为，**没有确认 E5 到达该状态**。E1–E5 的15k/最终PLY及final NPZ都没有accum/absolute accum/denom；不存在可读取的 E5 `.pth`。E6的另一算法checkpoint不能填补这一缺失。

## 最终尺度、过滤和透明度：实际全量数值

对6个最终 PLY各做一次顺序 CPU payload 读取，E1–E5为post-final、E6为iteration30000；没有读取巨量中间payload或重算权重。选择标量存为 `E*_final_scalar_cache.npy`，附列说明/源大小mtime/headerSHA以及**选择后的cache payload SHA**（不是整个原PLY校验）。全量分布在 [final_gaussian_properties.json](gaussian_analysis/final_gaussian_properties.json)，便于后续复用而不重复HDD读取。

官方 3D filter：`s'_i=sqrt(s_i²+filter²)`，`alpha_filtered=alpha_raw*prod(s_i/s'_i)`。opacity reset 把 filtered alpha压至≤.01后再乘倒数系数返回 raw空间；小sigma/highfilter可对应较大的raw alpha。原生renderer消费filtered scales/alpha；prune却用raw alpha。

| 实验 | 最终N | max sigma/own extent 中位 | raw alpha中位 | filtered alpha中位 | raw≥.05 且 filtered<.05 |
|---|---:|---:|---:|---:|---:|
| E1 | 600,458 | .00149795 | .999368 | .193514 | 104,434（17.39%） |
| E2 | 607,422 | .00140626 | .999393 | .230575 | 78,085（12.86%） |
| E3 | 1,036,619 | .00107606 | .998272 | .102026 | 285,659（27.56%） |
| E4 | 814,161 | .00110001 | .999006 | .133381 | 172,118（21.14%） |
| E5 | 5,153,165 | **.000105807** | .972636 | **.016239** | **3,055,537（59.29%）** |
| E6 | 119,836 | .00491971（2 tangent scales） | .970641 | 不适用 | 不适用 |

E5 归一化尺度中位是 E1 的约1/14.2，这排除了“只是extent整体2.504×自由尺度使绝对σ变小”的解释。E5 σ/extent>.01只有3,384（.0657%），E1有10,466（1.743%）；**5.15M主要不是大量巨大 sigma**，而是大量更小的 Gaussian。E6是2D表示，不能套GGGS三轴过滤公式或直接以sigma指标排序。

这是低 filtered-alpha 点在raw-only规则下可存活的直接参数证据，不是它们无图像贡献的证据。alpha通过前向透射叠加，许多小alpha仍可形成高覆盖/强opacity；**不得把59.29%称为无效点，也不得据此宣称删掉它们会修复blur**。最后15000后已停止raw-opacity prune，所以最终raw alpha低于.05也可以合法存在（E1 61,166、E5 290,993）。

## 黑背景与高 opacity：与实际渲染分开验证

以下颜色只用 clamped SH DC proxy `.5 + C0*f_dc`，brightness取三通道均值；实际 SH3 随视线颜色和多Gaussian叠加必须由真实renderer判断。

| 实验 | DC brightness<.03 且 raw alpha≥.5 | DC brightness<.03 且 filtered alpha≥.5 | 同时σ/extent>.01且raw alpha≥.5的近黑点 |
|---|---:|---:|---:|
| E1 | 75,653 | 56,381 | 8,544 |
| E2 | 76,435 | 59,655 | 16,948 |
| E3 | 2,058 | 825 | 15 |
| E4 | 19,107 | 7,433 | 84 |
| E5 | 929,078 | 366,276 | 3,261 |

E1/E5存在大量近黑高alpha Gaussian，E3的Mask/背景约束与更少近黑高alpha点同时出现；它们是联合不同loss配置的观察，不能从该表分离Mask loss与后期pruning的单独效果。E3后期5次只删除25,896，E4删除6,237；终点不同数量与颜色不单由“删得更多”解释。E3/E4仍可能有几何错层，不能等同完全解决植物薄结构。

黑 RGB 背景本身可被不透明黑Gaussian解释，单纯RGB loss不要求背景alpha为0。此机制与本轮真实E1/E5 alpha覆盖诊断可联合评估；不能只看黑背景RGB就认定空间透明。实际checkpoint渲染产物来自 [render_error_analysis/checkpoint_rerender_v1](render_error_analysis/checkpoint_rerender_v1)，而本表不是渲染、无伪造 alpha。

## 最小判别实验与当前结论

已准备 `scripts/optimization_instrumented_replay.py`，**未执行训练**。它包裹原官方方法，仅观察每事件 r/Q、denom0、abs0、eligible-zero、大小尺度分支和返回clone/split/prune计数，保留原随机调用；新增确定性reductions/同步会改变wall time但不主动消耗RNG。拒绝非空model_path和已存在events文件。默认600开始的event迭代映射注明仅适用于原调度；不要 `--smoke`，smoke会禁用增密和regularizer。

E5膨胀在7000→15000，1000步只覆盖早期600–900并跳过7000 regularizer，无法回答5.15M根因。本轮因此不为完成“短跑”而训练；准确判断历史zero-Q机制需要新增保持原配方、覆盖该后期窗口且实际记录事件的replay。历史无optimizer/RNG状态，不能把从7000 PLY继续训练伪装成原轨迹精确重放。

已有检查点的单因素 **offline filtered-alpha<.05删除后固定相机/RGB/剩余Gaussian重渲染** 已由主审计实际完成：固定同3视角（source0/95/189），E1删除27.58%，mean ΔFull/true-FG PSNR=−.00243/−.00744dB；E5删除64.94%，=−.08434/−.09265dB。删除比例包含raw<.05的点，故大于表中的raw≥.05且filtered<.05比例。结果见 [controlled_fixes/prune_effective_render](controlled_fixes/prune_effective_render)。没有全图或真实前景PSNR收益，Reject“直接删除这些点可修复当前blur”的简单方案；保留大量低filtered-alpha点是确认的机制，但它们不能整体视为可无损删除的冗余。此终点编辑控制不证明或否定历史Q=0。

当前分级：

- Confirmed：真实5.15M终点、同初始数量、增长时间窗口；官方raw-only prune、未使用screen-size条件；实际低filtered-alpha仍raw达保留阈值；归一化微小Gaussian分布和近黑高alpha参数存在。
- Rejected：E5初始预算更大是5.15M原因；简单extent倍数本身就是异常；标准PLY法线直接控制GGGS初始旋转/尺度；当前只用1000步即可解释后期5.15M；用E6≈E5图像指标即可排除所有表示/优化问题；offline直接删除filtered-alpha<.05可修复当前blur的简单方案。
- Unresolved：E5历史是否真正Q=0、zero-gradient split贡献比例；低filtered-alpha点的历史形成原因及是否补偿错误几何；7000 regularization、错误相机与增密之间的分别贡献。独立相机证据见 [02_camera_geometry_audit.md](02_camera_geometry_audit.md)，不能把其相机错误自动归为已证实的特定优化触发器。
