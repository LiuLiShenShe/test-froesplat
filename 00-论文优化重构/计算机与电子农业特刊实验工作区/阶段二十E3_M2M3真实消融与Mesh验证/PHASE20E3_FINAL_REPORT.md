# ForeSplat Phase20E-3：真实机制消融与植物 Mesh 验证

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

## 实际完成的工作

复用预先冻结的三个 M0（WanNianQing1/B_R0/Seed0、CaoMei2/B_R0/Seed0、HongZhang/B-C共享C_R1/Seed0）；六个新增 M1/M2 从相同 sparse points 重新训练到30000 iterations，原生29999次 optimizer steps。没有使用短测试 checkpoint、换种子或挑选有利 checkpoint。先完成三场景 M0 Mesh 和 WanNianQing1 启动诊断，再按固定顺序完成正式训练。

禁用剪枝/容量路径与标准 2DGS 等价，故预算为6次而非9次；源码证据见 reports/BASELINE_EQUIVALENCE.json。新增模型的初始 RNG、六类 Gaussian 张量、实际 alpha 与全部30000轮相机序列在 M1/M2 之间逐项一致；历史四类初始参数哈希另作独立验收，历史完整 RNG 状态仍缺失。

所有模型采用同一物理 GPU1。共享主机/磁盘负载、输入校验和逐步观察影响 wall-clock，不能把本次与历史训练时间差直接解释成算法加速；共同推理计时另见 M2M3_RENDER_TIMING_SUMMARY.csv。环境、扩展版本、GPU UUID 与哈希均保留。时间范围、PyTorch显存峰值与设备采样的区别详见 RESOURCE_METRIC_SPEC.md；这些峰值不是无观察开销的纯训练内存。

## 主 TSDF 协议

主提取在首次 GPU Mesh 工作和质量观察前冻结：bounded TSDF，depth_ratio=0，训练相机全集，冻结训练 alpha<0.5 的 depth=0；depth_trunc=2×相机半径，voxel=depth_trunc/1024，sdf_trunc=5×voxel，num_cluster=1。主参数不依模型质量修改。保留 raw 与 post；SH0 仅用于融合颜色，外观评价使用原生全SH渲染。

| 场景 | 融合相机数 | r | depth_trunc | voxel_size | sdf_trunc |
|---|---|---|---|---|---|
| WanNianQing1 | 186 | 3.0534939 | 6.1069878 | 0.0059638553 | 0.029819276 |
| CaoMei2 | 92 | 4.4754158 | 8.9508317 | 0.0087410466 | 0.043705233 |
| HongZhang | 186 | 2.7128606 | 5.4257212 | 0.0052985559 | 0.026492779 |

单位均为任意 COLMAP 单位；面积仅为其平方。没有物理尺度、独立3D GT、真值 Chamfer/F-score 或厘米级表型误差。

## 共同外观与固定前景代理

204个逐视角结果来自68个固定验证相机、9个模型。各场景表内是逐视角算术均值，验证视角不作为独立训练重复。全图采用共同 canonical undistorted raw RGB；前景采用完全相同的冻结 R0 ROI。ROI不是人工GT；composited SSIM/LPIPS受掩膜边界及卷积窗口影响。

| 场景 | 方法 | Gaussian | Full PSNR | SSIM | LPIPS | ROI PSNR | Comp. SSIM | Comp. LPIPS | ROI coverage | Outside RGB energy | Outside alpha |
|---|---|---|---|---|---|---|---|---|---|---|---|
| WanNianQing1 | M0 | 128,447 | 5.212587 | 0.260462 | 0.606881 | 23.082850 | 0.957907 | 0.098660 | 0.995379 | 0.001203 | 0.593034 |
| WanNianQing1 | M1 | 132,947 | 5.191331 | 0.257308 | 0.609313 | 22.057050 | 0.954420 | 0.101609 | 0.995924 | 0.001168 | 0.616589 |
| WanNianQing1 | M2 | 128,179 | 5.194446 | 0.257367 | 0.609306 | 22.306229 | 0.954614 | 0.101531 | 0.996043 | 0.001187 | 0.649949 |
| CaoMei2 | M0 | 130,164 | 4.955827 | 0.108415 | 0.718155 | 19.999550 | 0.963789 | 0.045793 | 0.999993 | 0.000188 | 0.977200 |
| CaoMei2 | M1 | 119,229 | 4.976612 | 0.111660 | 0.716857 | 21.815414 | 0.967165 | 0.043858 | 0.999035 | 0.000218 | 0.567434 |
| CaoMei2 | M2 | 116,663 | 4.966143 | 0.110115 | 0.717476 | 21.065523 | 0.965276 | 0.044580 | 0.999914 | 0.000210 | 0.887401 |
| HongZhang | M0 | 162,458 | 5.376382 | 0.292100 | 0.588685 | 20.322178 | 0.924833 | 0.143130 | 0.997171 | 0.001626 | 0.830942 |
| HongZhang | M1 | 147,594 | 5.381231 | 0.294562 | 0.586834 | 20.462522 | 0.927604 | 0.141139 | 0.996568 | 0.001668 | 0.734092 |
| HongZhang | M2 | 147,876 | 5.391593 | 0.296890 | 0.585257 | 21.324490 | 0.930223 | 0.138866 | 0.997097 | 0.001724 | 0.769282 |

## 主 Mesh 结构与一致性

下表包括全部原始/后处理 Mesh。post组件数由固定top-1规则直接影响，不能把组件减少解释成真实精度提高；小组件可能包含叶片。Mesh/Gaussian深度差只表示提取一致性。退化比例采用冻结的近零面积诊断阈值，与原生删除调用的判据不等同；post仍可有非零比例，没有据此额外修补。

| 场景 | 方法 | 阶段 | 顶点 | 面片 | 组件 | 最大组件面占比 | 开边 | 非流形边 | 退化面比例 | ROI可见覆盖 | 深度MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| WanNianQing1 | M0 | raw | 1,271,632 | 2,388,092 | 20,147 | 0.938626 | 121,058 | 0 | 0.000328 | 0.984294 | 0.020524 |
| WanNianQing1 | M0 | post | 1,135,076 | 2,241,526 | 1 | 1.000000 | 37,664 | 0 | 0.000137 | 0.983773 | 0.018237 |
| WanNianQing1 | M1 | raw | 1,398,086 | 2,454,080 | 51,052 | 0.894375 | 246,254 | 0 | 0.000353 | 0.984334 | 0.065784 |
| WanNianQing1 | M1 | post | 1,110,193 | 2,194,869 | 1 | 1.000000 | 33,295 | 0 | 0.000148 | 0.983411 | 0.051819 |
| WanNianQing1 | M2 | raw | 1,425,255 | 2,462,403 | 55,145 | 0.888023 | 285,255 | 0 | 0.000400 | 0.984370 | 0.065278 |
| WanNianQing1 | M2 | post | 1,105,787 | 2,186,671 | 1 | 1.000000 | 32,945 | 0 | 0.000143 | 0.983068 | 0.051245 |
| CaoMei2 | M0 | raw | 792,827 | 1,226,040 | 41,555 | 0.664840 | 282,896 | 0 | 0.000278 | 0.990304 | 0.088281 |
| CaoMei2 | M0 | post | 438,283 | 815,120 | 1 | 1.000000 | 73,304 | 0 | 0.000104 | 0.985554 | 0.117118 |
| CaoMei2 | M1 | raw | 754,317 | 1,288,056 | 21,321 | 0.743173 | 188,844 | 0 | 0.000174 | 0.987106 | 0.263768 |
| CaoMei2 | M1 | post | 506,085 | 957,248 | 1 | 1.000000 | 68,684 | 0 | 0.000077 | 0.981812 | 0.052365 |
| CaoMei2 | M2 | raw | 859,818 | 1,400,498 | 36,376 | 0.767188 | 255,514 | 0 | 0.000275 | 0.967284 | 0.159682 |
| CaoMei2 | M2 | post | 562,153 | 1,074,445 | 1 | 1.000000 | 65,089 | 0 | 0.000084 | 0.954027 | 0.157645 |
| HongZhang | M0 | raw | 2,575,549 | 4,296,140 | 126,275 | 0.766595 | 586,662 | 0 | 0.000409 | 0.981181 | 0.077134 |
| HongZhang | M0 | post | 1,672,621 | 3,293,400 | 1 | 1.000000 | 71,150 | 0 | 0.000188 | 0.977822 | 0.073957 |
| HongZhang | M1 | raw | 2,568,837 | 4,335,432 | 105,164 | 0.699867 | 591,686 | 0 | 0.000456 | 0.983156 | 0.112640 |
| HongZhang | M1 | post | 1,542,644 | 3,034,225 | 1 | 1.000000 | 64,365 | 0 | 0.000191 | 0.963314 | 0.063660 |
| HongZhang | M2 | raw | 2,204,370 | 3,713,630 | 96,581 | 0.804457 | 500,570 | 0 | 0.000454 | 0.981048 | 0.073113 |
| HongZhang | M2 | post | 1,517,300 | 2,987,455 | 1 | 1.000000 | 59,021 | 0 | 0.000193 | 0.966336 | 0.046810 |

包围盒、面积、法向代理、有效坐标比例、重投影P95和完整逐视角结果保留在 MESH_STRUCTURE_METRICS.csv、M2M3_MESH_COMPARISON.csv、metrics/。全部主Mesh可从真实PLY重新计算，最终独立重载证据位于 reports/MESH_INDEPENDENT_RELOAD.csv。

## 补充 TSDF 参数敏感性

预声明的0.5×、2× voxel仅作用三个M0；保持depth_trunc不变、sdf=5voxel，不重训且不替换主结果。不同分辨率引起结构统计变化不能被解释成真实准确率排序。

| 场景 | 补充参数 | 状态 | raw顶点 | raw面片 | raw组件 | 秒 |
|---|---|---|---|---|---|---|
| WanNianQing1 | sensitivity_voxel_0.5 | PASS | 5680101 | 10318691 | 121485 | 458.76 |
| WanNianQing1 | sensitivity_voxel_2 | PASS | 292098 | 564441 | 2626 | 69.79 |
| CaoMei2 | sensitivity_voxel_0.5 | PASS | 3438203 | 5253805 | 176540 | 195.67 |
| CaoMei2 | sensitivity_voxel_2 | PASS | 187921 | 296258 | 9032 | 19.24 |
| HongZhang | sensitivity_voxel_0.5 | PASS | 11443777 | 18561426 | 540639 | 787.32 |
| HongZhang | sensitivity_voxel_2 | PASS | 595146 | 1043066 | 23210 | 74.63 |

三个 M0 均完成预声明的 voxel 0.5x 和 2x 补充提取，6/6 个敏感性任务 PASS。改变 voxel 的同时按冻结规则改变 sdf=5*voxel，raw 顶点、面片、组件数量和运行时间均发生明显变化；因此 TSDF 协议对分辨率/截断耦合敏感。本试验不是准确率调参，也不能从两种补充分辨率选择最漂亮 Mesh 替换主结果。

## 十五项问题的实际回答

1. 是。WanNianQing1、CaoMei2、HongZhang 三个冻结 M0 均生成了非空 bounded-TSDF Mesh；新增 M1/M2 六个正式模型也全部生成并可重载。全部 9 个模型均有 checkpoint、PLY、Mesh 和结构复算记录。

2. 主 post Mesh 的顶点/面片分别为：WanNianQing1 M0 1,135,076/2,241,526，M1 1,110,193/2,194,869，M2 1,105,787/2,186,671；CaoMei2 M0 438,283/815,120，M1 506,085/957,248，M2 562,153/1,074,445；HongZhang M0 1,672,621/3,293,400，M1 1,542,644/3,034,225，M2 1,517,300/2,987,455。post 均为 1 个最大组件、有限顶点、0 条非流形边，但 raw Mesh 仍有 WanNianQing1 20,147–55,145、CaoMei2 21,321–41,555、HongZhang 96,581–126,275 个组件，开边和少量退化面仍存在。

3. 是，TSDF 对参数敏感。三场景的 voxel 0.5x/2x 补充实验均成功，但顶点/面片/组件和运行时间明显改变；由于 sdf 同步按 5*voxel 改变，不能把变化单独归因于 voxel，也不能据此挑选一个更有利的 Mesh 作为主结果。

4. 是。30/30 个正式额外剪枝事件均有 EVALUATED 分数、可用冻结 alpha 支持、预算和删除数量日志；M1/M2 每场景各 5 轮，删除数均大于 0。WanNianQing1、CaoMei2、HongZhang 的 M1/M2 总删除分别为 9,017/9,256、12,312/6,882、8,680/8,702。

5. 相对历史 M0，最终 Gaussian 数并不统一减少：WanNianQing1 M1 为 +4,500、M2 为 -268；CaoMei2 M1/M2 为 -10,935/-13,501；HongZhang M1/M2 为 -14,864/-14,582。由于 18,000 iteration 之前各运行轨迹已分化，这些最终数量不能单独解释为额外剪枝的因果效果；真实删除事件本身是可证实的。

6. held-out 全图 raw RGB 的 M1/M2 相对 M0 的 PSNR 变化为 WanNianQing1 -0.0213/-0.0181 dB、CaoMei2 +0.0208/+0.0103 dB、HongZhang +0.00485/+0.0152 dB；SSIM 与 LPIPS 也呈场景依赖方向。结果不支持统一的外观提升或退化结论。

7. 固定 R0 proxy ROI 的前景 PSNR 变化为 WanNianQing1 -1.026/-0.777 dB、CaoMei2 +1.816/+1.066 dB、HongZhang +0.140/+1.002 dB。前景覆盖变化均在约千分之一量级但方向和 outside-ROI alpha/RGB 能量不一致；ROI 是冻结预测代理而非独立 GT，故前景层级同样只能判为 MIXED。

8. Floor40 在 15/15 个 M2 事件中 active/evaluated，但累计 floor_blocked_count=0，三个场景各为 0。没有证据表明它在真实训练中改变了任何额外删除决定。

9. 没有。本次五轮 3% 预算下界始终高于全局 40% 容量下限，Floor40 未拦截删除，因此不能声称减少了实际过度剪枝。M2 与 M1 的差异不能归因于一次从未介入的 floor。

10. 有可观察但不统一的结构差异。post Mesh 的容量、边界、ROI 代理轮廓和 Gaussian 深度 MAE 在三场景方向不同；例如 CaoMei2 M1 的深度 MAE 优于 M0，而 M2 更差，HongZhang M2 的深度 MAE 较低。没有几何 GT 时，这些是结构一致性代理，不是精度排名。

11. 有。WanNianQing1 M1 虽然有 9,017 个额外删除，最终 Gaussian 仍比 M0 多 4,500；CaoMei2/HongZhang 的 M1/M2 最终数量较低，但 Mesh 和外观指标并不一致。因此“更紧凑”与“Mesh 更好”不能等同，M2M3 的容量效果是场景依赖的。

12. 目前不能把 M2M3 作为已证实的强独立创新。可以保留为实现忠实的 mask-supported weighted Gaussian pruning 和可选全局容量下限的机制 pilot；需要更多 seed、正式 baseline、控制介入前轨迹和独立几何评价后，才可能提升贡献强度。

13. 必须删除或改写 model-to-mask deficit、boundary-aware utility、bidirectional matching、local Floor40、额外 densification 以及由计数 floor 推导结构完整性的表述；保留并精确定义 alpha 中心投影支持度、各项惩罚的全局归一化、候选集合、3% 预算、额外剪枝调用和可选全局 floor 的真实代码路径。

14. Table 7 可新增本阶段三场景 pilot 的训练、共同 raw RGB、固定 ROI proxy、剪枝事件、Mesh 结构和描述性推理计时，但不能替换原四方法/20 序列平均值，也不能补齐缺失的公平 baseline。历史未验证数字继续标为未验证。

15. 四项表型仍缺少真实物理尺度及误差、尺度标定物与 COLMAP 坐标对应、重力/三维参考方向、株高/冠幅/叶长/叶宽的测量定义和器官身份、同次采集对象的人工实测值与单位、测量人员/日期/重复记录。下一阶段必须先补齐这些输入，不能把 COLMAP 任意单位换算成 cm 或 mm。

## 验收、恢复与边界

最终真实验收状态：PASS；21/21项通过。完整证据见 reports/FINAL_ACCEPTANCE.json。

全部204行外观/前景/背景代理由真实浮点重新计算，见 reports/FLOAT_METRIC_RECOMPUTATION.json。独立CPU检查9个checkpoint/PLY与18个主Mesh；5868条受保护文件和冻结配置逐个SHA256复查。测试日志、真实训练、剪枝和提取日志保留。准备错误、受限诊断及负结果见 EXECUTION_DEVIATIONS_AND_FAILURES.md。

恢复路径、命令、环境与逐文件SHA256见 RECOVERY_AND_REPRODUCTION.md、LARGE_ASSET_INDEX.csv 和 reports/RECOVERY_COMMANDS.json。大型文件位于独立外部目录，Git索引不能代替数据备份。只提交阶段脚本、配置、报告、小型图件与指标。

本阶段没有修改R1、SAM3、掩膜、GT、COLMAP、冻结划分、2DGS/M2M3生产数学逻辑或论文正文；没有将旧pilot当作本次结果。旧Table7的四方法20序列均值不被本次三场景pilot替换，详见 TABLE7_EVIDENCE_UPDATE.md。交付后立即停止，不开展Phase20F。

信息来源：本仓库冻结源码、真实运行日志、共同浮点渲染及真实Mesh；官方接口参考列于 MESH_EXTRACTION_PROTOCOL.md。科研判定记录的源文件SHA256见 reports/SCIENTIFIC_DECISION.json。
