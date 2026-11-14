# Development version record

## 2026-11-12：C12 主样本匹配对照完成，跨植物仍在开发验证

C12 的 C5-A/B/C/D 已全部从 E10 原始576,396点独立完成，同一208相机、原生GGGS滤波渲染器和960最大边进行统一评价。A删除212,580点，但前景MSE增加61.28%、暗前景MSE增加274.03%；B零修改。C通过37次分组修改，六轮后实际删除50,486点，背景Alpha mass降低38.0634%，FG-PSNR增加0.01007 dB，Thin Coverage不变。C的437次试删中有400次拒绝；局部照片约束、暗前景/边缘保护和最小分组收益仍然实际执行。

D的72个局部协方差优化检查点全部拒绝，最终原生PLY与C字节一致，没有最终通过的参数修改。D额外耗时5,737.43秒（+41.71%），目前主样本没有证明混合优化值得这项成本。C/D直接查看六个照片视角和八个自动自由视角后仍有俯视/斜视黑云、叶旁条带及碎点；不能宣称已清除全部黑点或全面优于历史C3。KNN在C/D的六轮中没有带来额外通过保护检查的删除候选。

完整主样本实测、模型路径和优化拒绝统计分别见`experiments/GGGS_Cleaner_Comparison/comparisons/MAIN_C0_C5_MEASURED_REPORT.md`、`MAIN_C0_C5_MEASURED_RESULTS.csv`及`MAIN_C5_TRIAL_AND_REFINEMENT_AUDIT.json`。19/19株原生GGGS 30K输入已通过格式/参数核验。六株开发集使用固定C12-C；十三株独立测试尚未开始，尚未推广正式全局版本。网纹草1提前与串行开发工作重叠执行，使用相同冻结源码/配置，完成后由未改动的批次resume路径重新核验；这是资源调度，不是按植株调参。所有时间保留共享工作站并发范围。

本轮通过Context7读取uv当前官方环境保留说明：https://github.com/astral-sh/uv/blob/main/docs/concepts/projects/sync.md 。博查搜索返回HTTP403；实验结论依据本地源码与实际测量，未用训练数据替代实测。

v1 candidates reduced total background mass but failed measured leaf-boundary coverage and local dark-foreground/geometry constraints on KongQueZhuYu. No heldout cleaner was run. These candidates were rejected and retained.

v2 applies one shared change: A2 must have no measured boundary/thin protection; background-dominant Gaussians with actual sensitive foreground contribution become A3 for bounded covariance optimization. This is a feature rule for every plant, not a Kong ROI, ID or world-space threshold. Six meaningful tests pass, including the high-background boundary-support regression. Source/config hashes are re-frozen under all_19_plants_v2. Previous v1 artifacts remain unchanged.

Independent heldout set remains the same 15 plants; no heldout outputs informed this change. Method benefit is still pending measured C/B and cross-plant results.


v3: among v2 A2, 44 Gaussians had >=2 actual footprint depth-support views. Whole-point invalidity was inconsistent with this evidence. The same global A2 rule now excludes multi-view depth-supported/occluded contributors; after a quality-rejected prune, actual foreground contributors are offered constrained local refinement. Thresholds and all plant input data stay unchanged. Seven native CUDA tests pass. No heldout output has been read. v2 remains archived under its original output root.


## 本轮 GGGS-only 对照与 C11 开发记录（尚未正式选型）

本轮统一输入 E10 原始 576,396 Gaussian；旧 v3 的 4/15 开发/测试记录保留为历史，本轮清理测试前固定 6 株开发、13 株独立测试，清单见 experiments/GGGS_Cleaner_Comparison/cross_plant_validation/plant_manifest.json。尚未启动独立测试，不使用测试结果调参。

C1/C2 原始、尺度、分级、KNN 归一化对照均已独立从原始 GGGS 完成。三轴滤波有效尺度、Opacity 和完整 SH 保留；SH DC 只是候选外观近似。外观规则去黑云同时损害前景照片和局部覆盖，不能直接用于生产删除。

C4 真实记录显示发现候选有效，但全批混合、全图收益阈值、前景/背景混合和局部深色/几何约束阻止接受。50% 批次质量通过但背景收益 .19146076% 低于 .2% 门槛。C5 保留质量约束，使用空间分组、真实原生贡献、受损像素驱动细分和删除后重新归因。

C11 从冻结的原始渲染建立照片相对误差/深度支持，支撑不足保守退回全 Mask；所有被排除像素另外受固定 1/255 RMSE 增量约束。梯度只提出拆分，仍由完整 208 相机原生重渲染接受。原先失败的 288 候选经自动细分产生 80 个通过候选，独立原型和完整运行复现一致 ID/收益，无原型 ID 或人工 ROI 被传入生产发现。

C11 A/B 已实际完成：A 删除但不满足前景质量约束，B 零修改。C/D 多轮比较仍在执行。当前所分析 C9–C11 轮次中 KNN 对安全剪枝的额外候选为零，不能宣称 KNN 已带来最终收益。混合协方差优化是否值得成本仍待完整 D 消融。

C11 使用现有工程的隔离源码快照避免污染正在运行的 C9/C10；正式通过开发/独立测试后再合回原工程，不能把开发快照称为另一套最终清理项目。批量接口新增可选两株并发，仍只在 GPU1，各株隔离原模型和全视角评价；状态及来源冻结经只读复核修正。正式全局参数选择仍待六株开发结果，最终可用模型/泛化结论尚未交付。
