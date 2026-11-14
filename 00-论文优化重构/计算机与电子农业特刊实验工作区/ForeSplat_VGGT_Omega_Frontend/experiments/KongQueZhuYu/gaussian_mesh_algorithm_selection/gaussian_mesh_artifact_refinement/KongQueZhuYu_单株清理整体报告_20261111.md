# KongQueZhuYu 单株清理整体报告

> 范围更正：本文件属于 2026-11-10 14:12 的旧单株任务。当前请求对应 16:30 发起的后续泛化任务，见 [泛化任务整体报告](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/reports/泛化任务整体报告_20261111.md)。

报告整理时间：2026-11-11 09:48:45 +0800。这是已有实验结果的核对报告，没有新增清理、训练或推理。

## 任务范围与操作时间

本报告对应 2026-11-10 14:12:06 发起的“Gaussian 伪影深度清理、P3 Mesh 定点修复与 Cleaned GGGS → GS2Mesh 组合验证”单株任务，范围仅为 KongQueZhuYu。

| 记录 | 本地时间（UTC+8） |
| --- | --- |
| 单株深度清理任务发起 | 2026-11-10 14:12:06 |
| 最终原生高斯 PLY 修改时间 | 2026-11-10 14:48:15 |
| 最终 SuperSplat 高斯 PLY 修改时间 | 2026-11-10 14:48:16 |
| G2 局部修复 Mesh 修改时间 | 2026-11-10 15:39:49 |
| 单株最终状态、最终判定报告修改时间 | 2026-11-10 15:59:32 |

后续 ForeSplat_AutoGaussianCleaner 泛化任务于 2026-11-10 16:30:32 发起，v3 批次在 17:15 启动，孔雀竹芋结果在 17:30 写出。用户当前要查询的是该泛化任务。本文件只保存 14:12 旧单株任务的历史情况。文件修改时间用于定位产物，操作发起时间来自本地用户操作记录；二者不能互相替代。

## 先查看的文件

- [SuperSplat 查看版：cleaned_gaussians_supersplat.ply](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/task0_gaussian_cleaning/best_cleaned/cleaned_gaussians_supersplat.ply)
- [原生高斯版：cleaned_gaussians.ply](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/task0_gaussian_cleaning/best_cleaned/cleaned_gaussians.ply)
- [当前推荐 Mesh：G2_targeted_cleaned_mesh.ply](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/task2_cleaned_gs2mesh/G2_BestCleaned_to_GS2Mesh/local_repair/G2_targeted_cleaned_mesh.ply)
- [删除点单独文件：removed_gaussians.ply](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/task0_gaussian_cleaning/best_cleaned/removed_gaussians.ply)

两份最终高斯 PLY 均为 510,248 点、128,584,052 字节，约 128.6 MB。原生文件保留 E10 中所保留点的全部原始字段；SuperSplat 版对有效尺度和透明度作查看器导出转换。已有四个代表视角的原生/导出渲染校验通过，RGB 与 Alpha 最大差小于 1/255；本次没有声称交互式查看器人工验收。

## Task 0：高斯清理结果

| 项目 | 结果 |
| --- | ---: |
| E10 原始高斯点 | 576,396 |
| P2 清理后点数 | 544,074 |
| Round 5 保留点数（含历史标记） | 513,840 |
| 最终植物＋花盆模型 | 510,248 |
| 相对 E10 累计删除 | 66,148（11.48%） |
| 相对 P2 新增删除 | 33,826 |
| 其中独立标记语义删除 | 3,592 |
| 固定八视角黑云像素总数：P2 → 最终 | 2,682,781 → 68,362 |
| 固定八视角黑云像素减少率 | 97.45% |

选中的是 Round 5 多轮重新归因背景层、再单独去除标记的版本。试验过的 Round 3 参数缩放没有进入最终模型。

97.45% 指固定八个自由相机中的黑云像素统计下降，不是高斯点删除比例，也不是三维物理面积。统计使用原 P3 轮廓作为保守参考；不能推断所有角度都已干净。

历史 208 视角中，P2 → Round 5 的背景 Alpha 为 0.331702 → 0.045932，前景 PSNR 为 19.474792 → 19.512784 dB。最终去除标记后，继续按包含标记的原 RAP 口径评价，前景 PSNR 为 18.680548 dB，低于 P2；这一回归必须保留。按补充的共同植物＋花盆区域，前景 PSNR 为 19.49629 → 19.53903 dB。不同 Mask 口径不能混作同一组提升。

清理质量结论：**有明显改善，尚未完全清干净。** 剩余约 4,011 个有明显黑色贡献的高斯，约 96.27% 的剩余黑色权重由前景/背景混合点承担。仍有细黑条、交叠黑层、盆沿错误亮纹理和局部深度变化；整点删除这些混合点可能伤及真实前景。

## Task 1 与 Task 2：Mesh 修复及组合验证

Task 1 删除原 P3 已定位独立漂浮环的 829 个三角面；指定 source157 区域的环已解决。另存去除独立标记的版本。叶片内部错误连接仍未解决，不能把局部环修复表述为整株 Mesh 验收通过。

Task 2 的 G1（P2 → GS2Mesh）和 G2（最终高斯 → GS2Mesh）均已实际完成 208 视角立体渲染、立体深度、融合和评价。G2 另做独立碎片局部清理，推荐查看 G2_targeted，文件头核对为 775,739 顶点、1,492,364 三角面。

| Mesh | IoU | 边界 F1 | 背景几何比例 | 封闭缺失投影均值（px） |
| --- | ---: | ---: | ---: | ---: |
| G0：原 E10 → GS2Mesh | 0.915556 | 0.151903 | 0.036240 | 358.31 |
| G1：P2 → GS2Mesh | 0.914219 | 0.150062 | 0.037201 | 326.98 |
| G2：最终高斯 → GS2Mesh | 0.918008 | 0.157959 | 0.033728 | 397.78 |
| G2_targeted：局部修复版 | 0.921293 | 0.163733 | 0.030164 | 347.89 |

G1 有小幅混合收益；G2 改善轮廓和背景，但未修复版缺失投影反而增加。G2_targeted 的缺失投影下降包含语义去标记影响，不能宣称叶片孔洞已被补齐。仍存在局部缺面、错误内部连接和盆沿纹理问题。

## 完成与质量判定

本次单株实验执行已经完成，产物与来源可追溯。当前高斯和 Mesh 均为改善后的查看候选；**尚未达到完整、干净三维资产的质量目标**。后续问题集中在混合贡献高斯、局部多视角深度冲突、叶片内部连接及盆沿纹理，需要分别处理。

## 本地依据

- [原始单株最终判定](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/FINAL_ARTIFACT_REFINEMENT_DECISION.md)
- [单株最终状态](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/FINAL_STATUS.json)
- [高斯清理分析及指标口径](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/task0_gaussian_cleaning/CLEANING_ANALYSIS.md)
- [高斯逐版本测量表](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/comparisons/gaussian_cleaning_metrics.csv)
- [最终高斯选择记录](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/task0_gaussian_cleaning/best_cleaned/best_manifest.json)
- [原生字段来源校验](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/task0_gaussian_cleaning/best_cleaned/NATIVE_LINEAGE_VERIFICATION.json)
- [原生与 SuperSplat 渲染校验](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/task0_gaussian_cleaning/best_cleaned/NATIVE_SUPERSPLAT_PARITY.json)
- [P3 局部修复分析](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/task1_p3_local_repair/P3_LOCAL_REPAIR_ANALYSIS.md)
- [清理高斯与 GS2Mesh 组合分析](/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/task2_cleaned_gs2mesh/COMBINATION_ANALYSIS.md)

博查检索返回 HTTP 403（余额/配额不足），搜索不可用。本报告依据本地操作记录、真实文件头及已有实验测量，没有以训练数据替代实验结果。
