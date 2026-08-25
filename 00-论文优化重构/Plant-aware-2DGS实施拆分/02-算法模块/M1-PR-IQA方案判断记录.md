# M1 H-VQG / PR-IQA 方案判断记录

更新日期：2026-05-20

## 1. 判断结论

原来的 PR-IQA 方案方向是对的，但需要修正结构：

- **不应**把 FFT、CLIP-IQA、mask completeness、COLMAP geometry 全部放进第一步统一打分。
- 如果第一步需要 FSAM3 mask 和 COLMAP 结果，就会出现前后依赖倒置。
- **建议**改为分层式质量守门员机制：

```text
Gate 1 Raw-view Quality Gate      : 原始帧初筛，不依赖 mask/COLMAP
Gate 2 Mask Reliability Gate      : FSAM3 后检查 mask 是否可靠
Gate 3 Geometry Reliability Gate  : COLMAP/SfM 后检查几何视角是否可靠
```

2026-05-20 复盘：分层 gate 的工程结构仍然合理，但 `KongQueZhuYu A6+M1/H-VQG hard filtering` 与 `A6+M1-reject-only/raw-mask` 均已证明 retained-list hard filtering 会破坏 foreground-object reconstruction。因此 H-VQG 不能再被理解为“按分数硬删视角”的模块，而应改成 soft view weighting 或 coverage-balanced anchor-view selection 的 reconstruction-effective foreground view selection。

最终建议纳入 M1 的不是 hard filtering，而是：

> **H-VQG: Hierarchical Reconstruction-oriented View Quality Gate**

中文：

> **分层式植物重建视图质量门控机制**

PR-IQA 可以保留为论文中的总称，但主设计应强调 hierarchical gates。

## 2. 为什么这样更合理

植物三维重建的质量问题不是单阶段问题：

| 阶段 | 可能失败的问题 | 对应 gate |
|---|---|---|
| 原始图像阶段 | 模糊、曝光差、低对比度 | Raw-view Gate |
| 分割阶段 | mask 错、mask 破碎、植株不完整 | Mask Gate |
| 几何阶段 | COLMAP 注册失败、姿态漂移、匹配不足 | Geometry Gate |
| Gaussian/mesh 阶段 | 背景 Gaussian、漂浮点、边缘膨胀 | M4 pruning + M5 edge-aware meshing |

这样可以解决原方案的逻辑问题：

```text
不是第一步就需要 mask/COLMAP，
而是每个阶段只使用当时已经可用的信息。
```

## 3. Gate 1 的定位

Gate 1 是真正的第一步。

输入：

```text
原始视频抽帧图像
```

指标：

- FFT sharpness
- exposure score
- contrast score
- image entropy
- CLIP-IQA perceptual quality

公式：

```text
Q_raw = alpha Q_fft + beta Q_clip + gamma Q_exposure + delta Q_contrast + eta Q_entropy
```

目标：

- 剔除严重模糊、过曝、欠曝、低对比度、明显低质量帧。
- 不做过强筛选，避免破坏视角覆盖。

## 4. Gate 2 的定位

Gate 2 在 FSAM3 后执行。

输入：

```text
Gate 1 保留图像 + FSAM3 mask
```

指标：

- foreground ratio
- largest component ratio
- hole ratio
- boundary smoothness
- temporal consistency
- mask confidence，如果可获取

目标：

- 识别清晰但 mask 错误的帧。
- 避免错误 mask 进入 M3 mask loss、M4 pruning 和 M5 meshing。

## 5. Gate 3 的定位

Gate 3 在 COLMAP/SfM 后执行。

输入：

```text
Gate 1/2 保留图像 + COLMAP/SfM 结果
```

指标：

- registration success
- matched keypoints
- inlier ratio
- reprojection error
- view coverage
- camera pose outlier

目标：

- 识别注册失败、重投影误差大、位姿异常的帧。
- 让最终进入 2DGS 的是 reconstruction-effective views。

## 6. 创新性分层

| 方案 | 创新强度 | 是否建议作为贡献 |
|---|---|---|
| FFT-only | 弱 | 作为基础分支 |
| CLIP-IQA-only | 弱 | 只作为消融 |
| FFT + CLIP-IQA | 中等 | 作为 Raw Gate 增强 |
| 单一 PR-IQA full 打分 | 有逻辑风险 | 不作为主设计 |
| hard-gated H-VQG | 负证据 | 作为简单硬筛强负对照 |
| reject-only raw/mask retained list | 负证据 | 作为 retained-list 弱负对照 |
| soft view weighting | 较强 | 当前下一版 M1 第一主候选 |
| coverage-balanced anchor selection | 较强 | 若必须删图，作为覆盖约束下的第二主候选 |

H-VQG 的创新性来自：

1. 面向 plant reconstruction，不是通用 IQA。
2. 阶段递进地评估 raw quality、mask reliability、geometry reliability。
3. 避免第一步依赖后续 mask/COLMAP 的逻辑矛盾。
4. 用 downstream reconstruction 和 phenotypic accuracy 验证。

## 7. 最小可行版本

如果时间有限，先实现：

1. Gate 1 full：FFT + CLIP-IQA + exposure + contrast。
2. Gate 2 basic：foreground ratio + largest component ratio + hole ratio。
3. Gate 3 basic：registration success + reprojection error + coverage constraint。

先不做：

- 复杂学习型权重。
- 大规模 prompt 搜索。
- MUSIQ / MANIQA / TOPIQ 全量外部 baseline。
- 复杂 view coverage optimization，但必须保留最小 coverage-balanced 分桶约束。

最小实验：

| Method | 必跑 |
|---|---|
| no filtering | yes |
| FFT-only | yes |
| hard-gated H-VQG | yes，作为强负对照 |
| reject-only raw/mask retained list | yes，作为弱负对照 |
| reject-only severe degradation | yes，必须是真正极端坏帧删除，不复用激进 retained list |
| soft view weighting | yes，下一版优先 |
| coverage-balanced selection | yes，若必须删图则必须先保覆盖 |

## 8. 论文写法

推荐英文：

> We propose a hierarchical reconstruction-oriented view quality gate that progressively evaluates raw perceptual quality, mask reliability, and multi-view geometric consistency, enabling quality control from raw frame selection to reconstruction-effective view selection.

推荐中文：

> 本文提出分层式植物重建视图质量门控机制，将原始图像质量、语义分割可靠性和多视角几何稳定性统一纳入输入筛选过程，实现从“清晰帧筛选”到“重建有效视角筛选”的转变。

不要写：

> 本文使用 CLIP-IQA 替代 FFT。

## 9. 风险与缓解

| 风险 | 缓解 |
|---|---|
| Raw Gate 过强导致视角覆盖不足 | 报告 retained ratio 和 registered views，支持 keep-ratio |
| CLIP-IQA 偏向视觉美学，不一定偏向重建 | 加入 Geometry Gate 和 downstream 验证 |
| Mask Gate 依赖 FSAM3 质量 | Gate 2 只作为可开关项，做消融 |
| Geometry Gate 只按误差删帧可能导致视角不均 | 加入 view coverage 或保留均匀采样约束 |

## 10. 引用核对状态

已初步确认以下方向合理，但正式写论文前需要重新下载 BibTeX 并核对作者、年份、venue：

- CLIP-IQA：*Exploring CLIP for Assessing the Look and Feel of Images*, arXiv:2207.12396。链接：https://arxiv.org/abs/2207.12396
- MUSIQ：*MUSIQ: Multi-Scale Image Quality Transformer*, ICCV 2021。链接：https://openaccess.thecvf.com/content/ICCV2021/html/Ke_MUSIQ_Multi-Scale_Image_Quality_Transformer_ICCV_2021_paper.html
- MANIQA：*MANIQA: Multi-Dimension Attention Network for No-Reference Image Quality Assessment*, CVPR Workshops 2022。链接：https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Yang_MANIQA_Multi-Dimension_Attention_Network_for_No-Reference_Image_Quality_Assessment_CVPRW_2022_paper.html
- TOPIQ：*TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment*, arXiv:2308.03060 / IEEE TPAMI。链接：https://arxiv.org/abs/2308.03060
