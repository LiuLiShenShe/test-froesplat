# M1：H-VQG / PR-IQA 分层式植物重建视图质量门控

## 1. 模块定位

M1 不再设计成“第一步就融合 FFT、CLIP、mask 和 COLMAP 的单一 PR-IQA 分数”。这种设计有逻辑问题：如果一开始就需要 FSAM3 mask 和 COLMAP 结果，那么它就不能作为真正的前置质量筛选。

因此 M1 更新为：

> **H-VQG: Hierarchical Reconstruction-oriented View Quality Gate**

中文：

> **分层式植物重建视图质量门控机制**

PR-IQA 仍可作为论文中的总称或子称：

> **PR-IQA / H-VQG: Plant Reconstruction-oriented Image Quality Assessment via Hierarchical View Quality Gates**

核心思想：

```text
Raw Gate      : 不依赖 mask/COLMAP，剔除明显坏原始帧
Mask Gate     : FSAM3 后检查 mask 是否可靠
Geometry Gate : COLMAP/SfM 后检查几何视角是否可靠
```

论文中角色：

- 从“图像是否清晰”升级为“该视角是否对植物 3D 重建有效”。
- 避免 CLIP-IQA 或 FFT-only 被认为只是调用现有 IQA 工具。
- 形成输入质量控制 → mask-constrained 2DGS → pruning → edge-aware meshing 的闭环。

工程中角色：

- Gate 1 会真正改变进入 FSAM3/COLMAP 的原始帧集合。
- Gate 2 会标记或剔除 mask 不可靠帧。
- Gate 3 会标记或剔除几何不可靠帧。
- 所有 gate 必须可关闭，默认关闭以严格运行 baseline。

## 2. 为什么不能把 PR-IQA 全部放在第一步

原先公式：

```text
Q_i = alpha Q_freq + beta Q_clip + gamma Q_plant + delta Q_geo
```

其中 `Q_plant` 依赖 FSAM3 mask，`Q_geo` 依赖 COLMAP 或 SfM 结果。如果把这个公式作为第一步，就会出现循环依赖：

```text
第一步筛帧需要 mask/COLMAP
但 mask/COLMAP 又需要先筛帧或先运行
```

正确做法是分阶段使用不同信息：

| 阶段 | 可用信息 | 对应 gate |
|---|---|---|
| 原始帧阶段 | RGB 图像本身 | Gate 1 Raw-view Quality Gate |
| 分割后 | RGB + FSAM3 mask | Gate 2 Mask Reliability Gate |
| SfM 后 | RGB + mask + COLMAP/SfM 统计 | Gate 3 Geometry Reliability Gate |

## 3. 总体流程

```text
Raw video frames
        ↓
Gate 1: Raw-view Quality Gate
FFT + CLIP-IQA + exposure + contrast + entropy
        ↓
FSAM3 mask generation
        ↓
Gate 2: Mask Reliability Gate
foreground ratio + hole ratio + component stability + temporal consistency
        ↓
SfM / COLMAP pose estimation
        ↓
Gate 3: Geometry Reliability Gate
registration + matches/inlier ratio + reprojection error + view coverage
        ↓
Selected reconstruction-effective views
        ↓
Mask-constrained 2DGS
        ↓
Topology-aware Gaussian pruning
        ↓
Edge-aware thin-leaf meshing
```

一句话：

> M1 既是第一步，也是后续质量守门员，但不同阶段使用不同信息。

## 4. Gate 1：Raw-view Quality Gate

Gate 1 是真正的前置步骤，不依赖 mask，也不依赖 COLMAP。

输入：

```text
原始视频抽帧图像
```

指标：

| 指标 | 是否需要 mask | 是否需要 COLMAP | 作用 |
|---|---|---|---|
| `Q_fft` | no | no | 检测 motion blur / defocus blur |
| `Q_clip` | no | no | CLIP-IQA 判断 clear / blurry / low-quality / well-exposed |
| `Q_exposure` | no | no | 检测过曝 / 欠曝 |
| `Q_contrast` | no | no | 检测低对比度 |
| `Q_entropy` | no | no | 估计图像信息量 |

公式：

```text
Q_raw = alpha Q_fft
      + beta  Q_clip
      + gamma Q_exposure
      + delta Q_contrast
      + eta   Q_entropy
```

保留规则：

```text
Keep_raw(I_i) = 1 if Q_raw(I_i) > tau_raw else 0
```

Gate 1 的目标不是精细排序所有好帧，而是剔除明显坏帧：

- 严重模糊。
- 严重过曝或欠曝。
- 极低对比度。
- 信息量极低。
- CLIP-IQA 判定明显低质量或不可识别植物。

## 5. Gate 2：Mask Reliability Gate

Gate 2 在 FSAM3 mask 生成后执行。

输入：

```text
通过 Gate 1 的 RGB 图像 + FSAM3 mask
```

指标：

| 指标 | 含义 |
|---|---|
| foreground ratio | 植株占画面比例是否合理 |
| largest component ratio | mask 是否碎裂 |
| hole ratio | mask 内部孔洞是否过多 |
| boundary smoothness | 边界是否异常破碎 |
| temporal consistency | 相邻帧 mask 是否突变 |
| mask confidence | FSAM3 输出置信度，如果可获取 |

公式：

```text
Q_mask = lambda_1 Q_area
       + lambda_2 Q_component
       + lambda_3 Q_hole
       + lambda_4 Q_boundary
       + lambda_5 Q_temporal
```

保留规则：

```text
Keep_mask(I_i) = 1 if Q_mask(I_i) > tau_mask else 0
```

Gate 2 主要处理：

- 图像清晰但 mask 错误。
- mask 把花盆、桌面、支架大量纳入。
- mask 断裂或碎片过多。
- mask 内部孔洞异常。
- 相邻帧 mask 突然大幅变化。
- 植株主体不完整。

## 6. Gate 3：Geometry Reliability Gate

Gate 3 在 SfM / COLMAP 后执行。

输入：

```text
通过 Gate 1 和 Gate 2 的图像 + SfM/COLMAP 结果
```

指标：

| 指标 | 含义 |
|---|---|
| registration success | 当前帧是否成功注册 |
| matched keypoints | 与相邻视角匹配点数量 |
| inlier ratio | RANSAC 内点比例 |
| reprojection error | 重投影误差 |
| view coverage | 视角覆盖是否均匀 |
| camera pose outlier | 相机位姿是否异常跳变 |

公式：

```text
Q_geo = mu_1 Q_reg
      + mu_2 Q_match
      + mu_3 Q_inlier
      - mu_4 E_reproj
      + mu_5 Q_coverage
```

保留规则：

```text
Keep_geo(I_i) = 1 if Q_geo(I_i) > tau_geo else 0
```

Gate 3 主要处理：

- 图像和 mask 都不错，但 COLMAP 注册失败。
- 当前帧匹配点很少。
- RANSAC 内点比例低。
- 重投影误差大。
- 相机位姿跳变，可能污染 2DGS 训练。
- 视角冗余或覆盖不均衡。

## 7. 实现接口

推荐参数：

```bash
--view_quality_mode {none,raw,raw_clip,hvqg}
--raw_gate_mode {none,fft,fft_clip,full}
--mask_gate_mode {none,basic,temporal}
--geo_gate_mode {none,registered,reproj,full}

--raw_gate_threshold <float>
--mask_gate_threshold <float>
--geo_gate_threshold <float>

--raw_gate_keep_ratio <float>
--mask_gate_keep_ratio <float>
--geo_gate_keep_ratio <float>

--raw_gate_weights <fft,clip,exposure,contrast,entropy>
--mask_gate_weights <area,component,hole,boundary,temporal>
--geo_gate_weights <reg,match,inlier,reproj,coverage>

--clip_iqa_model <name-or-path>
--clip_iqa_prompt_set {generic,plant,plant_reconstruction}
--mask_dir <path>
--colmap_dir <path>
--view_quality_report <path>
```

默认：

```bash
--view_quality_mode none
--raw_gate_mode none
--mask_gate_mode none
--geo_gate_mode none
```

不同模式：

| 模式 | 行为 | 用途 |
|---|---|---|
| `none` | 不启用任何 gate | baseline |
| `raw` | 只启用 Gate 1，通常 FFT/exposure/contrast | 前置初筛 |
| `raw_clip` | Gate 1 加 CLIP-IQA | 检查 CLIP perceptual branch |
| `hvqg` | Gate 1 + Gate 2 + Gate 3 | 完整分层质量门控 |

## 8. 输出文件

建议输出：

```text
数据管理/06-实验输出/<sample>/M1_hvqg/
├── gate1_raw/
│   ├── retained_frames.txt
│   ├── rejected_frames.txt
│   ├── raw_quality_scores.csv
│   └── rejected_raw_examples.png
├── gate2_mask/
│   ├── retained_frames.txt
│   ├── rejected_frames.txt
│   ├── mask_reliability_scores.csv
│   └── rejected_mask_examples.png
├── gate3_geometry/
│   ├── retained_frames.txt
│   ├── rejected_frames.txt
│   ├── geometry_reliability_scores.csv
│   └── rejected_geometry_examples.png
├── hvqg_final_frames.txt
├── hvqg_config.yaml
└── hvqg_summary.json
```

核心字段：

| 文件 | 关键字段 |
|---|---|
| `raw_quality_scores.csv` | image_name, Q_fft, Q_clip, Q_exposure, Q_contrast, Q_entropy, Q_raw, decision |
| `mask_reliability_scores.csv` | image_name, foreground_ratio, largest_component_ratio, hole_ratio, boundary_smoothness, temporal_consistency, Q_mask, decision |
| `geometry_reliability_scores.csv` | image_name, registered, matched_keypoints, inlier_ratio, reprojection_error, view_coverage, Q_geo, decision |

## 9. 消融设计

质量门控消融表：

| Method | Raw Gate | Mask Gate | Geometry Gate | Retained frames | Registered views | PSNR up | LPIPS down | Mesh time down | Trait MAPE down |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| no filtering | no | no | no | 100% |  |  |  |  |  |
| FFT only | yes | no | no |  |  |  |  |  |  |
| FFT + CLIP-IQA | yes | no | no |  |  |  |  |  |  |
| Raw + Mask Gate | yes | yes | no |  |  |  |  |  |  |
| Raw + Mask + Geometry Gate | yes | yes | yes |  |  |  |  |  |  |

Gate 1 子消融：

| Method | FFT | CLIP-IQA | Exposure | Contrast | Entropy |
|---|---|---|---|---|---|
| FFT only | yes | no | no | no | no |
| FFT + CLIP | yes | yes | no | no | no |
| Raw Gate full | yes | yes | yes | yes | yes |

Gate 2 子消融：

| Method | foreground ratio | component | hole | boundary | temporal |
|---|---|---|---|---|---|
| area only | yes | no | no | no | no |
| area + component + hole | yes | yes | yes | no | no |
| Mask Gate full | yes | yes | yes | yes | yes |

Gate 3 子消融：

| Method | registered | matches | inlier | reproj | coverage |
|---|---|---|---|---|---|
| registration only | yes | no | no | no | no |
| registration + reproj | yes | no | no | yes | no |
| Geometry Gate full | yes | yes | yes | yes | yes |

## 10. 指标

输入与 SfM：

- retained frame ratio after each gate
- registered images
- sparse points
- keypoint number
- matched keypoints
- inlier ratio
- reprojection error
- SfM failure rate
- view coverage uniformity

渲染与网格：

- PSNR
- SSIM
- LPIPS
- train time
- mesh time
- Gaussian number

表型：

- plant height MAE
- canopy width MAE
- leaf length MAE
- leaf width MAE/MAPE/Bias

## 11. 论文写法

2026-05-20 更新：当前 `A6+M1/H-VQG hard filtering` 与 `A6+M1-reject-only/raw-mask` 在 KongQueZhuYu 上均已形成负证据。简单 top-score hard filtering 显著降低 foreground-object 指标并提高 leakage；去掉 geometry hard delete 后，raw/mask retained-list 仍明显失败。这说明 M1 不能写成“高分图像筛选器”或 retained-list 硬删模块。后续论文写法应强调 reconstruction-effective foreground view selection，即优先保留多视角覆盖，并通过 soft view weighting 调节退化视角贡献；若必须删图，则必须先满足 coverage-balanced anchor-view 约束。

推荐英文：

> We propose a hierarchical reconstruction-oriented view quality gate that progressively evaluates raw perceptual quality, mask reliability, and multi-view geometric consistency. Unlike frequency-only filtering or generic IQA, the proposed gate controls view quality at the raw-image, semantic-mask, and SfM-geometry stages, enabling the downstream 2D Gaussian optimization to use reconstruction-effective plant views.

推荐中文：

> 本文提出分层式植物重建视图质量门控机制，分别在原始图像阶段、语义分割阶段和多视角几何阶段评估视角质量。与仅基于 FFT 的模糊检测或通用 IQA 不同，该机制将原始感知质量、mask 可靠性和 SfM 几何稳定性逐级纳入筛选过程，实现从“清晰帧筛选”到“重建有效视角筛选”的转变。

不建议写：

> We use CLIP-IQA to improve frame filtering.

## 12. 创新性边界

| 写法 | 创新强度 | 建议 |
|---|---|---|
| FFT-only | 弱 | 作为基础前置分支 |
| CLIP-IQA-only | 弱 | 只作为 Gate 1 消融 |
| FFT + CLIP-IQA | 中等 | 作为 Raw Gate 增强 |
| 单一 PR-IQA full 打分 | 有逻辑风险 | 不作为主设计 |
| hard-gated H-VQG | 负证据 | 作为 hard filtering 强负对照 |
| reject-only raw/mask retained list | 负证据 | 作为 retained-list 弱负对照 |
| soft view weighting | 较强 | 当前下一版 M1 第一主候选 |
| coverage-balanced anchor selection | 较强 | 若必须删图，作为覆盖约束下的第二主候选 |

H-VQG 的创新性来自：

1. 不做通用 IQA，而是面向 plant reconstruction。
2. 不把 mask/COLMAP 强行放在第一步，而是阶段递进。
3. 同时覆盖 raw image quality、mask reliability 和 geometry reliability。
4. 用 downstream reconstruction 和 phenotypic accuracy 验证每层 gate 的价值。

## 13. 验收标准

M1 成功的最低标准：

- Gate 1 可以在没有 mask/COLMAP 的情况下独立运行。
- Gate 2 可以在 FSAM3 mask 后独立运行。
- Gate 3 可以在 COLMAP 后独立运行。
- `none` 模式不改变 baseline。
- hard-gated H-VQG 与 reject-only raw/mask retained list 作为负对照保留，不计入有效 M1。
- M1-v2 / M1-v3 相比 no filtering、FFT-only、M1-hard 和 M1-reject-only 至少在以下 2 项改善：
  - COLMAP registered views 增加或 failure rate 降低。
  - retained views 覆盖更完整，empty angular bins 更少。
  - sparse points / inlier ratio / reprojection error 更稳定。
  - foreground PSNR/SSIM/LPIPS 改善或不下降。
  - leakage / outside 不超过 foreground-only 分离阈值。
  - mesh time 降低。
  - trait MAE/MAPE 降低。

风险：

- Raw Gate 或 Mask Gate retained-list 过强会破坏视角覆盖或训练/评估视角分布。
- Mask Gate 依赖 FSAM3，错误 mask 会误删可用帧。
- Geometry Gate 如果只保留注册成功帧，可能导致视角分布不均。
- CLIP-IQA 分支可能偏向“好看”，不一定偏向“可重建”。
- top-score hard filtering 可能删掉侧面、背面、顶部或遮挡补充视角，导致 foreground-object reconstruction 失败。

缓解：

- 每层必须报告 retained frame ratio。
- Gate 3 需要加入 view coverage，避免只按误差贪心删帧。
- 每个 gate 都支持 threshold、真正 reject-only severe degradation、coverage-balanced 和 soft-weighting 对照。
- 正式实验必须报告 no filtering / FFT-only / M1-hard / M1-reject-only / coverage-balanced / soft-weighting。
- M1 报告中必须增加 retained views、azimuth coverage、empty angular bins、min views per bin 和 pose graph connectivity。

## 14. 参考文献候选

引用需要在正式写作前再次核对 BibTeX：

- CLIP-IQA：Jianyi Wang, Kelvin C. K. Chan, Chen Change Loy, *Exploring CLIP for Assessing the Look and Feel of Images*, arXiv:2207.12396, 2022. https://arxiv.org/abs/2207.12396
- MUSIQ：Junjie Ke et al., *MUSIQ: Multi-Scale Image Quality Transformer*, ICCV 2021. https://openaccess.thecvf.com/content/ICCV2021/html/Ke_MUSIQ_Multi-Scale_Image_Quality_Transformer_ICCV_2021_paper.html
- MANIQA：Sidi Yang et al., *MANIQA: Multi-Dimension Attention Network for No-Reference Image Quality Assessment*, CVPR Workshops 2022. https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Yang_MANIQA_Multi-Dimension_Attention_Network_for_No-Reference_Image_Quality_Assessment_CVPRW_2022_paper.html
- TOPIQ：Chaofeng Chen et al., *TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment*, arXiv:2308.03060 / IEEE TPAMI. https://arxiv.org/abs/2308.03060
