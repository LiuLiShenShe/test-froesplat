# Plant-aware 2DGS 阶段思路与实验产物总结 v1.0

## 0. 文档定位

本文档用于汇总 Plant-aware 2DGS 从问题提出、任务重定义、阶段级消融、关键实验结果到当前论文产物的完整证据链。它不是单个实验记录，而是后续论文结构重写、结果章节撰写和图表重绘的总览入口。

建议阅读顺序：

```text
本文件：研究逻辑与证据链总览
  -> 03-实验设计/04-阶段消融对比与分析.md
  -> 04-论文写作与图表/01-论文结构重写任务.md
  -> 04-论文写作与图表/02-图表重绘任务.md
  -> 数据管理/05-评测结果/<各阶段结果>
```

本文档重点回答：

1. 原始 2DGS 为什么不适合植物 foreground-only 重建；
2. Plant-aware 2DGS 每一阶段具体改了什么；
3. 关键模块如何用公式表达；
4. 每一阶段由什么实验支撑；
5. 当前结果可以说明什么，不能说明什么；
6. 论文中应如何讲述这条方法故事线。

当前方法主线已经从普通 full-scene 2DGS 转为：

```text
Foreground-object reconstruction:
从多视角图像中重建由 mask 定义的植物前景对象，而不是先重建完整场景后再裁剪。
```

核心创新不是简单的 mask 后处理，而是将 2D Gaussian Splatting 的优化目标从 full-scene image reconstruction 改写为 mask-defined foreground-object reconstruction，并进一步通过 foreground track initialization、foreground-specific losses、soft view weighting 和 compact foreground cleanup 形成更稳定、更轻量的 plant-only Gaussian representation。

## 1. 总体研究逻辑

### 1.1 原始问题

原始 2DGS 的默认任务是 full-scene reconstruction。给定多视角图像，模型默认学习整张图像中的所有可见内容，包括植物、花盆、背景、桌面、墙体、支架和其他非目标区域。这对普通新视角合成是合理的，但对植物表型任务存在明显问题：

1. 植物表型只关心 plant foreground object；
2. full-scene 2DGS 会把 Gaussian 容量分配给背景；
3. 后续 plant-only asset export、mesh extraction 和 phenotype measurement 会受到背景残留污染；
4. 训练后再用 mask/pruning 删除背景，无法完全抵消训练阶段已经学习到的背景结构。

因此，本文不再把目标定义为“重建整张图”，而是定义为：

```text
只重建 mask 定义的 foreground plant object。
```

### 1.2 一句话创新

中文：

```text
本文的关键创新不是基于 mask 的后处理，而是将 2D Gaussian Splatting 从整图场景重建重新定义为 mask 约束的前景对象重建；进一步地，本文证明视图质量控制应采用 soft weighting 而不是 hard filtering，并结合 compact foreground cleanup 生成更轻量、更干净的 plant-only Gaussian representation。mask的来源是sam3，这里可以做一个不同prompt的分析。
```

英文：

```text
The key innovation is not mask-based post-processing, but reformulating 2D Gaussian Splatting from full-scene reconstruction into mask-defined foreground-object reconstruction. We further show that view quality should be incorporated through soft weighting rather than hard filtering, and combine it with compact foreground cleanup to produce a lighter and cleaner plant-only Gaussian representation.
```

## 2. 当前方法版本定义

### 2.1 Ours-core

Ours-core 对应 A6：

```text
A6 = foreground track initialization
   + foreground RGB loss
   + alpha mask loss
   + background opacity loss
```

Ours-core 的任务是证明：

```text
2DGS 可以从 full-scene reconstruction 被重新定义为 foreground-object reconstruction。
```

### 2.2 Ours-full / Ours-compact

Ours-full 或 Ours-compact 对应：

```text
A6 + M1-soft + M4
```

其中：

- A6 负责 foreground-object reconstruction 的核心任务重定义；
- M1-soft 在不删除视角的前提下，用 soft weighting 调整不同视角的训练贡献；
- M4 负责 compact foreground cleanup，减少 Gaussian 数量和背景泄漏，服务后续 export / mesh。

Ours-full 的论文角色不是显著提升 PSNR，而是在基本保持 foreground quality 的同时获得更少 Gaussian、更低 leakage 和更干净的 plant-only representation。

| 方法版本 | 组成 | 论文角色 |
|---|---|---|
| Ours-core | A6 | 核心 foreground-object reconstruction 方法 |
| Ours-full / Ours-compact | A6 + M1-soft + M4 | 实用紧凑版本，用于减少 Gaussian、降低泄漏、服务 export / mesh |

## 3. 评价指标体系

由于本文目标是 foreground-object reconstruction，不能只看 full-frame PSNR、SSIM、LPIPS。full-frame 指标会受到背景压黑或前景隔离的影响，不适合作为主指标。

当前核心指标为：

| 指标 | 方向 | 作用 |
|---|---:|---|
| PSNR_fg | ↑ | mask 前景区域的像素重建质量 |
| SSIM_fg | ↑ | mask 前景区域结构一致性 |
| LPIPS_fg / LPIPS_fg_black_bg | ↓ | 前景感知质量，通常在背景置黑后计算 |
| outside_nonblack_ratio_mean | ↓ | mask 外仍被渲染为非黑的比例，衡量背景残留 |
| leakage_energy_ratio_mean | ↓ | mask 外能量相对前景能量的泄漏比例 |
| gaussians_30000 | ↓ | 模型规模和后续 mesh 成本 |

当前 foreground-only 判定阈值暂定为：

```text
outside_nonblack_ratio_mean < 0.05
leakage_energy_ratio_mean < 0.10
```

只有同时满足上述条件，才能认为模型基本实现 foreground-only 分离。

## 4. 总体运行逻辑

```text
Input multi-view RGB images + plant masks（use sam3）
        ↓
SfM / COLMAP camera pose estimation
        ↓
Foreground track initialization
        ↓
Ours-core A6 foreground-object 2DGS training
        ↓
Foreground-specific loss optimization
        ↓
M1-soft view-weighted training enhancement
        ↓
M4 compact foreground cleanup / export
        ↓
Ours-full compact plant-only Gaussian representation
        ↓
M5 mesh-only evaluation / TSDF / boundary cleanup
        ↓
Future phenotype-ready mesh and trait measurement
```

当前论文主证据集中在 A6、M1-soft、M4 和 S18；M5/S19 已进入 mesh-only structural and efficiency evaluation 阶段，但尚未形成 leaf width 或 phenotype accuracy improvement 证据。

## 5. 阶段 A/B：Baseline 与任务重定义

### 5.1 原始 full-scene 2DGS

原始 2DGS 的默认优化目标可以抽象为：

```text
L_full = sum_i L_rgb(R_i, I_i)
```

其中：

- I_i 是第 i 个训练视角的整张图像；
- R_i 是模型渲染结果；
- RGB loss 在整张图像上计算。

这意味着背景区域也会参与监督，模型会被奖励去重建背景。

### 5.2 植物表型任务中的问题

对于植物表型任务，目标不是重建完整场景，而是得到 plant-only Gaussian / mesh。如果 full-scene 2DGS 学到背景，即使 mask 内 PSNR 不差，mask 外仍会有大量可见背景，导致：

1. plant-only asset 不干净；
2. mesh extraction 可能出现背景粘连；
3. 后处理 pruning 难以完全消除背景结构；
4. phenotype measurement 可能受到非植物结构污染。

### 5.3 Foreground-object reconstruction

本文提出 foreground-object reconstruction：

```text
从“整图场景重建”改为“由 mask 定义的植物前景对象重建”。
```

它不是训练后裁剪，而是在训练目标层面改变监督区域。

## 6. 阶段 B/D：Foreground-object objective 消融 A0-A6

### 6.1 阶段目标

这一阶段回答核心问题：

```text
背景问题能否通过后处理或简单正则解决？还是必须从训练目标层面改成 foreground-only？
```

### 6.2 方法定义表

| ID | 方法含义 | foreground init | foreground RGB loss | alpha mask loss | bg opacity loss | 论文角色 |
|---|---|---:|---:|---:|---:|---|
| A0 | full-scene 2DGS + foreground eval | no | no | no | no | full-scene 基线，证明背景竞争问题 |
| A1 | mask preprocess foreground training | no | implicit | no | no | 朴素前景训练 baseline |
| A2 | alpha mask loss only | no | no | yes | no | 只约束 silhouette |
| A3 | background opacity only | no | no | no | yes | 只压背景 opacity |
| A4 | alpha mask loss + background opacity | no | no | yes | yes | 只靠 alpha/bg，不改 RGB 监督 |
| A5 | foreground RGB loss + alpha mask loss + bg opacity | no | yes | yes | yes | foreground objective 的关键版本 |
| A6 | foreground track init + foreground RGB + alpha mask + bg opacity | yes | yes | yes | yes | 完整 foreground-only 核心方法 |

### 6.3 关键公式

原始整图 RGB loss：

```text
L_rgb_full = (1 / |Ω|) sum_{p in Ω} || R(p) - I(p) ||_1
```

Foreground RGB loss：

```text
L_rgb_fg = (1 / |Ω_fg|) sum_{p in Ω} M(p) || R(p) - I(p) ||_1
Ω_fg = {p | M(p) = 1}
```

Alpha mask loss：

```text
L_mask = (1 / |Ω|) sum_{p in Ω} | A(p) - M(p) |
```

Background opacity loss：

```text
L_bg = (1 / |Ω_bg|) sum_{p in Ω} (1 - M(p)) A(p)
Ω_bg = {p | M(p) = 0}
```

A6 总损失：

```text
L_A6 = L_rgb_fg + λ_mask L_mask + λ_bg L_bg + L_regularization
```

核心是 foreground RGB loss。alpha mask loss 和 background opacity loss 是辅助约束。

### 6.4 Foreground track initialization

A6 进一步引入 foreground track initialization。其思想是：

```text
不要让 full-scene COLMAP 背景点先进入模型，而是利用 3D point track 与多视角 mask 的投影交集筛出 foreground 初始化点。
```

可表示为：

```text
Keep(X_j) = 1, if (1 / |V_j|) sum_{i in V_j} M_i(π_i(X_j)) >= τ_track
```

其中：

- X_j 是 COLMAP 稀疏点；
- V_j 是能观测到 X_j 的视角集合；
- π_i(X_j) 是 X_j 投影到第 i 个视角的位置；
- M_i 是第 i 个视角的 foreground mask；
- τ_track 是 foreground track 保留阈值。

### 6.5 KongQueZhuYu A0-A6 结果

| ID | PSNR_fg | SSIM_fg | LPIPS_fg | outside | leakage | Gaussians | 是否 foreground-only |
|---|---:|---:|---:|---:|---:|---:|---|
| A0 | 24.2090 | 0.8514 | 0.0480 | 0.9908 | 1.2201 | 751,213 | no |
| A1 | 20.7291 | 0.7505 | 0.0696 | 0.0073 | 0.0042 | 263,108 | yes, but quality poor |
| A2 | 24.3422 | 0.8478 | 0.0491 | 0.9898 | 1.2260 | 768,067 | no |
| A3 | 24.7508 | 0.8672 | 0.0451 | 0.9900 | 1.2255 | 742,931 | no |
| A4 | 24.8126 | 0.8687 | 0.0445 | 0.9896 | 1.2266 | 763,266 | no |
| A5 | 25.1055 | 0.8561 | 0.0437 | 0.0294 | 0.0190 | 592,900 | yes |
| A6 | 25.0072 | 0.8548 | 0.0438 | 0.0294 | 0.0189 | 591,623 | yes |

### 6.6 结果逻辑

A0 的 PSNR_fg 为 24.2090，说明 full-scene 2DGS 能重建 mask 内植物区域；但 outside=0.9908、leakage=1.2201，说明 mask 外几乎全是可见背景。因此它不是 plant-only reconstruction。

A1 背景很干净，但 PSNR_fg 只有 20.7291，LPIPS_fg 达到 0.0696，说明简单把输入改成前景图不足以获得高质量前景重建。

A2-A4 的前景 PSNR 不差，但 leakage 仍在 1.22 左右。这说明如果 RGB loss 仍在整图上计算，模型仍会学习背景。alpha mask loss 和 bg opacity loss 不能单独改变任务目标。

A5 打开 foreground RGB loss 后，outside 降到 0.0294，leakage 降到 0.0190，同时 PSNR_fg 达到 25.1055。说明 foreground RGB supervision 是从 full-scene 到 foreground-object 的决定性转折。

A6 相比 A5 数值接近，但方法定义更完整，因为它从初始化阶段就使用 foreground track 过滤，使初始点云偏向植物对象。论文中应以 A6 作为 Ours-core。

## 7. E7 与 A6 的关系

E7 是 full-scene training 后再进行 mask/pruning/export 的路线。它试图回答：

```text
能不能先训练 full-scene，再用 mask 后处理得到 foreground object？
```

E7 不等同于 A6：

| 方法 | 训练目标 | 输出本质 | 论文角色 |
|---|---|---|---|
| E7 | full-scene reconstruction | full-scene Gaussian，之后尝试 mask/pruning/export | 强对照，说明后处理路线的上限和失败点 |
| A6 | foreground-object reconstruction | 训练目标本身就是 foreground object | 当前主方法 |

核心解释：

```text
后处理筛背景不是等价方案，因为 full-scene 模型在训练阶段已经将容量分配给背景结构。我们将优化目标从整图重建改写为前景对象重建，使模型从初始化、监督和正则化阶段都围绕 foreground object 收敛。
```

## 8. A6 三样本跨样本验证

### 8.1 实验目的

A0-A6 首先在 KongQueZhuYu 上证明了方法方向。随后需要验证 A6 是否只是单样本偶然，因此选择三类代表样本：

| 样本 | 角色 |
|---|---|
| KongQueZhuYu | 复杂背景 / 主样本 |
| XianKeLai1 | 薄叶 / 细结构 |
| CaoMei2 | 密集叶 / 遮挡 |

### 8.2 A6 三样本结果

| Sample | Role | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians |
|---|---|---:|---:|---:|---:|---:|---:|
| KongQueZhuYu | 复杂背景 / 主样本 | 25.0072 | 0.8548 | 0.0438 | 0.0294 | 0.0189 | 591,623 |
| XianKeLai1 | 薄叶 / 细结构 | 23.7276 | 0.8278 | 0.0309 | 0.0484 | 0.0379 | 253,827 |
| CaoMei2 | 密集叶 / 遮挡 | 25.0833 | 0.8121 | 0.0250 | 0.0147 | 0.0081 | 370,844 |

### 8.3 结果解释

三个样本均满足 foreground-only 阈值：

```text
outside < 0.05
leakage < 0.10
```

其中：

- KongQueZhuYu 前景质量较好，PSNR_fg 和 SSIM_fg 较高；
- CaoMei2 leakage 最低，是最干净的样本；
- XianKeLai1 outside=0.0484，接近阈值但仍通过，说明薄叶细结构是更难样本。

结论：

```text
A6 在复杂背景、薄叶细结构和密集遮挡三类代表样本上均成立，说明 foreground-object reconstruction 不是单样本偶然结果。
```

## 9. 阶段 C：M1 视图质量策略

### 9.1 原本设想

最初 M1 的想法是利用 raw quality、mask quality、geometry quality 等门控筛掉低质量图像，从而提升 foreground reconstruction。可以抽象为 hard filtering：

```text
Keep(I_i) = 1, if Q_i >= τ
Keep(I_i) = 0, if Q_i < τ
```

其中 Q_i 是视角质量分数。

### 9.2 M1-hard / reject-only 负证据

在 KongQueZhuYu 上，M1-hard 和 M1-reject-only 均失败：

| Variant | Eval images | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians |
|---|---:|---:|---:|---:|---:|---:|---:|
| A6 | 27 | 25.0072 | 0.8548 | 0.0438 | 0.0294 | 0.0189 | 591,623 |
| A6+M1-hard | 17 | 12.5478 | 0.6018 | 0.1179 | 0.1743 | 0.3020 | 597,116 |
| A6+M1-reject-only/raw-mask | 24 | 13.4557 | 0.6244 | 0.1115 | 0.1450 | 0.2848 | 579,612 |

结论：

```text
hard retained-list filtering 会破坏多视角覆盖，导致 foreground-object reconstruction 崩溃。
```

这说明 M1 不应该写成“筛掉低分图”，因为植物重建依赖完整视角覆盖。低质量视角仍可能包含关键几何信息。

### 9.3 M1-soft 修正

M1 从 hard filtering 改为 soft view weighting。不删除图像，而是在 foreground RGB loss 中加入视角权重：

```text
L_rgb_fg_soft = ( sum_i q_i L_rgb_fg(i) ) / ( sum_i q_i )
```

其中：

- q_i 是第 i 个视角的质量权重；
- 所有视角仍参与训练；
- 低质量视角梯度贡献降低，但其几何覆盖信息被保留。

### 9.4 M1-soft 正证据

KongQueZhuYu 上：

| Variant | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians |
|---|---:|---:|---:|---:|---:|---:|
| A6 | 25.0072 | 0.8548 | 0.0438 | 0.0294 | 0.0189 | 591,623 |
| A6+M1-soft | 24.9566 | 0.8543 | 0.0440 | 0.0284 | 0.0184 | 532,264 |

M1-soft 相对 A6：

- PSNR_fg 仅下降 0.0506 dB；
- SSIM_fg 仅下降 0.0005；
- LPIPS_fg 几乎不变；
- outside 和 leakage 略降；
- Gaussian 数量减少 59,359，约 10.03%。

结论：

```text
视图质量信息不能用于 hard delete，但可以用于 soft weighting。M1-soft 保留多视角覆盖，同时降低低质量视角的训练贡献。
```

## 10. 阶段 E：M4 compact foreground cleanup

### 10.1 模块定位

A6 已经解决 foreground-object reconstruction，但模型中仍存在一定冗余 Gaussian。M4 的目标不是改变训练目标，而是在 A6 后进行 compact cleanup / export。

M4 可以看作对 Gaussian 的后处理筛选：

```text
Keep(g_j) = 1, if Score(g_j) >= τ_g
Keep(g_j) = 0, if Score(g_j) < τ_g
```

其中 Score(g_j) 可综合：

```text
Score(g_j) = α M_j + β O_j + γ V_j + δ B_j + η C_j
```

含义：

- M_j：mask consistency；
- O_j：opacity；
- V_j：visibility / view coverage；
- B_j：brightness / color abnormality；
- C_j：connected / topology cue。

当前 M4 的论文定位应为：

```text
compactness / export cleanup module
```

而不是 foreground quality 提升模块。

### 10.2 M4 跨样本结果

XianKeLai1：

| Variant | Gaussians | PSNR_fg | SSIM_fg | LPIPS_fg | Leakage |
|---|---:|---:|---:|---:|---:|
| A6 | 253,827 | 23.7276 | 0.8278 | 0.0309 | 0.0379 |
| A6+M4 | 251,047 | 23.7256 | 0.8279 | 0.0310 | 0.0376 |

CaoMei2：

| Variant | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians |
|---|---:|---:|---:|---:|---:|---:|
| A6 | 25.0833 | 0.8121 | 0.0250 | 0.0147 | 0.0081 | 370,844 |
| A6+M4 | 25.0303 | 0.8108 | 0.0251 | 0.0144 | 0.0080 | 284,757 |

CaoMei2 上 Gaussian 减少 86,087，约 23.21%，foreground quality 基本不变。

### 10.3 M4 结论

```text
M4 可以安全减少 Gaussian 数量并轻微降低 leakage，但不是 foreground quality 提升模块。它应作为 A6 后的 practical compactness / export cleanup。
```

## 11. 阶段 S18：Ours-full / Ours-compact 三样本闭环

### 11.1 组合方法

S18 组合版为：

```text
Ours-full = A6 + M1-soft + M4
```

其目标是：

```text
在基本保持 A6 前景质量的同时，进一步降低 Gaussian 数量和背景泄漏。
```

### 11.2 三样本闭环结果

当前三样本结果已完成闭环：

```text
A6 Gaussian 总数：1,216,294
Ours-full Gaussian 总数：997,049
减少：219,245
减少比例：18.03%
平均 PSNR_fg 仅下降：0.0657 dB
```

说明：

```text
Ours-full 在三类代表样本上基本保持前景重建质量，同时显著减少 Gaussian 数量。
```

### 11.3 关键样本结果

CaoMei2：

| Variant | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians |
|---|---:|---:|---:|---:|---:|---:|
| A6 | 25.0833 | 0.8121 | 0.0250 | 0.0147 | 0.0081 | 370,844 |
| A6+M1-soft | 25.0046 | 0.8107 | 0.0253 | 0.0140 | 0.0077 | 249,944 |
| A6+M4 | 25.0303 | 0.8108 | 0.0251 | 0.0144 | 0.0080 | 284,757 |
| A6+M1-soft+M4 | 24.9718 | 0.8101 | 0.0252 | 0.0136 | 0.0076 | 246,452 |

组合版相对 A6：Gaussian 减少 124,392，约 33.54%；PSNR_fg 仅下降 0.1115 dB。

XianKeLai1：

| Variant | PSNR_fg | SSIM_fg | LPIPS_fg | Outside | Leakage | Gaussians |
|---|---:|---:|---:|---:|---:|---:|
| A6 | 23.7276 | 0.8278 | 0.0309 | 0.0484 | 0.0379 | 253,827 |
| A6+M1-soft+M4 | 23.7070 | 0.8273 | 0.0312 | 0.0479 | 0.0373 | 219,661 |

组合版相对 A6：Gaussian 减少 34,166，约 13.46%；PSNR_fg 仅下降 0.0206 dB，且 outside 仍小于 0.05。

### 11.4 S18 结论

```text
Ours-core = A6，证明任务重定义成立。
Ours-full / Ours-compact = A6 + M1-soft + M4，证明在三样本上可以以很小的前景质量代价换取更紧凑、更干净的 plant-only Gaussian representation。
```

## 12. 阶段 F / M5：Mesh-only structural and efficiency evaluation

### 12.1 M5 当前目标

A6 和 Ours-full 输出的是 plant-only Gaussian representation。植物表型任务最终需要 mesh，因此需要从 Gaussian 表示进入显式网格。

当前 M5 还不是 phenotype accuracy improvement，而是：

```text
mesh-only structural and efficiency evaluation
```

即评估不同网格策略对 mesh 顶点数、连通域、边界位移和 mesh wall time 的影响。

### 12.2 M5 变体

| Variant | 含义 |
|---|---|
| Standard TSDF | 标准 TSDF 网格化基线 |
| Smaller truncation | 缩小 truncation，生成更紧凑 mesh |
| Post-boundary cleanup | 对边界进行保守几何调整 |

原始 TSDF 可简写为：

```text
D(x) = ( sum_c w_c(x) d_c(x) ) / ( sum_c w_c(x) )
```

其中：

- x 是体素点；
- d_c(x) 是第 c 个相机视角下的局部截断符号距离；
- w_c(x) 是融合权重；
- D(x) 是全局 TSDF 值。

随后通过 Marching Cubes 提取 mesh。

### 12.3 S19 结构与效率结果

| Sample | Mesh variant | Vertices | Components | Largest comp. ratio | Small comps | Boundary edges | Boundary consistency | Mean disp. | P95 disp. | Mesh time/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KongQueZhuYu | Standard TSDF | 167,789 | 8 | 0.9920 | 5 | 12,088 | - | - | - | 53.33 |
| KongQueZhuYu | Smaller truncation | 147,665 | 20 | 0.9350 | 12 | 25,086 | - | - | - | 56.52 |
| KongQueZhuYu | Post-boundary | 167,789 | 8 | 0.9920 | 5 | 12,088 | 0.9631 | 0.0041 | 0.0222 | 58.26 |
| XianKeLai1 | Standard TSDF | 74,753 | 6 | 0.9488 | 0 | 6,956 | - | - | - | 78.15 |
| XianKeLai1 | Smaller truncation | 66,138 | 12 | 0.9487 | 5 | 9,763 | - | - | - | 78.57 |
| XianKeLai1 | Post-boundary | 74,753 | 6 | 0.9488 | 0 | 6,956 | 0.8278 | 0.0121 | 0.0376 | 97.10 |

时间口径：mesh-only wall time，不拆成 TSDF / cleanup 子阶段。该时间来自 `render.py --skip_train --skip_test`，包含加载、radiance reconstruction、TSDF、mesh extraction、cleanup 和可选 post-boundary 操作。

### 12.4 M5 结果解释

smaller truncation 在两个样本上都减少约 12% 顶点，但 connected components 增加：

```text
KongQueZhuYu: 8 -> 20
XianKeLai1: 6 -> 12
```

结论：

```text
smaller truncation 能生成更紧凑的网格，但会增加碎片化风险，不能直接写成质量提升。
```

post-boundary 保持连通域数量不变：

```text
KongQueZhuYu: 8 -> 8
XianKeLai1: 6 -> 6
```

但 XianKeLai1 的 boundary consistency 更低、mean displacement 和 P95 displacement 更高，说明薄叶细结构对边界调整更敏感。

结论：

```text
post-boundary 是结构保持型的保守边界调整；XianKeLai1 是更强的边界压力测试样本。
```

从 mesh time 看，post-boundary 会引入额外 wall time，尤其在 XianKeLai1 上从 78.15 s 增加到 97.10 s。说明薄叶细结构样本不仅边界调整更敏感，也可能带来更高处理开销。

### 12.5 当前不能写什么

当前 S19 不能写成：

```text
M5 提升了 leaf width accuracy。
M5 提升了 phenotype measurement accuracy。
smaller truncation 提升了 mesh quality。
```

只能写成：

```text
S19 提供了 mesh-only structural and efficiency evidence。
```

## 13. 当前已有产物清单

### 13.1 核心结果文档

| 产物 | 内容 |
|---|---|
| A0-A6 ablation summary | KongQueZhuYu foreground-object objective 消融 |
| representative_A6_summary | 三样本 A6 跨样本验证 |
| S14 M1 hard/reject-only summary | M1 retained-list hard filtering 负证据 |
| S16 M1 soft weighting summary | M1-soft 正证据 |
| S18 cross-sample summary | Ours-full / Ours-compact 三样本闭环 |
| S19 M5 summary | mesh-only structural + efficiency evaluation |

### 13.2 核心实验输出

| 阶段 | 状态 |
|---|---|
| A0-A6 | 已完成 |
| 三样本 A6 | 已完成 |
| M1-hard / reject-only | 已完成，负证据 |
| M1-soft | 已完成，正证据 |
| M4 跨样本 | 已完成 |
| S18 Ours-full 三样本 | 已完成 |
| S19 mesh-only | 已完成结构与时间评价 |

### 13.3 代码与测试产物

| 文件/脚本 | 作用 |
|---|---|
| compute_mesh_structure_metrics.py | 计算 mesh 结构指标 |
| test_compute_mesh_structure_metrics.py | 单元测试 |
| m5_mesh_structure_metrics.csv | S19 mesh 结构指标表 |
| m5_mesh_time.csv / json | S19 mesh wall time |
| mesh_time_logs/ | 6 份 mesh time log + time 文件 |

## 14. 论文故事线建议

论文可以按以下逻辑组织。

### 14.1 问题定义

原始 2DGS 是 full-scene reconstruction；植物表型需要 plant-only foreground reconstruction。full-scene 模型会把背景也重建出来，污染 plant-only asset、mesh 和表型测量。

### 14.2 反证链

1. A0 说明 full-scene mask 内质量不差，但背景泄漏严重；
2. A1 说明简单 mask preprocessing 会损伤前景质量；
3. A2-A4 说明只加 alpha 或 bg opacity regularization 不够；
4. E7 说明 full-scene 后处理不等价。

### 14.3 核心方法

A5 打开 foreground RGB loss，证明 foreground objective 是关键转折；A6 加入 foreground track initialization，形成完整 Ours-core。

### 14.4 跨样本稳定性

三样本 A6 均满足 foreground-only 阈值，证明 Ours-core 不是单样本偶然。

### 14.5 M1 机制修正

M1-hard 和 reject-only 是负证据，说明 hard delete 会破坏多视角覆盖。M1-soft 是正证据，说明 view quality 应作为 soft weight，而不是删图依据。

### 14.6 M4 实用清理

M4 不是主创新，而是 compactness/export cleanup，在基本不伤 foreground quality 的情况下减少 Gaussian。

### 14.7 Ours-full

S18 三样本闭环证明 Ours-full / Ours-compact 能在平均 PSNR_fg 仅下降 0.0657 dB 的情况下减少 18.03% Gaussian。

### 14.8 M5 mesh-readiness

S19 表明 mesh 阶段已可进行结构与效率评价。smaller truncation 更紧凑但有碎片化风险；post-boundary 保持连通域但带来边界调整和额外时间开销。当前只能写成 mesh-readiness evidence，还不能写 phenotype accuracy improvement。

## 15. 论文核心表格建议

| 表格 | 内容 | 当前状态 |
|---|---|---|
| Table 1 | Dataset summary | 待整理 |
| Table 2 | A0-A6 foreground-object objective ablation | 已有 KongQueZhuYu 完整结果 |
| Table 3 | A6 representative sample validation | 已有三样本结果 |
| Table 4 | M1 hard vs reject-only vs soft weighting | 已有 KongQueZhuYu 结果 |
| Table 5 | M4 compact cleanup ablation | 已有 XianKeLai1 / CaoMei2 |
| Table 6 | Ours-core vs Ours-full cross-sample summary | S18 三样本已完成 |
| Table 7 | M5 mesh-only structural and efficiency evaluation | S19 已完成初版 |

## 16. 论文核心图建议

| 图 | 内容 |
|---|---|
| Fig. 1 | 从 full-scene reconstruction 到 foreground-object reconstruction 的任务重定义示意图 |
| Fig. 2 | A0 / A1 / A5 / A6 可视化对比：背景泄漏、前景质量、foreground-only 输出 |
| Fig. 3 | A6 方法结构：foreground track initialization + foreground-specific losses |
| Fig. 4 | M1 hard filtering 失败 vs soft weighting 成功示意图 |
| Fig. 5 | M4 compact cleanup 前后 Gaussian 数量与泄漏对比 |
| Fig. 6 | Ours-core vs Ours-full 三样本定量柱状图 |
| Fig. 7 | S19 mesh variants：standard / smaller truncation / post-boundary 的网格结构对比 |

## 17. 当前最终结论

当前实验已经支撑以下结论：

1. A6 是 foreground-object reconstruction 的核心方法定义；
2. foreground RGB loss 是 A5 的关键转折，也是从 full-scene 转向 foreground-object 的核心机制；
3. A6 在复杂背景、薄叶细结构、密集遮挡三类样本上均满足 foreground-only 分离标准；
4. M1-hard 和 M1-reject-only 是负证据，说明 hard filtering 会破坏多视角覆盖；
5. M1-soft 是当前合理 M1 候选，说明视图质量应通过 soft weighting 调节训练贡献；
6. M4 是 compactness/export cleanup，在基本不伤前景质量的情况下减少 Gaussian；
7. Ours-full 在三样本上减少 18.03% Gaussian，平均 PSNR_fg 仅下降 0.0657 dB；
8. S19 已完成 mesh-only structural and efficiency evaluation，但不能提前写成 phenotype accuracy improvement。

最稳的一句话收口：

```text
本文将 2DGS 从 full-scene reconstruction 改写为 mask-defined foreground-object reconstruction，并证明 foreground-specific RGB supervision 是关键转折；在此基础上，soft view weighting 和 compact foreground cleanup 能在保持前景质量的同时减少 Gaussian 数量和背景泄漏，为后续 phenotype-ready mesh 提供更干净、更紧凑的 plant-only representation。
```

## 18. 下一步建议

### 18.1 短期下一步

1. 整理所有表格为论文格式；
2. 统一命名：Ours-core = A6，Ours-full = A6+M1-soft+M4；
3. 绘制 A0/A5/A6/E7 的核心对比图；
4. 绘制 M1-hard 失败和 M1-soft 成功的机制图；
5. 绘制 S18 三样本 Gaussian reduction / PSNR drop 图；
6. 绘制 S19 mesh structural evaluation 图。

### 18.2 中期下一步

1. 补 edge thickness proxy；
2. 如果可能，补 leaf width / leaf length phenotype measurement；
3. 如果 M5 指标稳定，再把 M5 从 mesh-readiness 推进到 phenotype-ready mesh。

### 18.3 写作边界

当前可以写：

```text
Ours-full reduces Gaussian count and leakage with minimal foreground quality degradation.
S19 provides mesh structural and efficiency evidence.
```

当前不能写：

```text
M5 improves leaf width measurement accuracy.
M5 proves phenotype accuracy improvement.
smaller truncation improves mesh quality.
```

这些需要后续 phenotype metrics 支撑。

## 19. 当前证据强度评估

### 19.1 已经比较稳的部分

Ours-core 的证据链已经成立。A0-A6 消融很清楚地把“full-scene 能重建前景但背景泄漏严重”“mask preprocessing 会损伤质量”“alpha/bg 正则不足以改变任务目标”“foreground RGB loss 是关键转折”这几件事拆开证明了。A6 三样本结果进一步说明该结论不是 KongQueZhuYu 单样本现象。

Ours-full 的阶段性证据也已经足够作为论文主结果之一。S18 三样本闭环显示 Gaussian 总数减少 18.03%，平均 PSNR_fg 仅下降 0.0657 dB，且 outside / leakage 未上升。这适合写成 practical compact variant，而不是新的核心任务定义。

M1 的叙事现在比最初更强。hard filtering 失败并不是坏结果，反而提供了关键负证据：植物多视角重建不能简单删图，因为 coverage 比单帧质量分数更重要。M1-soft 的正证据让模块逻辑变成“质量信息用于调权，而不是删视角”，这条故事线很自然。

### 19.2 需要降调或继续补强的部分

M4 当前更适合写成 compactness/export cleanup。它能减少 Gaussian 并略降 leakage，但不应被描述为显著提升 foreground reconstruction quality 的模块。

S19 当前已经从 mesh 入口升级为 mesh-only structural and efficiency evaluation，但仍不能写成 phenotype accuracy improvement。smaller truncation 的顶点减少同时伴随 components 和 boundary edges 增加，因此它最多说明 mesh 更紧凑且存在碎片化风险，不能直接写成质量更好。

M5 要成为强贡献，还需要 edge thickness proxy 或 phenotype measurement。尤其是 leaf width bias，如果后续能证明 post-boundary 或 edge-aware strategy 在不增加碎片化的前提下降低边界厚度代理或叶宽误差，M5 才能从 mesh-readiness 上升为 phenotype-ready mesh evidence。

### 19.3 写作策略建议

论文主创新建议聚焦在任务重定义：

```text
full-scene 2DGS -> mask-defined foreground-object 2DGS
```

贡献点可以采用两层结构：

1. Ours-core：foreground-object reconstruction 的核心方法；
2. Ours-full：soft view weighting + compact foreground cleanup 的实用紧凑版本。

M5 目前建议放在后段结果或扩展分析中，定位为 mesh-readiness / structural evaluation。如果后续补齐表型误差，再把它提升为第三个强贡献点。

### 19.4 当前风险

主要风险有三点：

1. 三样本数量仍然偏少。适合作为方法论文的代表样本证据，但若目标期刊要求强统计泛化，需要继续补样本或做更多 repeated runs。
2. foreground-only 阈值目前是内部定义，需要在论文中解释选择依据，最好与可视化结果或下游 mesh 污染风险关联起来。
3. M5 的结果目前是结构代理指标，不能替代表型人工测量。写作时必须严格避免把 mesh structural evidence 推导成 phenotype accuracy improvement。

### 19.5 总体判断

当前最成熟的论文主线是：

```text
A6 / Ours-core 解决任务定义问题；
M1-soft 修正视图质量策略；
M4 和 S18 形成 compact plant-only representation；
S19 打开 mesh-readiness 分析，但 phenotype-ready evidence 仍待补。
```

换句话说，现在已经足够支撑一篇以 Plant-aware 2DGS foreground-object reconstruction 为核心的方法论文主体实验。若希望论文标题和贡献继续强调 phenotypic measurement，则下一阶段必须补 edge thickness proxy 和 leaf-level phenotype accuracy。
