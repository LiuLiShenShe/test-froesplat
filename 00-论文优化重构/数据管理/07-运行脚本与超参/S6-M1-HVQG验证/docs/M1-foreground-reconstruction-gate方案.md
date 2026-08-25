# M1 Foreground Reconstruction View-quality Gate 方案

## 核心目标

M1 的目标不是筛选“看起来最好”的图片，而是筛选最有利于 **mask foreground object 三维重建** 的视角集合。

2026-05-20 更新：`KongQueZhuYu A6+M1/H-VQG hard filtering` 与 `A6+M1-reject-only/raw-mask` 已作为负对照完成。结果显示，简单 top-score hard filtering 会破坏多视角覆盖；即使去掉 geometry hard delete，raw/mask retained-list 仍使 foreground-object reconstruction 明显退化。因此当前 retained-list H-VQG 不能作为有效 M1；M1 下一版必须优先改为 soft view weighting，若必须删图则采用 coverage-balanced anchor-view selection。

最终 M1 不应优先输出激进 retained list，而应同时满足：

- 单帧足够清晰，曝光和对比度稳定；
- foreground object 完整、尺度合适、没有严重截断；
- mask 边界可靠，跨帧面积变化不过度跳变；
- COLMAP 几何可靠；
- 视角覆盖均匀，不只保留相近视角。

## Gate 设计

### Gate 1 Raw Quality

输入：原始 RGB。

指标：

- blur/FFT 或 Laplacian sharpness；
- exposure score；
- contrast score；
- entropy/texture richness；
- optional CLIP-IQA 或 NIQE/BRISQUE。

作用：只过滤明显坏帧，避免误删可用视角。

### Gate 2 Foreground Object Quality

输入：mask + RGB。

指标：

- foreground ratio：使用样本自适应分位区间，不固定偏好 0.35；
- largest component ratio：主体连通性；
- truncation penalty：mask 是否贴近图像边界，判断主体被截断；
- center offset：前景中心偏离图像中心的程度；
- boundary sharpness：mask 边界附近 RGB 梯度；
- temporal stability：相邻帧 foreground area / IoU / centroid 是否平滑。

作用：剔除前景过小、过大、截断、mask 异常或边界不可靠的视角。

### Gate 3 Geometry and Coverage

输入：COLMAP/SfM。

指标：

- registered；
- matched points；
- inlier ratio；
- reprojection error；
- camera baseline / neighbor angle；
- foreground-overlap coverage。

作用：保证被保留视角不仅质量高，还能形成稳定三维约束。

## Selection 策略

不要简单 top-K，也不要把 geometry gate 设计成只做 hard delete。推荐：

```text
1. 默认不删图，优先在 foreground RGB loss 中使用 view-quality soft weighting；
2. Gate 1/2 只删除严重坏帧，避免误删可用覆盖视角；
3. Gate 3 只硬删明显异常位姿，其余低分视角优先降权；
4. 若必须输出 retained list，则按相机轨迹、图像序号或 viewing angle 分桶；
5. 每个桶至少保留 anchor views，再在桶内选择高质量视角；
6. 若某桶没有足够高分视角，放宽阈值补齐覆盖；
7. 输出 retained/rejected examples、view weights 和 coverage 指标。
```

这样可以避免 hard filtering 造成侧面、背面、顶部或遮挡补充视角丢失。

## 消融矩阵

```text
M1-F0 no filtering
M1-F1 FFT-only / raw reject-only
M1-F2 Raw + foreground reject-only
M1-F3 hard-gated H-VQG
M1-F4 soft view weighting
M1-F5 coverage-balanced selection
```

其中 `M1-F3 hard-gated H-VQG` 和 raw/mask retained-list reject-only 只作为负对照保留，不作为当前主方法组件。

## 验收指标

M1 是否有效以 foreground-object 结果判断：

- `PSNR_fg`
- `SSIM_fg`
- `LPIPS_fg_black_bg`
- foreground Gaussian 数量
- foreground mesh 完整度
- 训练时间
- 表型误差

full-frame PSNR/SSIM/LPIPS 仅作辅助记录。

下一版 M1 必须额外报告：

- retained views；
- azimuth / trajectory coverage；
- empty angular bins；
- min views per bin；
- pose graph connectivity；
- train/test retained scope。

## KongQueZhuYu 当前负证据

当前 `A6+M1/H-VQG hard filtering` 使用已有 retained list：

```text
数据管理/05-评测结果/KongQueZhuYu/M1_hvqg_smoke_v2/
```

训练 / 渲染加载时的保留视角为：

| gate | train retained | test retained |
|---|---:|---:|
| raw gate | 163/183 | 24/27 |
| mask gate | 163/163 | 24/24 |
| geometry gate | 123/163 | 17/24 |

与 A6 对比：

| Version | PSNR_fg | SSIM_fg | LPIPS_fg | outside | leakage | Gaussians |
|---|---:|---:|---:|---:|---:|---:|
| A6 | 25.0072 | 0.8548 | 0.0438 | 0.0294 | 0.0189 | 591623 |
| A6+M1/H-VQG hard filtering | 12.5478 | 0.6018 | 0.1179 | 0.1743 | 0.3020 | 597116 |
| A6+M1-reject-only/raw-mask | 13.4557 | 0.6244 | 0.1115 | 0.1450 | 0.2848 | 579612 |

结论：

- 当前 H-VQG hard filtering 显著破坏 foreground-object reconstruction；
- 关闭 geometry hard delete 后，raw/mask retained-list 仍明显失败，说明问题不只来自 geometry gate；
- leakage 和 outside 均超过 foreground-only 分离阈值；
- M1-hard 的 Gaussian 数量没有降低，说明硬筛没有带来有效简化；
- 负结果不是证明 M1 方向错误，而是证明 M1 不能写成简单删除低分图或 retained-list 硬筛。

下一版 M1 应优先：

- 使用 soft view weighting 作为第一主候选；
- 使用 reject-only severe degradation 作为真正极端坏帧初筛，而不是复用激进 retained list；
- 若需要删图，则增加 coverage-balanced selection，保证每个角度 / 轨迹区间有 anchor views；
- 将 retained-list hard filtering 与 soft view weighting 做对照；
- 所有 M1 分支必须输出 foreground-object eval。
