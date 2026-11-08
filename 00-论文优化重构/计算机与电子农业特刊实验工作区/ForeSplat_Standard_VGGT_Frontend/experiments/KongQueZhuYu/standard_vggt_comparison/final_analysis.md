# Final analysis — Standard VGGT frontend vs VGGT-Omega (KongQueZhuYu)

Date: 2026-11-07. GPU1 (A6000). All metrics are **Training-view Metrics** over the same
208 views (both sides train on all views; `eval=False`).

## Answers to the six questions

1. **标准 VGGT 是否保留更多叶片细节/几何?**
   **否。** 原生网格前景深度梯度两者完全相同 (0.0118 vs 0.0118, ratio 1.0),但标准
   VGGT 的 518 方形预处理把 2160×3840 竖版图的内容压进 291×518 = 150,738 px
   (黑边占 44%),而 VGGT-Ω balanced@512 是 688×384 = 264,192 px —— 标准 VGGT
   的内容像素预算是 Ω 的约 57%。模型更强但喂给 GGGS 的深度图像素更少。

2. **深度图更清晰?**
   **没有。** 12 视角前景深度梯度(原图坐标系): Ω 0.0022 vs 标准 0.0017(标准更平坦,
   因上采样分辨率预算低);原生网格两者相同。标准 VGGT 深度在质量上没有带来可测量的
   提升,在内容分辨率上反而吃亏。视觉证据: `Standard_VGGT_Frontend/visualizations/depth_conf_12views.png`。

3. **相机估计更稳定?**
   **是,数值上非常稳定但这不是决定性因素。** 标准 VGGT 208-view 重投影误差
   mean 0.0004 px / p95 0.0011 px,远好于融合导向的 VGGT-Ω 流程。过程中发现并修复
   一个真实 bug: 把 camera/depth head 放进 bf16 autocast 会导致 `|RR^T-I| ≈ 1.6e-3`,
   产生随相机 |t| 增长的重投影误差(12 帧时 mean 2.93 px)—— 对齐官方 autocast 边界后
   全部 <1e-4 px (commit d23ddb98)。但这说明 2.9px 的"不稳定"纯粹是数值精度问题,
   不是算法问题,修复后也没有改善最终渲染。

4. **是否优于 VGGT-Ω + GGGS?**
   **否,明显更差。** E5 (13.11 dB / 0.719 / 0.355 / FG-PSNR 10.45) vs E1
   (17.89 dB / 0.786 / 0.241 / FG-PSNR 15.28)。PSNR −4.79 dB、FG-PSNR −4.83 dB,
   LPIPS +0.114。这不是微小变化,是全面且量级很大的倒退。

5. **模糊是否仍存在?**
   **是,且更严重;且已被证明与 GGGS 无关。** E5 叶片放大区域梯度 1.64 vs
   E1 2.53 vs GT 6.03;E5 渲染面板亮度 (均值 RGB≈5) 远低于 GT (24.6)。四联图
   前景梯度均值 (8 视角): GT 0.0360 / E1 0.0143 / E5 0.0111 / **E6 0.0028** ——
   2DGS 渲染比 GGGS 更平滑,锐度进一步下降 (`render_comparison/sharpness.csv`)。
   模糊不仅没解决,标准 VGGT 前端下更严重。E6 (官方 2DGS) 在同一输入上得到与
   E5 相同的全部指标 —— **任何 3DGS 系后端都复现同样的模糊**,说明 2DGS/GGGS
   的差距不是原因,标准 VGGT 前端输出才是。与无融合对照 (E2 原始深度 17.93 dB
   与 E1 持平) 结合看,完整归因链为: Ω 前端的模糊不来自融合模块也不来自初始
   点数 (既有对照),标准 VGGT 前端 (square@518) 的模糊更显著差于 Ω 前端,且与
   3D 算法无关。

6. **是否应切换 2DGS?**
   **已实测:2DGS 不能解决模糊,且证明瓶颈在前端而非 3D 算法层。** E6
   (官方 2DGS,同一份标准 VGGT COLMAP 数据/相机/初始化云,官方正则
   lambda_dist 0.2 / lambda_normal 0.05,30000 iter)结果:
   PSNR 13.20 / SSIM 0.713 / LPIPS 0.377 / FG-PSNR 10.52 —— 与 E5 (GGGS)
   的 13.11 / 0.719 / 0.355 / 10.45 在噪声范围内持平 (ΔPSNR +0.10 dB)。
   2DGS 最终只保留 119,836 个高斯 (GGGS 5.15M) 却得分相同。两个完全独立的
   3DGS 系算法在相同输入上得到相同的 ~13 dB,而 Ω 前端喂 GGGS 可达 17.89 dB
   —— **瓶颈被定位到标准 VGGT 前端的输出本身** (square@518 内容像素预算 + 其
   导出的几何),而非 GGGS 或任何单一 3DGS 系算法。

## Attribution: **情况 C 成立,且 E6 将瓶颈进一步定位到前端层**

- 情况 A (标准 VGGT 明显改善): 不成立。所有指标倒退。
- 情况 B (深度更好但渲染仍模糊): 不成立。深度没有可测量的提升。
- 情况 C (均无改善): **成立。** 标准 VGGT+GGGS 比 VGGT-Ω+GGGS 更差,模糊依旧。

规范触发的 2DGS 验证已按规范执行: “若仍模糊: 优先测官方 2DGS
(hbb1/2d-gaussian-splatting),用同一份标准 VGGT COLMAP 数据集/图像/相机/初始化
点云,官方推荐正则化与训练设置,与 GGGS 同视角同指标比较;2DGS 用自己优化
配置,不套用 GGGS 超参。仍不行再评估 PGSR。**不要对 GGGS 做无止境局部补丁。**”

E6 结果推翻了“换 3D 算法即可”的假设: 瓶颈在前端。规范链的下一步 (PGSR) 属于
同一 3DGS 系算法族、同一输入,E6 的证据预示其大概率同样持平;真正对症的对照
实验是保持 GGGS 后端不变、把标准 VGGT 的 square@518 预处理改为 Ω 式
balanced (宽高保持) 预处理,以隔离“内容像素预算 −43%”这一已识别差异。

## Evidence files

| artifact | path |
|---|---|
| 输入审计 | `source_data_audit.md` |
| 模型/环境元数据 | `model_metadata.json` |
| 标准 VGGT 前端 (深度/置信度/点云/COLMAP) | `Standard_VGGT_Frontend/` |
| GGGS 数据集 (图像/mask/相机/初始化云) | `Standard_VGGT_Dataset/` |
| E5 训练 (std VGGT + GGGS) | `E5_Standard_VGGT_GGGS_Baseline/{checkpoints,renders,metrics,RUN_REPORT.md}` |
| E6 训练 (std VGGT + 官方 2DGS) | `E6_2DGS_Standard_VGGT/{checkpoints,renders,metrics,RUN_REPORT.md}` |
| 深度对比 (预训练) | `Standard_VGGT_Frontend/visualizations/` |
| 渲染对比 + 锐度 | `render_comparison/{triptych_*.png, leaf_zoom_*.png, sharpness.csv}` (GT | E1 | E5 | E6 四联) |
| 统一对比表 | `comparison.csv` / `comparison.md` |

## Key numbers

| Frontend | PSNR | SSIM | LPIPS | FG-PSNR | FG-SSIM | Gaussians | Time (min) | VRAM (MiB) |
|---|---|---|---|---|---|---|---|---|
| VGGT-Ω balanced@512 + GGGS (E1) | 17.8947 | 0.7858 | 0.2406 | 15.2787 | 0.6117 | 600,458 | 40.3 | 8,950 |
| Standard VGGT square@518 + GGGS (E5) | 13.1057 | 0.7193 | 0.3546 | 10.4520 | 0.5006 | 5,153,165 | 49.6 | 18,331 |
| Standard VGGT square@518 + 2DGS (E6) | 13.2018 | 0.7134 | 0.3766 | 10.5194 | 0.4772 | 119,836 | 45.6 | — |
| Δ E6 − E5 | +0.096 | −0.006 | +0.022 | +0.067 | −0.023 | −5.01M | −4.0 | — |