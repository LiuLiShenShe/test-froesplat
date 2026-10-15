# Phase20E-3 共同参考与代理指标口径

正式评价直接调用 E2 的 `scripts/metric_core.py`，不修改其实现。每个模型使用同一场景、同一冻结 held-out 相机、同一 canonical undistorted raw RGB 和同一 R0_proxy Mask；原始文件与缩放后张量分别记录 SHA256。WanNianQing1 / CaoMei2 / HongZhang 分别为 27 / 14 / 27 个视角。每个场景每个方法只有一次 Seed0 训练，视角不是训练重复。

## A. Full raw RGB appearance fidelity

渲染背景为黑色；模型 RGB 为真实浮点渲染并裁剪至 [0,1]。参考为各组共同的 canonical raw RGB，按 E1/E2 的 Pillow 分通道默认缩放路径缩放，再除以 255。不把各自 masked RGB 当成独立 GT。PSNR 使用原 `utils/image_utils.py`；SSIM 使用原 `utils/loss_utils.py` 的 11×11、sigma=1.5 高斯卷积核与相同 padding；LPIPS 为安装的 0.1.4 包、VGG 网络和 v0.1 权重，`normalize=True`。每次新评价检查 E1 记录的包版本、源码及两份 VGG 权重 SHA256。

Full RGB 是全场景外观一致性。前景黑背景训练未拟合原始背景，因此不能将低全图 PSNR 直接解释为植物几何退化。

## B. Fixed proxy foreground appearance fidelity

ROI 是 E2 冻结且有来源记录的 R0 预测 Mask，二值最近邻缩放；所有方法完全共用，不随输出修改，也不是独立人工前景 GT。

令 ROI 像素集合为 R，RGB 通道为 c。`foreground_MSE = sum_{p∈R,c}(I-G)^2 / (3|R|)`；前景 PSNR 为 `-10 log10(MSE)`；绝对色差在同样的 `3|R|` 个样本上取绝对误差均值。掩膜外零误差不进入前景 MSE 分母。空 ROI 必须报错，不能替换或臆造。

Composited SSIM/LPIPS 将模型与共同 raw RGB 分别乘以同一个 ROI，ROI 外同时置黑，再调用相同原始指标。名称明确保留 `fixed_mask_composited_*` 和 proxy；卷积窗口/感受野跨越 ROI 边界，背景黑色区域仍进入这些指标，它们不是独立目标 GT 指标。

`foreground_coverage_proxy = mean_{p∈R}[alpha(p)>0.5]`；`missing_alpha_fraction = 1-coverage`。它们衡量模型 alpha 与预测区域的覆盖关系，不能证明真实叶片完整性。

## C. Background suppression

在相同 ROI 外计算 RGB 平方均值 `outside_ROI_RGB_energy`、alpha 均值及总和；alpha>0.5 的 ROI 外像素采用 8 邻接连通区域统计，记录面积≥16像素的组件数、最大面积及总像素数。预测 alpha 与 ROI 的 IoU 仅称 `proxy_silhouette_IoU`。外部 alpha 组件不自动等于真实 floating Gaussian。黑色 RGB 与非零 alpha 可同时出现，因此两类背景统计分别呈现。

## D. Descriptive geometric behavior

Gaussian 数量、sigmoid opacity 与 exp(scale) 分布来自真实模型。深度为原渲染器 depth_ratio=0 的 expected surface depth，单位仅为继承的任意 COLMAP 单位。深度分位数在 ROI、alpha>0.5、有限且正深度交集上计算。法向一致性是 Gaussian rend_normal 与由其深度得到的 surf_normal 的余弦/绝对余弦代理；二者并非独立几何观测。

M0 完整复用 E2 的 RGB/alpha float 文件及指标，同时保留 E2 为深度/法向重放时存储的 `geometry_replay_alpha`。复算严格区分：外观/ROI alpha 指标使用原 alpha，深度与法向支持使用重放 alpha。M1/M2 的两种 alpha 在同一次真实渲染中获得并相同；不把历史缓存替换成新渲染以美化对照。

Mesh-to-Gaussian depth 在同一冻结相机下 CPU raycast：共同支持区为 ROI、Gaussian 重放 alpha>0.5、两种深度均有限且为正。报告绝对差均值/中位数/P95及相对差均值。它只检验 Mesh 与输入 Gaussian 的一致性，不能作为真实三维精度。Mesh ROI 覆盖与 silhouette 也只是同一预测 ROI 的代理。

## 汇总与可复算

所有场景均为冻结视角的算术均值，不把 68 个相机作为 68 次独立训练。不构造 A/B/C/D 综合分数或排名，不进行仅凭三场景单种子的显著性声明。正式原始指标由保留的真实 float 张量计算；PNG 仅供展示。`scripts/recompute_metrics.py` 对全部 204 条模型/视角记录独立复算、逐项比较并写入 `reports/FLOAT_METRIC_RECOMPUTATION.csv`。大文件路径和 SHA256 由指标表与最终恢复索引提供。

来源：E2 `FOREGROUND_PROXY_METRIC_SPEC.md` 与 `scripts/metric_core.py`；E1 版本/权重记录；本地冻结 `utils/image_utils.py`、`utils/loss_utils.py`。没有新增人工 GT、真实几何 GT 或物理尺度。

训练日志中的原生 2DGS `training_report` 使用 loader 的黑背景 RGBA 派生 RGB，属于继承的内部训练诊断。它与本阶段共同 canonical raw RGB 主评价的参考不同，不可将其中 PSNR 当作正式全图结果，也不用于挑选 checkpoint。本阶段正式全图/前景/背景指标以 `M2M3_HELDOUT_METRICS.csv`、对应代理 CSV 和真实浮点复算记录为准。
