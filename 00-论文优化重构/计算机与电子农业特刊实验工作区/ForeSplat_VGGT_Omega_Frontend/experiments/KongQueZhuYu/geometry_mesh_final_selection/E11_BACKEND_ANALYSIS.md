# E11 Gaussian 后端正式对照分析

## 固定输入下的结论

在本株 KongQueZhuYu、208 个训练视角、Seed 0 和完整 30K 条件下，E10 标准 VGGT + GGGS 的照片重建优于 E11 标准 VGGT + 官方 2DGS。当前默认照片后端保留 GGGS。E11 的原生 TSDF 也没有表现出支持改用 2DGS 的薄表面优势；其 NKSR 补充实际运行两次均受显存限制，没有成功 Mesh，完整记录在 `meshes/supplementary/E11_2DGS_NKSR/RESOURCE_LIMIT.json`，不纳入几何排名。

| 全 208 帧逐视角均值 | E10 GGGS | E11 2DGS | E11−E10 |
|---|---:|---:|---:|
| PSNR dB | 23.900520 | 22.261472 | −1.639048 |
| FG-PSNR dB | 19.460452 | 17.592550 | −1.867902 |
| 前景内侧 5px 边缘 PSNR dB | 13.813681 | 12.736683 | −1.076999 |
| SSIM | 0.877556 | 0.863217 | −0.014339 |
| LPIPS Alex ↓ | 0.117086 | 0.137349 | +0.020263 |
| 前景梯度保留 | 0.731265 | 0.672860 | −0.058405 |
| 背景 Alpha 均值 | 0.378039 | 0.993141 | +0.615103 |
| 背景 Alpha>0.5 比例 | 0.382595 | 0.995759 | +0.613164 |
| Gaussian / Surfel | 576,396 | 248,629 | −327,767 |
| 完整训练命令秒 | 2,412.79 | 2,911.28 | +498.49 |

172/208 帧的 PSNR 与 FG-PSNR 低于 E10；205/208 帧的 LPIPS 和梯度保留更差。其余视角以及部分叶脉纹理存在局部收益，不能据此作全局推荐。208 帧来自同一株的相关训练观察，不以它们冒充独立重复实验，也未声称泛化测试性能。

## 相机与算法边界

E11 沿用 E6 官方实现及其 uv 环境，完整训练 30,000 iterations，SH=3、lambda_normal=0.05、lambda_dist=0.2、depth_ratio=0.0。原生增密、剪枝、opacity reset、优化器及几何正则保持不变，没有 EO、前景 RGB Loss、Alpha Loss、Mask Pruning 或额外深度先验。训练时源码快照、当前补丁 Diff 和实际参数均保存。

复用 E10 的修正 COLMAP 数据集、208 Frame ID、前景训练 RGB、1,252,383 个初始点、相机位姿和非中心内参。实际 Camera 为 width=1080、height=1920；每帧 K/W2C/Projection 与浮点 GT/Mask 哈希验证匹配。Pixel-center resize、非中心主点、ray unprojection、normal/depth 与 Mesh K 都已贯通验证，详见 `camera_interface/E11_INTERFACE_ANALYSIS.md`。实际旋转视角 CUDA 投影的 centroid 偏差低于 0.0025px，深度平面误差低于 1.1e-6；本轮不将结果差异归结为已排除的中心主点误用。

原生 surfel rasterizer 的投影 homography 已使用完整投影矩阵，本轮保留其核心 CUDA 数学。补丁增加相机参数接线和 expected/median 深度导出。实际 `train.py` 与上游固定 commit 匹配；其余 E6 源文件按哈希固定，未将混合旧目录误称为全部来自最新 checkout。

## 黑背景的几何

E11 RGB 背景接近黑色，但 Alpha 几乎充满背景。全 208 帧原生 TSDF raw 的轮廓 IoU=0.589656、背景外部几何比例=0.325091；仅删除少于 50 面组件的 post 为 IoU=0.618331、背景外部几何比例=0.289936。E10 对应 raw 为 0.875711 / 0.064957，post 为 0.882547 / 0.060210。两者 foreground coverage 都约 0.992，覆盖高不代表外部虚假几何少。

M5 raw 有 1,400,852 顶点、2,259,424 面、68,945 组件；post 保留 558 组件，删除 336,461 顶点和 260,633 面。M4 raw 为 697,688 顶点、1,310,705 面、4,871 组件；post 保留 93 组件。原始 Mesh 和所有删除位置保留，不默认只取最大组件。结构指标与二维轮廓均为代理证据，不能据此编造真实三维距离或叶片厚度精度。

本轮还完整运行同一 Alpha≥0.5 过滤协议下的原生 TSDF，结果近乎不变。E11 表面深度→NKSR 可选补充在保持 18,217,854 点和共同重建参数下，默认分配器及官方 expandable_segments 分配器各一次均 OOM；第二次已消除大部分闲置保留显存，但仍需申请 11.55GiB，超出当前可用容量。因此该项不作为成功 Mesh、不赋予零分，也不能用其硬件失败断言 NKSR 对 2DGS 一定更差。主要后端选择由完成的照片与原生 TSDF 证据支持。主 M5 仍为与官方提取兼容的原生 expected surface depth，主 M4 使用 GGGS median surface depth。

## 视觉与资源证据

已直接查看固定 0、53、105 的真实 native192px 叶片窗口组合（768×648）。E11 在个别局部保留条纹，但视角 53 有明显黑色条带横穿叶片/叶柄区域；E8/E10 的叶脉与边缘仍有残余模糊。后续 raw/post Mesh、Depth/Normal 与固定八视角对照一并保存在 `comparisons/visualizations/`，显示仅统一 nearest 缩小或保留 native crop，没有锐化和额外插值。

E11 Torch 峰值 allocated 2.6202 GiB、reserved 4.6836 GiB；E10 allocated 8.5801 GiB。E11 使用 E6 的 CPU 图像缓存和逐视角传 GPU，环境版本也不同，因此不能把差值解释为后端固有显存节省。完整训练墙钟 48.52min，相对 E10 40.21min 高约 20.7%；模型更小不等于训练更快或表面更准确。

## 可复核文件

`comparisons/rendering_comparison.csv`、`comparisons/E11_E10_paired_render_deltas.json`、`E11_2DGS_Corrected/per_view_metrics.csv`、`E11_2DGS_Corrected/training_complete.json`、`E11_2DGS_Corrected/actual_training_parameters.json`、`configs/actual_commands.jsonl`。208 帧 RGB/Depth/Normal/Alpha 的真实路径见各 `surface_inputs/*/rendered/` 索引；未重复复制大文件，也未重训 E8/E10。

来源：[官方 2DGS](https://github.com/hbb1/2d-gaussian-splatting)、[官方 GGGS](https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids)，本地真实训练、浮点数组和统一评价。博查检索因额度不足不可用；源码按固定版本直接核对。

主矩阵及Ω同源补充现已完成所有208视角raw/post评价；完整路线决策见`FINAL_PIPELINE_DECISION.md`。E11可选NKSR的两次同参数OOM明确记录，不计入成功Mesh。
