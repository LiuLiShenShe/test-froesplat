

全部完成并经过真实数据验证。以下为交付总结。

交付：ForeSplat 专用 VGGT-Ω 前端适配模块

代码根目录：00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/（基于 tomhol/vggt-omega 社区仓库，未改网络结构/权重加载/推理逻辑，未改 GGGS，未实现 NKSR）

1. 关键修改文件与位置

文件: vggt_omega/utils/load_fn.py
修改: 原 load_and_preprocess_images() 及私有助手逐字节保留；新增 load_and_preprocess_images_with_masks()（RGB 与
Mask 同一裁剪框/缩放/padding；Mask 用 NEAREST+零填充，禁用双三次）、_pixel_transform()（p_model=A·p_original，PIL
 像素中心约定）、_load_foreground_rgb_image()（RGBA→黑底合成，与 RAP-FSAM3 输出形式一致）、逐帧记录（含
pixel_transform）、transform_intrinsics_mo
────────────────────────────────────────
文件: visual_util.py
修改: 扩展社区 save_colmap()：新增
foreground_masks、preprocess_transforms、filter_depth_edges（默认关，保护叶缘）、return_stats 等参数；内参
K_original=A⁻¹K_model（不再是朴素 s_x/s_y
p_original=A⁻¹p_model；场景对齐前置到创建时一次施加（实测发现 pycolmap≥3.x 事后回写静默失效的社区
bug）；_build_valid_mask() 严格按 foregrouconf &
valid_depth，且置信度分位数只在前景点上统计；track 每点恰 1 个观测（单视角深度反投影，非 SfM
三角化，已在元数据/PLY header 显式声明）
────────────────────────────────────────
文件: foresplat_adapter/
修改: 新增：dataset.py（配对保序、前缀剥离）、inference.py（无 Gradio）、export.py（GGGS
目录装配）、ply.py、verify.py（直接调 GGGS readColmapSceneInfo）
────────────────────────────────────────
文件: tools/
修改: run_vggt_omega_gggs_export.py（主 CLI）、diagnose_reconstruction.py、visualise_export.py
────────────────────────────────────────
文件: tests/、docs/
修改: 19
项接口测试；VGGT_OMEGA_TO_GGGS.md（转换语义说明）、README_VGGT_OMEGA_GGGS_FRONTEND.md（环境/权重/命令）、RUN_REPOR
T.md（实测报告）

2. 真实运行结果（KongQueZhuYu RAP-FSAM3 21掩膜）

- 权重 sha256 校验通过（c02da418…7934），sunexpected
- 连续 20 帧 @512 → 1,137,200 个前景点：焦距跨帧差异 4.6%、跨视角深度中位误差 1.04%、5% 内点率
  80.6%、逐点重投影误差 <1e-5 px
- GGGS 官方读取验证通过：readColmapSceneInfo 全部检查 ✅，且 GaussianModel.create_from_pcd() 成功初始化 113.7 万高斯
- 可视化：runs/KongQueZhuYu_20frames_512/Gntcloud.png（可清晰辨认盆栽植株与花盆，仅目标植物无背景）、preview_mask_overlay.png、model_inputs/、model_masks/、predictions.npz
- 单元测试 19 passed（含 RGB 路径与社区 loader 逐元素一致、掩膜几何一致、A 与 PIL 重采样质心吻合）
- 明确未验证：GGGS train.py 完整训练（需编译其 CUDA 光栅化扩展，且本阶段不允许改 GGGS）；宽基线子集（stride 26）一致性显著下降，建议使用连续窗口——详见 RUN_REPORT §6

3. 运行命令

python tools/run_vggt_omega_gggs_export.py \
  --images-dir "$RAP/前景图" --masks-dir "
  --image-prefix-to-strip crop_ --mask-pre
  --checkpoint weights/vggt_omega_1b_512.pt \
  --output-dir <GGGS_Dataset> --image-resolution 512 \                                                              --confidence-percentile 20 --max-points >

重要钉版：pycolmap 必须 3.10.0（4.x/3.11+ 与社区 save_colmap() API 不兼容）；CUDA 12.6 编译 simple-knn 需补       #include <cfloat>（上游已知问题，本机从 /t

信息来源（均实际联网抓取/克隆）：github.com/tomhol/vggt-omega、modelscope.cn/datasets/hengyelyu/vggt-omega-weights、github.com/HKUST-SAIL/Gaussian-Splatting

【任务已结束】
完成：VGGT-Ω 前端适配（掩膜感知预处理 + 仿 19 项接口测试通过，并用真实 RAP-FSAM3数据完成 20 帧端到端运行——GGGS 可读取该数据集并成功初始化 113.7 万前景高斯点。