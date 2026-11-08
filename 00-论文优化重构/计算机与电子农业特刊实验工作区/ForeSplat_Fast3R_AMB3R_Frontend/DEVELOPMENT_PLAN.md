# ForeSplat Fast3R / AMB3R 前端开发计划

更新：2026-11-08（Asia/Shanghai）。仅验证 KongQueZhuYu；GPU 只允许物理 GPU1；依赖管理只用 uv。

## 当前执行记录

Stage 0–5全部完成。Fast3R全208单次全局推理；AMB3R OFF/ON全208同输入；原生SfM208/208注册（73KF/0unmapped）；混合SfM208/208注册（35KF/0unmapped）。完整原始输出已持久化，七方法定量/可视化、七问中文分析和接口说明已交付；七份Bundle v2审计全部通过。以下保留Stage 0审计时初始状态和原执行计划，最终实测状态见 experiments/KongQueZhuYu/run_registry.json 与 RUN_REPORT.md。

## 已核实输入与现有成果

- 用户确认 `/data/fj/F2DMAS/03-SAM/KongQueZhuYu` 是现有 RAP-SAM3/RAP-FSAM3 输出，无需重新分割。
- 210 个标准命名 RGB，210 个对应 `mask_NNNN.png`。`crop_0044.png`、`crop_0206.png` 无法解码，保留其清单和错误，不修改源文件。全部 208 个可读 RGB/Mask 为 2160×3840。
- 排序严格使用原始数字帧号。额外 `crop_0000_mask.png` 不作为 RGB；标准对应 Mask 使用 `mask_0000.png`。
- 旧 VGGT-Ω 全量结果位于同级 `ForeSplat_VGGT_Omega_Frontend/runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset`，208 帧。评价使用其 per_frame 原始深度预测，不能用已平均融合点云冒充原始预测。
- 本工作区旧 `Fast3R` 目录为空；已有 `frontends/fast3r/smoke_test.py` 需要修正官方返回值处理和实际输出字段。保留旧脚本，通过新独立 Runner 继续。

## 官方源码与论文

- Fast3R commit：`33104d4b5b8df43795ecded236194958bbdac572`。官方网络保留不变。权重本地 `weights/fast3r/model.safetensors`；逐次运行记录 SHA256。FAIR Noncommercial Research License。
- AMB3R commit：`92c4081f910f98e683503092b85301861519175e`。原始网络和 SfM 主流程保留不变；Adapter、输入/输出封装及记录器单独实现。
- 已直接获取 Fast3R 论文，读取 §3 模型/损失/坐标系与 §4 显存实现，并检查实际网络、双 DPT Head、多视角推理与 PnP 源码。官方文档已用 Context7 获取。
- 论文和源码的具体核验结论保存 `docs/*SOURCE_AUDIT.md`；不会仅依据 README 编写适配。
- 博查 API 已实际请求，返回 HTTP403 `You do not have enough money or package quota`，因此搜索发现服务不可用；后续证据来自直接获取的官方论文/源码，不伪称搜索成功。

## 执行与验收

1. 生成 `experiments/KongQueZhuYu/manifest.csv`，记录所有 210 帧的可读性、Mask 配对和 SHA256。统一输入保留完整视野，仿射像素变换矩阵含像素中心偏移；RGB bicubic，Mask nearest。默认不裁剪、腐蚀或调整曝光。
2. 新 Fast3R 冒烟保存 Global/Local Point Maps、Confidence、真实 PnP 状态及 Bundle。逐步增加视角，优先完整 208 视角在一次全局交互推理中完成。使用 BF16 和官方 Head 并行数设置；OOM 明确记录，不通过互不注册的分组拼接替代。
3. 修复 AMB3R 独立 uv 环境 CUDA 扩展兼容性。对相同输入/Mask/随机种子运行 `iters=0` 与 `iters=1`。优先全量；全量不可行时固定代表性子集并明确标注。
4. 调用原生 AMB3R-SfM，保留聚类、关键帧、粗注册、全局映射、非关键帧细化；记录全部失败或未映射帧。
5. `Fast3RForAMB3RSfM` 实现 `extract_amb3r_sfm_features` 与 `run_amb3r_sfm`；只使用 Fast3R 编码器特征。验证置信度原始数值范围和 C2W、Cluster 局部参考系传递。
6. 所有成功/失败实验进入统一比较表。评价覆盖率、叶缘覆盖率、密度、邻接可见区域一致性、置信度、轨迹连续性；自投影误差是相机/预测一致性指标，不能当作真实几何精度。扫描点云不用于配准或尺度修正。
7. 统一静态科学可视化用相同背景、颜色、点大小与观察轴。任意尺度模型仅在显示副本做明确标注的归一化，不改原始几何。无已配准 3D GT 时只展示 2D GT/原图，不能伪造 GT 点云比较。

## 统一接口

`frontends.common.load_inputs()` 返回输入张量、RGB、Mask、有效非 padding 像素和完整像素变换。
`export_bundle()` 保存原始预测、Global/Local 点图、Confidence、C2W/W2C、模型与原图内参、RGB/Mask、四种过滤 PLY、额外 q85 对照和每帧指标。Local 点云始终逐帧保存，不冒充全局对齐。
`FrontendBundle` 是后续 3DGS/Mesh 消费者的 CPU 读取接口。当前不宣称已验证 GGGS/COLMAP 兼容性，除非实际调用 `readColmapSceneInfo()`。

## 明确排除

本阶段不运行 EO、GGGS、2DGS/3DGS 训练、Mesh/NKSR、新分割、新平均融合算法或扫描点云真实尺度配准；不扩展到其他植物。

## 来源

- https://github.com/facebookresearch/fast3r
- https://arxiv.org/abs/2501.13928
- https://github.com/HengyiWang/amb3r
- https://arxiv.org/abs/2511.20343
- 本地源数据、manifest 与现有 VGGT-Ω RUN_REPORT。
