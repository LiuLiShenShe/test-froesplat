# 固定24帧背景恢复单因素控制

本实验比较RAP原前景RGB与“只在原Mask外恢复对应02-FFT RGB”的geometry输入。**它不是未经修改的original RGB**。原始分辨率2160×3840、源顺序、Mask、seed0、预处理、网络及原权重一致；原始图像/Mask均不写入，没有训练或重跑208。24源ID固定为：`0,1,19,20,38,39,57,58,76,77,95,96,114,115,133,134,152,153,170,171,189,190,208,209`。

baseline直接读取 `/data/fj/F2DMAS/03-SAM/KongQueZhuYu/crop_NNNN.png`；Mask读取对应 `mask_NNNN.png`。restored在原网格使用 `where(Mask, RAP_RGB, 02-FFT/NNNN.jpg)`，仅写新PNG。`prepared_sources.json` 保存实际文件SHA、尺寸和全部24帧PNG重读逐位断言：**Mask内变更总数0；Mask外与matched raw RGB全像素相同**。原JPEG与RAP的Mask内MAE最大仅.007213（0–255）；未用它替换RAP前景。模型RGB的后续bicubic/bilinear可在边缘混合改变的outside颜色，这是同一原预处理对新背景的预期响应，不能冒称模型网格的每个边缘foreground RGB仍逐位相同。模型Mask/A本身完全一致。

| 实际完成条件目录 | N | 原adapter调用秒 | peak allocated GiB |
|---|---:|---:|---:|
| Omega_RAP_baseline_24 | 24 | 5.836 | 7.869 |
| Omega_background_restored_24 | 24 | 3.430 | 7.882 |
| Standard_RAP_baseline_24 | 24 | 5.091 | 9.568 |
| Standard_background_restored_24 | 24 | 4.335 | 9.568 |

全部进程exit0，physicalGPU1已释放。时间是单次原adapter调用（含CPU输出），存在冷/热缓存差异，不能据此宣称背景恢复使模型加速。`inference_runs.csv` 和每条件metadata保存真实完整参数、dtype、time/peak。Ω保持384×688 balanced512；standard保持原square518，两套相机/depth流程均不执行point head。与70/110使用同一未修改模型调用和已有权重/uv环境，细节见 [view-count README](../view_count_control_v1/README.md)。

相机、原始depth/conf、精确float32模型RGB、nearest Mask、原/模型尺寸及A全部采用相同schema；每条件有 `cameras.json`、`cameras.npz`、`pixel_transforms.json`、`manifest.csv`、params/metadata/status/report及 `per_frame/*.npy`。统一CPU导出审计 [all_control_export_audit.json](../view_count_control_v1/all_control_export_audit.json) 实际passed；BG两model各24相同source的Mask SHA、A、dtype、seed、weights/sourceSHA全无差异，RGB差异由原始坐标完整pixel断言限定。

## 独立固定foreground匹配的实际结果

[evaluation/control_summary.json](evaluation/control_summary.json) 用同12对相邻source、同一旧RAP foreground SIFT/image-only RANSAC内点，**没有给恢复背景重新提特征以改变评价难度**。以下mask+finite基线，24个双向depth pair/36000查询；像素误差在shared384×688网格。

| 条件 | median Sampson(px) | depth→真实matched RGB(px) | ≤5%跨视角depth支持 |
|---|---:|---:|---:|
| Ω RAP | 9.802 | 23.300 | 52.69% |
| Ω 恢复Mask外背景 | .548 | 1.170 | 92.03% |
| Standard RAP | 37.150 | 117.290 | 23.78% |
| Standard 恢复Mask外背景 | .440 | 1.276 | 92.98% |

两个原模型在完全相同N/前景/Mask/权重下出现大幅几何收益，支持当前最早触发因素为**几何推理输入剥离背景上下文**，而非RAP叶片RGB被普遍模糊或模型架构本身必须更换。此前均匀降低N没有稳定修复，不能代替这个单因素证据。它尚不证明恢复背景的全208结果、扫描公制尺度、完整薄叶面精度或重新训练3DGS已通过。

## 可复用geometry RGB adapter

独立 [geometry_rgb_adapter.py](../geometry_rgb_adapter.py) 抽取实用算法：`restore_geometry_rgb(rap_rgb_path, mask_path, matched_raw_rgb_path, appearance_rgb_path=None, output_path=None)`。返回 `GeometryRGBResult`，含原网格uint8 geometry RGB、原bool Mask、metadata。要求数字源名匹配、原尺寸一致及0/255原Mask；不做crop/resize/exposure/erosion/model调用，不覆盖源文件，输出只允许新losslessPNG。

metadata明确分开 `geometry_rgb_source`（Mask外matched raw上下文）与 `downstream_appearance_rgb_source`（仍为未修改RAP RGB），两者坐标变换identity。后续几何可以利用背景，Foreground输出与appearance训练仍使用原RAP Mask/RGB；本阶段不运行新208或任何训练。

`validate_geometry_rgb_adapter.py` 已实际比较首/末source0/209：adapter输出与本实验原已保存PNG**全部像素相同**。结果 [geometry_rgb_adapter_validation.json](geometry_rgb_adapter_validation.json) 记录当前adapter SHA及完整两帧source-role元数据。

CPU准备中的一次重复seed关键字元数据错误已修复，完成的24像素准备重用，不重生成或改变图像；`prepare_attempt1_metadata_failed.log/json` 保存真实失败位置，发生在GPU模型加载/推理之前。全部最终条件status/报告是实际完成状态。综合归因见 [ROOT_CAUSE_ANALYSIS.md](../../ROOT_CAUSE_ANALYSIS.md)。
