# ForeSplat 盆栽与番茄评估指南

**当前状态：两批重建、尺度记录、叶网格和相应图像已在项目中。人工复核通过的身份对应可以用于定位模型；论文所需的最终质量评分、独立分割误差和同口径表型误差需要单独计算。** 论文以番茄为主要应用；19 株、9 类盆栽验证流程在多种植物形态上的适用性。42 个番茄是三个品种在多个时间点的扫描记录，不能写成 42 株独立植株。两批分别报告。

手量从 [公制测量录入与误差汇总.xlsx](/data/fj/F2DMAS/00-论文优化重构/数据管理/00-规范与索引/ForeSplat_公制测量录入与误差汇总.xlsx) 开始。`整株` 和 `叶片` 已列出样本、实际模型、参考来源及图像；填两侧数值和每项质控状态，`误差汇总` 自动计算有效配对数、MAE、RMSE 和平均偏差。初步自动值放在单独列，未经核验不会进入误差。未找到端点时留空。

## 1. 评估取样和分母

| 评估 | 样本 | 论文怎么报 |
| --- | --- | --- |
| 重建完成度与质量 | 19 株盆栽 + 42 个番茄扫描 | 两批真实 COLMAP、GGGS、TSDF 和植物网格均已完成，分别为 19/19、42/42。质量评分另算，按有有效测试视角或独立参考的实际分母报告；无厘米尺度的样本仍可参加不依赖尺度的评分。 |
| 整株公制表型 | 19 株盆栽 + 19 个通过尺度质控的番茄扫描 | 测植株可见高度和最大水平冠幅。38 是拥有厘米网格的样本数，株高和冠幅各报有效配对数。现有 18 个番茄株高初步配对使用官方点云派生值，需核对定义再定稿。 |
| 叶实例分割 | 盆栽 57 片指定叶；番茄对 42 个扫描中所有 245 片官方标注复叶开展评估 | 盆栽有独立轮廓后报告可见叶的二维掩膜 IoU；番茄用可可靠配准的官方逐点标注评估三维实例质量，未匹配项保留，公布纳入的扫描和叶数。身份核验不等于轮廓准确度。这个评估不必限于已标定的扫描。 |
| 单叶公制表型 | 盆栽 57 个确认对应；番茄的公制子集含 70 片官方复叶，其中 67 个有具体优先配对 | 先量同一叶的直线叶长、同一横向位置的直线叶宽，各报有效配对数和误差。面积仅报告有质控的观测单侧面积，取得同口径独立参考后才比较面积误差。 |

尺度标定：盆栽由多视角 2 cm 蓝块确定尺度（18 株自动轮廓、孔雀竹芋复用既有人工标注轮廓），19/19 通过，最终轮廓拟合 RMS 为 0.568–1.305 px。番茄以多视角标定靶恢复尺度，共 19/42 个扫描通过（Beefsteak 5/13、Big Boy Hybrid 8/13、Celebrity Hybrid 6/16），其余 23 个保留原重建单位。番茄从原 RGB 重跑 COLMAP，相机没有借用官方标定结果；官方扫描标签与点云用于独立对照，标定靶的已知几何提供模型尺度。`CH_07112025` 有公制模型、官方点云及逐点标签，但没有对应的 CloudGraph 条目；在参考点云单位和方向核实之前，该扫描不计算公制参考误差。

按官方逐点标注的器官类型，42 个番茄扫描共 245 片复叶、35 片子叶。复叶对应表中，184 条有独立 PLY（69 条厘米网格、115 条原单位网格；其中 10 条有歧义），61 条没有分配到模型。19 个公制扫描内的 70 片复叶中，67 条有明确优先候选（40 条高置信、27 条建议），另有 2 条歧义、1 条无匹配。作者已反馈查看了网页展示的叶片并认可对应；文件仍保留原建议、歧义和未匹配状态，也保留逐性状端点单独审核。现有自动预填仅覆盖 28 条中的 14 组直线叶长与 24 组中部横向宽度（有重叠），尚不是最终准确度。`BBH_06172025` 有独立的配准尺度提示，身份对应仍可使用，尺寸入表前先复核。预测候选数和 CloudGraph 复叶表中 `CL` 行数不是最终表型配对分母。

## 2. 怎么做

1. **重建评分。** 直接从批次表统计真实 COLMAP、GGGS、网格及尺度通过数。对比方法时，为同一扫描使用相同的测试 RGB、相机、分辨率和评估区域；核实测试视角没有用于被评估阶段。Gaussian 渲染可选 PSNR/SSIM；网格轮廓可选独立人工前景掩膜的 IoU 和边界 F1。已有番茄 `heldout_silhouettes.csv` 的对照是 SAM3 掩膜，只作为诊断。更细体素版是另一版网格，单独记录，不能把它当作原 57 片叶的几何来源。
2. **株高和冠幅。** 打开 `plant_only/plant_only_cm.ply`，只量植株，厘米模型不再乘比例。用 `calibration/reference_frame.json` 里的估计竖直方向作起点，检查后统一两端测量方向。株高用最高、最低可见点的竖直坐标差；冠幅用垂直于该方向的平面上、植株投影的最大点间距。参考端用同一口径；盆栽历史实测须核对同株和测法，番茄可在对应扫描的官方参考点云上重测。官方原始坐标需按 CloudGraph `meta.transform` 确认方向和单位，其配对脚本以 `100 / scale_div` 从原坐标转换为 cm；配准额外估计的比例只用来做身份关联，不能乘到模型侧的测量值上。官方 X/Y 冠幅不能直接当作最大水平冠幅。若原参考值是土面至顶点高度，也不要直接与可见跨度相减。
3. **叶实例分割。** 盆栽从 57 片叶的对应照片出发，先在原始 RGB 上独立描出可见叶边界，再比较预测投影掩膜。番茄优先用官方逐点标签：将预测网格和官方点云对齐到共同坐标，在固定的表面距离和共同观测范围内以参考点为单位计算实例 IoU，再做一对一匹配。在完整标注的评估范围内才能报告查准率、召回率和漏检；只复核指定叶片时，报告该子集的匹配和 IoU。盆栽二维 IoU 与番茄三维 IoU 分开列。官方番茄 `Compound Leaf-Node` 包括小叶及叶轴，叶柄与它分开标注；本轮不把 `Leaf_NNN` 小叶候选当成整片复叶真值。
4. **叶长、叶宽和叶面积。** 先配对同株或同次扫描、同一叶，再比较性状。叶长是相同叶基至叶尖的三维直线距离：盆栽从叶片基部到尖端；番茄复叶按官方器官边界，从复叶基部到顶端小叶尖。叶宽先在参考和模型上对齐同一横向测量部位，再取两侧叶缘点的三维直线距离。每对保存端点坐标及标记截图。现有曲面路径叶长和官方 `Thickness` 不直接代入。面积只统计一侧可见叶面；番茄整片复叶可汇总各小叶的单侧叶片面积，同一面不得重复计数，叶轴不计入面积。缺边、明显残缺或没有同口径独立参考的，保留观测值或质控状态，不填整叶面积误差。人工在同一份 ForeSplat PLY 上量的是人工与自动测量的一致性；重建尺寸误差需用实物记录或独立参考点云。
5. **录入与汇总。** 整株以同一 `数据集 + 植株/扫描 ID` 配对；叶片另用盆栽 `手写编号` 或番茄 `官方点标签` 加预测 ID。保留参考来源、工具、日期、模型版本、方向或端点截图和质控备注。每项两侧值均有效且质控通过后，按组报有效配对数、MAE、RMSE 和有符号偏差（模型值减参考值），绘制 1:1 对照图；长度用 cm、面积用 cm²。同株多叶和同株重复扫描保留分组标识，不当作独立植株。后续如计算区间，以可追溯的植株或时间序列为聚类单位。历史 Excel 两份的人工真值列一致、部分旧虚拟值不同；本次将核对后的真值配到新网格重测值，不复用旧虚拟值。

## 3. 数据位置

项目路径以 `/data/fj/F2DMAS` 为根目录，`<plant>` 为盆栽 ID，`<scan>` 为番茄扫描 ID。

| 要找什么 | 路径或打开入口 |
| --- | --- |
| 盆栽汇总 | [19 株批次表](/data/fj/F2DMAS/experiments/ForeSplat_MultiSpecies_Phenotyping/summary/all19_results.csv)；[重建总览图](/data/fj/F2DMAS/experiments/ForeSplat_MultiSpecies_Phenotyping/summary/all19_overview.png) |
| 盆栽尺度质控 | [A1 逐株标定表](/data/fj/F2DMAS/2026-11-18修改后论文/A1_逐株标定记录.csv)；[当前稿表 7](/data/fj/F2DMAS/2026-11-18修改后论文/ForeSplat_中英文对照.md:869) |
| 盆栽重建和整株测量 | `experiments/ForeSplat_MultiSpecies_Phenotyping/<plant>/` 下的 `gaussian/metric_gaussians.ply`、`mesh/mesh_cm.ply`、`plant_only/plant_only_cm.ply`、`phenotype/plant_measurements.csv`、`calibration/scale_calibration.json`；原图在 `common_input/images/` |
| 盆栽植物与花盆、制图来源 | 各株 `plant_only/plant_only_cm.ply` 和 `plant_only/pot_cm.ply` 是独立网格；`summary/` 里的 `rgb.png`、`gaussian.png`、`mesh.png`、`plant_only.png`、`instances.png`、`measurement.png` 是现有预览，见 [孔雀竹芋图像目录](/data/fj/F2DMAS/experiments/ForeSplat_MultiSpecies_Phenotyping/KongQueZhuYu/summary)。 |
| 盆栽 57 片模型 | [57 片路径总表](/data/fj/F2DMAS/experiments/ForeSplat_MultiSpecies_Phenotyping/summary/three_leaf_model_locations.csv) 的 `model_local_path` 是每片叶的绝对 PLY 路径，`viewer_local_path` 和 `original_rgb_local_path` 是查看和原图路径。实际模型在 `<plant>/leaf_instances/Leaf_NNN/leaf_cm.ply`。[旧网页索引](/data/fj/F2DMAS/experiments/ForeSplat_MultiSpecies_Phenotyping/summary/three_leaf_model_locations.html)的相对图片和下载链接依赖复核服务，离线使用路径总表或上面的 Excel。 |
| 盆栽人工参考 | [历史实测 Excel](/data/fj/F2DMAS/植株数据.xlsx)；[57 片配对和测量表](/data/fj/F2DMAS/experiments/ForeSplat_MultiSpecies_Phenotyping/summary/three_leaf_measurements.csv)；各株 `phenotype/manual_reference.csv`。同品种有多株，历史 Excel 的行须人工确认；已有自动叶长列不能当作本协议的直线叶长。 |
| 盆栽细体素版 | [细体素网格索引](/data/fj/F2DMAS/experiments/ForeSplat_MultiSpecies_Phenotyping/mesh_2dgs_official_voxel/mesh_index.html)；[该版批次表](/data/fj/F2DMAS/experiments/ForeSplat_MultiSpecies_Phenotyping/mesh_2dgs_official_voxel/all19_2dgs_voxel_results.csv)。网格在 `mesh_2dgs_official_voxel/<plant>/`。 |
| 番茄汇总及筛选 | [42 个扫描总览](/data/fj/F2DMAS/experiments/ForeSplat_TomatoPGT_Independent/summary/batch_index.html)；[批次表](/data/fj/F2DMAS/experiments/ForeSplat_TomatoPGT_Independent/summary/batch_results.csv) 里筛 `metric_mesh=True` 得 19 个扫描；[初步株高配对](/data/fj/F2DMAS/experiments/ForeSplat_TomatoPGT_Independent/summary/batch_height_pairs.csv)。 |
| 番茄复叶核验和图像 | [带原图和对照图的复核页](/data/fj/F2DMAS/experiments/ForeSplat_TomatoPGT_Independent/summary/compound_auto_review/index.html)；[逐片复叶路径表](/data/fj/F2DMAS/experiments/ForeSplat_TomatoPGT_Independent/summary/compound_auto_review/compound_match_suggestions.csv)，其中 `model_local_path` 是 PLY 绝对路径；[初步端点和尺寸表](/data/fj/F2DMAS/experiments/ForeSplat_TomatoPGT_Independent/summary/compound_auto_review/provisional_leaf_measurements.csv)。扫描 `summary/` 下也有与盆栽相同命名的图像和 `overview.png`。 |
| 番茄重建和整株测量 | `experiments/ForeSplat_TomatoPGT_Independent/<scan>/` 下的 `gaussian/metric_gaussians.ply`、`mesh/mesh_cm.ply`、`plant_only/plant_only_cm.ply`、`phenotype/plant_measurements.csv`、`calibration/scale_calibration.json`。未通过尺度质控的扫描保留 `gaussian/original_gaussians.ply`、`mesh/mesh_units.ply` 和 `plant_only/plant_only_units.ply`；`units` 不是 cm。原图可从各扫描 `raw_rgb/` 或[官方原始目录](/fj/end-paper/tomato_3d_benchmark/02_raw/TomatoPGT)打开。 |
| 番茄叶模型 | `<scan>/compound_leaf_instances/Compound_NNN/leaf_cm.ply` 是复叶候选，`<scan>/leaf_instances/Leaf_NNN/leaf_cm.ply` 是小叶候选。候选文件夹还有 `viewer.html`、`reference.png` 和 `reference_rgb.png`；复叶另有 `compound_leaf_instances/index.html`。未定标扫描用 `leaf_units.ply`。 |
| 番茄官方参考 | 有 CloudGraph 条目的扫描，其 `<scan>/summary/official_reference.json` 列出官方数据来源，`<scan>/phenotype/official_compound_traits.csv` 保存官方条目。官方点云在 `/fj/end-paper/tomato_3d_benchmark/03_standardized/TomatoPGT/<scan>/annotations/reference_cloud/official_point_cloud.ply`；逐点坐标和标签在同一 `annotations/semantic/` 的 `*_annotated_ext.txt`，整理后的逐点标签在 `/fj/end-paper/tomato_3d_benchmark/04_annotations/instances/TomatoPGT/<scan>_labels.csv`。 |

可先试一对样本：盆栽用 [孔雀竹芋整株 PLY](/data/fj/F2DMAS/experiments/ForeSplat_MultiSpecies_Phenotyping/KongQueZhuYu/plant_only/plant_only_cm.ply) 和 [1 号叶 PLY（Leaf_006）](/data/fj/F2DMAS/experiments/ForeSplat_MultiSpecies_Phenotyping/KongQueZhuYu/leaf_instances/Leaf_006/leaf_cm.ply)；番茄用 `BFS_06262025` 的 [整株 PLY](/data/fj/F2DMAS/experiments/ForeSplat_TomatoPGT_Independent/BFS_06262025/plant_only/plant_only_cm.ply)、[复叶候选索引](/data/fj/F2DMAS/experiments/ForeSplat_TomatoPGT_Independent/BFS_06262025/compound_leaf_instances/index.html)、[官方参考点云](/fj/end-paper/tomato_3d_benchmark/03_standardized/TomatoPGT/BFS_06262025/annotations/reference_cloud/official_point_cloud.ply) 和 [官方复叶条目](/data/fj/F2DMAS/experiments/ForeSplat_TomatoPGT_Independent/BFS_06262025/phenotype/official_compound_traits.csv)。先确认参考点云的长度单位和对齐方式，再把整个数据集按同一规则录入。番茄官方器官定义见 [TomatoPGT 工具手册](/fj/end-paper/tomato_3d_benchmark/02_raw/TomatoPGT/tomatoPGT-tools/README.pdf)，数据 DOI 为 `10.17632/72md54c7n7.1`。

## 4. 和原稿的关系

- [11 月 18 日原稿](/data/fj/F2DMAS/2026-11-18修改后论文/ForeSplat.tex:756)已有 21 条 RGB 序列上的 ForeSplat/2DGS 重建评分、效率对照，以及 21 个整株和 63 片叶的旧表型一致性结果。这批数值保留原实验口径，不能移到 19+42 的新样本上。
- [现稿](/data/fj/F2DMAS/2026-11-18修改后论文/ForeSplat_中英文对照.md:859)已写入盆栽 GGGS–TSDF 批次、厘米标定质控、植物和花盆分离，以及 798 个候选和 57 片身份确认。额外实物证据是可按原图编号追溯的独立 PLY、可复核原图、叠加掩膜和模型版本。
- 在此基础上新加了公开番茄应用：同一套批处理从 RGB 独立恢复 42 个扫描的相机和几何，产出公制子集，并在整片复叶层面对照官方点标签和三维点云。这是从盆栽移到经济作物的外部流程验证。原稿以 2DGS 为重建后端，这批采用 GGGS–TSDF，需在方法和比较表中明确；不能把完成率归因于原稿的 2DGS 优化，也不能沿用旧协议的速度或画质优势。

论文主张可写为：**我们检验了从普通多视角 RGB 获取可追溯三维植物器官的流程能否跨植物形态运行，并在公开番茄扫描上检验其从独立相机恢复到可测网格的应用价值。** 结果按“重建覆盖 → 公制质控 → 复叶分割和身份 → 同口径测量一致性”组织。盆栽保留为形态覆盖证据；番茄主文优先放同一扫描的 RGB、植物网格、复叶实例和官方对照；各性状的配对统计完成后再配散点图。逐样本尺度、版本、形态覆盖和用于复核的图像索引进入补充材料。现有图 8 保留，图 10 未制作或编号。
