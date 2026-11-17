# ForeSplat 全19株公制表型

数据入口为原始 `configs/plant_manifest.json`；development / heldout_validation 原样保留。所有重建产物仅保存在服务器 `experiments/ForeSplat_MultiSpecies_Phenotyping/`，原始数据和历史报告不覆盖。

在仓库根目录执行：

```bash
uv run --no-project --python experiments/COLMAP_Gaussian_Mesh_Selection/.venv-rap/bin/python run_colmap_phenotyping_all19.py --all
```

同一入口支持 `--plant DouBanLv1`、`--from-stage metric`、`--until-stage calibration`。阶段依次为 colmap、sam3、calibration、gggs、metric、tsdf、instances、phenotype、manual、report、validation。已有等价阶段复用；失败记录在各株 `logs/` 和 `status.json`，然后继续下一株。只有一个驱动器可持有运行锁，所有 GPU 工作共用物理 GPU1 的锁。没有 C1/C2、五算法选型或新的 Gaussian/Mesh 算法。

暂停后使用 `--all --resume` 跳过已经通过整株交付验证的植株，继续其余植株；每次运行配置另存历史。被停止的训练如果没有优化器检查点，需要从头完成30K，已经完成的相机、分割和标定阶段仍可复用。

查看实际状态：

```bash
cat experiments/ForeSplat_MultiSpecies_Phenotyping/summary/progress.json
cat experiments/ForeSplat_MultiSpecies_Phenotyping/summary/totals.json
```

`summary/all19_results.csv` 和 `conclusions.md` 区分正在运行、失败、公制完成、完成但存在交付限制。`input_inventory.json` 单独记录已有真实 SfM 与不宜复用的 Omega/VGGT 模型，不能把清点结果当成本次全流程完成数。

启动逐叶查看和人工参考录入：

```bash
uv run --no-project --python experiments/COLMAP_Gaussian_Mesh_Selection/.venv-rap/bin/python \
  00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/phenotyping/batch_annotation_server.py
```

访问 `http://127.0.0.1:8766/`。远程使用端口转发 `ssh -L 8766:127.0.0.1:8766 <服务器>`，再在本机浏览器打开此地址。也可以下载整个输出目录后直接打开 `summary/leaf_mesh_index.html`；逐叶页面包含完整原始几何，Plotly 资源在本地共享，无需联网。

每株 `leaf_instances/Leaf_NNN/` 中有 `leaf_cm.ply`、`reference.png`、`reference_rgb.png`、`instance.json`、`landmarks.json`、`viewer.html` 和原始面/顶点索引。尚未标定时只有 `leaf_units.ply`，页面明确标注模型单位。三维页面允许旋转，显示并点击原始顶点，计算沿三角网格边的最短曲面路径，下载选点 JSON。遇到不连通区域返回无路径，保留原始孔洞。

人工参考通过 `/review` 或 `phenotype/manual_reference.csv` 录入。只有明确确认同株、同叶、同次采集，且填写日期、测量工具与操作人的真实值进入误差计算。输入不全时 MAE、RMSE、R² 留空；误差表报告候选几何的适用限制。`/annotation?plant=<植物名>` 复用现有 SAM3 点、框、负向点、叶基叶尖及整株基部复核界面，支持动态宽高和全部稳定 Leaf ID。人工修改另存 `annotations/`；显式应用命令为：

```bash
uv run --no-project --python experiments/COLMAP_Gaussian_Mesh_Selection/.venv-rap/bin/python \
  00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/ForeSplat_AutoGaussianCleaner/phenotyping/batch_apply_corrections.py DouBanLv1
```

人工修正固定身份并重新融合可见原始三角面，旧派生文件保存到 `annotations/revisions/`。无人工确认不会编造完整叶片。自动候选基部、竖直方向和冠幅轴均在 `calibration/reference_frame.json` 中说明。

尺度约束：蓝块独立 SAM3 掩膜经过多视角三维立方体联合拟合；遮挡/无关蓝物体用固定像素残差和严格多数视角拒绝，不接受少量挑选出来的视角。历史边界提示单独标识，孔雀竹芋复用已核验历史标定。XYZ、三轴 log-scale、filter_3D 与相机平移同步按厘米转换，旋转、Opacity、SH 等逐字段保持不变。Mesh、Depth、voxel、SDF 及后续表型已在同一厘米坐标中，后续乘数恒为1。原始世界轴不旋转。

相机/RGB 约束：历史 COLMAP 输入图像的 SHA256 对应当前 FFT 图像；历史输入若是黑色背景的 PNG，则先定位唯一候选，再逐像素核对全部保留的非黑 RGB 像素。必要时联合重命名相机图像观测与 RGB，不能只凭同名匹配。无法核验或损坏的输入不进入训练；不足90%可核验视角时重跑真实 SfM。损坏历史 RAP 掩膜不读取为真值，优先查有效存档，否则在进入 Mesh 前用 SAM3 重做。没有合格原生 SfM 时使用现有 COLMAP 执行真实 SIFT/匹配/增量重建。

总览六个面板使用同一个实际相机视角，优先完整容纳观测到的植物，再比较画面中的投影大小。`summary/preview_view.json` 保存所选视角及范围；Gaussian 图片取已缓存的真实渲染，不额外训练或改变几何。

GPU 原生 GGGS 环境、SAM3 环境和既有 TSDF 环境分别复用，通过 uv 调用。GGGS 原生训练固定30K/SH3/SG0/seed0，保留最终原生 PLY；新任务不保存无使用需求的最终优化器副本。大型 PLY 采用经过独立 plyfile 校验的等价二进制读写，按约19.5 MiB/s节流，避免逐浮点 Python 解析；不会改变模型字段或算法。必要的可视化 HTML 与渲染缓存均保留来源。

原生 GGGS 渲染器还包含以模型单位写死的中值深度搜索范围0.4和近裁剪距离0.2。厘米推理必须把它们同步转换到0.4*s和0.2*s。`metric_renderer_units.py` 在输出目录隔离构建原生渲染器，仅暴露并转换这两个有长度量纲的参数；默认倍率1，训练扩展、搜索步骤和重建算法均保留原样。每株保存 `mesh/renderer_units.json`；原单位与厘米模型的实际渲染须通过深度值、有效范围、RGB及相机一致性检查。没有把厘米 PLY 再乘一次尺度。

验证重点为逐字段尺度/非长度参数、相机投影不变性、每叶源三角面及顶点坐标完全一致、面身份唯一、至少3视角身份支持、厘米值只在标定成功后出现。对照论文：<https://doi.org/10.1016/j.compag.2026.112467>，仅参照功能颗粒度。
