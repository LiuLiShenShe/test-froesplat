# 孔雀竹芋器官三维表型

本轮结果入口：[summary/index.html](summary/index.html)。先读 [summary/conclusions.md](summary/conclusions.md) 中的测量限制。

本轮已实际执行 2 cm 标定、SAM3 植物/盆分离、12 片叶子的跨视角关联及曲面厘米测量。所有单叶均有孔洞或碎片，当前 CSV 是局部观测值，不是完整叶片精度验证。整株基点与支撑法向待复核；完整叶片有效测量数为 0。没有确认的人工配对，误差指标留空。

只使用已有真实 COLMAP、GGGS 30K Original 和推荐 bounded TSDF Post。新网格是原始顶点、法向与三角面的子集，原文件不覆盖。不重训，不执行 C1/C2，不启动其余 18 株。物理 GPU1 串行，uv 使用已有环境；大型 Mesh/NPZ/图像留在服务器。

## 交互复核

在 `/data/fj/F2DMAS` 下运行：

```bash
uv run --no-project --python experiments/COLMAP_Gaussian_Mesh_Selection/.venv-rap/bin/python experiments/COLMAP_Phenotyping/KongQueZhuYu/scripts/annotation_server.py
```

打开 `http://127.0.0.1:8765`。从个人电脑连接服务器时可使用 SSH 端口转发 `-L 8765:127.0.0.1:8765`。服务默认仅监听服务器回环地址。支持视角选择、固定 Leaf ID、正负点、单叶框、SAM3 Mask、同 ID 其他视角、叶基/叶尖/边界和整株基点。叶片显示名称可以修改；固定 Leaf ID 用于对应实测表。SAM3 请求会等待 GPU1 锁并按需读取既有权重，可能需数分钟。

修正单独写入 `annotations/user_edits.json`、`annotations/user_masks/`。确认身份后勾选“已人工核对”再保存。二维点以相机射线首个原始 Mesh 交点提升到三维；遮挡处可能命中前方叶片，需换可见视角。叶基/叶尖必须落在对应叶片的同一个连通单侧观测片上，否则保留待复核状态。当前不使用点击来虚构缺失表面；边界点击记录作为人工复核资料，尚不直接触发三角面切分。修改整株基点后会生成 `annotations/plant_base.json`。

应用已确认修正：

```bash
uv run --no-project --python experiments/COLMAP_Gaussian_Mesh_Selection/.venv-rap/bin/python experiments/COLMAP_Phenotyping/KongQueZhuYu/scripts/apply_corrections.py
```

只更新派生叶片标签、Mesh、标志、CSV 与报告，不改历史掩膜、原始 RGB、源 Mesh 或相机。实际后端测试掩膜保存在 `annotations/validation/`，不是用户标注，也不会参与实例融合。

## 人工实测

补录 `phenotype/manual_reference.csv`：固定 LeafID、同株同次采集的 `capture_session`、`measured_at`、`measurement_tool`、`operator`、`pair_confirmed=true`。叶长沿中脉/中心线、宽度最大有效横向宽、面积扫描/面积仪测单侧；株高及冠幅需使用相同基点/轴。禁止按历史 Excel 叶片序号猜配对。

```bash
uv run --no-project --python experiments/COLMAP_Gaussian_Mesh_Selection/.venv-rap/bin/python experiments/COLMAP_Phenotyping/KongQueZhuYu/scripts/manual_errors.py
```

有明确配对才计算误差。当前无配对，`paired_measurements.csv` 只有表头，散点图标注 n=0，MAE/RMSE/Bias/相对误差/R² 留空。自动网格仍是局部曲面，即使补录后也须保留这一限制。

## 数据与复现

- `calibration/`：5 视角、8 角点、12 相邻边、2 cm 尺度和基准；历史轮廓参与拟合，不是独立精密标定真值。
- `semantic_masks/`：15 视角首轮后扩展至 83 视角，保存首轮与框/点修正结果。
- `plant_only/`：完整源面标签及植物/盆三角面子集；无支持的蓝块不导出模型。
- `leaf_instances/`：12 个独立叶片、原始面/顶点索引、固定 ID 掩膜、边界、曲面路径、三维身份验证。
- `phenotype/`：厘米单位 CSV、人工模板、配对与误差审计。
- `summary/`：图片、离线可旋转三维视图、报告、来源哈希与验证结果。
- `scripts/`：链接到 `ForeSplat_AutoGaussianCleaner/phenotyping/` 的必要实现。

现有环境依赖继承既有工程及其 `.pth`，不是可移植的新环境安装声明。完整阶段顺序为 `calibrate_cube.py` → `sam_inference.py --views pilot --leaves` → `prepare_guidance.py` → `sam_inference.py --views guided --correct` → `associate_leaves.py` → `leaf_sam_refine.py` → `fuse_semantics.py` → `extract_instances.py` → `review_geometry.py` → `measure_phenotypes.py` → `manual_errors.py` → `build_report.py` → `validate_delivery.py`。当前结果已生成；通常直接复核、应用修正即可，无须重跑全部推理。

资源 JSON 记录成功阶段，早期失败重试另见本地日志，不能将这些成功阶段的合计称为全部实际耗时。应用层 I/O 计量为每进程限速；不包含系统缓存、外部进程和全服务器总吞吐。来源与局部 SAM3 文件/权重 SHA256 见 `summary/source_manifest.json`。
