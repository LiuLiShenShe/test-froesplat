# 原版 GitHub Gaussian 清理器适配

在 ForeSplat_AutoGaussianCleaner 目录运行，使用现有 uv 环境 `../.venv`。没有新增依赖，没有安装 Taichi；清理过程看不到 CUDA 设备。依赖 NumPy/SciPy/scikit-learn 已由现有环境提供。

```bash
../.venv/bin/python -m external_cleaners.fetch_sources
../.venv/bin/python -m external_cleaners.run \
  --input /absolute/native/point_cloud.ply \
  --output /absolute/new_result/cleaned_gaussians.ply \
  --method corbeau --strength medium --io-limit-mib 20
```

`--method converter` 固定官方 `min_opacity=5`（5/255）、`sor_intensity=8`；`--method slim` 固定 prune 5th percentile，再 clean 的原版默认值。Corbeau 仅允许官方 medium/strong，先 medium，见到明显大云残留才 strong。

每次只有一次完整读、一次完整写。直接调用未修改的原版函数：Corbeau 返回 keep mask；另两种把 int64 original_id 当附加字段传过原版筛选，避免浮点坐标匹配或两阶段索引错位。splat-slim 输入属性转换 float32 遵循其原版 load_splat 定义；导出始终从原始记录取字节。没有改变 GGGS 的 filter_3D、SH、旋转、透明度或尺度。默认 CPU affinity 4 核，8 GiB 保守预检，拒绝覆盖源和既有结果。

结果只含一份最终 PLY 和小型 cleanup_report.json。资源计量覆盖限速读取、原版函数导入/计算、最终写出与 fsync；不含 Python/NumPy 启动、下载、独立核验和 GPU 评价。峰值 RSS 是该进程操作系统高水位。磁盘峰值报告为 Meter 观测的应用层尾随一秒吞吐，设备物理峰值未测量；内核回写不由应用限速器硬性约束。

当前固定的 3dgsconverter CPU 分支存在上游缺陷：remove_flyers 计算 mask 后没有赋回 self.data，因此按原样调用时 SOR 没有产生删除。本适配没有偷偷修复；报告明确标记。不能据该输出宣称“有效 CPU SOR 已完成”。

独立逐字节读回核验（不计入清理资源）：

```bash
../.venv/bin/python -m external_cleaners.verify \
  --plant-root experiments/external_cleaners_20261114/KongQueZhuYu
```

GPU1 最终评价入口，JSON 是方法名到最终 PLY 绝对路径的映射：

```bash
../.venv/bin/python -m fast_ply.evaluate \
  --scene experiments/Historical_TwoStage_Cleaner/scene_configs/KongQueZhuYu.json \
  --external-models reports/external_cleaners_20261114/KongQueZhuYu_models.json \
  --output /absolute/new_evaluation --views 16 --max-side 960
```

该模式只加载最终模型，跳过 Fast 消融；孔雀竹芋使用已有 fixed8 自由视角及 16 观测视角，C2 原阈值及真实 additional_rounds/round_05/gaussians.ply。复用 photometry/artifact_metrics，不改变黑像素或 PSNR 定义。每个方法保存相同相机的渲染 PNG 及 contact 图。再次执行需新目录，不覆盖评价 JSON。

源码版本、官方永久链接、检索失败状态见 `reports/external_cleaners_20261114/sources.json`；本地 upstream 为原版 Git checkout，算法源码不改。
