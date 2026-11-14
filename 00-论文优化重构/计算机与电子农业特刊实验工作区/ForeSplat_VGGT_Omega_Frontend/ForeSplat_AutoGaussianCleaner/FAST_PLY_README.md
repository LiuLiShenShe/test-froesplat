# Fast Gaussian Cleaner

**研究原型：已完成4株代表模型验证，低I/O工程目标成立，但去云不足、跨株边缘/薄结构指标下降，尚未通过通用清理质量验收。不要据此启动19株批处理。**

独立 CPU 原生 GGGS PLY 清理器。默认20 MiB/s应用层读写限速、8 GiB工作集预算；保留每条留下记录的全部原始字段。不会加载相机或GPU，不生成中间模型、逐点CSV/NPZ或默认SuperSplat副本。算法和当前实测限制见 [研究报告](FAST_PLY_RESEARCH.md)。

使用前端目录现有的 uv 环境（`../.venv`），本次没有安装新依赖；默认仅依赖 NumPy、SciPy，CPU模块不依赖plyfile或torch。以下命令在 `ForeSplat_AutoGaussianCleaner` 下执行：

```bash
../.venv/bin/python fast_clean_gggs.py \
  --input /absolute/path/to/native/point_cloud.ply \
  --output /absolute/path/to/new_result/cleaned_gaussians.ply \
  --io-limit-mib 20 --memory-budget-gib 8
```

可通过 `--config configs/fast_ply.json` 使用冻结JSON配置，显式CLI参数覆盖配置值。

只写 `cleaned_gaussians.ply` 和同目录 `cleanup_report.json`；已有结果、输入同路径、报告覆盖输入均拒绝。输入仅支持GGGS vertex-only二进制PLY（小端和大端）；附加字段原样保留。混合element、ASCII和列表字段显式拒绝。异常时清理本次partial文件，绝不替换原模型。不传output会在原模型同目录生成新名称，推荐显式独立结果目录。

低内存路径由预算自动选择：容量足够则一次读取+一次写入；否则仅存32字节/点特征、空间索引和keep，第二次顺序读源并分块写最终模型，不使用mmap/swap/scratch。预算低于估计的紧凑工作集时明确失败。预算是保守容量估计，不是操作系统强制RSS隔离；报告包含真实峰值RSS。`--io-limit-mib 0`关闭限速，默认不推荐在共享磁盘上使用。

报告区分逻辑流量、`/proc/self/io`缓存相关物理计数、CPU时间、墙钟时间和主动等待。峰值是应用层尾随1秒吞吐；不是设备物理峰值或全服务器带宽承诺。最终fsync/内核回写不由用户态速率器硬限制。读写块最大256KiB。

仅在需要查看时，单独生成SuperSplat版：

```bash
../.venv/bin/python -m fast_ply.export_viewer \
  --input /absolute/path/to/new_result/cleaned_gaussians.ply \
  --output /absolute/path/to/viewer/cleaned_supersplat.ply --io-limit-mib 20
```

该命令采用与原生GGGS相同的有效σ/α烘焙公式，分块执行；SG lobes无标准等价表示，显式拒绝。默认清理始终保留SG和未知字段，不擅自丢弃。

独立的最终评价明确使用GPU1，须在需要时单独启动：

```bash
../.venv/bin/python -m fast_ply.evaluate \
  --scene experiments/Historical_TwoStage_Cleaner/scene_configs/KongQueZhuYu.json \
  --output /absolute/path/to/paired_evaluation --views 16 --max-side 960
```

评价复用仓库原生Renderer和相机映射，原始/C1/C2/真实R5/fast采用相同视角与分辨率；现有历史自由相机另列。消融在内存中实现，不保存每种完整PLY。该命令不训练、不归因，不改变清理参数。评价额外依赖现有torch/CUDA/GGGS环境。RAP骨架、暗前景和P3轮廓均为代理，不是植物逐叶语义真值。

资源复现实验在 `scripts/fast_resource_validation.py` 中显式只允许孔雀竹芋、长寿花和网纹草。运行时指定 `--result-root /absolute/new_experiment`，可用 `--plants` 选择子集。网纹草使用2.5 GiB预算，其余默认8 GiB；算法参数相同。不会启动19株。已有输出会报错，应选择新的实验目录复测。

```bash
../.venv/bin/python -m pytest -q tests/fast_ply
```

测试覆盖filter公式、大小端/未知字段字节保真、输入不变、真实读写计数、限速绕过回归、强制低内存与一次读取结果一致、暗细结构保护、异常输入/覆盖拒绝和独立viewer烘焙等价。
