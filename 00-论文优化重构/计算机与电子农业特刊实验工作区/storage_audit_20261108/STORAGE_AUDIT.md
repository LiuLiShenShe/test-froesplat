# ForeSplat 项目存储审计与清理建议

统计时间：2026-11-08T12:40:29（服务器本地时间）。范围 `/data/fj/F2DMAS`，只读检查；没有删除、移动、压缩历史文件，没有启动模型或训练。所有诊断脚本由uv调用，未使用conda或GPU。

项目实际占用 **704.86 GiB**，共642,960个文件、58,152个目录。实际分配块按`st_blocks×512`统计、同一inode在同一目录树内只计一次，不跟随符号链接。扫描无权限/IO错误。

`df`显示数据盘约91%已用、约324 GiB剩余；系统盘约91%已用、约164 GiB剩余。实时字节快照见 [filesystem_usage.txt](filesystem_usage.txt)。清理 `/tmp` 或 `/home/test/.cache` 增加的是系统盘空间，不是数据盘空间。

## 优先清理候选

| 内容 | 可回收空间 | 所在盘 | 判断与前提 |
|---|---:|---|---|
| `__pycache__`、`.pytest_cache`生成缓存 | 2.10 GiB | 数据盘 | 低风险；只清生成缓存，不删源码、虚拟环境或实验目录。涉及13,832个缓存目录。 |
| 根目录 `hf_cache` 的Fast3R下载缓存 | 2.41 GiB | 数据盘 | 当前Fast3R/SfM运行器实际加载项目 `weights/fast3r`，不是此缓存；保留主权重。缓存未来可重新下载，不宣称两份权重已重新做完整SHA比对。 |
| `/tmp/foresplat_KongQueZhuYu/Fast3R_AMB3R_SfM_full208_original_dtype_20261107` | 48.72 GiB | 系统盘 | 低风险重复临时副本。正式结果已经持久化；当前5145源文件均存在于正式目录，1936个NPZ的entry CRC与压缩/未压缩尺寸全部相同，数据文件没有缺失/尺寸差异；3份元数据因finalize改变，属已知差异。 |
| 全局uv缓存 `/home/test/.cache/uv` | 完整清空上限 37.52 GiB | 系统盘 | 建议优先官方 `uv cache prune`；本次没有执行，实际prune收益未知。全清还会丢弃其他项目的下载/构建缓存，以后可能需下载和编译。 |

上述数据盘两项低风险候选合计 **4.51 GiB**。临时SfM副本和全uv缓存的系统盘合计上限 **86.23 GiB**；不要把不同盘收益合并为数据盘增加量，也不要将完整uv缓存空间等同于prune收益。

uv用法通过Context7读取 [官方缓存文档](https://docs.astral.sh/uv/concepts/cache/)；本机uv为0.10.12，最新文档有新版本缓存结构，本次只给出已有clean/prune操作，不推断新格式已经存在。使用uv自己的缓存命令，不手工拆其内部结构；当前项目的四套正式环境保留。

## 最大的有条件清理项：旧COLMAP中间图

目录：`11-COLMAP-fuse/run_20260302_153856/*/dense/stereo/`。

可清的是20个植物目录下的 **depth_maps / normal_maps / consistency_graphs**，共60目录、**184.23 GiB**。保留`stereo`的配置文本、原图、相机、数据库、日志、summary以及所有最终`fuse.ply`。

已读取20份`fuse.ply`的真实PLY头，全部存在且vertex>0；历史summary20行均为success。最终融合点云共0.82 GiB。完整核验见 [colmap_intermediate_validation.json](colmap_intermediate_validation.json)。这只验证最终文件存在且非空，不能证明每株重建质量优秀。

如果只需要保留已有融合结果，这批中间图可清理。若还要直接重做旧MVS融合、查看逐视角MVS深度/法向，先移到其他磁盘归档；删后必须重新计算中间图才能重融合。它们是旧COLMAP中间结果，**不是此次VGGT/Ω/Fast3R/AMB3R几何原始预测**。

此项加前述两项数据盘低风险缓存合计约 **188.75 GiB**，这是有条件可回收值，没有执行。

## 其余可归档或停用后清理的内容

- Fast3R smoke/ramp16/ramp64与AMB3R-SfM smoke32，整个目录上限 **4.04 GiB**。完整208帧结果已成功，保留Smoke日志、报告、配置和状态，才能考虑清理早期大型点云/预测；不是把现有208成果替换掉。
- SEEM、YOLO、SuGaR、旧SAM的四套`venv`合计 **29.46 GiB**。只有确认停用、保留版本冻结和自定义扩展构建来源后才能删除。它们是可重建环境，不是已确认重复数据；不删除当前Ω/标准VGGT/Fast3R/AMB3R环境，也不删用于读取E6与旧正式2DGS的两套2DGS环境。
- E5的最终PLY在`gaussian`和`checkpoints/final_gaussians`是同一inode的两条硬链接；NPZ也是如此。**只删一个路径回收0字节**。另外`iteration_30000`的PLY是独立inode，可能进一步去重约1.21 GiB，但本次没有完整SHA核验，不列为直接删除项；真实Checkpoint路径应保留。
- `/tmp/foresplat_root_cause_geometry_input/standard_model_inputs.npz`约0.21 GiB。有些旧审计绘图脚本实际读取它，因此不当作无依赖的垃圾；若清理，需要从保存数据重新生成。新的目标控制评价已经可以不用此缓存。

## 应保留的证据与原始数据

- `原始视频`、`01-FFmepg`、`02-FFT`、`03-SAM`、所有GT和人工标注。
- VGGT-Ω的19株原始预测、COLMAP Dataset及已完成E1–E6/无融合实验的最终Gaussian、相机、配置、日志、指标。
- 两套正式SfM的 `attempts/*full208*/local_predictions`合计 **86.11 GiB**。它们保存跨Cluster局部原始推理，是已交付的审计证据，**当前不建议直接删**；`attempts`不是失败临时目录，两套canonical符号链接都指向这里。
- 7份已验收Geometry Bundle及新 `root_cause_audit` 中的受控推理、原始重渲染、相机与定量证据。图/点云虽可派生，保留原始预测不等于必须重新跑网络才能复查。
- 当前模型主权重、四套新前端环境和两套2DGS环境。
- 根 `.git` **48.78 GiB**。其中约19.24 GiB loose objects、29.50 GiB pack以及约32.8 MiB Git报告garbage。不能删除objects/pack当缓存；可另做保留提交历史的受控打包优化，收益未测。涉及约48.8 GiB历史库，不能把它全部计为可回收空间。

硬链接还使`03-SAM`多处被正式Dataset引用：该目录显示6.65 GiB，但单独删除只回收约0.19 GiB，并会破坏原始路径。扫描包含8041个硬链接inode，报告中目录占用和删除收益刻意分开；父目录/子目录、跨目录硬链接行不应直接相加。

## 可复核清单与来源

- [目录实际占用与可回收估算](directories.csv)
- [逐项候选清单](cleanup_candidates.csv)
- [16 MiB以上文件清单](large_files.csv)
- [原始统计与扫描范围](inventory_summary.json)
- [临时与持久化Bundle当前核验](temporary_bundle_duplicate_verification.json)
- [uv缓存独立统计](external_uv_cache/inventory_summary.json)
- [Git对象统计](git_count_objects.txt)
- [来源记录](sources.json)

博查检索实际返回403（额度不足），没有搜索结果可引用；项目空间数字均来自本机文件系统，非训练数据推测。uv文档来自实际Context7检索的官方文档。没有执行任何清理命令。
