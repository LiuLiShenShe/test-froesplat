# M2M3-Floor40 三场景 Pilot 执行说明

## 目标

在副本仓库 `/data/fj/F2DMAS/2d-gaussian-splatting-m2m3-floor40` 中，对孔雀竹芋、长寿花和草莓运行前景感知 2DGS + M2M3/Floor40 pilot。原始仓库 `/data/fj/F2DMAS/2d-gaussian-splatting-main` 不参与修改。

## 目录约定

- 脚本目录：`/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验脚本`
- 输出目录：`/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出`
- 训练输出：`实验输出/runs/<RUN_TAG>/<SCENE>/<METHOD>`
- 日志输出：`实验输出/logs/<RUN_TAG>_<SCENE>_<METHOD>.log`
- 渲染评测日志：`实验输出/logs/<RUN_TAG>_<SCENE>_<METHOD>_render_iter30000.log` 和 `实验输出/logs/<RUN_TAG>_<SCENE>_<METHOD>_metrics_iter30000.log`
- 整理后的 mask：`实验输出/prepared_masks/<SCENE>`
- 有 mask 的视角 gate：`实验输出/prepared_gate_lists/<SCENE>_mask_matched.txt`
- 整理后的 view weights：`实验输出/prepared_view_weights/<SCENE>_view_weights.csv`
- capacity/点云大小汇总：`实验输出/reports/capacity_and_pointcloud_summary.csv`
- 完整评测汇总：`实验输出/reports/full_evaluation_summary_<RUN_TAG>.csv` 和 `实验输出/reports/full_evaluation_summary_<RUN_TAG>.md`
- 评测耗时记录：`实验输出/reports/evaluation_runtime_<RUN_TAG>.json`
- 每个模型的 render/GT 图：`实验输出/runs/<RUN_TAG>/<SCENE>/<METHOD>/test/ours_30000/{renders,gt,vis}`
- 每个模型的指标 JSON：`实验输出/runs/<RUN_TAG>/<SCENE>/<METHOD>/results.json` 和 `per_view.json`

## 场景清单

场景配置保存在 `scene_manifest.csv`：

- `KongQueZhuYu`：孔雀竹芋，使用 final locked COLMAP source + `/data/fj/F2DMAS/03-SAM/KongQueZhuYu` mask + soft view weights；当前 210/210 帧可匹配。
- `ChangShouHua2`：长寿花，使用实验专用 repaired COLMAP source + `/data/fj/F2DMAS/03-SAM/ChangShouHua2` mask；原始 final locked source 中 `images/0084.jpg` 和 `images/0085.jpg` 不能被 PIL 识别，因此在 `实验输出/prepared_sources/ChangShouHua2_repaired_colmap` 中复制 `images` 并用同尺寸 `images_rgba` 帧重建这两张 JPG，原始数据不改动。当前该 mask 与 source 有 190/212 帧可匹配，脚本会自动生成 matched-view gate list，并把初始化从 `foreground_track` 切换为 `foreground_mask`。
- `CaoMei2`：草莓，使用 final locked COLMAP source + `/data/fj/F2DMAS/03-SAM/CaoMei2` mask + soft view weights；当前 203/210 帧可匹配。

## 执行顺序

先检查路径：

```bash
cd /data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验脚本
bash 00_check_scene_paths.sh
```

准备 mask 软链接和标准化 view-weight 文件：

```bash
python 01_prepare_three_scene_assets.py \
  --manifest scene_manifest.csv \
  --output-root /data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出 \
  --scenes KongQueZhuYu,ChangShouHua2,CaoMei2 \
  --allow-missing-masks \
  --force
```

先看命令，不启动训练：

```bash
bash 02_run_three_scene_pilot.sh --mode smoke --dry-run
```

启动 smoke pilot：

```bash
CUDA_VISIBLE_DEVICES=0 bash 02_run_three_scene_pilot.sh --mode smoke
```

正式 30000 iteration：

```bash
CUDA_VISIBLE_DEVICES=0 bash 02_run_three_scene_pilot.sh --mode full
```

只跑某个场景或某个方法：

```bash
bash 02_run_three_scene_pilot.sh \
  --mode smoke \
  --scenes KongQueZhuYu \
  --methods fg2dgs_m2m3_floor40
```

汇总 capacity report 和最终点云大小：

```bash
python 03_collect_capacity_reports.py \
  --output-root /data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出
```

渲染测试集并汇总 PSNR / SSIM / LPIPS / 训练耗时 / 渲染耗时 / 点云数量 / PLY 大小：

```bash
CUDA_VISIBLE_DEVICES=0 python 04_evaluate_three_scene_full.py \
  --run-tag 20260610_215125_sam03_full
```

如果已经完成渲染和指标计算，只想重生成 CSV/Markdown 汇总：

```bash
CUDA_VISIBLE_DEVICES=0 python 04_evaluate_three_scene_full.py \
  --run-tag 20260610_215125_sam03_full \
  --skip-render \
  --skip-metrics
```

## 本次 03-SAM full run 产物位置

当前已完成 run tag：`20260610_215125_sam03_full`。

- 完整训练目录：`/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/runs/20260610_215125_sam03_full/`
- 训练和评测日志目录：`/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/logs/`
- capacity/点云大小汇总：`/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/reports/capacity_and_pointcloud_summary.csv`
- 完整评测 CSV：`/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/reports/full_evaluation_summary_20260610_215125_sam03_full.csv`
- 完整评测 Markdown：`/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/reports/full_evaluation_summary_20260610_215125_sam03_full.md`
- 评测耗时 JSON：`/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/reports/evaluation_runtime_20260610_215125_sam03_full.json`
- 每个模型的渲染输出：`/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/runs/20260610_215125_sam03_full/<SCENE>/<METHOD>/test/ours_30000/`
- 每个模型的平均指标：`/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/runs/20260610_215125_sam03_full/<SCENE>/<METHOD>/results.json`
- 每个模型的逐视角指标：`/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/runs/20260610_215125_sam03_full/<SCENE>/<METHOD>/per_view.json`

本次 `03-SAM` 资产整理摘要：

- `KongQueZhuYu`：210/210 mask 可用，无坏 mask。
- `ChangShouHua2`：190/212 mask 可用；`mask_0106.png` 不可读，另有 22 个视角在 source/mask 对齐后被 gate 跳过。
- `CaoMei2`：203/210 mask 可用；`0070, 0072, 0087, 0090, 0132, 0134, 0138` 对应 mask 不可读，因此这些视角和对应 GT 已跳过。

坏 mask 与匹配统计见：`/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/reports/asset_preparation_summary.json`
抽样检查图见：`/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/reports/sam03_mask_audit_contact_sheet.jpg`

## 结果分析结论

本次 `03-SAM` full run 的核心结论是：M2M3/Floor40 在三类植物上的表现明显依赖场景和 mask 形态，当前配置不能概括为“稳定压缩点云”，更准确的表述是“对部分场景有质量-容量再分配效果”。其中长寿花收益最明确，孔雀竹芋和草莓基本没有压缩收益。

| scene | method | PSNR | SSIM | LPIPS | points | PLY MB | size change vs baseline | conclusion |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `CaoMei2` | baseline | 5.199 | 0.1101 | 0.7042 | 245315 | 59.9 | 0.0% | baseline |
| `CaoMei2` | m2m3 | 5.198 | 0.1101 | 0.7041 | 248852 | 60.7 | +1.4% | 指标几乎不变，点云略增 |
| `CaoMei2` | m2m3_floor40 | 5.198 | 0.1101 | 0.7044 | 250033 | 61.0 | +1.9% | 指标几乎不变，点云略增 |
| `ChangShouHua2` | baseline | 15.386 | 0.4941 | 0.4439 | 131778 | 32.2 | 0.0% | baseline |
| `ChangShouHua2` | m2m3 | 15.625 | 0.5316 | 0.4272 | 147534 | 36.0 | +12.0% | 质量最佳，但容量变大 |
| `ChangShouHua2` | m2m3_floor40 | 15.510 | 0.5023 | 0.4349 | 126941 | 31.0 | -3.7% | 最好的质量-容量折中 |
| `KongQueZhuYu` | baseline | 6.251 | 0.2678 | 0.5566 | 532641 | 130.0 | 0.0% | baseline |
| `KongQueZhuYu` | m2m3 | 6.251 | 0.2679 | 0.5570 | 534751 | 130.5 | +0.4% | 与 baseline 基本一致 |
| `KongQueZhuYu` | m2m3_floor40 | 6.251 | 0.2678 | 0.5569 | 535116 | 130.6 | +0.5% | 与 baseline 基本一致 |

分场景判断：

- `ChangShouHua2` 是本批最值得写进论文讨论的正例。`m2m3` 相比 baseline 提升 `+0.239 dB` PSNR、`+0.0375` SSIM，并将 LPIPS 降低 `0.0167`，说明 M2M3 的拓扑/容量选择确实改善了重建质量；但代价是点云从 `131778` 增到 `147534`，PLY 从 `32.2 MB` 增到 `36.0 MB`。`m2m3_floor40` 的质量提升小一些，但点云降到 `126941`，PLY 降到 `31.0 MB`，更适合作为“质量提升同时轻微压缩”的折中设置。
- `KongQueZhuYu` 的三个方法几乎重合。M2M3 只带来 `0.0002 dB` 级别 PSNR 变化，LPIPS 还略高，点云反而增加约 `0.4-0.5%`。这说明在该 mask 和当前 pruning schedule 下，容量控制没有形成有效压缩。
- `CaoMei2` 的质量指标也几乎不变，M2M3/Floor40 反而使点云和 PLY 增加 `1.4-1.9%`。和上一批错误 mask 的“明显压缩”现象相比，这批正确 `03-SAM` mask 更能说明压缩效果对 mask 覆盖范围非常敏感。
- 本次所有 M2M3/Floor40 运行中 `capacity_total_blocked_by_floor=0`，说明 floor 下限没有真正触发阻挡；Floor40 的差异更可能来自容量策略和训练 densification/pruning 轨迹的交互，而不是 floor 直接保护了大量 Gaussian。

建议写法：这批结果适合作为 pilot/消融现象，结论应强调“场景自适应容量重分配”和“长寿花上质量-容量折中有效”，暂时不要把当前配置表述为跨场景稳定压缩方案。下一步如果要强化论文结论，建议增加 late-stage pruning 或 hard capacity target，并补多 seed/多样本重复实验。

## 逐模型具体产物位置

本次 run 根目录：

`/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/runs/20260610_215125_sam03_full`

下表中的路径均相对于该 run 根目录。每个模型的最终点云、测试集渲染图、GT、可视化图、平均指标和逐视角指标都已生成；baseline 没有 `capacity_control/` 目录是正常现象。

| scene | method | final PLY | renders / gt / vis | metrics JSON | capacity / pruning |
|---|---|---|---|---|---|
| `KongQueZhuYu` | `fg2dgs_baseline` | `KongQueZhuYu/fg2dgs_baseline/point_cloud/iteration_30000/point_cloud.ply` | `KongQueZhuYu/fg2dgs_baseline/test/ours_30000/{renders,gt,vis}`，27 张 | `KongQueZhuYu/fg2dgs_baseline/{results.json,per_view.json}` | `pruning/` |
| `KongQueZhuYu` | `fg2dgs_m2m3` | `KongQueZhuYu/fg2dgs_m2m3/point_cloud/iteration_30000/point_cloud.ply` | `KongQueZhuYu/fg2dgs_m2m3/test/ours_30000/{renders,gt,vis}`，27 张 | `KongQueZhuYu/fg2dgs_m2m3/{results.json,per_view.json}` | `capacity_control/`, `pruning/` |
| `KongQueZhuYu` | `fg2dgs_m2m3_floor40` | `KongQueZhuYu/fg2dgs_m2m3_floor40/point_cloud/iteration_30000/point_cloud.ply` | `KongQueZhuYu/fg2dgs_m2m3_floor40/test/ours_30000/{renders,gt,vis}`，27 张 | `KongQueZhuYu/fg2dgs_m2m3_floor40/{results.json,per_view.json}` | `capacity_control/`, `pruning/` |
| `ChangShouHua2` | `fg2dgs_baseline` | `ChangShouHua2/fg2dgs_baseline/point_cloud/iteration_30000/point_cloud.ply` | `ChangShouHua2/fg2dgs_baseline/test/ours_30000/{renders,gt,vis}`，24 张 | `ChangShouHua2/fg2dgs_baseline/{results.json,per_view.json}` | `pruning/` |
| `ChangShouHua2` | `fg2dgs_m2m3` | `ChangShouHua2/fg2dgs_m2m3/point_cloud/iteration_30000/point_cloud.ply` | `ChangShouHua2/fg2dgs_m2m3/test/ours_30000/{renders,gt,vis}`，24 张 | `ChangShouHua2/fg2dgs_m2m3/{results.json,per_view.json}` | `capacity_control/`, `pruning/` |
| `ChangShouHua2` | `fg2dgs_m2m3_floor40` | `ChangShouHua2/fg2dgs_m2m3_floor40/point_cloud/iteration_30000/point_cloud.ply` | `ChangShouHua2/fg2dgs_m2m3_floor40/test/ours_30000/{renders,gt,vis}`，24 张 | `ChangShouHua2/fg2dgs_m2m3_floor40/{results.json,per_view.json}` | `capacity_control/`, `pruning/` |
| `CaoMei2` | `fg2dgs_baseline` | `CaoMei2/fg2dgs_baseline/point_cloud/iteration_30000/point_cloud.ply` | `CaoMei2/fg2dgs_baseline/test/ours_30000/{renders,gt,vis}`，26 张 | `CaoMei2/fg2dgs_baseline/{results.json,per_view.json}` | `pruning/` |
| `CaoMei2` | `fg2dgs_m2m3` | `CaoMei2/fg2dgs_m2m3/point_cloud/iteration_30000/point_cloud.ply` | `CaoMei2/fg2dgs_m2m3/test/ours_30000/{renders,gt,vis}`，26 张 | `CaoMei2/fg2dgs_m2m3/{results.json,per_view.json}` | `capacity_control/`, `pruning/` |
| `CaoMei2` | `fg2dgs_m2m3_floor40` | `CaoMei2/fg2dgs_m2m3_floor40/point_cloud/iteration_30000/point_cloud.ply` | `CaoMei2/fg2dgs_m2m3_floor40/test/ours_30000/{renders,gt,vis}`，26 张 | `CaoMei2/fg2dgs_m2m3_floor40/{results.json,per_view.json}` | `capacity_control/`, `pruning/` |

## 方法标签

- `fg2dgs_baseline`：前景感知 2DGS，`capacity_control_mode=none`。
- `fg2dgs_m2m3`：启用 M2M3 topology score，但不启用 floor。
- `fg2dgs_m2m3_floor40`：启用 M2M3 topology score 和 `capacity_floor_ratio=0.4`。

## 当前注意点

- 训练脚本默认先整理资产；如果已整理且不想刷新，可以加 `--no-prepare`。
- `smoke` 模式为 19000 iteration，目的是越过默认 18000 pruning 起点并触发一次 pruning。
- `full` 模式为 30000 iteration，保存 7000 和 30000。
- 当前三套场景统一使用 `03-SAM` 目录下的 mask；如果后续替换 mask，只需要改 `scene_manifest.csv` 的 `raw_mask_dir`。
- 脚本会自动跳过不可读、近乎全空白、或近乎全满屏的坏 mask，并同步跳过对应视角和 GT。
- 长寿花当前自动只保留有有效 mask 的 190 个视角。
- 长寿花 source_dir 当前指向实验输出中的 repaired source，只修复不可读的 `0084.jpg` 和 `0085.jpg`；原始 final locked COLMAP 数据保持不变。

