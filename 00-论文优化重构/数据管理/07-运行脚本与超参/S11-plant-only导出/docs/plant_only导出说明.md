# Plant-only Gaussian 导出说明

## 定位

`plant-only` 是 E7/E8 之后的可选后处理分支，不重新训练模型。它读取训练输出中的 `point_cloud.ply`、`cameras.json` 和同一样本的 SAM/FSAM mask，通过多视角 mask 投票筛掉背景 Gaussian，并保留原 PLY 中全部 Gaussian 属性。

因此它可以直接接在当前 `KongQueZhuYu/E7_mask_pruning` 上继续跑。

口径说明：当前目标是 **mask foreground object**，不是 leaf-only，也不是去花盆后的纯叶片植株。只要对象被 mask 标为前景，就属于当前分支保留对象。

## KongQueZhuYu E7 默认命令

```bash
cd /data/fj/F2DMAS
python3 00-论文优化重构/数据管理/07-运行脚本与超参/S11-plant-only导出/scripts/export_plant_only_gaussians.py \
  --config 00-论文优化重构/数据管理/07-运行脚本与超参/S11-plant-only导出/configs/kongquezhu_e7_plant_only.json
```

## 当前输出

```text
/data/fj/F2DMAS/00-论文优化重构/数据管理/06-实验输出/KongQueZhuYu/E7_mask_pruning/plant_only/
├── plant_only_gaussians.ply
├── plant_only_report.json
└── plant_only_scores.npz
```

## 参数含义

- `min_observations`：一个 Gaussian 至少要被多少个 mask 相机有效投影到画面内。
- `foreground_ratio_threshold`：有效投影中落在植物 mask 内的比例阈值。
- `dilate_mask_px`：导出前对 mask 做轻微膨胀，避免薄叶边缘被过度切掉。
- `projection_mode=auto`：自动判断相机前方是正 Z 还是负 Z。

默认配置偏保守，目标是先得到可检查的 foreground-object PLY；如果可视化中仍有明显背景，可提高 `foreground_ratio_threshold` 或降低 `dilate_mask_px`。

## 与 E7/E8 的关系

- E7：训练阶段仍是 full-scene Gaussian 模型，M2/M3/M4 通过 mask loss 和 pruning 压制背景，但不会保证最终 PLY 只剩植物。
- plant-only / foreground-object：E7 训练完成后的导出分支，把 E7 的全场景 Gaussian 进一步按 mask 投票过滤为 foreground-object PLY。
- E8：可继续训练 full Plant-aware；若需要植物-only 结果，同样在 E8 输出目录上跑本脚本，把配置中的 `input_ply`、`cameras_json` 和 `output_dir` 改到 E8 即可。

## 评测记录

详见：

```text
数据管理/07-运行脚本与超参/S11-plant-only导出/docs/foreground_object导出与评测记录.md
数据管理/06-实验输出/KongQueZhuYu/E7_mask_pruning/plant_only/diagnostics/
```
