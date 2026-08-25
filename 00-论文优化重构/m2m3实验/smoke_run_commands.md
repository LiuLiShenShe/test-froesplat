# Smoke Run Commands: Foreground-Aware 2DGS + M2M3-Floor40

## Purpose

在正式 3-scene pilot 前，先用一个已经能跑通的植物场景做短训练 smoke run，确认：

- 副本仓库能正常启动；
- `capacity_control_mode=none` 不破坏原前景感知 2DGS 路径；
- `capacity_control_mode=m2m3_floor` 能创建 capacity report；
- floor 逻辑不会在第一轮 pruning 中产生明显异常。

## Repository

```bash
REPO=/data/fj/F2DMAS/2d-gaussian-splatting-m2m3-floor40
EXP=/data/fj/F2DMAS/00-论文优化重构/m2m3实验
```

## Fill These Paths First

请在运行前替换为真实路径：

```bash
SCENE=KongQueZhuYu
SOURCE=/path/to/colmap_scene/${SCENE}
MASK_DIR=/path/to/foreground_masks/${SCENE}
VIEW_WEIGHT_LIST=/path/to/view_weights/${SCENE}.csv
```

如果暂时不使用 view weights，可以删除 `--view_weight_mode` 和 `--view_weight_list` 两行。

## Smoke 1: Baseline Compatibility

目标：确认副本仓库在 `capacity_control_mode none` 下仍能启动并保存输出。

建议短跑到第一次 pruning 后一点点，例如 `--iterations 19000`，因为默认 M2M3/Floor 的关键逻辑在后期 pruning 阶段才触发。

```bash
cd ${REPO}
python train.py \
  -s ${SOURCE} \
  -m ${EXP}/runs/${SCENE}/smoke_fa2dgs_none \
  --iterations 19000 \
  --test_iterations 19000 \
  --save_iterations 19000 \
  --mask_mode alpha \
  --mask_dir ${MASK_DIR} \
  --init_pcd_mode foreground_track \
  --init_pcd_min_observations 3 \
  --init_pcd_foreground_threshold 0.9 \
  --use_foreground_rgb_loss \
  --use_mask_loss \
  --use_bg_opacity_loss \
  --lambda_mask 0.08 \
  --lambda_bg 0.02 \
  --mask_loss_type l1_dice \
  --mask_ignore_boundary_px 2 \
  --mask_loss_start_iter 500 \
  --mask_loss_warmup_iters 1500 \
  --view_weight_mode rgb_only \
  --view_weight_list ${VIEW_WEIGHT_LIST} \
  --pruning_mode mask \
  --pruning_start_iter 18000 \
  --pruning_interval 3000 \
  --pruning_max_remove_ratio 0.03 \
  --capacity_control_mode none
```

Expected:

```text
No capacity_control/ directory is required.
Training should follow the current foreground-aware 2DGS path.
```

## Smoke 2: M2M3-Floor40 Activation

目标：确认 M2M3-Floor40 能触发并写出报告。

```bash
cd ${REPO}
python train.py \
  -s ${SOURCE} \
  -m ${EXP}/runs/${SCENE}/smoke_fa2dgs_m2m3_floor40 \
  --iterations 19000 \
  --test_iterations 19000 \
  --save_iterations 19000 \
  --mask_mode alpha \
  --mask_dir ${MASK_DIR} \
  --init_pcd_mode foreground_track \
  --init_pcd_min_observations 3 \
  --init_pcd_foreground_threshold 0.9 \
  --use_foreground_rgb_loss \
  --use_mask_loss \
  --use_bg_opacity_loss \
  --lambda_mask 0.08 \
  --lambda_bg 0.02 \
  --mask_loss_type l1_dice \
  --mask_ignore_boundary_px 2 \
  --mask_loss_start_iter 500 \
  --mask_loss_warmup_iters 1500 \
  --view_weight_mode rgb_only \
  --view_weight_list ${VIEW_WEIGHT_LIST} \
  --pruning_mode mask \
  --pruning_start_iter 18000 \
  --pruning_interval 3000 \
  --capacity_control_mode m2m3_floor \
  --capacity_floor_ratio 0.4 \
  --capacity_floor_reference max_seen \
  --capacity_floor_start_iter 18000 \
  --m2m3_score_mode topology \
  --m2m3_region_mode foreground \
  --m2m3_max_remove_ratio 0.03 \
  --save_capacity_report \
  --capacity_report_interval 1000
```

Expected files:

```text
${EXP}/runs/${SCENE}/smoke_fa2dgs_m2m3_floor40/capacity_control/capacity_iter_18000.json
${EXP}/runs/${SCENE}/smoke_fa2dgs_m2m3_floor40/capacity_control/capacity_summary.json
```

## Quick Inspection

```bash
python - <<PY
import json
from pathlib import Path

path = Path("${EXP}") / "runs" / "${SCENE}" / "smoke_fa2dgs_m2m3_floor40" / "capacity_control" / "capacity_summary.json"
with path.open("r", encoding="utf-8") as f:
    data = json.load(f)
print(json.dumps(data, indent=2))
PY
```

Check:

- `capacity_control_mode` should be `m2m3_floor`.
- `max_seen_count` should be at least `initial_count`.
- `final_count` should not fall below the resolved floor.
- `rounds` should be greater than 0 after the first pruning iteration.
