# Implementation Log: Foreground-Aware 2DGS + M2M3-Floor40

## 2026-06-10

### Repository

Implemented only in the independent copy:

```text
/data/fj/F2DMAS/2d-gaussian-splatting-m2m3-floor40
```

Protected original repository was not modified:

```text
/data/fj/F2DMAS/2d-gaussian-splatting-main
```

### Modified Files

```text
2d-gaussian-splatting-m2m3-floor40/arguments/__init__.py
2d-gaussian-splatting-m2m3-floor40/train.py
```

### Added Parameters

Added to `OptimizationParams`:

```text
capacity_control_mode = "none"
capacity_floor_ratio = 0.4
capacity_floor_reference = "max_seen"
capacity_floor_count = 0
capacity_floor_start_iter = 0
capacity_floor_end_iter = 0
m2m3_score_mode = "topology"
m2m3_region_mode = "foreground"
m2m3_mask_weight = 3.0
m2m3_opacity_weight = 1.0
m2m3_brightness_weight = 1.0
m2m3_scale_weight = 0.5
m2m3_view_weight = 0.5
m2m3_max_remove_ratio = 0.03
save_capacity_report = False
capacity_report_interval = 1000
```

### Added Validation

`validate_plant_aware_args()` now checks:

- `capacity_control_mode` is one of `none`, `m2m3`, `m2m3_floor`.
- M2M3 capacity control cannot be enabled when `pruning_mode` is `none`.
- `capacity_floor_reference` is one of `initial`, `max_seen`, `current`.
- `capacity_floor_ratio` is in `[0, 1]`.
- `m2m3_score_mode` is one of `legacy`, `topology`.
- `m2m3_region_mode` is one of `foreground`, `all`.

### Added Controller Functions

Added to `train.py`:

- `capacity_control_enabled()`
- `init_capacity_state()`
- `update_capacity_state()`
- `normalized_positive()`
- `m2m3_prune_scores()`
- `capacity_floor_is_active()`
- `resolve_capacity_floor_count()`
- `select_top_prune_candidates()`
- `save_capacity_report()`
- `save_capacity_summary()`

### Integration Point

The first implementation is deliberately non-invasive:

- `scene/gaussian_model.py` was not changed.
- Standard densification remains unchanged.
- Existing foreground-aware loss remains unchanged.
- Existing plant-aware pruning still creates the candidate prune mask.
- M2M3/Floor40 intervenes only before the final `gaussians.prune_points(prune_mask)` call.

### Behaviour by Mode

#### `--capacity_control_mode none`

Uses the previous pruning logic and `pruning_max_remove_ratio`.

#### `--capacity_control_mode m2m3`

Uses M2M3 topology scores and `m2m3_max_remove_ratio` to select candidate removals from the existing pruning candidate mask.

#### `--capacity_control_mode m2m3_floor`

Uses M2M3 selection and then enforces:

```text
allowed_remove = max(0, current_count - floor_count)
```

where `floor_count` is computed from:

```text
capacity_floor_reference * capacity_floor_ratio
```

The default reference is `max_seen`, meaning the controller protects a fraction of the largest Gaussian capacity observed during training.

### Reports

When `--save_capacity_report` is enabled, reports are written to:

```text
<model_path>/capacity_control/
```

Expected files:

```text
capacity_iter_<iteration>.json
capacity_summary.json
```

Per-iteration reports include:

- requested candidate removals
- removals after M2M3 budget
- removals blocked by floor
- floor active state
- floor count
- capacity state summary
- removed-by-opacity / brightness / mask counts

`capacity_report_interval` is active. A per-iteration capacity report is written only when:

- the current iteration is divisible by `capacity_report_interval`;
- or the current iteration is the final training iteration;
- or `capacity_report_interval <= 0`.

The end-of-training `capacity_summary.json` is always written when capacity control and `--save_capacity_report` are enabled.

### Verification

Command:

```bash
cd /data/fj/F2DMAS/2d-gaussian-splatting-m2m3-floor40
python -m py_compile train.py arguments/__init__.py
```

Result:

```text
passed
```

Temporary top-level `__pycache__` directories created by the syntax check were removed.

### Added Smoke-Run Template

Created:

```text
/data/fj/F2DMAS/00-论文优化重构/m2m3实验/smoke_run_commands.md
```

This file contains short-run command templates for:

- baseline compatibility with `--capacity_control_mode none`;
- M2M3-Floor40 activation with capacity reports.

### Not Yet Done

- No real training run has been launched yet.
- No pilot metrics have been generated yet.
- No comparison tables have been populated yet.

## 2026-06-10: Three-Scene Pilot Scripts

### Added Script Files

Created under:

```text
/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验脚本
```

Files:

```text
scene_manifest.csv
scene_config.sh
00_check_scene_paths.sh
01_prepare_three_scene_assets.py
02_run_three_scene_pilot.sh
03_collect_capacity_reports.py
README_三场景pilot执行说明.md
```

### Output Layout

All generated assets and future training outputs are under:

```text
/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出
```

Subdirectories:

```text
prepared_masks/
prepared_gate_lists/
prepared_view_weights/
runs/
logs/
reports/
```

### Scene Handling

Pilot scenes:

- `KongQueZhuYu`: 210/210 masks matched; RAP-FSAM3 A5c masks; soft view weights matched 210/210.
- `ChangShouHua2`: 191/212 masks matched using `03-SAM/ChangShouHua2`; no soft view weights found. The runner uses a matched-view gate list and switches init from `foreground_track` to `foreground_mask` for this scene.
- `CaoMei2`: 210/210 masks matched; RAP-FSAM3 A5c masks; soft view weights matched 203/210, with default weight used for missing entries during training.

### Verification

Commands passed:

```bash
bash -n scene_config.sh
bash -n 00_check_scene_paths.sh
bash -n 02_run_three_scene_pilot.sh
python -m py_compile 01_prepare_three_scene_assets.py 03_collect_capacity_reports.py
bash 00_check_scene_paths.sh
bash 02_run_three_scene_pilot.sh --mode smoke --dry-run --scenes ChangShouHua2 --methods fg2dgs_m2m3_floor40
python 03_collect_capacity_reports.py --output-root /data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出
```

No training run was launched. Dry-run output directories were not created after the final runner fix.

### Minor Fix

`03_collect_capacity_reports.py` originally treated the empty `--output-csv` default as `Path("")`, which resolved to the current directory. This was fixed by using `None` as the default and writing to:

```text
/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出/reports/capacity_and_pointcloud_summary.csv
```
