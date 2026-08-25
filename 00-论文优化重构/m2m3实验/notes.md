# Notes: Current Foreground-Aware 2DGS State and M2M3 Integration Points

## Current Repository State

### Original repository

`/data/fj/F2DMAS/2d-gaussian-splatting-main`

该仓库已经不是纯原始 2DGS，而是包含前景植物感知改动：

- foreground / alpha mask 输入
- foreground point-cloud initialization
- foreground RGB supervision
- alpha mask loss
- background opacity loss
- view-quality soft weighting
- mask / topology / opacity / brightness pruning

### Independent copy

`/data/fj/F2DMAS/2d-gaussian-splatting-m2m3-floor40`

该副本已由原始仓库完整复制而来，后续 M2M3-Floor40 代码只应修改该副本。

## Existing Switches Observed

### Loading / dataset side

Located in:

`/data/fj/F2DMAS/2d-gaussian-splatting-main/arguments/__init__.py`

Existing relevant parameters:

- `--mask_mode`
- `--mask_dir`
- `--mask_pattern`
- `--mask_threshold`
- `--init_pcd_mode`
- `--init_pcd_min_observations`
- `--init_pcd_foreground_threshold`
- `--init_pcd_dilate_mask_px`
- `--init_pcd_max_cameras`
- `--init_pcd_chunk_size`

### Optimization side

Existing relevant parameters:

- `--use_mask_loss`
- `--use_bg_opacity_loss`
- `--use_foreground_rgb_loss`
- `--lambda_mask`
- `--lambda_bg`
- `--lambda_fg_rgb`
- `--foreground_bg_rgb_weight`
- `--foreground_rgb_crop_padding`
- `--mask_loss_type`
- `--mask_ignore_boundary_px`
- `--mask_loss_start_iter`
- `--mask_loss_warmup_iters`
- `--view_weight_mode`
- `--view_weight_list`
- `--view_weight_min`
- `--view_weight_max`
- `--view_weight_default`
- `--pruning_mode`
- `--pruning_start_iter`
- `--pruning_interval`
- `--pruning_opacity_threshold`
- `--pruning_brightness_threshold`
- `--pruning_mask_threshold`
- `--pruning_mask_max_views`
- `--pruning_max_remove_ratio`
- `--pruning_mask_score_weight`
- `--save_pruning_report`

## Existing Hook Points

### Training loop

Located in:

`/data/fj/F2DMAS/2d-gaussian-splatting-main/train.py`

Important functions:

- `validate_plant_aware_args()`
- `read_view_weights()`
- `rgb_reconstruction_loss()`
- `mask_loss_terms()`
- `gaussian_brightness()`
- `mask_consistency_scores()`
- `maybe_prune_gaussians()`
- `training()`

### Standard 2DGS densification / pruning

Located in:

`/data/fj/F2DMAS/2d-gaussian-splatting-main/scene/gaussian_model.py`

Important methods:

- `densify_and_clone()`
- `densify_and_split()`
- `densify_and_prune()`
- `prune_points()`
- `add_densification_stats()`

## Design Implication

M2M3-Floor40 can be added without disturbing the existing foreground-aware code if implemented as a new optional capacity controller:

- Default mode: off
- Existing foreground loss and pruning behaviour remains unchanged
- New mode only activates when `--capacity_control_mode` is set
- New reports are saved under the model output path, not the source repository root

## Suggested First Implementation Location

The least disruptive first implementation is:

1. Keep standard 2DGS densification logic unchanged.
2. Keep foreground-aware loss logic unchanged.
3. Extend `maybe_prune_gaussians()` or wrap its candidate deletion mask with an M2M3/Floor controller.
4. Enforce floor before calling `gaussians.prune_points(prune_mask)`.
5. Save diagnostic JSON files for each capacity-control iteration.

This path tests whether M2M3-Floor40 improves foreground-object 2DGS without rewriting the core Gaussian densification mechanics.

## Implemented in the M2M3 Copy

Implemented repository:

`/data/fj/F2DMAS/2d-gaussian-splatting-m2m3-floor40`

Modified files:

- `arguments/__init__.py`
- `train.py`

Implemented switches:

- `--capacity_control_mode none|m2m3|m2m3_floor`
- `--capacity_floor_ratio`
- `--capacity_floor_reference initial|max_seen|current`
- `--capacity_floor_count`
- `--capacity_floor_start_iter`
- `--capacity_floor_end_iter`
- `--m2m3_score_mode legacy|topology`
- `--m2m3_region_mode foreground|all`
- `--m2m3_mask_weight`
- `--m2m3_opacity_weight`
- `--m2m3_brightness_weight`
- `--m2m3_scale_weight`
- `--m2m3_view_weight`
- `--m2m3_max_remove_ratio`
- `--save_capacity_report`
- `--capacity_report_interval`

Implementation notes:

- Default `capacity_control_mode` is `none`, so copied baseline behaviour remains opt-in.
- If `capacity_control_mode` is enabled while `pruning_mode` remains `none`, training raises a clear error.
- M2M3 currently wraps the existing `maybe_prune_gaussians()` candidate-removal path.
- Standard densification in `gaussian_model.py` is unchanged.
- `m2m3_floor` enforces a global capacity floor before `gaussians.prune_points()` is called.
- Capacity reports are written to `<model_path>/capacity_control/`.
- Syntax check passed with `python -m py_compile train.py arguments/__init__.py`.

## Potential Later Implementation

If the first version works, a second version can move deeper:

- Add M2 diagnostics inside `densify_and_prune()`.
- Modulate `densify_grad_threshold` or `opacity_cull` by M2/M3 state.
- Apply region-aware capacity floors instead of a single global floor.

This should be treated as a second-stage extension because it touches more of the original 2DGS optimization loop.

## Three-Scene Pilot Script Notes

- Script directory: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验脚本`
- Output directory: `/data/fj/F2DMAS/00-论文优化重构/m2m3实验/实验输出`
- `KongQueZhuYu` and `CaoMei2` both have 210/210 mask matches with RAP-FSAM3 A5c downstream masks.
- `ChangShouHua2` has 191/212 mask matches with `03-SAM/ChangShouHua2`; the runner handles this by using `prepared_gate_lists/ChangShouHua2_mask_matched.txt` and `init_pcd_mode=foreground_mask`.
- Soft view weights are available for `KongQueZhuYu` and `CaoMei2`; `CaoMei2` has 203/210 matched weight entries, so the train script will use the configured default for missing entries.
- The generated runner defaults to `smoke` mode and requires `--mode full` for 30000-iteration runs.
