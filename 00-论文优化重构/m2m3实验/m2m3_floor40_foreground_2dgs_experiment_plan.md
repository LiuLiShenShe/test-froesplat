# M2M3-Floor40 on Foreground-Aware Standard 2DGS: Detailed Experiment Plan

## 1. Objective

This experiment tests whether the M2M3-Floor40 capacity-control algorithm can improve the current foreground-aware standard 2DGS pipeline for potted-plant reconstruction.

The current 2DGS code already reconstructs foreground plant objects using masks, foreground RGB supervision, alpha/background opacity losses, view weighting, foreground point-cloud initialization and mask-guided pruning. Therefore, this experiment should not be framed as "adding foreground awareness" again. The new question is:

> Given a foreground-aware 2DGS reconstruction target, can M2M3-Floor40 reduce redundant or harmful Gaussian primitives while preserving thin plant structures and improving the quality-capacity-mesh trade-off?

## 2. Scope Boundary

### In scope

- Standard multi-view / approximately 200-view 2DGS plant reconstruction.
- Existing foreground-aware 2DGS as the baseline.
- M2M3-current and M2M3-Floor variants implemented as optional switches.
- Plant-object quality, capacity, leakage and mesh efficiency evaluation.
- Pilot experiments on a small set of scenes before full expansion.

### Out of scope for the first stage

- Replacing the separate Sparse2DGS paper.
- Rewriting the entire 2DGS densification mechanism in the first implementation.
- Claiming new DTU/LLFF/Mip-NeRF360 general compression state of the art before external validation.
- Removing or altering existing foreground-aware loss behaviour.
- Modifying `/data/fj/F2DMAS/2d-gaussian-splatting-main`.

## 3. Repository and Output Isolation

### Original protected repository

```text
/data/fj/F2DMAS/2d-gaussian-splatting-main
```

This repository is treated as the protected current ForeSplat / foreground-aware 2DGS codebase. Do not modify it for M2M3-Floor40.

### M2M3 working repository

```text
/data/fj/F2DMAS/2d-gaussian-splatting-m2m3-floor40
```

All M2M3-Floor40 code modifications should be made here.

### Planning and experiment records

```text
/data/fj/F2DMAS/00-论文优化重构/m2m3实验
```

Recommended subdirectories:

```text
m2m3实验/
  task_plan.md
  notes.md
  m2m3_floor40_foreground_2dgs_experiment_plan.md
  implementation_log.md
  runs/
  reports/
  figures/
  tables/
```

## 4. Current Baseline Definition

The current baseline is not vanilla full-scene 2DGS. It is:

```text
Foreground-aware standard 2DGS
= standard 2DGS
+ foreground point-cloud initialization
+ foreground RGB supervision
+ alpha mask loss
+ background opacity suppression
+ optional view-quality weighting
+ optional mask/topology pruning
```

The M2M3 experiment should compare against this current foreground-aware baseline.

## 5. Proposed New Switches

Add M2M3 capacity control as disabled-by-default optimization parameters.

### Core switches

```text
--capacity_control_mode none
--capacity_control_mode m2m3
--capacity_control_mode m2m3_floor
```

Default:

```text
capacity_control_mode = "none"
```

### Floor parameters

```text
--capacity_floor_ratio 0.4
--capacity_floor_reference max_seen
--capacity_floor_start_iter 15000
--capacity_floor_end_iter 30000
```

Recommended defaults:

```text
capacity_floor_ratio = 0.4
capacity_floor_reference = "max_seen"
capacity_floor_start_iter = pruning_start_iter
capacity_floor_end_iter = iterations
```

Rationale:

- `max_seen` uses the largest Gaussian count observed after densification begins as the floor reference.
- `floor40` therefore means "do not prune below 40% of the observed capacity envelope".
- This is safer than using the initial COLMAP sparse point count, which may be too small for standard 2DGS.

### M2/M3 scoring parameters

```text
--m2m3_score_mode topology
--m2m3_region_mode foreground
--m2m3_mask_weight 3.0
--m2m3_opacity_weight 1.0
--m2m3_brightness_weight 1.0
--m2m3_scale_weight 0.5
--m2m3_view_weight 0.5
--m2m3_max_remove_ratio 0.03
```

These should be treated as engineering parameters, not all as paper-level hyperparameters. The paper can report the final fixed setting.

### Logging parameters

```text
--save_capacity_report
--capacity_report_interval 1000
```

Report files should be saved to:

```text
<model_path>/capacity_control/
```

Example report names:

```text
capacity_iter_18000.json
capacity_iter_21000.json
capacity_summary.json
```

## 6. Implementation Strategy

### Stage 1: Non-invasive capacity controller

This is the recommended first implementation.

Keep these unchanged:

- foreground-aware RGB loss
- mask loss
- background opacity loss
- view weighting
- foreground point-cloud initialization
- standard 2DGS densification

Modify only the pruning decision path:

1. Generate the candidate prune mask using the existing `maybe_prune_gaussians()` logic.
2. Compute or reuse candidate risk scores:
   - low opacity
   - low brightness
   - low foreground-mask consistency
   - oversized / unstable scale
   - low view support if available
3. Let M2 produce a diagnostic report:
   - current Gaussian count
   - candidate removal count
   - mask-risk count
   - opacity-risk count
   - brightness-risk count
   - thin-structure protection warning if candidate removals concentrate near mask boundaries
4. Let M3 select the final removals under a per-round budget.
5. If `capacity_control_mode == "m2m3_floor"`, enforce the capacity floor:

```text
allowed_remove = max(0, current_count - floor_count)
final_remove = min(candidate_remove, per_round_budget, allowed_remove)
```

6. Call `gaussians.prune_points(final_prune_mask)` only after M2/M3/floor filtering.

### Stage 2: Deeper densification integration

Only do this if Stage 1 results are positive.

Potential changes:

- M2 diagnostics influence `opacity_cull`.
- M2 diagnostics influence `densify_grad_threshold`.
- M3 controls clone/split/prune ratios.
- Region-level floors protect leaf-boundary or high-view-support foreground regions.

This stage is more powerful but more likely to disturb the current stable ForeSplat code.

## 7. Method Variants

### Pilot methods

| ID | Method name | Description |
| --- | --- | --- |
| BL | FA-2DGS | Current foreground-aware 2DGS baseline |
| M2M3 | FA-2DGS + M2M3-current | M2/M3 capacity update without explicit floor |
| Floor40 | FA-2DGS + M2M3-Floor40 | M2/M3 with 40% capacity floor |
| PostPrune | FA-2DGS + equal-count post-pruning | Optional diagnostic baseline |

### Optional floor sweep

Run only on 2-3 scenes:

| Variant | Floor ratio |
| --- | ---: |
| Floor30 | 0.30 |
| Floor40 | 0.40 |
| Floor50 | 0.50 |

Purpose:

```text
Verify whether floor40 remains a stable quality-capacity operating point under foreground-aware standard 2DGS.
```

Do not claim floor40 is universal unless the sweep supports it.

## 8. Scene Selection

Use scenes already known to run under the current standard 2DGS pipeline first.

From previous logs, successful scenes included:

```text
BaiZhang
CaoMei2
KongQueZhuYu
WangWenCao1
XiangPiShu1
XianKeLai1
XianKeLai3
```

### Recommended 3-scene pilot

| Scene | Reason |
| --- | --- |
| KongQueZhuYu | complex background, important foreground leakage case |
| XiangPiShu1 | large / thick structure, mesh size and redundancy likely visible |
| WangWenCao1 or XianKeLai1 | representative successful scene with manageable runtime |

### Expanded plant set

If pilot is positive, expand to all successful current scenes first:

```text
7-scene foreground-aware standard 2DGS set
```

Then decide whether to fix failed scenes or include the larger 20-scene plant set.

## 9. Training Command Templates

The exact source, mask and output paths should be filled scene by scene.

### Common variables

```bash
REPO=/data/fj/F2DMAS/2d-gaussian-splatting-m2m3-floor40
EXP=/data/fj/F2DMAS/00-论文优化重构/m2m3实验
SCENE=KongQueZhuYu
SOURCE=/path/to/colmap_or_scene/${SCENE}
MASK_DIR=/path/to/rap_fsam3_or_alpha_masks/${SCENE}
VIEW_WEIGHT_LIST=/path/to/view_weights/${SCENE}.csv
```

### Baseline: current foreground-aware 2DGS

```bash
cd ${REPO}
python train.py \
  -s ${SOURCE} \
  -m ${EXP}/runs/${SCENE}/fa2dgs_bl \
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
  --iterations 30000
```

### M2M3-current

```bash
cd ${REPO}
python train.py \
  -s ${SOURCE} \
  -m ${EXP}/runs/${SCENE}/fa2dgs_m2m3 \
  [same foreground-aware options as baseline] \
  --capacity_control_mode m2m3 \
  --m2m3_score_mode topology \
  --m2m3_region_mode foreground \
  --m2m3_max_remove_ratio 0.03 \
  --save_capacity_report
```

### M2M3-Floor40

```bash
cd ${REPO}
python train.py \
  -s ${SOURCE} \
  -m ${EXP}/runs/${SCENE}/fa2dgs_m2m3_floor40 \
  [same foreground-aware options as baseline] \
  --capacity_control_mode m2m3_floor \
  --capacity_floor_ratio 0.4 \
  --capacity_floor_reference max_seen \
  --capacity_floor_start_iter 18000 \
  --m2m3_score_mode topology \
  --m2m3_region_mode foreground \
  --m2m3_max_remove_ratio 0.03 \
  --save_capacity_report
```

## 10. Evaluation Metrics

### Rendering quality

Use foreground-aware evaluation if masks are available:

- PSNR_fg
- SSIM_fg
- LPIPS_fg
- full-image PSNR / SSIM / LPIPS as secondary metrics

### Foreground leakage / background suppression

- outside-mask non-black ratio
- leakage energy ratio
- rendered alpha outside foreground mask
- residual background Gaussian count if measurable

### Capacity and efficiency

- Gaussian count
- PLY size / checkpoint size
- training time
- mesh extraction time
- cleaned PLY size if using post-cleaner

### Mesh quality

- number of connected components
- largest-component ratio
- floating artifact count
- mesh vertex count
- mesh file size
- optional plant trait agreement if manual measurements are available

### Stability diagnostics

- per-round removed Gaussian count
- floor active iterations
- candidate removals blocked by floor
- mask-risk / opacity-risk / brightness-risk breakdown
- final capacity trajectory over iterations

## 11. Main Tables

### Pilot result table

| Scene | Method | PSNR_fg | SSIM_fg | LPIPS_fg | Outside ratio | Leakage | Gaussians | PLY MiB | Mesh time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KongQueZhuYu | FA-2DGS |  |  |  |  |  |  |  |  |
| KongQueZhuYu | M2M3 |  |  |  |  |  |  |  |  |
| KongQueZhuYu | M2M3-Floor40 |  |  |  |  |  |  |  |  |

### Capacity-control diagnostic table

| Scene | Method | Max Gaussians | Floor count | Final Gaussians | Removed total | Blocked by floor | Floor active rounds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |

### Floor sweep table

| Scene | Variant | Floor ratio | PSNR_fg | LPIPS_fg | Gaussians | PLY MiB | Observation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |

## 12. Success Criteria

The migration is worth expanding if M2M3-Floor40 satisfies at least one of the following patterns:

### Strong success

- PSNR_fg / SSIM_fg / LPIPS_fg are equal or better than FA-2DGS.
- Gaussian count or PLY size decreases by at least 30%.
- Leakage or floating artifacts decrease.
- Mesh extraction time decreases.

### Moderate success

- Rendering quality drops only slightly, but Gaussian/PLY/mesh time drops substantially.
- M2M3-current over-prunes some scenes, while Floor40 recovers quality.

### Weak / negative result

- Floor40 reduces capacity but significantly damages leaf boundaries or local plant structures.
- Floor40 cannot beat simple post-pruning.
- Standard foreground-aware 2DGS is already compact enough and M2M3 provides little benefit.

If results are weak, keep this as a negative/limitation experiment rather than turning it into a main paper claim.

## 13. Expected Effect

The likely improvement is not from more accurate foreground definition, because the baseline already uses foreground priors. The likely improvements are:

- fewer redundant foreground Gaussians
- fewer background or pot/table residual primitives
- better protection against destructive pruning of thin leaves
- smaller PLY / checkpoint size
- faster mesh extraction
- cleaner TSDF mesh inputs

Rendering PSNR may improve in some scenes, but the safer expected claim is:

```text
M2M3-Floor40 improves the quality-capacity-mesh efficiency trade-off of foreground-aware standard 2DGS.
```

## 14. Risks and Mitigation

### Risk 1: M2M3 duplicates existing mask pruning

Mitigation:

- Treat M2M3 as a controller over candidate pruning, not just another pruning rule.
- Report how many candidate removals were blocked by the floor.
- Include equal-count post-pruning as a diagnostic baseline.

### Risk 2: Floor40 is too high for dense-view 2DGS

Mitigation:

- Run floor30/floor40/floor50 on 2-3 pilot scenes.
- Frame floor40 as a fixed operating point only after validation.

### Risk 3: Floor40 protects bad foreground Gaussians

Mitigation:

- Use mask consistency and brightness/opacity scores to decide which Gaussians are preserved.
- Consider region-level floor only after global floor pilot.

### Risk 4: Code changes break current ForeSplat pipeline

Mitigation:

- Work only in the copied repository.
- Default `capacity_control_mode = none`.
- Keep all current command lines valid.
- Add a quick regression run using baseline options before enabling M2M3.

### Risk 5: Pilot is dominated by dataset-specific plant masks

Mitigation:

- Include scenes with different plant morphology and background complexity.
- If successful, test at least one non-plant or public benchmark subset later.

## 15. Implementation Checklist

### Code changes

- [x] Add new optimization parameters in `arguments/__init__.py`.
- [x] Add capacity state initialization in `training()`.
- [x] Add a capacity controller function in `train.py` or `utils/capacity_control_utils.py`.
- [x] Wrap `maybe_prune_gaussians()` candidate pruning with M2M3/floor logic.
- [x] Save per-iteration capacity-control JSON reports.
- [x] Preserve old behaviour when `--capacity_control_mode none`.
- [x] Add a short README section or experiment script under `m2m3实验`.

### Validation checks

- [ ] Baseline command still runs in the copied repository.
- [ ] `capacity_control_mode none` gives the same behaviour as current copied baseline.
- [ ] M2M3-current writes reports and changes Gaussian count.
- [ ] M2M3-Floor40 prevents pruning below the floor.
- [x] Reports contain enough information for paper tables.
- [x] Syntax check passes with `python -m py_compile train.py arguments/__init__.py`.

### Experiment checks

- [ ] 3-scene pilot completed.
- [ ] Metrics collected into one CSV.
- [ ] Visual panels generated for at least one success and one failure/mixed case.
- [ ] Decision made: expand / revise / stop.

## 16. Paper Framing if Successful

Recommended framing:

```text
Capacity-protected M2M3 control for foreground-aware 2D Gaussian Splatting.
```

Do not frame this as a replacement for the Sparse2DGS paper. Instead:

- Sparse2DGS paper: sparse-view capacity protection.
- Foreground-aware standard 2DGS paper/experiment: plant-object reconstruction with foreground priors and capacity-protected compact optimization.

Possible contribution statement:

```text
We further integrate M2M3-Floor40 into foreground-aware standard 2DGS, showing that capacity-protected Gaussian control can reduce redundant primitives and improve mesh-oriented reconstruction efficiency without sacrificing foreground rendering quality.
```

## 17. Immediate Next Step

The parameter skeleton and first non-invasive controller have been implemented in the copied repository. The next step is a smoke run:

1. Run one short baseline command with `capacity_control_mode none`.
2. Run one short `m2m3_floor` command on the same scene with `--save_capacity_report`.
3. Confirm that `<model_path>/capacity_control/capacity_summary.json` is created.
4. Inspect whether the reported `total_blocked_by_floor` and `final_count` behave as expected.

This keeps the migration reversible and prevents accidental damage to the existing foreground-aware 2DGS workflow.
