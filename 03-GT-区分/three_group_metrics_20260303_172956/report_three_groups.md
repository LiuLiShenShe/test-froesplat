# Three-Group Comparison Report

- Output dir: `/data/fj/03-GT/three_group_metrics_20260303_172956`
- Evaluation subset: common frames (`GT ∩ SAM ∩ SEEM`).

## 1) Global Basic Performance (Plant)

| Method | Plant mIoU | Plant F1 |
|---|---:|---:|
| SAM | 0.979868 | 0.983737 |
| SEEM | 0.940929 | 0.951142 |

## 2) Extreme Edge Precision (Plant)

| Method | Plant HD95 (px) | Plant Boundary F1 |
|---|---:|---:|
| SAM | 41.393716 | 0.389265 |
| SEEM | 281.823297 | 0.229724 |

## 3) Composite Semantic Alignment (Blue Block)

| Method | Block Recall (pixel-level) | Block Detection Rate (frame-level) |
|---|---:|---:|
| SAM | 0.985841 | 1.000000 |
| SEEM | 0.000000 | 0.117647 |

## Files

- Overall summary: `/data/fj/03-GT/three_group_metrics_20260303_172956/summary_overall_three_groups.csv`
- Per-plant summary: `/data/fj/03-GT/three_group_metrics_20260303_172956/summary_per_plant_three_groups.csv`
- Frame-level metrics: `/data/fj/03-GT/three_group_metrics_20260303_172956/frame_metrics_three_groups_common.csv`
- Common-frame counts: `/data/fj/03-GT/three_group_metrics_20260303_172956/common_frame_counts.csv`
