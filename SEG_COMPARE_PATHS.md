# Segmentation Compare Paths (Standard Naming)

## Input Root
- `/data/fj/F2DMAS/02-FFT`

## Method Entry Scripts
- ExG + Otsu: `/data/fj/F2DMAS/exg_otsu/run_01_exg_otsu.py`
- YOLOv8-Seg: `/data/fj/F2DMAS/yolov8_seg/run_02_yolov8_seg.py`
- SAM1 (AMG): `/data/fj/F2DMAS/segment-anything/run_03_sam1_amg.py`

## Default Output Roots
- ExG + Otsu: `/data/fj/F2DMAS/results_seg_compare/01_exg_otsu`
- YOLOv8-Seg: `/data/fj/F2DMAS/results_seg_compare/02_yolov8_seg`
- SAM1 (AMG): `/data/fj/F2DMAS/results_seg_compare/03_sam1_amg`

## 环境状态

YOLOv8-Seg 和 SAM1 的旧 `venv/` 已于 2026-11-14 按批准清单清理。后续重跑需使用 uv 重建环境，历史依赖版本及本地扩展来源记录保存在：

`/data/fj/F2DMAS_cleanup_20261114_165053/environment_records`
