# Segmentation Compare Paths (Standard Naming)

## Input Root
- `/data/fj/02-FFT`

## Method Entry Scripts
- ExG + Otsu: `/data/fj/exg_otsu/run_01_exg_otsu.py`
- YOLOv8-Seg: `/data/fj/yolov8_seg/run_02_yolov8_seg.py`
- SAM1 (AMG): `/data/fj/segment-anything/run_03_sam1_amg.py`

## Default Output Roots
- ExG + Otsu: `/data/fj/results_seg_compare/01_exg_otsu`
- YOLOv8-Seg: `/data/fj/results_seg_compare/02_yolov8_seg`
- SAM1 (AMG): `/data/fj/results_seg_compare/03_sam1_amg`

## Quick Run Examples
```bash
/data/fj/exg_otsu/venv/bin/python /data/fj/exg_otsu/run_01_exg_otsu.py --folder BaiZhang
/data/fj/yolov8_seg/venv/bin/python /data/fj/yolov8_seg/run_02_yolov8_seg.py --folder BaiZhang
/data/fj/segment-anything/venv/bin/python /data/fj/segment-anything/run_03_sam1_amg.py --folder BaiZhang
```

