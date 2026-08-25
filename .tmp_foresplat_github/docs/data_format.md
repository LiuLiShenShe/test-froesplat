# Data Format

ForeSplat expects one sequence folder per plant or acquisition run. A compact demo sequence is provided under `examples/demo_sequence/`.

## Sequence layout

```text
demo_sequence/
├── images/
│   ├── 000000.png
│   ├── 000001.png
│   └── 000002.png
├── masks/
│   ├── 000000.png
│   ├── 000001.png
│   └── 000002.png
├── cameras/
│   ├── transforms_demo.json
│   └── foreground_tracks.csv
└── scale_marker.json
```

## RGB frames

- Format: `.png`, `.jpg`, `.jpeg`, `.tif` or `.tiff`
- Naming: stable zero-padded frame names are recommended, for example `000000.png`
- Content: multiview RGB frames of the target potted plant

## Foreground-prior masks

- Format: single-channel PNG
- Values: `0` for background and `255` for foreground
- Alignment: masks must share the same width, height and frame order as `images/`
- Interpretation: the mask represents the potted-plant foreground object used for reconstruction; it may include the visible supporting container if the VFM consistently groups it with the plant.

## Camera manifest

`cameras/transforms_demo.json` follows a NeRF-style transform manifest:

```json
{
  "camera_model": "demo_pinhole",
  "frames": [
    {
      "file_path": "../images/000000.png",
      "mask_path": "../masks/000000.png",
      "w": 256,
      "h": 192,
      "fl_x": 217.6,
      "fl_y": 217.6,
      "cx": 128.0,
      "cy": 96.0,
      "transform_matrix": [[...]]
    }
  ]
}
```

For full experiments, this file can be generated from COLMAP outputs. The accompanying `foreground_tracks.csv` stores foreground-support summaries derived from mask-consistent sparse tracks.

## Scale marker

`scale_marker.json` stores the scene-to-metric scale:

```json
{
  "units_per_meter": 1.0,
  "reference_object": "demo scale marker",
  "note": "Replace with the calibrated reference used in the full experiment."
}
```

## Result files

The expected evaluation outputs are CSV files:

- `rendering_metrics.csv`: foreground PSNR/SSIM and leakage metrics
- `traits.csv`: virtual plant-trait measurements
- `phenotype_metrics.csv`: manual-virtual agreement metrics when manual measurements are available
