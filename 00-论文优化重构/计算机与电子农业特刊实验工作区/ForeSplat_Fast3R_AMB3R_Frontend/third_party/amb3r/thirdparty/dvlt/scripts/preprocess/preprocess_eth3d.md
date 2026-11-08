# ETH3D preprocessing (evaluation)

ETH3D high-resolution multi-view benchmark:
<https://www.eth3d.net/datasets#high-res-multi-view>.

## 1. Download

Fetches undistorted DSLR images + calibration, the distorted
(THIN_PRISM_FISHEYE) calibration, and per-scene GT depth (needs `wget`, `7z`):

```bash
bash src/dvlt/scripts/preprocess/eth3d/download.sh datasets/test/eth3d
```

Layout, one directory per scene:

```text
test/eth3d/
└── <scene>/
    ├── dslr_calibration_jpg/{cameras.txt, images.txt}            # distorted (THIN_PRISM_FISHEYE)
    ├── dslr_calibration_undistorted/{cameras.txt, images.txt}    # undistorted (PINHOLE) + poses
    ├── images/dslr_images_undistorted/*.JPG
    └── ground_truth_depth/dslr_images/*                          # distorted GT depth, float32 raw
```

## 2. Undistort depth

Maps the distorted GT depth onto the pinhole cameras (via `pycolmap`), writing
`.exr` to `ground_truth_depth/dslr_images_undistorted/`:

```bash
python -m dvlt.scripts.preprocess.eth3d.undistort_depth \
    --root datasets/test/eth3d
    # optional: --scenes courtyard delivery_area ...   (default: all)
    # optional: --override                             (re-process existing)
```

Preprocess-only deps: `pycolmap`, OpenEXR support in OpenCV.

Consumed by
[`test_datasets/eth3d.yaml`](../../config/experiments/data/test_datasets/eth3d.yaml),
read from `${user.data_root}/test/eth3d`.
