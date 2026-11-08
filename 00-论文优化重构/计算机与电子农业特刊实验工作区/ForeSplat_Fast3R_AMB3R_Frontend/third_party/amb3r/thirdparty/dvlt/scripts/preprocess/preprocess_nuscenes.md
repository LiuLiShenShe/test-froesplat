# NuScenes preprocessing (evaluation)

Converts raw NuScenes into one `scene.h5` per scene (CAM_FRONT images, sparse
LiDAR depth, camera extrinsics + intrinsics).

## 1. Download

Register at <https://www.nuscenes.org/>, accept the license, and download
(use **v1.0-trainval** — it contains the 50 eval scenes):

| Archive | Provides |
|---|---|
| `v1.0-trainval_meta.tgz` | metadata tables + `maps/` |
| `v1.0-trainval{01..10}_blobs.tgz` | `samples/` keyframes (CAM_FRONT + LIDAR_TOP) |
| `nuScenes-lidarseg-all-v1.0.tar.bz2` | lidarseg labels (filters noise/ego points) |

Extract all into one root (`--data_root`)
Use the helper script:

```bash
bash src/dvlt/scripts/preprocess/nuscenes_h5/extract.sh <archives_dir> [dest_root]
```

## 2. Raw data structure

```text
datasets/nuscenes_raw/
├── samples/
│   ├── CAM_FRONT/*.jpg          # images (read)
│   └── LIDAR_TOP/*.pcd.bin      # point clouds -> depth (read)
├── sweeps/                      # unused
├── maps/                        # unused
├── v1.0-trainval/               # metadata: scene/sample/sample_data/calibrated_sensor/ego_pose/lidarseg + category.json
└── lidarseg/v1.0-trainval/*_lidarseg.bin
```

## 3. Install (preprocess-only)

Optional dependencies for nuscenes preprocess:

```bash
pip install nuscenes-devkit pyquaternion
```

## 4. Run

```bash
python -m dvlt.scripts.preprocess.nuscenes_h5.preprocess \
    --data_root datasets/nuscenes_raw \
    --target_dir datasets/test/nuscenes \
    --split v1.0-trainval \
    --scene_split eval \
    --lidarseg_root datasets/nuscenes_raw/v1.0-trainval \
    --workers 8
```

Output:
`datasets/test/nuscenes/trainval/<scene>/scene.h5`, consumed by
[`test_datasets/nuscenes.yaml`](../../config/experiments/data/test_datasets/nuscenes.yaml).
