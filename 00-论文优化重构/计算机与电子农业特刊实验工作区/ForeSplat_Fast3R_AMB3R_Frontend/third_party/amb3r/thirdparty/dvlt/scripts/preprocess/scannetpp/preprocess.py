# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preprocess ScanNet++ DSLR captures into undistorted images + rendered metric depth.

Reads the raw release (``data/`` + ``splits/``) and writes the layout the dataset
configs expect, relative to ``user.data_root``:
``{output_root}/{train,test}/scannetpp/scannetpp_undistort/{scene}/``. Point the
configs' ``user.data_root`` at ``--output_root`` after running.

Expected ``--data_root`` layout (official ScanNet++ release)::

    data_root/
    ├── splits/
    │   ├── nvs_sem_train.txt              # scene ids, one per line
    │   └── nvs_sem_val.txt
    └── data/
        └── <SCENE_ID>/
            ├── scans/mesh_aligned_0.05.ply        # GT mesh (depth source)
            └── dslr/
                ├── resized_images/*.JPG           # fisheye RGB
                ├── colmap/images.txt              # world-to-camera poses
                └── nerfstudio/transforms.json     # fisheye intrinsics + k1..k4

Usage:
    # Both splits (nvs_sem_train -> train/, nvs_sem_val -> test/)
    python -m dvlt.scripts.preprocess.scannetpp.preprocess \
        --data_root datasets/scannetpp_raw \
        --output_root datasets

    # Only the eval split, on two specific scenes
    python -m dvlt.scripts.preprocess.scannetpp.preprocess \
        --data_root datasets/scannetpp_raw \
        --output_root datasets \
        --splits nvs_sem_val \
        --scene_ids 8b5caf3398 b20a261fdf
"""

import argparse

from dvlt.scripts.preprocess.scannetpp.processor import SPLIT_TO_SUBSET, ScanNetppProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="ScanNet++ DSLR depth-render + undistort preprocessing")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Raw ScanNet++ release root (contains data/ and splits/)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Processed dataset root (== user.data_root); writes {output_root}/{train,test}/scannetpp/...",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=list(SPLIT_TO_SUBSET),
        choices=list(SPLIT_TO_SUBSET),
        help="Splits to process (default: both)",
    )
    parser.add_argument(
        "--scene_ids",
        type=str,
        nargs="+",
        default=None,
        help="Optional explicit scene subset (default: all scenes in each split)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for rasterization",
    )
    parser.add_argument(
        "--bin_size",
        type=int,
        default=None,
        help="PyTorch3D rasterization bin size (default: safe heuristic)",
    )
    parser.add_argument(
        "--max_faces_per_bin",
        type=int,
        default=None,
        help=(
            "PyTorch3D per-bin face budget (default: safe heuristic). A smaller budget "
            "renders much faster, but too small silently drops faces and corrupts depth"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render scenes even if outputs already exist",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    processor = ScanNetppProcessor(
        data_root=args.data_root,
        output_root=args.output_root,
        splits=args.splits,
        scene_ids=args.scene_ids,
        device=args.device,
        overwrite=args.overwrite,
        bin_size=args.bin_size,
        max_faces_per_bin=args.max_faces_per_bin,
    )
    processor.run()


if __name__ == "__main__":
    main()
