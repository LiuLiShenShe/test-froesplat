# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Undistort ETH3D ground-truth depth maps onto the undistorted (pinhole) cameras.

ETH3D ships the high-resolution multi-view ground-truth depth in the *distorted*
DSLR frame. This script re-samples each depth map onto the corresponding
undistorted PINHOLE camera using the exact ``THIN_PRISM_FISHEYE -> PINHOLE``
mapping computed with pycolmap, and writes the result as ``.exr``.

For each scene under ``--root`` it reads:

    <scene>/dslr_calibration_jpg/cameras.txt          (THIN_PRISM_FISHEYE params)
    <scene>/dslr_calibration_jpg/images.txt           (image list + camera ids)
    <scene>/dslr_calibration_undistorted/cameras.txt  (PINHOLE params)
    <scene>/ground_truth_depth/dslr_images/<name>     (raw float32 distorted depth)

and writes:

    <scene>/ground_truth_depth/dslr_images_undistorted/<name>.exr

Usage:
    python -m dvlt.scripts.preprocess.eth3d.undistort_depth --root /path/to/eth3d
    python -m dvlt.scripts.preprocess.eth3d.undistort_depth --root /path/to/eth3d \
        --scenes courtyard office --override

Requires: ``pycolmap`` and OpenEXR support in OpenCV (enabled below).
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional


# OpenEXR support in OpenCV must be enabled before importing cv2.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
from tqdm import tqdm


try:
    import pycolmap
except ImportError:
    pycolmap = None


def load_eth3d_raw_depth(path) -> np.ndarray:
    """Load an ETH3D raw (distorted) ground-truth depth map."""
    height, width = 4032, 6048
    depth = np.fromfile(path, dtype=np.float32).reshape(height, width)
    depth = np.nan_to_num(depth, posinf=0.0, neginf=0.0, nan=0.0)
    return depth


def undistort_depth_maps(scene_folder: Path, override: bool = False) -> None:
    """Undistort a single scene's depth maps onto the undistorted pinhole cameras.

    Maps undistorted pixel coordinates back through the THIN_PRISM_FISHEYE model
    (via pycolmap) to sample the distorted ground-truth depth, then writes the
    result as ``.exr`` under ``ground_truth_depth/dslr_images_undistorted``.
    """
    if pycolmap is None:
        raise ImportError(
            "pycolmap is required for undistorting ETH3D depth maps. " "Install it with: pip install pycolmap"
        )

    undistorted_depth_dir = scene_folder / "ground_truth_depth" / "dslr_images_undistorted"
    undistorted_depth_dir.mkdir(parents=True, exist_ok=True)

    # Distorted cameras (THIN_PRISM_FISHEYE)
    distorted_cameras_txt_path = scene_folder / "dslr_calibration_jpg" / "cameras.txt"
    with open(distorted_cameras_txt_path, "r") as f:
        distorted_camera_lines = f.readlines()[3:]  # Skip header

    distorted_camera_params_dict = {}
    for line in distorted_camera_lines:
        parts = line.strip().split()
        distorted_camera_id = int(parts[0])
        distorted_params = list(map(float, parts[4:]))
        distorted_camera_params_dict[distorted_camera_id] = pycolmap.Camera(
            model="THIN_PRISM_FISHEYE",
            width=int(parts[2]),
            height=int(parts[3]),
            params=distorted_params,
        )

    # Undistorted cameras (PINHOLE)
    undistorted_cameras_txt_path = scene_folder / "dslr_calibration_undistorted" / "cameras.txt"
    with open(undistorted_cameras_txt_path, "r") as f:
        undistorted_camera_lines = f.readlines()[3:]  # Skip header

    undistorted_camera_params_dict = {}
    for line in undistorted_camera_lines:
        parts = line.strip().split()
        undistorted_camera_id = int(parts[0])
        undistorted_params = list(map(float, parts[4:]))
        undistorted_camera_params_dict[undistorted_camera_id] = pycolmap.Camera(
            model="PINHOLE",
            width=int(parts[2]),
            height=int(parts[3]),
            params=undistorted_params,
        )

    # Precompute distorted image coordinates for each camera ID
    distorted_img_coords_dict = {}
    for camera_id, undistorted_camera in undistorted_camera_params_dict.items():
        height, width = undistorted_camera.height, undistorted_camera.width
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        image_points = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)

        # cam_from_img returns normalized 2D points (z=1 plane); img_from_cam
        # expects 3D camera points, so homogenize with a unit z column.
        pinhole_world_pts = undistorted_camera.cam_from_img(image_points)
        pinhole_world_pts = np.concatenate([pinhole_world_pts, np.ones((pinhole_world_pts.shape[0], 1))], axis=1)

        distorted_camera = distorted_camera_params_dict[camera_id]
        distorted_img_coords = distorted_camera.img_from_cam(pinhole_world_pts)
        distorted_img_coords = np.clip(distorted_img_coords, 0, [width - 1, height - 1])
        distorted_img_coords = distorted_img_coords.astype(int)

        distorted_img_coords_dict[camera_id] = distorted_img_coords

    # Distorted images list
    distorted_images_txt_path = scene_folder / "dslr_calibration_jpg" / "images.txt"
    with open(distorted_images_txt_path, "r") as f:
        distorted_image_lines = f.readlines()[4:]  # Skip header

    for i in tqdm(range(0, len(distorted_image_lines), 2)):
        parts = distorted_image_lines[i].strip().split()
        _, _, _, _, _, _, _, _, camera_id, image_name = parts[:10]
        camera_id = int(camera_id)
        base_name = os.path.basename(image_name)

        undistorted_depth_path = undistorted_depth_dir / base_name.replace(".JPG", ".exr")
        if undistorted_depth_path.exists() and not override:
            continue

        depth_map_path = scene_folder / "ground_truth_depth" / "dslr_images" / base_name
        if not depth_map_path.exists():
            print(f"Warning: Depth map not found for {base_name}, skipping...")
            continue

        depth_map = load_eth3d_raw_depth(depth_map_path)

        if camera_id not in distorted_img_coords_dict:
            print(f"Warning: Camera ID {camera_id} not found in distorted_img_coords_dict, skipping...")
            continue
        distorted_img_coords = distorted_img_coords_dict[camera_id]

        undistorted_depth = depth_map[distorted_img_coords[:, 1], distorted_img_coords[:, 0]]

        undistorted_camera = undistorted_camera_params_dict[camera_id]
        height, width = undistorted_camera.height, undistorted_camera.width

        undistorted_depth = undistorted_depth.reshape(height, width).copy()

        cv2.imwrite(str(undistorted_depth_path), undistorted_depth.astype(np.float32))


def preprocess_eth3d_depth(root: str, scenes: Optional[List[str]] = None, override: bool = False) -> None:
    """Undistort the GT depth maps for every (or the requested) ETH3D scene."""
    root_path = Path(root)

    if scenes is None:
        scene_paths = sorted(p.parent for p in root_path.glob("*/dslr_calibration_undistorted"))
        scenes = [p.name for p in scene_paths]

    if len(scenes) == 0:
        raise ValueError(f"No scenes found in {root}")

    print(f"Found {len(scenes)} scenes to process: {scenes}")

    for scene_name in scenes:
        scene_root = root_path / scene_name
        print(f"\nProcessing scene: {scene_name}")

        original_depth_dir = scene_root / "ground_truth_depth" / "dslr_images"
        undistorted_depth_dir = scene_root / "ground_truth_depth" / "dslr_images_undistorted"

        if not original_depth_dir.exists():
            print(f"  No depth directory found for {scene_name}, skipping...")
            continue

        original_depth_files = list(original_depth_dir.glob("*.JPG"))
        files_to_process = [
            f.name
            for f in original_depth_files
            if not (undistorted_depth_dir / f.name.replace(".JPG", ".exr")).exists()
        ]

        if len(files_to_process) == 0 and not override:
            print(f"  All {len(original_depth_files)} depth maps already undistorted")
        else:
            if override:
                print(f"  Re-undistorting all {len(original_depth_files)} depth maps (override=True)...")
            else:
                print(f"  Undistorting {len(files_to_process)}/{len(original_depth_files)} depth maps...")
            undistort_depth_maps(scene_root, override=override)

    print("\nPreprocessing complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Undistort ETH3D ground-truth depth maps onto the pinhole cameras.")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing ETH3D scenes (e.g. ${user.data_root}/test/eth3d).",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        nargs="*",
        default=None,
        help="Specific scenes to process (default: all scenes under --root).",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Re-process existing undistorted depth maps.",
    )
    args = parser.parse_args()

    print("ETH3D depth undistortion")
    print(f"Root: {args.root}")
    print(f"Scenes: {args.scenes if args.scenes else 'all'}")
    print(f"Override: {args.override}")
    print("-" * 60)

    preprocess_eth3d_depth(args.root, args.scenes, override=args.override)


if __name__ == "__main__":
    main()
