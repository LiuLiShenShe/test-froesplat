# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, List, Union, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from . import CameraModel, DataField, DatasetMetadata, DataSource, DepthType
from .base import BaseDataset


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def rot90_pinhole_intrinsics(intrinsics: np.ndarray, hw: Tuple[int, int], factor: int) -> np.ndarray:
    """Rotate pinhole intrinsics ``[fx, fy, cx, cy]`` by ``factor`` * 90 degrees CCW.

    ``hw`` is the image ``(height, width)`` before rotation.
    """
    factor %= 4
    if factor == 0:
        return intrinsics

    h, w = hw
    fx, fy, cx, cy = intrinsics
    if factor == 1:
        return np.array([fy, fx, cy, w - cx], dtype=intrinsics.dtype)
    if factor == 2:
        return np.array([fx, fy, w - cx, h - cy], dtype=intrinsics.dtype)
    return np.array([fy, fx, h - cy, cx], dtype=intrinsics.dtype)


def rot_z_pose(
    pose: Union[np.ndarray, torch.Tensor], degree: Optional[float] = None, rad: Optional[float] = None
) -> Union[np.ndarray, torch.Tensor]:
    """Rotate a 4x4 pose CCW about the Z axis by ``degree`` or ``rad`` (exactly one)."""
    if (degree is None) == (rad is None):
        raise ValueError("Either degree or rad must be provided.")

    RT = np.eye(4)
    RT[:3, :3] = Rotation.from_euler("z", degree or rad, degrees=degree is not None).as_matrix()

    if isinstance(pose, torch.Tensor):
        return pose @ torch.from_numpy(RT).to(pose)
    return pose @ RT.astype(pose.dtype)


def pose_matrix_from_quaternion(pvec):
    """
    Get 4x4 pose matrix from quaternion (t, q)
    t = (tx, ty, tz)
    q = (qw, qx, qy, qz)
    """
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = Rotation.from_quat(pvec[3:], scalar_first=True).as_matrix()
    pose[:3, 3] = pvec[:3]
    return pose


# Dictionary of images that were originally portrait but are now landscape in the ETH3D dataset
originally_portrait_imgs_in_eth3d_dataset = {
    "delivery_area": ["DSC_0711.JPG", "DSC_0712.JPG", "DSC_0713.JPG", "DSC_0714.JPG"],
    "playground": [
        "DSC_0587.JPG",
        "DSC_0588.JPG",
        "DSC_0589.JPG",
        "DSC_0590.JPG",
        "DSC_0591.JPG",
        "DSC_0592.JPG",
    ],
    "relief": [
        "DSC_0427.JPG",
        "DSC_0428.JPG",
        "DSC_0429.JPG",
        "DSC_0430.JPG",
        "DSC_0431.JPG",
        "DSC_0432.JPG",
        "DSC_0433.JPG",
        "DSC_0434.JPG",
        "DSC_0435.JPG",
        "DSC_0436.JPG",
        "DSC_0437.JPG",
        "DSC_0438.JPG",
        "DSC_0439.JPG",
    ],
    "relief_2": [
        "DSC_0458.JPG",
        "DSC_0459.JPG",
        "DSC_0460.JPG",
        "DSC_0461.JPG",
        "DSC_0462.JPG",
        "DSC_0463.JPG",
        "DSC_0464.JPG",
        "DSC_0465.JPG",
        "DSC_0466.JPG",
        "DSC_0467.JPG",
        "DSC_0468.JPG",
    ],
}


class ETH3D(BaseDataset):
    """
    ETH3D dataset loader.

    Each scene is treated as a video, with each image as a frame.
    Loads poses, intrinsics, images and undistorted depth directly from ETH3D format.

    Expected directory structure:
    root/
        ├── courtyard/
        │   ├── dslr_calibration_undistorted/
        │   │   ├── cameras.txt
        │   │   ├── images.txt
        │   ├── ground_truth_depth/
    │   │   ├── dslr_images_undistorted/
    │   │       ├── DSC_0286.exr
        │   ├── images/
        │   │   ├── dslr_images_undistorted/
    │   │       ├── DSC_0286.JPG
        ├── delivery_area/
        │   ├── ...
    """

    def __init__(
        self,
        root_path: str,
        scenes: List[str] | None = None,
        load_depth: bool = True,
        rotate_portrait_images: bool = False,
        sort_frames: bool = True,
    ):
        """
        Args:
            root: Root directory containing ETH3D scenes
            scenes: List of scene names to load. If None, loads all scenes.
            load_depth: Whether to load depth maps
            rotate_portrait_images: Whether to rotate portrait images back to their original
                orientation. Default False, images returned as stored (rotated 90° CCW, landscape images).
            sort_frames: Whether to sort frames alphabetically by filename.

        Note:
            Depth maps must be undistorted beforehand:
            python -m dvlt.scripts.preprocess.eth3d.undistort_depth --root <root_path>
        """
        super().__init__()
        self.root = Path(root_path)
        self.load_depth = load_depth
        self.rotate_portrait_images = rotate_portrait_images
        self.sort_frames = sort_frames

        # Find all scenes
        if scenes is None:
            # Auto-detect scenes by looking for dslr_calibration_undistorted folders
            scene_paths = sorted([p.parent for p in self.root.glob("*/dslr_calibration_undistorted")])
            self.scenes = [p.name for p in scene_paths]
        else:
            self.scenes = scenes

        if len(self.scenes) == 0:
            raise ValueError(f"No scenes found in {root_path}")

        # Preprocess each scene
        self._preprocess_scenes()

    def _preprocess_scenes(self):
        """Load metadata for all scenes."""
        self.scene_data = []

        for scene_name in self.scenes:
            scene_root = self.root / scene_name

            # Load camera parameters
            cameras_dict = self._load_cameras(scene_root)

            # Load image metadata and poses
            frames = self._load_images(scene_root, cameras_dict)

            self.scene_data.append(
                {
                    "scene_name": scene_name,
                    "scene_root": scene_root,
                    "cameras": cameras_dict,
                    "frames": frames,
                }
            )

    def _load_cameras(self, scene_root: Path) -> dict:
        """Load camera parameters from cameras.txt."""
        cameras_txt = scene_root / "dslr_calibration_undistorted" / "cameras.txt"
        cameras_dict = {}

        with open(cameras_txt, "r") as f:
            lines = f.readlines()[3:]  # Skip header

        for line in lines:
            parts = line.strip().split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))

            cameras_dict[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }

        return cameras_dict

    def _load_images(self, scene_root: Path, cameras_dict: dict) -> list:
        """Load image metadata and poses from images.txt."""
        images_txt = scene_root / "dslr_calibration_undistorted" / "images.txt"
        frames = []

        with open(images_txt, "r") as f:
            lines = f.readlines()[4:]  # Skip header

        # Process every other line (skip POINTS2D lines)
        for i in range(0, len(lines), 2):
            parts = lines[i].strip().split()
            _, qw, qx, qy, qz, tx, ty, tz, camera_id, name = parts[:10]
            camera_id = int(camera_id)
            base_name = os.path.basename(name)

            # Get camera info
            camera_info = cameras_dict[camera_id]
            width = camera_info["width"]
            height = camera_info["height"]
            fx, fy, cx, cy = camera_info["params"][:4]

            # Get file paths
            image_path = scene_root / "images" / "dslr_images_undistorted" / base_name
            depth_path = (
                scene_root / "ground_truth_depth" / "dslr_images_undistorted" / base_name.replace(".JPG", ".exr")
            )

            # Skip if image doesn't exist
            if not image_path.exists():
                continue

            # Skip if depth doesn't exist and we need to load it
            if self.load_depth and not depth_path.exists():
                continue

            # Get camera pose (world to camera -> camera to world)
            w2c_pose = pose_matrix_from_quaternion(
                [float(tx), float(ty), float(tz), float(qw), float(qx), float(qy), float(qz)]
            )
            c2w_pose = np.linalg.inv(w2c_pose)

            # Check if image is originally portrait
            scene_name = scene_root.name
            is_portrait = (
                scene_name in originally_portrait_imgs_in_eth3d_dataset
                and base_name in originally_portrait_imgs_in_eth3d_dataset[scene_name]
            )

            # Adjust for portrait rotation
            # Images are stored rotated 90° CCW, so we apply 90° CW rotation to camera frame
            if is_portrait and self.rotate_portrait_images:
                intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)
                adjusted_intrinsics = rot90_pinhole_intrinsics(intrinsics, (height, width), factor=1)
                fx, fy, cx, cy = adjusted_intrinsics
                c2w_pose = rot_z_pose(c2w_pose, degree=-90)
                # Swap width and height
                width, height = height, width

            frames.append(
                {
                    "image_path": image_path,
                    "depth_path": depth_path if self.load_depth else None,
                    "c2w_pose": c2w_pose,
                    "intrinsics": np.array([fx, fy, cx, cy], dtype=np.float32),
                    "width": width,
                    "height": height,
                    "is_portrait": is_portrait,
                }
            )

        if self.sort_frames:
            frames.sort(key=lambda f: f["image_path"].name)
        return frames

    def _read_data(
        self,
        video_idx: int,
        frame_idxs: List[int],
        view_idxs: List[int],
        data_fields: List[DataField],
    ) -> dict[DataField, Any]:
        """Read data for specified frames."""
        scene_data = self.scene_data[video_idx]
        frames = scene_data["frames"]

        output_dict = {}
        output_dict[DataField.KEY] = scene_data["scene_name"]

        for data_field in data_fields:
            if data_field == DataField.KEY:
                pass  # Already included

            elif data_field == DataField.IMAGE_RGB:
                images = []
                for idx in frame_idxs:
                    frame = frames[idx]
                    img = Image.open(frame["image_path"])

                    # Rotate if portrait and rotation is enabled
                    if frame["is_portrait"] and self.rotate_portrait_images:
                        img = img.rotate(-90, expand=True)

                    # Convert to tensor [3, H, W], range [0, 1]
                    img_array = np.array(img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                    images.append(img_tensor)

                output_dict[data_field] = images

            elif data_field == DataField.DEPTH:
                if not self.load_depth:
                    raise ValueError("Depth loading is disabled")

                depths = []
                for idx in frame_idxs:
                    frame = frames[idx]
                    # Load depth from .exr file
                    depth_array = cv2.imread(str(frame["depth_path"]), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

                    # Handle multi-channel .exr (take first channel)
                    if len(depth_array.shape) == 3:
                        depth_array = depth_array[:, :, 0]

                    # Rotate if portrait and rotation is enabled
                    if frame["is_portrait"] and self.rotate_portrait_images:
                        depth_array = np.rot90(depth_array, k=3)

                    depth_tensor = torch.from_numpy(depth_array.copy()).float()
                    depths.append(depth_tensor)

                output_dict[data_field] = depths

            elif data_field == DataField.CAMERA_C2W_TRANSFORM:
                poses = []
                for idx in frame_idxs:
                    poses.append(torch.from_numpy(frames[idx]["c2w_pose"]).float())
                output_dict[data_field] = torch.stack(poses)

            elif data_field == DataField.CAMERA_INTRINSICS:
                intrinsics = []
                for idx in frame_idxs:
                    intrinsics.append(torch.from_numpy(frames[idx]["intrinsics"]).float())
                output_dict[data_field] = torch.stack(intrinsics)

            elif data_field == DataField.CAMERA_MODEL:
                # ETH3D uses PINHOLE model
                models = torch.full((len(frame_idxs),), CameraModel.PINHOLE.value, dtype=torch.uint8)
                output_dict[data_field] = models

            else:
                raise NotImplementedError(f"Data field {data_field} not supported")

        return output_dict

    def available_data_fields(self) -> List[DataField]:
        """Return available data fields."""
        fields = [
            DataField.KEY,
            DataField.IMAGE_RGB,
            DataField.CAMERA_C2W_TRANSFORM,
            DataField.CAMERA_INTRINSICS,
            DataField.CAMERA_MODEL,
        ]
        if self.load_depth:
            fields.extend([DataField.DEPTH])
        return fields

    def num_videos(self) -> int:
        """Return number of scenes."""
        return len(self.scenes)

    def num_views(self, video_idx: int) -> int:
        """Return number of views (always 1 for ETH3D)."""
        return 1

    def num_frames(self, video_idx: int, view_idx: int = 0) -> int:
        """Return number of frames in a scene."""
        return len(self.scene_data[video_idx]["frames"])

    def get_dataset_metadata(self) -> DatasetMetadata:
        """Return dataset metadata."""
        return DatasetMetadata(
            has_ordered_frames=False,
            fps=None,
            has_variable_size_images=True,  # Different cameras may have different resolutions
            is_metric_scale=True,
            depth_type=DepthType.METRIC,
            data_origin=DataSource.REAL,
        )
