# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
"""NuScenes H5 dataset loader.

Loads preprocessed NuScenes data from HDF5 files created by the nuscenes preprocessing scripts.
Each scene is stored as a separate h5 file with the following structure:
- images/{idx}: JPEG-encoded image bytes for each frame (CAM_FRONT only)
- depths: (num_frames, H, W) float32 sparse depth maps from LiDAR projection (0.0 = invalid)
- extrinsics_c2w: (num_frames, 4, 4) float64 camera-to-world transformation matrices
- intrinsics: (num_frames, 3, 3) float64 camera intrinsic matrices
"""

from pathlib import Path
from typing import Any, List, Optional

import cv2
import h5py
import numpy as np
import torch

from dvlt.common.pose import inverse_pose
from dvlt.data.sources.datasets import DataSource
from dvlt.data.sources.datasets.base import BaseDataset, DataField, DatasetMetadata, DepthType


# Reference NuScenes evaluation split (the 50 v1.0-trainval val scenes used for
# depth/pose benchmarking).
NUSCENES_EVAL_SCENES: tuple[str, ...] = (
    "scene-0013",
    "scene-0016",
    "scene-0035",
    "scene-0039",
    "scene-0094",
    "scene-0097",
    "scene-0100",
    "scene-0103",
    "scene-0106",
    "scene-0109",
    "scene-0268",
    "scene-0271",
    "scene-0274",
    "scene-0277",
    "scene-0330",
    "scene-0344",
    "scene-0519",
    "scene-0522",
    "scene-0552",
    "scene-0555",
    "scene-0558",
    "scene-0561",
    "scene-0564",
    "scene-0626",
    "scene-0630",
    "scene-0634",
    "scene-0637",
    "scene-0771",
    "scene-0778",
    "scene-0782",
    "scene-0794",
    "scene-0797",
    "scene-0800",
    "scene-0905",
    "scene-0908",
    "scene-0911",
    "scene-0914",
    "scene-0917",
    "scene-0921",
    "scene-0924",
    "scene-0927",
    "scene-0930",
    "scene-0963",
    "scene-0968",
    "scene-0972",
    "scene-1061",
    "scene-1064",
    "scene-1067",
    "scene-1070",
    "scene-1073",
)


class NuScenesH5(BaseDataset):
    """NuScenes dataset loader for preprocessed H5 files.

    Args:
        root_path: Path to the processed nuscenes data directory containing scene folders.
        split: Optional split name (e.g., "mini", "trainval"). If provided, looks for data
            in {root_path}/{split}/. If None, looks directly in root_path.
        scenes: Scene selection. ``"eval"`` (default) uses the baked-in
            :data:`NUSCENES_EVAL_SCENES` reference split; ``"all"`` uses every
            directory containing ``scene.h5``; a path string loads scene names from
            a text file (one per line); a list/tuple is used as the scene names.
        normalize_first_frame: If True, transforms camera poses so that frame 0 of each
            scene has identity pose (camera-to-world = I). Default True.
    """

    def __init__(
        self,
        root_path: str,
        split: Optional[str] = None,
        scenes: str | List[str] = "eval",
        normalize_first_frame: bool = True,
    ):
        super().__init__()
        self.root_path = Path(root_path)

        if split is not None:
            self.data_dir = self.root_path / split
        else:
            self.data_dir = self.root_path

        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        scene_names = self._resolve_scene_names(scenes)
        if scene_names is None:
            # Use every directory that contains scene.h5
            self.scene_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and (d / "scene.h5").exists()])
        else:
            self.scene_dirs = sorted(
                [self.data_dir / name for name in scene_names if (self.data_dir / name / "scene.h5").exists()]
            )

        if len(self.scene_dirs) == 0:
            raise ValueError(f"No valid scene directories found in {self.data_dir}")

        self.scene_names = [d.name for d in self.scene_dirs]
        self.normalize_first_frame = normalize_first_frame

        # Cache for number of frames per scene (lazy loaded)
        self._num_frames_cache: dict[int, int] = {}

    @staticmethod
    def _resolve_scene_names(scenes: str | List[str]) -> Optional[List[str]]:
        """Return the explicit list of scene names, or None to glob all scenes."""
        if isinstance(scenes, (list, tuple)):
            return list(scenes)
        if scenes == "eval":
            return list(NUSCENES_EVAL_SCENES)
        if scenes == "all":
            return None
        # Otherwise treat as a path to a text file (one scene name per line).
        with Path(scenes).open("r") as f:
            return [line.strip() for line in f if line.strip()]

    def get_dataset_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            has_ordered_frames=True,
            fps=2,  # NuScenes keyframes are at 2Hz
            has_variable_size_images=False,
            depth_type=DepthType.METRIC,
            is_metric_scale=True,
            data_origin=DataSource.REAL,
        )

    def available_data_fields(self) -> List[DataField]:
        return [
            DataField.KEY,
            DataField.IMAGE_RGB,
            DataField.CAMERA_C2W_TRANSFORM,
            DataField.CAMERA_INTRINSICS,
            DataField.DEPTH,
        ]

    def num_videos(self) -> int:
        return len(self.scene_dirs)

    def num_views(self, video_idx: int) -> int:
        return 1  # Only CAM_FRONT is stored in h5

    def num_frames(self, video_idx: int, view_idx: int = 0) -> int:
        if video_idx not in self._num_frames_cache:
            h5_path = self.scene_dirs[video_idx] / "scene.h5"
            with h5py.File(h5_path, "r") as f:
                # Get number of frames from depths array shape
                self._num_frames_cache[video_idx] = f["depths"].shape[0]
        return self._num_frames_cache[video_idx]

    def _decode_jpeg(self, jpeg_bytes: np.ndarray) -> np.ndarray:
        """Decode JPEG bytes to RGB image array.

        Args:
            jpeg_bytes: 1D uint8 array of JPEG-encoded bytes.

        Returns:
            RGB image array of shape (H, W, 3) with dtype uint8.
        """
        img = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _read_data(
        self,
        video_idx: int,
        frame_idxs: List[int],
        view_idxs: List[int],
        data_fields: List[DataField],
    ) -> dict[DataField, Any]:

        scene_name = self.scene_names[video_idx]
        h5_path = self.scene_dirs[video_idx] / "scene.h5"

        output_dict: dict[DataField, Any] = {DataField.KEY: scene_name}

        with h5py.File(h5_path, "r") as f:
            for data_field in data_fields:
                if data_field == DataField.KEY:
                    pass  # Already included

                elif data_field == DataField.IMAGE_RGB:
                    # Images are stored as JPEG bytes in images/{idx} datasets
                    images = []
                    for frame_idx in frame_idxs:
                        jpeg_bytes = f[f"images/{frame_idx}"][()]
                        img = self._decode_jpeg(jpeg_bytes)
                        # Convert to float [0, 1] and shape (3, H, W)
                        img_tensor = torch.from_numpy(img).float() / 255.0
                        img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
                        images.append(img_tensor)
                    output_dict[data_field] = torch.stack(images, dim=0)

                elif data_field == DataField.CAMERA_C2W_TRANSFORM:
                    # extrinsics_c2w is already camera-to-world
                    c2w = f["extrinsics_c2w"][frame_idxs]  # (N, 4, 4)

                    if self.normalize_first_frame:
                        c2w_frame0 = f["extrinsics_c2w"][0]  # (4, 4)
                        w2c_frame0 = inverse_pose(c2w_frame0)  # (4, 4)
                        c2w = np.einsum("ij,njk->nik", w2c_frame0, c2w)  # (N, 4, 4)

                    output_dict[data_field] = torch.from_numpy(c2w).float()

                elif data_field == DataField.CAMERA_INTRINSICS:
                    # intrinsics is (N, 3, 3), extract [fx, fy, cx, cy]
                    intrinsics = f["intrinsics"][frame_idxs]  # (N, 3, 3)
                    fx = intrinsics[:, 0, 0]
                    fy = intrinsics[:, 1, 1]
                    cx = intrinsics[:, 0, 2]
                    cy = intrinsics[:, 1, 2]
                    intrinsics_vec = np.stack([fx, fy, cx, cy], axis=-1)
                    output_dict[data_field] = torch.from_numpy(intrinsics_vec).float()

                elif data_field == DataField.DEPTH:
                    # depths is (num_frames, H, W), 0.0 indicates invalid/no data
                    depths = f["depths"][frame_idxs]  # (N, H, W)
                    output_dict[data_field] = torch.from_numpy(depths).float()

                else:
                    raise NotImplementedError(f"Data field {data_field} not implemented for NuScenesH5.")

        return output_dict

    def read_video_metadata(self, video_idx: int) -> dict[str, Any]:
        """Read metadata for a specific scene.

        Returns:
            Dictionary containing scene metadata like image dimensions.
        """
        h5_path = self.scene_dirs[video_idx] / "scene.h5"
        with h5py.File(h5_path, "r") as f:
            depths_shape = f["depths"].shape
            return {
                "scene_name": self.scene_names[video_idx],
                "num_frames": depths_shape[0],
                "image_height": depths_shape[1],
                "image_width": depths_shape[2],
            }
