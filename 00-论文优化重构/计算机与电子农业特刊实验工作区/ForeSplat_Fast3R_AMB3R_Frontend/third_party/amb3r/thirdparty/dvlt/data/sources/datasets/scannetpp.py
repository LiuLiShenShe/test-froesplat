# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data parser for ScanNet++ datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from dvlt.common.io import read_depth

from . import CameraModel, DataField, DatasetMetadata, DataSource, DepthType
from .base import BaseDataset


def load_from_json(path: Path) -> Dict[str, Any]:
    """Load data from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Data loaded from the JSON file.
    """
    with open(path, "r") as f:
        return json.load(f)


def load_split_file(split_file: Path) -> List[str]:
    """Load a split file containing sequence names.

    Args:
        split_file: Path to the split file.

    Returns:
        List of sequence names.
    """
    with open(split_file, "r") as f:
        return [line.strip() for line in f.readlines()]


class ScanNetpp(BaseDataset):
    """ScanNet++ Dataset implementation following BaseDataset format.

    ScanNet++ dataset (https://kaldir.vc.in.tum.de/scannetpp/) is a real-world 3D indoor dataset
    for semantics understanding and novel view synthesis.

    Default structure of the directory:
    .. code-block:: text
        root/
        ├── SCENE_ID0
            ├── dslr
                ├── resized_images
                ├── nerfstudio/transforms.json
        ├── SCENE_ID1/
        ...
    """

    def __init__(
        self,
        data_root: str,
        split_file: Optional[str] = None,
        images_dir: str = "dslr/resized_images",
        depth_dir: str = "dslr/resized_depth",
        transforms_path: str = "dslr/nerfstudio/transforms.json",
    ):
        """Initialize the ScanNet++ dataset.

        Args:
            data_root: Root directory containing all scenes.
            split_file: Path to the split file containing sequence names.
            images_dir: Relative path to the images directory.
            depth_dir: Relative path to the depth directory.
            transforms_path: Relative path to the transforms.json file.
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.images_dir = Path(images_dir)
        self.depth_dir = Path(depth_dir)
        self.transforms_path = Path(transforms_path)

        # Load sequence names from split file
        self.sequences = load_split_file(Path(split_file)) if split_file is not None else os.listdir(self.data_root)

        # Initialize scene data cache
        self._scene_cache = {}
        self._metadata_cache = {}
        self._poses_cache = {}

    def get_dataset_metadata(self) -> DatasetMetadata:
        """Get dataset metadata.

        This is for the DSLR image subset.
        """
        return DatasetMetadata(
            has_ordered_frames=False,
            fps=None,
            has_variable_size_images=False,
            depth_type=DepthType.METRIC,
            is_metric_scale=True,
            data_origin=DataSource.REAL,
        )

    def _load_scene(self, sequence_name: str) -> None:
        """Load a single scene's data if not already loaded.

        Args:
            sequence_name: Name of the scene to load.
        """
        if sequence_name in self._scene_cache:
            return

        sequence_dir = self.data_root / sequence_name
        if not sequence_dir.exists():
            raise ValueError(f"Scene directory {sequence_dir} does not exist")

        # Check if transforms.json exists
        transforms_file = sequence_dir / self.transforms_path
        if not transforms_file.exists():
            raise ValueError(f"Transforms file {transforms_file} does not exist")

        # Load metadata
        meta = load_from_json(transforms_file)
        data_dir = sequence_dir / self.images_dir
        depth_dir = sequence_dir / self.depth_dir

        # Process frames
        frames = meta["frames"] + meta["test_frames"]
        test_frames = [f["file_path"] for f in meta["test_frames"]]
        frames.sort(key=lambda x: x["file_path"])

        # Extract camera parameters
        fx = float(meta["fl_x"])
        fy = float(meta["fl_y"])
        cx = float(meta["cx"])
        cy = float(meta["cy"])
        height = int(meta["h"])
        width = int(meta["w"])

        # Undistorted DSLR captures are pinhole.
        camera_model = CameraModel.PINHOLE

        # Process poses
        poses = torch.from_numpy(np.array([frame["transform_matrix"] for frame in frames]).astype(np.float32))
        # Convert from OpenGL to OpenCV
        poses[:, 2, :] *= -1
        poses = poses[:, np.array([1, 0, 2, 3]), :]
        poses[:, 0:3, 1:3] *= -1

        # Split indices
        train_indices = []
        eval_indices = []
        for idx, frame in enumerate(frames):
            if frame["file_path"] in test_frames:
                eval_indices.append(idx)
            else:
                train_indices.append(idx)

        # Cache scene data
        self._scene_cache[sequence_name] = {
            "frames": frames,
            "data_dir": data_dir,
            "depth_dir": depth_dir,
            "train_indices": train_indices,
            "eval_indices": eval_indices,
        }

        self._metadata_cache[sequence_name] = {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "height": height,
            "width": width,
            "camera_model": camera_model,
        }

        self._poses_cache[sequence_name] = poses

    def available_data_fields(self) -> List[DataField]:
        """Return a list of available data fields in the dataset."""
        fields = [
            DataField.IMAGE_RGB,
            DataField.CAMERA_C2W_TRANSFORM,
            DataField.CAMERA_INTRINSICS,
            DataField.CAMERA_MODEL,
            DataField.DEPTH,
        ]
        return fields

    def num_videos(self) -> int:
        """Returns the number of videos in the dataset."""
        return len(self.sequences)

    def num_views(self, video_idx: int) -> int:
        """Returns the number of views in the video.

        Args:
            video_idx: Index of the video.

        Returns:
            Number of views in the video.
        """
        return 1

    def num_frames(self, video_idx: int, view_idx: int = 0) -> int:
        """Returns the number of frames in the given view.

        Args:
            video_idx: Index of the video.
            view_idx: Index of the view.

        Returns:
            Number of frames in the given view.
        """
        sequence_name = self.sequences[video_idx]
        self._load_scene(sequence_name)
        return len(self._scene_cache[sequence_name]["frames"])

    def read_video_metadata(self, video_idx: int) -> Dict[str, Any]:
        """Read metadata of the video.

        Args:
            video_idx: Index of the video.

        Returns:
            A dictionary containing metadata of the video.
        """
        sequence_name = self.sequences[video_idx]
        self._load_scene(sequence_name)
        metadata = self._metadata_cache[sequence_name]
        return {
            "camera_model": metadata["camera_model"],
            "fx": metadata["fx"],
            "fy": metadata["fy"],
            "cx": metadata["cx"],
            "cy": metadata["cy"],
            "height": metadata["height"],
            "width": metadata["width"],
        }

    def read_view_metadata(self, video_idx: int, view_idx: int) -> Dict[str, Any]:
        """Read metadata of the view.

        Args:
            video_idx: Index of the video.
            view_idx: Index of the view.

        Returns:
            A dictionary containing metadata of the view.
        """
        return self.read_video_metadata(video_idx)

    def _read_data(
        self,
        video_idx: int,
        frame_idxs: List[int],
        view_idxs: List[int],
        data_fields: List[DataField],
    ) -> Dict[DataField, Any]:
        """Read data from the dataset.

        Args:
            video_idx: Index of the video.
            frame_idxs: List of frame indices.
            view_idxs: List of view indices.
            data_fields: List of data fields to read.

        Returns:
            A dictionary mapping data fields to their values.
        """
        sequence_name = self.sequences[video_idx]
        self._load_scene(sequence_name)
        scene_data = self._scene_cache[sequence_name]
        metadata = self._metadata_cache[sequence_name]

        result: dict[DataField, Any] = {DataField.KEY: sequence_name}

        for field in data_fields:
            if field == DataField.IMAGE_RGB:
                images = []
                for frame_idx in frame_idxs:
                    frame = scene_data["frames"][frame_idx]
                    filepath = Path(frame["file_path"])
                    image_path = scene_data["data_dir"] / filepath
                    image = Image.open(image_path)
                    # Convert to tensor directly
                    image = torch.from_numpy(np.array(image)).float() / 255.0
                    images.append(image)
                result[field] = torch.stack(images).permute(0, 3, 1, 2).contiguous()

            elif field == DataField.CAMERA_C2W_TRANSFORM:
                result[field] = self._poses_cache[sequence_name][frame_idxs]

            elif field == DataField.CAMERA_INTRINSICS:
                intrinsics = torch.tensor(
                    [[metadata["fx"], metadata["fy"], metadata["cx"], metadata["cy"]]] * len(frame_idxs),
                    dtype=torch.float32,
                )
                result[field] = intrinsics

            elif field == DataField.CAMERA_MODEL:
                result[field] = torch.tensor([metadata["camera_model"].value] * len(frame_idxs), dtype=torch.uint8)

            elif field == DataField.DEPTH:
                depths = []
                for frame_idx in frame_idxs:
                    frame = scene_data["frames"][frame_idx]
                    filepath = Path(frame["file_path"]).with_suffix(".png")
                    depth_path = scene_data["depth_dir"] / filepath
                    depth = torch.from_numpy(read_depth(str(depth_path)))
                    depths.append(depth)
                result[field] = torch.stack(depths)

        return result

    def __len__(self) -> int:
        """Returns the total number of videos in the dataset."""
        return self.num_videos()
