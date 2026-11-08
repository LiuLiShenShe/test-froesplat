# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DTU dataset implementation treating views as video frames.

This module provides a DTU dataset class that treats different camera views
as different video frame indices, where each scan is a video and each view
is a frame in that video.
"""

import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from dvlt.common.pose import inverse_pose
from dvlt.data.sources.datasets import CameraModel, DataField, DatasetMetadata, DataSource, DepthType
from dvlt.data.sources.datasets.base import BaseDataset


class DTU(BaseDataset):
    """DTU dataset implementation treating views as video frames.

    This class implements the DTU dataset for multi-view stereo reconstruction,
    where each scan is treated as a video and different camera views are treated
    as different frames in that video.
    """

    def __init__(
        self,
        root_path: str,
        max_wh: Tuple[int, int] = (1600, 1200),
        light_idx: int = 3,
    ):
        """Initialize the DTU Video dataset.

        Args:
            root_path: Root directory of the DTU dataset.
            max_wh: Maximum width and height for test mode.
            light_idx: Light condition index (0-6 for train, 3 for test).
        """
        super().__init__()

        self.root_path = root_path

        self.max_wh = max_wh
        self.light_idx = light_idx

        # Build metadata
        self.scans = self._load_scans()
        self.view_ids = self._load_view_ids()

        # Cache for video metadata
        self._video_metadata_cache = {}

    def get_dataset_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            has_ordered_frames=False,
            fps=None,
            has_variable_size_images=False,
            depth_type=DepthType.METRIC,
            is_metric_scale=True,
            data_origin=DataSource.REAL,
        )

    def _load_scans(self) -> List[str]:
        """Automatically discover scans from the directory structure.

        Scans are directories that contain images/, cams/, depths/, and binary_masks/ subdirectories.

        Returns:
            A list of scan names.
        """
        if not os.path.exists(self.root_path):
            raise FileNotFoundError(f"Root path does not exist: {self.root_path}")

        scans = []
        for item in os.listdir(self.root_path):
            scan_path = os.path.join(self.root_path, item)
            if os.path.isdir(scan_path):
                # Check if this directory has the required subdirectories
                required_dirs = ["images", "cams", "depths", "binary_masks"]
                if all(os.path.exists(os.path.join(scan_path, d)) for d in required_dirs):
                    scans.append(item)

        scans = sorted(scans)  # Sort for consistent ordering

        if len(scans) == 0:
            raise ValueError(
                f"No valid scans found in {self.root_path}. Each scan must contain images/, cams/, depths/, and binary_masks/ subdirectories."
            )

        print(f"DTU: {len(scans)} scans (videos)")
        return scans

    def _load_view_ids(self) -> List[int]:
        """Load the list of view IDs from the images directory.

        Returns:
            A list of view IDs (0-indexed, matching image file indices).
        """
        # Load all images from the first scan's images directory
        if len(self.scans) == 0:
            raise ValueError("No scans found in dataset")

        scan = self.scans[0]
        images_dir = os.path.join(self.root_path, scan, "images")

        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        # Get all image files and sort them
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".JPG", ".PNG"))])

        if len(image_files) == 0:
            raise ValueError(f"No image files found in {images_dir}")

        # Extract view IDs from filenames (assuming format: 00000000.jpg, 00000001.jpg, etc.)
        view_ids = []
        for img_file in image_files:
            # Extract the numeric part from filename (e.g., "00000000.jpg" -> 0)
            base_name = os.path.splitext(img_file)[0]
            try:
                view_id = int(base_name)
                view_ids.append(view_id)
            except ValueError:
                # If filename doesn't match expected format, use index
                continue

        # If we couldn't parse any IDs, fall back to sequential indices
        if len(view_ids) == 0:
            view_ids = list(range(len(image_files)))

        # Sort to ensure consistent ordering
        view_ids = sorted(view_ids)

        print(f"DTU: {len(view_ids)} views (frames) per video (loaded from images directory)")
        return view_ids

    def __len__(self) -> int:
        """Return the number of videos in the dataset.

        Returns:
            Number of videos (scans) in the dataset.
        """
        return len(self.scans)

    def available_data_fields(self) -> List[DataField]:
        """Return a list of available data fields in the dataset.

        Returns:
            A list of available data fields.
        """
        fields = [
            DataField.IMAGE_RGB,
            DataField.DEPTH,
            DataField.CAMERA_C2W_TRANSFORM,
            DataField.CAMERA_INTRINSICS,
            DataField.CAMERA_MODEL,
        ]

        return fields

    def num_views(self, video_idx: int) -> int:
        """Return the number of views in the video.

        Args:
            video_idx: Index of the video.

        Returns:
            1, treat different views as different frames.
        """
        return 1

    def num_videos(self) -> int:
        """Return the number of videos in the dataset.

        Returns:
            Number of videos (scans).
        """
        return len(self.scans)

    def num_frames(self, video_idx: int, view_idx: int = 0) -> int:
        """Return the number of frames in the video.

        Args:
            video_idx: Index of the video.
            view_idx: Index of the view (not used).

        Returns:
            Number of frames (views) in the video.
        """
        return len(self.view_ids)

    def _get_scan_name(self, video_idx: int) -> str:
        """Get the scan name for a given video index.

        Args:
            video_idx: Index of the video.

        Returns:
            The scan name.
        """
        if video_idx < 0 or video_idx >= len(self.scans):
            raise ValueError(
                f"Video index {video_idx} is out of range. Maximum valid video index is {len(self.scans)-1}."
            )
        return self.scans[video_idx]

    def read_video_metadata(self, video_idx: int) -> Dict[str, Any]:
        """Read metadata of the video.

        Args:
            video_idx: Index of the video.

        Returns:
            A dictionary containing metadata of the video.
        """
        if video_idx in self._video_metadata_cache:
            return self._video_metadata_cache[video_idx]

        scan = self.scans[video_idx]

        metadata = {
            "scan": scan,
            "num_frames": len(self.view_ids),
            "frame_ids": self.view_ids.copy(),
        }

        self._video_metadata_cache[video_idx] = metadata
        return metadata

    def read_view_metadata(self, video_idx: int, view_idx: int) -> Dict[str, Any]:
        """Read metadata of the view.

        Args:
            video_idx: Index of the video.
            view_idx: Index of the view (always 0 for DTU Video).

        Returns:
            A dictionary containing metadata of the view.
        """
        scan = self.scans[video_idx]

        return {
            "scan": scan,
            "num_frames": len(self.view_ids),
            "light_idx": self.light_idx,
        }

    def _read_data(
        self,
        video_idx: int,
        frame_idxs: List[int],
        view_idxs: List[int],
        data_fields: List[DataField],
    ) -> Dict[DataField, Any]:
        """Read data from the dataset.

        Args:
            video_idx: Index of the video (scan).
            frame_idxs: List of frame indices (view indices).
            view_idxs: List of view indices (always [0] for DTU Video).
            data_fields: List of data fields to read.

        Returns:
            A dictionary mapping data fields to their values.
        """
        scan = self.scans[video_idx]

        # Initialize result dictionary
        result = {DataField.KEY: scan}

        # Read images
        if DataField.IMAGE_RGB in data_fields:
            images = []
            for frame_idx in frame_idxs:
                view_id = self.view_ids[frame_idx]
                # MVSnet format: images are in scan/images/{:0>8}.jpg
                img_filename = os.path.join(self.root_path, "{}/images/{:0>8}.jpg".format(scan, view_id))

                img = self._read_img(img_filename)
                images.append(img)

            # Stack images: (num_frames, C, H, W)
            result[DataField.IMAGE_RGB] = torch.from_numpy(np.stack(images)).float()

        # Read camera parameters
        if DataField.CAMERA_C2W_TRANSFORM in data_fields or DataField.CAMERA_INTRINSICS in data_fields:
            camera_c2w = []
            camera_intrinsics = []

            for frame_idx in frame_idxs:
                view_id = self.view_ids[frame_idx]

                # MVSnet format: cameras are in scan/cams/{:0>8}_cam.txt
                proj_mat_filename = os.path.join(self.root_path, "{}/cams/{:0>8}_cam.txt".format(scan, view_id))

                intrinsics, extrinsics = self._read_cam_file(proj_mat_filename)

                # MVSnet format stores world-to-camera (W2C); invert to camera-to-world (C2W).
                # Translation and depth are kept in millimetres.
                c2w = inverse_pose(extrinsics)

                camera_c2w.append(c2w)

                # Convert intrinsics to [fx, fy, cx, cy] format
                fx = intrinsics[0, 0]
                fy = intrinsics[1, 1]
                cx = intrinsics[0, 2]
                cy = intrinsics[1, 2]
                camera_intrinsics.append([fx, fy, cx, cy])

            if DataField.CAMERA_C2W_TRANSFORM in data_fields:
                result[DataField.CAMERA_C2W_TRANSFORM] = torch.from_numpy(np.stack(camera_c2w)).float()

            if DataField.CAMERA_INTRINSICS in data_fields:
                result[DataField.CAMERA_INTRINSICS] = torch.from_numpy(np.stack(camera_intrinsics)).float()

        # Set camera model
        if DataField.CAMERA_MODEL in data_fields:
            result[DataField.CAMERA_MODEL] = torch.tensor(
                [CameraModel.PINHOLE.value] * len(frame_idxs), dtype=torch.uint8
            )

        # Read depth maps
        if DataField.DEPTH in data_fields:
            depth_maps = []
            for frame_idx in frame_idxs:
                view_id = self.view_ids[frame_idx]
                # MVSnet format: depths are in scan/depths/{:0>8}.npy, in millimetres.
                depth_filename = os.path.join(self.root_path, "{}/depths/{:0>8}.npy".format(scan, view_id))
                depth = np.load(depth_filename)

                mask_filename = os.path.join(self.root_path, "{}/binary_masks/{:0>8}.png".format(scan, view_id))
                mask = cv2.imread(mask_filename, cv2.IMREAD_UNCHANGED) / 255.0
                mask = mask.astype(np.float32)

                mask[mask > 0.5] = 1.0
                mask[mask < 0.5] = 0.0

                mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
                kernel = np.ones((10, 10), np.uint8)  # Define the erosion kernel
                mask = cv2.erode(mask, kernel, iterations=1)
                depth = depth * mask
                depth_maps.append(depth)

            result[DataField.DEPTH] = torch.from_numpy(np.stack(depth_maps)).float()

        return result

    def _read_img(self, filename: str) -> np.ndarray:
        """Read an image from a file.

        Args:
            filename: Path to the image file.

        Returns:
            The image as a numpy array.
        """
        img = Image.open(filename)
        # Scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.0
        # H, W, C -> C, H, W
        np_img = np_img.transpose(2, 0, 1)
        return np_img

    def _read_cam_file(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read camera parameters from a file using MVSnet format parsing.

        This function uses the same parsing logic as the original dtu_load.py
        to ensure compatibility with MVSnet camera file format.

        Args:
            filename: Path to the camera file.

        Returns:
            A tuple containing (intrinsics 3x3, extrinsics 4x4).
        """
        with open(filename, "r") as f:
            cam = np.zeros((2, 4, 4))
            words = f.read().split()

            # read extrinsic (4x4 matrix from words[1:17])
            for i in range(0, 4):
                for j in range(0, 4):
                    extrinsic_index = 4 * i + j + 1
                    cam[0][i][j] = words[extrinsic_index]

            # read intrinsic (3x3 matrix from words[18:27])
            for i in range(0, 3):
                for j in range(0, 3):
                    intrinsic_index = 3 * i + j + 18
                    cam[1][i][j] = words[intrinsic_index]

            # Handle depth interval information (4th row of intrinsic matrix)
            if len(words) == 29:
                cam[1][3][0] = words[27]
                cam[1][3][1] = float(words[28])
                cam[1][3][2] = 192
                cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
            elif len(words) == 30:
                cam[1][3][0] = words[27]
                cam[1][3][1] = float(words[28])
                cam[1][3][2] = words[29]
                cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
            elif len(words) == 31:
                cam[1][3][0] = words[27]
                cam[1][3][1] = float(words[28])
                cam[1][3][2] = words[29]
                cam[1][3][3] = words[30]
            else:
                cam[1][3][0] = 0
                cam[1][3][1] = 0
                cam[1][3][2] = 0
                cam[1][3][3] = 0

            extrinsic = cam[0].astype(np.float32)
            intrinsic_full = cam[1].astype(np.float32)

            # Extract only the 3x3 intrinsic matrix
            intrinsics = intrinsic_full[:3, :3]

            return intrinsics, extrinsic
