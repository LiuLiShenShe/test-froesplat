# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Literal, NamedTuple

import cv2
import numpy as np
import torch

from dvlt.data.sources.datasets import CameraModel, DataField, DatasetMetadata, DataSource, DepthType

from .base import BaseDataset


class VideoData(NamedTuple):
    scene: str
    seq_name: str
    split: Literal["train", "test"]
    num_frames: int


class SevenScenes(BaseDataset):
    """Minimal reader for the Microsoft 7-Scenes dataset.

    Folder layout expected by this reader (matching the original release):

    ```
    {scene}/
        TrainSplit.txt
        TestSplit.txt
        seq-01/
            frame-000000.color.png
            frame-000000.depth.png
            frame-000000.pose.txt
            ...
        seq-02/
            ...
    ```

    Poses are provided as 4x4 camera-to-world matrices in metres.
    Depth images are uint16 where the stored value is depth in millimetres.
    RGB images are 640 x 480.
    """

    FOCAL_LENGTH = 525.0
    IMG_W, IMG_H = 640, 480

    INFOS = {
        "train": {
            "chess": [1, 2, 4, 6],
            "fire": [1, 2],
            "heads": [2],
            "office": [1, 3, 4, 5, 8, 10],
            "pumpkin": [2, 3, 6, 8],
            "redkitchen": [1, 2, 5, 7, 8, 11, 13],
            "stairs": [2, 3, 5, 6],
        },
        "test": {
            "chess": [3, 5],
            "fire": [3, 4],
            "heads": [1],
            "office": [2, 6, 7, 9],
            "pumpkin": [1, 7],
            "redkitchen": [3, 4, 6, 12, 14],
            "stairs": [1, 4],
        },
    }

    def __init__(
        self,
        data_root: str,
        split: Literal["train", "test"] = "train",
        scenes: list[str] | None = None,
        depth_source: Literal["proj", "raw"] = "proj",
        max_depth: float = 10.0,  # we follow pi3 https://github.com/yyfz/Pi3/blob/evaluation/datasets/sevenscenes.py#L161
        min_depth: float = 0.1,  # filter some very small depth values that dominate Abs/SqRel metrics
    ):
        super().__init__()

        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"{self.data_root} does not exist")

        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.split = split

        self._intrinsics = np.array(
            [self.FOCAL_LENGTH, self.FOCAL_LENGTH, self.IMG_W / 2.0, self.IMG_H / 2.0], dtype=np.float32
        )

        self._videos: list[VideoData] = list(self._discover_videos(self.data_root, split, scenes))

        if depth_source not in ("proj", "raw"):
            raise ValueError("depth_source must be 'proj' or 'raw'")
        self._depth_source = depth_source
        self.max_depth = max_depth
        self.min_depth = min_depth

    # ---------------------------------------------------------------------
    # Dataset interface helpers
    # ---------------------------------------------------------------------
    def _discover_videos(self, root: Path, split: str, scenes: list[str] | None = None) -> List[VideoData]:
        videos: list[VideoData] = []

        candidate_scenes = scenes if scenes is not None else [p.name for p in root.iterdir() if p.is_dir()]

        for scene in sorted(candidate_scenes):
            scene_dir = root / scene
            if not scene_dir.exists():
                raise FileNotFoundError(scene_dir)

            # Use INFOS to determine which sequences belong to the requested split
            if scene in self.INFOS[split]:
                # Get sequence numbers for this scene and split
                seq_numbers = self.INFOS[split][scene]
                # Convert numbers to sequence names (e.g., 1 -> "seq-01")
                seq_names = [f"seq-{num:02d}" for num in seq_numbers]
            else:
                # Scene not in INFOS for this split, skip it
                continue

            for seq in seq_names:
                seq_dir = scene_dir / seq
                if not seq_dir.exists():
                    continue
                # Count frames by colour images
                num_frames = len(list(seq_dir.glob("*.color.png")))
                videos.append(VideoData(scene, seq, split, num_frames))

        if not videos:
            raise RuntimeError("No videos found for SevenScenes")
        return videos

    # ------------------------------------------------------------------
    # Required BaseDataset API implementations
    # ------------------------------------------------------------------
    def get_dataset_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            has_ordered_frames=True,
            fps=30,  # original data captured ~30 Hz
            has_variable_size_images=False,
            depth_type=DepthType.METRIC,
            is_metric_scale=True,
            data_origin=DataSource.REAL,
        )

    def available_data_fields(self) -> List[DataField]:
        fields = [
            DataField.IMAGE_RGB,
            DataField.DEPTH,
            DataField.CAMERA_C2W_TRANSFORM,
            DataField.CAMERA_INTRINSICS,
            DataField.CAMERA_MODEL,
        ]
        return fields

    # -------------------------------
    # Dimensions helpers
    # -------------------------------
    def num_videos(self) -> int:
        return len(self._videos)

    def num_views(self, video_idx: int) -> int:  # monocular
        return 1

    def num_frames(self, video_idx: int, view_idx: int = 0) -> int:
        return self._videos[video_idx].num_frames

    # ------------------------------------------------------------------
    # Internal data loading helpers
    # ------------------------------------------------------------------

    def _seq_dir(self, video_data: VideoData) -> Path:
        return self.data_root / video_data.scene / video_data.seq_name

    def _rgb_path(self, video_data: VideoData, frame_idx: int) -> Path:
        return self._seq_dir(video_data) / f"frame-{frame_idx:06d}.color.png"

    def _depth_path(self, video_data: VideoData, frame_idx: int) -> Path:
        suffix = "depth.proj.png" if self._depth_source == "proj" else "depth.png"
        return self._seq_dir(video_data) / f"frame-{frame_idx:06d}.{suffix}"

    def _pose_path(self, video_data: VideoData, frame_idx: int) -> Path:
        return self._seq_dir(video_data) / f"frame-{frame_idx:06d}.pose.txt"

    # -----------------------------
    def _load_rgb(self, vd: VideoData, frame_idxs: list[int]) -> list[np.ndarray]:
        rgbs = []
        for i in frame_idxs:
            rgb = cv2.cvtColor(cv2.imread(str(self._rgb_path(vd, i)), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            rgbs.append(rgb.astype(np.uint8))
        return rgbs

    def _load_depth(self, vd: VideoData, frame_idxs: list[int]) -> list[np.ndarray]:
        depths = []
        for i in frame_idxs:
            depth = cv2.imread(str(self._depth_path(vd, i)), cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth = np.nan_to_num(depth.astype(np.float32), 0.0) / 1000.0
            depth[depth > self.max_depth] = 0
            depth[depth < self.min_depth] = 0
            depths.append(depth)
        return depths

    def _load_pose(self, vd: VideoData, frame_idxs: list[int]) -> np.ndarray:
        pose_mats = []
        for i in frame_idxs:
            pose_txt = np.loadtxt(self._pose_path(vd, i), dtype=np.float32).reshape(4, 4)
            pose_mats.append(pose_txt)
        return np.stack(pose_mats)

    # ------------------------------------------------------------------
    # Data access entry point
    # ------------------------------------------------------------------
    def _read_data(
        self,
        video_idx: int,
        frame_idxs: List[int],
        view_idxs: List[int],
        data_fields: List[DataField],
    ) -> Dict[DataField, Any]:
        if not all(v == 0 for v in view_idxs):
            raise NotImplementedError("SevenScenes only has a single view")

        vd = self._videos[video_idx]
        result: Dict[DataField, Any] = {}

        # Load depth first if requested, to get dimensions for RGB resizing
        depths = None
        if DataField.DEPTH in data_fields:
            depths = self._load_depth(vd, frame_idxs)

        for field in data_fields:
            if field == DataField.IMAGE_RGB:
                rgbs = self._load_rgb(vd, frame_idxs)
                # Resize RGB to match depth dimensions if depth is available
                if depths is not None and len(depths) > 0:
                    depth_h, depth_w = depths[0].shape[:2]
                    rgbs = [cv2.resize(rgb, (depth_w, depth_h), interpolation=cv2.INTER_LANCZOS4) for rgb in rgbs]
                result[field] = torch.stack([torch.tensor(r / 255.0, dtype=torch.float32) for r in rgbs]).permute(
                    0, 3, 1, 2
                )

            elif field == DataField.DEPTH:
                if depths is None:
                    depths = self._load_depth(vd, frame_idxs)
                result[field] = torch.stack([torch.tensor(d, dtype=torch.float32) for d in depths])

            elif field == DataField.CAMERA_C2W_TRANSFORM:
                poses = self._load_pose(vd, frame_idxs)
                result[field] = torch.from_numpy(poses)

            elif field == DataField.CAMERA_INTRINSICS:
                result[field] = torch.tensor(self._intrinsics.copy())[None].repeat(len(frame_idxs), 1)

            elif field == DataField.CAMERA_MODEL:
                result[field] = torch.tensor([CameraModel.PINHOLE.value] * len(frame_idxs), dtype=torch.uint8)

        # Unique key identifying the sample sequence
        result[DataField.KEY] = f"{vd.scene}_{vd.seq_name}"
        return result

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def __len__(self):
        return self.num_videos()
