# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class DepthType(Enum):
    METRIC = 0


class DataSource(Enum):
    REAL = 0
    SYNTHETIC = 1


@dataclass(frozen=True)
class DatasetMetadata:
    has_ordered_frames: bool
    fps: int | None
    has_variable_size_images: bool
    is_metric_scale: bool | None
    depth_type: DepthType | None
    data_origin: DataSource

    def __post_init__(self):
        if self.fps is not None and self.fps <= 0:
            raise ValueError("FPS must be positive.")
        if not self.has_ordered_frames and self.fps is not None:
            raise ValueError("FPS can only be set for ordered datasets.")
        if self.is_metric_scale is not None and self.depth_type is not None:
            if (self.depth_type == DepthType.METRIC) != self.is_metric_scale:
                raise ValueError(
                    f"Depth type (={self.depth_type}) and scene `metric` annotation (={self.is_metric_scale})"
                    " must be consistent"
                )


class CameraModel(Enum):
    PINHOLE = 1


class DataField(Enum):
    # str, unique per-video identifier (constant across frame/view indexing).
    KEY = "__key__"
    # [B, 3, H, W] (or list of [3, H, W]), float32 RGB in [0, 1].
    IMAGE_RGB = "image_rgb"
    # [B, 4, 4] float32 camera-to-world (OpenCV RDF convention).
    CAMERA_C2W_TRANSFORM = "camera_c2w_transform"
    # [B, 4] float32 pinhole intrinsics [fx, fy, cx, cy].
    CAMERA_INTRINSICS = "camera_intrinsics"
    # [B] uint8 camera model index.
    CAMERA_MODEL = "camera_model"
    # [B, H, W] (or list of [H, W]), float32 depth map.
    DEPTH = "depth"
    # [B, H, W] bool background mask (True where background); used by the adapter's depth_filter_background.
    BACKGROUND_MASK = "background_mask"
