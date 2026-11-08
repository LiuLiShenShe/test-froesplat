# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
"""VGGT model wrapper for dvlt evaluation.

Eval-only wrapper around the upstream `vggt` package
(https://github.com/facebookresearch/vggt). Training was supported by an
older internal version of this wrapper and has been removed for the public
release.
"""

from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger

from dvlt.common.amp import force_fp32
from dvlt.common.constants import DataField, PredictionField
from dvlt.common.geometry import depth_to_world_coords_points
from dvlt.common.pose import inverse_pose
from dvlt.model.base import Module
from dvlt.model_components import pose_enc_to_extri_intri
from dvlt.struct.util import extri_intri_to_cameras


try:
    from vggt.models.vggt import VGGT as _VGGTNet
except ImportError as e:
    raise ImportError(
        "VGGT baseline package not found. Install it: "
        "`pip install git+https://github.com/facebookresearch/vggt.git --no-deps`"
    ) from e


logger = get_logger(__name__)


class VGGT(Module):
    """VGGT model wrapper for dvlt evaluation.

    Example usage:
        python -m dvlt.scripts.test data=mixed_all model=vggt trainer.ckpt_dir=facebook/VGGT-1B
    """

    POSE_CONVENTION = "w2c"

    def __init__(
        self,
        *args,
        camera_head: bool = True,
        depth_head: bool = True,
        point_head: bool = True,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        dpt_chunk_size: Optional[int] = 24,
        # Accepted-but-ignored kwargs for backward compat with training-era configs.
        aggregator_config: Optional[Dict[str, Any]] = None,
        gradient_checkpointing_config: Optional[Dict[str, Any]] = None,
        conditioning_probs: Optional[Dict[str, float]] = None,
        loss: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self._enable_camera = camera_head
        self._enable_depth = depth_head
        self._enable_point = point_head
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.dpt_chunk_size = dpt_chunk_size
        self.pose_convention = self.POSE_CONVENTION
        super().__init__(*args, **kwargs)

    def build_model(self):
        return _VGGTNet(
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            enable_camera=self._enable_camera,
            enable_depth=self._enable_depth,
            enable_point=self._enable_point,
            enable_track=False,
        )

    def load_pretrained(
        self,
        pretrained_model_name_or_path: str,
        *args,
        filter: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        # Upstream `facebook/VGGT-1B` ships with `track_head.*` weights, but we
        # build the network with `enable_track=False`, so drop them before load.
        filter = list(filter or [])
        if not any("track_head" in p for p in filter):
            filter.append(r"^track_head\.")
        super().load_pretrained(pretrained_model_name_or_path, *args, filter=filter, **kwargs)

    @torch.no_grad()
    def predict(self, batch: dict, accelerator: Accelerator) -> dict:
        images = batch[DataField.IMAGES]
        raw = self.model(images)
        return self._postprocess_predictions(batch, raw)

    @force_fp32
    def _postprocess_predictions(self, batch: dict, raw: dict) -> dict:
        H, W = batch[DataField.IMAGES].shape[-2:]

        pose_enc = raw["pose_enc"] if "pose_enc" in raw else raw["pose_enc_list"][-1]
        extrinsics, intrinsics = pose_enc_to_extri_intri(pose_enc, (H, W))
        extrinsics_c2w = inverse_pose(extrinsics) if self.pose_convention == "w2c" else extrinsics

        cameras = [
            extri_intri_to_cameras(extr, intr, (H, W)) for extr, intr in zip(extrinsics_c2w, intrinsics)
        ]

        depths = raw["depth"].squeeze(-1) if "depth" in raw else None
        depths_conf = raw["depth_conf"].squeeze(-1) if "depth_conf" in raw else None
        world_points, _, _ = depth_to_world_coords_points(depths, extrinsics_c2w, intrinsics)

        predictions = {
            PredictionField.CAMERAS: cameras,
            PredictionField.DEPTHS: depths,
            PredictionField.DEPTHS_CONF: depths_conf,
            PredictionField.WORLD_POINTS: world_points,
        }

        if "world_points" in raw:
            predictions[PredictionField.WORLD_POINTS_DIRECT] = raw["world_points"]
            predictions[PredictionField.WORLD_POINTS_DIRECT_CONF] = raw["world_points_conf"].squeeze(-1)

        return predictions

    def train_step(self, *args, **kwargs):
        raise NotImplementedError("VGGT does not support training within dvlt")
