# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
"""MapAnything model wrapper for inference within the dvlt evaluation framework."""

from typing import Any, Dict, List, Optional

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch import nn

from dvlt.common.constants import DataField, PredictionField
from dvlt.model.base import Module
from dvlt.struct.util import extri_intri_to_cameras


try:
    from mapanything.models.mapanything import MapAnything
except ImportError as e:
    raise ImportError(
        "MapAnything baseline package not found. Install it: "
        "`pip install git+https://github.com/facebookresearch/map-anything.git --no-deps "
        "&& pip install uniception==0.1.6`"
    ) from e


logger = get_logger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MapAnythingWrapper(Module):
    """MapAnything inference wrapper.

    Usage:
        python -m dvlt.scripts.test model=mapanything trainer.ckpt_dir=facebook/map-anything
    """

    def __init__(
        self,
        *args,
        infer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        defaults = {
            "use_amp": True,
            "amp_dtype": "bf16",
            "apply_mask": False,
            "mask_edges": False,
            "apply_confidence_mask": False,
        }
        defaults.update(infer_kwargs or {})
        self.infer_kwargs = defaults
        super().__init__(*args, **kwargs)

    def build_model(self) -> nn.Module:
        return nn.Module()

    def load_pretrained(self, pretrained_model_name_or_path: str, **kwargs: Any) -> None:
        logger.info(f"Loading MapAnything from {pretrained_model_name_or_path}")
        self.model = MapAnything.from_pretrained(pretrained_model_name_or_path)

    def _batch_to_views(self, batch: dict) -> List[Dict[str, Any]]:
        """Convert a dvlt batch (images in [0,1]) to MapAnything's view-list format."""
        images = batch[DataField.IMAGES]  # [B, S, C, H, W]
        B, S, C, H, W = images.shape

        mean = torch.tensor(IMAGENET_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)

        views: list[dict[str, Any]] = []
        for s in range(S):
            img_norm = (images[:, s] - mean) / std  # [B, C, H, W]
            view: dict[str, Any] = {
                "img": img_norm,
                "data_norm_type": ["dinov2"] * B,
                "true_shape": [(H, W)] * B,
            }
            if DataField.INTRINSICS in batch:
                view["intrinsics"] = batch[DataField.INTRINSICS][:, s]
            views.append(view)

        return views

    @torch.no_grad()
    def predict(self, batch: dict, accelerator: Accelerator) -> dict:
        views = self._batch_to_views(batch)
        # disable autocast from accelerate because mapanything handles fp16 inference internally
        with torch.autocast("cuda", enabled=False):
            preds_list = self.model.infer(views, **self.infer_kwargs)

        H, W = batch[DataField.IMAGES].shape[-2:]

        world_points = torch.stack([p["pts3d"] for p in preds_list], dim=1)
        depth_z = torch.stack([p["depth_z"].squeeze(-1) for p in preds_list], dim=1)
        conf = torch.stack([p["conf"] for p in preds_list], dim=1)

        extrinsics_c2w = torch.stack([p["camera_poses"] for p in preds_list], dim=1)
        intrinsics = torch.stack([p["intrinsics"] for p in preds_list], dim=1)

        cameras = []
        for extr, intr in zip(extrinsics_c2w, intrinsics):
            cameras.append(extri_intri_to_cameras(extr, intr, (H, W)))

        return {
            PredictionField.WORLD_POINTS: world_points,
            PredictionField.DEPTHS: depth_z,
            PredictionField.DEPTHS_CONF: conf,
            PredictionField.CAMERAS: cameras,
        }

    def train_step(self, *args, **kwargs):
        raise NotImplementedError("MapAnything does not support training within dvlt")
