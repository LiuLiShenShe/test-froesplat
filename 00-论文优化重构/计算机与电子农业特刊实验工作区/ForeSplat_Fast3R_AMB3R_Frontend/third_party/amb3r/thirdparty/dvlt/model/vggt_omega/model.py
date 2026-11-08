# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VGGT-Omega model wrapper for dvlt evaluation.

Eval-only wrapper around the upstream `vggt_omega` package
(https://github.com/facebookresearch/vggt-omega).

Notes
-----
VGGT-Omega differs from VGGT in two important ways that this wrapper handles:

1. The pose encoding is 9D (translation, quaternion, FoV_h, FoV_w) and is
   decoded with ``vggt_omega.utils.pose_enc.encoding_to_camera``, which
   returns a 3x4 world-to-camera extrinsic and a 3x3 intrinsic.
2. The official checkpoints were trained at 512 (`VGGT-Omega-1B-512`) or 256
   (`VGGT-Omega-1B-256-Text-Alignment`) with patch size 16.
"""

from typing import Optional

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger

from dvlt.common.amp import force_fp32
from dvlt.common.constants import DataField, PredictionField
from dvlt.common.geometry import depth_to_world_coords_points
from dvlt.common.pose import inverse_pose
from dvlt.model.base import Module
from dvlt.struct.util import extri_intri_to_cameras


try:
    from vggt_omega.models import VGGTOmega as _VGGTOmegaNet
    from vggt_omega.utils.pose_enc import encoding_to_camera
except ImportError as e:
    raise ImportError(
        "VGGT-Omega baseline package not found. Install it: "
        "`pip install git+https://github.com/facebookresearch/vggt-omega.git --no-deps`"
    ) from e


logger = get_logger(__name__)


class VGGTOmega(Module):
    """VGGT-Omega model wrapper for dvlt evaluation.

    Example usage:
        python -m dvlt.scripts.test data=mixed_all model=vggt_omega \\
            trainer.ckpt_dir=facebook/VGGT-Omega
    """

    POSE_CONVENTION = "w2c"

    def __init__(
        self,
        *args,
        camera_head: bool = True,
        depth_head: bool = True,
        enable_alignment: bool = False,
        patch_size: int = 16,
        embed_dim: int = 1024,
        **kwargs,
    ):
        self._enable_camera = camera_head
        self._enable_depth = depth_head
        self._enable_alignment = enable_alignment
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.pose_convention = self.POSE_CONVENTION
        super().__init__(*args, **kwargs)

    def build_model(self):
        return _VGGTOmegaNet(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            enable_camera=self._enable_camera,
            enable_depth=self._enable_depth,
            enable_alignment=self._enable_alignment,
        )

    def load_pretrained(
        self,
        pretrained_model_name_or_path: str,
        *args,
        model_file: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Default HF asset name on `facebook/VGGT-Omega`. The repo ships two
        # checkpoints: `vggt_omega_1b_512.pt` (no text alignment) and
        # `vggt_omega_1b_256_text.pt` (with text alignment).
        if model_file is None:
            model_file = "vggt_omega_1b_256_text.pt" if self._enable_alignment else "vggt_omega_1b_512.pt"
        super().load_pretrained(pretrained_model_name_or_path, *args, model_file=model_file, **kwargs)

    @torch.no_grad()
    def predict(self, batch: dict, accelerator: Accelerator) -> dict:
        images = batch[DataField.IMAGES]
        raw = self.model(images)
        return self._postprocess_predictions(batch, raw)

    @force_fp32
    def _postprocess_predictions(self, batch: dict, raw: dict) -> dict:
        H, W = batch[DataField.IMAGES].shape[-2:]

        pose_enc = raw["pose_enc"]
        extrinsics, intrinsics = encoding_to_camera(pose_enc, (H, W), build_intrinsics=True)
        extrinsics_c2w = inverse_pose(extrinsics) if self.pose_convention == "w2c" else extrinsics

        cameras = [
            extri_intri_to_cameras(extr, intr, (H, W)) for extr, intr in zip(extrinsics_c2w, intrinsics)
        ]

        depths = raw["depth"].squeeze(-1) if "depth" in raw else None
        depths_conf = raw["depth_conf"] if "depth_conf" in raw else None
        world_points, _, _ = depth_to_world_coords_points(depths, extrinsics_c2w, intrinsics)

        predictions = {
            PredictionField.CAMERAS: cameras,
            PredictionField.DEPTHS: depths,
            PredictionField.DEPTHS_CONF: depths_conf,
            PredictionField.WORLD_POINTS: world_points,
        }
        return predictions

    def train_step(self, *args, **kwargs):
        raise NotImplementedError("VGGT-Omega does not support training within dvlt")
