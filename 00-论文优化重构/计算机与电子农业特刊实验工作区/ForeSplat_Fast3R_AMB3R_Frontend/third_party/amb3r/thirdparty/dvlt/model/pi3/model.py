# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional, Union

import torch
from accelerate import Accelerator

from dvlt.common.constants import DataField, PredictionField
from dvlt.common.projection import intrinsics_from_rays
from dvlt.model.base import Module
from dvlt.struct.util import extri_intri_to_cameras


try:
    from pi3.models.pi3 import Pi3 as Pi3Net
    from pi3.models.pi3x import Pi3X
except ImportError as e:
    raise ImportError(
        "Pi3 baseline package not found. Install it: " "`pip install git+https://github.com/yyfz/Pi3.git --no-deps`"
    ) from e


class Pi3(Module):
    """Pi3 model wrapper for dvlt evaluation.

    Usage example:
        python -m dvlt.scripts.test data=mixed_all model=pi3 trainer.ckpt_dir=yyfz233/Pi3
        python -m dvlt.scripts.test data=mixed_all model=pi3 trainer.ckpt_dir=yyfz233/Pi3X model.use_pi3x=True

    Args:
        pos_type: Positional encoding type for the base Pi3 backbone.
        decoder_size: Decoder size for the base Pi3 backbone.
        use_pi3x: If True, build the Pi3X variant (multi-modal, supports
            optional GT camera/intrinsics conditioning at inference).
        use_cam_cond_at_test: If True (and `use_pi3x=True`), feed
            `batch[EXTRINSICS_C2W]` (c2w) and `batch[INTRINSICS]` to Pi3X's
            forward as `poses` and `intrinsics`. No-op for the base Pi3 model.
            Default False.
    """

    def __init__(
        self,
        *args,
        pos_type: str = "rope100",
        decoder_size: str = "large",
        use_pi3x: bool = False,
        use_cam_cond_at_test: bool = False,
        **kwargs,
    ):
        self.pos_type = pos_type
        self.decoder_size = decoder_size
        self.use_pi3x = use_pi3x
        self.use_cam_cond_at_test = use_cam_cond_at_test
        super().__init__(*args, **kwargs)
        # Pi3 provides weights via HF: https://huggingface.co/yyfz233/Pi3

    # ---------------------------------------------------------------------
    # Module API ----------------------------------------------------------------
    # ---------------------------------------------------------------------

    def load_pretrained(
        self,
        pretrained_model_name_or_path: str,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        model_file: Optional[str] = None,
        strict: bool = False,
        remap: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        super().load_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            use_auth_token=use_auth_token,
            revision=revision,
            model_file="model.safetensors",
            strict=strict,
            remap=remap,
            **kwargs,
        )

    def build_model(self):
        """Instantiate the underlying Pi3 network."""
        if self.use_pi3x:
            return Pi3X()
        else:
            return Pi3Net(pos_type=self.pos_type, decoder_size=self.decoder_size)

    # ---------------------------------------------------------------------
    # Prediction -----------------------------------------------------------
    # ---------------------------------------------------------------------

    def predict(self, batch: dict, accelerator: Accelerator):
        # Pi3 expects images in B x N x 3 x H x W in range [0, 1]
        imgs = batch[DataField.IMAGES]
        device = imgs.device

        # Optional camera-input conditioning (Pi3X only). Pi3X expects c2w
        # poses (per upstream docstring) and pixel-space intrinsics, so we can
        # forward our batch fields directly.
        if self.use_pi3x and self.use_cam_cond_at_test:
            prediction = self.model(
                imgs,
                poses=batch.get(DataField.EXTRINSICS_C2W),
                intrinsics=batch.get(DataField.INTRINSICS),
            )
        else:
            prediction = self.model(imgs)

        # Parse outputs
        world_points = prediction["points"]  # (B, N, H, W, 3)
        local_points = prediction["local_points"]  # (B, N, H, W, 3)
        conf = prediction["conf"].squeeze(-1)  # (B, N, H, W)
        poses_c2w_all = prediction["camera_poses"]  # (B, N, 4, 4) camera->world

        B, N, _, H, W = imgs.shape

        # Depth: use z coordinate of local points
        depth = local_points[..., 2]  # (B, N, H, W)

        # Derive intrinsics from local pointmaps by converting to rays
        # Local points are in camera coordinates (x_cam, y_cam, z_cam)
        # Convert to normalized ray directions for intrinsics estimation
        # Normalize local points to get ray directions
        # Filter out invalid points (z <= 0) before normalizing
        valid_mask = local_points[..., 2] > 1e-6
        rays = local_points.clone()
        # Set invalid points to a backward direction to avoid division issues, rays estimation will handle invalid rays
        rays[~valid_mask] = torch.tensor([0.0, 0.0, -1.0], device=device)
        # Normalize to get ray directions
        rays = rays / (rays.norm(dim=-1, keepdim=True) + 1e-8)

        # Estimate intrinsics from rays: (B, N, H, W, 3) -> (B, N, 3, 3)
        intrinsics = intrinsics_from_rays(rays)

        # Build Cameras objects (one per batch element) with derived intrinsics
        cameras = []
        for extr_single, intr_single in zip(poses_c2w_all, intrinsics):
            cameras.append(extri_intri_to_cameras(extr_single, intr_single, (H, W)))
        # -------------------------------------------------------------------------

        # Convert outputs to expected PredictionField structure
        prediction_field = {
            PredictionField.WORLD_POINTS: world_points,
            PredictionField.DEPTHS: depth,
            PredictionField.DEPTHS_CONF: conf,
            PredictionField.CAMERAS: cameras,
        }
        return prediction_field

    # ---------------------------------------------------------------------
    # Training -------------------------------------------------------------
    # ---------------------------------------------------------------------

    def train_step(self, *args, **kwargs):
        raise NotImplementedError("Pi3 does not support training within dvlt")
