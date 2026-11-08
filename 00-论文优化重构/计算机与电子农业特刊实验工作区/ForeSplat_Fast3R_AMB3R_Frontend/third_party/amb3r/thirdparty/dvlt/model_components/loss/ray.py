# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ray loss for world-space ray prediction."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn

from dvlt.common.constants import DataField
from dvlt.model_components.loss.base import (
    BATCH_NORMALIZATION_SCALE_KEY,
    PRED_NORMALIZATION_SCALE_KEY,
)
from dvlt.model_components.loss.util import check_and_fix_inf_nan


class RayLoss(nn.Module):
    """Simple L1 loss for world-space ray predictions.

    Computes L1 loss between predicted rays and ground truth rays.
    GT rays should be pre-computed and stored in batch[DataField.WORLD_RAYS].

    Rays are 6-channel: [direction (3), origin (3)]
    """

    def __init__(
        self,
        pred_key: str = "rays",
        max_loss: float = 100.0,
    ):
        """Initialize RayLoss.

        Args:
            pred_key: Key in predictions for predicted rays (B, S, H, W, 6).
            max_loss: Maximum loss value for robustness (clamp).
        """
        super().__init__()
        self.pred_key = pred_key
        self.max_loss = max_loss

    def check_inputs(self, predictions: Dict[str, Any], batch: Dict[str, Any]) -> bool:
        """Check if required inputs are present."""
        if self.pred_key not in predictions or predictions[self.pred_key] is None:
            return False
        if DataField.WORLD_RAYS not in batch or batch[DataField.WORLD_RAYS] is None:
            return False
        return True

    def forward(
        self,
        predictions: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute ray loss.

        Args:
            predictions: Dict containing predicted rays.
            batch: Dict containing GT rays (DataField.WORLD_RAYS).

        Returns:
            Tuple of (per_sample_loss [B], loss_dict).
        """
        pred_rays = predictions[self.pred_key]  # (B, S, H, W, 6)
        gt_rays = batch[DataField.WORLD_RAYS]  # (B, S, H, W, 6)
        valid_mask = batch[DataField.POINT_MASKS]  # (B, S, H, W)
        B, S, H, W, _ = pred_rays.shape

        # Since world_point = origin + z_depth * direction, and direction
        # depends only on intrinsics/rotation (not scene scale), only the
        # origin (rays[3:6]) needs scene-scale normalization.
        pred_scale = predictions.get(PRED_NORMALIZATION_SCALE_KEY)
        batch_scale = batch.get(BATCH_NORMALIZATION_SCALE_KEY)
        if pred_scale is not None:
            s = pred_scale.view(-1, 1, 1, 1, 1)
            pred_rays = torch.cat([pred_rays[..., :3], pred_rays[..., 3:] / s], dim=-1)
        if batch_scale is not None:
            s = batch_scale.view(-1, 1, 1, 1, 1)
            gt_rays = torch.cat([gt_rays[..., :3], gt_rays[..., 3:] / s], dim=-1)

        # Per-view masking - view is valid if it has >100 valid pixels
        view_valid_mask = valid_mask.sum(dim=[-1, -2]) > 100  # (B, S)

        # Handle empty mask - return zeros for all samples
        # Use (pred_rays * 0) to preserve gradient flow through model parameters,
        # which is required for DDP gradient sync even when all masks are empty.
        if view_valid_mask.sum() == 0:
            zero = (pred_rays * 0).reshape(B, -1).mean(dim=1)  # [B] with grad_fn
            return zero, {"loss": zero.detach().mean()}

        # Simple L1 loss with robustness clamp
        loss = (pred_rays - gt_rays).abs()  # (B, S, H, W, 6)
        loss = loss.mean(dim=(-1, -2, -3))  # (B, S) - average over H, W, channels
        loss = loss.clamp(max=self.max_loss)  # (B, S)
        loss = check_and_fix_inf_nan(loss, "loss_ray")

        # Per-sample mean over valid views only
        # Mask invalid views with 0, then compute mean over valid views per sample
        loss = loss * view_valid_mask.float()  # Zero out invalid views
        num_valid_per_sample = view_valid_mask.sum(dim=1).clamp(min=1)  # (B,)
        per_sample_loss = loss.sum(dim=1) / num_valid_per_sample  # (B,)

        return per_sample_loss, {"loss": per_sample_loss.detach().mean()}
