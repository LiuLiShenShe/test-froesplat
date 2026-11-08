# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Depth loss with confidence weighting."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from dvlt.common.constants import DataField
from dvlt.model_components.loss.base import (
    BATCH_NORMALIZATION_SCALE_KEY,
    PRED_NORMALIZATION_SCALE_KEY,
)

from .common import ConfLoss
from .util import check_and_fix_inf_nan


class DepthLoss(ConfLoss):
    """Depth prediction loss with confidence weighting.

    Expects predictions to contain 'depth' and 'depth_conf'.
    Expects batch to contain depths and point masks.
    """

    def __init__(
        self,
        pred_key: str = "depth",
        conf_key: str = "depth_conf",
        gt_key: str = DataField.DEPTHS,
        grad_loss: Optional[str] = "grad",
        **kwargs,
    ):
        """Initialize DepthLoss.

        Args:
            pred_key: Key in predictions for depth values.
            conf_key: Key in predictions for depth confidence.
            gt_key: Key in batch for ground truth depth.
            grad_loss: Gradient loss type (default: "grad").
            **kwargs: Additional arguments passed to ConfLoss.
        """
        super().__init__(
            pred_key=pred_key,
            conf_key=conf_key,
            gt_key=gt_key,
            grad_loss=grad_loss,
            prefix="depth",
            **kwargs,
        )

    def _prepare_inputs(
        self,
        predictions: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        depth = predictions[self.pred_key]
        depth_conf = predictions[self.conf_key]

        # Ensure depth has trailing dimension
        if depth.dim() == 4:  # (B, S, H, W)
            depth = depth.unsqueeze(-1)

        gt_depth = check_and_fix_inf_nan(batch[self.gt_key], "gt_depth")
        gt_depth = gt_depth[..., None]

        # Apply normalization if scale factors are available
        pred_scale = predictions.get(PRED_NORMALIZATION_SCALE_KEY)
        batch_scale = batch.get(BATCH_NORMALIZATION_SCALE_KEY)

        if pred_scale is not None:
            depth = depth / pred_scale.view(-1, 1, 1, 1, 1)
        if batch_scale is not None:
            gt_depth = gt_depth / batch_scale.view(-1, 1, 1, 1, 1)

        return depth, depth_conf, gt_depth, batch[DataField.POINT_MASKS]
