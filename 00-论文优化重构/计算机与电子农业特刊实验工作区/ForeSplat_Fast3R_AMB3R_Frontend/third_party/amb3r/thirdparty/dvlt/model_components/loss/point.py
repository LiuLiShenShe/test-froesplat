# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pointmap loss with confidence weighting."""

from __future__ import annotations

from typing import Optional

from dvlt.common.constants import DataField

from .common import ConfLoss


class PointLoss(ConfLoss):
    """World point (pointmap) regression loss with confidence weighting.

    Expects predictions to contain 'world_points' and 'world_points_conf'.
    Expects batch to contain world points and point masks.
    """

    def __init__(
        self,
        pred_key: str = "world_points",
        conf_key: str = "world_points_conf",
        gt_key: str = DataField.WORLD_POINTS,
        grad_loss: Optional[str] = "normal",
        **kwargs,
    ):
        """Initialize PointLoss.

        Args:
            pred_key: Key in predictions for world point values.
            conf_key: Key in predictions for world point confidence.
            gt_key: Key in batch for ground truth world points.
            grad_loss: Gradient loss type (default: "normal").
            **kwargs: Additional arguments passed to ConfLoss.
        """
        super().__init__(
            pred_key=pred_key,
            conf_key=conf_key,
            gt_key=gt_key,
            grad_loss=grad_loss,
            prefix="point",
            **kwargs,
        )
