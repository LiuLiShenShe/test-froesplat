# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from torch import Tensor


def compute_intrinsics_metrics(intrinsics_pred: Tensor, intrinsics_gt: Tensor) -> dict[str, Tensor]:
    """
    Compute the metrics between the predicted and ground truth intrinsics.
    Currently, we only compute the focal length (x and y directions) error.

    Args:
        intrinsics_pred: predicted intrinsics with shape [N, 3, 3]
        intrinsics_gt: ground truth intrinsics with shape [N, 3, 3]

    Returns:
        metrics: dictionary of intrinsics metrics as scalar tensors
    """
    # Focal length error
    focal_x_pred = intrinsics_pred[:, 0, 0]
    focal_y_pred = intrinsics_pred[:, 1, 1]
    focal_x_gt = intrinsics_gt[:, 0, 0]
    focal_y_gt = intrinsics_gt[:, 1, 1]
    focal_x_error = (focal_x_gt - focal_x_pred).abs()
    focal_y_error = (focal_y_gt - focal_y_pred).abs()

    metrics = {
        "focal_x_error": focal_x_error.mean(),
        "focal_y_error": focal_y_error.mean(),
        "focal_mean_error": ((focal_x_error + focal_y_error) / 2.0).mean(),
    }
    return metrics
