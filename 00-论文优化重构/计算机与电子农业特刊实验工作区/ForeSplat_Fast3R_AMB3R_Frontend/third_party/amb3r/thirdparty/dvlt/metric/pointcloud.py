# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pointcloud evaluation metrics."""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor


def _chamfer_distance(
    x: Tensor,
    y: Tensor,
    single_directional: bool = False,
    point_reduction: Optional[str] = "mean",
    batch_reduction: Optional[str] = "mean",
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Chamfer distance between two point clouds via ``scipy.spatial.cKDTree``.

    Each batch element is processed independently on the CPU; the result is
    returned on ``x.device`` with ``x.dtype``.  Returns Euclidean (un-squared)
    nearest-neighbor distances, optionally reduced along the point axis and
    then the batch axis.

    Args:
        x: ``(B, P1, D)`` source points.
        y: ``(B, P2, D)`` target points.
        single_directional: if ``True``, only the ``x -> y`` direction is
            computed.
        point_reduction: ``"mean"`` or ``None`` (no reduction).
        batch_reduction: ``"mean"`` or ``None`` (no reduction).

    Returns:
        If ``point_reduction is None``: per-point distances.  A tuple
        ``(d_xy, d_yx)`` shaped ``(B, P1)`` and ``(B, P2)`` when bidirectional,
        else ``d_xy`` shaped ``(B, P1)``.  Otherwise a (possibly batch-reduced)
        scalar tensor; when bidirectional, the two directions are summed.
    """
    if point_reduction not in ("mean", None):
        raise ValueError(f"Unsupported point_reduction: {point_reduction!r}, expected 'mean' or None.")
    if batch_reduction not in ("mean", None):
        raise ValueError(f"Unsupported batch_reduction: {batch_reduction!r}, expected 'mean' or None.")

    device = x.device
    dtype = x.dtype
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    d_xy_list = []
    d_yx_list = []
    for b in range(x_np.shape[0]):
        d_xy, _ = cKDTree(y_np[b]).query(x_np[b], k=1)
        d_xy_list.append(d_xy)
        if not single_directional:
            d_yx, _ = cKDTree(x_np[b]).query(y_np[b], k=1)
            d_yx_list.append(d_yx)

    d_xy_t = torch.from_numpy(np.stack(d_xy_list)).to(device=device, dtype=dtype)
    if not single_directional:
        d_yx_t = torch.from_numpy(np.stack(d_yx_list)).to(device=device, dtype=dtype)

    if point_reduction is None:
        return d_xy_t if single_directional else (d_xy_t, d_yx_t)

    loss = d_xy_t.mean(dim=1) if single_directional else d_xy_t.mean(dim=1) + d_yx_t.mean(dim=1)
    return loss.mean() if batch_reduction == "mean" else loss


def compute_point_cloud_metrics(
    pred_points: Tensor,
    gt_points: Tensor,
    threshold: float = 0.01,
) -> Dict[str, Tensor]:
    """
    Compute precision, recall, and F-score metrics for point cloud evaluation.

    Implementation follows the metrics used in the DTU and Tanks & Temples benchmarks.

    Args:
        pred_points: Predicted point cloud with shape (N, 3).
        gt_points: Ground truth point cloud with shape (M, 3).
        threshold: Distance threshold for determining if a point is correctly predicted. Default: 0.01.

    Returns:
        Dictionary containing precision, recall, and F-score metrics.
    """
    assert pred_points.device == gt_points.device, "Input tensors must be on the same device"

    pred_points_batch = pred_points.unsqueeze(0)  # (1, N, 3)
    gt_points_batch = gt_points.unsqueeze(0)  # (1, M, 3)

    dist_pred_to_gt, dist_gt_to_pred = _chamfer_distance(
        pred_points_batch, gt_points_batch, point_reduction=None, batch_reduction=None
    )
    dist_pred_to_gt = dist_pred_to_gt.squeeze(0)  # (N,)
    dist_gt_to_pred = dist_gt_to_pred.squeeze(0)  # (M,)

    acc_mean = dist_pred_to_gt.mean()
    acc_med = dist_pred_to_gt.median()
    comp_mean = dist_gt_to_pred.mean()
    comp_med = dist_gt_to_pred.median()

    precision = (dist_pred_to_gt < threshold).float().mean()
    recall = (dist_gt_to_pred < threshold).float().mean()
    f_score = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "acc_mean": acc_mean,
        "acc_med": acc_med,
        "comp_mean": comp_mean,
        "comp_med": comp_med,
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
    }


def compute_chamfer_distance(
    pred_points: Tensor,
    gt_points: Tensor,
    bidirectional: bool = True,
) -> Tensor:
    """
    Compute Chamfer Distance between point clouds.

    Args:
        pred_points: Predicted point cloud with shape (N, 3).
        gt_points: Ground truth point cloud with shape (M, 3).
        bidirectional: If True, compute bidirectional Chamfer distance.

    Returns:
        Chamfer distance value.
    """
    assert pred_points.device == gt_points.device, "Input tensors must be on the same device"

    pred_points = pred_points.unsqueeze(0)
    gt_points = gt_points.unsqueeze(0)

    return _chamfer_distance(pred_points, gt_points, single_directional=not bidirectional)


def compute_point_cloud_metrics_multi_threshold(
    pred_points: Tensor,
    gt_points: Tensor,
    thresholds: Tuple[float, ...] = (0.01, 0.05, 0.1),
) -> Dict[str, Tensor]:
    """
    Compute precision, recall, and F-score metrics across multiple thresholds (in meters).

    Args:
        pred_points: Predicted point cloud with shape (N, 3).
        gt_points: Ground truth point cloud with shape (M, 3).
        thresholds: Tuple of distance thresholds for evaluation.
            Default: (0.01, 0.05, 0.1) meters.

    Returns:
        Dictionary containing metrics for each threshold.
    """
    assert pred_points.device == gt_points.device, "Input tensors must be on the same device"

    pred_points_batch = pred_points.unsqueeze(0)  # (1, N, 3)
    gt_points_batch = gt_points.unsqueeze(0)  # (1, M, 3)

    dist_pred_to_gt, dist_gt_to_pred = _chamfer_distance(
        pred_points_batch, gt_points_batch, point_reduction=None, batch_reduction=None
    )
    dist_pred_to_gt = dist_pred_to_gt.squeeze(0)  # (N,)
    dist_gt_to_pred = dist_gt_to_pred.squeeze(0)  # (M,)

    metrics = {}
    for threshold in thresholds:
        precision = (dist_pred_to_gt < threshold).float().mean()
        recall = (dist_gt_to_pred < threshold).float().mean()
        f_score = 2 * precision * recall / (precision + recall + 1e-8)
        metrics[f"precision@{threshold:.3f}m"] = precision
        metrics[f"recall@{threshold:.3f}m"] = recall
        metrics[f"f_score@{threshold:.3f}m"] = f_score

    f_scores = [metrics[f"f_score@{t:.3f}m"] for t in thresholds]
    metrics["f_score_avg"] = sum(f_scores) / len(f_scores)

    return metrics
