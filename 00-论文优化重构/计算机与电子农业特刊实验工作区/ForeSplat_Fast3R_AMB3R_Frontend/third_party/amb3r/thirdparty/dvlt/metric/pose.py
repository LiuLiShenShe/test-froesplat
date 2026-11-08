# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
"""Pose evaluation metrics used in SfM, i.e. relative pose error, rotation accuracy, translation accuracy, and AUC."""

import torch
from torch import Tensor

from dvlt.common.pose import inverse_pose
from dvlt.common.rotation import so3_relative_angle


def compute_pose_metrics(
    pred_poses: Tensor, gt_poses: Tensor, thresholds: tuple[int, ...] = (1, 5, 20)
) -> dict[str, Tensor]:
    """Evaluate pose estimation accuracy of predicted poses against ground truth poses.

    Args:
        pred_poses: Predicted poses with shape (N, 4, 4) or (B, N, 4, 4) in c2w format.
        gt_poses: Ground truth poses with shape (N, 4, 4) or (B, N, 4, 4) in c2w format.
        thresholds: List of thresholds for error metrics in degrees. Default: (1, 5, 20).

    Returns:
        dict: Dictionary containing various metrics as scalar tensors including mean rotation error,
            mean translation error, rotation/translation accuracies at different
            thresholds, and AUC score.
    """
    assert (
        pred_poses.shape == gt_poses.shape
    ), f"Predicted poses shape: {pred_poses.shape}, GT poses shape: {gt_poses.shape}"

    rel_rangle_deg, rel_tangle_deg = compute_rel_deg(pred_poses, gt_poses)

    metrics = {}
    metrics["R_error"] = rel_rangle_deg.mean()
    metrics["T_error"] = rel_tangle_deg.mean()

    for threshold in thresholds:
        metrics[f"Racc_{threshold}"] = torch.mean((rel_rangle_deg < threshold).float())
        metrics[f"Tacc_{threshold}"] = torch.mean((rel_tangle_deg < threshold).float())
        metrics[f"Auc_{threshold}"] = calculate_auc(rel_rangle_deg, rel_tangle_deg, threshold)
    return metrics


@torch.no_grad()
def compute_rel_deg(pred_poses: Tensor, gt_poses: Tensor) -> tuple[Tensor, Tensor]:
    """Compute relative rotation and translation angles between predicted and ground truth poses.

    Args:
        pred_poses: Predicted SE3 poses with shape (N, 4, 4) or (B, N, 4, 4) in c2w format.
        gt_poses: Ground truth SE3 poses with shape (N, 4, 4) or (B, N, 4, 4) in c2w format.

    Returns:
        tuple: A tuple containing:
            - rel_rangle_deg: Relative rotation angles in degrees.
            - rel_tangle_deg: Relative translation angles in degrees.

    Raises:
        ValueError: If the input shape is invalid.
    """
    # Generate pairwise indices to compute relative poses
    if len(pred_poses.shape) == 3:
        B, N = 1, pred_poses.shape[0]
    elif len(pred_poses.shape) == 4:
        B, N = pred_poses.shape[:2]
    else:
        raise ValueError(f"Invalid shape for pred_poses: {pred_poses.shape}")

    assert N > 1, "N must be greater than 1 for relative pose computation"

    pair_idx_i1, pair_idx_i2 = batched_all_pairs(B, N)
    pair_idx_i1 = pair_idx_i1.to(pred_poses.device)
    pair_idx_i2 = pair_idx_i2.to(pred_poses.device)

    # Compute relative camera poses between pairs
    # For each pose pair (i,j), we compute the relative pose from i to j
    relative_pose_gt = inverse_pose(gt_poses[pair_idx_i2]).bmm((gt_poses[pair_idx_i1]))
    relative_pose_pred = inverse_pose(pred_poses[pair_idx_i2]).bmm(pred_poses[pair_idx_i1])

    # Compute the difference in rotation between ground truth and predicted relative poses
    rel_rangle_deg = rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3])
    rel_tangle_deg = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3])

    return rel_rangle_deg, rel_tangle_deg


def rotation_angle(rot_gt: Tensor, rot_pred: Tensor, batch_size: int = None) -> Tensor:
    """Calculate rotation angle between ground truth and predicted rotation matrices.

    Args:
        rot_gt: Ground truth rotation matrices with shape (B, 3, 3) from c2w poses.
        rot_pred: Predicted rotation matrices with shape (B, 3, 3) from c2w poses.
        batch_size: Optional batch size to reshape the output. Default: None.

    Returns:
        Tensor: Rotation angle in degrees.
    """
    # Use direct acos calculation without linear extrapolation
    rel_angle_rad = so3_relative_angle(rot_gt, rot_pred)
    rel_rangle_deg = torch.rad2deg(rel_angle_rad)

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt: Tensor, tvec_pred: Tensor, batch_size: int = None) -> Tensor:
    """Calculate translation angle between ground truth and predicted translation vectors.

    Args:
        tvec_gt: Ground truth translation vectors with shape (B, 3) from c2w poses.
        tvec_pred: Predicted translation vectors with shape (B, 3) from c2w poses.
        batch_size: Optional batch size to reshape the output. Default: None.

    Returns:
        Tensor: Translation angle in degrees.
    """
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = torch.rad2deg(rel_tangle_deg)
    # Ensure the angle is the smaller of the two possible angles
    rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def batched_all_pairs(B, N) -> tuple[Tensor, Tensor]:
    """Generate all pairwise combinations of indices for batched data.

    Args:
        B: Batch size.
        N: Number of elements per batch.

    Returns:
        tuple: Two tensors containing indices for all combinations of pairs.
    """
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]

    return i1, i2


def compare_translation_by_angle(t_gt, t, eps=1e-15) -> Tensor:
    """Compute the angle between two translation vectors.

    Args:
        t_gt: Ground truth translation vectors with shape (B, 3) from c2w poses.
        t: Predicted translation vectors with shape (B, 3) from c2w poses.
        eps: Small epsilon value to avoid division by zero. Default: 1e-15.

    Returns:
        Tensor: Angle between normalized translation vectors in radians.
    """
    # dot product along last dim: (...,)
    dot = torch.sum(t_gt * t, dim=-1)
    # norms: (...,)
    na = torch.norm(t_gt, dim=-1)
    nb = torch.norm(t, dim=-1)
    # cosine of angle, clamped to [-1,1]
    cos = dot / (na.clamp(min=eps) * nb.clamp(min=eps))
    theta = torch.acos(torch.clamp(cos, -1.0, 1.0))
    return theta


def calculate_auc(r_error: Tensor, t_error: Tensor, max_threshold: int) -> Tensor:
    """Calculate the Area Under the Curve (AUC) for given error arrays.

    Args:
        r_error: Rotation error values in degrees computed from c2w poses.
        t_error: Translation error values in degrees computed from c2w poses.
        max_threshold: Maximum threshold value for binning the histogram.

    Returns:
        Tensor: AUC score calculated as the mean of cumulative sum of normalized histogram.
    """
    # Concatenate the error tensors along a new axis
    error_matrix = torch.stack((r_error, t_error), dim=1)

    # Compute the maximum error value for each pair
    max_errors, _ = torch.max(error_matrix, dim=1)

    # Calculate histogram of maximum error values
    histogram = torch.histc(max_errors, bins=max_threshold + 1, min=0, max=max_threshold)

    # Normalize the histogram
    num_pairs = float(max_errors.size(0))
    normalized_histogram = histogram / num_pairs

    # Compute and return the cumulative sum of the normalized histogram
    return torch.cumsum(normalized_histogram, dim=0).mean()
