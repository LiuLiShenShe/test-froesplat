# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in
# LICENSES/VGGT-LICENSE.txt in the root of this source tree.
#
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Common loss functions used across depth and point modules."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from dvlt.common.constants import DataField
from dvlt.model_components.loss.base import (
    BATCH_NORMALIZATION_SCALE_KEY,
    PRED_NORMALIZATION_SCALE_KEY,
)
from dvlt.model_components.loss.util import (
    check_and_fix_inf_nan,
    get_quantile_mask,
    point_map_to_normal,
)


class ConfLoss(nn.Module):
    """Confidence-weighted regression loss.

    Implements the loss:
        L = gamma * c * ||x - x_hat||_2 - alpha * log(c)
    """

    def __init__(
        self,
        pred_key: str,
        conf_key: str,
        gt_key: str,
        gamma: float = 1.0,
        alpha: float = 0.2,
        grad_loss: Optional[str] = None,
        grad_loss_synthetic_only: bool = True,
        valid_range: float = 0.98,
        disable_conf: bool = False,
        conf_grad_loss: bool = False,
        prefix: str = "",
    ):
        """Initialize ConfLoss.

        Args:
            pred_key: Key in predictions for predicted values.
            conf_key: Key in predictions for confidence values.
            gt_key: Key in batch for ground truth values.
            gamma: Linear weight on the data term.
            alpha: Weight on the negative-log confidence term.
            grad_loss: Gradient loss type - "grad", "normal", "edge", or None.
            grad_loss_synthetic_only: If True, only compute gradient loss for synthetic data.
            valid_range: Quantile filtering threshold (disabled if < 0).
            disable_conf: If True, ignore confidence (plain L2).
            conf_grad_loss: Use confidence for gradient loss.
            prefix: Prefix for loss dict keys and debug messages.
        """
        super().__init__()
        self.pred_key = pred_key
        self.conf_key = conf_key
        self.gt_key = gt_key
        self.gamma = gamma
        self.alpha = alpha
        self.grad_loss = grad_loss
        self.grad_loss_synthetic_only = grad_loss_synthetic_only
        self.valid_range = valid_range
        self.disable_conf = disable_conf
        self.conf_grad_loss = conf_grad_loss
        self.prefix = prefix

    def check_inputs(self, predictions: Dict[str, Any], batch: Dict[str, Any]) -> bool:
        if self.pred_key not in predictions or predictions[self.pred_key] is None:
            return False
        if self.conf_key not in predictions or predictions[self.conf_key] is None:
            return False
        return self.gt_key in batch and DataField.POINT_MASKS in batch

    def _prepare_inputs(
        self,
        predictions: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare prediction, confidence, ground truth, and mask tensors.

        Applies normalization if scale factors are present in batch.
        Override in subclass for custom preprocessing (e.g., adding dimensions).
        """
        pred = predictions[self.pred_key]
        conf = predictions[self.conf_key]
        gt = check_and_fix_inf_nan(batch[self.gt_key], f"gt_{self.prefix}")
        mask = batch[DataField.POINT_MASKS]

        # Apply normalization if scale factors are available
        pred_scale = predictions.get(PRED_NORMALIZATION_SCALE_KEY)
        batch_scale = batch.get(BATCH_NORMALIZATION_SCALE_KEY)

        if pred_scale is not None:
            pred = pred / pred_scale.view(-1, 1, 1, 1, 1)
        if batch_scale is not None:
            gt = gt / batch_scale.view(-1, 1, 1, 1, 1)

        return pred, conf, gt, mask

    def forward(
        self,
        predictions: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred, pred_conf, gt, valid_mask = self._prepare_inputs(predictions, batch)
        B = pred.shape[0]

        # Handle empty batches - return zeros for all samples
        # Use (pred * 0) to preserve gradient flow through model parameters,
        # which is required for DDP gradient sync even when all masks are empty.
        if not valid_mask.any():
            zero = (pred * 0).reshape(B, -1).mean(dim=1)  # [B] with grad_fn
            loss_dict: Dict[str, torch.Tensor] = {
                "loss_conf": zero.detach().mean(),
                "loss_reg": zero.detach().mean(),
                "loss_grad": zero.detach().mean(),
                "loss_conf_regularizer": zero.detach().mean(),
            }
            return zero, loss_dict

        # Compute regression loss (per-sample) with optional confidence weighting
        reg_loss_per_sample, loss_reg, conf_reg_mean = self._compute_reg_loss(pred, gt, valid_mask, pred_conf)

        if self.grad_loss_synthetic_only:
            grad_valid_mask = valid_mask & batch[DataField.IS_SYNTHETIC][..., None, None]
        else:
            grad_valid_mask = valid_mask

        loss_grad = self._compute_grad_loss(pred, gt, grad_valid_mask, pred_conf)

        # Add per-sample gradient loss
        per_sample_loss = reg_loss_per_sample + self.gamma * loss_grad  # [B] + [B] -> [B]

        loss_dict = {
            "loss_conf": per_sample_loss.detach().mean(),
            "loss_reg": loss_reg.detach().mean(),
            "loss_grad": loss_grad.detach().mean(),
            "loss_conf_regularizer": conf_reg_mean.detach(),
        }

        return per_sample_loss, loss_dict

    def _compute_reg_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        valid_mask: torch.Tensor,
        pred_conf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute regression loss per sample, with optional confidence weighting.

        Args:
            pred: Predictions of shape (B, S, H, W, C)
            gt: Ground truth of shape (B, S, H, W, C)
            valid_mask: Valid pixel mask of shape (B, S, H, W)
            pred_conf: Predicted confidence of shape (B, S, H, W)

        Returns:
            Tuple of (per_sample_loss [B], loss_reg_vec for logging, conf_reg_mean scalar).
        """
        B = pred.shape[0]

        # Compute L2 diff
        diff = torch.norm(gt - pred, dim=-1)  # (B, S, H, W)
        diff = check_and_fix_inf_nan(diff, f"{self.prefix}_loss_reg")

        # For logging: flattened valid losses
        loss_reg_vec = diff[valid_mask]

        # Compute the loss to optimize (with or without confidence weighting)
        if self.disable_conf:
            loss = self.gamma * diff  # (B, S, H, W)
            keep_mask = valid_mask
            conf_reg_mean = torch.tensor(0.0, device=pred.device)
        else:
            conf_reg = torch.log(pred_conf)  # (B, S, H, W)
            loss = self.gamma * diff * pred_conf - self.alpha * conf_reg  # (B, S, H, W)
            conf_reg_mean = conf_reg[valid_mask].mean()

            # Start with valid_mask as our keep_mask
            keep_mask = valid_mask.clone()  # (B, S, H, W)

            # Optional quantile filtering (global)
            if self.valid_range > 0:
                valid_losses = loss[valid_mask]  # (N_total,)
                quantile_keep = get_quantile_mask(valid_losses, self.valid_range)
                keep_mask[valid_mask] = quantile_keep

            loss = check_and_fix_inf_nan(loss, f"{self.prefix}_conf_loss")

        # Compute per-sample mean
        flat_loss = loss.view(B, -1)
        flat_mask = keep_mask.view(B, -1)
        masked_loss = flat_loss * flat_mask.float()
        num_valid = flat_mask.sum(dim=1).clamp(min=1)
        per_sample_loss = masked_loss.sum(dim=1) / num_valid  # [B]

        return per_sample_loss, loss_reg_vec, conf_reg_mean

    def _compute_grad_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        valid_mask: torch.Tensor,
        conf: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradient loss per sample.

        Returns:
            Per-sample gradient loss of shape [B].
        """
        B, S, H, W, C = pred.shape
        loss_grad = pred.new_zeros(B)

        if self.grad_loss is not None:
            pred_flat = pred.reshape(B * S, H, W, C)
            gt_flat = gt.reshape(B * S, H, W, C)
            mask_flat = valid_mask.reshape(B * S, H, W)
            conf_flat = conf.reshape(B * S, H, W) if self.conf_grad_loss else None

            if self.grad_loss == "grad":
                loss_per_view = _loss_multi_scale(
                    _gradient_loss, pred_flat, gt_flat, mask_flat, conf=conf_flat, prefix=self.prefix
                )
            elif self.grad_loss == "normal":
                loss_per_view = _loss_multi_scale(
                    _normal_loss, pred_flat, gt_flat, mask_flat, conf=conf_flat, scales=3, prefix=self.prefix
                )
            elif self.grad_loss == "edge":
                loss_per_view = _loss_multi_scale(
                    _edge_loss, pred_flat, gt_flat, mask_flat, conf=conf_flat, prefix=self.prefix
                )
            else:
                raise ValueError(f"Unsupported grad_loss mode: {self.grad_loss}")

            # Reshape from (B*S,) to (B, S) and average over views per sample
            loss_per_view = loss_per_view.view(B, S)  # (B, S)
            loss_grad = loss_per_view.mean(dim=1)  # (B,)

        return loss_grad


def _loss_multi_scale(
    loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], str], torch.Tensor],
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    conf: Optional[torch.Tensor] = None,
    scales: int = 4,
    prefix: str = "",
) -> torch.Tensor:
    """Apply loss on progressively sub-sampled grids.

    Returns:
        Per-view losses of shape (N,) where N is the batch dimension of prediction.
    """
    N = prediction.shape[0]
    total = prediction.new_zeros(N)
    for scale in range(scales):
        step = pow(2, scale)
        total = total + loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None,
            prefix=prefix,
        )
    return total / scales


def _edge_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    conf: Optional[torch.Tensor] = None,
    prefix: str = "",
) -> torch.Tensor:
    """Edge-based loss returning per-sample losses.

    From, e.g. DA3: |∇_xD - ∇_xD_hat| + |∇_yD - ∇_yD_hat|

    Args:
        prediction: (N, H, W, C) predictions
        target: (N, H, W, C) targets
        mask: (N, H, W) valid mask
        conf: Optional (N, H, W) confidence weights
        prefix: Prefix for debug messages

    Returns:
        Per-sample losses of shape (N,).
    """
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])

    # Per-sample divisor for normalization
    divisor_per_sample = mask.sum(dim=(1, 2, 3)).clamp(min=1)  # (N,)

    pred_masked = torch.mul(mask, prediction)
    gt_masked = torch.mul(mask, target)

    # x-gradients
    grad_x_pred = pred_masked[:, :, 1:] - pred_masked[:, :, :-1]
    grad_x_gt = gt_masked[:, :, 1:] - gt_masked[:, :, :-1]

    # y-gradients
    grad_y_pred = pred_masked[:, 1:, :] - pred_masked[:, :-1, :]
    grad_y_gt = gt_masked[:, 1:, :] - gt_masked[:, :-1, :]

    # |∇D - ∇D_hat| with masking
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x_diff = torch.abs(grad_x_pred - grad_x_gt)
    grad_x_diff = torch.mul(mask_x, grad_x_diff)

    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y_diff = torch.abs(grad_y_pred - grad_y_gt)
    grad_y_diff = torch.mul(mask_y, grad_y_diff)

    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        grad_x_diff = grad_x_diff * conf[:, :, 1:]
        grad_y_diff = grad_y_diff * conf[:, 1:, :]

    # Per-sample loss: sum over spatial dims, normalize by per-sample divisor
    edge_loss = torch.sum(grad_x_diff, (1, 2, 3)) + torch.sum(grad_y_diff, (1, 2, 3))  # (N,)
    # Apply check_and_fix_inf_nan BEFORE division to match original behavior (clamps raw values)
    edge_loss = check_and_fix_inf_nan(edge_loss, f"{prefix}_edge_loss")
    edge_loss = edge_loss / divisor_per_sample  # (N,)
    return edge_loss


def _gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    conf: Optional[torch.Tensor] = None,
    prefix: str = "",
) -> torch.Tensor:
    """Multi-scale image-gradient L1 loss returning per-sample losses.

    See https://gist.github.com/dvdhfnr/732c26b61a0e63a0abc8a5d769dbebd0

    Args:
        prediction: (N, H, W, C) predictions
        target: (N, H, W, C) targets
        mask: (N, H, W) valid mask
        conf: Optional (N, H, W) confidence weights
        prefix: Prefix for debug messages

    Returns:
        Per-sample losses of shape (N,).
    """
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])

    # Per-sample divisor for normalization
    divisor_per_sample = mask.sum(dim=(1, 2, 3)).clamp(min=1)  # (N,)

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        grad_x = grad_x * conf[:, :, 1:]
        grad_y = grad_y * conf[:, 1:, :]

    # Per-sample loss: sum over spatial dims, normalize by per-sample divisor
    image_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))  # (N,)
    image_loss = check_and_fix_inf_nan(image_loss, f"{prefix}_gradient_loss")
    image_loss = image_loss / divisor_per_sample  # (N,)
    return image_loss


def _normal_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    cos_eps: float = 1e-8,
    conf: Optional[torch.Tensor] = None,
    prefix: str = "",
) -> torch.Tensor:
    """Normal-based loss comparing angles between predicted and GT normals.

    Args:
        prediction: (N, H, W, C) predictions
        target: (N, H, W, C) targets
        mask: (N, H, W) valid mask
        cos_eps: Epsilon for numerical stability
        conf: Optional (N, H, W) confidence weights
        prefix: Prefix for debug messages

    Returns:
        Per-sample losses of shape (N,).
    """
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals, gt_valids = point_map_to_normal(target, mask, eps=cos_eps)

    # all_valid: (4, N, H, W)
    all_valid = pred_valids & gt_valids

    # Compute dot product for all positions: (4, N, H, W)
    dot = torch.sum(pred_normals * gt_normals, dim=-1)
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)
    loss_map = 1 - dot  # (4, N, H, W)

    # Apply confidence if provided
    if conf is not None:
        conf_expanded = conf[None, ...].expand(4, -1, -1, -1)  # (4, N, H, W)
        loss_map = loss_map * conf_expanded

    # Compute per-sample mean over valid positions
    # Mask invalid positions with zeros
    loss_map = loss_map * all_valid.float()  # (4, N, H, W)

    # Sum over directions and spatial dims, divide by count
    loss_per_sample = loss_map.sum(dim=(0, 2, 3))  # (N,)
    valid_count_per_sample = all_valid.sum(dim=(0, 2, 3)).clamp(min=1).float()  # (N,)
    loss_per_sample = loss_per_sample / valid_count_per_sample  # (N,)

    loss_per_sample = check_and_fix_inf_nan(loss_per_sample, f"{prefix}_normal_loss")
    return loss_per_sample


def quaternion_loss(
    pred_quat: torch.Tensor,
    gt_quat: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda x, y: torch.abs(x - y),
) -> torch.Tensor:
    """Compute double cover aware loss: min(loss_fn(pred, gt), loss_fn(-pred, gt))."""
    gt_quat = F.normalize(gt_quat, dim=-1)
    return torch.minimum(loss_fn(pred_quat, gt_quat), loss_fn(-pred_quat, gt_quat))


def rotation_loss(
    pred_rot: torch.Tensor,
    gt_rot: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute the geodesic distance between rotation matrices."""
    residual = pred_rot.transpose(-2, -1) @ gt_rot
    trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
    cosine = (trace - 1) / 2
    return torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))
