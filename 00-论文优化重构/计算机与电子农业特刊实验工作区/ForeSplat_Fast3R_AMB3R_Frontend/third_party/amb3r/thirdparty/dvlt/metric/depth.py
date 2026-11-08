# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
"""Depth evaluation metrics.

- Absolute relative error (AbsRel)
- Squared relative error (SqRel)
- Root mean squared error (RMSE)
- Mean error (MAE)
- Percentage of pixels within error threshold (Delta1, Delta2, Delta3)

Alignment modes
---------------
- ``"median"``          - Median-based scale and shift. Fast, sequential closed-form (aligned with megasam)
- ``"scale_and_shift"`` - Scale + shift via L1 minimisation with Adam. Can be more accurate (aligned with pi3 / monst3r)
- ``"scale"``           - Scale-only via Weiszfeld IRLS. Stricter: exposes additive bias (aligned with pi3 / monst3r)
- ``"none"``            - No alignment (metric depth)

Report *scale* and *scale_and_shift* together: a large gap flags systematic additive bias.
"""

from typing import Literal, Optional

import torch
from torch import Tensor


# Valid alignment modes.
AlignMode = Literal["median", "scale", "scale_and_shift", "none"]


# ─────────────────────────── Alignment helpers ───────────────────────────


def align_median(pred: Tensor, gt: Tensor, mask: Tensor) -> Tensor:
    """Median-based scale-and-shift alignment.

    ``scale = median(gt / pred)`` then ``shift = median(gt - scale * pred)``.
    """
    pred_valid = pred[mask]
    gt_valid = gt[mask]
    if pred_valid.numel() == 0:
        return pred
    scale = torch.median(gt_valid / torch.clamp(pred_valid, min=1e-8))
    shift = torch.median(gt_valid - scale * pred_valid)
    return pred * scale + shift


def align_scale_weiszfeld(pred: Tensor, gt: Tensor, mask: Tensor, iterations: int = 10) -> Tensor:
    """Scale-only alignment using Weiszfeld iterative reweighted least squares.

    1. Initial scale ``s = mean(gt) / mean(pred)``
    2. 10 IRLS iterations refining ``s`` to minimise the weighted L1 residual
    3. Clamp ``s >= 1e-3``

    """
    pred_valid = pred[mask]
    gt_valid = gt[mask]
    if pred_valid.numel() == 0:
        return pred

    # Initial scale via mean ratio (matches Pi3 nanmean)
    s = torch.nanmean(gt_valid) / torch.nanmean(pred_valid)

    # Weiszfeld IRLS iterations
    for _ in range(iterations):
        residuals = s * pred_valid - gt_valid
        abs_residuals = residuals.abs() + 1e-8
        weights = 1.0 / abs_residuals
        s = torch.sum(weights * pred_valid * gt_valid) / torch.sum(weights * pred_valid**2)

    s = s.clamp(min=1e-3).detach()
    return pred * s


def align_scale_and_shift_lad(
    pred: Tensor,
    gt: Tensor,
    mask: Tensor,
    lr: float = 1e-4,
    max_iters: int = 1000,
    tol: float = 1e-6,
) -> Tensor:
    """Scale-and-shift alignment by minimising the L1 (LAD) objective with Adam.

    1. ``s_init = median(gt) / median(pred)``, ``t_init = 0``
    2. Adam optimiser minimises ``sum(|s * pred + t - gt|)``
    3. Early-stop when loss change < ``tol``

    """
    pred_valid = pred[mask].detach()
    gt_valid = gt[mask].detach()
    if pred_valid.numel() == 0:
        return pred

    s_init = (torch.median(gt_valid) / torch.median(pred_valid)).item()

    # Run the optimisation with gradients enabled — the outer test loop
    # typically disables autograd via torch.no_grad() / inference_mode().
    with torch.enable_grad():
        s = torch.tensor([s_init], requires_grad=True, device=pred.device, dtype=pred.dtype)
        t = torch.tensor([0.0], requires_grad=True, device=pred.device, dtype=pred.dtype)

        optimizer = torch.optim.Adam([s, t], lr=lr)
        prev_loss: float | None = None

        for _ in range(max_iters):
            optimizer.zero_grad()
            loss = torch.sum(torch.abs(s * pred_valid + t - gt_valid))
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            if prev_loss is not None and abs(prev_loss - current_loss) < tol:
                break
            prev_loss = current_loss

    return pred * s.detach() + t.detach()


def apply_alignment(pred: Tensor, gt: Tensor, mask: Tensor, align: AlignMode) -> Tensor:
    """Dispatch to the chosen alignment method and return the aligned prediction."""
    if align == "none":
        return pred
    elif align == "median":
        return align_median(pred, gt, mask)
    elif align == "scale":
        return align_scale_weiszfeld(pred, gt, mask)
    elif align == "scale_and_shift":
        return align_scale_and_shift_lad(pred, gt, mask)
    else:
        raise ValueError(f"Unknown alignment mode: {align!r}. Choose from: median, scale, scale_and_shift, none")


# ─────────────────────────── Metric computation ───────────────────────────


def compute_depth_metrics(
    pred_depth: Tensor,
    gt_depth: Tensor,
    valid_mask: Optional[Tensor] = None,
    align: AlignMode = "median",
) -> dict[str, Tensor]:
    """Compute depth evaluation metrics.

    Args:
        pred_depth: Predicted depth map of shape (N, H, W).
        gt_depth: Ground truth depth map of shape (N, H, W).
        valid_mask: Valid mask of shape (N, H, W). If ``None``, all pixels
            with ``gt_depth > 0`` are considered valid.
        align: Alignment mode — see module docstring for details.

    Returns:
        dict: Dictionary containing the calculated metrics as scalar tensors.
    """
    mask = valid_mask if valid_mask is not None else gt_depth > 0

    # Align predictions to ground truth
    pred_depth = apply_alignment(pred_depth, gt_depth, mask, align)

    # Apply mask
    pred_depth = pred_depth[mask]
    gt_depth = gt_depth[mask]

    abs_rel = torch.abs(pred_depth - gt_depth) / gt_depth
    sq_rel = (pred_depth - gt_depth) ** 2 / gt_depth
    rmse = torch.sqrt(torch.mean((pred_depth - gt_depth) ** 2))
    mae = torch.mean(torch.abs(pred_depth - gt_depth))

    max_ratio = torch.maximum(pred_depth / gt_depth, gt_depth / pred_depth)
    delta1 = (max_ratio < 1.25).float()
    delta2 = (max_ratio < 1.25**2).float()
    delta3 = (max_ratio < 1.25**3).float()

    metrics = {
        "AbsRel": abs_rel.mean(),
        "SqRel": sq_rel.mean(),
        "RMSE": rmse.mean(),
        "MAE": mae.mean(),
        "Delta1": delta1.mean(),
        "Delta2": delta2.mean(),
        "Delta3": delta3.mean(),
    }
    return metrics
