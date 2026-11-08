# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in
# LICENSES/VGGT-LICENSE.txt in the root of this source tree.
#
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from math import ceil, floor
from typing import Optional

import torch
from accelerate import logging
from torch.nn import functional as F

from dvlt.common.amp import force_fp32


logger = logging.get_logger(__name__)


def check_and_fix_inf_nan(loss_tensor: torch.Tensor, loss_name: str, hard_max: Optional[float] = 100) -> torch.Tensor:
    """
    Checks if 'loss_tensor' contains inf or nan. If it does, replace those
    values with zero and print a warning with the name of the loss tensor.

    Args:
        loss_tensor (torch.Tensor): The loss tensor to check.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float): Maximum allowed value for the loss tensor.
    Returns:
        torch.Tensor: The checked and fixed loss tensor, with inf/nan replaced by 0.
    """

    if torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any():
        logger.warn(f"{loss_name} has inf or nan. Setting those values to 0.")
        loss_tensor = torch.where(
            torch.isnan(loss_tensor) | torch.isinf(loss_tensor),
            torch.tensor(0.0, device=loss_tensor.device),
            loss_tensor,
        )

    if hard_max is not None:
        loss_tensor = torch.clamp(loss_tensor, min=-hard_max, max=hard_max)

    return loss_tensor


@force_fp32
def point_map_to_normal(
    point_map: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    point_map: (B, H, W, 3)  - 3D points laid out in a 2D grid
    mask:      (B, H, W)     - valid pixels (bool)

    Returns:
      normals: (4, B, H, W, 3)  - normal vectors for each of the 4 cross-product directions
      valids:  (4, B, H, W)     - corresponding valid masks
    """
    # Pad inputs to avoid boundary issues
    padded_mask = F.pad(mask, (1, 1, 1, 1), mode="constant", value=0)
    pts = F.pad(point_map.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=0).permute(0, 2, 3, 1)

    # Each pixel's neighbors
    center = pts[:, 1:-1, 1:-1, :]  # B,H,W,3
    up = pts[:, :-2, 1:-1, :]
    left = pts[:, 1:-1, :-2, :]
    down = pts[:, 2:, 1:-1, :]
    right = pts[:, 1:-1, 2:, :]

    # Direction vectors
    up_dir = up - center
    left_dir = left - center
    down_dir = down - center
    right_dir = right - center

    # Four cross products (shape: B,H,W,3 each)
    n1 = torch.cross(up_dir, left_dir, dim=-1)  # up x left
    n2 = torch.cross(left_dir, down_dir, dim=-1)  # left x down
    n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
    n4 = torch.cross(right_dir, up_dir, dim=-1)  # right x up

    # Validity for each cross-product direction
    # We require that both directions' pixels are valid
    v1 = padded_mask[:, :-2, 1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
    v2 = padded_mask[:, 1:-1, :-2] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:, 1:-1]
    v3 = padded_mask[:, 2:, 1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
    v4 = padded_mask[:, 1:-1, 2:] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2, 1:-1]

    # Stack them to shape (4,B,H,W,3), (4,B,H,W)
    normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
    valids = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

    # Normalize each direction's normal
    # shape is (4, B, H, W, 3), so dim=-1 is the vector dimension
    # clamp_min(eps) to avoid division by zero
    # lengths = torch.norm(normals, dim=-1, keepdim=True).clamp_min(eps)
    # normals = normals / lengths
    normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    # Zero out invalid entries so they don't pollute subsequent computations
    # normals = normals * valids.unsqueeze(-1)

    return normals, valids


def compute_normalization_scale(
    pts3d: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Compute scale factor for normalizing point cloud to unit average norm.

    Args:
        pts3d:      3-D point map of shape ``B,S,H,W,3``.
        valid_mask: Validity mask (``B,S,H,W``).
        eps:        Numerical stability constant.

    Returns:
        scale_factor: Shape ``(B,)`` - the average distance to origin per batch.
    """
    dist = pts3d.norm(dim=-1)

    dist_sum = (dist * valid_mask).sum(dim=[1, 2, 3])
    valid_count = valid_mask.sum(dim=[1, 2, 3])

    avg_scale = (dist_sum / (valid_count + eps)).clamp(min=eps, max=1e3)
    return avg_scale


def get_quantile_mask(
    loss_tensor: torch.Tensor,
    valid_range: float,
    min_elements: int = 1000,
    hard_max: float = 100,
) -> torch.Tensor:
    """Get boolean mask for values below quantile threshold.

    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss

    Returns:
        Boolean mask of same shape as loss_tensor (True = keep)
    """
    # If too small, keep everything
    if loss_tensor.numel() <= min_elements:
        return torch.ones_like(loss_tensor, dtype=torch.bool)

    # Clamp for quantile computation
    clamped = loss_tensor.clamp(max=hard_max)

    # For very large tensors, randomly sample to compute quantile threshold.
    # Using randint (with replacement) instead of randperm to avoid allocating
    # a full 100M+ permutation and a known CUDA illegal memory access bug.
    if loss_tensor.numel() > 100_000_000:
        indices = torch.randint(0, loss_tensor.numel(), (1_000_000,), device=loss_tensor.device)
        sampled = clamped.view(-1)[indices]
        quantile_thresh = torch_quantile(sampled.detach(), valid_range)
    else:
        quantile_thresh = torch_quantile(clamped.detach(), valid_range)

    quantile_thresh = min(quantile_thresh, hard_max)

    # Get mask of values below threshold (applied to full tensor)
    quantile_mask = clamped < quantile_thresh

    # Only apply if enough elements remain
    if quantile_mask.sum() > min_elements:
        return quantile_mask

    return torch.ones_like(loss_tensor, dtype=torch.bool)


def torch_quantile(
    input: torch.Tensor,
    q: float | torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!") from None

    # Sanitization: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Sanitization: inteporlation
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} " f"(got '{interpolation}')!"
        )

    # Sanitization: out
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Logic
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Rectification: keepdim
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)
