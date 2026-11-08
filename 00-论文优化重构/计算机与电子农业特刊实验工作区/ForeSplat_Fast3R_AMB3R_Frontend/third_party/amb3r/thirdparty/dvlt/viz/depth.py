# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Depth map visualization utilities."""

from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor

from dvlt.util.color import ColormapOptions, apply_depth_colormap


def overlay_depth_map(
    image: Union[Tensor, np.ndarray],
    depth_map: Union[Tensor, np.ndarray],
    alpha: float = 0.5,
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    colormap_options: Optional[ColormapOptions] = None,
) -> np.ndarray:
    """Overlay a depth map on an image.

    Args:
        image (Union[torch.Tensor, np.ndarray]): image shape (H, W, 3)
        depth_map (Union[torch.Tensor, np.ndarray]): depth map shape (H, W) or (H, W, 1)
        alpha (float): alpha value for the overlay. 0 means no overlay, 1 means full overlay. Defaults to 0.5.
        colormap_options (ColormapOptions): colormap options to use. Defaults to ColormapOptions("inferno_r").
    Returns:
        np.ndarray: overlayed image as numpy array (H, W, 3), uint8 0-255
    """
    if colormap_options is None:
        colormap_options = ColormapOptions("default")

    # Convert to torch if needed
    is_tensor = isinstance(image, torch.Tensor)
    if not is_tensor:
        image = torch.from_numpy(image.copy())

    if not isinstance(depth_map, torch.Tensor):
        depth_map = torch.from_numpy(depth_map.copy())

    # Normalize image to 0-1 if needed (likely uint8)
    if image.dtype == torch.uint8:
        image = image.float() / 255.0

    # Ensure depth_map is 3D (H, W, 1)
    if depth_map.dim() == 2:
        depth_map = depth_map.unsqueeze(-1)

    # Apply colormap to depth
    depth_map_colored = apply_depth_colormap(
        depth_map, near_plane=near_plane, far_plane=far_plane, colormap_options=colormap_options
    )

    # Perform the overlay
    overlayed = image * (1 - alpha) + depth_map_colored * alpha
    invalid_mask = (depth_map == 0).squeeze(-1)
    overlayed[invalid_mask] = image[invalid_mask]

    # Always convert to numpy
    return (overlayed * 255).cpu().numpy().astype(np.uint8)
