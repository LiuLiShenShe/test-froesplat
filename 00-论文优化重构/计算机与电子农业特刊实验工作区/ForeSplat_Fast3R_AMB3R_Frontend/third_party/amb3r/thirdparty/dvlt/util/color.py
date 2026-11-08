# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for visualizing outputs

Reference: https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/utils/colormaps.py
"""

from dataclasses import dataclass
from typing import Optional

import matplotlib
import torch
from torch import Tensor

from dvlt.common.constants import BLACK, WHITE
from dvlt.common.types import Colormaps


@dataclass(frozen=True)
class ColormapOptions:
    """Options for colormap"""

    colormap: Colormaps = "default"
    """ The colormap to use """
    normalize: bool = False
    """ Whether to normalize the input tensor image """
    colormap_min: float = 0
    """ Minimum value for the output colormap """
    colormap_max: float = 1
    """ Maximum value for the output colormap """
    invert: bool = False
    """ Whether to invert the output colormap """


def apply_colormap(
    image: Tensor,
    colormap_options: Optional[ColormapOptions] = None,
    eps: float = 1e-9,
) -> Tensor:
    """
    Applies a colormap to a tensor image.
    If single channel, applies a colormap to the image.
    If 3 channel, treats the channels as RGB.
    If more than 3 channel, applies a PCA reduction on the dimensions to 3 channels

    Args:
        image: Input tensor image.
        eps: Epsilon value for numerical stability.

    Returns:
        Tensor with the colormap applied.
    """
    if colormap_options is None:
        colormap_options = ColormapOptions()

    # default for rgb images
    if image.shape[-1] == 3:
        return image

    # rendering depth outputs
    if image.shape[-1] == 1 and torch.is_floating_point(image):
        output = image
        if colormap_options.normalize:
            output = output - torch.min(output)
            output = output / (torch.max(output) + eps)
        output = (
            output * (colormap_options.colormap_max - colormap_options.colormap_min) + colormap_options.colormap_min
        )
        output = torch.clip(output, 0, 1)
        if colormap_options.invert:
            output = 1 - output
        return apply_float_colormap(output, colormap=colormap_options.colormap)

    # rendering boolean outputs
    if image.dtype == torch.bool:
        return apply_boolean_colormap(image)

    if image.shape[-1] > 3:
        return apply_pca_colormap(image)

    raise NotImplementedError


def apply_float_colormap(image: Tensor, colormap: Colormaps = "viridis") -> Tensor:
    """Convert single channel to a color image.

    Args:
        image: Single channel image.
        colormap: Colormap for image.

    Returns:
        Tensor: Colored image with colors in [0, 1]
    """
    if colormap == "default":
        colormap = "turbo"

    image = torch.nan_to_num(image, 0)
    if colormap == "gray":
        return image.repeat(1, 1, 3)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return torch.tensor(matplotlib.colormaps[colormap].colors, device=image.device)[image_long[..., 0]]


def apply_depth_colormap(
    depth: Tensor,
    accumulation: Optional[Tensor] = None,
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    colormap_options: Optional[ColormapOptions] = None,
) -> Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        depth: Depth image (H, W, 1).
        accumulation: Ray accumulation used for masking vis.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.
        colormap: Colormap to apply.

    Returns:
        Colored depth image with colors in [0, 1]
    """
    if colormap_options is None:
        colormap_options = ColormapOptions()

    near_plane = near_plane if near_plane is not None else float(torch.min(depth))
    far_plane = far_plane if far_plane is not None else float(torch.max(depth))

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)

    colored_image = apply_colormap(depth, colormap_options=colormap_options)

    if accumulation is not None:
        colored_image = colored_image * accumulation + (1 - accumulation)

    return colored_image


def apply_boolean_colormap(
    image: Tensor,
    true_color: Tensor = WHITE,
    false_color: Tensor = BLACK,
) -> Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        image: Boolean image.
        true_color: Color to use for True.
        false_color: Color to use for False.

    Returns:
        Colored boolean image
    """

    colored_image = torch.ones(image.shape[:-1] + (3,))
    colored_image[image[..., 0], :] = true_color
    colored_image[~image[..., 0], :] = false_color
    return colored_image


def apply_pca_colormap(image: Tensor, pca_mat: Optional[Tensor] = None, ignore_zeros=True) -> Tensor:
    """Convert feature image to 3-channel RGB via PCA. The first three principle
    components are used for the color channels, with outlier rejection per-channel

    Args:
        image: image of arbitrary vectors
        pca_mat: an optional argument of the PCA matrix, shape (dim, 3)
        ignore_zeros: whether to ignore zero values in the input image (they won't affect the PCA computation)

    Returns:
        Tensor: Colored image
    """
    original_shape = image.shape
    image = image.view(-1, image.shape[-1])
    if ignore_zeros:
        valids = (image.abs().amax(dim=-1)) > 0
    else:
        valids = torch.ones(image.shape[0], dtype=torch.bool)

    if pca_mat is None:
        _, _, pca_mat = torch.pca_lowrank(image[valids, :], q=3, niter=20)
    assert pca_mat is not None
    image = torch.matmul(image, pca_mat[..., :3])
    d = torch.abs(image[valids, :] - torch.median(image[valids, :], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    m = 2.0  # this is a hyperparam controlling how many std dev outside for outliers
    rins = image[valids, :][s[:, 0] < m, 0]
    gins = image[valids, :][s[:, 1] < m, 1]
    bins = image[valids, :][s[:, 2] < m, 2]

    image[valids, 0] -= rins.min()
    image[valids, 1] -= gins.min()
    image[valids, 2] -= bins.min()

    image[valids, 0] /= rins.max() - rins.min()
    image[valids, 1] /= gins.max() - gins.min()
    image[valids, 2] /= bins.max() - bins.min()

    image = torch.clamp(image, 0, 1)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return image.view(*original_shape[:-1], 3)
