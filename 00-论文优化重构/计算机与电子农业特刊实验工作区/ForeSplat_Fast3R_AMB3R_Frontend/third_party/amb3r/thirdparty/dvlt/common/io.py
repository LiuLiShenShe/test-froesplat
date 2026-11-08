# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image
from torch import Tensor


def read_image_cv2(path: str, rgb: bool = True) -> np.ndarray:
    """
    Reads an image from disk using OpenCV, returning it as an RGB image array (H, W, 3).

    Args:
        path (str):
            File path to the image.
        rgb (bool):
            If True, convert the image to RGB.
            If False, leave the image in BGR/grayscale.

    Returns:
        np.ndarray or None:
            A numpy array of shape (H, W, 3) if successful,
            or None if the file does not exist or could not be read.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"File does not exist or is empty: {path}")
        return None

    img = cv2.imread(path)
    if img is None:
        print(f"Could not load image={path}. Retrying...")
        img = cv2.imread(path)
        if img is None:
            print("Retry failed.")
            return None

    if rgb:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def read_depth(
    path: str,
    height: Optional[int] = None,
    width: Optional[int] = None,
    scale_adjustment: float = 1.0,
    interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    """
    Reads a depth map from disk in either .exr or .png format. The .exr is loaded using OpenCV
    with the environment variable OPENCV_IO_ENABLE_OPENEXR=1. The .png is assumed to be a 16-bit
    PNG (converted from half float).

    Args:
        path (str):
            File path to the depth image. Must end with .exr or .png.
        height (int):
            Height of the depth image for resizing.
        width (int):
            Width of the depth image for resizing.
        scale_adjustment (float):
            A multiplier for adjusting the loaded depth values (default=1.0).
        interpolation (int):
            Interpolation method for resizing the depth image.

    Returns:
        np.ndarray:
            A float32 array (H, W) containing the loaded depth. Zeros or non-finite values
            may indicate invalid regions.

    Raises:
        ValueError:
            If the file extension is not supported.
    """
    if path.lower().endswith(".exr"):
        # Ensure OPENCV_IO_ENABLE_OPENEXR is set to "1"
        d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., 0]
        d[d > 1e9] = 0.0
    elif path.lower().endswith(".png"):
        d = _load_16bit_png_depth(path)
    else:
        raise ValueError(f'unsupported depth file name "{path}"')

    d = d * scale_adjustment
    d[~np.isfinite(d)] = 0.0

    if height is not None and width is not None:
        d = cv2.resize(d, (width, height), interpolation=interpolation)

    return d


def _load_16bit_png_depth(depth_png: str) -> np.ndarray:
    """
    Loads a 16-bit PNG as a half-float depth map (H, W), returning a float32 NumPy array.

    Implementation detail:
      - PIL loads 16-bit data as 32-bit "I" mode.
      - We reinterpret the bits as float16, then cast to float32.

    Args:
        depth_png (str):
            File path to the 16-bit PNG.

    Returns:
        np.ndarray:
            A float32 depth array of shape (H, W).
    """
    with Image.open(depth_png) as depth_pil:
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


def normalize_image(
    image: Union[np.ndarray, Tensor], target_height: Optional[int] = None, target_width: Optional[int] = None
) -> np.ndarray:
    """
    Standardizes an image to HWC uint8 format (0-255) from numpy array or tensor.

    Args:
        image: Input image as numpy array or tensor
        target_height: Optional height to resize to
        target_width: Optional width to resize to

    Returns:
        np.ndarray: HWC uint8 RGB image with values 0-255
    """
    # Handle tensors
    if isinstance(image, Tensor):
        img = image.detach().cpu().numpy()
    # Handle numpy arrays
    else:
        img = image.copy()  # Create a copy to avoid modifying the original

    # Convert CHW to HWC if needed
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)  # CHW -> HWC

    # Ensure 3 channels (RGB)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.concatenate([img] * 3, axis=2)

    # Normalize float values to 0-255 range
    if img.dtype in [np.float32, np.float64]:
        if np.max(np.abs(img)) <= 1.0:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        # Convert any other integer types to uint8
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Resize if dimensions provided
    if target_height is not None and target_width is not None:
        current_h, current_w = img.shape[:2]
        if current_h != target_height or current_w != target_width:
            img = cv2.resize(img, (target_width, target_height))

    return img


def normalize_depth(
    depth: Union[np.ndarray, Tensor],
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
    scale_adjustment: float = 1.0,
) -> np.ndarray:
    """
    Standardizes a depth map to HW float32 format from numpy array or tensor.

    Args:
        depth: Input depth as numpy array or tensor
        target_height: Optional height to resize to
        target_width: Optional width to resize to
        scale_adjustment: Optional scale factor for depth values

    Returns:
        np.ndarray: HW float32 depth map
    """
    # Handle tensors
    if isinstance(depth, Tensor):
        depth_map = depth.detach().cpu().numpy()
    # Handle numpy arrays
    else:
        depth_map = depth.copy()  # Create a copy to avoid modifying the original

    # Ensure 2D
    if depth_map.ndim == 3:
        if depth_map.shape[0] == 1:  # CHW format
            depth_map = depth_map.squeeze(0)
        elif depth_map.shape[-1] == 1:  # HWC format
            depth_map = depth_map.squeeze(-1)
        elif depth_map.shape[0] == 3 and depth_map.shape[2] > 3:  # Possibly RGB depth image
            # Convert to grayscale if it looks like an RGB image
            depth_map = np.mean(depth_map, axis=0) if depth_map.shape[0] == 3 else depth_map

    # Convert to float32
    depth_map = depth_map.astype(np.float32)

    # Apply scale adjustment if needed
    if scale_adjustment != 1.0:
        depth_map = depth_map * scale_adjustment

    # Handle NaN and Inf values
    depth_map[~np.isfinite(depth_map)] = 0.0

    # Resize if needed
    if target_height is not None and target_width is not None:
        current_h, current_w = depth_map.shape[:2]
        if current_h != target_height or current_w != target_width:
            depth_map = cv2.resize(depth_map, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    return depth_map
