"""Mask-aware standard-VGGT frontend (add-on; official VGGT code is untouched).

Coordinate model (built in ``build_affine``, tested in ``verify``):

    p_model518 = A p_original        K_original = A^{-1} K_model518

``A`` composes three steps:
  1. center-pad the original image to a square (max_dim x max_dim, black bg);
  2. bicubic-resize the square to 1024x1024 (model load resolution);
  3. ``run_VGGT`` bilinear-resizes 1024x1024 -> 518x518 (model input).

For a portrait image (W_orig=2160, H_orig=3840) max_dim=3840, so
``s = 518 / max_dim`` and ``A = [[s,0,pad_left*s],[0,s,pad_top*s],[0,0,1]]``.

NOTE: the official ``rename_colmap_recons_and_rescale_camera`` shifts point2D by
``(xy - original_coords[:2]) * max_dim/img_size`` where ``original_coords`` is in
**1024-space** but ``xy`` is a **518-space** point.  That is correct only when the
padding offset is zero (square inputs).  For padded inputs it is wrong, so we use
the exact affine ``A^{-1}`` everywhere instead.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def compute_pad(W: int, H: int) -> tuple[int, int, int]:
    """Center-pad ``W x H`` to a square; return (max_dim, pad_left, pad_top)."""
    max_dim = max(W, H)
    pad_left = (max_dim - W) // 2
    pad_top = (max_dim - H) // 2
    return max_dim, pad_left, pad_top


def build_affine(W: int, H: int, model_res: int = 518) -> tuple[np.ndarray, np.ndarray]:
    """Build ``A`` (original px -> model518 px) and ``A_inv`` (model518 px -> original px)."""
    max_dim, pad_left, pad_top = compute_pad(W, H)
    s = model_res / max_dim
    A = np.array([[s, 0.0, pad_left * s],
                  [0.0, s, pad_top * s],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    A_inv = np.array([[1.0 / s, 0.0, -pad_left],
                      [0.0, 1.0 / s, -pad_top],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
    return A, A_inv


def apply_affine(M: np.ndarray, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return (M[:2, :2] @ xy.T + M[:2, 2:3]).T


def load_and_preprocess_images_square_with_masks(
    image_path_list, mask_path_list, load_res: int = 1024, model_res: int = 518,
    mask_threshold: int = 127,
):
    """Load RGB + mask with IDENTICAL padding; RGB bicubic, mask NEAREST, mask pad = 0.

    RGB is composited on a BLACK background if RGBA (matches the VGGT-Omega frontend;
    the official loader uses white, which would break the black-background convention).

    Returns:
        images (N,3,load_res,load_res) float [0,1]  (input to run_VGGT, which
              interpolates to model_res internally),
        masks518 (N,model_res,model_res) bool  foreground mask at MODEL resolution,
        affines518 (N,3,3) float64 ``A`` mapping original px -> model518 px,
        original_sizes (N,2) int64 (W,H).
    """
    from torchvision import transforms as TF
    to_tensor = TF.ToTensor()

    images, masks518, affines518, sizes = [], [], [], []
    for img_path, mask_path in zip(image_path_list, mask_path_list):
        img = Image.open(img_path)
        if img.mode == "RGBA":  # black bg to match VGGT-Omega (official uses white)
            bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
            img = Image.alpha_composite(bg, img)
        img = img.convert("RGB")
        W, H = img.size

        max_dim, pad_left, pad_top = compute_pad(W, H)
        A518, _ = build_affine(W, H, model_res)
        affines518.append(A518)
        sizes.append((W, H))

        square = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))  # black pad
        square.paste(img, (pad_left, pad_top))
        square = square.resize((load_res, load_res), Image.Resampling.BICUBIC)
        images.append(to_tensor(square))

        m = Image.open(mask_path).convert("L")
        msq = Image.new("L", (max_dim, max_dim), 0)  # mask pad = 0 (background)
        msq.paste(m, (pad_left, pad_top))
        m518 = msq.resize((model_res, model_res), Image.Resampling.NEAREST)
        masks518.append(torch.from_numpy(np.array(m518)).float().div_(255.0).ge_(mask_threshold / 255.0))

    return (torch.stack(images), torch.stack(masks518),
            torch.from_numpy(np.stack(affines518)).double(), np.array(sizes, dtype=np.int64))
