"""Standalone VGGT-Omega inference (no Gradio dependency).

Replicates the community ``demo_gradio.py`` inference path exactly:

1. ``load_model``      - build :class:`VGGTOmega` and load the released
   checkpoint (``strict=False`` as in the community demo);
2. ``run_inference``   - forward pass + ``encoding_to_camera`` to obtain
   extrinsic / intrinsic matrices + depth back-projection point maps.

The returned ``predictions`` dict mirrors the community demo:

======================  =====================================================
key                     content
======================  =====================================================
``pose_enc``            (N, 9) camera encoding
``extrinsic``           (N, 3, 4) world-to-camera (OpenCV convention)
``intrinsic``           (N, 3, 3) at *model* resolution
``depth``               (N, H, W, 1) metric depth (exp of logits, always > 0)
``depth_conf``          (N, H, W) confidence, >= 1.0 (1 + exp(logits))
``images``              (N, 3, H, W) actual model input in [0, 1]
``world_points_from_depth`` (N, H, W, 3) depth back-projection in world space
======================  =====================================================
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera


def load_model(checkpoint_path: str, device: str | None = None) -> VGGTOmega:
    """Load the released VGGT-Omega checkpoint exactly like the community demo."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VGGTOmega(device=device).eval()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    return model.to(device)


def run_inference(model: VGGTOmega, images: torch.Tensor, device: str | None = None) -> dict:
    """Run VGGT-Omega on a (1, N, 3, H, W) or (N, 3, H, W) image tensor.

    Returns the community-style ``predictions`` dict with numpy tensors
    (batch dimension removed) plus ``extrinsic``, ``intrinsic`` and
    ``world_points_from_depth``.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    t_start = time.time()
    with torch.inference_mode():
        if device == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                predictions = model(images.to(device))
        else:
            predictions = model(images.to(device))

    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    predictions_np = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            predictions_np[key] = value

    predictions_np["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions_np["depth"],
        predictions_np["extrinsic"],
        predictions_np["intrinsic"],
    )

    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"[run_inference] finished in {time.time() - t_start:.2f} s "
          f"(depth: {predictions_np['depth'].shape}, conf range: "
          f"{float(predictions_np['depth_conf'].min()):.3f}..{float(predictions_np['depth_conf'].max()):.3f}, "
          f"depth range: {float(predictions_np['depth'].min()):.3f}..{float(predictions_np['depth'].max()):.3f})")
    return predictions_np


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray
) -> np.ndarray:
    """Back-project per-pixel depth into world points (identical to demo_gradio.py)."""
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )
