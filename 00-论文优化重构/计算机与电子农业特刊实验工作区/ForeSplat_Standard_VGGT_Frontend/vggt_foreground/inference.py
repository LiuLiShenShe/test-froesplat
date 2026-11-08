"""Standard VGGT inference (aggregator + camera_head + depth_head only).

Mirrors ``demo_colmap.py::run_VGGT`` exactly: load at 1024, interpolate to 518,
run the aggregator/camera/depth heads under bf16 autocast, unproject with the
official ``unproject_depth_map_to_point_map``.  No point head, no track head,
no BA.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


def load_model(checkpoint_path: str, device: str = "cuda") -> VGGT:
    model = VGGT()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    return model.to(device).eval()


def run_vggt_heads(model, images_load: torch.Tensor, model_res: int = 518,
                   device: str = "cuda") -> dict:
    """images_load: (N,3,load_res,load_res) on device. Returns per-frame numpy arrays."""
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.inference_mode():
        img_model = F.interpolate(images_load, size=(model_res, model_res), mode="bilinear", align_corners=False).unsqueeze(0)  # (1,N,3,518,518)
        # Official run_VGGT: ONLY the aggregator runs under autocast; camera_head,
        # pose conversion and depth_head run outside it in fp32. (Running the pose
        # conversion inside bf16 autocast makes quat_to_mat ~4e-3 non-orthonormal,
        # which injects a |t|-scaled reprojection error of O(1) px.)
        with torch.autocast(device_type="cuda", dtype=dtype):
            tokens, ps_idx = model.aggregator(img_model)
        pose_enc = model.camera_head(tokens)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, img_model.shape[-2:])
        depth_map, depth_conf = model.depth_head(tokens, img_model, ps_idx)
        torch.cuda.synchronize()
    # squeeze batch dim
    extrinsic = extrinsic.squeeze(0).float().cpu().numpy()        # (N,3,4) or (N,4,4)
    intrinsic = intrinsic.squeeze(0).float().cpu().numpy()        # (N,3,3)
    depth = depth_map.squeeze(0).float().cpu().numpy()            # (N,518,518,1)
    conf = depth_conf.squeeze(0).float().cpu().numpy()            # (N,518,518)
    return {"extrinsic": extrinsic, "intrinsic": intrinsic, "depth": depth, "conf": conf}


def world_points_from_depth(depth, extrinsic, intrinsic):
    return unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
