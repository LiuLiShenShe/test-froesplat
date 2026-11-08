# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DA3 model wrapper for dvlt evaluation.

Eval-only wrapper around the upstream `depth_anything_3` package
(https://github.com/ByteDance-Seed/Depth-Anything-3). All custom DA3 net /
head / refinement code has been removed for the public release.
"""

import os
from typing import Any, Dict

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger

from dvlt.common.amp import force_fp32
from dvlt.common.constants import DataField, PredictionField
from dvlt.common.geometry import depth_to_world_coords_points
from dvlt.common.pose import inverse_pose, to4x4
from dvlt.model.base import Module
from dvlt.struct.util import extri_intri_to_cameras


# Suppress DA3 INFO logs - must be set before importing depth_anything_3.
os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")

try:
    from depth_anything_3.model.da3 import DepthAnything3Net
except ImportError as e:
    raise ImportError(
        "DepthAnything3 baseline package not found. Install it: "
        "`pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git --no-deps`"
    ) from e


logger = get_logger(__name__)

# Keys consumed by upstream `DepthAnything3Net.__init__`. Anything else in `da3_cfg`
# (e.g. `use_ray_pose`, `ref_view_strategy`, `use_cam_enc_at_test`) is handled by
# the wrapper itself.
_NET_KWARGS = {"net", "head", "cam_dec", "cam_enc", "gs_head", "gs_adapter"}

# Upstream DA3's `DepthAnything3Net.forward` does NOT normalize images — its
# `InputProcessor` (utils/io/input_processor.py) applies ImageNet normalization
# at preprocessing time. Our dvlt data pipeline emits images in [0, 1], so we
# apply the same normalization here.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize_scene_for_cam_enc(extrinsics_c2w: torch.Tensor) -> torch.Tensor:
    """Normalize GT cameras to match `cam_enc`'s training distribution.

    Upstream DA3 trains `cam_enc` on scenes pre-normalized by the data
    pipeline (`normalize_scene=True`): all poses re-framed to the first
    camera and translations divided by a scene scale factor. At eval time
    our batches are not normalized, so we apply the same transform here.

    The scale factor is the **mean inter-camera translation magnitude** —
    pose-only, no GT depth used (so we don't leak ground-truth depth into the
    model's conditioning). This matches the depth-free fallback in Pi3X's
    internal pose normalization.

    Args:
        extrinsics_c2w: (B, S, 3/4, 4) c2w extrinsics. Must have S >= 2.

    Returns:
        (B, S, 4, 4) normalized c2w extrinsics.
    """
    if extrinsics_c2w.shape[-2] == 3:
        extrinsics_c2w = to4x4(extrinsics_c2w)
    B, S = extrinsics_c2w.shape[:2]
    device, dtype = extrinsics_c2w.device, extrinsics_c2w.dtype

    # 1) Re-frame: c2w' = inv(c2w[0]) @ c2w  (first camera at identity)
    first_cam_w2c = inverse_pose(extrinsics_c2w[:, 0:1])  # (B, 1, 4, 4)
    transformed = first_cam_w2c @ extrinsics_c2w  # (B, S, 4, 4)

    # 2) Mean inter-camera translation magnitude per batch element.
    if S > 1:
        scale_factors = transformed[:, 1:, :3, 3].norm(dim=-1).mean(dim=-1)  # (B,)
        scale_factors = scale_factors.clamp(min=1e-3, max=1e3)
    else:
        scale_factors = torch.ones(B, device=device, dtype=dtype)

    # 3) Apply scale to translations.
    out = transformed.clone()
    out[..., :3, 3] = out[..., :3, 3] / scale_factors.view(B, 1, 1)
    return out


class DA3(Module):
    """DA3 model wrapper for dvlt evaluation.

    Example usage:
        python -m dvlt.scripts.test --config-name da3-giant.yaml \
            trainer.ckpt_dir=depth-anything/DA3-GIANT-1.1

    Args:
        da3_cfg: DA3 model configuration. Keys consumed by the upstream
            `DepthAnything3Net.__init__` (`net`, `head`, `cam_dec`, `cam_enc`,
            `gs_head`, `gs_adapter`) build the model. The remaining keys
            understood by the wrapper:
              - `use_ray_pose` (bool, default True): derive camera pose from
                the ray head instead of the camera decoder.
              - `ref_view_strategy` (str, default "saddle_balanced"): reference
                view selection when num_views >= 3.
              - `use_cam_enc_at_test` (bool, default False): condition the
                forward pass on GT cameras from the batch (requires `cam_enc`
                to be configured).
              - `world_points_from_rays` (bool, default False): compute
                ``WORLD_POINTS`` directly as ``ray_origin + ray_direction *
                depth`` from the predicted (global, unnormalized) rays and
                depth, matching the training-style pointmap target. The DA3
                ray head emits at sub-resolution (~4x the patch grid, e.g.
                148x148 for a 518² input), so the rays are bilinearly
                upsampled to ``(H, W)`` before the multiply. When True, both
                ``WORLD_POINTS`` and ``WORLD_POINTS_DIRECT`` carry this
                rays+depth value (DA3 has no separate model-emitted direct
                pointmap to keep on the side). When False (default), keeps the
                current behavior: ``WORLD_POINTS`` from depth + fitted
                extrinsics/intrinsics, and ``WORLD_POINTS_DIRECT`` from the
                upstream output's own ``world_points`` (when present).
    """

    def __init__(self, *args, da3_cfg: Dict[str, Any], **kwargs):
        self.cfg = da3_cfg
        super().__init__(*args, **kwargs)
        self.model_file = "model.safetensors"
        # ImageNet stats for input normalization. Registered as non-persistent
        # buffers so they ride the model device but aren't saved in checkpoints.
        self.model.register_buffer("_img_mean", torch.tensor(_IMAGENET_MEAN).view(1, 1, 3, 1, 1), persistent=False)
        self.model.register_buffer("_img_std", torch.tensor(_IMAGENET_STD).view(1, 1, 3, 1, 1), persistent=False)
        if self.cfg.get("world_points_from_rays", False):
            self._install_ray_capture()

    def load_pretrained(self, pretrained_model_name_or_path: str, **kwargs: Any) -> None:
        if pretrained_model_name_or_path.startswith("depth-anything/"):
            # HF DA3 checkpoints save under a top-level `model.` namespace; strip
            # it so keys match `self.model.state_dict()` (which is the upstream
            # DepthAnything3Net's own keys: `backbone.*`, `head.*`, ...).
            kwargs.setdefault("remap", {})[r"^model\."] = ""
            kwargs["strict"] = False
        # Drop Gaussian-splat weights only when the corresponding submodule
        # wasn't built. Upstream `DepthAnything3Net.__init__` initializes
        # `self.gs_head, self.gs_adapter = None, None` and only re-assigns when
        # both are configured in `da3_cfg`, so a `None` attr is a reliable
        # signal that those keys are unexpected in the target state_dict.
        filter_patterns = list(kwargs.pop("filter", None) or [])
        if getattr(self.model, "gs_head", None) is None and r"^gs_head\." not in filter_patterns:
            filter_patterns.append(r"^gs_head\.")
        if getattr(self.model, "gs_adapter", None) is None and r"^gs_adapter\." not in filter_patterns:
            filter_patterns.append(r"^gs_adapter\.")
        if filter_patterns:
            kwargs["filter"] = filter_patterns
        super().load_pretrained(pretrained_model_name_or_path, **kwargs)

    def _transform_state_dict(self, state_dict: dict) -> dict:
        """Duplicate level-0 LayerNorm weights to alias levels 1/2/3.

        Upstream `DualDPT` builds `output_conv2_aux` as a `ModuleList` of four
        `nn.Sequential`s that all unpack the **same** `nn.LayerNorm` instance
        via `*ln_seq`, so `model.state_dict()` exposes that single shared tensor
        under four alias paths (`...0.2.*`, `...1.2.*`, `...2.2.*`, `...3.2.*`).
        HF DA3 checkpoints dedupe and store only the level-0 entry, which causes
        `load_state_dict(strict=False)` to emit cosmetic "missing key" warnings
        for the other three. Duplicating the level-0 entry into the alias paths
        before load suppresses those warnings (the underlying parameter is
        shared, so the values are identical either way).
        """
        src_prefix = "head.scratch.output_conv2_aux.0.2."
        expanded = dict(state_dict)
        for key in list(state_dict.keys()):
            if src_prefix in key:
                for level in (1, 2, 3):
                    alias = key.replace(src_prefix, f"head.scratch.output_conv2_aux.{level}.2.")
                    expanded.setdefault(alias, state_dict[key])
        return expanded

    def build_model(self):
        net_kwargs = {k: v for k, v in self.cfg.items() if k in _NET_KWARGS}
        return DepthAnything3Net(**net_kwargs)

    def _install_ray_capture(self) -> None:
        """Preserve ``output.ray`` / ``output.ray_conf`` past upstream's deletes.

        Upstream `DepthAnything3Net._process_ray_pose_estimation` (when
        `use_ray_pose=True`) and `_process_camera_estimation` (when a `cam_dec`
        is present) both `del output.ray` / `del output.ray_conf` after using
        them. We need them downstream for the rays+depth pointmap path, so we
        wrap each method to copy them onto the output as `pred_ray` /
        `pred_ray_conf` before the originals are dropped. The wrappers are
        installed as instance attributes so the base class behavior is
        untouched and other DA3 instances are unaffected.
        """
        model = self.model
        orig_ray_pose = model._process_ray_pose_estimation
        orig_cam_est = model._process_camera_estimation

        def _ray_pose_keep(output, height, width):
            ray = output.get("ray", None)
            ray_conf = output.get("ray_conf", None)
            result = orig_ray_pose(output, height, width)
            if ray is not None:
                result.pred_ray = ray
            if ray_conf is not None:
                result.pred_ray_conf = ray_conf
            return result

        def _cam_est_keep(feats, H, W, output):
            ray = output.get("ray", None)
            ray_conf = output.get("ray_conf", None)
            result = orig_cam_est(feats, H, W, output)
            if ray is not None:
                result.pred_ray = ray
            if ray_conf is not None:
                result.pred_ray_conf = ray_conf
            return result

        model._process_ray_pose_estimation = _ray_pose_keep
        model._process_camera_estimation = _cam_est_keep

    @staticmethod
    def _rays_to_world_points(pred_ray: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Compose world points from predicted rays and depth.

        DA3's ray head emits at sub-resolution (~4x the patch grid), while the
        depth head is at full image resolution. Bilinearly upsample the rays
        to match the depth resolution (``align_corners=True``, matching DA3's
        own normalized identity-ray grid convention) and apply
        ``points = ray_origin + ray_direction * depth`` — the training-style
        pointmap target. The 6-channel ray layout is
        ``[direction (xyz), origin (xyz)]`` with unnormalized direction whose
        z-component encodes the camera-plane depth scale (matching
        ``compute_world_rays``).
        """
        B, S, _, _, C = pred_ray.shape
        assert C == 6, f"pred_ray must have 6 channels (dir + origin), got {C}"
        H, W = depth.shape[-2:]
        rays_full = (
            F.interpolate(
                pred_ray.permute(0, 1, 4, 2, 3).reshape(B * S, C, *pred_ray.shape[2:4]),
                size=(H, W),
                mode="bilinear",
                align_corners=True,
            )
            .reshape(B, S, C, H, W)
            .permute(0, 1, 3, 4, 2)
        )
        ray_dir = rays_full[..., :3]
        ray_origin = rays_full[..., 3:6]
        return ray_origin + ray_dir * depth.unsqueeze(-1)

    @torch.no_grad()
    def predict(self, batch: dict, accelerator: Accelerator) -> dict:
        # Apply ImageNet normalization (upstream's `InputProcessor` does this at
        # preprocessing time; our batch images live in [0, 1]).
        images = (batch[DataField.IMAGES] - self.model._img_mean) / self.model._img_std

        # Camera input conditioning (default off). Two extra steps before the
        # forward when conditioning is on:
        #   1) Scene-normalize GT poses to match `cam_enc`'s training
        #      distribution (re-frame to first cam + scale by point/pose
        #      magnitudes; mirrors Pi3X's pose normalization).
        #   2) Convert c2w -> w2c, since upstream `CameraEnc.forward` expects
        #      w2c (it inverts back to c2w internally).
        extrinsics_w2c = None
        intrinsics = None
        if self.cfg.get("use_cam_enc_at_test", False):
            extr_c2w = batch.get(DataField.EXTRINSICS_C2W)
            if extr_c2w is not None:
                extrinsics_w2c = inverse_pose(_normalize_scene_for_cam_enc(extr_c2w))
            intrinsics = batch.get(DataField.INTRINSICS)

        output = self.model(
            images,
            extrinsics=extrinsics_w2c,
            intrinsics=intrinsics,
            use_ray_pose=bool(self.cfg.get("use_ray_pose", True)),
            ref_view_strategy=self.cfg.get("ref_view_strategy", "saddle_balanced"),
        )
        return self._postprocess_predictions(batch, output)

    @force_fp32
    def _postprocess_predictions(self, batch: dict, output) -> dict:
        H, W = batch[DataField.IMAGES].shape[-2:]

        # Upstream stores extrinsics as **w2c** in both branches (3x4 in
        # `use_ray_pose` mode, 4x4 in cam_dec mode; cf. depth_anything_3's own
        # GLB / COLMAP exporters which label and consume them as w2c). Our
        # downstream consumers (`extri_intri_to_cameras`, `depth_to_world_coords_points`)
        # expect c2w, so we normalize to 4x4 and invert here.
        extr_w2c = output.extrinsics
        if extr_w2c.shape[-2] == 3:
            extr_w2c = to4x4(extr_w2c)
        extrinsics_c2w = inverse_pose(extr_w2c)
        intr = output.intrinsics

        cameras = [extri_intri_to_cameras(e, i, (H, W)) for e, i in zip(extrinsics_c2w, intr)]

        depth = output.depth
        depth_conf = output.depth_conf

        preds = {
            PredictionField.CAMERAS: cameras,
            PredictionField.DEPTHS: depth,
            PredictionField.DEPTHS_CONF: depth_conf,
        }

        pred_ray = output.get("pred_ray", None) if self.cfg.get("world_points_from_rays", False) else None
        if pred_ray is not None:
            # Direct rays + depth pointmap (training-style). Both
            # ``WORLD_POINTS`` and ``WORLD_POINTS_DIRECT`` carry this — DA3 has
            # no separate model-emitted direct pointmap to keep on the side.
            with torch.autocast("cuda", enabled=False):
                world_points = self._rays_to_world_points(pred_ray, depth)
            preds[PredictionField.WORLD_POINTS] = world_points
            preds[PredictionField.WORLD_POINTS_DIRECT] = world_points
            preds[PredictionField.WORLD_POINTS_DIRECT_CONF] = depth_conf
        else:
            with torch.autocast("cuda", enabled=False):
                world_points, _, _ = depth_to_world_coords_points(depth, extrinsics_c2w, intr)
            preds[PredictionField.WORLD_POINTS] = world_points
            if "world_points" in output:
                preds[PredictionField.WORLD_POINTS_DIRECT] = output.world_points
                preds[PredictionField.WORLD_POINTS_DIRECT_CONF] = output.world_points_conf

        return preds

    def train_step(self, *args, **kwargs):
        raise NotImplementedError("DA3 does not support training within dvlt")
