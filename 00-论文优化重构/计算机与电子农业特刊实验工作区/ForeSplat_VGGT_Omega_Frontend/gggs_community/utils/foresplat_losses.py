"""ForeSplat foreground losses for GGGS (faithful port of the active 2DGS code).

Every function here is ported from the ForeSplat downstream training code that
was actually used for the R4bm05 / E4b runs
(``2d-gaussian-splatting-m2m3-floor40/train.py`` + ``utils/plant_aware_utils.py``):

* ``rgb_reconstruction_loss`` - foreground-weighted photometric loss with a
  mask-bounding-box crop for the SSIM term (``--use_foreground_rgb_loss``,
  ``--lambda_fg_rgb``, ``--foreground_bg_rgb_weight``,
  ``--foreground_rgb_crop_padding``).
* ``mask_loss_terms`` - alpha/mask loss (``l1_dice`` by default) and the
  background-opacity loss, both ramped in from ``mask_loss_start_iter`` over
  ``mask_loss_warmup_iters`` (``--use_mask_loss / --lambda_mask``,
  ``--use_bg_opacity_loss / --lambda_bg``).

Nothing else is added: no normal/distortion terms belong here, and the
manuscript modules that were never enabled in the actual runs (M2M3-GO, view
weighting) are deliberately *not* ported.

Normalisation notes (for the paper):
* the mask loss ``l1`` term is a ``masked_mean`` over the stable region, i.e.
  divided by the number of pixels actually used - it does not scale with
  foreground area;
* the dice term is normalised by ``sum(pred) + sum(target)``;
* the foreground RGB loss is a weighted mean with denominator
  ``sum(weight) * 3``, so a larger plant does not change the loss scale.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from utils.loss_utils import ssim


# --------------------------------------------------------------------------- #
# mask / region helpers (utils/plant_aware_utils.py)
# --------------------------------------------------------------------------- #
def erode_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return -F.max_pool2d(-mask[None], kernel_size=kernel, stride=1, padding=radius)[0]


def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return F.max_pool2d(mask[None], kernel_size=kernel, stride=1, padding=radius)[0]


def stable_mask_region(mask: torch.Tensor, boundary_px: int) -> torch.Tensor:
    """1 where the pixel is at least ``boundary_px`` away from the mask edge."""
    if boundary_px <= 0:
        return torch.ones_like(mask)
    dilated = dilate_mask(mask, boundary_px)
    eroded = erode_mask(mask, boundary_px)
    return ((dilated - eroded) < 0.5).float()


def masked_mean(value: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    if weight is None:
        return value.mean()
    denom = weight.sum().clamp_min(1.0)
    return (value * weight).sum() / denom


def weighted_rgb_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    denom = weight.sum().clamp_min(1.0) * value.shape[0]
    return (value * weight).sum() / denom


def ramp_weight(iteration: int, start_iter: int, warmup_iters: int) -> float:
    if iteration < start_iter:
        return 0.0
    if warmup_iters <= 0:
        return 1.0
    return min(1.0, float(iteration - start_iter + 1) / float(warmup_iters))


def crop_to_mask_bbox(image, gt_image, weight, padding):
    if padding < 0:
        return image, gt_image, weight
    mask_2d = weight[0] > 0.0
    coords = torch.nonzero(mask_2d, as_tuple=False)
    if coords.numel() == 0:
        return image, gt_image, weight
    y0 = max(int(coords[:, 0].min().item()) - padding, 0)
    y1 = min(int(coords[:, 0].max().item()) + padding + 1, image.shape[-2])
    x0 = max(int(coords[:, 1].min().item()) - padding, 0)
    x1 = min(int(coords[:, 1].max().item()) + padding + 1, image.shape[-1])
    return image[:, y0:y1, x0:x1], gt_image[:, y0:y1, x0:x1], weight[:, y0:y1, x0:x1]


# --------------------------------------------------------------------------- #
# losses (2d-gaussian-splatting-m2m3-floor40/train.py)
# --------------------------------------------------------------------------- #
def rgb_reconstruction_loss(image, gt_image, target_mask, opt):
    """ForeSplat foreground-weighted photometric loss.

    Identical to the 2DGS implementation: weighted L1 over the stable
    foreground (and optionally a small background weight) plus an SSIM term
    computed inside the mask bounding box with ``foreground_rgb_crop_padding``.
    """
    if not opt.use_foreground_rgb_loss:
        raise ValueError("foreground RGB loss requested without --use_foreground_rgb_loss")
    if target_mask is None:
        raise ValueError("Foreground RGB loss was enabled but this camera has no loaded mask.")

    target_mask = target_mask.to(image.device).float()
    if target_mask.dim() == 2:
        target_mask = target_mask[None]
    stable_region = stable_mask_region(target_mask, opt.mask_ignore_boundary_px).to(image.device)
    fg_weight = target_mask * stable_region
    bg_weight = (1.0 - target_mask) * stable_region * max(0.0, opt.foreground_bg_rgb_weight)
    rgb_weight = fg_weight + bg_weight

    ll1 = weighted_rgb_mean(torch.abs(image - gt_image), rgb_weight)
    crop_image, crop_gt, crop_weight = crop_to_mask_bbox(
        image, gt_image, rgb_weight, int(opt.foreground_rgb_crop_padding)
    )
    crop_image = crop_image * crop_weight
    crop_gt = crop_gt * crop_weight
    ssim_loss = 1.0 - ssim(crop_image[None], crop_gt[None])
    loss = ((1.0 - opt.lambda_dssim) * ll1 + opt.lambda_dssim * ssim_loss) * opt.lambda_fg_rgb
    return loss, ll1


def mask_loss_terms(render_alpha, target_mask, opt, iteration):
    """Mask (alpha) loss + background-opacity loss, with the warm-up ramp."""
    if target_mask is None:
        if opt.use_mask_loss or opt.use_bg_opacity_loss:
            raise ValueError("Mask loss was enabled but this camera has no loaded mask.")
        zero = render_alpha.sum() * 0.0
        return zero, zero, 0.0

    target_mask = target_mask.to(render_alpha.device).float()
    if target_mask.dim() == render_alpha.dim() - 1:
        target_mask = target_mask[None]
    stable_region = stable_mask_region(target_mask, opt.mask_ignore_boundary_px).to(render_alpha.device)
    schedule = ramp_weight(iteration, opt.mask_loss_start_iter, opt.mask_loss_warmup_iters)

    mask_loss = render_alpha.sum() * 0.0
    if opt.use_mask_loss and opt.lambda_mask > 0.0 and schedule > 0.0:
        if opt.mask_loss_type == "bce":
            mask_loss_raw = F.binary_cross_entropy(
                render_alpha.clamp(1e-4, 1.0 - 1e-4), target_mask, reduction="none"
            )
            mask_loss = masked_mean(mask_loss_raw, stable_region) * opt.lambda_mask * schedule
        elif opt.mask_loss_type == "dice":
            pred = render_alpha * stable_region
            target = target_mask * stable_region
            denom = pred.sum() + target.sum() + 1e-6
            dice = 1.0 - (2.0 * (pred * target).sum() + 1e-6) / denom
            mask_loss = dice * opt.lambda_mask * schedule
        elif opt.mask_loss_type == "l1_dice":
            l1 = masked_mean((render_alpha - target_mask).abs(), stable_region)
            pred = render_alpha * stable_region
            target = target_mask * stable_region
            denom = pred.sum() + target.sum() + 1e-6
            dice = 1.0 - (2.0 * (pred * target).sum() + 1e-6) / denom
            mask_loss = (0.5 * l1 + 0.5 * dice) * opt.lambda_mask * schedule
        else:
            mask_loss = masked_mean((render_alpha - target_mask).abs(), stable_region) * opt.lambda_mask * schedule

    bg_loss = render_alpha.sum() * 0.0
    if opt.use_bg_opacity_loss and opt.lambda_bg > 0.0 and schedule > 0.0:
        background = (1.0 - target_mask) * stable_region
        bg_loss = masked_mean(render_alpha, background) * opt.lambda_bg * schedule

    return mask_loss, bg_loss, schedule
