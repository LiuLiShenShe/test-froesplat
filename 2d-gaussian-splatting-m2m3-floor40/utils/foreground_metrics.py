"""Foreground-object reconstruction metrics for ForeSplat / M2M3-GO.

Full-image PSNR against the raw RGB frame is not a valid metric for a
foreground-object model whose background is intentionally black/empty.  This
module implements the manuscript's foreground criteria:

    PSNR_fg, SSIM_fg, LPIPS_fg        (inside the foreground prior)
    outside_mask_non_black_ratio      (background leakage area)
    leakage_energy_ratio              (background energy / foreground energy)

All functions accept [C,H,W] or [1,C,H,W] tensors in [0,1] on any device and
return Python floats, so they can be called from training, evaluation or tests.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-8


def _as_single(image: torch.Tensor) -> torch.Tensor:
    return image.unsqueeze(0) if image.dim() == 3 else image


def _as_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    return mask


def boundary_band(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Return the boundary band of a mask (dilate - erode) as a boolean tensor."""
    mask = _as_mask(mask)
    if radius <= 0:
        return torch.zeros_like(mask, dtype=torch.bool)
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(mask, kernel, stride=1, padding=radius)
    eroded = -F.max_pool2d(-mask, kernel, stride=1, padding=radius)
    return (dilated - eroded) > 0.5


def foreground_weight(mask: torch.Tensor, boundary_ignore: int = 0) -> torch.Tensor:
    weight = _as_mask(mask)
    if boundary_ignore > 0:
        weight = weight * (~boundary_band(mask, boundary_ignore)).float()
    return weight


def foreground_psnr(render: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                    boundary_ignore: int = 0) -> float:
    render = _as_single(render).float()
    target = _as_single(target).float()
    weight = foreground_weight(mask, boundary_ignore).to(render.device)
    squared = (render - target) ** 2
    denom = weight.sum().clamp_min(1.0) * render.shape[1]
    mse = (squared * weight).sum() / denom
    if mse <= 0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / float(mse)))


def foreground_ssim(render: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                    boundary_ignore: int = 0) -> float:
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return float("nan")
    render = _as_single(render).float().clamp(0.0, 1.0)
    target = _as_single(target).float().clamp(0.0, 1.0)
    weight = foreground_weight(mask, boundary_ignore)[0, 0].cpu().numpy() > 0.5
    if weight.sum() < 16:
        return float("nan")
    render_np = render[0].permute(1, 2, 0).cpu().numpy()
    target_np = target[0].permute(1, 2, 0).cpu().numpy()
    # SSIM needs at least 7x7 in each spatial dimension.
    if render_np.shape[0] < 7 or render_np.shape[1] < 7:
        return float("nan")
    _, similarity = structural_similarity(
        target_np, render_np, channel_axis=-1, data_range=1.0, full=True)
    if similarity.ndim == 3:
        similarity = similarity.mean(axis=-1)
    return float(similarity[weight].mean())


_LPIPS_IMPL: dict = {"resolved": False, "name": None, "call": None}


def _prepare_lpips(net_type: str) -> tuple[str | None, object]:
    """Resolve a working LPIPS implementation and cache it.

    Preference order:
      1. the vendored ``lpipsPyTorch`` used by the 2DGS baseline (expects [0,1])
      2. the ``lpips`` pip package (expects [-1,1])

    The vendored copy in this workspace is corrupted on disk (its
    ``modules/lpips.py`` contains log text, not Python), so the fallback is not
    hypothetical -- it is the backend that actually runs here.  The returned
    name is written into the evaluation JSON so every LPIPS number is traceable
    to the implementation that produced it.
    """
    if _LPIPS_IMPL["resolved"] and _LPIPS_IMPL["name"]:
        return _LPIPS_IMPL["name"], _LPIPS_IMPL["call"]
    _LPIPS_IMPL["resolved"] = True
    try:
        from lpipsPyTorch import lpips as vendored

        def call(render, target):
            return float(vendored(render, target, net_type=net_type).mean().item())

        _LPIPS_IMPL.update({"name": "lpipsPyTorch", "call": call})
        return "lpipsPyTorch", call
    except Exception:
        pass
    try:
        import lpips as lpips_pkg

        model = lpips_pkg.LPIPS(net=net_type, verbose=False)
        state = {"device": None}

        def call(render, target):
            # pip `lpips` expects inputs in [-1, 1]; ours are in [0, 1]
            if state["device"] != render.device:
                model.to(render.device)
                state["device"] = render.device
            value = model(render * 2.0 - 1.0, target * 2.0 - 1.0)
            return float(value.mean().item())

        _LPIPS_IMPL.update({"name": f"lpips-pip({net_type})", "call": call})
        return _LPIPS_IMPL["name"], call
    except Exception:
        _LPIPS_IMPL.update({"name": None, "call": None})
        return None, None


def lpips_backend_name(net_type: str = "vgg") -> str | None:
    """Name of the LPIPS implementation that would be used, or None."""
    name, _ = _prepare_lpips(net_type)
    return name


def foreground_lpips(render: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                     boundary_ignore: int = 0, net_type: str = "vgg") -> float:
    """LPIPS on the foreground crop with the background zeroed.

    LPIPS is optional: returns NaN if no LPIPS backend or weights are
    available, never a fabricated value.
    """
    backend, call = _prepare_lpips(net_type)
    if call is None:
        return float("nan")
    render = _as_single(render).float().clamp(0.0, 1.0)
    target = _as_single(target).float().clamp(0.0, 1.0)
    weight = foreground_weight(mask, boundary_ignore).to(render.device)
    coords = torch.nonzero(weight[0, 0] > 0.5, as_tuple=False)
    if coords.numel() == 0:
        return float("nan")
    y0, x0 = coords.min(dim=0).values.tolist()
    y1, x1 = coords.max(dim=0).values.tolist()
    pad = 4
    y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
    y1 = min(render.shape[-2] - 1, y1 + pad); x1 = min(render.shape[-1] - 1, x1 + pad)
    render = render[:, :, y0:y1 + 1, x0:x1 + 1] * weight[:, :, y0:y1 + 1, x0:x1 + 1]
    target = target[:, :, y0:y1 + 1, x0:x1 + 1] * weight[:, :, y0:y1 + 1, x0:x1 + 1]
    try:
        return float(call(render, target))
    except Exception:
        return float("nan")


def outside_mask_non_black_ratio(render: torch.Tensor, mask: torch.Tensor,
                                 black_threshold: float = 0.02) -> float:
    render = _as_single(render).float()
    weight = _as_mask(mask).to(render.device)
    outside = (1.0 - weight)
    count = float(outside.sum().item())
    if count <= 0:
        return 0.0
    pixel_max = (render * outside).max(dim=1, keepdim=True).values
    non_black = ((pixel_max > black_threshold).float() * outside).sum()
    return float((non_black / count).item())


def leakage_energy_ratio(render: torch.Tensor, target: torch.Tensor,
                         mask: torch.Tensor) -> float:
    """Background rendered energy divided by foreground target energy."""
    render = _as_single(render).float()
    target = _as_single(target).float()
    weight = _as_mask(mask).to(render.device)
    foreground_energy = (target ** 2 * weight).sum() / (weight.sum() * render.shape[1]).clamp_min(1.0)
    background = 1.0 - weight
    background_energy = (render ** 2 * background).sum() / (background.sum() * render.shape[1]).clamp_min(1.0)
    return float((background_energy / foreground_energy.clamp_min(EPS)).item())


def compute_foreground_metrics(render: torch.Tensor, target: torch.Tensor,
                               mask: torch.Tensor, boundary_ignore: int = 2,
                               with_lpips: bool = False) -> dict:
    metrics = {
        "psnr_fg": foreground_psnr(render, target, mask, boundary_ignore),
        "ssim_fg": foreground_ssim(render, target, mask, boundary_ignore),
        "outside_mask_non_black_ratio": outside_mask_non_black_ratio(render, mask),
        "leakage_energy_ratio": leakage_energy_ratio(render, target, mask),
        "foreground_fraction": float(_as_mask(mask).float().mean().item()),
    }
    if with_lpips:
        metrics["lpips_fg"] = foreground_lpips(render, target, mask, boundary_ignore)
    return metrics
