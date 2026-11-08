"""ForeSplat mask-guided pruning for GGGS (faithful port of the active code).

Ported from the ForeSplat downstream training code that produced the R4bm05 /
E4b runs (``2d-gaussian-splatting-m2m3-floor40/train.py``):

* ``mask_consistency_stats`` -> (M_j, V_j): foreground-mask projection hit rate
  and number of sampled views that see the primitive;
* ``multi_cue_survival`` -> the manuscript's §2.6 survival score
  ``P_keep = sigmoid(w_g . z_j - b_g)`` over the five cues
  ``[M_j, log O_j, log(1+V_j), B_j, C_j]`` (default ``pruning_survival_mode =
  multi_cue``, which is what the E4b recipe used);
* ``maybe_prune_gaussians`` -> the mask-mode pruning round with the
  ``pruning_max_remove_ratio`` budget.

The legacy ``legacy_mask`` mode (plain threshold on M_j) is kept so results can
be reproduced bitwise, but the experiments use the defaults above.

Capacity-control / M2M3-GO modules are **not** ported: they were separate
ablations, never part of the four ForeSplat-GGGS comparison.
"""

from __future__ import annotations

import json
import math
import os

import torch

SH_C0 = 0.28209479177387814


def gaussian_brightness(gaussians) -> torch.Tensor:
    """B_j: mean RGB implied by the SH DC term, clamped to [0, 1]."""
    dc = gaussians._features_dc.detach().squeeze(1)
    rgb = torch.clamp(dc * SH_C0 + 0.5, 0.0, 1.0)
    return rgb.mean(dim=1)


def topology_cue(xyz, grid_size=64, reference="mean"):
    """C_j: local topological support (own-grid-cell density vs occupied mean)."""
    n = int(xyz.shape[0])
    if n == 0:
        return torch.zeros((0,), dtype=torch.float32, device=xyz.device)
    g = max(int(grid_size), 1)
    lo = xyz.min(dim=0).values
    hi = xyz.max(dim=0).values
    span = (hi - lo).clamp_min(1e-6)
    grid = ((xyz - lo) / span * g).floor().clamp(0, g - 1).to(torch.long)
    keys = grid[:, 0] * g * g + grid[:, 1] * g + grid[:, 2]
    ids, inverse = torch.unique(keys, return_inverse=True)
    counts = torch.bincount(inverse, minlength=int(ids.numel())).to(torch.float32)
    occupied = counts[counts > 0]
    if occupied.numel() == 0:
        return torch.ones((n,), dtype=torch.float32, device=xyz.device)
    if str(reference) == "max":
        ref = float(occupied.max().item())
    elif str(reference) == "median":
        ref = float(occupied.median().item())
    else:
        ref = float(occupied.mean().item())
    ref = max(ref, 1.0)
    cue = (counts / ref).clamp(0.0, 1.0)
    return cue[inverse]


def _mask_consistency_core(gaussians, cameras, max_views):
    """Hits / visible counts of each Gaussian centre inside the foreground prior."""
    mask_cameras = [cam for cam in cameras if getattr(cam, "gt_mask", None) is not None]
    if not mask_cameras:
        return None, None, 0
    if max_views > 0:
        stride = max(1, len(mask_cameras) // max_views)
        mask_cameras = mask_cameras[::stride][:max_views]

    xyz_h = torch.cat([gaussians.get_xyz.detach(), torch.ones_like(gaussians.get_xyz[:, :1])], dim=1)
    hits = torch.zeros((gaussians.get_xyz.shape[0],), dtype=torch.float32, device="cuda")
    visible = torch.zeros_like(hits)

    for cam in mask_cameras:
        projected = xyz_h @ cam.full_proj_transform
        z = projected[:, 3]
        pix = projected[:, :2] / z[:, None].clamp_min(1e-6)
        in_view = (z > 0) & (pix[:, 0] > -1.0) & (pix[:, 0] < 1.0) & (pix[:, 1] > -1.0) & (pix[:, 1] < 1.0)
        if not torch.any(in_view):
            continue
        mask = cam.gt_mask.to("cuda")
        if mask.dim() == 2:
            mask = mask[None]
        sampled = torch.nn.functional.grid_sample(
            mask[None],
            pix[None, None],
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(-1)
        visible += in_view.float()
        hits += (sampled > 0.5).float() * in_view.float()
    return hits, visible, len(mask_cameras)


def mask_consistency_stats(gaussians, cameras, max_views):
    """(M_j, V_j) for the manuscript §2.6 pruning descriptor."""
    hits, visible, n_sampled = _mask_consistency_core(gaussians, cameras, max_views)
    if hits is None:
        return None, None, 0
    valid = visible > 0
    scores = torch.ones_like(hits)
    scores[valid] = hits[valid] / visible[valid].clamp_min(1.0)
    return scores, visible, n_sampled


def mask_consistency_scores(gaussians, cameras, max_views):
    hits, visible, _ = _mask_consistency_core(gaussians, cameras, max_views)
    if hits is None:
        return None
    valid = visible > 0
    scores = torch.ones_like(hits)
    scores[valid] = hits[valid] / visible[valid].clamp_min(1.0)
    return scores


def multi_cue_survival(gaussians, opt, mask_scores, visible_views, n_sampled_views, xyz=None):
    """P_keep(g_j) = sigmoid(w_g . z_j - b_g), manuscript §2.6 (verbatim port)."""
    eps = 1e-8
    xyz = gaussians.get_xyz.detach() if xyz is None else xyz
    if mask_scores is None:
        mask_scores = torch.ones((xyz.shape[0],), dtype=torch.float32, device=xyz.device)
    if visible_views is None:
        visible_views = torch.zeros((xyz.shape[0],), dtype=torch.float32, device=xyz.device)
    opacity = gaussians.get_opacity.detach().squeeze().clamp(eps, 1.0 - eps)
    if opacity.dim() > 1:
        opacity = opacity.reshape(-1)
    brightness = gaussian_brightness(gaussians).detach().float().clamp(0.0, 1.0).reshape(-1)
    topology = topology_cue(
        xyz,
        int(getattr(opt, "pruning_topology_grid", 64)),
        str(getattr(opt, "pruning_topology_reference", "mean")),
    )

    tau_m = float(opt.pruning_mask_threshold)
    tau_o = float(opt.pruning_opacity_threshold)
    log_o = torch.log(opacity + eps)
    log_o_ref = math.log(max(tau_o + eps, eps))
    scale_o = max(-log_o_ref, eps)
    z_m = (mask_scores.float() - tau_m) / max(1.0 - tau_m, eps)
    z_o = ((log_o - log_o_ref) / scale_o).clamp(-1.0, 1.0)
    v_ref = float(visible_views.max().item()) if visible_views.numel() else 0.0
    if v_ref <= 0.0:
        v_ref = float(max(int(n_sampled_views), 1))
    log_v = torch.log1p(visible_views.float().clamp_min(0.0))
    z_v = (log_v - math.log1p(v_ref)) / max(math.log1p(v_ref), eps)
    med_b = float(brightness.median().item())
    scale_b = max(med_b, 1.0 - med_b, eps)
    tau_b = float(opt.pruning_brightness_threshold)
    abs_deficit = (tau_b - brightness) / max(tau_b, eps)
    rel_deficit = (med_b - brightness) / scale_b
    z_b = (-torch.maximum(abs_deficit, rel_deficit)).clamp(-1.0, 1.0)
    med_c = float(topology.median().item())
    scale_c = max(med_c, float(topology.mean().item()), eps)
    z_c = ((topology - med_c) / scale_c).clamp(-1.0, 1.0)

    weights = [
        float(v)
        for v in str(getattr(opt, "pruning_cue_weights", "3.0,1.0,0.25,1.0,0.5"))
        .replace(";", ",")
        .split(",")
        if str(v).strip()
    ]
    if len(weights) != 5:
        raise ValueError("--pruning_cue_weights requires 5 comma-separated values [M, logO, logV, B, C]")
    logit = (
        weights[0] * z_m
        + weights[1] * z_o
        + weights[2] * z_v
        + weights[3] * z_b
        + weights[4] * z_c
        - float(getattr(opt, "pruning_keep_bias", 0.0))
    )
    keep = torch.sigmoid(logit)
    cues = {
        "M": mask_scores.float(),
        "O": opacity,
        "V": visible_views.float(),
        "B": brightness,
        "C": topology,
        "z_M": z_m,
        "z_O": z_o,
        "z_V": z_v,
        "z_B": z_b,
        "z_C": z_c,
        "logit": logit,
    }
    return keep, cues


def select_top_prune_candidates(prune_mask, score, max_remove):
    max_remove = int(max_remove)
    if max_remove <= 0:
        return torch.zeros_like(prune_mask)
    requested = int(prune_mask.sum().item())
    if requested <= max_remove:
        return prune_mask
    candidate_score = score.clone()
    candidate_score[~prune_mask] = -float("inf")
    topk = torch.topk(candidate_score, k=max_remove, largest=True).indices
    limited = torch.zeros_like(prune_mask)
    limited[topk] = True
    return torch.logical_and(prune_mask, limited)


def maybe_prune_gaussians(gaussians, opt, iteration, model_path, cameras=None, report_path=None):
    """Mask-mode pruning round (multi_cue or legacy_mask), as in the 2DGS code."""
    if opt.pruning_mode == "none":
        return None
    if iteration < opt.pruning_start_iter or iteration % opt.pruning_interval != 0:
        return None

    before = int(gaussians.get_xyz.shape[0])
    prune_mask = torch.zeros((before,), dtype=torch.bool, device="cuda")
    prune_score = torch.zeros((before,), dtype=torch.float32, device="cuda")
    survival_mode = str(getattr(opt, "pruning_survival_mode", "multi_cue"))
    survival_report = {}

    if survival_mode == "multi_cue":
        mask_scores, visible_views, n_sampled_views = mask_consistency_stats(
            gaussians, cameras or [], opt.pruning_mask_max_views
        )
        survival_scores, cues = multi_cue_survival(gaussians, opt, mask_scores, visible_views, n_sampled_views)
        tau_keep = float(getattr(opt, "pruning_survival_threshold", 0.45))
        prune_mask = survival_scores < tau_keep
        prune_score = -survival_scores.detach().float()
        survival_report = {
            "survival_mode": "multi_cue",
            "survival_threshold": tau_keep,
            "survival_mean": float(survival_scores.mean().item()),
            "survival_cue_mean": {
                key: float(value.mean().item()) for key, value in cues.items() if key.startswith("z_")
            },
        }
    else:
        if opt.pruning_mode in ("opacity", "topology"):
            removed = gaussians.get_opacity.detach().squeeze() < opt.pruning_opacity_threshold
            prune_mask = torch.logical_or(prune_mask, removed)
            prune_score += (opt.pruning_opacity_threshold - gaussians.get_opacity.detach().squeeze()).clamp_min(0.0)
        if opt.pruning_mode in ("brightness", "topology"):
            brightness = gaussian_brightness(gaussians)
            removed = brightness < opt.pruning_brightness_threshold
            prune_mask = torch.logical_or(prune_mask, removed)
            prune_score += (opt.pruning_brightness_threshold - brightness).clamp_min(0.0)
        if opt.pruning_mode in ("mask", "topology") and cameras:
            mask_scores = mask_consistency_scores(gaussians, cameras, opt.pruning_mask_max_views)
            if mask_scores is not None:
                removed = mask_scores < opt.pruning_mask_threshold
                prune_mask = torch.logical_or(prune_mask, removed)
                prune_score += (opt.pruning_mask_threshold - mask_scores).clamp_min(0.0) * max(
                    0.0, opt.pruning_mask_score_weight
                )
        survival_report = {"survival_mode": "legacy_mask"}

    requested = int(prune_mask.sum().item())
    max_remove = int(before * max(0.0, min(float(opt.pruning_max_remove_ratio), 1.0)))
    if max_remove > 0 and requested > max_remove:
        prune_mask = select_top_prune_candidates(prune_mask, prune_score, max_remove)

    removed = int(prune_mask.sum().item())
    if removed > 0:
        gaussians.prune_points(prune_mask)
        # GGGS' prune_points does not touch the mip-splatting 3D filter; a
        # standalone pruning round (outside densify_and_prune) must keep
        # filter_3D aligned with the Gaussian tensors or the next render fails.
        filter_3d = getattr(gaussians, "filter_3D", None)
        if filter_3d is not None and filter_3d.shape[0] == before:
            gaussians.filter_3D = filter_3d[~prune_mask]
    after = int(gaussians.get_xyz.shape[0])

    report = {
        "iteration": iteration,
        "mode": opt.pruning_mode,
        "gaussians_before": before,
        "gaussians_after": after,
        "removed": removed,
        "requested": requested,
        "max_remove": max_remove,
        **survival_report,
    }
    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        records = []
        if os.path.isfile(report_path):
            try:
                with open(report_path, encoding="utf-8") as handle:
                    records = json.load(handle)
            except (OSError, json.JSONDecodeError):
                records = []
        records.append(report)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
    return report
