#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import json
import csv
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.plant_aware_utils import masked_mean, ramp_weight, stable_mask_region
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def validate_plant_aware_args(dataset, opt):
    if (
        opt.use_mask_loss
        or opt.use_bg_opacity_loss
        or opt.use_foreground_rgb_loss
        or opt.lambda_mask > 0.0
        or opt.lambda_bg > 0.0
    ):
        if dataset.mask_mode == "none" or not dataset.mask_dir:
            raise ValueError(
                "Plant-aware foreground losses require masks. Set --mask_mode alpha/preprocess and --mask_dir <path>."
            )
    if dataset.mask_mode in ("preprocess", "alpha") and not dataset.mask_dir:
        raise ValueError("--mask_mode preprocess/alpha requires --mask_dir <path>.")
    if opt.view_weight_mode != "none" and not opt.view_weight_list:
        raise ValueError("--view_weight_mode requires --view_weight_list <csv/txt>.")
    if opt.capacity_control_mode not in ("none", "m2m3", "m2m3_floor"):
        raise ValueError("--capacity_control_mode must be one of: none, m2m3, m2m3_floor.")
    if opt.capacity_control_mode != "none" and opt.pruning_mode == "none":
        raise ValueError("--capacity_control_mode requires --pruning_mode to be enabled.")
    if opt.capacity_floor_reference not in ("initial", "max_seen", "current"):
        raise ValueError("--capacity_floor_reference must be one of: initial, max_seen, current.")
    if not 0.0 <= float(opt.capacity_floor_ratio) <= 1.0:
        raise ValueError("--capacity_floor_ratio must be in [0, 1].")
    if opt.m2m3_score_mode not in ("legacy", "topology"):
        raise ValueError("--m2m3_score_mode must be one of: legacy, topology.")
    if opt.m2m3_region_mode not in ("foreground", "all"):
        raise ValueError("--m2m3_region_mode must be one of: foreground, all.")


def enabled_path(path):
    return path is not None and str(path).strip() not in ("", "none", "None")


def normalize_image_key(name):
    base = os.path.basename(str(name).strip())
    stem, _ = os.path.splitext(base)
    return stem


def read_view_weights(path, min_weight, max_weight, default_weight):
    weights = {}
    if not enabled_path(path):
        return weights

    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = "image_name" in sample.splitlines()[0] or "weight" in sample.splitlines()[0]
        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("image_name") or row.get("name") or row.get("image")
                value = row.get("weight") or row.get("view_weight")
                if name is None or value is None:
                    continue
                weight = float(value)
                weights[normalize_image_key(name)] = min(max(weight, min_weight), max_weight)
        else:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) < 2:
                    continue
                weight = float(parts[1])
                weights[normalize_image_key(parts[0])] = min(max(weight, min_weight), max_weight)

    print(
        f"[Plant-aware] loaded {len(weights)} soft view weights from {path} "
        f"(range={min_weight:.3f}-{max_weight:.3f}, default={default_weight:.3f})"
    )
    return weights


def get_view_weight(view_weights, viewpoint_cam, default_weight):
    if not view_weights:
        return 1.0
    key = normalize_image_key(viewpoint_cam.image_name)
    return float(view_weights.get(key, default_weight))


def weighted_rgb_mean(value, weight):
    denom = weight.sum().clamp_min(1.0) * value.shape[0]
    return (value * weight).sum() / denom


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


def rgb_reconstruction_loss(image, gt_image, target_mask, opt):
    if not opt.use_foreground_rgb_loss:
        ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        return loss, ll1

    if target_mask is None:
        raise ValueError("Foreground RGB loss was enabled but this camera has no loaded mask.")

    target_mask = target_mask.to(image.device).float()
    stable_region = stable_mask_region(target_mask, opt.mask_ignore_boundary_px).to(image.device)
    fg_weight = target_mask * stable_region
    bg_weight = (1.0 - target_mask) * stable_region * max(0.0, opt.foreground_bg_rgb_weight)
    rgb_weight = fg_weight + bg_weight

    ll1 = weighted_rgb_mean(torch.abs(image - gt_image), rgb_weight)
    crop_image, crop_gt, crop_weight = crop_to_mask_bbox(
        image,
        gt_image,
        rgb_weight,
        int(opt.foreground_rgb_crop_padding),
    )
    crop_image = crop_image * crop_weight
    crop_gt = crop_gt * crop_weight
    ssim_loss = 1.0 - ssim(crop_image, crop_gt)
    loss = ((1.0 - opt.lambda_dssim) * ll1 + opt.lambda_dssim * ssim_loss) * opt.lambda_fg_rgb
    return loss, ll1


def mask_loss_terms(render_alpha, target_mask, opt, iteration):
    if target_mask is None:
        if opt.use_mask_loss or opt.use_bg_opacity_loss:
            raise ValueError("Mask loss was enabled but this camera has no loaded mask.")
        zero = render_alpha.sum() * 0.0
        return zero, zero, 0.0

    target_mask = target_mask.to(render_alpha.device).float()
    stable_region = stable_mask_region(target_mask, opt.mask_ignore_boundary_px).to(render_alpha.device)
    schedule = ramp_weight(iteration, opt.mask_loss_start_iter, opt.mask_loss_warmup_iters)

    mask_loss = render_alpha.sum() * 0.0
    if opt.use_mask_loss and opt.lambda_mask > 0.0 and schedule > 0.0:
        if opt.mask_loss_type == "bce":
            mask_loss_raw = torch.nn.functional.binary_cross_entropy(
                render_alpha.clamp(1e-4, 1.0 - 1e-4),
                target_mask,
                reduction="none",
            )
            mask_loss = masked_mean(mask_loss_raw, stable_region) * opt.lambda_mask * schedule
        elif opt.mask_loss_type == "dice":
            weight = stable_region
            pred = render_alpha * weight
            target = target_mask * weight
            denom = pred.sum() + target.sum() + 1e-6
            dice = 1.0 - (2.0 * (pred * target).sum() + 1e-6) / denom
            mask_loss = dice * opt.lambda_mask * schedule
        elif opt.mask_loss_type == "l1_dice":
            l1 = masked_mean((render_alpha - target_mask).abs(), stable_region)
            weight = stable_region
            pred = render_alpha * weight
            target = target_mask * weight
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


def gaussian_brightness(gaussians):
    dc = gaussians._features_dc.detach().squeeze(1)
    rgb = torch.clamp(dc * 0.28209479177387814 + 0.5, 0.0, 1.0)
    return rgb.mean(dim=1)


def capacity_control_enabled(opt):
    return opt.capacity_control_mode in ("m2m3", "m2m3_floor")


def init_capacity_state(gaussians):
    initial_count = int(gaussians.get_xyz.shape[0])
    return {
        "initial_count": initial_count,
        "max_seen_count": initial_count,
        "total_requested": 0,
        "total_removed": 0,
        "total_blocked_by_floor": 0,
        "rounds": 0,
        "floor_active_rounds": 0,
    }


def update_capacity_state(capacity_state, current_count):
    if capacity_state is None:
        return
    capacity_state["max_seen_count"] = max(int(capacity_state.get("max_seen_count", 0)), int(current_count))


def normalized_positive(values):
    values = values.detach().float().clamp_min(0.0)
    if values.numel() == 0:
        return values
    denom = values.max().clamp_min(1e-6)
    return values / denom


def m2m3_prune_scores(gaussians, opt, legacy_score, mask_scores=None):
    if opt.m2m3_score_mode == "legacy":
        return legacy_score.detach().float()

    score = torch.zeros_like(legacy_score, dtype=torch.float32)
    opacity = gaussians.get_opacity.detach().squeeze()
    opacity_penalty = (float(opt.pruning_opacity_threshold) - opacity).clamp_min(0.0)
    score += normalized_positive(opacity_penalty) * max(0.0, float(opt.m2m3_opacity_weight))

    brightness = gaussian_brightness(gaussians)
    brightness_penalty = (float(opt.pruning_brightness_threshold) - brightness).clamp_min(0.0)
    score += normalized_positive(brightness_penalty) * max(0.0, float(opt.m2m3_brightness_weight))

    if mask_scores is not None and opt.m2m3_region_mode == "foreground":
        mask_penalty = (float(opt.pruning_mask_threshold) - mask_scores).clamp_min(0.0)
        score += normalized_positive(mask_penalty) * max(0.0, float(opt.m2m3_mask_weight))

    scales = gaussians.get_scaling.detach().max(dim=1).values
    scale_penalty = (scales - scales.median()).clamp_min(0.0)
    score += normalized_positive(scale_penalty) * max(0.0, float(opt.m2m3_scale_weight))

    screen_penalty = gaussians.max_radii2D.detach().float().clamp_min(0.0)
    score += normalized_positive(screen_penalty) * max(0.0, float(opt.m2m3_view_weight))
    return score


def capacity_floor_is_active(opt, iteration):
    start_iter = int(opt.capacity_floor_start_iter) if int(opt.capacity_floor_start_iter) > 0 else int(opt.pruning_start_iter)
    end_iter = int(opt.capacity_floor_end_iter) if int(opt.capacity_floor_end_iter) > 0 else int(opt.iterations)
    return start_iter <= int(iteration) <= end_iter


def resolve_capacity_floor_count(opt, capacity_state, current_count):
    if int(opt.capacity_floor_count) > 0:
        return min(int(opt.capacity_floor_count), int(current_count))
    if opt.capacity_floor_reference == "initial":
        reference_count = int(capacity_state.get("initial_count", current_count))
    elif opt.capacity_floor_reference == "current":
        reference_count = int(current_count)
    else:
        reference_count = int(capacity_state.get("max_seen_count", current_count))
    floor_count = int(round(reference_count * float(opt.capacity_floor_ratio)))
    return min(max(floor_count, 0), int(current_count))


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


def save_capacity_report(model_path, report):
    capacity_dir = os.path.join(model_path, "capacity_control")
    os.makedirs(capacity_dir, exist_ok=True)
    with open(os.path.join(capacity_dir, f"capacity_iter_{report['iteration']}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def should_save_capacity_report(opt, iteration):
    interval = int(opt.capacity_report_interval)
    return interval <= 0 or int(iteration) % interval == 0 or int(iteration) == int(opt.iterations)


def save_capacity_summary(model_path, summary):
    capacity_dir = os.path.join(model_path, "capacity_control")
    os.makedirs(capacity_dir, exist_ok=True)
    with open(os.path.join(capacity_dir, "capacity_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def mask_consistency_scores(gaussians, cameras, max_views):
    mask_cameras = [cam for cam in cameras if cam.gt_alpha_mask is not None]
    if not mask_cameras:
        return None
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
        sampled = torch.nn.functional.grid_sample(
            cam.gt_alpha_mask.to("cuda")[None],
            pix[None, None],
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(-1)
        visible += in_view.float()
        hits += (sampled > 0.5).float() * in_view.float()

    valid = visible > 0
    scores = torch.ones_like(hits)
    scores[valid] = hits[valid] / visible[valid].clamp_min(1.0)
    return scores


def maybe_prune_gaussians(gaussians, opt, iteration, model_path, cameras=None, capacity_state=None):
    if opt.pruning_mode == "none":
        return None
    if iteration < opt.pruning_start_iter or iteration % opt.pruning_interval != 0:
        return None

    before = int(gaussians.get_xyz.shape[0])
    update_capacity_state(capacity_state, before)
    prune_mask = torch.zeros((before,), dtype=torch.bool, device="cuda")
    removed_by_opacity = torch.zeros_like(prune_mask)
    removed_by_brightness = torch.zeros_like(prune_mask)
    removed_by_mask = torch.zeros_like(prune_mask)
    mask_scores = None
    prune_score = torch.zeros((before,), dtype=torch.float32, device="cuda")

    if opt.pruning_mode in ("opacity", "topology"):
        removed_by_opacity = (gaussians.get_opacity.detach().squeeze() < opt.pruning_opacity_threshold)
        prune_mask = torch.logical_or(prune_mask, removed_by_opacity)
        prune_score += (opt.pruning_opacity_threshold - gaussians.get_opacity.detach().squeeze()).clamp_min(0.0)
    if opt.pruning_mode in ("brightness", "topology"):
        brightness = gaussian_brightness(gaussians)
        removed_by_brightness = brightness < opt.pruning_brightness_threshold
        prune_mask = torch.logical_or(prune_mask, removed_by_brightness)
        prune_score += (opt.pruning_brightness_threshold - brightness).clamp_min(0.0)
    if opt.pruning_mode in ("mask", "topology") and cameras is not None:
        mask_scores = mask_consistency_scores(gaussians, cameras, opt.pruning_mask_max_views)
        if mask_scores is not None:
            removed_by_mask = mask_scores < opt.pruning_mask_threshold
            prune_mask = torch.logical_or(prune_mask, removed_by_mask)
            mask_penalty = (opt.pruning_mask_threshold - mask_scores).clamp_min(0.0)
            prune_score += mask_penalty * max(0.0, opt.pruning_mask_score_weight)

    requested = int(prune_mask.sum().item())
    if requested <= 0:
        report = {
            "iteration": iteration,
            "mode": opt.pruning_mode,
            "capacity_control_mode": opt.capacity_control_mode,
            "gaussians_before": before,
            "gaussians_after": before,
            "removed": 0,
        }
        if capacity_control_enabled(opt) and opt.save_capacity_report and should_save_capacity_report(opt, iteration):
            save_capacity_report(model_path, report)
        return report

    capacity_report = {}
    if capacity_control_enabled(opt):
        m2m3_score = m2m3_prune_scores(gaussians, opt, prune_score, mask_scores)
        max_remove = int(before * max(0.0, min(float(opt.m2m3_max_remove_ratio), 1.0)))
        if max_remove > 0:
            prune_mask = select_top_prune_candidates(prune_mask, m2m3_score, max_remove)
        after_budget = int(prune_mask.sum().item())

        floor_active = opt.capacity_control_mode == "m2m3_floor" and capacity_floor_is_active(opt, iteration)
        floor_count = 0
        allowed_remove = before
        blocked_by_floor = 0
        if floor_active:
            floor_count = resolve_capacity_floor_count(opt, capacity_state or {}, before)
            allowed_remove = max(0, before - floor_count)
            if after_budget > allowed_remove:
                prune_mask = select_top_prune_candidates(prune_mask, m2m3_score, allowed_remove)
            blocked_by_floor = max(0, after_budget - int(prune_mask.sum().item()))

        capacity_report = {
            "m2m3_requested": requested,
            "m2m3_after_budget": after_budget,
            "m2m3_budget_blocked": max(0, requested - after_budget),
            "m2m3_score_mode": opt.m2m3_score_mode,
            "m2m3_region_mode": opt.m2m3_region_mode,
            "floor_active": floor_active,
            "floor_count": floor_count,
            "floor_allowed_remove": allowed_remove,
            "floor_blocked": blocked_by_floor,
            "capacity_floor_ratio": float(opt.capacity_floor_ratio),
            "capacity_floor_reference": opt.capacity_floor_reference,
        }
        if capacity_state is not None:
            capacity_state["rounds"] += 1
            capacity_state["total_requested"] += requested
            capacity_state["total_blocked_by_floor"] += blocked_by_floor
            if floor_active:
                capacity_state["floor_active_rounds"] += 1
    else:
        max_remove = int(before * max(0.0, min(opt.pruning_max_remove_ratio, 1.0)))
        if max_remove > 0 and requested > max_remove:
            prune_mask = select_top_prune_candidates(prune_mask, prune_score, max_remove)

    removed = int(prune_mask.sum().item())
    if removed <= 0:
        report = {
            "iteration": iteration,
            "mode": opt.pruning_mode,
            "capacity_control_mode": opt.capacity_control_mode,
            "gaussians_before": before,
            "gaussians_after": before,
            "removed": 0,
            **capacity_report,
        }
        if capacity_control_enabled(opt) and opt.save_capacity_report and should_save_capacity_report(opt, iteration):
            save_capacity_report(model_path, report)
        return report
    gaussians.prune_points(prune_mask)
    after = int(gaussians.get_xyz.shape[0])
    if capacity_state is not None and capacity_control_enabled(opt):
        capacity_state["total_removed"] += removed
    report = {
        "iteration": iteration,
        "mode": opt.pruning_mode,
        "capacity_control_mode": opt.capacity_control_mode,
        "gaussians_before": before,
        "gaussians_after": after,
        "removed": removed,
        "pruning_ratio": removed / max(before, 1),
        "removed_by_opacity": int(torch.logical_and(prune_mask, removed_by_opacity).sum().item()),
        "removed_by_brightness": int(torch.logical_and(prune_mask, removed_by_brightness).sum().item()),
        "removed_by_mask": int(torch.logical_and(prune_mask, removed_by_mask).sum().item()),
        **capacity_report,
    }
    if capacity_state is not None and capacity_control_enabled(opt):
        report["capacity_state"] = dict(capacity_state)
    if mask_scores is not None:
        report["mask_score_mean_removed"] = float(mask_scores[prune_mask].mean().item()) if removed > 0 else None
        report["mask_score_min_removed"] = float(mask_scores[prune_mask].min().item()) if removed > 0 else None
    if opt.save_pruning_report:
        pruning_dir = os.path.join(model_path, "pruning")
        os.makedirs(pruning_dir, exist_ok=True)
        with open(os.path.join(pruning_dir, f"pruning_iter_{iteration}.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    if capacity_control_enabled(opt) and opt.save_capacity_report and should_save_capacity_report(opt, iteration):
        save_capacity_report(model_path, report)
    return report


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    validate_plant_aware_args(dataset, opt)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    view_weights = read_view_weights(
        opt.view_weight_list,
        float(opt.view_weight_min),
        float(opt.view_weight_max),
        float(opt.view_weight_default),
    ) if opt.view_weight_mode != "none" else {}
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    capacity_state = init_capacity_state(gaussians) if capacity_control_enabled(opt) else None
    if capacity_state is not None:
        print(
            "[M2M3] capacity control enabled: "
            f"mode={opt.capacity_control_mode}, floor_ratio={float(opt.capacity_floor_ratio):.3f}, "
            f"reference={opt.capacity_floor_reference}, initial_count={capacity_state['initial_count']}"
        )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_mask_for_log = 0.0
    ema_bg_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        loss, Ll1 = rgb_reconstruction_loss(image, gt_image, viewpoint_cam.gt_alpha_mask, opt)
        view_weight = get_view_weight(view_weights, viewpoint_cam, opt.view_weight_default)
        loss = loss * view_weight
        Ll1 = Ll1 * view_weight
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()
        mask_loss, bg_loss, mask_schedule = mask_loss_terms(
            render_pkg["rend_alpha"],
            viewpoint_cam.gt_alpha_mask,
            opt,
            iteration,
        )
        if opt.view_weight_mode == "all_losses":
            mask_loss = mask_loss * view_weight
            bg_loss = bg_loss * view_weight

        # loss
        total_loss = loss + dist_loss + normal_loss + mask_loss + bg_loss
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_mask_for_log = 0.4 * mask_loss.item() + 0.6 * ema_mask_for_log
            ema_bg_for_log = 0.4 * bg_loss.item() + 0.6 * ema_bg_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                if opt.use_mask_loss or opt.use_bg_opacity_loss or opt.use_foreground_rgb_loss:
                    loss_dict["mask"] = f"{ema_mask_for_log:.{5}f}"
                    loss_dict["bg"] = f"{ema_bg_for_log:.{5}f}"
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                if opt.use_mask_loss or opt.use_bg_opacity_loss or opt.use_foreground_rgb_loss:
                    tb_writer.add_scalar('train_loss_patches/mask_loss', ema_mask_for_log, iteration)
                    tb_writer.add_scalar('train_loss_patches/bg_opacity_loss', ema_bg_for_log, iteration)
                    tb_writer.add_scalar('train_loss_patches/mask_schedule', mask_schedule, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            update_capacity_state(capacity_state, int(gaussians.get_xyz.shape[0]))

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            pruning_report = maybe_prune_gaussians(
                gaussians,
                opt,
                iteration,
                scene.model_path,
                scene.getTrainCameras(),
                capacity_state=capacity_state,
            )
            if pruning_report is not None and pruning_report.get("removed", 0) > 0:
                pruning_label = "M2M3 capacity pruning" if capacity_control_enabled(opt) else "Plant-aware pruning"
                print(
                    "\n[ITER {}] {} removed {}/{} Gaussians ({:.2%})".format(
                        iteration,
                        pruning_label,
                        pruning_report["removed"],
                        pruning_report["gaussians_before"],
                        pruning_report["pruning_ratio"],
                    )
                )
                if iteration in saving_iterations:
                    print("\n[ITER {}] Re-saving Gaussians after {}".format(iteration, pruning_label))
                    scene.save(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

    if capacity_state is not None and opt.save_capacity_report:
        final_count = int(gaussians.get_xyz.shape[0])
        update_capacity_state(capacity_state, final_count)
        summary = {
            "capacity_control_mode": opt.capacity_control_mode,
            "capacity_floor_ratio": float(opt.capacity_floor_ratio),
            "capacity_floor_reference": opt.capacity_floor_reference,
            "capacity_floor_count_override": int(opt.capacity_floor_count),
            "capacity_floor_start_iter": int(opt.capacity_floor_start_iter) if int(opt.capacity_floor_start_iter) > 0 else int(opt.pruning_start_iter),
            "capacity_floor_end_iter": int(opt.capacity_floor_end_iter) if int(opt.capacity_floor_end_iter) > 0 else int(opt.iterations),
            "m2m3_score_mode": opt.m2m3_score_mode,
            "m2m3_region_mode": opt.m2m3_region_mode,
            "m2m3_max_remove_ratio": float(opt.m2m3_max_remove_ratio),
            "initial_count": int(capacity_state.get("initial_count", 0)),
            "max_seen_count": int(capacity_state.get("max_seen_count", 0)),
            "final_count": final_count,
            "total_requested": int(capacity_state.get("total_requested", 0)),
            "total_removed": int(capacity_state.get("total_removed", 0)),
            "total_blocked_by_floor": int(capacity_state.get("total_blocked_by_floor", 0)),
            "rounds": int(capacity_state.get("rounds", 0)),
            "floor_active_rounds": int(capacity_state.get("floor_active_rounds", 0)),
        }
        save_capacity_summary(scene.model_path, summary)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
