"""ForeSplat-GGGS experiment training script (GGGS + optional EO / Foreground).

Structure:
* everything about the GGGS core (Gaussian representation, densification,
  mip-splatting 3D filter, multi-view NCC / depth-normal regularisation,
  photometric loss, optimiser) is copied verbatim from ``gggs_community/train.py``;
* the only additions are the two experiment factors, each with an explicit
  switch:
  * ``ExposureCompensator`` (``--use_decoupled_appearance 1``) - GGGS' native
    per-view 3x4 exposure, applied before the photometric loss;
  * ForeSplat foreground losses (``utils/foresplat_losses.py``) + mask-guided
    pruning (``utils/foresplat_pruning.py``), ported from the active ForeSplat
    downstream code.

Four configurations (E1..E4) are produced by the same script with different
flag combinations, so every shared hyper-parameter is identical by construction
and the differences are exactly {EO on/off} x {Foreground on/off}.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from argparse import ArgumentParser
from random import randint

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

GGGS_DIR = os.path.dirname(os.path.abspath(__file__))
if GGGS_DIR not in sys.path:
    sys.path.insert(0, GGGS_DIR)
from utils.cuda_preload import preload_cuda_ext_deps  # noqa: E402

preload_cuda_ext_deps()

from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from models.exposure_compensator import ExposureCompensator, apply_exposure  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.foresplat_losses import mask_loss_terms, rgb_reconstruction_loss  # noqa: E402
from utils.foresplat_pruning import maybe_prune_gaussians  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402
from utils.image_utils import psnr  # noqa: E402
from utils.loss_utils import L1_loss_appearance, l1_loss, ssim  # noqa: E402


class ForegroundParams:
    """Experiment-factor arguments (GGGS core stays untouched)."""

    def __init__(self, parser: ArgumentParser):
        parser.add_argument("--enable_eo", action="store_true", help="GGGS native per-view exposure compensation")
        parser.add_argument("--exposure_params_dir", default=None, help="Where to export the per-view exposure params")
        parser.add_argument("--mask_dir", default=None, help="Directory of per-view foreground masks (<image_name>.png)")
        parser.add_argument("--mask_ext", default=".png")
        parser.add_argument("--mask_threshold", type=float, default=0.5)
        parser.add_argument("--use_foreground_rgb_loss", action="store_true")
        parser.add_argument("--foreground_bg_rgb_weight", type=float, default=0.0)
        parser.add_argument("--foreground_rgb_crop_padding", type=int, default=12)
        parser.add_argument("--use_mask_loss", action="store_true")
        parser.add_argument("--use_bg_opacity_loss", action="store_true")
        parser.add_argument("--lambda_fg_rgb", type=float, default=1.0)
        parser.add_argument("--lambda_mask", type=float, default=0.0)
        parser.add_argument("--lambda_bg", type=float, default=0.0)
        parser.add_argument("--mask_loss_type", default="l1_dice")
        parser.add_argument("--mask_ignore_boundary_px", type=int, default=0)
        parser.add_argument("--mask_loss_start_iter", type=int, default=0)
        parser.add_argument("--mask_loss_warmup_iters", type=int, default=0)
        parser.add_argument("--pruning_mode", default="none", choices=["none", "mask", "opacity", "brightness", "topology"])
        parser.add_argument("--pruning_start_iter", type=int, default=15_000)
        parser.add_argument("--pruning_interval", type=int, default=1_000)
        parser.add_argument("--pruning_opacity_threshold", type=float, default=0.01)
        parser.add_argument("--pruning_brightness_threshold", type=float, default=0.03)
        parser.add_argument("--pruning_mask_threshold", type=float, default=0.2)
        parser.add_argument("--pruning_mask_max_views", type=int, default=12)
        parser.add_argument("--pruning_max_remove_ratio", type=float, default=0.05)
        parser.add_argument("--pruning_mask_score_weight", type=float, default=1.0)
        parser.add_argument("--pruning_survival_mode", default="multi_cue", choices=["multi_cue", "legacy_mask"])
        parser.add_argument("--pruning_survival_threshold", type=float, default=0.45)
        parser.add_argument("--pruning_cue_weights", default="3.0,1.0,0.25,1.0,0.5")
        parser.add_argument("--pruning_keep_bias", type=float, default=0.0)
        parser.add_argument("--pruning_topology_grid", type=int, default=64)
        parser.add_argument("--pruning_topology_reference", default="mean")
        parser.add_argument("--save_pruning_report", action="store_true")
        parser.add_argument("--metrics_csv", default=None, help="Where to append per-iteration training metrics")
        parser.add_argument("--metrics_interval", type=int, default=100)


class ForegroundNamespace:
    """Lightweight adapter exposing the parsed arguments as plain attributes."""

    def __init__(self, fg_args):
        # copy every parsed attribute, then keep the shared SSIM weight in sync
        for key, value in vars(fg_args).items():
            setattr(self, key, value)
        self.lambda_dssim = 0.2  # overwritten with opt.lambda_dssim in training()


def load_camera_masks(cameras, mask_dir: str, mask_ext: str, threshold: float) -> None:
    """Load the per-view foreground mask into every GGGS camera (``gt_mask``)."""
    missing = []
    for cam in cameras:
        name = os.path.splitext(cam.image_name)[0] + mask_ext
        path = os.path.join(mask_dir, name)
        if not os.path.isfile(path):
            missing.append(name)
            continue
        with Image.open(path) as img:
            mask = img.convert("L").resize((cam.image_width, cam.image_height), Image.Resampling.NEAREST)
        tensor = torch.from_numpy(np.array(mask)).float() / 255.0
        cam.gt_mask = (tensor >= threshold).float().unsqueeze(0).to("cuda")
    if missing:
        raise FileNotFoundError(f"{len(missing)} masks missing in {mask_dir}, e.g. {missing[:3]}")


def training(dataset, opt, pipe, fg, eo, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    eo.attach(gaussians, len(scene.getTrainCameras()))
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    kernel_size = dataset.kernel_size

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    trainCameras = scene.getTrainCameras().copy()
    if fg.mask_dir:
        load_camera_masks(trainCameras, fg.mask_dir, fg.mask_ext, fg.mask_threshold)
    fg.lambda_dssim = opt.lambda_dssim  # keep the shared SSIM weight identical across experiments
    if dataset.disable_filter3D:
        gaussians.reset_3D_filter()
    else:
        gaussians.compute_3D_filter(cameras=trainCameras)

    # ---- training metrics CSV -------------------------------------------- #
    metrics_writer = None
    metrics_file = None
    if fg.metrics_csv:
        os.makedirs(os.path.dirname(os.path.abspath(fg.metrics_csv)), exist_ok=True)
        metrics_file = open(fg.metrics_csv, "w", newline="", encoding="utf-8")
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow(
            ["iteration", "total_loss", "rgb_loss", "l1_loss", "ssim_loss", "mask_loss", "bg_loss",
             "depth_normal_loss", "ncc_loss", "geo_loss", "lr", "num_gaussians", "iter_time_ms", "mask_schedule"]
        )

    from utils.loss_utils import PatchMatch

    patchmatch = PatchMatch(
        opt.multi_view_patch_size,
        opt.multi_view_pixel_noise_th,
        kernel_size=kernel_size,
        pipe=pipe,
        debug=True,
        model_path=dataset.model_path,
        optimize_geo=opt.lambda_multi_view_geo > 0,
        optimize_ncc=opt.lambda_multi_view_ncc > 0,
    )
    viewpoint_stack = []
    ema_loss_for_log = 0.0
    ema_normal_loss_for_log = 0.0
    ema_ncc_loss_for_log = 0.0
    ema_mask_for_log = 0.0
    ema_bg_for_log = 0.0
    t_start = time.time()
    peak_mem = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            if gaussians.active_sh_degree == gaussians.max_sh_degree:
                gaussians.unlockSGdegree(100)
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        # GGGS train.py uses Python's random.randint (inclusive high); numpy's
        # randint is exclusive and raises ValueError when only one view remains.
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        reg_kick_on = iteration >= opt.regularization_from_iter
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size, require_depth=reg_kick_on)
        rendered_image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        gt_image = viewpoint_cam.original_image.cuda()

        # ---- photometric loss (GGGS core + optional EO + optional FG) ---- #
        target_mask = getattr(viewpoint_cam, "gt_mask", None)
        if eo.enabled:
            image_corrected = apply_exposure(rendered_image, gaussians, viewpoint_cam.uid)
            Ll1_render = l1_loss(image_corrected, gt_image)
        else:
            image_corrected = rendered_image
            Ll1_render = L1_loss_appearance(rendered_image, gt_image, gaussians, viewpoint_cam.uid)

        if fg.use_foreground_rgb_loss:
            rgb_loss, Ll1_render = rgb_reconstruction_loss(image_corrected, gt_image, target_mask, fg)
        else:
            rgb_loss = (1.0 - opt.lambda_dssim) * Ll1_render + opt.lambda_dssim * (
                1.0 - ssim(rendered_image.unsqueeze(0), gt_image.unsqueeze(0))
            )

        # ---- GGGS regularisers (unchanged) ------------------------------- #
        if reg_kick_on and opt.lambda_depth_normal > 0:
            depth_map = render_pkg["median_depth"]
            rendered_normal = render_pkg["normal"]
            from utils.graphics_utils import depth_to_normal

            depth_normal, valid_points = depth_to_normal(viewpoint_cam, depth_map)
            normal_error_map = 1 - torch.linalg.vecdot(rendered_normal, depth_normal, dim=0)
            depth_normal_loss = normal_error_map.masked_fill_(~valid_points.squeeze(), 0.0).mean()
        else:
            depth_normal = None
            depth_normal_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        if reg_kick_on and (opt.lambda_multi_view_ncc > 0 or opt.lambda_multi_view_geo > 0):
            from random import sample

            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[
                sample(viewpoint_cam.nearest_id, 1)[0]
            ]
            ncc_loss, geo_loss = patchmatch(gaussians, render_pkg, viewpoint_cam, nearest_cam, iteration, depth_normal)
        else:
            ncc_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
            geo_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # ---- ForeSplat foreground losses -------------------------------- #
        mask_loss, bg_loss, mask_schedule = mask_loss_terms(
            render_pkg["mask"], target_mask, fg, iteration
        )

        loss = (
            rgb_loss
            + opt.lambda_depth_normal * depth_normal_loss
            + opt.lambda_multi_view_ncc * ncc_loss
            + opt.lambda_multi_view_geo * geo_loss
            + mask_loss
            + bg_loss
        )
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_normal_loss_for_log = 0.4 * depth_normal_loss.item() + 0.6 * ema_normal_loss_for_log
            ema_ncc_loss_for_log = 0.4 * ncc_loss.item() + 0.6 * ema_ncc_loss_for_log
            ema_mask_for_log = 0.4 * mask_loss.item() + 0.6 * ema_mask_for_log
            ema_bg_for_log = 0.4 * bg_loss.item() + 0.6 * ema_bg_for_log
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1024 / 1024)

            if iteration % 10 == 0:
                postfix = {
                    "Loss": f"{ema_loss_for_log:.{4}f}",
                    "normal": f"{ema_normal_loss_for_log:.{4}f}",
                    "ncc": f"{ema_ncc_loss_for_log:.{4}f}",
                    "pts": f"{gaussians.get_xyz.shape[0]}",
                }
                if fg.use_mask_loss or fg.use_bg_opacity_loss:
                    postfix["mask"] = f"{ema_mask_for_log:.{4}f}"
                    postfix["bg"] = f"{ema_bg_for_log:.{4}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            if iteration % 100 == 0 and iteration in (1, 100):
                print(
                    f"\n[{iteration}] total {loss.item():.5f} rgb {rgb_loss.item():.5f} "
                    f"l1 {Ll1_render.item():.5f} mask {mask_loss.item():.5f} bg {bg_loss.item():.5f} "
                    f"pts {gaussians.get_xyz.shape[0]}",
                    flush=True,
                )

            if metrics_writer and iteration % fg.metrics_interval == 0:
                # exact algebraic recovery of the SSIM term from the two
                # reported quantities (both loss branches have the form
                # (1-l)*L1 + l*SSIM, optionally scaled by lambda_fg_rgb)
                ssim_term = (
                    rgb_loss.item() / max(fg.lambda_fg_rgb, 1e-8)
                    - (1.0 - opt.lambda_dssim) * Ll1_render.item()
                ) / max(opt.lambda_dssim, 1e-8)
                metrics_writer.writerow(
                    [iteration, f"{loss.item():.6f}", f"{rgb_loss.item():.6f}", f"{Ll1_render.item():.6f}",
                     f"{ssim_term:.6f}", f"{mask_loss.item():.6f}", f"{bg_loss.item():.6f}",
                     f"{depth_normal_loss.item():.6f}", f"{ncc_loss.item():.6f}", f"{geo_loss.item():.6f}",
                     f"{gaussians.optimizer.param_groups[0]['lr']:.3e}", gaussians.get_xyz.shape[0],
                     f"{iter_start.elapsed_time(iter_end):.2f}", f"{mask_schedule}"]
                )

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration in testing_iterations:
                torch.cuda.empty_cache()
                l1_test = psnr_test = 0.0
                for idx, viewpoint in enumerate(scene.getTrainCameras()):
                    result = render(viewpoint, gaussians, pipe, background, kernel_size)["render"]
                    image = torch.clamp(result, 0.0, 1.0)
                    gt = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)
                    l1_test += l1_loss(image, gt).mean().double().item()
                    psnr_test += psnr(image, gt).mean().double().item()
                n = len(scene.getTrainCameras())
                print(
                    f"\n[ITER {iteration}] TRAIN-VIEW L1 {l1_test / n:.5f} PSNR {psnr_test / n:.4f}",
                    flush=True,
                )
                torch.cuda.empty_cache()

            # Densification (GGGS core, unchanged)
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold
                    )
                    if dataset.disable_filter3D:
                        gaussians.reset_3D_filter()
                    else:
                        gaussians.compute_3D_filter(cameras=trainCameras)
                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter and not dataset.disable_filter3D:
                if iteration < opt.iterations - 100:
                    gaussians.compute_3D_filter(cameras=trainCameras)

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if fg.pruning_mode != "none":
                report = maybe_prune_gaussians(
                    gaussians, fg, iteration, scene.model_path,
                    scene.getTrainCameras(),
                    report_path=(
                        os.path.join(scene.model_path, "pruning_report.json")
                        if fg.save_pruning_report else None
                    ),
                )
                if report and report.get("removed", 0) > 0:
                    print(
                        f"\n[ITER {iteration}] mask pruning removed {report['removed']}/{report['gaussians_before']}"
                        f" -> {report['gaussians_after']} pts",
                        flush=True,
                    )

            if iteration in checkpoint_iterations:
                torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnt{iteration}.pth")

            if fg.metrics_csv and metrics_file:
                metrics_file.flush()

    if metrics_file:
        metrics_file.close()

    # ---- finalisation: exposure params + gaussian dump ------------------- #
    eo.export(
        gaussians,
        fg.exposure_params_dir or scene.model_path,
        [cam.image_name for cam in scene.getTrainCameras()],
    )
    final_dir = os.path.join(scene.model_path, "final_gaussians")
    os.makedirs(final_dir, exist_ok=True)
    gaussians.save_ply(os.path.join(final_dir, "point_cloud.ply"))
    np.savez(
        os.path.join(final_dir, "gaussians.npz"),
        xyz=gaussians.get_xyz.detach().cpu().numpy(),
        features_dc=gaussians._features_dc.detach().cpu().numpy(),
        features_rest=gaussians._features_rest.detach().cpu().numpy(),
        opacity=gaussians.get_opacity.detach().cpu().numpy(),
        scaling=gaussians.get_scaling.detach().cpu().numpy(),
        rotation=gaussians.get_rotation.detach().cpu().numpy(),
        num_gaussians=gaussians.get_xyz.shape[0],
        training_seconds=time.time() - t_start,
        peak_vram_mib=peak_mem,
    )
    print(f"\nTraining complete: {gaussians.get_xyz.shape[0]} Gaussians, peak VRAM {peak_mem:.0f} MiB")


def main(argv=None) -> int:
    parser = ArgumentParser(description="ForeSplat-GGGS experiment training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--smoke", action="store_true", help="Short sanity run")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    ForegroundParams(parser)
    args = parser.parse_args(argv)

    if args.smoke:
        args.iterations = min(args.iterations, 200)
        args.save_iterations = [args.iterations]
        args.test_iterations = [args.iterations]
        args.densify_until_iter = 0
        args.regularization_from_iter = 10**9
    else:
        args.save_iterations = sorted(set(args.save_iterations + [args.iterations]))
        args.test_iterations = sorted(set(args.test_iterations + [args.iterations]))

    fg = ForegroundNamespace(args)
    eo = ExposureCompensator(args.enable_eo)
    args.use_decoupled_appearance = eo.decoupled_appearance
    if fg.use_foreground_rgb_loss or fg.use_mask_loss or fg.use_bg_opacity_loss or fg.mask_dir:
        if not fg.mask_dir:
            parser.error("Foreground losses require --mask_dir")
    print(f"EO enabled: {eo.enabled} | masks: {args.mask_dir} | pruning: {fg.pruning_mode}", flush=True)

    os.makedirs(args.model_path, exist_ok=True)
    safe_state(args.quiet)
    training(
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        fg=fg,
        eo=eo,
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        checkpoint_iterations=args.checkpoint_iterations,
        checkpoint=args.start_checkpoint,
        debug_from=-1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
