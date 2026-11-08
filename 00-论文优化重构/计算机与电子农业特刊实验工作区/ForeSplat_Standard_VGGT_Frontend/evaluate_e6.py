#!/usr/bin/env python3
"""Evaluate the E6 official-2DGS baseline on the standard-VGGT COLMAP dataset.

Renders all 208 training views with the trained 2DGS model and computes the
EXACT same metrics as E1/E5 (evaluate_gggs_experiments.image_metrics +
foreground bbox logic), so the three frontends are compared at identical
views / resolution / metric definition (Training-view Metrics).
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

TWO_DGS_ROOT = "/data/fj/F2DMAS/2d-gaussian-splatting-main"
sys.path.insert(0, TWO_DGS_ROOT)

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
PLANT = "KongQueZhuYu"
STD_ROOT = os.path.join(ROOT, "experiments", PLANT, "standard_vggt_comparison")
E6_DIR = os.path.join(STD_ROOT, "E6_2DGS_Standard_VGGT")
E6_DATASET = os.path.join(STD_ROOT, "Standard_VGGT_Dataset")
EXP_ID = "E6_2DGS_Standard_VGGT"


def to_numpy(image):
    return image.detach().cpu().numpy().transpose(1, 2, 0)


def bbox_from_mask(cam, masks_dir):
    path = os.path.join(masks_dir, os.path.splitext(cam.image_name)[0] + ".png")
    with Image.open(path) as img:
        mask = np.array(img.convert("L").resize((cam.image_width, cam.image_height),
                                                Image.Resampling.NEAREST)) > 127
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def image_metrics(pred, gt, lpips_model, ssim_fn, psnr_fn):
    psnr = float(psnr_fn(gt, pred, data_range=1.0))
    ssim = float(ssim_fn(gt, pred, channel_axis=2, data_range=1.0))
    lp = float("nan")
    if lpips_model is not None:
        with torch.no_grad():
            a = torch.from_numpy(pred).permute(2, 0, 1)[None].cuda() * 2 - 1
            b = torch.from_numpy(gt).permute(2, 0, 1)[None].cuda() * 2 - 1
            lp = float(lpips_model(a, b).item())
    return psnr, ssim, lp


def _args_for(dataset_dir, model_path, resolution):
    ns = argparse.Namespace()
    ns.sh_degree = 3
    ns.source_path = dataset_dir
    ns.model_path = model_path
    ns.images = "images"
    ns.dataset = ""
    ns.resolution = resolution
    ns.white_background = False
    ns.data_device = "cuda"
    ns.eval = False
    ns.disable_filter3D = False
    ns.kernel_size = 0.0
    return ns


def _pipe():
    ns = argparse.Namespace()
    ns.convert_SHs_python = False
    ns.compute_cov3D_python = False
    ns.debug = False
    ns.depth_ratio = 0.0
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30_000)
    ap.add_argument("--resolution", type=int, default=2)
    ap.add_argument("--skip-mesh", action="store_true")
    args = ap.parse_args()

    safe_state(True)
    import lpips
    import skimage.metrics as skm

    lpips_model = lpips.LPIPS(net="alex", verbose=False).cuda().eval()

    dataset = _args_for(E6_DATASET, os.path.join(E6_DIR, "checkpoints"), args.resolution)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iters, shuffle=False)
    pipe = _pipe()
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    cameras = scene.getTrainCameras()
    renders_dir = os.path.join(E6_DIR, "renders")
    metrics_dir = os.path.join(E6_DIR, "metrics")
    os.makedirs(renders_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    masks_dir = os.path.join(E6_DATASET, "masks")
    rows = []
    psnrs, ssims, lps, fg_psnrs, fg_ssims = [], [], [], [], []
    for cam in cameras:
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, background)
            image = torch.clamp(pkg["render"], 0.0, 1.0)
        pred = to_numpy(image)
        gt = to_numpy(torch.clamp(cam.original_image, 0.0, 1.0))
        psnr, ssim, lp = image_metrics(pred, gt, lpips_model,
                                       skm.structural_similarity,
                                       skm.peak_signal_noise_ratio)
        psnrs.append(psnr); ssims.append(ssim); lps.append(lp)
        bbox = bbox_from_mask(cam, masks_dir)
        fg_psnr = fg_ssim = float("nan")
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            fg_psnr = float(skm.peak_signal_noise_ratio(gt[y0:y1, x0:x1], pred[y0:y1, x0:x1], data_range=1.0))
            fg_ssim = float(skm.structural_similarity(gt[y0:y1, x0:x1], pred[y0:y1, x0:x1], channel_axis=2, data_range=1.0))
            fg_psnrs.append(fg_psnr); fg_ssims.append(fg_ssim)
        rows.append({"image_name": cam.image_name, "psnr": psnr, "ssim": ssim, "lpips": lp,
                     "fg_psnr": fg_psnr, "fg_ssim": fg_ssim})

    with open(os.path.join(metrics_dir, "per_view_metrics.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": EXP_ID,
        "method": "Official 2DGS (hbb1) on Standard VGGT COLMAP dataset",
        "metric_scope": "training_views",
        "num_views": len(rows),
        "psnr": float(np.nanmean(psnrs)),
        "ssim": float(np.nanmean(ssims)),
        "lpips": float(np.nanmean(lps)),
        "fg_psnr": float(np.nanmean(fg_psnrs)) if fg_psnrs else None,
        "fg_ssim": float(np.nanmean(fg_ssims)) if fg_ssims else None,
        "num_gaussians": int(gaussians.get_xyz.shape[0]),
        "regularization": {"lambda_dist": 0.2, "lambda_normal": 0.05, "iterations": args.iters},
    }
    json.dump(summary, open(os.path.join(metrics_dir, "render_metrics.json"), "w", encoding="utf-8"), indent=2)
    print("[eval] E6 summary:", json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())