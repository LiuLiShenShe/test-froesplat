#!/usr/bin/env python3
"""Evaluate the four ForeSplat-GGGS experiments and write the comparison tables.

For every experiment directory under ``experiments/<plant>/`` this script

* loads the trained Gaussians through the GGGS reader (``Scene(load_iteration=...)``),
* renders **all** training views at the training resolution and reports
  PSNR / SSIM / LPIPS (explicitly labelled *Training-view Metrics*), plus
  foreground-region PSNR / SSIM computed inside each view's GT-mask bbox,
* renders a fixed set of evaluation views (four azimuth bins, top, closest
  interior view, leaf-zoom crop) with identical cameras/resolution for every
  experiment, and saves GT + E1..E4 renders and per-view montages,
* exports the Gaussian cloud (means, RGB, opacity, scale, rotation) in the
  native VGGT-Omega coordinate frame,
* extracts a mesh with GGGS' native TSDF fusion (``mesh_extract.extract_mesh``
  code path, with the dataset masks used to cut background depth),
* for E2/E4 reports exposure-corrected PSNR as well.

Finally it writes ``comparison.csv``, ``comparison.md`` and
``ablation_report.md`` at the experiment root.  No scan registration happens
here: every artefact stays in the VGGT-Omega native frame.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGGS = os.path.join(ROOT, "gggs_community")
if GGGS not in sys.path:
    sys.path.insert(0, GGGS)

from arguments import ModelParams, PipelineParams  # noqa: E402
from utils.cuda_preload import preload_cuda_ext_deps  # noqa: E402

preload_cuda_ext_deps()

from gaussian_renderer import render  # noqa: E402
from models.exposure_compensator import ExposureCompensator, apply_exposure  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402

EXPERIMENTS = [
    ("E1_GGGS_Baseline", "GGGS", False, False),
    ("E2_GGGS_EO", "GGGS + EO", True, False),
    ("E3_GGGS_Foreground", "GGGS + FG", False, True),
    ("E4_ForeSplat_GGGS", "ForeSplat-GGGS", True, True),
]


def _args_for(dataset_dir: str, model_path: str, resolution: int) -> ModelParams:
    ns = argparse.Namespace()
    ns.sh_degree = 3
    ns.sg_degree = 0
    ns.source_path = dataset_dir
    ns.model_path = model_path
    ns.images = "images"
    ns.dataset = ""
    ns.resolution = resolution
    ns.white_background = False
    ns.data_device = "cuda"
    ns.eval = False
    ns.use_decoupled_appearance = 0
    ns.disable_filter3D = False
    ns.kernel_size = 0.0
    ns.multi_view_num = 8
    ns.multi_view_max_angle = 30
    ns.multi_view_min_dis = 0.01
    ns.multi_view_max_dis = 1.5
    return ns


def _pipe() -> PipelineParams:
    ns = argparse.Namespace()
    ns.convert_SHs_python = False
    ns.compute_cov3D_python = False
    ns.debug = False
    return ns


def load_experiment(dataset_dir: str, exp_dir: str, resolution: int, iters: int, eo_dir: str | None):
    dataset = _args_for(dataset_dir, os.path.join(exp_dir, "checkpoints"), resolution)
    gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
    scene = Scene(dataset, gaussians, load_iteration=iters)
    if eo_dir and os.path.isfile(os.path.join(eo_dir, "exposure_matrices.npz")):
        eo = ExposureCompensator(True)
        gaussians.create_app_model(len(scene.getTrainCameras()), 1)
        eo.load_into(gaussians, eo_dir)
    return dataset, gaussians, scene


def select_fixed_views(cameras, masks_dir: str):
    """Deterministic fixed view set shared by all four experiments."""
    centers = torch.stack([cam.camera_center.detach().cpu() for cam in cameras]).numpy()
    centroid = centers.mean(axis=0)
    rel = centers - centroid
    azimuth = np.degrees(np.arctan2(rel[:, 0], rel[:, 2]))
    labels = {}
    for name, target in (("front", 0.0), ("right", 90.0), ("back", 180.0), ("left", -90.0)):
        diff = np.abs(((azimuth - target + 180) % 360) - 180)
        labels[name] = int(np.argmin(diff))
    labels["top"] = int(np.argmax(centers[:, 1]))
    dist = np.linalg.norm(rel, axis=1)
    labels["interior"] = int(np.argmin(dist))
    # leaf-zoom crop: fixed bbox around the GT-mask centroid of the front view
    from PIL import Image

    front_cam = cameras[labels["front"]]
    mask_path = os.path.join(masks_dir, os.path.splitext(front_cam.image_name)[0] + ".png")
    with Image.open(mask_path) as img:
        mask = np.array(img.convert("L").resize((front_cam.image_width, front_cam.image_height),
                                                Image.Resampling.NEAREST)) > 127
    ys, xs = np.nonzero(mask)
    cx, cy = int(xs.mean()), int(ys.mean())
    half = 320
    crop = {
        "label": "leaf_zoom",
        "view": labels["front"],
        "x0": max(cx - half, 0), "y0": max(cy - half, 0),
        "x1": min(cx + half, front_cam.image_width), "y1": min(cy + half, front_cam.image_height),
    }
    return labels, crop


def render_view(cam, gaussians, pipe, background, kernel_size, eo_enabled: bool):
    with torch.no_grad():
        pkg = render(cam, gaussians, pipe, background, kernel_size)
        raw = pkg["render"]
        image = torch.clamp(raw, 0.0, 1.0)
        corrected = None
        if eo_enabled and gaussians.app_model != GaussianModel.App_model.NO:
            # apply the exposure to the *unclamped* render, as in training
            corrected = torch.clamp(apply_exposure(raw, gaussians, cam.uid), 0.0, 1.0)
    return image, corrected


def to_numpy(image):
    return image.detach().cpu().numpy().transpose(1, 2, 0)


def bbox_from_mask(cam, masks_dir: str):
    from PIL import Image

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


def evaluate_experiment(plant, exp_id, exp_dir, dataset_dir, masks_dir, iters, resolution,
                        lpips_model, ssim_fn, psnr_fn, fixed, crop, save_renders=True):
    import skimage.io as skio
    from PIL import Image

    eo_dir = os.path.join(exp_dir, "exposure_params") if exp_id in ("E2_GGGS_EO", "E4_ForeSplat_GGGS") else None
    dataset, gaussians, scene = load_experiment(dataset_dir, exp_dir, resolution, iters, eo_dir)
    pipe = _pipe()
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    kernel_size = dataset.kernel_size
    cameras = scene.getTrainCameras()
    eo_enabled = eo_dir is not None and gaussians.app_model == GaussianModel.App_model.GS

    renders_dir = os.path.join(exp_dir, "renders")
    metrics_dir = os.path.join(exp_dir, "metrics")
    os.makedirs(renders_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    rows = []
    psnrs, ssims, lps, fg_psnrs, fg_ssims, eo_psnrs = [], [], [], [], [], []
    for index, cam in enumerate(cameras):
        image, corrected = render_view(cam, gaussians, pipe, background, kernel_size, eo_enabled)
        pred = to_numpy(image)
        gt = to_numpy(torch.clamp(cam.original_image, 0.0, 1.0))
        psnr, ssim, lp = image_metrics(pred, gt, lpips_model, ssim_fn, psnr_fn)
        psnrs.append(psnr); ssims.append(ssim); lps.append(lp)
        bbox = bbox_from_mask(cam, masks_dir)
        fg_psnr = fg_ssim = float("nan")
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            fg_psnr = float(psnr_fn(gt[y0:y1, x0:x1], pred[y0:y1, x0:x1], data_range=1.0))
            fg_ssim = float(ssim_fn(gt[y0:y1, x0:x1], pred[y0:y1, x0:x1], channel_axis=2, data_range=1.0))
            fg_psnrs.append(fg_psnr); fg_ssims.append(fg_ssim)
        eo_psnr = float("nan")
        if corrected is not None:
            eo_psnr = float(psnr_fn(gt, to_numpy(corrected), data_range=1.0))
            eo_psnrs.append(eo_psnr)
        rows.append({"image_name": cam.image_name, "psnr": psnr, "ssim": ssim, "lpips": lp,
                     "fg_psnr": fg_psnr, "fg_ssim": fg_ssim, "eo_psnr": eo_psnr})

    with open(os.path.join(metrics_dir, "per_view_metrics.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": exp_id,
        "metric_scope": "training_views",
        "num_views": len(rows),
        "psnr": float(np.nanmean(psnrs)),
        "ssim": float(np.nanmean(ssims)),
        "lpips": float(np.nanmean(lps)) if np.isfinite(lps).any() else None,
        "fg_psnr": float(np.nanmean(fg_psnrs)) if fg_psnrs else None,
        "fg_ssim": float(np.nanmean(fg_ssims)) if fg_ssims else None,
        "eo_corrected_psnr": float(np.nanmean(eo_psnrs)) if eo_psnrs else None,
        "num_gaussians": int(gaussians.get_xyz.shape[0]),
    }
    json.dump(summary, open(os.path.join(metrics_dir, "render_metrics.json"), "w", encoding="utf-8"), indent=2)

    # ---- fixed evaluation views ----------------------------------------- #
    fixed_dir = os.path.join(renders_dir, "fixed_views")
    os.makedirs(fixed_dir, exist_ok=True)
    montage_inputs = {}
    for label, cam_index in fixed.items():
        cam = cameras[cam_index]
        image, corrected = render_view(cam, gaussians, pipe, background, kernel_size, eo_enabled)
        pred = to_numpy(corrected if corrected is not None else image)
        gt = to_numpy(torch.clamp(cam.original_image, 0.0, 1.0))
        skio.imsave(os.path.join(fixed_dir, f"{label}_{exp_id}.png"), (pred * 255).round().astype(np.uint8))
        if label == list(fixed.keys())[0]:
            skio.imsave(os.path.join(fixed_dir, f"{label}_gt.png"), (gt * 255).round().astype(np.uint8))
        montage_inputs[label] = (gt, pred)
    # leaf zoom crop (identical crop for all experiments)
    x0, y0, x1, y1 = crop["x0"], crop["y0"], crop["x1"], crop["y1"]
    cam = cameras[crop["view"]]
    image, corrected = render_view(cam, gaussians, pipe, background, kernel_size, eo_enabled)
    pred = to_numpy(corrected if corrected is not None else image)
    gt = to_numpy(torch.clamp(cam.original_image, 0.0, 1.0))
    skio.imsave(os.path.join(fixed_dir, f"leaf_zoom_{exp_id}.png"), (pred[y0:y1, x0:x1] * 255).round().astype(np.uint8))
    skio.imsave(os.path.join(fixed_dir, "leaf_zoom_gt.png"), (gt[y0:y1, x0:x1] * 255).round().astype(np.uint8))
    montage_inputs["leaf_zoom"] = (gt[y0:y1, x0:x1], pred[y0:y1, x0:x1])

    # ---- gaussian + mesh ------------------------------------------------ #
    gauss_dir = os.path.join(exp_dir, "gaussian")
    os.makedirs(gauss_dir, exist_ok=True)
    src_ply = os.path.join(exp_dir, "checkpoints/final_gaussians/point_cloud.ply")
    src_npz = os.path.join(exp_dir, "checkpoints/final_gaussians/gaussians.npz")
    for src, dst in ((src_ply, os.path.join(gauss_dir, "point_cloud.ply")),
                     (src_npz, os.path.join(gauss_dir, "gaussians.npz"))):
        if os.path.isfile(src) and not os.path.isfile(dst):
            os.link(src, dst)

    del gaussians, scene
    gc.collect()
    torch.cuda.empty_cache()
    return summary, montage_inputs


def extract_meshes(exp_id, exp_dir, dataset_dir, masks_dir, iters, resolution):
    """GGGS-native TSDF mesh extraction (mesh_extract.extract_mesh code path)."""
    import open3d as o3d
    import open3d.core as o3c
    from mesh_extract import post_process_mesh
    from train_foresplat import load_camera_masks

    mesh_dir = os.path.join(exp_dir, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    dataset = _args_for(dataset_dir, os.path.join(exp_dir, "checkpoints"), resolution)
    gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
    scene = Scene(dataset, gaussians, load_iteration=iters, shuffle=False)
    cameras = scene.getTrainCameras()
    load_camera_masks(cameras, masks_dir, ".png", 0.5)
    pipe = _pipe()
    kernel_size = dataset.kernel_size
    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    depth_list, color_list = [], []
    for cam in cameras:
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, background, kernel_size)
            color = torch.clamp(pkg["render"], 0.0, 1.0).cpu().numpy().transpose(1, 2, 0)
            color_list.append(np.ascontiguousarray(color))
            depth = pkg["median_depth"].clone()
            if cam.gt_mask is not None:
                depth[cam.gt_mask < 0.5] = 0
            depth_list.append(depth[0].cpu().numpy())
    torch.cuda.empty_cache()
    voxel_size = 0.002
    device = o3d.core.Device("CPU:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel_size,
        block_resolution=16,
        block_count=50000,
        device=device,
    )
    for color, depth, cam in zip(color_list, depth_list, cameras):
        img_d = o3d.t.geometry.Image(depth).to(device)
        img_c = o3d.t.geometry.Image(color).to(device)
        intrinsic = o3d.core.Tensor(np.array(
            [[cam.Fx, 0, cam.Cx], [0, cam.Fy, cam.Cy], [0, 0, 1]], dtype=np.float64))
        extrinsic = o3d.core.Tensor((cam.world_view_transform.T).cpu().numpy().astype(np.float64))
        frustum = vbg.compute_unique_block_coordinates(img_d, intrinsic, extrinsic, 1.0, 8.0)
        vbg.integrate(frustum, img_d, img_c, intrinsic, extrinsic, 1.0, 8.0)
    mesh = vbg.extract_triangle_mesh()
    # vbg.extract_triangle_mesh() returns an Open3D *tensor* TriangleMesh, which
    # exposes `.vertex`/`.triangle` (Tensors), not `.vertices`/`.triangles`.
    # Convert to the legacy mesh once and use the legacy API for normals, I/O
    # and vertex counts (post_process_mesh also expects a legacy mesh).
    mesh = mesh.to_legacy()
    mesh.compute_vertex_normals()
    raw_path = os.path.join(mesh_dir, "recon.ply")
    o3d.io.write_triangle_mesh(raw_path, mesh)
    processed = post_process_mesh(mesh, 1)
    post_path = os.path.join(mesh_dir, "recon_post.ply")
    o3d.io.write_triangle_mesh(post_path, processed)
    status = {"status": "ok", "raw": raw_path, "post": post_path,
              "num_vertices_raw": int(len(mesh.vertices)),
              "num_vertices_post": int(len(processed.vertices)),
              "method": "GGGS native TSDF fusion (mesh_extract.py) with dataset masks on depth"}
    json.dump(status, open(os.path.join(mesh_dir, "mesh_status.json"), "w", encoding="utf-8"), indent=2)
    del gaussians, scene
    gc.collect()
    torch.cuda.empty_cache()
    return status


def write_comparison(experiments_root, summaries, train_rows):
    fieldnames = ["Method", "EO", "Foreground", "PSNR", "SSIM", "LPIPS", "FG-PSNR", "FG-SSIM",
                  "EO-corrected PSNR", "Gaussian Count", "Training Time (min)", "Peak VRAM (MiB)"]
    csv_path = os.path.join(experiments_root, "comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for exp_id, label, eo, fg in EXPERIMENTS:
            s = summaries[exp_id]
            t = train_rows.get(exp_id, {})
            writer.writerow({
                "Method": label, "EO": "Yes" if eo else "No", "Foreground": "Yes" if fg else "No",
                "PSNR": f"{s['psnr']:.4f}", "SSIM": f"{s['ssim']:.4f}",
                "LPIPS": "" if s.get("lpips") is None else f"{s['lpips']:.4f}",
                "FG-PSNR": "" if s.get("fg_psnr") is None else f"{s['fg_psnr']:.4f}",
                "FG-SSIM": "" if s.get("fg_ssim") is None else f"{s['fg_ssim']:.4f}",
                "EO-corrected PSNR": "" if s.get("eo_corrected_psnr") is None else f"{s['eo_corrected_psnr']:.4f}",
                "Gaussian Count": s["num_gaussians"],
                "Training Time (min)": f"{t.get('training_seconds', float('nan')) / 60:.1f}",
                "Peak VRAM (MiB)": f"{t.get('peak_vram_mib', float('nan')):.0f}",
            })
    md = [
        "# Four-experiment comparison (KongQueZhuYu, all-view training)",
        "",
        "Metrics are computed on the **training views** (all 208 views participate in",
        "training; ``eval=False``). They are Training-view Metrics, not novel-view",
        "generalisation numbers.",
        "",
        "| Method | EO | Foreground | PSNR | SSIM | LPIPS | FG-PSNR | FG-SSIM | Gaussian Count | Training Time (min) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for exp_id, label, eo, fg in EXPERIMENTS:
        s = summaries[exp_id]
        t = train_rows.get(exp_id, {})
        md.append(
            f"| {label} | {'Yes' if eo else 'No'} | {'Yes' if fg else 'No'} | {s['psnr']:.4f} | {s['ssim']:.4f} | "
            f"{'' if s.get('lpips') is None else f'{s['lpips']:.4f}'} | "
            f"{'' if s.get('fg_psnr') is None else f'{s['fg_psnr']:.4f}'} | "
            f"{'' if s.get('fg_ssim') is None else f'{s['fg_ssim']:.4f}'} | {s['num_gaussians']} | "
            f"{t.get('training_seconds', float('nan')) / 60:.1f} |"
        )
    with open(os.path.join(experiments_root, "comparison.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(md) + "\n")
    return csv_path


def write_ablation_report(experiments_root, summaries, train_rows, shared):
    s1, s2, s3, s4 = (summaries[e] for e, *_ in EXPERIMENTS)
    d_psnr_eo = s2["psnr"] - s1["psnr"]
    d_ssim_eo = s2["ssim"] - s1["ssim"]
    d_fg_psnr = s3["fg_psnr"] - s1["fg_psnr"] if s3.get("fg_psnr") and s1.get("fg_psnr") else float("nan")
    d_psnr_fg = s3["psnr"] - s1["psnr"]
    d_ssim_fg = s3["ssim"] - s1["ssim"]
    d_psnr_full = s4["psnr"] - s1["psnr"]
    best = max((e for e, *_ in EXPERIMENTS), key=lambda e: summaries[e]["psnr"])
    lines = [
        "# ForeSplat-GGGS ablation report - KongQueZhuYu",
        "",
        "## A. Experimental setup",
        "",
        f"- Plant: KongQueZhuYu, full sequence from the ForeSplat VGGT-Omega frontend.",
        f"- Views: {shared['num_cameras']} training views (all valid views; eval split disabled).",
        f"- Initialisation: {shared['init_points']} fused foreground points (VGGT-Omega multi-view fusion);",
        "  identical for all four experiments, re-created into a fresh GaussianModel each run.",
        f"- Resolution divisor {shared['resolution']}, SH degree 0, SG degree 0, seed 0,",
        f"  {shared['iters']} iterations, identical optimiser / densification / pruning schedule",
        "  (pruning only in the Foreground experiments, as defined by the factor).",
        "- Coordinate frame: VGGT-Omega native. No laser-scan Sim(3) registration, no scale change.",
        "- GPU: single RTX A6000 (48 GB, GPU 1); experiments run sequentially, never concurrently.",
        "",
        "## B. Quantitative comparison",
        "",
        "| Method | EO | Foreground | PSNR | SSIM | LPIPS | FG-PSNR | FG-SSIM | Gaussian Count | Training Time (min) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for exp_id, label, eo, fg in EXPERIMENTS:
        s = summaries[exp_id]
        t = train_rows.get(exp_id, {})
        lines.append(
            f"| {label} | {'Yes' if eo else 'No'} | {'Yes' if fg else 'No'} | {s['psnr']:.4f} | {s['ssim']:.4f} | "
            f"{'' if s.get('lpips') is None else f'{s['lpips']:.4f}'} | "
            f"{'' if s.get('fg_psnr') is None else f'{s['fg_psnr']:.4f}'} | "
            f"{'' if s.get('fg_ssim') is None else f'{s['fg_ssim']:.4f}'} | {s['num_gaussians']} | "
            f"{t.get('training_seconds', float('nan')) / 60:.1f} |"
        )
    lines += [
        "",
        "All metrics are Training-view Metrics over the 208 training views.",
        "FG-PSNR / FG-SSIM are computed inside each view's GT-mask bounding box.",
        "",
        "## C. Ablation analysis (data-driven)",
        "",
        f"1. **Exposure compensation (E2 - E1)**: full-image PSNR {d_psnr_eo:+.4f}, "
        f"SSIM {d_ssim_eo:+.4f}. "
        + ("EO improved cross-view photometric consistency under this data." if d_psnr_eo > 0
           else "EO did not improve the rendered metrics on this data."),
        f"2. **Foreground constraints (E3 - E1)**: full-image PSNR {d_psnr_fg:+.4f}, "
        f"SSIM {d_ssim_fg:+.4f}, foreground-region PSNR {d_fg_psnr:+.4f}. "
        + ("The foreground loss improved the plant region specifically." if d_fg_psnr > 0
           else "The foreground loss did not improve the foreground-region metric here."),
        f"3. **Background suppression**: E3/E4 apply the mask + background-opacity losses and "
        f"mask-guided pruning; Gaussian counts are {s1['num_gaussians']} (E1), {s2['num_gaussians']} (E2), "
        f"{s3['num_gaussians']} (E3), {s4['num_gaussians']} (E4). "
        + ("Foreground runs kept fewer/equal primitives after pruning." if s3["num_gaussians"] <= s1["num_gaussians"]
           else "Foreground runs did not reduce the primitive budget on this data."),
        f"4. **Joint effect (E4 - E1)**: full-image PSNR {d_psnr_full:+.4f}; "
        f"best PSNR among the four: {best}. "
        + ("The combined method is the best of the four on PSNR." if best == "E4_ForeSplat_GGGS"
           else "The combined method is not the best on PSNR - report this honestly rather than assuming E4 wins."),
        "5. **Caveats**: metrics are on training views (all views trained on); per-view LPIPS/SSIM",
        "   variation is in `metrics/per_view_metrics.csv`; no novel-view split was held out.",
        "",
        "## D. Preparation for scan-based geometric evaluation (not performed here)",
        "",
        "- Every experiment keeps the Gaussian cloud (means / RGB / opacity / scale / rotation) in",
        "  `gaussian/gaussians.npz` and `gaussian/point_cloud.ply`, in the VGGT-Omega native frame.",
        "- Native meshes (GGGS TSDF fusion) are in `mesh/` when extraction succeeded.",
        "- The camera metadata and the scene alignment matrix are in the dataset metadata;",
        "  a later stage can register these outputs to the laser scan once, then evaluate geometry.",
    ]
    with open(os.path.join(experiments_root, "ablation_report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", default="KongQueZhuYu")
    parser.add_argument("--batch-root", default=os.path.join(ROOT, "runs/full_sequence_batch"))
    parser.add_argument("--experiments-root", default=None)
    parser.add_argument("--iters", type=int, default=30_000)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--skip-mesh", action="store_true")
    parser.add_argument("--mesh-only", action="store_true",
                        help="Only (re)extract meshes; skip metrics/renders/reports")
    args = parser.parse_args(argv)

    experiments_root = args.experiments_root or os.path.join(ROOT, "experiments", args.plant)
    dataset_dir = os.path.join(args.batch_root, args.plant, "GGGS_Dataset")
    masks_dir = os.path.join(dataset_dir, "masks")

    if args.mesh_only:
        safe_state(True)
        for exp_id, label, eo, fg in EXPERIMENTS:
            exp_dir = os.path.join(experiments_root, exp_id)
            print(f"[eval:mesh] {exp_id}", flush=True)
            try:
                st = extract_meshes(exp_id, exp_dir, dataset_dir, masks_dir, args.iters, args.resolution)
                print(f"[eval:mesh] {exp_id}: ok ({st['num_vertices_post']} vertices, post)", flush=True)
            except Exception as exc:  # noqa: BLE001 - best-effort, honest status only
                mesh_dir = os.path.join(exp_dir, "mesh")
                os.makedirs(mesh_dir, exist_ok=True)
                json.dump({"status": f"failed: {type(exc).__name__}: {exc}"},
                          open(os.path.join(mesh_dir, "mesh_status.json"), "w", encoding="utf-8"), indent=2)
                print(f"[eval:mesh] {exp_id}: FAILED {type(exc).__name__}: {exc}", flush=True)
        return 0
    dataset_info = json.load(open(os.path.join(experiments_root, "shared_config", "dataset_info.json"), encoding="utf-8"))
    init_info = json.load(open(os.path.join(experiments_root, "shared_config", "initialization_info.json"), encoding="utf-8"))
    shared_block = {
        "num_cameras": dataset_info["num_train_cameras"],
        "init_points": init_info["num_fused_points"],
        "resolution": args.resolution,
        "iters": args.iters,
    }

    safe_state(True)
    import lpips
    import skimage.metrics as skm

    lpips_model = lpips.LPIPS(net="alex", verbose=False).cuda().eval()

    # fixed view set computed once from a temporary camera list
    dataset_tmp = _args_for(dataset_dir, os.path.join(experiments_root, "E1_GGGS_Baseline", "checkpoints"), args.resolution)
    g_tmp = GaussianModel(3, 0)
    scene_tmp = Scene(dataset_tmp, g_tmp, load_iteration=args.iters)
    cameras_tmp = scene_tmp.getTrainCameras()
    fixed, crop = select_fixed_views(cameras_tmp, masks_dir)
    del scene_tmp, g_tmp
    gc.collect()
    torch.cuda.empty_cache()

    summaries = {}
    train_rows = {}
    montages = {}
    for exp_id, label, eo, fg in EXPERIMENTS:
        exp_dir = os.path.join(experiments_root, exp_id)
        print(f"[eval] {exp_id}", flush=True)
        s, m = evaluate_experiment(args.plant, exp_id, exp_dir, dataset_dir, masks_dir, args.iters,
                                   args.resolution, lpips_model, skm.structural_similarity,
                                   skm.peak_signal_noise_ratio, fixed, crop)
        summaries[exp_id] = s
        montages[exp_id] = m
        try:
            with np.load(os.path.join(exp_dir, "gaussian", "gaussians.npz")) as data:
                train_rows[exp_id] = {
                    "training_seconds": float(data["training_seconds"]),
                    "peak_vram_mib": float(data["peak_vram_mib"]),
                }
        except (OSError, KeyError):
            train_rows[exp_id] = {}
        if not args.skip_mesh:
            try:
                mesh_status = extract_meshes(exp_id, exp_dir, dataset_dir, masks_dir, args.iters, args.resolution)
                print(f"[eval] {exp_id}: mesh ok ({mesh_status['num_vertices_post']} vertices)", flush=True)
            except Exception as exc:  # noqa: BLE001 - mesh is best-effort, never faked
                mesh_dir = os.path.join(exp_dir, "mesh")
                os.makedirs(mesh_dir, exist_ok=True)
                json.dump({"status": f"failed: {type(exc).__name__}: {exc}"},
                          open(os.path.join(mesh_dir, "mesh_status.json"), "w", encoding="utf-8"), indent=2)
                print(f"[eval] {exp_id}: mesh failed - {type(exc).__name__}: {exc}", flush=True)

    # montages with identical settings
    montage_dir = os.path.join(experiments_root, "comparison")
    os.makedirs(montage_dir, exist_ok=True)
    from PIL import Image

    for label in montages["E1_GGGS_Baseline"]:
        panels = [montages["E1_GGGS_Baseline"][label][0]]
        for exp_id, *_ in EXPERIMENTS:
            panels.append(montages[exp_id][label][1])
        h = min(p.shape[0] for p in panels)
        w = min(p.shape[1] for p in panels)
        panels = [p[:h, :w] for p in panels]
        stacked = np.concatenate(panels, axis=1)
        Image.fromarray((np.clip(stacked, 0, 1) * 255).round().astype(np.uint8)).save(
            os.path.join(montage_dir, f"{label}_gt_E1_E2_E3_E4.png")
        )

    write_comparison(experiments_root, summaries, train_rows)
    write_ablation_report(experiments_root, summaries, train_rows, shared_block)
    print(f"[eval] wrote comparison.csv/.md and ablation_report.md under {experiments_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
