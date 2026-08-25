#!/usr/bin/env python3
"""
Evaluate COLMAP dense reconstruction (fuse.ply) using PSNR / SSIM / LPIPS.

Method:
1. Load COLMAP cameras + registered images from model directory.
2. Reproject fused dense point cloud into each camera view (z-buffer).
3. Compare rendered image with GT image of the same view.

Notes:
- Missing pixels (no projected point) are excluded for PSNR.
- For SSIM/LPIPS, missing pixels are replaced by GT to avoid penalizing unknown regions.
- Default uses sparse split (every 8th view) to keep CPU runtime practical.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from plyfile import PlyData


DEFAULT_SUGAR_ROOT = Path("/data/fj/SuGaR-main")
DEFAULT_COLMAP_SCRIPT_DIR = Path("/data/fj/tools/colmap-cuda-src/scripts/python")


# Import external helper modules after path setup.
sys.path.insert(0, str(DEFAULT_SUGAR_ROOT))
sys.path.insert(0, str(DEFAULT_COLMAP_SCRIPT_DIR))

from gaussian_splatting.utils.loss_utils import ssim as compute_ssim  # noqa: E402
from gaussian_splatting.lpipsPyTorch.modules.lpips import LPIPS  # noqa: E402
import read_write_model as rw_model  # noqa: E402


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate COLMAP dense model metrics.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/data/fj/11-COLMAP-fuse/run_20260302_153856/CaoMei2/dense/sparse"),
        help="COLMAP sparse model dir (contains cameras.bin/images.bin/points3D.bin).",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("/data/fj/11-COLMAP-fuse/run_20260302_153856/CaoMei2/dense/images"),
        help="Ground-truth image directory used by COLMAP dense stage.",
    )
    parser.add_argument(
        "--ply-path",
        type=Path,
        default=Path("/data/fj/11-COLMAP-fuse/run_20260302_153856/CaoMei2/fuse.ply"),
        help="Dense fused point cloud path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/fj/08-Check"),
        help="Root directory for evaluation outputs.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=8,
        help="Evaluate every Nth registered image (1 means all).",
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=0,
        help="Max number of selected views to evaluate (0 means no limit).",
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.01,
        help="Skip a view if rendered valid pixel ratio is below this threshold.",
    )
    parser.add_argument(
        "--lpips-net",
        type=str,
        default="vgg",
        choices=["alex", "vgg", "squeeze"],
        help="LPIPS backbone.",
    )
    parser.add_argument(
        "--lpips-max-side",
        type=int,
        default=1024,
        help="Downsample side length cap before LPIPS (0 means no downsample).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Torch device for SSIM/LPIPS.",
    )
    return parser.parse_args()


def resolve_intrinsics(camera) -> Intrinsics:
    model = camera.model
    params = camera.params

    if model == "SIMPLE_PINHOLE":
        fx = fy = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
    elif model == "PINHOLE":
        fx = float(params[0])
        fy = float(params[1])
        cx = float(params[2])
        cy = float(params[3])
    elif model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE"}:
        # Dense stage commonly uses undistorted cameras; use first 3 params as pinhole.
        fx = fy = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
    elif model in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"}:
        # Dense stage commonly uses undistorted cameras; use first 4 params as pinhole.
        fx = float(params[0])
        fy = float(params[1])
        cx = float(params[2])
        cy = float(params[3])
    else:
        raise ValueError(f"Unsupported camera model: {model}")

    return Intrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=int(camera.width),
        height=int(camera.height),
    )


def load_fused_point_cloud(ply_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(ply_path))
    vert = ply["vertex"]
    xyz = np.stack([vert["x"], vert["y"], vert["z"]], axis=1).astype(np.float32)
    rgb = np.stack([vert["red"], vert["green"], vert["blue"]], axis=1).astype(np.float32) / 255.0
    return xyz, rgb


def zbuffer_render(
    xyz_world: np.ndarray,
    rgb_world: np.ndarray,
    rot_w2c: np.ndarray,
    t_w2c: np.ndarray,
    intr: Intrinsics,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render point cloud into camera with one-pixel splat and z-buffer.

    Returns:
      render_rgb: float32 [H, W, 3], initialized as zeros.
      valid_mask: bool [H, W]
    """
    h, w = intr.height, intr.width
    render = np.zeros((h, w, 3), dtype=np.float32)
    mask = np.zeros((h, w), dtype=bool)

    xyz_cam = xyz_world @ rot_w2c.T + t_w2c[None, :]
    z = xyz_cam[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return render, mask

    xyz_cam = xyz_cam[valid]
    z = z[valid]
    colors = rgb_world[valid]

    x = xyz_cam[:, 0] / z
    y = xyz_cam[:, 1] / z
    u = np.rint(intr.fx * x + intr.cx).astype(np.int32)
    v = np.rint(intr.fy * y + intr.cy).astype(np.int32)

    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(inside):
        return render, mask

    u = u[inside]
    v = v[inside]
    z = z[inside]
    colors = colors[inside]

    pix = v * w + u
    order = np.argsort(z, kind="mergesort")
    pix_sorted = pix[order]
    keep = np.empty_like(pix_sorted, dtype=bool)
    keep[0] = True
    keep[1:] = pix_sorted[1:] != pix_sorted[:-1]
    selected = order[keep]

    uu = u[selected]
    vv = v[selected]
    cc = colors[selected]

    render[vv, uu] = cc
    mask[vv, uu] = True
    return render, mask


def masked_psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    diff = pred[mask] - gt[mask]
    mse = float(np.mean(diff * diff))
    mse = max(mse, 1e-12)
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def to_chw_tensor(image_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image_hwc).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)


def maybe_downsample_for_lpips(
    pred: torch.Tensor, gt: torch.Tensor, max_side: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if max_side <= 0:
        return pred, gt
    _, _, h, w = pred.shape
    cur_max = max(h, w)
    if cur_max <= max_side:
        return pred, gt
    scale = max_side / float(cur_max)
    new_h = max(16, int(round(h * scale)))
    new_w = max(16, int(round(w * scale)))
    pred_ds = F.interpolate(pred, size=(new_h, new_w), mode="bilinear", align_corners=False)
    gt_ds = F.interpolate(gt, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return pred_ds, gt_ds


def select_views(images: Iterable, interval: int, max_views: int) -> List:
    sorted_images = sorted(images, key=lambda im: im.name)
    if interval < 1:
        interval = 1
    selected = [im for idx, im in enumerate(sorted_images) if idx % interval == 0]
    if max_views > 0:
        selected = selected[:max_views]
    return selected


def run_eval(args: argparse.Namespace) -> Tuple[dict, List[dict]]:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Make LPIPS use local cache path if available.
    local_hub_dir = DEFAULT_SUGAR_ROOT / "lpipsPyTorch" / "weights"
    if local_hub_dir.exists():
        torch.hub.set_dir(str(local_hub_dir))

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading model from: {args.model_dir}")
    cameras, images, _ = rw_model.read_model(str(args.model_dir))
    print(f"[INFO] Registered images: {len(images)}")

    print(f"[INFO] Loading fused point cloud: {args.ply_path}")
    xyz_world, rgb_world = load_fused_point_cloud(args.ply_path)
    print(f"[INFO] Fused points: {xyz_world.shape[0]}")

    selected_views = select_views(images.values(), args.eval_interval, args.max_views)
    print(f"[INFO] Selected views: {len(selected_views)} (interval={args.eval_interval})")

    lpips_model = LPIPS(net_type=args.lpips_net).to(device).eval()

    per_view: List[dict] = []
    with torch.no_grad():
        for idx, image_meta in enumerate(selected_views, start=1):
            intr = resolve_intrinsics(cameras[image_meta.camera_id])
            gt_path = args.image_dir / image_meta.name
            if not gt_path.exists():
                per_view.append(
                    {
                        "image_name": image_meta.name,
                        "status": "missing_gt",
                        "valid_ratio": 0.0,
                        "psnr": float("nan"),
                        "ssim": float("nan"),
                        "lpips": float("nan"),
                    }
                )
                print(f"[WARN] {idx}/{len(selected_views)} missing GT: {image_meta.name}")
                continue

            gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0
            if gt.shape[0] != intr.height or gt.shape[1] != intr.width:
                # Keep camera-model resolution as source of truth.
                gt = np.asarray(
                    Image.fromarray((gt * 255.0).astype(np.uint8)).resize(
                        (intr.width, intr.height), Image.BILINEAR
                    ),
                    dtype=np.float32,
                ) / 255.0

            rot = image_meta.qvec2rotmat().astype(np.float32)
            tvec = image_meta.tvec.astype(np.float32)
            pred, valid = zbuffer_render(xyz_world, rgb_world, rot, tvec, intr)

            valid_ratio = float(valid.mean())
            if valid_ratio < args.min_valid_ratio:
                per_view.append(
                    {
                        "image_name": image_meta.name,
                        "status": "low_valid_ratio",
                        "valid_ratio": valid_ratio,
                        "psnr": float("nan"),
                        "ssim": float("nan"),
                        "lpips": float("nan"),
                    }
                )
                print(
                    f"[WARN] {idx}/{len(selected_views)} {image_meta.name} "
                    f"skipped (valid_ratio={valid_ratio:.4f})"
                )
                continue

            psnr_val = masked_psnr(pred, gt, valid)

            # For SSIM/LPIPS, keep unknown regions equal to GT so they are neutral.
            pred_eval = pred.copy()
            pred_eval[~valid] = gt[~valid]

            pred_t = to_chw_tensor(pred_eval, device=device)
            gt_t = to_chw_tensor(gt, device=device)
            ssim_val = float(compute_ssim(pred_t, gt_t).item())

            pred_lp, gt_lp = maybe_downsample_for_lpips(pred_t, gt_t, args.lpips_max_side)
            lpips_val = float(lpips_model(pred_lp, gt_lp).item())

            per_view.append(
                {
                    "image_name": image_meta.name,
                    "status": "ok",
                    "valid_ratio": valid_ratio,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "lpips": lpips_val,
                }
            )

            print(
                f"[INFO] {idx}/{len(selected_views)} {image_meta.name} "
                f"valid={valid_ratio:.4f} psnr={psnr_val:.4f} ssim={ssim_val:.4f} lpips={lpips_val:.4f}"
            )

    ok_rows = [r for r in per_view if r["status"] == "ok"]
    mean_psnr = float(np.mean([r["psnr"] for r in ok_rows])) if ok_rows else float("nan")
    mean_ssim = float(np.mean([r["ssim"] for r in ok_rows])) if ok_rows else float("nan")
    mean_lpips = float(np.mean([r["lpips"] for r in ok_rows])) if ok_rows else float("nan")
    mean_valid_ratio = float(np.mean([r["valid_ratio"] for r in ok_rows])) if ok_rows else float("nan")

    summary = {
        "model_dir": str(args.model_dir),
        "image_dir": str(args.image_dir),
        "ply_path": str(args.ply_path),
        "num_registered_images": len(images),
        "num_selected_views": len(selected_views),
        "num_valid_views": len(ok_rows),
        "eval_interval": int(args.eval_interval),
        "min_valid_ratio": float(args.min_valid_ratio),
        "lpips_net": args.lpips_net,
        "lpips_max_side": int(args.lpips_max_side),
        "device": str(device),
        "mean_valid_ratio": mean_valid_ratio,
        "metrics": {
            "PSNR": mean_psnr,
            "SSIM": mean_ssim,
            "LPIPS": mean_lpips,
        },
    }
    return summary, per_view


def write_outputs(output_dir: Path, summary: dict, per_view: List[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = output_dir / "per_view_metrics.csv"
    fieldnames = ["image_name", "status", "valid_ratio", "psnr", "ssim", "lpips"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_view:
            writer.writerow(row)


def main() -> int:
    args = parse_args()

    run_name = f"colmap_dense_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_root / run_name

    t0 = time.time()
    summary, per_view = run_eval(args)
    elapsed = time.time() - t0
    summary["elapsed_seconds"] = elapsed
    write_outputs(output_dir, summary, per_view)

    m = summary["metrics"]
    print("[DONE] COLMAP dense evaluation finished")
    print(f"[DONE] Output dir: {output_dir}")
    print(
        f"[DONE] PSNR={m['PSNR']:.4f}, SSIM={m['SSIM']:.4f}, LPIPS={m['LPIPS']:.4f}, "
        f"valid_views={summary['num_valid_views']}/{summary['num_selected_views']}, "
        f"elapsed={elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
