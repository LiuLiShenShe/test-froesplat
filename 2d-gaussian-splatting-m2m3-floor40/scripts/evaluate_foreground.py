#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 21 — foreground-object reconstruction evaluation.

Renders a trained ForeSplat checkpoint over the evaluation cameras and scores it
with the manuscript's foreground criteria (utils/foreground_metrics.py) instead
of a full-image PSNR, which is meaningless for a model whose background is
intentionally black.

Usage (trainer venv, CUDA):

    python evaluate_foreground.py \
        --model_path  <output dir of train.py> \
        --source_path <dataset root with images/ and sparse/> \
        --iteration   30000 \
        --mask_dir    <RAP-FSAM3 mask dir> \
        --mask_pattern "mask_{stem}.png" \
        --output_json metrics/foreground_eval.json

Only real numbers are written.  Missing masks, missing renders or a missing
LPIPS backend produce explicit nulls / MISSING counts, never fabricated values.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

TRAINER = Path(__file__).resolve().parents[1]
if str(TRAINER) not in sys.path:
    sys.path.insert(0, str(TRAINER))
# upstream 2DGS submodules are imported by absolute name, mirroring train.py
for extra in (TRAINER / "submodules/simple-knn",
              TRAINER / "submodules/diff-surfel-rasterization"):
    if extra.exists() and str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scene import Scene  # noqa: E402
from gaussian_renderer import render, GaussianModel  # noqa: E402
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from utils import foreground_metrics as fm  # noqa: E402
from utils import plant_aware_utils as pu  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="ForeSplat foreground evaluation")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    # --mask_dir / --mask_pattern / --mask_threshold come from ModelParams;
    # they are forwarded to the mask loader as-is.
    parser.add_argument("--output_json", default="")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--with_lpips", action="store_true")
    parser.add_argument("--boundary_ignore", type=int, default=2)
    parser.add_argument("--quiet", action="store_true")
    return model, pipeline, parser


def main():
    model_group, pipeline_group, parser = build_parser()
    args = get_combined_args(parser)
    dataset, pipe = model_group.extract(args), pipeline_group.extract(args)

    if not args.mask_dir:
        raise SystemExit("--mask_dir is required: foreground metrics need the RAP-FSAM3 mask "
                         "that was used as the training input (no baseline substitution).")

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    cameras = scene.getTrainCameras() if args.split == "train" else scene.getTestCameras()
    if not cameras:
        raise SystemExit(f"no cameras in split {args.split!r}")

    per_view = []
    missing_masks = []
    with torch.no_grad():
        for cam in cameras:
            image_name = cam.image_name
            try:
                mask_path = pu.resolve_mask_path(args.mask_dir, image_name, args.mask_pattern)
            except FileNotFoundError:
                missing_masks.append(image_name)
                continue
            # Identity closure: prefer the exact tensor the trainer used, and
            # hash the file it came from so the metric stays traceable.
            if getattr(cam, "gt_alpha_mask", None) is not None:
                mask = cam.gt_alpha_mask.float()
            else:
                mask = pu.load_mask_tensor(
                    args.mask_dir, image_name,
                    (cam.image_width, cam.image_height),
                    pattern=args.mask_pattern, threshold=args.mask_threshold)
            mask = mask.to(background.device)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)

            rendering = render(cam, gaussians, pipe, background)["render"]
            gt_image = cam.original_image if hasattr(cam, "original_image") else cam.image
            gt_image = gt_image.to(rendering.device).float()
            if gt_image.shape[-2:] != rendering.shape[-2:]:
                raise SystemExit(
                    f"resolution mismatch for {image_name}: render {tuple(rendering.shape)} "
                    f"vs gt {tuple(gt_image.shape)}")
            if mask.shape[-2:] != rendering.shape[-2:]:
                import torch.nn.functional as F
                mask = F.interpolate(mask[None], size=rendering.shape[-2:], mode="nearest")[0]

            metrics = fm.compute_foreground_metrics(
                rendering, gt_image, mask, boundary_ignore=args.boundary_ignore,
                with_lpips=args.with_lpips)
            metrics["image"] = image_name
            metrics["mask_sha256"] = pu.file_sha256(mask_path)
            per_view.append(metrics)
            if not args.quiet:
                print(f"[{args.split}] {image_name}: psnr_fg={metrics['psnr_fg']:.3f} "
                      f"ssim_fg={metrics['ssim_fg']:.4f} "
                      f"leak={metrics['leakage_energy_ratio']:.4f}")

    if not per_view:
        raise SystemExit(f"no view could be evaluated; missing masks: {len(missing_masks)}")

    def aggregate(key):
        values = [m[key] for m in per_view if m.get(key) is not None and np.isfinite(m.get(key, float("nan")))]
        if not values:
            return None
        return {"mean": float(np.mean(values)), "min": float(np.min(values)),
                "max": float(np.max(values)), "count": int(len(values))}

    summary = {key: aggregate(key) for key in
               ("psnr_fg", "ssim_fg", "outside_mask_non_black_ratio",
                "leakage_energy_ratio", "foreground_fraction")}
    if args.with_lpips:
        summary["lpips_fg"] = aggregate("lpips_fg")
        # record which implementation produced the numbers (the vendored
        # lpipsPyTorch copy in this workspace is corrupted, so the pip `lpips`
        # backend is what actually runs)
        summary["lpips_backend"] = fm.lpips_backend_name()
    summary["evaluated_views"] = len(per_view)
    summary["missing_masks"] = missing_masks
    summary["split"] = args.split
    summary["model_path"] = str(args.model_path)
    summary["mask_dir"] = str(args.mask_dir)
    summary["mask_pattern"] = args.mask_pattern
    summary["boundary_ignore"] = args.boundary_ignore

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "per_view": per_view},
                                  indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
