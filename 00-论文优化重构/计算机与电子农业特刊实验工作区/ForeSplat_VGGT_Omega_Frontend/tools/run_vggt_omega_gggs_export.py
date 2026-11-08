#!/usr/bin/env python3
"""ForeSplat GGGS frontend export (command line, no Gradio).

Pipeline::

    RAP-FSAM3 images/ + masks/  ->  mask-aware VGGT-Omega preprocessing
                                ->  VGGT-Omega inference (camera + depth)
                                ->  foreground-mask-constrained depth back-projection
                                ->  COLMAP sparse model (original-resolution intrinsics)
                                ->  GGGS dataset (COLMAP + points3D.ply + metadata)

Example::

    python tools/run_vggt_omega_gggs_export.py \
        --images-dir /path/to/rap_fsam3/images \
        --masks-dir /path/to/rap_fsam3/masks \
        --checkpoint /path/to/vggt_omega_1b_512.pt \
        --output-dir /path/to/GGGS_Dataset \
        --image-resolution 512 \
        --preprocess-mode balanced \
        --confidence-percentile 20 \
        --max-points 1000000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from foresplat_adapter.dataset import collect_frames, foreground_ratios  # noqa: E402
from foresplat_adapter.export import export_gggs_dataset  # noqa: E402
from foresplat_adapter.inference import load_model, run_inference  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images_with_masks  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAP-FSAM3 -> VGGT-Omega -> COLMAP -> GGGS dataset (ForeSplat frontend)"
    )
    parser.add_argument("--images-dir", required=True, help="RAP-FSAM3 foreground image directory")
    parser.add_argument("--masks-dir", required=True, help="RAP-FSAM3 binary foreground mask directory")
    parser.add_argument("--checkpoint", required=True, help="VGGT-Omega checkpoint (.pt)")
    parser.add_argument("--output-dir", required=True, help="GGGS dataset output directory")

    parser.add_argument("--image-resolution", type=int, default=512, help="VGGT-Omega input resolution")
    parser.add_argument("--preprocess-mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument("--patch-size", type=int, default=16, help="must divide --image-resolution")

    parser.add_argument("--confidence-percentile", type=float, default=20.0,
                        help="confidence percentile (0-100) computed on foreground pixels only")
    parser.add_argument("--max-points", type=int, default=1000000, help="maximum number of 3D points")
    parser.add_argument("--min-conf", type=float, default=1e-5, help="absolute confidence floor")

    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--foreground-background", choices=["black", "white"], default="black",
                        help="background used when compositing RGBA foreground images")

    parser.add_argument("--image-prefix-to-strip", default=None,
                        help="prefix stripped from image stems before mask pairing (e.g. 'crop_')")
    parser.add_argument("--mask-prefix-to-strip", default=None,
                        help="prefix stripped from mask stems before mask pairing (e.g. 'mask_')")

    parser.add_argument("--frame-stride", type=int, default=1, help="subset every k-th view (order preserved)")
    parser.add_argument("--max-frames", type=int, default=None, help="cap the number of views")

    parser.add_argument("--filter-depth-edges", action="store_true",
                        help="enable depth-edge filtering (OFF by default: leaf silhouettes are real geometry)")
    parser.add_argument("--depth-edge-rtol", type=float, default=0.03)

    parser.add_argument("--no-align", action="store_true",
                        help="do not apply the community scene alignment transform")

    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--gggs-repo", default=None,
                        help="path to a GGGS clone for readColmapSceneInfo verification")
    parser.add_argument("--skip-verify", action="store_true", help="skip the GGGS reader verification")
    parser.add_argument("--verify-resolution", type=int, default=1)
    parser.add_argument("--save-predictions", action="store_true",
                        help="store the raw VGGT-Omega predictions (depth/conf/poses) as debug/predictions.npz")
    parser.add_argument("--save-log", default=None, help="write the run log to this path (default: output_dir/metadata/run_log.json)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = args.device if args.device != "auto" else ("cuda" if _cuda_available() else "cpu")
    run_args = dict(vars(args))
    run_args["device_resolved"] = device
    log_lines: list[str] = []

    def log(message: str) -> None:
        text = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(text, flush=True)
        log_lines.append(text)

    os.makedirs(args.output_dir, exist_ok=True)

    log(f"collecting frames from {args.images_dir} + {args.masks_dir}")
    frames = collect_frames(
        args.images_dir,
        args.masks_dir,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        image_prefix=args.image_prefix_to_strip,
        mask_prefix=args.mask_prefix_to_strip,
    )
    log(f"{len(frames)} frame(s) selected (order preserved)")
    image_paths = [f.image_path for f in frames]
    mask_paths = [f.mask_path for f in frames]
    log("foreground ratios (original resolution): " + ", ".join(
        f"{name}={ratio:.4f}" for name, ratio in zip(
            [f.image_name for f in frames], foreground_ratios(mask_paths, args.mask_threshold)
        )
    ))

    log("preprocessing images + masks (identical geometry)")
    images, masks_model, transforms = load_and_preprocess_images_with_masks(
        image_paths,
        mask_paths,
        mode=args.preprocess_mode,
        image_resolution=args.image_resolution,
        patch_size=args.patch_size,
        background=args.foreground_background,
        mask_threshold=args.mask_threshold,
    )
    log(f"model input tensor: {tuple(images.shape)}, mask tensor: {tuple(masks_model.shape)}, "
        f"foreground pixel fraction: {masks_model.float().mean().item():.4f}")

    log(f"loading checkpoint {args.checkpoint} on {device}")
    model = load_model(args.checkpoint, device=device)

    log("running VGGT-Omega inference")
    predictions = run_inference(model, images, device=device)

    if args.save_predictions:
        npz_path = os.path.join(args.output_dir, "debug", "predictions.npz")
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        np.savez_compressed(npz_path, **{k: v for k, v in predictions.items() if isinstance(v, np.ndarray)})
        log(f"raw predictions saved to {npz_path}")

    log("exporting GGGS dataset (foreground-mask-constrained COLMAP)")
    summary = export_gggs_dataset(
        predictions=predictions,
        transforms=transforms,
        masks_model=masks_model,
        frames=frames,
        output_dir=args.output_dir,
        conf_thres=args.confidence_percentile,
        max_points=args.max_points,
        filter_depth_edges=args.filter_depth_edges,
        depth_edge_rtol=args.depth_edge_rtol,
        apply_alignment=not args.no_align,
        min_conf=args.min_conf,
        mask_threshold=args.mask_threshold,
        seed=args.seed,
        image_resolution=args.image_resolution,
        preprocess_mode=args.preprocess_mode,
        checkpoint_path=args.checkpoint,
        run_args=run_args,
    )
    log(f"export finished: {summary}")

    report_path = args.save_log or os.path.join(args.output_dir, "metadata", "run_log.json")
    report = {
        "run_args": run_args,
        "num_frames": summary["num_frames"],
        "num_points3d": summary["num_points3d"],
        "per_frame_point2d": summary["per_frame_point2d"],
        "frames": [
            {
                "image_name": f.image_name,
                "source_image": f.image_path,
                "source_mask": f.mask_path,
            }
            for f in frames
        ],
        "log": log_lines,
    }
    if not args.skip_verify:
        if args.gggs_repo is None:
            log("skip GGGS verification: --gggs-repo not provided")
            report["gggs_verification"] = {"skipped": True, "reason": "no --gggs-repo"}
        else:
            try:
                from foresplat_adapter.verify import verify_gggs_dataset

                log(f"verifying dataset with GGGS reader from {args.gggs_repo}")
                verification = verify_gggs_dataset(args.output_dir, args.gggs_repo, args.verify_resolution)
                report["gggs_verification"] = verification
                log(f"GGGS verification all_checks_passed={verification['all_checks_passed']} "
                    f"points={verification['point_cloud']['num_points']}")
            except Exception as exc:  # noqa: BLE001
                log(f"GGGS verification failed: {exc!r}")
                report["gggs_verification"] = {"failed": True, "error": repr(exc)}
    else:
        report["gggs_verification"] = {"skipped": True, "reason": "--skip-verify"}

    _dump_json(report_path, report)
    log(f"run report written to {report_path}")
    return 0


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _dump_json(path: str, payload: dict) -> None:
    import json

    def _default(obj):
        if isinstance(obj, (np.floating, np.integer, np.bool_)):
            return obj.item()
        return str(obj)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_default)


if __name__ == "__main__":
    raise SystemExit(main())
