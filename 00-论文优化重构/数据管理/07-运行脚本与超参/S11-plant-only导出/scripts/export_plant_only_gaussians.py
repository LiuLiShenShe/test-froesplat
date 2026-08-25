#!/usr/bin/env python3
"""Export plant-only Gaussians from a trained 2DGS/3DGS PLY using multi-view masks.

The script is intentionally a post-processing step: it does not retrain the model
and it preserves every original PLY vertex property for the kept Gaussians.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


@dataclass(frozen=True)
class CameraView:
    image_name: str
    width: int
    height: int
    fx: float
    fy: float
    center: np.ndarray
    rotation_c2w: np.ndarray
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a plant-only Gaussian PLY by multi-view mask voting."
    )
    parser.add_argument("--config", type=Path, help="JSON config path.")
    parser.add_argument("--input-ply", type=Path)
    parser.add_argument("--cameras-json", type=Path)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mask-pattern")
    parser.add_argument("--mask-threshold", type=int)
    parser.add_argument("--dilate-mask-px", type=int)
    parser.add_argument("--erode-mask-px", type=int)
    parser.add_argument("--camera-stride", type=int)
    parser.add_argument("--max-cameras", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--min-observations", type=int)
    parser.add_argument("--foreground-ratio-threshold", type=float)
    parser.add_argument(
        "--projection-mode",
        choices=("auto", "positive_z", "negative_z"),
    )
    parser.add_argument("--near", type=float)
    parser.add_argument("--export-background", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def normalize_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "mask_pattern": "mask_{stem}.png",
        "mask_threshold": 127,
        "dilate_mask_px": 5,
        "erode_mask_px": 0,
        "camera_stride": 1,
        "max_cameras": 0,
        "chunk_size": 200_000,
        "min_observations": 3,
        "foreground_ratio_threshold": 0.35,
        "projection_mode": "auto",
        "near": 1.0e-6,
        "export_background": False,
        "dry_run": False,
    }
    config: dict[str, Any] = {}
    if args.config:
        with args.config.open("r", encoding="utf-8") as f:
            config = json.load(f)

    cli_values = {
        "input_ply": args.input_ply,
        "cameras_json": args.cameras_json,
        "mask_dir": args.mask_dir,
        "output_dir": args.output_dir,
        "mask_pattern": args.mask_pattern,
        "mask_threshold": args.mask_threshold,
        "dilate_mask_px": args.dilate_mask_px,
        "erode_mask_px": args.erode_mask_px,
        "camera_stride": args.camera_stride,
        "max_cameras": args.max_cameras,
        "chunk_size": args.chunk_size,
        "min_observations": args.min_observations,
        "foreground_ratio_threshold": args.foreground_ratio_threshold,
        "projection_mode": args.projection_mode,
        "near": args.near,
        "export_background": args.export_background,
        "dry_run": args.dry_run,
    }
    for key, value in cli_values.items():
        if value is not None:
            config[key] = value
    for key, value in defaults.items():
        config.setdefault(key, value)

    required = ("input_ply", "cameras_json", "mask_dir", "output_dir")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"Missing required config values: {', '.join(missing)}")

    path_keys = ("input_ply", "cameras_json", "mask_dir", "output_dir")
    for key in path_keys:
        config[key] = normalize_path(config[key])
    return config


def mask_path_for(mask_dir: Path, pattern: str, image_name: str) -> Path:
    stem = Path(image_name).stem
    return mask_dir / pattern.format(stem=stem, image_name=image_name)


def load_cameras(config: dict[str, Any]) -> tuple[list[CameraView], list[str]]:
    with Path(config["cameras_json"]).open("r", encoding="utf-8") as f:
        camera_entries = json.load(f)

    views: list[CameraView] = []
    missing_masks: list[str] = []
    for item in sorted(camera_entries, key=lambda x: int(x.get("id", 0))):
        mask_path = mask_path_for(
            Path(config["mask_dir"]), config["mask_pattern"], item["img_name"]
        )
        if not mask_path.exists():
            missing_masks.append(str(mask_path))
            continue
        views.append(
            CameraView(
                image_name=item["img_name"],
                width=int(item["width"]),
                height=int(item["height"]),
                fx=float(item["fx"]),
                fy=float(item["fy"]),
                center=np.asarray(item["position"], dtype=np.float32),
                rotation_c2w=np.asarray(item["rotation"], dtype=np.float32),
                mask_path=mask_path,
            )
        )

    stride = max(1, int(config.get("camera_stride", 1)))
    views = views[::stride]
    max_cameras = int(config.get("max_cameras", 0))
    if max_cameras > 0:
        views = views[:max_cameras]
    if not views:
        raise ValueError("No usable cameras with masks were found.")
    return views, missing_masks


def load_mask(view: CameraView, config: dict[str, Any]) -> np.ndarray:
    mask = Image.open(view.mask_path).convert("L")
    if mask.size != (view.width, view.height):
        mask = mask.resize((view.width, view.height), resample=Image.NEAREST)
    mask_arr = np.asarray(mask)
    mask_bin = mask_arr > int(config.get("mask_threshold", 127))

    dilate_px = int(config.get("dilate_mask_px", 0))
    erode_px = int(config.get("erode_mask_px", 0))
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        mask_bin = cv2.dilate(mask_bin.astype(np.uint8), kernel) > 0
    if erode_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)
        )
        mask_bin = cv2.erode(mask_bin.astype(np.uint8), kernel) > 0
    return mask_bin


def project_points(
    xyz: np.ndarray,
    view: CameraView,
    projection_mode: str,
    near: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # cameras.json stores camera-to-world rotation and camera center.
    cam_xyz = (xyz - view.center) @ view.rotation_c2w
    if projection_mode == "positive_z":
        depth = cam_xyz[:, 2]
    elif projection_mode == "negative_z":
        depth = -cam_xyz[:, 2]
    else:
        raise ValueError(f"Unsupported projection mode: {projection_mode}")

    valid = depth > near
    u = np.empty(xyz.shape[0], dtype=np.int32)
    v = np.empty(xyz.shape[0], dtype=np.int32)
    u.fill(-1)
    v.fill(-1)
    valid_idx = np.nonzero(valid)[0]
    if valid_idx.size:
        z = depth[valid_idx]
        u_float = view.fx * (cam_xyz[valid_idx, 0] / z) + (view.width * 0.5)
        v_float = view.fy * (cam_xyz[valid_idx, 1] / z) + (view.height * 0.5)
        inside = (
            (u_float >= 0.0)
            & (u_float < view.width)
            & (v_float >= 0.0)
            & (v_float < view.height)
        )
        inside_idx = valid_idx[inside]
        u[inside_idx] = u_float[inside].astype(np.int32)
        v[inside_idx] = v_float[inside].astype(np.int32)
        valid = np.zeros(xyz.shape[0], dtype=bool)
        valid[inside_idx] = True
    return u, v, valid


def score_projection_mode(
    xyz: np.ndarray,
    views: list[CameraView],
    config: dict[str, Any],
    mode: str,
) -> dict[str, float]:
    sample_count = min(xyz.shape[0], 50_000)
    if xyz.shape[0] > sample_count:
        rng = np.random.default_rng(20260518)
        sample_idx = rng.choice(xyz.shape[0], sample_count, replace=False)
        sample_xyz = xyz[sample_idx]
    else:
        sample_xyz = xyz

    inside_total = 0
    foreground_total = 0
    for view in views[: min(len(views), 24)]:
        mask = load_mask(view, config)
        u, v, valid = project_points(
            sample_xyz, view, mode, float(config.get("near", 1.0e-6))
        )
        if np.any(valid):
            inside_total += int(np.count_nonzero(valid))
            foreground_total += int(np.count_nonzero(mask[v[valid], u[valid]]))

    foreground_ratio = foreground_total / inside_total if inside_total else 0.0
    return {
        "inside_total": float(inside_total),
        "foreground_total": float(foreground_total),
        "foreground_ratio": foreground_ratio,
    }


def choose_projection_mode(
    xyz: np.ndarray, views: list[CameraView], config: dict[str, Any]
) -> tuple[str, dict[str, dict[str, float]]]:
    requested = str(config.get("projection_mode", "auto"))
    if requested != "auto":
        return requested, {}

    scores = {
        mode: score_projection_mode(xyz, views, config, mode)
        for mode in ("positive_z", "negative_z")
    }
    chosen = max(scores, key=lambda mode: scores[mode]["inside_total"])
    if scores[chosen]["inside_total"] <= 0:
        raise ValueError("Projection auto-detection found zero visible points.")
    return chosen, scores


def vote_gaussians(
    xyz: np.ndarray,
    views: list[CameraView],
    config: dict[str, Any],
    projection_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    obs_count = np.zeros(xyz.shape[0], dtype=np.uint16)
    fg_count = np.zeros(xyz.shape[0], dtype=np.uint16)
    chunk_size = int(config.get("chunk_size", 200_000))
    near = float(config.get("near", 1.0e-6))

    for view_idx, view in enumerate(views, start=1):
        mask = load_mask(view, config)
        for start in range(0, xyz.shape[0], chunk_size):
            end = min(start + chunk_size, xyz.shape[0])
            u, v, valid = project_points(xyz[start:end], view, projection_mode, near)
            if not np.any(valid):
                continue
            valid_global = valid
            obs_count[start:end][valid_global] += 1
            foreground = np.zeros(end - start, dtype=bool)
            foreground[valid_global] = mask[v[valid_global], u[valid_global]]
            fg_count[start:end][foreground] += 1
        if view_idx % 25 == 0 or view_idx == len(views):
            print(f"[plant-only] voted {view_idx}/{len(views)} cameras", flush=True)
    return obs_count, fg_count


def write_ply(path: Path, source_ply: PlyData, vertex_data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    elements = [PlyElement.describe(vertex_data, "vertex")]
    out_ply = PlyData(
        elements,
        text=False,
        byte_order="<",
        comments=list(source_ply.comments),
        obj_info=list(source_ply.obj_info),
    )
    out_ply.write(path)


def safe_quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    qs = np.quantile(values.astype(np.float64), [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    keys = ("min", "p25", "p50", "p75", "p90", "p99", "max")
    return {key: float(value) for key, value in zip(keys, qs)}


def main() -> None:
    args = parse_args()
    config = load_config(args)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plant-only] loading PLY: {config['input_ply']}")
    source_ply = PlyData.read(str(config["input_ply"]))
    vertex_data = source_ply["vertex"].data
    xyz = np.column_stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]]).astype(
        np.float32
    )

    views, missing_masks = load_cameras(config)
    projection_mode, projection_scores = choose_projection_mode(xyz, views, config)
    print(f"[plant-only] projection mode: {projection_mode}")
    print(f"[plant-only] cameras with masks: {len(views)}")

    obs_count, fg_count = vote_gaussians(xyz, views, config, projection_mode)
    fg_ratio = np.divide(
        fg_count,
        np.maximum(obs_count, 1),
        out=np.zeros_like(fg_count, dtype=np.float32),
        where=obs_count > 0,
    )
    min_obs = int(config.get("min_observations", 3))
    ratio_threshold = float(config.get("foreground_ratio_threshold", 0.35))
    keep = (obs_count >= min_obs) & (fg_ratio >= ratio_threshold)

    plant_path = output_dir / "plant_only_gaussians.ply"
    report_path = output_dir / "plant_only_report.json"
    score_path = output_dir / "plant_only_scores.npz"

    if not bool(config.get("dry_run", False)):
        write_ply(plant_path, source_ply, vertex_data[keep])
        if bool(config.get("export_background", False)):
            write_ply(output_dir / "background_gaussians.ply", source_ply, vertex_data[~keep])
        np.savez_compressed(
            score_path,
            obs_count=obs_count,
            fg_count=fg_count,
            fg_ratio=fg_ratio,
            keep=keep,
        )

    visible = obs_count >= min_obs
    report = {
        "input_ply": str(config["input_ply"]),
        "cameras_json": str(config["cameras_json"]),
        "mask_dir": str(config["mask_dir"]),
        "output_dir": str(output_dir),
        "plant_only_ply": str(plant_path),
        "score_npz": str(score_path),
        "projection_mode": projection_mode,
        "projection_auto_scores": projection_scores,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in config.items()
        },
        "num_input_gaussians": int(vertex_data.shape[0]),
        "num_kept_gaussians": int(np.count_nonzero(keep)),
        "num_removed_gaussians": int(vertex_data.shape[0] - np.count_nonzero(keep)),
        "kept_ratio": float(np.count_nonzero(keep) / vertex_data.shape[0]),
        "num_visible_min_obs": int(np.count_nonzero(visible)),
        "num_cameras_used": len(views),
        "num_missing_masks": len(missing_masks),
        "missing_masks_preview": missing_masks[:20],
        "obs_count_quantiles": safe_quantiles(obs_count),
        "fg_ratio_quantiles_all": safe_quantiles(fg_ratio),
        "fg_ratio_quantiles_visible": safe_quantiles(fg_ratio[visible]),
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
