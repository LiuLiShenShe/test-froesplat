#!/usr/bin/env python3
"""ExG + threshold baseline with unified target definition and post-processing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_INPUT_BASE_DIR = Path("/data/fj/02-FFT")
DEFAULT_OUTPUT_BASE_DIR = Path("/data/fj/results_seg_compare/01_exg_otsu")
DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_TARGET_DEFINITION = "green_plant_canopy"


def parse_extensions(raw: str) -> set[str]:
    exts = {part.strip().lower() for part in raw.split(",") if part.strip()}
    normalized = set()
    for ext in exts:
        normalized.add(ext if ext.startswith(".") else f".{ext}")
    return normalized or set(DEFAULT_EXTENSIONS)


def list_images(folder: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions
    )


def compute_exg_u8(image_bgr: np.ndarray) -> np.ndarray:
    image_f = image_bgr.astype(np.float32)
    b, g, r = image_f[:, :, 0], image_f[:, :, 1], image_f[:, :, 2]
    total = r + g + b + 1e-6
    exg = (2.0 * g - r - b) / total
    emin, emax = float(exg.min()), float(exg.max())
    if emax - emin < 1e-8:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    return ((exg - emin) / (emax - emin) * 255).astype(np.uint8)


def threshold_exg(exg_u8: np.ndarray, mode: str, percentile: float) -> tuple[np.ndarray, float]:
    if mode == "percentile":
        threshold = float(np.percentile(exg_u8, np.clip(percentile, 0.0, 100.0)))
        binary = (exg_u8 >= threshold).astype(np.uint8) * 255
        return binary, threshold
    threshold, binary = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, float(threshold)


def fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    h, w = mask_u8.shape
    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    return mask_u8 | cv2.bitwise_not(flood)


def remove_small_components(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask_u8
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 1:
        return mask_u8
    keep = np.zeros_like(mask_u8)
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            keep[labels == label] = 255
    return keep


def keep_largest_components(mask_u8: np.ndarray, keep_components: int) -> np.ndarray:
    if keep_components <= 0:
        return mask_u8
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 1:
        return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return mask_u8
    order = np.argsort(areas)[::-1]
    keep_ids = order[:keep_components] + 1
    keep_mask = np.isin(labels, keep_ids)
    return (keep_mask * 255).astype(np.uint8)


def postprocess_mask(
    mask_u8: np.ndarray,
    open_kernel_size: int,
    close_kernel_size: int,
    fill_holes_flag: bool,
    keep_components: int,
    min_area_ratio: float,
) -> np.ndarray:
    h, w = mask_u8.shape
    total_pixels = h * w
    min_area = max(int(total_pixels * max(min_area_ratio, 0.0)), 1)

    if open_kernel_size > 1:
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size)
        )
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel)

    if close_kernel_size > 1:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
        )
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)

    if fill_holes_flag:
        mask_u8 = fill_holes(mask_u8)

    mask_u8 = remove_small_components(mask_u8, min_area=min_area)
    mask_u8 = keep_largest_components(mask_u8, keep_components=keep_components)
    return mask_u8


def segment_image(
    image_path: Path,
    threshold_mode: str,
    threshold_percentile: float,
    pp_open_kernel_size: int,
    pp_close_kernel_size: int,
    pp_fill_holes: bool,
    pp_keep_components: int,
    pp_min_area_ratio: float,
) -> tuple[np.ndarray, dict]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        empty = np.zeros((1, 1), dtype=np.uint8)
        return empty, {"error": "cannot_read_image"}

    exg_u8 = compute_exg_u8(image)
    binary, threshold = threshold_exg(
        exg_u8, mode=threshold_mode, percentile=threshold_percentile
    )
    raw_cov = float(np.count_nonzero(binary) / binary.size * 100.0)

    binary = postprocess_mask(
        binary,
        open_kernel_size=max(pp_open_kernel_size, 1),
        close_kernel_size=max(pp_close_kernel_size, 1),
        fill_holes_flag=pp_fill_holes,
        keep_components=max(pp_keep_components, 0),
        min_area_ratio=max(pp_min_area_ratio, 0.0),
    )

    clean_cov = float(np.count_nonzero(binary) / binary.size * 100.0)
    n_final, _, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    info = {
        "threshold_mode": threshold_mode,
        "threshold_value": round(float(threshold), 3),
        "raw_coverage_pct": round(raw_cov, 2),
        "clean_coverage_pct": round(clean_cov, 2),
        "final_regions": max(int(n_final) - 1, 0),
    }
    return binary, info


def save_outputs(image_path: Path, mask_u8: np.ndarray, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    mask_path = output_dir / f"mask_{stem}.png"
    crop_path = output_dir / f"crop_{stem}.png"

    cv2.imwrite(str(mask_path), mask_u8)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    crop = image.copy()
    crop[mask_u8 == 0] = 0
    cv2.imwrite(str(crop_path), crop)
    return mask_path, crop_path


def process_folder(
    folder_name: str,
    input_base_dir: Path,
    output_base_dir: Path,
    threshold_mode: str,
    threshold_percentile: float,
    pp_open_kernel_size: int,
    pp_close_kernel_size: int,
    pp_fill_holes: bool,
    pp_keep_components: int,
    pp_min_area_ratio: float,
    extensions: set[str],
    force: bool,
    limit: int | None,
) -> dict:
    input_dir = input_base_dir / folder_name
    output_dir = output_base_dir / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir, extensions)
    if limit is not None and limit > 0:
        images = images[:limit]
    if not images:
        return {"folder": folder_name, "images": 0, "skipped": True, "reason": "no_images"}

    log_path = output_dir / "segmentation_log.json"
    if log_path.exists() and not force:
        try:
            existing = json.loads(log_path.read_text(encoding="utf-8"))
            if isinstance(existing, list) and len(existing) >= len(images):
                print(f"[SKIP] {folder_name}: {len(existing)} images already done")
                return {"folder": folder_name, "images": len(existing), "skipped": True}
        except json.JSONDecodeError:
            pass

    print(f"[RUN ] {folder_name}: {len(images)} images")
    started = time.time()
    records = []

    for idx, image_path in enumerate(images, start=1):
        t0 = time.time()
        mask, info = segment_image(
            image_path=image_path,
            threshold_mode=threshold_mode,
            threshold_percentile=threshold_percentile,
            pp_open_kernel_size=pp_open_kernel_size,
            pp_close_kernel_size=pp_close_kernel_size,
            pp_fill_holes=pp_fill_holes,
            pp_keep_components=pp_keep_components,
            pp_min_area_ratio=pp_min_area_ratio,
        )
        mask_path, crop_path = save_outputs(image_path, mask, output_dir)
        elapsed = time.time() - t0

        record = {
            "image": image_path.name,
            "mask_path": str(mask_path),
            "crop_path": str(crop_path),
            "time_sec": round(elapsed, 4),
            **info,
        }
        records.append(record)

        if idx == 1 or idx == len(images) or idx % 50 == 0:
            print(
                f"       [{idx:>4}/{len(images)}] "
                f"thr={record['threshold_value']:.1f} "
                f"cov={record['raw_coverage_pct']:.1f}%->{record['clean_coverage_pct']:.1f}% "
                f"t={record['time_sec']:.3f}s"
            )

    log_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    total = time.time() - started
    print(f"[DONE] {folder_name}: {total:.1f}s ({total / len(images):.3f}s/img)")
    return {"folder": folder_name, "images": len(images), "time_sec": round(total, 2)}


def discover_folders(input_base_dir: Path, extensions: set[str]) -> list[str]:
    folders: list[str] = []
    for path in sorted(input_base_dir.iterdir()):
        if path.is_dir() and list_images(path, extensions):
            folders.append(path.name)
    return folders


def validate_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ExG threshold baseline")
    parser.add_argument("--input-base-dir", type=Path, default=DEFAULT_INPUT_BASE_DIR)
    parser.add_argument("--output-base-dir", type=Path, default=DEFAULT_OUTPUT_BASE_DIR)
    parser.add_argument("--folder", type=str, default=None, help="Process a single folder only")
    parser.add_argument("--extensions", type=str, default=".jpg,.jpeg,.png,.bmp")
    parser.add_argument("--force", action="store_true", help="Re-run even if log exists")
    parser.add_argument("--limit", type=int, default=None, help="Process first N images only")

    parser.add_argument("--target-definition", type=str, default=DEFAULT_TARGET_DEFINITION)
    parser.add_argument(
        "--threshold-mode",
        type=str,
        choices=["otsu", "percentile"],
        default="otsu",
    )
    parser.add_argument("--threshold-percentile", type=float, default=65.0)

    parser.add_argument("--pp-open-kernel-size", type=int, default=3)
    parser.add_argument("--pp-close-kernel-size", type=int, default=9)
    parser.add_argument(
        "--pp-fill-holes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pp-keep-components", type=int, default=1)
    parser.add_argument("--pp-min-area-ratio", type=float, default=0.0005)
    args = parser.parse_args()

    extensions = parse_extensions(args.extensions)
    validate_dirs([args.input_base_dir])
    args.output_base_dir.mkdir(parents=True, exist_ok=True)

    folders = [args.folder] if args.folder else discover_folders(args.input_base_dir, extensions)
    if not folders:
        print("No folders with images were found.")
        return

    print(f"ExG baseline | folders={len(folders)} | target={args.target_definition}")
    print(f"Input:  {args.input_base_dir}")
    print(f"Output: {args.output_base_dir}")
    print("-" * 72)

    started = time.time()
    summary = []
    for index, folder in enumerate(folders, start=1):
        print(f"[{index}/{len(folders)}] {folder}")
        result = process_folder(
            folder_name=folder,
            input_base_dir=args.input_base_dir,
            output_base_dir=args.output_base_dir,
            threshold_mode=args.threshold_mode,
            threshold_percentile=args.threshold_percentile,
            pp_open_kernel_size=max(args.pp_open_kernel_size, 1),
            pp_close_kernel_size=max(args.pp_close_kernel_size, 1),
            pp_fill_holes=bool(args.pp_fill_holes),
            pp_keep_components=max(args.pp_keep_components, 0),
            pp_min_area_ratio=max(args.pp_min_area_ratio, 0.0),
            extensions=extensions,
            force=args.force,
            limit=args.limit,
        )
        summary.append(result)

    total = time.time() - started
    batch_summary = {
        "method": "ExG_threshold",
        "target_definition": args.target_definition,
        "total_folders": len(folders),
        "total_time_sec": round(total, 2),
        "parameters": {
            "threshold_mode": args.threshold_mode,
            "threshold_percentile": args.threshold_percentile,
            "pp_open_kernel_size": max(args.pp_open_kernel_size, 1),
            "pp_close_kernel_size": max(args.pp_close_kernel_size, 1),
            "pp_fill_holes": bool(args.pp_fill_holes),
            "pp_keep_components": max(args.pp_keep_components, 0),
            "pp_min_area_ratio": max(args.pp_min_area_ratio, 0.0),
        },
        "folders": summary,
    }
    summary_path = args.output_base_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")
    print("-" * 72)
    print(f"All done in {total:.1f}s. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
