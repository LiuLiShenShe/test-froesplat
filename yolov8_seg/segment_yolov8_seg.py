#!/usr/bin/env python3
"""YOLOv8-Seg baseline with unified target definition and post-processing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ultralytics is not installed. Install with: pip install ultralytics"
    ) from exc


DEFAULT_INPUT_BASE_DIR = Path("/data/fj/02-FFT")
DEFAULT_OUTPUT_BASE_DIR = Path("/data/fj/results_seg_compare/02_yolov8_seg")
DEFAULT_MODEL_PATH = Path("/data/fj/yolov8_seg/yolov8n-seg.pt")
DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_TARGET_DEFINITION = "green_plant_canopy"


def parse_extensions(raw: str) -> set[str]:
    exts = {part.strip().lower() for part in raw.split(",") if part.strip()}
    normalized = set()
    for ext in exts:
        normalized.add(ext if ext.startswith(".") else f".{ext}")
    return normalized or set(DEFAULT_EXTENSIONS)


def parse_class_ids(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        return None
    return {int(v) for v in values}


def list_images(folder: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions
    )


def resolve_class_name(names: dict | list | None, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def compute_exg_u8(image_bgr: np.ndarray) -> np.ndarray:
    image_f = image_bgr.astype(np.float32)
    b, g, r = image_f[:, :, 0], image_f[:, :, 1], image_f[:, :, 2]
    total = r + g + b + 1e-6
    exg = (2.0 * g - r - b) / total
    emin, emax = float(exg.min()), float(exg.max())
    if emax - emin < 1e-8:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    return ((exg - emin) / (emax - emin) * 255).astype(np.uint8)


def green_mask_from_exg(
    exg_u8: np.ndarray,
    mode: str,
    percentile: float,
    fixed_threshold: float,
) -> tuple[np.ndarray, float]:
    if mode == "percentile":
        threshold = float(np.percentile(exg_u8, np.clip(percentile, 0.0, 100.0)))
        return (exg_u8 >= threshold), threshold
    if mode == "fixed":
        threshold = float(np.clip(fixed_threshold, 0.0, 255.0))
        return (exg_u8 >= threshold), threshold
    threshold, binary = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (binary > 0), float(threshold)


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
    model: YOLO,
    image_path: Path,
    class_ids: set[int] | None,
    conf: float,
    iou: float,
    mask_threshold: float,
    imgsz: int,
    device: str,
    green_refine: bool,
    green_threshold_mode: str,
    green_threshold_percentile: float,
    green_threshold_fixed: float,
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

    h, w = image.shape[:2]
    result = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]

    total_det = int(len(result.boxes)) if result.boxes is not None else 0
    mask_data = None
    cls_ids = None
    if result.masks is not None:
        mask_data = result.masks.data.detach().cpu().numpy()
    if result.boxes is not None and result.boxes.cls is not None:
        cls_ids = result.boxes.cls.detach().cpu().numpy().astype(int)

    def build_merged(active_class_ids: set[int] | None) -> tuple[np.ndarray, int, list[str]]:
        merged_local = np.zeros((h, w), dtype=np.uint8)
        det_local = 0
        classes_local: list[str] = []
        if mask_data is None:
            return merged_local, det_local, classes_local
        for idx, mask_prob in enumerate(mask_data):
            cls_id = int(cls_ids[idx]) if cls_ids is not None and idx < len(cls_ids) else -1
            if active_class_ids is not None and cls_id not in active_class_ids:
                continue
            mask_bin = (mask_prob >= mask_threshold).astype(np.uint8)
            if mask_bin.shape != (h, w):
                mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
            merged_local |= mask_bin
            det_local += 1
            if cls_id >= 0:
                classes_local.append(resolve_class_name(result.names, cls_id))
        return merged_local, det_local, classes_local

    merged, selected_det, selected_classes = build_merged(class_ids)
    class_filter_fallback = False
    if selected_det == 0 and class_ids is not None and total_det > 0:
        merged, selected_det, selected_classes = build_merged(None)
        class_filter_fallback = True

    raw_cov_before_green = float(np.count_nonzero(merged) / merged.size * 100.0)
    green_threshold_value = None
    green_refine_fallback = False
    if green_refine:
        exg_u8 = compute_exg_u8(image)
        green_mask, green_threshold_value = green_mask_from_exg(
            exg_u8,
            mode=green_threshold_mode,
            percentile=green_threshold_percentile,
            fixed_threshold=green_threshold_fixed,
        )
        refined = ((merged > 0) & green_mask).astype(np.uint8)
        if refined.sum() == 0 and merged.sum() > 0:
            green_refine_fallback = True
        else:
            merged = refined

    raw_cov_after_green = float(np.count_nonzero(merged) / merged.size * 100.0)
    merged = postprocess_mask(
        merged * 255,
        open_kernel_size=max(pp_open_kernel_size, 1),
        close_kernel_size=max(pp_close_kernel_size, 1),
        fill_holes_flag=pp_fill_holes,
        keep_components=max(pp_keep_components, 0),
        min_area_ratio=max(pp_min_area_ratio, 0.0),
    )

    clean_coverage = float(np.count_nonzero(merged) / merged.size * 100.0)
    n_final, _, _, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)

    info = {
        "total_detections": total_det,
        "selected_detections": selected_det,
        "selected_classes": sorted(set(selected_classes)),
        "class_filter_fallback": class_filter_fallback,
        "green_refine_fallback": green_refine_fallback,
        "raw_coverage_before_green_pct": round(raw_cov_before_green, 2),
        "raw_coverage_after_green_pct": round(raw_cov_after_green, 2),
        "green_threshold_mode": green_threshold_mode if green_refine else "disabled",
        "green_threshold_value": None if green_threshold_value is None else round(float(green_threshold_value), 3),
        "clean_coverage_pct": round(clean_coverage, 2),
        "final_regions": max(int(n_final) - 1, 0),
    }
    return merged, info


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
    model: YOLO,
    folder_name: str,
    input_base_dir: Path,
    output_base_dir: Path,
    extensions: set[str],
    force: bool,
    class_ids: set[int] | None,
    conf: float,
    iou: float,
    mask_threshold: float,
    imgsz: int,
    device: str,
    green_refine: bool,
    green_threshold_mode: str,
    green_threshold_percentile: float,
    green_threshold_fixed: float,
    pp_open_kernel_size: int,
    pp_close_kernel_size: int,
    pp_fill_holes: bool,
    pp_keep_components: int,
    pp_min_area_ratio: float,
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
        mask_u8, info = segment_image(
            model=model,
            image_path=image_path,
            class_ids=class_ids,
            conf=conf,
            iou=iou,
            mask_threshold=mask_threshold,
            imgsz=imgsz,
            device=device,
            green_refine=green_refine,
            green_threshold_mode=green_threshold_mode,
            green_threshold_percentile=green_threshold_percentile,
            green_threshold_fixed=green_threshold_fixed,
            pp_open_kernel_size=pp_open_kernel_size,
            pp_close_kernel_size=pp_close_kernel_size,
            pp_fill_holes=pp_fill_holes,
            pp_keep_components=pp_keep_components,
            pp_min_area_ratio=pp_min_area_ratio,
        )
        mask_path, crop_path = save_outputs(image_path, mask_u8, output_dir)
        elapsed = time.time() - t0

        record = {
            "image": image_path.name,
            "mask_path": str(mask_path),
            "crop_path": str(crop_path),
            "time_sec": round(elapsed, 4),
            **info,
        }
        records.append(record)

        if idx == 1 or idx == len(images) or idx % 20 == 0:
            print(
                f"       [{idx:>4}/{len(images)}] "
                f"det={record['selected_detections']}/{record['total_detections']} "
                f"cov={record['raw_coverage_after_green_pct']:.1f}%->{record['clean_coverage_pct']:.1f}% "
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
    parser = argparse.ArgumentParser(description="YOLOv8-Seg baseline")
    parser.add_argument("--input-base-dir", type=Path, default=DEFAULT_INPUT_BASE_DIR)
    parser.add_argument("--output-base-dir", type=Path, default=DEFAULT_OUTPUT_BASE_DIR)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--extensions", type=str, default=".jpg,.jpeg,.png,.bmp")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Process first N images only")

    parser.add_argument("--target-definition", type=str, default=DEFAULT_TARGET_DEFINITION)
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--classes",
        type=str,
        default="",
        help="Comma-separated class IDs, e.g. '58'. Empty means all classes.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--mask-threshold", type=float, default=0.5)

    parser.add_argument(
        "--green-refine",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--green-threshold-mode",
        type=str,
        choices=["otsu", "percentile", "fixed"],
        default="otsu",
    )
    parser.add_argument("--green-threshold-percentile", type=float, default=65.0)
    parser.add_argument("--green-threshold-fixed", type=float, default=128.0)

    parser.add_argument("--pp-open-kernel-size", type=int, default=3)
    parser.add_argument("--pp-close-kernel-size", type=int, default=9)
    parser.add_argument(
        "--pp-fill-holes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pp-keep-components", type=int, default=0)
    parser.add_argument("--pp-min-area-ratio", type=float, default=0.0002)
    args = parser.parse_args()

    extensions = parse_extensions(args.extensions)
    class_ids = parse_class_ids(args.classes)

    validate_dirs([args.input_base_dir])
    args.output_base_dir.mkdir(parents=True, exist_ok=True)

    folders = [args.folder] if args.folder else discover_folders(args.input_base_dir, extensions)
    if not folders:
        print("No folders with images were found.")
        return

    print(f"YOLOv8-Seg | folders={len(folders)} | target={args.target_definition}")
    print(f"Input:  {args.input_base_dir}")
    print(f"Output: {args.output_base_dir}")
    print(f"Model:  {args.model} | Device: {args.device} | Classes: {class_ids}")
    print("-" * 72)

    model = YOLO(args.model)
    started = time.time()
    summary = []

    for index, folder in enumerate(folders, start=1):
        print(f"[{index}/{len(folders)}] {folder}")
        result = process_folder(
            model=model,
            folder_name=folder,
            input_base_dir=args.input_base_dir,
            output_base_dir=args.output_base_dir,
            extensions=extensions,
            force=args.force,
            class_ids=class_ids,
            conf=args.conf,
            iou=args.iou,
            mask_threshold=args.mask_threshold,
            imgsz=max(args.imgsz, 64),
            device=args.device,
            green_refine=bool(args.green_refine),
            green_threshold_mode=args.green_threshold_mode,
            green_threshold_percentile=args.green_threshold_percentile,
            green_threshold_fixed=args.green_threshold_fixed,
            pp_open_kernel_size=max(args.pp_open_kernel_size, 1),
            pp_close_kernel_size=max(args.pp_close_kernel_size, 1),
            pp_fill_holes=bool(args.pp_fill_holes),
            pp_keep_components=max(args.pp_keep_components, 0),
            pp_min_area_ratio=max(args.pp_min_area_ratio, 0.0),
            limit=args.limit,
        )
        summary.append(result)

    total = time.time() - started
    batch_summary = {
        "method": "YOLOv8-Seg",
        "target_definition": args.target_definition,
        "total_folders": len(folders),
        "total_time_sec": round(total, 2),
        "parameters": {
            "model": args.model,
            "classes": sorted(class_ids) if class_ids is not None else "all",
            "conf": args.conf,
            "iou": args.iou,
            "mask_threshold": args.mask_threshold,
            "imgsz": args.imgsz,
            "green_refine": bool(args.green_refine),
            "green_threshold_mode": args.green_threshold_mode,
            "green_threshold_percentile": args.green_threshold_percentile,
            "green_threshold_fixed": args.green_threshold_fixed,
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
