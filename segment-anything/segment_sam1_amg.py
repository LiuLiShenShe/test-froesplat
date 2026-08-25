#!/usr/bin/env python3
"""SAM1 AMG baseline with candidate scoring and unified post-processing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


DEFAULT_INPUT_BASE_DIR = Path("/data/fj/02-FFT")
DEFAULT_OUTPUT_BASE_DIR = Path("/data/fj/results_seg_compare/03_sam1_amg")
DEFAULT_CHECKPOINT = Path("/data/fj/segment-anything/checkpoints/sam_vit_b_01ec64.pth")
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


def mask_center_score(mask_bool: np.ndarray) -> float:
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return 0.0
    h, w = mask_bool.shape
    cx = float(xs.mean()) / max(w - 1, 1)
    cy = float(ys.mean()) / max(h - 1, 1)
    dist = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5 / 0.70710678
    return float(np.clip(1.0 - dist, 0.0, 1.0))


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    if union <= 0:
        return 0.0
    return inter / union


def build_generator(args: argparse.Namespace) -> SamAutomaticMaskGenerator:
    sam = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    _ = sam.to(device=args.device)

    kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
        "output_mode": "binary_mask",
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return SamAutomaticMaskGenerator(sam, **kwargs)


def select_candidates(
    masks: list[dict],
    green_mask: np.ndarray,
    image_shape: tuple[int, int],
    min_area_ratio: float,
    max_area_ratio: float,
    select_k: int,
    target_area_ratio: float,
    duplicate_iou_threshold: float,
    w_green: float,
    w_center: float,
    w_model: float,
    w_area: float,
) -> tuple[list[np.ndarray], list[dict]]:
    h, w = image_shape
    total_pixels = h * w
    min_area = max(int(total_pixels * max(min_area_ratio, 0.0)), 1)
    max_area = max(int(total_pixels * max(max_area_ratio, 0.0)), 1)
    target_area_ratio = float(max(target_area_ratio, 1e-6))

    candidates = []
    for m in masks:
        area = int(m["area"])
        if area < min_area or area > max_area:
            continue
        mask_bool = m["segmentation"].astype(bool)
        if mask_bool.sum() == 0:
            continue

        area_ratio = float(area / total_pixels)
        green_ratio = float(green_mask[mask_bool].mean()) if mask_bool.any() else 0.0
        center_score = mask_center_score(mask_bool)
        model_score = float(
            0.5 * float(m.get("predicted_iou", 0.0)) + 0.5 * float(m.get("stability_score", 0.0))
        )
        area_score = float(
            np.clip(1.0 - abs(area_ratio - target_area_ratio) / target_area_ratio, 0.0, 1.0)
        )

        score = (
            w_green * green_ratio
            + w_center * center_score
            + w_model * model_score
            + w_area * area_score
        )

        candidates.append(
            {
                "mask": mask_bool,
                "score": float(score),
                "green_ratio": green_ratio,
                "center_score": center_score,
                "model_score": model_score,
                "area_ratio": area_ratio,
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    if not candidates:
        return [], []

    selected_masks: list[np.ndarray] = []
    selected_meta: list[dict] = []
    dup_thr = float(np.clip(duplicate_iou_threshold, 0.0, 1.0))

    for item in candidates:
        mask_bool = item["mask"]
        is_duplicate = any(mask_iou(mask_bool, s) >= dup_thr for s in selected_masks)
        if is_duplicate:
            continue
        selected_masks.append(mask_bool)
        selected_meta.append(item)
        if len(selected_masks) >= max(select_k, 1):
            break

    if not selected_masks:
        selected_masks = [c["mask"] for c in candidates[: max(select_k, 1)]]
        selected_meta = candidates[: max(select_k, 1)]

    return selected_masks, selected_meta


def segment_image(
    generator: SamAutomaticMaskGenerator,
    image_path: Path,
    candidate_min_area_ratio: float,
    candidate_max_area_ratio: float,
    candidate_select_k: int,
    candidate_target_area_ratio: float,
    candidate_duplicate_iou_threshold: float,
    score_w_green: float,
    score_w_center: float,
    score_w_model: float,
    score_w_area: float,
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
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        empty = np.zeros((1, 1), dtype=np.uint8)
        return empty, {"error": "cannot_read_image"}

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    masks = generator.generate(image_rgb)

    exg_u8 = compute_exg_u8(image_bgr)
    green_mask, green_threshold_value = green_mask_from_exg(
        exg_u8,
        mode=green_threshold_mode,
        percentile=green_threshold_percentile,
        fixed_threshold=green_threshold_fixed,
    )

    selected_masks, selected_meta = select_candidates(
        masks=masks,
        green_mask=green_mask,
        image_shape=(h, w),
        min_area_ratio=candidate_min_area_ratio,
        max_area_ratio=candidate_max_area_ratio,
        select_k=max(candidate_select_k, 1),
        target_area_ratio=max(candidate_target_area_ratio, 1e-6),
        duplicate_iou_threshold=float(np.clip(candidate_duplicate_iou_threshold, 0.0, 1.0)),
        w_green=score_w_green,
        w_center=score_w_center,
        w_model=score_w_model,
        w_area=score_w_area,
    )

    merged = np.zeros((h, w), dtype=np.uint8)
    for m in selected_masks:
        merged |= m.astype(np.uint8)

    raw_cov_before_green = float(np.count_nonzero(merged) / merged.size * 100.0)
    green_refine_fallback = False
    if green_refine:
        refined = ((merged > 0) & green_mask).astype(np.uint8)
        refined_cov = float(np.count_nonzero(refined) / refined.size * 100.0)
        min_keep_cov = max(1.0, raw_cov_before_green * 0.2)
        if refined_cov < min_keep_cov and raw_cov_before_green > 0.0:
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

    top_score = float(selected_meta[0]["score"]) if selected_meta else 0.0
    info = {
        "masks_total": len(masks),
        "masks_selected": len(selected_masks),
        "top_candidate_score": round(top_score, 4),
        "raw_coverage_before_green_pct": round(raw_cov_before_green, 2),
        "raw_coverage_after_green_pct": round(raw_cov_after_green, 2),
        "green_refine_fallback": green_refine_fallback,
        "green_threshold_mode": green_threshold_mode if green_refine else "disabled",
        "green_threshold_value": round(float(green_threshold_value), 3),
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
    generator: SamAutomaticMaskGenerator,
    folder_name: str,
    input_base_dir: Path,
    output_base_dir: Path,
    extensions: set[str],
    force: bool,
    candidate_min_area_ratio: float,
    candidate_max_area_ratio: float,
    candidate_select_k: int,
    candidate_target_area_ratio: float,
    candidate_duplicate_iou_threshold: float,
    score_w_green: float,
    score_w_center: float,
    score_w_model: float,
    score_w_area: float,
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
            generator=generator,
            image_path=image_path,
            candidate_min_area_ratio=candidate_min_area_ratio,
            candidate_max_area_ratio=candidate_max_area_ratio,
            candidate_select_k=candidate_select_k,
            candidate_target_area_ratio=candidate_target_area_ratio,
            candidate_duplicate_iou_threshold=candidate_duplicate_iou_threshold,
            score_w_green=score_w_green,
            score_w_center=score_w_center,
            score_w_model=score_w_model,
            score_w_area=score_w_area,
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

        if idx == 1 or idx == len(images) or idx % 10 == 0:
            print(
                f"       [{idx:>4}/{len(images)}] "
                f"masks={record['masks_selected']}/{record['masks_total']} "
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
    parser = argparse.ArgumentParser(description="SAM1 AMG baseline")
    parser.add_argument("--input-base-dir", type=Path, default=DEFAULT_INPUT_BASE_DIR)
    parser.add_argument("--output-base-dir", type=Path, default=DEFAULT_OUTPUT_BASE_DIR)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--extensions", type=str, default=".jpg,.jpeg,.png,.bmp")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Process first N images only")

    parser.add_argument("--target-definition", type=str, default=DEFAULT_TARGET_DEFINITION)
    parser.add_argument("--model-type", type=str, default="vit_b")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--stability-score-offset", type=float, default=1.0)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-n-layers", type=int, default=1)
    parser.add_argument("--crop-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-overlap-ratio", type=float, default=0.3413333333)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=1)
    parser.add_argument("--min-mask-region-area", type=int, default=0)

    parser.add_argument("--candidate-min-area-ratio", type=float, default=0.002)
    parser.add_argument("--candidate-max-area-ratio", type=float, default=0.80)
    parser.add_argument("--candidate-select-k", type=int, default=12)
    parser.add_argument("--candidate-target-area-ratio", type=float, default=0.08)
    parser.add_argument("--candidate-duplicate-iou-threshold", type=float, default=0.95)
    parser.add_argument("--score-w-green", type=float, default=0.55)
    parser.add_argument("--score-w-center", type=float, default=0.25)
    parser.add_argument("--score-w-model", type=float, default=0.15)
    parser.add_argument("--score-w-area", type=float, default=0.05)

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
    parser.add_argument("--pp-min-area-ratio", type=float, default=0.0005)
    args = parser.parse_args()

    extensions = parse_extensions(args.extensions)
    validate_dirs([args.input_base_dir, args.checkpoint])
    args.output_base_dir.mkdir(parents=True, exist_ok=True)

    folders = [args.folder] if args.folder else discover_folders(args.input_base_dir, extensions)
    if not folders:
        print("No folders with images were found.")
        return

    print(f"SAM1 AMG | folders={len(folders)} | target={args.target_definition}")
    print(f"Input:      {args.input_base_dir}")
    print(f"Output:     {args.output_base_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {args.device}")
    print("-" * 72)

    generator = build_generator(args)
    started = time.time()
    summary = []

    for index, folder in enumerate(folders, start=1):
        print(f"[{index}/{len(folders)}] {folder}")
        result = process_folder(
            generator=generator,
            folder_name=folder,
            input_base_dir=args.input_base_dir,
            output_base_dir=args.output_base_dir,
            extensions=extensions,
            force=args.force,
            candidate_min_area_ratio=max(args.candidate_min_area_ratio, 0.0),
            candidate_max_area_ratio=max(args.candidate_max_area_ratio, 0.0),
            candidate_select_k=max(args.candidate_select_k, 1),
            candidate_target_area_ratio=max(args.candidate_target_area_ratio, 1e-6),
            candidate_duplicate_iou_threshold=float(
                np.clip(args.candidate_duplicate_iou_threshold, 0.0, 1.0)
            ),
            score_w_green=float(args.score_w_green),
            score_w_center=float(args.score_w_center),
            score_w_model=float(args.score_w_model),
            score_w_area=float(args.score_w_area),
            green_refine=bool(args.green_refine),
            green_threshold_mode=args.green_threshold_mode,
            green_threshold_percentile=float(args.green_threshold_percentile),
            green_threshold_fixed=float(args.green_threshold_fixed),
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
        "method": "SAM1_AMG",
        "target_definition": args.target_definition,
        "total_folders": len(folders),
        "total_time_sec": round(total, 2),
        "parameters": {
            "model_type": args.model_type,
            "device": args.device,
            "points_per_side": args.points_per_side,
            "pred_iou_thresh": args.pred_iou_thresh,
            "stability_score_thresh": args.stability_score_thresh,
            "candidate_min_area_ratio": max(args.candidate_min_area_ratio, 0.0),
            "candidate_max_area_ratio": max(args.candidate_max_area_ratio, 0.0),
            "candidate_select_k": max(args.candidate_select_k, 1),
            "candidate_target_area_ratio": max(args.candidate_target_area_ratio, 1e-6),
            "candidate_duplicate_iou_threshold": float(
                np.clip(args.candidate_duplicate_iou_threshold, 0.0, 1.0)
            ),
            "score_w_green": float(args.score_w_green),
            "score_w_center": float(args.score_w_center),
            "score_w_model": float(args.score_w_model),
            "score_w_area": float(args.score_w_area),
            "green_refine": bool(args.green_refine),
            "green_threshold_mode": args.green_threshold_mode,
            "green_threshold_percentile": float(args.green_threshold_percentile),
            "green_threshold_fixed": float(args.green_threshold_fixed),
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
