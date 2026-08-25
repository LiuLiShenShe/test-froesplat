#!/usr/bin/env python3
"""Compute extended by-target segmentation metrics for SAM vs SEEM."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

METHODS = ("SAM", "SEEM")
TARGETS = ("plant", "block")


@dataclass
class PixelCounters:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    n_frames: int = 0

    def add(self, tp: int, fp: int, fn: int, tn: int) -> None:
        self.tp += int(tp)
        self.fp += int(fp)
        self.fn += int(fn)
        self.tn += int(tn)
        self.n_frames += 1


@dataclass
class FramePresenceCounters:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    n_frames: int = 0

    def add(self, gt_present: bool, pred_present: bool) -> None:
        self.n_frames += 1
        if gt_present and pred_present:
            self.tp += 1
        elif (not gt_present) and pred_present:
            self.fp += 1
        elif gt_present and (not pred_present):
            self.fn += 1
        else:
            self.tn += 1


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def metrics_from_confusion(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    fpr = safe_div(fp, fp + tn)
    npv = safe_div(tn, tn + fn)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    iou_fg = safe_div(tp, tp + fp + fn)
    iou_bg = safe_div(tn, tn + fp + fn)
    miou_binary = (iou_fg + iou_bg) / 2.0
    balanced_acc = (recall + specificity) / 2.0
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "npv": npv,
        "f1": f1,
        "iou_fg": iou_fg,
        "miou_binary": miou_binary,
        "balanced_acc": balanced_acc,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(areas)) + 1
    return labels == largest_idx


def shape_to_mask(shape: dict, h: int, w: int) -> np.ndarray:
    points = shape.get("points", [])
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points) < 3:
        return mask > 0
    pts = np.asarray(points, dtype=np.float32)
    pts = np.round(pts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [pts], 255)
    return mask > 0


def load_gt_masks(json_path: Path, block_area_ratio_thresh: float) -> tuple[np.ndarray, np.ndarray, str]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    h = int(data["imageHeight"])
    w = int(data["imageWidth"])
    total = float(h * w)

    plant = np.zeros((h, w), dtype=bool)
    block = np.zeros((h, w), dtype=bool)
    largest = None
    largest_area = -1

    for shape in data.get("shapes", []):
        sm = shape_to_mask(shape, h, w)
        area = int(sm.sum(dtype=np.int64))
        if area <= 0:
            continue
        if area / total < block_area_ratio_thresh:
            block |= sm
        else:
            plant |= sm
        if area > largest_area:
            largest_area = area
            largest = sm

    if not plant.any() and largest is not None:
        plant = largest.copy()
        block = np.logical_and(block, ~plant)

    image_rel = data.get("imagePath", json_path.with_suffix(".jpg").name)
    return plant, block, image_rel


def read_combined_pred_mask(
    method: str,
    plant_name: str,
    frame_stem: str,
    sam_root: Path,
    seem_root: Path,
    target_shape: tuple[int, int],
    seem_threshold: int,
) -> np.ndarray | None:
    if method == "SAM":
        pred_path = sam_root / plant_name / f"mask_{frame_stem}.png"
    else:
        pred_path = seem_root / plant_name / f"{frame_stem}.jpg"

    if not pred_path.exists():
        return None
    img = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.ndim == 3:
        if pred_path.suffix.lower() in {".jpg", ".jpeg"}:
            gray = np.max(img, axis=2)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if gray.shape != target_shape:
        gray = cv2.resize(gray, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    if pred_path.suffix.lower() in {".jpg", ".jpeg"}:
        return gray > seem_threshold
    return gray > 127


def make_blue_mask(image_bgr: np.ndarray, delta_r: int, delta_g: int, b_min: int) -> np.ndarray:
    b, g, r = cv2.split(image_bgr)
    b16 = b.astype(np.int16)
    g16 = g.astype(np.int16)
    r16 = r.astype(np.int16)
    return (b16 >= g16 + delta_g) & (b16 >= r16 + delta_r) & (b >= b_min)


def split_pred_targets(
    pred_combined: np.ndarray,
    image_bgr: np.ndarray,
    pred_block_max_area_ratio: float,
    pred_block_min_pixels: int,
    pred_block_min_blue_ratio: float,
    blue_delta_r: int,
    blue_delta_g: int,
    blue_b_min: int,
    keep_largest_plant: bool,
) -> tuple[np.ndarray, np.ndarray]:
    blue = make_blue_mask(image_bgr, blue_delta_r, blue_delta_g, blue_b_min)
    total_pixels = float(pred_combined.size)

    block = np.zeros_like(pred_combined, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(pred_combined.astype(np.uint8), 8)
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < pred_block_min_pixels:
            continue
        if area / total_pixels > pred_block_max_area_ratio:
            continue
        comp = labels == idx
        blue_ratio = safe_div(int(np.logical_and(comp, blue).sum(dtype=np.int64)), area)
        if blue_ratio >= pred_block_min_blue_ratio:
            block |= comp

    plant = np.logical_and(pred_combined, ~block)
    if keep_largest_plant:
        plant = keep_largest_component(plant)
    return plant, block


def boundary_map(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    if not m.any():
        return m > 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(m, kernel, iterations=1)
    b = np.logical_and(m > 0, eroded == 0)
    return b


def boundary_metrics(gt: np.ndarray, pred: np.ndarray, tolerance_px: int) -> tuple[float, float, float, float]:
    gt_b = boundary_map(gt)
    pr_b = boundary_map(pred)

    gt_n = int(gt_b.sum(dtype=np.int64))
    pr_n = int(pr_b.sum(dtype=np.int64))

    if gt_n == 0 and pr_n == 0:
        return 1.0, 1.0, 1.0, 0.0
    if gt_n == 0 or pr_n == 0:
        diag = float(np.hypot(gt.shape[0], gt.shape[1]))
        return 0.0, 0.0, 0.0, diag

    if tolerance_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1, 2 * tolerance_px + 1))
        gt_d = cv2.dilate(gt_b.astype(np.uint8), k, iterations=1) > 0
        pr_d = cv2.dilate(pr_b.astype(np.uint8), k, iterations=1) > 0
    else:
        gt_d = gt_b
        pr_d = pr_b

    b_precision = safe_div(int(np.logical_and(pr_b, gt_d).sum(dtype=np.int64)), pr_n)
    b_recall = safe_div(int(np.logical_and(gt_b, pr_d).sum(dtype=np.int64)), gt_n)
    b_f1 = safe_div(2.0 * b_precision * b_recall, b_precision + b_recall)

    inv_gt = (~gt_b).astype(np.uint8)
    inv_pr = (~pr_b).astype(np.uint8)
    dist_to_gt = cv2.distanceTransform(inv_gt, cv2.DIST_L2, 3)
    dist_to_pr = cv2.distanceTransform(inv_pr, cv2.DIST_L2, 3)
    d_pr_gt = dist_to_gt[pr_b]
    d_gt_pr = dist_to_pr[gt_b]
    hd95 = float(max(np.percentile(d_pr_gt, 95), np.percentile(d_gt_pr, 95)))

    return b_precision, b_recall, b_f1, hd95


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def component_stats(mask: np.ndarray) -> tuple[int, float]:
    if not mask.any():
        return 0, 0.0
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return 0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    total = float(areas.sum())
    n_comp = int(len(areas))
    largest_ratio = float(areas.max() / total) if total > 0 else 0.0
    return n_comp, largest_ratio


def confusion(gt: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.logical_and(gt, pred).sum(dtype=np.int64))
    fp = int(np.logical_and(~gt, pred).sum(dtype=np.int64))
    fn = int(np.logical_and(gt, ~pred).sum(dtype=np.int64))
    tn = int(np.logical_and(~gt, ~pred).sum(dtype=np.int64))
    return tp, fp, fn, tn


def mean_without_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> tuple[float, float, float]:
    vals = values.astype(np.float64)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(vals.mean())
    if vals.size == 1:
        return mean, mean, mean
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boot_means = vals[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    return mean, lo, hi


def paired_permutation_test(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_perm: int = 20000
) -> tuple[float, float]:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    valid = ~(np.isnan(a) | np.isnan(b))
    a = a[valid]
    b = b[valid]
    if a.size == 0:
        return float("nan"), float("nan")
    diff = a - b
    obs = float(diff.mean())
    if np.allclose(diff, 0):
        return obs, 1.0
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diff.size))
    perm_means = (signs * diff[None, :]).mean(axis=1)
    p = float((np.sum(np.abs(perm_means) >= abs(obs)) + 1) / (n_perm + 1))
    return obs, p


def main() -> None:
    parser = argparse.ArgumentParser(description="Extended by-target metrics for current SAM/SEEM outputs.")
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, default=Path("/data/fj/03-GT"))
    parser.add_argument("--sam-root", type=Path, default=Path("/data/fj/03-SAM"))
    parser.add_argument("--seem-root", type=Path, default=Path("/data/fj/03-SEEM"))
    parser.add_argument("--seem-threshold", type=int, default=10)
    parser.add_argument("--gt-block-area-ratio-thresh", type=float, default=0.03)
    parser.add_argument("--pred-block-max-area-ratio", type=float, default=0.03)
    parser.add_argument("--pred-block-min-pixels", type=int, default=200)
    parser.add_argument("--pred-block-min-blue-ratio", type=float, default=0.20)
    parser.add_argument("--blue-delta-r", type=int, default=20)
    parser.add_argument("--blue-delta-g", type=int, default=20)
    parser.add_argument("--blue-b-min", type=int, default=50)
    parser.add_argument("--boundary-tolerance-px", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--perm-samples", type=int, default=20000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--no-keep-largest-plant", action="store_true")
    args = parser.parse_args()

    keep_largest_plant = not args.no_keep_largest_plant
    rng = np.random.default_rng(args.random_seed)

    run_name = f"extended_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = args.eval_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_rows: list[dict[str, object]] = []
    block_detect_rows: list[dict[str, object]] = []

    per_plant_pixel = {p: {m: {t: PixelCounters() for t in TARGETS} for m in METHODS} for p in []}
    per_plant_block_presence = {p: {m: FramePresenceCounters() for m in METHODS} for p in []}

    overall_pixel = {m: {t: PixelCounters() for t in TARGETS} for m in METHODS}
    overall_block_presence = {m: FramePresenceCounters() for m in METHODS}

    plants = sorted(d.name for d in args.gt_root.iterdir() if d.is_dir() and any(d.glob("*.json")))
    per_plant_pixel = {p: {m: {t: PixelCounters() for t in TARGETS} for m in METHODS} for p in plants}
    per_plant_block_presence = {p: {m: FramePresenceCounters() for m in METHODS} for p in plants}

    common_frame_counts = []

    for plant in plants:
        gt_dir = args.gt_root / plant
        gt_jsons = sorted(gt_dir.glob("*.json"))

        missing = {m: 0 for m in METHODS}
        common = 0
        common_block_positive = 0

        for json_path in gt_jsons:
            frame = json_path.stem
            gt_plant, gt_block, image_rel = load_gt_masks(json_path, args.gt_block_area_ratio_thresh)
            gt_targets = {"plant": gt_plant, "block": gt_block}

            image = cv2.imread(str(gt_dir / image_rel), cv2.IMREAD_COLOR)
            if image is None:
                image = np.zeros((gt_plant.shape[0], gt_plant.shape[1], 3), dtype=np.uint8)
            elif image.shape[:2] != gt_plant.shape:
                image = cv2.resize(image, (gt_plant.shape[1], gt_plant.shape[0]), interpolation=cv2.INTER_LINEAR)

            pred_targets_by_method = {}
            for method in METHODS:
                pred_comb = read_combined_pred_mask(
                    method=method,
                    plant_name=plant,
                    frame_stem=frame,
                    sam_root=args.sam_root,
                    seem_root=args.seem_root,
                    target_shape=gt_plant.shape,
                    seem_threshold=args.seem_threshold,
                )
                if pred_comb is None:
                    pred_targets_by_method[method] = None
                    missing[method] += 1
                    continue
                pr_plant, pr_block = split_pred_targets(
                    pred_combined=pred_comb,
                    image_bgr=image,
                    pred_block_max_area_ratio=args.pred_block_max_area_ratio,
                    pred_block_min_pixels=args.pred_block_min_pixels,
                    pred_block_min_blue_ratio=args.pred_block_min_blue_ratio,
                    blue_delta_r=args.blue_delta_r,
                    blue_delta_g=args.blue_delta_g,
                    blue_b_min=args.blue_b_min,
                    keep_largest_plant=keep_largest_plant,
                )
                pred_targets_by_method[method] = {"plant": pr_plant, "block": pr_block}

            if any(pred_targets_by_method[m] is None for m in METHODS):
                continue

            common += 1
            gt_block_present = bool(gt_block.any())
            if gt_block_present:
                common_block_positive += 1

            for method in METHODS:
                pred_block_present = bool(pred_targets_by_method[method]["block"].any())
                per_plant_block_presence[plant][method].add(gt_block_present, pred_block_present)
                overall_block_presence[method].add(gt_block_present, pred_block_present)
                block_detect_rows.append(
                    {
                        "plant": plant,
                        "frame": frame,
                        "method": method,
                        "gt_block_present": int(gt_block_present),
                        "pred_block_present": int(pred_block_present),
                    }
                )

                for target in TARGETS:
                    gt = gt_targets[target]
                    pred = pred_targets_by_method[method][target]
                    tp, fp, fn, tn = confusion(gt, pred)
                    mm = metrics_from_confusion(tp, fp, fn, tn)
                    b_prec, b_rec, b_f1, hd95 = boundary_metrics(gt, pred, args.boundary_tolerance_px)
                    n_comp, largest_ratio = component_stats(pred)
                    gt_area = int(gt.sum(dtype=np.int64))
                    pred_area = int(pred.sum(dtype=np.int64))

                    area_rel_err = float("nan")
                    centroid_dist_px = float("nan")
                    centroid_dist_norm = float("nan")
                    if target == "block" and gt_area > 0:
                        area_rel_err = abs(pred_area - gt_area) / float(gt_area)
                        c_gt = centroid(gt)
                        c_pr = centroid(pred)
                        if c_gt is not None and c_pr is not None:
                            d = float(np.hypot(c_gt[0] - c_pr[0], c_gt[1] - c_pr[1]))
                            diag = float(np.hypot(gt.shape[0], gt.shape[1]))
                            centroid_dist_px = d
                            centroid_dist_norm = safe_div(d, diag)

                    row = {
                        "plant": plant,
                        "frame": frame,
                        "method": method,
                        "target": target,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn,
                        "precision": mm["precision"],
                        "recall": mm["recall"],
                        "specificity": mm["specificity"],
                        "fpr": mm["fpr"],
                        "npv": mm["npv"],
                        "balanced_acc": mm["balanced_acc"],
                        "f1": mm["f1"],
                        "iou_fg": mm["iou_fg"],
                        "miou_binary": mm["miou_binary"],
                        "boundary_precision": b_prec,
                        "boundary_recall": b_rec,
                        "boundary_f1": b_f1,
                        "hd95_px": hd95,
                        "gt_area": gt_area,
                        "pred_area": pred_area,
                        "pred_components": n_comp,
                        "pred_largest_component_ratio": largest_ratio,
                        "area_rel_error": area_rel_err,
                        "centroid_dist_px": centroid_dist_px,
                        "centroid_dist_norm": centroid_dist_norm,
                        "gt_positive": int(gt_area > 0),
                    }
                    frame_rows.append(row)

                    per_plant_pixel[plant][method][target].add(tp, fp, fn, tn)
                    overall_pixel[method][target].add(tp, fp, fn, tn)

        common_frame_counts.append(
            {
                "plant": plant,
                "gt_frames": len(gt_jsons),
                "common_frames": common,
                "common_block_positive_frames": common_block_positive,
                "missing_sam_frames": missing["SAM"],
                "missing_seem_frames": missing["SEEM"],
            }
        )

    frame_fields = [
        "plant",
        "frame",
        "method",
        "target",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "specificity",
        "fpr",
        "npv",
        "balanced_acc",
        "f1",
        "iou_fg",
        "miou_binary",
        "boundary_precision",
        "boundary_recall",
        "boundary_f1",
        "hd95_px",
        "gt_area",
        "pred_area",
        "pred_components",
        "pred_largest_component_ratio",
        "area_rel_error",
        "centroid_dist_px",
        "centroid_dist_norm",
        "gt_positive",
    ]
    write_csv(out_dir / "frame_metrics_extended_common.csv", frame_rows, frame_fields)
    write_csv(
        out_dir / "frame_block_detection_common.csv",
        block_detect_rows,
        ["plant", "frame", "method", "gt_block_present", "pred_block_present"],
    )
    write_csv(
        out_dir / "common_frame_counts.csv",
        common_frame_counts,
        ["plant", "gt_frames", "common_frames", "common_block_positive_frames", "missing_sam_frames", "missing_seem_frames"],
    )

    # Per-plant summary
    per_plant_rows = []
    for plant in plants:
        plant_rows = [r for r in frame_rows if r["plant"] == plant]
        for method in METHODS:
            for target in TARGETS:
                rows = [r for r in plant_rows if r["method"] == method and r["target"] == target]
                pix = per_plant_pixel[plant][method][target]
                mm = metrics_from_confusion(pix.tp, pix.fp, pix.fn, pix.tn)
                per_plant_rows.append(
                    {
                        "plant": plant,
                        "method": method,
                        "target": target,
                        "eval_frames": len(rows),
                        "precision": mm["precision"],
                        "recall": mm["recall"],
                        "specificity": mm["specificity"],
                        "fpr": mm["fpr"],
                        "npv": mm["npv"],
                        "balanced_acc": mm["balanced_acc"],
                        "f1": mm["f1"],
                        "iou_fg": mm["iou_fg"],
                        "miou_binary": mm["miou_binary"],
                        "mean_boundary_f1": mean_without_nan([float(r["boundary_f1"]) for r in rows]),
                        "mean_hd95_px": mean_without_nan([float(r["hd95_px"]) for r in rows]),
                        "mean_pred_components": mean_without_nan([float(r["pred_components"]) for r in rows]),
                        "mean_largest_component_ratio": mean_without_nan(
                            [float(r["pred_largest_component_ratio"]) for r in rows]
                        ),
                        "mean_area_rel_error": mean_without_nan([float(r["area_rel_error"]) for r in rows]),
                        "mean_centroid_dist_norm": mean_without_nan([float(r["centroid_dist_norm"]) for r in rows]),
                    }
                )
    per_plant_fields = [
        "plant",
        "method",
        "target",
        "eval_frames",
        "precision",
        "recall",
        "specificity",
        "fpr",
        "npv",
        "balanced_acc",
        "f1",
        "iou_fg",
        "miou_binary",
        "mean_boundary_f1",
        "mean_hd95_px",
        "mean_pred_components",
        "mean_largest_component_ratio",
        "mean_area_rel_error",
        "mean_centroid_dist_norm",
    ]
    write_csv(out_dir / "summary_per_plant_extended_common.csv", per_plant_rows, per_plant_fields)

    # Overall summary + bootstrap CI
    overall_rows = []
    ci_rows = []
    for method in METHODS:
        for target in TARGETS:
            rows = [r for r in frame_rows if r["method"] == method and r["target"] == target]
            pix = overall_pixel[method][target]
            mm = metrics_from_confusion(pix.tp, pix.fp, pix.fn, pix.tn)
            overall_rows.append(
                {
                    "method": method,
                    "target": target,
                    "eval_frames": len(rows),
                    "precision": mm["precision"],
                    "recall": mm["recall"],
                    "specificity": mm["specificity"],
                    "fpr": mm["fpr"],
                    "npv": mm["npv"],
                    "balanced_acc": mm["balanced_acc"],
                    "f1": mm["f1"],
                    "iou_fg": mm["iou_fg"],
                    "miou_binary": mm["miou_binary"],
                    "mean_boundary_f1": mean_without_nan([float(r["boundary_f1"]) for r in rows]),
                    "mean_hd95_px": mean_without_nan([float(r["hd95_px"]) for r in rows]),
                    "mean_pred_components": mean_without_nan([float(r["pred_components"]) for r in rows]),
                    "mean_largest_component_ratio": mean_without_nan(
                        [float(r["pred_largest_component_ratio"]) for r in rows]
                    ),
                    "mean_area_rel_error": mean_without_nan([float(r["area_rel_error"]) for r in rows]),
                    "mean_centroid_dist_norm": mean_without_nan([float(r["centroid_dist_norm"]) for r in rows]),
                }
            )

            for metric in ["f1", "iou_fg", "boundary_f1", "hd95_px", "specificity", "fpr"]:
                vals = np.asarray([float(r[metric]) for r in rows], dtype=np.float64)
                mean, lo, hi = bootstrap_ci(vals, rng=rng, n_boot=args.bootstrap_samples)
                ci_rows.append(
                    {
                        "method": method,
                        "target": target,
                        "metric": metric,
                        "mean": mean,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "n_frames": int(np.sum(~np.isnan(vals))),
                    }
                )

    overall_fields = [
        "method",
        "target",
        "eval_frames",
        "precision",
        "recall",
        "specificity",
        "fpr",
        "npv",
        "balanced_acc",
        "f1",
        "iou_fg",
        "miou_binary",
        "mean_boundary_f1",
        "mean_hd95_px",
        "mean_pred_components",
        "mean_largest_component_ratio",
        "mean_area_rel_error",
        "mean_centroid_dist_norm",
    ]
    write_csv(out_dir / "summary_overall_extended_common.csv", overall_rows, overall_fields)
    write_csv(
        out_dir / "summary_overall_bootstrap_ci_common.csv",
        ci_rows,
        ["method", "target", "metric", "mean", "ci95_low", "ci95_high", "n_frames"],
    )

    # Block detection presence metrics
    block_presence_rows = []
    for scope, plants_scope in [("overall", plants), *[(p, [p]) for p in plants]]:
        for method in METHODS:
            if scope == "overall":
                c = overall_block_presence[method]
            else:
                c = per_plant_block_presence[scope][method]
            mm = metrics_from_confusion(c.tp, c.fp, c.fn, c.tn)
            block_presence_rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "frames": c.n_frames,
                    "tp": c.tp,
                    "fp": c.fp,
                    "fn": c.fn,
                    "tn": c.tn,
                    "detection_precision": mm["precision"],
                    "detection_recall": mm["recall"],
                    "detection_f1": mm["f1"],
                    "false_alarm_rate": mm["fpr"],
                    "specificity": mm["specificity"],
                }
            )
    write_csv(
        out_dir / "block_presence_detection_summary.csv",
        block_presence_rows,
        [
            "scope",
            "method",
            "frames",
            "tp",
            "fp",
            "fn",
            "tn",
            "detection_precision",
            "detection_recall",
            "detection_f1",
            "false_alarm_rate",
            "specificity",
        ],
    )

    # Paired statistical comparison SAM vs SEEM on common frames
    stat_rows = []
    for target in TARGETS:
        sam_rows = sorted(
            [r for r in frame_rows if r["method"] == "SAM" and r["target"] == target],
            key=lambda x: (x["plant"], x["frame"]),
        )
        seem_rows = sorted(
            [r for r in frame_rows if r["method"] == "SEEM" and r["target"] == target],
            key=lambda x: (x["plant"], x["frame"]),
        )
        if len(sam_rows) != len(seem_rows):
            continue
        for metric in ["f1", "iou_fg", "boundary_f1", "hd95_px"]:
            a = np.asarray([float(r[metric]) for r in sam_rows], dtype=np.float64)
            b = np.asarray([float(r[metric]) for r in seem_rows], dtype=np.float64)
            diff_mean, p_val = paired_permutation_test(a, b, rng=rng, n_perm=args.perm_samples)
            # paired bootstrap CI for mean diff
            valid = ~(np.isnan(a) | np.isnan(b))
            d = (a - b)[valid]
            if d.size == 0:
                ci_lo = float("nan")
                ci_hi = float("nan")
            elif d.size == 1:
                ci_lo = float(d[0])
                ci_hi = float(d[0])
            else:
                idx = rng.integers(0, d.size, size=(args.bootstrap_samples, d.size))
                boot = d[idx].mean(axis=1)
                ci_lo = float(np.percentile(boot, 2.5))
                ci_hi = float(np.percentile(boot, 97.5))

            stat_rows.append(
                {
                    "target": target,
                    "metric": metric,
                    "n_pairs": int(d.size),
                    "mean_diff_sam_minus_seem": diff_mean,
                    "ci95_low": ci_lo,
                    "ci95_high": ci_hi,
                    "perm_p_value": p_val,
                }
            )
    write_csv(
        out_dir / "sam_vs_seem_paired_tests_common.csv",
        stat_rows,
        ["target", "metric", "n_pairs", "mean_diff_sam_minus_seem", "ci95_low", "ci95_high", "perm_p_value"],
    )

    # Markdown report
    # pull key rows
    key = {(r["method"], r["target"]): r for r in overall_rows}
    block_det_key = {(r["scope"], r["method"]): r for r in block_presence_rows}
    report_lines = [
        "# Extended Metrics Report (Common Frames)",
        "",
        f"- Source eval dir: `{args.eval_dir}`",
        f"- Output dir: `{out_dir}`",
        f"- Boundary tolerance: `{args.boundary_tolerance_px}` px",
        f"- Bootstrap samples: `{args.bootstrap_samples}`",
        f"- Permutation samples: `{args.perm_samples}`",
        "",
        "## Overall By Target",
        "",
        "| Method | Target | F1 | IoU_fg | Precision | Recall | Specificity | FPR | Boundary F1 | HD95(px) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        for target in TARGETS:
            r = key[(method, target)]
            report_lines.append(
                f"| {method} | {target} | {r['f1']:.6f} | {r['iou_fg']:.6f} | {r['precision']:.6f} | "
                f"{r['recall']:.6f} | {r['specificity']:.6f} | {r['fpr']:.6f} | {r['mean_boundary_f1']:.6f} | "
                f"{r['mean_hd95_px']:.4f} |"
            )

    report_lines += [
        "",
        "## Block Presence Detection (Frame-Level)",
        "",
        "| Method | Precision | Recall | F1 | False Alarm Rate | Specificity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        r = block_det_key[("overall", method)]
        report_lines.append(
            f"| {method} | {r['detection_precision']:.6f} | {r['detection_recall']:.6f} | "
            f"{r['detection_f1']:.6f} | {r['false_alarm_rate']:.6f} | {r['specificity']:.6f} |"
        )

    report_lines += [
        "",
        "## Paired Tests (SAM - SEEM, Common Frames)",
        "",
        "| Target | Metric | N | Mean Diff | 95% CI | Permutation p |",
        "|---|---|---:|---:|---|---:|",
    ]
    for r in stat_rows:
        report_lines.append(
            f"| {r['target']} | {r['metric']} | {r['n_pairs']} | {r['mean_diff_sam_minus_seem']:.6f} | "
            f"[{r['ci95_low']:.6f}, {r['ci95_high']:.6f}] | {r['perm_p_value']:.6g} |"
        )

    report_lines += [
        "",
        "## Metric Meanings (指标释义)",
        "",
        "- `precision`: 预测为前景的像素里，真实为前景的比例（越高越少误检）。",
        "- `recall`: 真实前景像素里，被预测到的比例（越高越少漏检）。",
        "- `specificity`: 真实背景像素里，被正确预测为背景的比例。",
        "- `fpr`: 假阳性率，等于 `1 - specificity`（越低越好）。",
        "- `npv`: 预测为背景的像素里，真实为背景的比例。",
        "- `balanced_acc`: `(recall + specificity) / 2`，前景/背景均衡准确度。",
        "- `f1`: 二值分割 Dice/F1（两者等价）。",
        "- `iou_fg`: 前景 IoU。",
        "- `miou_binary`: 二类平均 IoU（前景 IoU 与背景 IoU 的平均）。",
        "- `boundary_precision / boundary_recall / boundary_f1`: 边界级精度/召回/F1（在容差像素内匹配）。",
        "- `hd95_px`: 边界 Hausdorff 95% 距离（像素单位，越低越好）。",
        "- `pred_components`: 预测掩码连通域个数（越多通常越碎）。",
        "- `pred_largest_component_ratio`: 最大连通域面积占预测总前景面积比例（越接近 1 越集中）。",
        "- `area_rel_error`: 目标面积相对误差 `|A_pred - A_gt| / A_gt`（主要用于 block）。",
        "- `centroid_dist_px / centroid_dist_norm`: 预测与 GT 质心距离（像素/归一化对角线）。",
        "- `detection_*` (`block_presence_detection_summary.csv`): 帧级“是否检测到方块”的检测指标。",
        "- `mean_diff_sam_minus_seem`: 配对统计中 SAM 减 SEEM 的平均差；`perm_p_value` 为置换检验 p 值。",
        "",
        "## File Locations (结果文件位置)",
        "",
        f"- Frame-level extended metrics: `{out_dir / 'frame_metrics_extended_common.csv'}`",
        f"- Frame-level block presence flags: `{out_dir / 'frame_block_detection_common.csv'}`",
        f"- Per-plant summary: `{out_dir / 'summary_per_plant_extended_common.csv'}`",
        f"- Overall summary: `{out_dir / 'summary_overall_extended_common.csv'}`",
        f"- Overall bootstrap 95% CI: `{out_dir / 'summary_overall_bootstrap_ci_common.csv'}`",
        f"- Block presence detection summary: `{out_dir / 'block_presence_detection_summary.csv'}`",
        f"- SAM vs SEEM paired tests: `{out_dir / 'sam_vs_seem_paired_tests_common.csv'}`",
        f"- Common-frame counts and missing frames: `{out_dir / 'common_frame_counts.csv'}`",
        "",
        "## Notes",
        "",
        "- `Dice` equals `F1` for binary segmentation, so only `F1` is reported.",
        "- HD95 is computed on mask boundaries; lower is better.",
        "- For block target, area and centroid errors are reported in CSV summaries.",
    ]
    (out_dir / "report_extended.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[DONE] Extended metrics generated: {out_dir}")


if __name__ == "__main__":
    main()
