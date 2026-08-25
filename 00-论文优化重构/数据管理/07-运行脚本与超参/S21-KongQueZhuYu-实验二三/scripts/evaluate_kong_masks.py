#!/usr/bin/env python3
"""Evaluate KongQueZhuYu experiment masks against prepared GT masks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def read_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if shape is not None and img.shape != shape:
        img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return img > 127


def confusion(gt: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.logical_and(gt, pred).sum(dtype=np.int64))
    fp = int(np.logical_and(~gt, pred).sum(dtype=np.int64))
    fn = int(np.logical_and(gt, ~pred).sum(dtype=np.int64))
    tn = int(np.logical_and(~gt, ~pred).sum(dtype=np.int64))
    return tp, fp, fn, tn


def pixel_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    iou_fg = safe_div(tp, tp + fp + fn)
    iou_bg = safe_div(tn, tn + fp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_fg": iou_fg,
        "miou": (iou_fg + iou_bg) / 2.0,
    }


def boundary_map(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    m = mask.astype(np.uint8)
    eroded = cv2.erode(m, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return np.logical_and(m > 0, eroded == 0)


def boundary_f1(gt: np.ndarray, pred: np.ndarray, tolerance_px: int) -> float:
    gt_b = boundary_map(gt)
    pr_b = boundary_map(pred)
    gt_n = int(gt_b.sum(dtype=np.int64))
    pr_n = int(pr_b.sum(dtype=np.int64))
    if gt_n == 0 and pr_n == 0:
        return 1.0
    if gt_n == 0 or pr_n == 0:
        return 0.0
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1, 2 * tolerance_px + 1)
    )
    gt_d = cv2.dilate(gt_b.astype(np.uint8), k, iterations=1) > 0
    pr_d = cv2.dilate(pr_b.astype(np.uint8), k, iterations=1) > 0
    precision = safe_div(int(np.logical_and(pr_b, gt_d).sum(dtype=np.int64)), pr_n)
    recall = safe_div(int(np.logical_and(gt_b, pr_d).sum(dtype=np.int64)), gt_n)
    return safe_div(2 * precision * recall, precision + recall)


def hd95(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_b = boundary_map(gt)
    pr_b = boundary_map(pred)
    if not gt_b.any() and not pr_b.any():
        return 0.0
    diag = float(np.hypot(gt.shape[0], gt.shape[1]))
    if not gt_b.any() or not pr_b.any():
        return diag
    dist_to_pr = cv2.distanceTransform((~pr_b).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_gt = cv2.distanceTransform((~gt_b).astype(np.uint8), cv2.DIST_L2, 3)
    d1 = dist_to_pr[gt_b]
    d2 = dist_to_gt[pr_b]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def component_count(mask: np.ndarray, min_area_ratio: float = 0.0005) -> int:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return 0
    min_area = max(1, int(mask.size * min_area_ratio))
    return int(sum(int(stats[i, cv2.CC_STAT_AREA]) >= min_area for i in range(1, n)))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_method(
    method_name: str,
    mask_dir: Path,
    gt_dir: Path,
    image_dir: Path,
    boundary_tol: int,
) -> tuple[dict, list[dict]]:
    gt_paths = sorted(gt_dir.glob("mask_*.png"))
    frame_rows = []
    tp = fp = fn = tn = 0
    areas = []
    comps = []
    temporal_ious = []
    hd95_values = []
    bf_values = []
    missing = []
    prev_pred: np.ndarray | None = None

    for gt_path in gt_paths:
        stem = gt_path.stem.removeprefix("mask_")
        pred_path = mask_dir / f"mask_{stem}.png"
        gt = read_mask(gt_path)
        if not pred_path.exists():
            missing.append(stem)
            continue
        pred = read_mask(pred_path, gt.shape)
        c = confusion(gt, pred)
        tp += c[0]
        fp += c[1]
        fn += c[2]
        tn += c[3]
        mm = pixel_metrics(*c)
        h = hd95(gt, pred)
        b = boundary_f1(gt, pred, boundary_tol)
        area_ratio = float(pred.sum() / pred.size)
        comp = component_count(pred)
        temporal_iou = ""
        if prev_pred is not None:
            union = np.logical_or(prev_pred, pred).sum()
            temporal_iou = safe_div(int(np.logical_and(prev_pred, pred).sum()), int(union))
            temporal_ious.append(float(temporal_iou))
        prev_pred = pred
        outside_nonblack = safe_div(c[1], c[1] + c[3])
        leakage_energy = safe_div(c[1], gt.size)
        areas.append(area_ratio)
        comps.append(comp)
        hd95_values.append(h)
        bf_values.append(b)
        frame_rows.append(
            {
                "method": method_name,
                "frame": stem,
                "mask_path": str(pred_path),
                "gt_path": str(gt_path),
                "precision": mm["precision"],
                "recall": mm["recall"],
                "f1": mm["f1"],
                "miou": mm["miou"],
                "iou_fg": mm["iou_fg"],
                "hd95_px": h,
                "boundary_f1": b,
                "area_ratio": area_ratio,
                "component_count": comp,
                "temporal_iou": temporal_iou,
                "outside_nonblack_ratio": outside_nonblack,
                "leakage_energy": leakage_energy,
            }
        )

    summary_metrics = pixel_metrics(tp, fp, fn, tn)
    summary = {
        "method": method_name,
        "mask_dir": str(mask_dir),
        "gt_frames": len(gt_paths),
        "eval_frames": len(frame_rows),
        "missing_frames": ";".join(missing),
        "precision": summary_metrics["precision"],
        "recall": summary_metrics["recall"],
        "f1": summary_metrics["f1"],
        "miou": summary_metrics["miou"],
        "iou_fg": summary_metrics["iou_fg"],
        "hd95_px": float(np.mean(hd95_values)) if hd95_values else "",
        "boundary_f1": float(np.mean(bf_values)) if bf_values else "",
        "temporal_iou": float(np.mean(temporal_ious)) if temporal_ious else "",
        "area_cv": safe_div(float(np.std(areas)), float(np.mean(areas))) if areas else "",
        "component_count_mean": float(np.mean(comps)) if comps else "",
        "outside_nonblack_ratio": safe_div(fp, fp + tn),
        "leakage_energy": safe_div(fp, len(frame_rows) * gt_paths[0].stat().st_size) if False else safe_div(fp, (len(frame_rows) * read_mask(gt_paths[0]).size) if frame_rows else 0),
    }
    return summary, frame_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--method", action="append", nargs=2, metavar=("NAME", "MASK_DIR"), required=True)
    parser.add_argument("--boundary_tol", type=int, default=3)
    args = parser.parse_args()

    summary_rows = []
    frame_rows = []
    for name, mask_dir_raw in args.method:
        summary, rows = evaluate_method(name, Path(mask_dir_raw), args.gt_dir, args.image_dir, args.boundary_tol)
        summary_rows.append(summary)
        frame_rows.extend(rows)

    summary_fields = [
        "method",
        "mask_dir",
        "gt_frames",
        "eval_frames",
        "missing_frames",
        "precision",
        "recall",
        "f1",
        "miou",
        "iou_fg",
        "hd95_px",
        "boundary_f1",
        "temporal_iou",
        "area_cv",
        "component_count_mean",
        "outside_nonblack_ratio",
        "leakage_energy",
    ]
    frame_fields = [
        "method",
        "frame",
        "mask_path",
        "gt_path",
        "precision",
        "recall",
        "f1",
        "miou",
        "iou_fg",
        "hd95_px",
        "boundary_f1",
        "area_ratio",
        "component_count",
        "temporal_iou",
        "outside_nonblack_ratio",
        "leakage_energy",
    ]
    write_csv(args.output_dir / "summary_metrics.csv", summary_rows, summary_fields)
    write_csv(args.output_dir / "frame_metrics.csv", frame_rows, frame_fields)
    (args.output_dir / "summary_metrics.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Evaluated {len(summary_rows)} methods. Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
