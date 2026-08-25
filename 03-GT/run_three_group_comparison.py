#!/usr/bin/env python3
"""Run compact comparison with only three requested metric groups."""

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


@dataclass
class PixelCounter:
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
class BlockDetectCounter:
    gt_positive_frames: int = 0
    detected_positive_frames: int = 0

    def add(self, gt_positive: bool, pred_positive: bool) -> None:
        if gt_positive:
            self.gt_positive_frames += 1
            if pred_positive:
                self.detected_positive_frames += 1


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def confusion(gt: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.logical_and(gt, pred).sum(dtype=np.int64))
    fp = int(np.logical_and(~gt, pred).sum(dtype=np.int64))
    fn = int(np.logical_and(gt, ~pred).sum(dtype=np.int64))
    tn = int(np.logical_and(~gt, ~pred).sum(dtype=np.int64))
    return tp, fp, fn, tn


def f1_from_confusion(tp: int, fp: int, fn: int) -> float:
    return safe_div(2 * tp, 2 * tp + fp + fn)


def miou_from_confusion(tp: int, fp: int, fn: int, tn: int) -> float:
    iou_fg = safe_div(tp, tp + fp + fn)
    iou_bg = safe_div(tn, tn + fp + fn)
    return (iou_fg + iou_bg) / 2.0


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


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 2:
        return mask
    largest_idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return labels == largest_idx


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
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < pred_block_min_pixels:
            continue
        if area / total_pixels > pred_block_max_area_ratio:
            continue
        comp = labels == i
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
    eroded = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
    return np.logical_and(m > 0, eroded == 0)


def boundary_f1_and_hd95(gt: np.ndarray, pred: np.ndarray, tol_px: int) -> tuple[float, float]:
    gt_b = boundary_map(gt)
    pr_b = boundary_map(pred)

    gt_n = int(gt_b.sum(dtype=np.int64))
    pr_n = int(pr_b.sum(dtype=np.int64))

    if gt_n == 0 and pr_n == 0:
        return 1.0, 0.0
    if gt_n == 0 or pr_n == 0:
        diag = float(np.hypot(gt.shape[0], gt.shape[1]))
        return 0.0, diag

    if tol_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_px + 1, 2 * tol_px + 1))
        gt_d = cv2.dilate(gt_b.astype(np.uint8), k, iterations=1) > 0
        pr_d = cv2.dilate(pr_b.astype(np.uint8), k, iterations=1) > 0
    else:
        gt_d = gt_b
        pr_d = pr_b

    b_prec = safe_div(int(np.logical_and(pr_b, gt_d).sum(dtype=np.int64)), pr_n)
    b_rec = safe_div(int(np.logical_and(gt_b, pr_d).sum(dtype=np.int64)), gt_n)
    b_f1 = safe_div(2.0 * b_prec * b_rec, b_prec + b_rec)

    dist_to_gt = cv2.distanceTransform((~gt_b).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_pr = cv2.distanceTransform((~pr_b).astype(np.uint8), cv2.DIST_L2, 3)
    hd95 = float(max(np.percentile(dist_to_gt[pr_b], 95), np.percentile(dist_to_pr[gt_b], 95)))

    return b_f1, hd95


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact three-group comparison metrics.")
    parser.add_argument("--gt-root", type=Path, default=Path("/data/fj/03-GT"))
    parser.add_argument("--sam-root", type=Path, default=Path("/data/fj/03-SAM"))
    parser.add_argument("--seem-root", type=Path, default=Path("/data/fj/03-SEEM"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/fj/03-GT"))
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--seem-threshold", type=int, default=10)
    parser.add_argument("--gt-block-area-ratio-thresh", type=float, default=0.03)
    parser.add_argument("--pred-block-max-area-ratio", type=float, default=0.03)
    parser.add_argument("--pred-block-min-pixels", type=int, default=200)
    parser.add_argument("--pred-block-min-blue-ratio", type=float, default=0.20)
    parser.add_argument("--blue-delta-r", type=int, default=20)
    parser.add_argument("--blue-delta-g", type=int, default=20)
    parser.add_argument("--blue-b-min", type=int, default=50)
    parser.add_argument("--boundary-tolerance-px", type=int, default=3)
    parser.add_argument("--no-keep-largest-plant", action="store_true")
    args = parser.parse_args()

    keep_largest_plant = not args.no_keep_largest_plant
    run_name = args.run_name or f"three_group_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = args.output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    plants = sorted(d.name for d in args.gt_root.iterdir() if d.is_dir() and any(d.glob("*.json")))

    overall_plant = {m: PixelCounter() for m in METHODS}
    overall_block = {m: PixelCounter() for m in METHODS}
    overall_detect = {m: BlockDetectCounter() for m in METHODS}
    overall_boundary = {m: [] for m in METHODS}
    overall_hd95 = {m: [] for m in METHODS}

    per_plant_rows = []
    frame_rows = []
    count_rows = []

    for plant in plants:
        gt_dir = args.gt_root / plant
        gt_jsons = sorted(gt_dir.glob("*.json"))

        plant_plant = {m: PixelCounter() for m in METHODS}
        plant_block = {m: PixelCounter() for m in METHODS}
        plant_detect = {m: BlockDetectCounter() for m in METHODS}
        plant_boundary = {m: [] for m in METHODS}
        plant_hd95 = {m: [] for m in METHODS}

        missing = {m: 0 for m in METHODS}
        common_frames = 0
        common_block_positive = 0

        for json_path in gt_jsons:
            frame = json_path.stem
            gt_plant, gt_block, image_rel = load_gt_masks(json_path, args.gt_block_area_ratio_thresh)
            gt_targets = {"plant": gt_plant, "block": gt_block}

            img = cv2.imread(str(gt_dir / image_rel), cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((gt_plant.shape[0], gt_plant.shape[1], 3), dtype=np.uint8)
            elif img.shape[:2] != gt_plant.shape:
                img = cv2.resize(img, (gt_plant.shape[1], gt_plant.shape[0]), interpolation=cv2.INTER_LINEAR)

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
                    image_bgr=img,
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

            common_frames += 1
            gt_block_present = bool(gt_block.any())
            if gt_block_present:
                common_block_positive += 1

            for method in METHODS:
                pred_targets = pred_targets_by_method[method]

                tp, fp, fn, tn = confusion(gt_targets["plant"], pred_targets["plant"])
                plant_plant[method].add(tp, fp, fn, tn)
                overall_plant[method].add(tp, fp, fn, tn)

                bf1, hd95 = boundary_f1_and_hd95(gt_targets["plant"], pred_targets["plant"], args.boundary_tolerance_px)
                plant_boundary[method].append(bf1)
                plant_hd95[method].append(hd95)
                overall_boundary[method].append(bf1)
                overall_hd95[method].append(hd95)

                btp, bfp, bfn, btn = confusion(gt_targets["block"], pred_targets["block"])
                plant_block[method].add(btp, bfp, bfn, btn)
                overall_block[method].add(btp, bfp, bfn, btn)

                pred_block_present = bool(pred_targets["block"].any())
                plant_detect[method].add(gt_block_present, pred_block_present)
                overall_detect[method].add(gt_block_present, pred_block_present)

                frame_rows.append(
                    {
                        "plant": plant,
                        "frame": frame,
                        "method": method,
                        "plant_f1": f1_from_confusion(tp, fp, fn),
                        "plant_miou": miou_from_confusion(tp, fp, fn, tn),
                        "plant_boundary_f1": bf1,
                        "plant_hd95_px": hd95,
                        "block_recall_pixel": safe_div(btp, btp + bfn),
                        "block_detected": int(pred_block_present),
                        "gt_block_present": int(gt_block_present),
                    }
                )

        count_rows.append(
            {
                "plant": plant,
                "gt_frames": len(gt_jsons),
                "common_frames": common_frames,
                "common_block_positive_frames": common_block_positive,
                "missing_sam_frames": missing["SAM"],
                "missing_seem_frames": missing["SEEM"],
            }
        )

        for method in METHODS:
            pp = plant_plant[method]
            pb = plant_block[method]
            pd = plant_detect[method]
            per_plant_rows.append(
                {
                    "plant": plant,
                    "method": method,
                    "common_frames": common_frames,
                    "common_block_positive_frames": pd.gt_positive_frames,
                    "plant_f1": f1_from_confusion(pp.tp, pp.fp, pp.fn),
                    "plant_miou": miou_from_confusion(pp.tp, pp.fp, pp.fn, pp.tn),
                    "plant_boundary_f1_mean": mean_or_nan(plant_boundary[method]),
                    "plant_hd95_mean_px": mean_or_nan(plant_hd95[method]),
                    "block_recall_pixel": safe_div(pb.tp, pb.tp + pb.fn),
                    "block_detection_rate": safe_div(pd.detected_positive_frames, pd.gt_positive_frames),
                }
            )

    overall_rows = []
    for method in METHODS:
        pp = overall_plant[method]
        pb = overall_block[method]
        pd = overall_detect[method]
        overall_rows.append(
            {
                "method": method,
                "common_frames": pp.n_frames,
                "common_block_positive_frames": pd.gt_positive_frames,
                "plant_f1": f1_from_confusion(pp.tp, pp.fp, pp.fn),
                "plant_miou": miou_from_confusion(pp.tp, pp.fp, pp.fn, pp.tn),
                "plant_boundary_f1_mean": mean_or_nan(overall_boundary[method]),
                "plant_hd95_mean_px": mean_or_nan(overall_hd95[method]),
                "block_recall_pixel": safe_div(pb.tp, pb.tp + pb.fn),
                "block_detection_rate": safe_div(pd.detected_positive_frames, pd.gt_positive_frames),
            }
        )

    write_csv(
        out_dir / "summary_overall_three_groups.csv",
        overall_rows,
        [
            "method",
            "common_frames",
            "common_block_positive_frames",
            "plant_f1",
            "plant_miou",
            "plant_boundary_f1_mean",
            "plant_hd95_mean_px",
            "block_recall_pixel",
            "block_detection_rate",
        ],
    )
    write_csv(
        out_dir / "summary_per_plant_three_groups.csv",
        per_plant_rows,
        [
            "plant",
            "method",
            "common_frames",
            "common_block_positive_frames",
            "plant_f1",
            "plant_miou",
            "plant_boundary_f1_mean",
            "plant_hd95_mean_px",
            "block_recall_pixel",
            "block_detection_rate",
        ],
    )
    write_csv(
        out_dir / "frame_metrics_three_groups_common.csv",
        frame_rows,
        [
            "plant",
            "frame",
            "method",
            "plant_f1",
            "plant_miou",
            "plant_boundary_f1",
            "plant_hd95_px",
            "block_recall_pixel",
            "block_detected",
            "gt_block_present",
        ],
    )
    write_csv(
        out_dir / "common_frame_counts.csv",
        count_rows,
        ["plant", "gt_frames", "common_frames", "common_block_positive_frames", "missing_sam_frames", "missing_seem_frames"],
    )

    k = {r["method"]: r for r in overall_rows}
    report_lines = [
        "# Three-Group Comparison Report",
        "",
        f"- Output dir: `{out_dir}`",
        "- Evaluation subset: common frames (`GT ∩ SAM ∩ SEEM`).",
        "",
        "## 1) Global Basic Performance (Plant)",
        "",
        "| Method | Plant mIoU | Plant F1 |",
        "|---|---:|---:|",
        f"| SAM | {k['SAM']['plant_miou']:.6f} | {k['SAM']['plant_f1']:.6f} |",
        f"| SEEM | {k['SEEM']['plant_miou']:.6f} | {k['SEEM']['plant_f1']:.6f} |",
        "",
        "## 2) Extreme Edge Precision (Plant)",
        "",
        "| Method | Plant HD95 (px) | Plant Boundary F1 |",
        "|---|---:|---:|",
        f"| SAM | {k['SAM']['plant_hd95_mean_px']:.6f} | {k['SAM']['plant_boundary_f1_mean']:.6f} |",
        f"| SEEM | {k['SEEM']['plant_hd95_mean_px']:.6f} | {k['SEEM']['plant_boundary_f1_mean']:.6f} |",
        "",
        "## 3) Composite Semantic Alignment (Blue Block)",
        "",
        "| Method | Block Recall (pixel-level) | Block Detection Rate (frame-level) |",
        "|---|---:|---:|",
        f"| SAM | {k['SAM']['block_recall_pixel']:.6f} | {k['SAM']['block_detection_rate']:.6f} |",
        f"| SEEM | {k['SEEM']['block_recall_pixel']:.6f} | {k['SEEM']['block_detection_rate']:.6f} |",
        "",
        "## Files",
        "",
        f"- Overall summary: `{out_dir / 'summary_overall_three_groups.csv'}`",
        f"- Per-plant summary: `{out_dir / 'summary_per_plant_three_groups.csv'}`",
        f"- Frame-level metrics: `{out_dir / 'frame_metrics_three_groups_common.csv'}`",
        f"- Common-frame counts: `{out_dir / 'common_frame_counts.csv'}`",
    ]
    (out_dir / "report_three_groups.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[DONE] Three-group metrics generated: {out_dir}")


if __name__ == "__main__":
    main()
