#!/usr/bin/env python3
"""Generate by-target comparison visuals (integrated + split panels) per plant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

METHODS = ("SAM", "SEEM")
PANEL_NAMES = [
    "01_original",
    "02_gt_plant",
    "03_sam_plant",
    "04_seem_plant",
    "05_gt_block",
    "06_sam_block",
    "07_seem_block",
    "08_metrics",
]


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


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


def load_gt_masks(json_path: Path, gt_block_area_ratio_thresh: float) -> tuple[np.ndarray, np.ndarray, Path]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    h = int(data["imageHeight"])
    w = int(data["imageWidth"])
    total = float(h * w)

    plant_mask = np.zeros((h, w), dtype=bool)
    block_mask = np.zeros((h, w), dtype=bool)
    largest_area = -1
    largest_shape_mask = None

    for shape in data.get("shapes", []):
        sm = shape_to_mask(shape, h, w)
        area = int(sm.sum(dtype=np.int64))
        if area <= 0:
            continue
        if area / total < gt_block_area_ratio_thresh:
            block_mask |= sm
        else:
            plant_mask |= sm
        if area > largest_area:
            largest_area = area
            largest_shape_mask = sm

    if not plant_mask.any() and largest_shape_mask is not None:
        plant_mask = largest_shape_mask.copy()
        block_mask = np.logical_and(block_mask, ~plant_mask)

    image_rel = data.get("imagePath", json_path.with_suffix(".jpg").name)
    return plant_mask, block_mask, json_path.parent / image_rel


def read_combined_pred_mask(
    method: str,
    plant: str,
    frame: str,
    sam_root: Path,
    seem_root: Path,
    target_shape: tuple[int, int],
    seem_threshold: int,
) -> np.ndarray | None:
    if method == "SAM":
        pred_path = sam_root / plant / f"mask_{frame}.png"
    else:
        pred_path = seem_root / plant / f"{frame}.jpg"
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


def make_blue_mask(img_bgr: np.ndarray, delta_r: int, delta_g: int, b_min: int) -> np.ndarray:
    b, g, r = cv2.split(img_bgr)
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
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(areas)) + 1
    return labels == largest_idx


def split_pred_targets(
    pred_combined: np.ndarray,
    img_bgr: np.ndarray,
    pred_block_max_area_ratio: float,
    pred_block_min_pixels: int,
    pred_block_min_blue_ratio: float,
    blue_delta_r: int,
    blue_delta_g: int,
    blue_b_min: int,
    keep_largest_plant: bool,
) -> tuple[np.ndarray, np.ndarray]:
    blue = make_blue_mask(img_bgr, blue_delta_r, blue_delta_g, blue_b_min)
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


def metrics_f1(gt: np.ndarray, pred: np.ndarray) -> float:
    tp = int(np.logical_and(gt, pred).sum(dtype=np.int64))
    fp = int(np.logical_and(~gt, pred).sum(dtype=np.int64))
    fn = int(np.logical_and(gt, ~pred).sum(dtype=np.int64))
    return safe_div(2 * tp, 2 * tp + fp + fn)


def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = img_bgr.copy().astype(np.float32)
    color = np.array(color_bgr, dtype=np.float32)
    m = mask.astype(bool)
    out[m] = (1.0 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def resize_keep_aspect(img_bgr: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = float(target_w) / float(w)
    target_h = max(1, int(round(h * scale)))
    return cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)


def label_panel(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_metrics_panel(
    shape_hw: tuple[int, int],
    frame: str,
    sam_plant_f1: float | None,
    seem_plant_f1: float | None,
    sam_block_f1: float | None,
    seem_block_f1: float | None,
    notes: str,
) -> np.ndarray:
    h, w = shape_hw
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    lines = [
        f"Frame: {frame}",
        f"SAM Plant F1 : {sam_plant_f1:.4f}" if sam_plant_f1 is not None else "SAM Plant F1 : N/A",
        f"SEEM Plant F1: {seem_plant_f1:.4f}" if seem_plant_f1 is not None else "SEEM Plant F1: N/A",
        f"SAM Block F1 : {sam_block_f1:.4f}" if sam_block_f1 is not None else "SAM Block F1 : N/A",
        f"SEEM Block F1: {seem_block_f1:.4f}" if seem_block_f1 is not None else "SEEM Block F1: N/A",
        notes,
    ]
    y = 48
    for ln in lines:
        cv2.putText(panel, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (230, 230, 230), 2, cv2.LINE_AA)
        y += 48
    return panel


def add_frame_header(grid: np.ndarray, frame: str) -> np.ndarray:
    h, w = grid.shape[:2]
    bar_h = 40
    out = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    out[bar_h:, :, :] = grid
    cv2.putText(out, f"Frame {frame}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate integrated and split by-target comparison visuals.")
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, default=Path("/data/fj/03-GT"))
    parser.add_argument("--sam-root", type=Path, default=Path("/data/fj/03-SAM"))
    parser.add_argument("--seem-root", type=Path, default=Path("/data/fj/03-SEEM"))
    parser.add_argument("--subset", type=str, default="common", choices=["common", "all"])
    parser.add_argument("--panel-width", type=int, default=320)
    parser.add_argument("--seem-threshold", type=int, default=10)
    parser.add_argument("--gt-block-area-ratio-thresh", type=float, default=0.03)
    parser.add_argument("--pred-block-max-area-ratio", type=float, default=0.03)
    parser.add_argument("--pred-block-min-pixels", type=int, default=200)
    parser.add_argument("--pred-block-min-blue-ratio", type=float, default=0.20)
    parser.add_argument("--blue-delta-r", type=int, default=20)
    parser.add_argument("--blue-delta-g", type=int, default=20)
    parser.add_argument("--blue-b-min", type=int, default=50)
    parser.add_argument("--no-keep-largest-plant", action="store_true")
    args = parser.parse_args()

    keep_largest_plant = not args.no_keep_largest_plant
    eval_dir = args.eval_dir
    plants = sorted(d.name for d in eval_dir.iterdir() if d.is_dir())

    for plant in plants:
        gt_dir = args.gt_root / plant
        if not gt_dir.is_dir():
            continue

        out_root = eval_dir / plant / "target_compare"
        out_frames = out_root / "frames"
        out_panels = out_root / "panels"
        out_root.mkdir(parents=True, exist_ok=True)
        out_frames.mkdir(parents=True, exist_ok=True)
        out_panels.mkdir(parents=True, exist_ok=True)

        frame_items = []
        for json_path in sorted(gt_dir.glob("*.json")):
            frame = json_path.stem
            gt_plant, gt_block, image_path = load_gt_masks(json_path, args.gt_block_area_ratio_thresh)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                h, w = gt_plant.shape
                image = np.zeros((h, w, 3), dtype=np.uint8)
            elif image.shape[:2] != gt_plant.shape:
                image = cv2.resize(image, (gt_plant.shape[1], gt_plant.shape[0]), interpolation=cv2.INTER_LINEAR)

            pred_targets = {}
            missing_methods = []
            for method in METHODS:
                pred_comb = read_combined_pred_mask(
                    method=method,
                    plant=plant,
                    frame=frame,
                    sam_root=args.sam_root,
                    seem_root=args.seem_root,
                    target_shape=gt_plant.shape,
                    seem_threshold=args.seem_threshold,
                )
                if pred_comb is None:
                    pred_targets[method] = None
                    missing_methods.append(method)
                    continue
                pred_plant, pred_block = split_pred_targets(
                    pred_combined=pred_comb,
                    img_bgr=image,
                    pred_block_max_area_ratio=args.pred_block_max_area_ratio,
                    pred_block_min_pixels=args.pred_block_min_pixels,
                    pred_block_min_blue_ratio=args.pred_block_min_blue_ratio,
                    blue_delta_r=args.blue_delta_r,
                    blue_delta_g=args.blue_delta_g,
                    blue_b_min=args.blue_b_min,
                    keep_largest_plant=keep_largest_plant,
                )
                pred_targets[method] = {"plant": pred_plant, "block": pred_block}

            if args.subset == "common" and any(pred_targets[m] is None for m in METHODS):
                continue

            sam_plant = pred_targets["SAM"]["plant"] if pred_targets["SAM"] is not None else np.zeros_like(gt_plant)
            sam_block = pred_targets["SAM"]["block"] if pred_targets["SAM"] is not None else np.zeros_like(gt_block)
            seem_plant = pred_targets["SEEM"]["plant"] if pred_targets["SEEM"] is not None else np.zeros_like(gt_plant)
            seem_block = pred_targets["SEEM"]["block"] if pred_targets["SEEM"] is not None else np.zeros_like(gt_block)

            sam_plant_f1 = metrics_f1(gt_plant, sam_plant) if pred_targets["SAM"] is not None else None
            seem_plant_f1 = metrics_f1(gt_plant, seem_plant) if pred_targets["SEEM"] is not None else None
            sam_block_f1 = metrics_f1(gt_block, sam_block) if pred_targets["SAM"] is not None else None
            seem_block_f1 = metrics_f1(gt_block, seem_block) if pred_targets["SEEM"] is not None else None

            notes = "Missing: none" if not missing_methods else f"Missing: {','.join(missing_methods)}"
            p1 = label_panel(resize_keep_aspect(image, args.panel_width), "Original")
            p2 = label_panel(resize_keep_aspect(overlay_mask(image, gt_plant, (0, 255, 255), 0.55), args.panel_width), "GT Plant")
            p3 = label_panel(resize_keep_aspect(overlay_mask(image, sam_plant, (0, 255, 0), 0.55), args.panel_width), "SAM Plant")
            p4 = label_panel(
                resize_keep_aspect(overlay_mask(image, seem_plant, (255, 255, 0), 0.55), args.panel_width), "SEEM Plant"
            )
            p5 = label_panel(resize_keep_aspect(overlay_mask(image, gt_block, (0, 165, 255), 0.60), args.panel_width), "GT Block")
            p6 = label_panel(resize_keep_aspect(overlay_mask(image, sam_block, (0, 255, 0), 0.60), args.panel_width), "SAM Block")
            p7 = label_panel(
                resize_keep_aspect(overlay_mask(image, seem_block, (255, 255, 0), 0.60), args.panel_width), "SEEM Block"
            )
            metric_base = make_metrics_panel(
                shape_hw=p1.shape[:2],
                frame=frame,
                sam_plant_f1=sam_plant_f1,
                seem_plant_f1=seem_plant_f1,
                sam_block_f1=sam_block_f1,
                seem_block_f1=seem_block_f1,
                notes=notes,
            )
            p8 = label_panel(metric_base, "Metrics")

            panels = [p1, p2, p3, p4, p5, p6, p7, p8]
            row1 = np.hstack(panels[:4])
            row2 = np.hstack(panels[4:])
            grid = np.vstack([row1, row2])

            grid_path = out_frames / f"{frame}.jpg"
            cv2.imwrite(str(grid_path), grid)

            frame_panel_dir = out_panels / frame
            frame_panel_dir.mkdir(parents=True, exist_ok=True)
            for name, panel in zip(PANEL_NAMES, panels):
                cv2.imwrite(str(frame_panel_dir / f"{name}.jpg"), panel)

            frame_items.append((frame, grid))

        if frame_items:
            integrated = np.vstack([add_frame_header(g, f) for f, g in frame_items])
            cv2.imwrite(str(out_root / "integrated.jpg"), integrated)
        else:
            blank = np.zeros((200, 800, 3), dtype=np.uint8)
            cv2.putText(blank, "No frames selected for this subset.", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(str(out_root / "integrated.jpg"), blank)

    print(f"[DONE] Visualizations generated under: {eval_dir}")


if __name__ == "__main__":
    main()
