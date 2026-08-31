#!/usr/bin/env python3
"""Evaluate SAM/SEEM separately for plant and blue calibration block."""

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
class Counters:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    n_eval_frames: int = 0

    def add(self, tp: int, fp: int, fn: int, tn: int) -> None:
        self.tp += int(tp)
        self.fp += int(fp)
        self.fn += int(fn)
        self.tn += int(tn)
        self.n_eval_frames += 1


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def metrics_from_confusion(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    iou_fg = safe_div(tp, tp + fp + fn)
    iou_bg = safe_div(tn, tn + fp + fn)
    miou_binary = (iou_fg + iou_bg) / 2.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_fg": iou_fg,
        "miou_binary": miou_binary,
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


def load_gt_target_masks(json_path: Path, block_area_ratio_thresh: float) -> tuple[np.ndarray, np.ndarray, str]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    h = int(data["imageHeight"])
    w = int(data["imageWidth"])
    total = float(h * w)

    plant_mask = np.zeros((h, w), dtype=bool)
    block_mask = np.zeros((h, w), dtype=bool)
    largest_mask = None
    largest_area = -1

    for shape in data.get("shapes", []):
        sm = shape_to_mask(shape, h, w)
        area = int(sm.sum(dtype=np.int64))
        if area <= 0:
            continue
        area_ratio = area / total
        if area_ratio < block_area_ratio_thresh:
            block_mask |= sm
        else:
            plant_mask |= sm
        if area > largest_area:
            largest_area = area
            largest_mask = sm

    # Safety fallback: if all shapes were classified as block, force largest one to plant.
    if not plant_mask.any() and largest_mask is not None:
        plant_mask = largest_mask.copy()
        block_mask = np.logical_and(block_mask, ~plant_mask)

    image_path = data.get("imagePath", json_path.with_suffix(".jpg").name)
    return plant_mask, block_mask, image_path


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


def split_pred_into_targets(
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

    block_mask = np.zeros_like(pred_combined, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(pred_combined.astype(np.uint8), 8)
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < pred_block_min_pixels:
            continue
        area_ratio = area / total_pixels
        if area_ratio > pred_block_max_area_ratio:
            continue
        comp = labels == idx
        blue_ratio = safe_div(int(np.logical_and(comp, blue).sum(dtype=np.int64)), area)
        if blue_ratio >= pred_block_min_blue_ratio:
            block_mask |= comp

    plant_mask = np.logical_and(pred_combined, ~block_mask)
    if keep_largest_plant:
        plant_mask = keep_largest_component(plant_mask)
    return plant_mask, block_mask


def confusion(gt_mask: np.ndarray, pred_mask: np.ndarray) -> tuple[int, int, int, int]:
    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool)
    tp = int(np.logical_and(gt, pred).sum(dtype=np.int64))
    fp = int(np.logical_and(~gt, pred).sum(dtype=np.int64))
    fn = int(np.logical_and(gt, ~pred).sum(dtype=np.int64))
    tn = int(np.logical_and(~gt, ~pred).sum(dtype=np.int64))
    return tp, fp, fn, tn


def make_counter_table() -> dict[str, dict[str, Counters]]:
    return {m: {t: Counters() for t in TARGETS} for m in METHODS}


def append_frame_row(
    rows: list[dict[str, object]],
    plant: str,
    frame: str,
    method: str,
    target: str,
    subset: str,
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> None:
    mm = metrics_from_confusion(tp, fp, fn, tn)
    rows.append(
        {
            "plant": plant,
            "frame": frame,
            "method": method,
            "target": target,
            "subset": subset,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": mm["precision"],
            "recall": mm["recall"],
            "f1": mm["f1"],
            "iou_fg": mm["iou_fg"],
            "miou_binary": mm["miou_binary"],
        }
    )


def append_summary_rows(
    out_rows: list[dict[str, object]],
    counters: dict[str, dict[str, Counters]],
    gt_frames_by_target: dict[str, int],
    plant_name: str,
    subset_name: str,
    missing_frames_by_method: dict[str, int] | None = None,
) -> None:
    for method in METHODS:
        for target in TARGETS:
            cnt = counters[method][target]
            mm = metrics_from_confusion(cnt.tp, cnt.fp, cnt.fn, cnt.tn)
            row = {
                "plant": plant_name,
                "subset": subset_name,
                "method": method,
                "target": target,
                "gt_frames": gt_frames_by_target[target],
                "eval_frames": cnt.n_eval_frames,
                "precision": mm["precision"],
                "recall": mm["recall"],
                "f1": mm["f1"],
                "miou": mm["miou_binary"],
                "iou_fg": mm["iou_fg"],
            }
            if missing_frames_by_method is not None:
                row["missing_frames"] = int(missing_frames_by_method.get(method, 0))
            out_rows.append(row)


def rows_from_overall_counters(counters: dict[str, dict[str, Counters]], subset_name: str) -> list[dict[str, object]]:
    rows = []
    for method in METHODS:
        for target in TARGETS:
            cnt = counters[method][target]
            mm = metrics_from_confusion(cnt.tp, cnt.fp, cnt.fn, cnt.tn)
            rows.append(
                {
                    "subset": subset_name,
                    "method": method,
                    "target": target,
                    "eval_frames": cnt.n_eval_frames,
                    "precision": mm["precision"],
                    "recall": mm["recall"],
                    "f1": mm["f1"],
                    "miou": mm["miou_binary"],
                    "iou_fg": mm["iou_fg"],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SAM and SEEM separately for plant and blue block.")
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
    parser.add_argument("--no-keep-largest-plant", action="store_true")
    args = parser.parse_args()

    run_name = args.run_name or f"eval_targets_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    keep_largest_plant = not args.no_keep_largest_plant

    frame_fields = [
        "plant",
        "frame",
        "method",
        "target",
        "subset",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "f1",
        "iou_fg",
        "miou_binary",
    ]
    summary_fields = [
        "plant",
        "subset",
        "method",
        "target",
        "gt_frames",
        "eval_frames",
        "missing_frames",
        "precision",
        "recall",
        "f1",
        "miou",
        "iou_fg",
    ]
    summary_common_fields = [c for c in summary_fields if c != "missing_frames"]
    overall_fields = ["subset", "method", "target", "eval_frames", "precision", "recall", "f1", "miou", "iou_fg"]

    gt_plants = sorted(d.name for d in args.gt_root.iterdir() if d.is_dir() and any(d.glob("*.json")))

    per_plant_all_rows: list[dict[str, object]] = []
    per_plant_common_rows: list[dict[str, object]] = []
    per_plant_block_pos_common_rows: list[dict[str, object]] = []
    overall_all = make_counter_table()
    overall_common = make_counter_table()
    overall_block_pos_common = {m: Counters() for m in METHODS}
    frame_count_rows: list[dict[str, object]] = []

    for plant in gt_plants:
        gt_dir = args.gt_root / plant
        out_plant_dir = run_dir / plant
        out_plant_dir.mkdir(parents=True, exist_ok=True)

        gt_jsons = sorted(gt_dir.glob("*.json"))
        stems = [p.stem for p in gt_jsons]
        frame_rows_all: list[dict[str, object]] = []
        frame_rows_common: list[dict[str, object]] = []
        frame_rows_block_pos_common: list[dict[str, object]] = []

        missing_by_method: dict[str, list[str]] = {m: [] for m in METHODS}
        plant_all = make_counter_table()
        plant_common = make_counter_table()
        plant_block_pos_common = {m: Counters() for m in METHODS}

        gt_frame_counts = {"plant": len(stems), "block": 0}
        frame_cache: dict[str, dict[str, object]] = {}

        for json_path in gt_jsons:
            stem = json_path.stem
            gt_plant, gt_block, image_rel = load_gt_target_masks(json_path, args.gt_block_area_ratio_thresh)
            gt_targets = {"plant": gt_plant, "block": gt_block}
            if gt_block.any():
                gt_frame_counts["block"] += 1

            image_path = gt_dir / image_rel
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                image_bgr = np.zeros((gt_plant.shape[0], gt_plant.shape[1], 3), dtype=np.uint8)
            elif image_bgr.shape[:2] != gt_plant.shape:
                image_bgr = cv2.resize(
                    image_bgr,
                    (gt_plant.shape[1], gt_plant.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            pred_targets_by_method: dict[str, dict[str, np.ndarray] | None] = {}
            for method in METHODS:
                pred_comb = read_combined_pred_mask(
                    method=method,
                    plant_name=plant,
                    frame_stem=stem,
                    sam_root=args.sam_root,
                    seem_root=args.seem_root,
                    target_shape=gt_plant.shape,
                    seem_threshold=args.seem_threshold,
                )
                if pred_comb is None:
                    missing_by_method[method].append(stem)
                    pred_targets_by_method[method] = None
                    continue

                pred_plant, pred_block = split_pred_into_targets(
                    pred_combined=pred_comb,
                    image_bgr=image_bgr,
                    pred_block_max_area_ratio=args.pred_block_max_area_ratio,
                    pred_block_min_pixels=args.pred_block_min_pixels,
                    pred_block_min_blue_ratio=args.pred_block_min_blue_ratio,
                    blue_delta_r=args.blue_delta_r,
                    blue_delta_g=args.blue_delta_g,
                    blue_b_min=args.blue_b_min,
                    keep_largest_plant=keep_largest_plant,
                )
                pred_targets = {"plant": pred_plant, "block": pred_block}
                pred_targets_by_method[method] = pred_targets

                for target in TARGETS:
                    tp, fp, fn, tn = confusion(gt_targets[target], pred_targets[target])
                    plant_all[method][target].add(tp, fp, fn, tn)
                    overall_all[method][target].add(tp, fp, fn, tn)
                    append_frame_row(
                        frame_rows_all,
                        plant=plant,
                        frame=stem,
                        method=method,
                        target=target,
                        subset="all_available",
                        tp=tp,
                        fp=fp,
                        fn=fn,
                        tn=tn,
                    )

            frame_cache[stem] = {
                "gt": gt_targets,
                "pred": pred_targets_by_method,
                "gt_block_positive": bool(gt_block.any()),
            }

        common_stems = sorted(stem for stem in stems if all(frame_cache[stem]["pred"][m] is not None for m in METHODS))
        common_block_positive_stems = sorted(stem for stem in common_stems if frame_cache[stem]["gt_block_positive"])

        for stem in common_stems:
            gt_targets = frame_cache[stem]["gt"]
            for method in METHODS:
                pred_targets = frame_cache[stem]["pred"][method]
                for target in TARGETS:
                    tp, fp, fn, tn = confusion(gt_targets[target], pred_targets[target])
                    plant_common[method][target].add(tp, fp, fn, tn)
                    overall_common[method][target].add(tp, fp, fn, tn)
                    append_frame_row(
                        frame_rows_common,
                        plant=plant,
                        frame=stem,
                        method=method,
                        target=target,
                        subset="common",
                        tp=tp,
                        fp=fp,
                        fn=fn,
                        tn=tn,
                    )

        for stem in common_block_positive_stems:
            gt_block = frame_cache[stem]["gt"]["block"]
            for method in METHODS:
                pred_block = frame_cache[stem]["pred"][method]["block"]
                tp, fp, fn, tn = confusion(gt_block, pred_block)
                plant_block_pos_common[method].add(tp, fp, fn, tn)
                overall_block_pos_common[method].add(tp, fp, fn, tn)
                append_frame_row(
                    frame_rows_block_pos_common,
                    plant=plant,
                    frame=stem,
                    method=method,
                    target="block",
                    subset="block_positive_common",
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    tn=tn,
                )

        write_csv(out_plant_dir / "frame_metrics_by_target_all.csv", frame_rows_all, frame_fields)
        write_csv(out_plant_dir / "frame_metrics_by_target_common.csv", frame_rows_common, frame_fields)
        write_csv(
            out_plant_dir / "frame_metrics_block_positive_common.csv",
            frame_rows_block_pos_common,
            frame_fields,
        )

        append_summary_rows(
            out_rows=per_plant_all_rows,
            counters=plant_all,
            gt_frames_by_target=gt_frame_counts,
            plant_name=plant,
            subset_name="all_available",
            missing_frames_by_method={m: len(set(missing_by_method[m])) for m in METHODS},
        )

        append_summary_rows(
            out_rows=per_plant_common_rows,
            counters=plant_common,
            gt_frames_by_target={"plant": len(common_stems), "block": len(common_stems)},
            plant_name=plant,
            subset_name="common",
            missing_frames_by_method=None,
        )

        for method in METHODS:
            cnt = plant_block_pos_common[method]
            mm = metrics_from_confusion(cnt.tp, cnt.fp, cnt.fn, cnt.tn)
            per_plant_block_pos_common_rows.append(
                {
                    "plant": plant,
                    "subset": "block_positive_common",
                    "method": method,
                    "target": "block",
                    "gt_frames": len(common_block_positive_stems),
                    "eval_frames": cnt.n_eval_frames,
                    "precision": mm["precision"],
                    "recall": mm["recall"],
                    "f1": mm["f1"],
                    "miou": mm["miou_binary"],
                    "iou_fg": mm["iou_fg"],
                }
            )

        missing_lines = []
        for method in METHODS:
            miss = sorted(set(missing_by_method[method]))
            missing_lines.append(f"{method}_missing_frames={','.join(miss) if miss else '(none)'}")
        missing_lines.append(f"common_frames={','.join(common_stems) if common_stems else '(none)'}")
        missing_lines.append(
            f"common_block_positive_frames={','.join(common_block_positive_stems) if common_block_positive_stems else '(none)'}"
        )
        (out_plant_dir / "missing_frames_by_target_eval.txt").write_text("\n".join(missing_lines) + "\n", encoding="utf-8")

        frame_count_rows.append(
            {
                "plant": plant,
                "gt_frames": len(stems),
                "gt_block_positive_frames": gt_frame_counts["block"],
                "common_frames": len(common_stems),
                "common_block_positive_frames": len(common_block_positive_stems),
            }
        )

    write_csv(run_dir / "summary_per_plant_by_target_all.csv", per_plant_all_rows, summary_fields)
    write_csv(run_dir / "summary_per_plant_by_target_common.csv", per_plant_common_rows, summary_common_fields)
    write_csv(
        run_dir / "summary_per_plant_block_positive_common.csv",
        per_plant_block_pos_common_rows,
        summary_common_fields,
    )

    overall_all_rows = rows_from_overall_counters(overall_all, "all_available")
    overall_common_rows = rows_from_overall_counters(overall_common, "common")
    overall_block_pos_rows: list[dict[str, object]] = []
    for method in METHODS:
        cnt = overall_block_pos_common[method]
        mm = metrics_from_confusion(cnt.tp, cnt.fp, cnt.fn, cnt.tn)
        overall_block_pos_rows.append(
            {
                "subset": "block_positive_common",
                "method": method,
                "target": "block",
                "eval_frames": cnt.n_eval_frames,
                "precision": mm["precision"],
                "recall": mm["recall"],
                "f1": mm["f1"],
                "miou": mm["miou_binary"],
                "iou_fg": mm["iou_fg"],
            }
        )

    write_csv(run_dir / "summary_overall_by_target_all.csv", overall_all_rows, overall_fields)
    write_csv(run_dir / "summary_overall_by_target_common.csv", overall_common_rows, overall_fields)
    write_csv(run_dir / "summary_overall_block_positive_common.csv", overall_block_pos_rows, overall_fields)
    write_csv(
        run_dir / "frame_count_summary.csv",
        frame_count_rows,
        ["plant", "gt_frames", "gt_block_positive_frames", "common_frames", "common_block_positive_frames"],
    )

    report_lines = [
        "# SAM vs SEEM By-Target Evaluation Report",
        "",
        f"- Run directory: `{run_dir}`",
        f"- GT root: `{args.gt_root}`",
        f"- SAM root: `{args.sam_root}`",
        f"- SEEM root: `{args.seem_root}`",
        f"- SEEM threshold: `{args.seem_threshold}`",
        f"- GT block area threshold: `{args.gt_block_area_ratio_thresh}`",
        f"- Pred block rules: `area_ratio <= {args.pred_block_max_area_ratio}`, `area >= {args.pred_block_min_pixels}`, "
        f"`blue_ratio >= {args.pred_block_min_blue_ratio}`",
        f"- Blue mask rule: `B>=G+{args.blue_delta_g}`, `B>=R+{args.blue_delta_r}`, `B>={args.blue_b_min}`",
        f"- Keep largest plant component: `{keep_largest_plant}`",
        "",
        "## Overall (Common Frames)",
        "",
        "| Method | Target | Eval Frames | Precision | Recall | F1 | mIoU | IoU_fg |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in overall_common_rows:
        report_lines.append(
            f"| {row['method']} | {row['target']} | {row['eval_frames']} | {row['precision']:.6f} | "
            f"{row['recall']:.6f} | {row['f1']:.6f} | {row['miou']:.6f} | {row['iou_fg']:.6f} |"
        )

    report_lines += [
        "",
        "## Overall (Block Positive Common Frames)",
        "",
        "| Method | Eval Frames | Precision | Recall | F1 | mIoU | IoU_fg |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall_block_pos_rows:
        report_lines.append(
            f"| {row['method']} | {row['eval_frames']} | {row['precision']:.6f} | {row['recall']:.6f} | "
            f"{row['f1']:.6f} | {row['miou']:.6f} | {row['iou_fg']:.6f} |"
        )

    report_lines += [
        "",
        "## Per Plant (Block Positive Common Frames)",
        "",
        "| Plant | Method | Eval Frames | F1 | mIoU | IoU_fg |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in per_plant_block_pos_common_rows:
        report_lines.append(
            f"| {row['plant']} | {row['method']} | {row['eval_frames']} | {row['f1']:.6f} | "
            f"{row['miou']:.6f} | {row['iou_fg']:.6f} |"
        )

    report_lines += [
        "",
        "## Notes",
        "",
        "- Plant and block are scored independently.",
        "- AP/mAP is still not computed (no instance-level confidence ranking output).",
    ]

    (run_dir / "report_by_target.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[DONE] By-target evaluation completed: {run_dir}")


if __name__ == "__main__":
    main()
