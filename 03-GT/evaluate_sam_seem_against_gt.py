#!/usr/bin/env python3
"""Evaluate SAM / SEEM segmentation against manual LabelMe GT."""

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


def json_to_gt_mask(json_path: Path) -> np.ndarray:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    h = int(data["imageHeight"])
    w = int(data["imageWidth"])
    mask = np.zeros((h, w), dtype=np.uint8)
    for shape in data.get("shapes", []):
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        pts = np.asarray(points, dtype=np.float32)
        pts = np.round(pts).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        cv2.fillPoly(mask, [pts], 255)
    return mask > 0


def read_pred_mask(pred_path: Path, target_shape: tuple[int, int], seem_threshold: int) -> np.ndarray | None:
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


def confusion(gt_mask: np.ndarray, pred_mask: np.ndarray) -> tuple[int, int, int, int]:
    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool)
    tp = int(np.logical_and(gt, pred).sum(dtype=np.int64))
    fp = int(np.logical_and(~gt, pred).sum(dtype=np.int64))
    fn = int(np.logical_and(gt, ~pred).sum(dtype=np.int64))
    tn = int(np.logical_and(~gt, ~pred).sum(dtype=np.int64))
    return tp, fp, fn, tn


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = image_bgr.copy()
    color = np.array(color_bgr, dtype=np.float32)
    mask_bool = mask.astype(bool)
    out_f = out.astype(np.float32)
    out_f[mask_bool] = (1.0 - alpha) * out_f[mask_bool] + alpha * color
    return out_f.astype(np.uint8)


def make_error_map(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    h, w = gt_mask.shape
    err = np.zeros((h, w, 3), dtype=np.uint8)
    tp = np.logical_and(gt_mask, pred_mask)
    fp = np.logical_and(~gt_mask, pred_mask)
    fn = np.logical_and(gt_mask, ~pred_mask)
    err[tp] = (0, 180, 0)
    err[fp] = (0, 0, 255)
    err[fn] = (255, 0, 0)
    return err


def label_panel(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 46), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def load_or_blank(image_path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    h, w = shape_hw
    if img is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


def save_visualization(
    out_path: Path,
    image_path: Path,
    gt_mask: np.ndarray,
    sam_mask: np.ndarray,
    seem_mask: np.ndarray,
    vis_max_width: int,
) -> None:
    h, w = gt_mask.shape
    image = load_or_blank(image_path, (h, w))
    gt_overlay = overlay_mask(image, gt_mask, (0, 255, 255), 0.55)
    sam_overlay = overlay_mask(image, sam_mask, (0, 255, 0), 0.55)
    seem_overlay = overlay_mask(image, seem_mask, (255, 255, 0), 0.55)
    sam_err = make_error_map(gt_mask, sam_mask)
    seem_err = make_error_map(gt_mask, seem_mask)

    panels = [
        label_panel(image, "Original"),
        label_panel(gt_overlay, "GT"),
        label_panel(sam_overlay, "SAM"),
        label_panel(seem_overlay, "SEEM"),
        label_panel(sam_err, "SAM Error TP/FP/FN"),
        label_panel(seem_err, "SEEM Error TP/FP/FN"),
    ]

    scale = min(1.0, float(vis_max_width) / float(w))
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        panels = [cv2.resize(p, (new_w, new_h), interpolation=cv2.INTER_AREA) for p in panels]

    row1 = np.hstack(panels[:3])
    row2 = np.hstack(panels[3:])
    grid = np.vstack([row1, row2])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)


def append_metric_row(
    rows: list[dict[str, object]],
    plant: str,
    stem: str,
    method: str,
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> None:
    mm = metrics_from_confusion(tp, fp, fn, tn)
    rows.append(
        {
            "plant": plant,
            "frame": stem,
            "method": method,
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
    plant: str,
    gt_frames: int,
    counters_by_method: dict[str, Counters],
    missing_by_method: dict[str, list[str]],
) -> None:
    for method in METHODS:
        cnt = counters_by_method[method]
        mm = metrics_from_confusion(cnt.tp, cnt.fp, cnt.fn, cnt.tn)
        missing_n = len(set(missing_by_method.get(method, [])))
        out_rows.append(
            {
                "plant": plant,
                "method": method,
                "gt_frames": gt_frames,
                "eval_frames": cnt.n_eval_frames,
                "missing_frames": missing_n,
                "precision": mm["precision"],
                "recall": mm["recall"],
                "f1": mm["f1"],
                "miou": mm["miou_binary"],
                "iou_fg": mm["iou_fg"],
            }
        )


def overall_rows_from_counters(overall: dict[str, Counters]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in METHODS:
        cnt = overall[method]
        mm = metrics_from_confusion(cnt.tp, cnt.fp, cnt.fn, cnt.tn)
        rows.append(
            {
                "method": method,
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
    parser = argparse.ArgumentParser(description="Evaluate SAM/SEEM against manual GT.")
    parser.add_argument("--gt-root", type=Path, default=Path("/data/fj/03-GT"))
    parser.add_argument("--sam-root", type=Path, default=Path("/data/fj/03-SAM"))
    parser.add_argument("--seem-root", type=Path, default=Path("/data/fj/03-SEEM"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/fj/03-GT"))
    parser.add_argument("--seem-threshold", type=int, default=10)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--vis-max-width", type=int, default=960)
    args = parser.parse_args()

    run_name = args.run_name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    gt_plants = sorted(d.name for d in args.gt_root.iterdir() if d.is_dir() and any(d.glob("*.json")))

    frame_fields = [
        "plant",
        "frame",
        "method",
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
        "method",
        "gt_frames",
        "eval_frames",
        "missing_frames",
        "precision",
        "recall",
        "f1",
        "miou",
        "iou_fg",
    ]
    overall_fields = ["method", "eval_frames", "precision", "recall", "f1", "miou", "iou_fg"]

    summary_rows_all: list[dict[str, object]] = []
    summary_rows_common: list[dict[str, object]] = []
    overall_all: dict[str, Counters] = {m: Counters() for m in METHODS}
    overall_common: dict[str, Counters] = {m: Counters() for m in METHODS}
    common_count_rows: list[dict[str, object]] = []

    for plant in gt_plants:
        gt_dir = args.gt_root / plant
        sam_dir = args.sam_root / plant
        seem_dir = args.seem_root / plant
        out_plant_dir = run_dir / plant
        out_plant_dir.mkdir(parents=True, exist_ok=True)

        gt_jsons = sorted(gt_dir.glob("*.json"))
        gt_stems = [p.stem for p in gt_jsons]
        frame_rows_all: list[dict[str, object]] = []
        frame_rows_common: list[dict[str, object]] = []
        missing_frames: dict[str, list[str]] = {m: [] for m in METHODS}
        plant_all: dict[str, Counters] = {m: Counters() for m in METHODS}
        plant_common: dict[str, Counters] = {m: Counters() for m in METHODS}
        frame_cache: dict[str, dict[str, object]] = {}

        for json_path in gt_jsons:
            stem = json_path.stem
            gt_mask = json_to_gt_mask(json_path)
            h, w = gt_mask.shape
            pred_paths = {
                "SAM": sam_dir / f"mask_{stem}.png",
                "SEEM": seem_dir / f"{stem}.jpg",
            }
            pred_masks: dict[str, np.ndarray | None] = {}
            for method in METHODS:
                pred_masks[method] = read_pred_mask(pred_paths[method], (h, w), args.seem_threshold)
                if pred_masks[method] is None:
                    missing_frames[method].append(stem)
                    continue
                tp, fp, fn, tn = confusion(gt_mask, pred_masks[method])
                plant_all[method].add(tp, fp, fn, tn)
                overall_all[method].add(tp, fp, fn, tn)
                append_metric_row(frame_rows_all, plant, stem, method, tp, fp, fn, tn)

            frame_cache[stem] = {
                "gt": gt_mask,
                "pred": pred_masks,
                "img_path": gt_dir / f"{stem}.jpg",
            }

        common_stems = sorted(
            stem
            for stem in gt_stems
            if stem in frame_cache
            and frame_cache[stem]["pred"]["SAM"] is not None
            and frame_cache[stem]["pred"]["SEEM"] is not None
        )

        for stem in common_stems:
            gt_mask = frame_cache[stem]["gt"]
            for method in METHODS:
                pred_mask = frame_cache[stem]["pred"][method]
                tp, fp, fn, tn = confusion(gt_mask, pred_mask)
                plant_common[method].add(tp, fp, fn, tn)
                overall_common[method].add(tp, fp, fn, tn)
                append_metric_row(frame_rows_common, plant, stem, method, tp, fp, fn, tn)

            vis_path = out_plant_dir / "visualizations" / f"{stem}.jpg"
            save_visualization(
                out_path=vis_path,
                image_path=frame_cache[stem]["img_path"],
                gt_mask=gt_mask,
                sam_mask=frame_cache[stem]["pred"]["SAM"],
                seem_mask=frame_cache[stem]["pred"]["SEEM"],
                vis_max_width=args.vis_max_width,
            )

        write_csv(out_plant_dir / "frame_metrics.csv", frame_rows_all, frame_fields)
        write_csv(out_plant_dir / "frame_metrics_common.csv", frame_rows_common, frame_fields)

        append_summary_rows(summary_rows_all, plant, len(gt_stems), plant_all, missing_frames)
        append_summary_rows(summary_rows_common, plant, len(common_stems), plant_common, {m: [] for m in METHODS})

        missing_lines = []
        for method in METHODS:
            missing = sorted(set(missing_frames[method]))
            missing_lines.append(f"{method}_missing_frames={','.join(missing) if missing else '(none)'}")
        missing_lines.append(f"common_frames={','.join(common_stems) if common_stems else '(none)'}")
        (out_plant_dir / "missing_frames.txt").write_text("\n".join(missing_lines) + "\n", encoding="utf-8")

        common_count_rows.append(
            {
                "plant": plant,
                "gt_frames": len(gt_stems),
                "common_frames": len(common_stems),
            }
        )

    write_csv(run_dir / "summary_per_plant.csv", summary_rows_all, summary_fields)
    write_csv(run_dir / "summary_per_plant_common.csv", summary_rows_common, summary_fields)
    write_csv(run_dir / "summary_overall.csv", overall_rows_from_counters(overall_all), overall_fields)
    write_csv(run_dir / "summary_overall_common.csv", overall_rows_from_counters(overall_common), overall_fields)
    write_csv(run_dir / "common_frame_counts.csv", common_count_rows, ["plant", "gt_frames", "common_frames"])

    overall_all_rows = overall_rows_from_counters(overall_all)
    overall_common_rows = overall_rows_from_counters(overall_common)

    report_lines = [
        "# GT vs SAM/SEEM Evaluation Report",
        "",
        f"- Run directory: `{run_dir}`",
        f"- GT root: `{args.gt_root}`",
        f"- SAM root: `{args.sam_root}`",
        f"- SEEM root: `{args.seem_root}`",
        f"- SEEM binarization threshold: `{args.seem_threshold}` (`max(R,G,B) > threshold`)",
        f"- Visualization width per panel: `{args.vis_max_width}`",
        "- AP/mAP: not computed (current data has no per-instance confidence/ranking outputs).",
        "",
        "## Overall (All Available Frames Per Method)",
        "",
        "| Method | Eval Frames | Precision | Recall | F1 | mIoU | IoU_fg |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall_all_rows:
        report_lines.append(
            f"| {row['method']} | {row['eval_frames']} | {row['precision']:.6f} | {row['recall']:.6f} | "
            f"{row['f1']:.6f} | {row['miou']:.6f} | {row['iou_fg']:.6f} |"
        )

    report_lines += [
        "",
        "## Overall (Common Frames Only: GT∩SAM∩SEEM)",
        "",
        "| Method | Eval Frames | Precision | Recall | F1 | mIoU | IoU_fg |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall_common_rows:
        report_lines.append(
            f"| {row['method']} | {row['eval_frames']} | {row['precision']:.6f} | {row['recall']:.6f} | "
            f"{row['f1']:.6f} | {row['miou']:.6f} | {row['iou_fg']:.6f} |"
        )

    report_lines += [
        "",
        "## Per Plant (Common Frames Only)",
        "",
        "| Plant | Method | GT Frames | Eval Frames | Missing Frames | F1 | mIoU | IoU_fg |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows_common:
        report_lines.append(
            f"| {row['plant']} | {row['method']} | {row['gt_frames']} | {row['eval_frames']} | "
            f"{row['missing_frames']} | {row['f1']:.6f} | {row['miou']:.6f} | {row['iou_fg']:.6f} |"
        )

    report_lines += [
        "",
        "## Per Plant (All Available Frames Per Method)",
        "",
        "| Plant | Method | GT Frames | Eval Frames | Missing Frames | F1 | mIoU | IoU_fg |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows_all:
        report_lines.append(
            f"| {row['plant']} | {row['method']} | {row['gt_frames']} | {row['eval_frames']} | "
            f"{row['missing_frames']} | {row['f1']:.6f} | {row['miou']:.6f} | {row['iou_fg']:.6f} |"
        )

    (run_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[DONE] Evaluation completed: {run_dir}")


if __name__ == "__main__":
    main()
