#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build four-sample prompt-robustness tables for experiment 2.

The script reuses the existing P1-P5 candidate masks from the four-sample
representative benchmark and evaluates them with the same 2D segmentation
metrics used by experiment 1.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
BENCH_ROOT = WORKSPACE_DIR / "06-实验一四样本代表集/E1_Representative4_PottedPlant_VFM"
RESULT_DIR = WORKSPACE_DIR / "04-结果表格模板"

SAMPLES = ["KongQueZhuYu", "DouBanLv1", "ChangShouHua2", "CaoMei1"]
FRAMES = ["0000", "0025", "0050", "0075", "0100"]

PROMPTS = {
    "P1": {
        "text": "green plant",
        "subdir": "P1_绿色植物",
        "summary_failure": "召回不足，部分样本偏保守",
    },
    "P2": {
        "text": "entire plant excluding pot",
        "subdir": "P2_整株去花盆",
        "summary_failure": "四样本中最稳，但仍存在边界误差和少量泄漏",
    },
    "P3": {
        "text": "leaves and stems",
        "subdir": "P3_叶和茎",
        "summary_failure": "多样本主体召回不足或空掩膜",
    },
    "P4": {
        "text": "crop seedling",
        "subdir": "P4_作物幼苗",
        "summary_failure": "非幼苗/成熟盆栽形态下系统性欠分割",
    },
    "P5": {
        "text": "plant body without background",
        "subdir": "P5_去背景植物体",
        "summary_failure": "去背景语义过保守，多样本空掩膜",
    },
}


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def read_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if shape is not None and mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


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
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1)
    return np.logical_and(mask, eroded == 0)


def boundary_f1(gt: np.ndarray, pred: np.ndarray, tolerance_px: int = 3) -> float:
    gt_b = boundary_map(gt)
    pred_b = boundary_map(pred)
    gt_count = int(gt_b.sum(dtype=np.int64))
    pred_count = int(pred_b.sum(dtype=np.int64))
    if gt_count == 0 and pred_count == 0:
        return 1.0
    if gt_count == 0 or pred_count == 0:
        return 0.0
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * tolerance_px + 1, 2 * tolerance_px + 1),
    )
    gt_dilated = cv2.dilate(gt_b.astype(np.uint8), kernel, iterations=1) > 0
    pred_dilated = cv2.dilate(pred_b.astype(np.uint8), kernel, iterations=1) > 0
    precision = safe_div(int(np.logical_and(pred_b, gt_dilated).sum(dtype=np.int64)), pred_count)
    recall = safe_div(int(np.logical_and(gt_b, pred_dilated).sum(dtype=np.int64)), gt_count)
    return safe_div(2 * precision * recall, precision + recall)


def hd95(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_b = boundary_map(gt)
    pred_b = boundary_map(pred)
    if not gt_b.any() and not pred_b.any():
        return 0.0
    diag = float(np.hypot(gt.shape[0], gt.shape[1]))
    if not gt_b.any() or not pred_b.any():
        return diag
    dist_to_pred = cv2.distanceTransform((~pred_b).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_gt = cv2.distanceTransform((~gt_b).astype(np.uint8), cv2.DIST_L2, 3)
    d1 = dist_to_pred[gt_b]
    d2 = dist_to_gt[pred_b]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def component_count(mask: np.ndarray, min_area_ratio: float = 0.0005) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return 0
    min_area = max(1, int(mask.size * min_area_ratio))
    return int(sum(int(stats[i, cv2.CC_STAT_AREA]) >= min_area for i in range(1, n)))


def summarize(rows: list[dict[str, object]]) -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    hd95_values: list[float] = []
    boundary_values: list[float] = []
    areas: list[float] = []
    components: list[int] = []
    temporal_ious: list[float] = []

    for row in rows:
        tp += int(row["tp"])
        fp += int(row["fp"])
        fn += int(row["fn"])
        tn += int(row["tn"])
        hd95_values.append(float(row["hd95_px"]))
        boundary_values.append(float(row["boundary_f1"]))
        areas.append(float(row["area_ratio"]))
        components.append(int(row["component_count"]))
        if row["temporal_iou"] != "":
            temporal_ious.append(float(row["temporal_iou"]))

    pixel = pixel_metrics(tp, fp, fn, tn)
    shape = rows[0]["shape"] if rows else (0, 0)
    assert isinstance(shape, tuple)
    return {
        **pixel,
        "hd95_px": float(np.mean(hd95_values)) if hd95_values else 0.0,
        "boundary_f1": float(np.mean(boundary_values)) if boundary_values else 0.0,
        "temporal_iou": float(np.mean(temporal_ious)) if temporal_ious else 0.0,
        "area_cv": safe_div(float(np.std(areas)), float(np.mean(areas))) if areas else 0.0,
        "component_count_mean": float(np.mean(components)) if components else 0.0,
        "outside_nonblack_ratio": safe_div(fp, fp + tn),
        "leakage_energy": safe_div(fp, len(rows) * shape[0] * shape[1]) if rows else 0.0,
        "eval_frames": len(rows),
    }


def fmt(value: object, digits: int = 6) -> str:
    if value == "":
        return ""
    return f"{float(value):.{digits}f}"


def classify_failure(prompt: str, row: dict[str, object]) -> tuple[str, str]:
    f1 = float(row["f1"])
    recall = float(row["recall"])
    area_ratio = float(row["area_ratio"])
    leakage = float(row["leakage_energy"])

    if area_ratio < 1e-6:
        if prompt == "P4":
            return "成熟植株欠分割", "幼苗提示未覆盖非幼苗或成熟盆栽主体，输出近空掩膜"
        if prompt in {"P3", "P5"}:
            return "主体召回不足", "语义提示过保守，目标植物主体未被召回"
        return "空掩膜", "候选掩膜为空或接近为空"
    if recall < 0.5:
        return "主体召回不足", "目标植物主体召回不足，无法作为稳定前景先验"
    if leakage > 0.01:
        return "背景或花盆泄漏", "GT 外前景像素较多，可能包含花盆、桌面或背景"
    if f1 < 0.8:
        return "提示词敏感", "该提示词在此帧上明显低于 P2 默认提示"
    return "可用", "掩膜整体可用，但仍需结合边界和泄漏指标判断"


def evaluate_all() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    frame_rows: list[dict[str, object]] = []
    rap_root = BENCH_ROOT / "rap_runs/RAP-FSAM3-v2"

    for sample in SAMPLES:
        prev_by_prompt: dict[str, np.ndarray] = {}
        for frame in FRAMES:
            image_path = BENCH_ROOT / "selected_frames" / f"{sample}_{frame}.jpg"
            gt_path = BENCH_ROOT / "gt_masks" / f"mask_{sample}_{frame}.png"
            gt = read_mask(gt_path)
            for prompt, info in PROMPTS.items():
                mask_path = rap_root / sample / "候选掩膜" / str(info["subdir"]) / f"mask_{frame}.png"
                pred = read_mask(mask_path, gt.shape)
                tp, fp, fn, tn = confusion(gt, pred)
                metrics = pixel_metrics(tp, fp, fn, tn)
                temporal_iou: float | str = ""
                if prompt in prev_by_prompt:
                    prev = prev_by_prompt[prompt]
                    union = np.logical_or(prev, pred).sum()
                    temporal_iou = safe_div(int(np.logical_and(prev, pred).sum()), int(union))
                prev_by_prompt[prompt] = pred

                row: dict[str, object] = {
                    "prompt": prompt,
                    "prompt_text": info["text"],
                    "sample": sample,
                    "frame_id": frame,
                    "frame": f"{sample}_{frame}",
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "gt_path": str(gt_path),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "shape": gt.shape,
                    **metrics,
                    "hd95_px": hd95(gt, pred),
                    "boundary_f1": boundary_f1(gt, pred),
                    "area_ratio": float(pred.sum() / pred.size),
                    "component_count": component_count(pred),
                    "temporal_iou": temporal_iou,
                    "outside_nonblack_ratio": safe_div(fp, fp + tn),
                    "leakage_energy": safe_div(fp, gt.size),
                }
                failure_type, failure_description = classify_failure(prompt, row)
                row["failure_type"] = failure_type
                row["failure_description"] = failure_description
                frame_rows.append(row)

    summary_rows: list[dict[str, object]] = []
    for prompt, info in PROMPTS.items():
        rows = [row for row in frame_rows if row["prompt"] == prompt]
        metrics = summarize(rows)
        summary_rows.append(
            {
                "提示词编号": prompt,
                "提示词文本": info["text"],
                "样本范围": "FourSample_GT20",
                "序列数": len(SAMPLES),
                "帧数": len(FRAMES) * len(SAMPLES),
                "标注帧数": len(rows),
                "F1": fmt(metrics["f1"]),
                "mIoU": fmt(metrics["miou"]),
                "HD95像素": fmt(metrics["hd95_px"]),
                "边界F分数": fmt(metrics["boundary_f1"]),
                "时序IoU": fmt(metrics["temporal_iou"]),
                "面积变异系数": fmt(metrics["area_cv"]),
                "每帧连通域数": fmt(metrics["component_count_mean"]),
                "外部非黑比例": fmt(metrics["outside_nonblack_ratio"]),
                "泄漏能量": fmt(metrics["leakage_energy"]),
                "主要失败模式": info["summary_failure"],
                "备注": "复用四样本代表集 P1-P5 候选掩膜；仅做 2D segmentation robustness，不进入 2DGS",
            }
        )

    sample_rows: list[dict[str, object]] = []
    for sample in SAMPLES:
        for prompt, info in PROMPTS.items():
            rows = [row for row in frame_rows if row["sample"] == sample and row["prompt"] == prompt]
            metrics = summarize(rows)
            sample_rows.append(
                {
                    "样本名": sample,
                    "提示词编号": prompt,
                    "提示词文本": info["text"],
                    "帧数": len(rows),
                    "F1": fmt(metrics["f1"]),
                    "mIoU": fmt(metrics["miou"]),
                    "HD95像素": fmt(metrics["hd95_px"]),
                    "边界F分数": fmt(metrics["boundary_f1"]),
                    "时序IoU": fmt(metrics["temporal_iou"]),
                    "面积变异系数": fmt(metrics["area_cv"]),
                    "每帧连通域数": fmt(metrics["component_count_mean"]),
                    "外部非黑比例": fmt(metrics["outside_nonblack_ratio"]),
                    "泄漏能量": fmt(metrics["leakage_energy"]),
                    "主要现象": info["summary_failure"],
                }
            )

    return summary_rows, sample_rows, frame_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> int:
    summary_rows, sample_rows, frame_rows = evaluate_all()

    write_csv(
        RESULT_DIR / "实验二_提示词鲁棒性四样本汇总表.csv",
        summary_rows,
        [
            "提示词编号",
            "提示词文本",
            "样本范围",
            "序列数",
            "帧数",
            "标注帧数",
            "F1",
            "mIoU",
            "HD95像素",
            "边界F分数",
            "时序IoU",
            "面积变异系数",
            "每帧连通域数",
            "外部非黑比例",
            "泄漏能量",
            "主要失败模式",
            "备注",
        ],
    )
    write_csv(
        RESULT_DIR / "实验二_提示词鲁棒性四样本逐样本表.csv",
        sample_rows,
        [
            "样本名",
            "提示词编号",
            "提示词文本",
            "帧数",
            "F1",
            "mIoU",
            "HD95像素",
            "边界F分数",
            "时序IoU",
            "面积变异系数",
            "每帧连通域数",
            "外部非黑比例",
            "泄漏能量",
            "主要现象",
        ],
    )

    frame_output_rows: list[dict[str, object]] = []
    for row in frame_rows:
        frame_output_rows.append(
            {
                "样本名": row["sample"],
                "帧编号": row["frame_id"],
                "提示词编号": row["prompt"],
                "提示词文本": row["prompt_text"],
                "F1": fmt(row["f1"]),
                "mIoU": fmt(row["miou"]),
                "HD95像素": fmt(row["hd95_px"]),
                "边界F分数": fmt(row["boundary_f1"]),
                "面积比例": fmt(row["area_ratio"]),
                "连通域数": row["component_count"],
                "时序IoU": fmt(row["temporal_iou"]) if row["temporal_iou"] != "" else "",
                "外部非黑比例": fmt(row["outside_nonblack_ratio"]),
                "泄漏能量": fmt(row["leakage_energy"]),
                "原图路径": row["image_path"],
                "掩膜路径": row["mask_path"],
                "人工标注路径": row["gt_path"],
                "失败类型": row["failure_type"],
                "失败描述": row["failure_description"],
            }
        )
    write_csv(
        RESULT_DIR / "实验二_提示词鲁棒性四样本逐帧表.csv",
        frame_output_rows,
        [
            "样本名",
            "帧编号",
            "提示词编号",
            "提示词文本",
            "F1",
            "mIoU",
            "HD95像素",
            "边界F分数",
            "面积比例",
            "连通域数",
            "时序IoU",
            "外部非黑比例",
            "泄漏能量",
            "原图路径",
            "掩膜路径",
            "人工标注路径",
            "失败类型",
            "失败描述",
        ],
    )

    print(f"Wrote {len(summary_rows)} summary rows")
    print(f"Wrote {len(sample_rows)} sample rows")
    print(f"Wrote {len(frame_output_rows)} frame rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
