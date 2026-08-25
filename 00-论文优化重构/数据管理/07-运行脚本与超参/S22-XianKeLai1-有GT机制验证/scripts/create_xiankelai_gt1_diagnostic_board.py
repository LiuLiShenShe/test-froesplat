#!/usr/bin/env python3
"""Create a diagnostic board for the labeled XianKeLai1 RAP-FSAM3 frame."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PANEL_SIZE = (760, 960)
GT_AMBER = np.array([238, 170, 46], dtype=np.float32)
PRED_BLUE = np.array([68, 119, 170], dtype=np.float32)
PRED_GREEN = np.array([0, 158, 115], dtype=np.float32)
TP_GREEN = np.array([0, 158, 115], dtype=np.float32)
FP_RED = np.array([213, 94, 0], dtype=np.float32)
FN_BLUE = np.array([0, 114, 178], dtype=np.float32)
REMOVED_ORANGE = np.array([230, 159, 0], dtype=np.float32)
ADDED_PURPLE = np.array([117, 112, 179], dtype=np.float32)
THIN_MAGENTA = np.array([204, 121, 167], dtype=np.float32)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.6,
        "axes.spines.left": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.spines.bottom": False,
        "xtick.bottom": False,
        "ytick.left": False,
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L")) > 127
    if shape is not None and mask.shape != shape:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def blend(image: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    out = image.astype(np.float32).copy()
    out[mask] = out[mask] * (1.0 - alpha) + color * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_contour(image: np.ndarray, mask: np.ndarray, color: np.ndarray, thickness: int = 4) -> np.ndarray:
    out = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, tuple(int(v) for v in color.tolist()), thickness)
    return out


def union_bbox(masks: list[np.ndarray]) -> tuple[int, int, int, int]:
    canvas = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        canvas |= mask
    if not canvas.any():
        h, w = canvas.shape
        return 0, 0, w, h
    ys, xs = np.where(canvas)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    pad_ratio: float = 0.12,
    target_aspect: float = PANEL_SIZE[0] / PANEL_SIZE[1],
) -> tuple[int, int, int, int]:
    h, w = image_shape
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    pad = int(max(bw, bh) * pad_ratio)
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    bw = x1 - x0
    bh = y1 - y0
    if bw / bh < target_aspect:
        bw = bh * target_aspect
    else:
        bh = bw / target_aspect
    x0 = int(round(cx - bw / 2))
    x1 = int(round(cx + bw / 2))
    y0 = int(round(cy - bh / 2))
    y1 = int(round(cy + bh / 2))
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > w:
        x0 -= x1 - w
        x1 = w
    if y1 > h:
        y0 -= y1 - h
        y1 = h
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def crop_rgb(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return np.asarray(Image.fromarray(image[y0:y1, x0:x1]).resize(PANEL_SIZE, Image.Resampling.LANCZOS))


def crop_mask(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    crop = Image.fromarray((mask[y0:y1, x0:x1].astype(np.uint8) * 255))
    return np.asarray(crop.resize(PANEL_SIZE, Image.Resampling.NEAREST)) > 127


def overlay_panel(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, color: np.ndarray) -> np.ndarray:
    out = blend(image, pred, color, 0.42)
    out = draw_contour(out, gt, GT_AMBER, 5)
    return draw_contour(out, pred, color, 4)


def gt_panel(image: np.ndarray, gt: np.ndarray) -> np.ndarray:
    out = blend(image, gt, GT_AMBER, 0.24)
    return draw_contour(out, gt, GT_AMBER, 5)


def error_panel(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = np.clip(image.astype(np.float32) * 0.45 + 255.0 * 0.55, 0, 255).astype(np.uint8)
    out = blend(out, gt & pred, TP_GREEN, 0.82)
    out = blend(out, ~gt & pred, FP_RED, 0.86)
    out = blend(out, gt & ~pred, FN_BLUE, 0.86)
    out = draw_contour(out, gt, GT_AMBER, 4)
    return draw_contour(out, pred, TP_GREEN, 3)


def difference_panel(image: np.ndarray, before: np.ndarray, after: np.ndarray) -> np.ndarray:
    out = np.clip(image.astype(np.float32) * 0.42 + 255.0 * 0.58, 0, 255).astype(np.uint8)
    out = blend(out, before & after, TP_GREEN, 0.45)
    out = blend(out, before & ~after, REMOVED_ORANGE, 0.82)
    out = blend(out, ~before & after, ADDED_PURPLE, 0.82)
    return out


def thin_panel(image: np.ndarray, gt: np.ndarray, thin: np.ndarray) -> np.ndarray:
    out = np.clip(image.astype(np.float32) * 0.48 + 255.0 * 0.52, 0, 255).astype(np.uint8)
    out = blend(out, gt, GT_AMBER, 0.18)
    out = blend(out, thin, THIN_MAGENTA, 0.95)
    out = draw_contour(out, gt, GT_AMBER, 4)
    return draw_contour(out, thin, THIN_MAGENTA, 5)


def metric_lookup(summary_csv: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in read_csv(summary_csv):
        method = row["method"]
        out[method] = {}
        for key, value in row.items():
            if key in {"method", "mask_dir", "missing_frames"}:
                continue
            out[method][key] = float(value) if value else float("nan")
    return out


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tif"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_root",
        type=Path,
        default=Path("00-论文优化重构/数据管理/05-评测结果/S22_XianKeLai1_RAP_FSAM3_GT1"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("00-论文优化重构/计算机与电子农业特刊实验工作区/05-图件与论文映射/仙客来_RAP-FSAM3_GT1机制诊断"),
    )
    args = parser.parse_args()

    image = read_rgb(args.eval_root / "selected_frames" / "0000.jpg")
    gt = read_mask(args.eval_root / "gt_masks" / "mask_0000.png", image.shape[:2])
    method_root = args.eval_root / "method_masks"
    p2 = read_mask(method_root / "P2_candidate" / "mask_0000.png", image.shape[:2])
    a1 = read_mask(method_root / "A1_selected" / "mask_0000.png", image.shape[:2])
    a2 = read_mask(method_root / "A2_spnp" / "mask_0000.png", image.shape[:2])
    a5 = read_mask(method_root / "A5_full" / "mask_0000.png", image.shape[:2])
    thin = read_mask(method_root / "A3_thin_restored" / "mask_0000.png", image.shape[:2])
    bbox = expand_bbox(union_bbox([gt, p2, a1, a2, a5]), image.shape[:2])

    image_c = crop_rgb(image, bbox)
    gt_c = crop_mask(gt, bbox)
    p2_c = crop_mask(p2, bbox)
    a1_c = crop_mask(a1, bbox)
    a2_c = crop_mask(a2, bbox)
    a5_c = crop_mask(a5, bbox)
    thin_c = crop_mask(thin, bbox)

    panels = [
        ("RGB + GT", gt_panel(image_c, gt_c)),
        ("P2 candidate", overlay_panel(image_c, gt_c, p2_c, PRED_BLUE)),
        ("A1 selected", error_panel(image_c, gt_c, a1_c)),
        ("A2/A5", error_panel(image_c, gt_c, a5_c)),
        ("A2 - A1", difference_panel(image_c, a1_c, a2_c)),
        ("A3 restored", thin_panel(image_c, gt_c, thin_c)),
    ]
    metrics = metric_lookup(args.eval_root / "stage_metrics" / "summary_metrics.csv")

    fig, axes = plt.subplots(1, len(panels), figsize=(183 / 25.4, 58 / 25.4), gridspec_kw={"wspace": 0.04})
    for ax, (title, panel) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title, fontsize=7.2, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "XianKeLai1 GT1 diagnostic: P2 remains stronger than selected P3 on the labeled frame "
        f"(P2 F1={metrics['P2_candidate']['f1']:.3f}; A1={metrics['A1_selected']['f1']:.3f}; A5={metrics['A5_full']['f1']:.3f})",
        y=0.985,
        fontsize=8.0,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Error: TP green, FP red, FN blue. Difference: shared green, removed orange, added purple. "
        "A3 restored pixels are magenta.",
        ha="center",
        va="bottom",
        fontsize=6.2,
    )
    save_figure(fig, args.output_root / "figures" / "Fig_XianKeLai1_GT1_RAP_FSAM3_diagnostic")

    note = f"""# 仙客来 GT1 RAP-FSAM3 机制诊断

## 数据范围

- 数据集：XianKeLai1
- 已找到人工标注：`03-GT/XianKeLai1/0000.json`
- 本次只使用已有标注，不新增手工标注。

## 关键结果

| 阶段 | F1 | mIoU | Recall | Boundary F1 | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| P2 candidate | {metrics['P2_candidate']['f1']:.3f} | {metrics['P2_candidate']['miou']:.3f} | {metrics['P2_candidate']['recall']:.3f} | {metrics['P2_candidate']['boundary_f1']:.3f} | 该帧对 GT 最好 |
| A1 selected | {metrics['A1_selected']['f1']:.3f} | {metrics['A1_selected']['miou']:.3f} | {metrics['A1_selected']['recall']:.3f} | {metrics['A1_selected']['boundary_f1']:.3f} | 多提示词选择到 P3，召回下降 |
| A2 SPNP | {metrics['A2_spnp']['f1']:.3f} | {metrics['A2_spnp']['miou']:.3f} | {metrics['A2_spnp']['recall']:.3f} | {metrics['A2_spnp']['boundary_f1']:.3f} | 相对 A1 有小幅边界修正 |
| A5 full | {metrics['A5_full']['f1']:.3f} | {metrics['A5_full']['miou']:.3f} | {metrics['A5_full']['recall']:.3f} | {metrics['A5_full']['boundary_f1']:.3f} | 与 A2/A3 相同 |

## 解释

这帧不支持“A3-A5 在仙客来上明显起效”的强结论。A3 只恢复 4 个像素，A4/A5 几何反馈为 ok、未触发重提示，最终掩膜与 A2 相同。

它支持的结论更克制：

- 已有实现链路可以在仙客来上跑通，并能输出阶段日志、几何反馈和重提示记录。
- 当前 GT1 标注显示，提示词选择策略在小叶/花盆场景上仍可能选错，后续应优先修正 A1 评分或增加仙客来标注帧。
- 如果论文只基于孔雀竹芋 GT5，能够验证 RAP-FSAM3 的 A2 结构化正负提示有效，但不足以证明 A3-A5 在所有形态上都有独立增益。
"""
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "诊断说明.md").write_text(note, encoding="utf-8")
    print(f"Wrote diagnostic board to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
