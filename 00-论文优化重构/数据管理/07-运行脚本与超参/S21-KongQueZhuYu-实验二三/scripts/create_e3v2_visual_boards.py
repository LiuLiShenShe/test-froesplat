#!/usr/bin/env python3
"""Create RAP-FSAM3-v2 mechanism boards for experiment 3."""

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
PRED_BLUE = np.array([68, 119, 170], dtype=np.float32)
PRED_GREEN = np.array([0, 158, 115], dtype=np.float32)
GT_AMBER = np.array([238, 170, 46], dtype=np.float32)
TP_GREEN = np.array([0, 158, 115], dtype=np.float32)
FP_RED = np.array([213, 94, 0], dtype=np.float32)
FN_BLUE = np.array([0, 114, 178], dtype=np.float32)
REMOVED_ORANGE = np.array([230, 159, 0], dtype=np.float32)
ADDED_PURPLE = np.array([117, 112, 179], dtype=np.float32)


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L")) > 127
    if shape and mask.shape != shape:
        mask = cv2.resize(
            mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
        ) > 0
    return mask


def read_rgb_resized(path: Path, shape: tuple[int, int]) -> np.ndarray:
    img = read_rgb(path)
    if img.shape[:2] != shape:
        img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return img


def safe_float(value: str) -> float | str:
    if value == "":
        return ""
    try:
        return float(value)
    except ValueError:
        return value


def metric_map(path: Path) -> dict[tuple[str, str], dict[str, float | str]]:
    out: dict[tuple[str, str], dict[str, float | str]] = {}
    for row in read_csv_rows(path):
        key = (row["method"], row.get("frame", ""))
        out[key] = {name: safe_float(value) for name, value in row.items()}
    return out


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

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    bw = x1 - x0
    bh = y1 - y0
    if bw / bh < target_aspect:
        bw = bh * target_aspect
    else:
        bh = bw / target_aspect
    x0 = int(round(cx - bw / 2.0))
    x1 = int(round(cx + bw / 2.0))
    y0 = int(round(cy - bh / 2.0))
    y1 = int(round(cy + bh / 2.0))

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


def union_bbox(masks: list[np.ndarray]) -> tuple[int, int, int, int]:
    canvas = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        canvas |= mask
    if not canvas.any():
        h, w = canvas.shape
        return 0, 0, w, h
    ys, xs = np.where(canvas)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def crop_resize_rgb(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    crop = Image.fromarray(image[y0:y1, x0:x1])
    return np.asarray(crop.resize(PANEL_SIZE, Image.Resampling.LANCZOS))


def crop_resize_mask(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    crop = Image.fromarray((mask[y0:y1, x0:x1].astype(np.uint8) * 255))
    return np.asarray(crop.resize(PANEL_SIZE, Image.Resampling.NEAREST)) > 127


def blend_mask(image: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    out = image.astype(np.float32).copy()
    out[mask] = out[mask] * (1.0 - alpha) + color * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_contour(image: np.ndarray, mask: np.ndarray, color: np.ndarray, thickness: int = 4) -> np.ndarray:
    out = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, tuple(int(x) for x in color.tolist()), thickness)
    return out


def gt_panel(image: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return draw_contour(blend_mask(image, gt, GT_AMBER, 0.24), gt, GT_AMBER, 5)


def overlay_panel(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, color: np.ndarray) -> np.ndarray:
    out = blend_mask(image, pred, color, 0.42)
    out = draw_contour(out, gt, GT_AMBER, 5)
    return draw_contour(out, pred, color, 4)


def error_panel(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = np.clip(image.astype(np.float32) * 0.45 + 255.0 * 0.55, 0, 255).astype(np.uint8)
    out = blend_mask(out, gt & pred, TP_GREEN, 0.82)
    out = blend_mask(out, ~gt & pred, FP_RED, 0.86)
    out = blend_mask(out, gt & ~pred, FN_BLUE, 0.86)
    out = draw_contour(out, gt, GT_AMBER, 4)
    return draw_contour(out, pred, TP_GREEN, 3)


def difference_panel(image: np.ndarray, before: np.ndarray, after: np.ndarray) -> np.ndarray:
    out = np.clip(image.astype(np.float32) * 0.42 + 255.0 * 0.58, 0, 255).astype(np.uint8)
    out = blend_mask(out, before & after, TP_GREEN, 0.45)
    out = blend_mask(out, before & ~after, REMOVED_ORANGE, 0.82)
    out = blend_mask(out, ~before & after, ADDED_PURPLE, 0.86)
    return out


def geometry_delta_panel(image: np.ndarray, before: np.ndarray, after: np.ndarray) -> np.ndarray:
    diff = before ^ after
    if diff.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        halo = cv2.dilate(diff.astype(np.uint8), k, iterations=2) > 0
    else:
        halo = diff
    out = np.clip(image.astype(np.float32) * 0.36 + 255.0 * 0.64, 0, 255).astype(np.uint8)
    out = blend_mask(out, halo, ADDED_PURPLE, 0.30)
    out = blend_mask(out, before & ~after, REMOVED_ORANGE, 0.88)
    out = blend_mask(out, ~before & after, ADDED_PURPLE, 0.92)
    return out


def format_metric(value: float | str, digits: int = 3) -> str:
    if value == "":
        return "NA"
    return f"{float(value):.{digits}f}"


def set_image_axis(ax: plt.Axes, image: np.ndarray, title: str, tag: str) -> None:
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, pad=3.0, fontsize=7.1)
    ax.text(
        0.035,
        0.045,
        tag,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.1,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.8},
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tif"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_panel_image(image: np.ndarray, stem: Path) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    tif_path = stem.with_suffix(".tif")
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    pil_image.save(png_path, dpi=(600, 600))
    pil_image.save(tif_path, dpi=(600, 600), compression="tiff_lzw")
    return png_path, tif_path


def bbox_record(bbox: tuple[int, int, int, int]) -> dict[str, int]:
    x0, y0, x1, y1 = bbox
    return {
        "crop_x0": x0,
        "crop_y0": y0,
        "crop_x1": x1,
        "crop_y1": y1,
        "crop_width": x1 - x0,
        "crop_height": y1 - y0,
    }


def prepare_panels(
    image_path: Path,
    gt_path: Path,
    before_path: Path,
    after_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    image = read_rgb(image_path)
    gt = read_mask(gt_path, image.shape[:2])
    before = read_mask(before_path, image.shape[:2])
    after = read_mask(after_path, image.shape[:2])
    bbox = expand_bbox(union_bbox([gt, before, after]), image.shape[:2])
    return (
        crop_resize_rgb(image, bbox),
        crop_resize_mask(gt, bbox),
        crop_resize_mask(before, bbox),
        crop_resize_mask(after, bbox),
        bbox,
    )


def export_panel_group(
    panel_root: Path,
    figure: str,
    row_key: str,
    panels: list[tuple[str, str, np.ndarray]],
    bbox: tuple[int, int, int, int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    frame_dir = panel_root / figure / row_key
    for order, (panel_key, panel_label, image) in enumerate(panels, start=1):
        stem = frame_dir / f"{order:02d}_{panel_key}"
        png_path, tif_path = save_panel_image(image, stem)
        rows.append(
            {
                "figure": figure,
                "row_key": row_key,
                "panel_order": order,
                "panel_key": panel_key,
                "panel_label": panel_label,
                "png_path": str(png_path),
                "tif_path": str(tif_path),
                **bbox_record(bbox),
            }
        )
    return rows


def top_geometry_delta_frames(delta_csv: Path, n: int = 2) -> list[str]:
    rows = read_csv_rows(delta_csv)
    ranked = sorted(rows, key=lambda r: float(r["几何修正像素比例"]), reverse=True)
    return [Path(row["图像"]).stem for row in ranked[:n]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(
            "00-论文优化重构/计算机与电子农业特刊实验工作区/05-图件与论文映射/实验三_RAP-FSAM3v2新版图板"
        ),
    )
    args = parser.parse_args()

    s21 = Path("00-论文优化重构/数据管理/05-评测结果/S21_KongQueZhuYu_E2_E3")
    s22 = Path("00-论文优化重构/数据管理/05-评测结果/S22_XianKeLai1_RAP_FSAM3_GT1")
    mask_root = Path("00-论文优化重构/数据管理/03-分割Mask/05-RAP-FSAM3掩膜")
    a1s_dir = mask_root / "XianKeLai1_A1s_语义门控_候选复用_GT1冒烟" / "最终掩膜"
    a2_dir = mask_root / "E3v2_KongQueZhuYu_GT5_A2_A1s结构化正负提示" / "最终掩膜"
    a5c_dir = mask_root / "E3v2_KongQueZhuYu_GT5_A5c_完整RAPFSAM3v2" / "最终掩膜"
    delta_csv = mask_root / "E3v2_KongQueZhuYu_GT5_A5c_完整RAPFSAM3v2" / "corrective_geometry_delta.csv"

    xi_metrics = metric_map(s22 / "v2_a1s_a5c_metrics" / "frame_metrics.csv")
    kong_metrics = metric_map(s21 / "E3v2_KongQueZhuYu_GT5_full_metrics" / "frame_metrics.csv")
    kong_frames = top_geometry_delta_frames(delta_csv, n=2)

    rows_to_plot = [
        {
            "row_key": "XianKeLai1_0000_A1_old_vs_A1s",
            "tag": "XianKeLai1 0000",
            "before_label": "A1 old",
            "after_label": "A1s",
            "image": s22 / "selected_frames" / "0000.jpg",
            "gt": s22 / "gt_masks" / "mask_0000.png",
            "before": s22 / "method_masks" / "A1_selected" / "mask_0000.png",
            "after": a1s_dir / "mask_0000.png",
            "before_metric": xi_metrics[("A1_old", "0000")],
            "after_metric": xi_metrics[("A1s", "0000")],
            "delta_title": "A1s - A1",
        }
    ]
    for frame in kong_frames:
        rows_to_plot.append(
            {
                "row_key": f"KongQueZhuYu_{frame}_A2_vs_A5c",
                "tag": f"KongQueZhuYu {frame}",
                "before_label": "A2",
                "after_label": "A5c",
                "image": s21 / "selected_frames" / f"{frame}.jpg",
                "gt": s21 / "gt_masks" / f"mask_{frame}.png",
                "before": a2_dir / f"mask_{frame}.png",
                "after": a5c_dir / f"mask_{frame}.png",
                "before_metric": kong_metrics[("A2", frame)],
                "after_metric": kong_metrics[("A5c", frame)],
                "delta_title": "A5c - A2",
            }
        )

    fig, axes = plt.subplots(
        len(rows_to_plot),
        6,
        figsize=(183 / 25.4, 148 / 25.4),
        gridspec_kw={"wspace": 0.035, "hspace": 0.13},
    )
    if len(rows_to_plot) == 1:
        axes = np.expand_dims(axes, axis=0)

    panel_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    figure_name = "Fig_E3v2_A1s_A5c_mechanisms"

    for row_idx, spec in enumerate(rows_to_plot):
        image, gt, before, after, bbox = prepare_panels(spec["image"], spec["gt"], spec["before"], spec["after"])
        before_metric = spec["before_metric"]
        after_metric = spec["after_metric"]
        before_title = (
            f"{spec['before_label']} mask\n"
            f"F1={format_metric(before_metric['f1'])}, HD95={format_metric(before_metric['hd95_px'], 1)}"
        )
        after_title = (
            f"{spec['after_label']} mask\n"
            f"F1={format_metric(after_metric['f1'])}, HD95={format_metric(after_metric['hd95_px'], 1)}"
        )
        panels = [
            ("rgb_gt", "RGB + GT", gt_panel(image, gt)),
            ("before_mask", f"{spec['before_label']} mask", overlay_panel(image, gt, before, PRED_BLUE)),
            ("before_error", f"{spec['before_label']} error", error_panel(image, gt, before)),
            ("after_mask", f"{spec['after_label']} mask", overlay_panel(image, gt, after, PRED_GREEN)),
            ("after_error", f"{spec['after_label']} error", error_panel(image, gt, after)),
            ("delta", str(spec["delta_title"]), geometry_delta_panel(image, before, after)),
        ]
        titles = [
            "RGB + GT",
            before_title,
            f"{spec['before_label']} error",
            after_title,
            f"{spec['after_label']} error",
            str(spec["delta_title"]),
        ]
        for col_idx, (_, _, panel) in enumerate(panels):
            set_image_axis(
                axes[row_idx, col_idx],
                panel,
                titles[col_idx] if row_idx == 0 else titles[col_idx].split("\n")[0],
                str(spec["tag"]) if col_idx == 0 else "",
            )
        panel_rows.extend(export_panel_group(args.output_root / "individual_panels", figure_name, str(spec["row_key"]), panels, bbox))
        selection_rows.append(
            {
                "figure": figure_name,
                "row_key": spec["row_key"],
                "tag": spec["tag"],
                "selection_rule": "XianKeLai1 mechanism row plus top-2 KongQueZhuYu A5c geometry-delta frames",
            }
        )
        metric_rows.extend(
            [
                {
                    "figure": figure_name,
                    "row_key": spec["row_key"],
                    "method": spec["before_label"],
                    "f1": before_metric["f1"],
                    "miou": before_metric["miou"],
                    "hd95_px": before_metric["hd95_px"],
                    "boundary_f1": before_metric["boundary_f1"],
                    "leakage_energy": before_metric["leakage_energy"],
                },
                {
                    "figure": figure_name,
                    "row_key": spec["row_key"],
                    "method": spec["after_label"],
                    "f1": after_metric["f1"],
                    "miou": after_metric["miou"],
                    "hd95_px": after_metric["hd95_px"],
                    "boundary_f1": after_metric["boundary_f1"],
                    "leakage_energy": after_metric["leakage_energy"],
                },
            ]
        )

    fig.suptitle(
        "Experiment 3 v2: semantic-gated selection and reconstruction-consistent correction",
        y=0.988,
        fontsize=8.7,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.010,
        "Error: TP green, FP red, FN blue. Delta: shared mask halo, orange removed by the later module, purple added by the later module.",
        ha="center",
        va="bottom",
        fontsize=6.4,
    )
    save_figure(fig, args.output_root / "figures" / figure_name)

    write_csv(
        args.output_root / "source_data" / "selected_frames.csv",
        selection_rows,
        ["figure", "row_key", "tag", "selection_rule"],
    )
    write_csv(
        args.output_root / "source_data" / "figure_metrics_summary.csv",
        metric_rows,
        ["figure", "row_key", "method", "f1", "miou", "hd95_px", "boundary_f1", "leakage_energy"],
    )
    write_csv(
        args.output_root / "source_data" / "individual_panel_index.csv",
        panel_rows,
        [
            "figure",
            "row_key",
            "panel_order",
            "panel_key",
            "panel_label",
            "png_path",
            "tif_path",
            "crop_x0",
            "crop_y0",
            "crop_x1",
            "crop_y1",
            "crop_width",
            "crop_height",
        ],
    )
    note = f"""# 实验三 RAP-FSAM3-v2 新版图板说明

## 存放内容

- `figures/{figure_name}.*`：实验三 v2 机制图板，包含 `XianKeLai1` 的 A1 old vs A1s 语义门控纠错，以及 `KongQueZhuYu` 的 A2 vs A5c 几何一致性补充。
- `individual_panels/`：图板拆分后的无标题单独面板。
- `source_data/figure_metrics_summary.csv`：入图方法的逐帧指标。
- `source_data/selected_frames.csv`：入图行和选帧规则。
- `source_data/individual_panel_index.csv`：单独面板路径和裁剪坐标。

## 选帧规则

- `XianKeLai1 0000`：固定使用 GT1 机制诊断帧，展示旧 A1 错选 P3 后召回不足，A1s 修正为 P2。
- `KongQueZhuYu {", ".join(kong_frames)}`：按 A5c `corrective_geometry_delta.csv` 中几何修正像素比例排序，选取 top-2 帧。

## 当前结论

- A1s 的强证据来自 `XianKeLai1` GT1：A1 old F1 为 0.860，A1s F1 为 0.963。
- A2 是 `KongQueZhuYu` GT5 的主分割增益来源；A5c 在 A2 基础上产生真实几何 delta，并对 HD95/边界 F 分数有小幅补充作用。
- 该图板用于替代旧版 `Fig_E3_ablation_A0_vs_A2A5` 的正式论文用途；旧图板只保留为阶段八诊断图。
"""
    (args.output_root / "图板说明.md").write_text(note, encoding="utf-8")
    print(f"Created E3v2 visual board under {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
