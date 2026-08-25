#!/usr/bin/env python3
"""Create publication-ready visual boards for KongQueZhuYu experiments 2 and 3."""

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
FAIL_GRAY = np.array([140, 140, 140], dtype=np.float32)
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


def metric_map(summary_csv: Path) -> dict[str, dict[str, float | str]]:
    rows = read_csv_rows(summary_csv)
    out: dict[str, dict[str, float | str]] = {}
    for row in rows:
        method = row["method"]
        out[method] = {}
        for key, value in row.items():
            if key in {"method", "mask_dir", "missing_frames"}:
                out[method][key] = value
            else:
                try:
                    out[method][key] = float(value) if value != "" else ""
                except ValueError:
                    out[method][key] = value
    return out


def frame_metric_map(frame_csv: Path) -> dict[tuple[str, str], dict[str, float | str]]:
    rows = read_csv_rows(frame_csv)
    out: dict[tuple[str, str], dict[str, float | str]] = {}
    for row in rows:
        key = (row["method"], row["frame"])
        out[key] = {}
        for name, value in row.items():
            if name in {"method", "frame", "mask_path", "gt_path"}:
                out[key][name] = value
            else:
                out[key][name] = float(value) if value != "" else ""
    return out


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    pad_ratio: float = 0.10,
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


def draw_contour(
    image: np.ndarray,
    mask: np.ndarray,
    color: np.ndarray,
    thickness: int = 4,
) -> np.ndarray:
    out = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, tuple(int(x) for x in color.tolist()), thickness)
    return out


def gt_panel(image: np.ndarray, gt: np.ndarray) -> np.ndarray:
    out = blend_mask(image, gt, GT_AMBER, 0.24)
    return draw_contour(out, gt, GT_AMBER, 5)


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
    out = blend_mask(out, ~before & after, ADDED_PURPLE, 0.82)
    return out


def format_metric(value: float | str, digits: int = 3) -> str:
    if value == "":
        return "NA"
    return f"{float(value):.{digits}f}"


def set_image_axis(ax: plt.Axes, image: np.ndarray, title: str, frame: str | None = None) -> None:
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, pad=3.0, fontsize=7.3)
    if frame:
        ax.text(
            0.035,
            0.045,
            f"Frame {frame}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.5,
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


def prepare_frame_panels(
    eval_root: Path,
    stem: str,
    method_dirs: list[Path],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], tuple[int, int, int, int]]:
    image = read_rgb(eval_root / "selected_frames" / f"{stem}.jpg")
    gt = read_mask(eval_root / "gt_masks" / f"mask_{stem}.png", image.shape[:2])
    masks = [read_mask(method_dir / f"mask_{stem}.png", image.shape[:2]) for method_dir in method_dirs]
    bbox = expand_bbox(union_bbox([gt, *masks]), image.shape[:2])
    image_c = crop_resize_rgb(image, bbox)
    gt_c = crop_resize_mask(gt, bbox)
    masks_c = [crop_resize_mask(mask, bbox) for mask in masks]
    return image_c, gt_c, masks_c, bbox


def auto_select_difference_frames(
    eval_root: Path,
    method_a: Path,
    method_b: Path,
    n: int = 2,
) -> list[str]:
    scores: list[tuple[float, str]] = []
    for gt_path in sorted((eval_root / "gt_masks").glob("mask_*.png")):
        stem = gt_path.stem.removeprefix("mask_")
        a = read_mask(method_a / f"mask_{stem}.png")
        b = read_mask(method_b / f"mask_{stem}.png", a.shape)
        union = np.logical_or(a, b).sum()
        score = float(np.logical_xor(a, b).sum() / union) if union else 0.0
        scores.append((score, stem))
    scores.sort(reverse=True)
    return [stem for _, stem in scores[:n]]


def export_panel_group(
    panel_root: Path,
    figure: str,
    experiment: str,
    frame: str,
    panels: list[tuple[str, str, np.ndarray]],
    bbox: tuple[int, int, int, int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    frame_dir = panel_root / figure / f"frame_{frame}"
    for order, (panel_key, panel_label, image) in enumerate(panels, start=1):
        stem = frame_dir / f"{frame}_{order:02d}_{panel_key}"
        png_path, tif_path = save_panel_image(image, stem)
        rows.append(
            {
                "figure": figure,
                "experiment": experiment,
                "frame": frame,
                "panel_order": order,
                "panel_key": panel_key,
                "panel_label": panel_label,
                "png_path": str(png_path),
                "tif_path": str(tif_path),
                **bbox_record(bbox),
            }
        )
    return rows


def create_e2_individual_panels(eval_root: Path, panel_root: Path, frames: list[str]) -> list[dict[str, object]]:
    method_root = eval_root / "method_masks"
    p2_dir = method_root / "E2_P2"
    p4_dir = method_root / "E2_P4"
    rows: list[dict[str, object]] = []
    for stem in frames:
        image, gt, masks, bbox = prepare_frame_panels(eval_root, stem, [p2_dir, p4_dir])
        p2, p4 = masks
        panels = [
            ("rgb_gt", "RGB + GT", gt_panel(image, gt)),
            ("p2_mask", "P2 mask", overlay_panel(image, gt, p2, PRED_BLUE)),
            ("p2_error", "P2 error", error_panel(image, gt, p2)),
            ("p4_mask", "P4 mask", overlay_panel(image, gt, p4, FAIL_GRAY)),
            ("p4_error", "P4 error", error_panel(image, gt, p4)),
        ]
        rows.extend(
            export_panel_group(
                panel_root,
                "E2_prompt_P2_vs_P4",
                "experiment_2_prompt_robustness",
                stem,
                panels,
                bbox,
            )
        )
    return rows


def create_e3_individual_panels(eval_root: Path, panel_root: Path, frames: list[str]) -> list[dict[str, object]]:
    method_root = eval_root / "method_masks"
    a0_dir = method_root / "E3_A0_single_P2"
    a2_dir = method_root / "E3_A2_spnp"
    a5_dir = method_root / "E3_A5_full_rap_fsam3"
    rows: list[dict[str, object]] = []
    for stem in frames:
        image, gt, masks, bbox = prepare_frame_panels(eval_root, stem, [a0_dir, a2_dir, a5_dir])
        a0, a2, a5 = masks
        a25 = a5 if np.array_equal(a2, a5) else a2
        panels = [
            ("rgb_gt", "RGB + GT", gt_panel(image, gt)),
            ("a0_mask", "A0 mask", overlay_panel(image, gt, a0, PRED_BLUE)),
            ("a0_error", "A0 error", error_panel(image, gt, a0)),
            ("a2a5_mask", "A2/A5 mask", overlay_panel(image, gt, a25, PRED_GREEN)),
            ("a2a5_error", "A2/A5 error", error_panel(image, gt, a25)),
            ("a2a5_minus_a0", "A2/A5 - A0", difference_panel(image, a0, a25)),
        ]
        rows.extend(
            export_panel_group(
                panel_root,
                "E3_ablation_A0_vs_A2A5",
                "experiment_3_rap_fsam3_ablation",
                stem,
                panels,
                bbox,
            )
        )
    return rows


def create_e2_board(eval_root: Path, figure_dir: Path) -> list[str]:
    metrics = metric_map(eval_root / "E2_prompt_metrics" / "summary_metrics.csv")
    method_root = eval_root / "method_masks"
    p2_dir = method_root / "E2_P2"
    p4_dir = method_root / "E2_P4"
    frames = ["0000", "0025"]
    fig, axes = plt.subplots(
        len(frames),
        5,
        figsize=(183 / 25.4, 104 / 25.4),
        gridspec_kw={"wspace": 0.035, "hspace": 0.12},
    )
    titles = ["RGB + GT", "P2 mask", "P2 error", "P4 mask", "P4 error"]
    for row, stem in enumerate(frames):
        image, gt, masks, _ = prepare_frame_panels(eval_root, stem, [p2_dir, p4_dir])
        p2, p4 = masks
        panels = [
            gt_panel(image, gt),
            overlay_panel(image, gt, p2, PRED_BLUE),
            error_panel(image, gt, p2),
            overlay_panel(image, gt, p4, FAIL_GRAY),
            error_panel(image, gt, p4),
        ]
        for col, panel in enumerate(panels):
            set_image_axis(axes[row, col], panel, titles[col] if row == 0 else "", stem if col == 0 else None)

    fig.suptitle(
        "Experiment 2: prompt robustness on KongQueZhuYu "
        f"(P2 F1={format_metric(metrics['P2']['f1'])}, mIoU={format_metric(metrics['P2']['miou'])}; "
        f"P4 F1={format_metric(metrics['P4']['f1'])})",
        y=0.985,
        fontsize=8.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Overlay: GT contour in amber; prediction fill/contour in method color. "
        "Error: TP green, FP red, FN blue.",
        ha="center",
        va="bottom",
        fontsize=6.4,
    )
    save_figure(fig, figure_dir / "Fig_E2_prompt_P2_vs_P4")
    return frames


def create_e3_board(eval_root: Path, figure_dir: Path) -> list[str]:
    metrics = metric_map(eval_root / "E3_ablation_metrics" / "summary_metrics.csv")
    method_root = eval_root / "method_masks"
    a0_dir = method_root / "E3_A0_single_P2"
    a2_dir = method_root / "E3_A2_spnp"
    a5_dir = method_root / "E3_A5_full_rap_fsam3"
    frames = auto_select_difference_frames(eval_root, a0_dir, a2_dir, n=2)
    fig, axes = plt.subplots(
        len(frames),
        6,
        figsize=(183 / 25.4, 108 / 25.4),
        gridspec_kw={"wspace": 0.035, "hspace": 0.12},
    )
    titles = ["RGB + GT", "A0 mask", "A0 error", "A2/A5 mask", "A2/A5 error", "A2/A5 - A0"]
    for row, stem in enumerate(frames):
        image, gt, masks, _ = prepare_frame_panels(eval_root, stem, [a0_dir, a2_dir, a5_dir])
        a0, a2, a5 = masks
        a25 = a5 if np.array_equal(a2, a5) else a2
        panels = [
            gt_panel(image, gt),
            overlay_panel(image, gt, a0, PRED_BLUE),
            error_panel(image, gt, a0),
            overlay_panel(image, gt, a25, PRED_GREEN),
            error_panel(image, gt, a25),
            difference_panel(image, a0, a25),
        ]
        for col, panel in enumerate(panels):
            set_image_axis(axes[row, col], panel, titles[col] if row == 0 else "", stem if col == 0 else None)

    fig.suptitle(
        "Experiment 3: RAP-FSAM3 ablation "
        f"(A0 F1={format_metric(metrics['A0']['f1'])}, leakage={format_metric(metrics['A0']['leakage_energy'], 4)}; "
        f"A2/A5 F1={format_metric(metrics['A5']['f1'])}, leakage={format_metric(metrics['A5']['leakage_energy'], 4)})",
        y=0.985,
        fontsize=8.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Error: TP green, FP red, FN blue. Difference: shared mask green, removed by A2/A5 orange, added by A2/A5 purple.",
        ha="center",
        va="bottom",
        fontsize=6.4,
    )
    save_figure(fig, figure_dir / "Fig_E3_ablation_A0_vs_A2A5")
    return frames


def export_source_data(
    eval_root: Path,
    output_root: Path,
    e2_frames: list[str],
    e3_frames: list[str],
    panel_rows: list[dict[str, object]],
) -> None:
    source_dir = output_root / "source_data"
    e2_summary = read_csv_rows(eval_root / "E2_prompt_metrics" / "summary_metrics.csv")
    e3_summary = read_csv_rows(eval_root / "E3_ablation_metrics" / "summary_metrics.csv")
    summary_rows = []
    for figure_name, keep_methods, rows in [
        ("Fig_E2_prompt_P2_vs_P4", {"P2", "P4"}, e2_summary),
        ("Fig_E3_ablation_A0_vs_A2A5", {"A0", "A2", "A5"}, e3_summary),
    ]:
        for row in rows:
            if row["method"] in keep_methods:
                summary_rows.append({"figure": figure_name, **row})
    write_csv(source_dir / "figure_metrics_summary.csv", summary_rows, ["figure", *e2_summary[0].keys()])

    selection_rows = []
    for frame in e2_frames:
        selection_rows.append(
            {
                "figure": "Fig_E2_prompt_P2_vs_P4",
                "frame": frame,
                "selection_rule": "fixed representative GT frames",
            }
        )
    for frame in e3_frames:
        selection_rows.append(
            {
                "figure": "Fig_E3_ablation_A0_vs_A2A5",
                "frame": frame,
                "selection_rule": "top-2 A0-vs-A2 mask XOR ratio frames",
            }
        )
    write_csv(source_dir / "selected_frames.csv", selection_rows, ["figure", "frame", "selection_rule"])
    write_csv(
        source_dir / "individual_panel_index.csv",
        panel_rows,
        [
            "figure",
            "experiment",
            "frame",
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


def write_board_note(output_root: Path, e2_frames: list[str], e3_frames: list[str]) -> None:
    note = f"""# 实验二/三可视化图板说明

## 存放内容

- `figures/Fig_E2_prompt_P2_vs_P4.*`：实验二提示词鲁棒性例图，展示 P2 与 P4 的掩膜和误差差异。
- `figures/Fig_E3_ablation_A0_vs_A2A5.*`：实验三模块消融例图，展示 A0 与 A2/A5 的掩膜、误差和差异图。
- `individual_panels/`：两张整体图板拆出的单独干净面板，按图板和帧号分目录存放。
- `source_data/figure_metrics_summary.csv`：两张图板使用的汇总指标。
- `source_data/selected_frames.csv`：入图帧号和选帧规则。
- `source_data/individual_panel_index.csv`：单独面板路径、面板标签和原图裁剪坐标。

## 选帧规则

- 实验二：固定使用代表帧 {", ".join(e2_frames)}，突出 P2 可用、P4 失效的提示词差异。
- 实验三：自动选择 A0 与 A2 掩膜 XOR 比例最高的两帧，本次为 {", ".join(e3_frames)}。

## 图例约定

- 叠加图：GT 轮廓为 amber，预测掩膜为方法色。
- 误差图：TP 为 green，FP 为 red，FN 为 blue。
- A2/A5-A0 差异图：共享区域为 green，被 A2/A5 移除区域为 orange，被 A2/A5 新增区域为 purple。

## 单独面板命名

- `individual_panels/E2_prompt_P2_vs_P4/frame_<帧号>/<帧号>_<序号>_<面板>.png|tif`
- `individual_panels/E3_ablation_A0_vs_A2A5/frame_<帧号>/<帧号>_<序号>_<面板>.png|tif`
- 单独面板不嵌入标题和帧号，便于后续论文排版；具体含义见 `source_data/individual_panel_index.csv`。

## 当前结论

- 实验二支持“提示词会显著影响 FSAM3 前景掩膜质量”：P2 在孔雀竹芋 GT5 上稳定可用，P4 基本失效。
- 实验三支持“RAP-FSAM3 的结构化正负提示模块有效”：A2/A5 相比 A0 提升 F1 与 mIoU，并降低背景泄漏；A3-A5 在当前 GT5 子集上主要体现为保护性流程和记录机制，二值指标与 A2 持平。
"""
    (output_root / "图板说明.md").write_text(note, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_root",
        type=Path,
        default=Path("00-论文优化重构/数据管理/05-评测结果/S21_KongQueZhuYu_E2_E3"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(
            "00-论文优化重构/计算机与电子农业特刊实验工作区/05-图件与论文映射/实验二三_RAP-FSAM3图板"
        ),
    )
    args = parser.parse_args()

    figure_dir = args.output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    e2_frames = create_e2_board(args.eval_root, figure_dir)
    e3_frames = create_e3_board(args.eval_root, figure_dir)
    panel_root = args.output_root / "individual_panels"
    panel_rows = []
    panel_rows.extend(create_e2_individual_panels(args.eval_root, panel_root, e2_frames))
    panel_rows.extend(create_e3_individual_panels(args.eval_root, panel_root, e3_frames))
    export_source_data(args.eval_root, args.output_root, e2_frames, e3_frames, panel_rows)
    write_board_note(args.output_root, e2_frames, e3_frames)
    print(f"Created visual boards under {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
