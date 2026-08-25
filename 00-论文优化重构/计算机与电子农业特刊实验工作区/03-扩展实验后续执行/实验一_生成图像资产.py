#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate visual assets for experiment-one S23 segmentation benchmark.

Outputs:
- Per-frame, per-method prediction overlays.
- Per-frame, per-method error maps.
- A representative failure-case panel.
- A metric bar figure for the completed S23 methods.
- Source-data copies and a paper-ready summary table under the paper workspace.

The script uses only csv + Pillow + OpenCV + NumPy so it can run in the current
SAM3 environment without adding plotting dependencies.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
PAPER_ROOT = WORKSPACE_DIR.parent
DATA_ROOT = PAPER_ROOT / "数据管理"
BENCH_ROOT = DATA_ROOT / "05-评测结果/S23_Experiment1_VFM_Benchmark"
VIS_ROOT = WORKSPACE_DIR / "05-图件与论文映射/实验一_视觉基础模型横向对比"
SOURCE_DATA_ROOT = VIS_ROOT / "source_data"


METHOD_ORDER = [
    "RAP-FSAM3-v2",
    "UNet_fewshot_seqcv",
    "DeepLabV3PlusLite_fewshot_seqcv",
    "SAM3_P2",
    "SAM2_oracle_box",
    "CLIPSeg_P2",
    "GroundedSAM1_Plant",
    "Florence2_RES_P2",
    "GroundedSAM2_Plant",
]

DISPLAY_NAME = {
    "Florence2_RES_P2": "Florence-2",
    "CLIPSeg_P2": "CLIPSeg",
    "SAM2_oracle_box": "SAM2 oracle",
    "SAM3_P2": "SAM3 P2",
    "GroundedSAM1_Plant": "Grounded-SAM",
    "GroundedSAM2_Plant": "Grounded-SAM2",
    "RAP-FSAM3-v2": "RAP-FSAM3-v2",
    "UNet_fewshot_seqcv": "U-Net few-shot",
    "DeepLabV3PlusLite_fewshot_seqcv": "DeepLabv3+ lite",
}

METHOD_COLORS = {
    "Florence2_RES_P2": (224, 122, 95),
    "CLIPSeg_P2": (242, 177, 52),
    "SAM2_oracle_box": (156, 106, 222),
    "SAM3_P2": (47, 109, 181),
    "GroundedSAM1_Plant": (196, 78, 82),
    "GroundedSAM2_Plant": (162, 62, 72),
    "RAP-FSAM3-v2": (78, 159, 80),
    "UNet_fewshot_seqcv": (66, 150, 164),
    "DeepLabV3PlusLite_fewshot_seqcv": (84, 126, 203),
}

GT_COLOR = (0, 214, 201)
FP_COLOR = (222, 77, 77)
FN_COLOR = (62, 126, 214)
TP_COLOR = (75, 170, 90)
TEXT_COLOR = (34, 34, 34)
GRID_COLOR = (216, 216, 216)


@dataclass(frozen=True)
class FrameItem:
    sample: str
    frame: str
    stem: str
    image: Path
    gt_mask: Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_SMALL = font(18)
FONT_MED = font(24)
FONT_MED_BOLD = font(24, bold=True)
FONT_BIG = font(30, bold=True)


def load_frames() -> list[FrameItem]:
    rows = read_csv(BENCH_ROOT / "manifest.csv")
    frames: list[FrameItem] = []
    for row in rows:
        stem = f"{row['sample']}_{row['frame']}"
        frames.append(
            FrameItem(
                sample=row["sample"],
                frame=row["frame"],
                stem=stem,
                image=Path(row["image"]),
                gt_mask=Path(row["gt_mask"]),
            )
        )
    return frames


def load_summary() -> dict[str, dict[str, str]]:
    rows = read_csv(BENCH_ROOT / "metrics/summary_metrics.csv")
    return {row["method"]: row for row in rows}


def load_frame_metrics() -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(BENCH_ROOT / "metrics/frame_metrics.csv")
    return {(row["method"], row["frame"]): row for row in rows}


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if shape is not None and img.shape != shape:
        img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return img > 127


def resize_rgb(rgb: np.ndarray, width: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if w == width:
        return rgb
    height = max(1, int(round(h * width / w)))
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def resize_mask(mask: np.ndarray, width: int) -> np.ndarray:
    h, w = mask.shape[:2]
    if w == width:
        return mask
    height = max(1, int(round(h * width / w)))
    out = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return out > 0


def contours(mask: np.ndarray, thickness: int = 4) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    found, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, found, -1, (255, 255, 255), thickness)
    return out > 0


def blend_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32)
    out[mask] = out[mask] * (1.0 - alpha) + color_arr * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_contour(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int = 4) -> np.ndarray:
    out = rgb.copy()
    found, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr, found, -1, color[::-1], thickness)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def add_label(rgb: np.ndarray, title: str, subtitle: str = "") -> Image.Image:
    img = Image.fromarray(rgb)
    pad = 16
    label_h = 72 if subtitle else 48
    canvas = Image.new("RGB", (img.width, img.height + label_h), (255, 255, 255))
    canvas.paste(img, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 10), title, fill=TEXT_COLOR, font=FONT_MED_BOLD)
    if subtitle:
        draw.text((pad, 40), subtitle, fill=(88, 88, 88), font=FONT_SMALL)
    return canvas


def save_image(img: Image.Image, path: Path, tiff: bool = False, pdf: bool = False) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = [path]
    img.save(path, dpi=(600, 600))
    if tiff:
        tif_path = path.with_suffix(".tif")
        img.save(tif_path, dpi=(600, 600), compression="tiff_lzw")
        written.append(tif_path)
    if pdf:
        pdf_path = path.with_suffix(".pdf")
        img.save(pdf_path, resolution=600.0)
        written.append(pdf_path)
    return written


def make_overlay(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray, method: str) -> np.ndarray:
    out = blend_mask(rgb, pred, METHOD_COLORS[method], 0.42)
    out = draw_contour(out, gt, GT_COLOR, thickness=5)
    out = draw_contour(out, pred, METHOD_COLORS[method], thickness=3)
    return out


def make_gt_overlay(rgb: np.ndarray, gt: np.ndarray) -> np.ndarray:
    out = blend_mask(rgb, gt, (92, 181, 125), 0.28)
    return draw_contour(out, gt, GT_COLOR, thickness=5)


def make_error_map(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    base = (rgb.astype(np.float32) * 0.42 + 255 * 0.58).astype(np.uint8)
    tp = np.logical_and(gt, pred)
    fp = np.logical_and(~gt, pred)
    fn = np.logical_and(gt, ~pred)
    out = blend_mask(base, tp, TP_COLOR, 0.38)
    out = blend_mask(out, fp, FP_COLOR, 0.78)
    out = blend_mask(out, fn, FN_COLOR, 0.78)
    out = draw_contour(out, gt, GT_COLOR, thickness=3)
    return out


def metric_subtitle(metric: dict[str, str]) -> str:
    return (
        f"F1 {float(metric['f1']):.3f} | mIoU {float(metric['miou']):.3f} | "
        f"leak {float(metric['leakage_energy']):.4f}"
    )


def paper_table_rows(summary: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        row = summary[method]
        rows.append(
            {
                "method": DISPLAY_NAME[method],
                "method_dir": method,
                "F1": f"{float(row['f1']):.4f}",
                "mIoU": f"{float(row['miou']):.4f}",
                "HD95_px": f"{float(row['hd95_px']):.2f}",
                "leakage_energy": f"{float(row['leakage_energy']):.6f}",
                "precision": f"{float(row['precision']):.4f}",
                "recall": f"{float(row['recall']):.4f}",
                "boundary_F1": f"{float(row['boundary_f1']):.4f}",
                "component_count_mean": f"{float(row['component_count_mean']):.2f}",
            }
        )
    return rows


def archive_source_data() -> list[dict[str, object]]:
    SOURCE_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    source_files = [
        BENCH_ROOT / "manifest.csv",
        BENCH_ROOT / "metrics/summary_metrics.csv",
        BENCH_ROOT / "metrics/summary_metrics.json",
        BENCH_ROOT / "metrics/frame_metrics.csv",
    ]
    rows: list[dict[str, object]] = []
    for src in source_files:
        dst = SOURCE_DATA_ROOT / src.name
        shutil.copy2(src, dst)
        rows.append({"asset_type": "source_data", "method": "all", "frame": "all", "path": str(dst)})

    table_path = SOURCE_DATA_ROOT / "paper_summary_metrics.csv"
    write_csv(
        table_path,
        [
            "method",
            "method_dir",
            "F1",
            "mIoU",
            "HD95_px",
            "leakage_energy",
            "precision",
            "recall",
            "boundary_F1",
            "component_count_mean",
        ],
        paper_table_rows(load_summary()),
    )
    rows.append({"asset_type": "paper_table", "method": "all", "frame": "summary", "path": str(table_path)})
    return rows


def generate_per_frame_assets(panel_width: int) -> list[dict[str, object]]:
    frames = load_frames()
    metrics = load_frame_metrics()
    rows: list[dict[str, object]] = []
    for frame in frames:
        rgb_full = read_rgb(frame.image)
        gt_full = read_mask(frame.gt_mask, rgb_full.shape[:2])
        rgb = resize_rgb(rgb_full, panel_width)
        gt = resize_mask(gt_full, panel_width)
        gt_img = add_label(make_gt_overlay(rgb, gt), f"{frame.stem} | GT")
        gt_path = VIS_ROOT / "gt_overlays" / f"{frame.stem}_gt_overlay.png"
        save_image(gt_img, gt_path)
        rows.append({"asset_type": "gt_overlay", "method": "GT", "frame": frame.stem, "path": str(gt_path)})

        for method in METHOD_ORDER:
            pred_path = BENCH_ROOT / "method_masks" / method / f"mask_{frame.stem}.png"
            pred_full = read_mask(pred_path, rgb_full.shape[:2])
            pred = resize_mask(pred_full, panel_width)
            metric = metrics[(method, frame.stem)]
            label = DISPLAY_NAME[method]
            subtitle = metric_subtitle(metric)

            overlay = add_label(make_overlay(rgb, gt, pred, method), label, subtitle)
            overlay_path = VIS_ROOT / "overlays" / method / f"{frame.stem}_{method}_overlay.png"
            save_image(overlay, overlay_path)
            rows.append({"asset_type": "overlay", "method": method, "frame": frame.stem, "path": str(overlay_path)})

            error = add_label(make_error_map(rgb, gt, pred), f"{label} error", "red FP | blue FN | green TP")
            error_path = VIS_ROOT / "error_maps" / method / f"{frame.stem}_{method}_error.png"
            save_image(error, error_path)
            rows.append({"asset_type": "error_map", "method": method, "frame": frame.stem, "path": str(error_path)})
    return rows


def make_contact_sheet(images: list[Image.Image], cols: int, gap: int = 14, bg: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    if not images:
        raise ValueError("No images for contact sheet")
    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)
    rows = math.ceil(len(images) / cols)
    canvas = Image.new("RGB", (cols * cell_w + (cols + 1) * gap, rows * cell_h + (rows + 1) * gap), bg)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = gap + c * (cell_w + gap) + (cell_w - img.width) // 2
        y = gap + r * (cell_h + gap) + (cell_h - img.height) // 2
        canvas.paste(img, (x, y))
    return canvas


def generate_frame_contact_sheets(panel_width: int) -> list[dict[str, object]]:
    frames = load_frames()
    metrics = load_frame_metrics()
    rows: list[dict[str, object]] = []
    for frame in frames:
        rgb_full = read_rgb(frame.image)
        gt_full = read_mask(frame.gt_mask, rgb_full.shape[:2])
        rgb = resize_rgb(rgb_full, panel_width)
        gt = resize_mask(gt_full, panel_width)
        panels = [add_label(make_gt_overlay(rgb, gt), "Image + GT", frame.stem)]
        for method in METHOD_ORDER:
            pred = resize_mask(read_mask(BENCH_ROOT / "method_masks" / method / f"mask_{frame.stem}.png", rgb_full.shape[:2]), panel_width)
            panels.append(add_label(make_overlay(rgb, gt, pred, method), DISPLAY_NAME[method], metric_subtitle(metrics[(method, frame.stem)])))
        sheet = make_contact_sheet(panels, cols=4)
        path = VIS_ROOT / "contact_sheets" / f"{frame.stem}_method_overlay_grid.png"
        written = save_image(sheet, path, tiff=True)
        rows.extend(
            {
                "asset_type": "contact_sheet" if item == path else f"contact_sheet_{item.suffix.lstrip('.')}",
                "method": "all",
                "frame": frame.stem,
                "path": str(item),
            }
            for item in written
        )
    return rows


def generate_failure_panel(panel_width: int) -> list[dict[str, object]]:
    cases = [
        ("KongQueZhuYu_0050", "Florence2_RES_P2", "Severe miss in one viewpoint"),
        ("KongQueZhuYu_0000", "GroundedSAM1_Plant", "Detector boxes leak into background"),
        ("XianKeLai1_0000", "SAM2_oracle_box", "Oracle box can still over-segment fine structures"),
    ]
    frame_map = {f.stem: f for f in load_frames()}
    metrics = load_frame_metrics()
    row_panels: list[Image.Image] = []
    for stem, method, reason in cases:
        frame = frame_map[stem]
        rgb_full = read_rgb(frame.image)
        gt_full = read_mask(frame.gt_mask, rgb_full.shape[:2])
        pred_full = read_mask(BENCH_ROOT / "method_masks" / method / f"mask_{stem}.png", rgb_full.shape[:2])
        rap_full = read_mask(BENCH_ROOT / "method_masks/RAP-FSAM3-v2" / f"mask_{stem}.png", rgb_full.shape[:2])
        rgb = resize_rgb(rgb_full, panel_width)
        gt = resize_mask(gt_full, panel_width)
        pred = resize_mask(pred_full, panel_width)
        rap = resize_mask(rap_full, panel_width)
        fail_metric = metrics[(method, stem)]
        rap_metric = metrics[("RAP-FSAM3-v2", stem)]
        panels = [
            add_label(make_gt_overlay(rgb, gt), f"{stem}", reason),
            add_label(make_overlay(rgb, gt, pred, method), DISPLAY_NAME[method], metric_subtitle(fail_metric)),
            add_label(make_error_map(rgb, gt, pred), "Failure error map", "red FP | blue FN"),
            add_label(make_overlay(rgb, gt, rap, "RAP-FSAM3-v2"), "RAP-FSAM3-v2", metric_subtitle(rap_metric)),
        ]
        row_panels.extend(panels)
    sheet = make_contact_sheet(row_panels, cols=4)
    path = VIS_ROOT / "figures" / "Fig_E1_failure_cases.png"
    written = save_image(sheet, path, tiff=True, pdf=True)
    return [
        {
            "asset_type": "failure_panel" if item == path else f"failure_panel_{item.suffix.lstrip('.')}",
            "method": "selected",
            "frame": "selected",
            "path": str(item),
        }
        for item in written
    ]


def draw_horizontal_metric_panel(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    values: list[tuple[str, float]],
    max_value: float,
    lower_better: bool,
) -> None:
    draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 255), outline=(226, 226, 226), width=1)
    draw.text((x + 16, y + 12), title, fill=TEXT_COLOR, font=FONT_MED_BOLD)
    draw.text((x + 16, y + 42), "lower better" if lower_better else "higher better", fill=(100, 100, 100), font=FONT_SMALL)
    label_w = 150
    bar_x = x + label_w + 22
    bar_w = w - label_w - 92
    top = y + 80
    step = max(30, (h - 100) // len(values))
    for idx, (method, value) in enumerate(values):
        yy = top + idx * step
        name = DISPLAY_NAME[method]
        color = METHOD_COLORS[method]
        draw.text((x + 16, yy - 3), name, fill=TEXT_COLOR, font=FONT_SMALL)
        draw.line([bar_x, yy + 10, bar_x + bar_w, yy + 10], fill=GRID_COLOR, width=1)
        bw = 0 if max_value <= 0 else int(round(bar_w * value / max_value))
        draw.rectangle([bar_x, yy, bar_x + bw, yy + 20], fill=color)
        label = f"{value:.4f}" if value < 0.1 else f"{value:.3f}"
        draw.text((bar_x + bar_w + 10, yy - 2), label, fill=TEXT_COLOR, font=FONT_SMALL)


def svg_metric_bars(summary: dict[str, dict[str, str]], path: Path, width: int = 1700, height: int = 1160) -> None:
    panels = [
        ("F1", "f1", 1.0, False),
        ("mIoU", "miou", 1.0, False),
        ("HD95 px", "hd95_px", max(float(summary[m]["hd95_px"]) for m in METHOD_ORDER) * 1.05, True),
        ("Leakage energy", "leakage_energy", max(float(summary[m]["leakage_energy"]) for m in METHOD_ORDER) * 1.10, True),
    ]
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.small{font-size:18px}.title{font-size:24px;font-weight:700}.note{font-size:17px;fill:#666}</style>',
    ]
    panel_w, panel_h = 800, 520
    positions = [(50, 40), (890, 40), (50, 610), (890, 610)]
    for (title, key, maxv, lower), (x, y) in zip(panels, positions):
        chunks.append(f'<rect x="{x}" y="{y}" width="{panel_w}" height="{panel_h}" fill="white" stroke="#e3e3e3"/>')
        chunks.append(f'<text class="title" x="{x+16}" y="{y+34}">{title}</text>')
        chunks.append(f'<text class="note" x="{x+16}" y="{y+62}">{"lower better" if lower else "higher better"}</text>')
        label_w, bar_x, bar_w = 150, x + 172, panel_w - 242
        top, step = y + 100, 60
        for idx, method in enumerate(METHOD_ORDER):
            value = float(summary[method][key])
            yy = top + idx * step
            color = "#%02x%02x%02x" % METHOD_COLORS[method]
            bw = 0 if maxv <= 0 else int(round(bar_w * value / maxv))
            label = f"{value:.4f}" if value < 0.1 else f"{value:.3f}"
            chunks.append(f'<text class="small" x="{x+16}" y="{yy+17}">{DISPLAY_NAME[method]}</text>')
            chunks.append(f'<line x1="{bar_x}" y1="{yy+11}" x2="{bar_x+bar_w}" y2="{yy+11}" stroke="#d8d8d8"/>')
            chunks.append(f'<rect x="{bar_x}" y="{yy}" width="{bw}" height="22" fill="{color}"/>')
            chunks.append(f'<text class="small" x="{bar_x+bar_w+10}" y="{yy+18}">{label}</text>')
    chunks.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(chunks), encoding="utf-8")


def generate_metric_bars() -> list[dict[str, object]]:
    summary = load_summary()
    width, height = 1700, 1160
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    panels = [
        ("F1", "f1", 1.0, False),
        ("mIoU", "miou", 1.0, False),
        ("HD95 px", "hd95_px", max(float(summary[m]["hd95_px"]) for m in METHOD_ORDER) * 1.05, True),
        ("Leakage energy", "leakage_energy", max(float(summary[m]["leakage_energy"]) for m in METHOD_ORDER) * 1.10, True),
    ]
    positions = [(50, 40), (890, 40), (50, 610), (890, 610)]
    for (title, key, maxv, lower), (x, y) in zip(panels, positions):
        values = [(method, float(summary[method][key])) for method in METHOD_ORDER]
        draw_horizontal_metric_panel(draw, x, y, 800, 520, title, values, maxv, lower)
    out = VIS_ROOT / "figures" / "Fig_E1_metric_bars.png"
    written = save_image(img, out, tiff=True, pdf=True)
    svg_metric_bars(summary, out.with_suffix(".svg"))
    rows = [
        {
            "asset_type": "metric_bars" if item == out else f"metric_bars_{item.suffix.lstrip('.')}",
            "method": "all",
            "frame": "summary",
            "path": str(item),
        }
        for item in written
    ]
    rows.append({"asset_type": "metric_bars_svg", "method": "all", "frame": "summary", "path": str(out.with_suffix(".svg"))})
    return rows


def markdown_table(summary: dict[str, dict[str, str]]) -> str:
    lines = [
        "| Method | F1 | mIoU | HD95 px | Leakage energy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in paper_table_rows(summary):
        lines.append(
            f"| {row['method']} | {row['F1']} | {row['mIoU']} | "
            f"{row['HD95_px']} | {row['leakage_energy']} |"
        )
    return "\n".join(lines)


def write_readme(asset_rows: list[dict[str, object]]) -> None:
    summary = load_summary()
    frame_count = len(load_frames())
    overlay_count = sum(1 for row in asset_rows if row["asset_type"] == "overlay")
    error_count = sum(1 for row in asset_rows if row["asset_type"] == "error_map")
    readme = f"""# 实验一 视觉基础模型横向对比图像资产

## 存放内容

- `figures/Fig_E1_failure_cases.*`：代表性失败案例图板，包含原图/GT、失败方法、错误热区和 RAP-FSAM3-v2 对照。
- `figures/Fig_E1_metric_bars.*`：F1、mIoU、HD95 和泄漏能量四指标横向柱状图。
- `contact_sheets/`：{frame_count} 帧的原图+GT+各方法 overlay 汇总图。
- `overlays/`：每帧每方法预测掩膜叠加图，共 {overlay_count} 张。
- `error_maps/`：每帧每方法错误热区图，共 {error_count} 张。红色为 FP，蓝色为 FN，绿色为 TP。
- `gt_overlays/`：每帧 GT 叠加图。
- `source_data/`：生成图件所用 `manifest.csv`、`summary_metrics.csv/json`、`frame_metrics.csv` 和论文表格 `paper_summary_metrics.csv`。
- `figure_asset_index.csv`：本目录全部图像和数据资产索引。

## 当前核心结果

{markdown_table(summary)}

## 选帧与图件逻辑

- 评测口径：同一 6 帧 GT 子集，包含 `KongQueZhuYu` GT5 和 `XianKeLai1` GT1。
- 主张：RAP-FSAM3-v2 在同口径横向分割表中取得最高 F1/mIoU，并明显压低 HD95 与泄漏能量。
- 失败案例图板突出三类外部模型风险：Florence-2 局部严重漏检、Grounded-SAM 检测框带入背景和花盆、SAM2 oracle 框仍可能过分割细结构。
- 论文使用时建议优先引用 `figures/` 中的总图，再按需要从 `contact_sheets/` 或 `overlays/` 抽取单帧面板。
"""
    (VIS_ROOT / "图板说明.md").write_text(readme, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate S23 visual assets for experiment one.")
    parser.add_argument("--panel-width", type=int, default=900)
    parser.add_argument("--sheet-width", type=int, default=520)
    args = parser.parse_args()

    asset_rows: list[dict[str, object]] = []
    asset_rows.extend(archive_source_data())
    asset_rows.extend(generate_per_frame_assets(args.panel_width))
    asset_rows.extend(generate_frame_contact_sheets(args.sheet_width))
    asset_rows.extend(generate_failure_panel(args.sheet_width))
    asset_rows.extend(generate_metric_bars())
    write_readme(asset_rows)

    write_csv(
        VIS_ROOT / "figure_asset_index.csv",
        ["asset_type", "method", "frame", "path"],
        asset_rows,
    )
    print(f"Wrote {len(asset_rows)} visual assets under {VIS_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
