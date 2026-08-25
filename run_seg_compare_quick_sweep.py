#!/usr/bin/env python3
"""Quick parameter sweeps (3 configs per method) and preview image generation."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/data/fj")
INPUT_ROOT = ROOT / "02-FFT"
OUTPUT_ROOT = ROOT / "results_seg_compare" / "sweeps_quick"

EXG_PY = ROOT / "exg_otsu" / "venv" / "bin" / "python"
YOLO_PY = ROOT / "yolov8_seg" / "venv" / "bin" / "python"
SAM_PY = ROOT / "segment-anything" / "venv" / "bin" / "python"

EXG_RUN = ROOT / "exg_otsu" / "run_01_exg_otsu.py"
YOLO_RUN = ROOT / "yolov8_seg" / "run_02_yolov8_seg.py"
SAM_RUN = ROOT / "segment-anything" / "run_03_sam1_amg.py"


def run_cmd(args: list[str]) -> None:
    print("[RUN]", " ".join(args))
    subprocess.run(args, check=True)


def read_image_or_blank(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    if path.exists():
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    return np.zeros(shape, dtype=np.uint8)


def make_overlay(original: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    overlay = original.copy()
    mask_bool = mask_u8 > 0
    overlay[mask_bool] = (0.35 * overlay[mask_bool] + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
    return overlay


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def save_method_preview(
    method_root: Path,
    folder: str,
    first_image: Path,
    config_names: list[str],
    out_path: Path,
) -> None:
    original = cv2.imread(str(first_image), cv2.IMREAD_COLOR)
    if original is None:
        return
    h, w = original.shape[:2]

    rows = []
    stem = first_image.stem
    for cfg_name in config_names:
        cfg_dir = method_root / cfg_name / folder
        mask_path = cfg_dir / f"mask_{stem}.png"
        crop_path = cfg_dir / f"crop_{stem}.png"

        mask_u8 = read_image_or_blank(mask_path, (h, w, 3))
        if mask_u8.ndim == 3:
            mask_gray = cv2.cvtColor(mask_u8, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask_u8
        crop = read_image_or_blank(crop_path, original.shape)
        overlay = make_overlay(original, mask_gray)

        panel = np.hstack(
            [
                add_label(original, f"{cfg_name} | original"),
                add_label(overlay, f"{cfg_name} | overlay"),
                add_label(crop, f"{cfg_name} | crop"),
            ]
        )
        rows.append(panel)

    if rows:
        grid = np.vstack(rows)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), grid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick sweeps for 3 segmentation methods.")
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=5, help="Run first N images")
    parser.add_argument("--device-yolo", type=str, default="cpu")
    parser.add_argument("--device-sam", type=str, default="cpu")
    args = parser.parse_args()

    folder_dir = args.input_root / args.folder
    images = sorted([p for p in folder_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        raise FileNotFoundError(f"No images found in {folder_dir}")
    first_image = images[0]

    exg_cfgs = [
        ("exg_cfg1_otsu", ["--threshold-mode", "otsu", "--pp-close-kernel-size", "9"]),
        ("exg_cfg2_p60", ["--threshold-mode", "percentile", "--threshold-percentile", "60", "--pp-close-kernel-size", "7"]),
        ("exg_cfg3_p70", ["--threshold-mode", "percentile", "--threshold-percentile", "70", "--pp-close-kernel-size", "5", "--no-pp-fill-holes"]),
    ]
    yolo_cfgs = [
        ("yolo_cfg1_default", ["--conf", "0.25", "--iou", "0.7", "--green-refine"]),
        ("yolo_cfg2_loose", ["--conf", "0.15", "--iou", "0.6", "--green-refine", "--green-threshold-mode", "percentile", "--green-threshold-percentile", "60"]),
        ("yolo_cfg3_nogreen", ["--conf", "0.25", "--iou", "0.7", "--no-green-refine"]),
    ]
    sam_cfgs = [
        ("sam_cfg1_default", ["--green-refine", "--candidate-select-k", "3"]),
        ("sam_cfg2_focus", ["--green-refine", "--candidate-select-k", "1", "--candidate-target-area-ratio", "0.08", "--score-w-center", "0.35", "--score-w-green", "0.4"]),
        ("sam_cfg3_nogreen", ["--no-green-refine", "--candidate-select-k", "3"]),
    ]

    exg_root = args.output_root / args.folder / "01_exg_otsu"
    yolo_root = args.output_root / args.folder / "02_yolov8_seg"
    sam_root = args.output_root / args.folder / "03_sam1_amg"

    for cfg_name, extra in exg_cfgs:
        out_dir = exg_root / cfg_name
        run_cmd(
            [
                str(EXG_PY),
                str(EXG_RUN),
                "--input-base-dir",
                str(args.input_root),
                "--output-base-dir",
                str(out_dir),
                "--folder",
                args.folder,
                "--limit",
                str(args.limit),
                "--force",
                *extra,
            ]
        )

    for cfg_name, extra in yolo_cfgs:
        out_dir = yolo_root / cfg_name
        run_cmd(
            [
                str(YOLO_PY),
                str(YOLO_RUN),
                "--input-base-dir",
                str(args.input_root),
                "--output-base-dir",
                str(out_dir),
                "--folder",
                args.folder,
                "--limit",
                str(args.limit),
                "--device",
                args.device_yolo,
                "--force",
                *extra,
            ]
        )

    for cfg_name, extra in sam_cfgs:
        out_dir = sam_root / cfg_name
        run_cmd(
            [
                str(SAM_PY),
                str(SAM_RUN),
                "--input-base-dir",
                str(args.input_root),
                "--output-base-dir",
                str(out_dir),
                "--folder",
                args.folder,
                "--limit",
                str(args.limit),
                "--device",
                args.device_sam,
                "--force",
                *extra,
            ]
        )

    save_method_preview(
        method_root=exg_root,
        folder=args.folder,
        first_image=first_image,
        config_names=[name for name, _ in exg_cfgs],
        out_path=args.output_root / args.folder / "preview_exg.png",
    )
    save_method_preview(
        method_root=yolo_root,
        folder=args.folder,
        first_image=first_image,
        config_names=[name for name, _ in yolo_cfgs],
        out_path=args.output_root / args.folder / "preview_yolo.png",
    )
    save_method_preview(
        method_root=sam_root,
        folder=args.folder,
        first_image=first_image,
        config_names=[name for name, _ in sam_cfgs],
        out_path=args.output_root / args.folder / "preview_sam.png",
    )
    print(f"[DONE] Sweep outputs saved under {args.output_root / args.folder}")


if __name__ == "__main__":
    main()

