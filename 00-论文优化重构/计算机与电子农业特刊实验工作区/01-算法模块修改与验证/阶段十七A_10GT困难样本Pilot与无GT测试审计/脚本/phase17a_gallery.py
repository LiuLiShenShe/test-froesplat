#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 17A — Step 9: Qualitative gallery (grids, PNG).

Track A (07_gallery/trackA/):
  For each of the 10 GT frames, a 1×6 row image:
    [image | image+GT overlay | P00 | P10 | P01 | P11]
  (GT overlay = green mask over the raw frame. Variant panels = raw frame
   with the variant's final mask overlaid.)
  Plus one contact sheet stacking all 10 rows.

Track B (07_gallery/trackB/):
  For each TEST sample, choose the frame with the LARGEST mask_change_ratio
  across U-variants (max divergence from U00) and render
    [image | U00 + overlay | U10 + overlay | U01 + overlay | U11 + overlay]
  NO GT is shown (Track B has no GT).
  Plus one contact sheet.

Each variant panel = 3840×2160 raw frame blended with the green final mask.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PHASE17A = Path(__file__).resolve().parent.parent
WORKSPACE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证")
PHASE16 = WORKSPACE / "阶段十六_HardCase_GT构建与锁定"
GT_DIR = PHASE16 / "GT_potted_clean_challenge"
TRACK_A_DIR = PHASE17A / "02_trackA_variants"
TRACK_B_DIR = PHASE17A / "03_trackB_variants"
INPUT_A = PHASE17A / "00_input_trackA"
INPUT_B = PHASE17A / "01_input_trackB"
GALLERY = PHASE17A / "07_gallery"
BEHAVIOR = PHASE17A / "05_behavior"

TA_D = {"P00": "P00_Control", "P10": "P10_A6", "P01": "P01_A7", "P11": "P11_A6A7"}
TB_D = {"U00": "U00_Control", "U10": "U10_A6", "U01": "U01_A7", "U11": "U11_A6A7"}
PANEL_W = 640   # downscale 3840 → 640 (0.1667x) for the gallery


def load_img(path: Path) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    return cv2.imread(str(path))


def load_mask(path: Path) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img > 127


def overlay_green(img: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    vis = img.copy()
    if mask is not None and mask.any():
        green = vis.copy()
        green[mask] = (0, 220, 0)
        vis = cv2.addWeighted(vis, 0.6, green, 0.4, 0)
    return vis


def panel(img: np.ndarray) -> np.ndarray:
    """Uniform downscale to a fixed width, preserving aspect."""
    scale = PANEL_W / img.shape[1]
    return cv2.resize(img, (PANEL_W, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def stack_h(imgs: list[np.ndarray]) -> np.ndarray:
    assert imgs, "empty panel list"
    h = max(i.shape[0] for i in imgs)
    out = []
    for i in imgs:
        if i.shape[0] < h:
            pad = np.zeros((h - i.shape[0], i.shape[1], 3), dtype=np.uint8)
            i = np.vstack([i, pad])
        out.append(i)
    return np.hstack(out)


def main() -> None:
    (GALLERY / "trackA").mkdir(parents=True, exist_ok=True)
    (GALLERY / "trackB").mkdir(parents=True, exist_ok=True)

    # ---------- Track A ----------
    a_rows: list[np.ndarray] = []
    gt_frames = [("BaiZhang", "0033"), ("BaiZhang", "0034"), ("BaiZhang", "0035"),
                 ("BaiZhang", "0036"), ("BaiZhang", "0037"),
                 ("DouBanLv2", "0037"), ("DouBanLv2", "0038"), ("DouBanLv2", "0039"),
                 ("DouBanLv2", "0040"), ("DouBanLv2", "0041")]
    for sample, frame in gt_frames:
        stem = f"{sample}_{frame}"
        raw = load_img(INPUT_A / f"{stem}.jpg")
        if raw is None:
            print(f"  ⚠ raw missing: {stem}")
            continue
        gt = load_mask(GT_DIR / sample / f"mask_potted_clean_{frame}.png")
        panels = [panel(raw), panel(overlay_green(raw, gt))]
        for v, vd in TA_D.items():
            vmask = load_mask(TRACK_A_DIR / vd / "最终掩膜" / f"mask_{stem}.png")
            panels.append(panel(overlay_green(raw, vmask)))
        row = stack_h(panels)
        out_path = GALLERY / "trackA" / f"trackA_{stem}.jpg"
        cv2.imwrite(str(out_path), row)
        a_rows.append(row)
        print(f"  trackA {stem} → {out_path.name}")

    if a_rows:
        sheet = stack_h(a_rows)
        cv2.imwrite(str(GALLERY / "trackA" / "trackA_contact_sheet.jpg"), sheet)
        print(f"  trackA contact sheet → 07_gallery/trackA/trackA_contact_sheet.jpg")

    # ---------- Track B ----------
    tbd = {}
    if (BEHAVIOR / "trackB_behavior.csv").exists():
        import csv
        with (BEHAVIOR / "trackB_behavior.csv").open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["variant"] == "U00":
                    continue
                key = (r["sample"], r["frame"])
                try:
                    ratio = float(r["mask_change_ratio"])
                except (ValueError, TypeError):
                    continue
                if key not in tbd or ratio > tbd[key][0]:
                    tbd[key] = (ratio, r["variant"])

    b_rows: list[np.ndarray] = []
    by_sample: dict[str, tuple[str, float]] = {}
    for (sample, frame), (ratio, _) in sorted(tbd.items()):
        cur = by_sample.get(sample)
        if cur is None or ratio > cur[1]:
            by_sample[sample] = (frame, ratio)

    for sample in sorted(by_sample):
        frame, ratio = by_sample[sample]
        # frame column already contains the full stem "sample_frame"
        stem = frame
        raw = load_img(INPUT_B / f"{stem}.jpg")
        if raw is None:
            continue
        panels = [panel(raw)]
        for v, vd in TB_D.items():
            vmask = load_mask(TRACK_B_DIR / vd / "最终掩膜" / f"mask_{stem}.png")
            panels.append(panel(overlay_green(raw, vmask)))
        row = stack_h(panels)
        out_path = GALLERY / "trackB" / f"trackB_{stem}.jpg"
        cv2.imwrite(str(out_path), row)
        b_rows.append(row)
        print(f"  trackB {stem} (max_div ratio={ratio:.5f}) → {out_path.name}")

    if b_rows:
        sheet = stack_h(b_rows)
        cv2.imwrite(str(GALLERY / "trackB" / "trackB_contact_sheet.jpg"), sheet)
        print(f"  trackB contact sheet → 07_gallery/trackB/trackB_contact_sheet.jpg")


if __name__ == "__main__":
    main()