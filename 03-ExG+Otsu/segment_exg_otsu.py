"""
ExG + Otsu Segmentation Baseline (Optimized with OpenCV)
Traditional agricultural vision: ExG = 2g - r - b, then Otsu threshold.
"""

import os
import time
import json
import numpy as np
import cv2

INPUT_BASE_DIR = r"D:\CAAS\02-FFT"
OUTPUT_BASE_DIR = r"D:\CAAS\03-ExG+Otsu"
CLOSING_KERNEL_SIZE = 15
MIN_REGION_RATIO = 0.001


def segment_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, {"error": "cannot read"}
    h, w = img.shape[:2]
    img_f = img.astype(np.float32)
    B, G, R = img_f[:,:,0], img_f[:,:,1], img_f[:,:,2]
    total = R + G + B + 1e-6
    exg = (2.0 * G - R - B) / total
    emin, emax = exg.min(), exg.max()
    if emax - emin < 1e-6:
        return np.zeros((h, w), dtype=np.uint8), {"otsu_threshold": 0, "raw_coverage_pct": 0, "clean_coverage_pct": 0, "final_regions": 0}
    exg_u8 = ((exg - emin) / (emax - emin) * 255).astype(np.uint8)
    threshold, binary = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    raw_cov = np.count_nonzero(binary) / binary.size * 100
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSING_KERNEL_SIZE, CLOSING_KERNEL_SIZE))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    filled = binary.copy()
    mask_ff = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(filled, mask_ff, (0, 0), 255)
    binary = binary | cv2.bitwise_not(filled)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = np.argmax(areas) + 1
        binary = ((labels == largest) * 255).astype(np.uint8)
    clean_cov = np.count_nonzero(binary) / binary.size * 100
    n_f, _, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return binary, {"otsu_threshold": float(threshold), "raw_coverage_pct": round(raw_cov, 2), "clean_coverage_pct": round(clean_cov, 2), "final_regions": max(n_f - 1, 0)}


def process_folder(folder_name, force=False):
    input_dir = os.path.join(INPUT_BASE_DIR, folder_name)
    output_dir = os.path.join(OUTPUT_BASE_DIR, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    images = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))])
    log_path = os.path.join(output_dir, "segmentation_log.json")
    if not force and os.path.exists(log_path):
        with open(log_path, 'r') as f:
            existing = json.load(f)
        if len(existing) >= len(images):
            print(f"  [SKIP] {folder_name}: done ({len(existing)})")
            return
    print(f"  Processing {folder_name}: {len(images)} images")
    log = []
    t_start = time.time()
    for idx, img_name in enumerate(images):
        img_path = os.path.join(input_dir, img_name)
        t0 = time.time()
        mask, info = segment_image(img_path)
        out_fn = img_name.replace('.jpg', '.png').replace('.JPG', '.png')
        cv2.imwrite(os.path.join(output_dir, f"mask_{out_fn}"), mask)
        orig = cv2.imread(img_path)
        crop = orig.copy()
        crop[mask == 0] = 0
        cv2.imwrite(os.path.join(output_dir, f"crop_{out_fn}"), crop)
        elapsed = time.time() - t0
        info["image"] = img_name
        info["time_sec"] = round(elapsed, 4)
        log.append(info)
        if (idx + 1) % 50 == 0 or idx == 0 or idx == len(images) - 1:
            print(f"    [{idx+1}/{len(images)}] thr={info['otsu_threshold']:.0f} cov={info['raw_coverage_pct']:.1f}%->{info['clean_coverage_pct']:.1f}% t={elapsed:.3f}s")
    total_time = time.time() - t_start
    print(f"  Done {folder_name}: {total_time:.1f}s, {total_time/max(len(images),1):.3f}s/img")
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)


def get_all_folders():
    folders = []
    for name in sorted(os.listdir(INPUT_BASE_DIR)):
        p = os.path.join(INPUT_BASE_DIR, name)
        if os.path.isdir(p):
            if any(f.lower().endswith(('.jpg', '.png')) for f in os.listdir(p)):
                folders.append(name)
    return folders


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    t_global = time.time()
    folders = [args.folder] if args.folder else get_all_folders()
    print(f"ExG+Otsu Segmentation | {len(folders)} folders")
    print(f"Input: {INPUT_BASE_DIR} -> Output: {OUTPUT_BASE_DIR}")
    print("=" * 60)
    for i, folder in enumerate(folders):
        print(f"[{i+1}/{len(folders)}] {folder}")
        process_folder(folder, force=args.force)
    total = time.time() - t_global
    print(f"\nAll done! {len(folders)} folders in {total:.1f}s")
    summary = {"method": "ExG+Otsu", "total_folders": len(folders), "total_time_sec": round(total, 1),
               "parameters": {"closing_kernel_size": CLOSING_KERNEL_SIZE, "min_region_ratio": MIN_REGION_RATIO}}
    with open(os.path.join(OUTPUT_BASE_DIR, "batch_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
