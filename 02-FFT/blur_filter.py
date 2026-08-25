"""
FFT 模糊检测 + 智能过滤：去除模糊帧，同时保证 COLMAP 连续性。

核心策略：
  1. FFT 高频能量比 + Laplacian 方差 双指标检测模糊
  2. 最大连续删除帧数限制（max_consecutive_drop）保证帧间重叠
  3. 被迫保留时，选连续模糊帧中最清晰的一张
  4. 输出筛选后的图片到新文件夹，保持文件名连续

用法：
  python blur_filter.py --input_dir D:/CAAS/01-FFmepg/万年青1 --output_dir D:/CAAS/02-FFT/万年青1
  python blur_filter.py --input_dir D:/CAAS/01-FFmepg  (批量处理所有子文件夹)
"""

import os
import sys
import cv2
import numpy as np
import argparse
import shutil
import json
from pathlib import Path
from datetime import datetime


# ============================================================
# 模糊度评估
# ============================================================

def fft_blur_score(image_gray: np.ndarray) -> float:
    """
    FFT 高频能量比。值越小越模糊。
    将灰度图做 2D FFT，计算高频区域能量占比。
    """
    h, w = image_gray.shape
    f = np.fft.fft2(image_gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # 中心低频掩码：半径为 min(h,w) 的 10%
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * 0.10)
    Y, X = np.ogrid[:h, :w]
    low_freq_mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= radius ** 2

    total_energy = np.sum(magnitude ** 2) + 1e-8
    low_energy = np.sum(magnitude[low_freq_mask] ** 2)
    high_energy_ratio = 1.0 - low_energy / total_energy

    return float(high_energy_ratio)


def laplacian_variance(image_gray: np.ndarray) -> float:
    """
    Laplacian 方差。值越小越模糊。
    经典的模糊检测方法：cv2.Laplacian 后取方差。
    """
    lap = cv2.Laplacian(image_gray, cv2.CV_64F)
    return float(lap.var())


def imread_unicode(image_path: str) -> np.ndarray:
    """用 numpy 读取文件字节再解码，绕过 cv2.imread 不支持中文路径的问题。"""
    with open(image_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def compute_sharpness(image_path: str) -> dict:
    """
    计算一张图片的清晰度分数（两种指标）。
    """
    img = imread_unicode(image_path)
    if img is None:
        return {"fft": 0.0, "laplacian": 0.0, "combined": 0.0}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 缩放到统一大小以加速（保持比例）
    max_dim = 512
    h, w = gray.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    fft_score = fft_blur_score(gray)
    lap_score = laplacian_variance(gray)

    return {
        "fft": fft_score,
        "laplacian": lap_score,
    }


# ============================================================
# 智能过滤：保证 COLMAP 连续性
# ============================================================

def smart_filter(
    scores: list,
    blur_threshold_percentile: float = 25.0,
    max_consecutive_drop: int = 2,
) -> list:
    """
    智能过滤算法。

    参数：
        scores: list of dict, 每帧的 {"index": int, "file": str, "fft": float, "laplacian": float}
        blur_threshold_percentile: 模糊阈值百分位（低于此百分位的认为是模糊）
        max_consecutive_drop: 最大连续删除帧数（保证 COLMAP 重叠）

    返回：
        keep_indices: 保留帧的索引列表

    策略：
        1. 根据两个指标的加权综合分，计算阈值
        2. 低于阈值标记为模糊
        3. 从前到后扫描，如果当前帧模糊：
           - 如果已经连续丢弃了 max_consecutive_drop 帧 → 强制保留当前帧
             （或在连续模糊段中选最清晰的）
           - 否则丢弃
        4. 首尾帧强制保留（COLMAP 需要）
    """
    n = len(scores)
    if n == 0:
        return []

    # 归一化两个指标到 [0, 1] 再加权
    fft_vals = np.array([s["fft"] for s in scores])
    lap_vals = np.array([s["laplacian"] for s in scores])

    # Min-max 归一化
    fft_min, fft_max = fft_vals.min(), fft_vals.max()
    lap_min, lap_max = lap_vals.min(), lap_vals.max()

    fft_norm = (fft_vals - fft_min) / (fft_max - fft_min + 1e-8)
    lap_norm = (lap_vals - lap_min) / (lap_max - lap_min + 1e-8)

    # 综合分：50% FFT + 50% Laplacian
    combined = 0.5 * fft_norm + 0.5 * lap_norm

    # 阈值：取百分位
    threshold = np.percentile(combined, blur_threshold_percentile)

    # 标记模糊帧
    is_blurry = combined < threshold

    # 应用连续删除约束
    keep = []
    consecutive_drops = 0
    drop_buffer = []  # 缓存连续的模糊帧信息

    for i in range(n):
        if i == 0 or i == n - 1:
            # 首尾帧强制保留
            if drop_buffer:
                # 如果之前有缓存的模糊帧，清空（不需额外保留，因为当前帧被保留了）
                drop_buffer = []
                consecutive_drops = 0
            keep.append(i)
            consecutive_drops = 0
            continue

        if not is_blurry[i]:
            # 清晰帧：保留，重置计数
            if drop_buffer:
                drop_buffer = []
                consecutive_drops = 0
            keep.append(i)
            consecutive_drops = 0
        else:
            # 模糊帧
            drop_buffer.append((i, combined[i]))
            consecutive_drops += 1

            if consecutive_drops >= max_consecutive_drop:
                # 连续丢弃达上限，从 buffer 中选最清晰的保留
                best_idx = max(drop_buffer, key=lambda x: x[1])[0]
                keep.append(best_idx)
                drop_buffer = []
                consecutive_drops = 0

    keep.sort()
    return keep, combined, threshold, is_blurry


# ============================================================
# 主处理逻辑
# ============================================================

def process_folder(
    input_dir: str,
    output_dir: str,
    blur_percentile: float = 25.0,
    max_consecutive_drop: int = 2,
    copy_mode: str = "copy",
):
    """
    处理单个文件夹：检测模糊 → 智能过滤 → 输出。
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 获取所有图片，按文件名排序
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in extensions
    ])

    if not image_files:
        print(f"  [跳过] {input_dir} 中没有找到图片")
        return

    total = len(image_files)
    print(f"  共 {total} 张图片，开始清晰度检测...")

    # Step 1: 计算所有图片的清晰度分数
    scores = []
    for idx, img_file in enumerate(image_files):
        s = compute_sharpness(str(img_file))
        s["index"] = idx
        s["file"] = img_file.name
        scores.append(s)
        if (idx + 1) % 50 == 0:
            print(f"    已检测 {idx + 1}/{total}...")

    # Step 2: 智能过滤
    keep_indices, combined_scores, threshold, is_blurry = smart_filter(
        scores,
        blur_threshold_percentile=blur_percentile,
        max_consecutive_drop=max_consecutive_drop,
    )

    removed = total - len(keep_indices)
    blurry_count = int(np.sum(is_blurry))
    print(f"  检测到 {blurry_count} 张模糊帧（阈值百分位: {blur_percentile}%）")
    print(f"  保留 {len(keep_indices)} 张，删除 {removed} 张")
    print(f"  （其中 {blurry_count - removed} 张模糊帧因连续性约束被强制保留）")

    # Step 3: 输出
    output_path.mkdir(parents=True, exist_ok=True)

    for new_idx, orig_idx in enumerate(keep_indices):
        src = str(image_files[orig_idx])
        # 保持连续编号
        ext = image_files[orig_idx].suffix
        dst = str(output_path / f"{new_idx:04d}{ext}")
        if copy_mode == "copy":
            shutil.copy2(src, dst)
        elif copy_mode == "move":
            shutil.move(src, dst)
        elif copy_mode == "symlink":
            os.symlink(src, dst)

    # Step 4: 写入日志
    log_data = {
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "timestamp": datetime.now().isoformat(),
        "total_frames": total,
        "blurry_detected": blurry_count,
        "kept_frames": len(keep_indices),
        "removed_frames": removed,
        "forced_kept": blurry_count - removed,
        "blur_percentile": blur_percentile,
        "max_consecutive_drop": max_consecutive_drop,
        "threshold": float(threshold),
        "per_frame": [
            {
                "file": scores[i]["file"],
                "fft": round(scores[i]["fft"], 6),
                "laplacian": round(scores[i]["laplacian"], 4),
                "combined_norm": round(float(combined_scores[i]), 6),
                "is_blurry": bool(is_blurry[i]),
                "kept": i in set(keep_indices),
            }
            for i in range(total)
        ],
    }

    log_file = output_path / "filter_log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    print(f"  日志已写入: {log_file}")
    return log_data


def main():
    parser = argparse.ArgumentParser(
        description="FFT 模糊检测 + 智能过滤（保 COLMAP 连续性）"
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="输入目录。如果包含子文件夹（每个子文件夹是一组图片），则批量处理"
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="输出目录。默认为 D:\\CAAS\\02-FFT"
    )
    parser.add_argument(
        "--blur_percentile", type=float, default=25.0,
        help="模糊阈值百分位（0-100）。低于此百分位认为模糊。默认 25"
    )
    parser.add_argument(
        "--max_consecutive_drop", type=int, default=2,
        help="最大连续删除帧数。超过则强制保留最清晰帧。默认 2"
    )
    parser.add_argument(
        "--mode", choices=["copy", "move", "symlink"], default="copy",
        help="输出模式：copy(复制)、move(移动)、symlink(软链接)。默认 copy"
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_root = Path(args.output_dir) if args.output_dir else Path(r"D:\CAAS\02-FFT")

    # 判断是单个文件夹还是批量
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    has_images = any(f.suffix.lower() in image_exts for f in input_path.iterdir() if f.is_file())

    if has_images:
        # 单文件夹模式
        out_dir = output_root / input_path.name if args.output_dir is None else output_root
        print(f"\n{'='*60}")
        print(f"处理: {input_path.name}")
        process_folder(
            str(input_path), str(out_dir),
            args.blur_percentile, args.max_consecutive_drop, args.mode
        )
    else:
        # 批量模式：遍历子文件夹
        subdirs = sorted([
            d for d in input_path.iterdir()
            if d.is_dir() and d.name not in {"ffmpeg_bin"}
        ])
        print(f"批量模式：找到 {len(subdirs)} 个子文件夹")

        summary = []
        for subdir in subdirs:
            out_dir = output_root / subdir.name
            print(f"\n{'='*60}")
            print(f"处理: {subdir.name}")
            result = process_folder(
                str(subdir), str(out_dir),
                args.blur_percentile, args.max_consecutive_drop, args.mode
            )
            if result:
                summary.append({
                    "name": subdir.name,
                    "total": result["total_frames"],
                    "kept": result["kept_frames"],
                    "removed": result["removed_frames"],
                })

        # 打印汇总
        print(f"\n{'='*60}")
        print("汇总：")
        print(f"{'文件夹':<15} {'原始':>6} {'保留':>6} {'删除':>6} {'删除率':>8}")
        print("-" * 50)
        total_orig = total_kept = 0
        for s in summary:
            rate = s["removed"] / s["total"] * 100 if s["total"] else 0
            print(f"{s['name']:<15} {s['total']:>6} {s['kept']:>6} {s['removed']:>6} {rate:>7.1f}%")
            total_orig += s["total"]
            total_kept += s["kept"]
        total_rm = total_orig - total_kept
        total_rate = total_rm / total_orig * 100 if total_orig else 0
        print("-" * 50)
        print(f"{'合计':<15} {total_orig:>6} {total_kept:>6} {total_rm:>6} {total_rate:>7.1f}%")


if __name__ == "__main__":
    main()
