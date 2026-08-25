#!/usr/bin/env python3
"""Run hierarchical view-quality gates for plant reconstruction."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path("/data/fj/F2DMAS")
REPO_ROOT = ROOT / "2d-gaussian-splatting-main"
COLMAP_ROOT = ROOT / "00-论文优化重构" / "数据管理" / "02-位姿COLMAP" / "03-final_locked"
FFT_ROOT = ROOT / "02-FFT"
MASK_ROOT = ROOT / "03-SAM"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def min_max(values: list[float]) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-8:
        return [1.0 for _ in values]
    return [float((v - lo) / (hi - lo)) for v in arr]


def invert_norm(values: list[float]) -> list[float]:
    return [1.0 - v for v in min_max(values)]


def name_aliases(stem: str) -> list[str]:
    values = [stem]
    for prefix in ("crop_", "mask_"):
        if stem.startswith(prefix):
            values.append(stem[len(prefix) :])
    return values


def select_rows(rows: list[dict], score_key: str, threshold: float, keep_ratio: float) -> set[str]:
    if not rows:
        return set()
    ratio = max(0.0, min(float(keep_ratio), 1.0))
    if ratio < 1.0:
        keep_n = max(1, int(round(len(rows) * ratio)))
        selected = sorted(rows, key=lambda row: row[score_key], reverse=True)[:keep_n]
        cutoff = selected[-1][score_key]
        for row in rows:
            row["accepted"] = row[score_key] >= cutoff
            row["accept_rule"] = f"top_{ratio:.3f}"
        return {row["image_name"] for row in rows if row["accepted"]}

    for row in rows:
        row["accepted"] = row[score_key] >= threshold
        row["accept_rule"] = f"threshold_{threshold:.3f}"
    return {row["image_name"] for row in rows if row["accepted"]}


def exposure_score(gray: np.ndarray) -> tuple[float, float]:
    mean = float(gray.mean() / 255.0)
    score = max(0.0, 1.0 - abs(mean - 0.5) / 0.5)
    return mean, score


def contrast_score(gray: np.ndarray) -> float:
    return float(gray.std() / 64.0)


def entropy_score(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / max(hist.sum(), 1.0)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


def mask_metrics(mask: np.ndarray) -> dict[str, float]:
    mask_bin = (mask > 127).astype(np.uint8)
    h, w = mask_bin.shape
    area_ratio = float(mask_bin.mean())
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    component_area = 0.0
    if num_labels > 1:
        component_area = float(stats[1:, cv2.CC_STAT_AREA].max()) / float(mask_bin.sum() + 1e-6)
    holes = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) - mask_bin
    hole_ratio = float(holes.sum()) / float(mask_bin.sum() + 1e-6)
    boundary = cv2.morphologyEx(mask_bin, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    boundary_ratio = float(boundary.sum()) / float((h * w) + 1e-6)
    return {
        "foreground_ratio": area_ratio,
        "largest_component_ratio": component_area,
        "hole_ratio": hole_ratio,
        "boundary_ratio": boundary_ratio,
    }


def temporal_score(metrics: list[dict[str, float]]) -> list[float]:
    if not metrics:
        return []
    diffs = [0.0]
    for i in range(1, len(metrics)):
        prev = metrics[i - 1]
        cur = metrics[i]
        diff = (
            abs(cur["foreground_ratio"] - prev["foreground_ratio"])
            + abs(cur["largest_component_ratio"] - prev["largest_component_ratio"])
            + abs(cur["hole_ratio"] - prev["hole_ratio"])
        )
        diffs.append(diff)
    return invert_norm(diffs)


def read_fft_log(sample: str) -> dict[str, dict]:
    path = FFT_ROOT / sample / "filter_log.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {Path(item["file"]).stem: item for item in data["per_frame"]}


def read_colmap_stats(sample: str, colmap_dir: Path | None = None) -> dict[str, dict]:
    loader_path = REPO_ROOT / "scene" / "colmap_loader.py"
    spec = importlib.util.spec_from_file_location("hvqg_colmap_loader", loader_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    scene_dir = colmap_dir if colmap_dir is not None else COLMAP_ROOT / sample
    sparse = scene_dir / "sparse" / "0"
    images = module.read_extrinsics_binary(str(sparse / "images.bin"))
    _, _, errors = module.read_points3D_binary(str(sparse / "points3D.bin"))
    point_error_mean = float(np.asarray(errors).mean()) if len(errors) else 0.0
    stats = {}
    for img in images.values():
        registered = int(np.sum(img.point3D_ids >= 0))
        total = int(len(img.point3D_ids))
        inlier_ratio = float(registered / total) if total else 0.0
        stats[Path(img.name).stem] = {
            "registered": 1.0,
            "matched_points": float(registered),
            "inlier_ratio": inlier_ratio,
            "reprojection_error": point_error_mean,
        }
    return stats


def raw_gate(sample: str, image_dir: Path, out_dir: Path, threshold: float, keep_ratio: float) -> set[str]:
    fft_log = read_fft_log(sample)
    rows = []
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = image_path.stem
        gray = np.array(Image.open(image_path).convert("L"))
        exposure_mean, exposure = exposure_score(gray)
        contrast = contrast_score(gray)
        entropy = entropy_score(gray)
        fft_entry = {}
        for alias in name_aliases(stem):
            if alias in fft_log:
                fft_entry = fft_log[alias]
                break
        rows.append(
            {
                "image_name": stem,
                "fft_score": float(fft_entry.get("combined_norm", 0.0)),
                "is_fft_blurry": bool(fft_entry.get("is_blurry", False)),
                "fft_kept": bool(fft_entry.get("kept", True)),
                "exposure_mean": exposure_mean,
                "exposure_score": exposure,
                "contrast_score": contrast,
                "entropy_score": entropy,
            }
        )

    contrast_norm = min_max([r["contrast_score"] for r in rows])
    entropy_norm = min_max([r["entropy_score"] for r in rows])
    for row, c_norm, e_norm in zip(rows, contrast_norm, entropy_norm):
        row["contrast_norm"] = c_norm
        row["entropy_norm"] = e_norm
        row["raw_score"] = float(
            0.4 * row["fft_score"]
            + 0.2 * row["exposure_score"]
            + 0.2 * c_norm
            + 0.2 * e_norm
        )
    keep = select_rows(rows, "raw_score", threshold, keep_ratio)

    out_csv = out_dir / "raw_quality_scores.csv"
    if not rows:
        out_csv.write_text("", encoding="utf-8")
        (out_dir / "raw_gate_retained.txt").write_text("", encoding="utf-8")
        return set()
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "raw_gate_retained.txt").write_text("\n".join(sorted(keep)) + "\n", encoding="utf-8")
    return keep


def mask_gate(
    sample: str,
    retained: set[str],
    out_dir: Path,
    threshold: float,
    keep_ratio: float,
    mask_dir: Path | None = None,
    mask_pattern: str = "mask_{stem}.png",
) -> set[str]:
    mask_root = mask_dir if mask_dir is not None else MASK_ROOT / sample
    rows = []
    for stem in sorted(retained):
        mask_path = mask_root / mask_pattern.format(stem=stem)
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        metrics = mask_metrics(mask)
        rows.append({"image_name": stem, **metrics})

    if not rows:
        (out_dir / "mask_reliability_scores.csv").write_text("", encoding="utf-8")
        (out_dir / "mask_gate_retained.txt").write_text("", encoding="utf-8")
        return set()

    temporal = temporal_score(rows)
    area_pref = [1.0 - abs(r["foreground_ratio"] - 0.35) / 0.35 for r in rows]
    hole_inv = invert_norm([r["hole_ratio"] for r in rows])
    boundary_inv = invert_norm([r["boundary_ratio"] for r in rows])
    for row, area, hole, boundary, temp in zip(rows, area_pref, hole_inv, boundary_inv, temporal):
        row["area_score"] = max(0.0, area)
        row["hole_score"] = hole
        row["boundary_score"] = boundary
        row["temporal_score"] = temp
        row["mask_score"] = float(
            0.35 * row["area_score"]
            + 0.30 * row["largest_component_ratio"]
            + 0.20 * row["hole_score"]
            + 0.15 * row["temporal_score"]
        )
    keep = select_rows(rows, "mask_score", threshold, keep_ratio)

    out_csv = out_dir / "mask_reliability_scores.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "mask_gate_retained.txt").write_text("\n".join(sorted(keep)) + "\n", encoding="utf-8")
    return keep


def geometry_gate(
    sample: str,
    retained: set[str],
    out_dir: Path,
    threshold: float,
    keep_ratio: float,
    colmap_dir: Path | None = None,
) -> set[str]:
    colmap = read_colmap_stats(sample, colmap_dir)
    rows = []
    if not retained:
        (out_dir / "geometry_reliability_scores.csv").write_text("", encoding="utf-8")
        (out_dir / "geometry_gate_retained.txt").write_text("", encoding="utf-8")
        return set()
    matched_norm = min_max([colmap.get(stem, {}).get("matched_points", 0.0) for stem in retained])
    reproj_inv = invert_norm([colmap.get(stem, {}).get("reprojection_error", 1e9) for stem in retained])
    for stem, matched, reproj in zip(sorted(retained), matched_norm, reproj_inv):
        stat = colmap.get(stem, {})
        row = {
            "image_name": stem,
            "registered": float(stat.get("registered", 0.0)),
            "matched_points": float(stat.get("matched_points", 0.0)),
            "matched_norm": matched,
            "inlier_ratio": float(stat.get("inlier_ratio", 0.0)),
            "reprojection_error": float(stat.get("reprojection_error", 0.0)),
            "reprojection_score": reproj,
        }
        row["geo_score"] = float(
            0.35 * row["registered"]
            + 0.25 * row["matched_norm"]
            + 0.25 * row["inlier_ratio"]
            + 0.15 * row["reprojection_score"]
        )
        rows.append(row)

    keep = select_rows(rows, "geo_score", threshold, keep_ratio)
    out_csv = out_dir / "geometry_reliability_scores.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "geometry_gate_retained.txt").write_text("\n".join(sorted(keep)) + "\n", encoding="utf-8")
    return keep


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--mask_dir", default=None)
    parser.add_argument("--mask_pattern", default="mask_{stem}.png")
    parser.add_argument("--colmap_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--raw_threshold", type=float, default=0.45)
    parser.add_argument("--mask_threshold", type=float, default=0.45)
    parser.add_argument("--geo_threshold", type=float, default=0.55)
    parser.add_argument("--raw_keep_ratio", type=float, default=1.0)
    parser.add_argument("--mask_keep_ratio", type=float, default=1.0)
    parser.add_argument("--geo_keep_ratio", type=float, default=1.0)
    args = parser.parse_args()

    sample = args.sample
    image_dir = Path(args.image_dir) if args.image_dir else (COLMAP_ROOT / sample / "images")
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    colmap_dir = Path(args.colmap_dir) if args.colmap_dir else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_keep = raw_gate(sample, image_dir, out_dir, args.raw_threshold, args.raw_keep_ratio)
    mask_keep = mask_gate(
        sample,
        raw_keep,
        out_dir,
        args.mask_threshold,
        args.mask_keep_ratio,
        mask_dir,
        args.mask_pattern,
    )
    geo_keep = geometry_gate(sample, mask_keep, out_dir, args.geo_threshold, args.geo_keep_ratio, colmap_dir)

    report = {
        "sample": sample,
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir) if mask_dir is not None else str(MASK_ROOT / sample),
        "mask_pattern": args.mask_pattern,
        "colmap_dir": str(colmap_dir) if colmap_dir is not None else str(COLMAP_ROOT / sample),
        "raw_kept": len(raw_keep),
        "mask_kept": len(mask_keep),
        "geometry_kept": len(geo_keep),
        "final_keep_ratio_vs_raw_gate": len(geo_keep) / max(len(raw_keep), 1),
        "thresholds": {
            "raw": args.raw_threshold,
            "mask": args.mask_threshold,
            "geometry": args.geo_threshold,
        },
        "keep_ratios": {
            "raw": args.raw_keep_ratio,
            "mask": args.mask_keep_ratio,
            "geometry": args.geo_keep_ratio,
        },
    }
    (out_dir / "hvqg_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
