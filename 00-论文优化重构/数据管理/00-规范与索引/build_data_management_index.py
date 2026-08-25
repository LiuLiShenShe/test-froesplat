#!/usr/bin/env python3
"""Build the new Plant-aware 2DGS data management index.

The script centralizes existing project resources under
``00-论文优化重构/数据管理`` using symlinks. It does not modify or delete the
legacy data folders.
"""

from __future__ import annotations

import csv
import json
import struct
from pathlib import Path


ROOT = Path("/data/fj/F2DMAS")
DATA_ROOT = ROOT / "00-论文优化重构" / "数据管理"

SAMPLES = [
    "BaiZhang",
    "CaoMei1",
    "CaoMei2",
    "ChangShouHua1",
    "ChangShouHua2",
    "ChangShouHua3",
    "DouBanLv1",
    "DouBanLv2",
    "DouBanLv3",
    "HongZhang",
    "KongQueZhuYu",
    "WanNianQing1",
    "WanNianQing2",
    "WangWenCao1",
    "WangWenCao2",
    "XianKeLai1",
    "XianKeLai2",
    "XianKeLai3",
    "XiangPiShu1",
    "XiangPiShu2",
]

GT_SAMPLES = {
    "CaoMei1",
    "ChangShouHua2",
    "DouBanLv1",
    "KongQueZhuYu",
    "XianKeLai1",
}


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return len([
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])


def read_count_from_bin(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        data = f.read(8)
    if len(data) != 8:
        return 0
    return struct.unpack("<Q", data)[0]


def colmap_stats(scene_path: Path) -> dict[str, object]:
    sparse0 = scene_path / "sparse" / "0"
    registered = read_count_from_bin(sparse0 / "images.bin")
    points3d = read_count_from_bin(sparse0 / "points3D.bin")
    images = count_images(scene_path / "input") or count_images(scene_path / "images")
    rate = round(registered / images * 100, 1) if images else 0.0
    if registered == 0 or points3d == 0:
        status = "missing_or_fail"
    elif rate >= 70.0:
        status = "ok"
    elif rate >= 30.0:
        status = "warn"
    else:
        status = "fail"
    return {
        "status": status,
        "input_images": images,
        "registered": registered,
        "registration_rate": rate,
        "points3d": points3d,
    }


def relink(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        target.unlink()
    target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def load_rerun_report() -> dict[str, object]:
    path = ROOT / "04-COLMAP-rerun-original" / "rerun_report.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def best_rerun_path(sample: str, rerun_report: dict[str, object]) -> Path | None:
    entry = rerun_report.get(sample)
    if not isinstance(entry, dict):
        return None
    best = entry.get("best")
    if not isinstance(best, dict) or best.get("status") != "success":
        return None
    matcher = str(best.get("matcher", ""))
    if not matcher:
        return None
    path = ROOT / "04-COLMAP-rerun-original" / sample / matcher
    return path if path.exists() else None


def main() -> None:
    index_dir = DATA_ROOT / "00-规范与索引"
    index_dir.mkdir(parents=True, exist_ok=True)
    rerun_report = load_rerun_report()

    relink(ROOT / "08-Check", DATA_ROOT / "05-评测结果" / "01-existing_metrics" / "08-Check")
    relink(
        ROOT / "03-GT" / "three_group_metrics_20260303_172956",
        DATA_ROOT / "05-评测结果" / "01-existing_metrics" / "three_group_metrics_20260303_172956",
    )
    relink(
        ROOT / "04-COLMAP-rerun-original" / "rerun_report.json",
        DATA_ROOT / "02-位姿COLMAP" / "rerun_report.json",
    )

    rows: list[dict[str, object]] = []
    for sample in SAMPLES:
        raw_path = ROOT / "01-FFmepg" / sample
        fft_path = ROOT / "02-FFT" / sample
        current_colmap_path = ROOT / "04-COLMAP" / sample
        rerun_path = best_rerun_path(sample, rerun_report)

        relink(raw_path, DATA_ROOT / "01-输入图像" / "01-raw_frames" / sample)
        relink(fft_path, DATA_ROOT / "01-输入图像" / "02-fft_frames" / sample)
        relink(ROOT / "03-GT" / sample, DATA_ROOT / "03-分割Mask" / "01-gt_masks" / sample)
        relink(ROOT / "03-SAM" / sample, DATA_ROOT / "03-分割Mask" / "02-sam_masks" / sample)
        relink(ROOT / "03-SEEM" / sample, DATA_ROOT / "03-分割Mask" / "03-seem_masks" / sample)
        relink(current_colmap_path, DATA_ROOT / "02-位姿COLMAP" / "01-current_ok" / sample)
        if rerun_path:
            relink(rerun_path, DATA_ROOT / "02-位姿COLMAP" / "02-rerun_original_candidates" / sample)

        relink(ROOT / "05-2DGS-new" / sample, DATA_ROOT / "04-重建结果" / "01-2dgs_gaussians_existing" / sample)
        relink(ROOT / "06-MESH-new" / sample, DATA_ROOT / "04-重建结果" / "02-2dgs_mesh_existing" / sample)
        relink(ROOT / "07-SuGaR-GS" / sample, DATA_ROOT / "04-重建结果" / "03-3dgs_sugar_gaussians_existing" / sample)
        relink(ROOT / "07-SuGaR-Mesh" / sample, DATA_ROOT / "04-重建结果" / "04-3dgs_sugar_mesh_existing" / sample)

        current_stats = colmap_stats(current_colmap_path)
        rerun_stats: dict[str, object] = {}
        rerun_entry = rerun_report.get(sample)
        if isinstance(rerun_entry, dict) and isinstance(rerun_entry.get("best"), dict):
            rerun_stats = rerun_entry["best"]

        selected_path = rerun_path or (current_colmap_path if current_stats["status"] == "ok" else None)
        if selected_path:
            relink(selected_path, DATA_ROOT / "02-位姿COLMAP" / "03-final_locked" / sample)

        rows.append({
            "sample": sample,
            "raw_frames": count_images(raw_path),
            "fft_frames": count_images(fft_path),
            "has_gt": "yes" if sample in GT_SAMPLES else "no",
            "current_colmap_status": current_stats["status"],
            "current_registered": current_stats["registered"],
            "current_registration_rate": current_stats["registration_rate"],
            "current_points3d": current_stats["points3d"],
            "rerun_colmap_status": rerun_stats.get("status", ""),
            "rerun_registered": rerun_stats.get("registered", ""),
            "rerun_registration_rate": rerun_stats.get("registration_rate", ""),
            "rerun_points3d": rerun_stats.get("points3d", ""),
            "selected_colmap_path": str(selected_path) if selected_path else "",
            "notes": "rerun_candidate_preferred" if rerun_path else "",
        })

    csv_path = index_dir / "dataset_index.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = index_dir / "dataset_index.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
