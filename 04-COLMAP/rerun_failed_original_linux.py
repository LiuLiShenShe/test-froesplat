#!/usr/bin/env python3
"""Re-run failed COLMAP scenes with FFT-kept original RGB frames.

This script is intentionally non-destructive: it writes to
``04-COLMAP-rerun-original/`` and never modifies the existing ``04-COLMAP``
results. It first tries sequential matching, then falls back to exhaustive
matching if the registration rate is still low.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import time
from pathlib import Path


DEFAULT_SAMPLES = [
    "BaiZhang",
    "CaoMei2",
    "ChangShouHua1",
    "ChangShouHua3",
    "KongQueZhuYu",
    "XiangPiShu2",
]


def count_images_bin(path: Path) -> int:
    with path.open("rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def count_points3d_bin(path: Path) -> int:
    with path.open("rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def run_command(cmd: list[str], log_path: Path, cwd: Path | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(f"\n[exit_code] {proc.returncode}\n")
    return proc.returncode


def copy_input_images(sample: str, source_root: Path, run_dir: Path) -> int:
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    src_dir = source_root / sample
    files = sorted(
        p for p in src_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    for src in files:
        shutil.copy2(src, input_dir / src.name)
    return len(files)


def sparse_model_stats(model_dir: Path) -> dict[str, int] | None:
    cameras = model_dir / "cameras.bin"
    images = model_dir / "images.bin"
    points = model_dir / "points3D.bin"
    if not (cameras.exists() and images.exists() and points.exists()):
        return None
    return {
        "registered": count_images_bin(images),
        "points3d": count_points3d_bin(points),
    }


def select_best_model(sparse_root: Path) -> tuple[Path | None, dict[str, int]]:
    best_dir: Path | None = None
    best_stats: dict[str, int] = {"registered": 0, "points3d": 0}

    if not sparse_root.exists():
        return None, best_stats

    for model_dir in sorted(p for p in sparse_root.iterdir() if p.is_dir()):
        stats = sparse_model_stats(model_dir)
        if stats is None:
            continue
        key = (stats["registered"], stats["points3d"])
        best_key = (best_stats["registered"], best_stats["points3d"])
        if key > best_key:
            best_dir = model_dir
            best_stats = stats
    return best_dir, best_stats


def normalize_undistorted_sparse(run_dir: Path) -> None:
    sparse_dir = run_dir / "sparse"
    sparse_zero = sparse_dir / "0"
    sparse_zero.mkdir(parents=True, exist_ok=True)

    for item in list(sparse_dir.iterdir()):
        if item.name == "0":
            continue
        if item.is_file():
            target = sparse_zero / item.name
            if target.exists():
                target.unlink()
            shutil.move(str(item), str(target))


def run_colmap_attempt(
    sample: str,
    attempt_dir: Path,
    matcher: str,
    colmap: str,
    source_root: Path,
) -> dict[str, object]:
    if attempt_dir.exists():
        shutil.rmtree(attempt_dir)
    attempt_dir.mkdir(parents=True)
    attempt_dir = attempt_dir.resolve()
    log_dir = attempt_dir / "logs"

    input_count = copy_input_images(sample, source_root, attempt_dir)
    database = attempt_dir / "distorted" / "database.db"
    distorted_sparse = attempt_dir / "distorted" / "sparse"
    distorted_sparse.mkdir(parents=True, exist_ok=True)

    start = time.time()

    commands = [
        (
            "feature_extractor",
            [
                colmap,
                "feature_extractor",
                "--database_path", str(database),
                "--image_path", str(attempt_dir / "input"),
                "--ImageReader.single_camera", "1",
                "--ImageReader.camera_model", "OPENCV",
                "--SiftExtraction.use_gpu", "0",
                "--SiftExtraction.max_image_size", "3200",
                "--SiftExtraction.max_num_features", "12000",
                "--SiftExtraction.estimate_affine_shape", "1",
                "--SiftExtraction.domain_size_pooling", "1",
            ],
        ),
        (
            f"{matcher}_matcher",
            [
                colmap,
                f"{matcher}_matcher",
                "--database_path", str(database),
                "--SiftMatching.use_gpu", "0",
                "--SiftMatching.guided_matching", "1",
            ],
        ),
        (
            "mapper",
            [
                colmap,
                "mapper",
                "--database_path", str(database),
                "--image_path", str(attempt_dir / "input"),
                "--output_path", str(distorted_sparse),
                "--Mapper.ba_global_function_tolerance", "0.000001",
                "--Mapper.multiple_models", "1",
                "--Mapper.max_num_models", "5",
            ],
        ),
    ]

    if matcher == "sequential":
        commands[1][1].extend([
            "--SequentialMatching.overlap", "20",
            "--SequentialMatching.quadratic_overlap", "1",
            "--SequentialMatching.loop_detection", "0",
        ])

    for name, cmd in commands:
        code = run_command(cmd, log_dir / f"{name}.log", cwd=attempt_dir)
        if code != 0:
            elapsed = time.time() - start
            return {
                "status": "command_failed",
                "failed_step": name,
                "matcher": matcher,
                "input_images": input_count,
                "time_seconds": round(elapsed, 1),
            }

    best_model, best_stats = select_best_model(distorted_sparse)
    if best_model is None:
        elapsed = time.time() - start
        return {
            "status": "no_sparse_model",
            "matcher": matcher,
            "input_images": input_count,
            "time_seconds": round(elapsed, 1),
        }

    undistort_cmd = [
        colmap,
        "image_undistorter",
        "--image_path", str(attempt_dir / "input"),
        "--input_path", str(best_model),
        "--output_path", str(attempt_dir),
        "--output_type", "COLMAP",
    ]
    code = run_command(undistort_cmd, log_dir / "image_undistorter.log", cwd=attempt_dir)
    if code != 0:
        elapsed = time.time() - start
        return {
            "status": "undistort_failed",
            "matcher": matcher,
            "input_images": input_count,
            "best_model": best_model.name,
            **best_stats,
            "registration_rate": round(best_stats["registered"] / input_count * 100, 1) if input_count else 0.0,
            "time_seconds": round(elapsed, 1),
        }

    normalize_undistorted_sparse(attempt_dir)
    final_stats = sparse_model_stats(attempt_dir / "sparse" / "0") or best_stats
    images_dir = attempt_dir / "images"
    undistorted_images = len([
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]) if images_dir.exists() else 0

    elapsed = time.time() - start
    rate = final_stats["registered"] / input_count * 100 if input_count else 0.0
    return {
        "status": "success",
        "matcher": matcher,
        "input_images": input_count,
        "best_model": best_model.name,
        "registered": final_stats["registered"],
        "registration_rate": round(rate, 1),
        "points3d": final_stats["points3d"],
        "undistorted_images": undistorted_images,
        "time_seconds": round(elapsed, 1),
    }


def process_sample(
    sample: str,
    output_root: Path,
    source_root: Path,
    colmap: str,
    min_rate: float,
) -> dict[str, object]:
    sample_root = output_root / sample
    sample_root.mkdir(parents=True, exist_ok=True)

    attempts = []
    for matcher in ("sequential", "exhaustive"):
        attempt_dir = sample_root / matcher
        result = run_colmap_attempt(sample, attempt_dir, matcher, colmap, source_root)
        attempts.append(result)
        if (
            result.get("status") == "success"
            and float(result.get("registration_rate", 0.0)) >= min_rate
            and int(result.get("points3d", 0)) > 0
        ):
            break

    best = max(
        attempts,
        key=lambda r: (
            int(r.get("registered", 0)),
            int(r.get("points3d", 0)),
            -1 if r.get("status") != "success" else 0,
        ),
    )
    return {"sample": sample, "best": best, "attempts": attempts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="*", default=DEFAULT_SAMPLES)
    parser.add_argument("--source-root", default="02-FFT")
    parser.add_argument("--output-root", default="04-COLMAP-rerun-original")
    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--min-rate", type=float, default=70.0)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    source_root = Path(args.source_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    report_path = output_root / "rerun_report.json"
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as f:
            report: dict[str, object] = json.load(f)
    else:
        report = {}

    for sample in args.samples:
        print(f"\n=== {sample} ===", flush=True)
        result = process_sample(
            sample=sample,
            output_root=output_root,
            source_root=source_root,
            colmap=args.colmap,
            min_rate=args.min_rate,
        )
        report[sample] = result
        best = result["best"]
        print(json.dumps(best, ensure_ascii=False, indent=2), flush=True)

        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
