#!/usr/bin/env python3
"""Low-memory COLMAP rerun for XiangPiShu2.

The normal rerun script uses full CPU parallelism plus affine shape and DSP
SIFT options. XiangPiShu2 is a high-resolution scene, and both the sequential
and exhaustive attempts were killed by the OS during feature extraction. This
script keeps the original failed report entries, writes low-memory attempts
under ``04-COLMAP-rerun-original/XiangPiShu2/*_lowmem``, and updates the main
``rerun_report.json`` only after each attempt finishes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path


ROOT = Path("/data/fj/F2DMAS")
SAMPLE = "XiangPiShu2"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def count_images_bin(path: Path) -> int:
    with path.open("rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def count_points3d_bin(path: Path) -> int:
    with path.open("rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def run_command(cmd: list[str], log_path: Path, cwd: Path, num_threads: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(num_threads)
    env["OPENBLAS_NUM_THREADS"] = str(num_threads)
    env["MKL_NUM_THREADS"] = str(num_threads)

    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        log.write(f"\n[exit_code] {proc.returncode}\n")
    return proc.returncode


def copy_input_images(source_dir: Path, run_dir: Path) -> int:
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
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
    best_stats = {"registered": 0, "points3d": 0}
    if not sparse_root.exists():
        return None, best_stats
    for model_dir in sorted(p for p in sparse_root.iterdir() if p.is_dir()):
        stats = sparse_model_stats(model_dir)
        if stats is None:
            continue
        if (stats["registered"], stats["points3d"]) > (best_stats["registered"], best_stats["points3d"]):
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


def run_attempt(
    matcher_label: str,
    matcher_cmd: str,
    output_root: Path,
    source_root: Path,
    colmap: str,
    max_image_size: int,
    max_num_features: int,
    num_threads: int,
    sequential_overlap: int,
) -> dict[str, object]:
    run_dir = output_root / SAMPLE / matcher_label
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    run_dir = run_dir.resolve()

    start = time.time()
    input_count = copy_input_images(source_root / SAMPLE, run_dir)
    log_dir = run_dir / "logs"
    database = run_dir / "distorted" / "database.db"
    distorted_sparse = run_dir / "distorted" / "sparse"
    distorted_sparse.mkdir(parents=True, exist_ok=True)

    feature_cmd = [
        colmap,
        "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(run_dir / "input"),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "OPENCV",
        "--SiftExtraction.use_gpu", "0",
        "--SiftExtraction.num_threads", str(num_threads),
        "--SiftExtraction.max_image_size", str(max_image_size),
        "--SiftExtraction.max_num_features", str(max_num_features),
        "--SiftExtraction.estimate_affine_shape", "0",
        "--SiftExtraction.domain_size_pooling", "0",
    ]
    code = run_command(feature_cmd, log_dir / "feature_extractor.log", run_dir, num_threads)
    if code != 0:
        return {
            "status": "command_failed",
            "failed_step": "feature_extractor",
            "matcher": matcher_label,
            "input_images": input_count,
            "time_seconds": round(time.time() - start, 1),
        }

    match_cmd = [
        colmap,
        matcher_cmd,
        "--database_path", str(database),
        "--SiftMatching.use_gpu", "0",
        "--SiftMatching.num_threads", str(num_threads),
        "--SiftMatching.guided_matching", "1",
    ]
    if matcher_cmd == "sequential_matcher":
        match_cmd.extend([
            "--SequentialMatching.overlap", str(sequential_overlap),
            "--SequentialMatching.quadratic_overlap", "1",
            "--SequentialMatching.loop_detection", "0",
        ])
    code = run_command(match_cmd, log_dir / f"{matcher_label}_matcher.log", run_dir, num_threads)
    if code != 0:
        return {
            "status": "command_failed",
            "failed_step": f"{matcher_label}_matcher",
            "matcher": matcher_label,
            "input_images": input_count,
            "time_seconds": round(time.time() - start, 1),
        }

    mapper_cmd = [
        colmap,
        "mapper",
        "--database_path", str(database),
        "--image_path", str(run_dir / "input"),
        "--output_path", str(distorted_sparse),
        "--Mapper.ba_global_function_tolerance", "0.000001",
        "--Mapper.multiple_models", "1",
        "--Mapper.max_num_models", "5",
    ]
    code = run_command(mapper_cmd, log_dir / "mapper.log", run_dir, num_threads)
    if code != 0:
        return {
            "status": "command_failed",
            "failed_step": "mapper",
            "matcher": matcher_label,
            "input_images": input_count,
            "time_seconds": round(time.time() - start, 1),
        }

    best_model, best_stats = select_best_model(distorted_sparse)
    if best_model is None:
        return {
            "status": "no_sparse_model",
            "matcher": matcher_label,
            "input_images": input_count,
            "time_seconds": round(time.time() - start, 1),
        }

    undistort_cmd = [
        colmap,
        "image_undistorter",
        "--image_path", str(run_dir / "input"),
        "--input_path", str(best_model),
        "--output_path", str(run_dir),
        "--output_type", "COLMAP",
    ]
    code = run_command(undistort_cmd, log_dir / "image_undistorter.log", run_dir, num_threads)
    if code != 0:
        return {
            "status": "undistort_failed",
            "matcher": matcher_label,
            "input_images": input_count,
            "best_model": best_model.name,
            **best_stats,
            "registration_rate": round(best_stats["registered"] / input_count * 100, 1) if input_count else 0.0,
            "time_seconds": round(time.time() - start, 1),
        }

    normalize_undistorted_sparse(run_dir)
    final_stats = sparse_model_stats(run_dir / "sparse" / "0") or best_stats
    undistorted_images = len([
        p for p in (run_dir / "images").iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]) if (run_dir / "images").exists() else 0
    return {
        "status": "success",
        "matcher": matcher_label,
        "input_images": input_count,
        "best_model": best_model.name,
        "registered": final_stats["registered"],
        "registration_rate": round(final_stats["registered"] / input_count * 100, 1) if input_count else 0.0,
        "points3d": final_stats["points3d"],
        "undistorted_images": undistorted_images,
        "time_seconds": round(time.time() - start, 1),
    }


def choose_best(attempts: list[dict[str, object]]) -> dict[str, object]:
    return max(
        attempts,
        key=lambda item: (
            1 if item.get("status") == "success" else 0,
            int(item.get("registered", 0)),
            int(item.get("points3d", 0)),
        ),
    )


def update_report(report_path: Path, new_attempts: list[dict[str, object]]) -> dict[str, object]:
    report: dict[str, object]
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)
    else:
        report = {}

    entry = report.get(SAMPLE)
    old_attempts: list[dict[str, object]] = []
    if isinstance(entry, dict) and isinstance(entry.get("attempts"), list):
        old_attempts = [item for item in entry["attempts"] if isinstance(item, dict)]

    seen = {str(item.get("matcher", "")) for item in new_attempts}
    merged_attempts = [item for item in old_attempts if str(item.get("matcher", "")) not in seen]
    merged_attempts.extend(new_attempts)

    report[SAMPLE] = {
        "sample": SAMPLE,
        "best": choose_best(merged_attempts),
        "attempts": merged_attempts,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report[SAMPLE]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(ROOT / "02-FFT"))
    parser.add_argument("--output-root", default=str(ROOT / "04-COLMAP-rerun-original"))
    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--max-image-size", type=int, default=2000)
    parser.add_argument("--max-num-features", type=int, default=6000)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--sequential-overlap", type=int, default=20)
    parser.add_argument("--min-rate", type=float, default=70.0)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    source_root = Path(args.source_root).resolve()
    report_path = output_root / "rerun_report.json"

    attempts: list[dict[str, object]] = []
    for matcher_label, matcher_cmd in (
        ("sequential_lowmem", "sequential_matcher"),
        ("exhaustive_lowmem", "exhaustive_matcher"),
    ):
        print(f"\n=== {SAMPLE}: {matcher_label} ===", flush=True)
        result = run_attempt(
            matcher_label=matcher_label,
            matcher_cmd=matcher_cmd,
            output_root=output_root,
            source_root=source_root,
            colmap=args.colmap,
            max_image_size=args.max_image_size,
            max_num_features=args.max_num_features,
            num_threads=args.num_threads,
            sequential_overlap=args.sequential_overlap,
        )
        attempts.append(result)
        updated = update_report(report_path, attempts)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if (
            result.get("status") == "success"
            and float(result.get("registration_rate", 0.0)) >= args.min_rate
            and int(result.get("points3d", 0)) > 0
        ):
            break

    print(f"\nUpdated report entry:\n{json.dumps(updated, ensure_ascii=False, indent=2)}", flush=True)
    print(f"\nReport: {report_path}", flush=True)


if __name__ == "__main__":
    main()
